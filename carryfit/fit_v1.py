"""carryfit 第一版(只做两人抬一个箱子, 11-05), 留着复现 runs/couplelift_r1。新版是 fit.py。

输入(全来自室内流程的一轮, 外加物体网格):
  <r1>/02_da3/cameras.json     每个关键帧的相机(OpenCV, DA3 世界, 米); 内参是 input_hw 分辨率, 按 x / y 分别缩放到视频分辨率
                               (和 pipeline/body_video.py 一样)
  <r1>/02_da3/frame_map.json   关键帧 -> 视频帧号
  <r1>/02_da3/world_up.json    世界里"天"的方向
  <r1>/10_people_world.npz     每个关键帧每个人的 MHR 网格(DA3 世界)
  <r1>/03_masks/<词>.npz       SAM3 跟踪的 mask(视频分辨率 RLE), 取 --oid 那个实例
  pipeline/mhr_parts.npz       hand_w: MHR 顶点的手部权重 -> 每人两只手
  --mesh                       物体网格(SAM 3D Objects 出的 glb)

求解(所有关键帧一起):
  未知数: 物体整体一个缩放 s(网格最长边 = s 米); 每帧旋转(6D) + 平移
  初值: 两人手的中点 = 物体中心; 物体最长边沿两人手的连线; 另一条边朝天(两种: 中边朝天 / 短边朝天, 都试, 取损失小的);
        s = 两人手距的中位数
  损失: 轮廓   nvdiffrast 渲物体轮廓, 和 SAM3 mask 算 soft IoU; 被人挡住的像素不算(人的轮廓用我们的人体网格渲, 外扩 2 像素)
        接触   每只手到物体表面的最近距离, 鲁棒损失 d^2/(d^2+5cm^2) —— 没拿着的手离得远, 损失饱和, 不会把物体拽过去
        不穿插 人体顶点进入物体(物体按定向长方体近似)的深度
        平滑   平移 / 旋转的二阶差分
输出 <out>/: carry_poses.npz, metrics.json, preview.mp4(原图 | 渲染 | 叠加), 有 --ref 时再出 compare.mp4(上我们 / 下参考)

  conda activate fpose; PYOPENGL_PLATFORM=egl python carryfit/fit.py --r1 <r1 目录> --word box --oid 1 \
      --mesh <glb> --out carryfit/runs/<名字> [--ref <参考预览 mp4>]
"""
import argparse
import json
import os
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import trimesh
import nvdiffrast.torch as dr
from pycocotools import mask as mu
from scipy.sparse.csgraph import connected_components

SAM3D = "/home/ubuntu/sam3d"
DEV = "cuda"
RW, RH = 480, 270          # 优化时渲染分辨率(视频的 1/4)


# ---------------------------------------------------------------- 读数据
def load_inputs(r1, word, oid):
    cams = json.load(open(f"{r1}/02_da3/cameras.json"))["cameras"]
    k2v = json.load(open(f"{r1}/02_da3/frame_map.json"))["frames"]
    up = np.array(json.load(open(f"{r1}/02_da3/world_up.json"))["up"], float)
    keys = sorted(cams)
    md = np.load(f"{r1}/03_masks/{word}.npz", allow_pickle=True)
    H, W = [int(x) for x in md["hw"]]
    Ks, w2c = [], []
    for k in keys:
        c = cams[k]
        h, w = c["input_hw"]
        Ks.append(np.diag([W / w, H / h, 1.0]) @ np.array(c["intrinsic_3x3"], float))
        w2c.append(np.array(c["extrinsic_w2c_4x4"], float).reshape(4, 4))
    n = len(keys)
    masks = np.zeros((n, H, W), bool)
    has = np.zeros(n, bool)
    meta = md["meta"]
    for i in np.where(meta["oid"] == oid)[0]:
        f = int(meta["frame"][i])
        masks[f] = mu.decode({"counts": md["counts"][i], "size": [H, W]}).astype(bool)
        has[f] = masks[f].any()
    pw = np.load(f"{r1}/10_people_world.npz")
    faces = pw["faces"].astype(np.int32)
    people = {}
    vf_of_key = np.array([k2v[k] for k in keys])
    for f in pw.files:
        if f.endswith("_V"):
            pid = f[:-2]
            V, vf = pw[f], pw[f"{pid}_vf"]
            row = {int(v): j for j, v in enumerate(vf)}
            out = np.full((n, V.shape[1], 3), np.nan, np.float32)
            for i, v in enumerate(vf_of_key):
                if int(v) in row:
                    out[i] = V[row[int(v)]]
            people[pid] = out
    # 真实人体 mask(body4d 跟踪的, 每个关键帧一张, 颜色区分人, 非零 = 有人); 用来判断箱子哪里被人挡住
    pm_dir = f"{r1}/body4d/02_body4d/masks"
    pmask = None
    if os.path.isdir(pm_dir) and len(os.listdir(pm_dir)) == n:
        pmask = np.stack([cv2.imread(f"{pm_dir}/{i:08d}.png", cv2.IMREAD_UNCHANGED).reshape(H, W, -1).max(-1) > 0
                          for i in range(n)])
    return dict(keys=keys, vf=vf_of_key, K=np.stack(Ks), w2c=np.stack(w2c), up=up / np.linalg.norm(up),
                masks=masks, has_mask=has, H=H, W=W, people=people, faces=faces, pmask=pmask)


def hand_groups(faces, nv):
    """MHR 的手部顶点(hand_w > 0.5)按网格连通性分成两只手"""
    hw = np.load(f"{SAM3D}/pipeline/mhr_parts.npz")["hand_w"]
    idx = np.where(hw > 0.5)[0]
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keep = np.isin(e[:, 0], idx) & np.isin(e[:, 1], idx)
    A = sp.coo_matrix((np.ones(keep.sum()), (e[keep, 0], e[keep, 1])), shape=(nv, nv))
    _, lab = connected_components(A, directed=False)
    L = lab[idx]
    big = [c for c, _ in sorted(((c, (L == c).sum()) for c in np.unique(L)), key=lambda x: -x[1])[:2]]
    return [idx[L == c] for c in big]


def load_box_mesh(path):
    """网格转到自己的定向包围盒系: 轴对齐、居中, x 最长 y 中 z 最短, 最长边归一成 1"""
    m = trimesh.load(path, force="mesh", process=False)
    T, ext = trimesh.bounds.oriented_bounds(m)
    v = trimesh.transform_points(m.vertices, T)
    order = np.argsort(-ext)
    v, ext = v[:, order], ext[order]
    v = v / ext[0]
    P = np.eye(4)[list(order) + [3]]                       # 轴重排
    if np.linalg.det(P[:3, :3] @ T[:3, :3]) < 0:           # 重排成了镜像: 三角形朝向会整个翻到里面(渲染时近处的面被当背面剔掉)
        P[2] *= -1                                         # 最短轴反向, 变回正常旋转; 盒子对称, 形状不变
        v[:, 2] *= -1
    A = np.diag([1 / ext[0]] * 3 + [1.0]) @ P @ T          # 原始网格(glb 场景坐标) -> 局部系
    half = ext / ext[0] / 2
    mm = trimesh.Trimesh(v, m.faces, process=False)
    pts, _ = trimesh.sample.sample_surface(mm, 1500, seed=0)
    return dict(V=v.astype(np.float32), F=np.asarray(m.faces, np.int32), half=half.astype(np.float32), A=A,
                pts=pts.astype(np.float32), ext_ratio=(ext / ext[0]).astype(np.float32))


# ---------------------------------------------------------------- 几何小工具
def r6_to_mat(x):
    b1 = F.normalize(x[..., :3], dim=-1)
    a2 = x[..., 3:]
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], -1)          # 列 = 物体局部 x/y/z 在世界里的方向


def to_clip(X, w2c, K, Wr, Hr, near=0.05, far=50.0):
    """世界点 (n, V, 3) -> nvdiffrast clip 坐标; 图像第 0 行对应 ndc_y = -1, 所以 v 直接映射, 不翻转"""
    Xc = torch.einsum("nij,nvj->nvi", w2c[:, :3, :3], X) + w2c[:, None, :3, 3]
    z = Xc[..., 2].clamp(min=1e-4)
    u = K[:, None, 0, 0] * Xc[..., 0] / z + K[:, None, 0, 2]
    v = K[:, None, 1, 1] * Xc[..., 1] / z + K[:, None, 1, 2]
    zc = Xc[..., 2]
    return torch.stack([(2 * u / Wr - 1) * zc, (2 * v / Hr - 1) * zc,
                        zc * (far + near) / (far - near) - 2 * far * near / (far - near), zc], -1).contiguous()


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--word", required=True)
    ap.add_argument("--oid", type=int, required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="参考预览视频(同样 120 个关键帧, 3 列), 拼对比用")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--w_con", type=float, default=0.3, help="接触(手贴物体)在细调阶段的权重; 粗调阶段是它的 0.2 倍")
    ap.add_argument("--w_ends", type=float, default=10.0, help="两人抬: 每人的手落在物体长边一端附近(端面往里 15cm 到往外 5cm)")
    ap.add_argument("--w_pen", type=float, default=10.0, help="穿插(超过容差的部分)权重")
    ap.add_argument("--w_up", type=float, default=0.05, help="物体朝天那根轴别歪的权重(抬起过程中会斜, 别太大)")
    ap.add_argument("--w_shape", type=float, default=0.1, help="三向比例贴近 SAM 3D 网格的权重")
    ap.add_argument("--pen_tol", type=float, default=0.04, help="穿插容差(米): 人体进物体不超过这么深不罚(箱子本来贴着身体, 重建有几厘米误差)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    D = load_inputs(a.r1, a.word, a.oid)
    n, H, W = len(D["keys"]), D["H"], D["W"]
    box = load_box_mesh(a.mesh)
    pids = sorted(D["people"])
    nv = D["people"][pids[0]].shape[1]
    hands = hand_groups(D["faces"], nv)
    assert len(pids) == 2, f"第一版只处理两个人抬一个东西, 这里有 {len(pids)} 个人"

    # ---- 张量
    w2c = torch.tensor(D["w2c"], dtype=torch.float32, device=DEV)
    has = torch.tensor(D["has_mask"], device=DEV)
    rng = np.random.default_rng(0)
    PV = {}
    for p in pids:
        V = D["people"][p].copy()
        miss = np.isnan(V[:, 0, 0])
        V[miss] = np.array([0, 0, -100.0])             # 这帧没这个人: 扔到相机后面
        PV[p] = torch.tensor(V, device=DEV)
    hand_sub = [rng.choice(h, 80, replace=False) for h in hands]
    Hpts = torch.stack([PV[p][:, hs] for p in pids for hs in hand_sub], 1)          # (n, 4, 80, 3)
    body_sub = rng.choice(nv, 2000, replace=False)
    Bpts = torch.cat([PV[p][:, body_sub] for p in pids], 1)                          # (n, 4000, 3)
    dmin = torch.cdist(Bpts, Hpts.mean(2)).amin(-1)                                  # (n, 4000) 到最近一只手的距离
    pen_w = ((dmin - 0.20) / 0.10).clamp(0, 1)
    BV = torch.tensor(box["V"], device=DEV)
    BF = torch.tensor(box["F"], device=DEV)
    BP = torch.tensor(box["pts"][rng.choice(len(box["pts"]), 800, replace=False)], device=DEV)
    half = torch.tensor(box["half"], device=DEV)
    glctx = dr.RasterizeCudaContext()

    # 两个人合成一个网格(算遮挡用; 每个分辨率的遮挡在 stage_tensors 里渲, 外扩 2 像素)
    PF = torch.tensor(np.concatenate([D["faces"] + i * nv for i in range(len(pids))]), dtype=torch.int32, device=DEV)

    # ---- 初值: 两人手的中点 / 连线
    hc = Hpts.mean(2)                                   # (n, 4, 3) 每只手的中心
    h1, h2 = hc[:, :2].mean(1), hc[:, 2:].mean(1)       # 每个人两只手的中心
    axis = F.normalize(h2 - h1, dim=-1)
    upw = torch.tensor(D["up"], dtype=torch.float32, device=DEV).expand(n, 3)
    upo = F.normalize(upw - (upw * axis).sum(-1, keepdim=True) * axis, dim=-1)
    third = torch.cross(axis, upo, dim=-1)
    # 中心: 沿 SAM3 mask 质心的视线, 深度取两人手中点的深度(他们托着箱子底, 箱子在手上方, 不能直接用手的中点)
    hm = ((h1 + h2) / 2).cpu().numpy()
    center0 = hm.copy()
    for i in range(n):
        if not D["has_mask"][i]:
            continue
        ys, xs = np.nonzero(D["masks"][i])
        Rw, tw = D["w2c"][i][:3, :3], D["w2c"][i][:3, 3]
        z = (Rw @ hm[i] + tw)[2]
        K = D["K"][i]
        pc = np.array([(xs.mean() - K[0, 2]) / K[0, 0] * z, (ys.mean() - K[1, 2]) / K[1, 1] * z, z])
        center0[i] = Rw.T @ (pc - tw)
    center0 = torch.tensor(center0, dtype=torch.float32, device=DEV)
    s0 = float(torch.median((h2 - h1).norm(dim=-1))) * 1.1

    def stage_tensors(rw, rh, blur):
        """某个分辨率下的: 内参、观测 mask、人遮挡(外扩)、高斯核"""
        kk = torch.tensor(D["K"] * np.array([[rw / W], [rh / H], [1]]), dtype=torch.float32, device=DEV)
        ob = torch.tensor(np.stack([cv2.resize(m.astype(np.float32), (rw, rh), interpolation=cv2.INTER_AREA)
                                    for m in D["masks"]]), device=DEV)
        with torch.no_grad():
            if D["pmask"] is not None:
                oc = torch.tensor(np.stack([cv2.resize(m.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
                                            for m in D["pmask"]]), dtype=torch.float32, device=DEV)
            else:
                rr, _ = dr.rasterize(glctx, to_clip(torch.cat([PV[p] for p in pids], 1), w2c, kk, rw, rh), PF, (rh, rw))
                oc = (rr[..., 3] > 0).float()
            oc = (F.max_pool2d(oc[:, None], 5, 1, 2)[:, 0] > 0).float()
        g = None
        if blur > 0:
            r_ = int(3 * blur)
            x = torch.arange(-r_, r_ + 1, device=DEV, dtype=torch.float32)
            g1 = torch.exp(-x ** 2 / (2 * blur ** 2)); g1 = g1 / g1.sum()
            g = (g1[:, None] * g1[None, :])[None, None]
            ob = F.conv2d(ob[:, None], g, padding=r_)[:, 0]
        return dict(K=kk, obs=ob, valid=1 - oc, g=g, rw=rw, rh=rh)

    STAGES = [(stage_tensors(240, 135, 4.0), 0.2 * a.w_con, a.iters // 2), (stage_tensors(RW, RH, 0.0), a.w_con, a.iters - a.iters // 2)]

    def solve(variant, iters):
        R0 = torch.stack([axis, upo, third], -1) if variant == "中边朝天" else torch.stack([axis, -third, upo], -1)
        r6 = torch.cat([R0[..., 0], R0[..., 1]], -1).clone().requires_grad_(True)
        t = center0.clone().requires_grad_(True)
        up_ax = 1 if variant == "中边朝天" else 2                     # 物体局部哪根轴朝天
        logs = torch.full((3,), float(np.log(s0)), device=DEV, requires_grad=True)   # 长 / 中 / 短 三个方向各自缩放
        opt = torch.optim.Adam([{"params": [r6, t], "lr": 0.01}, {"params": [logs], "lr": 0.005}])
        hist = []
        sched = [(st, wc) for st, wc, k in STAGES for _ in range(k)] + [(STAGES[-1][0], STAGES[-1][1])]
        for it in range(len(sched)):
            st, w_con = sched[it]
            R = r6_to_mat(r6)
            s = logs.exp()                                          # (3,)
            Xw = torch.einsum("nij,vj->nvi", R, BV * s) + t[:, None]
            clip = to_clip(Xw, w2c, st["K"], st["rw"], st["rh"])
            rast, _ = dr.rasterize(glctx, clip, BF, (st["rh"], st["rw"]))
            sil = dr.antialias((rast[..., 3:4] > 0).float(), rast, clip, BF)[..., 0]
            if st["g"] is not None:
                sil = F.conv2d(sil[:, None], st["g"], padding=st["g"].shape[-1] // 2)[:, 0]
            obs, valid = st["obs"], st["valid"]
            inter = (sil * obs * valid).sum((1, 2))
            union = ((sil + obs - sil * obs) * valid).sum((1, 2)).clamp(min=1.0)
            l_sil = ((1 - inter / union) * has).sum() / has.sum().clamp(min=1)
            # 接触: 每只手到物体表面采样点的最近距离
            Pw = torch.einsum("nij,vj->nvi", R, BP * s) + t[:, None]                  # (n, 800, 3)
            d = torch.cdist(Hpts.reshape(n, -1, 3), Pw).reshape(n, 4, 80, -1).amin((2, 3))   # (n, 4)
            l_con = (d ** 2 / (d ** 2 + 0.05 ** 2)).mean()
            # 不穿插: 人体点在物体(定向长方体)里的深度
            q = torch.einsum("nji,nvj->nvi", R, Bpts - t[:, None])
            depth = ((half * s - q.abs()).amin(-1) - a.pen_tol).clamp(min=0)
            l_pen = (pen_w * depth ** 2).sum(1).mean()
            # 平滑
            acc_t = t[2:] - 2 * t[1:-1] + t[:-2]
            acc_R = R[2:] - 2 * R[1:-1] + R[:-2]
            l_smo = (acc_t ** 2).sum(-1).mean() + 0.1 * (acc_R ** 2).sum((-1, -2)).mean()
            l_up = (1 - (R[:, :, up_ax] @ upw[0]) ** 2).mean()       # 物体朝天那根轴别歪(软约束)
            # 两人抬: 每个人两只手的中心, 沿物体长边的坐标应在端面附近
            ax_c = torch.stack([torch.einsum("ni,ni->n", R[:, :, 0], hp - t) for hp in (h1, h2)], 1).abs()   # (n, 2)
            hl = 0.5 * s[0]
            l_ends = (F.relu(ax_c - (hl + 0.05)) ** 2 + F.relu((hl - 0.15) - ax_c) ** 2).mean()
            l_shape = ((logs - logs.mean()) ** 2).sum()             # 三向比例别离 SAM 3D 网格太远
            loss = (l_sil + w_con * l_con + a.w_pen * l_pen + 300.0 * l_smo + a.w_up * l_up + a.w_shape * l_shape
                    + a.w_ends * l_ends)
            iters = len(sched) - 1
            if it % 100 == 0 or it == iters:
                hist.append(dict(it=it, loss=float(loss), sil=float(l_sil), con=float(l_con), pen=float(l_pen),
                                 smo=float(l_smo), up=float(l_up), s=[round(float(x), 3) for x in s]))
            if it == iters:
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
        return dict(R=R.detach(), t=t.detach(), s=logs.exp().detach(), hist=hist, loss=hist[-1]["loss"])

    res = {v: solve(v, a.iters) for v in ["中边朝天", "短边朝天"]}
    best = min(res, key=lambda v: res[v]["loss"])
    sol = res[best]
    for v in res:
        print(v, "最终损失", round(res[v]["loss"], 4), res[v]["hist"][-1])
    t_fit = time.time() - t0

    # ---- 评估(960x540, 物体和人一起渲, 做遮挡): 可见箱子 vs SAM3 mask
    EW, EH = 960, 540
    Ke = torch.tensor(D["K"] * np.array([[EW / W], [EH / H], [1]]), dtype=torch.float32, device=DEV)
    obs_e = np.stack([cv2.resize(m.astype(np.uint8), (EW, EH), interpolation=cv2.INTER_NEAREST) for m in D["masks"]]).astype(bool)

    pm_e = None
    if D["pmask"] is not None:
        pm_e = np.stack([cv2.resize(m.astype(np.uint8), (EW, EH), interpolation=cv2.INTER_NEAREST) for m in D["pmask"]]).astype(bool)
        pm_e = F.max_pool2d(torch.tensor(pm_e[:, None], dtype=torch.float32), 5, 1, 2)[:, 0].numpy() > 0

    def vis_iou(R, t, s):
        with torch.no_grad():
            Xw = torch.einsum("nij,vj->nvi", R, BV * s) + t[:, None]
            allv = torch.cat([Xw] + [PV[p] for p in pids], 1)
            allf = torch.cat([BF, PF + BV.shape[0]], 0)
            rast, _ = dr.rasterize(glctx, to_clip(allv, w2c, Ke, EW, EH), allf, (EH, EW))
            tid = rast[..., 3].long()
            visb = ((tid > 0) & (tid <= BF.shape[0])).cpu().numpy()
            rb, _ = dr.rasterize(glctx, to_clip(Xw, w2c, Ke, EW, EH), BF, (EH, EW))
            silb = (rb[..., 3] > 0).cpu().numpy()
        out, out2 = [], []
        for i in range(n):
            if not D["has_mask"][i]:
                out.append(np.nan); out2.append(np.nan); continue
            out.append((visb[i] & obs_e[i]).sum() / max((visb[i] | obs_e[i]).sum(), 1))
            if pm_e is not None:
                v = ~pm_e[i]
                out2.append((silb[i] & obs_e[i] & v).sum() / max(((silb[i] | obs_e[i]) & v).sum(), 1))
        return np.array(out), (np.array(out2) if pm_e is not None else None)

    R_init = torch.stack([axis, upo, third], -1) if best == "中边朝天" else torch.stack([axis, -third, upo], -1)
    iou0, iou0b = vis_iou(R_init, center0, s0)
    iou1, iou1b = vis_iou(sol["R"], sol["t"], sol["s"])
    with torch.no_grad():
        Pw = torch.einsum("nij,vj->nvi", sol["R"], BP * sol["s"]) + sol["t"][:, None]
        dh = torch.cdist(Hpts.reshape(n, -1, 3), Pw).reshape(n, 4, 80, -1).amin((2, 3)).cpu().numpy() * 100
        q = torch.einsum("nji,nvj->nvi", sol["R"], Bpts - sol["t"][:, None])
        pen_all = (half * sol["s"] - q.abs()).amin(-1).clamp(min=0)
        pen = (pen_all * (pen_w > 0)).amax(1).cpu().numpy() * 100        # 手附近 25 cm 以外的身体
    segs = [(0, 30), (30, 60), (60, 90), (90, n)]
    seg_name = [f"关键帧 {a_}-{b_ - 1}(视频 {D['vf'][a_]}-{D['vf'][b_ - 1]})" for a_, b_ in segs]
    met = dict(
        选的初值=best, 用时秒=round(t_fit, 1), 物体尺寸米=[round(float(x) * float(y), 3) for x, y in zip(box["ext_ratio"], sol["s"])],
        可见轮廓IoU_初值={sn: round(float(np.nanmedian(iou0[a_:b_])), 3) for sn, (a_, b_) in zip(seg_name, segs)},
        可见轮廓IoU_拟合后={sn: round(float(np.nanmedian(iou1[a_:b_])), 3) for sn, (a_, b_) in zip(seg_name, segs)},
        真人以外轮廓IoU_初值=({sn: round(float(np.nanmedian(iou0b[a_:b_])), 3) for sn, (a_, b_) in zip(seg_name, segs)}
                         if iou0b is not None else None),
        真人以外轮廓IoU_拟合后=({sn: round(float(np.nanmedian(iou1b[a_:b_])), 3) for sn, (a_, b_) in zip(seg_name, segs)}
                          if iou1b is not None else None),
        四只手到箱子表面_cm_中位={sn: [round(float(x), 1) for x in np.median(dh[a_:b_], 0)] for sn, (a_, b_) in zip(seg_name, segs)},
        人穿进箱子最深_cm_中位={sn: round(float(np.median(pen[a_:b_])), 1) for sn, (a_, b_) in zip(seg_name, segs)},
        IoU低于05的帧=[int(i) for i in np.where(iou1 < 0.5)[0]],
        优化过程={v: res[v]["hist"] for v in res},
    )
    json.dump(met, open(f"{a.out}/metrics.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in met.items() if k != "优化过程"}, ensure_ascii=False, indent=1))

    # ---- 存位姿: 物体局部(定向包围盒系, 最长边 = 1) -> 世界
    M = np.tile(np.eye(4), (n, 1, 1))
    M[:, :3, :3] = sol["R"].cpu().numpy() * sol["s"].cpu().numpy()[None, None, :]
    M[:, :3, 3] = sol["t"].cpu().numpy()
    np.savez(f"{a.out}/carry_poses.npz", keys=np.array(D["keys"]), vf=D["vf"], local_to_world=M, scale=sol["s"].cpu().numpy(),
             mesh=a.mesh, mesh_to_local=box["A"], box_V=box["V"], box_F=box["F"], iou=iou1, hand_dist_cm=dh, pen_cm=pen)

    # ---- 预览: 原图 | 渲染(人 + 物体) | 叠加, 同 pipeline/body_video.py 的样子
    render_preview(a, D, pids, box, M)
    print("总用时", round(time.time() - t0, 1), "s")


def render_preview(a, D, pids, box, M):
    import imageio
    import pyrender
    TW = 640
    TH = int(round(TW * D["H"] / D["W"]))
    TH -= TH % 2
    r = pyrender.OffscreenRenderer(TW, TH)
    COL = {pids[0]: (230, 120, 40), pids[1]: (40, 160, 230)}
    frames_dir = os.path.join(os.path.dirname(a.r1.rstrip("/")), "frames_r1")
    wr = imageio.get_writer(f"{a.out}/preview.mp4", fps=8, codec="libx264", quality=8, macro_block_size=1)
    ref = None
    if a.ref:
        ref = imageio.get_reader(a.ref)
        wc = imageio.get_writer(f"{a.out}/compare.mp4", fps=8, codec="libx264", quality=8, macro_block_size=1)
    for i, k in enumerate(D["keys"]):
        sc = pyrender.Scene(bg_color=[0.96, 0.96, 0.96, 1.0], ambient_light=[0.35, 0.35, 0.35])
        K = D["K"][i] * np.array([[TW / D["W"]], [TH / D["H"]], [1]])
        cam = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=0.05, zfar=60)
        pose = np.linalg.inv(D["w2c"][i]) @ np.diag([1, -1, -1, 1])
        sc.add(cam, pose=pose)
        sc.add(pyrender.DirectionalLight(intensity=4.0), pose=pose)
        for p in pids:
            V = D["people"][p][i]
            if np.isnan(V[0, 0]):
                continue
            pm = trimesh.Trimesh(V, D["faces"], process=False)
            pm.visual.vertex_colors = np.array(list(COL[p]) + [255], np.uint8)
            sc.add(pyrender.Mesh.from_trimesh(pm, smooth=True))
        bm = trimesh.Trimesh(trimesh.transform_points(box["V"], M[i]), box["F"], process=False)
        bm.visual.vertex_colors = np.array([205, 170, 120, 255], np.uint8)
        sc.add(pyrender.Mesh.from_trimesh(bm, smooth=False))
        col, _ = r.render(sc)
        real = cv2.resize(cv2.cvtColor(cv2.imread(f"{frames_dir}/{k}"), cv2.COLOR_BGR2RGB), (TW, TH))
        row = np.hstack([real, col, (0.5 * real + 0.5 * col).astype(np.uint8)])
        cv2.putText(row, f"carryfit  {k}  video frame {D['vf'][i]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        wr.append_data(row)
        if ref is not None:
            try:
                rf = ref.get_data(i)
            except IndexError:
                rf = np.zeros_like(row)
            rf = cv2.resize(rf, (row.shape[1], row.shape[0]))
            cv2.putText(rf, "reference", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 80), 2)
            wc.append_data(np.vstack([row, rf]))
    wr.close()
    if ref is not None:
        wc.close()


if __name__ == "__main__":
    main()
