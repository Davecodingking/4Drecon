"""carryfit(第二版, 11-06): 人拿过的物体(端着 / 几个人合抬 / 骑 / 扔出去), 每个关键帧求它的位置、朝向, 外加一个固定大小。
第一版(只做两人抬箱子)在 fit_v1.py。一条命令跑全套(重跟 mask -> 生成网格 -> 拟合 -> 写 blend)用 carry.sh。

输入(全来自流水线的一轮, 室内室外一样, 外加物体网格):
  <r1>/02_da3/cameras.json      每个关键帧的相机(OpenCV, DA3 世界, 米); 内参按 x / y 分别缩放到视频分辨率
  <r1>/02_da3/frame_map.json    关键帧 -> 视频帧号, fps, 帧图片目录
  <r1>/02_da3/world_up.json     天的方向 + 地面高度
  <r1>/02_da3/depth/            DA3 深度(只在没人拿着的帧定起点的远近)
  <r1>/10_people_world.npz      每个关键帧每个人的 MHR 网格(DA3 世界)
  <r1>/body4d/02_body4d/masks   真人 mask(算物体哪里被人挡住)
  <r1>/03_masks/<词>.npz        SAM3 跟踪的 mask, 取 --oid 那个实例
  carryfit/parts.npz            MHR 的手 / 脚顶点(make_parts.py 出的)
  --mesh                        物体网格(SAM 3D Objects 出的 glb)

做法(所有关键帧一起优化, nvdiffrast 可微渲染):
  1. 接触: 每帧每只手 / 脚投到画面上, 离物体 mask 够近、远近也对得上(先估出人和 DA3 深度的整体偏差) -> 第一遍"碰着";
     粗拟合后按三维距离再判一遍, 带迟滞(< 8cm 算碰上, > 20cm 才算松开)。脚只在同一个人的手拿着时算(骑车踩踏板)。
     物体级"拿着" = 有手碰着, 或者从碰着的帧连续下来、物体 mask 一直挨着人的 mask(重建的手可能离得远, 东西还贴着人就没放手)。
  2. 起点: 中心沿 mask 质心的视线, 远近取碰着的手(没人拿时取 DA3 深度 + 偏差); 大小按 mask 外接框和远近估;
     朝向以"拿着它的人"的身体朝向为参照, 24 个直角朝向先在抽出的 <=40 帧、低分辨率上批量初筛, 留 3 个用全部帧粗拟合、再细调。
  3. 损失:
     轮廓       渲染轮廓和 SAM3 mask 的 soft IoU, 真人挡住的像素不算
     握点       每只碰着的手 / 脚在物体上有个握点(物体局部坐标, 每帧一个, 一起优化): 握点在世界里的位置 = 手 / 脚的位置;
                握点贴表面; 握点可以在物体上慢慢滑(重建的人会慢慢漂, 两个人的手距 15 秒里能差 50cm, 刚体两头不能同时钉死)
     身体朝向   拿着时, 物体在"拿它的人"的身体朝向里转得慢(人转身箱子跟着转); 拿它的人只有在手全松开时才换
     不穿插     人体(碰着的手 / 脚附近以外)进物体的深度, 物体内外用 SDF 网格判
     地面       物体最低点不低于地面
     运动       平移加速度: 脱手又离地时 = 重力, 其它时候 = 0; 角加速度 = 0(鲁棒, 接住 / 扔出那一下不会被抹平)
     三向比例   贴近 SAM 3D 网格
输出 <out>/: carry_poses.npz(和第一版同格式, 外加接触 / 阶段), metrics.json, preview.mp4(原图 | 渲染 | 叠加, 绿圈 = 碰着的手 / 脚)

  carryfit/run.sh --r1 <r1> --word tray --oid 0 --mesh <glb> --out carryfit/runs/<名字> [--ref <参考预览 mp4>]
"""
import argparse
import json
import os
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import nvdiffrast.torch as dr
from pycocotools import mask as mu
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SAM3D = os.path.dirname(HERE)
DEV = "cuda"
G = 9.81


# ================================================================ 读数据
def load_inputs(r1, word, oid, RW, mask_path=None):
    da3 = f"{r1}/02_da3"
    cams = json.load(open(f"{da3}/cameras.json"))["cameras"]
    fm = json.load(open(f"{da3}/frame_map.json"))
    k2v, fps = fm["frames"], float(fm.get("fps", 30.0))
    wu = json.load(open(f"{da3}/world_up.json"))
    up = np.array(wu["up"], float)
    up /= np.linalg.norm(up)
    keys = sorted(cams)
    n = len(keys)
    md = np.load(mask_path or f"{r1}/03_masks/{word.replace(' ', '_')}.npz", allow_pickle=True)
    H, W = [int(x) for x in md["hw"]]
    RH = int(round(RW * H / W))
    Ks, w2c = [], []
    for k in keys:
        c = cams[k]
        h, w = c["input_hw"]
        Ks.append(np.diag([W / w, H / h, 1.0]) @ np.array(c["intrinsic_3x3"], float))
        w2c.append(np.array(c["extrinsic_w2c_4x4"], float).reshape(4, 4))
    masks = np.zeros((n, H, W), bool)
    meta = md["meta"]
    kidx = {int(os.path.splitext(k)[0]): i for i, k in enumerate(keys)}   # mask 的 frame = 帧文件名里的号, 不是第几个关键帧
    for i in np.where(meta["oid"] == oid)[0]:
        f = kidx.get(int(meta["frame"][i]))
        if f is None:
            continue
        masks[f] = mu.decode({"counts": md["counts"][i], "size": [H, W]}).astype(bool)
    has = masks.reshape(n, -1).sum(1) >= 50
    # DA3 深度, 缩到工作分辨率
    depth = np.full((n, RH, RW), np.nan, np.float32)
    for i, k in enumerate(keys):
        p = f"{da3}/depth/{os.path.splitext(k)[0]}_depth.npy"
        if os.path.exists(p):
            depth[i] = cv2.resize(np.load(p).astype(np.float32), (RW, RH), interpolation=cv2.INTER_NEAREST)
    pw = np.load(f"{r1}/10_people_world.npz")
    faces = pw["faces"].astype(np.int32)
    vf_of_key = np.array([k2v[k] for k in keys])
    people = {}
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
    pm_dir = f"{r1}/body4d/02_body4d/masks"
    pmask = None
    if os.path.isdir(pm_dir) and len(os.listdir(pm_dir)) >= n:
        pmask = np.stack([cv2.imread(f"{pm_dir}/{i:08d}.png", cv2.IMREAD_UNCHANGED).reshape(H, W, -1).max(-1) > 0
                          for i in range(n)])
    return dict(keys=keys, vf=vf_of_key, fps=fps, frames_dir=fm.get("frames_dir"), video=fm.get("video"),
                K=np.stack(Ks), w2c=np.stack(w2c), up=up, gh=wu.get("ground_h"), masks=masks, has_mask=has,
                H=H, W=W, RW=RW, RH=RH, depth=depth, people=people, faces=faces, pmask=pmask)


def load_mesh(path, n_faces=3000):
    """网格转到自己的定向包围盒系: 居中, x 最长 y 中 z 最短, 最长边归一成 1。另出: 简化网格(拟合时渲染用)、表面采样点、SDF 网格"""
    m = trimesh.load(path, force="mesh", process=False)
    T, ext = trimesh.bounds.oriented_bounds(m)
    order = np.argsort(-ext)
    ext = ext[order]
    P = np.eye(4)[list(order) + [3]]
    if np.linalg.det(P[:3, :3] @ T[:3, :3]) < 0:          # 换轴成了镜像(行列式 -1): 最短轴反向, 变回正常旋转
        P[2] *= -1
    A = np.diag([1 / ext[0]] * 3 + [1.0]) @ P @ T         # 原始网格(glb 场景坐标) -> 局部系
    v = trimesh.transform_points(m.vertices, A)
    full = trimesh.Trimesh(v, m.faces, process=False)
    import open3d as o3d
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(np.asarray(m.faces)))
    if len(m.faces) > n_faces:
        om = om.simplify_quadric_decimation(n_faces)
    sv, sf = np.asarray(om.vertices, np.float32), np.asarray(om.triangles, np.int32)
    pts, _ = trimesh.sample.sample_surface(full, 2000, seed=0)
    # SDF: 局部系 [-0.6, 0.6]^3, 48^3 格; 网格不一定封闭, 符号用多条射线投票
    g = np.linspace(-0.6, 0.6, 48, dtype=np.float32)
    Q = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(np.asarray(m.faces)))))
    sdf = sc.compute_signed_distance(o3d.core.Tensor(Q), nsamples=5).numpy().reshape(48, 48, 48)
    return dict(V=sv, F=sf, Vfull=v.astype(np.float32), Ffull=np.asarray(m.faces, np.int32), A=A,
                pts=pts.astype(np.float32), ext_ratio=(ext / ext[0]).astype(np.float32), sdf=sdf.astype(np.float32))


# ================================================================ 几何小工具
def r6_to_mat(x):
    b1 = F.normalize(x[..., :3], dim=-1)
    a2 = x[..., 3:]
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], -1)     # 列 = 物体局部 x/y/z 在世界里的方向


def to_clip(X, w2c, K, Wr, Hr, near=0.05, far=80.0):
    """世界点 (m, V, 3) -> nvdiffrast clip; 图像第 0 行 = ndc_y -1, 不翻转"""
    Xc = torch.einsum("nij,nvj->nvi", w2c[:, :3, :3], X) + w2c[:, None, :3, 3]
    z = Xc[..., 2].clamp(min=1e-4)
    u = K[:, None, 0, 0] * Xc[..., 0] / z + K[:, None, 0, 2]
    v = K[:, None, 1, 1] * Xc[..., 1] / z + K[:, None, 1, 2]
    zc = Xc[..., 2]
    return torch.stack([(2 * u / Wr - 1) * zc, (2 * v / Hr - 1) * zc,
                        zc * (far + near) / (far - near) - 2 * far * near / (far - near), zc], -1).contiguous()


def project(X, w2c, K):
    """numpy: 世界点 (n, P, 3) -> 像素 (n, P, 2) 和相机深度 (n, P)"""
    Xc = np.einsum("nij,npj->npi", w2c[:, :3, :3], X) + w2c[:, None, :3, 3]
    z = Xc[..., 2]
    uv = np.stack([K[:, None, 0, 0] * Xc[..., 0] / z + K[:, None, 0, 2], K[:, None, 1, 1] * Xc[..., 1] / z + K[:, None, 1, 2]], -1)
    return uv, z


def huber(x, s):
    """伪 Huber: 小的像平方, 大的线性增长、不封顶(连跳几米要很贵; 接住 / 扔出那一两帧的突变代价有限)"""
    return torch.sqrt(1 + (x / s) ** 2) - 1


def rho(x, s):
    """鲁棒损失 x^2/(x^2+s^2): 小误差像平方, 大误差封顶 1"""
    return x ** 2 / (x ** 2 + s ** 2)


def runs_of(b):
    """布尔序列 -> [(起, 止)] 连续 True 段(止不含)"""
    out, i = [], 0
    while i < len(b):
        if b[i]:
            j = i
            while j < len(b) and b[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def clean(b, gap, minlen):
    """接触序列去噪: 补上不超过 gap 帧的缺口, 去掉短于 minlen 帧的段"""
    b = b.copy()
    for a_, b_ in runs_of(~b):
        if a_ > 0 and b_ < len(b) and b_ - a_ <= gap:
            b[a_:b_] = True
    for a_, b_ in runs_of(b):
        if b_ - a_ < minlen:
            b[a_:b_] = False
    return b


def cube_rotations():
    out = []
    import itertools
    for p in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for r, c in enumerate(p):
                M[r, c] = sg[r]
            if np.linalg.det(M) > 0:
                out.append(M)
    return np.array(out)                                   # 24 个


def kabsch(A, B):
    """B ≈ R A + c, 返回 R"""
    a, b = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


# ================================================================ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--word", required=True)
    ap.add_argument("--oid", type=int, required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--masks", default=None, help="物体 mask 文件(默认 <r1>/03_masks/<词>.npz; retrack.py 重跟过的就给它的输出)")
    ap.add_argument("--ref", default=None, help="参考预览视频(同样的关键帧, 3 列), 拼对比用")
    ap.add_argument("--iters_screen", type=int, default=120, help="24 个朝向初筛(抽帧、低分辨率)")
    ap.add_argument("--iters_coarse", type=int, default=150)
    ap.add_argument("--iters_fine", type=int, default=250)
    ap.add_argument("--keep", type=int, default=3, help="初筛后留几个朝向")
    ap.add_argument("--w_sil", type=float, default=1.0, help="轮廓(soft IoU)")
    ap.add_argument("--w_cham", type=float, default=1.0, help="轮廓(二维倒角距离, 细东西靠它)")
    ap.add_argument("--body_rate", type=float, default=3.0, help="身体朝向一项的转速尺度(弧度/秒): 转得比这快才明显罚")
    ap.add_argument("--w_grasp", type=float, default=1.0, help="握点不变(手); 脚是它的 --foot_frac 倍")
    ap.add_argument("--foot_frac", type=float, default=0.3)
    ap.add_argument("--w_surf", type=float, default=0.5, help="握点贴表面")
    ap.add_argument("--w_slip", type=float, default=1.0, help="握点在物体上滑动(重建的人会慢慢漂, 允许握点慢慢滑)")
    ap.add_argument("--slip_v", type=float, default=0.1, help="握点滑动速度的尺度(米/秒)")
    ap.add_argument("--w_pen", type=float, default=10.0, help="不穿插(超过容差的部分)")
    ap.add_argument("--pen_tol", type=float, default=0.03)
    ap.add_argument("--w_gnd", type=float, default=1.0, help="不低于地面")
    ap.add_argument("--w_acc", type=float, default=10.0, help="平移加速度(离手时); 拿着时乘 --held_frac")
    ap.add_argument("--w_rot", type=float, default=1.0, help="角加速度(离手时); 拿着时乘 --held_frac")
    ap.add_argument("--held_frac", type=float, default=1.0)
    ap.add_argument("--w_body", type=float, default=0.5, help="拿着时, 物体相对拿它的人的身体朝向转得慢")
    ap.add_argument("--w_shape", type=float, default=0.1)
    ap.add_argument("--touch_max", type=float, default=0.4, help="手松开后, 物体还贴着人的话最多再算几秒'拿着'")
    ap.add_argument("--w_depth", type=float, default=0.0, help="物体中心远近贴近 mask 里的 DA3 深度(鲁棒, 尺度 0.3m)")
    ap.add_argument("--smooth", choices=["rho", "huber"], default="rho", help="运动平滑: rho 封顶 / huber 线性增长")
    ap.add_argument("--no_preview", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0_all = time.time()
    torch.manual_seed(0)

    D = load_inputs(a.r1, a.word, a.oid, RW=480, mask_path=a.masks)
    n, H, W, RW, RH = len(D["keys"]), D["H"], D["W"], D["RW"], D["RH"]
    box = load_mesh(a.mesh)
    pids = sorted(D["people"])
    nv = D["people"][pids[0]].shape[1]
    parts = np.load(f"{HERE}/parts.npz")
    up, gh = D["up"], D["gh"]
    tsec = (D["vf"] - D["vf"][0]) / D["fps"]                     # 每个关键帧的时间(秒)
    dtm_ = float(np.median(np.diff(tsec)))
    GAP, MINLEN, MERGE = max(2, round(0.3 / dtm_)), max(3, round(0.2 / dtm_)), max(3, round(1.0 / dtm_))
    print(f"[输入] {n} 关键帧, 有 mask {int(D['has_mask'].sum())} 帧, 人 {pids}, 视频 {W}x{H}, fps {D['fps']}, "
          f"关键帧间隔中位 {dtm_:.3f}s, 地面 {gh}", flush=True)

    # ---- 每帧每个部位(人 x 左右手 / 左右脚)的中心
    rng = np.random.default_rng(0)
    PART_KINDS = ["hand_L", "hand_R", "foot_L", "foot_R"]
    meta = [(p, k) for p in pids for k in PART_KINDS]
    sub = {k: rng.choice(parts[k], 60, replace=False) for k in PART_KINDS}
    PC = np.stack([D["people"][p][:, sub[k]].mean(1) for p, k in meta], 1)        # (n, P, 3), 缺人 = NaN
    NP = len(meta)
    is_hand = np.array([k.startswith("hand") for _, k in meta])
    person_of = np.array([pids.index(p) for p, _ in meta])

    def tidy(c):
        """接触矩阵去噪; 脚只在同一个人的手拿着时算"""
        c = c.copy()
        for p in range(NP):
            c[:, p] = clean(c[:, p], GAP, MINLEN)
        for p in np.where(~is_hand)[0]:
            c[:, p] &= c[:, is_hand & (person_of == person_of[p])].any(1)
            c[:, p] = clean(c[:, p], GAP, MINLEN)
        return c

    # ---- 第一遍接触: 画面上离 mask 够近, 远近和 DA3 深度对得上(先估出人和 DA3 深度之间的整体偏差)
    Kr = D["K"] * np.array([[RW / W], [RH / H], [1]])
    uv, zc = project(np.nan_to_num(PC, nan=0.0), D["w2c"], Kr)
    mr = np.stack([cv2.resize(m.astype(np.uint8), (RW, RH), interpolation=cv2.INTER_NEAREST) for m in D["masks"]]).astype(bool)
    cand = []
    for i in range(n):
        if not D["has_mask"][i] or not mr[i].any():
            continue
        dtm = cv2.distanceTransform((~mr[i]).astype(np.uint8), cv2.DIST_L2, 3)
        thr = max(3.0, 0.15 * np.sqrt(mr[i].sum()))
        ys, xs = np.nonzero(mr[i])
        for p in range(NP):
            if np.isnan(PC[i, p, 0]) or zc[i, p] <= 0.05:
                continue
            u, v = uv[i, p]
            if not (0 <= u < RW and 0 <= v < RH) or dtm[int(v), int(u)] > thr:
                continue
            sel = (xs - u) ** 2 + (ys - v) ** 2 <= (thr * 2) ** 2
            zz = D["depth"][i][ys[sel], xs[sel]] if sel.any() else D["depth"][i][ys, xs]
            cand.append((i, p, zc[i, p] - (np.nanmedian(zz) if np.isfinite(zz).any() else np.nan)))
    dzs = np.array([x[2] for x in cand]) if cand else np.zeros(0)
    off = float(np.nanmedian(dzs)) if np.isfinite(dzs).any() else 0.0
    near2d = np.zeros((n, NP), bool)
    for i, p, dz in cand:
        near2d[i, p] = (not np.isfinite(dz)) or abs(dz - off) < max(0.4, 0.2 * zc[i, p])
    cont = tidy(near2d)
    print(f"[接触 1] 人比 DA3 深度整体远 {off:+.2f} m; 拿着的帧 {int(cont[:, is_hand].any(1).sum())}/{n}; "
          + ", ".join(f"{p}{k}:{int(cont[:, j].sum())}" for j, (p, k) in enumerate(meta) if cont[:, j].any()), flush=True)

    # ---- 物体级"拿着": 手碰着的帧, 以及从这些帧连续下来、物体 mask 一直挨着人的 mask 的帧
    #      (重建的手可能离得远, 但物体还贴着人就没放手; 扔出去的东西很快在画面上和人分开)
    touch = np.zeros(n, bool)
    if D["pmask"] is not None:
        pm_t = np.stack([cv2.resize(m.astype(np.uint8), (RW, RH), interpolation=cv2.INTER_NEAREST) for m in D["pmask"]]).astype(bool)
        for i in range(n):
            if mr[i].any():
                touch[i] = (ndimage.binary_dilation(mr[i], iterations=4) & pm_t[i]).any()

    def held_of(c):
        h = c[:, is_hand].any(1)
        out = h.copy()
        for rng_ in (range(n), range(n - 1, -1, -1)):
            on, since = False, None
            for i in rng_:
                if h[i]:
                    on, since = True, tsec[i]
                elif on:
                    on = ((touch[i] and D["has_mask"][i]) or not D["has_mask"][i]) and abs(tsec[i] - since) <= a.touch_max
                out[i] |= on
        return out

    # ---- 起点: 平移(沿 mask 质心视线; 远近 = 碰着的手, 没有就用 DA3 深度 + 偏差), 大小
    pm_r = None
    if D["pmask"] is not None:
        pm_r = np.stack([cv2.resize(m.astype(np.uint8), (RW, RH), interpolation=cv2.INTER_NEAREST) for m in D["pmask"]]).astype(bool)
    zfill = np.full(n, np.nan)
    zda3 = np.full(n, np.nan)                 # 每帧 mask 里 DA3 深度的中位数(已加人和 DA3 的整体偏差), 拟合时当远近的观测
    cxy = np.full((n, 2), np.nan)
    for i in range(n):
        if not D["has_mask"][i] or not mr[i].any():
            continue
        ys, xs = np.nonzero(mr[i])
        cxy[i] = xs.mean(), ys.mean()
        er = ndimage.binary_erosion(mr[i], iterations=1)
        if pm_r is not None:
            er &= ~pm_r[i]
        zz = D["depth"][i][er] if er.sum() >= 10 else D["depth"][i][mr[i]]
        if np.isfinite(zz).any():
            zda3[i] = np.nanmedian(zz) + off
        hc = cont[i] & is_hand
        zfill[i] = zc[i, hc].mean() if hc.any() else zda3[i]
    ok = np.isfinite(zfill) & np.isfinite(cxy[:, 0])
    assert ok.sum() >= 2, "有 mask 的帧太少"
    t_init = np.full((n, 3), np.nan)
    for i in np.where(ok)[0]:
        K = Kr[i]
        pc = np.array([(cxy[i, 0] - K[0, 2]) / K[0, 0] * zfill[i], (cxy[i, 1] - K[1, 2]) / K[1, 1] * zfill[i], zfill[i]])
        t_init[i] = D["w2c"][i][:3, :3].T @ (pc - D["w2c"][i][:3, 3])
    idx = np.where(ok)[0]
    for c in range(3):
        t_init[:, c] = np.interp(tsec, tsec[idx], t_init[idx, c])
    ss = [max(np.ptp(np.nonzero(mr[i])[1]), np.ptp(np.nonzero(mr[i])[0])) * zfill[i] / Kr[i][0, 0] for i in idx]
    s0 = float(np.clip(np.percentile(ss, 80), 0.1, 3.0))

    # ---- 参照朝向: 每帧"拿着它的人"(这一帧手碰着最多的人; 没人碰就沿用上一个)的骨盆朝向, 只留绕天的转角
    mp = np.load(f"{SAM3D}/pipeline/mhr_parts.npz")
    pidx, rest = mp["pelvis_idx"], mp["rest_pelvis"].astype(float)
    yaw_of = {}
    for j, p in enumerate(pids):
        Y = np.full((n, 3, 3), np.nan)
        for i in range(n):
            if np.isnan(D["people"][p][i, 0, 0]):
                continue
            fwd = kabsch(rest, D["people"][p][i][pidx].astype(float)) @ np.array([0, 0, 1.0])
            fh = fwd - (fwd @ up) * up
            fh /= np.linalg.norm(fh) + 1e-9
            Y[i] = np.stack([np.cross(up, fh), up, fh], -1)
        yaw_of[j] = Y

    def holder_frames(c):
        """每帧的拿着它的人(下标, -1 = 没有) 和他的朝向; 没人的帧用相机朝向"""
        hc = np.stack([c[:, is_hand & (person_of == j)].sum(1) for j in range(len(pids))], 1)     # (n, 人)
        tot = hc.sum(0)
        hold = np.full(n, -1)
        last = -1
        for i in range(n):
            if last >= 0 and hc[i, last] > 0:          # 粘性: 当前的人还有手碰着就不换(两人合抬时不来回跳)
                hold[i] = last
            elif hc[i].max() > 0:                      # 换人: 这一帧碰着的人里, 全段碰得最多的那个
                hold[i] = int(np.argmax(np.where(hc[i] > 0, tot, -1)))
            else:
                hold[i] = last
            last = hold[i]
        first = next((h for h in hold if h >= 0), -1)
        hold[hold < 0] = first
        Fr = np.zeros((n, 3, 3))
        for i in range(n):
            Y = yaw_of[hold[i]][i] if hold[i] >= 0 else np.full((3, 3), np.nan)
            if np.isnan(Y[0, 0]):
                fwd = np.linalg.inv(D["w2c"][i])[:3, 2]
                fh = fwd - (fwd @ up) * up
                fh /= np.linalg.norm(fh) + 1e-9
                Y = np.stack([np.cross(up, fh), up, fh], -1)
            Fr[i] = Y
        return hold, Fr

    hold1, Fr = holder_frames(cont)
    hc_count = [int(cont[:, is_hand & (person_of == j)].sum()) for j in range(len(pids))]
    holder = pids[int(np.argmax(hc_count))] if max(hc_count) > 0 else None
    print(f"[起点] 起始大小(最长边) {s0:.2f} m; 手接触帧 {dict(zip(pids, hc_count))}", flush=True)

    # ================================================================ 张量
    T = lambda x, dt=torch.float32: torch.tensor(np.asarray(x), dtype=dt, device=DEV)
    upT = T(up)
    BV, BF = T(box["V"]), T(box["F"], torch.int32)
    BP = T(box["pts"][rng.choice(len(box["pts"]), 600, replace=False)])
    SDF = T(box["sdf"])[None, None]                                      # (1,1,48,48,48), 下标 [x,y,z]
    glctx = dr.RasterizeCudaContext()
    PV = {}
    for p in pids:
        V = D["people"][p].copy()
        V[np.isnan(V[:, 0, 0])] = np.array([0, 0, -100.0])
        PV[p] = T(V)
    PF = T(np.concatenate([D["faces"] + j * nv for j in range(len(pids))]), torch.int32)
    hand_all = np.concatenate([parts["hand_L"], parts["hand_R"]])
    body_ok = np.setdiff1d(np.arange(nv), np.concatenate([hand_all, parts["foot_L"], parts["foot_R"]]))
    body_sub = rng.choice(body_ok, 1200, replace=False)
    ALL = dict(w2c=T(D["w2c"]), has=T(D["has_mask"].astype(np.float32)), PC=T(np.nan_to_num(PC, nan=0.0)),
               Bpts=torch.cat([PV[p][:, body_sub] for p in pids], 1), Fr=T(Fr), zda3=T(np.nan_to_num(zda3, nan=-1.0)), hold=T(hold1, torch.long), t=T(tsec))

    def set_holder(c):
        h, Fr_ = holder_frames(c)
        ALL["Fr"], ALL["hold"] = T(Fr_), T(h, torch.long)

    def stage_tensors(rw, rh, blur):
        kk = T(D["K"] * np.array([[rw / W], [rh / H], [1]]))
        ob = T(np.stack([cv2.resize(m.astype(np.float32), (rw, rh), interpolation=cv2.INTER_AREA) for m in D["masks"]]))
        with torch.no_grad():
            if D["pmask"] is not None:
                oc = T(np.stack([cv2.resize(m.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST) for m in D["pmask"]]))
            else:
                rr, _ = dr.rasterize(glctx, to_clip(torch.cat([PV[p] for p in pids], 1), ALL["w2c"], kk, rw, rh), PF, (rh, rw))
                oc = (rr[..., 3] > 0).float()
            oc = (F.max_pool2d(oc[:, None], 5, 1, 2)[:, 0] > 0).float()
            oc = oc * (ob < 0.5).float()          # 物体 mask 自己的像素不算遮挡(托盘压在手上时两者重叠, 信 SAM3 的物体 mask)
            # 人的网格在这个像素的深度(没有人 = 2): 人只能挡住他身后的东西, 物体在人前面时那块像素要和 mask 对得上
            rr, _ = dr.rasterize(glctx, to_clip(torch.cat([PV[p] for p in pids], 1), ALL["w2c"], kk, rw, rh), PF, (rh, rw))
            pz = torch.where(rr[..., 3] > 0, rr[..., 2], torch.full_like(rr[..., 2], 2.0))
        g = None
        if blur > 0:
            r_ = int(3 * blur)
            x = torch.arange(-r_, r_ + 1, device=DEV, dtype=torch.float32)
            g1 = torch.exp(-x ** 2 / (2 * blur ** 2))
            g1 = g1 / g1.sum()
            g = (g1[:, None] * g1[None, :])[None, None]
            ob = F.conv2d(ob[:, None], g, padding=r_)[:, 0]
        # 二维倒角距离用: mask 的距离图(像素), mask 上的采样点, 每帧的距离尺度
        dts, mpts, mw, spx = [], [], [], []
        for m in D["masks"]:
            mm = cv2.resize(m.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
            dts.append(cv2.distanceTransform(1 - mm, cv2.DIST_L2, 3) if mm.any() else np.zeros((rh, rw), np.float32))
            ys, xs = np.nonzero(mm)
            sel = rng.choice(len(xs), min(150, len(xs)), replace=False) if len(xs) else np.zeros(0, int)
            pts = np.zeros((150, 2), np.float32); w_ = np.zeros(150, np.float32)
            pts[:len(sel)] = np.stack([xs[sel], ys[sel]], -1); w_[:len(sel)] = 1
            mpts.append(pts); mw.append(w_); spx.append(max(2.0, 0.1 * np.sqrt(mm.sum())))
        return dict(K=kk, obs=ob, valid=1 - oc, pz=pz, g=g, rw=rw, rh=rh, dt=T(np.stack(dts))[:, None], mpts=T(np.stack(mpts)),
                    mw=T(np.stack(mw)), spx=T(spx))

    def view(fi, st):
        """取一部分关键帧(初筛时抽帧)"""
        fi_t = torch.tensor(fi, device=DEV)
        v = {k: x[fi_t] for k, x in ALL.items()}
        v.update(K=st["K"][fi_t], obs=st["obs"][fi_t], valid=st["valid"][fi_t], pz=st["pz"][fi_t], dt=st["dt"][fi_t],
                 mpts=st["mpts"][fi_t], mw=st["mw"][fi_t], spx=st["spx"][fi_t], g=st["g"], rw=st["rw"], rh=st["rh"], fi=np.asarray(fi))
        return v

    def segments(c, ts):
        """接触矩阵 -> 握点: 同一部位 1 秒内断开又碰上的算同一次握持, 共用一个握点。返回握点数, 以及 (握点号, 帧, 部位) 表"""
        sf, sp_, sid, info = [], [], [], []
        for p in range(NP):
            last_end, k = None, None
            for s_, e_ in runs_of(c[:, p]):
                if last_end is None or ts[s_] - ts[last_end - 1] > 1.0:
                    k = len(info)
                    info.append([p, s_, e_])
                info[k][2] = e_
                last_end = e_
                for i in range(s_, e_):
                    sf.append(i); sp_.append(p); sid.append(k)
        return info, T(sf, torch.long), T(sp_, torch.long), T(sid, torch.long)

    def solve(v, c, held_np, R0, t0_, logs0, anc0, iters, use_pen, lr=0.01, fix_scale=False):
        """B 个起点一起优化。held_np = 物体级"拿着"(见 held_of)"""
        B, n_ = R0.shape[0], len(v["fi"])
        info, sf, sp_, sid = segments(c, v["t"].cpu().numpy())
        held_f = T(held_np.astype(np.float32))
        post = [(s_, e_) for s_, e_ in runs_of(~held_np) if s_ > 0]   # 脱手后到下次被拿之前
        dt = v["t"][1:] - v["t"][:-1]
        dtm = (dt[1:] + dt[:-1]) / 2
        foot_w = T(np.where(is_hand, 1.0, a.foot_frac).astype(np.float32))
        hand_f = T(c[:, is_hand].any(1).astype(np.float32))           # 手真碰着(不含"贴着人")
        both_held = hand_f[1:] * hand_f[:-1] * (v["hold"][1:] == v["hold"][:-1]).float()   # 身体朝向一项只在手碰着时算; 换手的那一下不算
        r6 = torch.cat([R0[..., 0], R0[..., 1]], -1).clone().requires_grad_(True)
        t = t0_.clone().requires_grad_(True)
        logs = logs0.clone().requires_grad_(True)
        anc = anc0.clone().requires_grad_(True)                       # 每个 (握持, 帧) 一个握点
        same = (sid[1:] == sid[:-1]) if len(sid) > 1 else torch.zeros(0, dtype=torch.bool, device=DEV)
        m1, m2 = torch.where(same)[0], torch.where(same)[0] + 1        # 同一次握持里相邻的两帧
        dt_pair = (v["t"][sf[m2]] - v["t"][sf[m1]]).clamp(min=1e-3)
        opt = torch.optim.Adam([{"params": [r6, t], "lr": lr}, {"params": [anc] if fix_scale else [logs, anc], "lr": 0.005}])
        if use_pen:
            with torch.no_grad():
                dpc = torch.cdist(v["Bpts"], v["PC"]) + (1 - T(c.astype(np.float32)))[:, None, :] * 100
                wpen = ((dpc.amin(-1) - 0.25) / 0.10).clamp(0, 1)                 # 碰着的手 / 脚 25cm 内不算
        for it in range(iters + 1):
            R = r6_to_mat(r6)                                            # (B, n, 3, 3)
            s = logs.exp()                                               # (B, 3)
            Xw = torch.einsum("bnij,vj,bj->bnvi", R, BV, s) + t[:, :, None]
            clip = to_clip(Xw.reshape(B * n_, -1, 3), v["w2c"].repeat(B, 1, 1), v["K"].repeat(B, 1, 1), v["rw"], v["rh"])
            rast, _ = dr.rasterize(glctx, clip, BF, (v["rh"], v["rw"]))
            sil = dr.antialias((rast[..., 3:4] > 0).float(), rast, clip, BF)[..., 0]
            if v["g"] is not None:
                sil = F.conv2d(sil[:, None], v["g"], padding=v["g"].shape[-1] // 2)[:, 0]
            sil = sil.reshape(B, n_, v["rh"], v["rw"])
            obs = v["obs"][None]
            with torch.no_grad():                                        # 物体在人前面的像素不算"被人挡住"
                oz = rast[..., 2].reshape(B, n_, v["rh"], v["rw"])
                front = (rast[..., 3] > 0).reshape(B, n_, v["rh"], v["rw"]) & (v["pz"][None] < 1.5) & (oz < v["pz"][None])
                valid = torch.maximum(v["valid"][None], front.float())
            inter = (sil * obs * valid).sum((2, 3))
            union = ((sil + obs - sil * obs) * valid).sum((2, 3)).clamp(min=1.0)
            l_sil = ((1 - inter / union) * v["has"]).sum(1) / v["has"].sum().clamp(min=1)
            if len(info):
                Aw = torch.einsum("bmij,bmj->bmi", R[:, sf], anc * s[:, None]) + t[:, sf]
                d = (Aw - v["PC"][sf, sp_][None]).norm(dim=-1)
                l_grasp = (rho(d, 0.05) * foot_w[sp_]).mean(1)
                dsurf = torch.cdist(anc * s[:, None], BP[None] * s[:, None]).amin(-1)
                l_surf = rho(F.relu(dsurf - 0.03), 0.03).mean(1)
                slip = ((anc[:, m2] - anc[:, m1]) * s[:, None]).norm(dim=-1) / dt_pair
                l_slip = rho(slip / a.slip_v, 1.0).mean(1) if len(m1) else torch.zeros(B, device=DEV)
            else:
                l_grasp = l_surf = l_slip = torch.zeros(B, device=DEV)
            Pw = torch.einsum("bnij,vj,bj->bnvi", R, BP, s) + t[:, :, None]
            hgt = (Pw @ upT).amin(-1)                                    # (B, n) 物体最低点高度
            # 二维倒角距离: 表面采样点投到画面上要落在 mask 里(正向), mask 上的点要有投影点在附近(反向)。
            # 细东西(滑板车的杆、侧着的托盘)稍微偏一点 IoU 就是 0、没梯度, 距离有
            Xc = torch.einsum("nij,bnvj->bnvi", v["w2c"][:, :3, :3], Pw) + v["w2c"][None, :, None, :3, 3]
            zz_ = Xc[..., 2].clamp(min=0.05)
            uvp = torch.stack([v["K"][None, :, None, 0, 0] * Xc[..., 0] / zz_ + v["K"][None, :, None, 0, 2],
                               v["K"][None, :, None, 1, 1] * Xc[..., 1] / zz_ + v["K"][None, :, None, 1, 2]], -1)   # (B, n, V, 2)
            gxy = torch.stack([uvp[..., 0] / v["rw"] * 2 - 1, uvp[..., 1] / v["rh"] * 2 - 1], -1).reshape(B * n_, 1, -1, 2)
            d_in = F.grid_sample(v["dt"].repeat(B, 1, 1, 1), gxy, align_corners=False, padding_mode="border").reshape(B, n_, -1)
            vis = F.grid_sample(v["valid"][:, None].repeat(B, 1, 1, 1), gxy, align_corners=False, padding_mode="border").reshape(B, n_, -1)
            spx = v["spx"][None, :, None]
            fwd = (rho(d_in / spx, 1.0) * vis).sum(-1) / vis.sum(-1).clamp(min=1)
            d_back = torch.cdist(v["mpts"][None].expand(B, -1, -1, -1), uvp).amin(-1)                       # (B, n, 150)
            bwd = (rho(d_back / spx, 1.0) * v["mw"][None]).sum(-1) / v["mw"][None].sum(-1).clamp(min=1)
            l_cham = ((fwd + bwd) * v["has"][None]).sum(1) / v["has"].sum().clamp(min=1)
            l_gnd = huber(F.relu(gh - 0.01 - hgt), 0.03).mean(1) if gh is not None else torch.zeros(B, device=DEV)
            with torch.no_grad():
                free_air = torch.zeros_like(hgt)
                if gh is not None:
                    airb = ((hgt - gh) > 0.05).float()
                    for s_, e_ in post:
                        free_air[:, s_:e_] = torch.cumprod(airb[:, s_:e_], 1)
            vel = (t[:, 1:] - t[:, :-1]) / dt[None, :, None]
            acc = (vel[:, 1:] - vel[:, :-1]) / dtm[None, :, None]
            target = -G * upT * free_air[:, 1:-1, None]
            wmov = held_f[1:-1] * a.held_frac + (1 - held_f[1:-1])
            sm = huber if a.smooth == "huber" else rho
            # 没人拿着时, 0.5g 以内的加速度不罚(落地后滑行、弹跳要减速); 拿着时不留余量(手说了算, 要顺)
            dz = (1 - held_f[1:-1]) * 0.5
            l_acc = (sm(F.relu((acc - target).norm(dim=-1) / G - dz), 1.0) * wmov).mean(1)
            Qr = R[:, :-1].transpose(-1, -2) @ R[:, 1:]
            om = torch.stack([Qr[..., 2, 1] - Qr[..., 1, 2], Qr[..., 0, 2] - Qr[..., 2, 0], Qr[..., 1, 0] - Qr[..., 0, 1]], -1) / 2
            om = om / dt[None, :, None]
            alpha = (om[:, 1:] - om[:, :-1]) / dtm[None, :, None]
            l_rot = (sm(alpha.norm(dim=-1) / 30.0, 1.0) * wmov).mean(1)
            # 物体中心要在相机前面(跑到相机后面就渲染不出来, 轮廓项没梯度, 拉不回来)
            zc_ = torch.einsum("nj,bnj->bn", v["w2c"][:, 2, :3], t) + v["w2c"][None, :, 2, 3]
            l_cam = (F.relu(0.3 - zc_) ** 2).mean(1) * 100
            okz = (v["zda3"] > 0).float()
            l_depth = (rho((zc_ - v["zda3"][None]) / 0.3, 1.0) * okz).sum(1) / okz.sum().clamp(min=1)
            # 拿着时: 物体在"拿它的人的身体朝向"里的姿态变化慢
            Qb = v["Fr"].transpose(-1, -2)[None] @ R
            rate = (Qb[:, 1:] - Qb[:, :-1]).norm(dim=(-1, -2)) / dt[None]
            l_body = (rho(rate / a.body_rate, 1.0) * both_held).sum(1) / both_held.sum().clamp(min=1)
            l_shape = ((logs - logs.mean(1, keepdim=True)) ** 2).sum(1)
            if use_pen:
                q = torch.einsum("bnji,bnvj->bnvi", R, v["Bpts"][None] - t[:, :, None]) / s[:, None, None]
                sd = F.grid_sample(SDF.expand(B, -1, -1, -1, -1), (q / 0.6).flip(-1).reshape(B, 1, 1, -1, 3), align_corners=True,
                                   padding_mode="border")[:, 0, 0, 0].reshape(B, n_, -1) * s.mean(1)[:, None, None]
                l_pen = (wpen[None] * F.relu(-sd - a.pen_tol) ** 2).sum(-1).mean(1)
            else:
                l_pen = torch.zeros(B, device=DEV)
            per = (l_cam + a.w_depth * l_depth + a.w_sil * l_sil + a.w_cham * l_cham + a.w_grasp * l_grasp + a.w_surf * l_surf + a.w_slip * l_slip + a.w_gnd * l_gnd + a.w_acc * l_acc + a.w_rot * l_rot
                   + a.w_body * l_body + a.w_shape * l_shape + a.w_pen * l_pen)
            if it == iters:
                break
            opt.zero_grad()
            per.sum().backward()
            opt.step()
        terms = dict(depth=l_depth, sil=l_sil, cham=l_cham, grasp=l_grasp, surf=l_surf, slip=l_slip, gnd=l_gnd, acc=l_acc, rot=l_rot, body=l_body, shape=l_shape, pen=l_pen)
        return dict(R=R.detach(), t=t.detach(), logs=logs.detach(), loss=per.detach(),
                    terms={k: x.detach() for k, x in terms.items()}, info=info)

    def init_anchors(v, c, R, t, s):
        """握点起点: 部位在物体局部系里的平均位置, 再吸到最近的表面点"""
        info, sf, sp_, sid = segments(c, v["t"].cpu().numpy())
        B = R.shape[0]
        if not len(info):
            return torch.zeros(B, 0, 3, device=DEV)
        loc = torch.einsum("bmji,bmj->bmi", R[:, sf], v["PC"][sf, sp_][None] - t[:, sf]) / s[:, None]
        acc_ = torch.zeros(B, len(info), 3, device=DEV).index_add_(1, sid, loc)
        cnt = torch.zeros(len(info), device=DEV).index_add_(0, sid, torch.ones_like(sid, dtype=torch.float32))
        u = acc_ / cnt[None, :, None]
        return BP[torch.cdist(u * s[:, None], BP[None] * s[:, None]).argmin(-1)][:, sid]       # 每帧一个, 起点 = 这次握持的平均

    def to_all(R, t, fi):
        """抽帧上的解插值到所有关键帧(按时间)"""
        ts = tsec[fi]
        Rn, tn = R.cpu().numpy(), t.cpu().numpy()
        Ro = np.zeros((R.shape[0], n, 3, 3)); to = np.zeros((R.shape[0], n, 3))
        for c_ in range(3):
            to[..., c_] = np.stack([np.interp(tsec, ts, tn[b, :, c_]) for b in range(R.shape[0])])
            for r_ in range(3):
                Ro[..., r_, c_] = np.stack([np.interp(tsec, ts, Rn[b, :, r_, c_]) for b in range(R.shape[0])])
        Ro = r6_to_mat(T(np.concatenate([Ro[..., 0], Ro[..., 1]], -1)))
        return Ro, T(to)

    ST_S = stage_tensors(160, int(round(160 * H / W)), 3.0)
    ST_C = stage_tensors(240, int(round(240 * H / W)), 2.0)
    ST_F = stage_tensors(RW, RH, 0.0)
    allf = np.arange(n)

    # ---- 初筛: 24 个直角朝向, 抽 <=40 帧, 低分辨率
    t_c = time.time()
    fs = np.unique(np.linspace(0, n - 1, min(n, 40)).round().astype(int))
    vs = view(fs, ST_S)
    H24 = T(cube_rotations())
    B = len(H24)
    R0 = torch.einsum("nij,bjk->bnik", vs["Fr"], H24)
    t0s = T(t_init[fs])[None].expand(B, -1, -1).contiguous()
    logs0 = torch.full((B, 3), float(np.log(s0)), device=DEV)
    cs = cont[fs]
    rs = solve(vs, cs, held_of(cont)[fs], R0, t0s, logs0, init_anchors(vs, cs, R0, t0s, logs0.exp()), a.iters_screen, use_pen=False)
    order = torch.argsort(rs["loss"]).cpu().numpy()[:a.keep]
    print(f"[初筛] {time.time() - t_c:.1f}s, 24 个朝向 损失 {rs['loss'].min():.3f}..{rs['loss'].max():.3f}, "
          f"留 {[round(float(rs['loss'][j]), 3) for j in order]}", flush=True)

    # ---- 粗拟合: 留下的几个, 所有帧
    t_c = time.time()
    va = view(allf, ST_C)
    keep = torch.tensor(order, device=DEV)
    Rk, tk = to_all(rs["R"][keep], rs["t"][keep], fs)
    lk = rs["logs"][keep]
    rc = solve(va, cont, held_of(cont), Rk, tk, lk, init_anchors(va, cont, Rk, tk, lk.exp()), a.iters_coarse, use_pen=False)
    b0 = int(torch.argmin(rc["loss"]))
    print(f"[粗拟合] {time.time() - t_c:.1f}s, 损失 {[round(float(x), 3) for x in rc['loss']]}; 最好那个的各项 "
          + str({k: round(float(x[int(torch.argmin(rc['loss']))]), 3) for k, x in rc['terms'].items()}), flush=True)

    # ---- 第二遍接触: 按粗拟合最好的那个, 用三维距离
    with torch.no_grad():
        Pw = torch.einsum("nij,vj->nvi", rc["R"][b0], BP * rc["logs"][b0].exp()) + rc["t"][b0][:, None]
        d3 = torch.cdist(ALL["PC"], Pw).amin(-1).cpu().numpy()
    d3[np.isnan(PC[..., 0])] = np.inf
    # 迟滞: 离得 < 8cm(脚 6cm)或第一遍判了接触且 < 15cm 才算碰上; 碰上之后离开 20cm 以上才算松手(重建的手有几厘米到十几厘米的误差)
    enter = (d3 < np.where(is_hand, 0.08, 0.06)[None]) | (cont & (d3 < 0.15))
    stay = d3 < 0.20
    c2 = np.zeros_like(enter)
    for p in range(NP):
        on = False
        for i in range(n):
            on = enter[i, p] or (on and stay[i, p])
            c2[i, p] = on
        # 反着再扫一遍: 碰上之前那几帧如果一直很近, 也算(抬起的瞬间)
        on = False
        for i in range(n - 1, -1, -1):
            on = c2[i, p] or (on and stay[i, p])
            c2[i, p] = on
    cont2 = tidy(c2)
    print(f"[接触 2] 拿着的帧 {int(cont2[:, is_hand].any(1).sum())}/{n}; "
          + ", ".join(f"{p}{k}:{int(cont2[:, j].sum())}" for j, (p, k) in enumerate(meta) if cont2[:, j].any()), flush=True)

    # ---- 细调
    set_holder(cont2)
    t_f = time.time()
    vf_ = view(allf, ST_F)
    rf = solve(vf_, cont2, held_of(cont2), rc["R"], rc["t"], rc["logs"], init_anchors(vf_, cont2, rc["R"], rc["t"], rc["logs"].exp()),
               a.iters_fine, use_pen=True, lr=0.005)
    bi = int(torch.argmin(rf["loss"]))
    print(f"[细调] {time.time() - t_f:.1f}s, 损失 {[round(float(x), 3) for x in rf['loss']]} -> 选第 {bi} 个", flush=True)
    R_s, t_s, s_s = rf["R"][bi], rf["t"][bi], rf["logs"][bi].exp()
    terms = {k: round(float(x[bi]), 4) for k, x in rf["terms"].items()}

    # ---- 评估用: 960 宽, 物体单独渲, 真人的像素(物体 mask 以外)不算
    w2c = ALL["w2c"]
    EW, EH = 960, int(round(960 * H / W))
    Ke = T(D["K"] * np.array([[EW / W], [EH / H], [1]]))
    obs_e = np.stack([cv2.resize(m.astype(np.uint8), (EW, EH), interpolation=cv2.INTER_NEAREST) for m in D["masks"]]).astype(bool)
    pm_e = None
    if D["pmask"] is not None:
        pm_e = np.stack([cv2.resize(m.astype(np.uint8), (EW, EH), interpolation=cv2.INTER_NEAREST) for m in D["pmask"]]).astype(bool)
        pm_e = F.max_pool2d(torch.tensor(pm_e[:, None], dtype=torch.float32), 5, 1, 2)[:, 0].numpy() > 0
        pm_e &= ~obs_e

    def iou_of(R, t, s):
        with torch.no_grad():
            Xw = torch.einsum("nij,vj->nvi", R, BV * s) + t[:, None]
            rb, _ = dr.rasterize(glctx, to_clip(Xw, w2c, Ke, EW, EH), BF, (EH, EW))
            silb = (rb[..., 3] > 0).cpu().numpy()
        out = np.full(n, np.nan)
        for i in range(n):
            if D["has_mask"][i]:
                vv = ~pm_e[i] if pm_e is not None else np.ones_like(obs_e[i])
                out[i] = (silb[i] & obs_e[i] & vv).sum() / max(((silb[i] | obs_e[i]) & vv).sum(), 1)
        return out

    # ---- 修补: 轮廓连续对不上的片段(放下 / 扔出 / 翻倒之后, 朝向和拿着时不一样, 局部优化转不过去),
    #      只拿这一段按相机方向重试 24 个直角朝向, 轮廓明显更好就换上, 最后整体再细调一遍抹平接缝
    t_r = time.time()
    iou_now = iou_of(R_s, t_s, s_s)
    bad = clean(D["has_mask"] & (np.nan_to_num(iou_now, nan=1.0) < 0.4), 3, 4)
    Fcam = np.zeros((n, 3, 3))
    for i in range(n):
        fwd = np.linalg.inv(D["w2c"][i])[:3, 2]
        fh = fwd - (fwd @ up) * up
        fh /= np.linalg.norm(fh) + 1e-9
        Fcam[i] = np.stack([np.cross(up, fh), up, fh], -1)
    fixed = []
    heldA = held_of(cont2)
    pieces = [(a_, min(a_ + 34, e_)) for s_, e_ in runs_of(bad) for a_ in range(s_, e_, 34)]   # 长片段切成 <=40 帧一块(24 个朝向一起渲, 显存)
    for s_, e_ in pieces:
        fi = np.arange(max(0, s_ - 3), min(n, e_ + 3))
        vw = view(fi, ST_C)
        cw = cont2[fi]
        R0w = torch.einsum("nij,bjk->bnik", T(Fcam[fi]), H24)
        l0w = torch.log(s_s)[None].expand(len(H24), -1).contiguous()
        best_ = None
        for t_src in (t_s[fi], T(t_init[fi])):            # 位置起点两种: 现在的解 / 一开始按 mask 质心视线 + 远近估的
            t0w = t_src[None].expand(len(H24), -1, -1).contiguous()
            r_ = solve(vw, cw, heldA[fi], R0w, t0w, l0w, init_anchors(vw, cw, R0w, t0w, l0w.exp()), 120, use_pen=False, fix_scale=True)
            sc0_ = r_["terms"]["sil"] + r_["terms"]["cham"]
            if best_ is None or float(sc0_.min()) < float(best_[1].min()):
                best_ = (r_, sc0_)
        rw_, sc_ = best_
        Rc, tc, lc = R_s[fi][None], t_s[fi][None], torch.log(s_s)[None]
        cur = solve(vw, cw, heldA[fi], Rc, tc, lc, init_anchors(vw, cw, Rc, tc, lc.exp()), 0, use_pen=False, fix_scale=True)
        c0 = float(cur["terms"]["sil"][0] + cur["terms"]["cham"][0])
        j = int(torch.argmin(sc_))
        print(f"    修补 {s_}-{e_ - 1}: 现在 {c0:.3f}, 24 个朝向 {float(sc_.min()):.3f}..{float(sc_.max()):.3f}", flush=True)
        if os.environ.get("CF_DEBUG_REPAIR"):
            np.savez(f"{os.environ['CF_DEBUG_REPAIR']}_{s_}.npz", fi=fi, R=rw_["R"][j].cpu().numpy(), t=rw_["t"][j].cpu().numpy(),
                     s=s_s.cpu().numpy(), terms={k: float(x[j]) for k, x in rw_["terms"].items()})
        if float(sc_[j]) < c0 - 0.05:
            R_s, t_s = R_s.clone(), t_s.clone()
            R_s[torch.tensor(fi, device=DEV)] = rw_["R"][j]
            t_s[torch.tensor(fi, device=DEV)] = rw_["t"][j]
            fixed.append(f"{s_}-{e_ - 1}: {c0:.2f}->{float(sc_[j]):.2f}")
    if fixed:
        vf_ = view(allf, ST_F)
        rr = solve(vf_, cont2, heldA, R_s[None], t_s[None], torch.log(s_s)[None],
                   init_anchors(vf_, cont2, R_s[None], t_s[None], s_s[None]), 150, use_pen=True, lr=0.003)
        R_s, t_s, s_s = rr["R"][0], rr["t"][0], rr["logs"][0].exp()
        terms = {k: round(float(x[0]), 4) for k, x in rr["terms"].items()}
    print(f"[修补] {time.time() - t_r:.1f}s, 对不上的片段 {[f'{s_}-{e_ - 1}' for s_, e_ in runs_of(bad)]}, 换上 {fixed}", flush=True)
    t_fit = time.time() - t0_all

    # ================================================================ 评估
    with torch.no_grad():
        Pw = torch.einsum("nij,vj->nvi", R_s, BP * s_s) + t_s[:, None]
        dpart = torch.cdist(ALL["PC"], Pw).amin(-1).cpu().numpy() * 100
        q = torch.einsum("nji,nvj->nvi", R_s, ALL["Bpts"] - t_s[:, None]) / s_s
        sd = F.grid_sample(SDF, (q / 0.6).flip(-1).reshape(1, 1, 1, -1, 3), align_corners=True, padding_mode="border").reshape(n, -1)
        dpc = torch.cdist(ALL["Bpts"], ALL["PC"]) + (1 - T(cont2.astype(np.float32)))[:, None, :] * 100
        pen = ((-sd * s_s.mean()).clamp(min=0) * (dpc.amin(-1) > 0.25)).amax(1).cpu().numpy() * 100
        hgt = ((Pw @ upT).amin(-1) - (gh if gh is not None else 0)).cpu().numpy() * 100
    iou = iou_of(R_s, t_s, s_s)
    heldF = held_of(cont2)
    phase = np.where(heldF, "拿着", np.where(hgt > 5, "空中", "放着"))
    gd = [dpart[i, p] for i in range(n) for p in range(NP) if cont2[i, p] and is_hand[p]]
    fd = [dpart[i, p] for i in range(n) for p in range(NP) if cont2[i, p] and not is_hand[p]]
    med = lambda x: round(float(np.nanmedian(x)), 3) if np.isfinite(x).any() else None
    met = dict(
        用时秒=round(t_fit, 1), 物体尺寸米=[round(float(x) * float(y), 3) for x, y in zip(box["ext_ratio"], s_s)],
        拿着它的人=holder, 阶段={ph: int((phase == ph).sum()) for ph in ["拿着", "空中", "放着"]},
        阶段时间线="".join({"拿着": "H", "空中": "A", "放着": "."}[p] for p in phase),
        握点=[dict(部位=f"{meta[p][0]}{meta[p][1]}", 关键帧=[int(s_), int(e_ - 1)]) for p, s_, e_ in rf["info"]],
        轮廓IoU_真人以外_中位=med(iou),
        轮廓IoU_分阶段={ph: med(iou[phase == ph]) for ph in ["拿着", "空中", "放着"]},
        轮廓IoU_按四段={f"关键帧{s_}-{e_ - 1}": med(iou[s_:e_]) for s_, e_ in zip(np.linspace(0, n, 5).astype(int)[:-1], np.linspace(0, n, 5).astype(int)[1:])},
        碰着的手到物体表面_cm_中位=round(float(np.median(gd)), 1) if gd else None,
        碰着的脚到物体表面_cm_中位=round(float(np.median(fd)), 1) if fd else None,
        人穿进物体最深_cm_中位=round(float(np.median(pen)), 1), 人穿进物体最深_cm_最大=round(float(np.max(pen)), 1),
        物体最低点离地_cm_最小=round(float(np.min(hgt)), 1) if gh is not None else None,
        IoU低于05的关键帧=[int(i) for i in np.where(iou < 0.5)[0]],
        最终各项损失=terms,
    )
    json.dump(met, open(f"{a.out}/metrics.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in met.items() if k != "握点"}, ensure_ascii=False))

    M = np.tile(np.eye(4), (n, 1, 1))
    M[:, :3, :3] = R_s.cpu().numpy() * s_s.cpu().numpy()[None, None, :]
    M[:, :3, 3] = t_s.cpu().numpy()
    np.savez(f"{a.out}/carry_poses.npz", keys=np.array(D["keys"]), vf=D["vf"], local_to_world=M, scale=s_s.cpu().numpy(),
             mesh=a.mesh, mesh_to_local=box["A"], box_V=box["Vfull"], box_F=box["Ffull"], iou=iou, part_dist_cm=dpart,
             pen_cm=pen, contact=cont2, parts=np.array([f"{p}:{k}" for p, k in meta]), phase=phase)
    if not a.no_preview:
        render_preview(a, D, pids, box, M, PC, cont2, phase, iou)
    print("总用时", round(time.time() - t0_all, 1), "s")


def frame_image(D, i, k):
    p = os.path.join(D["frames_dir"] or "", k)
    if os.path.exists(p):
        return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    cap = cv2.VideoCapture(D["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(D["vf"][i]))
    ok, im = cap.read()
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if ok else np.zeros((D["H"], D["W"], 3), np.uint8)


def render_preview(a, D, pids, box, M, PC, cont, phase, iou):
    import imageio
    import pyrender
    TW = 640
    TH = int(round(TW * D["H"] / D["W"]))
    TH -= TH % 2
    r = pyrender.OffscreenRenderer(TW, TH)
    pal = [(230, 120, 40), (40, 160, 230), (120, 200, 90), (200, 90, 200), (220, 200, 60), (90, 200, 200), (160, 160, 160)]
    COL = {p: pal[j % len(pal)] for j, p in enumerate(pids)}
    fps = int(np.clip(round(D["fps"] * len(D["keys"]) / max(1, D["vf"][-1] - D["vf"][0] + 1)), 4, 30))   # 按真实速度播
    wr = imageio.get_writer(f"{a.out}/preview.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=1)
    ref = imageio.get_reader(a.ref) if a.ref else None
    wc = imageio.get_writer(f"{a.out}/compare.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=1) if a.ref else None
    Kt = D["K"] * np.array([[TW / D["W"]], [TH / D["H"]], [1]])
    uvp, zp = project(np.nan_to_num(PC, nan=0.0), D["w2c"], Kt)
    for i, k in enumerate(D["keys"]):
        sc = pyrender.Scene(bg_color=[0.96, 0.96, 0.96, 1.0], ambient_light=[0.35, 0.35, 0.35])
        K = Kt[i]
        cam = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=0.05, zfar=80)
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
        bm = trimesh.Trimesh(trimesh.transform_points(box["Vfull"], M[i]), box["Ffull"], process=False)
        bm.visual.vertex_colors = np.array([205, 170, 120, 255], np.uint8)
        sc.add(pyrender.Mesh.from_trimesh(bm, smooth=False))
        col, _ = r.render(sc)
        real = cv2.resize(frame_image(D, i, k), (TW, TH))
        ov = (0.5 * real + 0.5 * col).astype(np.uint8)
        for p in np.where(cont[i])[0]:
            if zp[i, p] > 0:
                cv2.circle(ov, tuple(int(x) for x in uvp[i, p]), 6, (0, 255, 0), 2)
        row = np.hstack([real, col, ov])
        ph = {"拿着": "held", "空中": "in air", "放着": "resting"}[phase[i]]
        cv2.putText(row, f"carryfit  {k}  frame {D['vf'][i]}  {ph}  IoU {iou[i]:.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        wr.append_data(row)
        if wc is not None:
            try:
                rf = ref.get_data(i)
            except IndexError:
                rf = np.zeros_like(row)
            rf = cv2.resize(rf, (row.shape[1], row.shape[0]))
            cv2.putText(rf, "reference", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 80), 2)
            wc.append_data(np.vstack([row, rf]))
    wr.close()
    if wc is not None:
        wc.close()


if __name__ == "__main__":
    main()
