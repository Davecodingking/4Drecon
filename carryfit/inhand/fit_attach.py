"""carryfit 第三版(挂手版): 拿着的时候, 物体是"握持系"(hand.py)的子节点。

第二版里物体是世界里每帧自由的刚体, 手只是一个能滑的握点 -> 手到表面 2 cm、IoU 也好看, 但物体在手里乱转乱滑
(metrics.py 量出来"在手上的帧占比"四个案例全是 0)。这里反过来:

    物体位姿(帧 i) = 握持系(人, 手, 帧 i) · G(这一段握持, 固定) [· 手内自由度(帧 i)]

每段握持只求一个 6 自由度的 G, 这一段所有帧的轮廓一起投票 —— 模糊帧、被挡的帧不单独决定什么, 被 G 带着走。
哪只手、哪几帧拿着: 掌心投到画面上挨着物体 mask 的连续帧算一段"候选握持"(或者 --grip 由人指定);
候选对不对由拟合结果验: 物体真跟着这只手走, 挂上去轮廓才对得上。

重建的手腕不一定准(托盘: 手臂不动只求 G, IoU 0.86, 和第二版逐帧自由拟合的 0.88 持平; 雨伞: 0.45 对 0.66, 不够)。
所以每帧再给手腕一个小修正, 让物体的轮廓反过来修手:
    手腕'(帧 i) = [R_腕 · exp(dR_i),  t_腕 + 相机系里的 dT_i]
  dR 小而且随时间变得慢; dT 在画面平面里几乎不许动(手在画面上的位置是可信的), 沿视线方向可以动(远近不可信)。
  物体永远刚性地挂在修正后的手上, 所以"在手上"是构造出来的; 代价是手偏离原来的重建, 偏了多少如实报出来。
  --frozen 关掉修正(阶段 A, 用来量"手腕本身够不够准")。
修正后的手腕是"目标", 由 arm_ik.py 用手臂的关节参数实现(改的是这个人的 MHR 参数, 人的网格跟着变)。
多只手同时拿(两手端托盘): 每只手各有自己的 G, 各自"提议"一个物体位姿, 联合阶段要求这些提议一致 —— 闭链靠这一项。

  carryfit/run.sh carryfit/inhand/fit_attach.py --r1 <r1> --word umbrella --oid 0 --mesh <glb> --masks <重跟的 mask.npz> \
      --out <目录> [--init_poses <第二版的 carry_poses.npz>] [--grip '[{"人":"p1","手":"hand_L","关键帧":[0,150]}]']
输出: carry_poses.npz(和第二版同格式, 多了 attach_parent), wrist_targets.npz(每段握持修正后的手腕, 给 arm_ik.py),
      attach.json(每段握持的结果和手腕修正量)。之后: arm_ik.py -> metrics.py --people_params 出指标和四格视频。
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fit as v2  # noqa: E402   第二版: 读数据、网格、渲染小工具
import hand as hd  # noqa: E402

DEV = "cuda"
T = lambda x, dt=torch.float32: torch.tensor(np.asarray(x), dtype=dt, device=DEV)


# ================================================================ 每个分辨率一份的观测
def make_stage(D, PV, PF, glctx, rw, rh, blur, rng):
    H, W, n = D["H"], D["W"], len(D["keys"])
    kk = T(D["K"] * np.array([[rw / W], [rh / H], [1]]))
    w2c = T(D["w2c"])
    ob = T(np.stack([cv2.resize(m.astype(np.float32), (rw, rh), interpolation=cv2.INTER_AREA) for m in D["masks"]]))
    with torch.no_grad():
        rr = torch.cat([dr.rasterize(glctx, v2.to_clip(PV[i:i + 16], w2c[i:i + 16], kk[i:i + 16], rw, rh), PF, (rh, rw))[0]
                        for i in range(0, n, 16)])
        if D["pmask"] is not None:
            oc = T(np.stack([cv2.resize(m.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST) for m in D["pmask"]]))
        else:
            oc = (rr[..., 3] > 0).float()
        oc = (F.max_pool2d(oc[:, None], 5, 1, 2)[:, 0] > 0).float() * (ob < 0.5).float()
        pz = torch.where(rr[..., 3] > 0, rr[..., 2], torch.full_like(rr[..., 2], 2.0))
    g = None
    if blur > 0:
        r_ = int(3 * blur)
        x = torch.arange(-r_, r_ + 1, device=DEV, dtype=torch.float32)
        g1 = torch.exp(-x ** 2 / (2 * blur ** 2))
        g1 = g1 / g1.sum()
        g = (g1[:, None] * g1[None, :])[None, None]
        ob = F.conv2d(ob[:, None], g, padding=r_)[:, 0]
    dts, mpts, mw, spx = [], [], [], []
    for m in D["masks"]:
        mm = cv2.resize(m.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
        dts.append(cv2.distanceTransform(1 - mm, cv2.DIST_L2, 3) if mm.any() else np.zeros((rh, rw), np.float32))
        ys, xs = np.nonzero(mm)
        sel = rng.choice(len(xs), min(150, len(xs)), replace=False) if len(xs) else np.zeros(0, int)
        pts = np.zeros((150, 2), np.float32)
        w_ = np.zeros(150, np.float32)
        pts[:len(sel)] = np.stack([xs[sel], ys[sel]], -1)
        w_[:len(sel)] = 1
        mpts.append(pts); mw.append(w_); spx.append(max(2.0, 0.1 * np.sqrt(mm.sum())))
    return dict(K=kk, w2c=w2c, obs=ob, valid=1 - oc, pz=pz, g=g, rw=rw, rh=rh, dt=T(np.stack(dts))[:, None],
                mpts=T(np.stack(mpts)), mw=T(np.stack(mw)), spx=T(spx))


def sub(st, fi):
    fi_t = torch.tensor(np.asarray(fi), device=DEV)
    v = {k: (x[fi_t] if torch.is_tensor(x) and x.dim() > 0 and k != "g" else x) for k, x in st.items()}
    return v


def image_terms(R, t, s, v, box, glctx):
    """R (B,n,3,3), t (B,n,3), s (B,3) -> 每帧的 轮廓项 (B,n)、二维倒角项 (B,n)、"在相机前面"项 (B,n)。和第二版同一套算法。"""
    B, n_ = R.shape[:2]
    rw, rh = v["rw"], v["rh"]
    Xw = torch.einsum("bnij,vj,bj->bnvi", R, box["BV"], s) + t[:, :, None]
    clip = v2.to_clip(Xw.reshape(B * n_, -1, 3), v["w2c"].repeat(B, 1, 1), v["K"].repeat(B, 1, 1), rw, rh)
    rast, _ = dr.rasterize(glctx, clip, box["BF"], (rh, rw))
    sil = dr.antialias((rast[..., 3:4] > 0).float(), rast, clip, box["BF"])[..., 0]
    if v["g"] is not None:
        sil = F.conv2d(sil[:, None], v["g"], padding=v["g"].shape[-1] // 2)[:, 0]
    sil = sil.reshape(B, n_, rh, rw)
    obs = v["obs"][None]
    with torch.no_grad():
        oz = rast[..., 2].reshape(B, n_, rh, rw)
        front = (rast[..., 3] > 0).reshape(B, n_, rh, rw) & (v["pz"][None] < 1.5) & (oz < v["pz"][None])
        valid = torch.maximum(v["valid"][None], front.float())
    inter = (sil * obs * valid).sum((2, 3))
    union = ((sil + obs - sil * obs) * valid).sum((2, 3)).clamp(min=1.0)
    l_sil = 1 - inter / union
    Pw = torch.einsum("bnij,vj,bj->bnvi", R, box["BP"], s) + t[:, :, None]
    Xc = torch.einsum("nij,bnvj->bnvi", v["w2c"][:, :3, :3], Pw) + v["w2c"][None, :, None, :3, 3]
    zz = Xc[..., 2].clamp(min=0.05)
    uvp = torch.stack([v["K"][None, :, None, 0, 0] * Xc[..., 0] / zz + v["K"][None, :, None, 0, 2],
                       v["K"][None, :, None, 1, 1] * Xc[..., 1] / zz + v["K"][None, :, None, 1, 2]], -1)
    gxy = torch.stack([uvp[..., 0] / rw * 2 - 1, uvp[..., 1] / rh * 2 - 1], -1).reshape(B * n_, 1, -1, 2)
    d_in = F.grid_sample(v["dt"].repeat(B, 1, 1, 1), gxy, align_corners=False, padding_mode="border").reshape(B, n_, -1)
    vis = F.grid_sample(v["valid"][:, None].repeat(B, 1, 1, 1), gxy, align_corners=False, padding_mode="border").reshape(B, n_, -1)
    spx = v["spx"][None, :, None]
    fwd = (v2.rho(d_in / spx, 1.0) * vis).sum(-1) / vis.sum(-1).clamp(min=1)
    d_back = torch.cdist(v["mpts"][None].expand(B, -1, -1, -1), uvp).amin(-1)
    bwd = (v2.rho(d_back / spx, 1.0) * v["mw"][None]).sum(-1) / v["mw"][None].sum(-1).clamp(min=1)
    zc = torch.einsum("nj,bnj->bn", v["w2c"][:, 2, :3], t) + v["w2c"][None, :, 2, 3]
    return l_sil, fwd + bwd, F.relu(0.3 - zc) ** 2 * 100


# ================================================================ 把人对回画面
def voc_index(bgr):
    """body4d 的人物 mask 是调色板 PNG(VOC 调色板, 编号 = 跟踪编号), cv2 读出来是 BGR -> 还原成编号图"""
    pal = np.zeros((256, 3), np.int64)
    for i in range(256):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j); g |= ((c >> 1) & 1) << (7 - j); b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        pal[i] = (b, g, r)
    lut = {int(pal[i, 0] * 65536 + pal[i, 1] * 256 + pal[i, 2]): i for i in range(255, -1, -1)}
    k = bgr[..., 0].astype(np.int64) * 65536 + bgr[..., 1].astype(np.int64) * 256 + bgr[..., 2]
    out = np.zeros(k.shape, np.uint8)
    for kk in np.unique(k):
        out[k == kk] = lut.get(int(kk), 255)
    return out


def anchor_person(Vw, faces, inst, other, D, glctx, tsec, rw=200, iters=160):
    """世界系里摆好的人(修过脚、平滑过、挪过)投回画面常常和画面上的人差十几到几十厘米 —— 手就不在画面上该在的地方。
    给这个人每帧求一个相机系里的平移 d (n,3), 让他的轮廓对上自己的实例 mask(别人的像素不算)。只平移不转, 沿视线的分量管大小。
    返回 d(没 mask 的帧按时间插值), 以及对齐前后的轮廓 IoU 中位。"""
    n, H, W = len(Vw), D["H"], D["W"]
    rh = int(round(rw * H / W))
    fr = np.array([i for i in range(n) if not np.isnan(Vw[i, 0, 0]) and inst[i].sum() >= 400])
    if len(fr) < 3:
        return np.zeros((n, 3)), None, None
    K = D["K"][fr] * np.array([[rw / W], [rh / H], [1]])
    E = D["w2c"][fr]
    Vc = np.einsum("nij,nvj->nvi", E[:, :3, :3], Vw[fr]) + E[:, None, :3, 3]
    ob = np.stack([cv2.resize(inst[i].astype(np.float32), (rw, rh), interpolation=cv2.INTER_AREA) for i in fr])
    ot = np.stack([cv2.resize(other[i].astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST) for i in fr])
    # 起点: 外接框的高度比定远近、中心差定平移。mask 贴着画面边(人被画面截断)的帧不这么估, 从 0 起
    d0 = np.zeros((len(fr), 3))
    for q in range(len(fr)):
        ys, xs = np.nonzero(ob[q] > 0.5)
        if len(xs) < 20 or xs.min() <= 1 or ys.min() <= 1 or xs.max() >= rw - 2 or ys.max() >= rh - 2:
            continue
        z = Vc[q, :, 2]
        u = K[q, 0, 0] * Vc[q, :, 0] / z + K[q, 0, 2]
        v = K[q, 1, 1] * Vc[q, :, 1] / z + K[q, 1, 2]
        sc = np.clip(np.ptp(ys) / max(np.ptp(v), 1.0), 0.6, 1.6)
        zc = float(np.median(z))
        d0[q] = [((xs.min() + xs.max()) / 2 - (u.min() + u.max()) / 2) * zc / sc / K[q, 0, 0],
                 ((ys.min() + ys.max()) / 2 - (v.min() + v.max()) / 2) * zc / sc / K[q, 1, 1], zc * (1 / sc - 1)]
    Vc_t, K_t, I4 = T(Vc), T(K), torch.eye(4, device=DEV)[None].expand(len(fr), -1, -1).contiguous()
    F_t = T(faces, torch.int32)
    obs, valid = T(ob), 1 - T(ot)
    d = T(d0).clone().requires_grad_(True)
    opt = torch.optim.Adam([d], lr=0.01)
    dt = T(np.diff(tsec[fr])).clamp(min=1e-3)

    def blur(x, sg):
        r_ = int(3 * sg)
        xx = torch.arange(-r_, r_ + 1, device=DEV, dtype=torch.float32)
        g1 = torch.exp(-xx ** 2 / (2 * sg ** 2)); g1 = g1 / g1.sum()
        return F.conv2d(x[:, None], (g1[:, None] * g1[None, :])[None, None], padding=r_)[:, 0]

    def iou_now(dd, sg):
        out = []
        for c0 in range(0, len(fr), 24):
            sl = slice(c0, c0 + 24)
            clip = v2.to_clip(Vc_t[sl] + dd[sl][:, None], I4[sl], K_t[sl], rw, rh)
            rast, _ = dr.rasterize(glctx, clip, F_t, (rh, rw))
            sil = dr.antialias((rast[..., 3:4] > 0).float(), rast, clip, F_t)[..., 0]
            o = obs[sl]
            if sg > 0:
                sil, o = blur(sil, sg), blur(o, sg)
            inter = (sil * o * valid[sl]).sum((1, 2))
            out.append(inter / ((sil + o - sil * o) * valid[sl]).sum((1, 2)).clamp(min=1.0))
        return torch.cat(out)

    with torch.no_grad():
        before = float(iou_now(torch.zeros_like(d), 0).median())
    for it in range(iters):
        sg = 4.0 if it < iters // 2 else 1.5
        loss = (1 - iou_now(d, sg)).mean() + 0.3 * rho2(sq((d[1:] - d[:-1]) / dt[:, None]), 1.0).mean() + 0.05 * (sq(d) / 0.3 ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        i0 = iou_now(torch.zeros_like(d), 0).cpu().numpy()
        i1 = iou_now(d, 0).cpu().numpy()
    dn = d.detach().cpu().numpy()
    # 逐帧把关: 对完没变好的帧、挪得离谱的帧(人被画面截断 / 被挡住大半时会跑飞)不信, 从信得过的帧按时间插值
    good = (i1 >= i0) & (np.linalg.norm(dn, axis=1) <= 0.6)
    if good.sum() < 3:
        return np.zeros((n, 3)), before, before
    out = np.stack([np.interp(tsec, tsec[fr[good]], dn[good, c]) for c in range(3)], 1)
    return out, before, float(np.median(np.where(good, i1, i0)))


# ================================================================ 手腕修正 + 握持
def expm_so3(w):
    K = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = w[..., 2], -w[..., 1], w[..., 0]
    return torch.linalg.matrix_exp(K)


def hand_pose(hp, dR, dT):
    """hp: Rw (n,3,3) tw (n,3) 手腕 -> 世界, Hm (3,3) o (3,) 握持系在手腕里, Rc2w (n,3,3)。dR dT (B,n,3)。
    -> 修正后的手腕 (B,n,..) 和握持系 (B,n,..)"""
    Rw2 = hp["Rw"][None] @ expm_so3(dR)
    tw2 = hp["tw"][None] + torch.einsum("nij,bnj->bni", hp["Rc2w"], dT)
    return Rw2, tw2, Rw2 @ hp["Hm"], tw2 + torch.einsum("bnij,j->bni", Rw2, hp["o"])


def sq(x):
    return (x ** 2).sum(-1)


def rho2(x2, s):
    """鲁棒损失, 传进来的是平方: x^2/(x^2+s^2)"""
    return x2 / (x2 + s ** 2)


def hub2(x2, s):
    """不封顶的鲁棒损失(伪 Huber), 传进来的是平方: 小的像平方, 大的线性增长 —— 偏得越远越贵, 不会"反正都错了"就放弃"""
    return torch.sqrt(1 + x2 / s ** 2) - 1


def obj_pose(Rg, tg, GR, Gt, q):
    """握持系 (B,n) · 手内自由度 q (B,n,4) · G (B) -> 物体位姿 (B,n)。
    q = [绕掌心转的转动向量(握持系里, 3), 沿物体长轴滑的距离(米)]。只有细长物体(rod)开: 松松握着的杆子会绕握点转
    (伞搭在肩上 / 垂着 / 甩起来, 相对手差 90 度以上, 手腕拧不出来), 也会在手里滑、绕自己的轴转。握点始终钉在掌心。
    长轴 = 物体局部系的 x(load_mesh 保证 x 最长)。不是细长物体时 q 恒为 0。"""
    Rp = Rg @ expm_so3(q[..., :3])
    R = Rp @ GR[:, None]
    t = tg + torch.einsum("bnij,bnj->bni", Rp, Gt[:, None] + GR[:, None, :, 0] * q[..., 3:4])
    return R, t


def inhand_reg(q, tt, a):
    l = a.w_q * (rho2(sq(q[..., :3]), 1.0) + (q[..., 3] / a.q_slide) ** 2).mean(1)
    if q.shape[1] > 1:
        dt = (tt[1:] - tt[:-1]).clamp(min=1e-3)[None]
        l = l + a.w_qs * (rho2(sq((q[:, 1:, :3] - q[:, :-1, :3]) / dt[..., None]), a.q_spin_rate) + rho2(((q[:, 1:, 3] - q[:, :-1, 3]) / dt) ** 2, a.q_slide_vel)).mean(1)
    return l


def wrist_reg(dR, dT, tt, a):
    """手腕修正的代价 (B,): 转角小、画面平面里几乎不动(不封顶)、沿视线可以动、随时间变得慢"""
    l = a.w_wr * rho2(sq(dR), a.wr_ang).mean(1) + a.w_xy * hub2(sq(dT[..., :2]), a.wr_xy).mean(1) + a.w_z * ((dT[..., 2] / a.wr_z) ** 2).mean(1)
    if dR.shape[1] > 1:
        dt = (tt[1:] - tt[:-1]).clamp(min=1e-3)[None, :, None]
        l = l + a.w_sm * (rho2(sq((dR[:, 1:] - dR[:, :-1]) / dt), a.wr_rate).mean(1) + rho2(sq((dT[:, 1:] - dT[:, :-1]) / dt), a.wr_vel).mean(1))
    return l


def solve_G(v, hp, GR0, Gt0, logs0, dR0, dT0, q0, box, glctx, a, iters, lr=0.01, fix_scale=False, free=True, inhand=False, fix_G=False):
    """一段握持, B 个起点一起: 求 G(物体局部 -> 握持系)、物体大小、(free 时)每帧的手腕修正。"""
    r6 = torch.cat([GR0[..., 0], GR0[..., 1]], -1).clone().requires_grad_(True)
    Gt = Gt0.clone().requires_grad_(True)
    logs = logs0.clone().requires_grad_(True)
    dR = dR0.clone().requires_grad_(free)
    dT = dT0.clone().requires_grad_(free)
    q = q0.clone().requires_grad_(inhand)
    groups = [] if fix_G else [{"params": [r6, Gt], "lr": lr}]
    if inhand:
        groups.append({"params": [q], "lr": lr})
    if not fix_scale:
        groups.append({"params": [logs], "lr": 0.005})
    if free:
        groups.append({"params": [dR, dT], "lr": lr})
    opt = torch.optim.Adam(groups) if groups else None
    if opt is None:
        iters = 0
    for it in range(iters + 1):
        GR = v2.r6_to_mat(r6)
        s = logs.exp()
        _, _, Rg, tg = hand_pose(hp, dR, dT)
        R, t = obj_pose(Rg, tg, GR, Gt, q)
        l_sil, l_cham, l_cam = image_terms(R, t, s, v, box, glctx)
        palm = -torch.einsum("bji,bj->bi", GR, Gt)
        dpalm = torch.cdist(palm[:, None], box["BP"][None] * s[:, None]).amin(-1)[:, 0]
        img = (a.w_sil * l_sil + a.w_cham * l_cham).mean(1)
        per = img + l_cam.mean(1) + a.w_palm * v2.rho(F.relu(dpalm - a.palm_tol), 0.05) + a.w_shape * sq(logs - logs.mean(1, keepdim=True))
        if free:
            per = per + wrist_reg(dR, dT, hp["t"], a)
        if inhand:
            per = per + inhand_reg(q, hp["t"], a)
        if it == iters:
            break
        opt.zero_grad()
        per.sum().backward()
        opt.step()
    return dict(GR=GR.detach(), Gt=Gt.detach(), logs=logs.detach(), dR=dR.detach(), dT=dT.detach(), q=q.detach(), loss=per.detach(), img=img.detach(),
                palm_cm=dpalm.detach() * 100)


def interp_frames(x, t_src, t_dst):
    """(B, n_src, C) 按时间插值到 n_dst 帧"""
    xn = x.detach().cpu().numpy()
    out = np.stack([np.stack([np.interp(t_dst, t_src, xn[b, :, c]) for c in range(xn.shape[2])], -1) for b in range(xn.shape[0])])
    return T(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--word", required=True)
    ap.add_argument("--oid", required=True, help="实例号; 同一个物体跨镜头被拆成了几个编号时用逗号连起来(例: 0,6,8,5), mask 合在一起当一个物体")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--masks", default=None, help="重跟过的 mask.npz; 多个实例号时用逗号一一对应(空着的位置用 03_masks 的)")
    ap.add_argument("--init_poses", default=None, help="第二版的 carry_poses.npz: 多给几个 G 的起点, 没人拿的帧沿用它的位姿")
    ap.add_argument("--grip", default=None, help='人指定的握持段, JSON: [{"人":"p1","手":"hand_L","关键帧":[起,止]}]; 给了就不自动找')
    ap.add_argument("--frozen", action="store_true", help="手腕不修正(阶段 A)")
    ap.add_argument("--no_anchor", action="store_true", help="不把人对回画面(直接用世界系里摆好的人)")
    ap.add_argument("--iters_screen", type=int, default=120)
    ap.add_argument("--iters_coarse", type=int, default=150)
    ap.add_argument("--iters_fine", type=int, default=200)
    ap.add_argument("--iters_joint", type=int, default=200)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--w_sil", type=float, default=1.0)
    ap.add_argument("--w_cham", type=float, default=1.0)
    ap.add_argument("--w_palm", type=float, default=0.3, help="掌心贴着物体表面")
    ap.add_argument("--palm_tol", type=float, default=0.03)
    ap.add_argument("--w_shape", type=float, default=2.0, help="三向缩放贴近网格自己的比例。0.1 时托盘被拉成 0.44x0.43x0.14(网格 1:0.59:0.19); 2.0 守住比例, IoU 不变")
    ap.add_argument("--w_wr", type=float, default=0.3, help="手腕修正的转角")
    ap.add_argument("--wr_ang", type=float, default=0.6, help="转角的尺度(弧度)")
    ap.add_argument("--w_xy", type=float, default=1.0, help="手在画面平面里的挪动")
    ap.add_argument("--wr_xy", type=float, default=0.03, help="画面平面里挪动的尺度(米)")
    ap.add_argument("--w_z", type=float, default=0.1, help="手沿视线方向的挪动")
    ap.add_argument("--wr_z", type=float, default=0.3)
    ap.add_argument("--w_sm", type=float, default=0.5, help="手腕修正随时间变得慢")
    ap.add_argument("--wr_rate", type=float, default=2.0, help="修正转角变化的尺度(弧度/秒)")
    ap.add_argument("--wr_vel", type=float, default=0.5, help="修正位置变化的尺度(米/秒)")
    ap.add_argument("--w_cons", type=float, default=2.0, help="几只手对同一个物体的位姿要一致")
    ap.add_argument("--inhand", choices=["auto", "none", "rod"], default="auto", help="手内自由度: auto = 细长物体(rod)允许绕掌心转、沿杆滑")
    ap.add_argument("--w_q", type=float, default=0.05)
    ap.add_argument("--q_slide", type=float, default=0.15, help="沿杆滑动的尺度(米)")
    ap.add_argument("--w_qs", type=float, default=0.5)
    ap.add_argument("--q_spin_rate", type=float, default=6.0, help="绕杆转的角速度尺度(弧度/秒)")
    ap.add_argument("--q_slide_vel", type=float, default=0.3, help="沿杆滑的速度尺度(米/秒)")
    ap.add_argument("--w_pen", type=float, default=10.0)
    ap.add_argument("--pen_tol", type=float, default=0.03)
    ap.add_argument("--w_acc", type=float, default=1.0, help="物体运动平滑(只在联合阶段)")
    ap.add_argument("--min_sec", type=float, default=0.4, help="候选握持段至少这么长")
    ap.add_argument("--tie_deg", type=float, default=25.0, help="同一只手的两段握持, G 的朝向差在这以内就共用")
    ap.add_argument("--tie_m", type=float, default=0.06)
    ap.add_argument("--actor_sim", type=float, default=0.85, help="跨镜头认人: 两个(人号, 镜头)的衣着外观直方图相似度(巴氏系数)到这个数就连起来, 连通的算同一个人")
    ap.add_argument("--near", type=float, default=0.3, help="掌心离 mask 多近算候选(乘 mask 面积的平方根)")
    ap.add_argument("--near_len", type=float, default=0.35, help="掌心离 mask 多近算候选(乘 mask 的长边)")
    ap.add_argument("--ext_sec", type=float, default=1.5, help="每段握持的提议前后各多延这么久, 让逐帧选择有得比")
    ap.add_argument("--gap_min_iou", type=float, default=0.2, help="中间的短空档重新拟合后, 轮廓中位到这个数就挂回手上")
    ap.add_argument("--detach_margin", type=float, default=0.25, help="离手(第二版的自由位姿)的轮廓要比挂着好出这么多才算离手")
    ap.add_argument("--switch_cost", type=float, default=0.15, help="逐帧选择里换一次的代价 = 这个数 x 最短握持帧数")
    ap.add_argument("--hold_len", type=float, default=1.0, help="没判上的帧: 最近握着它的那只手离 mask 不超过这么多个物体长, 就还算拿着")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0_all = time.time()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    free = not a.frozen

    oids = [int(x) for x in str(a.oid).split(",")]
    mpaths = (a.masks.split(",") if a.masks else []) + [""] * len(oids)
    D = v2.load_inputs(a.r1, a.word, oids[0], RW=480, mask_path=mpaths[0] or None)
    for o_, mp_ in zip(oids[1:], mpaths[1:]):                   # 别的实例号: 只要它的 mask, 并进来
        D2 = v2.load_inputs(a.r1, a.word, o_, RW=480, mask_path=mp_ or None)
        D["masks"] |= D2["masks"]
        D["has_mask"] |= D2["has_mask"]
        del D2
    n, H, W, RW, RH = len(D["keys"]), D["H"], D["W"], D["RW"], D["RH"]
    bx = v2.load_mesh(a.mesh)
    people, faces = hd.load_people(a.r1)
    pids = sorted(people)
    for p in pids:                                   # 人的网格和手腕用同一份参数算(椅子位移已加), 不用 load_inputs 读的那份
        row = hd.rows_for(people[p], D["vf"])
        Vp = np.full((n, people[p]["V"].shape[1], 3), np.nan, np.float32)
        Vp[row >= 0] = people[p]["V"][row[row >= 0]]
        D["people"][p] = Vp
    tsec = (D["vf"] - D["vf"][0]) / D["fps"]
    dtm_ = float(np.median(np.diff(tsec)))
    GAP, MINLEN = max(2, round(0.3 / dtm_)), max(4, round(a.min_sec / dtm_))
    cls0 = hd.shape_class(bx["ext_ratio"])
    inhand = a.inhand == "rod" or (a.inhand == "auto" and cls0 == "rod")
    print(f"[输入] {n} 关键帧, 有 mask {int(D['has_mask'].sum())}, 人 {pids}, 关键帧间隔 {dtm_:.3f}s, 手腕修正 {'开' if free else '关'}, "
          f"网格形状类 {cls0}, 手内自由度 {'绕掌心转 + 沿杆滑' if inhand else '无'}", flush=True)

    glctx = dr.RasterizeCudaContext()
    Rc2w = np.transpose(D["w2c"][:, :3, :3], (0, 2, 1))
    Kr = D["K"] * np.array([[RW / W], [RH / H], [1]])
    mr = np.stack([cv2.resize(m.astype(np.uint8), (RW, RH), interpolation=cv2.INTER_NEAREST) for m in D["masks"]]).astype(bool)

    # ---- 把可能拿着它的人对回画面: 握姿要在"画面上的手"上求, 求出来的是相对手的量, 之后原样挂回世界系的手上
    SHOT = hd.shots_of(a.r1, D["keys"], D["fps"])                     # 每个关键帧在第几个镜头
    APP = {}                                                       # (人号, 镜头) -> 衣着外观(Lab 直方图), 跨镜头认人用
    ANCH = {p: np.zeros((n, 3)) for p in pids}
    pm_dir = f"{a.r1}/body4d/02_body4d/masks"
    ident = {}
    try:
        ident = json.load(open(f"{a.r1}/10_people.json"))["身份"]["人"]
    except Exception:
        pass
    if not a.no_anchor and os.path.isdir(pm_dir) and len(os.listdir(pm_dir)) >= n and ident:
        inst_all = np.stack([voc_index(cv2.imread(f"{pm_dir}/{i:08d}.png", cv2.IMREAD_COLOR)) for i in range(n)])
        mdil = np.stack([cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(2, RW // 30)) for m in mr]).astype(bool)
        if len(set(SHOT.tolist())) > 1:                             # 多镜头: 每个人在每个镜头里长什么样(实例 mask 里的像素, Lab 粗直方图)
            for p in pids:
                tracks = ident.get(p[1:], [])
                for sh in sorted(set(SHOT.tolist())):
                    fi = [i for i in np.where(SHOT == sh)[0] if not np.isnan(D["people"][p][i, 0, 0]) and np.isin(inst_all[i], tracks).sum() > 2000]
                    if not fi or not tracks:
                        continue
                    h_ = np.zeros((8, 4, 4))
                    for i in np.array(fi)[np.unique(np.linspace(0, len(fi) - 1, min(len(fi), 6)).round().astype(int))]:
                        im = v2.frame_image(D, i, D["keys"][i])
                        im = cv2.resize(im, (inst_all.shape[2] // 4, inst_all.shape[1] // 4), interpolation=cv2.INTER_AREA)
                        mk = cv2.resize(np.isin(inst_all[i], tracks).astype(np.uint8), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                        lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[mk]
                        hh, _ = np.histogramdd(lab.astype(float), bins=(8, 4, 4), range=((0, 256), (0, 256), (0, 256)))
                        h_ += hh / max(hh.sum(), 1)
                    APP[(p, int(sh))] = h_ / h_.sum()
        for p in pids:
            tracks = ident.get(p[1:], [])
            Vp = D["people"][p]
            # 只对"投影出来的外接框碰得到物体 mask"的人做(别的人不可能拿着它)
            hit = 0
            for i in np.where(D["has_mask"] & ~np.isnan(Vp[:, 0, 0]))[0]:
                uv, z = v2.project(Vp[i][None, ::50], D["w2c"][i:i + 1], Kr[i:i + 1])
                x0, y0, x1, y1 = [int(np.clip(c, 0, lim)) for c, lim in zip((uv[0, :, 0].min(), uv[0, :, 1].min(), uv[0, :, 0].max(), uv[0, :, 1].max()), (RW - 1, RH - 1, RW - 1, RH - 1))]
                hit += bool(mdil[i, y0:y1 + 1, x0:x1 + 1].any())
            if not tracks or hit < 3:
                continue
            inst = np.isin(inst_all, tracks)
            other = (inst_all > 0) & ~inst
            ANCH[p], b_, a_ = anchor_person(Vp, D["faces"], inst, other, D, glctx, tsec)
            if b_ is None or a_ <= b_:
                ANCH[p] = np.zeros((n, 3))
                print(f"[对回画面] {p}: 没对上(实例 mask 太少, 或者对完反而更差), 不动", flush=True)
                continue
            mv = np.linalg.norm(ANCH[p], axis=1)
            print(f"[对回画面] {p}: 人的轮廓 IoU 中位 {b_:.2f} -> {a_:.2f}; 挪了 中位 {np.median(mv) * 100:.0f} cm, p90 {np.percentile(mv, 90) * 100:.0f} cm "
                  f"(其中沿视线 中位 {np.median(np.abs(ANCH[p][:, 2])) * 100:.0f} cm)", flush=True)
            D["people"][p] = Vp + np.einsum("nij,nj->ni", Rc2w, ANCH[p])[:, None]
    ANCH_W = {p: np.einsum("nij,nj->ni", Rc2w, ANCH[p]) for p in pids}
    # 身份组: 10_people.json 里共用同一个跟踪编号的人号算同一个人(换镜头以后人号会变) —— 同一个人同一只手的握姿可以跨镜头共用
    IDG = {p: j for j, p in enumerate(pids)}
    for p in pids:
        for q_ in pids:
            if set(ident.get(p[1:], [])) & set(ident.get(q_[1:], [])):
                lo, hi = sorted((IDG[p], IDG[q_]))
                IDG = {k_: (lo if v_ == hi else v_) for k_, v_ in IDG.items()}

    def same_actor(p1, s1, p2, s2):
        """(人号 p1, 镜头 s1) 和 (p2, s2) 是不是同一个人。同一个镜头里只认人号。跨镜头: 人号不可信(SAM3 的跟踪编号会跨着硬切接到别人
        身上 —— 杀破狼的 p1 在镜头 0 是吴京、镜头 1 是甄子丹), 看衣着: 外观直方图的相似度够高, 而且比 s2 里别的人都像。
        没有外观(没有实例 mask / 只有一个镜头)时退回"共用跟踪编号的人号算同一个人"。"""
        if s1 == s2:
            return p1 == p2
        if (p1, s1) not in ACTOR or (p2, s2) not in ACTOR:
            return IDG[p1] == IDG[p2] and not APP
        return ACTOR[(p1, s1)] == ACTOR[(p2, s2)]

    # 把 (人号, 镜头) 按外观连成"演员": 相似度到 actor_sim 的连一条边(同一个镜头里的两个人不连), 连通的算同一个人。
    # 用连通而不是两两比: 同一个人在不同镜头里光照差得多, 两两比会卡在门槛上(杀破狼吴京 6 个镜头两两 0.74~0.93, 和甄子丹之间最高 0.72)。
    ACTOR = {u: j for j, u in enumerate(sorted(APP))}
    for u in sorted(APP):
        for w in sorted(APP):
            if u < w and u[1] != w[1] and float(np.sqrt(APP[u] * APP[w]).sum()) >= a.actor_sim:
                lo, hi = sorted((ACTOR[u], ACTOR[w]))
                ACTOR = {k_: (lo if v_ == hi else v_) for k_, v_ in ACTOR.items()}

    same_holder = lambda r1_, r2_: r1_["hk"][1] == r2_["hk"][1] and same_actor(r1_["hk"][0], r1_["shot"], r2_["hk"][0], r2_["shot"])
    if APP:
        groups_ = {}
        for u, g_ in ACTOR.items():
            groups_.setdefault(g_, []).append(f"{u[0]}@{u[1]}")
        print("[身份] 按衣着外观连成的演员(人号@镜头): " + " | ".join(", ".join(v_) for v_ in groups_.values()), flush=True)

    # ---- 每只手: 手腕和握持系, 对到关键帧(这帧没这个人 = NaN)。WR = 对回画面的(拟合用), WR_W = 世界系原样的(最后挂回去用)
    HANDS = [(p, k) for p in pids for k in hd.SIDE]
    WR, WR_W, GRIP = {}, {}, {}
    for p, k in HANDS:
        row = hd.rows_for(people[p], D["vf"])
        ok = row >= 0
        Rw = np.full((n, 3, 3), np.nan); tw = np.full((n, 3), np.nan)
        Rw[ok], tw[ok] = people[p]["wrist"][k][0][row[ok]], people[p]["wrist"][k][1][row[ok]]
        Hm, o = people[p]["grip_local"][k]
        WR_W[(p, k)] = (Rw, tw, Hm, o)
        tw = tw + ANCH_W[p]
        WR[(p, k)] = (Rw, tw, Hm, o)
        GRIP[(p, k)] = (Rw @ Hm, tw + Rw @ o)

    def hp_of(hk, fi, world=False):
        Rw, tw, Hm, o = (WR_W if world else WR)[hk]
        return dict(Rw=T(Rw[fi]), tw=T(tw[fi]), Hm=T(Hm), o=T(o), Rc2w=T(Rc2w[fi]), t=T(tsec[fi]))

    # ---- 候选握持: 掌心投到画面上离物体 mask 够近。放得宽 —— 只是候选, 对不对由拟合验
    cxy = np.full((n, 2), np.nan)
    DT, LMAJ = {}, {}
    for i in range(n):
        if D["has_mask"][i] and mr[i].any():
            ys, xs = np.nonzero(mr[i])
            cxy[i] = xs.mean(), ys.mean()
            DT[i] = cv2.distanceTransform((~mr[i]).astype(np.uint8), cv2.DIST_L2, 3)
            LMAJ[i] = max(cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))[1])     # mask 的长边(像素)
    runs = []
    if a.grip:
        for g in json.loads(a.grip):
            runs.append(((g["人"], g["手"]), int(g["关键帧"][0]), int(g["关键帧"][1]) + 1))
    else:
        # 每只手每帧: 掌心离 mask 多远, 用 mask 的长边当单位(细长的东西握在端头外一点, 用面积当单位会判成"没挨着":
        # 雨伞开头 40 帧右手握着伞柄端, 掌心在 mask 端头外 0.1~0.3 个伞长, 按 0.3*sqrt(面积) 一帧都不算)
        REL = {hk: np.full(n, np.nan) for hk in HANDS}
        NEAR = {hk: np.zeros(n, bool) for hk in HANDS}
        STRICT = {hk: np.zeros(n, bool) for hk in HANDS}
        for hk in HANDS:
            tg = GRIP[hk][1]
            uv, zc = v2.project(np.nan_to_num(tg, nan=0.0)[:, None], D["w2c"], Kr)
            uv, zc = uv[:, 0], zc[:, 0]
            for i in DT:
                if np.isnan(tg[i, 0]) or zc[i] <= 0.05:
                    continue
                u, v_ = np.clip(uv[i, 0], 0, RW - 1), np.clip(uv[i, 1], 0, RH - 1)
                dpx = DT[i][int(v_), int(u)] + np.hypot(uv[i, 0] - u, uv[i, 1] - v_)       # 掌心在画面外: 加上到画面边的距离
                REL[hk][i] = dpx / LMAJ[i]
                tight = max(3.0, a.near * np.sqrt(mr[i].sum()), 0.10 * Kr[i][0, 0] / zc[i])
                STRICT[hk][i] = dpx <= tight
                NEAR[hk][i] = dpx <= max(tight, a.near_len * LMAJ[i])
        # 标了"被人拿着"的东西默认一直在手上: 哪只手都没判上的帧, 只要离它最近(按时间)的那只握着它的手这一帧离 mask 不超过一个物体长,
        # 就还算这只手拿着。真离手(扔出去 / 放下)= 连续好几帧离所有手都超过一个物体长, 那些帧才不挂。
        # 滞回: 严格条件(贴着 mask)连续成段才算"握上了"; 握上之后前后用宽松条件(离 mask 不超过 near_len 个物体长)延伸。
        # 只满足宽松条件的手(旁边路过的人)不算 —— 滑板车旁边别人的手就是这么混进来过。
        for hk in HANDS:
            seed = v2.clean(STRICT[hk], GAP, MINLEN)
            grown = seed.copy()
            for rng_ in (range(1, n), range(n - 2, -1, -1)):
                step = -1 if rng_.step == 1 else 1
                for i in rng_:
                    if not grown[i] and grown[i + step] and SHOT[i] == SHOT[i + step] and (NEAR[hk][i] or (i not in DT and not np.isnan(GRIP[hk][1][i, 0]))):
                        grown[i] = True
            NEAR[hk] = grown
        anyn = np.stack([NEAR[hk] for hk in HANDS], 1)                       # (n, 手)
        held_by = np.full(n, -1)
        for i in range(n):
            if anyn[i].any():
                held_by[i] = int(np.argmin(np.where(anyn[i], [REL[hk][i] for hk in HANDS], np.inf)))
        known = np.where(held_by >= 0)[0]
        for i in range(n):
            if held_by[i] >= 0 or not len(known):
                continue
            ks_ = known[SHOT[known] == SHOT[i]]                       # 只看同一个镜头里的
            if not len(ks_):
                continue
            j = ks_[np.argmin(np.abs(ks_ - i))]
            hk = HANDS[held_by[j]]
            present = not np.isnan(GRIP[hk][1][i, 0])
            if present and (i not in DT or REL[hk][i] <= a.hold_len):
                NEAR[hk][i] = True
        for hk in HANDS:
            for s_, e_ in v2.runs_of(v2.clean(NEAR[hk], GAP, MINLEN)):
                cut = [s_] + [i for i in range(s_ + 1, e_) if SHOT[i] != SHOT[i - 1]] + [e_]      # 一段握持不跨镜头: 换了镜头人号可能已经是别人
                runs += [(hk, c0, c1) for c0, c1 in zip(cut[:-1], cut[1:])]
    frames_of = lambda hk, s_, e_: np.array([i for i in range(s_, e_) if i in DT and not np.isnan(GRIP[hk][1][i, 0])])
    runs = [r for r in runs if len(frames_of(*r)) >= MINLEN]
    runs.sort(key=lambda r: -(r[2] - r[1]))
    print("[候选握持] " + ("; ".join(f"{p}:{k} {s_}-{e_ - 1}" for (p, k), s_, e_ in runs) or "没有"), flush=True)
    assert runs, "没有找到任何一只手挨着这个物体的连续帧; 用 --grip 指定"

    # ---- 张量
    nv = people[pids[0]]["V"].shape[1]
    PVn = np.zeros((n, len(pids) * nv, 3), np.float32)
    for j, p in enumerate(pids):
        V = D["people"][p].copy()
        V[np.isnan(V[:, 0, 0])] = np.array([0, 0, -100.0])
        PVn[:, j * nv:(j + 1) * nv] = V
    PV = T(PVn)
    PF = T(np.concatenate([D["faces"] + j * nv for j in range(len(pids))]), torch.int32)
    box = dict(BV=T(bx["V"]), BF=T(bx["F"], torch.int32), BP=T(bx["pts"][rng.choice(len(bx["pts"]), 600, replace=False)]))
    SDF = T(bx["sdf"])[None, None]
    ST_S = make_stage(D, PV, PF, glctx, 160, int(round(160 * H / W)), 3.0, rng)
    ST_C = make_stage(D, PV, PF, glctx, 240, int(round(240 * H / W)), 2.0, rng)
    ST_F = make_stage(D, PV, PF, glctx, RW, RH, 0.0, rng)
    H24 = T(v2.cube_rotations())
    parts_v2 = np.load(f"{os.path.dirname(HERE)}/parts.npz")
    body_ok = np.setdiff1d(np.arange(nv), np.concatenate([parts_v2[k] for k in ("hand_L", "hand_R", "foot_L", "foot_R")]))
    body_sub = rng.choice(body_ok, 1200, replace=False)
    Bpts = np.concatenate([PVn[:, j * nv:(j + 1) * nv][:, body_sub] for j in range(len(pids))], 1)       # (n, 人*1200, 3)

    init = np.load(a.init_poses, allow_pickle=True) if a.init_poses else None
    if init is not None:
        Li = init["local_to_world"].astype(float)
        si = np.linalg.norm(Li[:, :3, :3], axis=1).mean(0)
        Rio = Li[:, :3, :3] / np.linalg.norm(Li[:, :3, :3], axis=1, keepdims=True)

    # ---- 逐段: 初筛(24 朝向 [+ 第二版的几帧], 抽帧, 低分辨率) -> 留几个粗拟合 -> 细调。每段先各自带着物体大小求
    res = []
    for (hk, s_, e_) in runs:
        Rg_n, tg_n = GRIP[hk]
        fr = frames_of(hk, s_, e_)
        _, zh = v2.project(tg_n[fr][:, None], D["w2c"][fr], Kr[fr])
        zh = zh[:, 0]
        pc = np.stack([(cxy[fr, 0] - Kr[fr, 0, 2]) / Kr[fr, 0, 0] * zh, (cxy[fr, 1] - Kr[fr, 1, 2]) / Kr[fr, 1, 1] * zh, zh], -1)
        cw = np.einsum("nji,nj->ni", D["w2c"][fr][:, :3, :3], pc - D["w2c"][fr][:, :3, 3])
        t0 = np.median(np.einsum("nji,nj->ni", Rg_n[fr], cw - tg_n[fr]), 0)       # 物体中心在握持系里(mask 质心视线上、和掌心一样远)
        ss = [max(np.ptp(np.nonzero(mr[i])[1]), np.ptp(np.nonzero(mr[i])[0])) * z / Kr[i][0, 0] for i, z in zip(fr, zh)]
        s0 = float(np.clip(np.percentile(ss, 80), 0.1, 3.0))
        GR0, Gt0 = [H24], [T(t0)[None].expand(24, -1)]
        if init is not None:
            io = np.nan_to_num(init["iou"][fr], nan=0.0)
            pick = []
            for j in np.argsort(-io):
                if all(abs(fr[j] - fr[q]) >= max(3, len(fr) // 8) for q in pick):
                    pick.append(j)
                if len(pick) == 5:
                    break
            GR0.append(T(np.einsum("nji,njk->nik", Rg_n[fr[pick]], Rio[fr[pick]])))
            Gt0.append(T(np.einsum("nji,nj->ni", Rg_n[fr[pick]], Li[fr[pick], :3, 3] - tg_n[fr[pick]])))
        GR0, Gt0 = torch.cat(GR0), torch.cat(Gt0).contiguous()
        B = len(GR0)
        logs0 = torch.full((B, 3), float(np.log(s0)), device=DEV)
        if init is not None:
            logs0[24:] = T(np.log(si))[None]
        t_c = time.time()
        fs = fr[np.unique(np.linspace(0, len(fr) - 1, min(len(fr), 40)).round().astype(int))]
        z = lambda b, m, c=3: torch.zeros(b, m, c, device=DEV)
        rs = solve_G(sub(ST_S, fs), hp_of(hk, fs), GR0, Gt0, logs0, z(B, len(fs)), z(B, len(fs)), z(B, len(fs), 4), box, glctx, a,
                     a.iters_screen, free=free, inhand=False)                      # 初筛不开手内自由度: 先让 24 个朝向公平比
        order = torch.argsort(rs["loss"])[:a.keep]
        rc = solve_G(sub(ST_C, fr), hp_of(hk, fr), rs["GR"][order], rs["Gt"][order], rs["logs"][order],
                     interp_frames(rs["dR"][order], tsec[fs], tsec[fr]), interp_frames(rs["dT"][order], tsec[fs], tsec[fr]),
                     z(len(order), len(fr), 4), box, glctx, a, a.iters_coarse, free=free, inhand=inhand)
        rf = solve_G(sub(ST_F, fr), hp_of(hk, fr), rc["GR"], rc["Gt"], rc["logs"], rc["dR"], rc["dT"], rc["q"], box, glctx, a, a.iters_fine,
                     lr=0.005, free=free, inhand=inhand)
        bi = int(torch.argmin(rf["loss"]))
        res.append(dict(hk=hk, s=s_, e=e_, shot=int(SHOT[fr[0]]), fr=fr, GR=rf["GR"][bi], Gt=rf["Gt"][bi], logs=rf["logs"][bi], dR=rf["dR"][bi], dT=rf["dT"][bi], q=rf["q"][bi],
                        loss=float(rf["loss"][bi]), img=float(rf["img"][bi]), start=("第二版的帧" if int(order[bi]) >= 24 else "24 朝向")))
        print(f"[握持 {hk[0]}:{hk[1]} {s_}-{e_ - 1}] {len(fr)} 帧, {B} 个起点, {time.time() - t_c:.1f}s; 初筛 {float(rs['loss'].min()):.3f}.."
              f"{float(rs['loss'].max()):.3f}; 细调 {[round(float(x), 3) for x in rf['loss']]} -> 轮廓项 {res[-1]['img']:.3f}, 起点来自 {res[-1]['start']}", flush=True)

    # ---- 物体大小: 拿对得最好的那段(够长的里面)的
    longest = max(len(r["fr"]) for r in res)
    ref = min([r for r in res if len(r["fr"]) >= 0.5 * longest], key=lambda r: r["img"])
    logs = ref["logs"].clone()
    # 候选里明显挂不上的段(轮廓项比最好的段差很多)不进联合阶段: 那只手只是路过
    keep_r = [r for r in res if r["img"] <= max(1.6 * ref["img"], ref["img"] + 0.25)]
    # 对得差但和留下的某段是"同一个人同一只手"的不丢: 多半是那个镜头糊, 它会借清楚镜头的握姿
    keep_r = [r for r in res if r in keep_r or any(same_holder(r, q_) for q_ in keep_r)]
    for r in res:
        if r not in keep_r:
            print(f"    丢掉 {r['hk'][0]}:{r['hk'][1]} {r['s']}-{r['e'] - 1}: 轮廓项 {r['img']:.3f} 比最好的段({ref['img']:.3f})差太多", flush=True)
    res = keep_r
    # 同一只手的几段握持, G 差不多就共用一个: 中间几帧没看见(被挡 / 没 mask)不等于换了握法, 少一组未知数、段和段之间也不会跳
    group = list(range(len(res)))
    for i in range(len(res)):
        for j in range(i):
            if same_holder(res[i], res[j]) and group[j] == j:
                cosang = ((res[i]["GR"].T @ res[j]["GR"]).trace() - 1) / 2
                ang = float(torch.rad2deg(torch.acos(cosang.clamp(-1, 1))))
                dist = float((res[i]["Gt"] - res[j]["Gt"]).norm())
                similar = dist < a.tie_m and (inhand or ang < a.tie_deg)       # 细长物体绕掌心能转, 握姿只比握在杆上哪儿
                weak = len(res[i]["fr"]) < max(3 * MINLEN, 0.5 * len(res[j]["fr"])) or res[i]["img"] > 1.25 * res[j]["img"]
                if similar or weak:
                    group[i] = j
                    ri, rj = res[i], res[j]
                    print(f"    {ri['hk'][0]}:{ri['hk'][1]} {ri['s']}-{ri['e'] - 1} 用 {rj['hk'][0]}:{rj['hk'][1]} {rj['s']}-{rj['e'] - 1} 的握姿("
                          + (f"差不多: {ang:.0f} 度, {dist * 100:.1f} cm" if similar else f"这段太短 / 对得差: {len(ri['fr'])} 帧, 轮廓项 {ri['img']:.2f} 对 {rj['img']:.2f}") + ")", flush=True)
                    if not similar:                               # 握姿换了: 这一段的手腕修正 / 手内自由度按新握姿重拟(握姿和大小不动)
                        zz = lambda c_: torch.zeros(1, len(ri["fr"]), c_, device=DEV)
                        rr = solve_G(sub(ST_C, ri["fr"]), hp_of(ri["hk"], ri["fr"]), rj["GR"][None], rj["Gt"][None], logs[None], zz(3), zz(3), zz(4),
                                     box, glctx, a, a.iters_coarse, fix_scale=True, free=free, inhand=inhand, fix_G=True)
                        ri.update(GR=rj["GR"], Gt=rj["Gt"], dR=rr["dR"][0], dT=rr["dT"][0], q=rr["q"][0], img=float(rr["img"][0]))
                    break

    # ---- 联合阶段: 所有握持段一起。每帧的物体位姿 = 主握持的提议; 别的握持的提议要和它一致; 加不穿人、运动平滑
    def primary_of(scores):
        sets = [set(r["all"].tolist()) for r in res]
        cover = [[k for k in range(len(res)) if i in sets[k]] for i in range(n)]
        prim = np.full(n, -1)
        cur = -1
        for i in range(n):
            if not cover[i]:
                cur = -1
                continue
            best = max(cover[i], key=lambda k: scores[k])
            if cur not in cover[i] or scores[best] > scores[cur] + 0.1:
                cur = best
            prim[i] = cur
        return prim

    for r in res:                                     # 这一段握持盖住的所有帧(人在就算, 有没有 mask 都算)
        r["all"] = np.array([i for i in range(r["s"], r["e"]) if not np.isnan(GRIP[r["hk"]][1][i, 0])])
    prim = primary_of([-r["img"] for r in res])
    U = np.array([i for i in range(n) if prim[i] >= 0 and i in DT])
    pos_in_U = {int(i): q for q, i in enumerate(U)}
    vU = sub(ST_F, U)
    tU = T(tsec[U])
    hps = [hp_of(r["hk"], r["fr"]) for r in res]
    idxU = [torch.tensor([pos_in_U[int(i)] for i in r["fr"] if int(i) in pos_in_U], device=DEV) for r in res]
    selU = [torch.tensor([q for q, i in enumerate(r["fr"]) if int(i) in pos_in_U], device=DEV) for r in res]
    isprim = [T((prim[r["fr"][s.cpu().numpy()]] == k).astype(np.float32)) for k, (r, s) in enumerate(zip(res, selU))]
    BptsU = T(Bpts[U])
    with torch.no_grad():                                               # 不穿人: 挂着物体的那些手附近 25cm 不算
        gp = torch.full((len(U), len(res), 3), 1e3, device=DEV)
        for k, r in enumerate(res):
            gp[idxU[k], k] = T(GRIP[r["hk"]][1][r["fr"]])[selU[k]]
        wpen = ((torch.cdist(BptsU, gp).amin(-1) - 0.25) / 0.10).clamp(0, 1)
    P = dict(r6=[torch.cat([r["GR"][:, 0], r["GR"][:, 1]], -1).clone().requires_grad_(True) for r in res],
             Gt=[r["Gt"].clone().requires_grad_(True) for r in res],
             dR=[r["dR"].clone().requires_grad_(free) for r in res], dT=[r["dT"].clone().requires_grad_(free) for r in res],
             q=[r["q"].clone().requires_grad_(inhand) for r in res])
    for k, g in enumerate(group):
        P["r6"][k], P["Gt"][k] = P["r6"][g], P["Gt"][g]
    uniq = lambda xs: list({id(x): x for x in xs}.values())
    logs = logs.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": uniq(P["r6"]) + uniq(P["Gt"]) + (P["dR"] + P["dT"] if free else []) + (P["q"] if inhand else []), "lr": 0.003},
                            {"params": [logs], "lr": 0.002}])
    dtU = (tU[1:] - tU[:-1]).clamp(min=1e-3)
    adj = (dtU < 2.5 * dtm_).float()                                    # 只在相邻的关键帧之间算平滑
    t_j = time.time()
    for it in range(a.iters_joint + 1):
        s = logs.exp()
        RU = torch.zeros(len(U), 3, 3, device=DEV)
        tUo = torch.zeros(len(U), 3, device=DEV)
        l_cons = torch.zeros((), device=DEV)
        l_reg = torch.zeros((), device=DEV)
        l_palm = torch.zeros((), device=DEV)
        props = []
        for k, r in enumerate(res):
            GR = v2.r6_to_mat(P["r6"][k])
            _, _, Rg, tg = hand_pose(hps[k], P["dR"][k][None], P["dT"][k][None])
            Rk, tk = obj_pose(Rg, tg, GR[None], P["Gt"][k][None], P["q"][k][None])
            Rk, tk = Rk[0][selU[k]], tk[0][selU[k]]
            if inhand:
                l_reg = l_reg + inhand_reg(P["q"][k][None], hps[k]["t"], a)[0] * len(r["fr"]) / len(U)
            props.append((Rk, tk))
            m = isprim[k]
            RU = RU.index_add(0, idxU[k], Rk * m[:, None, None])
            tUo = tUo.index_add(0, idxU[k], tk * m[:, None])
            if free:
                l_reg = l_reg + wrist_reg(P["dR"][k][None], P["dT"][k][None], hps[k]["t"], a)[0] * len(r["fr"]) / len(U)
            palm = -(GR.T @ P["Gt"][k])
            l_palm = l_palm + v2.rho(F.relu(torch.cdist(palm[None], box["BP"] * s).amin() - a.palm_tol), 0.05)
        for k in range(len(res)):                                       # 不是主握持的帧: 这只手提议的位姿要和物体实际的位姿一致
            Rk, tk = props[k]
            w_ = 1 - isprim[k]
            if w_.sum() > 0:
                # 单向: 物体的位姿由主握持和画面定, 这一项只拉"别的手"(它的 G 和手腕修正)去够物体 —— 重建得差的那只手不许把物体带歪
                l_cons = l_cons + ((hub2(sq(tk - tUo[idxU[k]].detach()), 0.03)
                                    + hub2(sq((Rk - RU[idxU[k]].detach()).reshape(-1, 9)) / 2, 0.2)) * w_).sum() / len(U)
        l_sil, l_cham, l_cam = image_terms(RU[None], tUo[None], s[None], vU, box, glctx)
        q = torch.einsum("nji,nvj->nvi", RU, BptsU - tUo[:, None]) / s
        sd = F.grid_sample(SDF, (q / 0.6).flip(-1).reshape(1, 1, 1, -1, 3), align_corners=True, padding_mode="border").reshape(len(U), -1) * s.mean()
        l_pen = (wpen * F.relu(-sd - a.pen_tol) ** 2).sum(-1).mean()
        vel = (tUo[1:] - tUo[:-1]) / dtU[:, None]
        acc = (vel[1:] - vel[:-1]) / ((dtU[1:] + dtU[:-1]) / 2)[:, None]
        Qr = RU[:-1].transpose(-1, -2) @ RU[1:]
        om = torch.stack([Qr[:, 2, 1] - Qr[:, 1, 2], Qr[:, 0, 2] - Qr[:, 2, 0], Qr[:, 1, 0] - Qr[:, 0, 1]], -1) / 2 / dtU[:, None]
        alpha = (om[1:] - om[:-1]) / ((dtU[1:] + dtU[:-1]) / 2)[:, None]
        w2 = adj[1:] * adj[:-1]
        l_acc = ((v2.rho(acc.norm(dim=-1) / 9.81, 1.0) + v2.rho(alpha.norm(dim=-1) / 30.0, 1.0)) * w2).sum() / w2.sum().clamp(min=1) if len(U) > 2 else torch.zeros((), device=DEV)
        img = (a.w_sil * l_sil + a.w_cham * l_cham).mean()
        tot = (img + l_cam.mean() + a.w_cons * l_cons + l_reg + a.w_palm * l_palm + a.w_pen * l_pen + a.w_acc * l_acc
               + a.w_shape * sq(logs - logs.mean()))
        if it == a.iters_joint:
            break
        opt.zero_grad()
        tot.backward()
        opt.step()
    s_np = logs.exp().detach().cpu().numpy()
    print(f"[联合] {len(res)} 段握持, {len(U)} 帧, {time.time() - t_j:.1f}s: 轮廓项 {float(img):.3f}, 一致 {float(l_cons):.4f}, 手腕修正 {float(l_reg):.4f}, "
          f"穿人 {float(l_pen):.5f}, 平滑 {float(l_acc):.4f}; 大小 {np.round(bx['ext_ratio'] * s_np, 3).tolist()} m", flush=True)

    # ---- 结果: 每帧的物体位姿, 每段握持修正后的手腕, 逐帧 IoU(960 宽, 真人的像素不算; 和第二版同一把尺)
    EW, EH = 960, int(round(960 * H / W))
    Ke = T(D["K"] * np.array([[EW / W], [EH / H], [1]]))
    w2c = T(D["w2c"])
    obs_e = np.stack([cv2.resize(m.astype(np.uint8), (EW, EH), interpolation=cv2.INTER_NEAREST) for m in D["masks"]]).astype(bool)
    pm_e = None
    if D["pmask"] is not None:
        pm_e = np.stack([cv2.resize(m.astype(np.uint8), (EW, EH), interpolation=cv2.INTER_NEAREST) for m in D["pmask"]]).astype(bool)
        pm_e = F.max_pool2d(torch.tensor(pm_e[:, None], dtype=torch.float32), 5, 1, 2)[:, 0].numpy() > 0
        pm_e &= ~obs_e

    def iou_of(M, idx):
        out = np.full(len(idx), np.nan)
        with torch.no_grad():
            for c0 in range(0, len(idx), 32):
                ii = idx[c0:c0 + 32]
                Mt = T(M[c0:c0 + 32])
                Xw = torch.einsum("nij,vj->nvi", Mt[:, :3, :3], box["BV"]) + Mt[:, None, :3, 3]
                rb, _ = dr.rasterize(glctx, v2.to_clip(Xw, w2c[ii], Ke[ii], EW, EH), box["BF"], (EH, EW))
                silb = (rb[..., 3] > 0).cpu().numpy()
                for q_, i in enumerate(ii):
                    if D["has_mask"][i]:
                        vv = ~pm_e[i] if pm_e is not None else np.ones_like(obs_e[i])
                        out[c0 + q_] = (silb[q_] & obs_e[i] & vv).sum() / max(((silb[q_] | obs_e[i]) & vv).sum(), 1)
        return out

    # ---- 每段握持的"提议": 它盖住的帧, 前后再各多延 ext 帧(延出去的帧、没 mask 的帧: 手腕修正和手内自由度按时间插值 / 取端头的值)。
    #      两份位姿: 挂在"对回画面的手"上(和画面比 IoU 用) / 挂在世界系原样的手上(同样的 G、手腕修正、手内自由度; 导出用)
    EXT = max(2, round(a.ext_sec / dtm_))
    PROP = []
    with torch.no_grad():
        for k, r in enumerate(res):
            tw_ = WR[r["hk"]][1]
            ia = np.array([i for i in range(max(0, r["s"] - EXT), min(n, r["e"] + EXT)) if not np.isnan(tw_[i, 0]) and SHOT[i] == r["shot"]])
            dRa, dTa, qa = [interp_frames(P[x][k][None], tsec[r["fr"]], tsec[ia]) for x in ("dR", "dT", "q")]
            GRk = v2.r6_to_mat(P["r6"][k])[None]
            Ms = {}
            for world in (False, True):
                Rw2, tw2, Rg, tg = hand_pose(hp_of(r["hk"], ia, world=world), dRa, dTa)
                Rk, tk = obj_pose(Rg, tg, GRk, P["Gt"][k][None], qa)
                M = np.tile(np.eye(4), (len(ia), 1, 1))
                M[:, :3, :3] = Rk[0].cpu().numpy() * s_np[None, None, :]
                M[:, :3, 3] = tk[0].cpu().numpy()
                Ms[world] = M
            PROP.append(dict(ia=ia, Mi=Ms[False], Mw=Ms[True], Rw=Rw2[0].cpu().numpy(), tw=tw2[0].cpu().numpy(),      # Rw / tw: 世界系里修正后的手腕
                             iou=iou_of(Ms[False], ia), iou_w=iou_of(Ms[True], ia)))

    # ---- 每帧到底挂在哪段握持上, 还是已经离手: 逐帧比轮廓, 带切换代价(动态规划)。
    #      前面的候选判定只看"掌心离 mask 近不近", 两只手都近的时候会挂错手(雨伞关键帧 34-39 挂到了左手), "默认一直在手上"又会让
    #      扔出去的东西粘在手上(滑板车)。这里让画面说了算: 选项 = 每段握持的提议 / 离手(第二版的自由位姿, 要比挂着好出
    #      detach_margin 才算; 没给第二版就没有这个选项) / 什么都没有。
    K = len(res)
    NEG = -1e9
    S = np.full((n, K + 2), NEG)
    hasm = np.array([i in DT for i in range(n)])
    for k, pr in enumerate(PROP):
        S[pr["ia"], k] = np.where(hasm[pr["ia"]], np.nan_to_num(pr["iou"], nan=0.0), 0.0)
    if init is not None:
        S[:, K] = np.where(hasm, np.nan_to_num(init["iou"], nan=0.0) - a.detach_margin, 0.0)
    S[:, K + 1] = -0.3
    lam = float(np.clip(a.switch_cost * MINLEN, 0.5, 3.0))
    best, back = np.full((n, K + 2), NEG), np.zeros((n, K + 2), int)
    best[0] = S[0]
    for i in range(1, n):
        prev = best[i - 1]
        jb = int(np.argmax(prev))
        for o in range(K + 2):
            if S[i, o] <= NEG / 2:
                continue
            if prev[o] >= prev[jb] - lam:
                best[i, o], back[i, o] = S[i, o] + prev[o], o
            else:
                best[i, o], back[i, o] = S[i, o] + prev[jb] - lam, jb
    choice = np.zeros(n, int)
    choice[-1] = int(np.argmax(best[-1]))
    for i in range(n - 1, 0, -1):
        choice[i - 1] = back[i, choice[i]]
    # ---- 夹在两段握持中间的短空档(被判成离手 / 什么都没有, 不超过 ext 帧): 人没松过手, 物体不该在这一两秒里飘着。
    #      拿前一段、后一段各自的握姿(G 不动), 只放开手腕修正和手内自由度, 在空档的帧上重新拟合; 哪只手对得好挂回哪只手。
    #      两只手都对不上(轮廓中位低于 gap_min_iou)才留着离手。排在最前 / 最后的离手段不动 —— 那是还没拿起来 / 扔出去了。
    for g0, g1 in [(s_, e_ - 1) for s_, e_ in v2.runs_of(choice >= K)]:
        if g0 == 0 or g1 == n - 1 or g1 - g0 + 1 > EXT:
            continue
        gi = np.arange(g0, g1 + 1)
        gm = np.array([i for i in gi if i in DT])
        cands = []
        for k in dict.fromkeys([int(choice[g0 - 1]), int(choice[g1 + 1])]):
            pos = {int(i): q_ for q_, i in enumerate(PROP[k]["ia"])}
            if not all(int(i) in pos for i in gi):
                continue
            r = res[k]
            with torch.no_grad():
                GRk = v2.r6_to_mat(P["r6"][k])[None]
            if len(gm):
                st0 = [interp_frames(P[x][k][None], tsec[r["fr"]], tsec[gm]) for x in ("dR", "dT", "q")]
                rs = solve_G(sub(ST_C, gm), hp_of(r["hk"], gm), GRk.detach(), P["Gt"][k].detach()[None], logs.detach()[None], *st0, box, glctx, a,
                             150, fix_scale=True, free=free, inhand=inhand, fix_G=True)
                st = [interp_frames(rs[x], tsec[gm], tsec[gi]) for x in ("dR", "dT", "q")]
            else:
                st = [interp_frames(P[x][k][None], tsec[r["fr"]], tsec[gi]) for x in ("dR", "dT", "q")]
            new = {}
            with torch.no_grad():
                for world in (False, True):
                    Rw2, tw2, Rg, tg = hand_pose(hp_of(r["hk"], gi, world=world), st[0], st[1])
                    Rk, tk = obj_pose(Rg, tg, GRk, P["Gt"][k][None], st[2])
                    M = np.tile(np.eye(4), (len(gi), 1, 1))
                    M[:, :3, :3] = Rk[0].cpu().numpy() * s_np[None, None, :]
                    M[:, :3, 3] = tk[0].cpu().numpy()
                    new["Mw" if world else "Mi"] = M
                new.update(Rw=Rw2[0].cpu().numpy(), tw=tw2[0].cpu().numpy(), iou=iou_of(new["Mi"], gi), iou_w=iou_of(new["Mw"], gi))
            sc = float(np.nanmedian(new["iou"])) if np.isfinite(new["iou"]).any() else 0.0
            cands.append((sc, k, [pos[int(i)] for i in gi], new))
        if not cands:
            continue
        sc, k, qi, new = max(cands, key=lambda c: c[0])
        ok_ = sc >= a.gap_min_iou or not len(gm)
        print(f"    空档 {g0}-{g1}: 重新拟合后轮廓中位 " + ", ".join(f"{res[c[1]]['hk'][1]} {c[0]:.2f}" for c in cands)
              + (f" -> 挂回 {res[k]['hk'][0]}:{res[k]['hk'][1]}" if ok_ else f" -> 都低于 {a.gap_min_iou}, 留着离手"), flush=True)
        if ok_:
            for key in ("Mi", "Mw", "Rw", "tw", "iou", "iou_w"):
                PROP[k][key][qi] = new[key]
            choice[gi] = k
    att = choice < K
    prim = np.where(att, choice, -1)

    M_img, M_all = np.tile(np.eye(4), (n, 1, 1)), np.tile(np.eye(4), (n, 1, 1))
    iou_all, iou_world = np.full(n, np.nan), np.full(n, np.nan)
    for k, (r, pr) in enumerate(zip(res, PROP)):
        pos = {int(i): q_ for q_, i in enumerate(pr["ia"])}
        sel = np.array([i for i in pr["ia"] if choice[i] == k], int)
        qs = [pos[int(i)] for i in sel]
        M_img[sel], M_all[sel], iou_all[sel], iou_world[sel] = pr["Mi"][qs], pr["Mw"][qs], pr["iou"][qs], pr["iou_w"][qs]
    for k, (r, pr) in enumerate(zip(res, PROP)):
        # 这只手算"拿着"的帧: 物体挂在它身上的帧; 加上它自己那段里物体挂在别的手上、但它的提议和物体实际的位姿对得上的帧(两只手
        # 一起拿)。对不上的(逐帧选择里输掉的候选: 这只手其实没拿着)不算 —— 不进指标, 手臂 IK 也不把这只手往物体上拉。
        pos = {int(i): q_ for q_, i in enumerate(pr["ia"])}
        hold = []
        for i in r["all"]:
            if choice[i] == k:
                hold.append(i)
            elif att[i]:
                q_ = pos[int(i)]
                near_ = np.linalg.norm(pr["Mi"][q_][:3, 3] - M_img[i][:3, 3]) < 0.15
                fits = (not hasm[i]) or np.nan_to_num(pr["iou"][q_], nan=0.0) >= np.nan_to_num(iou_all[i], nan=0.0) - a.detach_margin
                if near_ and fits:
                    hold.append(i)
        r["hold"] = np.unique(np.concatenate([np.array([i for i in pr["ia"] if choice[i] == k], int), np.array(hold, int)])).astype(int)
        r["hold_q"] = [pos[int(i)] for i in r["hold"]]
    if init is not None:
        Mi = Li.copy()
        Mi[:, :3, :3] = Rio * s_np[None, None, :]
        M_all[~att], M_img[~att] = Mi[~att], Mi[~att]
        iou_world[~att], iou_all[~att] = init["iou"][~att], init["iou"][~att]
    elif att.any():
        ai = np.where(att)[0]
        for i in np.where(~att)[0]:
            M_all[i] = M_img[i] = M_all[ai[np.argmin(np.abs(ai - i))]]
    # ---- 哪段握持都没盖住、也没有"离手"可选的帧(没给第二版; 物体在这些镜头里没被跟上): 标了"被人拿着"的东西默认还在手上 ——
    #      拿着它的那个人(同一身份的人号)只要在画面里, 就按离得最近(按时间)的那段握持的握姿挂在他同一只手上, 手内自由度沿用那段端头的值。
    carried = np.zeros(n, bool)
    carry_name = {}
    if res and att.any():
        for i in np.where(choice == K + 1)[0]:
            best_ = None
            for k, (r, pr) in enumerate(zip(res, PROP)):
                held_k = np.where(choice == k)[0]
                if not len(held_k):
                    continue
                j = held_k[np.argmin(np.abs(held_k - i))]
                for p in pids:
                    hk2 = (p, r["hk"][1])
                    if (not np.isnan(WR[hk2][1][i, 0]) and same_actor(r["hk"][0], r["shot"], p, int(SHOT[i]))
                            and (best_ is None or abs(j - i) < best_[0])):
                        best_ = (abs(j - i), k, j, hk2)
            if best_ is None:
                continue
            _, k, j, hk2 = best_
            r, pr = res[k], PROP[k]
            with torch.no_grad():
                qj = interp_frames(P["q"][k][None], tsec[r["fr"]], tsec[[j]])
                z3 = torch.zeros(1, 1, 3, device=DEV)
                for world, M in ((False, M_img), (True, M_all)):
                    _, _, Rg, tg = hand_pose(hp_of(hk2, np.array([i]), world=world), z3, z3)
                    Rk, tk = obj_pose(Rg, tg, v2.r6_to_mat(P["r6"][k])[None], P["Gt"][k][None], qj)
                    M[i, :3, :3] = Rk[0, 0].cpu().numpy() * s_np[None, :]
                    M[i, :3, 3] = tk[0, 0].cpu().numpy()
            carried[i] = True
            carry_name[int(i)] = (f"{hk2[0]}:{hk2[1]}", group[k])
        att = att | carried
        if init is None and att.any():                          # 剩下还是哪儿都挂不上的帧(拿着它的人不在画面里): 停在最近一次挂着的位姿
            ai = np.where(att)[0]
            for i in np.where(~att)[0]:
                M_all[i] = M_img[i] = M_all[ai[np.argmin(np.abs(ai - i))]]
    print(f"[逐帧选择] 切换代价 {lam:.2f}: " + "".join(str(c) if c < K else ("-" if c == K else ".") for c in choice)
          + f"   (数字 = 第几段握持, - = 离手, . = 没有)", flush=True)
    PARTS = [f"{p}:{k}" for p in pids for k in ("hand_L", "hand_R", "foot_L", "foot_R")]
    cont = np.zeros((n, len(PARTS)), bool)
    hold_run = np.full((n, len(PARTS)), -1, np.int16)                   # 每只手每帧是按哪一段握持(哪个 G)拿着的: 指标按它分段
    parent = np.array([""] * n, dtype=object)
    targets, segs, nt = {}, [], 0
    for k, r in enumerate(res):
        name = f"{r['hk'][0]}:{r['hk'][1]}"
        cont[r["hold"], PARTS.index(name)] = True
        hold_run[r["hold"], PARTS.index(name)] = group[k]
        parent[prim == k] = name
        ang = np.degrees(P["dR"][k].detach().norm(dim=-1).cpu().numpy())
        dT = P["dT"][k].detach().cpu().numpy()
        qk = P["q"][k].detach().cpu().numpy()
        if len(r["hold"]):                                  # 给手臂 IK 的目标: 这只手拿着的那些帧上, 修正后的手腕
            targets[f"{nt}_name"], targets[f"{nt}_kf"], targets[f"{nt}_vf"] = name, r["hold"], D["vf"][r["hold"]]
            targets[f"{nt}_Rw"], targets[f"{nt}_tw"] = PROP[k]["Rw"][r["hold_q"]], PROP[k]["tw"][r["hold_q"]]
            nt += 1
        io = iou_all[r["fr"]]
        segs.append(dict(手=name, 关键帧=[int(r["s"]), int(r["e"] - 1)], 帧数=int(len(r["fr"])), 当主握持的帧=int((prim == k).sum()), 这只手拿着的帧=int(len(r["hold"])),
                         IoU中位=round(float(np.nanmedian(io)), 3) if np.isfinite(io).any() else None,
                         手腕修正_转角度_中位=round(float(np.median(ang)), 1), 手腕修正_转角度_p90=round(float(np.percentile(ang, 90)), 1),
                         手腕修正_画面内cm_中位=round(float(np.median(np.linalg.norm(dT[:, :2], axis=1))) * 100, 1),
                         手腕修正_沿视线cm_中位=round(float(np.median(np.abs(dT[:, 2]))) * 100, 1), 起点=r["start"], G组=int(group[k]),
                         手内_绕掌心转度_最大=round(float(np.degrees(np.linalg.norm(qk[:, :3], axis=1).max())), 1),
                         手内_沿杆滑cm_范围=round(float(np.ptp(qk[:, 3])) * 100, 1),
                         G_R=np.round(v2.r6_to_mat(P["r6"][k]).detach().cpu().numpy(), 4).tolist(), G_t=np.round(P["Gt"][k].detach().cpu().numpy(), 4).tolist()))
    for i, (nm, g_) in carry_name.items():                     # 没看见、沿用握姿的帧: 也算这只手拿着(手臂不动, 不给 IK 目标)
        cont[i, PARTS.index(nm)] = True
        hold_run[i, PARTS.index(nm)] = g_
        parent[i] = nm
    np.savez(f"{a.out}/wrist_targets.npz", n=nt, **targets)
    phase = np.where(att, "拿着", init["phase"] if init is not None else "空中")
    np.savez(f"{a.out}/carry_poses.npz", keys=np.array(D["keys"]), vf=D["vf"], local_to_world=M_all, scale=s_np, mesh=a.mesh,
             mesh_to_local=bx["A"], box_V=bx["Vfull"], box_F=bx["Ffull"], iou=iou_all, contact=cont, parts=np.array(PARTS),
             phase=phase, attach_parent=parent.astype(str), hold_run=hold_run, local_to_world_img=M_img, iou_world=iou_world,
             anchor_cam=np.stack([ANCH[p] for p in pids]), anchor_people=np.array(pids))
    med = lambda x: round(float(np.nanmedian(x)), 3) if np.isfinite(x).any() else None
    summ = dict(版本="挂手版(每段一个固定 G" + (", 手腕带修正)" if free else ", 手腕冻结)"), 用时秒=round(time.time() - t0_all, 1),
                形状类=hd.shape_class(bx["ext_ratio"] * s_np), 物体尺寸米=[round(float(x), 3) for x in bx["ext_ratio"] * s_np],
                挂在手上的帧=int(att.sum()), 其中没看见沿用握姿的帧=int(carried.sum()), 没挂上的帧=int((~att).sum()),
                没挂上的关键帧段=[[int(s_), int(e_ - 1)] for s_, e_ in v2.runs_of(~att)], 轮廓IoU_挂着的帧_中位=med(iou_all[att]),
                轮廓IoU_挂回世界系的手_中位=med(iou_world[att]),
                第二版同样这些帧_IoU中位=med(init["iou"][att]) if init is not None else None, 握持段=segs)
    json.dump(summ, open(f"{a.out}/attach.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "握持段"}, ensure_ascii=False))
    for g in segs:
        print("   ", json.dumps({k: v for k, v in g.items() if not k.startswith("G_")}, ensure_ascii=False))
    print("总用时", round(time.time() - t0_all, 1), "s")


if __name__ == "__main__":
    main()
