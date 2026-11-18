"""补腿: 把 ~/sam3d/proclegs(程序化补腿 v24: 规则脚印 + 接回看得见的腿)接到主流水线的 MHR 人上(2026-11-11)。

为什么这样接(不走 SMPL-X 往返):
  proclegs 的"规划"(什么时候迈哪只脚、落在哪、抬多高、骨盆压多少、和看得见的腿怎么接)只用到 骨盆路线 + 朝向 +
  一组腿/脚的尺寸(proclegs.Body 的十几个数), 和 SMPL-X 无关; SMPL-X 只在它最后一步 IK 和存盘里用到。
  所以: 规划原样复用(import proclegs, 调 run(), 把 Body 换成按这个人的 MHR 骨架量出来的尺寸), 只重写最后一步 ——
  MHR 的腿关节参数用梯度下降对准规划出的脚(脚跟 / 脚底中段 / 前掌内外 / 脚尖 5 个点) + 膝盖朝脚尖。
  往返的话要先把 MHR 拟合成 SMPL-X(RoHM/ours/fit_smplx.py, 慢, 要 rohm 环境), 回来再拟合一次; 两副骨架脚长、踝高不同,
  照 SMPL-X 的踝去摆 MHR 的脚, 踩地时会滑、会穿地或悬空几厘米。这里规划用的就是 MHR 自己的脚, 脚印钉住就是钉住。

一直开, 按数据决定(classify):
  每帧"脚看不看得见" = 调用方给的 feet_ok(主流水线 body_into_scene.feet_visible: 脚投回原图, 在画面里且落在自己的分割上);
  短于 VIS_MERGE_S 的进出并掉(同 proclegs/detect_legs.py)。
  A 全程看得见: 不动。  B 全程看不见: 整段规划。
  C 有时看得见: 看得见的帧用模型的腿(踩地的脚钉成脚印、不滑), 看不见的段规划、两头接上(proclegs join='plan')。
  坐着的人(坐着的帧 > SEATED_SKIP)不补, 交给座位逻辑。镜头切点 / 坏相机(seg)分段各自规划; 短于 MIN_SEG_S 的段不补。
只改: 两条腿的旋转参数(髋 3 + 膝 1 + 踝/脚 4 + 前掌 2, 见 LEG_COLS) 和根的高度(proclegs 腿够不着时把骨盆往下压, 上半身跟着);
  根的水平位置、朝向、上半身的参数一个数不动。
"""
import os, sys, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import mhr_body as M

PROCLEGS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "proclegs"))
VIS_MERGE_S = 0.5      # 看得见/看不见短于这么久的进出并掉(同 detect_legs.py)
SEATED_SKIP = 0.3      # 坐着的帧超过这个比例: 不补(交给座位逻辑)
MIN_SEG_S = 1.0        # 一段(镜头)短于这么久迈不了一步: 不规划, 看不见腿的帧改成"站着"(见 fill_legs 里 太短 那一支)
MAX_GAP_S = 0.75       # 相邻两帧隔这么久以上(人出画 / 被挡又回来) = 断开分段规划; 不断开的话 30 fps 重采样会把空档当成慢慢走
GLIDE_V = (0.10, 0.60) # 看得见的帧里, 脚贴着地(< 4 cm)却以这个速度(m/s)在挪: 模型的腿是"定住的姿势跟着身体滑", 不是观测 ——
GLIDE_S = 0.3          #   连着超过 GLIDE_S 秒就当看不见、交给规划。(office r1_sam31 人 2 0.3~2 s: 两脚都"踩着"跟骨盆一起以 0.23 m/s 滑,
                       #   钉成脚印后人往前走腿够不着, 骨盆被压低 8.6 cm)
FPS30 = 30             # proclegs 按 30 fps 规划(常数写死在它里面); 规划完按调用方的帧时间取回
DZ_MAX = 1.0        # 补腿时骨盆最多挪多少(m), 再多就不补那一帧(见 fill_legs)
IK_ITERS = 300
IK_LR = 0.05
W_KNEE = 0.3           # 膝盖朝脚尖 这一项的权重(cm^2 量级; 点的误差是 cm)
W_SMOOTH = 0.02        # 腿参数时间二阶差分
W_HYPER = 100.0        # 膝盖反弯(铰链参数 < 0)的罚
SIDES = ("l", "r")
# 每条腿参与 IK 的参数列(2026-11-11 逐列查过 parameter_transform):
#   upleg rx/ry/rz, lowleg rz(膝, 铰链, 正 = 往前弯), foot rx(小腿扭转)/ry, talocrural rz/rx, ball rz/rx
LEG_COLS = {"l": [59, 60, 61, 62, 63, 122, 64, 124, 67, 125], "r": [50, 51, 52, 53, 54, 126, 55, 128, 58, 129]}
KNEE_COL = {"l": 62, "r": 53}
# The first ball parameter supplies toe flexion. The second twists the skin
# laterally and narrows the forefoot under linear blend skinning.
BALL_TWIST = (125, 129)
PTS = ("heel", "mid", "ball_in", "ball_out", "toe")
PT_W = np.array([1.0, 1.0, 1.0, 1.0, 0.5])
# mid = 脚底中段(足弓下面)那个点: 没有它的话 IK 会用前掌关节把前掌/脚尖"折"回目标, 脚掌中段陷进地 2.8 cm(假人测试里量到)


def _proclegs():
    if PROCLEGS not in sys.path:
        sys.path.insert(0, PROCLEGS)
    import proclegs as PL
    return PL


# ------------------------------------------------------------------ 坐标: DA3 世界 <-> proclegs 的 W(z 朝上, 地面 z=0)
def _basis(UP):
    e3 = np.asarray(UP, float) / np.linalg.norm(UP)
    a = np.array([1.0, 0, 0]) if abs(e3[0]) < 0.9 else np.array([0, 0, 1.0])
    e1 = a - (a @ e3) * e3; e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(e3, e1), e3])          # 行: W 的三根轴在世界里的方向(右手)


def _gh_off(GH, nd):
    """地面高 -> 要减 / 加的 (…,3) 偏移。GH 标量 = 一块平地(老做法); (T,) = 每帧人脚下的地面高(有地形时, 2026-11-11 加)。
    nd: 点数组的维数((T,3) 是 2, (T,k,3) 是 3)。"""
    G = np.asarray(GH, float)
    if G.ndim == 0:
        return np.array([0, 0, float(G)])
    off = np.zeros(G.shape + (3,)); off[..., 2] = G
    return off.reshape((len(G),) + (1,) * max(nd - 2, 0) + (3,))


def to_W(X, B3, GH):
    X = np.asarray(X)
    return X @ B3.T - _gh_off(GH, X.ndim)


def from_W(Y, B3, GH):
    Y = np.asarray(Y)
    return (Y + _gh_off(GH, Y.ndim)) @ B3


# ------------------------------------------------------------------ MHR 骨架(只算骨架, 不蒙皮: 比 forward 快几十倍, 能求导)
def _qmat(q):
    """四元数 xyzw (...,4) -> 旋转矩阵 (...,3,3)。torch。"""
    import torch
    x, y, z, w = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))



def distal_prior(source, solve_columns):
    """Retain the observed local foot shape within a soft angular allowance.
    
    The ankle can roll and the metatarsals can flex at toe-off. A match to
    the sole markers alone is ambiguous about which joint supplied the bend.
    These allowances limit changes *from the source*, including when the
    source already contains a kick or a raised heel.
    """
    coordinates = [(c, degrees) for side in SIDES
                   for c, degrees in zip(LEG_COLS[side][4:9], (18., 15., 24., 12., 15.))
                   if c in solve_columns]
    columns, allowance = zip(*coordinates)
    indices = [solve_columns.index(c) for c in columns]
    return indices, source[:,list(columns)], source.new_tensor(np.deg2rad(allowance))


def skel_torch(params_t, dev):
    """(T,204) torch -> 骨架状态 (T,127,8) [平移 cm, 四元数 xyzw, 缩放], 和 mhr_body.forward 给的一样(骨架和体型无关)。"""
    import torch
    ch = M.model(dev).character_torch
    jp = ch.model_parameters_to_joint_parameters(torch.cat([params_t, torch.zeros(len(params_t), M.NSHAPE, device=dev,
                                                                                    dtype=params_t.dtype)], 1))
    return ch.joint_parameters_to_skeleton_state(jp)


def _attach(S_np, j, X):
    """把(静止姿势里)的点 X (n,3) 米 挂到关节 j 上: 返回关节局部坐标 (n,3) cm-单位缩放前。"""
    t, q, s = S_np[j, :3], S_np[j, 3:7], S_np[j, 7]
    R = Rot.from_quat(q).as_matrix()
    return (X * 100.0 - t) @ R / s


def _place(S, j, L):
    """torch: 骨架状态 S (T,127,8), 关节 j, 局部点 L (n,3) -> 世界点 (T,n,3) 米。"""
    R = _qmat(S[:, j, 3:7]); t = S[:, j, :3]; s = S[:, j, 7:8]
    return (t[:, None, :] + s[:, None, :] * (L[None] @ R.transpose(1, 2))) / 100.0


# ------------------------------------------------------------------ 这个人的腿 / 脚尺寸(静止姿势量)
def geometry(shape, P0, dev="cpu"):
    """量这个人(体型 shape、骨头长度 = P0 里的长度参数和骨骼缩放)的腿和脚:
    脚上 5 个点(脚跟 / 脚底中段 / 前掌内 / 前掌外 / 脚尖, 脚底那一层的顶点)在"脚自己的放平坐标"(x 左 y 上 z 前, 原点 = 踝)里的位置,
    以及它们挂在哪个关节上(IK 用); 拉直了的大腿/小腿向量(proclegs.Body 用)。"""
    sk = M.skeleton(dev); names = [str(n) for n in sk["names"]]; par = sk["parents"].astype(int)
    rest = np.array(P0, np.float64)[None].copy()
    lc = M.length_params(dev)
    rest[:, :M.NPOSE] = 0.0; rest[:, lc] = np.asarray(P0)[lc]
    V, S = M.forward(shape, rest, False, dev); V, S = V[0], S[0]
    # 每个顶点权重最大的关节
    o = np.lexsort((-sk["sk_w"], sk["sk_v"])); v_, j_ = sk["sk_v"][o], sk["sk_j"][o]
    first = np.r_[True, v_[1:] != v_[:-1]]; jmax = np.full(len(V), -1); jmax[v_[first]] = j_[first]
    J = lambda n: S[names.index(n), :3] / 100.0

    def below(root):
        out = {names.index(root)}
        changed = True
        while changed:
            changed = False
            for k, p_ in enumerate(par):
                if p_ in out and k not in out:
                    out.add(k); changed = True
        return out
    g = dict(names=names, height=float(V[:, 1].max() - V[:, 1].min()), side={})
    for sd in SIDES:
        foot_j = below(f"{sd}_talocrural")
        fv = np.where(np.isin(jmax, list(foot_j)))[0]
        F = V[fv]; ymin = F[:, 1].min()
        sole = fv[F[:, 1] < ymin + 0.012]
        ank, ball_j = J(f"{sd}_talocrural"), J(f"{sd}_ball")
        heel = sole[np.argmin(V[sole, 2])]
        near = sole[np.abs(V[sole, 2] - ball_j[2]) < 0.015]
        near = near if len(near) >= 2 else sole[np.argsort(np.abs(V[sole, 2] - ball_j[2]))[:6]]
        b_in, b_out = near[np.argmin(np.abs(V[near, 0]))], near[np.argmax(np.abs(V[near, 0]))]
        low2 = fv[F[:, 1] < ymin + 0.02]
        toe = low2[np.argmax(V[low2, 2])]
        zm = (V[heel, 2] + ball_j[2]) / 2                                # 脚跟和前掌中间
        midc = sole[np.abs(V[sole, 2] - zm) < 0.02]
        midc = midc if len(midc) else sole
        mid = midc[np.argmin(V[midc, 1])]
        vids = dict(heel=int(heel), mid=int(mid), ball_in=int(b_in), ball_out=int(b_out), toe=int(toe))
        # 脚的放平坐标: 前 = 脚跟 -> 前掌中点 的水平方向, 上 = y; 再绕左右轴转, 让脚跟和前掌一样高
        bm = (V[b_in] + V[b_out]) / 2
        f = bm - V[heel]; f[1] = 0; f /= np.linalg.norm(f)
        y = np.array([0, 1.0, 0]); x = np.cross(y, f)
        Rf = np.stack([x, y, f], 1)                                   # 列: 脚的 x 左 / y 上 / z 前 在模型里的方向
        loc = {k: Rf.T @ (V[i] - ank) for k, i in vids.items()}
        a = np.arctan2(((loc["ball_in"] + loc["ball_out"]) / 2)[1] - loc["heel"][1],
                       ((loc["ball_in"] + loc["ball_out"]) / 2)[2] - loc["heel"][2])
        Rp = Rot.from_rotvec([a, 0, 0]).as_matrix()                  # 绕 x 转 a: 前掌比脚跟高 a 的话转平
        loc = {k: Rp @ v for k, v in loc.items()}
        allf = (V[fv] - ank) @ Rf @ Rp.T                              # 整只脚所有顶点, 放平坐标
        ankle_h = float(-allf[:, 1].min())                            # 踝离脚底 = 到最低那个顶点(只看 3 个点会高 1.5 cm, 脚陷进地)
        # 这几个点保持自己真实的高度(脚跟/前掌那几个顶点比整只脚最低的顶点高 1 cm 左右; 硬按到脚底平面, 整只脚就陷进地 1~1.5 cm)
        # 挂到关节上(IK 时按骨架算这 4 个点, 不用蒙皮)
        att = {k: (int(jmax[i]), _attach(S, int(jmax[i]), V[i][None])[0]) for k, i in vids.items()}
        hip, knee = J(f"{sd}_upleg"), J(f"{sd}_lowleg")
        t, s = knee - hip, ank - knee; v = t + s
        tgt = np.array([0.0, v[1], v[2]]); tgt *= np.linalg.norm(v) / np.linalg.norm(tgt)
        Ral = Rot.align_vectors(tgt[None], v[None])[0].as_matrix()  # 去掉静止姿势的外展(MHR 静止姿势两脚叉开 38 cm)
        g["side"][sd] = dict(vids=vids, loc=loc, att=att, ankle_h=float(ankle_h), hip=hip, t0=Ral @ t, s0=Ral @ s,
                             j=dict(hip=names.index(f"{sd}_upleg"), knee=names.index(f"{sd}_lowleg"),
                                    ank=names.index(f"{sd}_talocrural")))
    return g


def make_body(g):
    """按 MHR 这个人的尺寸做一个 proclegs.Body(属性和公式照抄 Body.__init__, 只是数从 MHR 量)。
    布局用 SMPL-X 的关节号(proclegs 里写死的 LEGS): 0 骨盆(这里 = 两髋中点), 1/2 髋, 4/5 膝, 7/8 踝, 10/11 前掌。脚底 y = 0。"""
    PL = _proclegs()
    body = PL.Body.__new__(PL.Body)
    J = np.zeros((22, 3))
    for i, sd in enumerate(SIDES):
        G = g["side"][sd]; h, k, a, toe = PL.LEGS[i]
        v = G["t0"] + G["s0"]
        J[h] = [G["hip"][0], G["ankle_h"] - v[1], 0.0]
        J[k] = J[h] + G["t0"]; J[a] = J[k] + G["s0"]
        bm = (G["loc"]["ball_in"] + G["loc"]["ball_out"]) / 2
        J[toe] = J[a] + np.array([bm[0], 0.0, bm[2]])
    J[0] = [0.0, (J[1, 1] + J[2, 1]) / 2, 0.0]
    body.J = J; body.height = g["height"]
    sole = 0.0
    hip_h = float(np.mean([J[h, 1] for h, _, _, _ in PL.LEGS]) - sole)
    body.H = hip_h * PL.HIP2H
    body.pelvis_h = J[0, 1] - sole
    body.ankle_h = float(np.mean([J[a, 1] - sole for _, _, a, _ in PL.LEGS]))
    body.thigh = [float(np.linalg.norm(J[k] - J[h])) for h, k, _, _ in PL.LEGS]
    body.shin = [float(np.linalg.norm(J[a] - J[k])) for _, k, a, _ in PL.LEGS]
    body.hip_drop = float(np.mean([J[h, 1] - J[a, 1] for h, _, a, _ in PL.LEGS]))
    rel = np.array([J[a] - J[0] for _, _, a, _ in PL.LEGS])
    body.lat = float(np.mean(np.abs(rel[:, 0]))); body.fwd = float(np.mean(rel[:, 2]))
    body.t0 = [J[k] - J[h] for h, k, _, _ in PL.LEGS]
    body.s0 = [J[a] - J[k] for _, k, a, _ in PL.LEGS]
    body.Luse = [body.hip_ankle(i, PL.KNEE_STAND) for i in (0, 1)]
    leg = min(body.Luse)
    body.heel, body.ball = [], []
    for i, sd in enumerate(SIDES):
        G = g["side"][sd]; a = PL.LEGS[i][2]; toe = PL.LEGS[i][3]
        body.heel.append(np.array([J[a, 0] + G["loc"]["heel"][0], sole, J[a, 2] + G["loc"]["heel"][2]]))
        body.ball.append(np.array([J[toe, 0], sole, J[toe, 2]]))
    rise = abs(body.ball[0][2] - J[PL.LEGS[0][2], 2]) * np.sin(PL.REACH_HEEL)
    body.reach = float(np.sqrt(max(leg ** 2 - (body.hip_drop - PL.DROP_MAX - rise) ** 2, 0)))
    body.reach_side = float(np.sqrt(max(leg ** 2 - (body.hip_drop - PL.DROP_MAX) ** 2, 0)))
    rise_b = abs(body.ball[0][2] - J[PL.LEGS[0][2], 2]) * np.sin(PL.HEEL_OFF_MAX)
    body.reach_back = float(np.sqrt(max(leg ** 2 - (body.hip_drop - PL.DROP_MAX - rise_b) ** 2, 0)))
    return body


# ------------------------------------------------------------------ 分类
def runs_of(mask):
    m = np.concatenate([[0], np.asarray(mask, int), [0]]); e = np.flatnonzero(np.diff(m))
    return list(zip(e[::2], e[1::2]))


def merge_short(vis, t, min_s=VIS_MERGE_S):
    """看得见/看不见短于 min_s 秒的一段并进两边(两边都是另一种状态时才并)。"""
    vis = np.asarray(vis, bool).copy()
    for _ in range(2):
        for val in (False, True):
            for a, b in runs_of(vis == val):
                if a == 0 or b == len(vis):
                    continue
                if t[b - 1] - t[a] + (t[1] - t[0] if len(t) > 1 else 0) < min_s:
                    vis[a:b] = not val
    return vis


def classify(feet_ok, t, seated=None):
    vis = merge_short(feet_ok, t)
    if seated is not None and np.mean(seated) > SEATED_SKIP:
        return "坐着", vis
    return ("A" if vis.all() else "B" if not vis.any() else "C"), vis


# ------------------------------------------------------------------ 观测(看得见的帧的腿, C 类用)
def observe(S_np, g, B3, GH, t):
    """MHR 骨架(每帧)-> proclegs make_obs 要的: 踝 (T,2,3) W、脚朝向 (T,2)、俯仰 (T,2, >0 脚跟抬)、踩地 (T,2)。"""
    T = len(S_np)
    ank = np.zeros((T, 2, 3)); fyaw = np.zeros((T, 2)); phi = np.zeros((T, 2)); low = np.zeros((T, 2))
    for i, sd in enumerate(SIDES):
        G = g["side"][sd]
        ank[:, i] = to_W(S_np[:, G["j"]["ank"], :3] / 100.0, B3, GH)
        P = {}
        for k in ("heel", "ball_in", "ball_out", "toe"):   # (mid 只给 IK 用)
            j, L = G["att"][k]
            R = Rot.from_quat(S_np[:, j, 3:7]).as_matrix()
            P[k] = to_W((S_np[:, j, :3] + S_np[:, j, 7:8] * np.einsum("tij,j->ti", R, L)) / 100.0, B3, GH)
        bm = (P["ball_in"] + P["ball_out"]) / 2
        d = bm - P["heel"]
        fyaw[:, i] = np.arctan2(d[:, 1], d[:, 0])
        Lhb = np.linalg.norm(((G["loc"]["ball_in"] + G["loc"]["ball_out"]) / 2 - G["loc"]["heel"]))
        phi[:, i] = np.arcsin(np.clip((P["heel"][:, 2] - bm[:, 2]) / Lhb, -1, 1))
        low[:, i] = np.minimum.reduce([P["heel"][:, 2], bm[:, 2], P["toe"][:, 2]])
    hip_w = np.stack([to_W(S_np[:, g["side"][sd]["j"]["hip"], :3] / 100.0, B3, GH) for sd in SIDES], 1)
    rel_ank = ank - hip_w
    v = np.zeros((T, 2)); v_rel = np.zeros((T, 2))
    if T > 1:
        dt = np.gradient(t)
        v = np.linalg.norm(np.gradient(ank[:, :, :2], axis=0), axis=2) / np.maximum(dt, 1e-6)[:, None]
        v_rel = np.linalg.norm(np.gradient(rel_ank, axis=0), axis=2) / np.maximum(dt, 1e-6)[:, None]
    c = (low < 0.04) & (v < GLIDE_V[0])
    # 定住的姿势跟着身体滑: 两只脚都贴地(< 4 cm)、都在滑、且腿相对髋几乎没动
    glide = ((low < 0.04) & (v >= GLIDE_V[0]) & (v < GLIDE_V[1]) & (v_rel < 0.14)).all(1)
    return ank, np.unwrap(fyaw, axis=0), phi, c, glide


# ------------------------------------------------------------------ 主函数
def turning_intervals(yaw, visible, time, shot):
    """Recognize sustained turns inside continuous, unseen portions of one shot."""
    from scipy.ndimage import gaussian_filter1d
    groups = np.cumsum(np.r_[0, (np.diff(time)>MAX_GAP_S) | (np.diff(shot)!=0)])
    turning = np.zeros(len(time), bool)
    for group in np.unique(groups):
        for start, end in runs_of((groups==group) & ~visible):
            if end-start < 4 or time[end-1]-time[start] < MIN_SEG_S:
                continue
            t = time[start:end]
            dt = np.median(np.diff(t))
            heading = gaussian_filter1d(np.unwrap(yaw[start:end]), max(1.,.08/dt), mode="nearest")
            left = np.interp(t-.25,t,heading)
            right = np.interp(t+.25,t,heading)
            fast = np.abs(right-left) >= np.pi/2
            # Include the approach and release for step planning; leave long
            # unrelated standing or walking intervals on their original path.
            for first,last in runs_of(fast):
                lo = np.searchsorted(t,t[first]-MIN_SEG_S)
                hi = np.searchsorted(t,t[last-1]+MIN_SEG_S,side="right")
                if hi-lo > 1 and t[hi-1]-t[lo] >= MIN_SEG_S:
                    turning[start+lo:start+hi] = True
    return turning


def planned_swing_samples(times, contact, visible, steps):
    """Retain the planner's actual liftoff and landing times across a cropped interval."""
    result = np.full(contact.shape, np.nan)
    for step in steps:
        side, lift, landing = int(step[0]), step[1], step[2]
        if landing <= lift:
            continue
        phase = (times-lift) / (landing-lift)
        moving = (phase>0.) & (phase<1.) & ~contact[:,side] & ~visible
        result[moving,side] = phase[moving]
    return result


def fill_legs(params, shape, t, feet_ok, UP, GH, seated=None, seg=None, dev="cuda", log=print, bad=None, ground_at=None, registered_starts=(), source_time=None, shot=None, thighs=None, has_calibrated_root=False, ankle_seen=None):
    """一个人整段: 补腿。params (T,204) 世界系参数(已整理好位置/朝向), shape (45,), t (T,) 秒(段之间可以有大空档),
    feet_ok (T,) 这帧脚看不看得见, UP/GH 天和地面高, seated (T,) 坐没坐, seg (T,) 段号(镜头 / 坏相机)。
    返回 (新参数, 报告 dict)。A 类 / 坐着 / 段太短: 原样返回。"""
    import torch
    t0_ = time.time()
    params = np.asarray(params, np.float64); T = len(params); t = np.asarray(t, float)
    seg = np.zeros(T, int) if seg is None else np.asarray(seg)
    seg = np.cumsum(np.r_[0, (seg[1:] != seg[:-1]) | (np.diff(t) > MAX_GAP_S)])   # 空档也断开
    if seated is not None and np.mean(seated) > SEATED_SKIP:
        return params, dict(类="坐着", 帧数=T, 不补=f"坐着的帧 {np.mean(seated):.0%} > {SEATED_SKIP:.0%}")
    PL = _proclegs()
    B3 = _basis(UP); up = B3[2]
    g = geometry(shape, params[0], dev)
    with torch.no_grad():
        S_np = skel_torch(torch.tensor(params, dtype=torch.float32, device=dev), dev).cpu().numpy()
    ank_o, fyaw_o, phi_o, c_o, glide = observe(S_np, g, B3, GH, t)
    fo = np.asarray(feet_ok, bool).copy(); n_glide = 0
    for a, b_ in runs_of(fo & glide):
        if t[b_ - 1] - t[a] >= GLIDE_S:
            fo[a:b_] = False; n_glide += b_ - a
    kind, vis = classify(fo, t)
    rep = dict(类=kind, 看得见的帧=int(np.sum(vis)), 帧数=T, 看得见但脚在滑_当看不见的帧=int(n_glide))
    if kind == "A":
        rep["不补"] = "全程看得见"
        return params, rep
    body = make_body(g)
    names = g["names"]
    hipL, hipR = S_np[:, names.index("l_upleg"), :3] / 100, S_np[:, names.index("r_upleg"), :3] / 100
    hc = to_W((hipL + hipR) / 2, B3, GH)
    left = to_W(hipL, B3, 0) - to_W(hipR, B3, 0); left[:, 2] = 0
    fwd = np.cross(left, np.array([0, 0, 1.0]))
    yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))

    # 规划结果(按调用方的帧取回): 踝 W、脚朝向、俯仰、踩地、骨盆高度改动
    tgt = dict(ank=np.full((T, 2, 3), np.nan), fyaw=np.zeros((T, 2)), phi=np.zeros((T, 2)), c=np.zeros((T, 2), bool), dz=np.zeros(T))
    done = np.zeros(T, bool); segrep = []
    swing_phase = np.full((T, 2), np.nan)
    logs = []

    class _Dummy:
        faces = np.zeros((1, 3), np.int32)

    def _fake_bj(model, P):
        n = len(P["transl"]); return np.zeros((n, 25, 3)), np.zeros((n, 1, 3))
    turning = np.zeros(T,bool)
    if source_time is not None and shot is not None:
        source_time = np.asarray(source_time,float)
        shot = np.asarray(shot)
        turning = turning_intervals(yaw, vis, source_time, shot)
        if turning.any():
            # Camera-confidence changes can interrupt position smoothing. A dancer
            # still completes a turn across those boundaries in the same shot.
            split = ((seg[:-1]!=seg[1:]) & ~(turning[:-1]&turning[1:]))
            split |= (turning[:-1]!=turning[1:]) | (shot[:-1]!=shot[1:]) | (np.diff(source_time)>MAX_GAP_S)
            seg = np.cumsum(np.r_[0,split])
            t = source_time
    registered_shots = {shot[first] for first in registered_starts} if shot is not None else set()
    for sid in np.unique(seg):
        idx = np.where(seg == sid)[0]
        if len(idx) < 3 or t[idx[-1]] - t[idx[0]] < MIN_SEG_S:
            # ★ 2026-11-11 太短迈不了步: 看不见腿、没坐着的帧 -> 站着: 两脚在两髋正下方、平踩在地上、脚尖朝骨盆朝向;
            #   骨盆高过"腿站直够得着"就往下压; 坏相机的帧(位置不可信, bad)骨盆直接放到站着的高度(往上往下都行)。
            #   hitch 13~15 秒近景(1.7 秒、又被坏相机切成 4/3/1 帧): 胖子腿出画, 模型猜成半蹲, 相机正常的帧飘 30 cm、坏相机的帧蹲在地上。
            #   逐帧的"坐着"(竖直跨度小)这里不用: 看不见腿的帧跨度是拿猜的腿量的 —— hitch 胖子 14.6/14.8 秒腿猜成半蹲, 跨度 1.43 m
            #   被判成坐着, 一帧飘 33 cm 一帧蹲着没补。这个人坐不坐由上面整个人的判断定(坐着的帧 > SEATED_SKIP 整个人不补)。
            ks = [int(k) for k in idx if not vis[k]]
            if not ks:
                segrep.append(dict(段=int(sid), 帧=len(idx), 不补="太短")); continue
            reach = float(min(body.Luse)) + float(body.ankle_h)              # 站着时两髋中点离地(膝盖按 KNEE_STAND 微弯)
            for k in ks:
                Rk = PL.upright(yaw[k])
                for i in (0, 1):
                    a_ = hc[k] + Rk @ np.array([PL.SIDE[i] * body.lat, 0.0, body.fwd]); a_[2] = body.ankle_h
                    tgt["ank"][k, i] = a_
                tgt["fyaw"][k] = yaw[k]; tgt["phi"][k] = 0.0; tgt["c"][k] = True
                tgt["dz"][k] = (reach - hc[k, 2]) if (bad is not None and bool(np.asarray(bad)[k])) else min(0.0, reach - hc[k, 2])
                done[k] = True
            # A short standing fragment has no planned steps. Keep its foot
            # targets in world XY, releasing the anchor if the hips leave reach.
            for lo, hi in runs_of(np.isin(idx, ks)):
                frames = idx[lo:hi]
                anchor = frames[0]
                for k in frames:
                    if np.linalg.norm(hc[k, :2] - hc[anchor, :2]) > min(body.Luse) * .45:
                        anchor = k
                    tgt["ank"][k, :, :2] = tgt["ank"][anchor, :, :2]
                    tgt["fyaw"][k] = yaw[anchor]
            segrep.append(dict(段=int(sid), 帧=len(idx), 补法="太短, 站着", 站着的帧=len(ks)))
            continue
        if vis[idx].all():
            segrep.append(dict(段=int(sid), 帧=len(idx), 不补="这段全看得见")); continue
        ts = t[idx]
        t30 = np.arange(0.0, ts[-1] - ts[0] + 1e-9, 1.0 / FPS30)
        n30 = len(t30)
        lin = lambda Y: np.stack([np.interp(t30, ts - ts[0], Y[:, j]) for j in range(Y.shape[1])], 1)
        nearest = np.clip(np.searchsorted(ts - ts[0], t30 - 0.5 / FPS30), 0, len(idx) - 1)
        hc30 = lin(hc[idx]); yaw30 = np.interp(t30, ts - ts[0], yaw[idx])
        if turning[idx].all():
            from scipy.ndimage import median_filter, gaussian_filter1d
            # Reject single-frame camera-position impulses before they turn
            # into compulsory steps. Keep the signed unwrapped heading.
            hc30 = gaussian_filter1d(median_filter(hc30,size=(5,1),mode="nearest"),2.,axis=0,mode="nearest")
        go = np.stack([Rot.from_matrix(PL.upright(y)).as_rotvec() for y in yaw30])
        s = np.concatenate([go, hc30 - body.J[0], np.zeros((n30, 10)), np.zeros((n30, 63))], 1)
        obs = None
        if vis[idx].any():
            vis30 = vis[idx][nearest]
            a30 = np.stack([lin(ank_o[idx, i]) for i in (0, 1)], 1)
            obs = PL.make_obs(vis30, c_o[idx][nearest], a30, lin(fyaw_o[idx]), lin(phi_o[idx]))
        oldB, oldbj = PL.Body, PL.body_joints
        PL.Body, PL.body_joints = (lambda model, betas: body), _fake_bj
        try:
            fitted_plant = bool(registered_shots) and shot[idx[0]] in registered_shots and turning[idx].all()
            res = PL.run(s, t30, model=_Dummy(), log=logs.append, obs=obs, join="plan",
                         registered_contacts=fitted_plant)
        except Exception as e:
            # ★ 2026-11-12 规划器的边界情况(jw4 r4_30fps 人 5: 看不见的段开头另一只脚正迈着一步, proclegs 里 StopIteration,
            #   整个摆人那步崩掉; 那个已在 proclegs 修了)。以后再遇到: 这一段不补, 记进报告, 别的段 / 别的人照常。
            log(f"  补腿: 段 {int(sid)} 规划失败({type(e).__name__}: {str(e)[:80]}), 这段不补")
            segrep.append(dict(段=int(sid), 帧=len(idx), 不补=f"规划失败 {type(e).__name__}"))
            continue
        finally:
            PL.Body, PL.body_joints = oldB, oldbj
        k30 = np.clip(np.round((ts - ts[0]) * FPS30).astype(int), 0, n30 - 1)
        tgt["ank"][idx] = res["feet"][k30]; tgt["fyaw"][idx] = res["foot_yaw"][k30]; tgt["phi"][idx] = res["foot_pitch"][k30]
        swing_phase[idx] = planned_swing_samples(t30[k30], res["contact"][k30], vis[idx], res["steps"])
        tgt["c"][idx] = res["contact"][k30]
        dz_seg = (res["pelvis_z_out"] - res["pelvis_z_in"])[k30]
        miss = ~vis[idx]
        tgt["dz"][idx[miss]] = dz_seg[miss]
        done[idx[miss]] = True
        segrep.append(dict(段=int(sid), 帧=len(idx), 秒=round(float(ts[-1] - ts[0]), 2), 步数=int(len(res["steps"])),
                           看得见的帧=int(vis[idx].sum()), 骨盆压低_cm=[round(float(-tgt["dz"][idx].max()) * 100, 1),
                                                                  round(float(-tgt["dz"][idx].min()) * 100, 1)]))
    root_adjust = np.zeros((T, 3))
    # Once camera registration has reconciled the world frames, bridge the
    # unobserved stance at the edit. Preserve observed legs and large real cuts.
    roots = M.root_pos(params) + tgt["dz"][:, None] * up
    for first in registered_starts:
        prev = first - 1
        if prev < 1 or not (done[prev] and done[first]) or vis[first]:
            continue
        residual = roots[prev] - roots[first]
        if np.linalg.norm(residual) > min(body.Luse):
            continue
        # Elapsed time excludes the artificial gaps inserted at bad-camera
        # boundaries. Ramp out over the shot even if the confidence changes.
        velocity = (roots[prev] - roots[prev-1]) / max(t[prev]-t[prev-1], .001)
        velocity *= min(1., 1./max(np.linalg.norm(velocity), .001))
        elapsed = 0.
        cadence = np.median(np.diff(t)[(np.diff(t)>0) & (np.diff(t)<MAX_GAP_S)])
        for k in range(first, T):
            if k > first:
                elapsed += min(t[k]-t[k-1], cadence) if seg[k] != seg[k-1] else t[k]-t[k-1]
            if elapsed > .6 or not done[k] or vis[k]:
                break
            strength = np.exp(-.5*(elapsed/.18)**2)
            root_adjust[k] = strength*(roots[prev]+elapsed*velocity-roots[k])
            for side in range(2):
                if tgt["c"][prev, side]:
                    blend = np.exp(-.5*(max(0.,elapsed-.08)/.12)**2)
                    tgt["ank"][k, side, :2] += blend*(tgt["ank"][prev, side, :2]-tgt["ank"][k, side, :2])
                    # Release heading continuously on the shortest arc. An
                    # abrupt unlock can swing a planted toe across the floor.
                    old_yaw = tgt["fyaw"][prev, side]
                    new_yaw = tgt["fyaw"][k, side]
                    difference = np.arctan2(np.sin(old_yaw-new_yaw),np.cos(old_yaw-new_yaw))
                    tgt["fyaw"][k, side] += blend*difference
    rep["段"] = segrep
    # ★ 2026-11-12 骨盆挪动上限: 补腿是修腿的姿势(悬空 / 半蹲的腿踩回地上, 通常挪几厘米到三四十厘米), 不是搬人。
    #   要挪超过 DZ_MAX 说明脚下的地面不可信(深度自相矛盾 / 地形补错), 这一帧不补, 人留在原位。
    #   joker 3 号镜头: 人在相机下 7 m(身体大小和深度图在人身上都这么说), 同一张深度图里他身后的街在 14 m,
    #   地形按街面补, 补腿把骨盆压低 7.1 m 去贴它, 这个镜头对齐 4.5% -> 324%。
    # Outside a geometrically registered support, a guessed footprint has
    # no independent evidence for changing the height of the whole actor.
    inferred_height = done & ~vis & ~turning & np.asarray(has_calibrated_root, dtype=bool)
    for first in registered_starts:
        segment = np.asarray(shot) if shot is not None else seg
        inferred_height &= segment != segment[first]
    tgt["dz"][inferred_height] = 0.
    big = done & (np.abs(tgt["dz"]) > DZ_MAX)
    if big.any():
        done &= ~big; rep["骨盆要挪太多_不补的帧"] = int(big.sum())
    if not done.any():
        rep["不补"] = "没有能补的段"
        return params, rep

    # ---------------- IK: MHR 腿参数对准规划出的脚
    out = params.copy()
    pz = M.root_pos(out) + tgt["dz"][:, None] * up[None] + root_adjust
    out = M.set_root(out, None, pz)
    tg = np.zeros((T, 2, len(PTS), 3)); fdir = np.zeros((T, 2, 3))
    # ★ 2026-11-12 ground_at(P 世界点 (N,3), 帧) -> 地面高(沿天), 给了才用: 脚的目标按这只脚自己所在位置的地面高摆,
    #   不按骨盆正下方那一个高度(GH[k])。规划是在一块平地上做的; 台阶上脚离骨盆前后 30 cm 就可能是另一级 ——
    #   joker 推特写 133~150 帧脚比交付的台阶低 8~10 cm(一级 11.4 cm), 1 号镜头 10% 分位 -8 cm。
    #   踩着的脚: 脚下那一级的高(整段踩着高度不变); 迈步的脚: 起脚那级 -> 落脚那级按时间过渡(不在半空跳一级)。
    #   骨盆 / 上半身不动(规划和骨盆高度照旧), 只动腿。不给 = 老做法。
    gfoot = None
    if ground_at is not None:
        GHa = np.broadcast_to(np.asarray(GH, float), (T,)).copy(); gfoot = np.full((T, 2), np.nan); kk = np.arange(T)
        for i in (0, 1):
            Gs = g["side"][SIDES[i]]
            for k in np.where(vis & c_o[:, i])[0]:
                sole_v = from_W(ank_o[k, i][None], B3, 0.0)
                gs_v = np.asarray(ground_at(sole_v, k), float).ravel()
                if np.isfinite(gs_v).any():
                    gfoot[k, i] = float(np.nanmax(gs_v))
            for k in np.where(done & tgt["c"][:, i])[0]:
                A = tgt["ank"][k, i]; Rf = PL.upright(tgt["fyaw"][k, i]) @ PL.rx(tgt["phi"][k, i])
                # 整个脚底(脚跟 / 脚心 / 前掌两侧 / 脚尖 + 踝正下方)取最高那一级: 跨在台阶边上时脚平放在上一级边沿、
                #   伸出去的部分悬着(下楼梯前掌搭在台阶边上那样), 不插进上一级。只看踝正下方时 joker 29 帧脚底陷进台阶 > 3 cm,
                #   往下看镜头(下段楼梯一级 16.9 cm)最深 17 cm。
                sole = np.array([A + Rf @ Gs["loc"][nm] for nm in PTS] + [np.asarray(A, float)]); sole[:, 2] = 0.0
                gs = np.asarray(ground_at(from_W(sole, B3, 0.0), k), float).ravel()
                g_ = float(np.nanmax(gs)) if np.isfinite(gs).any() else np.nan
                if np.isfinite(g_) and g_ < GHa[k]:
                    # 往下一级: 骨盆不动、腿伸得到(髋-踝 <= Luse)的才放下去。规划按平地定骨盆, 踩地的腿几乎是直的 ——
                    #   第一版不管够不够得着, joker 3 号镜头 8 帧脚悬在下一级上方 11 cm、IK 95% 分位 2.6 -> 6.3 cm。
                    hp = hc[k] + PL.upright(yaw[k]) @ np.array([PL.SIDE[i] * body.lat, 0.0, 0.0])
                    dmax = np.sqrt(max(body.Luse[i] ** 2 - float(np.sum((A[:2] - hp[:2]) ** 2)), 0.0)) - (hp[2] + tgt["dz"][k] - A[2])
                    g_ = GHa[k] - min(GHa[k] - g_, max(dmax, 0.0))
                gfoot[k, i] = g_
            stance = ((done & tgt["c"][:, i]) | (vis & c_o[:, i])) & np.isfinite(gfoot[:, i])
            for k in np.where(done & ~tgt["c"][:, i])[0]:
                same = stance & (seg == seg[k]); a = np.where(same & (kk < k))[0]; b_ = np.where(same & (kk > k))[0]
                if len(a) and len(b_):
                    gfoot[k, i] = float(np.interp(t[k], [t[a[-1]], t[b_[0]]], [gfoot[a[-1], i], gfoot[b_[0], i]]))
                elif len(a) or len(b_):
                    gfoot[k, i] = gfoot[a[-1] if len(a) else b_[0], i]
            nf = ~np.isfinite(gfoot[:, i]); gfoot[nf, i] = GHa[nf]
    for i, sd in enumerate(SIDES):
        G = g["side"][sd]
        for k in np.where(done)[0]:
            C = PL.upright(tgt["fyaw"][k, i]); R = C @ PL.rx(tgt["phi"][k, i]); A = tgt["ank"][k, i]
            for pi, nm in enumerate(PTS):
                q = A + R @ G["loc"][nm]
                if nm == "toe" and tgt["c"][k, i] and tgt["phi"][k, i] > 0:   # 抬脚跟时脚趾留在地上(同 proclegs)
                    bm = (G["loc"]["ball_in"] + G["loc"]["ball_out"]) / 2
                    q = A + R @ bm + C @ (G["loc"]["toe"] - bm)
                tg[k, i, pi] = from_W(q, B3, gfoot[k, i] if gfoot is not None else (GH if np.ndim(GH) == 0 else GH[k]))
            fdir[k, i] = (C @ np.array([0, 0, 1.0])) @ B3
    cols = LEG_COLS["l"] + LEG_COLS["r"]
    base = torch.tensor(out, dtype=torch.float32, device=dev)
    x = torch.tensor(out[:, cols], dtype=torch.float32, device=dev, requires_grad=True)
    m_done = torch.tensor(done, device=dev)
    TG = torch.tensor(tg, dtype=torch.float32, device=dev); FD = torch.tensor(fdir, dtype=torch.float32, device=dev)
    att = [[(g["side"][sd]["att"][nm][0], torch.tensor(g["side"][sd]["att"][nm][1][None], dtype=torch.float32, device=dev))
            for nm in PTS] for sd in SIDES]
    jn = [g["side"][sd]["j"] for sd in SIDES]
    if thighs is not None:
        from leg_observations import projected_thigh_error
        th = {k: torch.as_tensor(value, dtype=torch.float32, device=dev) for k,value in thighs.items()}
        hip_ids = [leg["hip"] for leg in jn]
        knee_ids = [leg["knee"] for leg in jn]
    wp = torch.tensor(PT_W, dtype=torch.float32, device=dev)
    same = torch.tensor(np.r_[False, seg[1:] == seg[:-1]] & (done | np.r_[False, done[:-1]]), device=dev)
    kcol = [cols.index(KNEE_COL[sd]) for sd in SIDES]
    twist_cols = [cols.index(c) for c in BALL_TWIST]
    side_col_idx = [[cols.index(c) for c in LEG_COLS[sd]] for sd in SIDES]
    seen_leg = None
    if ankle_seen is not None:
        seen_leg = np.asarray(ankle_seen, bool) & ~glide[:, None] & (np.abs(tgt["dz"]) < 0.015)[:, None]
    distal_indices, source_foot, foot_slack = distal_prior(base,cols)
    opt = torch.optim.Adam([x], lr=IK_LR)

    def full(xx):
        P = base.clone(); P[:, cols] = xx; return P
    for it in range(IK_ITERS):
        opt.zero_grad()
        S = skel_torch(full(x), dev)
        loss = 0.0
        if thighs is not None:
            knee_error = projected_thigh_error(S[:,hip_ids,:3]/100., S[:,knee_ids,:3]/100.,
                                               th["projection"],th["size"],th["relative"])
            loss = 120000.*(knee_error*th["confidence"])[m_done].mean()
        for i in (0, 1):
            pts = torch.cat([_place(S, j, L) for j, L in att[i]], 1)                   # (T,5,3) 米
            e = ((pts - TG[:, i]) * 100.0).pow(2).sum(-1)                               # cm^2
            footprint = (e * wp).sum(-1)
            if thighs is not None:
                # The foot is inferred. It can complete the unseen shin
                # but must yield to a measured, moving femur.
                footprint = footprint * (1.-.90*th["confidence"][:,i])
            loss = loss + footprint[m_done].mean()
            hip, knee, ank = S[:, jn[i]["hip"], :3] / 100, S[:, jn[i]["knee"], :3] / 100, S[:, jn[i]["ank"], :3] / 100
            u = ank - hip; u = u / u.norm(dim=-1, keepdim=True)
            d = (knee - hip); d = d - (d * u).sum(-1, keepdim=True) * u
            n1 = FD[:, i] - (FD[:, i] * u).sum(-1, keepdim=True) * u; n1 = n1 / (n1.norm(dim=-1, keepdim=True) + 1e-6)
            kd = ((d - d.norm(dim=-1, keepdim=True) * n1) * 100.0).pow(2).sum(-1)
            loss = loss + W_KNEE * kd[m_done].mean()
            loss = loss + W_HYPER * torch.relu(-x[:, kcol[i]]).pow(2)[m_done].mean()
        deviation = (x[:,distal_indices]-source_foot).abs()
        loss += 180*torch.relu(deviation-foot_slack).square()[m_done].mean()
        if T > 2:
            d2 = x[2:] - 2 * x[1:-1] + x[:-2]
            ok2 = same[2:] & same[1:-1]
            if ok2.any():
                loss = loss + W_SMOOTH * (d2.pow(2).sum(-1) * 100.0)[ok2].mean()
        loss.backward()
        with torch.no_grad():
            x.grad[~m_done] = 0
            if seen_leg is not None:
                for i in (0, 1):
                    if seen_leg[:, i].any():
                        x.grad[torch.as_tensor(seen_leg[:, i], device=dev)[:, None] & torch.isin(torch.arange(len(cols), device=dev), torch.tensor(side_col_idx[i], device=dev))[None, :]] = 0
            # Keep the observed metatarsal twist while solving heel/toe roll.
            x.grad[:,twist_cols] = 0
        opt.step()
    with torch.no_grad():
        S = skel_torch(full(x), dev)
        err = []
        for i in (0, 1):
            pts = torch.cat([_place(S, j, L) for j, L in att[i]], 1)
            err.append(((pts - TG[:, i]) * 100.0).norm(dim=-1).cpu().numpy())
        err = np.stack(err, 1)[done]                                                       # (n,2,5) cm
    out[:, cols] = x.detach().cpu().numpy()
    if seen_leg is not None:
        for i, sd in enumerate(SIDES):
            mk = done & seen_leg[:, i]
            if mk.any():
                out[np.ix_(mk, LEG_COLS[sd])] = params[np.ix_(mk, LEG_COLS[sd])]
    out[~done] = params[~done]
    rep.update(IK_点误差_cm=dict(中位=round(float(np.median(err)), 2), p95=round(float(np.percentile(err, 95)), 2),
                                最大=round(float(err.max()), 2)),
               骨盆压低_cm=dict(中位=round(float(-np.median(tgt["dz"][done])) * 100, 1), 最大=round(float(-tgt["dz"][done].min()) * 100, 1)),
               补的帧=int(done.sum()), 秒=round(time.time() - t0_, 1))
    rep["_规划"] = dict(contact=tgt["c"], done=done, vis=vis, sole_world=tg, turning=turning & done, swing_phase=swing_phase)                                       # 给检查用(不进 JSON)
    log(f"  补腿: {kind} 类, 补 {int(done.sum())}/{T} 帧, 步数 {[s_.get('步数') for s_ in segrep]}, "
        f"IK 点误差 中位 {rep['IK_点误差_cm']['中位']} cm 最大 {rep['IK_点误差_cm']['最大']} cm, "
        f"骨盆压低 最大 {rep['骨盆压低_cm']['最大']} cm, {rep['秒']} s")
    return out, rep


# ------------------------------------------------------------------ 检查(数字)
def leg_metrics(params, shape, t, UP, GH, contact, seg=None, use=None, dev="cuda"):
    """补腿前后都用这一套量: 真网格(带姿势修正项)。contact (T,2) 用补腿规划出的踩地(两版用同一份, 才可比)。
    use (T,) 只量这些帧(补了的帧)。滑动: 踩地(连续两帧都踩、同一段)的脚, 第 k 帧最低的那个顶点到 k+1 帧的水平速度 cm/s。
    悬空: 踩地帧脚底最低点离地 cm。穿地: 所有帧脚最低点低于地面 > 1 cm 的帧数。膝角: 髋-膝-踝夹角(180 = 直)。"""
    V, S = M.forward(shape, params, True, dev)
    sk = M.skeleton(dev); names = [str(n) for n in sk["names"]]
    o = np.lexsort((-sk["sk_w"], sk["sk_v"])); v_, j_ = sk["sk_v"][o], sk["sk_j"][o]
    first = np.r_[True, v_[1:] != v_[:-1]]; jmax = np.full(V.shape[1], -1); jmax[v_[first]] = j_[first]
    par = sk["parents"].astype(int)
    up = np.asarray(UP, float) / np.linalg.norm(UP)
    seg = np.zeros(len(t), int) if seg is None else np.asarray(seg)
    outd = {}
    for i, sd in enumerate(SIDES):
        root = names.index(f"{sd}_talocrural"); fj = {root}
        for _ in range(10):
            fj |= {k for k, p_ in enumerate(par) if p_ in fj}
        fv = np.where(np.isin(jmax, list(fj)))[0]
        F = V[:, fv]; h = F @ up - (np.asarray(GH, float)[:, None] if np.ndim(GH) else GH)
        lowk = np.argmin(h, 1); low = h[np.arange(len(h)), lowk]
        c = np.asarray(contact[:, i], bool) & (np.ones(len(t), bool) if use is None else np.asarray(use, bool))
        pair = c[1:] & c[:-1] & (seg[1:] == seg[:-1])
        # 滑动: 第 k 帧最低的那个顶点, 到 k+1 帧水平挪了多少(抬脚跟时最低点从脚跟换到前掌, 不算滑)
        a0 = F[np.arange(len(h) - 1), lowk[:-1]]; a1 = F[np.arange(1, len(h)), lowk[:-1]]
        dd = a1 - a0; dd -= np.outer(dd @ up, up)
        sp = np.linalg.norm(dd, axis=1) / np.maximum(np.diff(t), 1e-6) * 100
        allm = np.ones(len(t), bool) if use is None else np.asarray(use, bool)
        hip, knee, ank = (S[:, names.index(f"{sd}_{n}"), :3] / 100 for n in ("upleg", "lowleg", "talocrural"))
        a1, a2 = hip - knee, ank - knee
        kang = np.degrees(np.arccos(np.clip((a1 * a2).sum(1) / np.linalg.norm(a1, axis=1) / np.linalg.norm(a2, axis=1), -1, 1)))
        outd[sd] = dict(滑动_cms=[round(float(np.median(sp[pair])), 1), round(float(np.percentile(sp[pair], 95)), 1)] if pair.any() else None,
                        踩地帧悬空_cm=[round(float(np.median(low[c]) * 100), 1), round(float(np.percentile(low[c], 95) * 100), 1)] if c.any() else None,
                        穿地帧_gt1cm=int((low[allm] < -0.01).sum()), 最低_cm=round(float(low[allm].min() * 100), 1),
                        膝角=[round(float(kang[allm].min()), 0), round(float(np.median(kang[allm])), 0), round(float(kang[allm].max()), 0)],
                        _low=low)
    allm = np.ones(len(t), bool) if use is None else np.asarray(use, bool)
    both_air = (outd["l"]["_low"] > 0.08) & (outd["r"]["_low"] > 0.08) & allm
    for sd in SIDES:
        outd[sd].pop("_low")
    outd["两脚都离地gt8cm_帧"] = int(both_air.sum())
    return outd, V, S


# ------------------------------------------------------------------ 看不见的脚在地下: 身体不动, 只把腿弯上来 (2026-11-12)
# ★ 起因 jw4 甄子丹坐地上的特写(4.8~11.6 秒): 腿出画, 他坐着(坐着的帧 48% > SEATED_SKIP), fill_legs 不补;
#   模型猜的脚在地下 6~54 cm。以前地面约束按"最低点"把整个人往上抬 -> 每帧抬得不一样(上下跳)、人投到画面上方
#   (落在分割上 0.7 -> 0.1~0.2)。身体的位置是看着估的、是对的, 错的只是猜出来的腿 —— 所以修腿, 不挪人。
# 规则(都是之前踩过的坑):
#   - 只动"脚不可信"的帧: 看不见、也不是 fill_legs 按脚印摆的(调用方给 active)。看得见的腿是观测(lovelynight 脱鞋、
#     翘二郎腿), 补腿摆好的腿已经踩在地上(hitch 13 秒), 都不碰。
#   - 只改两条腿的参数(LEG_COLS); 根位置、朝向、上半身一个数不动(挪人 = 和原图对不齐: jw4 上下跳, joker 推特写 2.9% -> 178%)。
#   - 最小改动: 脚刚好出地面就停(不是摆成站姿 / 盘腿), 改动按时间连贯(罚的是"改动量"的帧间差, 原来的动作细节留着)。
#   - 像人那样收腿: 脚只往上收(水平位置罚), 靠前后弯髋 + 弯膝(髋的 rz、膝); 往外劈腿(髋 ry)、扭大腿(髋 rx)、扭脚贵。
#     第一版只罚"改动量", 假人整个放低 25 cm 时它选了两条腿各往外劈 37° —— 参数改得最少, 但不是人会做的。
#   - 要抬超过 FOOT_LIFT_MAX 的帧不修: 那么深不是腿猜错, 是地面高或人的位置错了(joker 台阶脚印比平面高 3.4 m), 不拿腿去凑。
#   - 只管"在地下", 不管"悬空": 脚不可信时判不了是不是腾空(goodfellas 人 4 只拍到上半身)。
FOOT_LIFT_MAX = 0.7    # 米: 脚要抬这么多以上的帧不修
CLAMP_TOL = 0.02       # 米: 脚在地下这么多以内不管(同 body_motion.GROUND_TOL)
CLAMP_ITERS = 300
W_CL_REG = 50.0        # 改动量(弧度^2)的权重: 脚出地面之后就靠它停下
W_CL_DT = 200.0        # 改动量帧间差(弧度^2)的权重
W_CL_HOR = 1.0         # 脚(5 个脚底点)水平挪动(cm^2)的权重: 脚往上收, 不往前后左右甩
# 每列改动的相对代价(LEG_COLS 的顺序): 髋 rx(扭) / ry(外展) / rz(前后弯), 膝, 脚 rx/ry(扭), 踝 rz(勾脚) / rx, 前掌 rz / rx
#   (2026-11-12 量过: 髋 ry +30° 膝盖往外 21 cm, rz +30° 膝盖往后 21 cm, rx 只扭不挪)
CL_COL_W = np.array([20.0, 20.0, 1.0, 1.0, 20.0, 20.0, 2.0, 20.0, 5.0, 5.0])
KNEE_MAX = np.radians(150)
HIP_EXT_MAX = np.radians(20)   # 髋 rz 正 = 大腿往后(伸髋); 往后最多这么多(原来就更往后的不管)
W_CL_PEN = 10.0        # 脚在地下(cm^2)的权重
W_CL_LIM = 1e5         # 关节限位(弧度^2): 必须压过 cm^2 那几项 —— 第二版用 W_HYPER(100)时, 为了让脚不往前甩,
                       #   它把膝盖反弯 81°、大腿往后 44°(Z 字形)


def pin_registered_support(params, shape, vf, starts, feet, up, terrain, fps, dev, planned=None, support_side=None):
    """Close residual stance error using the skinned sole, after approximate joint IK."""
    pose = params.copy()
    for first in starts:
        if first < 2: continue
        rows = np.flatnonzero((vf >= vf[first]-2) & (vf <= vf[first]+round(.35*fps)))
        V, _ = M.forward(shape, pose[rows], True, dev)
        cut = int(np.flatnonzero(rows == first)[0])
        drift = [np.linalg.norm((V[cut-1,ix]-V[cut-2,ix]).mean(0)) for ix in feet]
        level = [np.min(V[cut-1,ix]@up - terrain.model_h(V[cut-1,ix])) for ix in feet]
        side = int(support_side) if support_side is not None else int(np.argmin(np.abs(level)+np.array(drift)))
        if abs(level[side]) > .03 or drift[side]*fps > .35: continue
        cols = LEG_COLS[SIDES[side]][:4]
        ix = feet[side]
        anchor = V[cut-1,ix].mean(0)
        for j in range(cut, len(rows)):
            k = rows[j]; dt = (vf[k]-vf[first])/fps
            if planned is not None and not planned[k]: break
            sole = V[j,ix]; origin = sole.mean(0)
            if np.linalg.norm(origin-anchor) > .5: break
            weight = np.exp(-.5*(max(0.,dt-.08)/.09)**2)
            target = origin + weight*(anchor-origin)
            def residual(v):
                a = target-v.mean(0)
                gap = np.min(v@up-terrain.model_h(v))
                return a - (a@up + gap)*up
            state = pose[k].copy(); err = residual(sole)
            for _ in range(10):
                if np.linalg.norm(err) < .002: break
                probes = np.repeat(state[None],len(cols),axis=0)
                probes[np.arange(len(cols)),cols] += .003
                verts,_ = M.forward(shape,probes,True,dev)
                jac = np.column_stack([(err-residual(v[ix]))/.003 for v in verts])
                step = jac.T @ np.linalg.solve(jac@jac.T+.0004*np.eye(3), err)
                trials = np.repeat(state[None],3,axis=0)
                trials[:,cols] += np.outer([1.,.5,.25],np.clip(step,-.2,.2))
                trials[:,cols[-1]]=np.clip(trials[:,cols[-1]],0.,KNEE_MAX)
                vs,_ = M.forward(shape,trials,True,dev)
                errors=np.array([residual(v[ix]) for v in vs])
                best=int(np.argmin(np.linalg.norm(errors,axis=1)))
                if np.linalg.norm(errors[best]) >= np.linalg.norm(err): break
                state,err=trials[best],errors[best]
            pose[k]=state
    return pose


def clear_planned_feet(params, shape, vertices, planned, sides, up, surface, dev):
    """Resolve residual collisions after IK against the exported collider.
    Search only hip/knee flexion. Evaluate the skinned sole so stair edges,
    ankle rotation and pose correctives are included in the contact test.
    """
    out = params.copy()
    world = vertices.copy()
    for sd, ix in zip(SIDES, sides):
        hip, knee = LEG_COLS[sd][2], KNEE_COL[sd]
        for k in np.flatnonzero(planned):
            sole = world[k, ix]
            gap = float(np.min(sole @ up - surface(sole)))
            if not -FOOT_LIFT_MAX <= gap < -CLAMP_TOL:
                continue
            # Coarse-to-fine search with the original pose as a candidate.
            base = out[k].copy()
            best = base.copy()
            def cost(v, angles):
                clear = np.min(v @ up - surface(v))
                displacement = np.mean(np.linalg.norm(v - sole, axis=1)**2)
                return (100 * max(-clear - .005, 0.)**2
                        + 100 * max(clear - .01, 0.)**2 + displacement + .002 * np.sum(angles**2))
            score = cost(sole, np.zeros(2))
            for step in np.radians([15., 5., 1.5]):
                moves = np.array([(h, n) for h in range(-2, 3) for n in range(-2, 3)]) * step
                trials = np.repeat(best[None], len(moves), axis=0)
                trials[:, [hip, knee]] += moves
                valid = ((trials[:, knee] >= min(base[knee], 0.))
                         & (trials[:, knee] <= max(base[knee], KNEE_MAX))
                         & (trials[:, hip] <= max(base[hip], HIP_EXT_MAX)))
                trials = trials[valid]
                if not len(trials):
                    continue
                verts, _ = M.forward(shape, trials, True, dev)
                for trial, v in zip(trials, verts):
                    s = cost(v[ix], (trial - base)[[hip, knee]])
                    if s < score:
                        score, best = s, trial.copy()
                        world[k] = v
            out[k] = best
    return out, world


def clamp_feet(params, shape, active, UP, GH, seg=None, dev="cuda", tol=CLAMP_TOL, lift_max=FOOT_LIFT_MAX):
    """params (T,204) 世界系; active (T,) 脚不可信的帧; GH 标量或 (T,)(有地形时每帧人脚下的地面高)。
    返回 (params, report)。只改 active 帧、且所在那一段(连续的 active 帧)里有脚在地下的; 只改两条腿的参数。"""
    import torch
    params = np.asarray(params, np.float64); T = len(params)
    active = np.asarray(active, bool)
    seg = np.zeros(T, int) if seg is None else np.asarray(seg)
    up = np.asarray(UP, float) / np.linalg.norm(UP)
    G = np.broadcast_to(np.asarray(GH, float), (T,)).copy()
    rep = dict(改的帧=0, 太深不修的帧=0)
    if not active.any():
        return params, rep
    g = geometry(shape, params[0], dev)
    att = [[(g["side"][sd]["att"][nm][0], torch.tensor(g["side"][sd]["att"][nm][1][None], dtype=torch.float32, device=dev))
            for nm in PTS] for sd in SIDES]
    U = torch.tensor(up, dtype=torch.float32, device=dev); Gt = torch.tensor(G, dtype=torch.float32, device=dev)

    def feet(P_t):
        S = skel_torch(P_t, dev)
        return torch.cat([torch.cat([_place(S, j, L) for j, L in att[i]], 1) for i in (0, 1)], 1)  # (T,10,3) 米

    def heights(P_t):
        return feet(P_t) @ U - Gt[:, None]                                                           # (T,10)
    with torch.no_grad():
        F0 = feet(torch.tensor(params, dtype=torch.float32, device=dev))
        h0 = (F0 @ U - Gt[:, None]).cpu().numpy()
    need = -h0.min(1)                                                                                # 最低那个点要抬多少
    deep = active & (need > lift_max)
    below = active & ~deep & (need > tol)
    rep["太深不修的帧"] = int(deep.sum())
    # 要改的帧 = 连续的 active(不含太深的)段里, 有脚在地下的那些段
    act = active & ~deep
    opt = np.zeros(T, bool); i = 0
    while i < T:
        if not act[i]:
            i += 1; continue
        j = i
        while j + 1 < T and act[j + 1] and seg[j + 1] == seg[i]:
            j += 1
        if below[i:j + 1].any():
            opt[i:j + 1] = True
        i = j + 1
    if not opt.any():
        return params, rep
    cols = LEG_COLS["l"] + LEG_COLS["r"]
    kcol = [cols.index(KNEE_COL[sd]) for sd in SIDES]
    hcol = [cols.index(LEG_COLS[sd][2]) for sd in SIDES]                                             # 髋 rz(前后弯)
    base = torch.tensor(params, dtype=torch.float32, device=dev)
    x0 = base[:, cols].clone()
    dx = torch.zeros_like(x0, requires_grad=True)
    m_opt = torch.tensor(opt, device=dev)
    pair = torch.tensor(np.r_[False, (seg[1:] == seg[:-1]) & (opt[1:] | opt[:-1])], device=dev)      # 这一帧和前一帧之间
    cw = torch.tensor(np.concatenate([CL_COL_W, CL_COL_W]), dtype=torch.float32, device=dev)
    adam = torch.optim.Adam([dx], lr=0.05)
    for it in range(CLAMP_ITERS):
        adam.zero_grad()
        P = base.clone(); P[:, cols] = x0 + dx * m_opt[:, None]
        Fp = feet(P)
        h = (Fp @ U - Gt[:, None]) * 100.0                                                           # cm
        loss = W_CL_PEN * torch.relu(-h).pow(2).sum(-1)[m_opt].sum()
        dF = (Fp - F0) * 100.0; dF = dF - (dF @ U)[..., None] * U                                    # 水平挪动, cm
        loss = loss + W_CL_HOR * dF.pow(2).sum(-1).sum(-1)[m_opt].sum()
        loss = loss + W_CL_REG * (dx.pow(2) * cw).sum(-1)[m_opt].sum()
        d1 = (dx * m_opt[:, None])[1:] - (dx * m_opt[:, None])[:-1]
        loss = loss + W_CL_DT * d1.pow(2).sum(-1)[pair[1:]].sum()
        kn = x0[:, kcol] + dx[:, kcol]
        k0 = torch.clamp(x0[:, kcol], min=0.0)                                                      # 原来就反弯的不往回罚(只罚变得更反)
        lim = torch.relu(k0 - kn).pow(2) + torch.relu(kn - KNEE_MAX).pow(2)
        hz = x0[:, hcol] + dx[:, hcol]; h0z = torch.clamp(x0[:, hcol], min=float(HIP_EXT_MAX))
        lim = lim + torch.relu(hz - h0z).pow(2)
        loss = loss + W_CL_LIM * lim.sum(-1)[m_opt].sum()
        loss.backward()
        adam.step()
    out = params.copy()
    out[np.ix_(opt, cols)] = (x0 + dx).detach().cpu().numpy()[opt]
    with torch.no_grad():
        h1 = heights(torch.tensor(out, dtype=torch.float32, device=dev)).cpu().numpy()
    rep.update(改的帧=int(opt.sum()), 脚在地下的帧=int(below.sum()),
               要抬_cm=dict(中位=round(float(np.median(need[below])) * 100, 1), 最大=round(float(need[below].max()) * 100, 1)) if below.any() else None,
               改后最低_cm=round(float(h1[opt].min()) * 100, 1),
               膝盖改动_度=round(float(np.degrees(np.abs(out[opt][:, [KNEE_COL["l"], KNEE_COL["r"]]] - params[opt][:, [KNEE_COL["l"], KNEE_COL["r"]]]).max())), 1))
    return out, rep
