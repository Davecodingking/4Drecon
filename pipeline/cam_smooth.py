#!/usr/bin/env python
"""相机轨迹时间平滑(2026-11-11, irish / spl 上用户两次说镜头抖)。DA3 逐帧估相机, 估计本身在抖:
  spl 位置抖 6.8 cm(中位, p90 20.7 cm), 比相机每步真实移动 5.0 cm 还大; irish 相机几乎不动(每步 1.1 cm)却抖 3.8 cm。
  渲染回图 / 验收视频 / FBX 里的相机都直接用它, 所以画面在抖, 原片不抖。
★ 平滑前验证过不是"深度已经把相机的抖抵消了"(那样只平滑相机会让场景抖起来): 画面左右两边静止的像素用同一份深度反投影,
  平滑相机后世界坐标反而更稳 —— irish 5.1 -> 4.2 cm, spl 20.8 -> 18.7 cm, singin 3.0 -> 2.8 cm(scout_1110/cam_smooth_test.py)。

做法: 每个镜头内(切点读 shot_cuts.json, 不跨镜头)、跳过坏相机帧(bad_frames.json, 保持原样), 相机中心和朝向(四元数)
各做高斯平滑, sigma = CAM_SMOOTH 个 DA3 帧(DA3 是隔几帧抽的, 抖动是逐个样本的白噪声, 按样本数定窗口)。
只改 cameras.json 的外参, 内参 / 深度不动。原件存 cameras_unsmoothed.json; 已经平滑过(标记 相机平滑)就跳过, 不叠加平滑。
"""
import argparse, json, os, shutil, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("--r1", required=True)
ap.add_argument("--sigma", type=float, default=2.0)
ap.add_argument("--redo", action="store_true", help="从 cameras_unsmoothed.json 重新平滑")
a = ap.parse_args()
D = f"{a.r1}/02_da3"; p = f"{D}/cameras.json"
if a.redo and os.path.exists(f"{D}/cameras_unsmoothed.json"):
    shutil.copy(f"{D}/cameras_unsmoothed.json", p)
doc = json.load(open(p))
if a.sigma <= 0:
    print("[相机平滑] CAM_SMOOTH=0, 不平滑"); sys.exit(0)
if doc.get("相机平滑") and not a.redo:
    print(f"[相机平滑] 已经平滑过({doc['相机平滑']}), 跳过 —— 要重来先从 cameras_unsmoothed.json 还原"); sys.exit(0)
if not os.path.exists(f"{D}/cameras_unsmoothed.json"):
    shutil.copy(p, f"{D}/cameras_unsmoothed.json")
cams = doc["cameras"]
fm = json.load(open(f"{D}/frame_map.json")); fps = fm.get("fps") or 25.0; vf = fm["frames"]
cuts = json.load(open(f"{D}/shot_cuts.json"))["cuts"] if os.path.exists(f"{D}/shot_cuts.json") else []
bad = set(json.load(open(f"{D}/bad_frames.json")).get("坏帧", [])) if os.path.exists(f"{D}/bad_frames.json") else set()
allk = sorted(cams, key=lambda k: vf[k])
keys = [k for k in allk if k not in bad]
shot = np.searchsorted(cuts, np.array([vf[k] / fps for k in keys]) + 1e-3, side="right")
shot_all = dict(zip(allk, np.searchsorted(cuts, np.array([vf[k] / fps for k in allk]) + 1e-3, side="right")))
moved_t, moved_r, n = [], [], 0
for s in sorted(set(shot)):
    ks = [k for k, q in zip(keys, shot) if q == s]
    if len(ks) < 3:
        continue
    E = np.array([cams[k]["extrinsic_w2c_4x4"] for k in ks], float)
    R = E[:, :3, :3]; C = -np.einsum("nji,nj->ni", R, E[:, :3, 3])
    q = Rot.from_matrix(R).as_quat()
    for i in range(1, len(q)):                              # 四元数同号, 平均才有意义
        if q[i] @ q[i - 1] < 0: q[i] = -q[i]
    Cs = gaussian_filter1d(C, a.sigma, axis=0, mode="nearest")
    qs = gaussian_filter1d(q, a.sigma, axis=0, mode="nearest"); qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    Rs = Rot.from_quat(qs).as_matrix()
    for k, Ri, Ci, R0, C0 in zip(ks, Rs, Cs, R, C):
        M = np.eye(4); M[:3, :3] = Ri; M[:3, 3] = -Ri @ Ci
        cams[k]["extrinsic_w2c_4x4"] = M.tolist()
        if cams[k].get("camera_center_world") is not None:
            cams[k]["camera_center_world"] = Ci.tolist()
        moved_t.append(float(np.linalg.norm(Ci - C0))); moved_r.append(float(np.degrees((Rot.from_matrix(Ri).inv() * Rot.from_matrix(R0)).magnitude())))
        n += 1
# ★ 坏帧: 同一镜头里前后好帧(已平滑)插值 —— 原样留着的话和邻帧一跳 10~40 cm(spl 第 2~5 秒)
#   只补"两头都有好帧、两头相隔不超过 MAX_GAP_S 秒"的; 其余(镜头开头 / 结尾整段坏, 或者坏得太长)保持原样。
#   ★ 11-11 第一版对只有一头有好帧的坏帧直接抄那一头的相机: joker 2 号镜头开头推特写 39 帧全坏, 相机被冻在一个位置,
#     人摆偏 600 px(占身高 22%); irish 结尾 18 帧同样被冻住。
MAX_GAP_S = 1.2   # spl 6 号镜头 277→302 帧缺口 1.04 s, 插值后已验收过
HOLD_S = 1.2        # 镜头头 / 尾的坏帧离最近好帧不超过这么久(s): 抄那一帧的相机(spl 2 号镜头尾 0.6 s); 更长 = 下面按段匀速拟合
#   ★ 11-12 0.6 -> 1.2(和 MAX_GAP_S 一样): irish 2 号镜头开头 7 个连续坏帧(291~310 帧, 0.63 s), 294~310 抄了好帧的相机,
#   291 离好帧 0.73 s 没抄、也不够 4 帧拟合, 留着估错的位置, 和下一帧一跳 66 cm。同一串坏帧要么一起抄, 要么一起拟合。
from scipy.spatial.transform import Slerp
nb = nkeep = 0
E_ = lambda q: np.array(cams[q]["extrinsic_w2c_4x4"], float)


def _set(k, R, Cc):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = -R @ Cc
    cams[k]["extrinsic_w2c_4x4"] = M.tolist()
    if cams[k].get("camera_center_world") is not None: cams[k]["camera_center_world"] = Cc.tolist()


# ★ 2026-11-12 补不了的长坏段按这一段自己的相机拟合一条匀速运动: 位置、朝向(相对平均朝向的旋转向量)、焦距(取对数)对时间
#   直线拟合, 残差超过 3 倍 MAD 的帧不参与(迭代 3 次)。joker 2 号镜头(3.64~6.71 秒, 推特写): 脸占满画面、背景虚,
#   VGGT-Ω 逐帧位置跳 40 cm、朝向跳 2.5°、焦距跳 1.6%, 原样留着整段在抖; 用户看过原片, 这段场景几乎不动。
#   不抄别的镜头的相机(11-11 那样做过, 人摆偏 22%)。人是按每帧相机摆的, 换相机后人跟着走, 投回原图不偏。
#   旁边有好帧的(anchors)当锚点(权重 ANCHOR_W)一起拟合, 拟合出来的线接得上好帧; 锚点自己不改。
LONG_MIN = 4        # 没有锚点时, 一段至少这么多帧才拟合
ANCHOR_W = 5


def _fit_run(ks, anchors=()):
    allk_ = list(ks) + [q for q in anchors for _ in range(ANCHOR_W)]
    isA = np.r_[np.zeros(len(ks), bool), np.ones(len(allk_) - len(ks), bool)]
    t = np.array([vf[k] / fps for k in allk_], float); t = t - t[~isA].mean(); X = np.c_[np.ones_like(t), t]
    E = np.array([cams[k]["extrinsic_w2c_4x4"] for k in allk_], float); R0 = E[:, :3, :3]
    C = -np.einsum("nji,nj->ni", R0, E[:, :3, 3])
    Rm = Rot.from_matrix(R0[~isA]).mean(); rv = (Rm.inv() * Rot.from_matrix(R0)).as_rotvec()
    lf = np.log([cams[k]["intrinsic_3x3"][0][0] for k in allk_])
    keep = np.ones(len(allk_), bool)
    for _ in range(3):
        bc = np.linalg.lstsq(X[keep], C[keep], rcond=None)[0]; br = np.linalg.lstsq(X[keep], rv[keep], rcond=None)[0]
        rc = np.linalg.norm(C - X @ bc, axis=1); rr = np.linalg.norm(rv - X @ br, axis=1)
        kr = keep & ~isA
        k2 = isA | ((rc <= 3 * np.median(rc[kr]) * 1.4826 + 1e-6) & (rr <= 3 * np.median(rr[kr]) * 1.4826 + 1e-9))
        if (k2 & ~isA).sum() < (1 if len(anchors) else 3) or (k2 == keep).all():
            break
        keep = k2
    bf = np.linalg.lstsq(X[keep], lf[keep], rcond=None)[0]
    n = len(ks); Cf = (X @ bc)[:n]; Rf = (Rm * Rot.from_rotvec((X @ br)[:n])).as_matrix(); ff = np.exp(X @ bf)[:n]
    for k, Ri, Ci, fi in zip(ks, Rf, Cf, ff):
        _set(k, Ri, Ci)
        K = np.array(cams[k]["intrinsic_3x3"], float); r_ = K[1][1] / K[0][0]; K[0][0] = fi; K[1][1] = fi * r_
        cams[k]["intrinsic_3x3"] = K.tolist()
    return dict(帧=[int(vf[ks[0]]), int(vf[ks[-1]])], 帧数=len(ks), 锚点=len(anchors), 参与拟合=int((keep & ~isA).sum()),
                原始位置残差cm_中位=round(float(np.median(rc[:n])) * 100, 1), 原始朝向残差度_中位=round(float(np.degrees(np.median(rr[:n]))), 2),
                速度_m每秒=round(float(np.linalg.norm(bc[1])), 3))


# ★ 2026-11-12 坏帧按"一串连续坏帧"整体处理(同一镜头内): 以前逐帧判, 同一串一半抄好帧、一半原样留着 ——
#   irish 2 号镜头开头 291 帧没抄、294 抄了, 一跳 66 cm; 最后一个镜头 554 以后 3 帧超出抄的时限又不够拟合, 一跳 104 cm。
#   两头都有好帧且相隔 <= MAX_GAP_S: 整串插值; 只有一头有好帧且整串离它 <= HOLD_S: 整串抄它; 其余: 整串匀速拟合(旁边的好帧当锚点)。
#   整个镜头都是坏帧(joker 推特写): 没有锚点, 按这一段自己拟合。
pos = {k: i for i, k in enumerate(allk)}
runs, cur = [], []
for k in allk:
    if k in bad and cur and pos[k] == pos[cur[-1]] + 1 and shot_all[k] == shot_all[cur[-1]]:
        cur.append(k)
    else:
        if cur: runs.append(cur)
        cur = [k] if k in bad else []
if cur: runs.append(cur)
fits = []
for run in runs:
    s_ = shot_all[run[0]]; g = [q for q in keys if shot_all[q] == s_]
    prv = [q for q in g if vf[q] < vf[run[0]]]; nxt = [q for q in g if vf[q] > vf[run[-1]]]
    a_ = prv[-1] if prv else None; b_ = nxt[0] if nxt else None
    if a_ is not None and b_ is not None and (vf[b_] - vf[a_]) / fps <= MAX_GAP_S:
        Ra, Rb = E_(a_)[:3, :3], E_(b_)[:3, :3]; Ca, Cb = -Ra.T @ E_(a_)[:3, 3], -Rb.T @ E_(b_)[:3, 3]
        sl = Slerp([0, 1], Rot.from_matrix([Ra, Rb]))
        for k in run:
            f_ = (vf[k] - vf[a_]) / (vf[b_] - vf[a_]); _set(k, sl(f_).as_matrix(), (1 - f_) * Ca + f_ * Cb); nb += 1
    elif (a_ is not None and b_ is None and (vf[run[-1]] - vf[a_]) / fps <= HOLD_S) or \
         (b_ is not None and a_ is None and (vf[b_] - vf[run[0]]) / fps <= HOLD_S):
        q = a_ if a_ is not None else b_; R = E_(q)[:3, :3]; Cc = -R.T @ E_(q)[:3, 3]
        for k in run:
            _set(k, R, Cc); nb += 1
    else:
        anc = [x for x in (a_, b_) if x is not None]
        if (anc and len(run) + len(anc) >= 3) or len(run) >= LONG_MIN:
            fits.append(_fit_run(run, anc))
        else:
            nkeep += len(run)
doc["相机平滑"] = dict(sigma_帧=a.sigma, 平滑帧=n, 坏帧插值=nb, 坏帧不动=nkeep, 长坏段匀速拟合=fits, 镜头=len(set(shot)),
                    挪动_位置cm_中位=round(float(np.median(moved_t)) * 100, 2) if moved_t else 0,
                    挪动_朝向度_中位=round(float(np.median(moved_r)), 3) if moved_r else 0)
json.dump(doc, open(p, "w"), ensure_ascii=False, indent=1)
print(f"[相机平滑] {json.dumps(doc['相机平滑'], ensure_ascii=False)} -> {p}")
