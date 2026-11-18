#!/usr/bin/env python
"""楼梯深度修正(2026-11-11, joker 起头): 从楼梯底往上拍的镜头, DA3 把楼梯压成缓坡; 按画面逐行把深度挪到真实坡度的楼梯面上。

量到的事实(scout_1110/stairs_probe.py, joker r2, 楼梯掩码 stairs/staircase 的点):
  镜头 1、4 从下往上拍: 楼梯段坡度 11~12.5°;  镜头 3 从上往下拍: 29°, 还看得出中间的平台(台阶立面看得见, 竖直面占 40%)。
  往上拍时台阶立面被踏面挡住, 深度把楼梯抹成了斜坡 —— 坡度是错的, 投影(画面里的位置)是对的。
做法:
  1. 每个镜头: 楼梯掩码的点 -> 水平主方向 = 楼梯走向(朝上坡) -> 坡度。
  2. 真坡度 = 台阶立面看得最清楚(竖直面占比最高)的那个镜头量到的坡度, 限在 [MIN_DEG, MAX_DEG]; 量不到用 31°(常规 17/28 cm)。
  3. 坡度比真坡度缓 TOL_DEG 以上的镜头: 楼梯面 = 过锚点、按真坡度的斜面。锚点 = 这个镜头里人脚踩的位置(人的远近按身体大小
     核对过, 那里的深度是准的); 没有人才退回楼梯底。★ 先试过锚在楼梯底: joker 镜头 1 深度要整体缩到 43%, 人就该在画面里
     大一倍 —— 和人的身体大小矛盾, 说明 DA3 在人那里是准的, 错的是"楼梯底准"这个假设。每帧楼梯像素的视线和它求交
     -> 新深度 / 旧深度 = 这个像素的比例 -> 每一行取中位(画面里一行 ≈ 楼梯上同一个远近) -> 行间补齐、平滑 ->
     整行所有像素(人、扶手、墙)一起乘。只改深度不改相机, 画面投影不变; 人的远近也是从深度取的, 跟着楼梯一起到位。
  原深度备份在 02_da3/depth_before_stairs/; 已经修过(02_stairs_fix.json 在)就跳过。
"""
import argparse, json, os, shutil, sys
import numpy as np, cv2
from pycocotools import mask as mask_util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terrain import frame_points, basis

WORDS = ("stairs", "staircase")
MIN_DEG, MAX_DEG, DEFAULT_DEG = 22.0, 40.0, 31.0
TOL_DEG = 5.0
MIN_PTS = 20000

ap = argparse.ArgumentParser()
ap.add_argument("--r1", required=True)
ap.add_argument("--dry", action="store_true")
ap.add_argument("--da3", default="", help="改这份 02_da3 的拷贝(试验用), 报告也写进它; 默认改 <r1>/02_da3")
a = ap.parse_args()
D = a.da3 or f"{a.r1}/02_da3"; REP = f"{a.da3}/stairs_fix.json" if a.da3 else f"{a.r1}/02_stairs_fix.json"
if os.path.exists(REP) and not a.dry:
    print(f"[楼梯深度] 已经修过({REP}), 跳过 —— 要重来先从 02_da3/depth_before_stairs/ 还原"); sys.exit(0)
cams = json.load(open(f"{D}/cameras.json"))["cameras"]
up = np.array(json.load(open(f"{D}/world_up.json"))["up"], float); up /= np.linalg.norm(up); e1, e2 = basis(up)
fm = json.load(open(f"{D}/frame_map.json")); fps = float(fm.get("fps") or 25.0); vf = fm["frames"]
cuts = json.load(open(f"{D}/shot_cuts.json"))["cuts"] if os.path.exists(f"{D}/shot_cuts.json") else []
bad = set(json.load(open(f"{D}/bad_frames.json")).get("坏帧", [])) if os.path.exists(f"{D}/bad_frames.json") else set()
M = {}
for w in WORDS:
    p = f"{a.r1}/03_masks/{w}.npz"
    if not os.path.exists(p): continue
    z = np.load(p, allow_pickle=True); hw = [int(x) for x in z["hw"]]
    for r, c in zip(z["meta"], z["counts"]): M.setdefault(int(r["frame"]), []).append((c, hw))
if not M:
    print("[楼梯深度] 没有楼梯掩码(stairs / staircase), 不修"); sys.exit(0)


def stair_mask(k):
    rows = M.get(int(vf[k]))
    if not rows: return None
    m = np.zeros(rows[0][1], bool)
    for c, hw in rows: m |= mask_util.decode({"counts": c, "size": hw}).astype(bool)
    return m


shot_of = lambda k: int(np.searchsorted(cuts, vf[k] / fps + 1e-3, side="right"))
by = {}
for k in sorted(cams, key=lambda k: vf[k]):
    if os.path.exists(f"{D}/depth/{k[:-4]}_depth.npy"): by.setdefault(shot_of(k), []).append(k)

# 0. 每个镜头里人脚踩的位置(10_people_world.npz: 每人每帧网格最低点, 取中位)
FEET, FEETN = {}, {}
_pw = f"{a.r1}/10_people_world.npz"
if os.path.exists(_pw):
    _z = np.load(_pw, allow_pickle=True); _acc = {}
    for _k in _z.files:
        if not _k.endswith("_V"): continue
        for _V, _f in zip(_z[_k], _z[_k[:-2] + "_vf"]):
            _s = int(np.searchsorted(cuts, int(_f) / fps + 1e-3, side="right"))
            _acc.setdefault(_s, []).append(_V[np.argmin(_V @ up)])
    for _s, _L in _acc.items():
        if len(_L) >= 5: FEET[_s] = np.median(np.array(_L), 0); FEETN[_s] = len(_L)

# 1. 每个镜头的楼梯: 走向、坡度、锚点
fit = {}
for s, ks in by.items():
    good = [k for k in ks if k not in bad and int(vf[k]) in M]
    if not good: continue
    good = [good[i] for i in np.unique(np.linspace(0, len(good) - 1, min(20, len(good))).astype(int))]
    P_, N_ = [], []
    for k in good:
        P, N, C = frame_points(D, k, cams[k], stride=2, skip=~stair_mask(k)); P_.append(P); N_.append(N)
    P = np.concatenate(P_); N = np.concatenate(N_)
    if len(P) < MIN_PTS: continue
    uv = np.c_[P @ e1, P @ e2]; h = P @ up; c = uv.mean(0)
    ax = np.linalg.svd(uv - c, full_matrices=False)[2][0]
    t = (uv - c) @ ax
    k_ = np.linalg.lstsq(np.c_[t, np.ones_like(t)], h, rcond=None)[0][0]
    if k_ < 0: ax, t, k_ = -ax, -t, -k_
    t0 = float(np.percentile(t, 3)); near = (t >= t0) & (t < t0 + 1.0)
    anc = "楼梯底"; h0 = float(np.median(h[near]))
    if FEET.get(s) is not None:                                  # 人脚: 锚点挪到人站的地方
        F = FEET[s]; t0 = float((np.r_[F @ e1, F @ e2] - c) @ ax); h0 = float(F @ up); anc = f"人脚({FEETN[s]} 帧中位)"
    fit[s] = dict(ax=ax, c=c, t0=t0, h0=h0, anc=anc, deg=float(np.degrees(np.arctan(k_))),
                  vert=float((np.abs(N @ up) < 0.3).mean()), n=len(P))
if not fit:
    print("[楼梯深度] 楼梯点太少, 不修"); sys.exit(0)
ref = max(fit, key=lambda s: fit[s]["vert"])
true_deg = fit[ref]["deg"] if MIN_DEG <= fit[ref]["deg"] <= MAX_DEG else DEFAULT_DEG
src = f"镜头 {ref + 1}(台阶立面占 {fit[ref]['vert']:.0%})" if MIN_DEG <= fit[ref]["deg"] <= MAX_DEG else f"常规 {DEFAULT_DEG}°(镜头 {ref + 1} 量到 {fit[ref]['deg']:.1f}° 不在 {MIN_DEG}~{MAX_DEG}°)"
kk = np.tan(np.radians(true_deg))
rep = dict(真坡度_度=round(true_deg, 1), 来源=src, 各镜头={})

# 2. 逐帧逐行修深度
if not a.dry:
    bk = f"{D}/depth_before_stairs"; os.makedirs(bk, exist_ok=True)
for s, f in sorted(fit.items()):
    info = dict(楼梯点=f["n"], 量到坡度_度=round(f["deg"], 1), 竖直面=round(f["vert"], 2), 锚点=f["anc"])
    if f["deg"] >= true_deg - TOL_DEG:
        info["结果"] = "坡度对, 不修"; rep["各镜头"][s + 1] = info; continue
    A = f["ax"][0] * e1 + f["ax"][1] * e2; off = float(f["c"] @ f["ax"]) + f["t0"]
    done, scales = 0, []
    for k in by[s]:
        m = stair_mask(k)
        if m is None: continue
        dp = f"{D}/depth/{k[:-4]}_depth.npy"; d = np.load(dp).astype(np.float64); H, W = d.shape
        mm = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) & (d > 0)
        if mm.sum() < 200: continue
        cam = cams[k]; K = np.array(cam["intrinsic_3x3"], float)
        if "input_hw" in cam:
            h0_, w0_ = cam["input_hw"]; sx, sy = W / w0_, H / h0_
        else:
            sx = sy = W / (K[0, 2] * 2)
        fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy
        E = np.array(cam["extrinsic_w2c_4x4"], float); R, tt = E[:3, :3], E[:3, 3]; C = -R.T @ tt
        ys, xs = np.nonzero(mm)
        dc = np.c_[(xs - cx) / fx, (ys - cy) / fy, np.ones(len(xs))]          # 相机系视线, z=1
        dw = dc @ R                                                            # 世界系(没归一, z 分量=1 对应深度)
        # 斜面: C·up + λ dw·up = h0 + kk (C·A + λ dw·A - off)   ->   λ 就是新深度(dw 的相机 z 分量是 1)
        den = dw @ up - kk * (dw @ A)
        lam = (f["h0"] + kk * (C @ A - off) - C @ up) / np.where(np.abs(den) < 1e-6, np.nan, den)
        z_old = d[ys, xs]; ratio = lam / z_old
        ok = np.isfinite(ratio) & (lam > 0) & (ratio > 0.3) & (ratio < 3.0)
        if ok.sum() < 200: continue
        rows = np.full(H, np.nan)
        for y in np.unique(ys[ok]):
            sel = ok & (ys == y)
            if sel.sum() >= 3: rows[y] = np.median(ratio[sel])
        have = np.isfinite(rows)
        if have.sum() < 5: continue
        yy = np.arange(H)
        rows = np.interp(yy, yy[have], rows[have])                             # 没有楼梯的行: 接最近的楼梯行
        rows = np.convolve(np.pad(rows, 7, mode="edge"), np.ones(15) / 15, mode="valid")    # 行间平滑
        scales.append((float(rows.min()), float(np.median(rows)), float(rows.max())))
        if not a.dry:
            if not os.path.exists(f"{bk}/{k[:-4]}_depth.npy"): shutil.copy(dp, f"{bk}/{k[:-4]}_depth.npy")
            np.save(dp, (d * rows[:, None]).astype(np.load(dp, mmap_mode="r").dtype))
        done += 1
    info.update(结果=f"修了 {done} 帧" if done else "没有能修的帧",
                深度比例_最小=round(min(x[0] for x in scales), 2) if scales else None,
                深度比例_中位=round(float(np.median([x[1] for x in scales])), 2) if scales else None,
                深度比例_最大=round(max(x[2] for x in scales), 2) if scales else None)
    rep["各镜头"][s + 1] = info
    print(f"[楼梯深度] 镜头 {s + 1}: {json.dumps(info, ensure_ascii=False)}", flush=True)
print(f"[楼梯深度] 真坡度 {true_deg:.1f}°, 来源 {src}")
json.dump(rep, open(REP if not a.dry else f"{a.r1}/debug_1111/stairs_fix_dry.json", "w"), ensure_ascii=False, indent=1)
