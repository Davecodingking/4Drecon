#!/usr/bin/env python
"""柱子: 不走生成, 从抠图点云量出每根柱子的截面, 交给 room_shell.py 盖成地板到天花板的方柱。

为什么不生成 (2026-09-22 office 试过, 用户拍板): 柱子从地通到顶, 画面里几乎总被上下边截掉,
量出来的高度不对; 抠出来是一条白长条, 生成出来多半不像。而柱子的形状本来就简单。
没有柱子, 挂在柱子上的急救箱就悬在半空, 家具和人也能穿过柱子的位置。

量法 (试验版和画面对照过: 几面都拍到的边缘差 4~6 cm):
  1. 每个柱子实例用**所有帧**的抠图反投影 (取点规则抄自 boxes_from_masks.backproj: 自适应腐蚀、
     置信度、MAD 剔穿透; 那边改了这里要跟着改)。量盒子那步只用面积最大的 8 帧, 看到的面少。
  2. 反投影前扣掉同一帧里别的物体的抠图 —— 挂在柱子上的急救箱/灭火器/标牌。
  3. 截面只认"竖着覆盖够多高度"的 3 cm 格子: 柱子从地到顶都有点, 挂件只在某个高度有。
     (office 的急救箱在第 58 帧以前没有自己的抠图, 光扣抠图扣不干净。)
  4. 竖向不到 1 m、点太少、截面超过 1.2 m 的不算柱子 (office 里 4.3 m / 9.3 m 长的其实是墙或梁)。
  5. 截面重叠或相距不到 15 cm 的实例是同一根 (同一根柱子被跟踪切成几段), 合并。
  6. 截面按房间网格对齐 —— z 朝上系已经做过绕 z 的轴对齐, 墙基本沿 x/y 走。
  7. 几面都拍到的(短边 >= 0.6 长边), 截面就是量到的。只拍到一面的, 记下看到的那面多宽、相机在哪一侧,
     由 room_shell 按**看到的宽度**补成方柱, 补在相机看不到的那一侧, 碰到墙就停 (墙的位置那一步才有)。
     ★ 不按"拍全的柱子多粗"统一补: office 3、4 号本来就是 16/21 cm 的细柱子, 按统一的 44 cm 补宽后
       和画面差 16~18 cm, 按各自看到的宽度只差 3~6 cm。

产物 09_柱子.json (z 朝上系, 米): {"柱子": [{ids, lo, hi, both, thin, unseen, wide, n, frames}], ...}
没有柱子的抠图也照样写 (空列表), 下游不用分情况。
"""
import argparse, glob, json, os, sys
import numpy as np, cv2, trimesh
from pycocotools import mask as mask_util
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "outdoor", "scripts"))
from world_frame import usable_frames

ap = argparse.ArgumentParser()
ap.add_argument("--r1", required=True, help="run 目录")
ap.add_argument("--words", default="column,pillar", help="算柱子的词(逗号分隔)")
ap.add_argument("--out", default="", help="默认 <r1>/09_柱子.json")
ap.add_argument("--min-points", type=int, default=2000)
ap.add_argument("--min-height", type=float, default=1.0, help="点的竖向跨度(m)不到这个不算柱子")
ap.add_argument("--max-side", type=float, default=1.2, help="截面长边(m)超过这个不算柱子")
ap.add_argument("--merge-gap", type=float, default=0.15, help="截面相距不到这么多(m)的实例合成一根")
a = ap.parse_args()
R1 = a.r1; DA3 = f"{R1}/02_da3"; MK = f"{R1}/03_masks"
OUT = a.out or f"{R1}/09_柱子.json"
WORDS = [w.strip() for w in a.words.split(",") if w.strip()]
CONF_MIN = float(os.environ.get("CONF_MIN", "2.0"))          # 和 boxes_from_masks 的默认一致

files = [f"{MK}/{w.replace(' ', '_')}.npz" for w in WORDS]
files = [f for f in files if os.path.exists(f)]
if not files:
    json.dump({"柱子": [], "说明": f"没有 {WORDS} 的抠图"}, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"[柱子] 没有 {WORDS} 的抠图, 不盖柱子 -> {OUT}"); sys.exit(0)

cams = json.load(open(f"{DA3}/cameras.json"))["cameras"]
HAVE = usable_frames(DA3, "量柱子")
zj = json.load(open(f"{R1}/03_点云_z朝上.json")); Rz, tz, sz = np.array(zj["R"]), np.array(zj["t"]), float(zj["s"])
camz = np.asarray(zj["camera_centers_new"], float)
V = np.asarray(trimesh.load(f"{R1}/03_点云_z朝上.ply").vertices)
CEIL = float(np.percentile(V[:, 2], 97))                      # 和 room_shell 的墙高同一个取法

def backproj(mask, fi):
    """抄自 boxes_from_masks.backproj (那边改了这里跟着改)。返回 DA3 世界系的点。"""
    fn = f"{fi:04d}.jpg"
    dep = np.load(f"{DA3}/depth/{fi:04d}_depth.npy"); cnf = np.load(f"{DA3}/depth/{fi:04d}_conf.npy")
    h, w = dep.shape
    r = int(np.clip(np.sqrt(mask.sum()) / 25, 1, 6))
    m = cv2.erode(mask.astype(np.uint8), np.ones((r * 2 + 1,) * 2, np.uint8))
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    m &= (cnf >= CONF_MIN) & np.isfinite(dep) & (dep > 0)
    if m.sum() < 12: return None
    d = dep[m]; med = np.median(d); mad = np.median(np.abs(d - med)) + 1e-6
    k = np.abs(d - med) < 4 * mad
    v, u = np.nonzero(m); v, u, d = v[k], u[k], d[k]
    K = np.array(cams[fn]["intrinsic_3x3"])
    P = np.stack([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d], 1)
    c2w = np.linalg.inv(np.array(cams[fn]["extrinsic_w2c_4x4"]).reshape(4, 4))
    return (c2w[:3, :3] @ P.T).T + c2w[:3, 3]

# ---- 每帧别的物体的抠图并起来 (挂件要从柱子的抠图里扣掉) ----
OTHER = {}
for f in sorted(glob.glob(f"{MK}/*.npz")):
    z = np.load(f, allow_pickle=True)
    if str(z["word"]) in WORDS or len(z["meta"]) == 0: continue
    hw = [int(x) for x in z["hw"]]
    for i, (o, fi, ar, e) in enumerate(z["meta"]):
        if ar <= 0: continue
        m = mask_util.decode({"counts": z["counts"][i], "size": hw}).astype(bool); fi = int(fi)
        OTHER[fi] = (OTHER[fi] | m) if fi in OTHER else m

# ---- 每个实例: 所有帧反投影, 转到 z 朝上系 ----
inst = {}
for f in files:
    z = np.load(f, allow_pickle=True); meta, counts = z["meta"], z["counts"]; hw = [int(x) for x in z["hw"]]
    w = str(z["word"])
    for o in sorted(set(int(x) for x in meta["oid"])):
        P, fr = [], []
        for i in np.nonzero(meta["oid"] == o)[0]:
            fi = int(meta["frame"][i])
            if fi not in HAVE or meta["area"][i] <= 0: continue
            m = mask_util.decode({"counts": counts[i], "size": hw}).astype(bool)
            if fi in OTHER: m &= ~OTHER[fi]
            p = backproj(m, fi) if m.sum() > 50 else None
            if p is not None: P.append(p); fr.append(fi)
        if P: inst[f"{w}#{o}"] = dict(P=sz * np.concatenate(P) @ Rz.T + tz, frames=fr)

def footprint(Pz):
    """截面(水平 5~95 分位): 只认竖着覆盖够多高度的 3 cm 格子 —— 柱子从地到顶都有点, 挂件只在某个高度有。"""
    m = (Pz[:, 2] > 0.3) & (Pz[:, 2] < CEIL - 0.3); Pm = Pz[m] if m.sum() > 50 else Pz
    cell = np.floor(Pm[:, :2] / 0.03).astype(np.int64); zb = np.floor(Pm[:, 2] / 0.10).astype(np.int64)
    uk, inv = np.unique((cell[:, 0] << 32) ^ (cell[:, 1] & 0xFFFFFFFF), return_inverse=True)
    if len(uk) < 20000:
        order = np.lexsort((zb, inv)); iv, zz = inv[order], zb[order]
        new = np.r_[True, (iv[1:] != iv[:-1]) | (zz[1:] != zz[:-1])]
        cov = np.bincount(iv[new], minlength=len(uk))          # 每个格子占了几个 10 cm 高度段
        good = cov[inv] >= 0.3 * max(1, len(np.unique(zb)))
        if good.sum() > 50: Pm = Pm[good]
    Q = Pm[:, :2]
    return np.percentile(Q, 5, 0), np.percentile(Q, 95, 0)

# ---- 像不像柱子 ----
ok = []
for k, d in inst.items():
    lo, hi = footprint(d["P"]); zlo, zhi = np.percentile(d["P"][:, 2], [5, 95]); side = hi - lo
    why = [s for s, c in (("点太少", len(d["P"]) < a.min_points), (f"竖向只有 {zhi - zlo:.1f} m", zhi - zlo < a.min_height),
                          (f"截面长边 {side.max():.1f} m", side.max() > a.max_side)) if c]
    d.update(lo=lo, hi=hi)
    print(f"[柱子] {k}: {len(d['P'])} 点 / {len(d['frames'])} 帧, 截面 {np.round(side, 2).tolist()} m, "
          f"竖向 {zlo:.2f}~{zhi:.2f} m -> " + ("算" if not why else "不算: " + ", ".join(why)))
    if not why: ok.append(k)

# ---- 截面挨着的是同一根 ----
par = {k: k for k in ok}
def root(x):
    while par[x] != x: x = par[x]
    return x
for i, p in enumerate(ok):
    for q in ok[i + 1:]:
        gap = np.maximum(inst[p]["lo"] - inst[q]["hi"], inst[q]["lo"] - inst[p]["hi"]).max()
        if gap < a.merge_gap: par[root(q)] = root(p)
groups = {}
for k in ok: groups.setdefault(root(k), []).append(k)

cols = []
for g in groups.values():
    Pz = np.concatenate([inst[k]["P"] for k in g]); lo, hi = footprint(Pz); side = hi - lo
    frs = sorted(set(f for k in g for f in inst[k]["frames"]))
    both = bool(side.min() >= 0.6 * side.max())
    thin, wide = int(np.argmin(side)), int(np.argmax(side))
    ctr = (lo + hi) / 2; cm = camz[frs, :2].mean(0)
    unseen = -1 if cm[thin] > ctr[thin] else 1                 # 相机在 + 侧, 看不到的是 - 侧
    cols.append(dict(ids=g, lo=lo.tolist(), hi=hi.tolist(), both=both, thin=thin, unseen=unseen,
                     wide=float(side[wide]), n=int(len(Pz)), frames=frs))
    print(f"[柱子] 合成一根: {g}, 截面 {np.round(side, 2).tolist()} m, "
          + ("几面都拍到" if both else f"只拍到一面(看不到的在 {'+-'[unseen < 0]}{'xy'[thin]} 侧), 按看到的 {side[wide]:.2f} m 补"))
json.dump({"柱子": cols, "天花板": CEIL, "词": WORDS, "坐标": "z 朝上系, 地板 z=0"},
          open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"[柱子] {len(inst)} 个实例 -> {len(ok)} 个像柱子 -> {len(cols)} 根 -> {OUT}")
