#!/usr/bin/env python
"""同一帧下, 把 PCA 版和 WildDet3D 版场景各渲一次, 取轮廓(mask), 和 SAM3 的真实 mask 比。
渲染用的相机、内参、翻轴和 render_back.py 完全一致, 只是取深度缓冲当 mask 而不是取彩色。
输出 5 联: 原图 | SAM3 真值 | PCA 版 | WD 版 | PCA/WD 差异图。"""
import sys, os, json, numpy as np, trimesh, cv2
# 算法模块(world_frame / word_rank / review_decisions)住在 data/outdoor/scripts/,
# 这个脚本在 tools/ —— 用仓库根定位, 别用 __file__ 所在目录(软链接会把它解析到 tools/)。
# 算法模块(world_frame / word_rank / review_decisions)住在 data/outdoor/scripts/,
# 这个脚本在 tools/ —— realpath 解析真身之后往上一层就是仓库根。
# 用 realpath 不用 abspath: 万一哪天又有人建软链接, abspath 会往上数到错的目录去。
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "data", "outdoor", "scripts"))
from world_frame import load_up, export_T, inverse_T
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
from pycocotools import mask as mask_util

R1  = os.environ["R1"]
FR  = os.environ["FR"]
FI  = int(os.environ.get("FI", "120"))
OUT = os.environ["OUT"]
EXCLUDE = set(w.strip() for w in os.environ.get("EXCLUDE", "").split(",") if w.strip())
W, H = 1280, 720

cams = json.load(open(f"{R1}/02_da3/cameras.json"))["cameras"]
fn = f"{FI:04d}.jpg"
if fn not in cams:
    fn = min(cams, key=lambda k: abs(int(k[:4]) - FI))
    print(f"[注意] 第 {FI} 帧没位姿, 改用最近的 {fn}")
cam = cams[fn]; K = np.array(cam["intrinsic_3x3"])
c2w = np.linalg.inv(np.array(cam["extrinsic_w2c_4x4"]).reshape(4, 4))
gl  = c2w @ np.diag([1, -1, -1, 1])
sx, sy = W / (K[0, 2] * 2), H / (K[1, 2] * 2)
pc = pyrender.IntrinsicsCamera(fx=K[0, 0]*sx, fy=K[1, 1]*sy, cx=K[0, 2]*sx, cy=K[1, 2]*sy)

# 组装那步导出时用的是"把真实的天转到 +y"的罗德里格斯旋转 R3, 不是老的 diag(1,-1,-1)。
# 要渲回原相机就必须反着乘 R3, 乘错的话整个场景绕着某根轴转掉(carpark 差 29 度)。
# 逆旋转从 world_frame 拿, 和组装那边同源
_UP, _GH, _FB = load_up(f"{R1}/02_da3", "mask对照")
INV=inverse_T(_UP)          # 和组装的 export_T 互逆

def silhouette(glb):
    """渲一个 glb, 返回 (轮廓bool, 彩色图)。地面那块单独剔掉, 不然整幅都是前景。"""
    s = trimesh.load(f"{R1}/{glb}")
    sc = trimesh.Scene()
    for n in s.graph.nodes_geometry:
        if "ground" in n.lower() or "floor" in n.lower(): continue
        T, g = s.graph[n]; m = s.geometry[g].copy(); m.apply_transform(T)
        sc.add_geometry(m, node_name=n)
    sc.apply_transform(INV)
    ps = pyrender.Scene(bg_color=[0,0,0,1], ambient_light=[.5]*3)
    for name, g in sc.geometry.items():
        for node in sc.graph.nodes_geometry:
            T, gname = sc.graph[node]
            if gname == name: ps.add(pyrender.Mesh.from_trimesh(g, smooth=False), pose=T)
    n1 = ps.add(pc, pose=gl); n2 = ps.add(pyrender.DirectionalLight(intensity=4.0), pose=gl)
    r = pyrender.OffscreenRenderer(W, H); col, dep = r.render(ps); r.delete()
    return (dep > 0), col

# SAM3 真值: 该帧上所有非排除词的实例 mask 并起来
gt = np.zeros((H, W), bool); ngt = 0
for f in sorted(os.listdir(f"{R1}/03_masks")):
    if not f.endswith(".npz"): continue
    if f[:-4] in EXCLUDE: continue
    z = np.load(f"{R1}/03_masks/{f}", allow_pickle=True)
    meta, counts, (mh, mw) = z["meta"], z["counts"], z["hw"]
    for i, r_ in enumerate(meta):
        if r_["area"] > 0 and int(r_["frame"]) == int(fn[:4]):
            m = mask_util.decode({"counts": counts[i], "size": [int(mh), int(mw)]}).astype(bool)
            gt |= cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            ngt += 1

pca, pca_col = silhouette(os.environ.get("SCENE_A", "07_scene_phy.glb"))
wd,  wd_col  = silhouette(os.environ.get("SCENE_B", "17_scene_wd_phy.glb"))

def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0
def cov(a, b):   # b 覆盖了多少 a
    return float((a & b).sum() / a.sum()) if a.sum() else 0.0

print(f"帧 {fn}  SAM3 真值实例 {ngt} 个, 占画面 {100*gt.mean():.1f}%")
for nm, m in (("PCA", pca), ("WD ", wd)):
    print(f"  {nm}: 占画面 {100*m.mean():5.1f}%  IoU(对真值) {iou(m,gt):.3f}  "
          f"真值被盖住 {100*cov(gt,m):5.1f}%  渲出来落在真值外 {100*(1-cov(m,gt)):5.1f}%")
print(f"  两版之间 IoU {iou(pca,wd):.3f}")

real = cv2.cvtColor(cv2.imread(f"{FR}/{fn}"), cv2.COLOR_BGR2RGB)
real = cv2.resize(real, (W, H))
def over(m, c):
    o = real.copy(); o[m] = (0.35*real[m] + 0.65*np.array(c)).astype(np.uint8); return o
diff = real.copy()
diff[pca & ~wd] = (0.3*real[pca & ~wd] + 0.7*np.array([255,60,60])).astype(np.uint8)   # 只 PCA 有
diff[wd & ~pca] = (0.3*real[wd & ~pca] + 0.7*np.array([60,140,255])).astype(np.uint8)  # 只 WD 有
diff[pca & wd]  = (0.3*real[pca & wd]  + 0.7*np.array([150,150,150])).astype(np.uint8) # 都有

panels = [(real, f"{fn}  real"),
          (over(gt, [255,255,0]),  f"SAM3 gt masks ({ngt} inst)"),
          (over(pca,[255,60,60]),  f"PCA  IoU {iou(pca,gt):.3f}  recall {100*cov(gt,pca):.0f}%"),
          (over(wd, [60,140,255]), f"WD   IoU {iou(wd,gt):.3f}  recall {100*cov(gt,wd):.0f}%"),
          (diff,                   f"red=PCA only  blue=WD only  gray=both  IoU {iou(pca,wd):.3f}")]
SZ = 900
rows = []
for img, cap in panels:
    t = cv2.resize(img, (SZ, int(SZ*H/W)))
    cv2.rectangle(t, (0,0), (SZ,26), (0,0,0), -1)
    cv2.putText(t, cap, (8,19), 0, 0.55, (255,255,255), 1, cv2.LINE_AA)
    rows.append(t)
grid = np.vstack([np.hstack(rows[0:2]), np.hstack(rows[2:4]),
                  np.hstack([rows[4], np.zeros_like(rows[4])])])
cv2.imwrite(OUT, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
print("saved", OUT)
