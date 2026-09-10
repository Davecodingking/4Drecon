#!/usr/bin/env python
"""闸 1 的图: 一眼看出地基对不对。三格 —— 抽帧一览 / 点云俯视+相机轨迹 / 点云侧视+地面线。

要看的三件事:
  抽帧    —— 素材本身对不对, 有没有抽到黑帧
  俯视    —— 相机走了多远。轨迹是一小团 = 没视差, 朝向和尺度都会不可靠
  侧视    —— 拟合出来的地面(红线)是不是真的贴着地面点, 而不是穿过物体中间
"""
import os, sys, json, numpy as np, cv2
# 算法模块(world_frame / word_rank / review_decisions)住在 data/outdoor/scripts/,
# 这个脚本在 tools/ —— realpath 解析真身之后往上一层就是仓库根。
# 用 realpath 不用 abspath: 万一哪天又有人建软链接, abspath 会往上数到错的目录去。
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "data", "outdoor", "scripts"))
from world_frame import load_up

R1  = os.environ["R1"]; FR = os.environ["FR"]; OUT = os.environ["OUT"]
D3  = next((f"{R1}/{n}" for n in ("02_da3", "02_DA3点云") if os.path.isdir(f"{R1}/{n}")), f"{R1}/02_da3")
W   = int(os.environ.get("GW", "1500"))

cams = json.load(open(f"{D3}/cameras.json"))["cameras"]
ks   = sorted(cams)
# ★ 这是闸 1 唯一那张"地基对不对"的图, 人就靠它判断天/地拟合。
#   缺 world_up.json 时按老假设画, 会画出一张**看起来完全正常**的图 ——
#   兜底错在最不该错的地方, 一行 stdout 的警告会被日志淹掉。所以: 一律走 world_frame
#   这一份实现(默认硬停), 真要凑合看得显式 ALLOW_UP_FALLBACK=1, 那时把警告压到图上,
#   因为唯一拦得住"拿它当依据"的地方就是这张图本身。
UP, GH, _UPFB = load_up(D3, "闸1地基图")

# --- 点云: 抽几帧反投影 (不读 points.ply, 那个不一定有) ---
P = []
for k in [ks[i] for i in np.linspace(0, len(ks)-1, min(10, len(ks))).astype(int)]:
    dp = f"{D3}/depth/{k[:4]}_depth.npy"
    if not os.path.exists(dp): continue
    d = np.load(dp); c = np.load(f"{D3}/depth/{k[:4]}_conf.npy")
    h, w = d.shape; K = np.array(cams[k]["intrinsic_3x3"])
    v, u = np.mgrid[0:h:3, 0:w:3]; v, u = v.ravel(), u.ravel()
    dd, cc = d[v, u], c[v, u]
    m = np.isfinite(dd) & (dd > 0) & (cc >= np.median(c))
    v, u, dd = v[m], u[m], dd[m]
    Q = np.stack([(u-K[0,2])*dd/K[0,0], (v-K[1,2])*dd/K[1,1], dd], 1)
    c2w = np.linalg.inv(np.array(cams[k]["extrinsic_w2c_4x4"]).reshape(4,4))
    P.append((c2w[:3,:3] @ Q.T).T + c2w[:3,3])
P = np.concatenate(P)
C = np.array([cams[k]["camera_center_world"] for k in ks])

e1 = np.array([1,0,0.]) - UP*UP[0]; e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1)
hgt = P @ UP

def panel(ax, ay, title, color_by_height):
    """把点画到一张图上。ax/ay 是两个投影方向。"""
    H = int(W*0.52)
    img = np.full((H, W, 3), 24, np.uint8)
    X, Y = P @ ax, P @ ay
    lo = np.percentile(np.stack([X, Y], 1), 1, axis=0); hi = np.percentile(np.stack([X, Y], 1), 99, axis=0)
    sc = min(W*0.92/max(hi[0]-lo[0], 1e-6), H*0.92/max(hi[1]-lo[1], 1e-6))
    def to2d(x, y): return (np.int32((x-lo[0])*sc + W*0.04), np.int32((y-lo[1])*sc + H*0.04))
    qx, qy = to2d(X, Y)
    m = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
    if color_by_height:
        t = np.clip((hgt - np.percentile(hgt,2))/max(np.percentile(hgt,98)-np.percentile(hgt,2),1e-6),0,1)
        col = cv2.applyColorMap(np.uint8(255*(1-t))[:,None], cv2.COLORMAP_VIRIDIS)[:,0,:]
        img[qy[m], qx[m]] = col[m]
    else:
        img[qy[m], qx[m]] = (110,110,110)
    # 相机轨迹
    cx, cy = to2d(C @ ax, C @ ay)
    for i in range(len(cx)-1):
        cv2.line(img, (cx[i],cy[i]), (cx[i+1],cy[i+1]), (60,220,255), 2, cv2.LINE_AA)
    cv2.circle(img, (cx[0],cy[0]), 7, (60,255,60), -1); cv2.circle(img, (cx[-1],cy[-1]), 7, (60,60,255), -1)
    # 地面线 (只在侧视图上有意义)
    # 侧视图的纵轴传的是 -UP: 图像 y 是向下增长的, 而高度是向上的, 不取反整张图会上下颠倒
    if not color_by_height and GH is not None and abs(ay @ UP) > 0.9:
        gy = int((-GH - lo[1])*sc + H*0.04)
        if 0 <= gy < H:
            cv2.line(img, (0,gy), (W,gy), (60,60,255), 2)
            cv2.putText(img, f"fitted ground  {GH:.2f} m", (10,gy-8), 0, 0.6, (60,60,255), 2, cv2.LINE_AA)
    # 侧视图的横向跨度远大于高度, 按等比缩放会剩下一大半空白 —— 裁到有内容的部分
    nz = np.nonzero(img.max(2) > 40)[0]
    if len(nz): img = img[:min(int(nz.max()) + 40, H)]
    cv2.rectangle(img,(0,0),(W,36),(0,0,0),-1)
    cv2.putText(img, title, (10,26), 0, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return img

span = float(np.linalg.norm(C[0]-C[-1]))
top  = panel(e1, e2, f"TOP VIEW   camera path {span:.1f} m (green=start red=end)   color=height", True)
side = panel(e1, -UP, f"SIDE VIEW   red line = fitted ground plane  (sky is up)", False)

# 抽帧一览
fs = sorted(f for f in os.listdir(FR) if f.endswith(".jpg"))
pick = [fs[i] for i in np.linspace(0, len(fs)-1, 8).astype(int)]
th = []
for f in pick:
    im = cv2.imread(f"{FR}/{f}"); im = cv2.resize(im, (W//4, int(W//4*9/16)))
    cv2.rectangle(im,(0,0),(im.shape[1],24),(0,0,0),-1)
    cv2.putText(im, f, (6,17), 0, 0.5, (255,255,255), 1, cv2.LINE_AA); th.append(im)
sheet = np.vstack([np.hstack(th[:4]), np.hstack(th[4:])])

def _pad(a, w):
    return a if a.shape[1] == w else cv2.copyMakeBorder(a,0,0,0,w-a.shape[1],cv2.BORDER_CONSTANT,value=(24,24,24))
grid = np.vstack([_pad(x, W) for x in (sheet, top, side)])
if _UPFB:
    # 图上的字必须 ASCII: cv2 默认字体画中文只出一排问号, 而且不报错
    cv2.rectangle(grid, (0, 0), (grid.shape[1], 44), (0, 0, 200), -1)
    cv2.putText(grid, "WARNING: no world_up.json - drawn with legacy -y up. "
                      "DO NOT judge the ground from this image.",
                (12, 31), 0, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite(OUT, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(f"[闸1图] 点云 {len(P)} 点, 相机 {len(C)} 帧, 轨迹首尾 {span:.2f} m -> {OUT}")
