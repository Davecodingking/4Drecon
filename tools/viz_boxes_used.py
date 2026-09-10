#!/usr/bin/env python
"""把"组装真正用的那批盒子"画出来。上排投到几帧原图上, 下排俯视图。

盒子是怎么来的 (boxes_from_masks.py:fit_box) ——
  1. 把 mask 里的像素按深度反投影成三维点 (每个实例取面积最大的若干帧, 合并)
  2. 竖直方向单独拿出来: hgt = P·UP
  3. 只对**水平投影**做 PCA, 最大特征向量当长轴; 短轴 = UP × 长轴
  4. 沿 (长轴, 短轴, UP) 三个方向取极值, min/max 之差就是尺寸
所以 PCA 只定水平朝向, 三个尺寸都是极值, 不是 PCA 的特征值。竖直方向压根没参与 PCA。

用法:
  python tools/viz_boxes_used.py --r1 <结果目录> --fr <帧目录> --out x.jpg [--frames 0,20,40,59]
"""
import argparse, json, os, sys, colorsys
import numpy as np, cv2
# 算法模块(world_frame / word_rank / review_decisions)住在 data/outdoor/scripts/,
# 这个脚本在 tools/ —— realpath 解析真身之后往上一层就是仓库根。
# 用 realpath 不用 abspath: 万一哪天又有人建软链接, abspath 会往上数到错的目录去。
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "data", "outdoor", "scripts"))
from world_frame import load_up

ap = argparse.ArgumentParser()
ap.add_argument("--r1", required=True)
ap.add_argument("--fr", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--boxes", default="", help="默认 06_all_boxes_merged.json, 没有就退回 06_all_boxes.json")
ap.add_argument("--frames", default="", help="逗号分隔; 空=从可用帧里均匀取 4 帧")
ap.add_argument("--exclude", default="screen,person,pedestrian,sofa,couch")
ap.add_argument("--w", type=int, default=960)
a = ap.parse_args()

EXCL = {w.strip() for w in a.exclude.split(",") if w.strip()}
bp = a.boxes or (f"{a.r1}/06_all_boxes_merged.json" if
                 os.path.exists(f"{a.r1}/06_all_boxes_merged.json") else f"{a.r1}/06_all_boxes.json")
B = json.load(open(bp))
cams = json.load(open(f"{a.r1}/02_da3/cameras.json"))["cameras"]
# "天"走 world_frame 这一份实现: 自己 json.load 加静默兜底的写法, 缺文件时会画出一张
# 看起来完全正常、但所有盒子都沿歪掉的方向立着的图 —— 而这张图恰恰是用来核对
# "组装真正用的是哪批盒子"的。load_up 缺文件直接停, 要凑合跑得显式 ALLOW_UP_FALLBACK=1。
UP, _GH, _FB = load_up(f"{a.r1}/02_da3", "组装用盒子图")
print(f"[天] UP={np.round(UP,3).tolist()}" + ("  ★ 走了兜底, 别拿这张图当依据" if _FB else ""))

words = sorted(w for w in B if w not in EXCL)
COL = {w: tuple(int(255*c) for c in colorsys.hsv_to_rgb(i/max(len(words),1), .85, 1))[::-1]
       for i, w in enumerate(words)}
items = [(w, b) for w in words for b in B[w]]
print(f"[盒子] {bp.split('/')[-1]}: {len(words)} 个词 / {len(items)} 个实例 (排除了 {sorted(EXCL)})")

def corners(b):
    """盒子八个角。长轴 a1 在水平面内, a2 = UP x a1, 第三个方向就是 UP。"""
    c = np.array(b["中心"]); a1 = np.array(b["长轴"], float)
    a1 = a1 - UP*(a1@UP); a1 /= np.linalg.norm(a1); a2 = np.cross(UP, a1)
    L, W, H = np.array(b["尺寸"])/2
    return np.array([c + sx*L*a1 + sy*W*a2 + sz*H*UP
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])

E = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]

# ---------- 上排: 投到原图 ----------
avail = sorted(int(k[:4]) for k in cams)
fr = [int(x) for x in a.frames.split(",") if x.strip()] or \
     [avail[int(round(i*(len(avail)-1)/3))] for i in range(4)]
tiles = []
for fi in fr:
    # --frames 是人手敲的, 编号不一定落在有位姿的那批帧上; 原来 fn not in cams 直接 continue,
    # 结果是四格图悄悄少一格。挪到最近的有位姿的帧, 图和位姿一起挪, 免得框画错位。
    fn = f"{fi:04d}.jpg"
    if fn not in cams:
        fn = min(cams, key=lambda k: abs(int(k[:4]) - fi))
        print(f"[注意] 第 {fi} 帧没位姿, 改用最近的 {fn}")
    img = cv2.imread(f"{a.fr}/{fn}")
    if img is None:
        print(f"[注意] 读不到帧 {a.fr}/{fn}, 这一格跳过")
        continue
    K = np.array(cams[fn]["intrinsic_3x3"], float)
    w2c = np.array(cams[fn]["extrinsic_w2c_4x4"], float).reshape(4, 4)
    dh, dw = cams[fn]["input_hw"]
    sx, sy = img.shape[1]/dw, img.shape[0]/dh
    ov = img.copy()
    for w, b in items:
        P = (w2c[:3,:3] @ corners(b).T).T + w2c[:3,3]
        if (P[:,2] <= 0.05).any(): continue          # 有角在相机后面, 投影会翻折
        uv = (K @ P.T).T; uv = uv[:,:2]/uv[:,2:3]
        uv[:,0] *= sx; uv[:,1] *= sy
        for i,j in E:
            cv2.line(ov, tuple(uv[i].astype(int)), tuple(uv[j].astype(int)), COL[w], 2, cv2.LINE_AA)
        cv2.putText(ov, f"{w}#{b['obj_id']}", tuple(uv[:,:2].min(0).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, COL[w], 1, cv2.LINE_AA)
    img = cv2.addWeighted(ov, .85, img, .15, 0)
    cv2.putText(img, fn, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,255), 2, cv2.LINE_AA)
    tiles.append(cv2.resize(img, (a.w//2, int(a.w/2*img.shape[0]/img.shape[1]))))
top = np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4])]) if len(tiles) >= 4 else np.hstack(tiles)

# ---------- 下排: 俯视图 ----------
e1 = np.array([1,0,0.]); e1 = e1 - UP*(e1@UP)
if np.linalg.norm(e1) < 1e-6: e1 = np.array([0,0,1.]) - UP*(np.array([0,0,1.])@UP)
e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1)
allc = np.array([p for _, b in items for p in corners(b)])
xy = np.c_[allc@e1, allc@e2]
lo, hi = xy.min(0)-1, xy.max(0)+1
S = min(top.shape[1]/(hi[0]-lo[0]), top.shape[1]*0.75/(hi[1]-lo[1]))
TW, TH = int((hi[0]-lo[0])*S)+1, int((hi[1]-lo[1])*S)+1
plan = np.full((TH, TW, 3), 28, np.uint8)
to = lambda p: (int((p[0]-lo[0])*S), TH-1-int((p[1]-lo[1])*S))
for w, b in items:
    C = corners(b); q = np.c_[C@e1, C@e2]
    hull = cv2.convexHull(np.array([to(p) for p in q], np.int32))
    cv2.polylines(plan, [hull], True, COL[w], 2, cv2.LINE_AA)
    ctr = np.array(b["中心"]); cv2.putText(plan, f"{w}#{b['obj_id']}", to((ctr@e1, ctr@e2)),
                                          cv2.FONT_HERSHEY_SIMPLEX, .4, COL[w], 1, cv2.LINE_AA)
cv2.putText(plan, f"TOP VIEW  {len(items)} boxes  {(hi[0]-lo[0]):.1f} x {(hi[1]-lo[1]):.1f} m",
            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2, cv2.LINE_AA)
if plan.shape[1] != top.shape[1]:
    plan = cv2.resize(plan, (top.shape[1], int(plan.shape[0]*top.shape[1]/plan.shape[1])))
cv2.imwrite(a.out, np.vstack([top, plan]), [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"[done] -> {a.out}")
