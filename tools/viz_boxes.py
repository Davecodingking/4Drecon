#!/usr/bin/env python
"""把 PCA 的盒子和 WildDet3D 的盒子直接画出来 —— 画的是盒子本身, 不是摆好的场景。
上排: 投影到某一帧的原图上, 线框。
下排: 俯视图, 点云(灰) + 盒子的地面投影(彩), 看盒子有没有套住点。"""
import os, json, numpy as np, cv2
import sys
# 算法模块(world_frame / word_rank / review_decisions)住在 data/outdoor/scripts/,
# 这个脚本在 tools/ —— realpath 解析真身之后往上一层就是仓库根。
# 用 realpath 不用 abspath: 万一哪天又有人建软链接, abspath 会往上数到错的目录去。
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "data", "outdoor", "scripts"))
from world_frame import load_up
R1  = os.environ["R1"]; FR = os.environ["FR"]
FI  = int(os.environ.get("FI", "120")); OUT = os.environ["OUT"]
EXCLUDE = set(w.strip() for w in os.environ.get("EXCLUDE","").split(",") if w.strip())

cams = json.load(open(f"{R1}/02_da3/cameras.json"))["cameras"]
# FI 默认 120, 而位姿只覆盖 DA3 真正跑过的那批帧 —— 硬索引会 KeyError 到看不懂。
# 挪到最近的有位姿的帧(和 mask_compare.py 同一个写法), 图和位姿一起挪。
fn = f"{FI:04d}.jpg"
if fn not in cams:
    fn = min(cams, key=lambda k: abs(int(k[:4]) - FI))
    print(f"[注意] 第 {FI} 帧没位姿, 改用最近的 {fn}")
cam = cams[fn]
K = np.array(cam["intrinsic_3x3"], float); dh, dw = cam["input_hw"]
w2c = np.array(cam["extrinsic_w2c_4x4"], float).reshape(4,4)
UP, _GH, _FB = load_up(f"{R1}/02_da3", "画盒子")
img = cv2.imread(f"{FR}/{fn}")
# imread 读不到只返回 None, 下一行 .shape 会崩成 AttributeError, 看不出是路径的事
if img is None:
    raise SystemExit(f"[失败] 读不到帧 {FR}/{fn} —— FR 指的帧目录里有这一帧吗?")
H, W = img.shape[:2]

def proj(P):
    q = (w2c[:3,:3] @ P.T).T + w2c[:3,3]
    ok = q[:,2] > 0.05
    u = K[0,0]*q[:,0]/np.maximum(q[:,2],1e-6) + K[0,2]
    v = K[1,1]*q[:,1]/np.maximum(q[:,2],1e-6) + K[1,2]
    return np.stack([u*W/dw, v*H/dh], 1), ok

def corners(b):
    c = np.array(b["中心"]); a1 = np.array(b["长轴"])
    a1 = a1 - UP*(a1@UP); a1 /= np.linalg.norm(a1); a2 = np.cross(UP, a1)
    s = np.array(b["尺寸"])/2
    return np.array([c + a1*s[0]*i + a2*s[1]*j + UP*s[2]*k
                     for i in (-1,1) for j in (-1,1) for k in (-1,1)])
E = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]

# 水平面的两个基, 用来做俯视图
e1 = np.array([1,0,0.]) - UP*UP[0]; e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1)

def load(bf):
    B = json.load(open(f"{R1}/{bf}"))
    return [(w,b) for w,lst in B.items() if w not in EXCLUDE for b in lst]

def color(w):
    h = abs(hash(w)) % 180
    return tuple(int(x) for x in cv2.cvtColor(np.uint8([[[h,220,255]]]), cv2.COLOR_HSV2BGR)[0,0])

# 点云(俯视用): DA3 的全局点云, 抽稀
import struct
pts = None
p = f"{R1}/02_da3/points.ply"
if os.path.exists(p):
    try:
        import trimesh; pc = trimesh.load(p); pts = np.asarray(pc.vertices)
        if len(pts) > 400000: pts = pts[np.random.default_rng(0).choice(len(pts),400000,replace=False)]
    except Exception as e: print("点云读不了:", e)

panels = []
for tag, bf in (("PCA", "06_all_boxes.json"), ("WildDet3D", "06_all_boxes_wilddet.json")):
    items = load(bf)
    # --- 上: 投到原图 ---
    top = img.copy(); drawn = 0
    for w, b in items:
        C = corners(b); uv, ok = proj(C)
        if ok.sum() < 8: continue
        col = color(w); drawn += 1
        for i,j in E:
            cv2.line(top, tuple(np.int32(uv[i])), tuple(np.int32(uv[j])), col, 2, cv2.LINE_AA)
    cv2.rectangle(top,(0,0),(W,44),(0,0,0),-1)
    cv2.putText(top, f"{tag}  boxes projected onto {fn}   ({drawn}/{len(items)} in view)",
                (10,32), 0, 1.0, (255,255,255), 2, cv2.LINE_AA)
    # --- 下: 俯视 ---
    SZ = W; tv = np.full((int(H), SZ, 3), 30, np.uint8)
    allpt = [np.stack([pts@e1, pts@e2],1)] if pts is not None else []
    allpt += [np.stack([corners(b)@e1, corners(b)@e2],1) for _,b in items]
    A = np.concatenate(allpt)
    lo = np.percentile(A,0.5,axis=0); hi = np.percentile(A,99.5,axis=0)
    sc = min(SZ*0.92/max(hi[0]-lo[0],1e-6), H*0.92/max(hi[1]-lo[1],1e-6))
    def to2d(Q):
        return np.int32(np.stack([(Q[:,0]-lo[0])*sc + SZ*0.04, (Q[:,1]-lo[1])*sc + H*0.04],1))
    if pts is not None:
        q = to2d(np.stack([pts@e1, pts@e2],1))
        m = (q[:,0]>=0)&(q[:,0]<SZ)&(q[:,1]>=0)&(q[:,1]<H)
        tv[q[m,1], q[m,0]] = (120,120,120)
    for w, b in items:
        C = corners(b); q = to2d(np.stack([C@e1, C@e2],1))
        hull = cv2.convexHull(q)
        cv2.polylines(tv, [hull], True, color(w), 2, cv2.LINE_AA)
        a1 = np.array(b["长轴"]); a1 = a1-UP*(a1@UP); a1/=np.linalg.norm(a1)
        c  = np.array(b["中心"]); tip = c + a1*b["尺寸"][0]/2
        pq = to2d(np.stack([np.array([c@e1,tip@e1]), np.array([c@e2,tip@e2])],1).T if False else
                  np.array([[c@e1,c@e2],[tip@e1,tip@e2]]))
        cv2.arrowedLine(tv, tuple(pq[0]), tuple(pq[1]), (255,255,255), 1, cv2.LINE_AA, tipLength=0.3)
    cv2.rectangle(tv,(0,0),(SZ,44),(0,0,0),-1)
    cv2.putText(tv, f"{tag}  top view: gray = DA3 point cloud, arrow = long axis  ({len(items)} boxes)",
                (10,32), 0, 1.0, (255,255,255), 2, cv2.LINE_AA)
    panels.append(np.vstack([top, tv]))
    print(f"{tag:10s} {len(items)} 个盒子, 该帧画面内 {drawn} 个")
grid = np.hstack(panels)
grid = cv2.resize(grid, (grid.shape[1]//2, grid.shape[0]//2))
cv2.imwrite(OUT, grid, [cv2.IMWRITE_JPEG_QUALITY, 92]); print("saved", OUT)
