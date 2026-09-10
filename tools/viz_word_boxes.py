"""同一个词, PCA 盒子 vs WD3 召回盒子, 并排画在同一帧上。

"天"从 world_frame 拿, 不写死 —— 老假设 [0,-1,0] 实测在 beach 偏 6.53°、carpark 偏 29.2°,
框上标的高度就是沿"天"量的, 方向歪了这个数字跟着错, 而这张图正是拿来比高度的。

用法: viz_word_boxes.py --run-dir <o1> --frames <dir> --word chair --frames-idx 24,27 --out x.jpg
"""
import argparse, json, os, sys
import numpy as np, cv2
# 算法模块(world_frame / word_rank / review_decisions)住在 data/outdoor/scripts/,
# 这个脚本在 tools/ —— realpath 解析真身之后往上一层就是仓库根。
# 用 realpath 不用 abspath: 万一哪天又有人建软链接, abspath 会往上数到错的目录去。
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "data", "outdoor", "scripts"))
from world_frame import load_up

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--word", required=True)
ap.add_argument("--frames-idx", default="24,27")
ap.add_argument("--out", required=True)
ap.add_argument("--width", type=int, default=1000)
a = ap.parse_args()

R = a.run_dir
UP, _GH, _ = load_up(f"{R}/02_da3", "词盒子对比图")
cams = json.load(open(f"{R}/02_da3/cameras.json"))["cameras"]
pca = json.load(open(f"{R}/06_all_boxes.json")).get(a.word, [])
rec = [r for r in json.load(open(f"{R}/13_recall.json"))["召回"] if r["词"] == a.word]

EDGES = [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]
def corners(d):
    c = np.array(d["中心"]); L, W_, H_ = d["尺寸"]
    e1 = np.array(d["长轴"]); e1 = e1 - UP*(e1@UP); e1 /= max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(UP, e1)
    return np.array([c + i*L/2*e1 + j*W_/2*e2 + k*H_/2*(-UP)
                     for i in (-1,1) for j in (-1,1) for k in (-1,1)])

def draw(img, items, col, K, w2c):
    n = 0
    H, W = img.shape[:2]
    for d in items:
        P = (w2c[:3,:3] @ corners(d).T).T + w2c[:3,3]
        if (P[:,2] < .1).any(): continue
        uv = (K @ P.T).T; uv = uv[:,:2]/uv[:,2:3]
        if uv[:,0].max() < 0 or uv[:,0].min() > W or uv[:,1].max() < 0 or uv[:,1].min() > H: continue
        for i, j in EDGES:
            cv2.line(img, tuple(uv[i].astype(int)), tuple(uv[j].astype(int)), col, 2, cv2.LINE_AA)
        p = uv.mean(0).astype(int)
        cv2.putText(img, f"{d['尺寸'][2]:.2f}", (p[0]-22, p[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, f"{d['尺寸'][2]:.2f}", (p[0]-22, p[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, col, 1, cv2.LINE_AA)
        n += 1
    return n

rows = []
for fi in [int(x) for x in a.frames_idx.split(",")]:
    # 抽帧编号和 DA3 出位姿的编号不一定一一对应(抽帧密、位姿稀), 硬索引 cams 会 KeyError。
    # 换成最近的那一帧, 而且原图和位姿一起挪 —— 只挪位姿的话图和框会错位。
    fn = f"{fi:04d}.jpg"
    if fn not in cams:
        fn = min(cams, key=lambda k: abs(int(k[:4]) - fi))
        print(f"[注意] 第 {fi} 帧没位姿, 改用最近的 {fn}")
        fi = int(fn[:4])
    base = cv2.imread(os.path.join(a.frames, fn))
    if base is None:
        raise SystemExit(f"[失败] 读不到帧 {os.path.join(a.frames, fn)} —— 帧目录 --frames 给对了吗?")
    H, W = base.shape[:2]
    cam = cams[fn]
    K = np.array(cam["intrinsic_3x3"], float).copy()
    K[0,:] *= W/(K[0,2]*2); K[1,:] *= H/(K[1,2]*2)
    w2c = np.array(cam["extrinsic_w2c_4x4"]).reshape(4,4)
    L, Rr = base.copy(), base.copy()
    nl = draw(L, pca, (80,255,80), K, w2c)
    nr = draw(Rr, rec, (255,90,220), K, w2c)
    cv2.putText(L, f"frame {fi}  PCA (point cloud): {nl} x {a.word}", (12,30),
                cv2.FONT_HERSHEY_SIMPLEX, .85, (80,255,80), 2, cv2.LINE_AA)
    cv2.putText(Rr, f"frame {fi}  WildDet3D recall: {nr} x {a.word}", (12,30),
                cv2.FONT_HERSHEY_SIMPLEX, .85, (255,90,220), 2, cv2.LINE_AA)
    row = np.hstack([L, Rr]); s = (a.width*2)/row.shape[1]
    rows.append(cv2.resize(row, (a.width*2, int(row.shape[0]*s))))
    print(f"[frame {fi}] PCA {nl}, WD3 {nr}   (数字是盒子高度 m)")

cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
print("->", a.out)
