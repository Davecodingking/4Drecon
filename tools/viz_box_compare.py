"""把同一个物体的几种盒子画在同一帧上比对。

盒子来源都是 DA3 世界系, 字段 中心/长轴/尺寸([长,宽,高])。
"天"从 world_frame 拿, 不写死 —— 老假设 [0,-1,0] 实测在 beach 偏 6.53°、carpark 偏 29.2°,
而这张图正是人用来挑"哪一版盒子对"的依据, 三版一起沿歪掉的方向立起来是看不出来的。
用法: viz_box_compare.py --run-dir <o1> --frames <dir> --word "grand piano" \
        --frames-idx 42,48,53,57 --out x.jpg
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
ap.add_argument("--frames-idx", default="42,48,53,57")
ap.add_argument("--out", required=True)
ap.add_argument("--width", type=int, default=960)
a = ap.parse_args()

R = a.run_dir
UP, _GH, _ = load_up(f"{R}/02_da3", "盒子对比图")
cams = json.load(open(f"{R}/02_da3/cameras.json"))["cameras"]

def pick(path):
    if not os.path.exists(path): return None
    b = json.load(open(path))
    for k, v in b.items():
        if k == a.word and v: return v[0]
    return None

SRC = [("PCA (point cloud)",     pick(f"{R}/06_all_boxes.json"),          (80, 255, 80)),
       ("WildDet3D geometric",   pick(f"{R}/06_all_boxes_wilddet.json"),  (60, 160, 255))]

det = f"{R}/12_wd_detections.json"
if os.path.exists(det):
    p = [x for x in json.load(open(det))["检测"] if x["词"] == a.word]
    # 直接取 48 个检测的中位数是错的: 里面混着别处的假阳性(分数低但位置差十几米),
    # 中位数会落到两堆之间的空地上。按位置贪心聚类, 取成员最多的那一簇。
    if p:
        cl = []
        for x in sorted(p, key=lambda d: -d["分"]):
            c = np.array(x["中心"]); hit = None
            for g in cl:
                if np.linalg.norm(c - g["c"]) < 1.5: hit = g; break
            if hit is None: cl.append(dict(c=c, m=[x]))
            else: hit["m"].append(x); hit["c"] = np.median([y["中心"] for y in hit["m"]], 0)
        p = max(cl, key=lambda g: len(g["m"]))["m"]
        print(f"    文本模式: {len(p)} 个检测聚成主簇 (原 {len(json.load(open(det))['检测'])} 中属于 {a.word} 的那些)")
        A = np.array([x["长轴"] for x in p]); s = np.sign(A @ A[0]); s[s == 0] = 1
        v = (A * s[:, None]).mean(0); v /= max(np.linalg.norm(v), 1e-9)
        SRC.append(("WildDet3D text-mode", dict(中心=np.median([x["中心"] for x in p], 0).tolist(),
                                                长轴=v.tolist(),
                                                尺寸=np.median([x["尺寸"] for x in p], 0).tolist()),
                    (255, 90, 220)))

EDGES = [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]
def corners(d):
    c = np.array(d["中心"]); L, W_, H_ = d["尺寸"]
    e1 = np.array(d["长轴"]); e1 = e1 - UP * (e1 @ UP); e1 /= max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(UP, e1)
    return np.array([c + i*L/2*e1 + j*W_/2*e2 + k*H_/2*(-UP)
                     for i in (-1,1) for j in (-1,1) for k in (-1,1)])

rows = []
for fi in [int(x) for x in a.frames_idx.split(",")]:
    # 抽帧编号和 DA3 出位姿的编号不一定一一对应(抽帧密、位姿稀), 硬索引 cams 会 KeyError。
    # 换成最近的那一帧, 而且原图和位姿一起挪 —— 只挪位姿的话图和框会错位。
    fn = f"{fi:04d}.jpg"
    if fn not in cams:
        fn = min(cams, key=lambda k: abs(int(k[:4]) - fi))
        print(f"[注意] 第 {fi} 帧没位姿, 改用最近的 {fn}")
        fi = int(fn[:4])
    img = cv2.imread(os.path.join(a.frames, fn))
    if img is None:
        raise SystemExit(f"[失败] 读不到帧 {os.path.join(a.frames, fn)} —— 帧目录 --frames 给对了吗?")
    H, W = img.shape[:2]
    cam = cams[fn]
    K = np.array(cam["intrinsic_3x3"], float).copy()
    K[0,:] *= W/(K[0,2]*2); K[1,:] *= H/(K[1,2]*2)
    w2c = np.array(cam["extrinsic_w2c_4x4"]).reshape(4,4)
    for name, d, col in SRC:
        if d is None: continue
        P = (w2c[:3,:3] @ corners(d).T).T + w2c[:3,3]
        if (P[:,2] < .1).any(): continue
        uv = (K @ P.T).T; uv = uv[:,:2] / uv[:,2:3]
        for i, j in EDGES:
            cv2.line(img, tuple(uv[i].astype(int)), tuple(uv[j].astype(int)), col, 3, cv2.LINE_AA)
    y = 34
    cv2.putText(img, f"frame {fi}  [{a.word}]", (12, y), cv2.FONT_HERSHEY_SIMPLEX, .9, (255,255,255), 2, cv2.LINE_AA)
    for name, d, col in SRC:
        if d is None: continue
        y += 34
        L, W_, H_ = d["尺寸"]
        cv2.putText(img, f"{name}:  {L:.2f} x {W_:.2f} x {H_:.2f} m", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, col, 2, cv2.LINE_AA)
    s = a.width / W
    rows.append(cv2.resize(img, (a.width, int(H*s))))

cv2.imwrite(a.out, np.vstack([np.hstack(rows[:2]), np.hstack(rows[2:4])]) if len(rows) == 4
            else np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 93])
for name, d, col in SRC:
    if d is not None:
        print(f"{name:22s} {np.round(d['尺寸'],2)}")
print("->", a.out)
