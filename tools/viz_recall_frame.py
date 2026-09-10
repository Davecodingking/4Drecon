"""三栏对比图: 原图 | SAM3 抠到的 mask | 加上 WildDet3D 召回的盒子。

召回实例每个只在"最佳帧"上有一张 mask, 所以第三栏画的是把它的 3D 盒子投影回这一帧,
这样任意机位都能看出召回补上了哪些东西。SAM3 已有的画实心 mask, 召回的画线框。

坐标系: 13_recall.json 的 中心/尺寸/长轴 都在 DA3 世界系。尺寸 = [长(沿长轴), 宽, 高]。
"天"从 world_frame 拿, 不写死 —— 老假设 [0,-1,0] 实测在 beach 偏 6.53°、carpark 偏 29.2°,
而这张图是人拿来判断"召回补上了哪些东西"的依据, 盒子沿歪掉的方向立起来看不出问题。
"""
import argparse, colorsys, json, os, sys
import numpy as np, cv2
from pycocotools import mask as mask_util
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
ap.add_argument("--frames-idx", default="0,26,53,79")
ap.add_argument("--out", required=True)
ap.add_argument("--min-area", type=int, default=200)
ap.add_argument("--width", type=int, default=900)
a = ap.parse_args()

R1 = a.run_dir
UP, _GH, _ = load_up(f"{R1}/02_da3", "召回三栏图")
cams = json.load(open(f"{R1}/02_da3/cameras.json"))["cameras"]
rec = json.load(open(f"{R1}/13_recall.json"))["召回"]
mdir = f"{R1}/03_masks"

npz = {}
for f in sorted(os.listdir(mdir)):
    if f.endswith(".npz"):
        z = np.load(f"{mdir}/{f}", allow_pickle=True)
        if len(z["meta"]) and z["meta"]["area"].max() > 0:
            npz[str(z["word"])] = z

words = sorted(set(npz) | {d["词"] for d in rec})
COL = {w: tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / max(len(words), 1), .85, 1.)[::-1])
       for i, w in enumerate(words)}

def corners(d):
    """世界系 8 个角点。"""
    c = np.array(d["中心"]); L, W_, H_ = d["尺寸"]
    e1 = np.array(d["长轴"]); e1 = e1 - UP * (e1 @ UP)
    e1 /= max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(UP, e1)
    return np.array([c + i * L / 2 * e1 + j * W_ / 2 * e2 + k * H_ / 2 * (-UP)
                     for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])

EDGES = [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]

panels = []
for fi in [int(x) for x in a.frames_idx.split(",")]:
    # 抽帧编号和 DA3 出位姿的编号不一定一一对应(抽帧密、位姿稀), 硬索引 cams 会 KeyError。
    # 换成最近的那一帧, 而且原图/mask/位姿三者一起挪 —— 只挪位姿的话图和框会错位。
    fn = f"{fi:04d}.jpg"
    if fn not in cams:
        fn = min(cams, key=lambda k: abs(int(k[:4]) - fi))
        print(f"[注意] 第 {fi} 帧没位姿, 改用最近的 {fn}")
        fi = int(fn[:4])
    img = cv2.imread(os.path.join(a.frames, fn))
    if img is None:
        raise SystemExit(f"[失败] 读不到帧 {os.path.join(a.frames, fn)} —— 帧目录 --frames 给对了吗?")
    H, W = img.shape[:2]
    sam = img.copy()
    hits = []
    for word, z in npz.items():
        meta, counts = z["meta"], z["counts"]
        h, w = [int(x) for x in z["hw"]]
        for i, r in enumerate(meta):
            if int(r["frame"]) != fi or int(r["area"]) < a.min_area:
                continue
            m = mask_util.decode({"counts": counts[i], "size": [h, w]}).astype(bool)
            if (h, w) != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), cv2.INTER_NEAREST).astype(bool)
            hits.append((word, int(r["oid"]), m))
    for word, oid, m in hits:
        sam[m] = (.45 * np.array(COL[word]) + .55 * sam[m]).astype(np.uint8)
        cv2.drawContours(sam, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)[0], -1, COL[word], 2)

    both = sam.copy()
    cam = cams[fn]
    K = np.array(cam["intrinsic_3x3"], float).copy()
    K[0, :] *= W / (K[0, 2] * 2); K[1, :] *= H / (K[1, 2] * 2)
    w2c = np.array(cam["extrinsic_w2c_4x4"]).reshape(4, 4)
    n_new = 0
    for d in rec:
        P = (w2c[:3, :3] @ corners(d).T).T + w2c[:3, 3]
        if (P[:, 2] < .1).any():                     # 有角点在相机后面, 投影会翻折
            continue
        uv = (K @ P.T).T; uv = uv[:, :2] / uv[:, 2:3]
        if uv[:, 0].max() < 0 or uv[:, 0].min() > W or uv[:, 1].max() < 0 or uv[:, 1].min() > H:
            continue
        col = COL[d["词"]]
        for i, j in EDGES:
            cv2.line(both, tuple(uv[i].astype(int)), tuple(uv[j].astype(int)), col, 2, cv2.LINE_AA)
        p = uv.mean(0).astype(int)
        cv2.putText(both, d["词"], (p[0] - 30, p[1]), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(both, d["词"], (p[0] - 30, p[1]), cv2.FONT_HERSHEY_SIMPLEX, .45, col, 1, cv2.LINE_AA)
        n_new += 1

    row = np.hstack([img, sam, both])
    s = (a.width * 3) / row.shape[1]
    row = cv2.resize(row, (a.width * 3, int(row.shape[0] * s)))
    cv2.putText(row, f"frame {fi}   photo | SAM3 only: {len(hits)} masks | + WildDet3D recall: {n_new} boxes",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
    panels.append(row)
    print(f"[frame {fi}] SAM3 {len(hits)} 个 mask, 召回盒子 {n_new} 个")

cv2.imwrite(a.out, np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 92])
print("->", a.out)
