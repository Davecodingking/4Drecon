#!/usr/bin/env python
"""汇报 slide 里要现做的几张图 -> docs/slides/assets/

用 sam3d 环境跑(要 pycocotools + cv2):
    ~/miniforge3/envs/sam3d/bin/python docs/slides/make_assets.py

图上的字一律英文 —— cv2 的默认字体画不出中文, 会变成问号而且不报错。
"""
import os, numpy as np, cv2
from pycocotools import mask as mask_util

SAM3D = os.path.expanduser("~/sam3d")
DATA = f"{SAM3D}/data/outdoor"
OUT = f"{SAM3D}/docs/slides/assets"
os.makedirs(OUT, exist_ok=True)
GREY = 210


def load_word(npz):
    """-> {oid: [(frame, area, 解码函数), ...]}"""
    z = np.load(npz, allow_pickle=True)
    meta, counts = z["meta"], z["counts"]
    h, w = [int(x) for x in z["hw"]]
    by = {}
    for i, r in enumerate(meta):
        dec = (lambda c=counts[i]: mask_util.decode({"counts": c, "size": [h, w]}).astype(bool))
        by.setdefault(int(r["oid"]), []).append((int(r["frame"]), int(r["area"]), dec))
    return by


def frame_img(fr_dir, fi, shape=None):
    img = cv2.imread(f"{fr_dir}/{fi:04d}.jpg")
    return img


def fit(m, img):
    H, W = img.shape[:2]
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


def cutout(img, m, pad=0.12, size=320):
    """灰底抠图, 和联络图同一种画法(灰 210 / 外扩 12% / 居中)。"""
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    s = int(max(x1 - x0, y1 - y0) * (1 + 2 * pad)) + 2
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    canvas = np.full((s, s, 3), GREY, np.uint8)
    src = np.where(m[..., None], img, GREY).astype(np.uint8)
    sx0, sy0 = cx - s // 2, cy - s // 2
    H, W = m.shape
    ax0, ay0, ax1, ay1 = max(sx0, 0), max(sy0, 0), min(sx0 + s, W), min(sy0 + s, H)
    canvas[ay0 - sy0:ay1 - sy0, ax0 - sx0:ax1 - sx0] = src[ay0:ay1, ax0:ax1]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def label(img, text, color=(40, 40, 40)):
    out = cv2.copyMakeBorder(img, 0, 44, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(out, text, (12, img.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- P3 出租车: 只拍到一面 vs 生成的完整车身
fr = cv2.imread(f"{SAM3D}/sam-3d-objects/my_images/ts_r1_taxi/frame.png")
mk = cv2.imread(f"{SAM3D}/sam-3d-objects/my_images/ts_r1_taxi/mask.png", 0) > 127
mk = fit(mk, fr)
ys, xs = np.nonzero(mk)
x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
mx, my = int((x1 - x0) * 0.6), int((y1 - y0) * 0.6)
ctx = fr.copy()
c = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
cv2.drawContours(ctx, c, -1, (60, 220, 60), 3)
H, W = fr.shape[:2]
ctx = ctx[max(y0 - my, 0):min(y1 + my, H), max(x0 - mx, 0):min(x1 + mx, W)]
cv2.imwrite(f"{OUT}/p3_taxi_photo.jpg", ctx, [cv2.IMWRITE_JPEG_QUALITY, 92])
# 生成的 mesh: timesquare 的 mesh 一览图里 taxi 那一格(每格 340, 一行三个词各两视角)
g = cv2.imread(f"{SAM3D}/data/outdoorOutput/timesquare/05_meshes.jpg")
tile = g[14 * 340 + 44:15 * 340 - 10, 1360:2040]      # 顶上 44 px 是格子标题, 裁掉
cv2.imwrite(f"{OUT}/p3_taxi_mesh.jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("P3 ok", ctx.shape, tile.shape)

# ---------------------------------------------------------------- P6 同一个垃圾桶被两个词各抠一遍
FR_B = f"{DATA}/dataset/beach/frames"
A = load_word(f"{DATA}/result/beach/r1/03_masks/bin.npz")[10]
B = {f: (a, d) for f, a, d in load_word(f"{DATA}/result/beach/r1/03_masks/trash_can.npz")[10]}
common = [(f, a + B[f][0], d, B[f][1]) for f, a, d in A if f in B]
fi, _, da, db = max(common, key=lambda t: t[1])
img = frame_img(FR_B, fi)
ma, mb = fit(da(), img), fit(db(), img)
iou = (ma & mb).sum() / max((ma | mb).sum(), 1)
ys, xs = np.nonzero(ma | mb)
x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
pad = int(max(x1 - x0, y1 - y0) * 1.2)
H, W = img.shape[:2]
# 下边只留很少: 这段素材右下角带着系统水印("激活 Windows"), 往下多留就会裁进来
box = (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + int(pad * 0.3), H))
panels = []
for m, name, col in ((ma, "word: bin  #10", (255, 140, 30)), (mb, "word: trash can  #10", (40, 90, 255))):
    ov = img.copy()
    ov[m] = (0.5 * np.array(col) + 0.5 * ov[m]).astype(np.uint8)
    cc = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(ov, cc, -1, col, 2)
    crop = ov[box[1]:box[3], box[0]:box[2]]
    crop = cv2.resize(crop, (420, int(420 * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_CUBIC)
    panels.append(label(crop, name, col))
both = np.hstack([panels[0], np.full((panels[0].shape[0], 16, 3), 255, np.uint8), panels[1]])
cv2.imwrite(f"{OUT}/p6_dedup.jpg", both, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"P6 ok frame {fi} IoU {iou:.3f} common frames {len(common)}")

# ---------------------------------------------------------------- P11 pedestrian: 留下的人 vs 闸口判掉的候选
FR_P = f"{DATA}/dataset/pedestrian/frames"
P = load_word(f"{DATA}/result/pedestrian/r2/03_masks/person.npz")
KEEP, DROP = [0, 1, 2], [5, 6, 8, 13, 15]
tiles = []
for oid in KEEP + DROP:
    f, a, d = max(P[oid], key=lambda t: t[1])
    im = frame_img(FR_P, f)
    t = cutout(im, fit(d(), im), size=260)
    dropped = oid in DROP
    t = label(t, f"{'dropped' if dropped else 'kept'}  #{oid}", (30, 50, 200) if dropped else (60, 140, 60))
    if dropped:
        t = cv2.rectangle(t, (1, 1), (t.shape[1] - 2, t.shape[0] - 2), (30, 50, 200), 3)
    tiles.append(t)
gap = np.full((tiles[0].shape[0], 10, 3), 255, np.uint8)
row = [tiles[0]]
for i, t in enumerate(tiles[1:], 1):
    row += [np.full((t.shape[0], 40 if i == len(KEEP) else 10, 3), 255, np.uint8), t]
cv2.imwrite(f"{OUT}/p11_person.jpg", np.hstack(row), [cv2.IMWRITE_JPEG_QUALITY, 92])
print("P11 ok", {oid: max(P[oid], key=lambda t: t[1])[:2] for oid in KEEP + DROP})

# ---------------------------------------------------------------- P7 召回: 判掉的 vs 召回成功的
# 判掉的: timesquare 核对图第一行前三格 —— 同一辆卡车(truck#2)被当成三个"新物体", IoU 都是 0.74
ts = cv2.imread(f"{SAM3D}/data/outdoorOutput/95_recall_check_2d_timesquare.jpg")
cv2.imwrite(f"{OUT}/p7_rejected.jpg", ts[0:456, 0:1296], [cv2.IMWRITE_JPEG_QUALITY, 92])
# 召回成功的: beach r2 核对图第三格(热狗车, IoU 0.13, 底下是空的) + 框喂回 SAM3 补抠出的 mask
bc = cv2.imread(f"{DATA}/result/beach/r2/95_recall_check_2d.jpg")
tile = bc[0:590, 866:1290]
rm = cv2.imread(f"{DATA}/result/beach/r2/14_recall_masks/be_r2_hot_dog_cart_10001.jpg")
rm = rm[300:640, 960:1360]            # 只取热狗车那一块, 右下角的水印不进来
rm = cv2.resize(rm, (int(rm.shape[1] * 590 / rm.shape[0]), 590), interpolation=cv2.INTER_CUBIC)
cv2.rectangle(rm, (0, 0), (rm.shape[1] - 1, 30), (0, 0, 0), -1)
cv2.putText(rm, "re-masked by SAM3", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(f"{OUT}/p7_recalled.jpg", np.hstack([tile, np.full((590, 12, 3), 255, np.uint8), rm]),
            [cv2.IMWRITE_JPEG_QUALITY, 92])
print("P7 ok")

# ---------------------------------------------------------------- P12 闸 4 的两张决策台截图
# 截图本身是无头浏览器截的(pedestrian r2 / beach r2 的 --stage post), 这里只裁。
# 召回表只留弹层左半(右半是空的); 召回卡片只留两张卡 —— beach r2 早于召回去向表,
# 顶栏会显示 "Recall 0/8 kept", 和下面的 2 个召回候选对不上, 所以顶栏不要。
t = cv2.imread(f"{OUT}/p12_recall_table.png")
if t is not None:
    cv2.imwrite(f"{OUT}/p12_recall_table_crop.png", t[55:2000, 262:1720])
c = cv2.imread(f"{OUT}/p12_recalled_cards.png")
if c is not None:
    cv2.imwrite(f"{OUT}/p12_recalled_cards_crop.png", c[1327:1973, 37:878])
print("P12 crops", t is not None, c is not None)

# ---------------------------------------------------------------- 现成大图里裁一块
# 渲染回图是 5 行 x 3 栏(实拍 | 重建 | 叠加), 每行 239 px 高。
# 每行顶上 24 px 有调试小字("0080.jpg real | rebuilt 50/50"), 逐行裁掉再拼
def rows(src, r0, r1, dst, top=24):
    im = cv2.imread(src)
    h = im.shape[0] / 5
    parts = [im[int(r * h) + top:int((r + 1) * h)] for r in range(r0, r1)]
    cv2.imwrite(dst, np.vstack(parts), [cv2.IMWRITE_JPEG_QUALITY, 92])
rows(f"{SAM3D}/data/outdoorOutput/timesquare/08_render_back_phy.jpg", 2, 4, f"{OUT}/p1_timesquare.jpg")
rows(f"{SAM3D}/data/indoorOutput/cyberpunk/o1/08_render_back_phy.jpg", 1, 3, f"{OUT}/p10_cyberpunk.jpg")
# pedestrian 第 0088 帧: 行人和出租车挤在一起, 最能看出校准后人和车的大小关系对不对
rows(f"{SAM3D}/data/outdoorOutput/pedestrian/08_render_back_phy.jpg", 2, 3, f"{OUT}/p9_pedestrian.jpg")
# 流程图里闸 2 那一整块(闸 -> 三个判断 -> 回跑 -> 汇回主线), 坐标来自 pipeline_figure.py,
# PNG 是按 2 倍导出的(3000 宽)
fig = cv2.imread(f"{OUT}/pipeline_figure.png")
cv2.imwrite(f"{OUT}/p12_gate2.png", fig[1260:2270, 280:2000])
print("crops ok")
