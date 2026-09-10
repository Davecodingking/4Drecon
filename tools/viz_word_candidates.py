"""某个词的全部候选 + 最终选中, mask 画在**整张原图**上 (不是抠图特写)。

抠图特写看不出这个东西在场景里的哪个位置、旁边是什么, 判断不了抠得对不对。
候选顺序和 verify_picks.py 的排序一致 (面积大且不贴边优先), 编号也一致 ——
就是 VLM 做选择题时看到的那个编号。

用法: viz_word_candidates.py --run-dir <o1> --frames <frames_o1> --word lamp --out x.jpg
"""
import argparse, json, os
import numpy as np, cv2
from pycocotools import mask as mask_util

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--word", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--topk", type=int, default=8)
ap.add_argument("--min-area-pct", type=float, default=0.05)
ap.add_argument("--cols", type=int, default=3)
ap.add_argument("--cell", type=int, default=760)
ap.add_argument("--zoom", type=float, default=4.0,
                help="裁剪窗口是 mask 外接框的几倍。1=只有物体, 大=看得到周围环境。0=不裁, 出整帧")
ap.add_argument("--min-win", type=float, default=0.18,
                help="裁剪窗口至少占画面宽度的多少, 挡住太小的物体被放大成马赛克")
a = ap.parse_args()

R = a.run_dir
z = np.load(f"{R}/03_masks/{a.word.replace(' ', '_')}.npz", allow_pickle=True)
meta, counts = z["meta"], z["counts"]
H, W = [int(x) for x in z["hw"]]
have = {int(f[:4]) for f in os.listdir(f"{R}/02_da3/depth") if f.endswith("_depth.npy")}

# 和 verify_picks.py:55-64 一样的打分: 面积大、不贴边、活帧多
live = {}
for i, (oid, fi, ar, e) in enumerate(meta):
    if ar > 0 and int(fi) in have:
        live.setdefault(int(oid), []).append((i, int(fi), int(ar), float(e)))
cands = []
for oid, rows in live.items():
    if len(rows) < 2:
        continue
    nf = min(len(rows) / 10, 1.0)
    i, fi, ar, e = max(rows, key=lambda t: t[2] * (1 - min(t[3] * 4, 0.9)))
    cands.append((ar * (1 - min(e * 4, 0.9)) * nf, oid, i, fi, ar, e, len(rows)))
cands.sort(reverse=True)
cands = [c for c in cands if 100 * c[4] / (H * W) >= a.min_area_pct][:a.topk]

pick = json.load(open(f"{R}/04_picks/picks.json")).get(a.word, {})
tiles = []
for k, (sc_, oid, i, fi, ar, e, nfr) in enumerate(cands, 1):
    img = cv2.imread(os.path.join(a.frames, f"{fi:04d}.jpg"))
    m = mask_util.decode({"counts": counts[i], "size": [H, W]}).astype(bool)
    if m.shape != img.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    chosen = (oid == pick.get("obj_id") and fi == pick.get("frame"))
    col = (60, 230, 60) if chosen else (60, 90, 255)
    # ★ 只画轮廓和框, 不给里面上色 —— 半透明填充会盖住物体本身的样子, 看不出抠的是什么。
    # 这一版就是因为填了色, 把烛台和灯柱都看成一团红方块, 判断不了对错。
    cv2.drawContours(img, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)[0], -1, col, 2)
    ys, xs = np.nonzero(m)
    cv2.rectangle(img, (xs.min() - 6, ys.min() - 6), (xs.max() + 6, ys.max() + 6), col, 2)
    ih, iw = img.shape[:2]
    if a.zoom > 0:
        # 以 mask 为中心裁一块, 边长 = 外接框 x zoom, 但不小于画面宽的 min-win ——
        # 这个场景的候选常常只占 0.06% 画面 (40x60 像素), 只裁物体本身会放大成马赛克,
        # 而看整帧又根本找不到它在哪。带一圈环境才判断得了抠得对不对。
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = max((xs.max() - xs.min()), (ys.max() - ys.min())) * a.zoom / 2
        half = max(half, iw * a.min_win / 2)
        ar_ = ih / iw                                   # 保持原始画面比例
        hw = min(half, iw / 2); hh = min(half * ar_, ih / 2)
        x0 = int(np.clip(cx - hw, 0, iw - 2 * hw)); y0 = int(np.clip(cy - hh, 0, ih - 2 * hh))
        img = img[y0:y0 + int(2 * hh), x0:x0 + int(2 * hw)]
    s = a.cell / img.shape[1]
    img = cv2.resize(img, (a.cell, max(int(img.shape[0] * s), 1)))
    tag = f"#{k}  obj{oid} f{fi}  {100*ar/(H*W):.2f}%" + ("   <<< PICKED" if chosen else "")
    for c2, t2 in (((0, 0, 0), 5), (col, 2)):
        cv2.putText(img, tag, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, c2, t2, cv2.LINE_AA)
    if chosen:
        cv2.rectangle(img, (2, 2), (img.shape[1] - 3, img.shape[0] - 3), col, 6)
    tiles.append(img)

if not tiles:
    raise SystemExit(f"[失败] {a.word} 没有候选")
h = max(t.shape[0] for t in tiles)
tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in tiles]
rows = []
for i in range(0, len(tiles), a.cols):
    r = tiles[i:i + a.cols]
    while len(r) < a.cols:
        r.append(np.full_like(tiles[0], 255))
    rows.append(np.hstack(r))
out = np.vstack(rows)
for c2, t2 in (((0, 0, 0), 6), ((255, 255, 255), 2)):
    cv2.putText(out, f'"{a.word}"  {len(tiles)} candidates   green = picked',
                (14, h * len(rows) - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.1, c2, t2, cv2.LINE_AA)
cv2.imwrite(a.out, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(f"[{a.word}] {len(tiles)} 个候选, 选中 obj{pick.get('obj_id')} / frame {pick.get('frame')}")
print("->", a.out)
