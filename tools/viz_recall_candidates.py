"""召回实例的抠图, 轮廓 + 框画在原帧上, 局部放大。

召回和 SAM3 跟踪不是一回事: SAM3 对这些词返回 0 实例, 是 WildDet3D 全图检测找到位置、
再把 2D 框喂回 SAM3 补抠出来的。所以它没有"候选池"和 VLM 选择题, 每个实例就一张 mask。
选谁去生成 mesh 由 mask_recall.py 定 —— 一个词只派抠得最大的那个当代表 (其余复用)。

用法: viz_recall_candidates.py --run-dir <o1> --frames <frames_o1> --word sofa --out x.jpg
"""
import argparse, json, os
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--word", required=True, help="逗号分隔可以一次看多个词")
ap.add_argument("--out", required=True)
ap.add_argument("--zoom", type=float, default=4.0)
ap.add_argument("--min-win", type=float, default=0.18)
ap.add_argument("--cols", type=int, default=3)
ap.add_argument("--cell", type=int, default=760)
a = ap.parse_args()

R = a.run_dir
GEN = os.path.expanduser("~/sam3d/sam-3d-objects")
ready = json.load(open(f"{R}/14_recall_ready.json"))
jobs = set(open(f"{R}/logs/recall_jobs.txt").read().split())
words = [w.strip() for w in a.word.split(",") if w.strip()]
rows_ = [x for x in ready if x["词"] in words]
rows_.sort(key=lambda x: (words.index(x["词"]), -x["面积占比"]))
if not rows_:
    raise SystemExit(f"[失败] 召回里没有 {words}")

tiles = []
for k, x in enumerate(rows_, 1):
    n = x["生成名"]
    d = f"{GEN}/my_images/{n}"
    img = cv2.imread(f"{d}/frame.png")
    m = cv2.imread(f"{d}/mask.png", 0) > 127
    ih, iw = img.shape[:2]
    gen = n in jobs                      # 这个是被派去生成 mesh 的代表
    col = (60, 230, 60) if gen else (60, 90, 255)
    cv2.drawContours(img, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)[0], -1, col, 2)
    ys, xs = np.nonzero(m)
    cv2.rectangle(img, (xs.min() - 6, ys.min() - 6), (xs.max() + 6, ys.max() + 6), col, 2)
    if a.zoom > 0:
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) * a.zoom / 2
        half = max(half, iw * a.min_win / 2)
        ar_ = ih / iw
        hw = min(half, iw / 2); hh = min(half * ar_, ih / 2)
        x0 = int(np.clip(cx - hw, 0, iw - 2 * hw)); y0 = int(np.clip(cy - hh, 0, ih - 2 * hh))
        img = img[y0:y0 + int(2 * hh), x0:x0 + int(2 * hw)]
    s = a.cell / img.shape[1]
    img = cv2.resize(img, (a.cell, max(int(img.shape[0] * s), 1)))
    sz = "x".join(f"{v:.2f}" for v in x["尺寸"])
    tag = (f'#{k} {x["词"]} obj{x["obj_id"]} f{x["最佳帧"]} {x["面积占比"]:.2f}%  '
           f'iou{x.get("框重合", 0):.2f}  {sz}m' + ("   <<< GENERATED" if gen else "   (reuse)"))
    for c2, t2 in (((0, 0, 0), 5), (col, 2)):
        cv2.putText(img, tag, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, c2, t2, cv2.LINE_AA)
    if gen:
        cv2.rectangle(img, (2, 2), (img.shape[1] - 3, img.shape[0] - 3), col, 6)
    tiles.append(img)

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
    cv2.putText(out, f'recall: {", ".join(words)}   {len(tiles)} instances   green = mesh generated',
                (14, h * len(rows) - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c2, t2, cv2.LINE_AA)
cv2.imwrite(a.out, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(f"[{', '.join(words)}] {len(tiles)} 个召回实例 -> {a.out}")
