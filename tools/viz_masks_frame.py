"""把某一帧上所有 SAM3 跟踪到的 mask 画出来。

用法: python viz_masks_frame.py --run-dir <o1> --frames <frames_o1> --frame 26 --out x.jpg
mask 存的是 pycocotools RLE, 每条记录是 (oid, frame, area, edge)。
帧号是**跟踪序列内的序号**(0..N-1), 不是原视频帧号; 映射见 frames_*_映射.json。
"""
import argparse, colorsys, json, os
import numpy as np, cv2
from pycocotools import mask as mask_util

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--frame", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-area", type=int, default=200)
a = ap.parse_args()

img = cv2.imread(os.path.join(a.frames, f"{a.frame:04d}.jpg"))
H, W = img.shape[:2]
mdir = os.path.join(a.run_dir, "03_masks")

# 收集这一帧上的所有 mask
hits = []
for f in sorted(os.listdir(mdir)):
    if not f.endswith(".npz"):
        continue
    z = np.load(os.path.join(mdir, f), allow_pickle=True)
    meta, counts = z["meta"], z["counts"]
    h, w = [int(x) for x in z["hw"]]
    word = str(z["word"])
    for i, r in enumerate(meta):
        if int(r["frame"]) != a.frame or int(r["area"]) < a.min_area:
            continue
        m = mask_util.decode({"counts": counts[i], "size": [h, w]}).astype(bool)
        if (h, w) != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        hits.append((word, int(r["oid"]), int(r["area"]), m))

hits.sort(key=lambda x: -x[2])                       # 大的先画, 小的压在上面
words = sorted({h[0] for h in hits})
COL = {w: tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / max(len(words), 1), 0.85, 1.0)[::-1])
       for i, w in enumerate(words)}

ov = img.copy()
for word, oid, area, m in hits:
    ov[m] = (0.45 * np.array(COL[word]) + 0.55 * ov[m]).astype(np.uint8)
for word, oid, area, m in hits:
    c = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(ov, c, -1, COL[word], 2)
    ys, xs = np.nonzero(m)
    cv2.putText(ov, f"{word}#{oid}", (int(xs.mean()) - 40, int(ys.mean())),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(ov, f"{word}#{oid}", (int(xs.mean()) - 40, int(ys.mean())),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL[word], 1, cv2.LINE_AA)

out = np.hstack([img, ov])
cv2.putText(out, f"frame {a.frame}   {len(hits)} masks / {len(words)} words",
            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite(a.out, out)

from collections import Counter
print(f"[frame {a.frame}] {len(hits)} 个 mask, {len(words)} 个词")
for w, n in Counter(h[0] for h in hits).most_common():
    ar = [h[2] for h in hits if h[0] == w]
    print(f"   {w:14s} {n:2d} 个   面积 {min(ar)/(H*W)*100:.2f}~{max(ar)/(H*W)*100:.2f}%")
print("->", a.out)
