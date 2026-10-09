#!/usr/bin/env python
"""新旧两个验收视频并排: 原片 | 老(网格上修) | 新(一手数据上修)。两个输入都是 body_video.py 出的三栏视频, 取前两栏。
用法: python compare_video.py <老.mp4> <新.mp4> <输出.mp4> [标题]"""
import os, subprocess, sys
import cv2, imageio_ffmpeg, numpy as np

old, new, out = sys.argv[1:4]
title = sys.argv[4] if len(sys.argv) > 4 else ""
assert not os.path.exists(out), f"{out} 已存在, 不覆盖"
co, cn = cv2.VideoCapture(old), cv2.VideoCapture(new)
fps = co.get(cv2.CAP_PROP_FPS)
enc, n, BAR = None, 0, 30
while True:
    ok1, a = co.read(); ok2, b = cn.read()
    if not (ok1 and ok2):
        break
    w = a.shape[1] // 3
    row = np.hstack([a[:, :w], a[:, w:2 * w], b[:, w:2 * w]])
    canvas = np.zeros((row.shape[0] + BAR + (row.shape[0] + BAR) % 2, row.shape[1], 3), np.uint8)
    canvas[BAR:BAR + row.shape[0]] = row
    for i, lab in enumerate((f"input  {title}", "old: fixes on meshes", "new: fixes on first-hand joint data")):
        cv2.putText(canvas, lab, (i * w + 10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if enc is None:
        enc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", f"{canvas.shape[1]}x{canvas.shape[0]}", "-r", f"{fps:.3f}", "-i", "-",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], stdin=subprocess.PIPE)
    enc.stdin.write(canvas.tobytes()); n += 1
enc.stdin.close(); enc.wait()
print(f"{out}  {n} 帧 @ {fps:.2f} fps  {canvas.shape[1]}x{canvas.shape[0]}")
