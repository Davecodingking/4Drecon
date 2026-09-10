#!/usr/bin/env python
"""视频 -> 抽稀+等比缩放的帧序列, 给 SAM-Body4D 用。

为什么不能直接把原片丢给它 —— 三个理由, 都是量出来的:

  1. Diffusion-VAS 的底子是 Stable Video Diffusion, 上游仓库明确按 ~8fps 喂。
     我们的素材是 30fps, 隔 4 帧取一 = 7.5fps, 时间轴正好整除, 不用重采样。
  2. 论文自测是 480x854 / 90 帧 / 5 人 / A100-80G。我们只有 32G, 原片 1080p / 269 帧
     直接喂进去, 光 SAM3 的 video_segments 那个 dict (每帧每人一张 bool 全图)
     就是 GB 级。
  3. SAM3 传播的显存和帧数 x 人数 x 面积成正比, 这是唯一能线性砍的地方。

★ 补零四位。这个仓库踩过的最大的坑是 DA3 按字典序读帧, 0,1,10,100,... 把视频读乱。
  这里还多一条: offline_app.py 对目录输入做的是 glob('*.jpg') 然后 image_list.sort(),
  纯字典序; 不补零, RGB 序列和后面的 mask 序列会各排各的, 而且一声不吭。
  (注意他们自己写出来的 images/ 是八位补零, 和我们这里的四位不是一回事, 别搞混。)

★ 宽高比一律等比缩放, 不补黑边也不裁。理由是读了 Diffusion-VAS 的代码之后确定的:
  - 它内部 resize 到固定的 512x1024 是纯拉伸, 不管你喂什么比例一律压成 2:1,
    出来再拉回原尺寸。所以补黑边挡不住这次压缩, 只是让被压的内容里多一半是黑边。
  - 补黑边还会主动坏事: MoGe-2 是从整幅图估焦距的, 黑边把视场角估歪,
    歪的焦距直接进 SAM 3D Body 的相机, 网格的深度和尺度跟着错。
  - 中心裁到 2:1 会切掉 mayday 16% 的宽度, 打斗场景边上的人正好在那儿。
  结论: 保持真实像素比是唯一不主动引入错误的选项。

用法:
  python body4d_frames.py --video a.mp4 --out 帧目录 [--fps 7.5] [--long-side 1024]
                          [--max-frames 96] [--start-sec 0] [--quality 95]
"""
import argparse, json, os, sys
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--out", required=True, help="帧目录; 同时写一份 frames.json 记录抽了哪些帧")
ap.add_argument("--fps", type=float, default=7.5, help="目标帧率。stride = round(源fps/它)")
ap.add_argument("--stride", type=int, default=0, help="非 0 则直接指定隔几帧取一, 忽略 --fps")
ap.add_argument("--long-side", type=int, default=1024, help="长边缩到多少。0 = 不缩")
ap.add_argument("--max-frames", type=int, default=96, help="帧数上限")
ap.add_argument("--start-sec", type=float, default=0.0)
ap.add_argument("--quality", type=int, default=95)
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
if not cap.isOpened():
    raise SystemExit(f"[失败] 打不开视频: {a.video}")
SW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
SH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
NF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

stride = a.stride if a.stride > 0 else max(1, int(round(FPS / a.fps)))
start = int(round(a.start_sec * FPS))

# 目标尺寸: 等比, 长边定到 long_side。不缩就是原尺寸。
if a.long_side and max(SW, SH) > a.long_side:
    s = a.long_side / max(SW, SH)
    OW, OH = int(round(SW * s)), int(round(SH * s))
else:
    OW, OH = SW, SH

src_aspect, out_aspect = SW / SH, OW / OH
print(f"[源] {SW}x{SH} 比例 {src_aspect:.4f}  {NF} 帧 @ {FPS:.2f}fps")
print(f"[抽] 隔 {stride} 帧取一 -> {FPS/stride:.2f}fps, 从第 {start} 帧起, 上限 {a.max_frames} 帧")
print(f"[缩] -> {OW}x{OH} 比例 {out_aspect:.4f}  (误差 {abs(out_aspect/src_aspect-1)*100:.3f}%)")

# ★ 自查: 等比缩放算错了要当场炸, 不要等到下游才发现人被压扁。
#   0.5% 的容差是留给整数取整的 (1066 -> 430 那种)。
if abs(out_aspect / src_aspect - 1) > 0.005:
    raise SystemExit(f"[失败] 缩放不等比: 源 {src_aspect:.4f} -> 出 {out_aspect:.4f}。"
                     f"这是本脚本自己的 bug, 不是素材问题")

os.makedirs(a.out, exist_ok=True)
for f in os.listdir(a.out):
    if f.endswith((".jpg", ".json")):
        os.remove(os.path.join(a.out, f))   # 换配置就该换一整套帧, 不能和上一轮混着

kept, i, n = [], 0, 0
while n < a.max_frames:
    ok, fr = cap.read()
    if not ok:
        break
    if i >= start and (i - start) % stride == 0:
        if (OW, OH) != (SW, SH):
            # INTER_AREA 是缩小的正确选择: 它是区域平均, 不会像 INTER_LINEAR 那样丢细节产生锯齿。
            # 远处的人本来就只有几十个像素, 这里省不得。
            fr = cv2.resize(fr, (OW, OH), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{a.out}/{n:04d}.jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        kept.append(i)
        n += 1
    i += 1
cap.release()

meta = dict(video=os.path.abspath(a.video), src_fps=FPS, src_wh=[SW, SH], src_frames=NF,
            stride=stride, start_frame=start, out_fps=FPS / stride,
            out_wh=[OW, OH], src_aspect=src_aspect, out_aspect=out_aspect,
            n_frames=n, kept_indices=kept)
json.dump(meta, open(f"{a.out}/frames.json", "w"), ensure_ascii=False, indent=1)
print(f"[done] 抽出 {n} 帧 -> {a.out}  (0000.jpg .. {n-1:04d}.jpg, 四位补零)")
