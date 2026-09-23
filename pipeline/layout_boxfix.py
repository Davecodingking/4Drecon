#!/usr/bin/env python
"""盒子掰正 + 去穿模, 独立成步 (原先藏在 SpatialGen/prepare_spatiallm_scene_5090.py 里).

下游每一个消费者 (hybrid_match / hybrid_place / floor_texture / wall_texture) 读的都是
"修复后"的布局, 不是 SpatialLM 的原始输出。这一步以前是 SpatialGen 输入准备脚本的副作用,
去掉 SpatialGen 那条支线就一起丢了 —— 所以在这里单拎出来。

算法不重写, 直接 import 那个脚本里的 fix_boxes (它自身只用 numpy + shapely,
顶层 import 也很轻, spatiallm 环境两者都有):
  1) 掰正:   水平转角离 0/90 的倍数 <= snap_deg 就吸附 (点云漂移让 SpatialLM 估歪角度)
  2) 去穿模: 两盒水平相交且高度重叠 -> 小盒向自己中心等比缩, 最多缩 30% (椅子插桌子里那种)
  carpet/curtain 不参与去穿模 (地毯本来被压着, 窗帘贴窗)

墙/门/窗行原样保留, bbox_N 的编号也原样保留 —— 下游全靠这个 idx 对应物体。

★ 物体链有对应的一步: data/outdoor/scripts/assemble_scene.py 的 YAW_SNAP / snap_yaw()。
  那边吸附目标不是 0/90 —— 物体链的 assemble 跑在未旋转的 DA3 系里, 坐标轴和墙无关,
  所以它从房间壳的墙线里按长度加权数出建筑主方向再吸。同一件事, 两个坐标系。
  改这里的掰正规则时想想那边。

用法: python layout_boxfix.py --layout 04_布局.txt --out 05_布局_掰正.txt
"""
import argparse, os, re, sys

SPATIALGEN = os.path.expanduser("~/sam3d/SpatialGen")

ap = argparse.ArgumentParser()
ap.add_argument("--layout", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--snap-deg", type=float, default=30.0, help="转角离 0/90 倍数多少度以内就吸附")
a = ap.parse_args()

if os.path.abspath(a.layout) == os.path.abspath(a.out):
    sys.exit("[错误] --out 不能覆盖 --layout, 换个输出名")

sys.path.insert(0, SPATIALGEN)
from prepare_spatiallm_scene_5090 import fix_boxes   # noqa: E402

lines = [l.rstrip("\n") for l in open(a.layout)]
idx, boxes = [], []
for i, line in enumerate(lines):
    m = re.match(r"(\w+)=Bbox\((.*)\)\s*$", line.strip())
    if not m: continue
    v = m.group(2).split(",")
    idx.append(i); boxes.append((v[0], *[float(x) for x in v[1:]]))

if not boxes:
    print("[info] 布局里没有 Bbox, 原样复制")
    open(a.out, "w").write("\n".join(lines) + "\n"); sys.exit(0)

fixed = fix_boxes(boxes, snap_deg=a.snap_deg)
changed = 0
for i, old, new in zip(idx, boxes, fixed):
    if old == new: continue
    name = lines[i].split("=")[0]
    lines[i] = f"{name}=Bbox({new[0]}," + ",".join(f"{x}" for x in new[1:]) + ")"
    changed += 1

open(a.out, "w").write("\n".join(lines) + "\n")
print(f"[done] {len(boxes)} 个盒子, 改了 {changed} 个 -> {a.out}")
