#!/usr/bin/env python
"""把召回实例的盒子并进主盒子文件, 让它们能进场景组装。

召回实例的盒子来自 WildDet3D (13_recall.json), 格式和 06_all_boxes.json 一致,
但字段名不同(它是检测阶段的记法), 这里翻译一下再合并。
只并"补抠出 mask 且真的生成了 mesh"的那些 —— 有盒子没 mesh 摆不出东西。
输出 06_all_boxes_merged.json, 原文件不动, 想切回去只要换 BOXES 环境变量。
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_frame import load_up, export_T, inverse_T
R1  = os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
TAG = os.environ.get("TAG","ts_r1")
GEN = os.environ.get("GEN_OUT", os.path.expanduser("~/sam3d/sam-3d-objects/outputs"))
SRC = os.environ.get("BOXES", f"{R1}/06_all_boxes.json")
OUT = os.environ.get("MERGED", f"{R1}/06_all_boxes_merged.json")
# ★ 天方向以前写死 [0,-1,0], 而依赖表里却声明这一步依赖 up —— 依赖表在说假话。
#   后果: 俯拍场景真实的天偏 26~29°, 这里算出来的朝向/高度系统性歪掉。
UP, _GH, _FB  = load_up(f"{R1}/02_da3", "召回并盒子")

base = json.load(open(SRC))
n0 = sum(len(v) for v in base.values())
_rp = f"{R1}/14_recall_ready.json"
# ★ 这一步现在排在 assemble 之前, 崩了整条链出不了场景。没有召回结果不是错误 ——
#   原样把基础盒子复制成 merged, 让下游照常走。
if not os.path.exists(_rp):
    # 变量名是 OUT, 不是 MERGED —— MERGED 只是那个环境变量的名字。
    # 写成 MERGED 的话这条"让下游照常走"的兜底分支自己 NameError, 正好在没有召回结果时炸。
    json.dump(base, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"[跳过] 没有 {os.path.basename(_rp)}, 这一轮没有召回结果。"
          f"原样复制 {n0} 个盒子 -> {os.path.basename(OUT)}")
    raise SystemExit(0)
ready = json.load(open(_rp))
added = skipped = 0
for a in ready:
    name = a["生成名"]
    if not os.path.exists(f"{GEN}/{name}/{name}.glb"):
        skipped += 1; continue                      # 没生成出 mesh 的不并, 摆不出东西
    ctr = np.array(a["中心"]); ext = np.array(a["尺寸"])
    base.setdefault(a["词"], []).append(dict(
        中心=ctr.tolist(), 长轴=a["长轴"], 尺寸=ext.tolist(),
        底高=float(ctr@UP - ext[2]/2), obj_id=a["obj_id"],
        点数=0, 用帧=a["见帧数"], 总帧=a["见帧数"],
        朝向分歧=a.get("朝向分歧",0), 来源="召回"))
    added += 1
json.dump(base, open(OUT,"w"), ensure_ascii=False, indent=1)
print(f"[合并] 原有 {n0} + 召回 {added} = {sum(len(v) for v in base.values())} 个实例"
      f"  (没生成出 mesh 跳过 {skipped})")
print(f"[done] -> {OUT}")
