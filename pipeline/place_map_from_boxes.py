#!/usr/bin/env python
"""从 06_盒子映射.json + 已生成的 mesh, 拼出 hybrid_place_5090.py --map 要的那一行。

规则和 outdoor 的 assemble_scene.pick_mesh 一样: 有实例专属 mesh 就用它, 没有就用同类共用的。
"同一个名字出现多次 = 复用" 是 hybrid_place 天然支持的 —— 它逐条 idx:名字 加载 glb 再按
各自盒子的 yaw/尺寸归位, 所以 20 把餐椅共用一个 mesh 各自缩放, place 脚本一行不用改。

用法: python place_map_from_boxes.py --map-json 06_盒子映射.json --tag cyb_o1 \
        --outputs ~/sam3d/sam-3d-objects/outputs --out logs/place_map.txt
"""
import argparse, json, os, collections

ap = argparse.ArgumentParser()
ap.add_argument("--map-json", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--outputs", default=os.path.expanduser("~/sam3d/sam-3d-objects/outputs"))
ap.add_argument("--out", required=True)
a = ap.parse_args()

have = {d for d in os.listdir(a.outputs) if os.path.exists(f"{a.outputs}/{d}/{d}.glb")}
m = json.load(open(a.map_json))
pairs, miss, stat = [], collections.Counter(), collections.Counter()
for idx in sorted(m, key=int):
    w = m[idx]["word"]; oid = m[idx].get("obj_id")
    u = w.replace(" ", "_")
    solo = f"{a.tag}_{u}_{oid}"
    cls  = f"{a.tag}_{u}"
    name = solo if solo in have else (cls if cls in have else None)
    if name is None: miss[w] += 1; continue
    pairs.append(f"{idx}:{name}"); stat[("专属" if name == solo else "复用") + " " + w] += 1

open(a.out, "w").write(",".join(pairs) + "\n")
print(f"[done] {len(pairs)} 个盒子有 mesh -> {a.out}")
for k, v in stat.most_common(): print(f"    {k}: {v}")
if miss: print(f"[warn] 没有 mesh 因而摆不了的: {dict(miss)}")
