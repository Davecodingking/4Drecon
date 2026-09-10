#!/usr/bin/env python
"""把物体场景和楼合成一个 glb, 顺带报一下互相穿模的情况(AABB 重叠)。"""
import os, sys, numpy as np, trimesh
R1=os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
SRC=os.environ.get("SRC","07_scene_phy.glb,10_buildings.glb").split(",")
DST=os.environ.get("DST","11_scene_full.glb")
out=trimesh.Scene(); box={}
for f in SRC:
    s=trimesh.load(f"{R1}/{f}")
    tag=f.split("_")[0]
    for n in s.graph.nodes_geometry:
        T,g=s.graph[n]; m=s.geometry[g].copy(); m.apply_transform(T)
        nm=f"{tag}:{n}"
        out.add_geometry(m,node_name=nm); box[nm]=m.bounds
out.export(f"{R1}/{DST}")
print(f"[done] {len(box)} 个体 -> {R1}/{DST}")
# 楼 vs 物体 的 AABB 重叠体积
bl=[k for k in box if k.startswith("10")]; ob=[k for k in box if not k.startswith("10")]
def ov(a,b):
    lo=np.maximum(a[0],b[0]); hi=np.minimum(a[1],b[1]); d=hi-lo
    return float(np.prod(d)) if (d>0).all() else 0.0
rep={}
for b in bl:
    hit=[(ov(box[b],box[o]),o) for o in ob]; hit=[h for h in hit if h[0]>0.5]
    if hit: rep[b]=sorted(hit,reverse=True)
print(f"\n[穿模] {len(rep)}/{len(bl)} 栋楼的包围盒吃到了物体")
for b,h in sorted(rep.items(), key=lambda kv:-sum(x[0] for x in kv[1]))[:12]:
    print(f"  {b:28s} 压住 {len(h):3d} 个, 最大重叠 {h[0][0]:8.1f} m3  ({h[0][1]})")
