import os, json, numpy as np, trimesh
from world_frame import load_up, export_R
R1="/home/jincheng/sam3d/data/outdoor/result/timesquare/r1"
cams=json.load(open(f"{R1}/02_da3/cameras.json"))["cameras"]
C=np.array([np.linalg.inv(np.array(v["extrinsic_w2c_4x4"]).reshape(4,4))[:3,3] for v in cams.values()])
# ★ 这里曾经写死 diag(1,-1,-1): 组装那边 2026-09-08 改成按实际天方向算, 这份抄本没跟着,
#   于是"相机落在楼盒子里没有"这个判断是在两个错开的坐标系里做的(carpark 差 29.2°),
#   楼越高越容易误判。和组装同源就不会再漂。topview.py 已经修过同一份。
_UP,_GH,_FB=load_up(f"{R1}/02_da3","楼体诊断")
F=export_R(_UP)
Cg=C@F.T
s=trimesh.load(f"{R1}/10_buildings.glb")
print(f"相机轨迹 {len(C)} 个, 跨度 {np.linalg.norm(C.max(0)-C.min(0)):.1f} m")
rows=[]
for n in s.graph.nodes_geometry:
    T,g=s.graph[n]; m=s.geometry[g].copy(); m.apply_transform(T)
    lo,hi=m.bounds; ctr=(lo+hi)/2
    inside=((Cg>=lo)&(Cg<=hi)).all(1).sum()
    d=np.linalg.norm(Cg-ctr,axis=1).min()
    rows.append((inside,d,n,(hi-lo)))
for inside,d,n,e in sorted(rows,key=lambda r:-r[0]):
    flag="  ← 相机在楼里面" if inside else ""
    print(f"{n:14s} 最近相机 {d:7.1f} m  盒子 {np.round(e,1)}  相机落在盒内 {inside:3d}/{len(C)} 帧{flag}")
