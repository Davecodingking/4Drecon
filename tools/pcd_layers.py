#!/usr/bin/env python
"""点云分层俯视图 —— 判断"桌椅到底有没有重建出来"的体检图。

整体俯视图会被地板和墙淹掉, 看不出家具。按高度切三层就一目了然:
  0.35~0.85m  桌面层    —— 餐桌/茶几/吧台面应该在这层显形成一块块
  0.85~1.60m  椅背层    —— 椅背/吧凳/盆栽应该是一个个团
  >1.60m      墙柱窗层  —— 墙/柱/垂挂物/窗外

★ 怎么读: 好的点云在前两层是**一块块紧凑的团**; 如果看到的是从相机位置向外辐射的
  **扇形拖尾**, 说明各帧深度没有配准到一起 (长推轨 + 雾 + 暗最容易这样), 家具已经被
  抹成糊, 再往下跑 SpatialLM 也只能认出恰好还成团的东西 (盆栽/垂挂物)。

用法: python pcd_layers.py --ply 03_点云_z朝上.ply [--zup 03_点云_z朝上.json] --out 图.jpg
"""
import argparse, json
import numpy as np, trimesh, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True, help="z 朝上、地板 z=0 的点云")
ap.add_argument("--zup", default="", help="变换记录 json, 给了就把相机轨迹画上去")
ap.add_argument("--out", required=True)
ap.add_argument("--max-pts", type=int, default=300000, help="每层最多画多少点")
a = ap.parse_args()

p = trimesh.load(a.ply)
P = np.asarray(p.vertices)
C = np.asarray(p.colors)[:, :3] / 255.0 if p.colors is not None and len(p.colors) else None
cam = np.array(json.load(open(a.zup))["camera_centers_new"]) if a.zup else None

LAYERS = [(0.35, 0.85, "0.35-0.85m  table tops"),
          (0.85, 1.60, "0.85-1.60m  chair backs / stools / plants"),
          (1.60, 99.0, ">1.60m  walls / columns / hangings")]

fig, ax = plt.subplots(1, 3, figsize=(21, 7))
for k, (lo, hi, t) in enumerate(LAYERS):
    m = (P[:, 2] > lo) & (P[:, 2] < hi)
    idx = np.where(m)[0]
    if len(idx) > a.max_pts: idx = np.random.choice(idx, a.max_pts, replace=False)
    ax[k].scatter(P[idx, 0], P[idx, 1], s=0.15, c=(C[idx] if C is not None else "k"))
    if cam is not None: ax[k].plot(cam[:, 0], cam[:, 1], "r-", lw=1.5, label="camera path")
    ax[k].set_title(f"{t}   ({m.sum()/len(P)*100:.0f}% of points)")
    ax[k].set_aspect("equal"); ax[k].grid(alpha=.3)
ax[0].legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.savefig(a.out, dpi=70)
print(f"[done] -> {a.out}")
