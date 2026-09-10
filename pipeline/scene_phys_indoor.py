#!/usr/bin/env python
"""物理修正(室内版): 拟合地面 + 落地 + 推开穿模。读 00_整屋.glb, 出 00_整屋_phy.glb。

骨架照抄 data/outdoor/scripts/scene_phys.py, 三处按室内改:

  1) 竖直轴是 z 不是 y。outdoor 的 assemble_scene.py 导出时做过 y-up 翻转,
     hybrid_place_5090.py 直接 scene.export(), 保持 da3_to_spatiallm 那个 z 朝上系。
  2) 场景里没有 ground 节点。outdoor 的 assemble_scene.py 会铺一片进去,
     物体链这边没做, 所以 scene_phys.py 的 `assert ground is not None` 直接过不去。
     -> 这个脚本自己拟合并铺一片。
  3) 落地先找支撑面。outdoor 是单一地面标量, 会把桌上的东西传送到地板;
     这里沿用 hybrid_place_5090.py --settle 的做法: 按底面从低到高处理,
     先看有没有别的物体能当台面(XY 投影重叠 >= --overlap), 有就落台面, 没有才落地面。

地面用 RANSAC 拟合所有物体的盒底, 不能用最小二乘 —— 挂高的东西(这里是 lamp, 最高 1.32m;
outdoor 那边是墙上的广告牌)会把平面带歪, assemble_scene.py 注释里记着实测被带偏 1.51m。

用法: python scene_phys_indoor.py --glb 07_整屋/00_整屋.glb [--out .../00_整屋_phy.glb]
"""
import argparse, itertools, os
import numpy as np, trimesh

ap = argparse.ArgumentParser()
ap.add_argument("--glb", required=True)
ap.add_argument("--out", default="", help="默认在原名后加 _phy")
ap.add_argument("--snap", type=float, default=0.8,
                help="底面离支撑面在这个范围内才吸附, 超过视为本就悬空 (m)")
ap.add_argument("--iters", type=int, default=12, help="推开穿模的迭代轮数")
ap.add_argument("--margin", type=float, default=0.02, help="推开后留的缝 (m)")
ap.add_argument("--overlap", type=float, default=0.5, help="XY 投影重叠多少才算压在它身上")
ap.add_argument("--support-labels", default="table,dining_table,side_table,bar_counter,counter,"
                "sofa,couch,booth_seat,shelf,planter",
                help="能给别的物体当台面的类别 (按节点名里的类名匹配)")
ap.add_argument("--hung-labels", default="screen,lamp",
                help="挂墙/吊顶物, 不落地 (灯多半是吊灯/壁灯)")
ap.add_argument("--no-ground", action="store_true", help="不往场景里铺地面片")
a = ap.parse_args()
UP = 2                                        # ★ z 朝上, 不是 outdoor 的 y
OUT = a.out or a.glb.replace(".glb", "_phy.glb")

sc = trimesh.load(a.glb)
items = []
for node in sc.graph.nodes_geometry:
    T, gname = sc.graph[node]
    g = sc.geometry[gname].copy(); g.apply_transform(T)
    if "ground" in node or node.startswith("wall_"): continue   # 墙和已有地面不参与
    items.append([node, g])
print(f"[读入] {len(items)} 个物体")

def cls_of(name):                             # obj12_dining_table -> dining_table
    p = name.split("_", 1)
    return p[1] if len(p) > 1 else name
SUPP = set(x.strip() for x in a.support_labels.split(",") if x.strip())
HUNG = set(x.strip() for x in a.hung_labels.split(",") if x.strip())

# ---- 地面: RANSAC 拟合所有盒底 (最小二乘会被挂高的物体带歪) ----
pts = np.array([[g.centroid[0], g.centroid[1], g.vertices[:, UP].min()] for _, g in items])
A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
rng = np.random.default_rng(0); best = None
for _ in range(600):
    i = rng.choice(len(pts), 3, replace=False)
    try: c = np.linalg.solve(A[i], pts[i, 2])
    except np.linalg.LinAlgError: continue
    n = int((np.abs(A @ c - pts[:, 2]) < 0.5).sum())
    if best is None or n > best[0]: best = (n, c)
inl = np.abs(A @ best[1] - pts[:, 2]) < 0.5
sol, *_ = np.linalg.lstsq(A[inl], pts[inl, 2], rcond=None)
def gz(x, y): return float(sol[0] * x + sol[1] * y + sol[2])
res = np.abs(A[inl] @ sol - pts[inl, 2])
print(f"[地面] {len(pts)} 个盒底中 {inl.sum()} 个在地面上, 残差中位 {np.median(res):.3f} m")
print(f"       平面 z = {sol[0]:+.4f}x {sol[1]:+.4f}y {sol[2]:+.3f}  "
      f"(场景中心处 z = {gz(pts[:,0].mean(), pts[:,1].mean()):+.3f} m)")

# ---- 落地: 从低到高, 先找台面, 找不到才落地面 ----
snapped = skipped = onsupp = 0
for d in sorted(items, key=lambda d: d[1].vertices[:, UP].min()):
    name, g = d
    if cls_of(name) in HUNG:
        print(f"  [落地] {name}: 挂墙/吊顶物, 跳过"); continue
    V = g.vertices; z_bot = float(V[:, UP].min())
    lo, hi = V[:, :2].min(0), V[:, :2].max(0)
    target = gz(*g.centroid[:2]); tname = "地面"
    for o_name, og in items:                                  # 找更高的支撑台面
        if og is g or cls_of(o_name) not in SUPP: continue
        Vo = og.vertices
        lo2, hi2 = Vo[:, :2].min(0), Vo[:, :2].max(0)
        ov = np.minimum(hi, hi2) - np.maximum(lo, lo2)
        if (ov <= 0).any(): continue
        if (ov[0] * ov[1]) / max((hi[0]-lo[0]) * (hi[1]-lo[1]), 1e-6) < a.overlap: continue
        z_top = float(Vo[:, UP].max())
        if z_top <= z_bot + 0.02 and z_top > target: target, tname = z_top, o_name
    dz = target - z_bot
    if abs(dz) > a.snap:
        print(f"  [落地] {name}: 离{tname} {dz*100:+.0f}cm 超过 {a.snap*100:.0f}cm, 视为本就悬空")
        skipped += 1; continue
    if abs(dz) > 1e-4:
        t = np.zeros(3); t[UP] = dz; g.apply_translation(t)
    snapped += 1
    if tname != "地面": onsupp += 1
print(f"[落地] {snapped} 个吸附 (其中 {onsupp} 个落在别的物体台面上), {skipped} 个视为本就悬空")

# ---- 推开穿模: 世界系 AABB, 只水平推, 体积大的当静止 ----
for it_ in range(a.iters):
    worst = 0.0; hit = 0
    for i, j in itertools.combinations(range(len(items)), 2):
        gi, gj = items[i][1], items[j][1]
        lo1, hi1 = gi.bounds; lo2, hi2 = gj.bounds
        ov = np.minimum(hi1, hi2) - np.maximum(lo1, lo2)
        if (ov <= 0).any(): continue
        hit += 1
        ax = 0 if ov[0] < ov[1] else 1                        # ★ 水平面是 x/y (outdoor 是 x/z)
        push = ov[ax] + a.margin; worst = max(worst, ov[ax])
        sign = 1.0 if gi.centroid[ax] > gj.centroid[ax] else -1.0
        vi, vj = np.prod(gi.extents), np.prod(gj.extents)
        wi, wj = vj/(vi+vj), vi/(vi+vj)                       # 小的多让
        t = np.zeros(3); t[ax] = push*sign*wi;  gi.apply_translation(t)
        t = np.zeros(3); t[ax] = -push*sign*wj; gj.apply_translation(t)
    if worst < 0.01: break
print(f"[推开] {it_+1} 轮, 最后一轮最深穿模 {worst:.3f} m, {hit} 对仍相交")

# 推开后可能被抬离支撑面, 再吸一次 (只对地面, 台面关系推开后可能已经变了)
for name, g in items:
    if cls_of(name) in HUNG: continue
    dz = gz(*g.centroid[:2]) - float(g.vertices[:, UP].min())
    if abs(dz) <= a.snap:
        t = np.zeros(3); t[UP] = dz; g.apply_translation(t)

out = trimesh.Scene()
for name, g in items: out.add_geometry(g, node_name=name)
for node in sc.graph.nodes_geometry:                           # 墙原样带回去
    if node.startswith("wall_"):
        T, gname = sc.graph[node]
        w = sc.geometry[gname].copy(); w.apply_transform(T)
        out.add_geometry(w, node_name=node)
if not a.no_ground:
    allv = np.vstack([g.vertices for _, g in items])
    lo, hi = allv[:, :2].min(0) - 1.0, allv[:, :2].max(0) + 1.0
    corners = [(lo[0],lo[1]), (hi[0],lo[1]), (hi[0],hi[1]), (lo[0],hi[1])]
    V = np.array([[x, y, gz(x, y)] for x, y in corners])
    ground = trimesh.Trimesh(V, np.array([[0,1,2],[0,2,3]]), process=False)
    ground.visual.face_colors = [130, 130, 135, 255]
    out.add_geometry(ground, node_name="ground")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.export(OUT)
print(f"[done] -> {OUT} ({os.path.getsize(OUT)/1e6:.0f}MB)")
