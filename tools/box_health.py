#!/usr/bin/env python
"""盒子体检: 每个盒子里到底有没有真东西。

来历是 Hybrid r2 的 obj9 教训 —— SpatialLM 报了一个 3.75m 的 curtain, 盒内 31k 个点
看着很"实", 但那些点沿长轴均匀铺满全长, 全是壁画墙的墙面点漏进了 15cm 的薄盒;
真窗帘只有一窄条。照单生成的结果是把一块窄布拉伸 9 倍盖住整面墙。

两个指标:
  点/立方米   盒子里有多少实测点。太低 = 空盒子, LM 凭空补的
  长轴铺满度  把盒内点投到最长边上分 8 格, 有多少格是"有货"的。
              真物体集中在几格 (0.2~0.9); **接近 1.00 = 铺满全长**, 多半是吸进了
              墙面点或雾, 不是一个物体

判定 (阈值是在 cyberpunk 上定的, 换场景可调):
  ★真  密度 > 3000 且 铺满度 < 0.95
  ?    密度 > 800  (存疑, 人眼过一遍)
  空   其余 (别拿去生成)

用法: python box_health.py --layout 05_布局_修复后.txt --pcd 03_点云_z朝上.ply [--csv 体检表.csv]
"""
import argparse, csv, re
import numpy as np, trimesh

ap = argparse.ArgumentParser()
ap.add_argument("--layout", required=True)
ap.add_argument("--pcd", required=True, help="z朝上点云, 和 layout 同坐标系")
ap.add_argument("--csv", default="")
ap.add_argument("--min-density", type=float, default=3000.0)
ap.add_argument("--max-fill", type=float, default=0.95)
a = ap.parse_args()

P = np.asarray(trimesh.load(a.pcd).vertices)
rows = []
for line in open(a.layout):
    m = re.match(r"bbox_(\d+)=Bbox\(([^,]+),([^)]*)\)", line.strip())
    if not m: continue
    v = [float(x) for x in m.group(3).split(",")]
    c, yaw, s = np.array(v[0:3]), v[3], np.array(v[4:7])
    Rz = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
    L = P - c; L = np.c_[L[:, :2] @ Rz.T, L[:, 2]]
    inb = (np.abs(L) <= s / 2).all(1); n = int(inb.sum())
    dens = n / max(np.prod(s), 1e-6)
    ax = int(np.argmax(s))
    h = np.histogram(L[inb, ax], bins=8, range=(-s[ax] / 2, s[ax] / 2))[0] if n > 50 else np.zeros(8)
    fill = float((h > max(n * 0.02, 5)).sum() / 8) if n > 50 else 0.0
    verdict = "真" if dens > a.min_density and fill < a.max_fill else ("存疑" if dens > 800 else "空")
    rows.append(dict(idx=int(m.group(1)), cls=m.group(2), n=n, density=round(dens),
                     fill=round(fill, 2), size="x".join(f"{x:.2f}" for x in s), verdict=verdict))

rows.sort(key=lambda r: -r["density"])
print(f"{'盒':>4} {'类别':<14} {'盒内点':>8} {'点/m3':>8} {'铺满度':>6}  {'尺寸':<18} 判定")
for r in rows:
    print(f"{r['idx']:>4} {r['cls']:<14} {r['n']:>8} {r['density']:>8} {r['fill']:>6.2f}  {r['size']:<18} {r['verdict']}")
k = {v: sum(1 for r in rows if r["verdict"] == v) for v in ("真", "存疑", "空")}
print(f"\n[done] {len(rows)} 个盒子: 真 {k['真']} / 存疑 {k['存疑']} / 空 {k['空']}")
if a.csv:
    with open(a.csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"       -> {a.csv}")
