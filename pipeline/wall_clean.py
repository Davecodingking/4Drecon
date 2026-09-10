#!/usr/bin/env python
"""清洗 SpatialLM 的墙布局: 过滤碎墙 + 合并共线 + 补厚度。两条室内链共用。

为什么必须有这一步 —— cyberpunk 的 arch 模式出了 117 面墙, 其中 98 面短于 0.3m,
中位长度只有 0.16m。真正围出房间的只有 9 面 (>=1m, 占总长 69%)。碎墙有两个害处:
  1. 地板的足迹是用奇偶射线法从墙线算的, 碎墙会在射线上制造大量伪交点, 足迹变成筛子
  2. 每面碎墙都要建一个 mesh, 场景里几百个 3cm 的小盒子

thickness 是 SpatialLM 硬编码的 0.0 (SpatialLM/spatiallm/layout/layout.py:116
"assume the walls has a thickness of 0.0 for now"), 拿来当厚度会得到零厚度的纸片 ——
纸片的 AABB 在法向上 extent 恰好为 0, 物理推开循环第一句 `if (ov<=0).any(): continue`
永远成立, 一次都推不动。所以厚度必须在这里补上。

只改 Wall 行。Bbox / Door / Window 原样保留 —— LM 链的布局里还有家具盒子和门窗。

用法:
  python wall_clean.py --layout 09_墙布局.txt --out 09_墙布局_清洗.txt \
      [--min-len 1.0] [--ang 5] [--perp 0.15] [--gap 0.5] [--thick 0.10]
"""
import argparse, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--layout", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-len", type=float, default=1.0, help="短于这个的墙丢掉")
ap.add_argument("--ang", type=float, default=5.0, help="合并共线的角度容差(度)")
ap.add_argument("--perp", type=float, default=0.15, help="合并共线的垂距容差(m)")
ap.add_argument("--gap", type=float, default=0.5, help="两段之间允许的间隙(m)")
ap.add_argument("--thick", type=float, default=0.10, help="墙厚。SpatialLM 给的恒为 0")
a = ap.parse_args()

WALL = re.compile(r"(wall_\d+)=Wall\(([^)]*)\)")

walls, others = [], []
for line in open(a.layout, encoding="utf-8"):
    line = line.rstrip("\n")
    m = WALL.match(line.strip())
    if not m:
        if line.strip():
            others.append(line)
        continue
    v = [float(x) for x in m.group(2).split(",")]
    # Wall(ax,ay,az, bx,by,bz, height, thickness) —— 恰好 8 个数
    walls.append(dict(a=np.array(v[0:2]), b=np.array(v[3:5]), z0=v[2], h=v[6]))

n0 = len(walls)
L0 = sum(float(np.linalg.norm(w["b"] - w["a"])) for w in walls)
print(f"[输入] {n0} 面墙, 总长 {L0:.1f}m" + (f", 另有 {len(others)} 行非墙内容" if others else ""))

# ---------- 1. 过滤碎墙 ----------
keep = [w for w in walls if np.linalg.norm(w["b"] - w["a"]) >= a.min_len]
L1 = sum(float(np.linalg.norm(w["b"] - w["a"])) for w in keep)
print(f"[过滤] 丢掉短于 {a.min_len}m 的 {n0 - len(keep)} 面 -> {len(keep)} 面, "
      f"总长 {L1:.1f}m (保留 {L1/max(L0,1e-9)*100:.0f}%)")

# ---------- 2. 合并共线 ----------
# 判据三条同时成立才合并: 方向夹角小(折到 0~90, 因为墙没有正反)、一条的端点到另一条
# 所在直线的垂距小、两段在共同方向上的投影区间重叠或间隙够小。
# 合并后取两端最远的投影点, 所以多段接力也能一次合成一整面。
def merge_pass(ws, ang_th, perp_th, gap_th):
    used = [False] * len(ws)
    out = []
    for i, w in enumerate(ws):
        if used[i]:
            continue
        grp = [w]
        used[i] = True
        changed = True
        while changed:                      # 反复扫, 让 A-B 合并后还能吸收 C
            changed = False
            # 当前组的方向和基点用组内最长的一段定
            lead = max(grp, key=lambda x: np.linalg.norm(x["b"] - x["a"]))
            d = lead["b"] - lead["a"]
            Lg = float(np.linalg.norm(d))
            u = d / Lg
            n = np.array([-u[1], u[0]])
            for j, o in enumerate(ws):
                if used[j]:
                    continue
                d2 = o["b"] - o["a"]
                L2 = float(np.linalg.norm(d2))
                u2 = d2 / L2
                cos = abs(float(u @ u2))
                if np.degrees(np.arccos(np.clip(cos, 0, 1))) > ang_th:
                    continue
                # 垂距: 两个端点都要贴着 lead 所在的直线
                perp = np.abs(np.array([o["a"] - lead["a"], o["b"] - lead["a"]]) @ n)
                if perp.max() > perp_th:
                    continue
                # 投影区间的间隙
                t_g = np.concatenate([[(x["a"] - lead["a"]) @ u, (x["b"] - lead["a"]) @ u]
                                      for x in grp])
                t_o = np.array([(o["a"] - lead["a"]) @ u, (o["b"] - lead["a"]) @ u])
                gap = max(t_o.min() - t_g.max(), t_g.min() - t_o.max(), 0.0)
                if gap > gap_th:
                    continue
                grp.append(o)
                used[j] = True
                changed = True
        # 组内所有端点投到 lead 方向上, 取最远的两个当新墙的两端
        lead = max(grp, key=lambda x: np.linalg.norm(x["b"] - x["a"]))
        d = lead["b"] - lead["a"]
        u = d / float(np.linalg.norm(d))
        P = np.array([p for x in grp for p in (x["a"], x["b"])])
        t = (P - lead["a"]) @ u
        out.append(dict(a=lead["a"] + t.min() * u, b=lead["a"] + t.max() * u,
                        z0=float(np.median([x["z0"] for x in grp])),
                        h=float(np.median([x["h"] for x in grp])), n=len(grp)))
    return out

merged = merge_pass(keep, a.ang, a.perp, a.gap)
L2 = sum(float(np.linalg.norm(w["b"] - w["a"])) for w in merged)
print(f"[合并] {len(keep)} -> {len(merged)} 面, 总长 {L2:.1f}m")
for i, w in enumerate(sorted(merged, key=lambda x: -np.linalg.norm(x["b"] - x["a"]))):
    L = float(np.linalg.norm(w["b"] - w["a"]))
    ang = np.degrees(np.arctan2(*(w["b"] - w["a"])[::-1])) % 180
    print(f"   长 {L:6.2f}m  朝向 {ang:5.1f}°  ({w['a'][0]:6.2f},{w['a'][1]:6.2f}) -> "
          f"({w['b'][0]:6.2f},{w['b'][1]:6.2f})  高 {w['h']:.2f}m  (并了 {w['n']} 段)")

# ---------- 3. 输出 ----------
with open(a.out, "w", encoding="utf-8") as f:
    for i, w in enumerate(merged):
        f.write(f"wall_{i}=Wall({w['a'][0]},{w['a'][1]},{w['z0']},"
                f"{w['b'][0]},{w['b'][1]},{w['z0']},{w['h']},{a.thick})\n")
    for line in others:
        f.write(line + "\n")
print(f"[done] 厚度统一写 {a.thick}m -> {a.out}")

# ---------- 4. 闭合性体检 ----------
# 地板的足迹是奇偶射线法从墙线算的, 要求墙首尾相接围成闭合环。围不上足迹就会漏。
# 这里不修, 只报告 —— 围不上就说明这个场景 (开放式/玻璃幕墙) 本来就没有闭合房间,
# 该退回"墙的 AABB 当地板范围", 而不是硬凑一个环。
if merged:
    ends = np.array([p for w in merged for p in (w["a"], w["b"])])
    D = np.linalg.norm(ends[:, None, :] - ends[None, :, :], axis=2)
    np.fill_diagonal(D, 1e9)
    for k in range(0, len(ends), 2):        # 同一面墙的两端不算互相接上
        D[k, k + 1] = D[k + 1, k] = 1e9
    near = (D.min(1) < 0.5).sum()
    print(f"[闭合性] {len(ends)} 个端点里 {near} 个能在 0.5m 内找到另一面墙的端点"
          f" ({near/len(ends)*100:.0f}%)")
    if near < len(ends):
        print(f"   ⚠ {len(ends)-near} 个端点悬空 —— 围不成闭合环, 地板足迹会漏。"
              f"这种场景该用墙的 AABB 当地板范围")
