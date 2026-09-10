#!/usr/bin/env python
"""胶水: outdoor 链的 06_all_boxes.json -> Hybrid 的 layout txt。

为什么要转: 两条链的盒子格式和坐标系都不一样。
  outdoor  06_all_boxes.json  DA3 世界系(=首帧相机, OpenCV y 朝下), {中心, 长轴, 尺寸, 底高}
  Hybrid   layout txt         z 朝上、地板 z=0,  bbox_N=Bbox(cls, cx,cy,cz, yaw, sx,sy,sz)
桥梁就是 da3_to_spatiallm.py 存的 03_点云_z朝上.json: X_new = s*R @ X_old + t。

为什么要转过去: hybrid_place_5090.py 比 outdoor 的 assemble_scene.py+scene_phys.py 强得多 ——
  朝向 k*90° 搜索(不是长轴直接映射)、逐轴缩放(不是单一标量)、
  --settle 搜支撑面(不是吸到一个标量地面, 那会把桌上的东西传送到地板)。
所以量盒子用 outdoor 那条, 摆放用 Hybrid 那条。

用法: python boxes_to_layout.py --boxes 06_all_boxes.json --zup 03_点云_z朝上.json \
        --out 06_布局.txt [--exclude screen] [--map-out 06_盒子映射.json]
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--boxes", required=True)
ap.add_argument("--zup", required=True, help="da3_to_spatiallm.py 存的变换记录")
ap.add_argument("--out", required=True)
ap.add_argument("--exclude", default="", help="逗号分隔, 这些词不进布局")
ap.add_argument("--map-out", default="", help="额外存一份 bbox序号 -> (词, obj_id) 的对照")
ap.add_argument("--outputs", default=os.path.expanduser("~/sam3d/sam-3d-objects/outputs"))
ap.add_argument("--tag", default="", help="给了就做类内尺度统一 (需要读该类的 mesh)")
ap.add_argument("--picks", default="", help="04_picks/picks.json, 类内尺度的基准实例")
ap.add_argument("--per-instance", default="potted_palm,potted plant,tree,plants",
                help="这些类不做类内统一 (棕榈/树高矮是真实差异, 压平会把大的压矮)")
ap.add_argument("--no-uniform", action="store_true", help="关掉类内尺度统一")
a = ap.parse_args()

EX = set(x.strip() for x in a.exclude.split(",") if x.strip())
zt = json.load(open(a.zup))
s = float(zt.get("s", 1.0)); R = np.array(zt["R"]); t = np.array(zt["t"])

def to_zup(p):  return s * (R @ np.asarray(p, float)) + t
def dir_zup(v): return R @ np.asarray(v, float)          # 方向只转不平移, 尺度不影响朝向

boxes = json.load(open(a.boxes))

# ---- 类内尺寸统一 (assemble_scene.py 的 CLASS_SCALE 同一个意思, 但只用实测盒子) ----
# 为什么要有: 盒子的长边不可信 —— PCA 长轴会把一串挨着的凳子当成一个, 实测
# bar_stool 长边 0.32~3.85m 差 12 倍, chair 差 8 倍, lamp 差 11 倍; 高度反而稳
# (bar_stool 0.40~0.64), assemble_scene.py 注释: "高度几乎总能看全, 长度会被遮挡截短"。
#
# ★ 用类内**中位数**, 不用 mesh 的长宽比。试过后者, 被 table 的 mesh 坑了 ——
#   那个 mesh 是从"只抠到桌面"的图生成的, 长宽比 5:1, 按高度定尺度后桌子变成 2.96m。
#   outdoor 那边只算一个标量作用在 mesh 上、不覆盖盒子尺寸, 所以没这个问题。
#   中位数只吃实测数据, 天然抗那 12 倍的离群。
PER_INST = set(x.strip() for x in a.per_instance.split(",") if x.strip())
CLASS_SIZE = {}
if not a.no_uniform:
    for word in boxes:
        if word in EX or word in PER_INST or word.replace(" ", "_") in PER_INST: continue
        S = np.array([b["尺寸"] for b in boxes[word]])
        if len(S) < 2: continue
        med = np.median(S, axis=0)
        CLASS_SIZE[word] = med
        print(f"[类尺寸] {word:14s} n={len(S)}  长边 {S[:,:2].max(1).min():.2f}~"
              f"{S[:,:2].max(1).max():.2f} -> 统一 {med[0]:.2f}x{med[1]:.2f}x{med[2]:.2f}")

lines, mapping, n = [], {}, 0
for word in sorted(boxes):
    if word in EX: continue
    cls = word.replace(" ", "_")                          # Hybrid 的解析按逗号切, 词里不能有逗号
    for b in boxes[word]:
        c_new = to_zup(b["中心"])
        # outdoor 的 尺寸 = [沿长轴, 沿次轴, 高]; 高那一维在 z 朝上系里还是高
        L, W, H = b["尺寸"]
        if word in CLASS_SIZE: L, W, H = [float(x) for x in CLASS_SIZE[word]]
        ax = dir_zup(b["长轴"]); ax[2] = 0.0              # 长轴压平到水平面
        nrm = np.linalg.norm(ax)
        yaw = float(np.arctan2(ax[1], ax[0])) if nrm > 1e-6 else 0.0
        lines.append(f"bbox_{n}=Bbox({cls},{c_new[0]:.6f},{c_new[1]:.6f},{c_new[2]:.6f},"
                     f"{yaw:.6f},{L*s:.6f},{W*s:.6f},{H*s:.6f})")
        mapping[str(n)] = dict(word=word, obj_id=b.get("obj_id"), 点数=b.get("点数"),
                               同组候选=b.get("同组候选", []))
        n += 1

open(a.out, "w").write("\n".join(lines) + "\n")
if a.map_out: json.dump(mapping, open(a.map_out, "w"), ensure_ascii=False, indent=1)
print(f"[done] {n} 个盒子 -> {a.out}" + (f"  (排除 {sorted(EX)})" if EX else ""))
