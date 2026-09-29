#!/usr/bin/env python
"""物理修正: 落地 + 推开穿模。读 07_scene.glb, 出 07_scene_phy.glb。

落地: 每个物体的最低点吸到地面上。本来就该悬空的(雨篷/红绿灯/树冠)按"离地超过阈值就不管"放过。
穿模: 用世界系 AABB 找重叠, 沿最小平移方向水平推开; 体积大的当静止, 小的让路; 迭代几轮。
不用物理引擎, 纯几何 —— 室外地面是一个平面, 比室内简单。

节点分三类, 靠名字认 (assemble_scene 建的时候就是这么命名的):
  ground / floor   地面, 只当高度参照, 不参与推开
  wall_*           ★ 静态碰撞体: 不落地、不被推、原样带到输出, 只把物体往屋里推
  其余             可动物体
墙为什么不能走 AABB —— 斜墙的 AABB 是覆盖整条对角线的大方块, 会把几米外的物体乱推;
轴对齐的零厚度墙 AABB 在法向上 extent 恰好为 0, `if (ov<=0).any(): continue` 永远成立,
一次都推不动。所以墙走 2D 线段-有向距离, 判据和 layout_clash_fix_5090.py 一致: 物体中心投到墙的 u/n
局部系, t 落在 (0,L) 内且 |有向距离| 小于 半宽+半墙厚 才算碰, 沿 n 推出去。
线段是从 wall_* 节点的几何直接反推的 (水平投影做 SVD 取主方向), 不再读 json ——
省掉复算 zup->DA3->FLIP 那条变换链, 少一处能对不上的地方。
推的方向取"物体现在在墙的哪一侧就往哪一侧推", 即隔断墙的双向推开。房间壳是点云足迹
围出来的, 物体本来就都在里面, 所以这等价于"推回屋内"; 真有物体在屋外, 这条会把它推得
更远 —— 那种情况是盒子量错了, 该在量盒子那步修, 不该靠物理硬拽。
场景里没有 wall_* 节点就整段跳过 (室外没有墙, 行为逐位不变)。
"""
import os, sys, re, numpy as np, trimesh
R1   = os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
SNAP = float(os.environ.get("SNAP","0.8"))    # 底面离支撑面在这个范围内才吸附
# 支撑面: 东西可以落在这些类的顶面上, 不是只能落地 (桌上的杯子不该被传送到地板)
SUPP = {x.strip() for x in os.environ.get("SUPPORT_LABELS","").split(",") if x.strip()}
# 挂墙/吊顶物: 本来就不落地
HUNG = {x.strip() for x in os.environ.get("HUNG_LABELS","").split(",") if x.strip()}
OVERLAP = float(os.environ.get("SUPPORT_OVERLAP","0.35"))   # 水平重叠占自身多少才算被托住
# ★ 给了 HUNG_LABELS 就关掉"离得远=本来悬空"这条兜底: 那条规则分不开"量歪了"和"真的挂着",
#   实测 37 个物体里 32 个被它当成吊灯放过, 全悬在半空(最高 98cm)。既然已经能按词点名
#   哪些是挂的, 就不该再让距离替它猜。没给 HUNG_LABELS 的(室外)保持原行为, 逐位不变。
FREE_FALL = bool(HUNG)
ITERS= int(os.environ.get("ITERS","12"))
MARGIN=float(os.environ.get("MARGIN","0.02")) # 推开后留的缝

SRC=os.environ.get("SCENE_IN","07_scene.glb")
sc=trimesh.load(f"{R1}/{SRC}")
# 收集每个节点的世界系网格 (07_scene.glb 已按 glTF 约定 y 朝上)
items=[]; ground=None; walls=[]
for node in sc.graph.nodes_geometry:
    T,gname=sc.graph[node]
    g=sc.geometry[gname].copy(); g.apply_transform(T)
    # ★ 必须整名匹配, 不能用 startswith —— "floor_lamp_10012".startswith("floor") 是真,
    #   落地灯会被当成地面吞掉 (实测: 组装 37 个物体, 物理之后只剩 36, 少的就是它)。
    #   节点名的规矩见 assemble_scene: 物体是 "<词>_<实例号>", 房间壳是 "floor"/"wall_<N>"。
    base=re.sub(r"_\d+$", "", node.split("/")[-1])
    if base in ("ground", "floor"): ground=g if ground is None else ground
    elif base == "wall": walls.append([node, g])
    else: items.append([node, g])
assert ground is not None, "场景里没有 ground / floor 节点"
gy=float(np.median(ground.vertices[:,1]))     # 地面高度 (平面近似水平)
print(f"[地面] y = {gy:.2f} m,  {len(items)} 个物体, {len(walls)} 面墙(静态)")

# ---- 墙线: 2D 线段 + 半厚, 在 xz 平面上 (glTF y 朝上) ----
# 墙线 json 存的是 z 朝上系的 (x,y); 导出到 glTF 之后水平面变成 (x,z)。
# 所以这里不直接吃 json 的坐标, 改从 wall_* 节点的实际几何反推 —— 那是场景里真正的位置,
# 不用再复算一遍 zup->DA3->FLIP 的变换链, 少一处能对不上的地方。
segs=[]
for name,g in walls:
    P=g.vertices[:,[0,2]]                      # 水平投影
    c=P.mean(0); Q=P-c
    d=np.linalg.svd(Q, full_matrices=False)[2][0]; d=d/np.linalg.norm(d)  # 主方向=墙的走向
    t=Q@d; half_w=float(np.abs(Q@np.array([-d[1],d[0]])).max())   # 垂直方向的最大偏移=半厚
    segs.append(dict(a=c+t.min()*d, b=c+t.max()*d, d=d, L=float(t.max()-t.min()),
                     n=np.array([-d[1],d[0]]), hw=half_w, name=name))
if segs:
    print(f"[墙] 从几何反推出 {len(segs)} 条墙线, 长 "
          f"{min(s_['L'] for s_ in segs):.2f}~{max(s_['L'] for s_ in segs):.2f}m, "
          f"半厚 {np.median([s_['hw'] for s_ in segs]):.3f}m")

def push_out_of_walls(g):
    """物体和墙相交就沿墙法向推到屋内一侧。返回是否推过。"""
    b=g.bounds; ctr=np.array([(b[0][0]+b[1][0])/2, (b[0][2]+b[1][2])/2])
    rad=float(max(b[1][0]-b[0][0], b[1][2]-b[0][2]))/2
    mv=np.zeros(2)
    for s_ in segs:
        q=ctr-s_["a"]; t=float(q@s_["d"])
        if t<-rad or t>s_["L"]+rad: continue    # 投影落在墙段外, 不算碰
        sd=float(q@s_["n"])                     # 有向距离
        need=rad+s_["hw"]+MARGIN-abs(sd)
        if need<=0: continue
        mv += s_["n"]*np.sign(sd if abs(sd)>1e-6 else 1.0)*need
    if np.linalg.norm(mv)<1e-6: return False
    g.apply_translation([mv[0],0,mv[1]]); return True

# ---- 落地: 从低到高, 先找支撑台面, 找不到才落地面 ----
# 顺序必须从低到高 —— 桌子先落到地上, 桌上的东西才能落到它已经摆正的桌面上。
cls_of=lambda n: re.sub(r"_\d+$","",n.split("/")[-1])
snapped=onsupp=0; skipped=[]
for name,g in sorted(items, key=lambda d: float(d[1].vertices[:,1].min())):
    if cls_of(name) in HUNG: skipped.append((cls_of(name),"挂的")); continue
    V=g.vertices; low=float(V[:,1].min())
    lo,hi=V[:,[0,2]].min(0), V[:,[0,2]].max(0)
    target=gy; tname="地面"
    # ★ 台面自己不落在别的台面上 —— 否则桌子会摞到桌子上 (实测 table_10 落到了
    #   dining_table_5 的桌面上, 悬空 48cm)。只有"被摆在台面上的小东西"才找台面。
    for on,og in ([] if cls_of(name) in SUPP else items):
        if og is g or cls_of(on) not in SUPP: continue
        Vo=og.vertices; lo2,hi2=Vo[:,[0,2]].min(0), Vo[:,[0,2]].max(0)
        ov=np.minimum(hi,hi2)-np.maximum(lo,lo2)
        if (ov<=0).any(): continue
        if (ov[0]*ov[1])/max((hi[0]-lo[0])*(hi[1]-lo[1]),1e-9) < OVERLAP: continue
        z_top=float(Vo[:,1].max())
        if z_top <= low+0.02 and z_top > target: target,tname=z_top,cls_of(on)
    d=low-target
    if not FREE_FALL and abs(d)>SNAP:
        skipped.append((cls_of(name), round(d,2))); continue
    g.apply_translation([0,-d,0]); snapped+=1; onsupp += tname!="地面"
print(f"[落地] {snapped} 个吸附 (其中 {onsupp} 个落在别的物体台面上), "
      f"{len(skipped)} 个没落" + (f" (挂的: {sorted({x[0] for x in skipped})})" if FREE_FALL else
                                  f" (离支撑面超过 {SNAP}m)"))

# ---- 推开穿模 ----
def aabb(g): b=g.bounds; return b[0],b[1]
vol=[np.prod(g.extents) for _,g in items]
order=np.argsort(vol)[::-1]                   # 体积大的先固定
moved=0
for it_ in range(ITERS):
    worst=0.0; hit=0
    for a in range(len(items)):
        for b in range(a+1,len(items)):
            lo1,hi1=aabb(items[a][1]); lo2,hi2=aabb(items[b][1])
            ov=np.minimum(hi1,hi2)-np.maximum(lo1,lo2)
            if (ov<=0).any(): continue
            hit+=1
            # 只水平推: 在 x / z 里挑代价最小的方向
            ax=0 if ov[0]<ov[2] else 2
            push=ov[ax]+MARGIN; worst=max(worst,ov[ax])
            sign=1.0 if (items[a][1].centroid[ax] > items[b][1].centroid[ax]) else -1.0
            va,vb=np.prod(items[a][1].extents), np.prod(items[b][1].extents)
            wa=vb/(va+vb); wb=va/(va+vb)      # 小的多让
            t=np.zeros(3); t[ax]=push*sign*wa; items[a][1].apply_translation(t); 
            t=np.zeros(3); t[ax]=-push*sign*wb; items[b][1].apply_translation(t)
            moved+=1
    if worst<0.01: break
print(f"[推开] {it_+1} 轮, 最后一轮最深穿模 {worst:.3f} m, 共 {hit} 对仍相交")

# ---- 推回屋内 ----
# 放在物体之间推开之后: 先让物体互相让开, 再统一把越界的推回来。反过来做的话,
# 推回屋内的位移会被下一轮物体互推再顶出去。
if segs:
    nw=sum(push_out_of_walls(g) for _,g in items)
    print(f"[墙] {nw}/{len(items)} 个物体穿墙, 已推回屋内")

# 推开后可能被抬离地面, 再吸一次 (只吸回地面; 台面关系推开之后可能已经不成立了)
# ★ FREE_FALL 关(室外)时这里必须和老代码逐字一致: abs(d)<=SNAP 就吸, 埋地的(d<0)也要抬回来。
for name,g in items:
    if cls_of(name) in HUNG: continue
    d=float(g.vertices[:,1].min())-gy
    if abs(d)<=SNAP: g.apply_translation([0,-d,0])

out=trimesh.Scene()
for name,g in items: out.add_geometry(g,node_name=name)
out.add_geometry(ground,node_name="ground" if not walls else "floor")
for name,g in walls: out.add_geometry(g,node_name=name)   # ★ 原样带过去, 一个顶点都没动
dst=f"{R1}/"+os.environ.get("PHY_OUT", SRC.replace(".glb","_phy.glb")); out.export(dst)
print(f"[done] -> {dst}")
