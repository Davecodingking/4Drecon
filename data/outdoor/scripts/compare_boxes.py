#!/usr/bin/env python
"""现行 PCA 出框 vs WildDet3D 出框 的对照表。

要回答的就一件事: WildDet3D 值不值得换上去。分四项看 ——
  1. 检出数   同一批实例, 两边各出了多少框, 谁漏了谁
  2. 尺寸     逐实例比三边, 重点是 WildDet3D 是不是真的更大 (PCA 只量得到朝相机那面)
  3. 朝向     两边长轴的夹角分布, 以及 WildDet3D 自己跨帧的正反分歧率
  4. 遮挡分档 按实例在最佳帧里露出多少分三档, 分别看尺寸 —— 如果只有露得多的那档准,
              说明"补全被遮挡部分"这个能力在我们的素材上不成立, 那它就没用
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_frame import load_up, export_T, inverse_T
R1 = os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
A  = json.load(open(f"{R1}/06_all_boxes.json"))              # PCA
B  = json.load(open(f"{R1}/06_all_boxes_wilddet.json"))      # WildDet3D
# ★ 天方向以前写死 [0,-1,0], 而依赖表里却声明这一步依赖 up —— 依赖表在说假话。
#   后果: 俯拍场景真实的天偏 26~29°, 这里算出来的朝向/高度系统性歪掉。
UP, _GH, _FB = load_up(f"{R1}/02_da3", "盒子对照")
# 真值只有这几类查得到公开尺寸(米, 长x宽x高), 只用来看"谁更接近", 不是评测基准
REAL = {"taxi":(4.9,1.8,1.5), "car":(4.6,1.8,1.5), "van":(5.4,2.0,2.2),
        "truck":(7.0,2.4,3.0), "street lamp":(None,None,6.0), "tree":(None,None,6.0),
        "trash can":(None,None,0.9), "barrier":(2.0,None,1.1), "bench":(1.8,None,0.9)}

pa = {(w,b["obj_id"]): b for w,bs in A.items() for b in bs}
pb = {(w,b["obj_id"]): b for w,bs in B.items() for b in bs}
both = sorted(set(pa) & set(pb)); only_a = set(pa)-set(pb); only_b = set(pb)-set(pa)

print("=" * 74)
print("1. 检出数")
print(f"  PCA        {len(pa):4d} 个实例")
print(f"  WildDet3D  {len(pb):4d} 个实例")
print(f"  两边都有   {len(both):4d}   只有PCA {len(only_a):3d}   只有WildDet3D {len(only_b):3d}")
if only_a:
    from collections import Counter
    print(f"  WildDet3D 漏掉的类: {dict(Counter(w for w,_ in only_a).most_common(6))}")

print("\n" + "=" * 74)
print("2. 尺寸 (只列两边都有的类; 长=尺寸[0] 高=尺寸[2], 单位米)")
print(f"  {'类':14s} {'n':>3s}  {'PCA长':>6s} {'WD3长':>6s} {'真值':>6s}   {'PCA高':>6s} {'WD3高':>6s} {'真值':>6s}")
rows = {}
for k in both: rows.setdefault(k[0], []).append(k)
for w in sorted(rows, key=lambda w: -len(rows[w])):
    ks = rows[w]
    la = np.median([pa[k]["尺寸"][0] for k in ks]); lb = np.median([pb[k]["尺寸"][0] for k in ks])
    ha = np.median([pa[k]["尺寸"][2] for k in ks]); hb = np.median([pb[k]["尺寸"][2] for k in ks])
    rl, _, rh = REAL.get(w, (None,None,None))
    f = lambda v: f"{v:6.2f}" if v is not None else "     -"
    print(f"  {w:14s} {len(ks):3d}  {la:6.2f} {lb:6.2f} {f(rl)}   {ha:6.2f} {hb:6.2f} {f(rh)}")

print("\n" + "=" * 74)
print("3. 朝向")
ang = []
for k in both:
    u = np.array(pa[k]["长轴"]); v = np.array(pb[k]["长轴"])
    u = u-UP*(u@UP); v = v-UP*(v@UP)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6: continue
    ang.append(np.degrees(np.arccos(np.clip(u@v/(nu*nv), -1, 1))))
ang = np.array(ang)
print(f"  两边长轴夹角 (n={len(ang)}): 中位 {np.median(ang):.1f}°")
for lo,hi in [(0,15),(15,60),(60,120),(120,165),(165,181)]:
    n = int(((ang>=lo)&(ang<hi)).sum())
    tag = "  <- 轴一致, 正反相同" if hi==15 else ("  <- 轴一致, 正反相反(PCA 定不了正反, 这里是它随机)" if lo==165 else
          ("  <- 轴都对不上" if lo==60 else ""))
    print(f"    {lo:3d}-{hi:3d}°  {n:4d} 个 ({100*n/max(len(ang),1):4.1f}%){tag}")
same_axis = int(((ang<15)|(ang>165)).sum())
print(f"  轴对得上的(夹角<15° 或 >165°): {same_axis}/{len(ang)} = {100*same_axis/max(len(ang),1):.1f}%")
fl = [pb[k].get("朝向分歧",0) for k in both]
print(f"  WildDet3D 自己跨帧的正反分歧率: 中位 {np.median(fl):.2f}, "
      f"完全一致的 {sum(1 for x in fl if x==0)}/{len(fl)}")

print("\n" + "=" * 74)
print("4. 按遮挡分档 (露出比例 = 最佳帧 mask 面积 / 2D框面积, 越小说明被挡越多)")
import glob
from pycocotools import mask as mask_util
vis = {}
for w in {k[0] for k in both}:
    f = f"{R1}/03_masks/{w.replace(' ','_')}.npz"
    if not os.path.exists(f): continue
    z = np.load(f, allow_pickle=True); meta, counts = z["meta"], z["counts"]
    for k in [k for k in both if k[0]==w]:
        rws = [(int(meta['area'][i]), i, float(meta['edge'][i])) for i in range(len(meta))
               if int(meta['oid'][i])==k[1] and meta['area'][i]>0]
        if not rws: continue
        a,i,e = max(rws, key=lambda t: t[0]*(1-min(t[2]*4,.9)))
        m = mask_util.decode({"counts":counts[i],"size":[int(z["hw"][0]),int(z["hw"][1])]}).astype(bool)
        ys,xs = np.nonzero(m)
        bb = max((xs.max()-xs.min()+1)*(ys.max()-ys.min()+1), 1)
        vis[k] = a/bb
tiers = [(">70%",.7,1.01), ("40~70%",.4,.7), ("<40%",0,.4)]
# ★ 不能用 "WD3 长 / PCA 长" 当指标: 那个比值把两种错混在一起了。折叠椅 PCA 量出 2.60m、
# 行人 1.47m、路灯 2.74m —— 这些 PCA 本身就是虚高的, WD3 变小是变对了, 比值却显示"更差"。
# 只在查得到真值的类上比"离真值差多少", 才是能判读的数。
def err(k, src):
    w = k[0]; rl,_,rh = REAL.get(w,(None,None,None))
    e = []
    if rl: e.append(abs(src[k]["尺寸"][0]-rl)/rl)
    if rh: e.append(abs(src[k]["尺寸"][2]-rh)/rh)
    return np.mean(e) if e else None
HAS = [k for k in both if REAL.get(k[0]) and any(REAL[k[0]][i] for i in (0,2))]
print(f"  有公开真值可比的实例: {len(HAS)}/{len(both)}  "
      f"(类: {sorted({k[0] for k in HAS})})")
print(f"\n  {'档':10s} {'n':>4s}  {'PCA平均误差':>11s} {'WD3平均误差':>11s}   {'轴对得上':>8s}")
for name, lo, hi in tiers:
    ks = [k for k in HAS if k in vis and lo <= vis[k] < hi]
    if not ks: print(f"  {name:10s} {0:4d}   (这一档没有带真值的实例)"); continue
    ea = [err(k,pa) for k in ks]; eb = [err(k,pb) for k in ks]
    aa = []
    for k in ks:
        u=np.array(pa[k]["长轴"]); v=np.array(pb[k]["长轴"])
        u=u-UP*(u@UP); v=v-UP*(v@UP)
        if min(np.linalg.norm(u),np.linalg.norm(v))<1e-6: continue
        d=np.degrees(np.arccos(np.clip(u@v/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1)))
        aa.append(d<15 or d>165)
    print(f"  {name:10s} {len(ks):4d}  {100*np.mean(ea):10.1f}% {100*np.mean(eb):10.1f}%   "
          f"{100*np.mean(aa) if aa else float('nan'):7.1f}%")

print("\n  另外单独看一眼没有真值、但常识上能判断的几类 (PCA 明显虚高的):")
for w in ["folding chair","table","pedestrian","street lamp","traffic light"]:
    ks=[k for k in both if k[0]==w]
    if not ks: continue
    print(f"    {w:15s} PCA长 {np.median([pa[k]['尺寸'][0] for k in ks]):5.2f}m  "
          f"WD3长 {np.median([pb[k]['尺寸'][0] for k in ks]):5.2f}m   "
          f"<- {'椅子/桌子/人/灯杆不可能有 PCA 那么长' if True else ''}")

print("\n判读: 三档的 WD3 误差都明显低于 PCA 才说明'补全被遮挡部分'在我们的素材上成立;")
print("      只有 >70% 那一档好, 就说明这个能力对我们没用。")
