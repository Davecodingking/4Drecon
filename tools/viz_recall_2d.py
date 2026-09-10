#!/usr/bin/env python
"""把"召回的到底是不是新东西"直接摆在照片上看。
对每个召回实例, 打开它自己的最佳帧, 画:
  黄框 = WildDet3D 当初在这张照片上画的原始 2D 框(没经过三维)
  彩色 = SAM3 在同一帧上已有的 mask
右上角标最高的 IoU —— 高就说明 SAM3 其实已经有了, 这是重复; 低才是真漏检。
现行代码是拿三维距离判的, 这张图是拿二维判的, 用来看两者差多少。"""
import os, json, numpy as np, cv2
from pycocotools import mask as mask_util
R1=os.environ["R1"]; FR=os.environ["FR"]; OUT=os.environ["OUT"]
N=int(os.environ.get("N","12")); COLS=int(os.environ.get("COLS","4"))
# ★ 只画**真能进入下一步**的。
#   13_recall.json 是聚类刚出来那一份(8 个), 后面 mask_recall.py 还有两道筛:
#     - 补抠出来太小(< REC_MIN_PCT)的扔掉
#     - 词在 EXCLUDE 里的、同词非代表的不生成
#   画前者的话, 人看着 8 个做决策, 系统实际只认 4 个 —— 人挑中的那个可能根本进不去,
#   而且不报错。这张图是人唯一的决策依据, 它必须和下游认的东西一致。
_rec = json.load(open(f"{R1}/13_recall.json"))["召回"]
_ready = f"{R1}/14_recall_ready.json"
_jobs_p = f"{R1}/logs/recall_jobs.txt"
if os.path.exists(_ready):
    _live = {a["生成名"] for a in json.load(open(_ready))}
    if os.path.exists(_jobs_p):
        _live &= set(open(_jobs_p).read().split())
    _tag = os.environ.get("TAG", "")
    def _name(a): return f"{_tag}_{a['词'].replace(' ','_')}_{a['obj_id']}"
    _drop = [a for a in _rec if _name(a) not in _live]
    rec = [a for a in _rec if _name(a) in _live]
    if _drop:
        print(f"[不画] {len(_drop)} 个进不了下一步(补抠太小 / 在 EXCLUDE 里 / 同词非代表): "
              + ", ".join(f"{a['词']}#{a['obj_id']}" for a in _drop))
else:
    rec = _rec

# 每帧的 SAM3 mask 索引
byframe={}
for f in sorted(os.listdir(f"{R1}/03_masks")):
    if not f.endswith(".npz"): continue
    z=np.load(f"{R1}/03_masks/{f}",allow_pickle=True)
    meta,counts,(H,W)=z["meta"],z["counts"],z["hw"]; w=f[:-4]
    for i,r in enumerate(meta):
        if r["area"]>0: byframe.setdefault(int(r["frame"]),[]).append((w,int(r["oid"]),counts[i],int(H),int(W)))

def iou(a,b):
    u=(a|b).sum(); return float((a&b).sum()/u) if u else 0.0

rows=[]; scores=[]
for a in rec:
    fi=int(a["最佳帧"]); x0,y0,x1,y1=[float(v) for v in a["框2d"]]
    img=cv2.imread(f"{FR}/{fi:04d}.jpg")
    if img is None: continue
    H,W=img.shape[:2]
    wd=np.zeros((H,W),bool); wd[int(max(y0,0)):int(min(y1,H)), int(max(x0,0)):int(min(x1,W))]=True
    best=(0.0,None)
    vis=img.copy()
    for w,oid,cnt,mh,mw in byframe.get(fi,[]):
        m=mask_util.decode({"counts":cnt,"size":[mh,mw]}).astype(bool)
        if (mh,mw)!=(H,W): m=cv2.resize(m.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
        v,u=np.nonzero(m)
        if len(u)==0: continue
        # 只画和 WD 框有交集的, 免得糊满
        if not wd[v,u].any(): continue
        s=iou(m,wd)
        if s>best[0]: best=(s,f"{w}#{oid}")
        vis[m]=(0.45*vis[m]+0.55*np.array([80,220,80])).astype(np.uint8)
    cv2.rectangle(vis,(int(x0),int(y0)),(int(x1),int(y1)),(0,230,255),3)
    pad=int(0.35*max(x1-x0,y1-y0))+20
    cx0,cy0=max(int(x0)-pad,0),max(int(y0)-pad,0); cx1,cy1=min(int(x1)+pad,W),min(int(y1)+pad,H)
    crop=vis[cy0:cy1, cx0:cx1]
    if crop.size==0: continue
    scores.append(best[0])
    rows.append((best[0], crop, f"{a['词']} f{fi}  IoU {best[0]:.2f}" + (f" vs {best[1]}" if best[1] else "  (nothing from SAM3 here)")))

rows.sort(key=lambda r:-r[0])                       # IoU 高的排前面 = 最像重复的
sel=rows[:N]
S=420; tiles=[]
for _,crop,cap in sel:
    h,w=crop.shape[:2]; sc=S/max(w,1); t=cv2.resize(crop,(S,max(int(h*sc),1)))
    t=cv2.copyMakeBorder(t,30,6,6,6,cv2.BORDER_CONSTANT,value=(0,0,0))
    cv2.putText(t,cap,(8,21),0,0.55,(255,255,255),1,cv2.LINE_AA); tiles.append(t)
hmax=max(t.shape[0] for t in tiles)
tiles=[cv2.copyMakeBorder(t,0,hmax-t.shape[0],0,0,cv2.BORDER_CONSTANT,value=(0,0,0)) for t in tiles]
grid=np.vstack([np.hstack(tiles[i:i+COLS]) for i in range(0,len(tiles)-len(tiles)%COLS,COLS)])
cv2.imwrite(OUT,grid,[cv2.IMWRITE_JPEG_QUALITY,92])
s=np.array(scores)
print(f"{len(s)} 个召回实例, 用二维 IoU 重判:")
for t in (0.3,0.5,0.7):
    print(f"  IoU > {t}: {int((s>t).sum()):3d} 个 —— SAM3 其实已经有了, 属于重复")
print(f"  IoU = 0 (SAM3 那块完全空的): {int((s==0).sum())} 个 —— 真漏检")
print("saved",OUT)
