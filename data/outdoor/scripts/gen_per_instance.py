#!/usr/bin/env python
"""对不该复用的类, 每个实例各自裁一张最好的图去生成 mesh.
够大的单独生成; 太小的(生成必是垃圾)继续复用同类里最清楚的那个。"""
import os, sys, json, numpy as np, cv2
from pycocotools import mask as mask_util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imgio import imwrite_stable
import review_decisions as review
TAG=os.environ.get("TAG","ts_r1")
R1=os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
FR=os.environ.get("FR","/home/jincheng/sam3d/data/outdoor/dataset/timesquare/frames")
SOLO=set(x for x in os.environ.get("SOLO","car").split(",") if x)
KEEP=set(x for x in os.environ.get("SOLO_KEEP","").split(",") if x)   # 白名单: 可合并, 共用一个 mesh
# SOLO 和 EXCLUDE 可能同时含一个词 (cyberpunk 上 sofa 就是), 排除优先 —— 否则逐实例
# 生成一堆 mesh, 组装时全被扔。
EXCLUDE=set(x.strip() for x in os.environ.get("EXCLUDE","").split(",") if x.strip())
TH=float(os.environ.get("TH","0.5"))
BOXES=os.environ.get("BOXES", f"{R1}/06_all_boxes.json")
d=json.load(open(BOXES))
if "*" in SOLO: SOLO=set(d)-KEEP          # "*" = 除白名单外全部逐实例生成
SOLO=SOLO-EXCLUDE                         # ★ 必须在 "*" 展开之后减, 否则 "*" 场景漏掉
print(f"[SOLO] {len(SOLO)} 类逐实例生成, 白名单(共用mesh) {sorted(KEEP)}"
      + (f", EXCLUDE 挡掉 {sorted(EXCLUDE & (set(d)|SOLO))}" if EXCLUDE else ""))
OV_INST=review.overrides_inst(R1)     # {(词, 实例号): 帧号} —— 人在闸口点名换的那张抠图
# ★ VLM 逐实例校验的否决名单(verify_picks.py 那一段写的)。
#   以前逐实例这条路上一个机器校验都没有: 共用的词验代表那一张, solo 的词一张都不验,
#   于是图越多越没人管。pedestrian/r2 实测 person 15 个里 3 个根本不是人。
#   文件不在就当空的 —— 老轮次没有这份, 不该因此跑不了。
_VP = f"{R1}/04_picks/vlm_verify_inst.json"
VLM_REJ = ({w: set(v) for w, v in (json.load(open(_VP)).get("否决") or {}).items()}
           if os.path.exists(_VP) else {})
if VLM_REJ:
    print(f"[VLM否决] {sum(len(v) for v in VLM_REJ.values())} 个实例不生成: "
          + json.dumps({k: sorted(v) for k, v in VLM_REJ.items()}, ensure_ascii=False))
jobs=[]
for w in SOLO:
    z=np.load(f"{R1}/03_masks/{w.replace(' ','_')}.npz",allow_pickle=True)
    meta=z["meta"]; counts=z["counts"]; H,W=z["hw"]
    for b in d.get(w,[]):
        oid=b["obj_id"]
        if oid in VLM_REJ.get(w, ()):      # VLM 说这不是这个词, 别烧这份 GPU
            continue
        rows=[(i,int(a),int(f),float(e)) for i,(o,f,a,e) in enumerate(meta) if int(o)==oid and a>0]
        if not rows: continue
        i,a,fi,e = max(rows, key=lambda t: t[1]*(1-min(t[3]*4,0.9)))   # 面积大且不贴边
        # ★ 人在闸口点名换帧的, 压过上面这个自动挑法。以前这一步完全不看人的决定,
        #   而 SOLO="*" 的场景里绝大多数候选都走这条 —— 等于闸口的"改用"对它们全是空操作。
        _want = OV_INST.get((w, oid))
        if _want is not None:
            _hit=[r for r in rows if r[2]==_want]
            if _hit:
                i,a,fi,e = _hit[0]
                print(f"[人工指定] {w}#{oid} 改用第 {fi} 帧")
            else:
                # 人点的那一帧这个实例根本没被跟踪到。静默退回自动帧 = 人以为换了其实没换,
                # 这正是这一整类问题的形状, 所以说出来。
                print(f"[人工指定] ★ {w}#{oid} 点的第 {_want} 帧没有这个实例的 mask, "
                      f"退回自动挑的第 {fi} 帧 —— 换个帧号重填")
        pct=100*a/(H*W)
        if pct<TH:
            if _want is not None:
                print(f"[人工指定] ★ {w}#{oid} 第 {fi} 帧只占 {pct:.2f}% < TH={TH}, 仍然不生成")
            continue
        m=mask_util.decode({"counts":counts[i],"size":[int(H),int(W)]}).astype(np.uint8)
        name=f"{TAG}_{w.replace(' ','_')}_{oid}"
        dd=os.path.expanduser(f"~/sam3d/sam-3d-objects/my_images/{name}"); os.makedirs(dd,exist_ok=True)
        imwrite_stable(f"{dd}/frame.png", cv2.imread(f"{FR}/{fi:04d}.jpg"))
        imwrite_stable(f"{dd}/mask.png", m*255)
        jobs.append(name); print(f"[{w}#{oid}] 第{fi}帧 占{pct:.2f}% -> {name}")
open(f"{R1}/logs/solo_jobs.txt","w").write("\n".join(jobs))
print(f"\n[done] {len(jobs)} 个待单独生成")
