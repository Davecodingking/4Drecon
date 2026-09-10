#!/usr/bin/env python
"""把 SAM3 视频跟踪抠到的每个实例, 在它最大的那一帧裁出来放灰底上, 拼成特写一览."""
import os, numpy as np, cv2, torch
from sam3.model_builder import build_sam3_video_predictor
FRAMES=os.environ.get("FR","/home/jincheng/sam3d/data/outdoor/dataset/timesquare/frames"); OUT=os.environ.get("OD","/home/jincheng/sam3d/data/outdoor/result/timesquare/sam3_probe")
CKPT=os.path.expanduser("~/sam3d/sam3/checkpoints/sam3.pt")
WORDS=os.environ.get("WORDS","hedge,door,sign,tree,trash can,bicycle,street lamp").split(",")
H,W=cv2.imread(f"{FRAMES}/0000.jpg").shape[:2]
pred=build_sam3_video_predictor(checkpoint_path=CKPT)
sid=pred.handle_request(request=dict(type="start_session",resource_path=FRAMES))["session_id"]
items=[]
for w in WORDS:
    pred.handle_request(request=dict(type="reset_session",session_id=sid))
    pred.handle_request(request=dict(type="add_prompt",session_id=sid,frame_index=0,text=w))
    best={}; life={}
    for r in pred.handle_stream_request(request=dict(type="propagate_in_video",session_id=sid)):
        fi=r["frame_index"]; o=r["outputs"]
        for k,oid in enumerate(o["out_obj_ids"].tolist()):
            m=o["out_binary_masks"][k]; a=int(m.sum())
            if a==0: continue
            life[oid]=life.get(oid,0)+1
            if oid not in best or a>best[oid][0]: best[oid]=(a,fi,m.copy())
    for oid,(a,fi,m) in best.items():
        items.append((w,oid,life[oid],a,fi,m))
    print(f"[{w}] {len(best)} 个", flush=True)
# 按面积排, 只画活过 5 帧的
items=[t for t in items if t[2]>=int(os.environ.get("MINF","5")) and t[3]>=float(os.environ.get("MINA","0"))*H*W]; items.sort(key=lambda t:-t[3])
S=300; cols=6; rows=(len(items)+cols-1)//cols
sheet=np.full((rows*S,cols*S,3),200,np.uint8)
for k,(w,oid,nf,a,fi,m) in enumerate(items):
    img=cv2.imread(f"{FRAMES}/{fi:04d}.jpg"); m=m.astype(bool)
    ys,xs=np.nonzero(m); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    pad=int(0.08*max(y1-y0,x1-x0))+4
    y0,y1,x0,x1=max(0,y0-pad),min(H,y1+pad),max(0,x0-pad),min(W,x1+pad)
    cut=np.full_like(img[y0:y1,x0:x1],200); mm=m[y0:y1,x0:x1]
    cut[mm]=img[y0:y1,x0:x1][mm]
    h,ww=cut.shape[:2]; s=min((S-30)/h,(S-30)/ww); cut=cv2.resize(cut,(int(ww*s),int(h*s)))
    r_,c_=k//cols,k%cols; oy=r_*S+25+(S-25-cut.shape[0])//2; ox=c_*S+(S-cut.shape[1])//2
    sheet[oy:oy+cut.shape[0],ox:ox+cut.shape[1]]=cut
    cv2.putText(sheet,f"{w}#{oid} {nf}f {100*a/(H*W):.1f}%",(c_*S+6,r_*S+18),0,0.45,(20,20,20),1)
cv2.imwrite(f"{OUT}/01_cutouts.jpg",sheet,[cv2.IMWRITE_JPEG_QUALITY,90])
print("saved",f"{OUT}/01_cutouts.jpg",len(items),"个")
