#!/usr/bin/env python
"""室外视频词表试圈: 逐个文字提示跑 SAM3 视频跟踪, 统计每个词圈出几个实例、活多少帧, 存叠加图."""
import os, sys, time, json
import numpy as np, cv2, torch
from sam3.model_builder import build_sam3_video_predictor

FRAMES = os.environ.get("FR","/home/jincheng/sam3d/data/outdoor/dataset/timesquare/frames")
OUT    = os.environ.get("OD","/home/jincheng/sam3d/data/outdoor/result/timesquare/sam3_probe")
CKPT   = os.path.expanduser("~/sam3d/sam3/checkpoints/sam3.pt")
WORDS  = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "car","bicycle","trash can","door","sign","street lamp",
    "hedge","tree","window","pipe","building","road"]
os.makedirs(OUT, exist_ok=True)
nf = len([f for f in os.listdir(FRAMES) if f.endswith(".jpg")])
H,W = cv2.imread(f"{FRAMES}/0000.jpg").shape[:2]
print(f"[info] {nf} 帧, {len(WORDS)} 个词", flush=True)

t0=time.time()
pred = build_sam3_video_predictor(checkpoint_path=CKPT)
print(f"[info] 模型载入 {time.time()-t0:.0f}s, 显存 {torch.cuda.memory_allocated()/2**30:.1f}G", flush=True)

sid = pred.handle_request(request=dict(type="start_session", resource_path=FRAMES))["session_id"]
PAL = np.array([[230,60,60],[60,200,90],[70,130,240],[240,180,50],[190,90,220],
                [50,210,210],[240,120,180],[150,200,60],[250,150,80],[120,120,255]],np.uint8)
report={}
for w in WORDS:
  try:
    pred.handle_request(request=dict(type="reset_session", session_id=sid))
    t=time.time()
    pred.handle_request(request=dict(type="add_prompt", session_id=sid, frame_index=0, text=w))
    per={}
    for r in pred.handle_stream_request(request=dict(type="propagate_in_video", session_id=sid)):
        per[r["frame_index"]] = r["outputs"]
    # 统计: 每个 obj_id 活了多少帧, 平均占画面比例
    life={}; area={}
    for fi,o in per.items():
        for k,oid in enumerate(o["out_obj_ids"].tolist()):
            m=o["out_binary_masks"][k]
            a=int(m.sum())
            if a>0: life[oid]=life.get(oid,0)+1; area[oid]=area.get(oid,0)+a
    ids=sorted(life,key=lambda i:-life[i])
    report[w]=dict(n=len(ids), sec=round(time.time()-t,1),
                   top=[dict(id=i, frames=life[i], pct=round(100*area[i]/life[i]/(H*W),1)) for i in ids[:8]])
    print(f"[{w}] {len(ids)} 个实例, {time.time()-t:.0f}s  " +
          " ".join(f"#{i}:{life[i]}帧/{100*area[i]/life[i]/(H*W):.1f}%" for i in ids[:6]), flush=True)
    # 叠加图: 6 个时间点
    picks=np.linspace(0,nf-1,6).round().astype(int)
    tiles=[]
    for fi in picks:
        img=cv2.cvtColor(cv2.imread(f"{FRAMES}/{fi:04d}.jpg"),cv2.COLOR_BGR2RGB)
        o=per.get(int(fi))
        if o is not None:
            for k,oid in enumerate(o["out_obj_ids"].tolist()):
                m=o["out_binary_masks"][k].astype(bool)
                if m.any():
                    c=PAL[oid%len(PAL)]
                    img[m]=(0.45*img[m]+0.55*c).astype(np.uint8)
                    ys,xs=np.nonzero(m)
                    cv2.putText(img,str(oid),(int(xs.mean()),int(ys.mean())),0,1.2,(255,255,0),3)
        tiles.append(cv2.resize(img,(240,427)))
    sheet=np.hstack(tiles)
    cv2.putText(sheet,f"{w}  ({len(ids)} inst)",(8,28),0,0.9,(255,255,0),2)
    cv2.imwrite(f"{OUT}/{w.replace(' ','_')}.jpg",cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,88])
  except torch.OutOfMemoryError:
    print(f"[{w}] 显存爆了, 跳过", flush=True); torch.cuda.empty_cache()
json.dump(report,open(f"{OUT}/report.json","w"),ensure_ascii=False,indent=1)
print("[done]",OUT, f"总耗时 {time.time()-t0:.0f}s", f"峰值显存 {torch.cuda.max_memory_allocated()/2**30:.1f}G")
