#!/usr/bin/env python
"""每一类挑一个最好的实例, 挑它最完整的一帧, 存 frame.png + mask.png 给 SAM3D.
挑帧标准: mask 面积最大(通常也最不被挡) 且不贴画面边(贴边=被裁掉一半)."""
import os, json, numpy as np, cv2, torch
from sam3.model_builder import build_sam3_video_predictor

FR   = "/home/jincheng/sam3d/data/outdoor/dataset/timesquare/frames"
OUT  = "/home/jincheng/sam3d/data/outdoor/result/timesquare/picks"
CKPT = os.path.expanduser("~/sam3d/sam3/checkpoints/sam3.pt")
WORDS = open("/home/jincheng/sam3d/data/outdoor/result/timesquare/09_vocab_main.txt").read().split(",")  # 纯 VLM 输出
os.makedirs(OUT, exist_ok=True)
H,W = cv2.imread(f"{FR}/0000.jpg").shape[:2]
pred = build_sam3_video_predictor(checkpoint_path=CKPT)
sid  = pred.handle_request(request=dict(type="start_session", resource_path=FR))["session_id"]

DEP=os.environ.get("DA3DIR","/home/jincheng/sam3d/data/outdoor/result/timesquare/da3")+"/depth"
HAVE_DEPTH = {int(f[:4]) for f in os.listdir(DEP) if f.endswith("_depth.npy")}  # 只在 DA3 真跑过的帧里挑

def edge_penalty(m):
    """贴边比例: 物体被画面裁掉多少"""
    return max(m[0,:].mean(), m[-1,:].mean(), m[:,0].mean(), m[:,-1].mean())

picked = {}
for w in WORDS:
    pred.handle_request(request=dict(type="reset_session", session_id=sid))
    pred.handle_request(request=dict(type="add_prompt", session_id=sid, frame_index=0, text=w))
    life, best = {}, {}
    for r in pred.handle_stream_request(request=dict(type="propagate_in_video", session_id=sid)):
        fi, o = r["frame_index"], r["outputs"]
        for k, oid in enumerate(o["out_obj_ids"].tolist()):
            m = o["out_binary_masks"][k]; a = int(m.sum())
            if a == 0 or fi not in HAVE_DEPTH: continue
            life[oid] = life.get(oid,0)+1
            ep = edge_penalty(m.astype(bool))
            score = a * (1 - min(ep*4, 0.9))          # 贴边重罚
            if oid not in best or score > best[oid][0]:
                best[oid] = (score, a, fi, m.copy(), ep)
    # 只在活够 20 帧的实例里选, 取得分最高的
    cands = [(oid,)+best[oid] for oid in best if life[oid] >= int(os.environ.get("MINF","10"))]
    if not cands:
        print(f"[{w}] 没有活够 20 帧的实例, 跳过", flush=True); continue
    oid, score, a, fi, m, ep = max(cands, key=lambda t: t[1])
    name = w.replace(" ","_")
    d = os.path.expanduser(f"~/sam3d/sam-3d-objects/my_images/ts_{name}"); os.makedirs(d, exist_ok=True)
    img = cv2.imread(f"{FR}/{fi:04d}.jpg")
    cv2.imwrite(f"{d}/frame.png", img)
    cv2.imwrite(f"{d}/mask.png", (m.astype(np.uint8)*255))
    ov = img.copy(); mb = m.astype(bool)
    ov[mb] = (0.5*ov[mb] + 0.5*np.array([60,60,230])).astype(np.uint8)
    cv2.imwrite(f"{OUT}/{name}.jpg", ov, [cv2.IMWRITE_JPEG_QUALITY,88])
    picked[w] = dict(obj_id=int(oid), frame=int(fi), 面积占比=round(100*a/(H*W),2),
                     活帧数=life[oid], 贴边=round(float(ep),3), 候选数=len(cands), 目录=d)
    print(f"[{w}] 选实例#{oid} 第{fi}帧 占{100*a/(H*W):.2f}% 活{life[oid]}帧 贴边{ep:.3f} (共{len(cands)}个候选)", flush=True)
json.dump(picked, open(f"{OUT}/picks.json","w"), ensure_ascii=False, indent=1)
print(f"\n[done] {len(picked)} 类已备好 -> ~/sam3d/sam-3d-objects/my_images/ts_*")
