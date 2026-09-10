#!/usr/bin/env python
"""每个词的每个实例, 跨它出现的所有帧反投影并合并点云, 拟合带朝向的盒子.
跨帧合并能补回单帧只看到一面的缺口。输出 all_boxes.json 供整场景组装。"""
import os, json, numpy as np, cv2, torch
from sam3.model_builder import build_sam3_video_predictor

ROOT = "/home/jincheng/sam3d/data/outdoor/result/timesquare"
FR   = "/home/jincheng/sam3d/data/outdoor/dataset/timesquare/frames"
DA3  = os.environ.get("DA3DIR", f"{ROOT}/da3")
CKPT = os.path.expanduser("~/sam3d/sam3/checkpoints/sam3.pt")
WORDS= open(f"{ROOT}/09_vocab_main.txt").read().split(",")   # 纯 VLM 输出, 无人工增删
HAVE = {int(f[:4]) for f in os.listdir(f"{DA3}/depth") if f.endswith("_depth.npy")}
cams = json.load(open(f"{DA3}/cameras.json"))["cameras"]
MAXF = 8          # 每个实例最多用几帧合并 (取面积最大的几帧)

def backproj(mask, fname, conf_min=2.0):
    dep=np.load(f"{DA3}/depth/{fname[:-4]}_depth.npy"); cnf=np.load(f"{DA3}/depth/{fname[:-4]}_conf.npy")
    h,w=dep.shape
    r=int(np.clip(np.sqrt(mask.sum())/25,1,6))
    m=cv2.erode(mask.astype(np.uint8),np.ones((r*2+1,)*2,np.uint8))
    m=cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
    m&=(cnf>=conf_min)&np.isfinite(dep)&(dep>0)
    if m.sum()<12: return None
    d=dep[m]; med=np.median(d); mad=np.median(np.abs(d-med))+1e-6
    k=np.abs(d-med)<4*mad
    v,u=np.nonzero(m); v,u,d=v[k],u[k],d[k]
    K=np.array(cams[fname]["intrinsic_3x3"])
    P=np.stack([(u-K[0,2])*d/K[0,0],(v-K[1,2])*d/K[1,1],d],1)
    c2w=np.linalg.inv(np.array(cams[fname]["extrinsic_w2c_4x4"]).reshape(4,4))
    return (c2w[:3,:3]@P.T).T+c2w[:3,3]

# 世界系里的"上"方向: DA3 世界=首帧相机系(OpenCV, y down) -> 上 = -y
raise SystemExit(
    "[已废弃] 这个脚本已被 boxes_from_masks.py 取代(见 outdoor_pipeline.py 开头的注释)。\n"
    "  它还写死着 UP=[0,-1,0], 在俯拍场景上偏 26~29°, 随手跑出来的盒子是错的。\n"
    "  要量盒子请跑: python data/outdoor/scripts/boxes_from_masks.py")
UP = np.array([0,-1,0.])

def fit_box(P):
    """水平面内 PCA 定 yaw, 竖直方向单独量高度"""
    hgt = P@UP
    Q = P - np.outer(hgt, UP)                      # 投到水平面
    c2 = Q.mean(0); R = Q-c2
    C = R.T@R/len(R)
    ev,V = np.linalg.eigh(C)
    ax = V[:,np.argsort(ev)[::-1]]                 # 水平主轴 (含 UP 方向的退化分量)
    a1 = ax[:,0]-UP*(ax[:,0]@UP); a1/=np.linalg.norm(a1)
    a2 = np.cross(UP,a1)
    pr = np.stack([P@a1, P@a2, hgt],1)
    lo,hi = pr.min(0), pr.max(0)
    ctr = a1*(lo[0]+hi[0])/2 + a2*(lo[1]+hi[1])/2 + UP*(lo[2]+hi[2])/2
    return dict(中心=ctr.tolist(), 长轴=a1.tolist(), 尺寸=(hi-lo).tolist(),
                底高=float(lo[2]), 点数=len(P))

pred=build_sam3_video_predictor(checkpoint_path=CKPT)
sid=pred.handle_request(request=dict(type="start_session",resource_path=FR))["session_id"]
all_boxes={}
for w in WORDS:
    pred.handle_request(request=dict(type="reset_session",session_id=sid))
    pred.handle_request(request=dict(type="add_prompt",session_id=sid,frame_index=0,text=w))
    per={}
    for r in pred.handle_stream_request(request=dict(type="propagate_in_video",session_id=sid)):
        fi,o=r["frame_index"],r["outputs"]
        if fi not in HAVE: continue
        for k,oid in enumerate(o["out_obj_ids"].tolist()):
            m=o["out_binary_masks"][k]; a=int(m.sum())
            if a>0: per.setdefault(oid,[]).append((a,fi,m.copy()))
    boxes=[]
    for oid,lst in per.items():
        if len(lst)<int(os.environ.get("MINF","4")): continue
        lst.sort(key=lambda t:-t[0])
        pts=[]
        for a,fi,m in lst[:MAXF]:
            P=backproj(m,f"{fi:04d}.jpg")
            if P is not None: pts.append(P)
        if not pts: continue
        P=np.concatenate(pts)
        # 剔掉离群帧: 距整体中位中心太远的点
        med=np.median(P,0); P=P[np.linalg.norm(P-med,axis=1)<np.percentile(np.linalg.norm(P-med,axis=1),95)]
        b=fit_box(P); b["obj_id"]=int(oid); b["帧数"]=len(lst); b["用帧"]=len(pts)
        boxes.append(b)
    all_boxes[w]=boxes
    sz=[np.round(b["尺寸"],2).tolist() for b in boxes[:4]]
    print(f"[{w}] {len(boxes)} 个实例  前几个尺寸: {sz}", flush=True)
OUTNAME=f"{ROOT}/"+os.environ.get("BOXOUT","06_all_boxes.json")
json.dump(all_boxes, open(OUTNAME,"w"), ensure_ascii=False, indent=1)
print(f"\n[done] 共 {sum(len(v) for v in all_boxes.values())} 个实例 -> {OUTNAME}")
