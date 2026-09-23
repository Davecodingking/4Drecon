#!/usr/bin/env python
"""把逐栋生成的楼按点云盒子摆回去, 单独导出 building.glb。
和物体走同一条链, 只是每栋各是各的 mesh(每栋楼都独一无二, 不复用)。"""
import sys, os, json, numpy as np, cv2, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_frame import load_up, export_T, inverse_T, usable_frames
from pycocotools import mask as mask_util
TAG=os.environ.get("TAG","ts_r1")
R1=os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
DA3=f"{R1}/02_da3"; OUTS=os.environ.get("GEN_OUT", os.path.expanduser("~/sam3d/sam-3d-objects/outputs"))
cams=json.load(open(f"{DA3}/cameras.json"))["cameras"]
HAVE=usable_frames(DA3, "建筑点云")      # 跳过位姿不可信的帧
# 天方向以前写死 [0,-1,0], 而物体那条链早就按实际算了 —— 楼和物体拼进同一个 glb 时
# 物体对、楼歪。现在两边同源。
UP, _GH, _FB = load_up(DA3, "摆楼")
z=np.load(f"{R1}/03_masks/building.npz",allow_pickle=True)
meta=z["meta"]; counts=z["counts"]; H,W=z["hw"]

def backproj(mask, fi, conf_min=float(os.environ.get("CONF","1.0"))):
    dep=np.load(f"{DA3}/depth/{fi:04d}_depth.npy"); cnf=np.load(f"{DA3}/depth/{fi:04d}_conf.npy")
    h,w=dep.shape
    r=int(np.clip(np.sqrt(mask.sum())/25,1,6))
    m=cv2.erode(mask.astype(np.uint8),np.ones((r*2+1,)*2,np.uint8))
    m=cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
    m&=(cnf>=conf_min)&np.isfinite(dep)&(dep>0)
    if m.sum()<12: return None
    d=dep[m]; med=np.median(d); mad=np.median(np.abs(d-med))+1e-6
    k=np.abs(d-med)<4*mad
    v,u=np.nonzero(m); v,u,d=v[k],u[k],d[k]
    K=np.array(cams[f"{fi:04d}.jpg"]["intrinsic_3x3"])
    P=np.stack([(u-K[0,2])*d/K[0,0],(v-K[1,2])*d/K[1,1],d],1)
    c2w=np.linalg.inv(np.array(cams[f"{fi:04d}.jpg"]["extrinsic_w2c_4x4"]).reshape(4,4))
    return (c2w[:3,:3]@P.T).T+c2w[:3,3]

def fit_box(P):
    hgt=P@UP; Q=P-np.outer(hgt,UP); R=Q-Q.mean(0)
    ev,V=np.linalg.eigh(R.T@R/len(R)); ax=V[:,np.argsort(ev)[::-1]]
    a1=ax[:,0]-UP*(ax[:,0]@UP); a1/=np.linalg.norm(a1); a2=np.cross(UP,a1)
    pr=np.stack([P@a1,P@a2,hgt],1); lo,hi=pr.min(0),pr.max(0)
    return (a1*(lo[0]+hi[0])/2+a2*(lo[1]+hi[1])/2+UP*(lo[2]+hi[2])/2), a1, (hi-lo), lo[2]

# --- 和 assemble_scene.py 同一个坑 ---
# 下面 T 的写法默认"mesh 的长边在自己的 +Z 上", 但 SAM3D 不保证: 同一批里有的在 Z 有的在 X,
# 在 X 的摆上去就横 90°(carpark 的 container 14 个里 10 个中招)。摆之前先量再转。
YAW_MIN_AR=float(os.environ.get("YAW_MIN_AR","1.15"))
YAW_FIX=os.environ.get("YAW_FIX","1")=="1"
_yaw_n=[0,0]
def yaw_to_z(m):
    if not YAW_FIX: return np.eye(4)
    V=m.vertices[:,[0,2]]; e=V.max(0)-V.min(0)
    if max(e)/max(min(e),1e-6) < YAW_MIN_AR: _yaw_n[1]+=1; return np.eye(4)
    Q=V-V.mean(0); ev,W=np.linalg.eigh(Q.T@Q); a=W[:,np.argmax(ev)]
    th=np.arctan2(a[0],a[1]); ct,st=np.cos(-th),np.sin(-th)
    T=np.eye(4); T[0,0],T[0,2],T[2,0],T[2,2]=ct,st,-st,ct
    _yaw_n[0]+=int(abs(np.degrees(th))>45)
    return T

live={}
for i,(o,f,a,e) in enumerate(meta):
    if a>0 and int(f) in HAVE: live.setdefault(int(o),[]).append((int(a),int(f),i))

scene=trimesh.Scene(); info={}
for oid,rows in sorted(live.items()):
    n=f"{TAG}_building_{oid}"
    p=f"{OUTS}/{n}/{n}.glb"
    if not os.path.exists(p): continue
    rows.sort(key=lambda t:-t[0]); pts=[]
    for a,fi,i in rows[:8]:
        m=mask_util.decode({"counts":counts[i],"size":[int(H),int(W)]}).astype(bool)
        P=backproj(m,fi)
        if P is not None: pts.append(P)
    if not pts: continue
    P=np.concatenate(pts); med=np.median(P,0); dd=np.linalg.norm(P-med,axis=1)
    P=P[dd<np.percentile(dd,95)]
    c,a1,ext,bot=fit_box(P)
    s=trimesh.load(p); m=s.to_geometry() if hasattr(s,"to_geometry") else s
    m.apply_translation(-m.bounds.mean(0))
    me=m.extents
    sc=ext[2]/max(me[1],1e-6)                     # 楼是立着的, 按高度对齐
    m.apply_scale(sc)
    m.apply_transform(yaw_to_z(m))                # ★ 先把 mesh 自己的长轴摆到 +Z, 同 assemble_scene
    fwd=a1-UP*(a1@UP); fwd/=np.linalg.norm(fwd); rgt=np.cross(UP,fwd)
    T=np.eye(4); T[:3,0],T[:3,1],T[:3,2]=rgt,UP,fwd
    m.apply_transform(T)
    bottom=c-UP*ext[2]/2
    h=m.vertices@UP; m.apply_translation(UP*((bottom@UP)-h.min()))
    dxz=c-m.bounds.mean(0); dxz=dxz-UP*(dxz@UP); m.apply_translation(dxz)
    scene.add_geometry(m,node_name=f"building_{oid}")
    info[oid]=dict(点数=len(P),盒子=np.round(ext,1).tolist(),楼高=round(float(ext[2]),1),
                   缩放后最高=round(float(m.extents[1]),1))
    print(f"  楼#{oid:<3d} 点{len(P):6d}  盒子 {np.round(ext,1)}  摆好后高 {m.extents[1]:.1f} m")
scene.apply_transform(export_T(UP))      # 导出按 glTF y 朝上, 和 assemble_scene 同源
dst=f"{R1}/10_buildings.glb"; scene.export(dst)
json.dump(info,open(f"{R1}/10_buildings.json","w"),ensure_ascii=False,indent=1)
print(f"[朝向] mesh 长轴对齐: {_yaw_n[0]} 栋原本歪了 45°以上, {_yaw_n[1]} 栋太方没动")
print(f"\n[done] {len(info)} 栋 -> {dst}")
