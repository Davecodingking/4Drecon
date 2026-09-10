#!/usr/bin/env python
"""从原视频的相机位置渲染组装好的场景, 和真实帧并排比对。"""
import sys, os, json, numpy as np, trimesh, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_frame import load_up, export_T, inverse_T
os.environ["PYOPENGL_PLATFORM"]="egl"
import pyrender

ROOT=os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
FR  =os.environ.get("FR","/home/jincheng/sam3d/data/outdoor/dataset/timesquare/frames")
cams=json.load(open(f"{ROOT}/02_da3/cameras.json"))["cameras"]
SCENE=os.environ.get("SCENE","07_scene.glb")
import glob
sc_tm=trimesh.Scene()
for _f in os.environ.get("SCENE","07_scene.glb").split(","):
    _s=trimesh.load(f"{ROOT}/{_f}")
    for _n in _s.graph.nodes_geometry:
        _T,_g=_s.graph[_n]; _m=_s.geometry[_g].copy(); _m.apply_transform(_T); sc_tm.add_geometry(_m,node_name=_f+_n)
# 导出时组装那步把"真实的天"转到了 +y(assemble_scene.py 末尾的罗德里格斯 R3),
# 渲染要对回原相机就得乘 R3 的逆。老代码写死 diag(1,-1,-1) 是 world_up 之前的假设,
# 天一旦不是首帧相机的 -y 就对不上: carpark 差 29.2°, beach 差 153.8°, 整个场景转掉。
# 逆旋转从 world_frame 拿, 和 assemble_scene 的 export_T 同源。
# ★ 这里曾经写死 diag(1,-1,-1): 组装那边改成按实际天方向算之后这里没跟着改,
#   carpark 差 29.2°、beach 差 153.8°, 所有验收对比图全部无效。
_UP, _GH, _FB = load_up(f"{ROOT}/02_da3", "渲染回图")
_INV = inverse_T(_UP)
sc_tm.apply_transform(_INV)
W,H=1280,720
# FRAMES 空字符串要当"没设"处理 —— 驱动里 RB_FRAMES 默认是 "", 直接 split 会得到 [''],
# int('') 抛 ValueError。空了就按帧数均匀取 5 帧, 比硬编码那串更适应短视频。
_fr=[int(x) for x in os.environ.get("FRAMES","").split(",") if x.strip()]
if not _fr:
    _n=len([f for f in os.listdir(FR) if f.endswith((".jpg",".png"))])
    _fr=sorted({int(round(i*(_n-1)/4)) for i in range(5)}) if _n>1 else [0]
    print(f"[帧] FRAMES 没给, 从 {_n} 帧里均匀取 {_fr}")
FRAMES=_fr

scene=pyrender.Scene(bg_color=[0.85,0.87,0.9,1.0], ambient_light=[.5]*3)
# ★ 半透明的东西(房间壳的灰墙)必须最后画, 而且要按材质的 alphaMode 混合。
#   pyrender 的 from_trimesh 会读 PBRMaterial 的 alphaMode, 但只有在几何本身带材质时才有;
#   带 ColorVisuals 的会被当成顶点色、alpha 被忽略 —— 那正是"灰墙渲染成白墙"的原因。
def _alpha(g):
    m=getattr(g.visual,"material",None); f=getattr(m,"baseColorFactor",None)
    return float(f[3])/(255.0 if max(f)>1.001 else 1.0) if f is not None and len(f)>3 else 1.0
order=sorted(sc_tm.geometry.items(), key=lambda kv: -_alpha(kv[1]))   # 不透明在前
for name,g in order:
    for node in sc_tm.graph.nodes_geometry:
        T,gname=sc_tm.graph[node]
        if gname==name: scene.add(pyrender.Mesh.from_trimesh(g,smooth=False),pose=T)
tiles=[]
r=pyrender.OffscreenRenderer(W,H)
for fi in FRAMES:
    fn=f"{fi:04d}.jpg"
    if fn not in cams: fn=min(cams,key=lambda k:abs(int(k[:4])-fi)); 
    cam=cams[fn]; K=np.array(cam["intrinsic_3x3"])
    sx=W/ (K[0,2]*2); sy=H/(K[1,2]*2)
    pc=pyrender.IntrinsicsCamera(fx=K[0,0]*sx, fy=K[1,1]*sy, cx=K[0,2]*sx, cy=K[1,2]*sy)
    c2w=np.linalg.inv(np.array(cam["extrinsic_w2c_4x4"]).reshape(4,4))
    gl=c2w@np.diag([1,-1,-1,1])                       # OpenCV -> OpenGL
    n1=scene.add(pc,pose=gl); n2=scene.add(pyrender.DirectionalLight(intensity=4.0),pose=gl)
    col,_=r.render(scene); scene.remove_node(n1); scene.remove_node(n2)
    real=cv2.cvtColor(cv2.imread(f"{FR}/{fn}"),cv2.COLOR_BGR2RGB)
    real=cv2.resize(real,(W,H))
    blend=(0.5*real+0.5*col).astype(np.uint8)
    SZ=int(os.environ.get("TILE","426")); row=np.hstack([cv2.resize(x,(SZ,int(SZ*9/16))) for x in (real,col,blend)])
    cv2.putText(row,f"{fn}   real | rebuilt | 50/50",(8,20),0,0.5,(255,255,0),1)
    tiles.append(row)
r.delete()
cv2.imwrite(f"{ROOT}/"+os.environ.get("RBOUT","08_render_back.jpg"),cv2.cvtColor(np.vstack(tiles),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90])
print("saved",f"{ROOT}/"+os.environ.get("RBOUT","08_render_back.jpg"))
