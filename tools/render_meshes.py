#!/usr/bin/env python
"""把生成的 glb 逐个渲染成灰底两视角(正面/斜后), 拼成一览图, 顺便报网格健康度."""
import os, glob, numpy as np, trimesh, cv2
TAG=os.environ.get("TAG","ts_r1")
os.environ["PYOPENGL_PLATFORM"]="egl"
import pyrender

# 生成结果目录写死绝对路径的话, 换机器 glob 到空 → 下面走"空 grid 就跳过"那条分支,
# 一张图都不出而且退出码是 0, 链上看不出少了东西。所以给个能被环境变量顶掉的口子。
OUTS=os.environ.get("GEN_OUT", os.path.expanduser("~/sam3d/sam-3d-objects/outputs"))
# R1 不给默认值: 手动跑忘了传时, 老的默认值会把图静默写进 timesquare 别人的结果目录里。
# 宁可直接 KeyError 停在这一行。
DST =os.environ["R1"]+"/05_meshes.jpg"
S=340
def shot(mesh, yaw):
    sc=pyrender.Scene(bg_color=[0.78,0.78,0.78,1.0], ambient_light=[.45]*3)
    m=mesh.copy(); m.apply_translation(-m.bounding_box.centroid)
    r=np.linalg.norm(m.bounding_box.extents)/2 or 1.0
    m.apply_scale(1.0/r)
    sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    cam=pyrender.PerspectiveCamera(yfov=np.pi/4.5)
    th=np.radians(yaw); el=np.radians(18); d=3.2
    eye=np.array([d*np.cos(el)*np.sin(th), d*np.sin(el), d*np.cos(el)*np.cos(th)])
    f=-eye/np.linalg.norm(eye); rgt=np.cross([0,1,0],-f); rgt/=np.linalg.norm(rgt); up=np.cross(-f,rgt)
    P=np.eye(4); P[:3,0],P[:3,1],P[:3,2],P[:3,3]=rgt,up,-f,eye
    sc.add(cam,pose=P); sc.add(pyrender.DirectionalLight(intensity=3.5),pose=P)
    r_=pyrender.OffscreenRenderer(S,S); col,_=r_.render(sc); r_.delete()
    return col

rows=[]
print(f"{'名字':18s} {'面数':>8s} {'部件':>5s} {'水密':>5s} {'尺寸(归一前)':>16s}")
for g in sorted(glob.glob(f"{OUTS}/{TAG}_*/*.glb")):
    n=os.path.basename(g)[:-4]
    sc=trimesh.load(g)
    mesh=sc.dump(concatenate=True) if hasattr(sc,'dump') else sc
    comp=mesh.split(only_watertight=False)
    print(f"{n:18s} {len(mesh.faces):8d} {len(comp):5d} {str(mesh.is_watertight):>5s} "
          f"{'x'.join(f'{v:.2f}' for v in mesh.bounding_box.extents):>16s}")
    tiles=[shot(mesh,0), shot(mesh,150)]
    t=np.hstack(tiles)
    cv2.putText(t,n[6:],(8,26),0,0.7,(20,20,20),2)
    rows.append(t)
while len(rows)%3: rows.append(np.full_like(rows[0],199))
grid=[np.hstack(rows[i:i+3]) for i in range(0,len(rows),3)]
# ★ 空列表要挡住。场景配 SOLO="*" 时一个类 mesh 都不生成(全部走逐实例, 而逐实例排在
# scale 之后), 这里拿到空 grid, np.vstack 直接 ValueError 把整条链打断。
# 这一步只是出一张一览图, 没有下游依赖它, 空就跳过, 不该成为闸门。
if not grid:
    print("[跳过] 一个类 mesh 都没有(这个场景全部走逐实例生成), 不出一览图")
    raise SystemExit(0)
cv2.imwrite(DST, cv2.cvtColor(np.vstack(grid),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90])
print("\nsaved",DST)
