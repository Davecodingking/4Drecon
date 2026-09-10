#!/usr/bin/env python
"""mesh 一览: 把 sam-3d-objects/outputs 下某个 tag 的所有 glb 排成一张图, 每格一个物体。

用来肉眼查"同一类是不是复用了同一个 mesh"—— 格子标题里的名字就是复用单位:
  <tag>_<词>            类 mesh, 这个词的所有实例共用
  <tag>_<词>_<obj_id>   solo mesh, 只有这一个实例用 (由 SOLO 配置决定)
  <tag>_<词>_rec<id>    召回补的 mesh (mask_recall 出的, SAM3 没跟到的物体)
所以一览里出现 "chair" 和 "chair_rec10021" 两种椅子, 说明召回那一支没有复用类 mesh。

渲染用 pyrender 的离屏后端 (EGL), 不需要显示器。相机固定在斜 45 度俯视,
按每个 mesh 自己的包围球放大, 所以图上大小不代表真实尺寸。
★ 底下那行是 mesh 自己的**长宽高比例**(最长边归一到 1), 不是米 —— SAM3D 出的 mesh
  本来就是归一化的, 真实尺寸是组装那步按盒子高度算标量缩放上去的。看比例够判断形状。

用法:
  python tools/viz_mesh_gallery.py --outputs sam-3d-objects/outputs --tag cyb_o1 \
      --out data/indoorOutput/cyberpunk/o1/05_meshes.jpg [--cols 6]
"""
import argparse, glob, os
import numpy as np, trimesh, cv2

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender

ap = argparse.ArgumentParser()
ap.add_argument("--outputs", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cols", type=int, default=6)
ap.add_argument("--cell", type=int, default=320)
a = ap.parse_args()

paths = sorted(glob.glob(f"{a.outputs}/{a.tag}_*/*.glb"))
print(f"[输入] {len(paths)} 个 mesh")

def render(m, S):
    sc = pyrender.Scene(bg_color=[0.13, 0.13, 0.15, 1.0], ambient_light=[.35, .35, .38])
    m = m.copy()
    m.apply_translation(-m.bounding_box.centroid)
    r = float(np.linalg.norm(m.extents)) / 2 or 1.0
    sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    # 斜 45 度俯视, 距离按包围球定 -> 每个 mesh 都刚好占满格子
    d = r * 2.6
    yaw, pit = np.radians(35), np.radians(22)
    eye = np.array([d*np.cos(pit)*np.sin(yaw), d*np.sin(pit), d*np.cos(pit)*np.cos(yaw)])
    f = -eye / np.linalg.norm(eye)
    rt = np.cross(f, [0, 1, 0.]); rt /= np.linalg.norm(rt)
    up = np.cross(rt, f)
    P = np.eye(4); P[:3, 0], P[:3, 1], P[:3, 2], P[:3, 3] = rt, up, -f, eye
    sc.add(pyrender.PerspectiveCamera(yfov=np.pi/4), pose=P)
    sc.add(pyrender.DirectionalLight(intensity=3.5), pose=P)
    col, _ = pyrender.OffscreenRenderer(S, S).render(sc)
    return col[:, :, ::-1].copy()

cells = []
for p in paths:
    n = os.path.basename(os.path.dirname(p))
    s = trimesh.load(p, force="scene")
    m = trimesh.util.concatenate([g for g in s.geometry.values()])
    img = render(m, a.cell)
    e = m.extents
    short = n[len(a.tag) + 1:]
    cv2.putText(img, short[:34], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"ratio {e[0]:.2f} x {e[1]:.2f} x {e[2]:.2f}", (6, a.cell - 10),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (140, 220, 255), 1, cv2.LINE_AA)
    cells.append(img)
    print(f"  {short:38s} {e[0]:5.2f} x {e[1]:5.2f} x {e[2]:5.2f}")

C = a.cols
rows = [np.hstack(cells[i:i+C] + [np.zeros((a.cell, a.cell, 3), np.uint8)] * (C - len(cells[i:i+C])))
        for i in range(0, len(cells), C)]
cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"[done] {len(cells)} 格 -> {a.out}")
