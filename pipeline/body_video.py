#!/usr/bin/env python
"""人放进场景之后的验收视频: 原视频 | 同一个相机看到的重建场景+人 | 两者叠加, 逐帧拼成 mp4。

为什么要它: 4 张静帧看不出人在走动时对不对得上 —— 脚有没有滑、人有没有忽大忽小、
坐着的人是不是一直坐在椅子上。人的顶点按 body_into_scene.py 的 10_people.json 重放,
和合成那步(--mode compose)是同一份数据, 看到的就是会进 glb 的样子。

用法:
  python body_video.py --r1 <场景结果目录> --body <body4d轮次目录> --scene <静态场景 glb> \
                       --fr <场景帧目录> [--out 09_people_check.mp4]
sam3d 环境 (pyrender EGL + trimesh + imageio-ffmpeg)。
"""
import os, sys, json, argparse, subprocess, tempfile, shutil
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np, trimesh, cv2, pyrender
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/sam3d/data/outdoor/scripts"))
import body_into_scene as B
from world_frame import inverse_T

ap = argparse.ArgumentParser()
ap.add_argument("--r1", required=True); ap.add_argument("--body", required=True)
ap.add_argument("--scene", required=True, help="静态场景 glb (相对 r1), 如 07_scene_phy.glb")
ap.add_argument("--fr", required=True, help="场景那套帧的目录 (文件名和 cameras.json 的键一致)")
ap.add_argument("--people", default="10_people.json")
ap.add_argument("--out", default="09_people_check.mp4")
ap.add_argument("--tile", type=int, default=640, help="每一栏的宽")
a = ap.parse_args()

people, faces, kept = B.load_body(a.body)
da3, cams, UP, GH, vid2key = B.load_scene(a.r1)
mesh_run = next(f"{a.body}/{d}" for d in ("04_mesh_occ1", "04_mesh_occ0")
                if os.path.isdir(f"{a.body}/{d}/mesh_4d_individual"))
roots = B.load_roots(mesh_run)
corr = json.load(open(f"{a.r1}/{a.people}"))["people"]
world = B.apply_corr(people, corr, cams, vid2key, kept, roots)        # DA3 世界系, 和相机同一个系
key2vid = {k: v for v, k in vid2key.items()}

# 静态场景: glb 是 glTF 系(天 +y), 乘 inverse_T 回到 DA3 世界系, 和 render_back 同一套
sc = trimesh.load(f"{a.r1}/{a.scene}")
INV = inverse_T(UP)
scene = pyrender.Scene(bg_color=[0.85, 0.87, 0.9, 1.0], ambient_light=[.5] * 3)
def _alpha(g):
    m = getattr(g.visual, "material", None); f = getattr(m, "baseColorFactor", None)
    return float(f[3]) / (255.0 if max(f) > 1.001 else 1.0) if f is not None and len(f) > 3 else 1.0
items = []
for n in sc.graph.nodes_geometry:
    T, g = sc.graph[n]; m = sc.geometry[g].copy(); m.apply_transform(T); m.apply_transform(INV)
    items.append(m)
for m in sorted(items, key=lambda g: -_alpha(g)):                   # 半透明的墙最后画
    scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))

k0 = sorted(cams)[0]; K0 = np.array(cams[k0]["intrinsic_3x3"])
AR = K0[0, 2] / K0[1, 2]
W = a.tile - a.tile % 2; H = int(round(W / AR)); H -= H % 2
r = pyrender.OffscreenRenderer(W, H)
COL = [(230, 120, 40), (40, 160, 230), (60, 190, 90), (200, 70, 160), (220, 200, 40), (120, 90, 220), (40, 200, 200)]
tmp = tempfile.mkdtemp(prefix="bodyvid_")
n = 0
for key in sorted(cams):
    vf = key2vid.get(key)
    c = cams[key]; K = np.array(c["intrinsic_3x3"]); h, w = c["input_hw"]
    sx, sy = W / w, H / h
    cam = pyrender.IntrinsicsCamera(fx=K[0, 0] * sx, fy=K[1, 1] * sy, cx=K[0, 2] * sx, cy=K[1, 2] * sy)
    c2w = np.linalg.inv(np.array(c["extrinsic_w2c_4x4"]).reshape(4, 4))
    pose = c2w @ np.diag([1, -1, -1, 1])
    added = [scene.add(cam, pose=pose), scene.add(pyrender.DirectionalLight(intensity=4.0), pose=pose)]
    for i, (pid, fr) in enumerate(sorted(world.items(), key=lambda x: int(x[0]))):
        if vf in fr:
            pm = trimesh.Trimesh(vertices=fr[vf], faces=faces, process=False)
            pm.visual.vertex_colors = np.array(list(COL[i % len(COL)]) + [255], np.uint8)
            added.append(scene.add(pyrender.Mesh.from_trimesh(pm, smooth=True)))
    col, _ = r.render(scene)
    for nd in added:
        scene.remove_node(nd)
    real = cv2.resize(cv2.cvtColor(cv2.imread(f"{a.fr}/{key}"), cv2.COLOR_BGR2RGB), (W, H))
    blend = (0.5 * real + 0.5 * col).astype(np.uint8)
    row = np.hstack([real, col, blend])
    cv2.putText(row, f"{key}  video frame {vf}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imwrite(f"{tmp}/{n:05d}.jpg", cv2.cvtColor(row, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    n += 1

fps = json.load(open(f"{a.body}/00_frames/frames.json"))["src_fps"] * len(cams) / max(1, max(key2vid.values()) + 1)
import imageio_ffmpeg
ff = imageio_ffmpeg.get_ffmpeg_exe()
out = f"{a.r1}/{a.out}"
# ★ H.264 + yuv420p + 宽高偶数, 浏览器才认 (cv2 的 mp4v 不报错但放不了)
subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", f"{fps:.3f}", "-i", f"{tmp}/%05d.jpg",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], check=True)
shutil.rmtree(tmp)
print(f"[done] {n} 帧 @ {fps:.2f}fps -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
