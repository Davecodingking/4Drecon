#!/usr/bin/env python
"""验收: 整屋 GLB 在几个真实相机位离屏渲染, 和原照片并排。

data/Hybrid/r2,r3/render_check.py 是同一件事的硬编码版 (机位写死、路径指向已经改名
删掉的 data/test_noblur*), 这里改成带参数的, 给 indoor_pipeline 的 render 步骤用。

坐标: cameras.json 的外参在 DA3 世界系, 整屋 GLB 在 z 朝上系, 所以要先左乘 zup 变换的逆。
      变换里的 s (尺度) 要乘进 R —— 漏了会渲染出一个大小对不上的屋子。
      pyrender 相机是 OpenGL 约定 (y 上 z 后), OpenCV 的 c2w 要右乘 diag(1,-1,-1,1)。

看图注意: 渲染结果没有天花板、也没有 SpatialLM 盒子之外的前景家具, 所以看起来会比照片
"往前冲", 这是错觉, 不是没对齐。判断对齐看桌角/栏杆/柱子的位置。

用法: python render_check_indoor.py --glb 00_整屋.glb --da3 02_DA3点云 \
        --zup 03_点云_z朝上.json --frames <提亮帧目录> --out 12_照片对比.jpg [--frames-idx 0,12,24,36]
"""
import argparse, json, os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np, trimesh, pyrender, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("--glb", required=True)
ap.add_argument("--da3", required=True)
ap.add_argument("--zup", required=True, help="03_点云_z朝上.json (变换记录)")
ap.add_argument("--frames", required=True, help="和 DA3 用的是同一套帧 (提亮帧)")
ap.add_argument("--out", required=True)
ap.add_argument("--frames-idx", default="", help="逗号分隔的帧序号 (按 images_used 顺序), 空=均匀取4个")
ap.add_argument("--n", type=int, default=4, help="不给 --frames-idx 时取几个机位")
ap.add_argument("--width", type=int, default=960, help="每格宽度, 太大图会几十 MB")
a = ap.parse_args()

cams = json.load(open(os.path.join(a.da3, "cameras.json")))
zt = json.load(open(a.zup))
T = np.eye(4); T[:3, :3] = zt.get("s", 1.0) * np.array(zt["R"]); T[:3, 3] = np.array(zt["t"])
T_inv = np.linalg.inv(T)

names = cams["images_used"]
picks = ([int(x) for x in a.frames_idx.split(",") if x.strip()] if a.frames_idx
         else np.linspace(0, len(names) - 1, a.n).round().astype(int).tolist())

tm = trimesh.load(a.glb)
scene = pyrender.Scene.from_trimesh_scene(tm, ambient_light=[0.75] * 3)

r = None; tiles = []
for fi in picks:
    name = names[fi]; c = cams["cameras"][name]
    w2c = np.array(c["extrinsic_w2c_4x4"]) @ T_inv
    pose = np.linalg.inv(w2c) @ np.diag([1., -1., -1., 1.])
    K = np.array(c["intrinsic_3x3"]); Hp, Wp = c["input_hw"]
    if r is None: r = pyrender.OffscreenRenderer(Wp, Hp)
    cam = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
    nc = scene.add(cam, pose=pose); nl = scene.add(pyrender.PointLight(intensity=20.0), pose=pose)
    img, _ = r.render(scene)
    scene.remove_node(nc); scene.remove_node(nl)
    photo = cv2.resize(np.asarray(Image.open(os.path.join(a.frames, name)).convert("RGB")), (Wp, Hp))
    pair = np.concatenate([photo, np.full((Hp, 6, 3), 255, np.uint8), img], axis=1)
    s = a.width * 2 / pair.shape[1]
    pair = cv2.resize(pair, (int(pair.shape[1] * s), int(pair.shape[0] * s)))
    cv2.putText(pair, f"frame {fi}  {name}   [left: photo   right: render]", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    tiles.append(pair)

sheet = np.concatenate(tiles, axis=0)
cv2.imwrite(a.out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(f"[done] {len(tiles)} 个机位 -> {a.out}")
