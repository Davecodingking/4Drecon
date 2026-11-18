"""把 carryfit 求出的"被拿着的物体"加进流程导出的 blend(12_motion/<场景>_scene_people.blend), 另存一份, 原文件不动。

坐标换算(和 pipeline/motion_export.py、pipeline/cam_track.py 一致):
  拟合结果  : 物体局部系 -> DA3 世界              carry_poses.npz 的 local_to_world(每关键帧)
  原始网格  : glb 场景坐标 -> 物体局部系           carry_poses.npz 的 mesh_to_local
  DA3 世界 -> glTF 世界(天 +y, 场景 glb 的系)     T = motion.npz 的 cam_c2w @ inv(DA3 的 c2w), 每帧都一样(这里核对)
  glTF 世界 -> Blender 世界(天 +z)               G2B
  Blender 导入 glb 时自己做了 G2B, 所以物体的 matrix_world = G2B · T · M_k · A · G2B⁻¹
关键帧打在原视频帧号上(blend 里人和相机也是这套帧号)。

  ~/venvs/bpy/bin/python carryfit/to_blend.py --r1 <r1> --poses carryfit/runs/<名字>/carry_poses.npz --name carried_box
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix

G2B = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])

ap = argparse.ArgumentParser()
ap.add_argument("--r1", required=True)
ap.add_argument("--poses", required=True)
ap.add_argument("--name", default="carried_box")
ap.add_argument("--out", default=None)
ap.add_argument("--src", default=None, help="在哪个 blend 上加(默认流程导出的那个; 加第二个物体时给上一次的输出)")
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])

scene_name = os.path.basename(os.path.dirname(os.path.abspath(a.r1).rstrip("/")))
src = a.src or f"{a.r1}/12_motion/{scene_name}_scene_people.blend"
out = a.out or os.path.join(os.path.dirname(a.poses), f"{scene_name}_scene_people_carry.blend")
P = np.load(a.poses, allow_pickle=True)
keys, vf, M, A = list(P["keys"]), P["vf"], P["local_to_world"], P["mesh_to_local"]

# DA3 世界 -> glTF 世界: 用相机反推, 核对每帧一致
mo = np.load(f"{a.r1}/12_motion/motion.npz")
cams = json.load(open(f"{a.r1}/02_da3/cameras.json"))["cameras"]
row = {int(v): i for i, v in enumerate(mo["cam_vf"])}
Ts = []
for k, v in zip(keys, vf):
    c2w_d = np.linalg.inv(np.array(cams[k]["extrinsic_w2c_4x4"], float))
    Ts.append(mo["cam_c2w"][row[int(v)]] @ np.linalg.inv(c2w_d))
Ts = np.array(Ts)
dev = np.abs(Ts - Ts[0]).max()
assert dev < 1e-6, f"DA3 -> 导出世界的变换每帧不一样(差 {dev}), 不能直接换"
T = Ts[0]

bpy.ops.wm.open_mainfile(filepath=src)
sc = bpy.context.scene
if a.name in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects[a.name], do_unlink=True)
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=str(P["mesh"]))
new = [o for o in bpy.data.objects if o not in before]
meshes = [o for o in new if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
obj.parent = None
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)   # 网格数据 = 导入后的 Blender 坐标
for o in new:
    if o.name in bpy.data.objects and o != obj:
        bpy.data.objects.remove(o, do_unlink=True)
obj.name = a.name
obj.rotation_mode = "QUATERNION"

for i, v in enumerate(vf):
    W = G2B @ T @ M[i] @ A @ np.linalg.inv(G2B)
    obj.matrix_world = Matrix(W.tolist())
    for path in ("location", "rotation_quaternion", "scale"):
        obj.keyframe_insert(data_path=path, frame=int(v))
if obj.animation_data and obj.animation_data.action:
    for fc in obj.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

# 自检: 抽几帧, 把 Blender 里的顶点反算回拟合用的物体局部系, 应正好落在归一化的盒子里(中心 0, 最长边半长 0.5)
dg = bpy.context.evaluated_depsgraph_get()
chk = []
for i in range(0, len(vf), max(1, len(vf) // 6)):
    sc.frame_set(int(vf[i]))
    Wm = np.array(obj.matrix_world)
    X = np.array([v.co[:] for v in obj.data.vertices]) @ Wm[:3, :3].T + Wm[:3, 3]
    q = (np.linalg.inv(G2B @ T @ M[i]) @ np.c_[X, np.ones(len(X))].T).T[:, :3]
    chk.append((round(float(np.abs((q.max(0) + q.min(0)) / 2).max()), 4), round(float(q.max(0)[0]), 4)))
print("自检(中心偏差, 最长边半长) 应为 (≈0, ≈0.5):", chk)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out))
print("saved", out)
