#!/usr/bin/env python
"""gravity_from_poles: 合成几根已知方向的竖杆 + 多个相机, 看能不能解回竖直、该用不该用判得对不对。"""
import json, os, sys, tempfile
import numpy as np
from pycocotools import mask as mu
import cv2

sys.path.insert(0, os.path.expanduser("~/sam3d/data/outdoor/scripts"))
import gravity_from_poles as G

H, W = 360, 640
K = np.array([[500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1]])
UP = np.array([0, -1.0, 0])                          # OpenCV 世界: -y 朝天(地面法线)


def ry(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def make_scene(root, g_true, n_poles):
    os.makedirs(f"{root}/02_da3"); os.makedirs(f"{root}/03_masks")
    cams, meta, counts = {}, [], []
    rng = np.random.default_rng(0)
    bases = [np.array([x, 1.5, z]) for x, z in zip(rng.uniform(-3, 3, n_poles), rng.uniform(6, 10, n_poles))]
    for f in range(0, 80, 4):                        # 相机边走边转头
        R = ry(np.radians(-15 + f * 0.4)); C = np.array([-2 + f * 0.05, 0, 0])
        w2c = np.eye(4); w2c[:3, :3] = R.T; w2c[:3, 3] = -R.T @ C
        cams[f"{f:04d}.jpg"] = dict(intrinsic_3x3=K.tolist(), extrinsic_w2c_4x4=w2c.tolist(), input_hw=[H, W])
        for oid, b in enumerate(bases):
            uv = []
            for p in (b, b + 1.2 * g_true):        # 1.2 m 高的杆子
                q = K @ (w2c[:3, :3] @ p + w2c[:3, 3]); uv.append(q[:2] / q[2])
            m = np.zeros((H, W), np.uint8)
            cv2.line(m, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), 1, 7)
            if m.sum() == 0: continue
            edge = float(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
            meta.append((oid, f, int(m.sum()), edge))
            counts.append(mu.encode(np.asfortranarray(m))["counts"])
    json.dump(dict(cameras=cams), open(f"{root}/02_da3/cameras.json", "w"))
    c = np.empty(len(counts), object); c[:] = counts
    np.savez(f"{root}/03_masks/bollard.npz", word="bollard", hw=np.array([H, W]), counts=c,
             meta=np.array(meta, dtype=[("oid", "<i4"), ("frame", "<i4"), ("area", "<i8"), ("edge", "<f4")]))


def run(g_true, n_poles):
    with tempfile.TemporaryDirectory() as t:
        make_scene(t, g_true, n_poles); return G.estimate(t, UP)


ang = lambda a, b: float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0, 1))))
g3 = np.array([np.sin(np.radians(3)), -np.cos(np.radians(3)), 0])           # 地面斜 3°

d = run(g3, 6)
assert d["用了"], d["原因"]
assert ang(d["gravity"], g3) < 0.3, f"解出来差 {ang(d['gravity'], g3):.2f}°"
assert abs(d["和地面法线差"] - 3) < 0.3, d["和地面法线差"]
print(f"斜 3°: 解回 {d['和地面法线差']}°, 和真值差 {ang(d['gravity'], g3):.2f}°, 去掉一根最多变 {d['去掉一根最多变']}° -> 用")

d = run(UP, 6)
assert not d["用了"] and "本来就平" in d["原因"], d["原因"]
print(f"平地: {d['原因']}")

d = run(g3, 2)
assert not d["用了"] and "只有 2 根" in d["原因"], d["原因"]
print(f"两根杆子: {d['原因']}")

# 套父节点转 glb: 只改 JSON, 转完包围盒的方向对
import trimesh
with tempfile.TemporaryDirectory() as t:
    trimesh.Scene([trimesh.creation.box((0.1, 2.0, 0.1))]).export(f"{t}/a.glb")
    R = G.rot([0, 1.0, 0], [np.sin(0.1), np.cos(0.1), 0]); G.tilt_glb(f"{t}/a.glb", f"{t}/b.glb", R)
    V = trimesh.load(f"{t}/b.glb", force="scene").to_geometry().vertices
    ax = np.linalg.svd(V - V.mean(0), full_matrices=False)[2][0]
    assert abs(ang(ax, [np.sin(0.1), np.cos(0.1), 0])) < 0.01, ax
print("tilt_glb: 转向正确")
print("== test_gravity_from_poles 全过 ==")
