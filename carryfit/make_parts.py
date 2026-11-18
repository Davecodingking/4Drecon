#!/usr/bin/env python
"""carryfit 要的身体部位顶点(一次性, body4d 环境), 存 carryfit/parts.npz:
  hand_L / hand_R   两只手(手腕往下)主权重的顶点, 和 pipeline/make_mhr_parts.py 认手的办法一样
  foot_L / foot_R   两只脚: 腿子树里、静止姿势下离脚底 12 cm 以内的顶点(脚掌 + 脚跟)
不改 pipeline/mhr_parts.npz —— 那个文件的 sha 被摆人那步记着, 改了会让所有人重算。

  ~/miniforge3/envs/body4d/bin/python ~/sam3d/carryfit/make_parts.py
"""
import os
import numpy as np
import torch

SRC = os.path.expanduser("~/sam3d/sam-body4d/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parts.npz")
m = torch.jit.load(SRC, map_location="cpu")
buf = {k: v.detach().cpu().numpy() for k, v in m.named_buffers()}
g = lambda k: buf["character_torch." + k]
rest = g("mesh.rest_vertices").astype(np.float64) / 100.0
par = g("skeleton.joint_parents").astype(int)
NV, NJ = len(rest), len(par)
W = np.zeros((NV, NJ))
np.add.at(W, (g("linear_blend_skinning.vert_indices_flattened"), g("linear_blend_skinning.skin_indices_flattened")),
          g("linear_blend_skinning.skin_weights_flattened"))
dom = W.argmax(1)
kids = {j: [k for k in range(NJ) if par[k] == j] for j in range(NJ)}
sub = lambda j: [j] + sum((sub(k) for k in kids[j]), [])
verts_of = lambda js: np.isin(dom, js)
root = [j for j in range(NJ) if par[j] < 0][0]
PELVIS = kids[root][0]
ch = sorted(kids[PELVIS], key=lambda k: rest[verts_of(sub(k)), 1].mean())
LEGS, UPPER = ch[:2], ch[2]
out = {}
for hip in LEGS:
    v = np.where(verts_of(sub(hip)))[0]
    foot = v[rest[v, 1] < rest[v, 1].min() + 0.12]
    side = "L" if rest[foot, 0].mean() > 0 else "R"
    out[f"foot_{side}"] = foot.astype(np.int32)
for s in (-1, 1):
    c = [k for k in sub(UPPER) if len(sub(k)) >= 20 and verts_of(sub(k)).any() and s * rest[verts_of(sub(k)), 0].mean() > 0.3]
    h = min(c, key=lambda k: len(sub(k)))
    v = np.where(W[:, sub(h)].sum(1) > 0.5)[0]
    out["hand_L" if rest[v, 0].mean() > 0 else "hand_R"] = v.astype(np.int32)
np.savez(OUT, **out)
print(OUT, {k: len(v) for k, v in out.items()})
