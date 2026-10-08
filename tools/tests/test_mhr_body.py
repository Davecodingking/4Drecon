#!/usr/bin/env python
"""pipeline/mhr_body.py: SAM 3D Body 的一手参数 <-> 网格/骨架, 相机系 <-> 世界系。
这些规则(根绕哪儿转、欧拉角约定、平移单位)是数值实验摸出来的(2026-10-08), 上游文档里没写 —— 钉在这里。
要 MHR 模型文件在盘上(sam-body4d/checkpoints)。跑: $PY tools/tests/test_mhr_body.py"""
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import mhr_body as M

rng = np.random.default_rng(0)
SHAPE = rng.normal(0, 0.5, 45)


def _params(T=4):
    p = np.zeros((T, 204)); p[:, 6:136] = rng.normal(0, 0.2, (T, 130)); p[:, 136:] = rng.normal(0, 0.05, 68)
    p[:, 3:6] = rng.uniform(-1.2, 1.2, (T, 3))
    return p


def _c2w(T=4):
    M4 = np.tile(np.eye(4), (T, 1, 1))
    M4[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, 1.0, (T, 3))).as_matrix(); M4[:, :3, 3] = rng.normal(0, 5, (T, 3))
    return M4


def test_root_turns_about_the_root_joint_and_moves_in_world_axes():
    p = _params(2); p[:, 3:6] = 0
    v0, s0 = M.forward(SHAPE, p)
    assert np.allclose(s0[:, 1, :3], M.ROOT_C, atol=1e-3), s0[:, 1, :3]          # 根关节固定在 (0, 92.4, 0) cm
    Q = Rotation.from_rotvec(rng.normal(0, 1.0, (2, 3))).as_matrix(); pos = rng.normal(0, 3, (2, 3))
    v1, s1 = M.forward(SHAPE, M.set_root(p, Q, pos))
    c = M.ROOT_C / 100.0
    want = np.einsum("tij,tnj->tni", Q, v0 - c) + pos[:, None]
    assert np.abs(v1 - want).max() < 1e-4, np.abs(v1 - want).max()               # 顶点 = Q (u - c) + 根关节位置
    assert np.abs(s1[:, 1, :3] / 100.0 - pos).max() < 1e-4                       # 根关节就在 pos
    assert np.abs(M.root_pos(M.set_root(p, Q, pos)) - pos).max() < 1e-9
    assert np.abs(M.root_rot(M.set_root(p, Q, pos)) - Q).max() < 1e-9


def test_world_params_reproduce_the_camera_path():
    """老路: 相机系网格(F·MHR + cam_t) 再用相机位姿搬进世界。新路: 世界系参数直接喂 MHR。两条路要一样。"""
    p, c2w, cam_t = _params(), _c2w(), rng.normal(0, 2, (4, 3)) + [0, 0, 8]
    old = np.einsum("tij,tnj->tni", c2w[:, :3, :3], M.cam_verts(SHAPE, p, cam_t)) + c2w[:, None, :3, 3]
    new, _ = M.forward(SHAPE, M.to_world(p, cam_t, c2w))
    assert np.abs(old - new).max() < 2e-4, np.abs(old - new).max()


def test_rigid_transform_of_world_params():
    p = M.to_world(_params(), rng.normal(0, 2, (4, 3)), _c2w())
    T4 = np.eye(4); T4[:3, :3] = Rotation.from_rotvec([0.3, -2.0, 0.5]).as_matrix(); T4[:3, 3] = [1.0, -2.0, 0.5]
    a, _ = M.forward(SHAPE, p); b, _ = M.forward(SHAPE, M.transform(p, T4))
    assert np.abs(a @ T4[:3, :3].T + T4[:3, 3] - b).max() < 2e-4


def test_root_euler_angles_stay_continuous():
    """人原地转一整圈: 朝向是连续的, 存下来的欧拉角也要连续(不然对参数做插值/平滑, 2π 的跳变会被当成猛转)。"""
    ang = np.linspace(0, 2 * np.pi * 1.5, 60)
    Q = Rotation.from_rotvec(np.outer(ang, [0.0, 1.0, 0.0])).as_matrix()
    e = M.set_root(np.zeros((60, 204)), Q, np.zeros((60, 3)))[:, 3:6]
    assert np.abs(np.diff(e, axis=0)).max() < 0.3, np.abs(np.diff(e, axis=0)).max()
    assert np.abs(M.root_rot(M.set_root(np.zeros((60, 204)), Q)) - Q).max() < 1e-9


def test_skeleton_plus_skinning_reproduces_the_mesh():
    """导动作文件靠这条: 关节全局变换 × 自带逆绑定 × 静止网格, 按蒙皮权重加起来 == MHR 自己出的网格(不含姿势修正项)。"""
    p = M.to_world(_params(1), np.array([[0.2, -0.1, 6.0]]), _c2w(1))
    v, s = M.forward(SHAPE, p, correctives=False)
    sk = M.skeleton(); rest = M.rest_verts(SHAPE)

    def mat(x):
        m = np.zeros(x.shape[:-1] + (4, 4))
        m[..., :3, :3] = Rotation.from_quat(x[..., 3:7]).as_matrix() * x[..., 7:8, None]
        m[..., :3, 3] = x[..., :3]; m[..., 3, 3] = 1
        return m
    K = mat(s[0]) @ mat(sk["inv_bind"])
    rh = np.c_[rest, np.ones(len(rest))]
    out = np.zeros((len(rest), 3))
    np.add.at(out, sk["sk_v"], np.einsum("nij,nj->ni", K[sk["sk_j"]], rh[sk["sk_v"]])[:, :3] * sk["sk_w"][:, None])
    assert np.abs(out / 100.0 - v[0]).max() < 1e-4, np.abs(out / 100.0 - v[0]).max()


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f(); print("ok", name)
