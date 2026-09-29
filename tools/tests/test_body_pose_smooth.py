#!/usr/bin/env python
"""body_into_scene 的姿势平滑: SAM 3D Body 是单帧模型, 四肢和身体朝向一帧一晃
(骨盆系里的高频抖动: soccer 1.4~2.1 cm/顶点, nfl1 抱摔那堆人 4~6.5 cm, 手脚 11~13 cm)。
以前只平滑"人在哪"(位置、远近), 姿势本身没人管。现在骨盆系里按时间平滑四肢、四元数平滑朝向,
动得快的顶点/转得快的帧少平滑(One Euro Filter 的思路) —— 抖动压掉, 扑救/飞扑不被抹软。
跑: $PY tools/tests/test_body_pose_smooth.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_into_scene as B

P = B.load_parts()
NV = int(P["pelvis"].max()) + 200
rng = np.random.default_rng(0)
BODY = rng.uniform(-0.4, 0.4, (NV, 3)) * [1, 2.2, 0.6]        # 一个假人: 随便一团点, 骨盆那圈用真的静止骨盆
BODY[P["pelvis"]] = P["rest_pelvis"]
LIMB = np.setdiff1d(np.arange(NV), P["pelvis"])[:150]           # 当"手脚"用的一小撮顶点


def _rot(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _walk(T=96, fps=24.0, noise=0.03, swing=None):
    """慢慢转身 + 往前走; noise: 四肢每帧的随机抖动(米); swing: (起, 止) 帧里手脚甩出去 1.2 m 又收回。"""
    t = np.arange(T) / fps
    clean, noisy = [], []
    for i in range(T):
        L = BODY.copy()
        if swing and swing[0] <= i < swing[1]:
            k = np.sin(np.pi * (i - swing[0]) / (swing[1] - swing[0]))
            L[LIMB] += [1.2 * k, 0, 0]
        V = L @ _rot(0.3 * t[i]).T + [0.8 * t[i], 1.0, 5.0]
        clean.append(V)
        J = L.copy()
        J[np.setdiff1d(np.arange(NV), P["pelvis"])] += rng.normal(0, noise, (NV - len(P["pelvis"]), 3))
        noisy.append(J @ _rot(0.3 * t[i] + rng.normal(0, 0.03)).T + [0.8 * t[i], 1.0, 5.0])
    return t, clean, noisy


def _err(A, Bs):
    return float(np.mean([np.linalg.norm(a - b, axis=1).mean() for a, b in zip(A, Bs)]))


def test_jitter_is_removed_and_pelvis_stays():
    t, clean, noisy = _walk()
    out = B.smooth_pose(noisy, t)
    assert _err(out, clean) < 0.5 * _err(noisy, clean), (_err(out, clean), _err(noisy, clean))
    c0 = np.stack([v[P["pelvis"]].mean(0) for v in noisy]); c1 = np.stack([v[P["pelvis"]].mean(0) for v in out])
    assert np.abs(c0 - c1).max() < 1e-9                           # 位置归位置那套管, 这里不动骨盆中心
    assert B.pose_jitter(out, t) < 0.5 * B.pose_jitter(noisy, t)


def test_fast_motion_is_kept():
    """手脚 0.25 秒里甩出去 1.2 m 又收回(扑救/飞扑那种速度): 平滑后甩出去的最远处要留住 85% 以上。"""
    t, clean, noisy = _walk(noise=0.0, swing=(40, 46))
    out = B.smooth_pose(clean, t)
    reach = lambda Vs: max(float((v[LIMB] - v[P["pelvis"]].mean(0)).mean(0) @ _rot(0.3 * ti)[:, 0]) for v, ti in zip(Vs, t))
    assert reach(out) >= 0.85 * reach(clean), (reach(out), reach(clean))


def test_short_track_untouched():
    t, clean, noisy = _walk(T=4)
    out = B.smooth_pose(noisy, t)
    assert all(np.array_equal(a, b) for a, b in zip(out, noisy))


def test_apply_corr_replays_pose_smooth():
    t, clean, noisy = _walk(T=30)
    I = np.eye(4).tolist()
    cams = {f"{i:04d}.jpg": {"extrinsic_w2c_4x4": I} for i in range(30)}
    vid2key, kept = {i: f"{i:04d}.jpg" for i in range(30)}, list(range(30))
    people = {"1": {b: noisy[b] @ B.GL2CV for b in range(30)}}      # ply 是 GL 系, apply_corr 会乘回来
    roots = {"1": {b: np.zeros(3) for b in range(30)}}
    zero = {str(b): [0, 0, 0] for b in range(30)}
    ps = dict(fps=24.0, sigma=B.POSE_SIGMA, sigma_min=B.POSE_SIGMA_MIN, v0=B.POSE_V0, w0=B.POSE_W0)
    w = B.apply_corr(people, {"1": dict(ratio=1.0, frames=list(range(30)), delta=zero, pose_smooth=ps)},
                     cams, vid2key, kept, roots)["1"]
    ref = B.smooth_pose(noisy, t)
    assert all(np.allclose(w[b], ref[b]) for b in range(30))
    w = B.apply_corr(people, {"1": dict(ratio=1.0, frames=list(range(30)), delta=zero)},    # 老文件没有这项: 不平滑
                     cams, vid2key, kept, roots)["1"]
    assert all(np.allclose(w[b], noisy[b]) for b in range(30))


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
