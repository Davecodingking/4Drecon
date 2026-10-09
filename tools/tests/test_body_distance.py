#!/usr/bin/env python
"""body_into_scene 的逐帧远近: 人离相机多远按**每一帧**的场景深度定, 不是每人一个固定比例 × 模型逐帧距离。
起因: soccer r2 守门员扑救倒地的 30 帧, SAM 3D Body 按"人在画面里多大"估的距离从 19 m 塌到 8~11 m,
场景深度一直是 19~22 m; 固定比例把他往相机拉近 10 m, 俯视图上一条长线。跑: $PY tools/tests/test_body_distance.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_into_scene as B


def test_model_distance_collapse_follows_scene_depth():
    bis = list(range(40))
    z = {b: (9.0 if 15 <= b < 25 else 19.0) for b in bis}          # 模型: 中间 10 帧塌到 9 m
    scene = {b: 21.0 for b in bis}                                  # 场景深度: 一直 21 m
    ratios = {b: scene[b] / z[b] for b in bis}
    R = float(np.median(list(ratios.values())))
    r = B.frame_ratios(bis, ratios, z, R)
    d = np.array([r[b] * z[b] for b in bis])
    assert np.allclose(d, 21.0), d                                  # 每一帧都在 21 m, 没有跳变


def test_single_bad_measurement_is_voted_out():
    bis = list(range(20))
    z = {b: 10.0 for b in bis}
    ratios = {b: 1.1 for b in bis}
    ratios[7] = 2.5                                                 # 某一帧 mask 糊到背景上, 量出 25 m
    r = B.frame_ratios(bis, ratios, z, 1.1)
    assert abs(r[7] * z[7] - 11.0) < 1e-6, r[7]                     # 时间中位数把它压掉


def test_frames_without_measurement_fall_back_to_person_ratio():
    bis = list(range(30))
    z = {b: 10.0 for b in bis}
    ratios = {b: 1.2 for b in range(5)}                             # 只有前 5 帧量到深度(后面被挡)
    r = B.frame_ratios(bis, ratios, z, 1.3)
    assert abs(r[0] - 1.2) < 1e-9 and r[29] == 1.3, (r[0], r[29])              # 窗口里没有测量 -> 用这个人的整体比例


def test_untrusted_frames_never_use_model_distance():
    """nfl1 人 3 片尾(09-28): 翻转补帧那段模型在两个人之间来回认, 人又被挡得只剩 100~500 像素、量不到场景深度。
    以前量不到就退回"整体比例 × 模型自己估的距离", 模型的距离一帧跳 ±1.8 m, 人跟着前后来回晃。
    模型不可信的帧量不到深度 -> 用前后量到的距离按时间接上, 不碰模型的距离。可信的帧照旧(见上一个测试)。"""
    bis = list(range(60))
    z = {b: (9.5 if b % 2 else 11.5) for b in bis}                  # 模型的距离一帧一跳, 来回 2 m
    ratios = {b: 10.0 / z[b] for b in range(20)}                    # 前 20 帧量到场景深度: 10 m
    ratios.update({b: 9.0 / z[b] for b in range(50, 60)})           # 最后 10 帧又量到: 9 m
    r = B.frame_ratios(bis, ratios, z, 1.0, trust=set(range(20)))  # 20 帧以后模型不可信
    d = np.array([r[b] * z[b] for b in bis])
    assert np.abs(np.diff(d)).max() < 0.1, np.abs(np.diff(d)).max()             # 不再一帧跳 2 m
    assert 9.0 - 1e-6 <= d[20:50].min() and d[20:50].max() <= 10.0 + 1e-6, d[20:50]   # 夹在前后量到的两个距离之间


def test_outlier_person_untrusted_frames_also_skip_model_distance():
    """自己量的深度离群的人(nfl1 人 4)不用自己的深度, 整段是 R × 模型的距离。翻转补帧的帧照样不能用模型的距离:
    09-28 人 4 补帧段模型的距离一帧跳 28 m, 退回它就一帧跳 13 m。按前后可信帧的距离接上。"""
    bis = list(range(60))
    z = {b: (10.0 if b < 20 or b >= 40 else (5.0 if b % 2 else 30.0)) for b in bis}   # 20~39 帧模型乱跳
    r = B.frame_ratios(bis, {}, z, 1.0, trust=set(bis) - set(range(20, 40)))        # {} = 自己量的深度不用
    d = np.array([r[b] * z[b] for b in bis])
    assert np.allclose(d, 10.0), d[18:42]


def test_far_depth_noise_is_smoothed_along_ray():
    """远处(20 m)VGGT 深度有 ±1 m 的噪声, 9 帧中位数会把它变成台阶: soccer r2 守门员 139->140 帧
    画面里只挪了 3 px, 距离却从 21.1 跳到 22.3 m。前后方向按秒平滑之后, 逐帧变化要收到 0.1 m 以内。"""
    fps = 21.0
    bis = list(range(120))
    t = {b: b / fps for b in bis}
    z = {b: 19.0 for b in bis}
    scene = {b: (21.0 if (b // 12) % 2 == 0 else 22.3) for b in bis}   # 12 帧一个台阶, 来回跳 1.3 m
    ratios = {b: scene[b] / z[b] for b in bis}
    r = B.frame_ratios(bis, ratios, z, 1.1, t=t)
    d = np.array([r[b] * z[b] for b in bis])
    assert np.abs(np.diff(d)).max() <= 0.1, np.abs(np.diff(d)).max()


def test_real_slow_approach_is_kept():
    """人真的在朝相机走(25 -> 15 m, 约 2 m/s): 平滑不能把它抹掉, 中段跟真值差 0.1 m 以内。"""
    fps = 21.0
    bis = list(range(100))
    t = {b: b / fps for b in bis}
    z = {b: 10.0 for b in bis}
    truth = {b: 25.0 - 10.0 * b / 99 for b in bis}
    r = B.frame_ratios(bis, {b: truth[b] / z[b] for b in bis}, z, 2.0, t=t)
    mid = bis[15:85]
    err = max(abs(r[b] * z[b] - truth[b]) for b in mid)
    assert err <= 0.1, err


def test_airborne_keeps_jumps_but_snaps_noise():
    """nfl1 人 5 飞扑: 2 秒里脚离地 0.4~0.8 m, 不能一帧帧按回地面; 走路的深度噪声(±0.1 m)和整体浮空照旧贴地。"""
    t = np.arange(120) / 24.0
    rng = np.random.default_rng(0)
    walk = B.smooth_track((0.3 + rng.uniform(-0.1, 0.1, 120))[:, None], t)[:, 0]   # 整体浮着 0.3 m + 噪声(流水线里先平滑)
    assert B.airborne(walk, t).max() == 0.0
    dive = np.zeros(120); dive[40:90] = 0.8                          # 第 40~89 帧飞在 0.8 m
    a = B.airborne(dive, t)
    assert abs(a[65] - 0.65) < 1e-9 and a[10] == 0.0 and a[110] == 0.0, (a[65], a[10], a[110])


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
