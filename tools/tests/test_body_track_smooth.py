#!/usr/bin/env python
"""body_into_scene 的位置平滑(smooth_track): 人在地面上的水平位置。
以前只有 5 帧中位 + 3 帧平均(前后约 0.1 秒), 只压单帧尖刺, 来回晃几帧的锯齿原样留着
(09-28 soccer r2 骨盆中心偏离平滑路径 2.5~5 cm)。现在按时间做局部二次拟合(高斯权重):
匀速、匀加速(起跑、扑出去)原样保留, 只去掉来回晃。跑: $PY tools/tests/test_body_track_smooth.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_into_scene as B

t = np.arange(120) / 24.0
rng = np.random.default_rng(0)


def test_zigzag_removed_on_straight_run():
    truth = np.stack([4.0 * t, 0 * t, 0.5 * t], 1)                  # 匀速 4 m/s 斜着跑
    zig = truth + np.stack([0 * t, 0 * t, 0.04 * (-1) ** np.arange(120)], 1) + rng.normal(0, 0.02, (120, 3)) * [1, 0, 1]
    out = B.smooth_track(zig, t)
    e0 = np.linalg.norm(zig - truth, axis=1).mean(); e1 = np.linalg.norm(out - truth, axis=1).mean()
    assert e1 < 0.35 * e0, (e1, e0)                               # 去掉 65% 以上


def test_acceleration_kept():
    """起跑/扑出去: 匀加速原样保留(二次拟合对抛物线是精确的), 包括两头。"""
    truth = np.stack([2.0 * t ** 2, 0 * t, -1.0 * t], 1)
    e = np.abs(B.smooth_track(truth, t) - truth).max(1)
    assert e[10:-10].max() < 1e-6                                  # 中间精确
    assert e.max() < 0.01, e.max()                                 # 片头片尾用直线拟合(防翘), 匀加速在那里差 0.5 cm


def test_track_start_does_not_curl_up():
    """office 人 2(约 6 fps): 片头只有一边有数据, 二次拟合在边上翘起来, 头两帧脚离地 38 cm。边上要稳。"""
    t6 = np.arange(35) / 6.0
    truth = np.stack([1.2 * t6, 0 * t6, 0 * t6], 1)
    P = truth + rng.normal(0, 0.05, truth.shape) * [1, 1, 1]
    e = np.abs(B.smooth_track(P, t6) - truth).max(1)
    assert e[:3].max() < 0.08 and e[-3:].max() < 0.08, (e[:3], e[-3:])


def test_low_fps_still_smoothed():
    """约 6 fps 时 0.08 秒的窗口里只有 3 帧, 以前凑不够拟合就整段不平滑。现在窗口至少 1.5 个帧间隔。"""
    t6 = np.arange(60) / 6.0
    truth = np.stack([1.0 * t6, 0 * t6, 0 * t6], 1)
    zig = truth + np.stack([0 * t6, 0.1 * (-1) ** np.arange(60), 0 * t6], 1)
    out = B.smooth_track(zig, t6)
    assert np.abs(out - truth)[5:-5].max() < 0.05, np.abs(out - truth)[5:-5].max()


def test_sudden_dive_small_lag():
    """守门员扑救: 站着不动, 突然 0.3 秒里横移 1.5 m 再停住。平滑后任何一帧离真值不超过 8 cm。"""
    x = np.clip((t - 2.0) / 0.3, 0, 1); x = x * x * (3 - 2 * x) * 1.5
    truth = np.stack([x, 0 * t, 0 * t], 1)
    err = np.abs(B.smooth_track(truth, t) - truth).max()
    assert err < 0.08, err


def test_single_frame_spikes_ignored():
    """nfl1 人 3: 跟着 mask 走的那段有单帧跳出去 1.8 m 又回来。平滑后离真值不超过 3 cm。"""
    truth = np.stack([1.0 * t, 0 * t, 0 * t], 1)
    P = truth.copy(); P[50] += [0, 0, 1.8]; P[70] += [1.0, 0, 0]; P[71] += [1.0, 0, 0]
    assert np.abs(B.smooth_track(P, t) - truth).max() < 0.03


def test_bad_stretch_never_pulled_to_origin():
    """nfl1 人 4(画面里 54 px, 深度量错): 一段连着十几帧跳出去 28 m。第二遍把整个窗口都当尖刺剔掉时,
    拟合退化成 0, 人被放到世界原点 —— 也就是第一台相机上(09-28 用户在 3D 页看到"相机上站了个人")。
    不管数据多烂, 输出都不能跑出输入的范围。"""
    P = np.stack([50.0 + t, 0 * t, 20.0 + 0 * t], 1)
    P[90:105] += [28.0, 0, 0]
    out = B.smooth_track(P, t)
    assert out[:, 0].min() >= 50.0 - 0.1 and out[:, 0].max() <= P[:, 0].max() + 0.1, (out[:, 0].min(), out[:, 0].max())
    assert np.abs(out[:, 2] - 20.0).max() < 0.1


def test_no_jump_where_edge_fit_changes():
    """片尾前一个 σ 以内从二次拟合换成直线(防翘)。窗口一长(补过的帧 σ 0.4 秒), 两种拟合在弯路上差几厘米,
    突然换的那一帧人跳一下(09-28 nfl1 人 3 片尾前 10 帧跳 4 cm)。要渐变过去: 弯路上逐帧步长的变化处处都小。"""
    truth = np.stack([0.8 * np.sin(1.2 * t), 0 * t, 0.5 * t], 1)    # 弯着走, 真实步长变化 0.2 cm/帧
    out = B.smooth_track(truth, t, sigma=0.4)
    acc = np.linalg.norm(np.diff(out, 2, axis=0), axis=1)
    assert acc.max() < 0.005, (acc.max(), int(acc.argmax()))


def test_short_track_untouched():
    P = rng.normal(0, 1, (4, 3))
    assert np.array_equal(B.smooth_track(P, t[:4]), P)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f(); print("ok", name)
