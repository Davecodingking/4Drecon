#!/usr/bin/env python
"""body_into_scene 的轨迹整理(tidy_track): 每帧有可信度, 骨盆中心三个方向一起平滑、贴地、腾空统一处理。
起因(09-28): 以前几层各干各的 —— 位置平滑只管水平, 上下交给 5 帧中位贴地, nfl1 人 3 躺着不动却一帧上下跳 22 cm;
腾空不知道哪些帧是翻转补帧换掉的, 把它们抬到空中 20 cm。跑: $PY tools/tests/test_body_tidy.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_into_scene as B

P = B.load_parts()
NV = int(P["pelvis"].max()) + 200
rng = np.random.default_rng(0)
BODY = rng.uniform(-0.4, 0.4, (NV, 3)) * [1, 2.2, 0.6]           # 假人: 高约 1.76 m, 骨盆那圈用真的静止骨盆
BODY[P["pelvis"]] = P["rest_pelvis"]
UP, GH = np.array([0.0, 1.0, 0.0]), 0.0
FOOT = -BODY[:, 1].min()                                           # 骨盆中心放这么高, 脚底正好在地上
T = 120
t = np.arange(T) / 24.0
bis = list(range(T))


def _person(h, x=None, zig=0.0):
    """h: 每帧骨盆比"脚正好着地"高出多少; x: 水平位置; zig: 左右来回晃的幅度。"""
    x = 0.5 * t if x is None else x
    return {b: BODY + [x[b], FOOT + h[b], 5.0 + zig * (-1) ** b] for b in bis}


def _final(W, conf):
    d = B.tidy_track(W, bis, t, conf, UP, GH)
    return [W[b] + d[b] for b in bis]


def _pelvis_h(Vs):
    return np.array([v[P["pelvis"]].mean(0)[1] for v in Vs])


def _feet(Vs):
    return np.array([v[:, 1].min() for v in Vs])


def test_unreliable_stretch_vertical_noise_gone():
    """nfl1 人 3: 翻转补帧那段动作定住、位置跟 mask 走, 上下噪声 ±10 cm。整理后骨盆稳住, 脚在地上, 不许飞起来。"""
    h = np.zeros(T); h[40:100] = rng.normal(0, 0.08, 60)
    conf = np.ones(T); conf[40:100] = B.CONF_SUB
    out = _final(_person(h), conf)
    ph = _pelvis_h(out)
    assert np.abs(np.diff(ph)).max() < 0.04, np.abs(np.diff(ph)).max()
    assert np.abs(_feet(out)).max() < 0.04, np.abs(_feet(out)).max()


def test_unreliable_stretch_wobble_removed():
    """nfl1 人 3 片尾(09-28): 补过的帧位置跟着 mask 露出来的那一小块走, 每零点几秒前后左右晃 10~20 cm,
    ±0.24 秒的窗口压不住(平滑后一帧还挪 15~22 cm)。不可信的帧参考更长的时间(可信度 1/5 -> 窗口 5 倍长):
    晃的部分去掉, 慢慢挪的部分留着; 可信的帧照旧。"""
    x = 0.3 * t                                                      # 慢慢挪, 0.3 m/s(一帧 1.25 cm)
    xw = x.copy(); xw[50:] += 0.15 * np.sin(2 * np.pi * t[50:] / 0.4)   # 补过的帧: 0.4 秒一个来回, 振幅 15 cm
    conf = np.ones(T); conf[50:] = B.CONF_SUB
    px = np.array([v[P["pelvis"]].mean(0)[0] for v in _final(_person(np.zeros(T), x=xw), conf)])
    assert np.abs(np.diff(px[50:])).max() < 0.03, np.abs(np.diff(px[50:])).max()
    assert np.abs(px[60:110] - x[60:110]).max() < 0.05, np.abs(px[60:110] - x[60:110]).max()
    assert np.abs(px[:40] - x[:40]).max() < 1e-6                    # 可信的帧(窗口碰不到补过的帧)一点不变


def test_no_pop_where_unreliable_stretch_starts():
    """可信 -> 补帧的交界: 窗口从 0.08 秒一下变成 0.4 秒, 两边估法不同, 人在交界那帧跳一下
    (09-28 nfl1 人 3 第 58 帧往下 4.9 cm、横着多挪 3 cm, 两三帧后弹回)。窗口要渐变, 交界处步长变化和别处一样小。"""
    r = np.random.default_rng(1)
    x = 0.8 * np.sin(1.2 * t)                                        # 弯着走
    noise = np.zeros((T, 3)); noise[50:] = r.normal(0, 0.05, (T - 50, 3))   # 补过的帧: mask 噪声 5 cm(三个方向)
    W = {b: BODY + [x[b], FOOT, 5.0 + 0.5 * t[b]] + noise[b] for b in bis}
    conf = np.ones(T); conf[50:] = B.CONF_SUB
    d = B.tidy_track(W, bis, t, conf, UP, GH)
    pc = np.array([(W[b] + d[b])[P["pelvis"]].mean(0) for b in bis])
    acc = np.linalg.norm(np.diff(pc, 2, axis=0), axis=1)
    assert acc[44:56].max() < 0.008, (acc[44:56].max(), int(acc[44:56].argmax()) + 45)


def test_reliable_dive_stays_in_air():
    """nfl1 人 5 飞扑: 可信的帧里脚离地 0.6 m 持续 1 秒多, 要留在空中。"""
    h = np.zeros(T); h[50:80] = 0.6 * np.sin(np.linspace(0, np.pi, 30))
    out = _final(_person(h), np.ones(T))
    assert _feet(out)[65] > 0.4, _feet(out)[65]
    assert abs(_feet(out)[10]) < 0.02 and abs(_feet(out)[110]) < 0.02


def test_unreliable_frames_cannot_float():
    """同样离地 0.6 m, 但那段是补过的(不可信): 不许当腾空抬起来。"""
    h = np.zeros(T); h[50:80] = 0.6 * np.sin(np.linspace(0, np.pi, 30))
    conf = np.ones(T); conf[50:80] = B.CONF_SUB
    out = _final(_person(h), conf)
    assert _feet(out).max() < 0.05, _feet(out).max()


def test_short_blip_is_not_airborne():
    """可信的帧里零星两帧脚离地 30 cm(nfl1 人 3 那种): 连着不够 0.4 秒, 当噪声贴回地面。"""
    h = np.zeros(T); h[60:62] = 0.3
    out = _final(_person(h), np.ones(T))
    assert _feet(out).max() < 0.05, _feet(out).max()


def test_low_fps_three_frames_is_not_airborne():
    """office 约 6 fps: 刚走进画面的 3 帧身体估歪了, 脚离地 35 cm —— 有 0.5 秒, 但只有 3 帧, 不够当"跳起来"。"""
    t6 = np.arange(40) / 6.0; b6 = list(range(40))
    h = np.zeros(40); h[:3] = [0.35, 0.3, 0.2]
    W = {b: BODY + [0.2 * b, FOOT + h[b], 5.0] for b in b6}
    d = B.tidy_track(W, b6, t6, np.ones(40), UP, GH)
    feet = np.array([(W[b] + d[b])[:, 1].min() for b in b6])
    assert feet.max() < 0.05, feet.max()


def test_left_right_zigzag_still_removed():
    zig = 0.04
    out = _final(_person(np.zeros(T), zig=zig), np.ones(T))
    z = np.array([v[P["pelvis"]].mean(0)[2] for v in out])
    assert np.abs(z - 5.0).mean() < 0.35 * zig, np.abs(z - 5.0).mean()


def test_low_weight_frames_count_less():
    """smooth_track 的可信度权重: 一段不可信的帧整体偏了 20 cm, 夹在可信的帧中间 —— 结果要被拉回两边那条线。"""
    truth = np.stack([1.0 * t, 0 * t, 0 * t], 1)
    Pz = truth.copy(); Pz[58:62] += [0, 0, 0.2]
    wt = np.ones(T); wt[58:62] = B.CONF_SUB
    a = np.abs(B.smooth_track(Pz, t)[60] - truth[60]).max()
    b = np.abs(B.smooth_track(Pz, t, wt=wt)[60] - truth[60]).max()
    assert b < 0.7 * a, (b, a)                                   # 实测 19 -> 12 cm。两头都有好帧的补帧段位置本来是插值的, 不会整体偏


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f(); print("ok", name)
