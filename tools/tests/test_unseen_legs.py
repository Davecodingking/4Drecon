"""看不见的腿去抖(body_motion.smooth_unseen_legs, 2026-11-12):
jw4 甄子丹坐地上的特写, 腿出画, 模型每帧重猜一次腿, 左膝 98° -> 73° -> 97° 一帧一跳; 关节平滑是按肢体速度放宽的,
把这种跳当成快动作放过去了。看不见的腿不是观测: 去掉单帧乱跳, 再按时间平滑。看得见的帧一个数不动。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_motion as BM

COLS = [53, 62]


def _P(knee):
    P = np.zeros((len(knee), 204)); P[:, 53] = np.radians(knee); P[:, 62] = np.radians(knee); P[:, 10] = 0.3
    return P


def test_single_frame_jumps_in_unseen_legs_are_removed():
    k = np.array([98, 97, 73, 97, 99, 98, 75, 98, 97, 98], float)        # 甄子丹左膝那样的单帧乱跳
    t = np.arange(len(k)) / 6.0
    out = BM.smooth_unseen_legs(_P(k), np.ones(len(k), bool), t, np.zeros(len(k), int), COLS)
    o = np.degrees(out[:, 62])
    assert np.abs(o - 97.5).max() < 6, o
    assert np.abs(np.diff(o)).max() < 4, o


def test_seen_frames_and_other_columns_untouched():
    k = np.array([98, 70, 100, 73, 97, 99, 75, 98, 97, 98], float)
    seen = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1], bool)
    P = _P(k)
    out = BM.smooth_unseen_legs(P, ~seen, np.arange(len(k)) / 6.0, np.zeros(len(k), int), COLS)
    assert np.array_equal(out[seen], P[seen])
    other = np.setdiff1d(np.arange(204), COLS)
    assert np.array_equal(out[:, other], P[:, other])


def test_real_slow_change_is_kept():
    k = np.linspace(20, 110, 25)                                           # 4 秒里慢慢蹲下去: 真动作, 不能被抹平
    t = np.arange(len(k)) / 6.0
    out = BM.smooth_unseen_legs(_P(k), np.ones(len(k), bool), t, np.zeros(len(k), int), COLS)
    o = np.degrees(out[:, 62])
    assert np.abs(o[3:-3] - k[3:-3]).max() < 3, o - k


def test_does_not_smooth_across_a_cut():
    k = np.array([30, 31, 30, 29, 30, 100, 101, 99, 100, 101], float)     # 切镜头: 两边是两个姿势
    seg = np.array([0] * 5 + [1] * 5)
    out = BM.smooth_unseen_legs(_P(k), np.ones(len(k), bool), np.arange(len(k)) / 6.0, seg, COLS)
    o = np.degrees(out[:, 62])
    assert o[:5].max() < 33 and o[5:].min() > 97, o


def test_same_seconds_at_30fps():
    # 窗口按秒算: 30 fps 时同样的乱跳(持续 1/6 秒 = 5 帧)也去掉
    k = np.full(60, 98.0); k[20:25] = 73.0
    t = np.arange(60) / 30.0
    out = BM.smooth_unseen_legs(_P(k), np.ones(60, bool), t, np.zeros(60, int), COLS)
    o = np.degrees(out[:, 62])
    assert np.abs(o - 98).max() < 6, o


if __name__ == "__main__":
    for f in [test_single_frame_jumps_in_unseen_legs_are_removed, test_seen_frames_and_other_columns_untouched,
              test_real_slow_change_is_kept, test_does_not_smooth_across_a_cut, test_same_seconds_at_30fps]:
        f(); print("ok", f.__name__)
    print("== test_unseen_legs 全过 ==")
