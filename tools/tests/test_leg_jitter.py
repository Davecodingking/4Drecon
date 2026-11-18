"""腿估翻的一小段(body_motion.leg_offmask_runs / fix_leg_runs, 2026-11-11 lovelynight Emma 3~4 秒):
脚落到自己分割外面、前后都对得上的短段要找出来; 单帧轻微对不上、腿看不见(出画/整段对不上)、太长、跨切镜头的不动。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_motion as BM

T = 60
t = np.arange(T) / 8.57
ones = np.ones(T)


def test_emma_like_run_is_found():
    onm = ones.copy(); onm[28:36] = [0.37, 0.65, 0.89, 0.46, 0.34, 0.74, 0.56, 0.89]   # Emma 实测
    w = BM.leg_offmask_runs(ones, onm, t)
    assert [(a, b) for a, b, _ in w] == [(27, 36)], w


def test_a_lucky_frame_inside_does_not_split_the_run():
    onm = ones.copy(); onm[28:36] = [0.37, 0.65, 0.97, 0.46, 0.34, 0.74, 0.56, 0.89]   # 第 30 帧碰巧对上
    w = BM.leg_offmask_runs(ones, onm, t)
    assert [(a, b) for a, b, _ in w] == [(27, 36)], w


def test_single_mild_dip_is_kept():
    onm = ones.copy(); onm[40] = 0.87
    assert BM.leg_offmask_runs(ones, onm, t) == []


def test_invisible_legs_are_not_touched():
    onm = np.zeros(T)                                   # 整段对不上 = 腿看不见, 补腿的事
    assert BM.leg_offmask_runs(ones, onm, t) == []
    onm = ones.copy(); onm[20:30] = 0.2; inf = ones.copy(); inf[20:30] = 0.0   # 出画
    assert BM.leg_offmask_runs(inf, onm, t) == []


def test_too_long_is_kept():
    onm = ones.copy(); onm[10:25] = 0.3                # 1.9 秒
    assert BM.leg_offmask_runs(ones, onm, t) == []


def test_needs_good_frames_on_both_sides():
    onm = ones.copy(); onm[0:5] = 0.3                  # 片头, 前面没有对得上的帧
    assert BM.leg_offmask_runs(ones, onm, t) == []
    onm = ones.copy(); onm[20:24] = 0.3; onm[24] = np.nan   # 后面一帧没法判
    assert BM.leg_offmask_runs(ones, onm, t) == []


def test_does_not_cross_a_cut():
    onm = ones.copy(); onm[20:24] = 0.3
    seg = np.zeros(T, int); seg[22:] = 1
    assert BM.leg_offmask_runs(ones, onm, t, seg) == []


def test_only_seated_frames_are_judged():
    onm = ones.copy(); onm[20:24] = 0.3
    sit = ones.astype(bool).copy(); sit[15:30] = False      # 这段站着(走路): 交给补腿, 不在这里过渡
    assert BM.leg_offmask_runs(ones, onm, t, use=sit) == []
    assert len(BM.leg_offmask_runs(ones, onm, t, use=ones.astype(bool))) == 1


def test_fix_interpolates_only_leg_columns():
    pose = np.zeros((T, 6)); pose[:, :3] = np.arange(T)[:, None]; pose[27, 3:] = 1.0; pose[36, 3:] = 2.0
    pose[28:36, 3:] = 9.0                              # 估翻的帧
    out = BM.fix_leg_runs(pose, [(27, 36, {})], t, np.array([3, 4, 5]))
    assert np.allclose(out[:, :3], pose[:, :3])        # 别的列不动
    assert np.all((out[28:36, 3:] > 1.0) & (out[28:36, 3:] < 2.0))
    assert np.all(np.diff(out[27:37, 3]) > 0)


if __name__ == "__main__":
    for f in [test_emma_like_run_is_found, test_a_lucky_frame_inside_does_not_split_the_run, test_single_mild_dip_is_kept, test_invisible_legs_are_not_touched,
              test_too_long_is_kept, test_needs_good_frames_on_both_sides, test_does_not_cross_a_cut, test_only_seated_frames_are_judged,
              test_fix_interpolates_only_leg_columns]:
        f(); print("ok", f.__name__)
    print("== test_leg_jitter 全过 ==")
