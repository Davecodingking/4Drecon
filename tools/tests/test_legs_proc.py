#!/usr/bin/env python
"""pipeline/legs_proc.py: 补腿(proclegs 规划 + MHR 腿 IK)。要 MHR 模型文件和 ~/sam3d/proclegs 在盘上, 有 GPU 更快。
假人: MHR 静止姿势(体型 0), 天 = +y, 沿 +z(人自己的前方)以 0.7 m/s 平移 4 秒, 10 fps —— 腿是定住的(= 模型猜的腿在滑)。
跑: $PY tools/tests/test_legs_proc.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import mhr_body as M
import legs_proc as LP

DEV = "cuda" if __import__("torch").cuda.is_available() else "cpu"
UP = np.array([0.0, 1.0, 0.0])
SHAPE = np.zeros(M.NSHAPE)


def walker(T=40, fps=10.0, v=0.7, gap_at=None, gap_s=0.0):
    t = np.arange(T) / fps
    if gap_at is not None:
        t[gap_at:] += gap_s
    P = np.zeros((T, 204))
    p0 = M.root_pos(P[:1])[0]
    P = M.set_root(P, np.repeat(np.eye(3)[None], T, 0), p0 + np.outer(t * v, [0, 0, 1.0]))
    V, _ = M.forward(SHAPE, P[:1], False, DEV)
    GH = float(V[0, :, 1].min())
    return P, t, GH


def _crouch(P, lift):
    """膝盖弯 60° 的"猜出来的腿", 整个人抬高 lift 米(= 飘在空中)。"""
    P = P.copy()
    for c in (LP.KNEE_COL["l"], LP.KNEE_COL["r"]):
        P[:, c] = np.radians(60)
    p = M.root_pos(P); p[:, 1] += lift
    return M.set_root(P, None, p)


def test_short_shot_with_unseen_legs_stands_on_the_floor():
    P, t, GH = walker(T=6, v=0.0)                 # 0.5 秒: 太短, 迈不了步
    P = _crouch(P, 0.30)
    out, rep = LP.fill_legs(P, SHAPE, t, np.zeros(len(t), bool), UP, GH, dev=DEV)
    V, _ = M.forward(SHAPE, out, False, DEV)
    low = V[:, :, 1].min(1) - GH
    assert np.abs(low).max() < 0.03, low           # 两脚踩地, 不飘
    assert np.all(np.abs(out[:, [LP.KNEE_COL["l"], LP.KNEE_COL["r"]]]) < np.radians(30)), np.degrees(out[:, [LP.KNEE_COL["l"], LP.KNEE_COL["r"]]])
    assert any(s_.get("补法") == "太短, 站着" for s_ in rep["段"]), rep


def test_short_shot_bad_camera_gets_standing_height():
    P, t, GH = walker(T=6, v=0.0)
    P = _crouch(P, -0.40)                          # 坏相机: 整个人被放低 40 cm
    out, rep = LP.fill_legs(P, SHAPE, t, np.zeros(len(t), bool), UP, GH, dev=DEV, bad=np.ones(len(t), bool))
    V0, _ = M.forward(SHAPE, walker(T=1, v=0.0)[0], False, DEV)
    V, _ = M.forward(SHAPE, out, False, DEV)
    head0 = V0[0, :, 1].max() - GH; head = V[:, :, 1].max(1) - GH
    assert np.abs(head - head0).max() < 0.06, (head, head0)   # 站着的身高(膝盖微弯), 不是蹲着
    assert np.abs(V[:, :, 1].min(1) - GH).max() < 0.03


def test_short_shot_frames_with_seen_legs_are_left_alone():
    P, t, GH = walker(T=6, v=0.0)
    P = _crouch(P, 0.0)
    ok = np.zeros(len(t), bool); ok[:] = False
    out, rep = LP.fill_legs(P, SHAPE, t, np.ones(len(t), bool), UP, GH, dev=DEV)   # 腿全看得见: A 类, 一帧不改
    assert np.array_equal(out, P), rep


def test_short_shot_guessed_crouch_counted_as_seated_still_stands():
    P, t, GH = walker(T=6, v=0.0)
    P = _crouch(P, 0.30)                          # 猜成半蹲又飘着: 竖直跨度小, 逐帧会被当成坐着
    sit = np.r_[np.zeros(5), np.ones(1)].astype(bool)
    out, rep = LP.fill_legs(P, SHAPE, t, np.zeros(len(t), bool), UP, GH, dev=DEV, seated=sit)
    V, _ = M.forward(SHAPE, out, False, DEV)
    assert np.abs(V[-1, :, 1].min() - GH) < 0.03   # 那一帧也站到地上


def _sink(P, d):
    """整个人往下放 d 米(= 猜的脚在地下 d 米, 骨盆位置当成是对的)。"""
    p = M.root_pos(P); p[:, 1] -= d
    return M.set_root(P, None, p)


def _leg_mask():
    m = np.zeros(204, bool); m[LP.LEG_COLS["l"] + LP.LEG_COLS["r"]] = True
    return m


def test_clamp_feet_bends_buried_legs_body_stays_put():
    # jw4 甄子丹坐地上的特写: 腿出画, 猜的脚在地下; 身体是对的(和原图对齐), 不许挪 —— 只把腿弯上来
    P, t, GH = walker(T=8, v=0.0)
    P = _sink(P, 0.25)
    out, rep = LP.clamp_feet(P, SHAPE, np.ones(len(t), bool), UP, GH, dev=DEV)
    assert np.array_equal(out[:, ~_leg_mask()], P[:, ~_leg_mask()])             # 根位置、朝向、上半身一个数不动
    V, _ = M.forward(SHAPE, out, False, DEV)
    low = V[:, :, 1].min(1) - GH
    assert low.min() > -0.03, low                                                 # 脚出了地面
    assert low.max() < 0.05, low                                                  # 不是抬到半空: 刚好落地
    assert np.all(out[:, [LP.KNEE_COL["l"], LP.KNEE_COL["r"]]] > np.radians(20))  # 是弯膝盖做到的
    assert np.abs(np.degrees(out[:, [60, 51, 59, 50]])).max() < 8, np.degrees(out[:, [60, 51, 59, 50]])  # 不是往外劈腿 / 扭大腿
    V0, _ = M.forward(SHAPE, P, False, DEV)
    f0 = V0[:, :, [0, 2]][np.arange(len(t))[:, None], V0[:, :, 1].argsort(1)[:, :50]].mean(1)
    f1 = V[:, :, [0, 2]][np.arange(len(t))[:, None], V[:, :, 1].argsort(1)[:, :50]].mean(1)
    assert np.linalg.norm(f1 - f0, axis=1).max() < 0.12, np.linalg.norm(f1 - f0, axis=1)   # 脚往上收, 不往前后左右甩
    assert rep["改的帧"] == len(t), rep


def test_clamp_feet_leaves_seen_frames_alone():
    P, t, GH = walker(T=8, v=0.0)
    P = _sink(P, 0.25)
    act = np.array([False, False, True, True, True, True, False, False])
    out, rep = LP.clamp_feet(P, SHAPE, act, UP, GH, dev=DEV)
    assert np.array_equal(out[~act], P[~act])                                     # 看得见的 / 补腿摆过的帧不碰
    assert rep["改的帧"] == int(act.sum())


def test_clamp_feet_leaves_feet_above_ground_alone():
    P, t, GH = walker(T=6, v=0.0)
    P = _crouch(P, 0.10)                                                          # 腿是猜的, 但脚不在地下(悬空 10 cm 也不管: 那是贴地的事)
    out, rep = LP.clamp_feet(P, SHAPE, np.ones(len(t), bool), UP, GH, dev=DEV)
    assert np.array_equal(out, P) and rep["改的帧"] == 0, rep


def test_clamp_feet_skips_frames_buried_too_deep():
    # 要抬的比 FOOT_LIFT_MAX 还多: 不是腿的问题, 是地面高 / 位置错了(joker 台阶脚印比平面高 3.4 m), 不拿腿去凑
    P, t, GH = walker(T=6, v=0.0)
    P = _sink(P, 1.2)
    out, rep = LP.clamp_feet(P, SHAPE, np.ones(len(t), bool), UP, GH, dev=DEV)
    assert np.array_equal(out, P), rep
    assert rep["太深不修的帧"] == len(t), rep


def test_clamp_feet_correction_is_smooth_over_time():
    # 埋得深浅逐帧不一样(0 -> 30 cm -> 0): 腿的改动前后连贯, 不一帧一跳
    P, t, GH = walker(T=9, v=0.0)
    d = np.array([0.0, 0.05, 0.12, 0.22, 0.30, 0.22, 0.12, 0.05, 0.0])
    p = M.root_pos(P); p[:, 1] -= d; P = M.set_root(P, None, p)
    out, rep = LP.clamp_feet(P, SHAPE, np.ones(len(t), bool), UP, GH, dev=DEV)
    V, _ = M.forward(SHAPE, out, False, DEV)
    assert (V[:, :, 1].min(1) - GH).min() > -0.03
    k = np.degrees(out[:, LP.KNEE_COL["l"]])
    top = int(np.argmax(d))
    assert np.all(np.diff(k[:top + 1]) >= -0.5) and np.all(np.diff(k[top:]) <= 0.5), k   # 跟着埋的深浅单峰走, 不来回抖
    assert np.abs(np.diff(k)).max() < 30, k


def test_feet_visible_all_the_time_is_left_alone():
    P, t, GH = walker(v=0.0)
    out, rep = LP.fill_legs(P, SHAPE, t, np.ones(len(t), bool), UP, GH, dev=DEV)
    assert rep["类"] == "A" and np.array_equal(out, P), rep


def test_sitting_person_is_left_alone():
    P, t, GH = walker()
    out, rep = LP.fill_legs(P, SHAPE, t, np.zeros(len(t), bool), UP, GH, seated=np.ones(len(t), bool), dev=DEV)
    assert rep["类"] == "坐着" and np.array_equal(out, P), rep


def test_invisible_legs_walk_with_planted_feet_and_nothing_else_moves():
    P, t, GH = walker()
    out, rep = LP.fill_legs(P, SHAPE, t, np.zeros(len(t), bool), UP, GH, dev=DEV)
    assert rep["类"] == "B" and rep["补的帧"] == len(t), rep
    np.testing.assert_array_equal(out[:,LP.BALL_TWIST], P[:,LP.BALL_TWIST])
    assert rep["段"][0]["步数"] >= 4, rep["段"]                     # 4 秒 0.7 m/s 走了 2.8 m
    assert rep["IK_点误差_cm"]["中位"] < 0.5, rep["IK_点误差_cm"]
    legs = LP.LEG_COLS["l"] + LP.LEG_COLS["r"]
    other = np.delete(np.arange(204), legs + [0, 1, 2])
    assert np.abs(out[:, other] - P[:, other]).max() < 1e-9          # 上半身、根的朝向一个数不动
    hor = M.root_pos(out) - M.root_pos(P); hor[:, 1] = 0
    assert np.abs(hor).max() < 1e-9                                  # 根的水平位置不动(只许改高度)
    c = rep["_规划"]["contact"]
    m, _, _ = LP.leg_metrics(out, SHAPE, t, UP, GH, c, dev=DEV)
    m0, _, _ = LP.leg_metrics(P, SHAPE, t, UP, GH, c, dev=DEV)
    for sd in "lr":
        assert m[sd]["滑动_cms"][0] < 2.0 and m0[sd]["滑动_cms"][0] > 50, (m[sd], m0[sd])   # 原来整条腿贴地滑 70 cm/s
        assert abs(m[sd]["踩地帧悬空_cm"][0]) < 1.5, m[sd]
        assert m[sd]["最低_cm"] > -3.0, m[sd]


def test_a_gap_in_the_frames_is_planned_as_two_pieces():
    P, t, GH = walker(T=40, gap_at=20, gap_s=3.0)                    # 人出画 3 秒再回来
    out, rep = LP.fill_legs(P, SHAPE, t, np.zeros(len(t), bool), UP, GH, dev=DEV)
    assert len([s for s in rep["段"] if "步数" in s]) == 2, rep["段"]


def test_visible_feet_that_glide_along_the_floor_are_not_trusted():
    # office r1_sam31 人 2: 脚"看得见", 但两只脚贴着地跟身体一起以 0.23 m/s 滑 -> 当看不见, 交给规划
    P, t, GH = walker(v=0.25)
    out, rep = LP.fill_legs(P, SHAPE, t, np.ones(len(t), bool), UP, GH, dev=DEV)
    assert rep["看得见但脚在滑_当看不见的帧"] > 0 and rep["类"] == "B", rep



def test_planned_sole_clears_one_raised_tread():
    poses, _, _ = walker(T=2, v=0.)
    world, _ = M.forward(SHAPE, poses, True, DEV)
    rest = M.rest_verts(SHAPE, DEV)
    low = rest[:, 1] < rest[:, 1].min() + 10.
    sides = [np.flatnonzero(low & (rest[:, 0] * sign > 0)) for sign in (1, -1)]
    left, right = sides
    floor = world[0, :, 1].min()
    tread = lambda v: np.where(v[:, 0] > 0, floor + .08, floor)
    out, fitted = LP.clear_planned_feet(poses, SHAPE, world, [True, False], sides, UP, tread, DEV)
    assert np.min(fitted[0, left] @ UP - tread(fitted[0, left])) > -.02
    assert np.array_equal(poses[1], out[1])
    assert np.array_equal(poses[:, :50], out[:, :50])
    assert np.allclose(fitted[0, right], world[0, right], atol=1e-5)
    assert np.allclose(fitted, M.forward(SHAPE, out, True, DEV)[0], atol=1e-5)

if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f(); print("ok", name)
