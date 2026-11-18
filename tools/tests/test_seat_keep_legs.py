"""坐着、脚看得到的人(body_into_scene.seat_keep_legs, 2026-11-11 lovelynight 长凳):
腿照原样, 竖直位置按屁股放在椅面上; 抬脚不把人往下拉; 腿比椅面长时脚不扎进地里; 认得出 bench。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_into_scene as B

P = B.load_parts()
N = len(P["lower_w"])
UP = np.array([0.0, 1.0, 0.0])
GH = -0.8
SEAT_H = 0.42


def person(butt, foot, T=40):
    """T 帧的假人: 屁股(seat 顶点)离地 butt[t], 最低的脚离地 foot[t], 其余顶点在 1 m 高。"""
    X = np.zeros((T, N, 3))
    X[:, :, 1] = GH + 1.0
    for t in range(T):
        X[t, P["low"], 1] = GH + foot[t]
        X[t, P["seat"], 1] = GH + butt[t]
    return X


def run(butt, foot, seated=None, delta=None):
    T = len(butt)
    X = person(butt, foot, T)
    t = np.arange(T) / 8.0
    seated = np.ones(T, bool) if seated is None else seated
    delta = np.zeros((T, 3)) if delta is None else delta
    d, diag = B.seat_keep_legs(X, delta, t, np.ones(T), UP, GH, SEAT_H, seated)
    Y = X + d[:, None, :]
    return (Y[:, P["seat"], 1].min(1) - GH), (Y[:, :, 1].min(1) - GH), diag


def test_foot_lift_does_not_pull_person_down():
    # 平时屁股 0.41、脚 0.0; 第 18-19 帧抬脚 17 cm(lovelynight 女的脱鞋), 屁股不动
    T = 40
    butt = np.full(T, 0.41); foot = np.zeros(T); foot[18:20] = 0.17
    sunk = np.zeros((T, 3)); sunk[18:20, 1] = -0.09   # 逐帧贴地已经把这 2 帧整个人拉低 9 cm(r1 实测)
    b, f, diag = run(butt, foot, delta=sunk)
    want = SEAT_H - B.SEAT_SINK
    assert np.abs(b - want).max() < 0.005, (b.min(), b.max())
    assert f[18:20].min() > 0.15, f[18:20]          # 抬起来的脚留在空中, 不是整个人沉下去
    print("抬脚:", diag)


def test_long_legs_do_not_go_through_the_floor():
    # 模型的腿比椅面长 5 cm(屁股放上椅面脚就要扎进地 5 cm): 脚优先, 屁股悬空 5 cm 也不让脚进地
    T = 40
    butt = np.full(T, 0.46); foot = np.zeros(T)      # 屁股到脚落差 0.46 > 椅面 0.42
    b, f, diag = run(butt, foot)
    assert f.min() > -0.005, f.min()
    assert diag["脚扎地_帧数"] == 0
    print("长腿:", diag)


def test_standing_frames_are_left_alone():
    T = 40
    butt = np.full(T, 0.80); foot = np.full(T, 0.05)
    seated = np.zeros(T, bool)
    b, f, _ = run(butt, foot, seated)
    assert np.abs(f - 0.05).max() < 1e-6 and np.abs(b - 0.80).max() < 1e-6


def test_bench_sofa_stool_count_as_seats():
    for n in ["bench_0", "park_bench_3", "wooden_bench", "sofa_1", "couch", "bar_stool_2", "office_chair_11", "chair"]:
        assert B.is_seat(n), n
    for n in ["table_3", "ground", "potted_plant_5", "bar_counter"]:
        assert not B.is_seat(n), n


def _pelvis_track(C):
    """(T,3) 骨盆中心 -> 假的顶点数组: 骨盆那几个顶点都放在中心上。"""
    P = B.load_parts(); Y = np.zeros((len(C), int(np.max(P["pelvis"])) + 1, 3))
    Y[:, P["pelvis"]] = np.asarray(C)[:, None, :]
    return Y


def test_seat_spread_ignores_drift_along_the_camera_ray():
    # jw4 甄子丹坐地上的特写: 推镜头, 人按每帧估的远近摆, 骨盆顺着视线前后漂 0.76 m —— 不是他在动
    UPv = np.array([0, 1.0, 0]); cam = np.array([0, 1.0, 0.0])
    d = np.array([0, -0.3, 1.0]); d /= np.linalg.norm(d)
    C = np.array([cam + d * s for s in np.linspace(1.4, 2.3, 30)])
    assert B.seat_spread(_pelvis_track(C), UPv) > 0.3                                     # 老量法: 当成在动
    assert B.seat_spread(_pelvis_track(C), UPv, cam=np.repeat(cam[None], 30, 0)) < 0.05


def test_seat_spread_still_sees_sideways_movement():
    # 跳舞 / 蹲着打: 横着挪, 和视线垂直 —— 照样算在动
    UPv = np.array([0, 1.0, 0]); cam = np.array([0, 1.0, -3.0])
    C = np.array([[x, 0.5, 0.0] for x in np.linspace(-0.8, 0.8, 30)])
    assert B.seat_spread(_pelvis_track(C), UPv, cam=np.repeat(cam[None], 30, 0)) > 0.6


if __name__ == "__main__":
    test_seat_spread_ignores_drift_along_the_camera_ray()
    test_seat_spread_still_sees_sideways_movement()
    test_foot_lift_does_not_pull_person_down()
    test_long_legs_do_not_go_through_the_floor()
    test_standing_frames_are_left_alone()
    test_bench_sofa_stool_count_as_seats()
    print("== test_seat_keep_legs 全过 ==")
