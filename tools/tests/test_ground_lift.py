"""地面硬约束(body_motion.ground_lift, 2026-11-11 hitch 13 秒坏相机的帧沉进地里 30~51 cm):
在地下的抬上来; 腾空(跳)不压; 坏相机的帧上下都贴地; 容差以内不动。
2026-11-12 jw4: 脚不可信(看不到、也不是补腿摆在地上的)的帧不按脚抬、不贴地, 只保证小腿和脚以外的身体不在地下。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_motion as BM


def test_below_ground_is_lifted():
    clear = np.array([0.0, -0.30, -0.51, -0.01, 0.0])
    lift = BM.ground_lift(clear)
    assert np.allclose(lift, [0, 0.30, 0.51, 0, 0])          # -1 cm 在容差内不动


def test_airborne_is_not_pushed_down():
    clear = np.array([0.0, 0.25, 0.40, 0.10, 0.0])            # 跳起来
    assert np.allclose(BM.ground_lift(clear), 0)


def test_unseen_feet_do_not_lift_the_body():
    # jw4 甄子丹坐地上的特写(2026-11-12): 腿出画, 模型猜的脚在地下 6~54 cm, 骨盆的位置是对的(和原图对齐)。
    # 按猜的脚抬 = 每帧抬得不一样(上下跳) + 人投到画面上方。脚不可信的帧不按脚抬。
    clear = np.array([-0.30, -0.06, -0.54, -0.30])
    seen = np.array([False, False, False, True])
    body = np.array([0.05, 0.10, 0.02, 0.40])
    assert np.allclose(BM.ground_lift(clear, seen=seen, body_clear=body), [0, 0, 0, 0.30])


def test_unseen_feet_butt_still_kept_above_ground():
    # 脚不可信也不许身上(小腿和脚以外)在地下: 按身体抬, 抬到贴地为止(jw1 倒地的人头 / 背进地)
    clear = np.array([-0.50, -0.40])
    seen = np.array([False, False])
    body = np.array([-0.12, -0.01])            # -1 cm 在容差内不动
    assert np.allclose(BM.ground_lift(clear, seen=seen, body_clear=body), [0.12, 0])


def test_unseen_feet_on_bad_camera_not_snapped():
    # 坏相机 + 脚不可信: 不再上下贴地(贴的是猜的脚), 只管身体不进地
    clear = np.array([0.30, -0.45, -0.45])
    bad = np.array([True, True, True])
    seen = np.array([False, False, True])
    body = np.array([0.50, 0.10, 0.30])
    assert np.allclose(BM.ground_lift(clear, bad, seen=seen, body_clear=body), [0, 0, 0.45])


def test_lower_leg_mask_is_shins_and_feet_only():
    import mhr_body as M
    m = BM.lower_leg_mask("cpu")
    V, _ = M.forward(np.zeros(M.NSHAPE), np.zeros((1, 204)), False, "cpu")
    y = V[0, :, 1]
    assert 0.05 < m.mean() < 0.3, m.mean()
    assert y[m].max() < 0.6 and y[~m].min() > 0.3, (y[m].max(), y[~m].min())   # 小腿和脚在膝盖以下, 别的都在上面


def test_no_seen_given_is_the_old_rule():
    clear = np.array([0.0, -0.45, 0.30, -0.30])
    bad = np.array([False, True, True, False])
    assert np.allclose(BM.ground_lift(clear, bad), [0, 0.45, -0.30, 0.30])


def test_bad_camera_frames_snap_both_ways():
    clear = np.array([0.0, -0.45, 0.30, 0.0])
    bad = np.array([False, True, True, False])
    assert np.allclose(BM.ground_lift(clear, bad), [0, 0.45, -0.30, 0])


if __name__ == "__main__":
    for f in [test_below_ground_is_lifted, test_airborne_is_not_pushed_down, test_bad_camera_frames_snap_both_ways,
              test_unseen_feet_do_not_lift_the_body, test_unseen_feet_butt_still_kept_above_ground,
              test_unseen_feet_on_bad_camera_not_snapped, test_no_seen_given_is_the_old_rule, test_lower_leg_mask_is_shins_and_feet_only]:
        f(); print("ok", f.__name__)
    print("== test_ground_lift 全过 ==")
