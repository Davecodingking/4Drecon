#!/usr/bin/env python
"""pipeline/body_motion.py: 在 SAM 3D Body 的一手参数(关节数据)上做修复 —— 不再从网格反推骨骼。
要 MHR 模型文件在盘上。跑: $PY tools/tests/test_body_motion.py"""
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_into_scene as B
import mhr_body as M
import body_motion as BM

T = 96
t = np.arange(T) / 24.0
G = {n: c - 6 for n, c in M.limb_groups().items()}            # 列下标: 关节参数那 130 列里的
NAMES = list(M.model().character_torch.parameter_transform.parameter_names)[:136]
COL = lambda name: NAMES.index(name) - 6


def test_limbs_are_grouped_and_fingers_and_lengths_left_out():
    assert set(G) == {"trunk", "c_neck", "l_clavicle", "r_clavicle", "l_upleg", "r_upleg"}, set(G)
    used = {NAMES[c + 6] for v in G.values() for c in v}
    assert "l_elbow_bend" in used and "r_knee_bend" in used
    assert not [n for n in used if "thumb" in n or "length" in n or "width" in n], used
    assert COL("l_lowarm_twist") in G["l_clavicle"] and COL("l_wrist_ry") in G["l_clavicle"]


def test_two_frame_arm_excursion_is_bridged():
    """nfl1 人 6 左臂第 65~66 帧: 前臂扭转从 25° 跳到 -135°, 手腕反着跳, 两帧后跳回。换成前后的过渡, 别的肢体不动。"""
    pose = np.zeros((T, 130)); pose[:, COL("l_lowarm_twist")] = np.radians(25) + 0.2 * t
    pose[:, COL("r_knee_bend")] = 0.5 * np.sin(2 * t)
    want = pose.copy()
    pose[40:42, COL("l_lowarm_twist")] = np.radians(-135); pose[40:42, COL("l_wrist_ry")] = np.radians(70)
    out, fixed = BM.fix_limb_spikes(pose, t, G)
    assert fixed == {"l_clavicle": [(40, 41)]}, fixed
    assert np.abs(out - want).max() < 1e-9, np.abs(out - want).max()          # 直线上的两帧被原样接回去


def test_real_motion_is_not_touched():
    """真的快动作不动: 一路弯过去(肘 0.1 秒里弯 100°, 1000°/秒)、快速来回挥(0.5 秒一个来回, ±80°)、
    换了姿势之后稳住的(一帧跳 90° 然后一直保持)。
    ★ 前两种试过按"一帧转 30° 以上就断开"判, 会被当成跳变整段换掉(nfl1 人 6 右臂 65~75 帧)。"""
    pose = np.zeros((T, 130))
    pose[:, COL("l_elbow_bend")] = np.radians(100) * np.clip((t - 1.0) / 0.1, 0, 1)
    pose[:, COL("l_uparm_rz")] = np.radians(80) * np.sin(2 * np.pi * t / 0.5)
    pose[50:, COL("r_uparm_ry")] = np.radians(90)                            # 跳过去之后稳住 1.9 秒
    out, fixed = BM.fix_limb_spikes(pose, t, G)
    assert fixed == {} and np.array_equal(out, pose), fixed


def test_track_ends_are_left_alone():
    """片头片尾只有一边有帧: 最后两帧突然弯了 120°, 分不清是毛刺还是动作刚开始 —— 不动。
    (一路挥过去的真动作, 起点离后面几帧的中位数天然很远, 用单边窗口会把片头误判成毛刺。)"""
    pose = np.zeros((T, 130)); pose[-2:, COL("l_knee_bend")] = np.radians(120)
    pose[:, COL("r_uparm_rz")] = np.radians(80) * np.sin(2 * np.pi * t / 0.5)
    out, fixed = BM.fix_limb_spikes(pose, t, G)
    assert fixed == {} and np.array_equal(out, pose), fixed


def test_three_frame_glitch_is_bridged_as_a_whole_limb():
    """nfl1 人 6 左臂 103~105 帧: 肘从 +60° 变成 -25°(反着弯)、上臂同时拧了 60°, 三帧后回来。
    肘在第一帧只偏了 40°(不到门槛), 但上臂那个角超了 —— 一条肢体里有一个角是毛刺, 这一帧整条肢体都换。"""
    pose = np.zeros((T, 130)); pose[:, COL("l_elbow_bend")] = np.radians(55); pose[:, COL("l_uparm_twist")] = np.radians(15)
    want = pose.copy()
    pose[40:43, COL("l_elbow_bend")] = np.radians([15, -18, -25]); pose[40:43, COL("l_uparm_twist")] = np.radians([-51, -46, 11])
    out, fixed = BM.fix_limb_spikes(pose, t, G)
    assert fixed == {"l_clavicle": [(40, 42)]}, fixed
    assert np.abs(out - want).max() < 1e-9


def test_low_fps_one_frame_excursion():
    """office 约 6.7 fps(一帧 0.15 秒): 单独一帧跳出去的换掉; 稳住两帧(0.3 秒)的当真。"""
    t6 = np.arange(40) / 6.66
    pose = np.zeros((40, 130)); pose[20, COL("r_elbow_bend")] = np.radians(150)
    pose[30:32, COL("l_elbow_bend")] = np.radians(150)
    out, fixed = BM.fix_limb_spikes(pose, t6, G)
    assert fixed == {"r_clavicle": [(20, 20)]}, fixed
    assert np.abs(out[:, COL("r_elbow_bend")]).max() < 1e-9 and out[30, COL("l_elbow_bend")] > 2.0


def test_whole_body_flip_fill_on_parameters():
    """翻转补帧作用在参数上: 两头有好帧 -> 关节线性过渡、根朝向球面插值; 只有一头 -> 关节和朝向都定住。位置用计划里的。"""
    bis = list(range(6))
    pose = np.arange(6)[:, None] * np.ones((6, 130)) * 0.1
    Q = Rotation.from_euler("y", np.arange(6)[:, None] * 30.0, degrees=True).as_matrix(); p = np.zeros((6, 3))
    occ = {"2": [1, 4, 1 / 3, [1.0, 0.0, 0.0]], "3": [1, 4, 2 / 3, [2.0, 0.0, 0.0]], "5": [4, 4, 0.0, [9.0, 0.0, 0.0]]}
    pose[2:4] = 99; Q[2:4] = np.eye(3); pose[5] = 99; Q[5] = np.eye(3)       # 坏帧里的数不该被用到
    po, Qo, pp = BM.fill_flips(pose, Q, p, bis, occ)
    assert np.allclose(po[2], 0.1 + 0.1) and np.allclose(po[3], 0.1 + 0.2), (po[2, 0], po[3, 0])
    ang = [np.degrees(Rotation.from_matrix(Qo[i]).magnitude()) for i in (2, 3, 5)]
    assert np.allclose(ang, [60, 90, 120], atol=1e-6), ang                   # 30° -> 120° 之间的 1/3、2/3; 片尾定在 120°
    assert np.allclose(po[5], 0.4) and np.allclose(pp[[2, 3, 5], 0], [1, 2, 9])


def test_fast_arm_is_kept_and_slow_jitter_removed():
    """姿势平滑作用在关节上: 手臂快速挥(手约 6 m/s)保得不比在网格上平滑差; 站着不动的腿上的抖动去掉大半。
    "快不快"看的是这个关节带动的肢体末端有多快, 不是关节自己的转速(那样肩、肘各被平滑一半, 手会落后十几厘米)。"""
    rng = np.random.default_rng(0)
    shape = np.zeros(45); scales = np.zeros(68)
    clean = np.zeros((T, 130))
    swing = np.radians(50) * np.sin(2 * np.pi * t / 0.6)                     # 0.6 秒一个来回
    clean[:, COL("l_uparm_rz")] = swing; clean[:, COL("l_elbow_bend")] = np.radians(40) + 0.5 * swing
    noisy = clean.copy(); legs = np.r_[G["l_upleg"], G["r_upleg"]]
    noisy[:, legs] += rng.normal(0, np.radians(3), (T, len(legs)))           # 腿上每帧 ±3° 的抖动
    out = BM.smooth_joints(noisy, t, BM.limb_speed_fn(shape, scales, t))

    def verts(ps):
        P0 = np.zeros((T, 204)); P0[:, 6:136] = ps
        return M.forward(shape, P0, False)[0]
    Vc, Vn, Vo = verts(clean), verts(noisy), verts(out)
    Vm = np.stack(B.smooth_pose(list(Vn), t))                                # 老办法: 在网格上平滑, 当参照
    sk = M.skeleton(); jn = [str(x) for x in sk["names"]]
    dom = np.zeros(Vc.shape[1], int); bw = np.zeros(Vc.shape[1])
    for v, j, w in zip(sk["sk_v"], sk["sk_j"], sk["sk_w"]):
        if w > bw[v]:
            bw[v], dom[v] = w, j
    hand, shin = dom == jn.index("l_wrist"), dom == jn.index("r_lowleg")
    assert hand.sum() > 50 and shin.sum() > 50
    sp = np.linalg.norm(np.diff(Vc[:, hand], axis=0), axis=2).max() * 24
    assert sp > 3.0, sp                                                      # 这个测试里的手确实算快
    lag = lambda V: np.linalg.norm(V[:, hand] - Vc[:, hand], axis=2)[5:-5].max()
    assert lag(Vo) < 0.06 and lag(Vo) <= 1.05 * lag(Vm), (lag(Vo), lag(Vm))  # 实测 5.1 cm, 网格上平滑是 5.8 cm
    jit = lambda V: np.linalg.norm(V[:, shin] - Vc[:, shin], axis=2)[5:-5].mean()
    assert jit(Vo) < 0.6 * jit(Vn), (jit(Vo), jit(Vn))                       # 小腿上的抖动去掉 40% 以上


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f(); print("ok", name)
