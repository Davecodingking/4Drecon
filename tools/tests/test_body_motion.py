#!/usr/bin/env python
"""pipeline/body_motion.py: 摆人那步 —— 在 SAM 3D Body 的关节数据上修人, 只算一遍(物理、网页、FBX 都读它的结果)。
要 MHR 模型文件在盘上。跑: $PY tools/tests/test_body_motion.py"""
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
sys.path.insert(0, os.path.expanduser("~/sam3d/data/outdoor/scripts"))
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
    """姿势平滑作用在关节上: 手臂快速挥(手约 6 m/s)最多落后 6 cm; 站着不动的腿上的抖动去掉大半。
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
    assert lag(Vo) < 0.06, lag(Vo)                                           # 实测 5.1 cm(10-09 之前直接在网格顶点上平滑是 5.8 cm)
    jit = lambda V: np.linalg.norm(V[:, shin] - Vc[:, shin], axis=2)[5:-5].mean()
    assert jit(Vo) < 0.6 * jit(Vn), (jit(Vo), jit(Vn))                       # 小腿上的抖动去掉 40% 以上


def test_root_orientation_jitter_is_smoothed_and_a_slow_turn_is_kept():
    """身体朝向一帧一晃(±2°)去掉一半以上; 慢慢转身(每秒 17°)原样留着。位置不归这里管。"""
    rng = np.random.default_rng(1)
    clean = Rotation.from_euler("y", (0.3 * t)[:, None]).as_matrix()
    noisy = (Rotation.from_matrix(clean) * Rotation.from_rotvec(rng.normal(0, np.radians(2), (T, 3)))).as_matrix()
    err = lambda Q: np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t", Q, clean) - 1) / 2, -1, 1)))[5:-5].mean()
    assert err(BM.smooth_rots(noisy, t)) < 0.5 * err(noisy), (err(BM.smooth_rots(noisy, t)), err(noisy))
    assert err(BM.smooth_rots(clean, t)) < 0.05


def test_tracks_shorter_than_five_frames_are_not_smoothed():
    rng = np.random.default_rng(2)
    pose = rng.normal(0, 0.1, (4, 130)); Q = Rotation.from_rotvec(rng.normal(0, 0.3, (4, 3))).as_matrix()
    assert np.array_equal(BM.smooth_joints(pose, t[:4], None), pose) and np.array_equal(BM.smooth_rots(Q, t[:4]), Q)


def _seat_case(T=9, seed=3):
    """一个坐着的人: 骨盆朝向每帧乱晃(±15°), 腿和上身每帧都不一样。"""
    rng = np.random.default_rng(seed)
    pose = rng.normal(0, 0.15, (T, 130))
    Q = (Rotation.from_euler("y", rng.normal(0.4, 0.1, (T, 1))) * Rotation.from_rotvec(rng.normal(0, 0.15, (T, 3)))).as_matrix()
    bis = list(range(10, 10 + T))
    yaw = rng.normal(0, 0.2, T); yaw[4] = 0.0
    fix = dict(ref=bis[4], freeze=False, sigma=1e-3, up=[0.0, 1.0, 0.0], seat=None,
               yaw={str(b): float(y) for b, y in zip(bis, yaw)}, pos={str(b): [0.1 * i, 0.5, 2.0] for i, b in enumerate(bis)})
    return pose, Q, bis, fix, yaw, rng.normal(0, 0.5, 45)


def _global_rots(shape, pose, Q, joints):
    P0 = np.zeros((len(pose), 204)); P0[:, 6:136] = pose
    S = M.forward(shape, M.set_root(P0, Q, np.zeros((len(pose), 3))), False)[1]
    jn = [str(x) for x in M.skeleton()["names"]]
    return {j: Rotation.from_quat(S[:, jn.index(j), 3:7]).as_matrix() for j in joints}


def test_bone_lengths_are_locked_to_the_median_of_reliable_frames():
    """管长度的参数: 每人全程一个值 = 可信帧的中位数。不可信的帧(被翻转补帧换掉的)不参与取值, 但一样改成这个值。别的参数一个不动。"""
    rng = np.random.default_rng(0); n, D = 9, 20; cols = np.array([3, 7, 11])
    pose = rng.normal(0, 1, (n, D)); reliable = np.ones(n, bool); reliable[[2, 5]] = False
    pose[2, cols], pose[5, cols] = 50.0, -50.0                                  # 不可信的两帧: 离谱的长度
    keep = pose.copy()
    out = BM.fix_bone_lengths(pose, reliable, cols)
    assert np.allclose(out[:, cols], np.median(keep[reliable][:, cols], axis=0))  # 每一帧都是同一个值, 没被那两帧带偏
    others = np.setdiff1d(np.arange(D), cols)
    assert np.array_equal(out[:, others], keep[:, others]) and np.array_equal(pose, keep)   # 别的参数没动; 传进来的数组也没被改


def test_bone_lengths_use_all_frames_when_none_is_reliable():
    """整段都是补出来的(一帧可信的都没有): 用全部帧取中位数, 不留着不管。"""
    rng = np.random.default_rng(1); pose = rng.normal(0, 1, (6, 10)); cols = np.array([1, 4])
    out = BM.fix_bone_lengths(pose, np.zeros(6, bool), cols)
    assert np.allclose(out[:, cols], np.median(pose[:, cols], axis=0))


def test_locked_lengths_make_every_bone_constant_in_the_real_skeleton():
    """拿真的 MHR 骨架验: 长度参数每帧不一样时, 骨头(关节到父关节的距离)跟着变; 锁住之后, 人怎么动每根骨头都一样长。
    也就是说 length_params() 找全了 —— 没有别的姿势参数还在偷偷改骨头长度。"""
    rng = np.random.default_rng(2); n = 12
    P = np.zeros((n, 204)); P[:, 6:136] = rng.normal(0, 0.3, (n, 130))            # 每帧一个随便的姿势
    cols = M.length_params()
    P[:, cols] = rng.normal(0, 0.5, (n, len(cols)))                              # 长度每帧都不一样
    par = M.skeleton()["parents"]; kid = np.where(par >= 0)[0]
    def bones(params):
        _, sk = M.forward(np.zeros(45), params, False)
        return np.linalg.norm(sk[:, kid, :3] - sk[:, par[kid], :3], axis=2)       # (帧, 骨头) 厘米
    assert np.ptp(bones(P), axis=0).max() > 0.5                                   # 修之前: 有骨头全程差出半厘米以上
    Q = P.copy(); Q[:, 6:136] = BM.fix_bone_lengths(P[:, 6:136], np.ones(n, bool), cols - 6)
    assert np.ptp(bones(Q), axis=0).max() < 1e-3, np.ptp(bones(Q), axis=0).max()  # 修之后: 每根骨头全程一样长


def test_person_merged_from_two_ids_gets_one_set_of_bone_scales():
    """核对身份把两个编号并成一个人: 上游给每个编号各锁了一套骨骼缩放, 并起来之后前后两段骨头不一样长。
    同一条规则(每人全程一个值)用在骨骼缩放上: 全程一样长, 用的是占多数的那个编号的值。"""
    rng = np.random.default_rng(3); n = 12
    P = np.zeros((n, 204)); P[:, 6:136] = rng.normal(0, 0.3, (n, 130)); P[:, M.length_params()] = 0.0
    P[:8, 136:], P[8:, 136:] = rng.normal(0, 0.2, 68), rng.normal(0, 0.2, 68)       # 前 8 帧来自一个编号, 后 4 帧来自另一个
    par = M.skeleton()["parents"]; kid = np.array([j for j in range(len(par)) if par[j] >= 0 and par[par[j]] >= 0])
    def bones(params):
        _, sk = M.forward(np.zeros(45), params, False)
        return np.linalg.norm(sk[:, kid, :3] - sk[:, par[kid], :3], axis=2)
    assert np.ptp(bones(P), axis=0).max() > 0.5
    Q = P.copy(); Q[:, 136:] = BM.fix_bone_lengths(P[:, 136:], np.ones(n, bool), np.arange(68))
    assert np.ptp(bones(Q), axis=0).max() < 1e-3 and np.allclose(Q[:, 136:], P[0, 136:])   # 全程一样长; 值 = 多数那一段的


def test_seat_fix_lower_body_is_the_reference_frame_turned_about_up():
    """坐姿修正搬到关节上: 下半身(根 + 两条腿, 连腿长/髋宽)整段是参考帧的, 只绕竖轴转 fix 里那个角;
    上身在世界里的朝向不跟着骨盆变 —— 根换了朝向, 差的那点由腰补上(老办法是上身顶点原样留着)。"""
    pose, Q, bis, fix, yaw, shape = _seat_case()
    out, Qn = BM.seat_fix_params(pose, Q, bis, fix)
    r = 4
    want_root = np.stack([Rotation.from_euler("y", a).as_matrix() @ Q[r] for a in yaw])
    assert np.abs(Qn - want_root).max() < 1e-9
    for name in ("l_upleg_ry", "r_knee_bend", "l_foot_bend", "leg_length_flexible", "hip_width_flexible"):
        assert np.abs(out[:, COL(name)] - pose[r, COL(name)]).max() < 1e-12, name
    js = ("c_spine0", "c_spine3", "c_head", "l_wrist", "r_lowleg")
    g0, g1 = _global_rots(shape, pose, Q, js), _global_rots(shape, out, Qn, js)
    deg = lambda A, B_: np.degrees(Rotation.from_matrix(np.einsum("tji,tjk->tik", A, B_)).magnitude()).max()
    for j in ("c_spine0", "c_spine3", "c_head", "l_wrist"):                 # 上身: 和修之前一样(sigma 取得极小 = 不平滑)
        assert deg(g0[j], g1[j]) < 0.01, (j, deg(g0[j], g1[j]))
    turned = np.stack([Rotation.from_euler("y", a).as_matrix() @ g0["r_lowleg"][r] for a in yaw])
    assert deg(turned, g1["r_lowleg"]) < 0.01                               # 小腿: 参考帧的, 绕竖轴转
    # 腰以上的关节角原样(只有腰那三个角被拿去补)
    assert np.abs(out[:, COL("l_elbow_bend")] - pose[:, COL("l_elbow_bend")]).max() < 1e-9
    assert np.abs(out[:, COL("spine1_ry_flexible")] - pose[:, COL("spine1_ry_flexible")]).max() < 1e-9


def test_seat_fix_upper_body_is_smoothed_over_neighbouring_frames():
    """上身每帧的动作保留, 但按相邻帧做一次时间平滑(和老办法同一个 sigma: 按 body4d 帧号算的 1 帧)。"""
    pose, Q, bis, fix, yaw, shape = _seat_case()
    fix = dict(fix, sigma=1.0)
    pose[:, COL("l_elbow_bend")] = 0.0; pose[4, COL("l_elbow_bend")] = 1.0   # 一帧的尖
    out, _ = BM.seat_fix_params(pose, Q, bis, fix)
    w = np.exp(-0.5 * np.arange(-3, 4.0) ** 2); w /= w.sum()
    assert np.allclose(out[3:6, COL("l_elbow_bend")], w[2:5], atol=1e-9), out[:, COL("l_elbow_bend")]   # 前后各 3 帧都在的那几帧
    assert 0 < out[1, COL("l_elbow_bend")] < 0.01 and out[0, COL("l_elbow_bend")] == 0                # 3 帧以外不沾


def test_seat_fix_frozen_person_is_the_reference_frame_throughout():
    """骨盆朝向估计整段都不可信(freeze)的人: 整个人定成参考帧, 不转。"""
    pose, Q, bis, fix, yaw, shape = _seat_case()
    out, Qn = BM.seat_fix_params(pose, Q, bis, dict(fix, freeze=True))
    assert np.abs(out - pose[4]).max() < 1e-12 and np.abs(Qn - Q[4]).max() < 1e-12


def test_seat_fix_position_follows_the_same_contact_rule():
    """位置: 水平按 fix["pos"]; 竖直量的是修完坐姿之后的网格 —— 没对上椅子的最低点踩在地上,
    对上椅子的屁股压进椅面 1 cm。(物理之后椅子挪了、人跟着挪的那一下不在这里: 见 test_people_store.py。)"""
    pose, Q, bis, fix, yaw, shape = _seat_case()
    P = B.load_parts(); UP = np.array([0.0, 1.0, 0.0]); GH = 0.2
    out, Qn = BM.seat_fix_params(pose, Q, bis, fix)
    P0 = np.zeros((len(bis), 204)); P0[:, 6:136] = out
    P0 = M.set_root(P0, Qn, np.zeros((len(bis), 3)))
    W = M.forward(shape, P0, True)[0]
    def placed(fx):
        return M.forward(shape, M.set_root(P0, None, BM.seat_root_pos(M.root_pos(P0), W, bis, fx, GH)), True)[0]
    V = placed(fix)                                                          # 没对上椅子
    want = np.array([fix["pos"][str(b)] for b in bis])
    assert np.abs(V[:, P["pelvis"]].mean(1)[:, [0, 2]] - want[:, [0, 2]]).max() < 1e-4
    assert np.abs((V @ UP).min(1) - GH).max() < 1e-4, (V @ UP).min(1) - GH    # 每一帧最低点都在地面上
    V = placed(dict(fix, seat=dict(chair="chair_1", 椅面离地=0.43)))
    assert np.abs((V @ UP)[:, P["seat"]].min(1) - (GH + 0.43 - B.SEAT_SINK)).max() < 1e-4
    assert np.abs(V[:, P["pelvis"]].mean(1)[:, [0, 2]] - want[:, [0, 2]]).max() < 1e-4


def test_seat_fix_refuses_frames_it_has_no_decision_for():
    """决定里缺了这个人某一帧的转角: 直接失败, 不悄悄跳过那一帧。"""
    pose, Q, bis, fix, yaw, shape = _seat_case()
    del fix["yaw"][str(bis[-1])]
    try:
        BM.seat_fix_params(pose, Q, bis, fix); assert False
    except SystemExit as e:
        assert "没有这几帧的决定" in str(e), e


# ---------------------------------------------------------------- 整个 solve(): 两个造出来的人
UP = np.array([0.0, -1.0, 0.0])            # 相机不动、平视: 世界 = 相机系(OpenCV, y 朝下), 天是 -y
GH = -1.5                                  # 相机离地 1.5 m
NS = 24                                    # 帧数; 视频帧号 = body4d 帧号
LEN = M.length_params()                    # 管骨头长度的 6 个姿势参数


def _scene():
    cams = {f"{i:04d}.jpg": {"extrinsic_w2c_4x4": np.eye(4).tolist()} for i in range(NS)}
    return cams, {i: f"{i:04d}.jpg" for i in range(NS)}, list(range(NS))


def _people(seed=5):
    """人 1 站着从左往右走; 人 2 坐着不动(大腿抬平、小腿垂下: 竖直跨度 1.36 m, 屁股到脚 0.42 m)。
    两个人的骨头长度每帧都不一样(模型逐帧各估各的), 关节上有一点抖动。"""
    rng = np.random.default_rng(seed)
    par = {"1": {}, "2": {}}
    for b in range(NS):
        for pid in par:
            P0 = np.zeros(204)
            P0[6:136] = rng.normal(0, np.radians(1), 130)
            P0[LEN] = rng.normal(0, 0.15, len(LEN)); P0[136:] = rng.normal(0, 0.02, 68)
            if pid == "2":
                for s in "lr":
                    P0[NAMES.index(f"{s}_upleg_rz")] += -1.5; P0[NAMES.index(f"{s}_knee_bend")] += 1.5
            ct = np.array([-1.0 + 0.05 * b, 1.5, 6.0]) if pid == "1" else np.array([2.0, 1.5 - 0.44, 5.0])
            par[pid][b] = (P0, ct)
    return par, {"1": np.zeros(45), "2": np.zeros(45)}


def _chair_under(par, shape):
    """一把椅子正放在人 2 屁股底下(glTF 系: 天 +y), 椅面离地 0.43 m。返回 (load_chairs 那种, chair_seats 那种)。"""
    from world_frame import export_T
    FL = export_T(UP)
    P0, ct = par["2"][0]
    V = M.forward(shape["2"], M.to_world(P0[None], ct[None], np.eye(4)[None]), True)[0][0]
    hip = (V[B.load_parts()["pelvis"]].mean(0) @ FL[:3, :3].T + FL[:3, 3])[[0, 2]]
    return {"chair_1": (hip, np.eye(2), np.array([0.25, 0.25]))}, {"chair_1": (0.43, hip, np.array([hip[0], 0.3, hip[1]]))}


def _solve(chairs=None, seats=None):
    cams, vid2key, kept = _scene()
    par, shape = _people()
    if chairs == "under":
        chairs, seats = _chair_under(par, shape)
    res, info, rep = BM.solve(par, shape, cams, vid2key, kept, UP, GH, "/没有这个目录", "/没有这个目录", 24.0, None,
                              chairs=chairs, seats=seats, seat_h_default=0.43)
    return par, res, info, rep


def test_solve_locks_bone_lengths_and_puts_feet_on_the_ground():
    par, res, info, rep = _solve()
    for pid in ("1", "2"):
        P = res[pid]["params"]
        assert np.ptp(P[:, LEN], axis=0).max() < 1e-12 and np.ptp(P[:, 136:], axis=0).max() < 1e-12    # 每人全程一个值
        raw = np.stack([par[pid][b][0] for b in range(NS)])
        assert np.allclose(P[0, LEN], np.median(raw[:, LEN], 0)) and np.ptp(raw[:, LEN], axis=0).min() > 0.1
    V = M.forward(res["1"]["shape"], res["1"]["params"], True)[0]
    feet = (V @ UP).min(1) - GH
    assert np.abs(feet).max() < 0.02, feet                                                 # 走路的人脚在地上
    assert rep["1"]["可疑"] == [] and rep["1"]["锁骨头长度_表面挪动_厘米"][1] > 0.5


def test_solve_tells_physics_who_walks_where_and_who_sits():
    """物理读的两样: 站/走着的人每帧一个脚印(路障); 坐着的人没有脚印。场景里没有椅子: 谁也不钉, 也不做坐姿修正。"""
    _, res, info, rep = _solve()
    assert info["1"]["seated"] is False and info["1"]["chair"] is None and len(info["1"]["footprints"]) == NS
    assert [f["vf"] for f in info["1"]["footprints"]] == list(range(NS))
    assert info["2"]["seated"] is True and info["2"]["footprints"] == [] and info["2"]["chair"] is None
    assert "seat_fix" not in info["2"] and "坐姿修正" not in rep["2"]
    assert rep["2"]["坐着的帧"] == NS and rep["1"]["坐着的帧"] == 0 and rep["1"]["路线脚印"] == NS


def test_solve_seats_the_sitting_person_on_the_chair_under_them():
    """有椅子: 坐着的人对上屁股底下那把(物理要钉住它); 坐姿是在锁好骨头的身体上定的, 定完直接改进参数 ——
    交付的参数生成的网格, 屁股正好压进椅面 1 cm, 下半身整段一个姿势。走路的人不受影响。"""
    _, res0, _, _ = _solve()
    _, res, info, rep = _solve(chairs="under")
    assert info["2"]["chair"] == "chair_1" and info["2"]["seat_fix"]["seat"] == dict(chair="chair_1", 椅面离地=0.43)
    assert info["1"]["chair"] is None and "seat_fix" not in info["1"]
    assert np.array_equal(res["1"]["params"], res0["1"]["params"])                         # 别人的椅子不影响走路的人
    P = res["2"]["params"]
    V = M.forward(res["2"]["shape"], P, True)[0]
    seat = (V @ UP)[:, B.load_parts()["seat"]].min(1) - GH
    assert np.abs(seat - (0.43 - B.SEAT_SINK)).max() < 1e-3, seat
    legs = np.r_[M.limb_groups()["l_upleg"], M.limb_groups()["r_upleg"]]
    assert np.ptp(P[:, legs], axis=0).max() < 1e-9                                         # 腿整段用同一帧的
    assert np.ptp(P[:, LEN], axis=0).max() < 1e-12                                         # 坐姿修正没把锁好的长度弄乱
    assert rep["2"]["坐姿修正"]["高度来源"] == "chair_1" and abs(rep["2"]["坐姿修正"]["屁股到椅面_后"] + B.SEAT_SINK) < 2e-3


def test_suspicious_people_are_named_with_the_reason():
    assert B.suspicious(1.0, 1.0, 1.0, 1.0, 0.3) == []
    why = B.suspicious(0.49, 0.97, 0.97, 0.4, 3.0)                                         # office 那件外套 + 另外两条
    assert len(why) == 3 and "深度离群" in why[0] and "0.49" in why[0] and "40%" in why[1] and "3.0 m" in why[2]


def test_identity_record_lists_where_every_track_id_went():
    """闸口上摆的那张表: 按跟踪编号, 每个编号的帧最后算谁的。这里 3 号在闸口被点掉了(出网格时 5、8 号重排成 3、4)。"""
    names = {"1": 1, "2": 2, "3": 5, "4": 8}
    who = {"1": [("1", b) for b in range(60)], "2": [("2", b) for b in range(30)] + [("4", b) for b in range(40, 60)],
           "5": [("3", b) for b in range(60)]}
    log = [dict(事="并进别人", 编号="8", 帧=list(range(40, 60)), 并进="2", 原因="位置接得上"),
           dict(事="不算人", 编号="5", 帧=[70], 原因="只有 1 帧")]
    rec = BM.identity_record(names, who, log, [3 * b for b in range(80)])
    assert rec["勾的人"] == [1, 2, 5, 8] and rec["人"] == {"1": [1], "2": [2, 8], "5": [5]}
    assert list(rec["去向"]) == ["1", "2", "5", "8"]
    assert rec["去向"]["8"] == [dict(事="并进别人", 人="2", 帧数=20, 视频帧=[120, 177], 原因="位置接得上")]
    assert rec["去向"]["2"] == [dict(事="人", 人="2", 帧数=30, 视频帧=[0, 87])]
    assert [x["事"] for x in rec["去向"]["5"]] == ["人", "不算人"] and rec["去向"]["5"][1]["视频帧"] == [210, 210]


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f(); print("ok", name)
