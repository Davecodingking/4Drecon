#!/usr/bin/env python
"""摆人(流水线的 people 步, 物理之前): 把 SAM 3D Body 出的人修好、放进重建的场景里。修在模型的**关节数据**上, 只算这一遍。

  SAM 3 跟踪 + SAM 3D Body  ->  每人每帧一组关节参数(相机系, 真米), 存在 <出网格目录>/mhr_4d_individual/(见 mhr_body.py)
  重建流水线                ->  每帧相机在世界里的位置和朝向、场景深度
  这个脚本                  ->  先核对身份(people_identity.py), 再逐人修: 得到世界系的参数 —— 喂给 MHR 就是网格和骨架

★ 修什么(每一项改的是什么):
    远近 / 位置 / 贴地      根关节的位置(远近按每一帧的场景深度, 不信模型按"人在画面里多大"猜的距离)
    认错人的帧(翻转补帧)    关节参数插值, 根朝向球面插值; 只有一头有好帧就定住
    骨头长度                姿势参数里管长度的 6 个 + 骨骼缩放 68 个, 每人全程一个值
    肢体跳变                单条手臂/腿的关节角跳出去又跳回来的几帧, 换成前后的过渡
    姿势抖动                每个关节参数按时间平滑, 根朝向按四元数平滑, 快动作少平滑
    坐姿(有椅子的场景)      腿的关节用稳定帧的, 根只绕竖轴转, 上身朝向由腰补回来; 屁股放在椅面上
  判据(哪几帧算认错人、远近怎么定、位置怎么整理、谁算坐着、坐哪把椅子)都是 body_into_scene.py 里的函数。

★ 为什么只在这里修(2026-10-09): 10-08 之前修在网格上(body_into_scene.solve_people), 要骨骼动画只能从网格反推;
  10-08 加了这条在关节数据上修的路给 FBX 用, 但网页验收视频、物理、3D 页还读网格那条 —— 同一个人修了两遍,
  后加的两项(骨头长度、肢体跳变)只有这边有: 网页上看到的人和交付的 FBX 身体表面平均差 1~3 cm, 个别帧几十厘米
  (nfl1 人 6 手脚跳变那几帧 69 cm)。现在网格那条删了, 所有下游读这里的结果(存取见 people_store.py)。

用法:
  python body_motion.py --r1 <场景结果目录> --body <body4d 轮次目录> [--scene-in 07_scene.glb] [--seat-fix off]
产物(<R1>/ 下, 格式见 people_store.py):
  10_people.json  10_people_params.npz  10_people_world.npz  10_people_body.npz;  <body>/06_identity/(核对过身份的人体数据)
sam3d 环境(torch + scipy + trimesh)。有 GPU 就用(流水线在这一步外面领卡)。
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/sam3d/data/outdoor/scripts"))
import body_into_scene as B
import mhr_body as M
import people_identity, people_store
from world_frame import export_T

POSE = slice(6, M.NPOSE)          # 各关节(含手)的参数; [0:6] 是根


def load_params(mesh_run):
    """{人: {帧: (params 204, cam_t 3)}}, {人: 体型 45}"""
    root = f"{mesh_run}/mhr_4d_individual"
    if not os.path.isdir(root):
        raise SystemExit(f"[失败] {root} 不存在 —— 这一轮出网格时还没存关节数据(2026-10-08 以后才存), 重跑 people_mesh")
    par, shape = {}, {}
    for pid in sorted(os.listdir(root), key=int):
        fr, sh = {}, []
        for f in sorted(os.listdir(f"{root}/{pid}")):
            z = np.load(f"{root}/{pid}/{f}")
            fr[int(f[:-4])] = (z["mhr_model_params"].astype(np.float64), z["pred_cam_t"].astype(np.float64))
            sh.append(z["shape_params"])
        if fr:
            par[pid] = fr
            shape[pid] = np.median(np.stack(sh), 0).astype(np.float64)       # 上游已经把体型定成同一个, 取中位只是防万一
    return par, shape


def _near_max(x, t, half):
    """每帧取前后 half 秒里最大的: 甩到最远那一刻速度是 0, 只看自己会被当成慢的。x (T,...)。"""
    return np.stack([x[np.abs(t - ti) <= half].max(0) for ti in t])


def limb_speed_fn(shape, scales, t, dev="cpu"):
    """给 smooth_joints 用: 关节参数 (T,130) -> 每个参数带动的那段肢体里最快的顶点速度 (T,130), m/s。
    速度在根不动时的网格上量(= 骨盆系里的身体)。"""
    t = np.asarray(t, float)

    def f(pose):
        P0 = np.zeros((len(t), M.NPOSE + M.NSCALE)); P0[:, POSE] = pose; P0[:, M.NPOSE:] = scales
        V0, _ = M.forward(shape, P0, False, dev)
        sv = np.linalg.norm(np.diff(V0, axis=0), axis=2) / np.maximum(np.diff(t), 1e-6)[:, None]
        return M.limb_speed(np.maximum(np.r_[sv[:1], sv], np.r_[sv, sv[-1:]]), dev)[:, POSE]
    return f


def smooth_joints(pose, t, limb_speed, sigma=B.POSE_SIGMA, sigma_min=B.POSE_SIGMA_MIN, v0=B.POSE_V0):
    """pose (T,D) 关节参数按时间平滑, 快动作少平滑(思路见 body_into_scene.py "姿势平滑"那节)。
    limb_speed(pose) -> (T,D): 每个参数带动的那段肢体里最快的顶点的速度(m/s), 见 mhr_body.limb_speed。
    ★ "快不快"不能看关节自己的转速: 一次挥臂是肩、肘、腕叠出来的, 每个关节单看都只是中等速度, 各被平滑掉一半,
      沿手臂累加到手上 —— 2026-10-08 nfl1 人 6 第 105 帧左手离模型原始动作 43 cm(直接在网格顶点上平滑是 18 cm)。
      要看的是它带动的肢体末端有多快(米/秒), 阈值 v0。"""
    pose = np.asarray(pose, float); t = np.asarray(t, float); T = len(t)
    if T < 5:
        return pose.copy()
    strong = B._gauss_t(pose, t, np.full(T, sigma))
    sp = limb_speed(strong)                                                  # 速度按强平滑版求, 不然抖动本身被当成"快"
    a = v0 / (v0 + _near_max(sp, t, 2 * sigma))
    return a * strong + (1 - a) * B._gauss_t(pose, t, np.full(T, sigma_min))


def smooth_rots(Q, t, sigma=B.POSE_SIGMA, sigma_min=B.POSE_SIGMA_MIN, w0=B.POSE_W0):
    """Q (T,3,3) 根朝向按四元数平滑, 转得快的帧少平滑。"""
    from scipy.spatial.transform import Rotation
    t = np.asarray(t, float); T = len(t)
    if T < 5:
        return np.array(Q, copy=True)
    dt = np.maximum(np.diff(t), 1e-6)
    q = Rotation.from_matrix(Q).as_quat()
    for i in range(1, T):                                                    # q 和 -q 同一个朝向, 先接成连续的
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    qs = B._gauss_t(q, t, np.full(T, sigma)); qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    w = 2 * np.arccos(np.clip(np.abs((qs[1:] * qs[:-1]).sum(1)), 0, 1)) / dt
    w = _near_max(np.maximum(np.r_[w[:1], w], np.r_[w, w[-1:]]), t, 2 * sigma)
    b = (w0 / (w0 + w))[:, None]
    q = b * qs + (1 - b) * B._gauss_t(q, t, np.full(T, sigma_min))
    return Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()


LIMB_SPIKE_DEG = 45.0    # 度: 一条肢体的关节角偏离"前后一小段的中位数"这么多 = 跳出去的毛刺
LIMB_SPIKE_HALF = 0.13   # 秒: 中位数看前后各这么久(24 fps 是前后各 3 帧 -> 最多认得出连着 3 帧的毛刺); 至少前后各 1 帧


def fix_limb_spikes(pose, t, groups, thr=LIMB_SPIKE_DEG, half=LIMB_SPIKE_HALF):
    """单条手臂/腿的关节角"跳出去又马上跳回来"的那几帧, 换成前后正常帧之间的过渡。pose (T,D); groups {肢体: 列下标}。
    返回 (新的 pose, {肢体: [(i0, i1), ...]})。
    ★ 为什么需要(2026-10-08): SAM 3D Body 逐帧独立估, 前臂拧 160° + 手腕反拧回来 和 不拧, 手的位置差不多 ——
      模型会一帧一个解(nfl1 人 6 左前臂 65~66 帧拧过去又拧回来; 103~105 帧肘反着弯 25°)。网格上几乎看不出来
      (老流程从没碰到), 但直接对关节角做平滑会让手臂从中间"绕"过去, 导出的动作文件里前臂瞬间拧一圈。
    ★ 判据是"偏离前后的中位数", 不是"变得快"(Hampel 滤波的思路): 真的快动作是顺着一个方向走的, 中位数就在它旁边;
      跳出去又回来的毛刺离中位数很远。试过照搬"整个人认错"那条(一帧转 30° 以上就断开、稳不住的段换掉):
      手腕和前臂在激烈动作里真能转这么快, 一段真动作被切成一串"稳不住的小段"整段换掉, 人 6 的手离原动作 1 m。
    ★ 一条肢体里只要有一个角是毛刺, 这条肢体这一帧的角全换 —— 它们是一起跳的(前臂拧过去、手腕反拧), 只换一个会拧坏。"""
    pose = np.array(pose, float, copy=True); t = np.asarray(t, float); T = len(t)
    fixed = {}
    if T < 3:
        return pose, fixed
    half = max(half, 1.01 * float(np.median(np.diff(t))))
    # ★ 窗口两边对称(前后各取一样多的帧): 片头片尾只有一边有帧, 一路挥过去的真动作在那儿天然离"后面几帧的中位数"很远,
    #   分不清是毛刺还是动作刚开始 —— 判不了就不动
    k = [min(int((t[:i] >= t[i] - half).sum()), int((t[i + 1:] <= t[i] + half).sum())) for i in range(T)]
    for name, cols in groups.items():
        x = pose[:, cols]
        med = np.stack([np.median(x[i - k[i]:i + k[i] + 1], 0) for i in range(T)])
        bad = np.degrees(np.abs(x - med)).max(1) > thr
        if not bad.any() or bad.all():
            continue
        good = np.nonzero(~bad)[0]
        idx = np.nonzero(bad)[0]
        for c in cols:                                           # 两头有正常帧: 按时间线性过渡; 只有一头: 定在那一帧(np.interp 的边界行为)
            pose[idx, c] = np.interp(t[idx], t[good], pose[good, c])
        fixed[name] = [(int(r[0]), int(r[-1])) for r in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)]
    return pose, fixed


def fix_bone_lengths(pose, reliable, cols):
    """决定骨头长度的参数(cols): 每人全程定成一个值。用在两处(见 solve):
      - 姿势参数里管长度的那 6 个(脊柱/脖子/手臂/腿的长度、肩宽、胯宽)
      - 骨骼缩放那 68 个
    ★ 为什么: SAM 3D Body 一帧只看一张图, 不知道前后帧是同一个人, 姿势里那 6 个数每帧各估各的(nfl1 人 6 的前臂差到近 10 cm)。
      上游的视频版把体型和骨骼缩放锁成了一个值, 那 6 个在姿势参数里, 漏了。而且它是按"编号"锁的: 一个人由几个编号并成
      (核对身份, 见 people_identity.py)时, 各段的骨骼缩放不一样(office r10 人 2 = 编号 2 + 6 + 10, 脚的骨头前后差 2 cm)。
      仿真和换骨架都要骨头长度固定。体型那 45 个数 load_params 已经按人取了中位, 这里不用管。
    取值 = 模型可信的帧(reliable: 没被翻转补帧换掉的)的中位数 —— 被换掉的帧模型本来就是错的, 不让它们参与;
    一帧可信的都没有(整段都是补的)就用全部帧。别的参数不动。
    pose (T,D) 关节参数; reliable (T,) bool; cols: 这几个参数在 pose 里的列。返回新的 pose。"""
    pose = np.array(pose, float, copy=True); reliable = np.asarray(reliable, bool)
    src = pose[reliable] if reliable.any() else pose
    pose[:, cols] = np.median(src[:, cols], axis=0)
    return pose


def lock_shift(shape, P_before, P_after, dev="cpu"):
    """锁骨头长度让身体表面挪了多少(厘米): 同一帧、根关节不动, 锁之前 vs 锁之后的网格。返回 (平均, 最大)。只给报告用。"""
    Va, _ = M.forward(shape, P_before, False, dev); Vb, _ = M.forward(shape, P_after, False, dev)
    d = np.linalg.norm(np.asarray(Vb) - np.asarray(Va), axis=2) * 100
    return float(d.mean()), float(d.max())


def fill_flips(pose, Q, p, bis, occ):
    """翻转补帧, 作用在参数上。occ: occ_plan 的结果 {帧: [A, B, w, 根关节位置]}。
    两头都有好帧(A != B): 关节参数线性过渡, 根朝向球面插值; 只有一头(A == B): 关节和朝向都定在那个好帧。位置用计划里的。"""
    from scipy.spatial.transform import Rotation
    pose, Q, p = pose.copy(), Q.copy(), p.copy()
    at = {b: i for i, b in enumerate(bis)}
    for k, (A, Bf, w, r) in occ.items():
        i, ia, ib = at[int(k)], at[A], at[Bf]
        pose[i] = (1 - w) * pose[ia] + w * pose[ib]
        rel = Rotation.from_matrix(Q[ia].T @ Q[ib]).as_rotvec()
        Q[i] = Q[ia] @ Rotation.from_rotvec(rel * w).as_matrix()
        p[i] = np.asarray(r)
    return pose, Q, p


_SEAT = {}
def _seat_consts(dev="cpu"):
    """坐姿修正要的骨架常数: 下半身的参数列、腰(c_spine0)三个角各由哪些参数加出来、腰的预旋转。"""
    if dev not in _SEAT:
        from scipy.spatial.transform import Rotation
        sk = M.skeleton(dev); names = [str(x) for x in sk["names"]]; par = sk["parents"].astype(int)
        PT = M.model(dev).character_torch.parameter_transform.parameter_transform.cpu().numpy()[:, POSE]
        leg = np.array([n in ("l_upleg", "r_upleg") for n in names])
        for j in range(1, len(names)):                       # 父关节的编号都比子关节小: 顺着往下把两条腿整棵标出来
            leg[j] |= leg[par[j]]
        lower = np.array([c for c in range(PT.shape[1]) if len(np.nonzero(PT[:, c])[0])
                          and leg[np.nonzero(PT[:, c])[0] // 7].all()])              # 只驱动腿上关节的参数(连腿长、髋宽)
        r = names.index("c_spine0") * 7
        A = PT[r + 3:r + 6]                                  # (3,130): 腰的 rx/ry/rz = A @ 关节参数
        flex = np.array([int(np.nonzero((A[k] == 1.0) & (np.count_nonzero(PT, axis=0) == 1))[0][0]) for k in range(3)])
        S = M.forward(np.zeros(M.NSHAPE), np.zeros((1, M.NPOSE + M.NSCALE)), False, dev)[1][0]
        G = lambda j: Rotation.from_quat(S[names.index(j), 3:7]).as_matrix()
        _SEAT[dev] = dict(lower=lower, A=A, flex=flex, pre=G("root").T @ G("c_spine0"))
    return _SEAT[dev]


def seat_fix_params(pose, Q, bis, fix, dev="cpu"):
    """坐姿修正做在关节数据上。fix 是 body_into_scene.seat_fix 刚算好的那份决定(solve 里, 同一个人的网格上算的):
    用哪一帧的腿、每帧绕竖轴转多少、定不定住 —— 这里只管"改的是什么":
      下半身(根 + 两条腿, 连腿长/髋宽)整段用参考帧的, 根只绕竖轴转 fix["yaw"];
      上身每帧的动作保留, 按相邻帧做一次时间平滑(sigma 按 body4d 帧号算);
      上身在世界里的朝向不跟着骨盆变: 根换了朝向, 差的那一点由腰(c_spine0)补上;
      定住(freeze)的人整段就是参考帧。
    pose (T,130) 关节参数, Q (T,3,3) 根在 DA3 世界里的朝向。返回 (pose, Q); 根关节的位置见 seat_root_pos。"""
    from scipy.spatial.transform import Rotation
    K = _seat_consts(dev); T = len(bis)
    miss = [b for b in bis if str(int(b)) not in fix["yaw"]]
    if miss or fix["ref"] not in bis:
        raise SystemExit(f"[失败] 坐姿修正里没有这几帧的决定: {miss or fix['ref']} —— 决定和这个人的帧对不上")
    r = list(bis).index(fix["ref"])
    if fix["freeze"]:
        return np.repeat(pose[r:r + 1], T, 0), np.repeat(Q[r:r + 1], T, 0)
    b = np.asarray(bis, float)
    wt = np.exp(-0.5 * ((b[:, None] - b[None, :]) / fix["sigma"]) ** 2)
    wt[np.abs(b[:, None] - b[None, :]) > 3 * fix["sigma"]] = 0.0
    wt /= wt.sum(1, keepdims=True)
    E = lambda e: Rotation.from_euler("xyz", e).as_matrix()
    q = Rotation.from_matrix(Q @ K["pre"] @ E(pose @ K["A"].T)).as_quat()    # 修之前腰在世界里的朝向 = 上身的朝向
    for i in range(1, T):                                                    # q 和 -q 同一个朝向, 先接成连续的
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    q = wt @ q
    Gs = Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()
    out = wt @ pose
    out[:, K["lower"]] = pose[r, K["lower"]]
    UP = np.asarray(fix["up"], float)
    Qn = np.stack([B._rot_up(UP, fix["yaw"][str(int(bb))]) @ Q[r] for bb in bis])
    e = M._continuous_euler(Rotation.from_matrix(K["pre"].T @ np.swapaxes(Qn, 1, 2) @ Gs).as_euler("xyz"))
    out[:, K["flex"]] += e - out @ K["A"].T                                   # 腰那三个角补到刚好
    return out, Qn


def seat_root_pos(p, W, bis, fix, GH):
    """坐着的人的根关节位置。p (T,3) 现在的根关节位置, W (T,N,3) 用它生成的网格, GH 地面高。
    水平: 骨盆(那圈顶点的中心)落在 fix["pos"] 上。
    竖直: 量的是修完坐姿之后的网格 —— 对上椅子的, 屁股压进椅面 SEAT_SINK; 没对上的, 最低点踩在地上。
          (直接抄 fix["pos"] 的高度的话, 两边的腿差几毫米, 脚就会陷进地面一点: office r8 人 8 陷 1.3 cm。)
    物理之后椅子挪了、人跟着挪的那一下不在这里加: 那时物理还没跑。见 people_store.py 的 _shift。"""
    P = B.load_parts(); UP = np.asarray(fix["up"], float)
    C = W[:, P["pelvis"]].mean(1)
    h = (W - C[:, None, :]) @ UP                                              # 每个顶点比骨盆高多少
    seat = fix.get("seat")
    y = (GH + seat["椅面离地"] - B.SEAT_SINK - h[:, P["seat"]].min(1)) if seat else (GH - h[:, P["low"]].min(1))
    pos = np.array([fix["pos"][str(int(b))] for b in bis])
    want = pos - np.outer(pos @ UP, UP) + np.outer(y, UP)
    return p + want - C


def solve(par, shape, cams, vid2key, kept, UP, GH, mesh_run, da3, fps, focals, dev="cpu",
          chairs=None, seats=None, seat_h_default=0.45):
    """修每个人。这是唯一的一遍: 物理、网页验收视频、3D 页、FBX 读的都是它的结果(见 people_store.py)。
    返回 (res, info, rep):
      res  {人: dict(frames, vf, params(DA3 世界系), shape, conf)}
      info {人: dict(ratio, frames, seated, chair, footprints[, seat_fix])} —— 10_people.json 的 people; 物理读椅子和脚印
      rep  {人: 报告}
    chairs: B.load_chairs() 的结果, 判"坐哪把椅子"用(没有椅子给 None);
    seats: B.chair_seats() 的结果, 给了才对坐着的人做坐姿修正(场景里一把椅子都没有就不做, 原因见下面第 4 步)。
    顺序: 远近 -> 翻转补帧 -> 锁骨头长度 -> 肢体跳变 -> 姿势平滑 -> 位置整理/贴地 -> 坐没坐/椅子/脚印 -> 坐姿修正。
    判据全是 body_into_scene 里那一套(flip_segments / frame_ratios / occ_plan / tidy_track / sit_and_route / seat_fix),
    它们吃世界系的网格 —— 网格由当时的参数生成。"""
    RATIO_TOL = B.RATIO_TOL        # 和算尺度时踢离群的人(ruler_people)是同一条线
    FLIP = export_T(UP)
    todo, all_r = {}, []
    for pid, fr in par.items():
        bis = sorted(b for b in fr if b < len(kept) and kept[b] in vid2key and vid2key[kept[b]] in cams)
        if not bis:
            continue
        rt = {b: fr[b][1] for b in bis}
        ratios = B.depth_ratio(pid, bis, rt, mesh_run, da3, vid2key, kept)
        todo[pid] = (bis, rt, ratios); all_r += list(ratios.values())
    R_all = float(np.median(all_r)) if all_r else 1.0
    out, info, rep = {}, {}, {}
    for pid, (bis, rt, ratios) in todo.items():
        T = len(bis)
        tt = np.array([kept[b] / fps for b in bis])
        tmap = {b: kept[b] / fps for b in bis}
        c2w = np.stack([np.linalg.inv(np.array(cams[vid2key[kept[b]]]["extrinsic_w2c_4x4"], float)) for b in bis])
        Pc = np.stack([par[pid][b][0] for b in bis]); ct = np.stack([rt[b] for b in bis])
        # ★ 先量全部人, 再逐人定比例: 场景尺度的偏差对所有人都一样, 各人的比例本该扎堆; 离全体中位超过 RATIO_TOL 的人
        #   多半是深度被前面的遮挡物带偏了(office 5 号: 别人 1.3~1.66, 他 0.68), 不信自己的, 用全体的
        R_own = float(np.median(list(ratios.values()))) if ratios else R_all
        R = R_own if abs(R_own / R_all - 1.0) <= RATIO_TOL else R_all
        # 0) 翻转段: 只看骨盆朝向。网格从参数生成(这时远近还没定, 朝向和远近无关)
        W0, _ = M.forward(shape[pid], M.to_world(Pc, ct, c2w), True, dev)
        rots = B.pelvis_rots(list(W0))
        flips = {b: (B._rot_deg(rots[i - 1], rots[i]) if i else 0.0) for i, b in enumerate(bis)}
        segs = B.flip_segments(rots, list(tt)) if focals is not None else []
        bad = {bis[i] for i0, i1 in segs for i in range(i0, i1 + 1)}
        # 1) 远近: 每帧按场景深度(模型不可信的帧不用模型的距离)
        rf = B.frame_ratios(bis, ratios if R == R_own else {}, {b: float(rt[b][2]) for b in bis}, R,
                            t=tmap, trust=set(bis) - bad)
        Pw = M.to_world(Pc, ct * np.array([rf[b] for b in bis])[:, None], c2w)
        pose, Q, p = Pw[:, POSE].copy(), M.root_rot(Pw), M.root_pos(Pw)
        # 1b) 翻转补帧: 换的是关节参数、根朝向、根关节位置
        occ = {}
        if segs:
            mroot = B.mask_roots(pid, bis, rt, rf, focals[pid], mesh_run, cams, vid2key, kept)
            occ = B.occ_plan(bis, segs, tmap, dict(zip(bis, p)), mroot)
            pose, Q, p = fill_flips(pose, Q, p, bis, occ)
        raw_p = p.copy()                                         # "整理之前"的位置: 补过翻转段的(补之前那几帧本来就不是他)
        # 1a') 骨头长度: 每人全程定成一个值(见 fix_bone_lengths) —— 姿势参数里管长度的那几个, 和骨骼缩放那一段
        reliable = np.array([str(b) not in occ for b in bis])
        P_len = Pw.copy(); P_len[:, POSE] = pose
        pose = fix_bone_lengths(pose, reliable, M.length_params(dev) - POSE.start)
        Pw = Pw.copy(); Pw[:, M.NPOSE:] = fix_bone_lengths(Pw[:, M.NPOSE:], reliable, np.arange(M.NSCALE))
        P_fix = Pw.copy(); P_fix[:, POSE] = pose
        len_shift = lock_shift(shape[pid], P_len, P_fix, dev)
        # 1b') 肢体级的毛刺: 单条手臂/腿的关节角跳出去又跳回来的那几帧, 换成前后的过渡(见 fix_limb_spikes)
        groups = {n: c - POSE.start for n, c in M.limb_groups(dev).items()}
        pose, limb_fix = fix_limb_spikes(pose, tt, groups)
        # 1c) 姿势平滑: 关节参数 + 根朝向, 根关节位置不动
        pose_raw = pose.copy()
        if T >= 5:
            pose, Q = smooth_joints(pose, tt, limb_speed_fn(shape[pid], Pw[0, M.NPOSE:], tt, dev)), smooth_rots(Q, tt)
        smooth_deg = round(float(np.degrees(np.median(np.abs(pose - pose_raw)))), 2) if T >= 5 else None
        # 2) 位置整理: 骨盆位置三个方向一起平滑 + 贴地 + 腾空(tidy_track 吃网格, 网格从整理好姿势的参数生成)
        Pw = Pw.copy(); Pw[:, POSE] = pose
        Pw = M.set_root(Pw, Q, p)
        W, _ = M.forward(shape[pid], Pw, True, dev)              # 姿势整理完、位置还没整理的人
        conf = np.array([B.CONF_SUB if str(b) in occ else 1.0 for b in bis])
        delta = B.tidy_track(dict(zip(bis, W)), bis, tt, conf, UP, GH)
        # 3) 坐没坐 / 坐哪把椅子 / 走过哪些地方: 物理拿去钉椅子、当路障
        seated, chair, prints = B.sit_and_route(list(W), delta, [kept[b] for b in bis], UP, GH, FLIP, chairs or {})
        feet0 = float(np.median((W @ UP).min(1) - GH))
        step0 = np.linalg.norm(np.diff(raw_p, axis=0), axis=1) if T > 1 else np.zeros(1)
        info[pid] = dict(ratio=R, frames=[int(b) for b in bis], seated=bool(chair is not None or seated.mean() > 0.5),
                         chair=chair, footprints=prints)
        # 4) 坐着的人换一种摆法(见 body_into_scene 的 seat_fix): 位置整理的结果不用; 椅子和脚印不动
        # ★ 怎么摆是在这份网格上定的(骨头已经锁好、肢体跳变已经修掉), 定完马上改参数 —— 以前是摆人那步在它自己的
        #   网格上定、导出那步照着抄, 两边的身体不是同一份(骨头长度不一样), 屁股到脚的落差差几厘米。
        # ★ 场景里一把椅子都没有就不做(09-28 nfl1): 飞扑/被撞倒的人竖直跨度也小于 SEAT_EXTENT, 会被当成坐着,
        #   腿换成整段里站着跑的那一帧、骨盆掰直 —— 横着飞的人被折成反着的蹲姿。office 有椅子, 照旧(人 3 没对上椅子也照旧)
        fix = seat_rep = None
        if seats and info[pid]["seated"]:
            st = seats.get(chair) if chair else None
            fix, diag = B.seat_fix(W, bis, UP, GH, FLIP, (chair, st[0], st[1]) if st else None, seat_h_default)
            info[pid]["seat_fix"] = fix
            Q_free, before = Q, W + delta[:, None, :]
            pose, Q = seat_fix_params(pose, Q, bis, fix, dev)
            Pw[:, POSE] = pose
            Pw = M.set_root(Pw, Q, p)
            W, _ = M.forward(shape[pid], Pw, True, dev)
            delta = seat_root_pos(p, W, bis, fix, GH) - p
            turn = np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t", Q_free, Q) - 1) / 2, -1, 1)))
            seat_rep = dict(diag, **B._seat_report(before, W + delta[:, None, :], UP, GH, fix),
                            骨盆朝向改动_度=[round(float(np.median(turn)), 1), round(float(turn.max()), 1)])
        p = p + delta
        Pw = M.set_root(Pw, Q, p)
        feet = ((W + delta[:, None, :]) @ UP).min(1) - GH
        step = np.linalg.norm(np.diff(p, axis=0), axis=1) if T > 1 else np.zeros(1)
        out[pid] = dict(frames=np.array(bis), vf=np.array([kept[b] for b in bis]), params=Pw, shape=shape[pid], conf=conf)
        rv = np.array(list(ratios.values())) if ratios else np.array([1.0])
        rep[pid] = dict(帧数=T, 远近比例=round(R, 3), 自己量的比例=round(R_own, 3), 用全体中位=bool(R != R_own),
                        全体中位=round(R_all, 3),
                        比例10_90=[round(float(np.percentile(rv, 10)), 3), round(float(np.percentile(rv, 90)), 3)],
                        量到深度的帧=len(ratios), 翻转补帧=B._fix_report(occ, flips, kept),
                        肢体跳变修复={n: [[int(kept[bis[i0]]), int(kept[bis[i1]])] for i0, i1 in v] for n, v in limb_fix.items()},
                        关节参数平滑_改动中位_度=smooth_deg,
                        锁骨头长度_表面挪动_厘米=[round(len_shift[0], 2), round(len_shift[1], 2)],
                        校正前脚离地中位=round(feet0, 3), 校正后脚离地中位=round(float(np.median(feet)), 3),
                        脚离地范围=[round(float(feet.min()), 3), round(float(feet.max()), 3)],
                        帧间位移最大_前=round(float(step0.max()), 3), 帧间位移最大_后=round(float(step.max()), 3),
                        坐着的帧=int(seated.sum()), 椅子=chair, 路线脚印=len(prints),
                        可信帧比例=round(float((conf >= 1.0).mean()), 2))
        rep[pid]["可疑"] = B.suspicious(R_own, R_all, R, rep[pid]["可信帧比例"], float(step0.max()))
        if seat_rep:
            rep[pid]["坐姿修正"] = seat_rep
    return out, info, rep


def identity_record(names, who, log, kept):
    """核对身份的结果, 按**跟踪编号**整理成闸口上能直接摆出来的样子(网页"人"那张表、驱动的闸口报告都读它):
      去向 {跟踪编号: [dict(事="人"/"并进别人"/"不算人", 人=人号, 帧数, 视频帧=[头, 尾], 原因), ...]}
    names: people_identity.track_ids(); who / log: people_identity.resolve() 的结果; kept: body4d 帧号 -> 原视频帧号。"""
    fate = {}
    for pid, items in who.items():
        bs = [b for rid, b in items if str(names[rid]) == pid]
        if bs:
            fate.setdefault(pid, []).append(dict(事="人", 人=pid, 帧数=len(bs), 视频帧=[int(kept[min(bs)]), int(kept[max(bs)])]))
    for x in log:
        row = dict(事=x["事"], 帧数=len(x["帧"]), 视频帧=[int(kept[x["帧"][0]]), int(kept[x["帧"][-1]])], 原因=x["原因"])
        if x["事"] == "并进别人":
            row["人"] = x["并进"]
        fate.setdefault(str(x["编号"]), []).append(row)
    for rows in fate.values():
        rows.sort(key=lambda r: r["视频帧"][0])
    return dict(勾的人=sorted(int(v) for v in set(names.values())), 编号对照=names,
                人={pid: sorted({int(names[rid]) for rid, _ in items}) for pid, items in who.items()},
                去向=dict(sorted(fate.items(), key=lambda kv: int(kv[0]))),
                说明="人号 = 跟踪编号(挑人卡片上的数字)。勾的人 = 这一次摆人时闸口上勾着的; 编号对照 = 出网格目录里的编号 -> 跟踪编号; "
                     "去向 = 每个跟踪编号的帧最后算谁的")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r1", required=True, help="场景结果目录(底下有 02_da3/ 和 07_scene.glb)")
    ap.add_argument("--body", required=True, help="body4d 那一轮(底下有 00_frames/ 和 04_mesh_occ*/)")
    ap.add_argument("--scene-in", default="07_scene.glb", help="找椅子的场景(物理之前的)")
    ap.add_argument("--seat-fix", choices=["on", "off"], default="on",
                    help="坐着的人按椅面定高度、腿取整段稳定姿势、上身整段平滑; off = 和站着的人一样逐帧整理")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    da3, cams, UP, GH, vid2key = B.load_scene(a.r1)
    fj = json.load(open(f"{a.body}/00_frames/frames.json")); kept, fps = fj["kept_indices"], fj["src_fps"]
    # 跟踪给的编号不等于人: 先按房间里的位置核对一遍(并掉同一个人的几个编号, 去掉不是人的), 修的是核对过的那份
    print("== 核对身份 (见 people_identity.py) ==")
    raw = people_identity.raw_run(a.body)
    if not os.path.isdir(f"{raw}/mhr_4d_individual"):
        raise SystemExit(f"[失败] {raw} 没有模型的关节数据(mhr_4d_individual; 2026-10-08 之前出的人体网格没存) —— 重跑 people_mesh")
    names = people_identity.track_ids(raw)
    who, id_log = people_identity.resolve(B.identity_tracks(raw, da3, cams, vid2key, kept, fps), UP,
                                          name=names, taken=people_identity.all_track_ids(a.body, raw))
    mesh_run = people_identity.write(a.body, raw, who, id_log)
    for x in id_log:
        vf = [kept[b] for b in x["帧"]]
        print(f"  编号 {x['编号']} 的 {len(vf)} 帧(视频第 {vf[0]}~{vf[-1]} 帧): "
              + (f"并进人 {x['并进']}" if x["事"] == "并进别人" else "不算人") + f" —— {x['原因']}")
    print(f"  跟踪给了 {len(names)} 个编号 {sorted(names.values())} -> 核对后 {len(who)} 个人: {list(who)}")

    print("== 摆人 (物理之前; 在模型的关节数据上修, 只算这一遍) ==")
    par, shape = load_params(mesh_run)
    print(f"  人体 {len(par)} 人, 每人 {[len(v) for v in par.values()]} 帧; 场景 {len(cams)} 个相机, 天 {np.round(UP, 3)}, "
          f"地面高 {GH:.2f} m, 视频 {fps:.2f} fps; 设备 {dev}")
    chairs = B.load_chairs(f"{a.r1}/{a.scene_in}")
    seats = B.chair_seats(f"{a.r1}/{a.scene_in}", GH) if a.seat_fix == "on" else None
    # 没坐在椅子正上方的人按"脚着地", 选腿时的目标落差用场景里椅面高度的中位数
    seat_h0 = float(np.median([s[0] for s in seats.values()])) if seats else 0.45
    res, info, rep = solve(par, shape, cams, vid2key, kept, UP, GH, mesh_run, da3, fps, B.load_focals(mesh_run), dev,
                           chairs=chairs, seats=seats, seat_h_default=seat_h0)
    if not res:
        raise SystemExit("[失败] 人体的帧和场景的相机一帧都对不上 —— 检查是不是同一段素材")
    for pid, q in rep.items():
        print(f"  人 {pid}: {q['帧数']:3d} 帧  远近 x{q['远近比例']:.3f} "
              + (f"(自己量的 {q['自己量的比例']} 离群, 改用全体中位) " if q["用全体中位"] else
                 f"({q['比例10_90'][0]}~{q['比例10_90'][1]}, {q['量到深度的帧']} 帧量到) ")
              + f" 脚离地 {q['校正前脚离地中位']:+.2f} -> {q['校正后脚离地中位']:+.2f} m {q['脚离地范围']}  "
              f"最大一步 {q['帧间位移最大_前']:.2f} -> {q['帧间位移最大_后']:.2f} m  "
              + (f"坐在 {q['椅子']} ({q['坐着的帧']} 帧)" if q["椅子"] else
                 (f"坐着 {q['坐着的帧']} 帧但没找到椅子" if q["坐着的帧"] > q["帧数"] / 2 else f"路线 {q['路线脚印']} 个脚印")))
        print(f"         关节平滑改动中位 {q['关节参数平滑_改动中位_度']}°"
              f"  锁骨头长度: 表面挪了平均 {q['锁骨头长度_表面挪动_厘米'][0]} / 最大 {q['锁骨头长度_表面挪动_厘米'][1]} cm")
        if q["可疑"]:
            print(f"         ★可疑: {'; '.join(q['可疑'])} —— 看验收那道闸\"人\"那张表里他的截图, 不是人就点掉")
        if q["肢体跳变修复"]:
            print("         肢体跳变: " + "; ".join(f"{n} {' '.join(f'{a}~{b}' for a, b in v)}" for n, v in q["肢体跳变修复"].items()))
        for g in q["翻转补帧"]:
            print(f"         翻转补帧: 第 {g['帧'][0]}~{g['帧'][1]} 帧({g['帧数']} 帧, 段里最大一翻 {g['最大一翻']}°) {g['补法']}")
        z = q.get("坐姿修正")
        if z:
            print(f"         坐姿修正: 高度按 {z['高度来源']}" + ("" if z["离椅面中心"] is None else f"(骨盆离椅面中心 {z['离椅面中心']} m)")
                  + f"  腿取第 {kept[z['参考帧']]} 帧({z['候选帧']} 帧候选, 落差 {z['落差']}/{z['目标落差']} m)"
                  + ("  ★朝向每帧乱翻(中位 %.0f°), 整段定住" % z["骨盆转角中位_度"] if z["定住"] else "")
                  + f"; 骨盆朝向改了 中位 {z['骨盆朝向改动_度'][0]}° 最大 {z['骨盆朝向改动_度'][1]}°"
                  + f"\n         骨盆离地 {z['骨盆离地_前']} -> {z['骨盆离地_后']}  头顶最大一跳 {z['头顶最大一跳_前']} -> {z['头顶最大一跳_后']} m"
                  + (f"  屁股到椅面 {z['屁股到椅面_前']:+.3f} -> {z['屁股到椅面_后']:+.3f} m" if "屁股到椅面_前" in z else "")
                  + f"  脚离地 {z['脚离地_后']}")
    # 落盘: 修好的参数 + 它生成的网格(网页那几样读) + 物理用的人体空气墙。网格就是参数喂给 MHR 的结果, 不另算
    FLIP = export_T(UP)
    verts = {pid: M.forward(r["shape"], r["params"], True, dev)[0] for pid, r in res.items()}
    body = B.people_body({pid: dict(zip(res[pid]["vf"].tolist(), verts[pid].astype(np.float64))) for pid in res}, FLIP)
    pinned = sorted({c["chair"] for c in info.values() if c["chair"]})
    meta = dict(floor_y=GH, pinned_chairs=pinned, people=info, 报告=rep,
                身份=identity_record(names, who, id_log, kept), 人体来源=people_identity.source_of(a.body),
                说明="人修在模型的关节数据上, 结果在 10_people_params.npz(参数)和 10_people_world.npz(它生成的网格), 读法见 "
                     "people_store.py。这里是给物理和闸口看的: footprints 是 glTF 系 xz 的矩形, 物理那步当路障; pinned_chairs 物理不推; "
                     "seat_fix 是坐着的人怎么摆的那份决定(已经做进参数里了, 留着备查); 报告和身份给闸口网页")
    faces = M.model(dev).character_torch.mesh.faces.cpu().numpy()
    people_store.save(a.r1, meta, res, verts, faces, body, B.BODY_PITCH, fps)
    print(f"  -> {a.r1}/{people_store.META}  (钉住的椅子 {pinned}, 路线脚印共 {sum(len(c['footprints']) for c in info.values())} 个)")
    print(f"  -> {a.r1}/{people_store.PARAMS}  {people_store.WORLD}  (每人每帧的关节参数, 和它生成的网格)")
    print(f"  -> {a.r1}/{people_store.BODY}  (人体空气墙: 每人整段扫过的身体内芯(表面往里收 {B.BODY_PITCH*100:.0f} cm), "
          f"{B.BODY_PITCH*100:.0f} cm 格子, 不含手; "
          + ", ".join(f"{k}: {len(v)}" for k, v in sorted(body.items(), key=lambda x: int(x[0]))) + ")")


if __name__ == "__main__":
    main()
