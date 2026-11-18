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
import legs_proc as LPROC
from observed_pose import hold_tracked_upper_pose
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


# ---------------------------------------------------------------- 腿估翻的一小段 (2026-11-11)
# ★ lovelynight Emma 3~4 秒(第 28~35 帧): 原片里一直左腿搭右腿、没动; 模型逐帧估, 左右膝谁高每一两帧翻一次, 腿一帧跳 6~18 cm,
#   脚伸到她自己的分割外面去(脚落在分割上的比例 0.34~0.89, 前后帧全是 1.00)。fix_limb_spikes 只修一两帧的毛刺, 漏了。
# ★ 为什么看分割、不看动作: 只看动作(两端一样 + 中间走得远 + 来回走)试过, 走路一个步态周期、踢踏舞、滑板车全被判成"抖"
#   (pedestrian 每人 2~6 段); 逐帧反向的比例也分不开(Emma 0.43, 踢踏舞 0.43~0.62)。真动作模型的脚跟着原图走, 落在分割上 = 1.00;
#   估翻的帧脚在原图上没人的地方。只在"前后都对得上"的短段里改: 整段对不上 = 腿看不见(出画 / 被挡), 那是补腿的事, 不在这里。
# ★ 只改坐着的帧(竖直跨度 < B.SEAT_EXTENT): 全盘粗查(132 人)不限坐站有 28 人被改, 多是走路/跑/踢踏舞里脚被别人或东西挡住的一两帧 ——
#   站着动的人短段线性过渡会把步子抹成滑步, 那该交给补腿(proclegs)按脚步规划; 坐着的人腿动得慢, 过渡就对。
LEG_OFF_GOOD = 0.95      # 脚落在自己分割上的比例到这么多 = 这一帧对得上
LEG_OFF_BAD = 0.7        # 一段里最差的一帧低于这个才动(Emma 0.34; 单独一帧 0.87 这种不动)
LEG_OFF_MAX_S = 1.2      # 秒: 两端(对得上的帧)之间最多这么长
LEG_IN_FRAME = 0.9       # 脚在画面里的比例到这么多才判(出画的帧是补腿的事)
LEG_OFF_GAP = 2          # 帧: 两段对不上的中间只夹着这么几帧对得上的, 并成一段 —— 估翻的那几帧里偶尔有一帧碰巧对上
                         #   (Emma 第 30 帧), 拿它当端点就是从一个翻了的姿势过渡(第一版实测 28~32 帧左右膝还是反的)


def leg_offmask_runs(inframe, onmask, t, seg=None, use=None, good=LEG_OFF_GOOD, bad=LEG_OFF_BAD, max_s=LEG_OFF_MAX_S,
                     in_frame=LEG_IN_FRAME, gap=LEG_OFF_GAP):
    """inframe/onmask: 每帧脚在画面里 / 落在自己分割上的比例(B.feet_mask_frac, nan = 没法判); t 秒; seg 每帧属于哪段(不跨切镜头)。
    use: 哪些帧可以判(坐着的帧), 别的帧当没法判。返回 [(a, b, 诊断)]: a、b 是两端对得上的帧(保留), 中间 a+1..b-1 是要换的帧。"""
    inframe, onmask, t = np.asarray(inframe, float), np.asarray(onmask, float), np.asarray(t, float)
    T = len(t); seg = np.zeros(T, int) if seg is None else np.asarray(seg)
    obs = np.isfinite(onmask) & (np.nan_to_num(inframe) >= in_frame)
    if use is not None:
        obs &= np.asarray(use, bool)
    ok = obs & (np.nan_to_num(onmask) >= good)
    off = obs & ~ok
    out, i = [], 0
    while i < T:
        if not off[i]:
            i += 1; continue
        j = i
        while True:
            if j + 1 < T and off[j + 1] and seg[j + 1] == seg[i]:
                j += 1; continue
            nxt = next((k for k in range(j + 1, min(T, j + 2 + gap)) if off[k]), None)   # 隔着不超过 gap 帧还有对不上的
            if nxt is not None and ok[j + 1:nxt].all() and (seg[j + 1:nxt + 1] == seg[i]).all():
                j = nxt; continue
            break
        a, b = i - 1, j + 1
        if (a >= 0 and b < T and ok[a] and ok[b] and seg[a] == seg[i] == seg[b]
                and t[b] - t[a] <= max_s and float(np.min(onmask[i:j + 1])) < bad):
            out.append((a, b, dict(最差一帧落在分割上=round(float(np.min(onmask[i:j + 1])), 2))))
        i = j + 1
    return out


def fix_leg_runs(pose, runs, t, cols):
    """runs 里每段中间的帧, 两条腿的关节参数换成两端之间按时间的线性过渡。返回新 pose。"""
    pose = np.array(pose, float, copy=True); t = np.asarray(t, float)
    for a, b, _ in runs:
        s = ((t[a + 1:b] - t[a]) / (t[b] - t[a]))[:, None]
        pose[a + 1:b, cols] = pose[a, cols] * (1 - s) + pose[b, cols] * s
    return pose


# ---------------------------------------------------------------- 地面硬约束 (2026-11-11)
# ★ hitch 13.2~14.0 秒(视频 332~349 帧)短镜头里相机估错(坏相机), 坏相机的帧按设计"留在模型自己放的位置"(不平滑、不贴地),
#   两人整个沉进地里: 骨盆 0.97/1.01 m -> 0.49/0.51 m, 脚在地面以下 30~51 cm。看不到脚的帧(DIST_KEEP)也按设计不贴地。
#   这些"不贴地"的豁免各有道理(和原图对齐), 但"人在地面以下"在哪种情况下都不对 —— 所以最后统一兜一道:
#     任何帧: 最低点低于地面超过 GROUND_TOL -> 整个人抬到地面(只抬不压: 跳起来、腾空不受影响);
#     坏相机的帧: 位置本来就不可信, 上下都贴地(最低点 = 地面)。
#   地面按 B.clearance(有 07_terrain.npz 就按脚下地形)。只动根的高度, 姿势一个数不动。
GROUND_TOL = 0.02   # 米: 穿地这么多以内不管(补腿 IK 的残差 1~3 cm、脚掌贴地的正常压入)
# ★ 2026-11-12 jw4 甄子丹坐地上的特写(4.8~11.6 秒): 腿出画, 模型猜的脚在地下 6~54 cm(每帧不一样), 骨盆的位置是对的
#   (离地 0.13~0.19 m, 投回原图顶点 70% 落在他的分割上)。按"最低点"抬 = 按猜的脚抬: 每帧抬得不一样 -> 上下跳;
#   抬 30~50 cm 人投到画面上方(落在分割上 0.7 -> 0.1~0.2), 抹鼻子的手和头都出了画。
#   所以只认可信的脚: 看得到(feet_visible), 或补腿(legs_proc)按脚印摆在地上的。脚不可信的帧不按脚抬、坏相机也不贴地,
#   只保证小腿和脚以外的身体(lower_leg_mask 以外的顶点)不在地下; 猜的小腿和脚进地交给 legs_proc.clamp_feet 弯腿。
#   hitch 13 秒那批坏相机的帧腿是补腿摆的(短镜头站着), 照旧上下贴地。
#   ★ 第一版只看屁股(贴椅面那圈顶点): jw1 倒在地上打的人头 / 背 / 胳膊进地 6~20 cm 没人管, 改成看全身(除了猜的小腿和脚)。


def ground_lift(clear, bad=None, tol=GROUND_TOL, seen=None, body_clear=None):
    """clear (T,) 每帧最低点离地(米, 负 = 在地下); bad (T,) 坏相机的帧。返回每帧整个人要往上挪多少(米)。
    seen (T,) 脚可信的帧; 给了的话, 脚不可信的帧只按 body_clear (T,)(小腿和脚以外的身体最低点离地)抬, 不按脚抬、不贴地。
    seen 不给 = 老规则(结果逐位不变)。"""
    clear = np.asarray(clear, float)
    lift = np.where(clear < -tol, -clear, 0.0)
    if bad is not None:
        bad = np.asarray(bad, bool)
        lift[bad] = -clear[bad]
    if seen is not None:
        bod = np.full(len(clear), np.inf) if body_clear is None else np.asarray(body_clear, float)
        lift = np.where(np.asarray(seen, bool), lift, np.where(bod < -tol, -bod, 0.0))
    return lift


# ---------------------------------------------------------------- 看不见的腿去抖 (2026-11-12)
# ★ jw4 甄子丹坐地上的特写: 腿出画, 模型每帧重猜一次腿, 左膝 98° -> 73° -> 97° 一帧一跳(膝盖平时每帧变 3~4°, 个别帧跳 20° 以上)。
#   关节平滑(smooth_joints)按肢体速度放宽 —— 真甩腿不能抹平 —— 于是把"猜的腿乱跳"当成快动作放过去了。
#   看不见的腿不是观测, 不该跟着每帧的猜测动: 脚不可信的帧(看不见、也不是补腿摆的)两条腿的参数先去单帧乱跳(中值),
#   再按时间平滑; 窗口按秒算(6 fps / 30 fps 一样)。前后看得见的帧当锚(参与滤波、本身不改), 不跨切点 / 坏相机段。
LEG_GUESS_MED_S = 0.5      # 秒: 中值窗口(短于它一半的乱跳去掉)
LEG_GUESS_SIGMA_S = 0.25   # 秒: 之后的高斯平滑


def smooth_unseen_legs(P, unseen, t, seg, cols, med_s=LEG_GUESS_MED_S, sig_s=LEG_GUESS_SIGMA_S):
    """P (T,204); unseen (T,) 腿是猜的帧; t (T,) 秒; seg (T,) 段号; cols 腿的参数列。返回新的 P(只改 unseen 帧的 cols)。"""
    from scipy.ndimage import median_filter, gaussian_filter1d
    P = np.asarray(P, float); out = P.copy()
    unseen = np.asarray(unseen, bool); t = np.asarray(t, float); seg = np.asarray(seg); T = len(P); cols = list(cols)
    i = 0
    while i < T:
        if not unseen[i]:
            i += 1; continue
        j = i
        while j + 1 < T and unseen[j + 1] and seg[j + 1] == seg[i]:
            j += 1
        a = i - 1 if i > 0 and seg[i - 1] == seg[i] else i                   # 前后看得见的帧当锚
        b = j + 1 if j + 1 < T and seg[j + 1] == seg[i] else j
        n = b - a + 1
        if n >= 3:
            dt = float(np.median(np.diff(t[a:b + 1]))) or 1.0 / 30
            k = max(3, int(round(med_s / dt)) | 1)
            k = min(k, n if n % 2 else n - 1)
            X = median_filter(P[a:b + 1][:, cols], size=(k, 1), mode="nearest")
            X = gaussian_filter1d(X, max(sig_s / dt, 0.5), axis=0, mode="nearest")
            out[i:j + 1, cols] = X[i - a:j - a + 1]
        i = j + 1
    return out


_LOWLEG = {}


def lower_leg_mask(dev="cpu"):
    """(N,) 顶点是不是小腿 / 脚(蒙皮权重最大的关节在 l_lowleg / r_lowleg 及以下)。和人无关, 只看模板的蒙皮。"""
    if dev not in _LOWLEG:
        sk = M.skeleton(dev); names = [str(n) for n in sk["names"]]; par = sk["parents"].astype(int)
        sub_ = {names.index("l_lowleg"), names.index("r_lowleg")}
        changed = True
        while changed:
            changed = False
            for k, p_ in enumerate(par):
                if p_ in sub_ and k not in sub_:
                    sub_.add(k); changed = True
        o = np.lexsort((-sk["sk_w"], sk["sk_v"])); v_, j_ = sk["sk_v"][o], sk["sk_j"][o]
        first = np.r_[True, v_[1:] != v_[:-1]]
        jm = np.full(int(v_.max()) + 1, -1); jm[v_[first]] = j_[first]
        _LOWLEG[dev] = np.isin(jm, list(sub_))
    return _LOWLEG[dev]


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


# ---------------------------------------------------------------- 腿长按地面定 (2026-11-10, 开关 --leg-fit)
# ★ 中景 / 近景里腿在画面外, SAM 3D Body 的腿是猜的。看得到的上半身它估得准(投回原图头顶差 6~8 px),
#   猜的腿偏短: hitch_cuts_indoor r1 两人脚离地 +0.13 / +0.15 m。以前的贴地(tidy_track)是把整个人往下拉,
#   上半身跟着沉, 投回原图头低了 80~100 px —— 人看着矮、相机看着高。
#   改成: 腿是猜的就按"脚落在地上"定腿长。MHR 的 leg_length_flexible 每 +1 脚底下移约 9.8 cm,
#   头顶不动、骨盆附近挪不到 1 cm(2026-11-10 两人实测), 上半身和原图的对齐不受影响。宽度不变。
LEG_COL = 135              # mhr_model_params 里 leg_length_flexible 的下标
LEG_FIT_MAX = 0.25         # 腿最多伸 / 缩这么多米; 再多就不是"猜短了", 是别的错(尺度、地面), 不该拿腿去填
LEG_FEET_SEEN = 0.5        # 看得到脚的帧超过这个比例: 腿是看着估的, 不动
SEAT_KEEP_LEGS_SEEN = 0.8  # ★ 2026-11-11 坐在椅子上、看得到脚的帧至少这么多: 腿照原样, 屁股按椅面定高(B.seat_keep_legs)。
                           #   比 LEG_FEET_SEEN 严: office r1_sam31 人 1 坐在办公桌后 0.52, 腿大半被桌子挡着, 照旧整段定腿
LEG_STAND_EXT = 1.3        # 头顶到脚底的跨度中位小于这个(米): 不像站着(坐 / 蹲), 脚离地不代表腿短, 不动
LEG_MIN_GAP = 0.02         # 差不到这么多就不动

# ---------------------------------------------------------------- 远近: 模型的距离不离谱就用模型的 (2026-11-10, 开关 --dist-keep)
# ★ 远近那步每帧按场景深度把人沿视线推远 / 拉近, 身体大小不变 —— 投回原图人就变大 / 变小、整体挪开。
#   hitch_cuts_indoor r1 逐步量(同一顶点投回原图, 和模型自己的结果比): 远近这一步之后普通帧偏 23 / 40 px,
#   某个镜头 236 px; 后面翻转补帧 / 骨长 / 平滑 / 腿长 加起来只差几个 px。逐帧比例 0.85~1.26。
#   模型的距离是按"人在画面里多大"算的, 和原图天然对齐; 场景深度的用处是兜住模型塌掉的帧(soccer 守门员扑救
#   距离从 19 m 掉到 8~11 m, 见 B.frame_ratios)。所以: 和场景差不到 DIST_KEEP_TOL 就用模型的(× 这个人的整体比例 R),
#   差得多才换成场景的。
DIST_KEEP_TOL = 0.25       # 和 B.RATIO_TOL(判一个人整体深度离群)同一条线
# ★ 2026-11-12 切点后开头几帧的模型焦距换成本镜头自己的(FOCAL_FIX_TOL: 差超过这么多才换, None = 不换)。
#   SAM 3D Body 的焦距是一段帧一起估的, 这段会跨切点: 新镜头开头几帧沿用上个镜头的焦距(值一模一样)。
#   joker 推特写开头 4 帧(视频 110~118)焦距 3296(上个镜头的; 本镜头 4794), 模型把人放在 2.5 m(场景深度 3.7 m);
#   远近那步换成场景深度只把人沿视线推远、焦距没换, 投回原图偏下半个人(45~50%)、横偏 0.2 m, 补腿再把骨盆抬 83 cm 去凑。
#   同样的情况: jw1 230~236、jw4 甄子丹 144~154、hitch 332~349、joker 355~358。
#   换法: 相机系横、纵不动, 深度 × 本镜头焦距 / 这帧焦距(人在画面里的位置和大小不变, 距离和本镜头其余帧一致)。
#   换成本镜头自己的焦距, 不换成场景焦距: 场景焦距在这种帧上也可能估错(jw1 230 帧场景 4894, 本镜头场景中位 4028;
#   11-12 第一版换成场景焦距, 这帧对齐 30% -> 116%, 而且没换的帧跟着整体比例变, 31~34 帧对齐变差)。
#   这个人的整体量照换之前的算, 其余帧一个数都不动: 远近比例 R 用换之前的比例; 腿长(LEG_FIT, 全程一个值)不看换过的帧 ——
#   joker 换了 355 / 358 两帧, 脚离地中位 3.1 -> 1.8 cm, 腿长那步的 2 cm 门槛一翻, 全片 170 帧整体低 3.4 cm、62 帧对齐变差。
#   门槛 15%: 差得少的(jw1 7 号镜头 10%)在 DIST_KEEP 容忍的误差里, 换了反而 7 帧变差; 跨切点沿用的一般差 30~45%。
FOCAL_FIX_TOL = 0.15
FOCAL_SAME = 0.01          # 和上个镜头最后一帧的焦距差不到这么多 = 沿用的
FOOT_GROUND = True         # 补腿的脚按脚下那一级台阶摆(只在交付地面模型时, 见 legs_proc.fill_legs 的 ground_at)
# ★ 2026-11-13 翻转段里按轮次指定"这几帧是真动作、用模型自己的"(FLIP_KEEP; 空 = 老做法)。
#   jw4 人 5 在 16.7 秒被打飞翻滚倒地: 骨盆一帧转 26~50°, 超过 FLIP_RATE(一帧 24°)被当成模型乱翻, 501~520 帧整段换掉;
#   后面没有好帧(人出画), 只能"动作定住在 500 帧(半空弯腰) + 跟 mask" —— 原片已经躺在地上, 模型还弯腰站着僵 1 秒。
#   模型原始输出 500~516 帧其实跟原片一致(debug_1113/p5_freeze/raw_500_530.jpg), 518 帧起才真错(一帧转 123°, 半截出画)。
#   配置: 00_run_config.json 的 覆盖.FLIP_KEEP = "人号:起-止[,起-止][;人号:...]"(原视频帧号, 闭区间)。
#   这些帧从翻转段里拿掉(信模型); 剩下的翻转段照旧补 —— 紧挨着的保留帧成了锚, 定住的是保留段最后一帧的姿势。
FLIP_KEEP = {}


def parse_flip_keep(txt):
    """"5:501-517;3:10-20,40-50" -> {"5": [(501, 517)], "3": [(10, 20), (40, 50)]}。空串 -> {}。"""
    out = {}
    for part in str(txt or "").split(";"):
        part = part.strip()
        if not part:
            continue
        pid, rngs = part.split(":", 1)
        for r in rngs.split(","):
            lo, hi = (int(x) for x in r.strip().split("-"))
            out.setdefault(pid.strip(), []).append((min(lo, hi), max(lo, hi)))
    return out


def flip_keep_mask(bis, kept, occ, ranges):
    """FLIP_KEEP 的帧 + 翻转补帧里两头锚点都是 FLIP_KEEP 帧的(定在它们上的帧) -> (T,) bool。ranges 空 -> 全 False。
    ★ 2026-11-13 这些帧是真动作(摔倒 / 翻滚 / 躺地), 补腿会把人当成走路、配一双站着的腿, 骨盆撑回站着的高度:
      jw4 人 5 躺在地上时重心离地 85 cm(关补腿 28 cm), 投回原图偏高 120~290 px。所以脚当看得见(贴地照做), 补腿不碰。"""
    T = len(bis)
    if not ranges:
        return np.zeros(T, bool)
    inr = lambda v: any(lo <= int(v) <= hi for lo, hi in ranges)
    out = np.array([inr(kept[b]) for b in bis], bool)
    for j, b in enumerate(bis):
        o = (occ or {}).get(str(b))
        if o is not None and inr(kept[o[0]]) and inr(kept[o[1]]):
            out[j] = True
    return out


def flip_keep_split(segs, vfs, ranges):
    """翻转段 [(i0, i1)](下标, 闭区间) 去掉原视频帧号落在 ranges 里的帧, 剩下的按连续下标重新切段。vfs: 每个下标的原视频帧号。"""
    if not ranges:
        return list(segs)
    keep = [any(lo <= int(v) <= hi for lo, hi in ranges) for v in vfs]
    out = []
    for i0, i1 in segs:
        cur = None
        for i in range(i0, i1 + 1):
            if keep[i]:
                if cur is not None:
                    out.append(tuple(cur)); cur = None
            elif cur is None:
                cur = [i, i]
            else:
                cur[1] = i
        if cur is not None:
            out.append(tuple(cur))
    return out


def focal_fix_cut(rt, focal, shot, tol=0.15, vf=None):
    """切点后开头几帧沿用了上个镜头焦距的, 换到本镜头自己的焦距(见 FOCAL_FIX_TOL)。
    rt {bi: 模型相机系根位置(米)}; focal {bi: 模型焦距}; shot {bi: 镜头号}。
    返回 (新 rt, 新焦距, {bi: 深度乘了多少}, 换过的帧(给了 vf 用视频帧号))。"""
    rt, fo, k_, nfix = dict(rt), dict(focal), {}, []
    by = {}
    for b in sorted(b for b in rt if focal.get(b)):
        by.setdefault(shot[b], []).append(b)
    for s_ in sorted(by):
        if s_ - 1 not in by:
            continue
        f_prev = focal[by[s_ - 1][-1]]
        same = lambda b: abs(focal[b] / f_prev - 1.0) <= FOCAL_SAME
        own = [focal[b] for b in by[s_] if not same(b)]
        if not own:
            continue
        f_own = float(np.median(own))
        for b in by[s_]:
            if not same(b):
                break                                  # 只管镜头开头连着的那几帧
            if abs(f_own / focal[b] - 1.0) > tol:
                k_[b] = f_own / focal[b]
                rt[b] = np.asarray(rt[b], float) * np.array([1.0, 1.0, k_[b]]); fo[b] = f_own
                nfix.append(int(vf[b]) if vf else int(b))
    return rt, fo, k_, sorted(nfix)


def fit_leg_length(shape, Pw, W, UP, GH, feet_ok, dev="cpu", use=None, vfs=None):
    """看不到脚的站着的人: 改腿长让脚底(各帧最低点的中位)落到地面。返回 (Pw, W, 报告)。
    Pw (T,204) 世界系参数, W (T,N,3) 它生成的网格。整段一个值(和锁骨头长度一致: 一个人腿长不该逐帧变)。"""
    use = np.ones(len(W), bool) if use is None or not np.any(use) else np.asarray(use, bool)  # 坏相机的帧不参与(世界里的位置是错的)
    h = W @ UP
    vfs = np.array([None] * len(W) if vfs is None else list(vfs), dtype=object)
    gap = float(np.median([B.clearance(w, UP, GH, v) for w, v in zip(W[use], vfs[use])]))   # 有地形按脚下(本镜头)地形量
    ext = float(np.median(h[use].max(1) - h[use].min(1)))
    seen = float(np.mean(np.asarray(feet_ok)[use])) if len(feet_ok) else 0.0
    r = dict(脚离地_前=round(gap, 3), 看得到脚的帧比例=round(seen, 2), 头到脚跨度中位=round(ext, 3))
    why = ("脚大多看得到, 腿不是猜的" if seen > LEG_FEET_SEEN else
           "不像站着(跨度 < %.1f m)" % LEG_STAND_EXT if ext < LEG_STAND_EXT else
           "差不到 %d cm" % round(LEG_MIN_GAP * 100) if abs(gap) < LEG_MIN_GAP else None)
    if why:
        r["不动"] = why
        return Pw, W, r
    idx = np.unique(np.linspace(0, len(Pw) - 1, min(8, len(Pw))).astype(int))
    Q = np.array(Pw[idx], copy=True); Q[:, LEG_COL] += 1.0
    W1, _ = M.forward(shape, Q, True, dev)
    slope = float(np.median(h[idx].min(1) - (W1 @ UP).min(1)))         # 参数每 +1 脚底下移多少米
    if slope < 0.02:
        r["不动"] = f"腿长参数带不动脚(每 +1 只下移 {slope * 100:.1f} cm)"
        return Pw, W, r
    want = float(np.clip(gap, -LEG_FIT_MAX, LEG_FIT_MAX))
    d = want / slope
    Pw = np.array(Pw, copy=True); Pw[:, LEG_COL] += d
    W, _ = M.forward(shape, Pw, True, dev)
    r.update(腿长参数改动=round(d, 3), 腿伸长_米=round(want, 3), 封顶=bool(want != gap),
             脚离地_后=round(float(np.median([B.clearance(w, UP, GH, v) for w, v in zip(W[use], vfs[use])])), 3),
             身高_后=round(float(np.median((W[use] @ UP).max(1) - (W[use] @ UP).min(1))), 3))
    return Pw, W, r


def solve(par, shape, cams, vid2key, kept, UP, GH, mesh_run, da3, fps, focals, dev="cpu",
          chairs=None, seats=None, seat_h_default=0.45, leg_fit=False, cuts=(), badcam=frozenset(),
          dist_keep=False, legs_proc_on=True, scene_in=None, r1_dir=None, pair_arm_on=True):
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
    foc_use = {}
    _cuts = np.asarray(cuts if cuts is not None else [], float)
    for pid, fr in par.items():
        bis = sorted(b for b in fr if b < len(kept) and kept[b] in vid2key and vid2key[kept[b]] in cams)
        if not bis:
            continue
        rt = {b: fr[b][1] for b in bis}
        kfix = {}
        if FOCAL_FIX_TOL and focals is not None and pid in focals:              # 切点后沿用上个镜头焦距的帧(见 focal_fix_cut)
            rt, foc_use[pid], kfix, nfix = focal_fix_cut(
                rt, focals[pid], {b: int(np.searchsorted(_cuts, kept[b] / fps + 1e-3, side="right")) for b in bis},
                tol=FOCAL_FIX_TOL, vf={b: int(kept[b]) for b in bis})
            if nfix:
                print(f"  人 {pid}: 切点后沿用上个镜头焦距的 {len(nfix)} 帧(视频第 {', '.join(map(str, nfix))} 帧)换到本镜头的焦距", flush=True)
        ratios = B.depth_ratio(pid, bis, rt, mesh_run, da3, vid2key, kept)
        r_R = {b: r * kfix.get(b, 1.0) for b, r in ratios.items()}                # 整体比例照换焦距之前的算(其余帧不动)
        todo[pid] = (bis, rt, ratios, r_R, kfix); all_r += list(r_R.values())
    R_all = float(np.median(all_r)) if all_r else 1.0
    out, info, rep = {}, {}, {}
    for pid, (bis, rt, ratios, r_R, kfix) in todo.items():
        T = len(bis)
        # Every image-space consumer must use the same calibrated distance and
        # focal pair as the world solve. Keep the camera observation independent
        # of the later scene-depth fit (rf).
        camera_track = [(par[pid][b][0], rt[b], *par[pid][b][2:]) for b in bis]
        focal_by_frame = foc_use.get(pid, (focals or {}).get(pid, {}))
        camera_focals = [focal_by_frame.get(b) for b in bis] if focal_by_frame else None
        # ★ 切镜头 / 坏相机: 时间轴在段边界拉开 SHOT_GAP 秒, 下面所有按时间的平滑都不跨段(见 B.time_segments)
        bc = np.array([vid2key[kept[b]] in badcam for b in bis])
        seg = B.time_segments([kept[b] for b in bis], fps, cuts, bc)
        # 虚拟间隔只供平滑使用; 翻转补帧按真实时间和显式段号计算。
        source_t = np.array([kept[b] / fps for b in bis])
        tt = source_t + B.SHOT_GAP * seg
        tmap = dict(zip(bis, tt))
        repair_time = dict(zip(bis, source_t))
        c2w = np.stack([np.linalg.inv(np.array(cams[vid2key[kept[b]]]["extrinsic_w2c_4x4"], float)) for b in bis])
        Pc = np.stack([par[pid][b][0] for b in bis]); ct = np.stack([rt[b] for b in bis])
        # ★ 先量全部人, 再逐人定比例: 场景尺度的偏差对所有人都一样, 各人的比例本该扎堆; 离全体中位超过 RATIO_TOL 的人
        #   多半是深度被前面的遮挡物带偏了(office 5 号: 别人 1.3~1.66, 他 0.68), 不信自己的, 用全体的
        R_own = float(np.median(list(r_R.values()))) if r_R else R_all
        R = R_own if abs(R_own / R_all - 1.0) <= RATIO_TOL else R_all
        # 0) 翻转段: 只看骨盆朝向。网格从参数生成(这时远近还没定, 朝向和远近无关)
        W0, _ = M.forward(shape[pid], M.to_world(Pc, ct, c2w), True, dev)
        rots = B.pelvis_rots(list(W0))
        flips = {b: (B._rot_deg(rots[i - 1], rots[i]) if i else 0.0) for i, b in enumerate(bis)}
        segs = B.flip_segments(rots, source_t, segments=seg) if focal_by_frame else []
        if FLIP_KEEP.get(str(pid)):                              # 见 FLIP_KEEP
            _n0 = sum(i1 - i0 + 1 for i0, i1 in segs)
            segs = flip_keep_split(segs, [kept[b] for b in bis], FLIP_KEEP[str(pid)])
            print(f"   人 {pid}: FLIP_KEEP {FLIP_KEEP[str(pid)]} -> 翻转段 {_n0} 帧 -> {sum(i1 - i0 + 1 for i0, i1 in segs)} 帧")
        bad = {bis[i] for i0, i1 in segs for i in range(i0, i1 + 1)}
        # 1) 远近: 每帧按场景深度(模型不可信的帧不用模型的距离)
        rf = B.frame_ratios(bis, ratios if R == R_own else {}, {b: float(rt[b][2]) for b in bis}, R,
                            t=tmap, trust=set(bis) - bad)
        n_keep = 0
        if dist_keep:                                            # 1') 模型的距离不离谱就用模型的(见 DIST_KEEP_TOL)
            keep = {b: abs(rf[b] / R - 1.0) <= DIST_KEEP_TOL for b in bis}
            # In a dense shot, an abrupt change in monocular range can drag the
            # pelvis despite locked feet. Use the depth-supported range on the
            # contiguous unstable interval, with a time-based taper at its ends.
            depth_weight = np.zeros(T)
            if any(cams[vid2key[kept[b]]].get("插值") for b in bis):
                scene_z = np.array([rf[b] * rt[b][2] for b in bis])
                model_z = ct[:, 2] * R
                for sid in np.unique(seg):
                    ix = np.flatnonzero(seg == sid)
                    if len(ix) < 3 or bc[ix].any():
                        continue
                    evidence = [j for j in ix if bis[j] in ratios]
                    if len(evidence) < 2:
                        continue
                    # Only suppress a model-only impulse where local measured
                    # scene range exists on both sides of the suspect interval.
                    err = model_z[ix] - scene_z[ix]
                    for j in np.flatnonzero(np.abs(np.diff(err)) > np.maximum(.25, 4.*np.diff(tt[ix]))):
                        k = ix[j+1]
                        if not (tt[evidence[0]] <= tt[k] <= tt[evidence[-1]]):
                            continue
                        depth_weight[ix] = np.maximum(depth_weight[ix], np.exp(-.5*((tt[ix]-tt[k])/B.DIST_SIGMA)**2))
            n_keep = sum(keep.values())
            rf = {b: ((1-depth_weight[j])*R + depth_weight[j]*rf[b] if keep[b] else rf[b])
                  for j,b in enumerate(bis)}
            if bc.any() and not bc.all() and ratios and R == R_own and any(
                    cams[vid2key[kept[b]]].get("插值") for b in bis):
                from range_continuity import bridge_range_fallbacks
                vf_range = np.asarray([kept[b] for b in bis])
                real_shot = np.searchsorted(_cuts, source_t + .001, side="right")
                breaks = (np.diff(real_shot) != 0) | (np.diff(vf_range) != 1)
                continuous_shot = np.cumsum(np.r_[0, breaks])
                range_clock = dict(zip(bis, source_t + B.SHOT_GAP * continuous_shot))
                z = dict(zip(bis, ct[:, 2]))
                reference = B.frame_ratios(bis, ratios, z, R, t=range_clock, trust=set(bis) - bad)
                current = np.asarray([rf[b] * z[b] for b in bis])
                scene_depth = np.asarray([reference[b] * z[b] for b in bis])
                measured = np.asarray([b in ratios and b not in bad for b in bis])
                dense = np.asarray([bool(cams[vid2key[kept[b]]].get("插值")) for b in bis])
                samples = np.asarray([ratios[b] * z[b] if measured[j] else np.nan for j, b in enumerate(bis)])
                joined = bridge_range_fallbacks(vf_range, real_shot, bc, current, scene_depth, measured, fps,
                                               dense=dense, depth_samples=samples)
                for j in np.flatnonzero(joined != current):
                    rf[bis[j]] = joined[j] / z[bis[j]]
        Pw = M.to_world(Pc, ct * np.array([rf[b] for b in bis])[:, None], c2w)
        pose, Q, p = Pw[:, POSE].copy(), M.root_rot(Pw), M.root_pos(Pw)
        # 1b) 翻转补帧: 换的是关节参数、根朝向、根关节位置
        occ = {}
        if segs:
            with_focal = [b for b in bis if focal_by_frame.get(b) and focal_by_frame[b]>0]
            mroot = dict.fromkeys(bis)
            mroot.update(B.mask_roots(pid, with_focal, rt, rf, focal_by_frame, mesh_run, cams, vid2key, kept))
            occ = B.occ_plan(bis, segs, repair_time, dict(zip(bis, p)), mroot, segments=seg)
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
        # 1b'') 腿估翻的一小段: 脚投回原图落到自己分割外面、前后都对得上 -> 两条腿用前后过渡(见 leg_offmask_runs)
        P_now = Pw.copy(); P_now[:, POSE] = pose; P_now = M.set_root(P_now, Q, p)
        W_now, S_now = M.forward(shape[pid], P_now, True, dev)
        rest_ = M.rest_verts(shape[pid], dev); fidx = np.where(rest_[:, 1] < rest_[:, 1].min() + 10.0)[0]
        inf_, onm_ = B.feet_mask_frac(W_now, bis, pid, mesh_run, cams, vid2key, kept, fidx)
        sit_ = np.ptp(W_now @ UP, axis=1) < B.SEAT_EXTENT
        leg_runs = leg_offmask_runs(inf_, onm_, tt, seg, use=sit_)
        leg_cols = np.concatenate([groups[n] for n in ("l_upleg", "r_upleg") if n in groups])
        if leg_runs and len(leg_cols):
            pose = fix_leg_runs(pose, leg_runs, tt, leg_cols)
        # 1b''') 近景特写里伸出画面外的手腕: 和出画的腿同理, 看不见的手每帧重猜会乱甩骗过 limb_speed, 先按中值+高斯去抖
        _cam_list = [cams[vid2key[kept[b]]] for b in bis]
        if T >= 5 and all("intrinsic_3x3" in cd and "extrinsic_w2c_4x4" in cd for cd in _cam_list):
            _jnames = [str(x) for x in M.skeleton(dev)["names"]]
            _E_all = np.array([np.asarray(cd["extrinsic_w2c_4x4"], float) for cd in _cam_list])
            _K_all = np.array([np.asarray(cd["intrinsic_3x3"], float) for cd in _cam_list])
            for _wname, _gname in (("l_wrist", "l_clavicle"), ("r_wrist", "r_clavicle")):
                if _wname in _jnames and _gname in groups:
                    _wj = _jnames.index(_wname)
                    _pw = S_now[:, _wj, :3] / 100.0
                    _pc = np.einsum("tij,tj->ti", _E_all[:, :3, :3], _pw) + _E_all[:, :3, 3]
                    _px = np.einsum("tij,tj->ti", _K_all, _pc)
                    _uv = _px[:, :2] / np.maximum(_px[:, 2:3], 0.05)
                    _wh = np.stack([2.0 * _K_all[:, 0, 2], 2.0 * _K_all[:, 1, 2]], axis=1)
                    _out_frm = (_pc[:, 2] <= 0.1) | (_uv[:, 0] < 4.0) | (_uv[:, 0] > _wh[:, 0] - 4.0) | (_uv[:, 1] < 4.0) | (_uv[:, 1] > _wh[:, 1] - 4.0)
                    if _out_frm.sum() >= 3 and not _out_frm.all():
                        pose = smooth_unseen_legs(pose, _out_frm, source_t, seg, groups[_gname], med_s=0.35, sig_s=0.18)
        # 1c) 姿势平滑: 关节参数与相机外参无关, 全程保留平滑; 坏相机帧的根朝向改在相机系里平滑再转回世界系(不跟着错外参拉歪, 也不退回毛刺原值)
        pose_raw = pose.copy(); Q_raw = Q.copy()
        if T >= 5:
            pose = smooth_joints(pose, tt, limb_speed_fn(shape[pid], Pw[0, M.NPOSE:], tt, dev))
            Q = smooth_rots(Q, tt)
            if bc.any():
                _w2c_r = np.swapaxes(c2w[:, :3, :3], 1, 2)
                _Q_cam_sm = smooth_rots(_w2c_r @ Q_raw, tt)
                Q[bc] = (c2w[:, :3, :3] @ _Q_cam_sm)[bc]
        smooth_deg = round(float(np.degrees(np.median(np.abs(pose - pose_raw)))), 2) if T >= 5 else None
        # 2) 位置整理: 骨盆位置三个方向一起平滑 + 贴地 + 腾空(tidy_track 吃网格, 网格从整理好姿势的参数生成)
        Pw = Pw.copy(); Pw[:, POSE] = pose
        Pw = M.set_root(Pw, Q, p)
        W, observed_skeleton = M.forward(shape[pid], Pw, True, dev)  # 姿势整理完、位置还没整理的人
        conf = np.array([B.CONF_SUB if str(b) in occ else 1.0 for b in bis])
        rest = M.rest_verts(shape[pid], dev)
        feet_idx = np.where(rest[:, 1] < rest[:, 1].min() + 10.0)[0]           # 静止姿势里离脚底 10 cm 以内的顶点
        feet_ok = B.feet_visible(W, bis, pid, mesh_run, cams, vid2key, kept, feet_idx)
        _orig_terrain = B.TERRAIN
        collider = None
        if legs_proc_on and scene_in:
            from stance_stabilizer import build_support_collider
            _r1 = r1_dir or os.path.dirname(da3)
            collider = build_support_collider(_r1, scene_in, UP, GH, _orig_terrain,
                                              [dict(W_da3=W, shape=shape[pid], feet_ok=feet_ok)], fps, dev=dev)
            if collider.boxes_gltf:
                B.TERRAIN = collider
        leg = None
        if leg_fit:                                              # 2a) 腿是猜的: 按地面定腿长, 再贴地(见 fit_leg_length)
            Pw, W, leg = fit_leg_length(shape[pid], Pw, W, UP, GH, feet_ok, dev, use=~bc & ~np.array([b in kfix for b in bis], bool),
                                        vfs=[kept[b] for b in bis])         # 换过焦距的帧不参与(见 FOCAL_FIX_TOL)
            pose = Pw[:, POSE].copy()                            # 坐姿修正那支会拿 pose 写回 Pw, 腿长要在里面
        keep_m = flip_keep_mask(bis, kept, occ, FLIP_KEEP.get(str(pid)))   # 见 flip_keep_mask: 信模型的帧, 脚当看得见, 补腿不碰
        if keep_m.any():
            feet_ok = np.asarray(feet_ok, bool) | keep_m
            print(f"   人 {pid}: FLIP_KEEP 信模型的帧 {int(keep_m.sum())} 个(含定在它们上的), 脚当看得见、补腿不碰", flush=True)
        delta = B.tidy_track(dict(zip(bis, W)), bis, tt, conf, UP, GH, feet_ok=feet_ok,
                             snap_ok=feet_ok if dist_keep else None, vfs=[kept[b] for b in bis], foot_indices=feet_idx)   # 和原图对齐优先: 看不到脚不贴地
        delta[bc] = 0.0                                          # 坏相机的帧不在世界系平滑/贴地; 连续段改在相机系平滑根位置
        if bc.any():
            _E_bc = np.array([np.asarray(cams[vid2key[kept[b]]]["extrinsic_w2c_4x4"], float) for b in bis])
            _p_cam = np.einsum("tij,tj->ti", _E_bc[:, :3, :3], p) + _E_bc[:, :3, 3]
            for _sid in np.unique(seg[bc]):
                _ix = np.flatnonzero((seg == _sid) & bc)
                if len(_ix) >= 5:
                    _p_cam_sm = B.smooth_track(_p_cam[_ix], tt[_ix])
                    delta[_ix] = np.einsum("tij,tj->ti", c2w[_ix, :3, :3], _p_cam_sm) + c2w[_ix, :3, 3] - p[_ix]
        # 3) 坐没坐 / 坐哪把椅子 / 走过哪些地方: 物理拿去钉椅子、当路障
        seated, chair, prints = B.sit_and_route(list(W), delta, [kept[b] for b in bis], UP, GH, FLIP, chairs or {})
        feet0 = float(np.median([B.clearance(w, UP, GH) for w in W]))
        step0 = np.linalg.norm(np.diff(raw_p, axis=0), axis=1) if T > 1 else np.zeros(1)
        if T > 1:
            step0[seg[1:] != seg[:-1]] = 0.0                     # 切点 / 坏相机交界那一步是换机位, 不算人跳
        info[pid] = dict(ratio=R, frames=[int(b) for b in bis], seated=bool(chair is not None or seated.mean() > 0.5),
                         chair=chair, footprints=prints)
        # 4) 坐着的人换一种摆法(见 body_into_scene 的 seat_fix): 位置整理的结果不用; 椅子和脚印不动
        # ★ 怎么摆是在这份网格上定的(骨头已经锁好、肢体跳变已经修掉), 定完马上改参数 —— 以前是摆人那步在它自己的
        #   网格上定、导出那步照着抄, 两边的身体不是同一份(骨头长度不一样), 屁股到脚的落差差几厘米。
        # ★ 场景里一把椅子都没有就不做(09-28 nfl1): 飞扑/被撞倒的人竖直跨度也小于 SEAT_EXTENT, 会被当成坐着,
        #   腿换成整段里站着跑的那一帧、骨盆掰直 —— 横着飞的人被折成反着的蹲姿。office 有椅子, 照旧(人 3 没对上椅子也照旧)
        fix = seat_rep = keep_rep = None
        st_k = seats.get(chair) if (seats and chair) else None
        # ★ 2026-11-11 jw4: 没坐在椅子上的"坐着"的人, 骨盆一直在动就不是坐着(蹲着打、爬), 不做坐姿修正 ——
        #   人 2 蹲着出拳, 被当成坐着、朝向判"乱翻"整段定住, 13 秒处一动不动。坐在椅子上的照旧。
        sit_move = (B.seat_spread(W + delta[:, None, :], UP, ~bc)
                    if (seats and info[pid]["seated"] and not st_k and not (chair and seats.get(chair))) else 0.0)
        if st_k and float(np.mean(feet_ok)) >= SEAT_KEEP_LEGS_SEEN:
            delta, keep_rep = B.seat_keep_legs(W, delta, tt, conf, UP, GH, st_k[0], seated, vfs=[kept[b] for b in bis])
            delta[bc] = 0.0
            info[pid]["seat_fix"] = dict(mode="keep_legs", seat=dict(chair=chair, 椅面离地=round(float(st_k[0]), 4)))
        elif seats and info[pid]["seated"] and sit_move <= B.SEAT_STILL_M:
            st = seats.get(chair) if chair else None
            fix, diag = B.seat_fix(W, bis, UP, GH, FLIP, (chair, st[0], st[1]) if st else None, seat_h_default,
                                   legs_seen=None if st else np.asarray(feet_ok, bool), bad=bc)
            info[pid]["seat_fix"] = fix
            Q_free, before = Q, W + delta[:, None, :]
            pose, Q = seat_fix_params(pose, Q, bis, fix, dev)
            if bc.any():                                         # 坏相机的帧: 朝向、位置用模型自己的(跟着错相机才投得回原图)
                Q = Q.copy(); Q[bc] = Q_free[bc]
            Pw[:, POSE] = pose
            Pw = M.set_root(Pw, Q, p)
            W, _ = M.forward(shape[pid], Pw, True, dev)
            delta = seat_root_pos(p, W, bis, fix, GH) - p
            delta[bc] = 0.0
            turn = np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t", Q_free, Q) - 1) / 2, -1, 1)))
            seat_rep = dict(diag, **B._seat_report(before, W + delta[:, None, :], UP, GH, fix),
                            骨盆朝向改动_度=[round(float(np.median(turn)), 1), round(float(turn.max()), 1)])
        p = p + delta
        Pw = M.set_root(Pw, Q, p)
        registration = os.path.join(os.path.dirname(da3), "contact_registration.json")
        contact_record = None
        root_registration = []
        calibrated_root = None
        if legs_proc_on and os.path.isfile(registration):
            with open(registration) as file:
                contact_record = json.load(file)
        if (contact_record and len(todo)==1 and getattr(B.TERRAIN, "landing_priority", False)
                and fix is None and keep_rep is None):
            from root_registration import registered_horizontal_path, camera_root_rays
            from PIL import Image
            frames = np.asarray([kept[b] for b in bis])
            shot_ids = np.searchsorted(_cuts,source_t+.001,side="right")
            names = list(M.skeleton(dev)["names"])
            joints = [names.index(n) for n in ("l_upleg","r_upleg","l_clavicle","r_clavicle")]
            torso = observed_skeleton[:,joints,:3]/100
            torso -= torso[:,:2].mean(1,keepdims=True)
            feet = []
            for sign in (1,-1):
                side = feet_idx[sign*rest[feet_idx,0]>0]
                sole = W[:,side]
                feet.append(sole[np.arange(T),np.argmin(sole@UP,axis=1)] + delta)
            mask_path = f"{mesh_run}/masks/{bis[0]:08d}.png"
            rays = None
            if camera_focals is not None and os.path.isfile(mask_path):
                with Image.open(mask_path) as image:
                    rays = camera_root_rays(camera_track,camera_focals,[cams[vid2key[int(f)]] for f in frames],
                                            image.size)
            registered_p, root_registration = registered_horizontal_path(
                p, frames, shot_ids, bc, UP, fps, B.TERRAIN, contact_record, torso, np.stack(feet,axis=1),rays,feet_ok)
            delta = delta + (registered_p-p)
            p = registered_p
            calibrated_root = registered_p.copy()
            Pw = M.set_root(Pw,None,p)
        # ★ 补腿(legs_proc, 2026-11-11, 一直开): 腿看不见 / 看得见但贴地在滑 的段按 proclegs 规划脚印、IK 摆 MHR 的腿;
        #   全程看得见、坐着(含上面做过坐姿修正 / 腿照原样的)不动。只改两条腿的参数和根高度, 根水平位置、朝向、上半身一个数不动。
        #   判据和数字见 legs_proc.py 开头; hitch 两人 / office 4 号、2 号的试跑见 <run>/debug_1111/legs_proc/README。
        legs_rep = None
        height_report = None
        mask_paths = [f"{mesh_run}/masks/{b:08d}.png" for b in bis]
        if legs_proc_on:
            sit = np.ones(T, bool) if (fix is not None or keep_rep) else np.asarray(seated, bool)
            # ★ 2026-11-12 没做坐姿修正的人: 按段(切点 / 坏相机分开)看骨盆, 一段里骨盆一直在动(离开中心 > SEAT_STILL_M)
            #   就不是坐着(跳舞屈膝、蹲着打), 这段的"坐着"作废。上面那条同样的规则只在场景里有椅子时才算, joker 没有椅子,
            #   身高按地面拟合成 1.41~1.47 m(SEAT_EXTENT 1.45)时 0%~69% 的帧被判坐着, 补腿整个人跳过、主镜头对齐 2.5%~11% 来回变。
            #   按段算: 镜头之间世界系对不上, 跨镜头算会把机位差当成走动; 坐地上不动的人(jw4)一段里骨盆基本不动, 不受影响。
            if fix is None and not keep_rep and sit.any():
                _sg = np.asarray(seg); _Y = W + delta[:, None, :]; _off = []
                for _s in np.unique(_sg):
                    _m = (_sg == _s) & ~np.asarray(bc, bool)
                    if _m.sum() < 5:      # 整段都是坏相机帧(推特写): cam_smooth 已拟合成匀速, 段内位置连贯, 用全部帧
                        _m = _sg == _s
                    if _m.sum() >= 5 and sit[_sg == _s].any():
                        _mv = B.seat_spread(_Y[_m], UP, cam=c2w[_m][:, :3, 3])   # 只量和视线垂直的(见 seat_spread)
                        if _mv > B.SEAT_STILL_M:
                            sit[_sg == _s] = False; _off.append((int(_s), round(_mv, 2)))
                if _off:
                    print(f"  人 {pid}: 坐着判定作废的段(骨盆一直在动, 段号, 离开中心 m): {_off}", flush=True)
            # ★ 2026-11-11 有地形(GROUND_MODE=terrain)时, 补腿按每帧人脚下(根关节正下方、这帧所在镜头)的地形高摆腿, 不再按一块平地 GH:
            #   joker 楼梯上补腿先把骨盆按平地压低最多 4.6 m, 再靠地面约束抬回来; 3 号镜头深度本身矛盾, 一压一抬人就偏了(对齐 288%)。
            #   相邻帧跨台阶时地形高一跳 11.5 cm, 按时间轻平滑(段内 sigma 1.5 帧)。没地形 = 老做法(标量 GH, 结果逐位不变)。
            GHk = GH
            if B.TERRAIN is not None:
                from scipy.ndimage import gaussian_filter1d as _g1
                _rp = M.root_pos(Pw)
                GHk = np.array([float(B.TERRAIN.height_at(_rp[j], vf=kept[b])[0]) for j, b in enumerate(bis)])
                _sg = np.asarray(seg)
                for _s in np.unique(_sg):
                    _m = _sg == _s
                    if _m.sum() > 2: GHk[_m] = _g1(GHk[_m], 1.5, mode="nearest")
            # ★ 2026-11-12 交付的是地面模型(平面 + 台阶网格)时, 补腿的脚按脚下那一级台阶摆(见 legs_proc.fill_legs 的 ground_at)
            _gat = None
            if FOOT_GROUND and B.TERRAIN is not None and getattr(B.TERRAIN, "model_mesh", False):
                _gat = lambda P_, k_: B.TERRAIN.model_h(P_)
            registered_starts = []
            if contact_record:
                registered = int(contact_record["shot"]) - 1
                shot_ids = np.searchsorted(_cuts, np.asarray([kept[b] for b in bis]) / fps + .001)
                registered_starts = np.flatnonzero((shot_ids[1:] == registered) &
                                                   (shot_ids[:-1] == registered-1)) + 1
            from leg_observations import measured_thighs
            track_cameras = [cams[vid2key[kept[b]]] for b in bis]
            measured = measured_thighs(Pc, ct, camera_focals, track_cameras, mask_paths,
                                       int(pid), np.searchsorted(_cuts,source_t+.001), fps, dev)
            from height_feasibility import standing_root_shift
            scene_frames = np.asarray([kept[b] for b in bis])
            scene_shot = np.searchsorted(_cuts, source_t+.001)
            protected = np.asarray(keep_m, bool).copy()
            if fix is not None or keep_rep is not None or chair is not None:
                protected |= np.asarray(sit,bool)
            if contact_record and len(todo)==1:
                protected |= scene_shot == int(contact_record['shot'])-1
            shift = standing_root_shift(Pw, shape[pid], W, delta, Pc, ct, camera_focals, track_cameras,
                                        mask_paths, int(pid), feet_ok, sit, scene_frames, scene_shot,
                                        fps, UP, B.TERRAIN, protected, root_registration, measured, dev)
            if np.any(shift):
                vertical = shift @ UP
                planar = shift - vertical[:,None]*UP
                height_report = {
                    "帧数": int((np.linalg.norm(shift,axis=1)>.0001).sum()),
                    "最大上移_cm": round(100*float(vertical.max(initial=0)),2),
                    "最大下移_cm": round(-100*float(vertical.min(initial=0)),2),
                    "最大水平校准_cm": round(100*float(np.linalg.norm(planar,axis=1).max()),2),
                }
                p = p + shift
                delta = delta + shift
                Pw = M.set_root(Pw, None, p)
                if calibrated_root is not None:
                    calibrated_root = calibrated_root + shift
            _Pw_keep = Pw.copy()
            Pw, legs_rep = LPROC.fill_legs(Pw, shape[pid], tt, feet_ok, UP, GHk, seated=sit, seg=seg, dev=dev,
                                           log=lambda s_: print(f"  人 {pid}:{s_.strip()}", flush=True), bad=bc, ground_at=_gat, registered_starts=registered_starts,
                                           source_time=source_t, shot=np.searchsorted(_cuts,source_t+.001), thighs=measured,
                                           has_calibrated_root=([f is not None and np.isfinite(f) and f > 0 for f in camera_focals]
                                                            if camera_focals is not None else False))
            if keep_m.any():                                     # 见 flip_keep_mask: 补腿不碰(看不见脚在滑那条规则也可能把它们改回看不见)
                Pw[keep_m] = _Pw_keep[keep_m]
                if not legs_rep.get("补的帧"):
                    p = M.root_pos(Pw)
            if legs_rep.get("补的帧"):
                p = M.root_pos(Pw)
                W, _ = M.forward(shape[pid], Pw, True, dev)
                delta = np.zeros_like(delta)                    # W 已经是摆好的; 下面 feet 用 W + delta 量
        # ★ 2026-11-12 脚不可信 = 看不见、也不是补腿按脚印摆的; 这些帧的腿是猜的: 先去抖(smooth_unseen_legs), 后面地面约束、收腿都认这一份
        done_ = np.asarray(legs_rep["_规划"]["done"], bool) if (legs_rep and "_规划" in legs_rep) else np.zeros(T, bool)
        vis_ = np.asarray(legs_rep["_规划"].get("vis", feet_ok), bool) if (legs_rep and "_规划" in legs_rep) else np.asarray(feet_ok, bool)
        guess_ = ~vis_ & ~done_
        guess_rep = None
        if legs_proc_on and guess_.any():
            lc_ = LPROC.LEG_COLS["l"] + LPROC.LEG_COLS["r"]
            Pn = smooth_unseen_legs(Pw, guess_, tt, seg, lc_)
            dd = np.degrees(np.abs(Pn[:, lc_] - Pw[:, lc_]).max(1))
            if dd.max() > 1e-6:
                k_ = [lc_.index(LPROC.KNEE_COL[s_]) for s_ in ("l", "r")]
                _adj = guess_[1:] & guess_[:-1] & (np.asarray(seg)[1:] == np.asarray(seg)[:-1])      # 相邻、都是猜的、同一段
                jump = lambda X: float(np.degrees(np.abs(np.diff(X[:, k_], axis=0))[_adj]).max()) if _adj.any() else 0.0
                guess_rep = dict(帧=int(guess_.sum()), 改动_度_中位最大=[round(float(np.median(dd[guess_])), 1), round(float(dd.max()), 1)],
                                 膝盖逐帧最大跳_度_前后=[round(jump(Pw[:, lc_]), 1), round(jump(Pn[:, lc_]), 1)])
                Pw = Pn; p = M.root_pos(Pw)
                W, _ = M.forward(shape[pid], Pw, True, dev); delta = np.zeros_like(delta)
        # ★ 地面硬约束(见 ground_lift): 不许在地下; 坏相机的帧上下都贴地。放在所有摆法(贴地/座位/补腿)之后
        clr = np.array([B.clearance(w, UP, GH, kept[b]) for w, b in zip(W + delta[:, None, :], bis)])
        # ★ 2026-11-12 有地形(GROUND_MODE=terrain)时: 坏相机的长段已由 badcam_ground.py 把相机连人一起挪到脚踩地,
        #   这里再"上下都贴地"只挪人、不挪相机, 投回原图就偏 —— joker 推特写(脸占满画面, 人离相机约 2 m)贴地挪了 -9~+547 cm,
        #   这段对齐 2.9% -> 178%。所以坏相机帧也只按"不许在地下"抬; 看不到脚的帧(腿出画, 脚是模型猜的)一律不抬:
        #   第一版留了"埋进地里 0.3 m 以上照样抬", 推特写的相机落在台阶侧边, 一部分帧的人算在台阶脚印里(比平面高 3.4 m),
        #   被整个抬 3.4 m, 投到画面上方 2000 px 外, 还是 178%。高低由 badcam_ground / shot_register 连相机一起定。
        #   没有地形 = 老做法(结果逐位不变)。
        if B.TERRAIN is not None:
            if getattr(B.TERRAIN, "model_mesh", False):
                soles = W[:, feet_idx] + delta[:, None, :]
                clr = np.array([np.min(v @ UP - B.TERRAIN.model_h(v)) for v in soles])
            lift = ground_lift(clr, None)
            # 看不到脚的好相机帧: 不按猜的脚抬, 但小腿和脚以外的身体(lower_leg_mask 以外)不许进脚下的地形(按逐帧地形量,
            #   和没地形那支的 body_clear 同一个意思; 那支按一块平地 GH 量)。坏相机帧不抬: 高低已由 badcam_ground 连相机一起定。
            _keep = ~lower_leg_mask(dev)
            body_clr = np.array([B.clearance(w[_keep], UP, GH, kept[b]) for w, b in zip(W + delta[:, None, :], bis)])
            lift_body = np.where((body_clr < -GROUND_TOL) & ~np.asarray(bc, bool), -body_clr, 0.0)
            lift = np.where(np.asarray(feet_ok, bool), lift, lift_body)
            if (lift_body > 0).any() and not np.asarray(feet_ok, bool).all():
                print(f"  人 {pid}: 看不到脚的帧里身体进地形 {int(((lift_body > 0) & ~np.asarray(feet_ok, bool)).sum())} 帧, 最多抬 {lift_body.max()*100:.0f} cm", flush=True)
        else:
            # ★ 2026-11-12 只按可信的脚抬(见 ground_lift): 看得到的, 加上补腿摆在地上的; 其余帧只管屁股不进地
            seen_ = np.asarray(feet_ok, bool).copy()
            if legs_rep and "_规划" in legs_rep:
                seen_ |= np.asarray(legs_rep["_规划"]["done"], bool)
            body_clr = ((W + delta[:, None, :])[:, ~lower_leg_mask(dev)] @ UP).min(1) - GH
            lift = ground_lift(clr, bc, seen=seen_, body_clear=body_clr)
        ground_rep = None
        if np.any(np.abs(lift) > 1e-6):
            p = p + np.outer(lift, UP); Pw = M.set_root(Pw, Q, p); delta = delta + np.outer(lift, UP)
            up_ = lift > 1e-6; bc_ = bc & (np.abs(lift) > 1e-6)
            ground_rep = dict(抬出地面的帧=int((up_ & ~bc).sum()), 最多抬_米=round(float(lift[up_ & ~bc].max()), 3) if (up_ & ~bc).any() else 0.0,
                              坏相机贴地的帧=int(bc_.sum()), 坏相机挪动_米=[round(float(lift[bc_].min()), 3), round(float(lift[bc_].max()), 3)] if bc_.any() else None)
        # ★ 2026-11-12 看不见、也没被补腿摆过的脚在地下: 身体不动, 只把腿弯上来(LPROC.clamp_feet, 规则和踩过的坑写在那里)。
        #   jw4 甄子丹坐地上(坐着 48%, 补腿不管), 腿出画, 猜的脚在地下 6~54 cm; 地面约束已经不按猜的脚抬人(见 ground_lift)。
        clamp_rep = None
        if legs_proc_on:
            act_ = guess_
            if act_.any():
                Pw, clamp_rep = LPROC.clamp_feet(Pw, shape[pid], act_, UP, GHk, seg=seg, dev=dev)
                if clamp_rep["改的帧"]:
                    W, _ = M.forward(shape[pid], Pw, True, dev); delta = np.zeros_like(delta); p = M.root_pos(Pw)
        if legs_proc_on and getattr(B.TERRAIN, "model_mesh", False) and done_.any():
            # IK uses sparse foot targets. At tread edges a sole vertex can
            # intersect the upper tread even when every IK target is satisfied.
            # Check both soles independently and keep the projected torso fixed.
            feet_sides = [feet_idx[rest[feet_idx, 0] * sign > 0] for sign in (1, -1)]
            Pw, W = LPROC.clear_planned_feet(Pw, shape[pid], W + delta[:, None, :],
                                               done_, feet_sides, UP, B.TERRAIN.model_h, dev)
            delta = np.zeros_like(delta)
        # ★ 2026-11-11 人不穿墙(室外墙, 见 body_into_scene.wall_push): 贴墙跳的帧人的远近估深十几厘米会穿进墙(singin 24/180 帧),
        #   沿墙法向推到相机那一侧。必须放在所有摆法最后: 放在贴地后面时, 补腿(legs_proc)按脚印重摆的腿又伸回了墙里。
        if B.WALLS is not None:
            _C = np.array([np.linalg.inv(np.asarray(cams[vid2key[kept[b]]]["extrinsic_w2c_4x4"], float))[:3, 3] for b in bis])
            _wp, _nw = B.wall_push(W + delta[:, None, :], UP, C=_C)
            _wp[bc] = 0.0
            if _nw:
                p = p + _wp; Pw = M.set_root(Pw, Q, p); delta = delta + _wp
                print(f"    人 {pid}: {_nw} 次(帧 x 墙)穿墙, 推到相机那一侧, 最多推 {np.linalg.norm(_wp, axis=1).max() * 100:.0f} cm", flush=True)
        if legs_proc_on and len(registered_starts) and getattr(B.TERRAIN, "model_mesh", False):
            sides = [feet_idx[rest[feet_idx, 0] * sign > 0] for sign in (1, -1)]
            Pw = LPROC.pin_registered_support(Pw, shape[pid], np.asarray([kept[b] for b in bis]),
                                               registered_starts, sides, UP, B.TERRAIN, fps, dev, planned=done_, support_side=contact_record.get("support_side"))
            W, _ = M.forward(shape[pid], Pw, True, dev)
            refine_frames = np.zeros(T, bool)
            for first in registered_starts:
                elapsed = (np.asarray([kept[b] for b in bis]) - kept[bis[first]]) / fps
                refine_frames |= (elapsed >= 0) & (elapsed <= .35) & done_
            Pw, W = LPROC.clear_planned_feet(Pw, shape[pid], W, refine_frames, sides, UP, B.TERRAIN.model_h, dev)
            delta = np.zeros_like(delta)
        source_frames = np.asarray([kept[b] for b in bis])
        frame_paths = [os.path.join(os.path.dirname(mesh_run), "00_frames", f"{b:04d}.jpg") for b in bis]
        if (legs_proc_on and getattr(B.TERRAIN, "landing_priority", False) and len(registered_starts)
                and np.all(np.diff(source_frames) == 1)
                and camera_focals is not None and all(f is not None and np.isfinite(f) and f > 0 for f in camera_focals)):
            import landing_motion
            from PIL import Image
            mask_path = f"{mesh_run}/masks/{bis[0]:08d}.png"
            with Image.open(mask_path) as mask_image:
                wh = mask_image.size
            Pw = landing_motion.refine(Pw,shape[pid],source_frames,
                                       cams,vid2key,_cuts,fps,UP,B.TERRAIN,contact_record,
                                       camera_track, camera_focals,wh,dev,
                                       plan=legs_rep.get("_规划") if legs_rep else None,
                                       masks=mask_paths, person_id=int(pid), thighs=measured,
                                       frame_paths=frame_paths, calibrated_root=calibrated_root, prefill_pose=_Pw_keep,
                                       verified_frames=[f for match in root_registration for f in match["joins"]])
            p = M.root_pos(Pw)
            W, _ = M.forward(shape[pid], Pw, True, dev)
            delta = np.zeros_like(delta)
        elif hold_tracked_upper_pose(Pw, M.limb_groups(dev), frame_paths, mask_paths, int(pid), source_frames, seg):
            W, _ = M.forward(shape[pid], Pw, True, dev)
            delta = np.zeros_like(delta)
        stab_rep = {}
        if legs_proc_on:
            from stance_stabilizer import SupportCollider, stabilize_actor_stances
            _act_col = collider if collider is not None else SupportCollider(UP, GH, _orig_terrain)
            _is_seated = bool(fix is not None or keep_rep is not None or (chair is not None)
                              or (np.mean(seated) > LPROC.SEATED_SKIP and sit_move <= B.SEAT_STILL_M))
            _cam_dicts = [cams[vid2key[kept[b]]] for b in bis]
            _w2c = [np.asarray(cd["extrinsic_w2c_4x4"], float) for cd in _cam_dicts] if all("extrinsic_w2c_4x4" in cd for cd in _cam_dicts) else None
            _K = [np.asarray(cd["intrinsic_3x3"], float) for cd in _cam_dicts] if all("intrinsic_3x3" in cd for cd in _cam_dicts) else None
            Pw, stab_rep = stabilize_actor_stances(
                Pw, shape[pid], source_frames, np.searchsorted(_cuts, source_t + .001, side="right"),
                bc, fps, UP, _act_col, dev=dev, cams_w2c=_w2c, cams_K=_K, seated=_is_seated
            )
            if stab_rep:
                p = M.root_pos(Pw)
                W, _ = M.forward(shape[pid], Pw, True, dev)
                delta = np.zeros_like(delta)
        if collider is not None and collider.pinned_supports:
            info[pid]["pinned_supports"] = collider.pinned_supports
        feet = np.array([B.clearance(w, UP, GH) for w in W + delta[:, None, :]])
        B.TERRAIN = _orig_terrain
        step = np.linalg.norm(np.diff(p, axis=0), axis=1) if T > 1 else np.zeros(1)
        if T > 1:
            step[seg[1:] != seg[:-1]] = 0.0
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
                        可信帧比例=round(float((conf >= 1.0).mean()), 2), 看得到脚的帧比例=round(float(feet_ok.mean()), 2))
        rep[pid]["可疑"] = B.suspicious(R_own, R_all, R, rep[pid]["可信帧比例"], float(step0.max()))
        if leg:
            rep[pid]["腿长"] = leg
        if dist_keep:
            rep[pid]["远近用模型的帧"] = [int(n_keep), T]
        rep[pid]["分段"] = dict(段数=int(seg.max()) + 1, 切点=len(cuts), 坏相机帧=int(bc.sum()))
        if height_report:
            rep[pid]["站姿高度校准"] = height_report
        if seat_rep:
            rep[pid]["坐姿修正"] = seat_rep
        if keep_rep:
            rep[pid]["坐着腿照原样"] = keep_rep
        if sit_move > B.SEAT_STILL_M:
            rep[pid]["坐着但在动"] = round(sit_move, 2)
        if legs_rep:
            rep[pid]["补腿"] = {k: v for k, v in legs_rep.items() if not k.startswith("_")}
        if ground_rep:
            rep[pid]["地面约束"] = ground_rep
        if root_registration:
            rep[pid]["根轨迹配准"] = root_registration
        if clamp_rep and (clamp_rep["改的帧"] or clamp_rep["太深不修的帧"]):
            rep[pid]["腿收出地面"] = clamp_rep
        if guess_rep:
            rep[pid]["猜的腿去抖"] = guess_rep
        if leg_runs:
            rep[pid]["腿估翻补帧"] = [dict(d, 视频帧=[int(kept[bis[a + 1]]), int(kept[bis[b - 1]])]) for a, b, d in leg_runs]
        if stab_rep:
            rep[pid]["支撑与防滑"] = stab_rep
    if pair_arm_on and len(out) >= 2:
        from pair_arm_repair import repair_pair_arm_penetration
        arm_rep = repair_pair_arm_penetration(out, fps, dev=dev)
        if arm_rep:
            for pid in out:
                mine = {k: v for k, v in arm_rep.items() if k.startswith(f"p{pid}_")}
                if mine:
                    rep[pid]["双人穿模局部修复"] = mine
    return out, info, rep


FILL_MAX_S = 1.5      # 秒: 同一个人中间缺帧(被挡住、mask 只剩碎片没出人)不超过这么久, 用前后两帧插值补上
FILL_CONF = 0.1       # 补出来的帧的可信度(比翻转补帧的 B.CONF_SUB 还低: 这几帧模型一眼都没看到他)
# ★ 2026-11-06 goodfellas_tray 人 4(端托盘的厨师)从女人身后走过, 被挡住的 13 帧 mask 只剩一条, 碎片帧不出人之后
#   他就消失了 0.3 秒再出现(闪)。被挡住不等于不在: 短空档按前后插值, 长空档(真出画、走开又回来)照旧不补。


def fill_gaps(r, kept, vid2key, cams, fps, max_s=FILL_MAX_S):
    """一个人的结果(solve 出的 dict)里, 中间不超过 max_s 秒的缺帧按前后两帧插值补上:
    根位置按时间线性、根朝向球面插值, 关节参数和骨骼缩放线性。返回 (新的 dict, 补了哪几段 [(头, 尾) 原视频帧号])。"""
    from scipy.spatial.transform import Rotation, Slerp
    bis, P, conf = list(r["frames"]), r["params"], r["conf"]
    nb, nP, nc, segs = [], [], [], []
    for i, b0 in enumerate(bis):
        nb.append(b0); nP.append(P[i]); nc.append(conf[i])
        if i + 1 == len(bis):
            break
        b1 = bis[i + 1]
        t0, t1 = kept[b0] / fps, kept[b1] / fps
        if b1 - b0 <= 1 or t1 - t0 > max_s:
            continue
        Q = M.root_rot(P[i:i + 2]); pos = M.root_pos(P[i:i + 2])
        sl = Slerp([t0, t1], Rotation.from_matrix(Q))
        add = [b for b in range(b0 + 1, b1) if b < len(kept) and kept[b] in vid2key and vid2key[kept[b]] in cams]
        for b in add:
            t = kept[b] / fps; w = (t - t0) / (t1 - t0)
            q = (1 - w) * P[i] + w * P[i + 1]
            nb.append(b); nc.append(FILL_CONF)
            nP.append(M.set_root(q[None], sl([t]).as_matrix(), ((1 - w) * pos[0] + w * pos[1])[None])[0])
        if add:
            segs.append((int(kept[add[0]]), int(kept[add[-1]])))
    o = np.argsort(nb)
    nb = np.array(nb)[o]
    out = dict(r, frames=nb, vf=np.array([kept[b] for b in nb]), params=np.array(nP)[o], conf=np.array(nc)[o])
    return out, segs


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
    ap.add_argument("--dist-keep", choices=["on", "off"], default="off",
                    help="远近: 模型的距离和场景深度差不到 DIST_KEEP_TOL 就用模型的(人和原图对齐); off = 每帧都按场景深度")
    ap.add_argument("--legs-proc", choices=["on", "off"], default="on",
                    help="补腿(legs_proc): 腿看不见 / 贴地在滑的段规划脚印 + IK, 全程看得见和坐着的不动; 默认一直开")
    ap.add_argument("--people-n", type=int, default=0,
                    help="人数上限(本轮配置 PEOPLE_N; 0 = 不知道几个人)。见 people_identity.cap_people")
    ap.add_argument("--leg-fit", choices=["on", "off"], default="off",
                    help="看不到脚的站着的人按'脚落在地上'定腿长, 上半身不动(见 fit_leg_length); off = 整个人往下拉贴地")
    a = ap.parse_args()
    try:                                                         # 见 FLIP_KEEP: 按轮次配置
        FLIP_KEEP.update(parse_flip_keep(json.load(open(f"{a.r1}/00_run_config.json")).get("覆盖", {}).get("FLIP_KEEP", "")))
    except (OSError, ValueError):
        pass
    if FLIP_KEEP:
        print("FLIP_KEEP(这几帧信模型, 不当翻转):", FLIP_KEEP)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fj = json.load(open(f"{a.body}/00_frames/frames.json")); kept, fps = fj["kept_indices"], fj["src_fps"]
    da3, cams, UP, GH, vid2key = B.load_scene(a.r1, dense=bool(fj.get("dense")))   # 人体全帧时中间帧的相机(dense_cams.py)
    # 跟踪给的编号不等于人: 先按房间里的位置核对一遍(并掉同一个人的几个编号, 去掉不是人的), 修的是核对过的那份
    print("== 核对身份 (见 people_identity.py) ==")
    raw = people_identity.raw_run(a.body)
    if not os.path.isdir(f"{raw}/mhr_4d_individual"):
        raise SystemExit(f"[失败] {raw} 没有模型的关节数据(mhr_4d_individual; 2026-10-08 之前出的人体网格没存) —— 重跑 people_mesh")
    names = people_identity.track_ids(raw)
    depth_checks = {}
    tracks = B.identity_tracks(raw, da3, cams, vid2key, kept, fps, depth_checks=depth_checks)

    def continuous_at_scene_depth(rid, a, b):
        # Recheck only a suspected jump. A second person or discontinuous mask
        # makes interpolated depth insufficient evidence for retaining the ID.
        from PIL import Image
        points = depth_checks[rid]
        if a not in points or b not in points or not (points[a][1] or points[b][1]):
            return False
        d = points[b][0] - points[a][0]
        if np.linalg.norm(d - (d @ UP) * UP) >= people_identity.SAME_M:
            return False
        masks = []
        for frame in (a, b):
            path = f"{raw}/masks/{frame:08d}.png"
            if not os.path.exists(path):
                return False
            with Image.open(path) as img:
                masks.append(np.array(img))
        return B.isolated_mask_pair(*masks, rid)

    frag_log, lost = [], {}
    for rid, bad in B.fragment_frames(raw, kept, fps).items():         # 碎片帧不出人(见 B.FRAG_*)
        lost[rid] = list(bad)
        bad = [b for b in bad if b in tracks.get(rid, {})]
        for b in bad:
            del tracks[rid][b]
        if not tracks.get(rid):
            tracks.pop(rid, None)
        for s0 in np.split(np.array(bad), np.where(np.diff(bad) > 1)[0] + 1) if bad else []:
            frag_log.append(dict(事="不算人", 编号=str(names.get(rid, rid)), 帧=[int(x) for x in s0],
                                 原因=f"mask 只剩一小块(不到画面 {B.FRAG_ABS:.1%}, 或不到他前后 {B.FRAG_WIN:.0f} 秒平常大小的 "
                                      f"{B.FRAG_REL:.0%}), 多半已经不是他"))
    # ★ 2026-11-11 位置不可信的帧不按瞬移切人: 坏相机的帧 + 隔着切点(jw4 甄子丹坐地上的特写, 坏相机那几帧位置一跳 4.1 m,
    #   被切成两个人, 验收视频 10 秒黄变粉)。见 people_identity.resolve 的 unreliable / shot_of
    _cuts_id = np.asarray(B.shot_cuts(da3), float); _badk = B.load_badcam(da3)
    _unrel = {b for b in range(len(kept)) if kept[b] in vid2key and vid2key[kept[b]] in _badk}
    who, id_log = people_identity.resolve(tracks, UP, name=names, taken=people_identity.all_track_ids(a.body, raw), lost=lost,
                                          unreliable=_unrel, max_people=a.people_n, check_depth_jump=continuous_at_scene_depth,
                                          shot_of=lambda b: int(np.searchsorted(_cuts_id, kept[b] / fps + 1e-3, side="right")))
    # ★ 2026-11-11 "几帧、很小"的误检不当人(joker 墙头的植物、jw4 头上的红斑), 判据见 B.tiny_people
    for pid, d in B.tiny_people(raw, who, kept, fps).items():
        for rid in sorted({r for r, _ in who[pid]}, key=str):
            frag_log.append(dict(事="不算人", 编号=str(names.get(rid, rid)), 帧=sorted(int(b) for r, b in who[pid] if r == rid),
                                 原因=f"核对完只有 {d['帧']} 帧({d['秒']} 秒), 掩码面积中位只占画面 {d['掩码面积中位_pct']}%"
                                      f"(< {B.TINY_AREA:.1%}): 远处的小东西被认成了人"))
        who.pop(pid)
    id_log = sorted(id_log + frag_log, key=lambda x: (int(x["编号"]), x["帧"][0]))
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
    cuts = B.shot_cuts(da3)
    print(f"  [切镜头] {len(cuts)} 个切点" + (f": {', '.join(f'{c:.1f}' for c in cuts)} 秒" if cuts else "(一镜到底)") + "; 坏相机 "
          f"{len(B.load_badcam(da3))} 帧 —— 平滑不跨段, 坏相机的帧用模型自己的位置", flush=True)
    # 没坐在椅子正上方的人按"脚着地", 选腿时的目标落差用场景里椅面高度的中位数
    seat_h0 = float(np.median([s[0] for s in seats.values()])) if seats else 0.45
    res, info, rep = solve(par, shape, cams, vid2key, kept, UP, GH, mesh_run, da3, fps, B.load_focals(mesh_run), dev,
                           chairs=chairs, seats=seats, seat_h_default=seat_h0, leg_fit=a.leg_fit == "on",
                           cuts=cuts, badcam=B.load_badcam(da3), dist_keep=a.dist_keep == "on",
                           legs_proc_on=a.legs_proc == "on", scene_in=a.scene_in, r1_dir=a.r1)
    if not res:
        raise SystemExit("[失败] 人体的帧和场景的相机一帧都对不上 —— 检查是不是同一段素材")
    any_gaps = False
    for pid in res:                                                        # 中间被挡住的短空档补上(见 fill_gaps)
        res[pid], segs = fill_gaps(res[pid], kept, vid2key, cams, fps)
        rep[pid]["补上被挡住的帧"] = [dict(帧=[a_, b_]) for a_, b_ in segs]
        info[pid]["frames"] = [int(b) for b in res[pid]["frames"]]
        if segs:
            any_gaps = True
            print(f"  人 {pid}: 中间缺帧补上 {', '.join(f'第 {a_}~{b_} 帧' for a_, b_ in segs)}(不超过 {FILL_MAX_S} 秒的空档, 前后插值)")
    if any_gaps and len(res) >= 2:
        from pair_arm_repair import repair_pair_arm_penetration
        repair_pair_arm_penetration(res, fps, dev=dev)
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
        if q.get("远近用模型的帧"):
            print(f"         远近: {q['远近用模型的帧'][0]}/{q['远近用模型的帧'][1]} 帧用模型自己的距离(和场景差不到 {DIST_KEEP_TOL:.0%})")
        if q.get("腿长"):
            L = q["腿长"]
            print(f"         腿长: " + (f"不动({L['不动']}; 脚离地 {L['脚离地_前']:+.2f} m)" if "不动" in L else
                  f"伸长 {L['腿伸长_米'] * 100:+.0f} cm(参数 {L['腿长参数改动']:+.2f}{', 封顶' if L['封顶'] else ''}), "
                  f"脚离地 {L['脚离地_前']:+.2f} -> {L['脚离地_后']:+.2f} m, 身高 {L['身高_后']:.2f} m"))
        if q["可疑"]:
            print(f"         ★可疑: {'; '.join(q['可疑'])} —— 看验收那道闸\"人\"那张表里他的截图, 不是人就点掉")
        if q["肢体跳变修复"]:
            print("         肢体跳变: " + "; ".join(f"{n} {' '.join(f'{a}~{b}' for a, b in v)}" for n, v in q["肢体跳变修复"].items()))
        for g in q["翻转补帧"]:
            print(f"         翻转补帧: 第 {g['帧'][0]}~{g['帧'][1]} 帧({g['帧数']} 帧, 段里最大一翻 {g['最大一翻']}°) {g['补法']}")
        for r in q.get("腿估翻补帧", []):
            print(f"         腿估翻: 视频第 {r['视频帧'][0]}~{r['视频帧'][1]} 帧脚落到自己分割外面(最差 {r['最差一帧落在分割上']}), 两条腿换成前后过渡")
        g2 = q.get("地面约束")
        if g2:
            print(f"         地面约束: 抬出地面 {g2['抬出地面的帧']} 帧(最多 {g2['最多抬_米'] * 100:.0f} cm)"
                  + (f", 坏相机贴地 {g2['坏相机贴地的帧']} 帧(挪 {g2['坏相机挪动_米'][0] * 100:+.0f}~{g2['坏相机挪动_米'][1] * 100:+.0f} cm)" if g2["坏相机贴地的帧"] else ""))
        g3 = q.get("猜的腿去抖")
        if g3:
            print(f"         猜的腿去抖: {g3['帧']} 帧(腿看不见), 膝盖逐帧最大跳 {g3['膝盖逐帧最大跳_度_前后'][0]}° -> {g3['膝盖逐帧最大跳_度_前后'][1]}°")
        c2 = q.get("腿收出地面")
        if c2:
            print(f"         腿收出地面: 看不见的脚在地下 {c2.get('脚在地下的帧', 0)} 帧"
                  + (f"(要抬中位 {c2['要抬_cm']['中位']} / 最多 {c2['要抬_cm']['最大']} cm)" if c2.get("要抬_cm") else "")
                  + f", 改了 {c2['改的帧']} 帧的腿(身体不动), 改后最低 {c2.get('改后最低_cm', 0)} cm"
                  + (f", 太深不修 {c2['太深不修的帧']} 帧" if c2["太深不修的帧"] else ""))
        L2 = q.get("补腿")
        if L2:
            print(f"         补腿: {L2.get('类')} 类" + (f", 不补({L2['不补']})" if L2.get("不补") else
                  f", 补 {L2.get('补的帧')}/{L2.get('帧数')} 帧, IK 点误差中位 {L2['IK_点误差_cm']['中位']} cm, 骨盆最多压低 {L2['骨盆压低_cm']['最大']} cm"))
        k = q.get("坐着腿照原样")
        if k:
            print(f"         坐着、脚看得到: 腿照原样, 屁股按椅面 {k['椅面离地']} m 定高度  屁股离地 {k['屁股离地_前']} -> {k['屁股离地_后']}"
                  f"  脚离地 {k['脚离地_前']} -> {k['脚离地_后']}" + (f"  ★脚扎地 {k['脚扎地_帧数']} 帧" if k["脚扎地_帧数"] else ""))
        if q.get("坐着但在动") is not None:
            print(f"         坐着(竖直跨度小)但没坐在椅子上、骨盆一直在动(离开中心 {q['坐着但在动']} m > {B.SEAT_STILL_M} m): 蹲着 / 爬, 不做坐姿修正")
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
    pinned = sorted({c["chair"] for c in info.values() if c["chair"]}
                    | {s for c in info.values() for s in c.get("pinned_supports", ())})
    pinned_sup = sorted({s for c in info.values() for s in c.get("pinned_supports", ())})
    meta = dict(floor_y=GH, pinned_chairs=pinned, pinned_supports=pinned_sup, people=info, 报告=rep,
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
