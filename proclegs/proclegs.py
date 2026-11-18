"""程序化补腿(rohm 环境, 有 smplx), 做法见 README.md。
输入 RoHM/ours/inputs/p<pid>_rohm_input.npz(原版结果拟合的 SMPL-X, 世界 W: z 朝上, 地面 z=0, 30 fps)
输出 outputs/p<pid>_proclegs<tag>_W.npz: SMPL-X 参数、顶点、22 关节、每帧两踝位置/朝向/俯仰/是否踩地、步子列表
用法: /home/ubuntu/miniforge3/envs/rohm/bin/python proclegs.py --pid 4 [--tag _v3]
"""
import argparse, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import gaussian_filter1d, minimum_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROHM = os.path.join(os.path.dirname(HERE), 'RoHM', 'ours')      # 原来写死 /home/ubuntu/sam3d, 换机器就找不到
sys.path.insert(0, ROHM)
from rohm_ours import smplx_model, body_joints

FPS = 30
# Murakami & Otaka 2017 (J Phys Ther Sci 29(4)), 表 1: 21 名健康成人(平均身高 1.667 m) 速度(m/min) -> 步长(m)
WR_SPEED = np.array([11.4, 20.9, 49.1, 79.0, 118.2]) / 60
WR_STEP = np.array([0.43, 0.47, 0.58, 0.70, 0.87])
WR_HEIGHT = 1.667
# 以下是假设, 看结果再调
SWING = (0.35, 0.65)         # 往前走的迈步时长 s(横着/倒着再乘 TS_DIR); 试过上限 0.75: 测出来几乎不变, 骨盆反而多沉 1.6 cm
TOE_OUT = np.radians(7)      # 脚尖外八
DS_MIN = 0.10                # 两脚都着地最短 s
DROP_MAX = 0.04              # 规划步长时: 骨盆最多压低多少 m
REACH_HEEL = np.radians(15)  # 规划步长时: 后脚离地前脚跟抬这么多(真人动捕往前走离地俯仰 +15°)
TOE_IN_MAX = np.radians(10)    # 落脚时脚尖相对骨盆: 最多内八这么多
TOE_OUT_MAX = np.radians(40)   # 最多外八这么多(转身时转向那一侧的脚先打开)
TOE_IN_STEP = np.radians(15)   # 踩着的脚因为身体转动变成内八超过这么多就迈
TOE_OUT_STEP = np.radians(45)  # 外八超过这么多就迈
PIVOT_IN, PIVOT_OUT = np.radians(20), np.radians(50)   # 来不及迈时, 踩着的脚超过这个内八/外八就绕前掌转(前掌不动, 脚跟转)
FEET_YAW_MAX = np.radians(70)  # 两只脚尖之间最多差这么多
KNEE_TO_PELVIS = 0.0           # 膝盖朝向 = 脚尖朝向(膝盖对着脚尖; v6 往骨盆偏 30% 会让外八的脚膝盖内扣)
PELVIS_YAW_SIGMA = 0.3         # 骨盆朝向平滑 s(上半身用腰部关节补回, 世界里不变)
# 下面按真人动捕量(AMASS 2300 步, mocap_stats.py / mocap_summary.py); 速度按身高归一(身高/s), 这个人大多在 0.15–0.30
# 三个一组的都是 (往前, 横着, 倒着), 按移动方向和朝向的夹角插值
SIDE_STEP = 0.72               # 横着走一次挪的距离 = 往前的 0.72 倍(真人 0.34 vs 0.47 身高)
BACK_STEP = 0.83               # 倒着走 0.83 倍(0.39 vs 0.47)
TS_DIR = (1.0, 0.6, 0.85)      # (v10 以前用) 迈步时长倍数
# v11 起步长和迈步时长直接查真人表(bench.py 里 134 段 AMASS 走路, 同一个检测方法量的; 速度档中点, 身高/s)。
# 真人走快了是把步子迈大, 迈步时长差不多(往前 0.5–0.6 s); v10 步子被"腿能伸多远"卡住, 走快时只能 0.3 s 一步, 腿甩得急 = 抖
V_TAB = [0.075, 0.225, 0.375, 0.525, 0.75, 0.95]      # 延伸到快走(0.95 = 1.7 m/s): 原来封顶 0.75, 人走快了脚跟不上, 骨盆被压到地上
STRIDE_TAB = ([0.36, 0.48, 0.56, 0.68, 0.83, 1.04], [0.24, 0.34, 0.36, 0.42, 0.55, 0.75], [0.19, 0.41, 0.46, 0.62, 0.73, 0.94])   # 一次挪(身高), 横着的样本少, 平滑过
TS_TAB = ([0.50, 0.58, 0.54, 0.53, 0.53, 0.53], [0.35, 0.40, 0.38, 0.38, 0.45, 0.45], [0.38, 0.50, 0.48, 0.50, 0.50, 0.55])       # 迈步时长 s(检测方法量的)
DS_MIN_TAB = [0.14, 0.12, 0.08, 0.06, 0.04, 0.03]  # 两脚都着地最短 s, 按速度(真人两脚着地: 慢 0.40, 中 0.23/0.13, 快 0.07 s); 原来固定 0.1, 快走时后脚被拉远、两脚叉开
DS_MID_TAB = [0.15, 0.15, 0.11, 0.065, 0.035, 0.025]   # 对称点用落地后多久的骨盆 = 真人两脚着地时长的一半(0.40/0.23/0.13/0.07 s); 原来固定 0.1, 快走时落点被推远
CYCLE_TAB = ([1.77, 1.83, 1.50, 1.30, 1.17, 1.03], [1.75, 1.60, 1.35, 1.23, 1.15, 1.03], [1.85, 1.53, 1.33, 1.23, 1.03, 0.98])  # 真人步态周期 s(迈步+踩地, 和检测方法无关)
SWING_FRAC = [0.34, 0.35, 0.37, 0.38, 0.40, 0.40]  # 迈步占步态周期的比例(步态教科书: 正常速度约 40%, 慢走更少)。
                               # 不再用检测方法量出来的迈步时长校准: 它对真人偏长(脚跟着地、蹬地时脚在动)、对我们偏短(迈步两头脚慢), 校准越调越长
BEAT_V = (0.25, 0.45)          # 走动时按节奏抬脚的权重: 速度(等效身高/s)从 0.25 到 0.45 由 0 过渡到 1。
                               # 快走时按"落后够一步才抬"会抬晚: 踩地 0.83 s(真人 0.67), 周期 1.33 s(真人 1.17), 步子大 14%, 两脚叉开、腿甩得急
LEAD_RATIO = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # 落地时 前脚离骨盆 / 后脚离骨盆。v22 用过真人值 [0.85 .. 0.62](走快了前脚近后脚远), v23 考场上 1.0 叉开更少
# 试过快走时骨盆高度用短窗口(让骨盆随步子起伏): 骨盆猛升降 0 -> 20 次/分钟、整体顿挫 1.3 -> 14, 撤回(v14)
Z_DROP_TOTAL = 0.12            # 安全底线: 骨盆最多比站直低这么多; 再够不着就让脚够不着(离地一点), 不能蹲到地上
K_STRIDE = 1.0
TS_ADD = 0.08
# 上面两张是"量出来的"真人值; 规划时要用的值由 calib_gait.py 按考场结果自动校准, 存在 gait_tables.json(有就用它)
STRIDE_MEAS = STRIDE_TAB       # 真人量出来的(校准不许偏离它太多, 见 calib_gait.py)
GAIT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gait_tables.json')
USE_GAIT_JSON = 0              # v23: 不用校准表。v22 的校准把慢速步长调到真人的 2.7 倍(迈不出来 -> 越调越大), 正常抬脚规则不触发, 脚总被拖到够不着才迈
if USE_GAIT_JSON and os.path.exists(GAIT_JSON):
    import json as _json
    _g = _json.load(open(GAIT_JSON))
    STRIDE_TAB, TS_TAB, K_STRIDE, TS_ADD = tuple(_g['stride']), tuple(_g['ts']), 1.0, 0.0
V_REF = [0.075, 0.225, 0.375, 0.525]                 # 速度档中点(身高/s)
HIP2H = 1.813                  # 动捕骨架 身高/髋高(1.72/0.949)。步态按腿长比(髋高), 不按身高: SMPL-X 髋关节偏高在骨盆里, 髋高/身高 0.48(动捕 0.55),
                               # 按身高比会让 SMPL-X 的人步子偏大、骨盆被压低。所有"身高"单位的表都乘 body.H = 髋高 x 1.813
MH_HEIGHT = 1.72                                     # 动捕骨架身高, 抬脚高度按身高换算
LIFT_TAB = ([8.4, 13.6, 14.8, 17.4], [8.2, 8.2, 7.8, 8.5], [5.4, 6.5, 7.0, 12.7])   # 踝最高离"放平时的高度" cm
LIFT_PEAK = (0.25, 0.45, 0.79)                       # 踝最高点在迈步的什么时刻: 往前早、倒着晚(倒着 ≈ 往前倒放)
LAND_TAB = ([-2.1, -4.3, -5.4, -10.2], [3.7, 3.9, 3.8, 3.1], [5.6, 6.6, 7.1, 11.5])  # 落地时脚的俯仰°: <0 脚跟先(脚尖翘), >0 脚尖先
OFF_TAB = ([10, 10, 17, 26], [7, 7, 6, 8], [1, 1, -3, -1])                         # 离地前脚的俯仰°: >0 脚跟先抬, <0 脚尖先抬
SETTLE_T = (0.08, 0.04, 0.08)  # 落地后多久放平 s
OFF_RAMP = 0.15                # 离地前多久开始抬脚跟/脚尖 s
BOB_TAB = [0.4, 1.1, 1.7, 2.8]  # 两脚着地时骨盆比单脚着地中点低多少 cm
DS_MID = 0.10                  # 走动时, 沿移动方向用"对称"定迈步: 这只脚若现在抬、落地后 DS_MID 秒(两脚着地那段中间)时,
                               # 它该落的位置 = 另一只脚关于骨盆的对称点; 这个点离它现在 >= 一步该有的长度就迈, 落在那里。
                               # 两脚着地时骨盆就在两脚中间(真人横着走中位偏 0.09; v8 0.33)。v8 按"离自己的标准位置落后多少"判断,
                               # 标准位置左右各偏 8 cm, 横着走时这个偏移正好在移动方向上: 前脚要等骨盆越过它才迈, 后脚被放到骨盆前面。
                               # v9 试过直接落在对称位置: 起步时两脚都在身体下方, 对称出来就不往前迈, 后面被迫迈大步, 骨盆最多多沉 9 cm
ALT_RECENT = 0.6               # 上一步落地不到这么久才强制左右轮流(9 s 处停了 1.2 s 再横挪, 轮流规则让后脚先迈, 骨盆越过了前脚)
EDGE_M = 0.02                  # 骨盆(左右方向)离某只脚不到 2 cm、快越过它 -> 那只脚往外迈(慢慢横挪时重心快出脚的范围; 14–15 s 两脚曾都在骨盆一侧)
TS_MIN = 0.28                  # 迈步最短 s(v8 有一步 0.20 s, 看起来是一下弹过去)
PRE_SWING = 0.17               # 往前/横着迈步离地前这么久内, 抬脚跟(绕前掌, 不滑)让膝盖先弯到迈步开头的角度(真人离地时膝盖已弯 ~30°; 不这样离地那一帧膝盖会从 10° 跳到 30°)
PEL_XY_SIGMA = 0.07            # 骨盆水平位置轻度平滑 s: 原始输入在 15–16.5 s(骨盆浮起那段)有 1–1.7 cm 的顿挫, 整个人跟着一跳
SWING_V_END = 1.5               # 迈步起止速度最多取这么大 m/s(每个分量)
KNEE_SMOOTH = 0.10             # 迈步时膝角按时间平滑(高斯 s), 两头权重 0: 伸直附近脚差 1 cm 膝盖就差十几度, 不平滑会一帧跳十几度
                               # 0.10 时膝角变化速度和真人一样(95% 160 vs 154 °/s), 代价是抬脚比真人低约 20%; 试过按差额补高度: 抬够了但又抖(195 °/s)
LIFT_AL_MAX = 4.0              # 抬脚曲线 u^al 的 al 上限: 倒着走最高点不要太靠后(否则前半段腿被拉直, 后半段膝盖猛弯 = 抖)
Z_MIN_WIN, Z_SIGMA = 15, 6     # 骨盆高度: 先取 0.5 s 窗口最小值, 再按 0.2 s 平滑
KNEE_STAND = np.radians(12)  # 站直时膝盖也弯 12°(从 SMPL-X 静止姿势算起); 腿最多伸到这个程度。5° 时太直, 脚跟抬 7° 膝盖就跳 16°(膝角对距离的敏感度和 sin(弯角) 成反比); 12° 骨盆只低约 0.5 cm
STILL = 0.08                 # 低于这个速度算站着 m/s
SETTLE = 0.10                # 站着时脚离该在的位置超过这么多就挪一步 m
LOOK_MAX = 1.0               # 落脚点最多往后看多少秒的路线(只用于沿移动方向; 左右按两脚着地那一刻的骨盆)
LAT_ACC_K = 0.04             # 转弯/横向加速时脚往外放: 两脚中点在骨盆的反方向 0.04 m 每 1 m/s^2(measure_lateral.py: 真人 -1/+1 m/s^2 -> +0.046/-0.052 m)
                             # v22 及以前左右也按 0.4 s 后的骨盆: 走弯路时脚被放到内侧 10-15 cm(真人在外侧), 骨盆偏一侧/叉开多 2-3 倍
FEET_GAP = 0.12              # 落脚时两踝左右至少隔开 m(两脚并排时; 一前一后时只要不交叉: FEET_GAP_WALK)
FEET_GAP_WALK = 0.02         # 两脚前后隔开 >= 0.25 m 时左右至少隔开这么多(真人往前走两脚着地 5% 分位 ~0.01 m)
WIDTH_V = [0.095, 0.225, 0.375, 0.525, 0.75]                       # 走路时两踝左右间距(和移动方向垂直, 两脚着地时中位), 速度档(等效身高/s)
WIDTH_TAB = ([0.173, 0.100, 0.081, 0.089, 0.089], [0.184, 0.164, 0.127, 0.130, 0.148])   # (往前, 倒着) m, 已是 office p4 身材(measure_lateral.py)
                             # v22 及以前走路也用站着的间距(0.156 m) + 至少隔 0.12 m: 往前走真人 0.09 m, 我们 0.17 m(2 倍)
HEEL_OFF_MAX = np.radians(35)  # 脚在身体后面够不着时: 脚跟最多抬这么多(绕前掌转)
TOE_UP_MAX = np.radians(20)    # 脚在身体前面够不着时: 脚尖最多翘这么多(绕脚跟转)
TOE_UP_PLAN = np.radians(15)   # 算骨盆高度时按翘 15° 算(倒着走时前脚最后是绕脚跟翘脚尖离地)

EDGE_GATE = 0.3              # 走动时"骨盆快越过这只脚"只在移动方向朝这只脚那侧的分量 >= 这个值时管(0.3 = 偏 17° 以上; v22 0.5)
LAT_FIX = 0                  # 试过没留下(考场/office 都没变好), 留着开关: 1: 落点左右按两脚着地那一刻的骨盆 + 加速度往外放(v23); 0: 按 0.4 s 后的骨盆(v22)
WIDTH_ON = 0                 # 试过没留下(叉开少了但骨盆偏一侧多了), 留着开关: 1: 走路时两脚左右间距查真人表(v23); 0: 用站着的(v22)
GAIT_V10 = 2                 # v23 = 2(慢走 v10 算法, 快走查真人表, 见 gait()); 1: 步长/迈步时长按 v10 的算法(Murakami 步长表 + 腿能伸多远封顶, 步子短了迈得快)
GAIT_BLEND = (0.35, 0.55)    # GAIT_V10 = 2 时的过渡速度(等效身高/s)
REACH_V10 = 1                # v23 = 1(v22 的 0 让 office 1.8 s 骨盆多沉 5 cm); 1: "脚够不着就迈"按 v10: 往后伸也按 0.30 m(不按抬脚跟 35° 的 0.38 m), 按半步之后的身体位置判断
# 试参数: 环境变量 PROCLEGS_OPT='{"EDGE_GATE": 0.7}' 覆盖上面任何一个大写常量(gate.py --opt 用它; 子进程会继承)
if os.environ.get('PROCLEGS_OPT'):
    import json as _json
    globals().update(_json.loads(os.environ['PROCLEGS_OPT']))

FP_TOL = 0.03                 # 接观测: 看不见的段里一步没迈过的脚, 离段尾观测的脚印超过这么多 m 才加一步过去
AIR_EARLY = 0.25              # 接观测: 段尾在空中的那只脚, 第一遍那步的离地比估计的早不超过这么多 s 就照第一遍的离地(迈得慢一点), 不推后
JOIN_W = 0.3                  # 看得见/看不见交界处: 观测和规划之间剩下的差(俯仰、踝位置)在这么多 s 内渐变掉

LEGS = [(1, 4, 7, 10), (2, 5, 8, 11)]   # 左, 右: 髋 膝 踝 脚尖(SMPL-X 关节号)
SIDE = np.array([1.0, -1.0])            # 模型 x 轴朝人的左边


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def runs_of(mask):
    """True 的连续段 [(开始, 结束(不含))]。"""
    m = np.concatenate([[0], np.asarray(mask, int), [0]]); e = np.flatnonzero(np.diff(m))
    return list(zip(e[::2], e[1::2]))


def make_obs(vis, c, ank, fyaw, phi):
    """腿看得见的帧的观测 -> plan/run 的 obs。vis (T,) 这帧腿看得见; c (T,2) 踩地; ank (T,2,3) 踝(SMPL-X 踝高约定, 放平时离地 body.ankle_h);
    fyaw (T,2) 脚朝向(proclegs 约定, 同 foot_pose); phi (T,2) 俯仰(>0 脚跟抬起)。
    每段踩地 = 一个脚印: 放平帧的踝水平位置中位数, 之后钉住(观测的脚滑在这里去掉)。"""
    vis = np.asarray(vis, bool); T = len(vis); c = np.asarray(c, bool) & vis[:, None]
    run = -np.ones((T, 2), int); fp = []
    land_k = np.full((T, 2), T); off_k = np.full((T, 2), -1)
    for i in (0, 1):
        for a, b in runs_of(c[:, i]):
            m = np.abs(phi[a:b, i]) < np.radians(5)
            m = m if m.sum() >= 2 else np.ones(b - a, bool)
            yw = fyaw[a:b, i][m]
            fp.append((np.median(ank[a:b, i, :2][m], 0), float(np.arctan2(np.sin(yw).mean(), np.cos(yw).mean())), a, b))
            run[a:b, i] = len(fp) - 1
        nxt, prev = T, -1
        for k in range(T - 1, -1, -1):                   # 迈步帧: 下一次落地的帧
            nxt = k if c[k, i] else nxt
            land_k[k, i] = nxt
        for k in range(T):                               # 迈步帧: 这次离地的帧(没看到离地 = -1)
            prev = k + 1 if c[k, i] else prev
            off_k[k, i] = prev
    sw = [(b - a) / FPS for i in (0, 1) for a, b in runs_of(vis & ~c[:, i]) if a > 0 and b < T and c[a - 1, i] and c[b, i]]
    return dict(vis=vis, c=c, ank=np.asarray(ank, float), fyaw=np.asarray(fyaw, float), phi=np.asarray(phi, float), run=run, fp=fp,
                land_k=land_k, off_k=off_k, ts_med=float(np.median(sw)) if len(sw) >= 2 else None)   # 看得见的完整迈步时长中位(这个人自己的)


def upright(psi):
    """模型坐标(x 左, y 上, z 前) -> W: 站直、面朝水平方向 psi 的人。"""
    f = np.array([np.cos(psi), np.sin(psi), 0.0]); up = np.array([0, 0, 1.0])
    return np.stack([np.cross(up, f), up, f], 1)


def rx(phi):
    return Rot.from_rotvec([phi, 0, 0]).as_matrix()


class Body:
    """静止姿势(全 0)下的几何量, 模型坐标, y 朝上。"""
    def __init__(self, model, betas):
        J, V = body_joints(model, {'global_orient': np.zeros((1, 3)), 'transl': np.zeros((1, 3)),
                                   'betas': betas[None], 'body_pose': np.zeros((1, 63))})
        self.J, V = J[0], V[0]
        sole = V[:, 1].min()
        self.height = V[:, 1].max() - sole
        hip_h = float(np.mean([self.J[h, 1] for h, _, _, _ in LEGS]) - sole)
        self.H = hip_h * HIP2H                                 # 查步态表用的"等效身高": 按髋高(腿长)换算, 和动捕骨架同比例
        self.pelvis_h = self.J[0, 1] - sole                  # 站直时骨盆离地
        self.ankle_h = float(np.mean([self.J[a, 1] - sole for _, _, a, _ in LEGS]))
        self.thigh = [np.linalg.norm(self.J[k] - self.J[h]) for h, k, _, _ in LEGS]
        self.shin = [np.linalg.norm(self.J[a] - self.J[k]) for _, k, a, _ in LEGS]
        self.hip_drop = float(np.mean([self.J[h, 1] - self.J[a, 1] for h, _, a, _ in LEGS]))   # 站直时髋比踝高多少
        rel = np.array([self.J[a] - self.J[0] for _, _, a, _ in LEGS])
        self.lat = float(np.mean(np.abs(rel[:, 0])))       # 站着时踝离骨盆的左右距离(左右取平均, 对称)
        self.fwd = float(np.mean(rel[:, 2]))               # 前后
        # 静止姿势的大腿/小腿向量。SMPL-X 站直时腿本来就不是一条直线(膝关节比髋、踝靠外几厘米),
        # IK 以它为准: 膝盖只绕自己的左右轴(x)弯, 不把这个弧度拧掉。
        self.t0 = [self.J[k] - self.J[h] for h, k, _, _ in LEGS]
        self.s0 = [self.J[a] - self.J[k] for _, k, a, _ in LEGS]
        self.Luse = [self.hip_ankle(i, KNEE_STAND) for i in (0, 1)]           # 腿最多伸到的髋-踝距离
        leg = min(self.Luse)
        # 脚掌上的两个支点(模型坐标, 在脚底高度): 脚跟 = 脚最靠后的点, 前掌 = 脚尖关节正下方
        w = model.lbs_weights.detach().cpu().numpy().argmax(1)
        self.heel, self.ball = [], []
        for _, _, a, toe in LEGS:
            ids = np.isin(w, [a, toe]) & (V[:, 1] < sole + 0.02)
            self.heel.append(np.array([self.J[a, 0], sole, V[ids, 2].min()]))
            self.ball.append(np.array([self.J[toe, 0], sole, self.J[toe, 2]]))
        # 踝离髋最远的水平距离: 骨盆压 DROP_MAX, 后脚离地前脚跟抬 REACH_HEEL(真人往前走离地时约 15°, 踝因此抬高)
        rise = abs(self.ball[0][2] - self.J[LEGS[0][2], 2]) * np.sin(REACH_HEEL)
        self.reach = float(np.sqrt(max(leg ** 2 - (self.hip_drop - DROP_MAX - rise) ** 2, 0)))
        self.reach_side = float(np.sqrt(max(leg ** 2 - (self.hip_drop - DROP_MAX) ** 2, 0)))   # 往侧面伸时抬脚跟帮不上
        rise_b = abs(self.ball[0][2] - self.J[LEGS[0][2], 2]) * np.sin(HEEL_OFF_MAX)
        self.reach_back = float(np.sqrt(max(leg ** 2 - (self.hip_drop - DROP_MAX - rise_b) ** 2, 0)))   # 脚在身体后面: 抬脚跟到 35°(真人快走后脚离骨盆 0.36 m)

    def hip_ankle(self, i, th):
        """膝盖从静止姿势弯 th 时, 髋到踝的距离。"""
        return float(np.linalg.norm(self.t0[i] + rx(th) @ self.s0[i]))

    def knee_for(self, i, D):
        """髋-踝距离 D -> 膝盖要弯多少(二分; 距离随弯曲单调变短)。"""
        lo, hi = 0.0, np.radians(170)
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if self.hip_ankle(i, mid) > D else (lo, mid)
        return (lo + hi) / 2


def gait_v10(v, body, u=None, psi=0.0):
    """v10 的步长/迈步时长(对照用): 步长查 Murakami 表、被腿能伸多远封顶、按方向缩; 迈步时长按缩完的步长算。返回 (Ts, d, lag 的分母)。"""
    scale, tscale = dir_scale(u, psi), dir_mix(u, psi, TS_DIR)
    reach = dir_mix(u, psi, (body.reach, body.reach_side, body.reach))
    L = np.interp(v, WR_SPEED, WR_STEP) * body.height / WR_HEIGHT
    if v < STILL:
        return max(SWING[1] * tscale, TS_MIN), min(L / 2, reach), L
    Ts = SWING[1] * tscale
    for _ in range(8):
        d = max(min(L - v * Ts / 2, reach) * scale, 0.05)
        Ts = float(np.clip(tscale * 0.8 * (d + v * Ts / 2) / v, max(SWING[0] * tscale, TS_MIN), SWING[1] * tscale))
    return Ts, d, 2 * d + v * Ts / 2


def gait(v, body, u=None, psi=0.0):
    """速度 v、移动方向 u、朝向 psi -> (迈步时长, 落脚时脚在身体前方多远 d, lag 的分母 = 一次挪多远)。
    GAIT_V10 = 2: 慢走用 v10 的算法(步子被腿能伸多远封顶, office 这类慢慢倒着/横着走不会被迫大步、骨盆不沉),
    快走用真人表(腿抖少一半); 速度(等效身高/s)在 GAIT_BLEND 之间线性过渡。"""
    if GAIT_V10 == 1:
        return gait_v10(v, body, u, psi)
    if GAIT_V10 == 2:
        w = float(np.clip((v / body.H - GAIT_BLEND[0]) / (GAIT_BLEND[1] - GAIT_BLEND[0]), 0, 1))
        if w <= 0:
            return gait_v10(v, body, u, psi)
        if w >= 1:
            return gait_tab(v, body, u, psi)
        return tuple((1 - w) * x + w * y for x, y in zip(gait_v10(v, body, u, psi), gait_tab(v, body, u, psi)))
    return gait_tab(v, body, u, psi)


def gait_tab(v, body, u=None, psi=0.0):
    """速度 v、移动方向 u、朝向 psi -> (迈步时长, 落脚时脚在身体前方多远 d, 一次挪多远)。直接查真人动捕表(bench.py 同一个检测方法量的),
    再换算成规划值(检测方法会把迈步两头脚贴地慢动的部分算成踩地, 量出来比规划的短)。"""
    vn = v / body.H
    tab = lambda T3: dir_mix(u, psi, tuple(np.interp(vn, V_TAB, x) for x in T3))
    stride = tab(STRIDE_TAB) * body.H * K_STRIDE
    Ts = max(np.interp(vn, V_TAB, SWING_FRAC) * tab(CYCLE_TAB), TS_MIN)        # 迈步时长 = 迈步占比 x 周期(都按真人)
    d = max((stride - v * Ts) / 2, 0.05)
    return Ts, d, stride


def dir_mix(u, psi, vals):
    """移动方向 u(单位向量, None = 站着) 和朝向 psi -> 在 (往前, 横着, 倒着) 三个值之间插值。"""
    if u is None:
        return vals[0]
    c = float(u @ np.array([np.cos(psi), np.sin(psi)]))
    f, sd, bk = vals
    return sd + ((f - sd) if c > 0 else (bk - sd)) * abs(c)


def dir_scale(u, psi):
    return dir_mix(u, psi, (1.0, SIDE_STEP, BACK_STEP))


def swing_style(A, B, psi, v, body):
    """一步的样子(按真人动捕): 踝最高多高、在什么时刻, 落地/离地时脚的俯仰, 落地后多少帧放平。"""
    dvec = B - A; dist = np.linalg.norm(dvec)
    u = dvec / dist if dist > 0.05 else None
    vn = v / body.H
    tab = lambda T3: dir_mix(u, psi, tuple(np.interp(vn, V_REF, t) for t in T3))
    H = tab(LIFT_TAB) / 100 * body.H / MH_HEIGHT * min(1.0, max(dist / 0.2, 0.4))   # 挪得很少的步抬得低
    return dict(H=H, up=dir_mix(u, psi, LIFT_PEAK), pl=np.radians(tab(LAND_TAB)), po=np.radians(tab(OFF_TAB)),
                settle=max(int(round(dir_mix(u, psi, SETTLE_T) * FPS)), 1))


def plan(t, pxy, yaw, body, obs=None, script=None):
    """离线规划脚印。pxy: 平滑后的骨盆水平位置, yaw: 平滑后的骨盆朝向。
    返回: 每帧两脚的脚印(踝的水平位置, 迈步中的帧是要落的那个)、脚朝向、是否踩地, 迈步列表, 速度。
    obs(make_obs, 可选, 腿有时看得见): 看得见的帧照观测(踩地的脚在观测脚印上); 看不见的段从观测的最后状态接着规划(还在空中的脚接着迈完);
    段尾接上下一段第一帧的观测, 分两遍: 第一遍照常规划(左右自然轮流); 再改尾巴(tail_script): 按观测定最后迈的是哪只脚
    (那时在空中的那只; 都踩着地 = 之后不先抬的那只), 它最后一步和另一只脚在它之前的最后一步落到观测脚印上(在空中的: 结束在观测的空中位置),
    后面多出来的步删掉; 第二遍照改好的步子表(script)走, 每帧的踩地/朝向/绕前掌转重新算。"""
    T = len(t)
    vel = np.gradient(pxy, t, axis=0); speed = np.linalg.norm(vel, axis=1)
    acc = np.gradient(vel, t, axis=0)
    toe = SIDE * TOE_OUT
    vis = obs['vis'] if obs is not None else np.zeros(T, bool)

    def half_width(k):
        """第 k 帧两踝左右间距的一半: 站着 = 静止姿势的; 走动时查真人表(往前/倒着), 不超过站着的。"""
        if speed[k] <= STILL or not WIDTH_ON:
            return body.lat
        u = vel[k] / speed[k]
        wf, wb = (np.interp(speed[k] / body.H, WIDTH_V, w) for w in WIDTH_TAB)
        return min(dir_mix(u, yaw[k], (wf, 2 * body.lat, wb)) / 2, body.lat)

    def neutral(i, k, ky=None):
        """第 k 帧这只脚放在身体下面该在的位置; 左右方向按第 ky 帧的朝向(默认 = k)。"""
        ky = k if ky is None else ky
        c, s = np.cos(yaw[ky]), np.sin(yaw[ky])
        return pxy[k] + SIDE[i] * half_width(ky) * np.array([-s, c]) + body.fwd * np.array([c, s])

    rch = lambda uu, ps: dir_mix(uu, ps, (body.reach, body.reach_side, body.reach if REACH_V10 else body.reach_back))       # 落点在身体 前/侧/后 能伸多远
    rch_err = lambda uu, ps: dir_mix(uu, ps, (body.reach if REACH_V10 else body.reach_back, body.reach_side, body.reach))   # 脚该往 前/侧/后 挪 = 脚在身体 后/侧/前
    pos = [neutral(i, 0) for i in (0, 1)]; fy = [yaw[0] + toe[i] for i in (0, 1)]
    swing = [None, None]; last_land, last = -1e9, 1
    fxy = np.zeros((T, 2, 2)); fyaw = np.zeros((T, 2)); contact = np.zeros((T, 2), bool)
    swings = []
    def ball_off(i, psi):                                     # 踝 -> 前掌 的水平偏移(W)
        dx, dz = (body.ball[i] - body.J[LEGS[i][2]])[[0, 2]]
        return dx * np.array([-np.sin(psi), np.cos(psi)]) + dz * np.array([np.cos(psi), np.sin(psi)])

    def land(i, kl, v, u, d, why_i):
        """这一步(落地帧 kl)落在哪: 位置看后面的路线 -> 沿移动方向落在对称点 -> 够得着 -> 不踩到另一只脚。返回 B, yB, dbg。"""
        look = min(d / speed[kl], LOOK_MAX) if speed[kl] > STILL else 0.0    # d 已按方向缩过
        kf = min(kl + int(round(look * FPS)), T - 1)
        yB = yaw[kf] + toe[i]
        yB = fy[1 - i] + float(np.clip(wrap(yB - fy[1 - i]), -FEET_YAW_MAX, FEET_YAW_MAX))
        out = float(np.clip(wrap(yB - yaw[kl]) * SIDE[i], -TOE_IN_MAX, TOE_OUT_MAX))   # 相对落脚时的骨盆: 内八少, 外八可以多
        yB = yaw[kl] + out * SIDE[i]
        B = neutral(i, kf, kl)                                               # 位置看后面的路线, 左右按落脚时的朝向
        dbg = dict(B0=B.copy(), kl=kl, kf=kf, u=None if u is None else u.copy())
        km2 = min(kl + int(round(np.interp(speed[kl] / body.H, V_TAB, DS_MID_TAB) * FPS)), T - 1)
        C = pxy[km2] + body.fwd * np.array([np.cos(yaw[km2]), np.sin(yaw[km2])])   # 那时身体下方(两踝平均该在的位置)
        lf = np.array([-np.sin(yaw[kl]), np.cos(yaw[kl])])
        if v > STILL and LAT_FIX == 1:
            # 和移动方向垂直的那个分量: 按两脚着地那一刻(km2)的骨盆 + 往加速度反方向放(转弯时脚在外侧)
            N = neutral(i, km2, kl) - LAT_ACC_K * (acc[km2] @ lf) * lf
            B = N + ((B - N) @ u) * u
        if v > STILL and why_i not in ('toe_in', 'toe_out', 'settle'):                        # 跟身体走的步: 沿移动方向落在对称点
                                                                     # (转脚尖的步不用: 它会被放到很远, 下一步两脚叉开 0.45 m, 骨盆沉 12 cm)
            rho = np.interp(v / body.H, V_TAB, LEAD_RATIO)
            Bm = C + rho * (C - pos[1 - i])                                  # 另一只脚关于骨盆的"对称"点, 前脚近后脚远
            B = B + ((Bm - B) @ u) * u
        if v > STILL and LAT_FIX == 2:
            # 和移动方向垂直的分量: 按"骨盆沿移动方向走到这只脚那一刻(kp)"的路线 + 往加速度反方向放(转弯时脚在外侧)。
            # 固定往后看 0.4 s(v22)会在走弯路时把脚放到内侧 10-15 cm(真人在外侧); 只看落地那一刻(LAT_FIX=1)又会在弯路上越走越偏
            kp = kl
            while kp < min(kl + int(LOOK_MAX * FPS), T - 1) and (pxy[kp] - B) @ u < 0:
                kp += 1
            N = neutral(i, kp, kl) - LAT_ACC_K * (acc[kp] @ lf) * lf
            B = N + ((B - N) @ u) * u
        dbg['Bmir'] = B.copy()
        nb = neutral(i, kl); eb = B - nb; nbe = np.linalg.norm(eb)          # 落地那一刻腿要够得着: 离那时身体下方不超过 reach
        lim = rch(eb / nbe if nbe > 1e-6 else None, yaw[kl])
        if nbe > lim:
            B = nb + eb * lim / nbe
        dbg['Bclamp'] = B.copy()
        side = np.array([-np.sin(yaw[kl]), np.cos(yaw[kl])]) * SIDE[i]      # 指向这只脚该在的那一侧
        gap = (B - pos[1 - i]) @ side
        along = abs((B - pos[1 - i]) @ np.array([np.cos(yaw[kl]), np.sin(yaw[kl])]))
        gmin = float(np.interp(along, [0.12, 0.25], [FEET_GAP, FEET_GAP_WALK])) if WIDTH_ON else FEET_GAP   # 并排时隔开, 一前一后时只要不交叉
        if gap < gmin:                                                       # 别落到另一只脚上或交叉
            B = B + (gmin - gap) * side
        return B, yB, dbg

    # 看不见的段尾要接上的观测: {段尾 kb(下一段看得见的第一帧): 每只脚 ('c', 脚印, 朝向) 或 ('a', 空中踝的水平位置, 朝向, 落地帧)}
    nxt_vis = np.full(T, T); tgt = {}
    if obs is not None:
        nv = T
        for k in range(T - 1, -1, -1):
            nv = k if vis[k] else nv
            nxt_vis[k] = nv
        for ka, kb in runs_of(~vis):
            if kb < T:
                tgt[kb] = [('c', obs['fp'][obs['run'][kb, i]][0], obs['fp'][obs['run'][kb, i]][1]) if obs['c'][kb, i]
                           else ('a', obs['ank'][kb, i, :2], obs['fyaw'][kb, i], obs['land_k'][kb, i]) for i in (0, 1)]

    def tail_script(sw1, fxy1, contact1):
        """第一遍的步子表 -> 第二遍用的 {离地帧: [步]}: 段尾接观测的每个看不见的段, 改最后两步、删掉后面多的。"""
        sc = {}
        for ka, kb in runs_of(~vis):
            if kb >= T:
                continue
            tg = tgt[kb]
            st = [dict(i=x['i'], k0=x['k0'], k1=x['k1'], A=np.array(x['A'], float), B=np.array(x['B'], float), yB=x['yB'], why=x['why'], air_start=x.get('air_start', False))
                  for x in sorted(sw1, key=lambda x: x['k0']) if ka - 1 <= x['k0'] < kb]
            air = [i for i in (0, 1) if tg[i][0] == 'a']
            if len(air) < 2:
                v = speed[kb - 1]; u = vel[kb - 1] / v if v > STILL else None
                Ts = gait(v, body, u, yaw[kb - 1])[0]; nT = max(int(round(Ts * FPS)), 1)
                nD = int(round(np.interp(v / body.H, V_TAB, DS_MIN_TAB) * FPS)); n02 = int(round(0.2 * FPS))
                def pos_ka(i):
                    # ka 时这只脚在哪: 踩着 = 脚印; 在空中 = 它这一步的落点。
                    # ★ 2026-11-12 (sam3d 主流水线 30 fps 人体): 这一步要是在 ka-1 之前就离地了, 不在 st 里(st 只收 ka-1 之后离地的),
                    #   原来的 next() 找不到 -> StopIteration, 整个摆人那步崩掉(jw4 r4_30fps 人 5)。去第一遍的全部步子里找包住 ka 的那一步;
                    #   还找不到就用 ka 时规划的位置。
                    if contact1[ka, i]:
                        return fxy1[ka, i]
                    for x in st:
                        if x['i'] == i:
                            return x['B']
                    for x in sw1:
                        if x['i'] == i and x['k0'] <= ka <= x['k1']:
                            return np.array(x['B'], float)
                    return fxy1[ka, i]
                # 最后迈的那只脚 f: 那时在空中的那只; 都踩着 = 之后先抬的那只的另一只
                f = air[0] if air else 1 - int(np.argmin([obs['fp'][obs['run'][kb, i]][3] for i in (0, 1)])); o = 1 - f
                jf = max([n for n, x in enumerate(st) if x['i'] == f], default=None)
                if jf is not None:
                    st = st[:jf + 1]                                                 # f 最后一步之后的都不要
                jo = max([n for n, x in enumerate(st) if x['i'] == o], default=None)
                if tg[f][0] == 'a':                                                  # 结束在观测的空中位置
                    # 已经在空中多久: 离地处(这只脚当时的脚印)到 kb 观测位置的距离 / kb 观测的水平速度, 0.15~0.5 s
                    A_f = st[jf]['A'] if jf is not None and not st[jf]['air_start'] else (st[jf]['B'] if jf is not None else pos_ka(f))
                    v_kb = np.linalg.norm(obs['ank'][min(kb + 1, T - 1), f, :2] - obs['ank'][kb, f, :2]) * FPS if kb + 1 < T and vis[kb + 1] else 1.0
                    k0a = kb - int(round(float(np.clip(np.linalg.norm(np.asarray(tg[f][1]) - A_f) / max(v_kb, 0.3), 0.15, 0.5)) * FPS))
                    if jf is not None and not st[jf]['air_start']:                # 第一遍那步的离地时刻离估计的不远就不挪(推后会让这只脚在后面多站几帧, 两脚叉开)
                        st[jf]['k0'] = max(st[jf]['k0'], k0a - int(round(AIR_EARLY * FPS)))
                    else:
                        k0a = max(k0a, st[jf]['k1'] + 1) if jf is not None else k0a
                        st.append(dict(i=f, k0=min(k0a, kb - 1), why='', air_start=False)); jf = len(st) - 1
                    st[jf].update(k1=kb, B=np.array(tg[f][1], float), yB=tg[f][2], why=st[jf]['why'] + '->接观测(空中)', air_end=True)
                else:
                    if jf is None and np.linalg.norm(pos_ka(f) - tg[f][1]) > FP_TOL:      # 整段没迈过、又离脚印远: 加一步
                        k0n = max(kb - nT - 1, (max(x['k1'] for x in st) + nD) if st else ka)
                        st.append(dict(i=f, k0=min(k0n, kb - n02), k1=kb - 1, why='', air_start=False)); jf = len(st) - 1
                    if jf is not None:
                        x = st[jf]
                        if x['k1'] > kb and not x['air_start']:                    # 要在 kb 前落下: 整步往前挪, 不压缩
                            sh = x['k1'] - kb; x['k0'] -= sh; x['k1'] = kb
                        x.update(B=np.array(tg[f][1], float), yB=tg[f][2], k1=min(x['k1'], kb), why=x['why'] + '->接观测')
                if tg[o][0] == 'c':                                                  # 另一只脚: 在 f 最后一步之前落到它的观测脚印
                    kf0 = st[jf]['k0'] if jf is not None else kb
                    if jo is None and np.linalg.norm(pos_ka(o) - tg[o][1]) > FP_TOL and kf0 - nD - max(kf0 - nD - nT, ka) >= n02:
                        st.append(dict(i=o, k0=max(kf0 - nD - nT, ka), k1=kf0 - nD, why='', air_start=False)); jo = len(st) - 1
                    if jo is not None:
                        x = st[jo]; x.update(B=np.array(tg[o][1], float), yB=tg[o][2], why=x['why'] + '->接观测')
                        if x['k1'] > kf0 - nD and not x['air_start']:
                            x['k1'] = max(kf0 - nD, x['k0'] + n02)
            for x in st:
                sc.setdefault(x['k0'], []).append(x)
        return sc

    if obs is not None and script is None and tgt:                   # 第一遍照常规划(script=False), 再改尾巴, 第二遍照步子表
        fxy1, _, contact1, sw1, _ = plan(t, pxy, yaw, body, obs, script=False)
        script = tail_script(sw1, fxy1, contact1)
    pivots = 0
    for k in range(T):
        if vis[k]:                                            # 看得见: 照观测(规划的最后一步到这帧为止)
            for i in (0, 1):
                swing[i] = None
                if obs['c'][k, i]:
                    xy, yw, kon, _ = obs['fp'][obs['run'][k, i]]
                    pos[i], fy[i] = xy.copy(), yw
                    if kon == k:
                        last_land, last = t[k], i
                fxy[k, i] = pos[i] if obs['c'][k, i] else obs['ank'][k, i, :2]
                fyaw[k, i] = obs['fyaw'][k, i]; contact[k, i] = obs['c'][k, i]
            continue
        scripted = isinstance(script, dict) and nxt_vis[k] < T   # 这段段尾要接观测, 第二遍: 照步子表
        if k > 0 and vis[k - 1]:                              # 刚进看不见的段: 观测里还在空中的脚接着迈完, 落点按规则定
            for i in (0, 1):
                if obs['c'][k - 1, i]:
                    continue
                e = next((e for e in (script or {}).get(k - 1, []) if e['i'] == i and e.get('air_start')), None) if scripted else None
                v = speed[k]; u = vel[k] / v if v > STILL else None
                Ts, d, _ = gait(v, body, u, yaw[k])
                Ts = obs['ts_med'] or Ts                                     # 迈步时长按这个人看得见时的(比查表准)
                ko = obs['off_k'][k - 1, i]
                el = (k - 1 - ko) / FPS if ko >= 0 else Ts / 2               # 已经在空中多久
                rem = max(Ts - el, 0.1)
                k1 = k - 1 + max(int(round(rem * FPS)), 1); kl = min(k1, T - 1)
                B, yB, dbg = land(i, kl, v, u, d, 'cont')
                # 落点: 规则的落点 和 按观测的运动外推的(减速到 0: 再走 速度 x 剩下时间 / 2) 按已迈了多少混合; 快落地了就基本照观测
                vo = (obs['ank'][k - 1, i, :2] - obs['ank'][k - 2, i, :2]) * FPS if k >= 2 and vis[k - 2] else np.zeros(2)
                w = float(np.clip(el / Ts, 0, 1))
                B = w * (obs['ank'][k - 1, i, :2] + vo * rem / 2) + (1 - w) * B
                yB = yB + w * wrap(obs['fyaw'][k - 1, i] - yB)
                if e is not None:
                    k1, B, yB = e['k1'], e['B'], e['yB']; kl = min(k1, T - 1)
                A = obs['ank'][k - 1, i, :2].copy()
                swing[i] = dict(i=i, k0=k - 1, k1=k1, A=A, B=B, dbg=dbg, yA=obs['fyaw'][k - 1, i], yB=yB, v=v, why='接着迈完', air_start=True,
                                **swing_style(A, B, yaw[(k - 1 + kl) // 2], v, body))
                swings.append(swing[i])
        for i in (0, 1):
            if swing[i] is not None and k >= swing[i]['k1']:
                pos[i], fy[i] = swing[i]['B'], swing[i]['yB']; swing[i] = None; last_land, last = t[k], i
            if swing[i] is None:                              # 身体转太快来不及迈: 踩着的脚绕前掌转
                out = wrap(fy[i] - yaw[k]) * SIDE[i]
                lim = -PIVOT_IN if out < -PIVOT_IN else (PIVOT_OUT if out > PIVOT_OUT else None)
                if lim is not None:
                    ball = pos[i] + ball_off(i, fy[i])
                    fy[i] = yaw[k] + lim * SIDE[i]
                    pos[i] = ball - ball_off(i, fy[i]); pivots += 1
        if scripted:                                          # 照改好的步子表迈
            for e in script.get(k, []):
                if e.get('air_start'):
                    continue
                i = e['i']
                if swing[i] is not None:                          # 上一步还没落(改过的表偶尔挤在一起): 先落下
                    pos[i], fy[i] = swing[i]['B'], swing[i]['yB']; swing[i] = None
                v = speed[k]; kl = min(e['k1'], T - 1)
                swing[i] = dict(i=i, k0=k, k1=e['k1'], A=pos[i].copy(), B=e['B'], dbg=None, yA=fy[i], yB=e['yB'], v=v, why=e['why'], air_end=e.get('air_end', False),
                                **swing_style(pos[i], e['B'], yaw[(k + kl) // 2], v, body))
                swings.append(swing[i])
        elif swing[0] is None and swing[1] is None and t[k] - last_land >= np.interp(speed[k] / body.H, V_TAB, DS_MIN_TAB) - 1e-6:
            # 往后看半个迈步时长: 按那时身体的位置判断要不要迈(离线才能这样; 真人起步也是脚先动)
            u0 = vel[k] / speed[k] if speed[k] > STILL else None
            km = min(k + int(round(gait(speed[k], body, u0, yaw[k])[0] * FPS / 2)), T - 1)
            v = speed[km]
            u = vel[km] / v if v > STILL else None
            Ts, d, stride = gait(v, body, u, yaw[km])                                # 查真人表: 走快了步子大, 迈步时长差不多
            need = np.zeros(2); why = ['', '']
            for i in (0, 1):
                e = neutral(i, km) - pos[i]
                out = wrap(fy[i] - yaw[km]) * SIDE[i]           # 脚尖相对骨盆: > 0 外八, < 0 内八
                ne = np.linalg.norm(e)
                e_now = neutral(i, k) - pos[i]; ne_now = np.linalg.norm(e_now)        # 够不着按"现在"算(按半步之后算会在快走时提前抬脚)
                if REACH_V10:
                    e_now, ne_now = e, ne
                if v > STILL:                                                         # 对称点离它现在多远(沿移动方向)
                    kmid = min(k + int(round((Ts + np.interp(v / body.H, V_TAB, DS_MID_TAB)) * FPS)), T - 1)
                    Cm = pxy[kmid] + body.fwd * np.array([np.cos(yaw[kmid]), np.sin(yaw[kmid])])
                    rho = np.interp(v / body.H, V_TAB, LEAD_RATIO)
                    lag = ((Cm + rho * (Cm - pos[1 - i])) - pos[i]) @ u / stride
                sd = np.array([-np.sin(yaw[km]), np.cos(yaw[km])]) * SIDE[i]
                o_i = (pxy[km] - pos[i]) @ sd                                        # >0: 骨盆已在这只脚外侧
                if v > STILL and u @ sd < EDGE_GATE:                                 # 只在站着或主要往这只脚那侧横移时管(往前走时骨盆本来就左右晃)
                    o_i = -1.0
                beat = 0.0
                if v > STILL and i != last:                                          # 按节奏: 另一只脚落地后过 (周期/2 - 迈步时长) 就抬
                    Tc = dir_mix(u, yaw[km], tuple(np.interp(v / body.H, V_TAB, x) for x in CYCLE_TAB))
                    wb = np.clip((v / body.H - BEAT_V[0]) / (BEAT_V[1] - BEAT_V[0]), 0, 1)
                    beat = wb * (t[k] - last_land) / max(Tc / 2 - Ts, 0.03)
                wb = np.clip((v / body.H - BEAT_V[0]) / (BEAT_V[1] - BEAT_V[0]), 0, 1) if v > STILL else 0.0   # 走快了只按节奏抬脚, 不按距离
                terms = dict(beat=beat, edge=1 + (o_i + EDGE_M) / 0.05 if o_i > -EDGE_M else 0.0, lag=(1 - wb) * lag if v > STILL else 0.0, lat=abs(e @ np.array([-u[1], u[0]])) / max(d, SETTLE) if v > STILL else 0.0,
                             settle=ne / SETTLE if v <= STILL else 0.0, reach=ne_now / rch_err(e_now / ne_now if ne_now > 1e-6 else None, yaw[k]),
                             toe_in=-out / TOE_IN_STEP, toe_out=out / TOE_OUT_STEP)
                why[i] = max(terms, key=terms.get)
                need[i] = max(need[i], terms[why[i]])
            i = int(np.argmax(need))
            if need[i] >= 1:
                swap = i == last and need[1 - i] >= 0.6 and t[k] - last_land < ALT_RECENT and why[i] != 'edge'   # 连续走时左右轮流; 停过再起步、或骨盆快越过这只脚(保平衡)时不轮流
                if swap:
                    i = 1 - i
                k1 = k + max(int(round(Ts * FPS)), 1); kl = min(k1, T - 1)
                B, yB, dbg = land(i, kl, v, u, d, why[i])
                swing[i] = dict(i=i, k0=k, k1=k1, A=pos[i].copy(), B=B, dbg=dbg, yA=fy[i], yB=yB, v=v, why=why[i] + ('(换脚 %.1f)' % need[i] if swap else ''),
                                **swing_style(pos[i], B, yaw[(k + kl) // 2], v, body))
                swings.append(swing[i])
        for i in (0, 1):
            s = swing[i]
            contact[k, i] = s is None or k == s['k0']
            fxy[k, i] = pos[i] if contact[k, i] else s['B']
            fyaw[k, i] = fy[i] if contact[k, i] else s['yB']
    print(f'踩着的脚绕前掌转: {pivots} 帧·脚')
    return fxy, fyaw, contact, swings, speed


def foot_pose(P, psi, phi, body, i):
    """脚放平时踝在 P(W)、朝向 psi; 俯仰 phi > 0 脚跟抬起(绕前掌转), < 0 脚尖翘起(绕脚跟转)。支点不动。
    返回 踝的位置, 脚的全局旋转。"""
    C = upright(psi); a0 = body.J[LEGS[i][2]]
    c0 = body.ball[i] if phi > 0 else body.heel[i]
    R = C @ rx(phi)
    return P + C @ (c0 - a0) + R @ (a0 - c0), R


def pitch_side(P, psi, Hp):
    """脚在髋前面(按脚尖方向) -> 只能绕脚跟翘脚尖(-1); 在后面 -> 抬脚跟(+1)。"""
    return -1.0 if (P[:2] - Hp[:2]) @ np.array([np.cos(psi), np.sin(psi)]) > 0 else 1.0


def min_pitch(P, psi, Hp, L, body, i):
    """踩地的脚: 够得着就放平; 够不着就抬脚跟/翘脚尖, 取刚好够得着的最小角度(1° 一档)。"""
    sgn = pitch_side(P, psi, Hp); lim = HEEL_OFF_MAX if sgn > 0 else TOE_UP_MAX
    for deg in range(int(np.degrees(lim)) + 1):
        phi = sgn * np.radians(deg)
        if np.linalg.norm(foot_pose(P, psi, phi, body, i)[0] - Hp) <= L:
            return phi
    return sgn * lim


def leg_ik(Hp, Tp, psi, Gf, G0, body, i, th=None):
    """髋位置 Hp、踝目标 Tp、膝盖朝向 psi、脚的全局旋转 Gf(都在 W) -> 髋/膝/踝的局部旋转(3x3)。返回 (三个旋转, 踝够不着的距离)。
    以 SMPL-X 静止姿势的腿为准: 膝盖只绕 x 轴弯 th(距离对上), 再把整条腿绕髋转过去: 髋->踝 对准目标, 膝盖朝 psi。"""
    D = np.linalg.norm(Tp - Hp); Dmax = body.hip_ankle(i, 0.0)
    miss = max(D - Dmax, 0.0)
    if th is None:
        th = 0.0 if miss > 0 else body.knee_for(i, D)
    else:                                                                      # 给了 th 就按它弯: 踝沿 髋->目标 方向挪到对应距离
        miss = 0.0; Dn = body.hip_ankle(i, th); Tp = Hp + (Tp - Hp) / D * Dn; D = Dn
    v = body.t0[i] + rx(th) @ body.s0[i]; v = v / np.linalg.norm(v)            # 静止姿势里 髋->踝 方向(弯了 th 之后)
    z = np.array([0, 0, 1.0]); n0 = z - (z @ v) * v; n0 /= np.linalg.norm(n0)  # 静止姿势里膝盖朝前
    u = (Tp - Hp) / D
    f = np.array([np.cos(psi), np.sin(psi), 0.0]); n1 = f - (f @ u) * u; n1 /= np.linalg.norm(n1)
    Gt = np.stack([u, n1, np.cross(u, n1)], 1) @ np.stack([v, n0, np.cross(v, n0)], 1).T   # 大腿的全局旋转
    Gs = Gt @ rx(th)
    return (G0.T @ Gt, rx(th), Gs.T @ Gf), miss


def constrain_swing_reach(ankles, hips, lengths, contact, visible, swings):
    """Fit inferred swing targets to leg reach while leaving each footprint in place.

    If a planted ankle is already outside reachable range, applying the full
    reach projection on the first airborne sample teleports the foot. Carry only
    the pre-existing endpoint error along the swing and release it continuously.
    """
    offset = ankles - hips
    horizontal = np.linalg.norm(offset[..., :2], axis=-1)
    limit = np.sqrt(np.maximum(0., np.square(lengths) - np.square(offset[..., 2])))
    excess = np.maximum(0., horizontal - limit)
    allowance = np.zeros_like(excess)
    n = len(ankles)
    for swing in swings:
        side, lo, hi = swing['i'], swing['k0'], swing['k1']
        if lo < 0 or lo >= n or hi <= lo:
            continue
        indices = np.arange(lo + 1, min(hi, n))
        phase = (indices - lo) / (hi - lo)
        blend = phase**3 * (10. - 15.*phase + 6.*phase**2)
        start = excess[lo, side]
        end = excess[hi, side] if hi < n else 0.
        allowance[indices, side] = (1. - blend)*start + blend*end
    moving = ~contact & ~visible[:, None]
    pull = np.maximum(0., excess - allowance) * moving
    ankles[..., :2] -= offset[..., :2] * (pull / np.maximum(horizontal, 1.e-9))[..., None]
    return int(np.count_nonzero(pull))


def run(s, t, model=None, log=print, obs=None, join='plan', registered_contacts=False):
    """s: (T, 79) SMPL-X 参数 [global_orient 3, transl 3, betas 10, body_pose 63], 世界 W(z 朝上, 地面 z=0), 30 fps; t: 时间 s。
    返回补好腿的结果(dict, 和 main 存的 npz 同样的键)。
    obs(make_obs, 可选): 腿有时看得见。看得见的帧用观测的腿(踩地的脚钉在脚印上, 不滑)。
      join='plan': 看不见的段接着观测规划, 段尾最后几步落到观测脚印上(见 plan)。
      join='cut'(对照组): 整段照常规划(不看观测), 看得见的帧直接换成观测, 交界处 JOIN_W 秒内把差值渐变掉。"""
    s = np.asarray(s, np.float64)
    go, transl, betas, pose = s[:, 0:3], s[:, 3:6], s[:, 6:16], s[:, 16:79].copy()
    model = smplx_model() if model is None else model
    body = Body(model, betas[0])
    T = len(t)
    pel = transl + body.J[0]                                                   # 骨盆关节在 W 里的位置
    # 骨盆朝向: SAM 3D Body 的骨盆朝向抖得厉害(腿看不见时它也是猜的), 腿跟着它会扭。
    # 把骨盆朝向平滑(水平朝向 + 前后左右倾斜分开平滑), 腰部关节(spine1)反向补回, 上半身在世界里的朝向不变。
    G_in = Rot.from_rotvec(go).as_matrix()
    f = G_in @ np.array([0, 0, 1.0])
    yaw_in = np.unwrap(np.arctan2(f[:, 1], f[:, 0]))
    yaw = gaussian_filter1d(yaw_in, PELVIS_YAW_SIGMA * FPS)
    tilt = np.stack([upright(y).T @ G for y, G in zip(yaw_in, G_in)])           # 去掉水平朝向后剩下的倾斜
    tilt = Rot.from_rotvec(gaussian_filter1d(Rot.from_matrix(tilt).as_rotvec(), 0.1 * FPS, axis=0)).as_matrix()
    G0 = np.stack([upright(y) @ Q for y, Q in zip(yaw, tilt)])
    for k in range(T):                                                          # spine1(关节 3)补回: G0 新 · L3 新 = G0 旧 · L3 旧
        pose[k, 6:9] = Rot.from_matrix(G0[k].T @ G_in[k] @ Rot.from_rotvec(pose[k, 6:9]).as_matrix()).as_rotvec()
    go = Rot.from_matrix(G0).as_rotvec()
    pxy = gaussian_filter1d(pel[:, :2], 0.25 * FPS, axis=0)

    obsp = obs if (obs is not None and join == 'plan') else None                # 规划时用不用观测
    vis = obsp['vis'] if obsp is not None else np.zeros(T, bool)
    fxy, fyaw, contact, swings, speed = plan(t, pxy, yaw, body, obsp)
    P = np.concatenate([fxy, np.full((T, 2, 1), body.ankle_h)], 2)              # 脚放平时踝的位置
    hip_rel = np.stack([np.einsum('tij,j->ti', G0, body.J[h] - body.J[0]) for h, _, _, _ in LEGS], 1)   # (T, 2, 3)
    Luse = body.Luse

    # 骨盆高度: 原来的高度(高过站直就压到站直; 站直按膝盖弯 KNEE_STAND 算) -> 踩地的腿抬脚跟/翘脚尖还够不着就再往下压 -> 平滑
    z_stand = body.pelvis_h - (body.hip_drop - min(Luse))
    z_goal = gaussian_filter1d(np.minimum(pel[:, 2], z_stand), 0.2 * FPS)
    zmax = np.full(T, np.inf)
    for k in range(T):
        for i in (0, 1):
            if not contact[k, i]:
                continue
            Hxy = pel[k, :2] + hip_rel[k, i, :2]
            sgn = pitch_side(P[k, i], fyaw[k, i], np.append(Hxy, 0))
            if vis[k]:                                                          # 看得见: 按观测的俯仰够(骨盆低一点, 不硬抬脚跟; 否则腿卡在最直、抬/不抬脚跟来回切, 膝角跳)
                A, _ = foot_pose(P[k, i], fyaw[k, i], obsp['phi'][k, i], body, i)
            else:
                A, _ = foot_pose(P[k, i], fyaw[k, i], sgn * (HEEL_OFF_MAX if sgn > 0 else TOE_UP_PLAN), body, i)
            dh = np.linalg.norm(A[:2] - Hxy)
            zmax[k] = min(zmax[k], A[2] + np.sqrt(max(Luse[i] ** 2 - dh ** 2, 0)) - hip_rel[k, i, 2])
    z = gaussian_filter1d(minimum_filter1d(np.minimum(z_goal, zmax), Z_MIN_WIN), Z_SIGMA)
    # 骨盆随步子起伏(真人: 两脚着地时低、单脚着地中点高, 幅度按速度查动捕表)。只往下压, 不会让腿够不着
    ds = gaussian_filter1d(contact.all(1).astype(float), 0.05 * FPS)
    bob = np.interp(gaussian_filter1d(speed, 0.5 * FPS) / body.H, V_REF, BOB_TAB) / 100 * ds
    bob = bob * gaussian_filter1d((~vis).astype(float), 0.1 * FPS)              # 看得见的帧: 骨盆是观测的, 本来就带起伏
    z = z - bob
    z = np.maximum(z, z_stand - Z_DROP_TOTAL)
    pel_out = np.concatenate([gaussian_filter1d(pel[:, :2], PEL_XY_SIGMA * FPS, axis=0), z[:, None]], 1)
    Hp = pel_out[:, None] + hip_rel                                            # (T, 2, 3) 髋

    # 踩地的脚: 最小俯仰; 迈步的脚: 从抬脚那帧的踝位置/俯仰 平滑过渡到落脚那帧的, 中间加抬脚高度
    # 自然的落地/离地俯仰(真人动捕): 往前走脚跟先落、脚跟先抬; 倒着走脚尖先落、脚尖先抬; 横着走脚尖略先。
    # 落地后 settle 帧内放平, 离地前 OFF_RAMP 内抬起。支点(脚跟或前掌)不动, 所以不滑。
    nat = np.zeros((T, 2)); n_off = int(round(OFF_RAMP * FPS))
    for sw in swings:
        i, k0, k1 = sw['i'], sw['k0'], sw['k1']
        for k in range(max(k0 - n_off, 0), k0 + 1):
            w = (k - (k0 - n_off)) / n_off; w = w * w * (3 - 2 * w)
            if contact[k, i] and abs(sw['po'] * w) > abs(nat[k, i]):
                nat[k, i] = sw['po'] * w
        for k in range(k1, min(k1 + sw['settle'] + 1, T)):
            w = 1 - (k - k1) / sw['settle']
            if contact[k, i] and abs(sw['pl'] * w) > abs(nat[k, i]):
                nat[k, i] = sw['pl'] * w
    phi = np.zeros((T, 2)); ank = np.zeros((T, 2, 3))
    for k in range(T):
        for i in (0, 1):
            if contact[k, i]:
                pr, pn = min_pitch(P[k, i], fyaw[k, i], Hp[k, i], Luse[i], body, i), nat[k, i]
                phi[k, i] = pn if pr == 0 else (np.sign(pr) * max(abs(pr), abs(pn)) if pr * pn > 0 else pr)   # 够不着优先
                ank[k, i] = foot_pose(P[k, i], fyaw[k, i], phi[k, i], body, i)[0]

    def apply_obs(o, ank, phi, fyaw):
        """看得见的帧换成观测: 踩地的脚 = 脚印(钉住) + 观测的俯仰/朝向; 迈步的脚 = 观测的踝。
        腿和规划的帧一样不许伸过 Luse(膝至少弯 KNEE_STAND): 观测的腿常是直的(真人 40% 帧膝角 > 172°), 伸直附近差几 mm 膝角就跳十几度。
        踩地的脚: 够不着就多抬脚跟/翘脚尖(同规划的帧); 迈步的脚: 踝水平往髋那边收。"""
        V = o['vis']
        for k in np.flatnonzero(V):
            for i in (0, 1):
                fyaw[k, i] = o['fyaw'][k, i]
                if o['c'][k, i]:
                    Pk = np.append(o['fp'][o['run'][k, i]][0], body.ankle_h); pn = o['phi'][k, i]
                    pr = min_pitch(Pk, fyaw[k, i], Hp[k, i], Luse[i], body, i)
                    phi[k, i] = pn if pr == 0 else (np.sign(pr) * max(abs(pr), abs(pn)) if pr * pn > 0 else pr)
                    ank[k, i] = foot_pose(Pk, fyaw[k, i], phi[k, i], body, i)[0]
        def swing_part():
            # 迈步的脚: 观测的踝 + 两头的差(离地/落地那帧 钉住的脚 - 观测)在迈步过程中线性过渡, 落地时正好接上钉住的脚印(不然踝高差几 cm, 落地那帧膝角跳)
            for i in (0, 1):
                for a, b in runs_of(V & ~o['c'][:, i]):
                    d0 = ank[a - 1, i] - o['ank'][a - 1, i] if a > 0 and V[a - 1] and o['c'][a - 1, i] else np.zeros(3)
                    d1 = ank[b, i] - o['ank'][b, i] if b < T and V[b] and o['c'][b, i] else np.zeros(3)
                    for k in range(a, b):
                        u = (k - a + 1) / (b - a + 1)
                        ank[k, i] = o['ank'][k, i] + (1 - u) * d0 + u * d1; phi[k, i] = o['phi'][k, i]
                        e = ank[k, i] - Hp[k, i]
                        if np.linalg.norm(e) > Luse[i]:
                            h = np.linalg.norm(e[:2]); hmax = np.sqrt(max(Luse[i] ** 2 - e[2] ** 2, 0))
                            ank[k, i, :2] = Hp[k, i, :2] + e[:2] * hmax / max(h, 1e-9)
        swing_part()
        # 离地前膝盖先弯(同规划的 PRE_SWING): 钉住的脚被腿长上限卡在最直, 真人离地前 2-3 帧膝盖已经在弯, 不这样离地那帧一下弯 15°。
        # 离地前 PRE_SWING 内逐渐多抬脚跟(绕前掌, 不滑), 让离地那一帧的膝角接上迈步第一帧的
        th0 = knee_raw(); n_pre = int(round(PRE_SWING * FPS)); moved = False
        for i in (0, 1):
            for a, b in runs_of(V & ~o['c'][:, i]):
                if a - n_pre < 0 or not (V[a - 1] and o['c'][a - 1, i]) or th0[a, i] - th0[a - 1, i] < np.radians(6):
                    continue
                jump = th0[a, i] - th0[a - 1, i]
                for k in range(a - n_pre + 1, a):
                    if not (V[k] and o['c'][k, i]):
                        continue
                    w = (k - (a - n_pre)) / n_pre; target = th0[k, i] + w * w * (3 - 2 * w) * jump
                    Pk = np.append(o['fp'][o['run'][k, i]][0], body.ankle_h)
                    for deg in range(int(np.degrees(max(phi[k, i], 0))), int(np.degrees(HEEL_OFF_MAX)) + 1):
                        A = foot_pose(Pk, fyaw[k, i], np.radians(deg), body, i)[0]; D = np.linalg.norm(A - Hp[k, i])
                        if D <= body.hip_ankle(i, 0.0) and body.knee_for(i, D) >= target:
                            break
                    if np.radians(deg) > phi[k, i]:
                        phi[k, i] = np.radians(deg); ank[k, i] = A; moved = True
        if moved:
            swing_part()
    def make_swings():
        """迈步的脚: 从抬脚那帧的踝位置/俯仰 平滑过渡到落脚那帧的, 中间加抬脚高度; 然后腿伸到头的往回收。"""
        for sw in swings:
            i, k0, k1 = sw['i'], sw['k0'], sw['k1']
            if k1 < T:
                a1, p1 = ank[k1, i], phi[k1, i]
            else:
                a1, p1 = foot_pose(np.append(sw['B'], body.ankle_h), sw['yB'], 0.0, body, i)[0], 0.0
            # 抬脚: 形状 u^al (1-u)^be, 最高点在 up(往前早、倒着晚), 两端速度为 0; 高度让踝最高离放平高度 H
            up = sw['up']
            al, be = (2.0, 2.0 * (1 - up) / up) if up < 0.5 else (min(2.0 * up / (1 - up), LIFT_AL_MAX), 2.0)
            up = al / (al + be)
            mu = up ** 3 * (10 - 15 * up + 6 * up ** 2)
            Hb = max(body.ankle_h + sw['H'] - (ank[k0, i, 2] + mu * (a1[2] - ank[k0, i, 2])), 0.02)
            # 起止速度接上离地前(抬脚跟)、落地后(绕脚跟/前掌放平)脚的实际运动(五次 Hermite, 两头加速度 0);
            # 原来两头速度为 0, 空中要急加速急减速 = 腿抖(快走时比真人多 13 倍)
            Tsw = (k1 - k0) / FPS; d01 = a1 - ank[k0, i]
            v0 = (ank[k0, i] - ank[max(k0 - 1, 0), i]) * FPS if k0 > 0 and (contact[k0 - 1, i] or vis[k0 - 1]) else np.zeros(3)   # 接观测: 观测的速度
            v1 = (ank[min(k1 + 1, T - 1), i] - a1) * FPS if k1 + 1 < T and (contact[min(k1 + 1, T - 1), i] or vis[k1 + 1]) else np.zeros(3)
            v0 = np.clip(v0, -SWING_V_END, SWING_V_END); v1 = np.clip(v1, -SWING_V_END, SWING_V_END)
            for k in range(k0 + 1, min(k1, T)):
                u = (k - k0) / (k1 - k0); m = u ** 3 * (10 - 15 * u + 6 * u ** 2)    # 水平: 最小加加速度曲线 + 两头速度项
                h1 = u - 6 * u ** 3 + 8 * u ** 4 - 3 * u ** 5; h4 = -4 * u ** 3 + 7 * u ** 4 - 3 * u ** 5
                bump = (u ** al * (1 - u) ** be) / (up ** al * (1 - up) ** be)
                ank[k, i] = ank[k0, i] + m * d01 + Tsw * (h1 * v0 + h4 * v1) + np.array([0, 0, Hb * bump])
                phi[k, i] = phi[k0, i] + m * (p1 - phi[k0, i])
                # 脚尖朝向按"相对骨盆"过渡: 骨盆转得快时脚跟着转, 不会在半空中落后成内八
                rA, rB = wrap(sw['yA'] - yaw[k0]), wrap(sw['yB'] - yaw[min(k1, T - 1)])
                fyaw[k, i] = yaw[k] + rA + m * (rB - rA)

        # 迈步途中腿不许伸到头: 伸直附近膝角对距离很敏感(差 1 cm 膝盖就差十几度), 会一下拉直又一下弯 = 抖。超出就把踝水平往髋那边收
        if registered_contacts:
            return constrain_swing_reach(ank, Hp, Luse, contact, vis, swings)
        pulled = 0
        for k in range(T):
            for i in (0, 1):
                if not contact[k, i] and not vis[k]:
                    e = ank[k, i] - Hp[k, i]
                    if np.linalg.norm(e) > Luse[i]:
                        h = np.linalg.norm(e[:2]); hmax = np.sqrt(max(Luse[i] ** 2 - e[2] ** 2, 0))
                        ank[k, i, :2] = Hp[k, i, :2] + e[:2] * hmax / max(h, 1e-9); pulled += 1
        return pulled

    def knee_raw():
        return np.array([[0.0 if np.linalg.norm(ank[k, i] - Hp[k, i]) > body.hip_ankle(i, 0.0) else body.knee_for(i, np.linalg.norm(ank[k, i] - Hp[k, i]))
                          for i in (0, 1)] for k in range(T)])

    if obsp is not None:
        apply_obs(obsp, ank, phi, fyaw)
    make_swings(); th = knee_raw()
    # 离地前膝盖先弯: 脚在髋后面(往前/横着走)时, 离地前 PRE_SWING 内逐渐多抬脚跟(绕前掌转, 前掌不动), 让离地那一帧膝角 = 迈步第一帧的
    n_pre = int(round(PRE_SWING * FPS)); pre = 0
    for sw in swings:
        i, k0 = sw['i'], sw['k0']
        if k0 + 2 >= T or k0 - n_pre < 0 or th[k0 + 1, i] - th[k0, i] < np.radians(6) or sw.get('air_start'):
            continue
        if pitch_side(P[k0, i], fyaw[k0, i], Hp[k0, i]) < 0:
            continue
        jump = th[k0 + 1, i] - th[k0, i]
        for k in range(k0 - n_pre + 1, k0 + 1):
            if not contact[k, i] or vis[k]:
                continue
            w = (k - (k0 - n_pre)) / n_pre; target = th[k, i] + w * w * (3 - 2 * w) * jump
            for deg in range(int(np.degrees(max(phi[k, i], 0))), int(np.degrees(HEEL_OFF_MAX)) + 1):
                A = foot_pose(P[k, i], fyaw[k, i], np.radians(deg), body, i)[0]; D = np.linalg.norm(A - Hp[k, i])
                if D <= body.hip_ankle(i, 0.0) and body.knee_for(i, D) >= target:
                    break
            if np.radians(deg) > phi[k, i]:
                phi[k, i] = np.radians(deg); ank[k, i] = A; pre += 1
    pulled = make_swings()
    if obs is not None:                                                          # 交界处: 观测和规划剩下的差渐变掉; 对照组在这里才换成观测
        ank_p, phi_p, fyaw_p = ank.copy(), phi.copy(), fyaw.copy()
        apply_obs(obs, ank, phi, fyaw)
        if obsp is None:
            contact[obs['vis']] = obs['c'][obs['vis']]
        W = max(int(round(JOIN_W * FPS)), 1); ss = lambda x: x * x * (3 - 2 * x)
        ex = lambda X, a, b: 2 * X[a] - X[b]                                     # 按上一帧的速度外推一帧(迈步中的脚不会停一帧)
        for ka, kb in runs_of(~obs['vis']):
            for i in (0, 1):
                if ka > 1:                                                       # 看得见 -> 看不见: 差值从 1 渐变到 0
                    da = ex(ank[:, i], ka - 1, ka - 2) - ank_p[ka, i]
                    dp = ex(phi[:, i], ka - 1, ka - 2) - phi_p[ka, i]; dy = wrap(ex(fyaw[:, i], ka - 1, ka - 2) - fyaw_p[ka, i])
                    for k in range(ka, min(ka + W, kb)):
                        w = 1 - ss((k - ka + 1) / W)
                        ank[k, i] = ank_p[k, i] + w * da; phi[k, i] = phi_p[k, i] + w * dp; fyaw[k, i] = fyaw_p[k, i] + w * dy
                if kb < T and kb - 2 >= ka:                                      # 看不见 -> 看得见: 差值从 0 渐变到 1
                    da = ank[kb, i] - ex(ank_p[:, i], kb - 1, kb - 2)
                    dp = phi[kb, i] - ex(phi_p[:, i], kb - 1, kb - 2); dy = wrap(fyaw[kb, i] - ex(fyaw_p[:, i], kb - 1, kb - 2))
                    for k in range(max(kb - W, ka), kb):
                        w = ss((k - (kb - W) + 1) / W)
                        ank[k, i] = ank[k, i] + w * da; phi[k, i] = phi[k, i] + w * dp; fyaw[k, i] = fyaw[k, i] + w * dy
    th = knee_raw()
    log(f'离地前多抬脚跟让膝盖先弯: {pre} 帧·脚; 迈步途中腿伸到头、把脚往回收: {pulled} 帧·脚')
    # 迈步时膝角平滑(只动空中的帧; 两头权重 sin(pi u) = 0, 离地/落地那一刻不变)
    th_s = th.copy()
    for sw in swings:
        i, k0, k1 = sw['i'], sw['k0'], min(sw['k1'], T - 1)
        a0, a1 = max(k0 - 4, 0), min(k1 + 5, T)
        sm = gaussian_filter1d(th[a0:a1, i], KNEE_SMOOTH * FPS)
        for k in range(k0 + 1, k1):
            if obs is not None and obs['vis'][k]:
                continue
            w = np.sin(np.pi * (k - k0) / (k1 - k0))
            th_s[k, i] = th[k, i] + w * (sm[k - a0] - th[k, i])
    miss = np.zeros((T, 2))
    for k in range(T):
        for i, (h, kn, an, toe) in enumerate(LEGS):
            Gf = upright(fyaw[k, i]) @ rx(phi[k, i])
            knee_dir = fyaw[k, i] + KNEE_TO_PELVIS * wrap(yaw[k] - fyaw[k, i])
            R3, miss[k, i] = leg_ik(Hp[k, i], ank[k, i], knee_dir, Gf, G0[k], body, i, None if contact[k, i] else th_s[k, i])
            for j, Rm in zip((h, kn, an), R3):
                pose[k, 3 * (j - 1):3 * j] = Rot.from_matrix(Rm).as_rotvec()
            pose[k, 3 * (toe - 1):3 * toe] = [-max(phi[k, i], 0), 0, 0]          # 抬脚跟时脚趾留在地上
    Pm = {'global_orient': go, 'transl': pel_out - body.J[0], 'betas': betas, 'body_pose': pose}
    J, V = body_joints(model, Pm)
    dev = np.linalg.norm(J[:, [7, 8]] - ank, axis=2)
    err = np.where(contact, dev, 0.0)                                           # 踩地的脚: 踝和目标差(应为 ~0)
    log(f'迈步中膝角平滑让踝离原路线 最大 {np.where(contact, 0.0, dev).max() * 100:.1f} cm')
    J_in, _ = body_joints(model, {'global_orient': s[:, 0:3], 'transl': transl, 'betas': betas, 'body_pose': s[:, 16:79]})
    up_shift = np.linalg.norm((J[:, 12:22] - J[:, [0]]) - (J_in[:, 12:22] - J_in[:, [0]]), axis=2).max(1)   # 脖子以上、手臂相对骨盆的位置变化
    steps = np.array([(sw['i'], t[sw['k0']], t[sw['k0']] + (sw['k1'] - sw['k0']) / FPS, *sw['A'], *sw['B'], sw['yB'], sw['v']) for sw in swings])
    log(f'{len(steps)} 步; 踝和目标差 最大 {err.max() * 1000:.1f} mm; 够不着 最大 {miss.max() * 100:.1f} cm '
        f'({np.mean(miss.max(1) > 0.005) * 100:.1f}% 帧 > 5 mm); 骨盆压低 中位 {np.median(pel[:, 2] - z) * 100:.1f} cm 最大 {np.max(pel[:, 2] - z) * 100:.1f} cm')
    log(f'骨盆朝向平滑: 和原来差 中位 {np.degrees(np.median(np.abs(yaw - yaw_in))):.0f}° 最大 {np.degrees(np.abs(yaw - yaw_in).max()):.0f}°; '
        f'上半身关节相对骨盆位置变化 最大 {up_shift.max() * 1000:.1f} mm')
    return dict(vertices=V.astype(np.float32), faces=model.faces.astype(np.int32), joints=J[:, :22].astype(np.float32), time_s=t,
                global_orient=go, transl=Pm['transl'], betas=betas, body_pose=pose, feet=ank, foot_yaw=fyaw, foot_pitch=phi, contact=contact,
                steps=steps, step_why=np.array([sw['why'] for sw in swings]), speed=speed, pelvis_z_in=pel[:, 2], pelvis_z_out=z,
                ik_miss=miss, ankle_err=err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--tag', default='', help='输出文件名后缀, 区分版本')
    a = ap.parse_args()
    d = np.load(f'{ROHM}/inputs/p{a.pid}_rohm_input.npz')
    s = d['smplx_world'].astype(np.float64)
    assert np.abs(s[:, 3:6] + Body(smplx_model(), s[0, 6:16]).J[0] - d['joints_world'][:, 0]).max() < 1e-3
    o = run(s, d['time_s'])
    out = f'{HERE}/outputs/p{a.pid}_proclegs{a.tag}_W.npz'
    os.makedirs(f'{HERE}/outputs', exist_ok=True)
    np.savez(out, **o)
    print('->', out)


if __name__ == '__main__':
    main()
