#!/usr/bin/env python
"""地形网格: 地面不是一个平面的场景(楼梯、台阶、斜坡)用它代替"一个高度的地面"。2026-11-11 加, joker 楼梯起头。

开关 GROUND_MODE(outdoor_pipeline CFG): plane = 老的一个平面(默认, 不碰这里) / terrain = 用这里。

只信"朝上的面"(深度图上逐像素算法向, 和天夹角小于阈值的才要: 地面、踏面; 墙面、台阶立面、人的侧面都进不来)。
建地形(尺度之后, 组装之前), 每个镜头各建一张:
  1. 普通地面: 好帧的朝上点(去掉人、栏杆 / 扶手的像素)按 5 cm 格子分箱, 每格取偏低的分位 -> 剔尖刺 -> 扔小块
     -> 0.5 m 以内的洞补上(人脚下被挡住的那块)。离量到的格子 1 m 以外 = 没看到过的地方, 用这个镜头的地面斜面外推,
     不再拿最近一格的高度往外铺(joker: 上坡那头的高度铺到人脚下, 摆人的地面硬约束把人抬起 99~313 cm)。
  2. 楼梯(有 stairs / staircase 掩码时): 只拿楼梯掩码里、朝上的点(踏面), RANSAC 拟一个斜面 —— 扶手、栏杆顶、
     侧边的石墩、消防梯都不在这个面上, 自动排除。斜面定方向、坡度、起止和宽度(内点最密的那一条, 不含两侧扶手区),
     然后直接生成一级级的方块台阶: 每级同高同深, 踏面水平、立面竖直、两侧边缘齐平, 级数 = 总高 / 17 cm。
     楼梯那块(两侧再加 0.3 m, 扶手立在那)从普通地面里挖掉, 换成这些方块; 查高度时那块返回所在那级的踏面高度。
  -> 07_terrain.npz(查高度用) + 07_terrain.json(报告) + 07_terrain_高度.jpg(俯视高度图)。
  网格由组装那步现做(Terrain.mesh = 普通地面 + 台阶方块), 放在 DA3 世界系, 和物体一起转正导出。

天不在这里定, 用 up 那步的结果(楼梯上 world_up.py 拟合不出平面, joker 用的是人工给的相机的上)。
★ 试过"取所有朝上小面的共同方向当天"(踏面互相平行): joker 上和相机的上差 13°, 而且算出来整段楼梯是平的 ——
  DA3 的深度分不出一级级台阶, 楼梯在深度里是一整个斜坡, 朝上小面的共同方向成了斜坡的法线, 不是重力。撤了。
★ 斜坡比真楼梯缓(joker 1 号镜头 10.5°, 真的约 30°): 仰拍的深度把楼梯压扁了, 没有独立线索能纠正(试过的都被数据否了)。
  台阶按深度的斜坡做, 人也按深度摆, 从原机位看对得上; 级数按 17 cm 一级算, 所以踏面会比真楼梯深。
★ 11-11 第一版台阶是把整块楼梯区域的高度表按 28 cm 量化: 每级 5 cm 高, 扶手、侧边石墩也被量化成台阶, 用户说很烂。
  换成上面的参数化方块台阶。

查高度: Terrain(path).height_at(P, vf) —— 世界点 P 的水平位置脚下的地面高度(沿 up), 给原视频帧号就用那个镜头的。
摆人(body_into_scene.clearance)、组装(物体落地)都用它。
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RES = 0.05          # 查高度的格子(m)
MESH_STEP = 2       # 普通地面网格隔几格取一个顶点(0.10 m); 台阶是单独的方块, 不靠格子
GROUND_STYLE = "heightmap"   # 交付的地面: heightmap = 高度表网格(老的) / plane_stairs = 一个平面 + 一个干净的台阶(见 build 末尾)。按本轮配置定, 见 _run_style
MESH_JUMP = 0.4     # 一个网格小方格四个角高差超过这个(m)就不连: 侧边石墩、对不齐的格子之间会被拉成一片片竖墙
                    # (joker 3 号镜头往下看, 画面上方一排锯齿竖片)
UP_DEG = 25.0       # 法向和天夹角小于这个才算"朝上的面"。UP_DEG_LEVELED: 镜头扶正(shot_level.py)过的用 40°: 扶正后 joker 楼梯是 25°,
                    #   深度分不出一级级台阶, 楼梯面的法向就是斜坡的法向, 卡在 25° 上一半被扔, 1 号镜头点不够没建成; 墙 90° 照样进不来
UP_DEG_LEVELED = 40.0
LOW_Q = 0.30        # 每格取高度的分位: 偏低, 压掉零碎的朝上小面
MIN_HITS = 3        # 一格至少这么多点才算量到
FILL_M = 0.5        # 空洞补多远(m): 人脚下被挡住的那块
HOLE_M = 3.0        # 离量到的格子这么近以内(m)的空格 = 被挡住的洞(人、物体挡的): 用周围量到的地面局部平滑补(高斯归一化卷积)
FAR_M = HOLE_M      # 再远 = 没看到过: 在量到的格子和"相机脚下"之间线性插值(HOLE_M~2*HOLE_M 之间渐变过去)
HOLE_SIGMA = 0.6    # 局部补洞的高斯宽度(m)
FOOT_R = 0.3        # 人脚下这么大半径(m)的格子按人的脚高定
FOOT_OVERRIDE = 0.5 # 那里量到的地面比脚低这么多(m)以上就以脚为准(深度穿过人看到的是他身后更低的地方)
HALF_BODY = 0.12    # 和 body_into_scene.HALF_BODY 一样: 深度图量的是身体正面, 模型的根在身体中间
MIN_BLOB_M2 = 1.0   # 连通块小于这个面积的扔掉(飞点)
SPIKE_WIN, SPIKE_M = 9, 0.25   # 尖刺: 离周围 9x9 格中位超过 0.25 m 的格子扔掉
STRIDE = 2          # 深度图隔几个像素取一个
EDGE_REL = 0.05     # 相邻像素深度跳变超过 5% = 物体边缘, 那里的法向是假的
PEOPLE_DILATE = 25  # 人的掩码外扩(原图像素)
RAIL_WORDS = ("railing", "railings", "handrail", "handrails", "banister", "guardrail", "fence")
RAIL_DILATE = 9     # 栏杆 / 扶手掩码外扩(掩码像素): 细杆子边上的深度是糊的
OBJ_DILATE = 5      # 其余物体掩码外扩(掩码像素)
STAIR_WORDS = ("stairs", "staircase", "steps", "stairway")
STAIR_RISE = 0.17   # 每级高(m), 常规楼梯 15~18 cm
STAIR_FIT_M = 0.12  # 拟楼梯斜面: 离面这么近的点才算踏面(扶手顶 ~0.9 m、侧边石墩都远在外面)
STAIR_MIN_DEG = 4.0 # 楼梯斜面坡度小于这个就不做台阶(那就是平地)
STAIR_MIN_PTS = 3000
STAIR_SIDE_M = 0.3  # 台阶两侧再挖掉这么宽(扶手 / 栏杆立在那, 普通地面在那里是乱的)
STAIR_BASE_M = 0.3  # 台阶方块往下埋多深(最低一级踏面以下)
STAIR_MIN_H = 0.45  # 楼梯总高至少这么多(约 3 级)才做台阶; joker 3 号镜头是特写, 楼梯只露一小角, 拟出 0.15 m 高的"楼梯"
STAIR_MIN_LEN = 2.0 # 楼梯沿坡至少这么长(m)
STAIR_MIN_INL = 0.7 # 楼梯掩码里朝上的点至少这么多落在斜面上(不够 = 掩码里乱, 不是一段清楚的楼梯)
STAIR_GAP_M = 1.0   # 横向: 内点之间小于这么宽的空档算同一段楼梯(栏杆立在台阶上, 栏杆那条没有点)
STAIR_PERIOD_MIN_AC = 0.2   # 画面里数台阶: 自相关峰至少这么高才信
STAIR_RISE_OK = (0.06, 0.25)
MERGE_TOL = 0.3     # 别的镜头和主镜头在重叠处的高差中位(m)不超过这个才把它量到的地面并进统一场景  # 数出来的每级高要在这个范围(m), 否则不信, 退回 17 cm 一级


def _run_style(r1):
    """交付的地面用哪种: 环境变量 GROUND_STYLE > 本轮配置 00_run_config.json 的"覆盖" > 默认 GROUND_STYLE。"""
    if os.environ.get("GROUND_STYLE"):
        return os.environ["GROUND_STYLE"]
    try:
        o = json.load(open(f"{r1}/00_run_config.json")).get("覆盖", {}) or {}
        if o.get("GROUND_STYLE"):
            return o["GROUND_STYLE"]
    except (OSError, ValueError):
        pass
    return GROUND_STYLE


def _cams(da3):
    return json.load(open(f"{da3}/cameras.json"))["cameras"]


def _bad(da3):
    p = f"{da3}/bad_frames.json"
    return set(json.load(open(p)).get("坏帧", [])) if os.path.exists(p) else set()


def frame_points(da3, k, cam, stride=STRIDE, skip=None):
    """一帧: 世界系的点和(朝着相机的)单位法向, 只留可信像素。返回 (P (N,3), N (N,3), C 相机中心)。
    skip: 这一帧不要的像素(bool 掩码或掩码列表, 任意分辨率, 比如人、栏杆), 缩到深度图大小后去掉。"""
    d = np.load(f"{da3}/depth/{k[:-4]}_depth.npy").astype(np.float64)
    cf = f"{da3}/depth/{k[:-4]}_conf.npy"
    conf = np.load(cf) if os.path.exists(cf) else None
    H, W = d.shape
    K = np.array(cam["intrinsic_3x3"], float)
    if "input_hw" in cam:
        h0, w0 = cam["input_hw"]; sx, sy = W / w0, H / h0
    else:
        sx = sy = W / (K[0, 2] * 2)
    fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy
    ys, xs = np.mgrid[0:H, 0:W]
    Pc = np.stack([(xs - cx) / fx * d, (ys - cy) / fy * d, d], -1)
    dx = Pc[1:-1, 2:] - Pc[1:-1, :-2]; dy = Pc[2:, 1:-1] - Pc[:-2, 1:-1]
    n = np.cross(dx, dy)
    dc = d[1:-1, 1:-1]
    jump = np.maximum(np.abs(d[1:-1, 2:] - d[1:-1, :-2]), np.abs(d[2:, 1:-1] - d[:-2, 1:-1])) > EDGE_REL * dc
    ok = (dc > 0) & (dc <= 35.0) & np.isfinite(dc) & ~jump
    if conf is not None:
        c_ = conf[1:-1, 1:-1]; ok &= c_ >= np.percentile(c_, 30)
    if skip is not None:
        import cv2
        for m in (skip if isinstance(skip, (list, tuple)) else [skip]):
            ok &= ~cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)[1:-1, 1:-1]
    sl = (slice(None, None, stride), slice(None, None, stride))
    P, n, ok = Pc[1:-1, 1:-1][sl].reshape(-1, 3), n[sl].reshape(-1, 3), ok[sl].reshape(-1)
    P, n = P[ok], n[ok]
    nn = np.linalg.norm(n, axis=1); good = nn > 1e-12
    P, n = P[good], n[good] / nn[good, None]
    n[(n * P).sum(1) > 0] *= -1                              # 看得到的面朝着相机(相机在原点)
    E = np.array(cam["extrinsic_w2c_4x4"], float); R, t = E[:3, :3], E[:3, 3]
    return (P - t) @ R, n @ R, -R.T @ t


def people_masks(r1):
    """人体跟踪的掩码: 原视频帧号 -> 取掩码的函数(bool, 原图大小, 已外扩)。没有人体那一轮 = {}。角色名(joker)也在里面, 不靠词表。"""
    body = f"{r1}/body4d"
    fj = f"{body}/00_frames/frames.json"
    if not os.path.exists(fj):
        return {}
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "pipeline"))
    import people_identity
    import cv2
    from PIL import Image
    raw = people_identity.raw_run(body)
    kept = json.load(open(fj))["kept_indices"]
    out = {}
    for bi, vf in enumerate(kept):
        p = f"{raw}/masks/{bi:08d}.png"
        if os.path.exists(p):
            out[int(vf)] = (lambda p=p: cv2.dilate((np.array(Image.open(p)) > 0).astype(np.uint8),
                                                   np.ones((2 * PEOPLE_DILATE + 1,) * 2, np.uint8)).astype(bool))
    return out


def people_feet(r1, da3, keys, cams, vfm, up):
    """人的脚 = 地面的证据。每帧每个人: SAM 3D Body 的人体网格(相机系), 远近按这个人掩码里的场景深度缩放
    (和摆人那步 body_into_scene.depth_ratio 同一个比例: (掩码深度中位 + 半个身体厚) / 模型给的距离), 每个顶点保持
    画面里的位置、深度乘这个比例, 反投影到世界系, 取沿 up 最低的 20 个顶点的平均 = 脚点。坏相机的帧不在 keys 里。
    返回 {原视频帧号: (N,3)}。
    ★ 11-12 加: 人脚下那块地被人自己挡住, 深度图在那里看到的是他身后远处的东西。joker 3 号镜头人在相机下约 7 m
      (身体大小、深度图在人身上都这么说), 身后的街在相机下 14 m, 地形按街补, 摆人 / 贴地 / 补腿都把人往下拽, 对齐 4.5% -> 288%。
    ★ 先试过"掩码最下沿 = 脚": joker 3 号镜头这个人的腿都被画框切掉(137 个好相机帧里 112 帧掩码碰底边), 只取到 11 个点,
      3 号镜头一个都没有。模型的网格腿出画也有, 所以改用网格。"""
    body = f"{r1}/body4d"; fj = f"{body}/00_frames/frames.json"
    if not os.path.exists(fj):
        return {}
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "pipeline"))
    import people_identity, cv2, trimesh
    from PIL import Image
    raw = people_identity.raw_run(body); kept = {int(v): i for i, v in enumerate(json.load(open(fj))["kept_indices"])}
    md = f"{raw}/mesh_4d_individual"
    pids = sorted(os.listdir(md)) if os.path.isdir(md) else []
    out = {}
    for k in keys:
        vf = int(vfm[k]); bi = kept.get(vf)
        mp = f"{raw}/masks/{bi:08d}.png" if bi is not None else None
        if not mp or not os.path.exists(mp):
            continue
        mk = np.array(Image.open(mp)); Hm_, Wm_ = mk.shape[:2]
        d = np.load(f"{da3}/depth/{k[:-4]}_depth.npy").astype(np.float64); H, W = d.shape
        c = cams[k]; K = np.array(c["intrinsic_3x3"], float)
        if "input_hw" in c:
            h0, w0 = c["input_hw"]; sx, sy = W / w0, H / h0
        else:
            sx = sy = W / (K[0, 2] * 2)
        fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy
        E = np.array(c["extrinsic_w2c_4x4"], float); R, t = E[:3, :3], E[:3, 3]
        pts = []
        for pid in pids:
            jp, pp = f"{raw}/focal_4d_individual/{pid}/{bi:08d}.json", f"{md}/{pid}/{bi:08d}.ply"
            if not (os.path.exists(jp) and os.path.exists(pp)):
                continue
            m = cv2.resize((mk == int(pid)).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            m = cv2.erode(m, np.ones((3, 3), np.uint8)).astype(bool)
            if m.sum() < 15:
                continue
            cj = json.load(open(jp)); tz, fm = float(cj["camera"][2]), float(cj["focal_length"])
            ratio = (float(np.median(d[m])) + HALF_BODY) / tz
            if not np.isfinite(ratio) or ratio <= 0:
                continue
            V = np.asarray(trimesh.load(pp, process=False).vertices)[::5] * np.array([1.0, -1.0, -1.0])   # ply 是 OpenGL 系
            V = V[V[:, 2] > 0.1]
            u = (fm * V[:, 0] / V[:, 2] + Wm_ / 2) * W / Wm_; v = (fm * V[:, 1] / V[:, 2] + Hm_ / 2) * H / Hm_
            z = V[:, 2] * ratio
            Pw = (np.c_[(u - cx) / fx * z, (v - cy) / fy * z, z] - t) @ R
            pts.append(Pw[np.argsort(Pw @ up)[:20]].mean(0))
        if pts:
            out[vf] = np.array(pts)
    return out


def word_masks(r1, words, dilate=0):
    """03_masks 里这些词的掩码: 返回 (取掩码的函数 原视频帧号 -> bool 或 None, 有掩码的帧数)。空掩码当没有。"""
    from pycocotools import mask as mask_util
    M = {}
    for w in words:
        p = f"{r1}/03_masks/{w}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True); hw = [int(x) for x in z["hw"]]
        for r, c in zip(z["meta"], z["counts"]):
            M.setdefault(int(r["frame"]), []).append((c, hw))

    def get(vf):
        rows = M.get(int(vf))
        if not rows:
            return None
        m = np.zeros(rows[0][1], bool)
        for c, hw in rows:
            m |= mask_util.decode({"counts": c, "size": hw}).astype(bool)
        if not m.any():
            return None
        if dilate:
            import cv2
            m = cv2.dilate(m.astype(np.uint8), np.ones((2 * dilate + 1,) * 2, np.uint8)).astype(bool)
        return m
    return get, sum(1 for f in M if get(f) is not None) if M else 0


def iter_frames(da3, max_frames=60, skip_of=None):
    """skip_of: 原视频帧号 -> 这一帧不要的掩码列表(人、栏杆)。"""
    cams, bad = _cams(da3), _bad(da3)
    keys = [k for k in sorted(cams) if k not in bad and os.path.exists(f"{da3}/depth/{k[:-4]}_depth.npy")]
    keys = [keys[i] for i in np.unique(np.linspace(0, len(keys) - 1, min(max_frames, len(keys))).astype(int))]
    vfm = json.load(open(f"{da3}/frame_map.json"))["frames"]
    for k in keys:
        yield k, frame_points(da3, k, cams[k], skip=skip_of(int(vfm[k])) if skip_of else None)


def basis(up):
    e1 = np.array([1.0, 0, 0]) - up * up[0]
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([0, 0, 1.0]) - up * up[2]
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


# 楼梯参数(每个镜头一行, 没有楼梯 = NaN): c0 + sl*s 是斜面高度, s = uv·d 沿上坡, w = uv·nw 横向
ST_KEYS = ("c0", "d0", "d1", "sl", "s0", "s1", "w0", "w1", "n")


def stair_boxes(st):
    """一组楼梯参数 -> [(s_a, s_b, w0, w1, 踏面高, 底高)], 每级一个方块。"""
    c0, d0, d1, sl, s0, s1, w0, w1, n = st
    n = int(n); run = (s1 - s0) / n; rise = sl * run
    base = c0 + sl * s0 - rise / 2 - STAIR_BASE_M
    return [(s0 + k * run, s0 + (k + 1) * run, w0, w1, c0 + sl * (s0 + (k + 0.5) * run), base) for k in range(n)]


class Terrain:
    """07_terrain.npz 的查询端。高度都沿 up, 和 world_up.json 的 ground_h 同一把尺子。
    ★ 分镜头: 多镜头的片子各镜头的深度对不齐(joker 4 个镜头, 合在一起建出满地尖刺、相机陷进地形), 所以每个镜头各建一张
      高度表(Hq[s]), 查的时候给原视频帧号就用它那个镜头的; 不给帧号 / 那个镜头没建成 = 主镜头(量到格子最多的那个)。
      网格(mesh, 交付的场景里那块地)只有主镜头的。"""

    def __init__(self, path):
        z = np.load(path)
        self.up, self.e1, self.e2 = z["up"], z["e1"], z["e2"]
        self.lo, self.res = z["lo"], float(z["res"])
        self.H = z["H"]                 # 主镜头: 量到 + 补过的格子, 其余 NaN(铺网格用; 楼梯那块已挖掉)
        self.Hq = z["Hq"]               # (镜头数, ny, nx): 每个镜头每一格都有值, 查询用(楼梯那块 = 台阶踏面高); 没建成的镜头整张 NaN
        self.cuts, self.fps, self.main = z["cuts"], float(z["fps"]), int(z["main"])
        self.ST = z["ST"] if "ST" in z.files else np.full((len(self.Hq), len(ST_KEYS)), np.nan)
        self.Hm = z["Hm"] if "Hm" in z.files else None   # (镜头数, ny, nx): 每个镜头铺网格用的高度(楼梯那块挖空)
        _md = int(z["model"]) if "model" in z.files else 0   # 地面模型(平面 + 台阶), 见 build: 1 = 人也查它; 2 = 只交付网格
        self.model = _md == 1          # 人查的高度 / 镜头对齐都按模型
        self.model_mesh = _md in (1, 2)  # 交付的地面网格 = 平面 + 台阶
        self.plane_h = float(z["plane_h"]) if "plane_h" in z.files else float("nan")
        self.landing_priority = bool(z["landing_priority"]) if "landing_priority" in z.files else False
        self.joined_flights = bool(z["joined_flights"]) if "joined_flights" in z.files else False
        self.extra = z["extra"] if "extra" in z.files else np.zeros((0, 6))   # 分层地面的下段楼梯 / 平台方块(主楼梯坐标)
        self.snap = z["snap"] if "snap" in z.files else np.zeros((0, 6))      # 立在地上的物体该落到的高度(不进网格, 见 _pedestals)

    def model_h(self, P):
        """交付的地面(平面 + 台阶 + 分层的下段楼梯 / 平台 / 底座柱)在 P 处的顶高; 没有地面模型 = 主镜头查询高度。
        组装场景时立在地上的物体(GROUNDED)落到这里 —— 落到查询高度表上会和看得到的网格对不上(路灯陷进台阶 2.8 m)。"""
        P = np.atleast_2d(np.asarray(P, float))
        if not self.model_mesh:
            return self.height_at(P)
        st = self.ST[self.main]; d = st[1] * self.e1 + st[2] * self.e2; nw = -st[2] * self.e1 + st[1] * self.e2
        ex = np.vstack([np.asarray(self.extra, float).reshape(-1, 6), np.asarray(self.snap, float).reshape(-1, 6)])
        ss, ww = P @ d, P @ nw
        height = _model_h_sw(ss, ww, st, ex, self.plane_h)
        if self.landing_priority and len(self.extra):
            # Below the landing, the registered street view observes the lower
            # flight and road. The extrapolated upper flight has no support there.
            below = ss <= self.extra[-1,1]
            height[below] = _model_h_sw(ss[below], ww[below], np.full_like(st,np.nan), ex, self.plane_h)
        return height

    def shot_of(self, vf):
        return int(np.searchsorted(self.cuts, vf / self.fps + 1e-3, side="right"))

    def height_at(self, P, vf=None):
        P = np.atleast_2d(np.asarray(P, float))
        s = self.main if vf is None else self.shot_of(vf)
        if s >= len(self.Hq) or not np.isfinite(self.Hq[s]).any():
            s = self.main
        G = self.Hq[s]
        uv = np.c_[P @ self.e1, P @ self.e2]
        ij = np.floor((uv - self.lo) / self.res).astype(int)
        i = np.clip(ij[:, 0], 0, G.shape[1] - 1); j = np.clip(ij[:, 1], 0, G.shape[0] - 1)
        return G[j, i]

    def stairs_mesh(self, shot=None, base_h=None):
        """这个镜头的台阶方块(trimesh, DA3 世界系); 没有楼梯 = None。base_h: 方块底面一律落到这个高度(实心到路面)。"""
        import trimesh
        st = self.ST[self.main if shot is None else shot]
        if not np.isfinite(st).all():
            return None
        d = st[1] * self.e1 + st[2] * self.e2
        nw = -st[2] * self.e1 + st[1] * self.e2
        if np.linalg.det(np.c_[d, nw, self.up]) < 0:
            raise AssertionError("台阶坐标系不是右手系")
        rows = [(sa, sb, w0, w1, top, min(base, base_h) if base_h is not None else base) for sa, sb, w0, w1, top, base in stair_boxes(st)]
        return self._boxes(rows, st)

    def _boxes(self, rows, st=None):
        """一组 (s_a, s_b, w0, w1, 顶高, 底高) 方块(主楼梯的 d / nw / up 坐标) -> trimesh。"""
        import trimesh
        st = self.ST[self.main] if st is None else st
        d = st[1] * self.e1 + st[2] * self.e2
        nw = -st[2] * self.e1 + st[1] * self.e2
        R = np.eye(4); R[:3, :3] = np.c_[d, nw, self.up]
        parts = []
        for sa, sb, w0, w1, top, base in rows:
            b = trimesh.creation.box(extents=[sb - sa, w1 - w0, top - base])
            T = R.copy(); T[:3, 3] = (sa + sb) / 2 * d + (w0 + w1) / 2 * nw + (top + base) / 2 * self.up
            b.apply_transform(T); parts.append(b)
        return trimesh.util.concatenate(parts)

    def mesh(self, step=MESH_STEP, shot=None, H=None):
        """DA3 世界系的地形网格(trimesh): 普通地面(只铺量到 / 补过的格子) + 台阶方块。默认主镜头; shot = 看某个镜头自己的。"""
        import trimesh
        # ★ 2026-11-12 交付的场景地面 = 一整块水平路面 + 实心台阶(底面落到路面), 用户定的:
        #   "地面就应该是拟合出来的平的路面, 楼梯是放在上面的"。原来是高度表铺网格 + 台阶方块: 台阶两侧各挖 0.3 m
        #   (STAIR_SIDE_M)没填、相邻格高差 > 0.4 m 就断开(MESH_JUMP), 楼梯和两侧地面之间一道道缝; 往下看的镜头里
        #   街面被高度表补出来的一片片地面盖住。有地面模型(self.model)时摆人 / 贴地 / 补腿查的高度(Hq)也是这个模型。
        if self.model_mesh and shot is None and H is None:
            hs = self.plane_h; ny, nx = self.H.shape; u0, v0 = self.lo - 30.0; u1, v1 = self.lo + np.array([nx, ny]) * self.res + 30.0
            #   ★ 四周各铺出去 30 m: 网格范围只盖量到的地面, 远处的车(joker car_2 在 s -23)底下是空的
            PV = np.array([uu_ * self.e1 + vv_ * self.e2 + hs * self.up for uu_, vv_ in ((u0, v0), (u1, v0), (u0, v1), (u1, v1))])
            m = trimesh.Trimesh(PV, np.array([[0, 1, 3], [0, 3, 2]]), process=False)
            if float(m.face_normals[0] @ self.up) < 0:
                m.faces = m.faces[:, ::-1]
            if getattr(self, "joined_flights", False):
                st = self.ST[self.main]
                upper = stair_boxes(st)
                if self.landing_priority and len(self.extra):
                    end = self.extra[-1,1]
                    upper = [(max(a,end),b,w0,w1,h,base) for a,b,w0,w1,h,base in upper if b > end]
                    # End the buried solid at the landing edge as well.
                rows = np.array(upper + list(map(tuple, self.extra)))
                # Split only at actual tread edges and emit one top height per
                # cell. Adjacent flight estimates can overlap after registration.
                xs = np.unique(rows[:, :2]); ys = np.unique(rows[:, 2:4])
                cells = []
                for x0,x1 in zip(xs[:-1],xs[1:]):
                    x = (x0+x1)/2
                    for y0,y1 in zip(ys[:-1],ys[1:]):
                        y = (y0+y1)/2
                        hits = rows[(rows[:,0]<=x)&(x<rows[:,1])&(rows[:,2]<=y)&(y<rows[:,3])]
                        if len(hits) and x1-x0 > 1e-6 and y1-y0 > 1e-6:
                            cells.append((x0,x1,y0,y1,hits[:,4].max(),hs))
                sm = self._boxes(cells) if cells else None
            else:
                sm = self.stairs_mesh(None, base_h=hs)
            if sm is not None:
                m = trimesh.util.concatenate([m, sm])
            if len(self.extra) and not getattr(self, "joined_flights", False):        # 分层: 下段楼梯 + 平台(实心到路面)
                m = trimesh.util.concatenate([m, self._boxes([tuple(r) for r in self.extra])])
            m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[130 / 255, 130 / 255, 135 / 255, 1.0], metallicFactor=0.0, roughnessFactor=0.9,
                doubleSided=True))
            return m
        if H is None:
            H = self.Hm[shot] if (shot is not None and self.Hm is not None) else self.H
        Hs = H[::step, ::step]
        rows, cols = Hs.shape
        jj, ii = np.mgrid[0:rows, 0:cols]
        u = self.lo[0] + (ii * step + 0.5) * self.res
        v = self.lo[1] + (jj * step + 0.5) * self.res
        V = (u[..., None] * self.e1 + v[..., None] * self.e2 + np.nan_to_num(Hs)[..., None] * self.up).reshape(-1, 3)
        ok = np.isfinite(Hs)
        q = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
        Hz = np.nan_to_num(Hs); c4 = np.stack([Hz[:-1, :-1], Hz[1:, :-1], Hz[:-1, 1:], Hz[1:, 1:]])
        q &= (c4.max(0) - c4.min(0)) <= MESH_JUMP
        a = (jj[:-1, :-1] * cols + ii[:-1, :-1])[q]
        b, c, d = a + 1, a + cols, a + cols + 1
        F = np.r_[np.c_[a, c, b], np.c_[b, c, d]]
        m = trimesh.Trimesh(V, F, process=False)
        m.remove_unreferenced_vertices()
        if len(m.faces) and float(np.median(m.face_normals @ self.up)) < 0:
            m.faces = m.faces[:, ::-1]
        sm = self.stairs_mesh(shot)
        if sm is not None:
            m = trimesh.util.concatenate([m, sm]) if len(m.faces) else sm
        m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[130 / 255, 130 / 255, 135 / 255, 1.0], metallicFactor=0.0, roughnessFactor=0.9,
            doubleSided=True))
        return m


def _cell_uv(lo, res, ny, nx):
    jj, ii = np.mgrid[0:ny, 0:nx]
    return lo[0] + (ii + 0.5) * res, lo[1] + (jj + 0.5) * res


def _grid(U, Hh, lo, nx, ny, res, feet=None, pfeet=None):
    """一组朝上点 -> 高度表: 每格偏低分位 -> 剔尖刺 -> 扔小块 -> 补洞 -> 没看到过的地方插值。
    feet: 这个镜头各帧"相机脚下"的点 (N,3) = (u, v, 相机高 - 相机离地), 没看到过的地方在量到的格子和它们之间线性插值。
    ★ 11-11 先试过按这个镜头的地面拟斜面外推: 楼梯的深度被压扁(11°, 真的约 30°), 斜面往镜头这边延长正好穿过相机,
      joker 4 号镜头相机离"地面" -0.12~0.02 m, 离镜头近的 joker 被地面硬约束抬起最多 196 cm。撤了。
      再之前是最近一格往外铺: 上坡那头的高度铺到人脚下, 抬人 99~313 cm。
    返回 (G 量到的, Gq 全图查询用, Gm 铺网格用, 量到格数, 剔掉的尖刺格数)。"""
    from scipy import ndimage
    ij = np.floor((U - lo) / res).astype(int)
    m = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    cid = ij[m, 1] * nx + ij[m, 0]; h = Hh[m]
    o = np.lexsort((h, cid)); cid, h = cid[o], h[o]
    uc, st, cnt = np.unique(cid, return_index=True, return_counts=True)
    keep = cnt >= MIN_HITS
    G = np.full(ny * nx, np.nan)
    G[uc[keep]] = h[st[keep] + np.floor(LOW_Q * (cnt[keep] - 1)).astype(int)]
    G = G.reshape(ny, nx); meas = np.isfinite(G)
    if not meas.any():
        return None
    _, idx0 = ndimage.distance_transform_edt(~meas, return_indices=True)
    med = ndimage.median_filter(G[idx0[0], idx0[1]], size=SPIKE_WIN)
    spike = meas & (np.abs(G - med) > SPIKE_M)
    meas &= ~spike
    lab, nl = ndimage.label(meas)
    if nl:
        area = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nl + 1)) * res * res
        meas &= np.isin(lab, 1 + np.nonzero(area >= MIN_BLOB_M2)[0])
    if not meas.any():
        return None
    G[~meas] = np.nan
    # 人的脚(pfeet (N,3) = u, v, 高): 脚下 FOOT_R 内的格子, 没量到 / 和量到的差 > FOOT_OVERRIDE 的, 按脚高(偏低分位)定
    if pfeet is not None and len(pfeet):
        fij = np.floor((pfeet[:, :2] - lo) / res).astype(int); r_ = int(np.ceil(FOOT_R / res))
        ok = (fij[:, 0] >= 0) & (fij[:, 0] < nx) & (fij[:, 1] >= 0) & (fij[:, 1] < ny); fij, fh = fij[ok], pfeet[ok, 2]
        FH = {}
        for (i_, j_), h_ in zip(fij, fh):
            for dj in range(-r_, r_ + 1):
                for di in range(-r_, r_ + 1):
                    if di * di + dj * dj <= r_ * r_ and 0 <= i_ + di < nx and 0 <= j_ + dj < ny:
                        FH.setdefault((j_ + dj, i_ + di), []).append(h_)
        for (j_, i_), hs in FH.items():
            h_ = float(np.quantile(hs, LOW_Q))
            if not meas[j_, i_] or h_ - G[j_, i_] > FOOT_OVERRIDE:
                G[j_, i_] = h_; meas[j_, i_] = True
    # ★ 没量到的格子一律用"周围的数据"做多尺度高斯归一化卷积补: 数据 = 量到的格子 + 相机脚下(相机高 - 相机离地)。
    #   离数据近就用窄的高斯(0.6 m), 补不上的放宽一倍再补, 已补的不动。
    #   试过的: 最近一格往外铺(上坡的高度铺到人脚下, 抬人 99~313 cm) / 按斜面外推(压扁的斜面穿过相机, 相机离地 -0.12 m) /
    #   量到的边缘点 + 相机脚下做全局线性插值(被远点主导: joker 3 号镜头人脚下离最近量到的地面约 3 m, 插进来的是 20 m 外
    #   街面的高度, 相机下 14 m, 人周围真量到的是相机下 7.7~8.1 m; 地面约束把人往下拽 4.8 m, 对齐 4.5% -> 343%)。
    known = meas.copy(); G0 = np.where(meas, G, 0.0)
    if feet is not None and len(feet):
        fij = np.floor((feet[:, :2] - lo) / res).astype(int)
        ok = (fij[:, 0] >= 0) & (fij[:, 0] < nx) & (fij[:, 1] >= 0) & (fij[:, 1] < ny)
        fij, fh = fij[ok], feet[ok, 2]
        fr = ~meas[fij[:, 1], fij[:, 0]]                      # 相机脚下其实量到了的, 信量到的
        G0[fij[fr, 1], fij[fr, 0]] = fh[fr]; known[fij[fr, 1], fij[fr, 0]] = True
    W0 = known.astype(float); G0 = np.where(known, G0, 0.0)
    todo = ~known; sg = HOLE_SIGMA / res; Gq = np.where(known, G0, np.nan)
    while todo.any() and sg < 4 * max(nx, ny):
        num = ndimage.gaussian_filter(G0, sg); den = ndimage.gaussian_filter(W0, sg)
        ok_ = todo & (den > 1e-4)
        Gq[ok_] = num[ok_] / den[ok_]; todo &= ~ok_
        sg *= 2
    Gq[todo] = float(np.median(G[meas]))
    Gq = np.where(meas, ndimage.median_filter(np.where(meas, G, Gq), size=3), Gq)
    dk = ndimage.distance_transform_edt(~known) * res
    Gm = np.where(dk <= HOLE_M, Gq, np.nan)                   # 网格铺到离数据 HOLE_M 以内(被挡住的洞、相机脚下也铺)
    return G, Gq, Gm, int(meas.sum()), int(spike.sum())


def _stair_points(r1, da3, keys_by_shot, cams, vfm, up, e1, e2, cu, skip_of):
    """每个镜头: 楼梯掩码里、朝上的点(去掉人和栏杆) -> {镜头: (uv (N,2), 高 (N,))}。没有楼梯掩码 = {}。"""
    get, n = word_masks(r1, STAIR_WORDS)
    if not n:
        return {}
    out = {}
    for s, ks in keys_by_shot.items():
        U, Hh = [], []
        for k in ks:
            m = get(vfm[k])
            if m is None:
                continue
            P, N, C = frame_points(da3, k, cams[k], skip=[~m] + (skip_of(int(vfm[k])) if skip_of else []))
            o = N @ up > cu
            U.append(np.c_[P[o] @ e1, P[o] @ e2]); Hh.append(P[o] @ up)
        if U and sum(map(len, Hh)) >= STAIR_MIN_PTS:
            out[s] = (np.concatenate(U), np.concatenate(Hh))
    return out


def stair_fit(U, Hh, seed=0, iters=400):
    """楼梯的踏面点 -> (楼梯参数 ST_KEYS 那一行, 报告)。拟不出 / 太平 -> (None, 报告)。
    RANSAC 斜面(离面 < STAIR_FIT_M 算内点, 坡度 < 45°) -> 最小二乘精修 -> 内点横向分布最密的那一条定宽 -> 纵向 1~99% 定起止。"""
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    if len(Hh) > 200000:
        i = rng.choice(len(Hh), 200000, replace=False); U, Hh = U[i], Hh[i]
    A = np.c_[np.ones(len(Hh)), U]
    best, nbest = None, -1
    for _ in range(iters):
        i = rng.choice(len(Hh), 3, replace=False)
        try:
            x = np.linalg.solve(A[i], Hh[i])
        except np.linalg.LinAlgError:
            continue
        if np.hypot(x[1], x[2]) > 1.0:
            continue
        k = int((np.abs(A @ x - Hh) < STAIR_FIT_M).sum())
        if k > nbest:
            best, nbest = x, k
    if best is None:
        return None, dict(台阶="拟不出斜面")
    inl = np.abs(A @ best - Hh) < STAIR_FIT_M
    for _ in range(2):
        x = np.linalg.lstsq(A[inl], Hh[inl], rcond=None)[0]
        inl = np.abs(A @ x - Hh) < STAIR_FIT_M
    c0, g = float(x[0]), x[1:]; sl = float(np.linalg.norm(g)); deg = float(np.degrees(np.arctan(sl)))
    if deg < STAIR_MIN_DEG:
        return None, dict(坡度_度=round(deg, 1), 台阶="坡度太小, 不做")
    d = g / sl
    # 楼梯走向 = 踏面点脚印的长轴(主成分), 不一定等于斜面的下降方向: 天差一点, 下降方向就往旁边偏(joker 4 号镜头偏 7°,
    # 按下降方向摆, 楼梯脚印一头宽出去一头缺一块)。脚印不够细长(长短轴比 < 2)就还用下降方向。
    X = U[inl] - U[inl].mean(0); ev, V = np.linalg.eigh(X.T @ X); a_ = V[:, 1] * np.sign(V[:, 1] @ d)
    turn = float(np.degrees(np.arccos(np.clip(a_ @ d, -1, 1))))
    if ev[1] >= 4 * ev[0] and turn <= 30:
        d = a_
    nw = np.array([-d[1], d[0]])
    s, w = U[inl] @ d, U[inl] @ nw

    def span(x, binw, min_frac, gap):
        """一维: 有点的格子(>= 峰值 min_frac)连成段, 段间空档 <= gap 就并起来, 取点最多的那段的 (lo, hi, 选中)。"""
        e = np.arange(x.min(), x.max() + 2 * binw, binw); hc = np.histogram(x, e)[0]
        occ = np.nonzero(hc >= max(min_frac * hc.max(), 5))[0]
        runs, a = [], occ[0]
        for i, j in zip(occ[:-1], occ[1:]):
            if (j - i - 1) * binw > gap:
                runs.append((a, i)); a = j
        runs.append((a, occ[-1]))
        a, b = max(runs, key=lambda r: hc[r[0]:r[1] + 1].sum())
        return e[a], e[b + 1], (x >= e[a]) & (x <= e[b + 1])
    wa, wb, sel = span(w, 0.1, 0.05, STAIR_GAP_M)
    sa, sb, sel2 = span(s[sel], 0.25, 0.01, STAIR_GAP_M)
    sel[np.nonzero(sel)[0][~sel2]] = False
    w0, w1 = (float(v) for v in np.percentile(w[sel], [0.5, 99.5]))
    s0, s1 = (float(v) for v in np.percentile(s[sel], [0.2, 99.8]))
    c0 = c0 + float(g @ nw) * (w0 + w1) / 2; sl = float(g @ d)      # 沿走向的中线: 高 = c0 + sl*s(踏面水平, 横向坡只剩几厘米)
    H = sl * (s1 - s0); n = max(3, int(round(H / STAIR_RISE)))
    st = np.array([c0, d[0], d[1], sl, s0, s1, w0, w1, n], float)
    rep = dict(坡度_度=round(deg, 1), 走向偏下降方向_度=round(turn, 1), 级数=n, 每级高_cm=round(H / n * 100, 1), 踏面深_cm=round((s1 - s0) / n * 100, 1),
               长_m=round(s1 - s0, 2), 宽_m=round(w1 - w0, 2), 总高_m=round(H, 2), 级数来源=f"按 {STAIR_RISE * 100:.0f} cm 一级",
               踏面点=int(sel.sum()), 楼梯掩码朝上点=int(len(Hh)), 在斜面上的占比=round(float(inl.mean()), 3))
    why = [t for t, bad in (("总高不够", H < STAIR_MIN_H), ("太短", s1 - s0 < STAIR_MIN_LEN),
                            ("斜面上的点太少", inl.mean() < STAIR_MIN_INL)) if bad]
    if why:
        rep["台阶"] = "不做: " + "、".join(why)
        return None, rep
    return st, rep


def stair_far_end(da3, keys, cams, vfm, st, up, e1, e2, mask_get):
    """楼梯远端: 远处的深度点稀 / 被置信度滤掉(joker 4 号镜头深度点到 s=24 m 就没了, 画面里楼梯还往上好几级)。
    改用画面: 每帧楼梯掩码的像素射线和楼梯斜面求交, 落在楼梯宽度内的取 97% 分位, 各帧取中位 = 远端。返回 s 或 None。"""
    c0, d0, d1, sl, s0, s1, w0, w1, n = st
    a3 = d0 * e1 + d1 * e2; n3 = -d1 * e1 + d0 * e2; far = []
    for k in keys:
        m = mask_get(vfm[k])
        if m is None:
            continue
        c = cams[k]; K = np.array(c["intrinsic_3x3"], float); h, w_ = c["input_hw"]; E = np.array(c["extrinsic_w2c_4x4"], float)
        ys, xs = np.nonzero(m[::4, ::4]); xs = (xs * 4 + 0.5) * w_ / m.shape[1]; ys = (ys * 4 + 0.5) * h / m.shape[0]
        R = E[:3, :3]; C = -R.T @ E[:3, 3]
        r = np.c_[(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones(len(xs))] @ R
        den = r @ up - sl * (r @ a3); t = (c0 + sl * (C @ a3) - C @ up) / np.where(np.abs(den) > 1e-9, den, np.nan)
        P = C + t[:, None] * r; ok = (t > 0) & np.isfinite(t)
        sv, wv = P[ok] @ a3, P[ok] @ n3; ok2 = (wv >= w0) & (wv <= w1) & (sv > s0)
        if ok2.sum() > 50:
            far.append(float(np.percentile(sv[ok2], 97)))
    return float(np.median(far)) if len(far) >= 3 else None


def stair_relevel(st, rep, per):
    """用画面里数出的周期重排(per 来自 stair_period); 数出来的每级高不在常理范围 -> 原样返回。"""
    if per is None:
        return st, rep
    st2 = stairs_from_period(st, per)
    rise = st2[3] * per["run"]
    rep = dict(rep, 画面数台阶=dict(周期_cm=round(per["run"] * 100, 1), 自相关=per["自相关"], 帧=per["帧"]))
    if not (STAIR_RISE_OK[0] <= rise <= STAIR_RISE_OK[1]):
        rep["画面数台阶"]["不用"] = f"每级高 {rise * 100:.1f} cm 不在 {STAIR_RISE_OK} m 内"
        return st, rep
    n = int(st2[8])
    rep.update(级数=n, 每级高_cm=round(rise * 100, 1), 踏面深_cm=round(per["run"] * 100, 1), 长_m=round(st2[5] - st2[4], 2),
               总高_m=round(rise * n, 2), 级数来源="画面里数的(台阶边的周期)")
    return st2, rep


def stair_period(da3, keys, cams, vfm, st, up, e1, e2, skip_of=None):
    """台阶级数从画面里数: 拟好的斜面在台阶宽度内取 15 条纵线投到画面上, 沿上坡方向逐厘米取亮度的纵向梯度(台阶边是一道道横线),
    去掉人的像素, 去趋势后求自相关 -> 各帧平均 -> 第一个峰 = 一级的深度(斜面上的, m); 再用 2~4 倍处的峰最小二乘精修;
    按这个周期把各帧的梯度折叠起来, 最亮的相位 = 台阶边的位置。
    返回 dict(run, phase(相对 s0), 自相关, 帧) 或 None(没有帧目录 / 不够明显)。"""
    import cv2
    c0, d0, d1, sl, s0, s1, w0, w1, n = st
    d3 = d0 * e1 + d1 * e2; n3 = -d1 * e1 + d0 * e2
    FR = json.load(open(f"{da3}/frame_map.json")).get("frames_dir")
    if not FR or not os.path.isdir(FR):
        return None
    S = np.arange(s0, s1, 0.01); L = len(S)
    if L < 300:
        return None
    lagmax = min(L // 2, 160)
    profs, acs = [], []
    for k in keys:
        p = f"{FR}/{int(vfm[k]):04d}.jpg"
        if not os.path.exists(p):
            continue
        c = cams[k]; K = np.array(c["intrinsic_3x3"], float); h, w_ = c["input_hw"]; E = np.array(c["extrinsic_w2c_4x4"], float)
        img = cv2.imread(p, 0).astype(float); sx, sy = img.shape[1] / w_, img.shape[0] / h
        gy = np.abs(cv2.Sobel(cv2.GaussianBlur(img, (3, 3), 0), cv2.CV_64F, 0, 1, ksize=3))
        bad = np.zeros(img.shape, bool)
        for msk in (skip_of(int(vfm[k])) if skip_of else []):
            bad |= cv2.resize(msk.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        prof = np.zeros(L); cnt = np.zeros(L)
        for wv in np.linspace(w0 + 0.1 * (w1 - w0), w1 - 0.1 * (w1 - w0), 15):
            P = S[:, None] * d3 + wv * n3 + (c0 + sl * S)[:, None] * up
            Pc = P @ E[:3, :3].T + E[:3, 3]; ok = Pc[:, 2] > 0.1
            x = (K[0, 0] * Pc[:, 0] / np.maximum(Pc[:, 2], 1e-6) + K[0, 2]) * sx
            y = (K[1, 1] * Pc[:, 1] / np.maximum(Pc[:, 2], 1e-6) + K[1, 2]) * sy
            ok &= (x >= 0) & (x < img.shape[1] - 1) & (y >= 0) & (y < img.shape[0] - 1)
            idx = np.nonzero(ok)[0]; xi, yi = x[idx].astype(int), y[idx].astype(int); g = ~bad[yi, xi]
            prof[idx[g]] += gy[yi[g], xi[g]]; cnt[idx[g]] += 1
        v = cnt > 5
        if v.sum() < 300:
            continue
        pr = np.where(v, prof / np.maximum(cnt, 1), 0.0)
        tr = np.convolve(pr, np.ones(100), "same") / np.maximum(np.convolve(v.astype(float), np.ones(100), "same"), 1)
        pp = np.where(v, pr - tr, 0.0)
        ac = np.correlate(pp, pp, "full")[L - 1:L - 1 + lagmax + 1]
        if ac[0] <= 0:
            continue
        profs.append(pp); acs.append(ac / ac[0])
    if len(acs) < 3:
        return None
    A = np.mean(acs, 0)
    pk = [l for l in range(15, lagmax) if A[l] > A[l - 1] and A[l] >= A[l + 1] and A[l] >= STAIR_PERIOD_MIN_AC]
    if not pk:
        return None
    l1 = pk[0]; num = den = 0.0
    for j in range(1, 5):                          # 精修: j 倍处附近的峰(抛物线插值), 过原点最小二乘
        a, b = int(j * l1 * 0.85), min(int(j * l1 * 1.15) + 1, lagmax)
        if b - a < 3:
            break
        l = a + int(np.argmax(A[a:b]))
        if l <= a or l >= b - 1 or l + 1 > lagmax:
            continue
        y0, y1, y2 = A[l - 1], A[l], A[l + 1]; dl = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if (y0 - 2 * y1 + y2) < 0 else 0.0
        num += j * (l + dl); den += j * j
    run = 0.01 * (num / den if den else l1)
    nb = max(int(round(run / 0.005)), 8); F = np.zeros(nb)       # 折叠求相位(0.5 cm 一格)
    ph = (((S - s0) % run) / run * nb).astype(int) % nb
    for pp in profs:
        np.add.at(F, ph, pp)
    F = np.convolve(np.r_[F[-2:], F, F[:2]], np.ones(5) / 5, "valid")
    return dict(run=float(run), phase=float((np.argmax(F) + 0.5) / nb * run), 自相关=round(float(A[l1]), 3), 帧=len(acs))


def stairs_from_period(st, per):
    """按画面里数出的周期和相位重排台阶: 台阶边落在 s0 + phase + j*run 上, 头尾各取离原起止最近的那道边。"""
    c0, d0, d1, sl, s0, s1, w0, w1, n = st
    run, phi = per["run"], per["phase"]
    j0 = int(np.ceil((-run / 2 - phi) / run)); a = s0 + phi + j0 * run
    n2 = max(3, int(round((s1 - a) / run)))
    return np.array([c0, d0, d1, sl, a, a + n2 * run, w0, w1, n2], float)


def apply_stairs(Gq, Gm, st, lo, res):
    """把楼梯那块(两侧再加 STAIR_SIDE_M)换掉: 查询表 = 所在那级的踏面高, 网格 = 挖空(由台阶方块补上)。"""
    c0, d0, d1, sl, s0, s1, w0, w1, n = st
    uu, vv = _cell_uv(lo, res, *Gq.shape)
    s = uu * d0 + vv * d1; w = -uu * d1 + vv * d0
    inside = (s >= s0) & (s <= s1) & (w >= w0 - STAIR_SIDE_M) & (w <= w1 + STAIR_SIDE_M)
    run = (s1 - s0) / n
    k = np.clip(np.floor((s - s0) / run), 0, n - 1)
    Gq = np.where(inside, c0 + sl * (s0 + (k + 0.5) * run), Gq)
    Gm = np.where(inside, np.nan, Gm)
    return Gq, Gm, int(inside.sum())


def build(r1, res=RES, max_frames=80):
    da3 = f"{r1}/02_da3"
    W = json.load(open(f"{da3}/world_up.json")); up = np.array(W["up"], float); up /= np.linalg.norm(up)
    _lv = json.load(open(f"{da3}/cameras.json")).get("镜头扶正") if os.path.exists(f"{da3}/cameras.json") else None
    _upd = UP_DEG_LEVELED if (_lv and any("扶正" in str(r.get("结论", "")) and "不扶" not in str(r.get("结论", "")) for r in _lv.get("各镜头", []))) else UP_DEG
    e1, e2 = basis(up); cu = np.cos(np.radians(_upd))
    fm = json.load(open(f"{da3}/frame_map.json")); fps = float(fm.get("fps") or 25.0); vfm = fm["frames"]
    cuts = np.array(json.load(open(f"{da3}/shot_cuts.json"))["cuts"] if os.path.exists(f"{da3}/shot_cuts.json") else [], float)
    PM = people_masks(r1)
    rail_get, n_rail = word_masks(r1, RAIL_WORDS, dilate=RAIL_DILATE)
    # ★ 11-11 其余检测到的物体(车、路灯、垃圾桶、消防梯……)的像素也不算地面: 车顶、消防梯平台都是朝上的面,
    #   joker 3 号镜头往下拍的街道上它们被建成一片片碎地面。03_masks 里的词都是有边界的物体(列词时就跳过了路面、墙这类背景),
    #   只有楼梯那几个词要留着当地面。
    obj_words = sorted({f[:-4] for f in os.listdir(f"{r1}/03_masks") if f.endswith(".npz")} - set(STAIR_WORDS) - set(RAIL_WORDS))         if os.path.isdir(f"{r1}/03_masks") else []
    obj_get, n_obj = word_masks(r1, obj_words, dilate=OBJ_DILATE)

    def skip_of(vf):
        out = []
        f = PM.get(int(vf))
        if f: out.append(f())
        r = rail_get(vf) if n_rail else None
        if r is not None: out.append(r)
        o = obj_get(vf) if n_obj else None
        if o is not None: out.append(o)
        return out

    U, Hh, SH, nall, nup = [], [], [], 0, 0
    KS = {}
    for k, (P, N, C) in iter_frames(da3, max_frames, skip_of=skip_of):
        sh = int(np.searchsorted(cuts, vfm[k] / fps + 1e-3, side="right"))
        KS.setdefault(sh, []).append(k)
        nall += len(P); s = N @ up > cu; nup += int(s.sum()); P = P[s]
        U.append(np.c_[P @ e1, P @ e2]); Hh.append(P @ up); SH.append(np.full(len(P), sh))
    U = np.concatenate(U); Hh = np.concatenate(Hh); SH = np.concatenate(SH)
    cams = _cams(da3)
    # 相机脚下: 相机高 - 相机离地(world_up.json 的"相机离地", up 那步定的; 没有就不加)。格子范围也要盖住相机(人常站在镜头跟前)
    cam_h = W.get("相机离地"); FEET = {}
    if cam_h:
        for k in sorted(cams):
            E = np.array(cams[k]["extrinsic_w2c_4x4"], float); C = -E[:3, :3].T @ E[:3, 3]
            sh = int(np.searchsorted(cuts, vfm[k] / fps + 1e-3, side="right"))
            FEET.setdefault(sh, []).append([C @ e1, C @ e2, C @ up - float(cam_h)])
    lo = np.percentile(U, 1, 0) - 0.5; hi = np.percentile(U, 99, 0) + 0.5
    if FEET:
        F_ = np.concatenate([np.array(v) for v in FEET.values()])[:, :2]
        lo = np.minimum(lo, F_.min(0) - 1.0); hi = np.maximum(hi, F_.max(0) + 1.0)
    nx, ny = (np.ceil((hi - lo) / res)).astype(int) + 1
    nshot = len(cuts) + 1
    Hq = np.full((nshot, ny, nx), np.nan, np.float32); ST = np.full((nshot, len(ST_KEYS)), np.nan)
    Gms, per = {}, []
    SP = _stair_points(r1, da3, KS, cams, vfm, up, e1, e2, cu, skip_of)
    _badk = _bad(da3)   # 脚点用全部好相机帧(建地形只抽了 max_frames 帧, 脚点便宜, 帧越多越稳)
    PF = people_feet(r1, da3, [k for k in sorted(cams) if k not in _badk and os.path.exists(f"{da3}/depth/{k[:-4]}_depth.npy")],
                     cams, vfm, up); PFS = {}
    for vf_, P_ in PF.items():
        PFS.setdefault(int(np.searchsorted(cuts, vf_ / fps + 1e-3, side="right")), []).append(np.c_[P_ @ e1, P_ @ e2, P_ @ up])
    PFS = {s: np.concatenate(v) for s, v in PFS.items()}
    grids = {s: (_grid(U[SH == s], Hh[SH == s], lo, nx, ny, res, np.array(FEET.get(s, []), float).reshape(-1, 3), PFS.get(s))
                 if (SH == s).sum() > 1000 else None) for s in range(nshot)}
    fits = {s: stair_fit(*SP[s]) for s in SP if grids.get(s) is not None}
    sget, _ = word_masks(r1, STAIR_WORDS)
    for s, (st, srep) in fits.items():                 # 远端按画面里楼梯掩码延长(最多再长一半)
        if st is None:
            continue
        fe = stair_far_end(da3, KS[s], cams, vfm, st, up, e1, e2, sget)
        if fe is not None and fe > st[5]:
            new = min(fe, st[5] + 0.5 * (st[5] - st[4]))
            # ★ 11-12 延长后按原来的级高重算级数: 以前只改长度不改级数, joker 主镜头 9.9 m 拉到 14.9 m 还是 28 级,
            #   每级被摊成 25 cm 高、53 cm 深(画面数周期那步因为 25 cm 超出常理没接手, 就一直错着)
            rise0 = st[3] * (st[5] - st[4]) / max(int(st[8]), 1)
            srep["远端按画面延长_m"] = round(new - st[5], 2); st[5] = new
            if rise0 > 1e-6:
                st[8] = max(1, int(round(st[3] * (st[5] - st[4]) / rise0)))
                srep.update(级数=int(st[8]), 长_m=round(float(st[5] - st[4]), 2), 总高_m=round(float(st[3] * (st[5] - st[4])), 2))
    pers = {s: stair_period(da3, KS[s], cams, vfm, f[0], up, e1, e2, skip_of) for s, f in fits.items() if f[0] is not None}
    best = max(((s, p) for s, p in pers.items() if p), key=lambda sp: sp[1]["自相关"], default=None)
    for s in range(nshot):
        r = grids[s]
        if r is None:
            per.append(dict(镜头=s + 1, 建成=False)); continue
        G, Gq, Gm, nmeas, nspike = r
        srep = None
        if s in fits:
            st, srep = fits[s]
            if st is not None:
                p = pers.get(s)
                if p is None and best is not None:       # 这个镜头数不出来(楼梯太远 / 太小): 同一段楼梯, 借数得最清楚那个镜头的周期, 相位不知道
                    p = dict(best[1], phase=0.0, 帧=0); srep["借用周期"] = f"{best[0] + 1} 号镜头"
                st, srep = stair_relevel(st, srep, p)
                Gq, Gm, ncut = apply_stairs(Gq, Gm, st, lo, res); ST[s] = st; srep["挖掉的格"] = ncut
        Hq[s] = Gq; Gms[s] = Gm
        v = Gm[np.isfinite(Gm)]
        per.append(dict(镜头=s + 1, 建成=True, 量到格=nmeas, 剔尖刺格=nspike, 补后格=int(np.isfinite(Gm).sum()),
                        高度_5_50_95=[round(float(np.percentile(v, q)), 2) for q in (5, 50, 95)] if len(v) else None,
                        台阶=srep))
    if not Gms:
        raise SystemExit("[失败] 没有一个镜头建得出地形(朝上的点太少)")
    main = max(Gms, key=lambda s: per[s]["量到格"])          # 主镜头 = 量到格子最多的(交付的场景里铺它的地)
    # 镜头对齐(pipeline/shot_register.py)挪过 / 转过的镜头: 已经放进主场景了, 人和物体都按主场景的地面(统一的场景)
    reg = json.load(open(f"{da3}/cameras.json")).get("镜头竖直对齐") or {}
    joined = [r["镜头"] - 1 for r in reg.get("各镜头", []) if r.get("转了_度") or r.get("竖直挪_m")]
    # 统一场景的地面 = 主镜头的 + 别的镜头量到、主镜头没量到的(先在两边都量到的格子上比高差, 中位差 < MERGE_TOL 才并, 并时补上这个差)。
    # joker: 1 / 2 号镜头和主镜头重叠 21054 / 5810 格, 高差中位 +0.05 / -0.05 m; 1 号拍到了楼梯底下的街道, 主镜头没有,
    # 不并的话 3 号镜头(转正后往下拍街道)画面是空的。转过 / 挪过的镜头(joined)自己的深度和主镜头对不上, 不拿来并。
    uu, vv = _cell_uv(lo, res, ny, nx)
    def _inst(st):
        if not np.isfinite(st).all(): return np.zeros((ny, nx), bool)
        sv = uu * st[1] + vv * st[2]; wv = -uu * st[2] + vv * st[1]
        return (sv >= st[4]) & (sv <= st[5]) & (wv >= st[6]) & (wv <= st[7])
    Gu = Gms[main].copy(); Hu = Hq[main].copy(); known_m = np.isfinite(Gu) | _inst(ST[main]); merged = []
    for s in sorted((s for s in Gms if s != main and s not in joined), key=lambda s: -np.isfinite(Gms[s]).sum()):
        ks_ = np.isfinite(Gms[s]) | _inst(ST[s]); ov = known_m & ks_
        if ov.sum() < 200:
            continue
        off = float(np.median((Hq[main] - Hq[s])[ov]))
        if abs(off) > MERGE_TOL:
            per[s]["并进统一场景"] = f"不并: 和主镜头高差 {off:+.2f} m"; continue
        add = ~known_m & np.isfinite(Gms[s])
        Gu[add] = Gms[s][add] + off; Hu[add] = Gu[add]; known_m |= add
        per[s]["并进统一场景"] = dict(高差_m=round(off, 2), 重叠格=int(ov.sum()), 并进格=int(add.sum())); merged.append(s)
    # 转过 / 挪过的镜头: 它自己的深度和主镜头对不上(高差不拿来判), 但统一场景里别人都没拍到的地方只有它拍到了
    # (joker 3 号镜头往下拍的街道, 12~25 m, 1/2/4 号镜头都没拍到 —— 不补的话 8 m 以外是空的, 人站在空地边缘后面被挡住)。
    # 只补"谁都没量到"的格子, 在重叠处对一个高差(中位)再补。
    for s in joined:
        if s == main or s not in Gms:
            continue
        ks_ = np.isfinite(Gms[s]) | _inst(ST[s]); ov = known_m & ks_
        if ov.sum() < 200:
            continue
        off = float(np.median((Hq[main] - Hq[s])[ov]))
        add = ~known_m & np.isfinite(Gms[s])
        # 不加高差: 这片只有它拍到, 它的人就站在这片上 —— 用它自己的高度, 人才既对得上原片又踩在看得见的地上;
        # 和别人量到的地方之间会有一道落差(joker 3 号镜头约 1.5 m), 网格在那里断开(MESH_JUMP)
        Gu[add] = Gms[s][add]; Hu[add] = Gu[add]; known_m |= add
        per[s]["补进统一场景"] = dict(交界高差_m=round(off, 2), 重叠格=int(ov.sum()), 补进格=int(add.sum()))
    Gms[main] = Gu; Hq[main] = Hu
    # ★ 11-11 试过让转过 / 挪过的镜头的人改站统一场景的地面: joker 3 号镜头的人被抬 4.5 m, 对齐从 4.5% 变成 307%(楼梯被压扁,
    #   和这个镜头自己的深度对不上)。撤了: 人还站在自己镜头的地面上(镜头整体转 / 挪过, 相对相机不变, 对齐不受影响)。
    for s in joined:
        if s != main and 0 <= s < nshot:
            per[s]["已对齐进统一场景"] = True
    # ★ 2026-11-12 地面模型(GROUND_STYLE=plane_stairs, 用户定的: "拟合一个平面和一个干净的台阶"):
    #   平面 = 主镜头相机脚下(相机高中位 - 相机离地); 台阶 = 主镜头拟合出来的参数化台阶, 按同样的级高 / 踏面往下补到平面。
    #   所有镜头的人都站在这个模型上(Hq 全换成模型); 交付的场景地面也只有它(Terrain.mesh)。各镜头自己量到的地面留在 Hm,
    #   只给镜头对齐(shot_register)用。joker(镜头扶正后): 楼梯 25.5°, 往下补 11 级, 起点正好落在主镜头相机脚下(差 0.26 m) ——
    #   相机站在楼梯底往上拍, 前后一致。高度表拼地面的老做法: 台阶两侧挖 0.3 m、高差断开, 一道道缝; 往下看的镜头里街被补出来的地盖住。
    model = None; plane_h = np.nan
    _sty = _run_style(r1)
    if _sty in ("plane_stairs", "plane_stairs_mesh") and np.isfinite(ST[main]).all():
        _bd = _bad(da3)
        Cz = [float((-np.array(c["extrinsic_w2c_4x4"], float)[:3, :3].T @ np.array(c["extrinsic_w2c_4x4"], float)[:3, 3]) @ up)
              for k, c in cams.items() if k not in _bd and int(np.searchsorted(cuts, vfm[k] / fps + 1e-3, side="right")) == main]
        if Cz:
            plane_h = float(np.median(Cz)) - float(cam_h or 1.5)
            st = ST[main].copy(); bx = stair_boxes(st); run = bx[0][1] - bx[0][0]
            rise = (bx[1][4] - bx[0][4]) if len(bx) > 1 else STAIR_RISE
            n_add = max(0, int(np.ceil((bx[0][4] - plane_h) / rise)) - 1)
            st[4] -= n_add * run; st[8] += n_add; ST[main] = st
            sv = uu * st[1] + vv * st[2]; wv = -uu * st[2] + vv * st[1]
            ins = (sv >= st[4]) & (sv <= st[5]) & (wv >= st[6]) & (wv <= st[7])
            # 人查的高度 = 所在那一级的踏面高(和网格一致)。★ 11-12 试过改成穿过踏面中点的连续斜面: 1 号镜头 0.8% -> 10.4%、
            #   3 号 1.6% -> 7.4%, 主镜头只 11% -> 8.7%, 撤了(主镜头偏的真原因见问题记录第 27 条: 坐着的判定翻来翻去)。
            kk = np.clip(np.floor((sv - st[4]) / run).astype(int), 0, int(st[8]) - 1)
            M = np.where(ins, st[0] + st[3] * (st[4] + (kk + 0.5) * run), plane_h).astype(np.float32)
            # plane_stairs_mesh(★ 11-12 用户定的): 只换交付的地面网格, 人还站各镜头自己量到的地面、相机一个不动 ——
            #   plane_stairs 让人也站模型, 要配镜头扶正 + 以人为锚挪相机, 物体是几个镜头的点拼的, 相机一挪就和画面对不上
            #   (主镜头路灯挡在人前面、两侧消防梯不见)。
            if _sty == "plane_stairs":
                Gms[main] = M.copy(); Hq[:] = M
            model = dict(平面高_m=round(plane_h, 2), 主镜头相机离平面_m=round(float(np.median(Cz)) - plane_h, 2), 往下补的级数=n_add,
                         台阶=dict(级数=int(st[8]), 每级高_cm=round(rise * 100, 1), 踏面深_cm=round(run * 100, 1),
                                 坡度_度=round(float(np.degrees(np.arctan(st[3]))), 1), 宽_m=round(float(st[7] - st[6]), 2),
                                 总高_m=round(float(st[3] * (st[5] - st[4])), 2)))
            # ★ 11-12 分层(用户: "楼梯是分段的""最低的平面上还有汽车垃圾桶"; 算法一直是找最低的地面):
            #   上面的平面只是主镜头相机脚下(= 主楼梯底下那块平台), joker 往下看的镜头量到的街比它低 7.5 m, 车(底高 -8.7~-9.2)
            #   停在街上 —— 只铺一个平面会把街整个盖住, 人看着站在大平地上。所以: 平面 = 量到的最低那层; 主楼梯底下的平台
            #   做成实心平台; 平台和街之间的那段楼梯按看得到它的镜头里楼梯掩码的点拟合; 主楼梯往上延长到顶上站在地上的物体。
            def _stand_h(P, vf_):      # 人在他那个镜头里实际站的高度(摆人查的就是这张 Hq)
                s_ = int(np.searchsorted(cuts, vf_ / fps + 1e-3, side="right")); G_ = Hq[s_] if np.isfinite(Hq[s_]).any() else Hq[main]
                ij = np.floor((np.c_[P @ e1, P @ e2] - lo) / res).astype(int)
                return G_[np.clip(ij[:, 1], 0, ny - 1), np.clip(ij[:, 0], 0, nx - 1)]
            foot_h = plane_h; extra, lay = _layers(r1, da3, cams, vfm, cuts, fps, up, Hh, SH, st, foot_h, skip_of, PF, _stand_h)
            if lay:
                model["分层"] = lay
                if lay.get("主楼梯往上补到_s") is not None:
                    st[8] = max(1, int(round(st[3] * (lay["主楼梯往上补到_s"] - st[4]) / rise))); st[5] = lay["主楼梯往上补到_s"]; ST[main] = st
                    model["台阶"].update(级数=int(st[8]), 总高_m=round(float(st[3] * (st[5] - st[4])), 2))
                if lay.get("路面高_m") is not None:
                    land_h = lay.get("平台", {}).get("高_m", foot_h)
                    k_more = int(round((foot_h - land_h) / rise)) if land_h < foot_h else 0
                    if k_more > 0:                     # 平台按人脚定得比"相机脚下"低: 主楼梯往下再补几级接到平台
                        st[4] -= k_more * run; st[8] += k_more; ST[main] = st
                        lay["平台"]["主楼梯往下再补"] = k_more
                        if len(extra):                 # 平台方块的 s 终点跟着主楼梯起点走
                            extra = np.asarray(extra, float); extra[-1, 1] = st[4]
                    plane_h = lay["路面高_m"]; model["平面高_m"] = round(plane_h, 2); model["平台高_m"] = round(float(land_h), 2)
            snap, prep = _pedestals(r1, up, st, extra, plane_h)
            if len(snap):
                model.setdefault("分层", {})["落地高度块"] = prep
    if model is None or "分层" not in model:
        extra = np.zeros((0, 6))
    if model is None or "落地高度块" not in model.get("分层", {}):
        snap = np.zeros((0, 6))
    # ★ 11-12 分层的下段楼梯 / 平台范围里, 所有镜头的查询高度(人站的、补腿 / 贴地查的)直接等于交付网格的顶高:
    #   平台高已经按人站的高定了(_layers), 但补腿按骨盆正下方查 Hq 摆脚, 和网格的格子 / 分位差几厘米, joker 往下看的镜头
    #   Joker 脚还是比平台低 7~9 cm(最多 17 cm)。在这块让两边是同一个数, 人站在哪看到的地就在哪。
    #   只填建成了的镜头: 第一版把推特写(没建成, 整张 NaN)也填了一块, 查高度不再退回主镜头, 补腿拿到 NaN, SVD 不收敛整步失败。
    if model is not None and len(extra):
        _st = ST[main]; _sv = uu * _st[1] + vv * _st[2]; _wv = -uu * _st[2] + vv * _st[1]
        _built = [s_ for s_ in range(nshot) if np.isfinite(Hq[s_]).any()]   # 没建成的镜头整张 NaN = 查的时候退回主镜头; 填了一块就不退了
        for sa, sb, w0, w1, top, base in np.asarray(extra, float).reshape(-1, 6):
            _m = (_sv >= sa) & (_sv <= sb) & (_wv >= w0) & (_wv <= w1)
            for s_ in _built:
                Hq[s_][_m] = top
    Hm = np.stack([Gms.get(s, np.full((ny, nx), np.nan)) for s in range(nshot)]).astype(np.float32)
    out = f"{r1}/07_terrain.npz"
    np.savez(out, up=up, e1=e1, e2=e2, lo=lo, res=res, H=Gms[main].astype(np.float32), Hq=Hq, ST=ST, Hm=Hm,
             cuts=cuts, fps=fps, main=main, plane_h=plane_h, model=(0 if model is None else 1 if _sty == "plane_stairs" else 2),
             extra=np.asarray(extra, float).reshape(-1, 6), snap=np.asarray(snap, float).reshape(-1, 6))
    rep = dict(格子_m=res, 范围_m=[round(float(x), 1) for x in (hi - lo)], 朝上点占比=round(nup / max(nall, 1), 3),
               去掉人的帧=len(PM), 去掉栏杆的帧=n_rail, 人的脚点=int(sum(len(v) for v in PF.values())), 去掉的物体词=obj_words, 相机离地_m=cam_h,
               镜头数=nshot, 主镜头=main + 1, 地面模型=model, 各镜头=per)
    json.dump(rep, open(f"{r1}/07_terrain.json", "w"), ensure_ascii=False, indent=1)
    _preview(Hq[main], Gms[main], f"{r1}/07_terrain_高度.jpg")
    print(f"[地形] {json.dumps(rep, ensure_ascii=False)}\n       -> {out}")
    return out


def _model_h_sw(s_, w_, st, extra, plane_h):
    """交付的地面(平面 + 主楼梯 + 额外方块)在主楼梯坐标 (s, w) 处的顶高。"""
    s_, w_ = np.asarray(s_, float), np.asarray(w_, float); h = np.full(s_.shape, plane_h, float)
    if np.isfinite(st).all():
        run = (st[5] - st[4]) / st[8]; ins = (s_ >= st[4]) & (s_ <= st[5]) & (w_ >= st[6]) & (w_ <= st[7])
        k = np.clip(np.floor((s_ - st[4]) / run), 0, st[8] - 1)
        h = np.where(ins, np.maximum(h, st[0] + st[3] * (st[4] + (k + 0.5) * run)), h)
    for sa, sb, w0, w1, top, base in np.asarray(extra, float).reshape(-1, 6):
        h = np.where((s_ >= sa) & (s_ <= sb) & (w_ >= w0) & (w_ <= w1), np.maximum(h, top), h)
    return h


PED_MIN = 0.5     # 立在地上的物体底高比脚下交付的地面高这么多以上, 才给它立底座柱


def _pedestals(r1, up, st, extra, plane_h):
    """立在地上的物体(GROUNDED, 06_all_boxes.json 的底高)底下交付的地面比它低 PED_MIN 以上 -> 记一块"落地高度"(盒子水平范围,
       顶 = 它的底高), 组装场景时它落到这个高度上。不进网格: 真正的柱子由物理那步(scene_phys.py)按摆好的网格的灯杆底立
       (按盒子范围立的柱子太粗, joker 路灯 7 的柱子穿进消防梯 4, 11.6% 的顶点在里面)。原来是: 立一根柱子
       从路面到它的底高(柱子盖住盒子的水平范围)。joker 主楼梯两边的路灯立在台阶两侧的矮墙上(离楼梯边 0.5~1.1 m), 交付的楼梯没有矮墙,
       底下是 9 m 下面的路面。★ 试过把整段楼梯加宽到盖住路灯: 撞上挂在楼上的消防梯, 3 个消防梯底部埋进楼梯 1.3~2.0 m, 撤了。"""
    bj = f"{r1}/06_all_boxes.json"
    if not os.path.exists(bj) or not np.isfinite(st).all():
        return np.zeros((0, 6)), {}
    e1, e2 = basis(up); dvec = st[1] * e1 + st[2] * e2; nwv = -st[2] * e1 + st[1] * e2
    try:                                    # 闸口 / VLM 否决、不进场景的物体不立柱子(joker 路灯 5 被否决, 它的柱子把旁边路灯 0 顶高 0.6 m)
        sys.path.insert(0, HERE); import review_decisions as _rv; rej = _rv.vlm_rejected(r1)
    except Exception:
        rej = {}
    rows, names = [], []
    for w, bs in json.load(open(bj)).items():
        if w not in GROUNDED or not isinstance(bs, list):
            continue
        for b in bs:
            bh = b.get("底高")
            if bh is None or b.get("obj_id") in rej.get(w, ()):
                continue
            # 柱子盖住整个盒子的水平范围: 灯有灯臂, 盒子中心不是灯杆底(只立在中心, 路灯 2 的灯杆底落在柱子外, 悬空 9.7 m)
            c = np.array(b["中心"], float); a = np.array(b.get("长轴", [1, 0, 0]), float); a = a - up * (a @ up); a /= max(np.linalg.norm(a), 1e-9)
            bb = np.cross(up, a); L, Wd = float(b["尺寸"][0]) / 2, float(b["尺寸"][1]) / 2
            cor = np.array([c + i * L * a + j * Wd * bb for i in (-1, 1) for j in (-1, 1)])
            ss, ww = cor @ dvec, cor @ nwv
            g = float(_model_h_sw(np.r_[ss, c @ dvec], np.r_[ww, c @ nwv], st, extra, plane_h).min())
            if bh - g > PED_MIN:
                rows.append((float(ss.min()) - 0.1, float(ss.max()) + 0.1, float(ww.min()) - 0.1, float(ww.max()) + 0.1, float(bh), plane_h))
                names.append(f"{w}#{b.get('obj_id')} 底高 {bh:+.2f}(下面原来 {g:+.2f})")
    return np.array(rows, float).reshape(-1, 6), {"个数": len(rows), "给谁": names}


LOW_LAYER_FRAC, LOW_LAYER_MIN = 0.02, 2000   # 最低那层: 朝上的点至少占这么多(比例 / 个数)才算一层地面
LEVEL_GAP = 1.0      # 最低那层比主楼梯底下的平台低这么多(m)以上 = 分层(中间有一段楼梯)
FLIGHT_DEG = (15.0, 45.0)   # 量出来的下段楼梯坡度要在这个范围内才用
GROUNDED = ("car", "truck", "van", "bus", "trash can", "street lamp", "lamp post", "bollard", "newspaper box", "cone",
            "fire hydrant", "bench", "mailbox", "parking meter", "potted plant")   # 立在地上的物体: 底高 = 那里的地面高


def _layers(r1, da3, cams, vfm, cuts, fps, up, Hh, SH, st, foot_h, skip_of, PF=None, stand_h=None):
    """分层地面(见 build 里调用处)。返回 (额外方块 (K,6) = (s_a, s_b, w0, w1, 顶高, 底高) 在主楼梯的 d / nw 坐标里, 报告)。
       1) 最低那层: 所有镜头朝上的点(已扣人 / 栏杆 / 物体)按 0.25 m 分层, 最低的一层(>= LOW_LAYER_FRAC / LOW_LAYER_MIN)取中位。
       2) 比平台低 LEVEL_GAP 以上 -> 平台和最低层之间有一段楼梯: 在最低层点最多的那个镜头里, 取楼梯掩码里的点(不卡法向:
          往下看时台阶面大多斜着, 25° 的朝上门槛会把整段扔掉), 每 1 m 取高度中位, 直线拟合 h(s);
          底 = 拟合线到最低层的位置, 顶 = 到平台高的位置; 级高约 STAIR_RISE。平台 = 下段楼梯顶到主楼梯底。宽 = 点的横向范围并上主楼梯。
       3) 主楼梯往上: 立在地上的物体(06_all_boxes.json 的底高)在主楼梯顶以外、底高离主楼梯斜面延长线 < 1 m 的, 主楼梯延长到最远那个。"""
    rep = {}; extra = []
    import json as _j
    e1, e2 = basis(up); dvec = st[1] * e1 + st[2] * e2; nwv = -st[2] * e1 + st[1] * e2
    # 3) 主楼梯往上延长
    bj = f"{r1}/06_all_boxes.json"
    if os.path.exists(bj):
        far = []
        for w, bs in _j.load(open(bj)).items():
            if w not in GROUNDED or not isinstance(bs, list):
                continue
            for b in bs:
                c = np.array(b["中心"], float); s_ = float(c @ dvec); w_ = float(c @ nwv); bh = b.get("底高")
                if bh is None or s_ <= st[5] or not (st[6] - 1.5 <= w_ <= st[7] + 1.5):
                    continue
                if abs(bh - (st[0] + st[3] * s_)) < 1.0 and s_ - st[5] < 20.0:
                    far.append((s_, w, b.get("obj_id")))
        if far:
            s_new = max(f[0] for f in far) + 0.5
            rep["主楼梯往上补到_s"] = round(s_new, 2); rep["往上补的依据"] = [f"{w}#{i} s={s_:.1f}" for s_, w, i in sorted(far)]
    # 1) 最低那层
    if not len(Hh):
        return np.zeros((0, 6)), rep
    bins = np.arange(np.floor(Hh.min()), Hh.max() + 0.25, 0.25); cnt, _ = np.histogram(Hh, bins)
    thr = max(LOW_LAYER_MIN, LOW_LAYER_FRAC * len(Hh)); i0 = next((i for i, c in enumerate(cnt) if c >= thr), None)
    if i0 is None:
        return np.zeros((0, 6)), rep
    sel = (Hh >= bins[i0] - 0.5) & (Hh < bins[i0] + 0.75); low_h = float(np.median(Hh[sel]))
    low_shot = int(np.bincount(SH[sel]).argmax())
    rep.update(最低一层_m=round(low_h, 2), 最低一层的镜头=low_shot + 1, 最低一层点数=int(sel.sum()))
    if low_h > foot_h - LEVEL_GAP:
        return np.zeros((0, 6)), rep       # 一层: 平面就是平台那层(主楼梯已往下补到它)
    # 2) 下段楼梯
    sget, _ = word_masks(r1, STAIR_WORDS); bad = _bad(da3)
    ks = [k for k in sorted(cams, key=lambda k: vfm[k]) if k not in bad and os.path.exists(f"{da3}/depth/{k[:-4]}_depth.npy")
          and int(np.searchsorted(cuts, vfm[k] / fps + 1e-3, side="right")) == low_shot]
    S_, H_, W_ = [], [], []
    for k in ks[::2]:
        vf = int(vfm[k]); sm = sget(vf)
        if sm is None:
            continue
        P, N, C = frame_points(da3, k, cams[k], skip=[~sm] + (skip_of(vf) if skip_of else []))
        if len(P):
            S_.append(P @ dvec); H_.append(P @ up); W_.append(P @ nwv)
    if not S_:
        rep["下段楼梯"] = "看不到最低层的那个镜头里没有楼梯掩码"; return np.zeros((0, 6)), rep
    S_, H_, W_ = np.concatenate(S_), np.concatenate(H_), np.concatenate(W_)
    keep = (H_ > low_h + 0.2) & (H_ < foot_h + 0.5); S_, H_, W_ = S_[keep], H_[keep], W_[keep]
    xs, ys = [], []
    for a in np.arange(np.floor(S_.min()) if len(S_) else 0, (S_.max() if len(S_) else 0) + 1, 1.0):
        b = (S_ >= a) & (S_ < a + 1)
        if b.sum() >= 30:
            xs.append(a + 0.5); ys.append(float(np.median(H_[b])))
    if len(xs) < 3:
        rep["下段楼梯"] = f"楼梯掩码里的点太少({len(S_)})"; return np.zeros((0, 6)), rep
    a_, b_ = np.polyfit(xs, ys, 1); deg = float(np.degrees(np.arctan(abs(a_))))
    if a_ <= 0 or not (FLIGHT_DEG[0] <= deg <= FLIGHT_DEG[1]):
        rep["下段楼梯"] = f"拟合坡度 {deg:.1f}° / 方向不对, 不用"; return np.zeros((0, 6)), rep
    # ★ 11-12 平台高按站在平台上的人脚(量到的)定, 不按"主镜头相机离地 1.5 m"推: joker 往下看的镜头 Joker 脚在 -1.45~-1.64 m,
    #   按相机推的平台 -1.38 m, 脚陷进去 7~18 cm; 而且平台起点(拟合线到 -1.38 的位置)正好在他脚边, 他一半的帧踩在下段楼梯最上面两级。
    #   人脚: 水平在下段楼梯顶附近到主楼梯底之间(s 拟合线到平台高 -2 m ~ 主楼梯起点)、高度离相机脚下 0.6 m 以内的。
    land_h = foot_h; s_g = (foot_h - b_) / a_
    if PF:
        F = np.concatenate([np.asarray(v, float).reshape(-1, 3) for v in PF.values()])
        FV = np.concatenate([[vf_] * len(np.asarray(v).reshape(-1, 3)) for vf_, v in PF.items()])
        fs, fw, fh = F @ dvec, F @ nwv, F @ up
        sel = (fs >= s_g - 2.0) & (fs <= st[4]) & (fw >= st[6] - 1.0) & (fw <= st[7] + 1.0) & (np.abs(fh - foot_h) < 0.6)
        if sel.sum() >= 10:
            # 取人实际站的高(他那个镜头的查询高度表在脚下的值, 摆人 / 贴地 / 补腿都按它): 第一版取脚点高度中位 -1.43 m,
            #   但他那个镜头的高度表在脚下按偏低分位取(-1.55 m 左右), 人摆好后脚还是比平台低 11~13 cm
            if stand_h is not None:
                hs_ = np.array([float(stand_h(F[i:i + 1], int(FV[i]))[0]) for i in np.nonzero(sel)[0]])
                hs_ = hs_[np.isfinite(hs_)]
            else:
                hs_ = fh[sel]
            if len(hs_) >= 10:
                land_h = float(np.median(hs_))
                rep["平台高按人脚"] = dict(脚点=int(sel.sum()), 人站的高_中位_m=round(land_h, 2), 脚点高_中位_m=round(float(np.median(fh[sel])), 2), 相机脚下_m=round(foot_h, 2))
    s_bot = (low_h - b_) / a_; s_top = min((land_h - b_) / a_, st[4])
    if s_top - s_bot < 1.0:
        rep["下段楼梯"] = "拟合出来太短"; return np.zeros((0, 6)), rep
    w0 = min(st[6], float(np.percentile(W_, 5))); w1 = max(st[7], float(np.percentile(W_, 95)))
    n = max(1, int(round((land_h - low_h) / STAIR_RISE))); rr = (land_h - low_h) / n; rn = (s_top - s_bot) / n
    for k in range(n):
        extra.append((s_bot + k * rn, s_bot + (k + 1) * rn, w0, w1, low_h + (k + 1) * rr, low_h))
    if st[4] - s_top > 0.05:
        extra.append((s_top, st[4], w0, w1, land_h, low_h))       # 平台(s 终点在 build 里跟主楼梯起点再对一次)
    rep.update(路面高_m=round(low_h, 2), 下段楼梯=dict(坡度_度=round(deg, 1), 级数=n, 每级高_cm=round(rr * 100, 1), 踏面深_cm=round(rn * 100, 1),
                                               s=[round(s_bot, 2), round(s_top, 2)], 宽_m=round(w1 - w0, 2), 拟合用的分段=len(xs)),
               平台=dict(高_m=round(land_h, 2), s=[round(s_top, 2), round(float(st[4]), 2)]))
    return np.array(extra, float), rep


def _preview(Gq, Gm, path):
    """俯视高度图: 网格铺到的格子(含台阶那块, 用查询表的踏面高)上色, 其余暗灰。"""
    import cv2
    show = np.isfinite(Gm)
    v = Gq[np.isfinite(Gq)]
    if not len(v):
        return
    a, b = np.percentile(v, 1), np.percentile(v, 99)
    g = np.clip((np.nan_to_num(Gq, nan=a) - a) / max(b - a, 1e-6), 0, 1)
    img = cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    img[~show] = (img[~show] * 0.35).astype(np.uint8)
    s = 800 / max(img.shape[:2])
    img = np.ascontiguousarray(cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                                          interpolation=cv2.INTER_NEAREST)[::-1])
    cv2.putText(img, f"terrain height {a:.2f} .. {b:.2f} m (top view, dim = not meshed)", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(path, img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--frames", type=int, default=80)
    a = ap.parse_args()
    build(a.r1, max_frames=a.frames)
