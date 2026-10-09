#!/usr/bin/env python
"""人进场景的两件事:
  1. 修人要用的**判据**(一组函数): 远近按场景深度、哪几帧算认错人、位置怎么整理和贴地、谁算坐着、坐哪把椅子、
     坐着的人怎么摆、物理用的人体空气墙。摆人那步(body_motion.py)调它们, 在模型的关节数据上修。
  2. **合成**(流水线的 people_glb 步, 物理之后, 就是这个文件的 main): 椅子被物理挪了就让坐着的人跟着挪、
     身高自检、场景里摘掉静态人形(-> 09_scene_nopeople.glb)、出几帧验收图。人读摆人那步存下来的(people_store.py)。
(会动的人的交付物是骨骼动画的 FBX, 见 motion_export.py。)

★ 2026-10-09 以前这里还有一整套"在网格上修人"(solve_people / apply_corr: 算一张校正表, 每个读者拿它在网格上重放一遍),
  给网页验收视频、物理、3D 页用; FBX 走 body_motion.py 在关节数据上另修一遍。同一个人修了两遍, 后加的修复只有一边有。
  网格那一套删了: 判据留在这里, 修只在 body_motion.py 做一遍。

★ 为什么要把人"摆进场景": SAM 3D Body 只会说"这个人在**我镜头前方**偏右 3.8 米、远 10.8 米",
  它不知道镜头本身在哪、朝哪、这一帧和下一帧镜头动了多少。所以它的产物只能贴回原片,
  拿不到"这个人在这条街上的哪个位置"。缺的那块正是重建流水线有的。

★ 顺带修一个现在就错的东西: 量盒子那步假设物体静止, 把一个人走过的一整条路合并成
  一个盒子。pedestrian 实测 14 个 person 里有 11.78x1.83x3.26m 的"人";
  realstreet 4 个 person 没一个尺寸对。所以走动的人**本来就不该**走静态物体那条路。

★ 被挡住的人: SAM 3D Body 会把挡在前面的人当成他重建(位置跳到别人那儿、动作换成别人的)。
  套得稳(跟着邻居走)的不动; 在两个人之间来回认、身体朝向一帧翻 30° 以上的段换掉 —— 见"翻转补帧"那节。

用法(合成; 流水线的 people_glb 步就是这一条):
  python body_into_scene.py --r1 <场景结果目录> --body <body4d轮次目录> [--render --fr <场景帧目录>]

坐标系(每一步都踩过坑, 别凭印象改):
  cameras.json         OpenCV: x右 y下 z朝前; 模型给的根平移(focal_4d_individual 的 camera)也是这个系
  摆人的结果           DA3 世界系(= 相机位姿所在的系), 米
  07_scene_phy.glb     assemble_scene 末尾乘过 export_T(UP), 真实的天在 +y -> 人要和它放一起得再乘 export_T(UP)

sam3d 环境(trimesh + pygltflib + cv2 + numpy)。纯 CPU。
"""
import argparse, glob, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import people_identity, people_store
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/sam3d/data/outdoor/scripts"))
from _common import load_frame_map
from world_frame import load_up, export_T

# 身高的合理区间。落在外面说明场景尺度是坏的 —— 不是把人缩放一下就能糊过去的:
# 人是绝对米制的, 错的是场景。office 那种相机离地 0.23m 的场景会在这里被挡住。
H_MIN, H_MAX = 1.40, 2.00
# 每个人"场景深度 / 人体模型给的距离"的比例本该扎堆(尺度偏差对所有人一样); 离全体中位超过这么多的人, 多半是深度被
# 遮挡物带偏了, 或者根本不是人(office 搭在椅背上的外套 0.49)。算尺度(world_scale.py)和摆人(body_motion.py)用同一条线。
RATIO_TOL = 0.25


# ------------------------------------------------------------------ 读
def load_scene(r1):
    """读场景这一半: 相机、天、帧号映射。"""
    da3 = f"{r1}/02_da3" if os.path.isdir(f"{r1}/02_da3") else f"{r1}/02_DA3点云"
    cams = json.load(open(f"{da3}/cameras.json"))["cameras"]
    UP, GH, _ = load_up(da3, "人体入场")
    fmap = load_frame_map(da3)                      # {帧文件名: 原视频帧号}
    vid2key = {v: k for k, v in fmap.items()}       # 反过来: 原视频帧号 -> 相机的键
    return da3, cams, np.asarray(UP, float), float(GH), vid2key


# ------------------------------------------------------------------ 体检
def measure(world, UP, GH):
    """每个人的体检: 身高、脚离地、帧间位移。这是唯一能判断"摆对了没有"的自查。"""
    rows = {}
    for pid, fr in sorted(world.items()):
        vfs = sorted(fr)
        h = [fr[v] @ UP for v in vfs]
        tall = np.array([float(x.max() - x.min()) for x in h])
        foot = np.array([float(x.min() - GH) for x in h])
        C = np.array([fr[v].mean(0) for v in vfs])
        step = np.linalg.norm(np.diff(C, axis=0), axis=1) if len(C) > 1 else np.zeros(1)
        rows[pid] = dict(帧数=len(vfs), 视频帧=[int(vfs[0]), int(vfs[-1])],
                         身高=round(float(np.median(tall)), 3),
                         身高抖动=round(float(tall.std()), 3),
                         脚离地=round(float(np.median(foot)), 3),
                         脚离地范围=[round(float(foot.min()), 3), round(float(foot.max()), 3)],
                         帧间位移=round(float(np.median(step)), 3),
                         帧间位移最大=round(float(step.max()), 3))
    return rows


# ------------------------------------------------------------------ 远近
# SAM 3D Body 自己估人离镜头多远(看人在画面里多大, 按人的体型反推), 不看我们的深度图。
# pedestrian 实测 5 个人里 3 个被放近 11~22%, 脚飘 15~40cm; 远近换成场景深度之后 4/5 的人脚自己落到 ±10cm 内。
# 修人的顺序(body_motion.solve): 远近按场景深度 -> 翻转补帧 -> 锁骨头长度 -> 肢体跳变 -> 姿势平滑
#   -> 位置整理 / 贴地 -> 判断谁坐着、坐哪把椅子、记下走过的路线 -> 坐着的人换一种摆法
HALF_BODY = 0.12     # 深度图量的是身体正面, 人体网格的根在身体中间, 差半个身体厚
SEAT_EXTENT = 1.45   # 一帧的竖直跨度小于这个 = 坐着 (站着的成年人 >= 1.55m, 坐着 1.2~1.35m)
FOOT_BAND = 0.6      # 走过的路线只取离地这么高以内的点: 脚和小腿占的地方才是"路",
                     # 手臂伸到桌面上方不算占路, 否则人一靠近桌子就把桌子推走


def identity_tracks(run, da3, cams, vid2key, kept, fps):
    """核对身份要的东西: 跟踪的每个编号、每个出了网格的帧在房间里的位置 {编号: {帧: (秒, 世界坐标)}}。
    位置 = 模型给的根平移按场景深度校正远近(这一帧量到了用这一帧的比例, 没量到用这个编号的中位数), 再用相机位姿放进世界。"""
    roots = load_roots(run)
    todo, all_r = {}, []
    for pid, rt in roots.items():
        bis = sorted(b for b in rt if b < len(kept) and kept[b] in vid2key and vid2key[kept[b]] in cams
                     and os.path.exists(f"{run}/mesh_4d_individual/{pid}/{b:08d}.ply"))
        if bis:
            todo[pid] = (bis, depth_ratio(pid, bis, rt, run, da3, vid2key, kept)); all_r += list(todo[pid][1].values())
    r_all = float(np.median(all_r)) if all_r else 1.0
    out = {}
    for pid, (bis, ratios) in todo.items():
        r_own = float(np.median(list(ratios.values()))) if ratios else r_all
        out[pid] = {}
        for b in bis:
            c2w = np.linalg.inv(np.array(cams[vid2key[kept[b]]]["extrinsic_w2c_4x4"], float))
            out[pid][b] = (kept[b] / fps, ratios.get(b, r_own) * roots[pid][b] @ c2w[:3, :3].T + c2w[:3, 3])
    return out


def load_roots(mesh_run):
    """每人每帧 SAM 3D Body 给的根平移 (相机系 OpenCV, 米): focal_4d_individual/<pid>/<bi>.json 的 camera。"""
    out = {}
    root = f"{mesh_run}/focal_4d_individual"
    for pid in os.listdir(root):
        d = f"{root}/{pid}"
        out[pid] = {int(f[:-5]): np.asarray(json.load(open(f"{d}/{f}"))["camera"], float)
                    for f in os.listdir(d) if f.endswith(".json")}
    return out


def depth_ratio(pid, bis, roots, mesh_run, da3, vid2key, kept):
    """每帧: (mask 里场景深度的中位数 + 半个身体厚) / 模型给的距离。返回 {bi: 比例}。
    ★ 人被挡了大半的帧 mask 很小, 量不准, 直接跳过 —— 最后每人只取一个中位数。"""
    import cv2
    from PIL import Image
    out = {}
    for bi in bis:
        if bi >= len(kept) or kept[bi] not in vid2key or bi not in roots:
            continue
        stem = vid2key[kept[bi]].rsplit(".", 1)[0]
        mp = f"{mesh_run}/masks/{bi:08d}.png"
        dp = f"{da3}/depth/{stem}_depth.npy"
        if not (os.path.exists(mp) and os.path.exists(dp)):
            continue
        dep = np.load(dp)
        m = (np.array(Image.open(mp)) == int(pid)).astype(np.uint8)
        m = cv2.resize(m, (dep.shape[1], dep.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = cv2.erode(m, np.ones((3, 3), np.uint8)).astype(bool)
        if m.sum() < 15:
            continue
        out[bi] = (float(np.median(dep[m])) + HALF_BODY) / float(roots[bi][2])
    return out


DIST_HALF = 0.17    # 秒: 逐帧场景距离先取前后各这么久的中位数(24 fps 约 ±4 帧, 即原来的 9 帧窗),
                    # 压掉单帧 mask 糊到背景上的测量。★ 09-28 从"9 帧"改成按秒: office 约 7 fps, 9 帧是 1.3 秒
DIST_SIGMA = 0.3    # 然后再按时间做高斯平滑的 σ(秒, ±2σ 约 1 秒)。只平滑"离相机多远"这一个方向:
                    # 远处 VGGT 深度有 ±1 m 的噪声, 中位数会把它变成台阶(soccer r2 守门员 139->140 帧
                    # 画面里挪 3 px, 距离 21.1 -> 22.3 m); 人前后方向 1/20 秒挪不了 1 m, 放心平滑。
                    # 左右上下(画面里的位置)不动 —— 那是准的, 扑救这种快动作靠它。按秒不按帧: 室内约 8 fps, 室外约 21 fps


def frame_ratios(bis, ratios, z, R, half=DIST_HALF, t=None, sigma=DIST_SIGMA, trust=None):
    """每帧的远近比例 {bi: r}: 人离相机 r*z 米, 由这一帧前后 half 秒里**场景深度量到的距离**的中位数定。
    ★ 2026-09-24 以前每人只取一个比例 R, 距离 = R × 模型逐帧估的距离 z。模型的 z 是按"人在画面里多大"
      猜的, 人一倒地/蜷起来就塌: soccer r2 守门员扑救那 30 帧 z 从 19 m 掉到 8~11 m, 场景深度一直 19~22 m,
      固定的 R 把他往相机拉近 10 m, 俯视图上一条长线。平滑后最大一步 3.02 m -> 0.54 m, 近处的人几乎不变。
    ★ 平滑的是距离(物理上连续), 不是比例 —— 比例在模型塌陷的边界上本来就该跳。
    窗口里一帧都没量到(被挡住) -> 模型这帧可信, 就用这个人的整体比例 R(= R × 模型的距离);
      模型不可信(trust 之外: 翻转补帧的帧, 模型在两个人之间来回认) -> 不碰模型的距离, 按时间接上前后已经定了的
      距离(量到的, 或可信帧的 R × 模型距离)。
      ★ 09-28 nfl1 人 3 片尾 36 帧被挡得只剩 100~500 像素、量不到深度, 退回模型的距离一帧跳 ±1.8 m, 人前后来回晃;
        人 4(自己量的深度离群, ratios 传空)补帧段模型的距离一帧跳 28 m。
    t: {bi: 秒}(原视频帧号 / fps), 给高斯平滑用; 不给就按 24 fps 算。trust: 模型距离可信的帧, 不给 = 全可信。"""
    t = t or {b: b / 24.0 for b in bis}
    meas = {b: ratios[b] * z[b] for b in ratios}
    med = {}
    for b in bis:
        win = [meas[x] for x in bis if x in meas and abs(t[x] - t[b]) <= half + 1e-9]
        if win:
            med[b] = float(np.median(win))
    have = [b for b in bis if b in med]
    tt = np.array([t[b] for b in have]); dd = np.array([med[b] for b in have])
    ds = {b: float((k * dd).sum() / k.sum()) for b in have
          for k in [np.exp(-0.5 * ((tt - t[b]) / sigma) ** 2)]}
    out = {}
    for b in bis:
        if b in ds:
            out[b] = ds[b] / z[b]
        elif trust is None or b in trust:
            out[b] = R
    known = [b for b in bis if b in out]
    for b in bis:
        if b not in out:
            out[b] = (float(np.interp(t[b], [t[x] for x in known], [out[x] * z[x] for x in known])) / z[b]
                      if known else R)
    return out


# ------------------------------------------------------------------ 翻转补帧
# ★ 人被挡住大半时, SAM 3D Body 有时会把**挡在前面的那个人**当成他重建。两种后果完全不同:
#   - 稳定地套到邻居身上(soccer r2 人 1 在 13~45 帧套到人 2 身上): 邻居就在旁边、深度差不多、动作连贯,
#     看着是平滑的, 用户验收说没问题 —— 不动。
#     ★ 09-27 试过按"网格投影落在自己 mask 上的比例 < 0.5"把这种段整段换掉(两头插值/定住):
#       人 1 被补成僵住的动作, 比套到邻居身上难看得多, 用户否了。
#   - 每帧在两种理解之间来回认(守门员扑倒时一个近 10 m、看着大三倍的球员从前面穿过):
#     身体朝向一帧翻 30~118°, 这才是"鬼畜"。露全身时三个人每帧最多转 14°(中位 1~2°),
#     被挡但套得稳的人 1 最多 21°。
#   判据: 骨盆朝向转得比 FLIP_RATE 还快就在这儿断开; 断开后连续稳不到 MIN_RUN_S 的小段 = 翻来翻去, 换掉。
#   怎么换见 occ_plan: 两头都有好帧 -> 插值; 只剩一头(守门员翻到片尾) -> 动作定在最后一个好帧、位置跟 mask 走
#   (守门员扑倒以后本来就躺在地上, 定住反而是对的)。换掉的帧在轨迹整理里算"不可信"(见 tidy_track)。
#   ★ 09-28 从"一帧 30°、稳 10 帧"改成按秒(24 fps 下一模一样): office 约 7 fps, 按帧的话正常转身也会被当成翻
FLIP_RATE = 720.0   # 度/秒(24 fps 一帧 30°)。露全身的人一帧最多转 14°, 即约 340°/秒
MIN_RUN_S = 0.4     # 秒(24 fps 约 10 帧)。守门员 147 帧以后最长的一段只有 8 帧; 正常的人整段一次都不断


def _rot_deg(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))


def pelvis_rots(Ws):
    """每帧骨盆朝向(按 MHR 骨盆那圈顶点和静止姿势对齐)。"""
    P = load_parts()
    return [_kabsch(P["rest_pelvis"], W[P["pelvis"]]) for W in Ws]


def flip_segments(rots, t=None, rate=FLIP_RATE, min_run=MIN_RUN_S):
    """rots: 按帧顺序的骨盆朝向; t: 秒(不给按 24 fps)。返回要换掉的段 [(i0, i1)](闭区间, 下标)。
    一段够长的都没有(从头翻到尾) -> 不知道该信谁, 返回 []。"""
    return short_runs([0.0] + [_rot_deg(rots[i - 1], rots[i]) for i in range(1, len(rots))], t, rate, min_run)


def short_runs(step, t=None, rate=FLIP_RATE, min_run=MIN_RUN_S):
    """step[i] = 第 i-1 帧到第 i 帧变了多少度(step[0] 不用); t: 秒(不给按 24 fps)。
    变得比 rate(度/秒)还快的地方断开, 断开后连着稳不到 min_run 秒的小段 = 跳来跳去, 返回这些段 [(i0, i1)](闭区间, 下标)。
    一段够长的都没有 -> 不知道该信谁, 返回 []。骨盆朝向(flip_segments)和各段肢体的关节角(body_motion)共用这一条判据。"""
    n = len(step)
    if n < 2:
        return []
    t = np.arange(n) / 24.0 if t is None else np.asarray(t, float)
    dt = np.diff(t)
    cuts = [0] + [i for i in range(1, n) if step[i] > rate * dt[i - 1]] + [n]
    runs = [(a, b - 1) for a, b in zip(cuts[:-1], cuts[1:])]
    long_ = lambda a, b: (b - a + 1) * float(np.median(dt)) >= min_run - 1e-9      # 帧数 × 帧间隔
    if not any(long_(a, b) for a, b in runs):
        return []
    segs = []
    for a, b in runs:
        if long_(a, b):
            continue
        if segs and a == segs[-1][1] + 1:
            segs[-1][1] = b
        else:
            segs.append([a, b])
    return [tuple(x) for x in segs]


def load_focals(mesh_run):
    """每人每帧 SAM 3D Body 自己估的焦距(像素), 和 load_roots 同一批 json。"""
    root = f"{mesh_run}/focal_4d_individual"
    return {pid: {int(f[:-5]): float(json.load(open(f"{root}/{pid}/{f}"))["focal_length"])
                  for f in os.listdir(f"{root}/{pid}") if f.endswith(".json")}
            for pid in os.listdir(root)}


def occ_plan(bis, segs, t, rootW, mask_root):
    """每个不可信帧怎么补: {str(bi): [A, B, w, 根的世界坐标]}。照着它改参数的是 body_motion.fill_flips。
    动作 = A、B 两个好帧之间按 w 过渡; 根的位置:
      两头都有好帧 -> 按时间线性插值(1~2 秒里人基本走直线, 比被挡时的任何观测都可靠);
      只有一头(片头/片尾) -> 动作定在那个好帧(A = B), 位置跟着自己的 mask 走:
        mask_root[bi] = 沿 mask 质心那条视线、按场景深度量的距离; 减去锚点那帧"mask 质心 vs 真根"的差,
        接缝处不跳。mask 空了(整个人被挡没了)就停在上一帧。
    一整段都不可信(没有锚) -> 不补, 返回里没有这段。"""
    out = {}
    for i0, i1 in segs:
        A = bis[i0 - 1] if i0 > 0 else None
        B = bis[i1 + 1] if i1 + 1 < len(bis) else None
        if A is None and B is None:
            continue
        seg = bis[i0:i1 + 1]
        if A is not None and B is not None:
            for b in seg:
                w = (t[b] - t[A]) / (t[B] - t[A])
                out[str(b)] = [A, B, round(w, 5), ((1 - w) * rootW[A] + w * rootW[B]).round(5).tolist()]
            continue
        K = A if A is not None else B
        mk = (mask_root or {}).get(K)
        off = rootW[K] - mk if mk is not None else np.zeros(3)
        prev = rootW[K]
        for b in (seg if A is not None else seg[::-1]):             # 从锚点往外走, "停在上一帧"才有上一帧
            m = (mask_root or {}).get(b)
            if m is not None:
                prev = m + off
            out[str(b)] = [K, K, 0.0, np.asarray(prev).round(5).tolist()]
    return out


def mask_roots(pid, bis, rt, rf, focal, mesh_run, cams, vid2key, kept):
    """逐帧"沿自己 mask 质心那条视线、按场景深度"的根 {bi: 世界坐标, mask 空了是 None}。
    质心按 body4d 的画面算(和网格同一个像素系, 主点在画面中心); 距离用 frame_ratios 那套
    (rf × 模型的 z = 自己 mask 里的场景深度)。"""
    from PIL import Image
    out = {}
    for bi in bis:
        mp = f"{mesh_run}/masks/{bi:08d}.png"
        own = (np.array(Image.open(mp)) == int(pid)) if os.path.exists(mp) else None
        if own is None or not own.any():
            out[bi] = None
            continue
        H, W_ = own.shape
        ys, xs = np.nonzero(own)
        z = rf[bi] * float(rt[bi][2])
        pc = np.array([(xs.mean() - W_ / 2) / focal[bi] * z, (ys.mean() - H / 2) / focal[bi] * z, z])
        c2w = np.linalg.inv(np.array(cams[vid2key[kept[bi]]]["extrinsic_w2c_4x4"], float))
        out[bi] = pc @ c2w[:3, :3].T + c2w[:3, 3]
    return out


def _fix_report(occ, flips, kept):
    """报告: 补了哪几段(原视频帧号)、怎么补的、段里最大的一翻(度)。"""
    bs = sorted(int(k) for k in occ)
    segs = []
    for b in bs:
        if segs and b == segs[-1][-1] + 1:
            segs[-1].append(b)
        else:
            segs.append([b])
    return [dict(帧=[int(kept[x[0]]), int(kept[x[-1]])], 帧数=len(x),
                 补法="两头插值" if occ[str(x[0])][0] != occ[str(x[0])][1] else "动作定住+跟mask",
                 最大一翻=round(max(flips.get(b, 0.0) for b in x))) for x in segs]


def _smooth(P, w_med=5, w_avg=3):
    """逐帧位置 (N,3): 先滑动中位数压掉单帧跳变, 再滑动平均去抖。两端用可用的邻居。"""
    n = len(P)
    if n < 3:
        return P.copy()
    med = np.array([np.median(P[max(0, i - w_med // 2):i + w_med // 2 + 1], 0) for i in range(n)])
    return np.array([med[max(0, i - w_avg // 2):i + w_avg // 2 + 1].mean(0) for i in range(n)])


# ------------------------------------------------------------------ 姿势平滑
# ★ SAM 3D Body 是单帧模型, 每帧从零估: 四肢和身体朝向一帧一晃。骨盆系里的高频抖动(09-28 实测)
#   soccer 1.4~2.1 cm/顶点, nfl1 抱摔那堆人 4~6.5 cm, 手脚 11~13 cm。SAM-Body4D 自带的平滑只在 8 帧一批里
#   往前看, 批和批不连。
#   做法(现在做在关节参数上, 见 body_motion.smooth_joints / smooth_rots; 这里只留两边共用的强度和按时间加权的平均):
#   各算两版 —— 强平滑(σ=POSE_SIGMA) 和 几乎不平滑(σ=POSE_SIGMA_MIN), 按**各自**动得多快混合: 慢的用强平滑版,
#   快的用原样(One Euro Filter 的思路, Casiez CHI 2012)。甩出去的那条腿保持原样, 身体别处照样去抖;
#   只按整个人的平均速度调, 一条腿甩得再快平均下来也不快, 会被抹软。
POSE_SIGMA = 0.08       # 秒: 慢的部分的平滑强度(24 fps 约 2 帧)
POSE_SIGMA_MIN = 0.02   # 秒: 快的部分基本不平滑(约半帧)
POSE_V0 = 1.5           # m/s: 骨盆系里一个顶点动到这么快, 两版各一半
POSE_W0 = 3.0           # rad/s: 身体朝向转到这么快(约 170°/s), 两版各一半


def _gauss_t(Y, t, sig):
    """Y: (T, ...) 按时间 t(秒)做高斯加权平均, 每帧用自己的 σ(秒)。"""
    out = np.empty_like(Y)
    for i in range(len(t)):
        m = np.abs(t - t[i]) <= 3 * sig[i]
        w = np.exp(-0.5 * ((t[m] - t[i]) / sig[i]) ** 2)
        out[i] = np.tensordot(w / w.sum(), Y[m], axes=1)
    return out


# ------------------------------------------------------------------ 轨迹整理(位置 + 贴地 + 腾空)
# ★ 09-28 以前这里是几层各干各的: 翻转补帧 / 腾空 / 位置平滑(只管水平) / 去尖刺 / 兜底 —— 互相不知道对方的
#   假设, 改一层踩一层: 位置平滑只管水平, 上下交给 5 帧中位贴地, nfl1 人 3 躺着不动却一帧上下跳 22 cm;
#   腾空不知道哪些帧是补过的, 把补过的帧抬到空中 20 cm; 去尖刺把整个窗口剔光, 人 4 被放到相机上。
#   现在统一成 tidy_track: 每帧先有一个可信度 conf(1 = 模型这帧可信; CONF_SUB = 翻转补帧换掉的帧, 动作是
#   定住/插值出来的、位置是插值或跟 mask 的), 然后
#     1) 骨盆中心**三个方向一起**做局部二次拟合(smooth_track), conf 当权重 —— 补过的帧少算;
#     2) 脚底离地高度同样平滑后再贴地 —— 贴地量本身是平滑的, 不再把噪声一帧帧带回来;
#     3) 高出平时脚底的部分当腾空留在空中(airborne), 只有可信的帧能判腾空, 平时脚底也只按可信的帧算。
#   以前的位置平滑是 _smooth(5 帧中位 + 3 帧平均, 前后约 0.1 秒): 只压单帧尖刺, 来回晃几帧的锯齿原样留着
#   (soccer r2 骨盆中心偏离平滑路径 2.5~5 cm, 相机本身很稳)。
#   局部二次拟合: 普通平均会把扑救的起点终点抹圆; 二次拟合对匀速、匀加速是精确的, 只去掉来回晃。
POS_SIGMA = 0.08   # 秒: 拟合窗口的高斯 σ(取前后 3σ)。测试里的取舍: 0.08 -> 锯齿剩 30%, 0.3 秒横移 1.5 m 的扑救
                   #   最多偏 4 cm; 0.15 -> 锯齿剩 22%, 扑救偏 16 cm。再宽锯齿只多去几个点, 快动作被抹得多
POS_MIN_FRAMES = 1.5   # σ 至少是这么多个帧间隔: office 约 6 fps(一帧 0.17 秒), 0.08 秒的窗口里只有 3 帧,
                       # 凑不够拟合就整段跳过不平滑 —— 09-28 office 走路的人上下一帧跳 49 cm。24 fps 时不起作用
POS_SPIKE = 0.3    # 米: 第一遍拟合后偏出去超过这么多的帧当尖刺, 第二遍不算它(正常锯齿、扑救都在 10 cm 以内)
CONF_SUB = 0.2     # 翻转补帧换掉的帧的可信度: 平滑里只算 1/5、窗口 5 倍长(见 tidy_track), 不能判腾空, 不参与"平时脚底"


def smooth_track(P, t, sigma=POS_SIGMA, spike=POS_SPIKE, wt=None):
    """P: (T,D) 逐帧位置; t: 秒; sigma: 秒, 一个数或逐帧给; wt: 每帧可信度(不给 = 全 1)。
    返回局部二次拟合后的位置。不到 5 帧的不动。
    ★ 拟合两遍, 第二遍把尖刺的权重压到 0(双平方权重): 二次拟合本身不防尖刺 —— nfl1 人 3 翻转补帧那段跟着 mask 走,
      人堆里 mask 忽大忽小, 有几帧单独跳出去 1~2 m 又回来, 直接拟合会被拖偏 1 m 多。
      先过 5 帧中位数也试过: 来回交替的锯齿中位数去不掉, 反而更差。
    ★ 两个兜底(09-28 nfl1 人 4: 一段连着十几帧跳出去 28 m, 第二遍把整个窗口都剔光了, 拟合退化成 0,
      人被放到世界原点 = 第一台相机上):
      - 窗口里还算数的帧不到 4 个, 不重拟, 用第一遍的结果;
      - 拟合相对本帧的第一遍结果做, 就算退化也只会退回它, 不会被拉向原点。"""
    P = np.asarray(P, float); t = np.asarray(t, float)
    if len(t) < 5:
        return P.copy()
    sig = np.maximum(np.broadcast_to(np.asarray(sigma, float), t.shape),
                     POS_MIN_FRAMES * float(np.median(np.diff(t))))
    wt = np.ones(len(t)) if wt is None else np.asarray(wt, float)
    out, rw = P.copy(), np.ones(len(t))
    for it in range(2):
        ref, out = out, out.copy()
        for i in range(len(t)):
            s = sig[i]
            m = np.abs(t - t[i]) <= 3 * s
            k = rw[m] * wt[m]
            if m.sum() < 4 or (k > 0).sum() < 4:
                continue
            d = t[m] - t[i]
            w = np.sqrt(np.exp(-0.5 * (d / s) ** 2) * k)[:, None]
            # 片头片尾窗口只有一边有数据, 二次曲线在边上会翘(09-28 office 人 2 头两帧脚离地 38 cm): 那里用直线。
            # 按"短的那一边伸出去几个 σ"渐变: 最边上一帧全用直线, 1σ 以上全用二次。不能在某一帧突然换 ——
            # 窗口一长两种拟合在弯路上差几厘米, 换的那一帧人跳一下(09-28 nfl1 人 3 片尾前 10 帧跳 4 cm)。
            # 实测(6 fps / 24 fps 加噪声各跑 100~200 次)片头片尾的偏差和原来的硬切一样, 弯路上的跳 3.6 -> 0.4 cm
            a = float(np.clip(min(-d.min(), d.max()) / s, 0, 1))
            A, y = np.stack([d ** p for p in range(3)], 1) * w, (P[m] - ref[i]) * w
            lin = np.linalg.lstsq(A[:, :2], y, rcond=None)[0][0]
            quad = np.linalg.lstsq(A, y, rcond=None)[0][0] if a > 0 else lin
            out[i] = ref[i] + a * quad + (1 - a) * lin
            # 二次拟合碰到台阶会冲过头(28 m 的跳变冲出去 0.8 m): 夹在窗口里这些位置的范围内
            out[i] = np.clip(out[i], P[m].min(0), P[m].max(0))
        rw = np.clip(1 - (np.linalg.norm(P - out, axis=1) / spike) ** 2, 0, None) ** 2
    return out


AIR_WIN = 2.0    # 秒: "平时脚底有多高"看前后各这么久(飞扑一整段约 2 秒, 窗要比它宽才看得到落地的帧)
AIR_PCT = 20     # 取这个百分位当平时的脚底
AIR_TOL = 0.15   # 米: 高出平时脚底不到这么多 = 深度噪声, 照旧贴地; 超出的部分当真离地
AIR_MIN_S = 0.4  # 秒: 离地要连着这么久才算腾空。一两帧的"离地"是噪声(nfl1 人 3 可信帧里零星两帧 15 cm);
                 # nfl1 人 5 飞扑 1.7 秒, 人跳起来的滞空也有 0.4 秒以上
AIR_MIN_FRAMES = 5   # 而且至少这么多帧: office 约 6 fps, 刚走进画面的 3 帧身体估歪了(脚离地 37 cm)也有 0.5 秒。
                     # 帧率低, 3 帧不够当"跳起来"的证据; 24 fps 时 0.4 秒已经是 10 帧, 这条不起作用


def airborne(f, t, win=AIR_WIN, pct=AIR_PCT, tol=AIR_TOL, ok=None):
    """每帧该留在空中多高(米)。f: 逐帧最低点离地(已平滑), t: 秒; ok: 哪些帧可信(不给 = 全可信)。
    ★ 09-28 以前每帧都把最低点压到地上: nfl1 人 5 飞扑时脚离地 0.4~0.8 m, 被一帧帧按回地面。
      走路的人高出平时脚底都在噪声以内, 结果和以前一样。
    ★ 不可信的帧不能判腾空, "平时脚底"也只按可信的帧算: nfl1 人 3 翻转补帧那段(动作定住、位置跟 mask),
      乱数据显示脚离地 20 cm, 以前就被当成腾空抬了起来。开关按时间软化(σ 0.1 秒), 进出那段时不会一帧掉下来。"""
    f, t = np.asarray(f, float), np.asarray(t, float)
    ok = np.ones(len(t), bool) if ok is None else np.asarray(ok, bool)
    base = []
    for ti in t:
        m = np.abs(t - ti) <= win
        base.append(np.percentile(f[m & ok] if (m & ok).any() else f[m], pct))
    air = np.maximum(0.0, f - np.array(base) - tol)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    i = 0
    while i < len(air):                                            # 连着离地不够 AIR_MIN_S 的一段 = 噪声
        j = i
        while j < len(air) and air[j] > 0:
            j += 1
        if j > i and ((j - i) * dt < AIR_MIN_S - 1e-9 or j - i < AIR_MIN_FRAMES):
            air[i:j] = 0.0
        i = max(j, i + 1)
    if not ok.all():
        air *= _gauss_t(ok.astype(float), t, np.full(len(t), 0.1))
    return air


def tidy_track(W, bis, t, conf, UP, GH):
    """一个人整段的位置整理(姿势已经整理完)。W: {bi: 世界系顶点}; t: 秒; conf: 每帧可信度(和 bis 同顺序)。
    返回每帧的整体平移 delta (T,3): 骨盆中心三个方向一起平滑 + 脚底贴地 + 可信的腾空留在空中。见本节开头。"""
    t, conf = np.asarray(t, float), np.asarray(conf, float)
    pel = load_parts()["pelvis"]
    P = np.array([W[b][pel].mean(0) for b in bis])
    # ★ 不可信的帧参考更长的时间: 可信度 1/5 -> 窗口 5 倍长(σ 0.4 秒, 前后约 1.2 秒)。只降权重不够 ——
    #   一整段都是补过的帧时大家一样轻, 互相比没差别(09-28 nfl1 人 3 片尾跟着 mask 露出的一小块走,
    #   ±0.24 秒的窗口里平滑完一帧还前后晃 15~22 cm)。可信的帧窗口不变, 快动作照旧保住。
    #   只用在骨盆位置上: 贴地量要紧跟(已经平滑过的)脚, 再用长窗口平滑一遍就跟不上, 补过的帧脚离地 11 cm。
    #   窗口在交界处按时间渐变(可信度先做 0.1 秒高斯): 一帧里从 0.08 秒跳到 0.4 秒, 两边估法对不齐,
    #   人在交界那帧跳一下(09-28 nfl1 人 3 第 58 帧往下 4.9 cm)
    sig = POS_SIGMA / np.maximum(_gauss_t(conf, t, np.full(len(t), 0.1)), CONF_SUB)
    delta = smooth_track(P, t, sigma=sig, wt=conf) - P
    foot = np.array([float(((W[b] + d) @ UP).min() - GH) for b, d in zip(bis, delta)])
    fs = smooth_track(foot[:, None], t, wt=conf)[:, 0]
    air = airborne(fs, t, ok=conf >= 1.0)
    return delta - (fs - air)[:, None] * UP[None, :]


# ------------------------------------------------------------------ 坐着的人
# ★ 为什么单独摆: 坐着的人腿被桌子挡住, SAM 3D Body 每帧重猜一次腿。贴地(tidy_track)会把腿的变化放大成
#   整个人上下跳 —— office 1 号: 头顶高度跳了 0.46m, 屁股陷进椅面 15cm; 可她的头顶相对骨盆一直很稳(全程只差 6cm)。
#   所以坐着的人换一种摆法:
#     下半身(骨盆+腿)整段用同一帧的姿势, 只像转椅子那样绕竖轴转; 屁股放在椅面上(没椅子就脚着地);
#     上身保留每帧的动作, 跟着骨盆走, 再做一次时间平滑。
#   看不见的腿"这一帧猜成什么样"一概不信, 只信整段里最像样的那一次。
#   这里的 seat_fix 只做**决定**(用哪一帧的腿、每帧转多少、放多高); 照着改关节参数的是 body_motion.seat_fix_params。
PARTS_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mhr_parts.npz")
SEAT_BAND = (0.30, 0.80)   # 椅面: 离地这个范围里面积最大的那层朝上的面
SEAT_XY_TOL = 0.35         # 骨盆离椅面中心超过这个 = 不在这把椅子上方, 改成脚着地
SEAT_SINK = 0.01           # 屁股压进坐垫 1cm, 免得看着悬空
DROP_TOL = 0.05            # 选腿: 屁股到脚的落差离椅面高度这么近的帧优先 —— 坐上去脚正好着地
FREEZE_STEP = 20.0         # 相邻帧骨盆转角的中位数超过这么多度 = 朝向估计本身不可信, 整段定住
UPPER_SIGMA = 1.0          # 上身时间平滑的高斯 σ, 单位是 body4d 的帧
_PARTS = None


def load_parts():
    """MHR 身体分区(make_mhr_parts.py 一次性产出): 骨盆顶点、贴椅面的顶点、下半身权重。"""
    global _PARTS
    if _PARTS is None:
        if not os.path.exists(PARTS_NPZ):
            raise SystemExit(f"[失败] 缺 {PARTS_NPZ}: 先在 body4d 环境跑一次 pipeline/make_mhr_parts.py")
        z = np.load(PARTS_NPZ)
        rp = z["rest_pelvis"].astype(np.float64)
        _PARTS = dict(lower_w=z["lower_w"].astype(np.float64)[:, None], pelvis=z["pelvis_idx"],
                      rest_pelvis=rp - rp.mean(0), seat=z["seat_idx"], low=z["low_idx"],
                      hand_w=(z["hand_w"].astype(np.float64) if "hand_w" in z.files else None))
    return _PARTS


def chair_seats(scene_glb, GH):
    """椅子的椅面: {节点名: (椅面离地高, 椅面中心 xz, 整把椅子包围盒中心 xyz)}, glTF 系(天 +y)。
    ★ 椅面 = 离地 0.30~0.80m 之间、面积最大的那一层朝上的面(1cm 一格统计)。靠背是竖的,
      扶手面积小, 都不会被当成椅面。office 三把椅子是同一个模型, 椅面都在 0.43m。"""
    import re as _re
    sc = trimesh.load(scene_glb)
    out = {}
    for node in sc.graph.nodes_geometry:
        name = node.split("/")[-1]
        if "chair" not in _re.sub(r"_\d+$", "", name):
            continue
        T, g = sc.graph[node]
        m = sc.geometry[g].copy(); m.apply_transform(T)
        n, area, cen = m.face_normals, m.area_faces, m.triangles_center
        h = cen[:, 1] - GH
        band = (n[:, 1] > 0.8) & (h > SEAT_BAND[0]) & (h < SEAT_BAND[1])
        if not band.any():
            continue
        edges = np.arange(SEAT_BAND[0], SEAT_BAND[1] + 1e-6, 0.01)
        hist, _ = np.histogram(h[band], bins=edges, weights=area[band])
        k = int(hist.argmax())
        sel = band & (h >= edges[k] - 0.02) & (h < edges[k + 1] + 0.02)
        out[name] = (float(np.average(h[sel], weights=area[sel])),
                     np.average(cen[sel][:, [0, 2]], axis=0, weights=area[sel]), m.bounds.mean(0))
    return out


def _kabsch(A, B):
    """A 已去中心。返回 R: B - B.mean ≈ A @ R.T (列向量写法 b = R a)。"""
    U, _, Vt = np.linalg.svd(A.T @ (B - B.mean(0)))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _rot_up(UP, a):
    """绕 UP 转 a 弧度(列向量写法)。"""
    K = np.array([[0, -UP[2], UP[1]], [UP[2], 0, -UP[0]], [-UP[1], UP[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _pelvis_track(X, P, UP):
    """每帧骨盆: 中心 (T,3)、朝向 (T,3,3)、水平朝向角 (T,) 弧度(已展开, 不会在 ±π 处跳)。"""
    C = X[:, P["pelvis"]].mean(1)
    R = np.stack([_kabsch(P["rest_pelvis"], x[P["pelvis"]]) for x in X])
    e1 = np.cross(UP, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(UP, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1)
    f = R[:, :, 2]                                   # 静止姿势里人面朝 +z
    return C, R, np.unwrap(np.arctan2(f @ e2, f @ e1))


def seat_fix(X, bis, UP, GH, FLIP, seat=None, seat_h_default=0.45):
    """坐着的人: 算整段的摆法, 返回 (可 JSON 化的 fix, 诊断)。
    X: (T,NV,3) 姿势整理完(骨头锁好、平滑过)、还没做位置整理和贴地的世界系顶点(body_motion.solve 里的 W)。
    seat: (椅子名, 椅面离地高, 椅面中心 xz[glTF]) 或 None。"""
    P = load_parts()
    T = len(X)
    C, R, yaw = _pelvis_track(X, P, UP)
    steps = [float(np.degrees(np.arccos(np.clip((np.trace(R[i].T @ R[i + 1]) - 1) / 2, -1, 1))))
             for i in range(T - 1)]
    freeze = bool(steps and np.median(steps) > FREEZE_STEP)
    # 1) 坐哪: 骨盆在椅面正上方(水平 SEAT_XY_TOL 以内)才信这把椅子, 否则按脚着地
    on, gap = None, None
    if seat is not None:
        pxz = np.median((C @ FLIP[:3, :3].T + FLIP[:3, 3])[:, [0, 2]], 0)
        gap = float(np.linalg.norm(pxz - seat[2]))
        if gap <= SEAT_XY_TOL:
            on = (seat[0], float(seat[1]))
    target = on[1] if on else seat_h_default
    # 2) 选腿: "屁股到脚的落差 ≈ 椅面高"的帧里, 取和其他帧最像的那一帧(中位代表)
    rel = X - C[:, None, :]
    h = rel @ UP
    drop = h[:, P["seat"]].min(1) - h[:, P["low"]].min(1)
    ok = np.where(np.abs(drop - target) <= DROP_TOL)[0]
    if len(ok) == 0:
        ok = np.array([int(np.argmin(np.abs(drop - target)))])
    idx = np.arange(X.shape[1])[::8] if freeze else P["low"][::4]
    Z = np.stack([rel[t][idx] @ _rot_up(UP, -yaw[t]).T for t in range(T)])   # 先转到同一个朝向, 只比"腿怎么放"
    ref = int(ok[int(np.argmin([np.linalg.norm(Z - Z[s], axis=2).mean(1).sum() for s in ok]))])
    # 3) 水平转向整段平滑(转椅子这种慢的保留, 单帧乱翻去掉); 定住的人不转
    yaw_s = np.full(T, yaw[ref]) if freeze else _smooth(yaw[:, None], w_med=9, w_avg=5)[:, 0]
    # 4) 位置: 水平重平滑(定住的人取整段中位); 竖直按椅面/地面一次定死 —— 下半身只绕竖轴转, 高度不变
    Ch = C - np.outer(C @ UP, UP)
    Ch = np.repeat(np.median(Ch, 0)[None], T, 0) if freeze else _smooth(Ch, w_med=9, w_avg=5)
    y = (GH + on[1] - SEAT_SINK - float(h[ref][P["seat"]].min())) if on else (GH - float(h[ref][P["low"]].min()))
    Pbar = Ch + y * UP
    fix = dict(ref=int(bis[ref]), freeze=freeze, sigma=UPPER_SIGMA, up=np.round(UP, 8).tolist(),
               seat=(dict(chair=on[0], 椅面离地=round(on[1], 4)) if on else None),
               yaw={str(int(b)): round(float(yaw_s[t] - yaw[ref]), 5) for t, b in enumerate(bis)},
               pos={str(int(b)): np.round(Pbar[t], 5).tolist() for t, b in enumerate(bis)})
    diag = dict(骨盆转角中位_度=round(float(np.median(steps)), 1) if steps else 0.0, 定住=freeze,
                参考帧=int(bis[ref]), 候选帧=int(len(ok)), 落差=round(float(drop[ref]), 3),
                目标落差=round(float(target), 3), 离椅面中心=(round(gap, 3) if gap is not None else None),
                高度来源=(on[0] if on else "地面(脚着地)"))
    return fix, diag


def _seat_report(old, new, UP, GH, fix):
    """坐着的人改前改后的体检。"""
    P = load_parts()
    def st(V):
        H = V @ UP - GH
        return (V[:, P["pelvis"]].mean(1) @ UP - GH, H.max(1), H.min(1), H[:, P["seat"]].min(1))
    a, b = st(old), st(new)
    r3 = lambda x: round(float(x), 3)
    rng = lambda x: [r3(x.min()), r3(x.max())]
    jump = lambda x: r3(np.abs(np.diff(x)).max()) if len(x) > 1 else 0.0
    out = dict(骨盆离地_前=rng(a[0]), 骨盆离地_后=rng(b[0]), 头顶离地_前=rng(a[1]), 头顶离地_后=rng(b[1]),
               头顶最大一跳_前=jump(a[1]), 头顶最大一跳_后=jump(b[1]), 脚离地_前=rng(a[2]), 脚离地_后=rng(b[2]),
               竖直跨度抖动_前=r3((a[1] - a[2]).std()), 竖直跨度抖动_后=r3((b[1] - b[2]).std()))
    if fix["seat"]:
        s = fix["seat"]["椅面离地"]
        out.update(屁股到椅面_前=r3(np.median(a[3]) - s), 屁股到椅面_后=r3(np.median(b[3]) - s))
    return out


def sit_and_route(Ws, delta, vfs, UP, GH, FLIP, chairs):
    """一个人整段: 哪几帧算坐着、坐的是哪把椅子、站/走着的帧踩过哪些地方。物理那步拿后两样钉椅子、当路障。
    Ws: 按帧排的世界系顶点(姿势整理完的); delta (T,3): 位置整理给的整体平移; vfs: 每帧的原视频帧号;
    chairs: load_chairs() 的结果(没有椅子给 {})。返回 (每帧坐没坐 (T,) bool, 椅子名或 None, 脚印列表)。
      坐着 = 这一帧的竖直跨度小于 SEAT_EXTENT; 椅子 = 坐着的帧里, 臀部那一层(离地 0.3~0.8 m)落在哪把椅子的
      水平矩形 0.4 m 以内, 票多的那把(而且过半的帧坐着);
      脚印 = 站/走着的帧, 离地 FOOT_BAND 以内的点的水平最小外接矩形(glTF 系 xz)。"""
    R3 = FLIP[:3, :3]
    seated = np.array([float(np.ptp(W @ UP)) for W in Ws]) < SEAT_EXTENT
    chair, votes, prints = None, {}, []
    for W, d, s_, vf in zip(Ws, delta, seated, vfs):
        V = (W + d) @ R3.T
        h = V[:, 1] - GH
        if s_:
            hip = V[(h > 0.3) & (h < 0.8)][:, [0, 2]]
            if len(hip) < 20:
                continue
            hc = hip.mean(0)
            best = None
            for name, (c, ax, hh) in chairs.items():
                q = hc - c
                gap = float(np.linalg.norm(np.maximum(np.abs([q @ ax[0], q @ ax[1]]) - hh, 0)))
                if gap < 0.4 and (best is None or gap < best[0]):
                    best = (gap, name)
            if best:
                votes[best[1]] = votes.get(best[1], 0) + 1
            continue
        low = V[h < FOOT_BAND][:, [0, 2]]
        if len(low) < 20:
            continue
        T2, e2 = trimesh.bounds.oriented_bounds_2D(low)
        Rr = T2[:2, :2]
        prints.append(dict(vf=int(vf), c=(-Rr.T @ T2[:2, 2]).round(4).tolist(),
                           ax=Rr.round(5).tolist(), h=(np.asarray(e2) / 2).round(4).tolist()))
    if votes and seated.mean() > 0.5:
        chair = max(votes, key=votes.get)
    return seated, chair, prints


def ruler_people(per, tol=RATIO_TOL):
    """算尺度时谁能当尺子。per: {编号: 这个人的比例中位}。离全体中位超过 tol 的踢掉, 返回 (留下的 {编号: 比例}, 踢掉的 {编号: 原因})。
    ★ 为什么(office r12, 2026-10-09): 外套被当成人 7, 比例 0.466 混在 8 个人里, 中位 0.978 -> 0.945, 整个场景多放大 3.5%。
      它要到第 5 段摆人之后才会被人在闸口上点掉, 可尺度第 3 段就乘进深度了, 不重算 —— 所以这一步自己就得把它踢出尺子。
    踢完一个不剩(两个人各执一词, 中位正好在中间)就一个都不踢: 分不出谁对, 留给闸口上的人看。"""
    if not per:
        return {}, {}
    med = float(np.median(list(per.values())))
    out = {p: f"离全体中位 {med:.3f} 差 {abs(r / med - 1):.0%}, 超过 {tol:.0%}" for p, r in per.items() if abs(r / med - 1.0) > tol}
    if len(out) == len(per):
        return dict(per), {}
    return {p: r for p, r in per.items() if p not in out}, out


def suspicious(R_own, R_all, R, conf_ok, jump):
    """可疑的人: 数据从头到尾就不对, 整理只能遮丑(nfl1 人 4: 画面里 54 px, 深度离群, 位置跳 28 m)。返回原因列表(空 = 不可疑)。
    只点名, 不自动删: office 那件搭在椅背上的外套(深度比例 0.49)和 nfl1 远处 20x60 像素的真人(0.73)都是"深度离群",
    分不开 —— 删不删是闸口上人看着截图拍板的事。
    R_own / R_all / R: 自己量的远近比例、全体中位、最后用的; conf_ok: 模型可信的帧占比; jump: 模型原始位置一帧最多跳多少米。"""
    return ([f"自己量的深度离群(比例 {R_own:.2f}, 全体 {R_all:.2f})"] if R != R_own else []) \
        + ([f"只有 {conf_ok:.0%} 的帧模型可信"] if conf_ok < 0.5 else []) \
        + ([f"模型原始位置一帧跳 {jump:.1f} m"] if jump > 2.0 else [])


def load_chairs(scene_glb):
    """场景里的椅子: {节点名: (水平最小外接矩形中心 xz, 两个轴, 半长)}, glTF 系。"""
    import re as _re
    sc = trimesh.load(scene_glb)
    out = {}
    for node in sc.graph.nodes_geometry:
        name = node.split("/")[-1]
        if "chair" not in _re.sub(r"_\d+$", "", name):
            continue
        T, g = sc.graph[node]
        m = sc.geometry[g].copy(); m.apply_transform(T)
        T2, e2 = trimesh.bounds.oriented_bounds_2D(m.vertices[:, [0, 2]])
        Rr = T2[:2, :2]
        out[name] = (-Rr.T @ T2[:2, 2], Rr.copy(), np.asarray(e2) / 2)
    return out


# ------------------------------------------------------------------ 人体空气墙
# ★ 为什么要它 (2026-09-22): 物理那步原来只拿"脚印"当路障 —— 每帧离地 0.6m 以内的点画一个
#   地面矩形, 家具压到就推开。人的身体本身从没进过穿模检测, 所以 office 2 号、7 号整个人站在
#   站立办公桌里(桌面 1.1m, 正好在腰上), 物理一眼都没看。现在把每个人整段动作扫过的身体
#   (MIME 的"自由空间"思路)切成 BODY_PITCH 的格子交给物理: 家具的实心格子碰到它就推开, 人不动。
#   手不算: 手搭在桌面、键盘上是正常接触, 不是穿模。坐着的人照样算 —— 他那把椅子物理钉住不推。
BODY_PITCH = 0.05


def people_body(world, FLIP):
    """{pid: 整段扫过的身体"内芯"格子 (N,3) int32}, glTF 系, 格子坐标 = floor(坐标 / BODY_PITCH)。
    ★ 内芯 = 每帧把人体填成实心, 再往里收一格(BODY_PITCH)。家具碰到内芯 = 压进人身体超过约 5cm 才算穿模;
      只碰到表面的(坐在自己椅子上、靠着桌沿、擦身走过)不算。人体模型和真实的人本来就差几厘米
      (office 实测真实点云只有 66~88% 落在模型 ±5cm 内), 贴着表面算穿模会把正常接触当成要推开的东西。
      手不算(手搭在桌面、键盘上是正常接触): 填实心时带着手(不然手腕是开口, 小臂填不实), 最后把手占的格子去掉。"""
    from scipy import ndimage
    P = load_parts()
    if P["hand_w"] is None:
        raise SystemExit(f"[失败] {PARTS_NPZ} 是老版本(没有手), 在 body4d 环境重跑一次 make_mhr_parts.py")
    hand = P["hand_w"] > 0.5
    out = {}
    for pid, fr in world.items():
        deep = []
        for v in fr:
            c = np.floor((fr[v] @ FLIP[:3, :3].T + FLIP[:3, 3]) / BODY_PITCH).astype(np.int64)
            lo = c.min(0) - 2; g = np.zeros(c.max(0) - lo + 3, bool); g[tuple((c - lo).T)] = True
            g = ndimage.binary_fill_holes(ndimage.binary_closing(g, iterations=1))
            core = np.argwhere(ndimage.binary_erosion(g, iterations=1)) + lo
            hk = {tuple(x) for x in np.unique(c[hand], axis=0)}
            deep.append(core[[tuple(x) not in hk for x in core]] if len(core) else core)
        out[pid] = np.unique(np.concatenate(deep), axis=0).astype(np.int32) if deep else np.zeros((0, 3), np.int32)
    return out


# ------------------------------------------------------------------ 去重
def dedup_static(r1, world, UP, thresh=1.5):
    """静态 person 实例 vs 4D 轨迹。返回 (要摘掉的 obj_id, 没对上的).

    ★ 规则: PERSON 走 4D 时**所有** person 实例都摘掉, 然后把"没有任何 4D 轨迹经过"
      的那些单独列出来报给人看。留重影是把坏数据藏起来 —— 而那些静态盒子本来就多半是
      坏的(走动的人被量成拖影), 没对上更可能说明 4D 跟踪漏了人, 那是需要知道的信息。"""
    p = f"{r1}/06_all_boxes.json"
    if not os.path.exists(p):
        return [], []
    boxes = json.load(open(p))
    persons = [b for w in boxes if w.strip().lower() in ("person", "pedestrian", "people")
               for b in boxes[w]]
    if not persons:
        return [], []
    traj = np.concatenate([np.array([fr[v].mean(0) for v in sorted(fr)])
                           for fr in world.values()]) if world else np.zeros((0, 3))
    drop, unmatched = [], []
    for b in persons:
        c = np.array(b["中心"], float)
        d = float(np.linalg.norm(traj - c, axis=1).min()) if len(traj) else 1e9
        drop.append(b["obj_id"])
        if d > thresh:
            unmatched.append(dict(obj_id=b["obj_id"], 尺寸=[round(x, 2) for x in b["尺寸"]],
                                  离最近的4D轨迹=round(d, 2)))
    return drop, unmatched


# ------------------------------------------------------------------ 打包
def static_scene(scene_glb, out, drop_ids):
    """场景里摘掉那几个静态 person(4D 的人顶替了它们), 存成 out(09_scene_nopeople.glb)。返回摘掉的节点名。

    ★ 这份要留在盘上: 验收图渲它, 动作导出(motion_export.py)的场景也是它, 不能用原来的
      07_scene_phy.glb —— 那里面还留着旧的静态人形, 渲出来是"新人+旧人同时在场",
      而且旧人正好填在 4D 覆盖不到的位置, 看着"人都在", 会**高估**效果。第一版就是这么渲的, 图比真实情况好看。
    ★ 2026-10-08 以前这里还往里追加顶点动画的人, 出一个 09_scene_4d.glb(一帧一组顶点, nfl1 154 MB)。
      会动的人现在走骨骼: motion_export.py 出场景 + 人的 FBX(12_motion/), 那个 glb 没人读, 不出了。
    """
    # ★ 必须**精确**匹配, 不能用 startswith: "person_1" 的前缀同时命中 person_1 /
    #   person_10 / person_12 / person_14。这次 drop_ids 恰好是全集所以没咬到人,
    #   但只要哪天改成"只摘一部分", 前缀匹配就会连坐删掉三个不该删的。
    want = {f"{w}_{i}" for i in drop_ids for w in ("person", "pedestrian")}
    sc = trimesh.load(scene_glb)
    keep = trimesh.Scene()
    dropped = []
    for node in sc.graph.nodes_geometry:
        base = node.rsplit(".", 1)[0]          # trimesh 重名时会加 .1 后缀
        if base in want or node in want:
            dropped.append(node); continue
        T, g = sc.graph[node]
        m = sc.geometry[g].copy(); m.apply_transform(T)
        keep.add_geometry(m, node_name=node)
    keep.export(out)
    return dropped


def dump_frame_glb(world, faces, FLIP, vf, path):
    """某一帧的人单独导一个静态 glb, 给渲染回图用(渲染器只会渲静止姿势)。"""
    sc = trimesh.Scene()
    n = 0
    for pid, fr in sorted(world.items()):
        if vf not in fr:
            continue
        V = fr[vf] @ FLIP[:3, :3].T + FLIP[:3, 3]
        sc.add_geometry(trimesh.Trimesh(vertices=V, faces=faces, process=False),
                        node_name=f"person4d_{pid}")
        n += 1
    if n:
        sc.export(path)
    return n


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r1", required=True, help="场景结果目录(底下有 02_da3/ 和 07_scene*.glb)")
    ap.add_argument("--body", required=True, help="body4d 的一轮, 如 sam-body4d/outputs/office/b1")
    ap.add_argument("--scene-in", default="07_scene.glb", help="物理之前的场景: 坐着的人是按它里面的椅子摆的")
    ap.add_argument("--scene-glb", default="07_scene_phy.glb", help="物理之后的场景: 摘掉静态人形、渲验收图用它")
    ap.add_argument("--out", default="09_scene_nopeople.glb", help="摘掉静态人形之后的场景")
    ap.add_argument("--render", action="store_true", help="顺便出渲染回图(调 render_back.py)")
    ap.add_argument("--render-frames", type=int, default=4)
    ap.add_argument("--fr", default=None, help="原始帧目录, --render 时要")
    a = ap.parse_args()

    da3, cams, UP, GH, vid2key = load_scene(a.r1)
    fps = json.load(open(f"{a.body}/00_frames/frames.json"))["src_fps"]
    FLIP = export_T(UP)
    print("== 合成 (物理之后) ==")
    meta = people_store.check_fresh(a.r1, people_identity.source_of(a.body))
    corr = meta["people"]
    # ★ 坐着的人是按物理之前那把椅子的椅面摆的。钉住的椅子物理不会推, 这里再核一遍: 同名椅子物理前后挪了超过 1 cm,
    #   人跟着椅子挪。位移每次都从两份场景重新量、单独存(people_store.save_shift): 不改摆人的结果, 重跑合成也不会加两遍;
    #   网页上的人(网格)和 FBX 里的人(参数)读的时候加的是同一个位移
    shifts = {}
    sat = {p: c["seat_fix"]["seat"]["chair"] for p, c in corr.items() if (c.get("seat_fix") or {}).get("seat")}
    if sat:
        s0, s1 = chair_seats(f"{a.r1}/{a.scene_in}", GH), chair_seats(f"{a.r1}/{a.scene_glb}", GH)
        for p, ch in sat.items():
            if ch in s0 and ch in s1:
                d = (s1[ch][2] - s0[ch][2]) @ FLIP[:3, :3]          # glTF 系位移 -> DA3 世界系
                if np.linalg.norm(d) > 0.01:
                    shifts[p] = d
                    print(f"  人 {p}: 椅子 {ch} 物理之后挪了 {np.linalg.norm(d):.3f}m, 人跟着挪")
    people_store.save_shift(a.r1, meta["solve_id"], shifts)
    world, faces = people_store.load_world(a.r1)
    rows = measure(world, UP, GH)
    tall = np.array([r["身高"] for r in rows.values()])
    bad = [(p, r["身高"]) for p, r in rows.items() if not H_MIN <= r["身高"] <= H_MAX
           and not corr[p]["seated"]]
    if bad:
        k = 1.70 / float(np.median(tall))
        raise SystemExit(
            f"[失败] 尺度自检没过: 站着的人身高 {[f'{h:.2f}' for _, h in bad]}m 落在 {H_MIN}~{H_MAX} 之外。\n"
            f"        人是绝对米制的, 所以错的是场景 —— 它大约差 {k:.2f} 倍(按 1.70m 反推)。")
    for pid, r in rows.items():
        print(f"  人 {pid}: 身高 {r['身高']:.2f}m  脚离地 {r['脚离地']:+.2f}m {r['脚离地范围']}  "
              f"最大一步 {r['帧间位移最大']:.2f}m" + ("  (坐着)" if corr[pid]["seated"] else ""))
    drop, unmatched = dedup_static(a.r1, world, UP)
    static_glb = f"{a.r1}/{a.out}"
    dropped = static_scene(f"{a.r1}/{a.scene_glb}", static_glb, drop)
    print(f"  {static_glb}  (没有人的场景)" + (f"; 摘掉静态 person {dropped}" if dropped else ""))
    rep = dict(场景=a.r1, 人体=a.body, 视频fps=fps, 人=rows, 摆人=meta["报告"],
               摘掉的静态person=drop, 没对上4D的=unmatched, 静态场景=os.path.basename(static_glb))
    json.dump(rep, open(f"{a.r1}/09_body4d_report.json", "w"), ensure_ascii=False, indent=1)

    if a.render:
        if not a.fr:
            raise SystemExit("[失败] --render 要给 --fr 原始帧目录")
        print("== 渲染回图 ==")
        for f in glob.glob(f"{a.r1}/09_people_f*.glb") + glob.glob(f"{a.r1}/09_render_4d_f*.jpg"):
            os.remove(f)                             # 上一次的: 人变了, 抽到的帧号不一样, 旧的会留在目录里冒充这一次的
        allvf = sorted({v for fr in world.values() for v in fr})
        pick = [allvf[int(round(i * (len(allvf) - 1) / max(1, a.render_frames - 1)))]
                for i in range(a.render_frames)]
        for vf in sorted(set(pick)):
            gp = f"{a.r1}/09_people_f{vf:04d}.glb"
            if not dump_frame_glb(world, faces, FLIP, vf, gp):
                continue
            # ★ 渲 09_scene_nopeople.glb, **不是** 07_scene_phy.glb, 理由见 static_scene。
            # ★ render_back 的 FRAMES 是**场景相机的编号**(cameras.json 的键 0037.jpg -> 37),
            #   不是原视频帧号。pedestrian 两者恰好一样, office 不一样(场景帧是隔 3.6 帧抽的)。
            ki = int(vid2key[vf].rsplit(".", 1)[0])
            env = dict(os.environ, R1=a.r1, FR=a.fr,
                       SCENE=f"{os.path.basename(static_glb)},{os.path.basename(gp)}",
                       RBOUT=f"09_render_4d_f{vf:04d}.jpg", FRAMES=str(ki))
            r = subprocess.run([sys.executable,
                                os.path.expanduser("~/sam3d/data/outdoor/scripts/render_back.py")],
                               env=env, capture_output=True, text=True)
            print(f"  帧 {vf:04d} -> 09_render_4d_f{vf:04d}.jpg"
                  + ("" if r.returncode == 0 else f"   ★失败: {r.stderr.strip().splitlines()[-1:]}"))


if __name__ == "__main__":
    main()
