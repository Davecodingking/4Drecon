#!/usr/bin/env python
"""人物动作(一手数据版): 在 SAM 3D Body 的一手参数上做修复, 直接得到世界系的参数 —— 喂给 MHR 就是网格和骨架。

★ 为什么有它(2026-10-08): body_into_scene.py 的修复都做在**网格**上(模型的关节数据当初没存), 要骨骼动作文件
  只能从网格反推: 关节角度不保证和模型算的一样, 局部差到十几厘米, 每人多花 7 分钟。
  这里换成一手数据(存在 <出网格目录>/mhr_4d_individual/, 见 mhr_body.py), 修复直接改参数:

    修复                现在(网格上)                    这里(参数上)
    远近 / 位置 / 贴地   整个人平移                      根关节的位置
    认错人的帧           两帧网格之间逐顶点过渡           关节参数插值, 根朝向球面插值; 只有一头就定住
    姿势抖动             骨盆系里逐顶点平滑               每个关节参数按时间平滑, 根朝向按四元数平滑

★ 判据一条没动, 全部调 body_into_scene 里现成的: 哪几帧算认错人(flip_segments)、远近按场景深度(frame_ratios)、
  补帧怎么补(occ_plan)、位置整理(tidy_track)。它们要吃世界系的网格 —— 现在的网格是从参数生成的。

★ 还没接进流水线: 不写 10_people.json, 不管椅子/坐姿/路线(那些还在 body_into_scene 里)。先对着同一轮的老结果
  比指标、出对比视频, 人看过再接。

用法:
  python body_motion.py --r1 <场景结果目录> --mesh-run <带 mhr_4d_individual 的出网格目录> --out <输出目录>
产物(--out 下):
  motion.npz        每人: 世界系(glTF, 天 +y)的参数 (T,204) + 体型 (45) + 视频帧号 —— 这就是修好的一手数据
  people_world.npz  每人每帧的网格顶点(DA3 世界系, 含姿势修正项), 给验收视频用
  report.json       每人的报告
sam3d 环境(torch + scipy + trimesh)。有 GPU 就用。
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/sam3d/data/outdoor/scripts"))
import body_into_scene as B
import mhr_body as M
from world_frame import export_T

POSE = slice(6, M.NPOSE)          # 各关节(含手)的参数; [0:6] 是根


def load_params(mesh_run):
    """{人: {帧: (params 204, cam_t 3)}}, {人: 体型 45}"""
    root = f"{mesh_run}/mhr_4d_individual"
    if not os.path.isdir(root):
        raise SystemExit(f"[失败] {root} 不存在 —— 这一轮出网格时还没存一手数据(2026-10-08 以后才存), 重跑出网格那步")
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
    速度在根不动时的网格上量(= 骨盆系里的身体), 和 smooth_pose 在骨盆系里量顶点速度是一回事。"""
    t = np.asarray(t, float)

    def f(pose):
        P0 = np.zeros((len(t), M.NPOSE + M.NSCALE)); P0[:, POSE] = pose; P0[:, M.NPOSE:] = scales
        V0, _ = M.forward(shape, P0, False, dev)
        sv = np.linalg.norm(np.diff(V0, axis=0), axis=2) / np.maximum(np.diff(t), 1e-6)[:, None]
        return M.limb_speed(np.maximum(np.r_[sv[:1], sv], np.r_[sv, sv[-1:]]), dev)[:, POSE]
    return f


def smooth_joints(pose, t, limb_speed, sigma=B.POSE_SIGMA, sigma_min=B.POSE_SIGMA_MIN, v0=B.POSE_V0):
    """pose (T,D) 关节参数按时间平滑, 快动作少平滑(和 smooth_pose 同一个思路, 对象从顶点换成关节)。
    limb_speed(pose) -> (T,D): 每个参数带动的那段肢体里最快的顶点的速度(m/s), 见 mhr_body.limb_speed。
    ★ "快不快"不能看关节自己的转速: 一次挥臂是肩、肘、腕叠出来的, 每个关节单看都只是中等速度, 各被平滑掉一半,
      沿手臂累加到手上 —— 2026-10-08 nfl1 人 6 第 105 帧左手离模型原始动作 43 cm(在网格上平滑是 18 cm)。
      要看的是它带动的肢体末端有多快, 阈值和顶点版同一个 v0。"""
    pose = np.asarray(pose, float); t = np.asarray(t, float); T = len(t)
    if T < 5:
        return pose.copy()
    strong = B._gauss_t(pose, t, np.full(T, sigma))
    sp = limb_speed(strong)                                                  # 速度按强平滑版求, 不然抖动本身被当成"快"
    a = v0 / (v0 + _near_max(sp, t, 2 * sigma))
    return a * strong + (1 - a) * B._gauss_t(pose, t, np.full(T, sigma_min))


def smooth_rots(Q, t, sigma=B.POSE_SIGMA, sigma_min=B.POSE_SIGMA_MIN, w0=B.POSE_W0):
    """Q (T,3,3) 根朝向按四元数平滑, 转得快的帧少平滑(照搬 smooth_pose 里朝向那一段)。"""
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
    """管骨头长度的那几个参数(cols): 模型每帧给同一个人估的都不一样, 仿真里骨架该是固定的。
    pose (T,D) 关节参数; reliable (T,) bool: 这一帧模型可信(没被翻转补帧换掉); cols: 这几个参数在 pose 里的列。
    返回新的 pose。现在什么都不改 —— 要不要定成每人一个常数、取什么值, 还没定。"""
    pose = np.array(pose, float, copy=True)
    # TODO(human): 每人每个长度参数定成一个常数。例如 pose[:, cols] = 某个统计量(pose[reliable][:, cols])。
    return pose


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


def solve(par, shape, cams, vid2key, kept, UP, GH, mesh_run, da3, fps, focals, dev="cpu"):
    """返回 {人: dict(frames, vf, params(世界系 DA3), shape, conf)}, 报告。流程和 body_into_scene.solve_people 一一对应。"""
    RATIO_TOL = 0.25
    todo, all_r = {}, []
    for pid, fr in par.items():
        bis = sorted(b for b in fr if b < len(kept) and kept[b] in vid2key and vid2key[kept[b]] in cams)
        if not bis:
            continue
        rt = {b: fr[b][1] for b in bis}
        ratios = B.depth_ratio(pid, bis, rt, mesh_run, da3, vid2key, kept)
        todo[pid] = (bis, rt, ratios); all_r += list(ratios.values())
    R_all = float(np.median(all_r)) if all_r else 1.0
    out, rep = {}, {}
    for pid, (bis, rt, ratios) in todo.items():
        T = len(bis)
        tt = np.array([kept[b] / fps for b in bis])
        tmap = {b: kept[b] / fps for b in bis}
        c2w = np.stack([np.linalg.inv(np.array(cams[vid2key[kept[b]]]["extrinsic_w2c_4x4"], float)) for b in bis])
        Pc = np.stack([par[pid][b][0] for b in bis]); ct = np.stack([rt[b] for b in bis])
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
        raw_p = p.copy()
        # 1b) 翻转补帧: 换的是关节参数、根朝向、根关节位置
        occ = {}
        if segs:
            mroot = B.mask_roots(pid, bis, rt, rf, focals[pid], mesh_run, cams, vid2key, kept)
            occ = B.occ_plan(bis, segs, tmap, dict(zip(bis, p)), mroot)
            pose, Q, p = fill_flips(pose, Q, p, bis, occ)
        # 1a') 骨头长度: 模型逐帧估的都不一样(见 fix_bone_lengths; 现在还没处理, 原样通过)
        pose = fix_bone_lengths(pose, np.array([str(b) not in occ for b in bis]), M.length_params(dev) - POSE.start)
        # 1b') 肢体级的毛刺: 单条手臂/腿的关节角跳出去又跳回来的那几帧, 换成前后的过渡(见 fix_limb_spikes)
        groups = {n: c - POSE.start for n, c in M.limb_groups(dev).items()}
        pose, limb_fix = fix_limb_spikes(pose, tt, groups)
        # 1c) 姿势平滑: 关节参数 + 根朝向, 根关节位置不动
        pose_raw = pose.copy()
        if T >= 5:
            pose, Q = smooth_joints(pose, tt, limb_speed_fn(shape[pid], Pw[0, M.NPOSE:], tt, dev)), smooth_rots(Q, tt)
        # 2) 轨迹整理: 骨盆位置三个方向一起平滑 + 贴地 + 腾空(tidy_track 吃网格, 网格从整理好姿势的参数生成)
        Pw = Pw.copy(); Pw[:, POSE] = pose
        Pw = M.set_root(Pw, Q, p)
        W, _ = M.forward(shape[pid], Pw, True, dev)
        conf = np.array([B.CONF_SUB if str(b) in occ else 1.0 for b in bis])
        delta = B.tidy_track(dict(zip(bis, W)), bis, tt, conf, UP, GH)
        p = p + delta
        Pw = M.set_root(Pw, Q, p)
        feet = np.array([float(((v + d) @ UP).min() - GH) for v, d in zip(W, delta)])
        step = np.linalg.norm(np.diff(p, axis=0), axis=1) if T > 1 else np.zeros(1)
        step0 = np.linalg.norm(np.diff(raw_p, axis=0), axis=1) if T > 1 else np.zeros(1)
        out[pid] = dict(frames=np.array(bis), vf=np.array([kept[b] for b in bis]), params=Pw, shape=shape[pid], conf=conf)
        rep[pid] = dict(帧数=T, 远近比例=round(R, 3), 自己量的比例=round(R_own, 3), 用全体中位=bool(R != R_own),
                        量到深度的帧=len(ratios), 翻转补帧=B._fix_report(occ, flips, kept),
                        肢体跳变修复={n: [[int(kept[bis[i0]]), int(kept[bis[i1]])] for i0, i1 in v] for n, v in limb_fix.items()},
                        关节参数平滑_改动中位_度=round(float(np.degrees(np.median(np.abs(pose - pose_raw)))), 2) if T >= 5 else None,
                        脚离地范围=[round(float(feet.min()), 3), round(float(feet.max()), 3)],
                        帧间位移最大_前=round(float(step0.max()), 3), 帧间位移最大_后=round(float(step.max()), 3),
                        可信帧比例=round(float((conf >= 1.0).mean()), 2))
    return out, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r1", required=True); ap.add_argument("--mesh-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--body", default="", help="body4d 那一轮(读 00_frames/frames.json); 不给就读 09_body4d_report.json")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    da3, cams, UP, GH, vid2key = B.load_scene(a.r1)
    body = a.body or json.load(open(f"{a.r1}/09_body4d_report.json"))["人体"]
    fj = json.load(open(f"{body}/00_frames/frames.json")); kept, fps = fj["kept_indices"], fj["src_fps"]
    par, shape = load_params(a.mesh_run)
    print(f"一手数据: {len(par)} 人, 每人 {[len(v) for v in par.values()]} 帧; 视频 {fps:.2f} fps; 设备 {dev}")
    res, rep = solve(par, shape, cams, vid2key, kept, UP, GH, a.mesh_run, da3, fps, B.load_focals(a.mesh_run), dev)
    FL = export_T(UP)
    mo, wv = dict(fps=np.float32(fps)), dict(fps=np.float32(fps))
    for pid, r in res.items():
        V, _ = M.forward(r["shape"], r["params"], True, dev)
        wv[f"p{pid}_V"] = V.astype(np.float32); wv[f"p{pid}_vf"] = r["vf"].astype(np.int32)
        mo[f"p{pid}_params"] = M.transform(r["params"], FL).astype(np.float32)        # glTF 世界系(天 +y), 导动作文件直接用
        mo[f"p{pid}_shape"] = r["shape"].astype(np.float32); mo[f"p{pid}_vf"] = r["vf"].astype(np.int32)
        mo[f"p{pid}_conf"] = r["conf"].astype(np.float32)
        q = rep[pid]
        print(f"  人 {pid}: {q['帧数']:3d} 帧  远近 x{q['远近比例']:.3f}" + (" (离群, 用全体中位)" if q["用全体中位"] else "")
              + f"  脚离地 {q['脚离地范围']}  最大一步 {q['帧间位移最大_前']:.2f} -> {q['帧间位移最大_后']:.2f} m"
              + f"  关节平滑改动中位 {q['关节参数平滑_改动中位_度']}°")
        if q["肢体跳变修复"]:
            print("         肢体跳变: " + "; ".join(f"{n} {' '.join(f'{a}~{b}' for a, b in v)}" for n, v in q["肢体跳变修复"].items()))
        for g in q["翻转补帧"]:
            print(f"         翻转补帧: 第 {g['帧'][0]}~{g['帧'][1]} 帧({g['帧数']} 帧, 最大一翻 {g['最大一翻']}°) {g['补法']}")
    np.savez_compressed(f"{a.out}/motion.npz", **mo)
    np.savez_compressed(f"{a.out}/people_world.npz", **wv)
    json.dump(rep, open(f"{a.out}/report.json", "w"), ensure_ascii=False, indent=1)
    print(f"-> {a.out}/motion.npz  people_world.npz  report.json")


if __name__ == "__main__":
    main()
