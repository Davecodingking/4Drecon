"""v23 补腿结果 -> UE 里的 office P4(rohm 环境)。
1. 用 UE 实际驱动的骨架重跑补腿: UE 里的 SMPL-X 人用平均体型骨架(betas=0, v1.1 neutral; 拟合出的体型只记录不驱动), 骨盆放在本人实际位置。
   p4 本人腿比平均体型短 1.1 cm, 直接把 v23 结果套上去脚会插进地里 -> 骨盆路线不变, 体型换成平均体型, 重跑 v23, 再用 office 关卡打分。
2. 转成 UE 那份 office_P4.npz 的格式(见 ue/scripts/mhr2smplx_official.py): glTF 世界(天 +y, 米), 每个视频帧一行(23.976 fps),
   poses (N,165): 身体 22 个关节用补腿结果, 下巴/眼睛/手用原文件; trans = 骨盆世界位置 - UE 骨架参考姿势的骨盆。
   补腿的世界 W(z 朝上) -> glTF: 两份数据里 p4 上半身关节在同一帧上对齐求出(只旋转+平移), 中位误差 > 3 cm 就停(说明两份不是同一版结果)。
3. 检查: 用 UE 骨架(ue/out/smplx_ref.json)正向算一遍, 脚着地时脚踝离 UE 地面多高、滑多快, 原版 vs 新版;
4. 看 p4 腿用的相机路线: 原视频相机离 p4 太近(1.5~2.3 m), 腿全程在画面外; 放在原相机位置(稍退), 抬高、低头、换 16 mm 广角(UE 里新序列用)。
输出: ue/out/smplx/office_P4_v23.npz(不覆盖原文件)、ue/out/smplx/legcam_v23.json、outputs/p4_proclegs_v23_ue_W.npz、outputs/to_ue_report.json
用法: python to_ue.py
"""
import contextlib, io, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import office_gate as OG

P = OG.load_impl(f'{HERE}/versions/proclegs_v23.py')
from rohm_ours import smplx_model, rest_pelvis          # proclegs 已把 RoHM/ours 加进 sys.path
import smplx

INP = '/home/ubuntu/sam3d/RoHM/ours/inputs/p4_rohm_input.npz'     # 补腿输入(W, 30 fps, SMPL-X 1.0, p4 体型)
CAM = '/home/ubuntu/sam3d/RoHM/ours/inputs/p4_smplx_Wy.npz'       # 关键帧相机(DA3 世界 -> 相机) 和 W <-> DA3
SMPLX11 = '/home/ubuntu/sam3d/redir/smplx_model/SMPLX_NEUTRAL.npz'
UE_IN = '/home/ubuntu/ue/out/smplx/office_P4.npz'
UE_OUT = '/home/ubuntu/ue/out/smplx/office_P4_v23.npz'
UE_REF = '/home/ubuntu/ue/out/smplx_ref.json'
FLOOR_Y = -1.517                                                    # UE 场景地面(glTF y, 米), 同 bake_smplx.py
LEGCAM = '/home/ubuntu/ue/out/smplx/legcam_v23.json'                # 看腿用的相机路线(UE 新序列用)
CAM_BACK, CAM_H, AIM_H, CAM_FOCAL = 0.3, 1.6, 0.85, 16.0           # 比原相机往后退多少、相机离地、对准点离地(米)、焦距 mm
UPPER = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]        # 对齐用: 骨盆、脊柱、脖子、锁骨、头、肩、肘、腕
PAR = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15,
       20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
       21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]
PUE = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])             # UE(cm, z 上) <-> SMPL/glTF(y 上), 自己的逆


def kabsch(A, B):
    """B ≈ A @ R.T + t(只旋转平移)。"""
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca, float(np.trace(np.diag(S) @ D) / ((A - ca) ** 2).sum())


def interp(t, x, tq):
    x = x.reshape(len(t), -1)
    return np.stack([np.interp(tq, t, x[:, i]) for i in range(x.shape[1])], 1)


def fk(poses, trans, J):
    """UE 骨架正向运动学(同 bake_smplx.fk_smplx): poses (N,165), trans (N,3) -> 世界关节 (N,55,3)。"""
    N = len(poses)
    R = Rot.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(N, 55, 3, 3)
    G = np.zeros_like(R); X = np.zeros((N, 55, 3)); G[:, 0] = R[:, 0]; X[:, 0] = J[0]
    for j in range(1, 55):
        p = PAR[j]; G[:, j] = G[:, p] @ R[:, j]; X[:, j] = X[:, p] + G[:, p] @ (J[j] - J[p])
    return X + trans[:, None]


def feet(X, fps, con=None):
    """脚踝(7 左 8 右): 着地帧上离 UE 地面多高、水平滑多快。着地 = 给定的 con, 或 脚踝高度 < 这只脚最低 + 2 cm。"""
    out = {}
    for i, (nm, a) in enumerate((('L', 7), ('R', 8))):
        h = X[:, a, 1] - FLOOR_Y
        v = np.linalg.norm(np.gradient(X[:, a][:, [0, 2]], axis=0), axis=1) * fps
        c = con[:, i].astype(bool) if con is not None else h < h.min() + 0.02
        out[nm] = dict(frames_pct=round(100 * c.mean(), 1), h_cm=[round(100 * float(np.percentile(h[c], q)), 1) for q in (5, 50, 95)],
                       slide_cms=[round(100 * float(np.percentile(v[c], q)), 1) for q in (50, 90)])
    return out


def main():
    rep = {}
    d = np.load(INP); s = d['smplx_world'].astype(np.float64); t = d['time_s']
    m10 = smplx_model()
    m11 = smplx.SMPLX(model_path=SMPLX11, gender='neutral', use_pca=False, flat_hand_mean=True)

    # ---- 1. 平均体型骨架上重跑 v23 ----
    J0p = rest_pelvis(m10, s[:1, 6:16]); J0m = rest_pelvis(m11, np.zeros((1, 10)))
    s2 = s.copy(); s2[:, 6:16] = 0; s2[:, 3:6] = s[:, 3:6] + J0p - J0m       # 骨盆世界位置不变
    with contextlib.redirect_stdout(io.StringIO()):
        o = P.run(s2, t, model=m11, log=lambda *a, **k: None)
    np.savez(f'{HERE}/outputs/p4_proclegs_v23_ue_W.npz', **o)
    sc = OG.scores(o); rows = OG.compare(sc, json.load(open(OG.BASE)))
    ok = not any(r[3] for r in rows)
    v23 = json.load(open(f'{HERE}/outputs/office_gate_v23.json'))
    print(f'1. 平均体型重跑 v23: office 关卡 {"通过" if ok else "不通过"}')
    print(f'   {"项目":<14}{"平均体型":>8}{"p4 体型":>8}{"v10":>8}')
    for k, x, b, w in rows:
        print(f'   {k:<14}{x:>8.1f}{v23[k]:>8.1f}{b:>8.1f}  {"变差" if w else ""}')
    rep['gate'] = dict(pass_=ok, **{k: x for k, x, _, _ in rows})

    # ---- 2. W -> glTF, 换成 UE 格式 ----
    u = np.load(UE_IN); fps = float(u['mocap_framerate']); NF = len(u['poses'])
    tk = u['key_vf'] / fps
    A = interp(t, d['joints_world'], tk).reshape(len(tk), -1, 3)[:, UPPER].reshape(-1, 3)
    B = u['key_joints'][:, UPPER].astype(np.float64).reshape(-1, 3)
    R, tt, scale = kabsch(A, B)
    res = np.linalg.norm(A @ R.T + tt - B, axis=1)
    tilt = float(np.degrees(np.arccos(np.clip((R @ [0, 0, 1.0])[1], -1, 1))))
    floor = float((R @ [*s[:, 3:5].mean(0), 0] + tt)[1] - FLOOR_Y)        # W 的地面(p4 脚下) 在 glTF 里比 UE 地面高多少
    print(f'2. W -> glTF 对齐(上半身 {len(UPPER)} 个关节 x {len(tk)} 个关键帧): 误差 中位 {100 * np.median(res):.1f} cm, '
          f'95% {100 * np.percentile(res, 95):.1f} cm, 最大 {100 * res.max():.1f} cm; 若允许缩放 = {scale:.3f}; '
          f'竖直方向差 {tilt:.2f}°; W 地面比 UE 地面高 {100 * floor:+.1f} cm')
    rep['align'] = dict(res_med_cm=100 * float(np.median(res)), res_p95_cm=100 * float(np.percentile(res, 95)), scale=scale,
                        tilt_deg=tilt, floor_cm=100 * floor, R=R.tolist(), t=tt.tolist())
    if np.median(res) > 0.03:
        sys.exit('对齐误差 > 3 cm: 补腿输入和 UE 那份 P4 不是同一版结果, 停')

    tq = np.clip(np.arange(NF) / fps, t[0], t[-1])
    go = (Rot.from_matrix(R) * Slerp(t, Rot.from_rotvec(o['global_orient']))(tq)).as_rotvec()
    bp = np.stack([Slerp(t, Rot.from_rotvec(o['body_pose'][:, 3 * j:3 * j + 3]))(tq).as_rotvec() for j in range(21)], 1)
    pel = interp(t, o['joints'][:, 0], tq) @ R.T + tt
    ref = json.load(open(UE_REF))['bones']
    names = [str(n) for n in u['joint_names']]
    J_ue = np.array([PUE @ np.array(ref[n]['pos']) / 100.0 for n in names])
    poses = u['poses'].astype(np.float64).reshape(NF, 55, 3).copy()
    poses[:, 0] = go; poses[:, 1:22] = bp
    poses = poses.reshape(NF, 165); trans = pel - J_ue[0]
    np.savez(UE_OUT, poses=poses.astype(np.float32), trans=trans.astype(np.float32), betas=u['betas'], gender=u['gender'],
             mocap_framerate=u['mocap_framerate'], key_vf=u['key_vf'], key_joints=u['key_joints'], fit_err=u['fit_err'],
             joint_names=u['joint_names'],
             说明=np.array('v23 补腿版(sam3d/proclegs/to_ue.py): 身体 22 个关节 = 在平均体型骨架上重跑的 v23 补腿, 下巴/眼睛/手 = office_P4.npz; '
                         '其余约定同 office_P4.npz(glTF 世界, 天 +y, 米, 每个视频帧一行, trans = 骨盆世界位置 - UE 骨架参考姿势骨盆)'))
    print(f'   -> {UE_OUT}  ({NF} 帧 @ {fps:.3f} fps)')

    # ---- 3. 检查 ----
    J11 = m11(betas=__import__('torch').zeros(1, 10)).joints[0, :22].detach().numpy()
    bone = np.abs((J_ue[:22] - J_ue[0]) - (J11 - J11[0])).max()
    X_new, X_old = fk(poses, trans, J_ue), fk(u['poses'].astype(np.float64), u['trans'].astype(np.float64), J_ue)
    ank = interp(t, o['joints'][:, 7:9], tq).reshape(NF, 2, 3) @ R.T + tt
    diff = np.linalg.norm(X_new[:, 7:9] - ank, axis=-1).max()
    con = o['contact'][np.abs(t[None] - tq[:, None]).argmin(1)]
    print(f'3. UE 骨架 vs SMPL-X 1.1 平均体型 关节差最大 {100 * bone:.2f} cm; UE 骨架上算的脚踝 vs 补腿结果 最大差 {100 * diff:.2f} cm')
    print(f'   站直时脚踝离地 {100 * P.Body(m11, np.zeros(10)).ankle_h:.1f} cm(着地时脚踝高度应在这附近)')
    rep['check'] = dict(bone_cm=100 * float(bone), ankle_vs_proclegs_cm=100 * float(diff),
                        old=feet(X_old, fps), new=feet(X_new, fps), new_planned=feet(X_new, fps, con))
    for k, lab in (('old', '原版(着地 = 脚踝 < 最低 + 2 cm)'), ('new', '新版(同一规则)'), ('new_planned', '新版(补腿自己的着地帧)')):
        print(f'   {lab}: ' + '; '.join(f'{f} 着地 {v["frames_pct"]}% 离地 5/50/95% {v["h_cm"]} cm 滑动 中位/90% {v["slide_cms"]} cm/s'
                                       for f, v in rep['check'][k].items()))

    # ---- 4. 看腿用的相机(UE 新序列用) ----
    # 原相机全程离 p4 只有 1.5~2.3 m, 腿在画面下面看不到。新相机: 基本就在原相机的位置(平滑 0.5 s, 往远离 p4 的方向只退 0.3 m;
    # 11-06 第一版退到离 p4 3 m, 通道拐弯处退进了隔墙后面), 离地 1.6 m, 低头对着 p4 骨盆正下方离地 0.85 m 处, 焦距 16 mm(胶片 36 mm 宽, 同原相机)。
    from scipy.ndimage import gaussian_filter1d
    c = np.load(CAM); w2c = c['w2c_da3']
    Cg = (-np.einsum('kba,kb->ka', w2c[:, :3, :3], w2c[:, :3, 3]) @ c['R_W_da3'].T + c['t_W_da3']) @ R.T + tt   # 原相机中心, glTF
    Cf = interp(c['vf'].astype(float), Cg, np.arange(NF))
    ps = gaussian_filter1d(pel, 0.5 * fps, axis=0)
    dirn = Cf - ps; dirn[:, 1] = 0; dirn /= np.linalg.norm(dirn, axis=1, keepdims=True)
    dirn = gaussian_filter1d(dirn, 1.5 * fps, axis=0); dirn /= np.linalg.norm(dirn, axis=1, keepdims=True)
    cam = gaussian_filter1d(Cf, 0.5 * fps, axis=0) + CAM_BACK * dirn; cam[:, 1] = FLOOR_Y + CAM_H
    aim = ps.copy(); aim[:, 1] = FLOOR_Y + AIM_H
    f = (aim - cam) / np.linalg.norm(aim - cam, axis=1, keepdims=True)
    d_ue = f @ PUE.T
    yaw = np.degrees(np.unwrap(np.arctan2(d_ue[:, 1], d_ue[:, 0]))); pitch = np.degrees(np.arcsin(d_ue[:, 2]))
    json.dump(dict(focal=CAM_FOCAL, loc=(100 * cam @ PUE.T).tolist(), pitch=pitch.tolist(), yaw=yaw.tolist(),
                   说明='看 p4 腿的相机(sam3d/proclegs/to_ue.py): UE 世界 cm / 度, 每个视频帧一个, 滚转 0'), open(LEGCAM, 'w'))
    rgt = np.cross(f, [0, 1.0, 0]); rgt /= np.linalg.norm(rgt, axis=1, keepdims=True); up = np.cross(rgt, f)
    q = X_new[:, :22] - cam[:, None]; z = (q * f[:, None]).sum(-1)
    fr = np.maximum(np.abs((q * rgt[:, None]).sum(-1) / z) / (18 / CAM_FOCAL),
                    np.abs((q * up[:, None]).sum(-1) / z) / (18 * 1072 / 1920 / CAM_FOCAL)).max(1)
    print(f'4. 看腿相机 -> {LEGCAM}: p4 全身关节离画面中心最远 = 半个画面的 中位 {np.median(fr):.2f} / 最大 {fr.max():.2f} 倍(< 1 = 都在画面里); '
          f'离原相机 {np.linalg.norm(cam - Cf, axis=1).min():.2f}~{np.linalg.norm(cam - Cf, axis=1).max():.2f} m')
    rep['legcam'] = dict(frame_ratio_med=float(np.median(fr)), frame_ratio_max=float(fr.max()))
    json.dump(rep, open(f'{HERE}/outputs/to_ue_report.json', 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
