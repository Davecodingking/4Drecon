"""补腿结果的指标(rohm 环境)。前两行和 RoHM 对比同一套算法(RoHM/ours/eval_raw.py), 后面是走路本身的检查。
用法: /home/ubuntu/miniforge3/envs/rohm/bin/python eval_legs.py --pid 4 [--tag _v2]
"""
import argparse, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/home/ubuntu/sam3d/RoHM/ours')
from rohm_ours import smplx_model, body_joints
from eval_raw import report, knee
from proclegs import gait, Body, WR_SPEED, WR_STEP, WR_HEIGHT, STILL

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    inp = np.load(f'/home/ubuntu/sam3d/RoHM/ours/inputs/p{a.pid}_rohm_input.npz'); s = inp['smplx_world']
    m = smplx_model()
    Ji, Vi = body_joints(m, {'global_orient': s[:, 0:3], 'transl': s[:, 3:6], 'betas': s[:, 6:16], 'body_pose': s[:, 16:79]})
    o = np.load(f'{HERE}/outputs/p{a.pid}_proclegs{a.tag}_W.npz')
    V, J, t = o['vertices'], o['joints'], o['time_s']
    report('输入(原版插值30fps)', Vi, Ji[:, :22], inp['time_s'])
    report('补腿' + a.tag, V, J, t, Ji[:, :22])

    print('\n-- 走路本身 --')
    st = o['steps']; sp = o['speed']; c = o['contact']
    body = Body(m, s[0, 6:16].astype(np.float64))
    moving = sp > STILL
    land = st[:, 2]
    print('步数 %d; 走动的时间 %.1f s, 平均速度 %.2f m/s -> 步频 %.0f 步/分 (真人表在这个速度: %.0f 步/分, 步长 %.2f m)'
          % (len(st), moving.sum() / 30, sp[moving].mean(), len(st) / (moving.sum() / 30) * 60,
             60 * sp[moving].mean() / (np.interp(sp[moving].mean(), WR_SPEED, WR_STEP) * body.height / WR_HEIGHT),
             np.interp(sp[moving].mean(), WR_SPEED, WR_STEP) * body.height / WR_HEIGHT))
    move = np.linalg.norm(st[:, 5:7] - st[:, 3:5], axis=1)
    print('每步脚移动距离: 中位 %.2f m, 最大 %.2f m; 迈步时长 中位 %.2f s' % (np.median(move), move.max(), np.median(st[:, 2] - st[:, 1])))
    print('同一只脚连着迈两次: %d 次' % np.sum(st[1:, 0] == st[:-1, 0]))
    print('两脚都离地的帧: %.1f%%; 只有一只脚着地: %.1f%%' % (100 * np.mean(c.sum(1) == 0), 100 * np.mean(c.sum(1) == 1)))
    dz = o['pelvis_z_in'] - o['pelvis_z_out']
    print('骨盆压低: 中位 %.1f cm, 95%% %.1f cm, 最大 %.1f cm (最大在 %.1f s)' % (np.median(dz) * 100, np.percentile(dz, 95) * 100, dz.max() * 100, t[dz.argmax()]))
    print('IK 够不着 最大 %.1f cm; 踝和目标差 最大 %.1f mm' % (o['ik_miss'].max() * 100, o['ankle_err'].max() * 1000))
    low = V[..., 2].min(1)
    print('穿地(最低点 < -1 cm)的帧: %.1f%%, 最低 %.1f cm' % (100 * np.mean(low < -0.01), low.min() * 100))
    gap = np.linalg.norm(J[:, 7, :2] - J[:, 8, :2], axis=1)
    print('两踝水平距离: 最小 %.2f m (在 %.1f s), 5%% 分位 %.2f m' % (gap.min(), t[gap.argmin()], np.percentile(gap, 5)))
    f = Rot.from_rotvec(o['global_orient']).as_matrix() @ np.array([0, 0, 1.0])
    yaw = np.arctan2(f[:, 1], f[:, 0])
    tw = np.degrees(np.abs((o['foot_yaw'] - yaw[:, None] + np.pi) % (2 * np.pi) - np.pi))
    print('脚尖和骨盆朝向差: 中位 %.0f°, 最大 %.0f°' % (np.median(tw), tw.max()))
    out = np.degrees((o['foot_yaw'] - yaw[:, None] + np.pi) % (2 * np.pi) - np.pi) * np.array([1, -1])   # > 0 外八, < 0 内八
    print('内八 > 10° 的帧: 左 %.0f%% 右 %.0f%%; 最内八 左 %.0f° 右 %.0f°' % (100 * np.mean(out[:, 0] < -10), 100 * np.mean(out[:, 1] < -10), -out[:, 0].min(), -out[:, 1].min()))
    zr = np.abs(np.gradient(o['pelvis_z_out'], t)) * 100
    ref = np.minimum(o['pelvis_z_in'], o['pelvis_z_out'].max())
    print('骨盆高度变化速度: 95%% %.0f cm/s, 最大 %.0f cm/s; 比(原版高度, 站直)取低的再低: 95%% %.1f cm, 最大 %.1f cm (在 %.1f s)'
          % (np.percentile(zr, 95), zr.max(), 100 * np.percentile(ref - o['pelvis_z_out'], 95), 100 * (ref - o['pelvis_z_out']).max(), t[(ref - o['pelvis_z_out']).argmax()]))
    if 'foot_pitch' in o.files:
        ph = np.degrees(o['foot_pitch'])[c]
        print('踩地时脚的俯仰: 抬脚跟的帧 %.0f%% (最大 %.0f°), 翘脚尖的帧 %.0f%% (最大 %.0f°)'
              % (100 * np.mean(ph > 0.5), ph.max(), 100 * np.mean(ph < -0.5), -ph.min()))
    kL, kR = knee(J, 1, 4, 7), knee(J, 2, 5, 8)
    print('膝角(180=直): 踩地时中位 L %.0f R %.0f; 迈步时最小 L %.0f R %.0f'
          % (np.median(kL[c[:, 0]]), np.median(kR[c[:, 1]]), kL[~c[:, 0]].min() if (~c[:, 0]).any() else 0, kR[~c[:, 1]].min() if (~c[:, 1]).any() else 0))


if __name__ == '__main__':
    main()
