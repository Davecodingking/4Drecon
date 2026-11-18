"""从真人动捕(AMASS, 已转成 MetaHuman 86 骨)量走路的规律, 给补腿规则定参数(rohm 环境)。
每一步按"脚移动方向和骨盆朝向的夹角"分: 往前(<45°)、横着(45–135°)、倒着(>135°)。
量: 步长(按身高归一)、迈步时长、步频、两脚着地占比、抬脚高度和最高点时刻、落脚时脚跟/脚尖/平着先着地、
    骨盆上下起伏、左右晃、水平转动、左右倾斜(和步子的相位)。
数据: /cfs/dataset/metahuman_dataset/AMASS/*.npz (pose 局部四元数 + 根平移, UE 坐标 cm)
骨架: /cfs/retarget/AnimationAssets/direct_npz/assets/metahuman_441.pt (骨长), FK 和坐标换算借用同目录 npz2pt_anims.py
用法: python mocap_stats.py [--limit N]  -> outputs/mocap_steps.npz + 打印汇总
"""
import argparse, glob, os, re, sys
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, binary_opening, binary_closing
sys.path.insert(0, '/cfs/retarget/AnimationAssets/direct_npz')
from npz2pt_anims import ue2cv_quat, fk_positions, T_MAT, DEFAULT_SKELETON

HERE = os.path.dirname(os.path.abspath(__file__))
AMASS = '/cfs/dataset/metahuman_dataset/AMASS'
H_TOL, V_TOL = 0.04, 0.5      # 踩地: 离最低处 < 4 cm 且水平速度 < 0.5 m/s(用往前走迈步约占 40% 校准)
NEED = ['pelvis', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r', 'foot_l', 'foot_r', 'ball_l', 'ball_r', 'head']


def pick_clips(limit):
    names = sorted(os.listdir(AMASS))
    bad = re.compile(r'run|jump|hop|skip|crawl|stair|beam|cartwheel|moonwalk|one_leg|oneleg|catwalk|drank|sneak|baggage|push|pull|cane|exhausted|kick|punch|dance', re.I)
    out = []
    for n in names:
        if bad.search(n):
            continue
        if n.startswith('ACCAD__') and re.search(r'walk|side_step|sidestep|backward', n, re.I):
            out.append(n)
        elif n.startswith('Eyes_Japan_Dataset__') and re.search(r'__(walk-(01|05|08|10|19|22|24|25|27|28|29)|turn-0[1235])-', n):
            out.append(n)
    bml = [n for n in names if n.startswith('BMLrub__') and re.search(r'normal_walk|circle_walk', n)]
    out += bml[::max(1, len(bml) // 40)]
    return out[:limit] if limit else out


class Skel:
    def __init__(self):
        sk = torch.load(DEFAULT_SKELETON, map_location='cpu', weights_only=False)
        self.names = [str(n) for n in sk['joint_names']]
        self.parents = [int(p) for p in sk['parents']]
        self.rest_rot = sk['rest_rotations'].numpy().astype(np.float64)
        self.rest_tra = sk['rest_translations'].numpy().astype(np.float64)
        self.idx = {n: i for i, n in enumerate(self.names)}
        # 只做到 NEED 这些骨头的祖先, 省时间
        keep = set()
        for n in NEED:
            j = self.idx[n]
            while j != -1:
                keep.add(j); j = self.parents[j]
        self.keep = sorted(keep)
        self.sub = {j: k for k, j in enumerate(self.keep)}
        self.sub_parents = [self.sub[self.parents[j]] if self.parents[j] != -1 else -1 for j in self.keep]

    def positions(self, path):
        d = np.load(path, allow_pickle=True)
        bones = [str(b) for b in d['bones'].tolist()]
        pose = ue2cv_quat(d['pose'].astype(np.float64)); trans = d['trans'].astype(np.float64)
        F = len(pose)
        rot = np.broadcast_to(self.rest_rot[self.keep], (F, len(self.keep), 4)).copy()
        tra = np.broadcast_to(self.rest_tra[self.keep], (F, len(self.keep), 3)).copy()
        for k, b in enumerate(bones):
            j = self.idx.get(b)
            if j in self.sub:
                rot[:, self.sub[j]] = pose[:, k]
        tra[:, self.sub[self.idx['pelvis']]] = (trans @ T_MAT.T) / 100.0
        if 'root' in self.idx and self.idx['root'] in self.sub:
            r = self.sub[self.idx['root']]; rot[:, r] = [0, 0, 0, 1.0]; tra[:, r] = 0
        P = fk_positions(rot, tra, self.sub_parents) @ T_MAT          # UE 轴(x 前, y 右, z 上), 米
        P[..., 1] *= -1                                                # -> 右手系: y 朝左
        return {n: P[:, self.sub[self.idx[n]]] for n in NEED}, float(d['fps'])


def runs(mask):
    """布尔序列 -> [(开始, 结束(不含))]"""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    e = np.flatnonzero(np.diff(m))
    return list(zip(e[::2], e[1::2]))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def analyze(J, fps):
    F = len(J['pelvis'])
    up = np.array([0, 0, 1.0])
    left = J['thigh_l'] - J['thigh_r']; left[:, 2] = 0; left /= np.linalg.norm(left, axis=1, keepdims=True)
    fwd = np.cross(left, up)
    yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    height = np.percentile((J['thigh_l'][:, 2] + J['thigh_r'][:, 2]) / 2, 90) * 1.813   # 按髋高(腿长)换算的等效身高, 动捕骨架 身高/髋高 = 1.813; 不同骨架之间才可比
    leg = np.median(np.linalg.norm(J['thigh_l'] - J['calf_l'], axis=1) + np.linalg.norm(J['calf_l'] - J['foot_l'], axis=1))
    pel = J['pelvis']
    pxy_s = gaussian_filter1d(pel[:, :2], 0.25 * fps, axis=0)
    vel = np.gradient(pxy_s, axis=0) * fps; speed = np.linalg.norm(vel, axis=1)
    floor = min(np.percentile(J['ball_l'][:, 2], 3), np.percentile(J['ball_r'][:, 2], 3))
    steps, contact = [], np.zeros((F, 2), bool)
    fl = {}
    for i, s in enumerate('lr'):
        A, B = J['foot_' + s], J['ball_' + s]
        a0 = np.percentile(A[:, 2], 3); b0 = np.percentile(B[:, 2], 3)
        va = np.linalg.norm(np.gradient(A[:, :2], axis=0), axis=1) * fps
        vb = np.linalg.norm(np.gradient(B[:, :2], axis=0), axis=1) * fps
        heel = (A[:, 2] < a0 + H_TOL) & (va < V_TOL)
        toe = (B[:, 2] < b0 + H_TOL) & (vb < V_TOL)
        c = binary_closing(binary_opening(heel | toe, np.ones(3)), np.ones(3))
        contact[:, i] = c
        # 脚的俯仰: 踝->前掌 连线和水平的夹角, 减去踩稳时的中位数 -> 0 = 放平, >0 脚跟抬起(脚尖低), <0 脚尖翘起
        fv = B - A
        pitch = np.degrees(np.arctan2(-fv[:, 2], np.linalg.norm(fv[:, :2], axis=1)))
        flat = np.median(pitch[heel & toe]) if (heel & toe).sum() > 5 else np.median(pitch[c])
        pitch = pitch - flat
        fl[s] = (A, B, heel, toe, pitch, a0)
    for i, s in enumerate('lr'):
        A, B, heel, toe, pitch, a0 = fl[s]
        cr = runs(contact[:, i])
        for (s0, e0), (s1, e1) in zip(cr[:-1], cr[1:]):
            k0, k1 = e0, s1                                            # 迈步: 离地 k0 .. 落地 k1
            Ts = (k1 - k0) / fps
            if Ts < 0.15 or Ts > 1.2 or k0 < 3 or k1 > F - 3:
                continue
            mid = (A[k0] + A[k1]) / 2
            dvec = A[k1, :2] - A[k0 - 1, :2]
            dist = np.linalg.norm(dvec)
            if dist < 0.05:
                continue
            km = (k0 + k1) // 2
            ang = np.degrees(abs(wrap(np.arctan2(dvec[1], dvec[0]) - yaw[km])))
            # 落脚方式: 看脚跟(踝)和前掌谁先回到这次踩稳时的高度(+1 cm)。差 < 1 帧算平着落
            st_end = min(e1, F)
            ah, bh = np.median(A[k1:st_end, 2]), np.median(B[k1:st_end, 2])
            kap = k0 + int(np.argmax(A[k0:k1, 2]))
            ta = next((k for k in range(kap, st_end) if A[k, 2] <= ah + 0.01), st_end)
            tb = next((k for k in range(kap, st_end) if B[k, 2] <= bh + 0.01), st_end)
            land = 'heel' if ta < tb else ('toe' if tb < ta else 'flat')
            kl = min(ta, tb)
            lift = A[k0:k1, 2] - a0
            stance_len = (e1 - s1) / fps
            # 这一步期间的速度: 骨盆从这只脚上一次落地到这次落地之间的平均
            kk0 = s0
            v = np.linalg.norm(pxy_s[k1] - pxy_s[kk0]) / max((k1 - kk0) / fps, 1e-3)
            dyaw = np.degrees(wrap(yaw[min(k1, F - 1)] - yaw[k0]))
            steps.append(dict(foot=i, k0=k0, k1=k1, Ts=Ts, dist=dist / height, ang=ang, land=land,
                              pitch_land=pitch[kl], land_lag=abs(ta - tb) / fps, pitch_off=pitch[k0 - 1], lift=lift.max(), lift_t=(np.argmax(lift) + 0.5) / max(k1 - k0, 1),
                              v=v / height, stance=stance_len, dyaw=dyaw))
    # 骨盆起伏: 去掉 1 s 的慢变化, 看和步子的关系
    z = pel[:, 2] - gaussian_filter1d(pel[:, 2], 0.5 * fps)
    latv = np.stack([-np.sin(yaw), np.cos(yaw)], 1)
    lat = ((pel[:, :2] - gaussian_filter1d(pel[:, :2], 0.5 * fps, axis=0)) * latv).sum(1)
    yaw_osc = np.degrees(yaw - gaussian_filter1d(yaw, 0.5 * fps))
    hl = J['thigh_l'] - J['thigh_r']
    obl = np.degrees(np.arctan2(hl[:, 2], np.linalg.norm(hl[:, :2], axis=1)))   # >0 左髋高
    obl = obl - gaussian_filter1d(obl, 0.5 * fps)
    return steps, dict(z=z, lat=lat, yaw_osc=yaw_osc, obl=obl, contact=contact, speed=speed / height, height=height, leg=leg, fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    clips = pick_clips(a.limit)
    print(len(clips), '段')
    sk = Skel()
    rows, osc = [], []
    for ci, n in enumerate(clips):
        try:
            J, fps = sk.positions(f'{AMASS}/{n}')
            st, o = analyze(J, fps)
        except Exception as e:
            print('跳过', n, e); continue
        for s in st:
            s['clip'] = ci; rows.append(s)
        c = o['contact']; fps = o['fps']
        # 每个"单脚着地"段的中点 vs 两脚着地段: 骨盆高度、左右偏、倾斜
        for i in (0, 1):
            for s0, e0 in runs(c[:, i] & ~c[:, 1 - i]):
                if e0 - s0 < 3:
                    continue
                km = (s0 + e0) // 2
                osc.append(dict(clip=ci, foot=i, dur=(e0 - s0) / fps, v=o['speed'][km],
                                z_mid=o['z'][km], lat_mid=o['lat'][km] * (1 if i == 0 else -1),     # >0 = 往踩地的那只脚那边
                                obl_mid=o['obl'][km] * (1 if i == 0 else -1),                        # >0 = 踩地那边髋高(迈步那边掉下去)
                                yaw_mid=o['yaw_osc'][km]))
        for s0, e0 in runs(c[:, 0] & c[:, 1]):
            if e0 - s0 < 1 or s0 == 0 or e0 == len(c):
                continue
            km = (s0 + e0) // 2
            osc.append(dict(clip=ci, foot=-1, dur=(e0 - s0) / fps, v=o['speed'][km], z_mid=o['z'][km], lat_mid=0, obl_mid=0, yaw_mid=0))
        if ci % 20 == 0:
            print(ci, n, len(st), '步')
    keys = rows[0].keys()
    np.savez(f'{HERE}/outputs/mocap_steps.npz', clips=np.array(clips), **{k: np.array([r[k] for r in rows]) for k in keys},
             **{'osc_' + k: np.array([r[k] for r in osc]) for k in osc[0].keys()})
    print(len(rows), '步 ->', f'{HERE}/outputs/mocap_steps.npz')


if __name__ == '__main__':
    main()
