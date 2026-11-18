"""腿抖不抖(rohm 环境): 膝角变化速度、膝/踝位置高频抖动(去掉 ~0.07 s 平滑后剩下的), 真人动捕和我们同一套算法、同速度段。
用法: python jitter_check.py --tags _v8 _v9
"""
import argparse, os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from mocap_stats import Skel, AMASS, pick_clips
from compare_mocap import SMPLX

HERE = os.path.dirname(os.path.abspath(__file__))


def measure(J, fps, H):
    out = {'rate': [], 'hf': []}
    pel = gaussian_filter1d(J['pelvis'][:, :2], 0.25 * fps, axis=0)
    sp = np.linalg.norm(np.gradient(pel, axis=0), axis=1) * fps / H
    m = (sp > 0.08) & (sp < 0.35)
    for s in 'lr':
        a = J['thigh_' + s] - J['calf_' + s]; b = J['foot_' + s] - J['calf_' + s]
        ang = np.degrees(np.arccos(np.clip((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1)))
        if fps != 30:                                                     # 统一到 30 fps 再算, 免得帧率影响高频
            t0 = np.arange(len(ang)) / fps; t1 = np.arange(0, t0[-1], 1 / 30)
            ang = np.interp(t1, t0, ang); kn = np.stack([np.interp(t1, t0, J['calf_' + s][:, d]) for d in range(3)], 1)
            mm = np.interp(t1, t0, m.astype(float)) > 0.5
        else:
            kn, mm = J['calf_' + s], m
        out['rate'].append((np.abs(np.gradient(ang)) * 30)[mm])
        out['hf'].append(np.linalg.norm(kn - gaussian_filter1d(kn, 2, axis=0), axis=1)[mm])
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True)
    a = ap.parse_args()
    sk = Skel(); R = {'rate': [], 'hf': []}
    for n in pick_clips(0):
        J, fps = sk.positions(f'{AMASS}/{n}')
        H = np.median(J['head'][:, 2]) + 0.10
        r = measure(J, fps, H)
        for k in R:
            R[k].append(r[k])
    R = {k: np.concatenate(v) for k, v in R.items()}
    show = lambda nm, r: print('%-10s 膝角变化速度 95%% %4.0f °/s, 99%% %4.0f, 最大 %4.0f | 膝位置高频抖 95%% %4.1f mm, 99%% %4.1f, 最大 %4.1f' % (
        nm, *np.percentile(r['rate'], [95, 99]), r['rate'].max(), *(1000 * np.percentile(r['hf'], [95, 99])), 1000 * r['hf'].max()))
    show('真人', R)
    for t in a.tags:
        o = np.load(f'{HERE}/outputs/p4_proclegs{t}_W.npz')
        J = {n: o['joints'][:, j].astype(np.float64) for n, j in SMPLX.items()}
        show('补腿' + t, measure(J, 30.0, 1.746))


if __name__ == '__main__':
    main()
