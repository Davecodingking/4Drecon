"""两脚着地时骨盆在不在两脚中间(rohm 环境)。按骨盆左右方向投影, 0 = 正中, 1 = 压在一只脚上, >1 = 在脚外。
真人用动捕横着走的段。用法: python center_check.py --tags _v8 _v9"""
import argparse, os, re, warnings
import numpy as np
from scipy.ndimage import gaussian_filter1d
from mocap_stats import Skel, analyze, AMASS, pick_clips
from compare_mocap import SMPLX
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))


def stats(J, contact, fps, H):
    l = J['thigh_l'] - J['thigh_r']; l[:, 2] = 0; l /= np.linalg.norm(l, axis=1, keepdims=True)
    pel = J['pelvis'][:, :2]; v = np.gradient(gaussian_filter1d(pel, 0.25 * fps, axis=0), axis=0) * fps; sp = np.linalg.norm(v, axis=1)
    side = np.abs((v * l[:, :2]).sum(1)) > 0.7 * np.maximum(sp, 1e-9)
    p = (pel * l[:, :2]).sum(1); aL = (J['foot_l'][:, :2] * l[:, :2]).sum(1); aR = (J['foot_r'][:, :2] * l[:, :2]).sum(1)
    m = side & (sp > 0.06 * H) & contact.all(1)
    span = np.abs(aL - aR); off = np.abs(p - (aL + aR) / 2) / np.maximum(span / 2, 1e-3)
    return off[m], span[m] / H


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tags', nargs='+', required=True); a = ap.parse_args()
    sk = Skel(); O, S = [], []
    for n in pick_clips(0):
        if re.search(r'side_?step', n, re.I):
            J, fps = sk.positions(f'{AMASS}/{n}'); _, o = analyze(J, fps)
            x, y = stats(J, o['contact'], fps, o['height']); O.append(x); S.append(y)
    show = lambda nm, o, s: print('%-9s 横着走、两脚着地 %4d 帧: 骨盆离两脚中点 中位 %.2f, 75%% %.2f, 90%% %.2f; 两脚左右叉开 中位 %.2f 身高' % (
        nm, len(o), np.median(o), np.percentile(o, 75), np.percentile(o, 90), np.median(s)))
    show('真人', np.concatenate(O), np.concatenate(S))
    for t in a.tags:
        d = np.load(f'{HERE}/outputs/p4_proclegs{t}_W.npz'); J = {n: d['joints'][:, j].astype(np.float64) for n, j in SMPLX.items()}
        _, o = analyze(J, 30.0); show('补腿' + t, *stats(J, o['contact'], 30.0, o['height']))


if __name__ == '__main__':
    main()
