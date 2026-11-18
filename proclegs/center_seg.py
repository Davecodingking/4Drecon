"""分段看两脚着地时骨盆在不在两脚中间(rohm 环境; 0 = 正中, 1 = 压在一只脚上, >1 两脚在骨盆同一侧)。用法: python center_seg.py --tags _v8 _v9"""
import argparse, os, warnings
import numpy as np
from scipy.ndimage import gaussian_filter1d
from mocap_stats import analyze
from compare_mocap import SMPLX
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
SEGS = [('1 s 横挪', 0.5, 2.5), ('9 s 横挪', 8, 10.6), ('15 s 边转边换方向', 14, 16.5)]
ap = argparse.ArgumentParser(); ap.add_argument('--tags', nargs='+', required=True); a = ap.parse_args()
for tag in a.tags:
    o = np.load(f'{HERE}/outputs/p4_proclegs{tag}_W.npz'); J = {n: o['joints'][:, j].astype(float) for n, j in SMPLX.items()}; t = o['time_s']
    _, oo = analyze(J, 30.0); c = oo['contact']
    l = J['thigh_l'] - J['thigh_r']; l[:, 2] = 0; l /= np.linalg.norm(l, axis=1, keepdims=True)
    pel = J['pelvis'][:, :2]; v = np.gradient(gaussian_filter1d(pel, 7.5, axis=0), axis=0) * 30; sp = np.linalg.norm(v, axis=1)
    p = (pel * l[:, :2]).sum(1); aL = (J['foot_l'][:, :2] * l[:, :2]).sum(1) - p; aR = (J['foot_r'][:, :2] * l[:, :2]).sum(1) - p
    off = np.abs((aL + aR) / 2) / np.maximum(np.abs(aL - aR) / 2, 1e-3)
    m = c.all(1) & (sp > 0.1)
    row = [f'{nm} 中位 {np.median(off[m & (t >= lo) & (t < hi)]):.2f}/同侧 {(off[m & (t >= lo) & (t < hi)] > 1).sum()} 帧' for nm, lo, hi in SEGS]
    print(f'{tag:5s} ' + ' | '.join(row) + f' | 全段 中位 {np.median(off[m]):.2f}/同侧 {(off[m] > 1).sum()}/{m.sum()} 帧')
