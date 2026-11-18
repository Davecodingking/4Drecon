"""真人转弯/加速时脚放在骨盆哪边(rohm 环境): 用考场存的真人关节(已缩放到 office p4 身材, 30 fps)。
两脚着地的帧: 两踝中点相对骨盆的左右偏移(按两髋连线的朝向, >0 = 往左) vs 骨盆的左右加速度(同一朝向, >0 = 往左加速)。
物理上: 要往左拐(往左加速), 地面得把人往左推, 脚就得在骨盆右边 -> 偏移 ≈ -k x 加速度。拟合 k(单位 s^2 = m / (m/s^2))。
用法: python measure_lateral.py [--tag base_v10]
"""
import argparse, glob, json, os, re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from mocap_stats import analyze

HERE = os.path.dirname(os.path.abspath(__file__))
HARD = re.compile(r'turn-05-turn|MartialArts|leap|jump|run', re.I)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='base_v10'); a = ap.parse_args()
    X, Y, V, F = [], [], [], []
    for f in sorted(glob.glob(f'{HERE}/outputs/bench/{a.tag}/clips/*.npz')):
        d = np.load(f)
        if HARD.search(str(d['name'])):
            continue
        J = {n[5:]: d[n].astype(np.float64) for n in d.files if n.startswith('real_')}
        _, o = analyze(J, 30.0); c = o['contact'].all(1)
        l = J['thigh_l'] - J['thigh_r']; l[:, 2] = 0; l /= np.linalg.norm(l, axis=1, keepdims=True)
        fw = np.stack([l[:, 1], -l[:, 0]], 1)
        p = gaussian_filter1d(J['pelvis'][:, :2], 0.25 * 30, axis=0)                      # 和 proclegs 一样平滑
        vel = np.gradient(p, axis=0) * 30; acc = np.gradient(vel, axis=0) * 30
        mid = (J['foot_l'][:, :2] + J['foot_r'][:, :2]) / 2 - J['pelvis'][:, :2]
        X.append((acc * l[:, :2]).sum(1)[c]); Y.append((mid * l[:, :2]).sum(1)[c])
        V.append(np.linalg.norm(vel, axis=1)[c]); F.append((mid * fw).sum(1)[c])
    X, Y, V = map(np.concatenate, (X, Y, V))
    m = V > 0.3
    k, b = np.polyfit(X[m], Y[m], 1)
    print(f'走动(>0.3 m/s) 两脚着地 {m.sum()} 帧: 两踝中点左右偏移 = {k:+.3f} x 左右加速度 {b:+.3f} m;  相关 {np.corrcoef(X[m], Y[m])[0, 1]:.2f}')
    for lo, hi in [(-3, -1), (-1, -0.5), (-0.5, -0.2), (-0.2, 0.2), (0.2, 0.5), (0.5, 1), (1, 3)]:
        mm = m & (X >= lo) & (X < hi)
        if mm.sum() > 20:
            print(f'  加速度 {lo:+.1f}..{hi:+.1f} m/s^2: {mm.sum():5d} 帧, 偏移 中位 {np.median(Y[mm]):+.3f} m')
    json.dump(dict(k=k, b=b), open(f'{HERE}/outputs/lateral_fit.json', 'w'))


if __name__ == '__main__':
    main()


def step_width(tag='base_v10'):
    """两脚着地时两踝左右间距(按两髋连线朝向), 按速度(等效身高/s)和方向分档: 中位、5%、95%。"""
    rows = []
    for f in sorted(glob.glob(f'{HERE}/outputs/bench/{tag}/clips/*.npz')):
        d = np.load(f)
        if HARD.search(str(d['name'])):
            continue
        J = {n[5:]: d[n].astype(np.float64) for n in d.files if n.startswith('real_')}
        _, o = analyze(J, 30.0); c = o['contact'].all(1)
        l = J['thigh_l'] - J['thigh_r']; l[:, 2] = 0; l /= np.linalg.norm(l, axis=1, keepdims=True)
        fw = np.stack([l[:, 1], -l[:, 0]], 1)
        p = gaussian_filter1d(J['pelvis'][:, :2], 7.5, axis=0); v = np.gradient(p, axis=0) * 30
        sp = np.linalg.norm(v, axis=1); H = np.percentile((J['thigh_l'][:, 2] + J['thigh_r'][:, 2]) / 2, 90) * 1.813
        ang = np.degrees(np.arccos(np.clip((v * fw).sum(1) / np.maximum(sp, 1e-9), -1, 1)))
        w = ((J['foot_l'][:, :2] - J['foot_r'][:, :2]) * l[:, :2]).sum(1)
        rows.append(np.stack([sp / H, ang, w], 1)[c])
    R = np.concatenate(rows)
    print('两脚着地时 左踝-右踝 左右距离(m, >0 = 没交叉), 按速度(等效身高/s)')
    for nm, m0 in (('往前', R[:, 1] < 45), ('横着', (R[:, 1] >= 45) & (R[:, 1] <= 135)), ('倒着', R[:, 1] > 135)):
        for lo, hi in [(0, 0.04), (0.04, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.9)]:
            m = m0 & (R[:, 0] >= lo) & (R[:, 0] < hi)
            if m.sum() > 50:
                print(f'  {nm} {lo:.2f}-{hi:.2f}: {m.sum():5d} 帧  中位 {np.median(R[m, 2]):.3f}  5% {np.percentile(R[m, 2], 5):.3f}  95% {np.percentile(R[m, 2], 95):.3f}')
