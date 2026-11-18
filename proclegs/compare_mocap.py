"""补腿结果和真人动捕(mocap_stats.py)用同一套测量比: 往前/横着/倒着 各自的步长、迈步时长、落脚方式、抬脚高度和时刻。
用法: python compare_mocap.py --pid 4 --tags _v7 _v8   (rohm 环境)
"""
import argparse, os
import numpy as np
from mocap_stats import analyze

HERE = os.path.dirname(os.path.abspath(__file__))
SMPLX = dict(pelvis=0, thigh_l=1, thigh_r=2, calf_l=4, calf_r=5, foot_l=7, foot_r=8, ball_l=10, ball_r=11, head=15)


def cats(d):
    return np.where(d['ang'] < 45, '往前', np.where(d['ang'] > 135, '倒着', '横着'))


def row(name, d, m):
    L = d['land'][m]
    return '%-10s %4d v%.2f | 一次挪 %.2f  迈步 %.2f s  占比 %.2f | 跟/尖/平 %3.0f/%3.0f/%3.0f%%  落地俯仰 %+5.1f°  离地俯仰 %+5.1f° | 抬踝 %4.1f cm @ %.2f' % (
        name, m.sum(), np.median(d['v'][m]), np.median(d['dist'][m]), np.median(d['Ts'][m]), np.median(d['Ts'][m] / (d['Ts'][m] + d['stance'][m])),
        100 * np.mean(L == 'heel'), 100 * np.mean(L == 'toe'), 100 * np.mean(L == 'flat'),
        np.median(d['pitch_land'][m]), np.median(d['pitch_off'][m]), 100 * np.median(d['lift'][m]), np.median(d['lift_t'][m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, default=4)
    ap.add_argument('--tags', nargs='+', required=True)
    a = ap.parse_args()
    real = np.load(f'{HERE}/outputs/mocap_steps.npz')
    ours = {}
    for t in a.tags:
        o = np.load(f'{HERE}/outputs/p{a.pid}_proclegs{t}_W.npz')
        J = {n: o['joints'][:, j].astype(np.float64) for n, j in SMPLX.items()}
        st, _ = analyze(J, 30.0)
        ours[t] = {k: np.array([s[k] for s in st]) for k in st[0]}
    vs = np.concatenate([d['v'] for d in ours.values()])
    lo, hi = np.percentile(vs, [10, 90])
    print('速度按身高归一; 我们的步速 10–90%% 在 %.2f–%.2f /s, 真人取同一速度段' % (lo, hi))
    rc = cats(real)
    for c in ['往前', '横着', '倒着']:
        print(row('真人', real, (rc == c) & (real['v'] >= lo) & (real['v'] <= hi)), c)
        for t, d in ours.items():
            m = cats(d) == c
            if m.sum():
                print(row('补腿' + t, d, m))


if __name__ == '__main__':
    main()
