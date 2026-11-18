"""考场里某类问题出现在什么情况下(rohm 环境): 按速度、移动方向(往前/横着/倒着)、转身快慢、是否在片段开头 1 s, 每分钟次数, 补腿 vs 真人。
用法: python bench_where.py v16c2 wide_stance off_center"""
import json, re, sys, glob, os
import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
HARD = re.compile(r'turn-05-turn|MartialArts|leap|jump|run', re.I)


def context(d):
    hm = (d['real_thigh_l'] + d['real_thigh_r']) / 2
    v = np.gradient(gaussian_filter1d(hm[:, :2], 7.5, axis=0), axis=0) * 30
    l = d['real_thigh_l'] - d['real_thigh_r']; yaw = np.unwrap(np.arctan2(-l[:, 0], l[:, 1]))
    H = np.percentile(hm[:, 2], 90) * 1.813
    sp = np.linalg.norm(v, axis=1) / H
    ang = np.degrees(np.abs((np.arctan2(v[:, 1], v[:, 0]) - yaw + np.pi) % (2 * np.pi) - np.pi))
    turn = np.degrees(np.abs(np.gradient(gaussian_filter1d(yaw, 5)))) * 30
    return sp, ang, turn


def main():
    tag, cats = sys.argv[1], sys.argv[2:]
    s = json.load(open(f'{HERE}/outputs/bench/{tag}/summary.json'))
    files = sorted(glob.glob(f'{HERE}/outputs/bench/{tag}/clips/*.npz'))
    groups = {'速度 <0.15': lambda sp, a, tu, t: sp < 0.15, '速度 0.15-0.45': lambda sp, a, tu, t: (sp >= 0.15) & (sp < 0.45),
              '速度 0.45-0.9': lambda sp, a, tu, t: sp >= 0.45,
              '往前(走动)': lambda sp, a, tu, t: (sp >= 0.15) & (a < 45), '横着(走动)': lambda sp, a, tu, t: (sp >= 0.15) & (a >= 45) & (a <= 135),
              '倒着(走动)': lambda sp, a, tu, t: (sp >= 0.15) & (a > 135),
              '转身 <30°/s': lambda sp, a, tu, t: tu < 30, '转身 30-90°/s': lambda sp, a, tu, t: (tu >= 30) & (tu < 90), '转身 >90°/s': lambda sp, a, tu, t: tu >= 90,
              '开头 1 s': lambda sp, a, tu, t: t < 1.0}
    tot = {g: [0.0, {c: [0, 0] for c in cats}] for g in groups}
    for f, p in zip(files, s['per_clip']):
        if HARD.search(p['name']):
            continue
        d = np.load(f); t = d['t']; sp, ang, turn = context(d)
        for g, fn in groups.items():
            m = fn(sp, ang, turn, t); tot[g][0] += m.sum() / 30 / 60
            for c in cats:
                for side, j in (('ev_real', 0), ('ev_ours', 1)):
                    for e in p[side]:
                        if e[0] == c:
                            k = min(int(round(e[4] * 30)), len(t) - 1)
                            tot[g][1][c][j] += int(m[k])
    print(f'{"情况":<16}{"分钟":>6}' + ''.join(f'{c + " 真人/补腿 次/分钟":>34}' for c in cats))
    for g, (mins, cc) in tot.items():
        print(f'{g:<16}{mins:>6.1f}' + ''.join(f'{cc[c][0] / max(mins, 1e-6):>20.1f} / {cc[c][1] / max(mins, 1e-6):<10.1f}' for c in cats))


if __name__ == '__main__':
    main()
