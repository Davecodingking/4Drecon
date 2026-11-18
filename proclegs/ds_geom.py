"""两脚着地时两脚的几何(rohm 环境): 按移动方向分解两踝距离(前后/左右), 真人 vs 补腿, 看"叉太开"是步子太长还是左右太宽。
用法: python ds_geom.py base_v10 v22chk v23a [--office]"""
import glob, os, re, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from mocap_stats import analyze

HERE = os.path.dirname(os.path.abspath(__file__))
HARD = re.compile(r'turn-05-turn|MartialArts|leap|jump|run', re.I)
OFFICE = re.compile(r'side_step|backward|lookfor|look_around|look_at|thinking|call|stand', re.I)


def geom(J):
    _, o = analyze(J, 30.0); c = o['contact'].all(1)
    p = gaussian_filter1d(J['pelvis'][:, :2], 7.5, axis=0); v = np.gradient(p, axis=0) * 30; sp = np.linalg.norm(v, axis=1)
    u = v / np.maximum(sp, 1e-9)[:, None]; n = np.stack([-u[:, 1], u[:, 0]], 1)
    dlt = J['foot_l'][:, :2] - J['foot_r'][:, :2]
    m = c & (sp > 0.3)
    return np.abs((dlt * u).sum(1))[m], np.abs((dlt * n).sum(1))[m], np.linalg.norm(dlt, axis=1)[m]


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith('--')]
    keep = (lambda n: not HARD.search(n) and OFFICE.search(n)) if '--office' in sys.argv else (lambda n: not HARD.search(n) and not OFFICE.search(n))
    print(f'{"":<10}{"前后 中位/95%":>16}{"左右 中位/95%":>16}{"距离 95%/99%":>16}  (m, 走动 >0.3 m/s, 两脚着地)')
    for k, tag in enumerate(tags):
        R, O = [], []
        for f in sorted(glob.glob(f'{HERE}/outputs/bench/{tag}/clips/*.npz')):
            d = np.load(f)
            if not keep(str(d['name']).replace('__', '/')):
                continue
            for side, L in (('real', R), ('ours', O)):
                if side == 'real' and k > 0:
                    continue
                L.append(geom({n[len(side) + 1:]: d[n].astype(np.float64) for n in d.files if n.startswith(side + '_')}))
        for nm, L in (('真人', R), (tag, O)):
            if not L:
                continue
            a, b, c = (np.concatenate([x[j] for x in L]) for j in range(3))
            print(f'{nm:<10}{np.median(a):>8.2f}/{np.percentile(a, 95):<7.2f}{np.median(b):>8.2f}/{np.percentile(b, 95):<7.2f}{np.percentile(c, 95):>8.2f}/{np.percentile(c, 99):<7.2f}')


if __name__ == '__main__':
    main()
