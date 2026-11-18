"""问题是哪类步子引起的(rohm 环境): 对考场片段重跑补腿, 每个"骨盆偏一侧/叉太开"事件找它前面最近落地的一步, 按那一步的触发原因统计。
用法: python event_causes.py [--group office|normal] [--workers 48]   (参数覆盖用环境变量 PROCLEGS_OPT, 同 gate.py --opt)
"""
import argparse, contextlib, io, os, re, sys
from collections import Counter
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
HARD = re.compile(r'turn-05-turn|MartialArts|leap|jump|run', re.I)
OFFICE = re.compile(r'side_step|backward|lookfor|look_around|look_at|thinking|call|stand', re.I)


def work(name):
    import bench
    from mocap_stats import AMASS
    from qa_legs import signals, events, load_thr, leg_len, SMPLX
    G = bench.G
    J, fps = G['sk'].positions(f'{AMASS}/{name}')
    t0 = np.arange(len(J['pelvis'])) / fps; t = np.arange(0, t0[-1], 1 / 30)
    if t[-1] < 3.0:
        return []
    J = {n: np.stack([np.interp(t, t0, v[:, d]) for d in range(3)], 1) for n, v in J.items()}
    r = G['hip_h'] / leg_len(J); J = {n: v * r for n, v in J.items()}
    s = bench.make_input(J, G['body'], G['betas'], False, 0)
    with contextlib.redirect_stdout(io.StringIO()):
        o = G['impl'].run(s, t, G['model'], log=lambda *a, **k: None)
    Jo = {n: o['joints'][:, j].astype(np.float64) for n, j in SMPLX.items()}
    ev = events(signals(Jo), load_thr()['thr'], t)
    st, why = o['steps'], o['step_why']
    out = []
    for k, a, b, pk, tp in ev:
        if k not in ('off_center', 'wide_stance'):
            continue
        land = st[:, 2]; j = np.flatnonzero(land <= tp + 1e-6)
        if not len(j):
            out.append((k, 'start', 'none', 0.0)); continue
        j = j[-1]
        prev = why[j - 1].split('(')[0] if j > 0 else 'none'
        out.append((k, why[j].split('(')[0], prev, float(tp - land[j])))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--group', default='office'); ap.add_argument('--workers', type=int, default=48)
    a = ap.parse_args()
    import bench
    from mocap_stats import pick_clips
    names = [n for n in pick_clips(0) if not HARD.search(n.replace('__', '/'))]
    names = [n for n in names if bool(OFFICE.search(n)) == (a.group == 'office')]
    with Pool(a.workers, initializer=bench.init) as pool:
        res = [e for r in pool.map(work, names, chunksize=1) for e in r]
    for k in ('off_center', 'wide_stance'):
        ev = [e for e in res if e[0] == k]
        print(f'{k}: {len(ev)} 次; 最近落地那一步的原因: ' + ', '.join(f'{w} {n}' for w, n in Counter(e[1] for e in ev).most_common()))
        print(f'   再前一步: ' + ', '.join(f'{w} {n}' for w, n in Counter(e[2] for e in ev).most_common()))
        dt = np.array([e[3] for e in ev]); print(f'   离落地多久 中位 {np.median(dt):.2f} s, 25–75% {np.percentile(dt, 25):.2f}–{np.percentile(dt, 75):.2f}')


if __name__ == '__main__':
    main()
