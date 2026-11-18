"""真人动捕当考场(rohm 环境): 从 AMASS 走路片段里只拿骨盆路线和朝向(假装腿看不见), 用 proclegs 补腿,
和这个人真实的腿用同一套体检(qa_legs.py)比。每改一次规则跑一遍: 看各类问题"次/分钟"和真人比, 以及问题最多的片段和时刻。

做法: 动捕(MetaHuman 骨架)统一到 30 fps -> 按腿长缩放到 office p4 的 SMPL-X 身材 -> 骨盆位置取两髋中点换算成 SMPL-X 骨盆关节,
朝向取两髋连线的水平朝向(不要前后左右倾斜) -> 上半身给 0(只看腿) -> proclegs.run。
--degrade: 把骨盆路线弄得像 office 的输入: 每秒只取 6.6 帧再线性插值, 每个取样点加噪声(位置 8 mm、高度 5 mm、朝向 4°)。
用法:
  python bench.py --tag v10 [--degrade] [--limit N] [--workers 48] [--recalib]
输出 outputs/bench/<tag>[_degrade]/summary.json 和 clips/*.npz(补腿关节、真实关节、事件)
阈值文件 outputs/bench/qa_thresholds.json: 第一次(或 --recalib)用真人的腿定, 之后各版本共用, 才能比。
"""
import argparse, contextlib, io, json, os, re, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
PCT = 99.5
G = {}


def load_impl(path):
    """要考的补腿程序(默认当前 proclegs.py; 旧版本放在 versions/ 里, 用 --impl 指定, 才能在同一个考场公平比)。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location('impl', path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def init(impl=None):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, HERE)
    from mocap_stats import Skel
    G['impl'] = load_impl(impl or f'{HERE}/proclegs.py')
    smplx_model, Body = G['impl'].smplx_model, G['impl'].Body
    G['sk'] = Skel(); G['model'] = smplx_model()
    d = np.load('/home/ubuntu/sam3d/RoHM/ours/inputs/p4_rohm_input.npz')
    G['betas'] = d['smplx_world'][0, 6:16].astype(np.float64)
    G['body'] = Body(G['model'], G['betas'])
    b = G['body']; G['hip_h'] = float((b.J[1, 1] + b.J[2, 1]) / 2 - (b.J[0, 1] - b.pelvis_h))   # 髋高(离脚底), 旧版本 Body 没有 .H


def short(n):
    return re.sub(r'_stageii\.npz$', '', n).replace('__', '/')


def make_input(J, body, betas, degrade, seed):
    """真人关节(已缩放, 30 fps) -> SMPL-X 输入 (T, 79)。"""
    upright = G['impl'].upright
    l = J['thigh_l'] - J['thigh_r']; l[:, 2] = 0; l /= np.linalg.norm(l, axis=1, keepdims=True)
    yaw = np.unwrap(np.arctan2(-l[:, 0], l[:, 1]))                          # 左 = (-sin, cos) -> yaw
    hipmid = (J['thigh_l'] + J['thigh_r']) / 2
    off = (body.J[1] + body.J[2]) / 2 - body.J[0]                            # 模型坐标里 骨盆 -> 两髋中点
    T = len(yaw)
    if degrade:
        rng = np.random.default_rng(seed)
        key = np.unique(np.round(np.arange(0, T - 1, 30 / 6.6)).astype(int).tolist() + [T - 1])
        hk = hipmid[key] + rng.normal(0, 1, (len(key), 3)) * [0.008, 0.008, 0.005]
        yk = yaw[key] + np.radians(4) * rng.normal(0, 1, len(key))
        hipmid = np.stack([np.interp(np.arange(T), key, hk[:, d]) for d in range(3)], 1)
        yaw = np.interp(np.arange(T), key, yk)
    G0 = np.stack([upright(y) for y in yaw])
    pel = hipmid - np.einsum('tij,j->ti', G0, off)
    s = np.zeros((T, 79))
    s[:, 0:3] = Rot.from_matrix(G0).as_rotvec(); s[:, 3:6] = pel - body.J[0]; s[:, 6:16] = betas
    return s


def work(args):
    name, degrade, seed = args
    from mocap_stats import AMASS, analyze
    from qa_legs import signals, leg_len, SMPLX
    run = G['impl'].run
    J, fps = G['sk'].positions(f'{AMASS}/{name}')
    t0 = np.arange(len(J['pelvis'])) / fps; t = np.arange(0, t0[-1], 1 / 30)
    if t[-1] < 3.0:
        return None
    J = {n: np.stack([np.interp(t, t0, v[:, d]) for d in range(3)], 1) for n, v in J.items()}
    body = G['body']
    r = G['hip_h'] / leg_len(J)                                              # 按髋高缩放到 SMPL-X 这个人
    J = {n: v * r for n, v in J.items()}
    path = np.linalg.norm(np.diff(J['pelvis'][:, :2], axis=0), axis=1).sum()
    if path < 0.5:
        return None
    s = make_input(J, body, G['betas'], degrade, seed)
    tic = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        o = run(s, t, G['model'], log=lambda *a, **k: None)
    Jo = {n: o['joints'][:, j].astype(np.float64) for n, j in SMPLX.items()}
    st_r, st_o = analyze(J, 30.0)[0], analyze(Jo, 30.0)[0]                       # 两边用同一个检测方法数步、量步
    steps_real, steps_det = len(st_r), len(st_o)
    keys = ('v', 'dist', 'Ts', 'ang', 'lift', 'stance')
    gait_r = np.array([[x[k] for k in keys] for x in st_r]).reshape(-1, len(keys)); gait_o = np.array([[x[k] for k in keys] for x in st_o]).reshape(-1, len(keys))
    return dict(name=name, dur=float(t[-1]), path=float(path), sec=time.time() - tic, t=t,
                sig_real=signals(J), sig_ours=signals(Jo), steps_real=steps_real, steps_ours=steps_det,
                z_real=J['pelvis'][:, 2] - J['pelvis'][:, 2].mean(), z_in=o['pelvis_z_in'], z_out=o['pelvis_z_out'],
                Jr={n: J[n].astype(np.float32) for n in SMPLX}, Jo={n: Jo[n].astype(np.float32) for n in SMPLX},
                why=o['step_why'], gait_r=gait_r, gait_o=gait_o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--degrade', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('BENCH_WORKERS', 48)))
    ap.add_argument('--recalib', action='store_true')
    ap.add_argument('--impl', default=None, help='要考的程序文件(默认 proclegs.py), 如 versions/proclegs_v10.py')
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    from mocap_stats import pick_clips
    from qa_legs import events, NAMES, THR_FILE
    clips = pick_clips(a.limit)
    tic = time.time()
    with Pool(a.workers, initializer=init, initargs=(a.impl,)) as pool:
        res = [r for r in pool.map(work, [(n, a.degrade, i) for i, n in enumerate(clips)], chunksize=1) if r is not None]
    print(f'{len(res)} 段(共 {sum(r["dur"] for r in res) / 60:.1f} 分钟), 用时 {time.time() - tic:.0f} s')
    os.makedirs(os.path.dirname(THR_FILE), exist_ok=True)
    if a.recalib or not os.path.exists(THR_FILE):
        thr = {}
        for k in NAMES:
            x = np.concatenate([r['sig_real'][k] for r in res])
            x = x[x > 0] if k in ('off_center', 'wide_stance') else x
            thr[k] = float(np.percentile(x, PCT))
        mins = sum(r['dur'] for r in res) / 60
        rate = {k: sum(sum(e[0] == k for e in events(r['sig_real'], thr, r['t'])) for r in res) / mins for k in NAMES}
        json.dump(dict(pct=PCT, thr=thr, real_rate=rate, real_rate_total=sum(rate.values()), clips=len(res)), open(THR_FILE, 'w'), indent=1, ensure_ascii=False)
        print('阈值 ->', THR_FILE)
    thr = json.load(open(THR_FILE))['thr']
    mins = sum(r['dur'] for r in res) / 60
    out = f'{HERE}/outputs/bench/{a.tag}' + ('_degrade' if a.degrade else '')
    os.makedirs(f'{out}/clips', exist_ok=True)
    per = []
    for i, r in enumerate(res):
        er, eo = events(r['sig_real'], thr, r['t']), events(r['sig_ours'], thr, r['t'])
        per.append(dict(name=short(r['name']), dur=r['dur'], steps_real=r['steps_real'], steps_ours=r['steps_ours'],
                        gait_r=r['gait_r'].tolist(), gait_o=r['gait_o'].tolist(), ev_real=er, ev_ours=eo, z_sink=float(np.max(np.minimum(r['z_in'], r['z_out'].max()) - r['z_out']))))
        np.savez_compressed(f'{out}/clips/{i:03d}.npz', name=r['name'], t=r['t'], why=r['why'],
                            **{'real_' + n: v for n, v in r['Jr'].items()}, **{'ours_' + n: v for n, v in r['Jo'].items()})
    print(f'{"项目":<22}{"阈值":>8}{"真人 次/分钟":>12}{"补腿 次/分钟":>12}   补腿问题最多的片段(次数, 第一次的时刻)')
    summ = {}
    for k in NAMES:
        nr = sum(sum(e[0] == k for e in p['ev_real']) for p in per); no = sum(sum(e[0] == k for e in p['ev_ours']) for p in per)
        worst = sorted(per, key=lambda p: -sum(e[0] == k for e in p['ev_ours']))[:3]
        ws = '; '.join(f'{p["name"].split("/")[-1][:28]}({sum(e[0] == k for e in p["ev_ours"])}, {next(e[1] for e in p["ev_ours"] if e[0] == k):.1f}s)'
                       for p in worst if any(e[0] == k for e in p['ev_ours']))
        summ[k] = dict(real=nr / mins, ours=no / mins)
        print(f'{NAMES[k]:<22}{thr[k]:>8.2f}{nr / mins:>12.1f}{no / mins:>12.1f}   {ws}')
    sr = np.array([p['steps_ours'] / max(p['steps_real'], 1) for p in per])
    zs = np.array([p['z_sink'] for p in per])
    print(f'步数 补腿/真人: 中位 {np.median(sr):.2f} (25–75% {np.percentile(sr, 25):.2f}–{np.percentile(sr, 75):.2f}); '
          f'骨盆被压低(比 输入和站直取低的 再低) 每段最大: 中位 {100 * np.median(zs):.1f} cm, 90% {100 * np.percentile(zs, 90):.1f} cm')
    json.dump(dict(tag=a.tag, degrade=a.degrade, minutes=mins, rates=summ, per_clip=per,
                   steps_ratio_median=float(np.median(sr)), z_sink_median=float(np.median(zs))),
              open(f'{out}/summary.json', 'w'), ensure_ascii=False, indent=1)
    print('->', out)


if __name__ == '__main__':
    main()
