"""接观测实验(rohm 环境): 腿"一会儿看得见、一会儿看不见"时, 补的腿能不能和看得见的真腿接上。
动捕考场(bench.py 同一批片段, 去掉跑跳等超纲的, 只要 >= 5 s; 先把地面拉平, 见 level_floor): 每段随机藏掉中间一截(从 25~45% 处开始, 长 25~45%),
藏掉的部分只给骨盆路线, 看得见的部分给真腿(按 SMPL-X 踝高换算的踝位置、脚朝向、俯仰、踩地), 三种做法比:
  A 整段补   不管看得见的腿, 整段按骨盆路线补(现在 v23 的做法)
  B 硬切     整段照常补, 看得见的帧换成真腿, 交界处 0.3 s 渐变(proclegs.run join='cut')
  C 接上     看不见的段从真腿最后的状态接着规划, 段尾最后几步落到真腿下一段的脚印上(join='plan')
指标(都只看"藏掉的部分 + 交界前后 0.5 s"):
  体检问题 次/分钟(qa_legs, 和真人同一窗口比); 交界前后 0.5 s 踩地的脚滑多快(支点 = 踝和前掌里慢的那个); 交界前后 0.3 s 膝角一帧最大变化;
  同一只脚连迈; 看得见的帧踝离真腿多远(A 会改掉真腿); 藏掉的帧踝离真腿多远(只作参考: 真腿怎么走本来猜不到)。
用法: python bench_join.py [--workers 48] [--limit N]   -> outputs/bench/join/summary.json、clips/*.npz
"""
import argparse, contextlib, io, json, os, re, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
HARD = re.compile(r'turn-05-turn|MartialArts|leap|jump|run', re.I)
OFFICE = re.compile(r'side_step|backward|lookfor|look_around|look_at|thinking|call|stand', re.I)
OUT = f'{HERE}/outputs/bench/join'
MODES = ('A', 'B', 'C')


def obs_from_joints(J, body, vis, impl):
    """真人关节(动捕骨架, 已缩放) -> 看得见的帧的观测(make_obs)。踝高换成 SMPL-X 的(放平时离地 body.ankle_h); 朝向/俯仰按 踝->前掌 连线。"""
    from mocap_stats import analyze
    _, o = analyze(J, 30.0); c = o['contact']
    T = len(vis); ank = np.zeros((T, 2, 3)); fyaw = np.zeros((T, 2)); phi = np.zeros((T, 2))
    for i, sd in enumerate('lr'):
        A, Bl = J['foot_' + sd], J['ball_' + sd]
        fv = Bl - A
        rest = body.ball[i] - body.J[impl.LEGS[i][2]]                         # SMPL-X 静止姿势 踝 -> 前掌(x 左, z 前)
        fyaw[:, i] = np.unwrap(np.arctan2(fv[:, 1], fv[:, 0])) - np.arctan2(rest[0], rest[2])
        elev = np.arctan2(-fv[:, 2], np.linalg.norm(fv[:, :2], axis=1))
        phi[:, i] = elev - (np.median(elev[c[:, i]]) if c[:, i].sum() > 5 else np.median(elev))
        flat = c[:, i] & (np.abs(phi[:, i]) < np.radians(5))
        z0 = np.median(A[flat, 2]) if flat.sum() > 5 else np.percentile(A[:, 2], 5)
        ank[:, i] = A; ank[:, i, 2] = A[:, 2] - z0 + body.ankle_h
    return impl.make_obs(vis, c, ank, fyaw, phi)


def level_floor(J):
    """动捕地面不平(这批片段 2 s 内最低的前掌高度 中位漂 2.3 cm, 1/4 超过 4 cm; 踩地检测按"离最低处 < 4 cm", 会整段漏掉踩地):
    按 2 s 滑动最低的前掌高度估地面, 整个人(所有关节)减掉地面的起伏。输入的骨盆和用来比的真腿一起拉平。"""
    from scipy.ndimage import minimum_filter1d, gaussian_filter1d
    fl = gaussian_filter1d(minimum_filter1d(np.minimum(J['ball_l'][:, 2], J['ball_r'][:, 2]), 61), 15)
    fl = fl - np.median(fl)
    return {n: v - np.stack([0 * fl, 0 * fl, fl], 1) for n, v in J.items()}


def knee_angle(J):
    out = []
    for s in 'lr':
        a = J['thigh_' + s] - J['calf_' + s]; b = J['foot_' + s] - J['calf_' + s]
        out.append(np.degrees(np.arccos(np.clip((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1))))
    return np.stack(out, 1)


def measure(Jx, Jr, t, vis, thr, impl):
    """一段的指标。Jx: 要量的腿(补腿结果或真人), Jr: 真人(算离真腿多远)。"""
    from mocap_stats import analyze
    from qa_legs import signals, events, NAMES
    T = len(t)
    segs = impl.runs_of(~vis)
    bnd = [ka for ka, kb in segs if ka > 0] + [kb for ka, kb in segs if kb < T]
    dist = np.min(np.abs(np.arange(T)[:, None] - np.array(bnd)[None]), 1)
    Bw = dist <= 15                                                            # 交界前后 0.5 s
    Wv = ~vis | Bw                                                             # 评估窗口
    ev = events(signals(Jx), thr, t)
    cnt = {k: sum(1 for e in ev if e[0] == k and Wv[min(int(round(e[4] * 30)), T - 1)]) for k in NAMES}
    _, o = analyze(Jx, 30.0); c = o['contact']
    slide = []
    for i, sd in enumerate('lr'):
        v = np.minimum(*[np.linalg.norm(np.gradient(Jx[n + sd][:, :2], axis=0), axis=1) * 30 for n in ('foot_', 'ball_')])   # 支点(踝和前掌里慢的那个): 滚脚时踝本来就动
        slide += list(100 * v[c[:, i] & Bw])
    ka_ = knee_angle(Jx); dk = np.abs(np.diff(ka_, axis=0, prepend=ka_[:1])).max(1)
    kj = [float(dk[max(b - 9, 0):b + 10].max()) for b in bnd]                 # 交界前后 0.3 s 膝角一帧最大变化
    offs = sorted([(k, i) for i in (0, 1) for a, b in impl.runs_of(c[:, i]) for k in [b] if b < T and Wv[min(b, T - 1)]])
    same = sum(1 for (k0, i0), (k1, i1) in zip(offs[:-1], offs[1:]) if i0 == i1)
    err = np.linalg.norm(np.stack([Jx['foot_l'][:, :2] - Jr['foot_l'][:, :2], Jx['foot_r'][:, :2] - Jr['foot_r'][:, :2]], 1), axis=2).mean(1)
    return dict(win_min=float(Wv.sum() / 30 / 60), events=cnt, slide=slide, knee_bnd=kj, same_foot=same,
                err_vis=float(np.median(err[vis & ~Bw])) if (vis & ~Bw).any() else np.nan, err_hid=float(np.median(err[~vis])))


def work(args):
    name, idx = args
    import bench
    from mocap_stats import AMASS
    from qa_legs import leg_len, SMPLX, load_thr
    G = bench.G; impl = G['impl']
    if HARD.search(name.replace('__', '/')):
        return None
    J, fps = G['sk'].positions(f'{AMASS}/{name}')
    t0 = np.arange(len(J['pelvis'])) / fps; t = np.arange(0, t0[-1], 1 / 30)
    if t[-1] < 5.0:
        return None
    J = {n: np.stack([np.interp(t, t0, v[:, d]) for d in range(3)], 1) for n, v in J.items()}
    r = G['hip_h'] / leg_len(J); J = {n: v * r for n, v in J.items()}
    J = level_floor(J)
    if np.linalg.norm(np.diff(J['pelvis'][:, :2], axis=0), axis=1).sum() < 0.5:
        return None
    T = len(t); rng = np.random.default_rng(idx)
    a = int(rng.uniform(0.25, 0.45) * T); L = int(rng.uniform(0.25, 0.45) * T)
    vis = np.ones(T, bool); vis[a:a + L] = False
    s = bench.make_input(J, G['body'], G['betas'], False, 0)
    obs = obs_from_joints(J, G['body'], vis, impl)
    thr = load_thr()['thr']
    res = dict(name=name, group='像office' if OFFICE.search(name) else '正常走路', dur=float(t[-1]), hid=(a / 30, (a + L) / 30))
    keep = dict(t=t, vis=vis, **{'real_' + n: J[n].astype(np.float32) for n in SMPLX})
    res['real'] = measure(J, J, t, vis, thr, impl)
    for m in MODES:
        with contextlib.redirect_stdout(io.StringIO()):
            o = impl.run(s, t, G['model'], log=lambda *a, **k: None, obs=None if m == 'A' else obs, join='cut' if m == 'B' else 'plan')
        Jo = {n: o['joints'][:, j].astype(np.float64) for n, j in SMPLX.items()}
        res[m] = measure(Jo, J, t, vis, thr, impl)
        keep.update({f'{m}_{n}': Jo[n].astype(np.float32) for n in SMPLX})
    os.makedirs(f'{OUT}/clips', exist_ok=True)
    np.savez_compressed(f'{OUT}/clips/{idx:03d}.npz', name=name, **keep)
    return res


def summarize(res, label):
    from qa_legs import NAMES
    print(f'\n== {label}: {len(res)} 段, 评估窗口共 {sum(r["real"]["win_min"] for r in res):.1f} 分钟(藏掉的部分 + 交界前后 0.5 s), 交界 {sum(len(r["real"]["knee_bnd"]) for r in res)} 处')
    cols = ('real',) + MODES
    hdr = dict(real='真人', A='A整段补', B='B硬切', C='C接上')
    print(f'{"":<26}' + ''.join(f'{hdr[c]:>10}' for c in cols))
    out = {}
    mins = sum(r['real']['win_min'] for r in res)
    for k in NAMES:
        row = [sum(r[c]['events'][k] for r in res) / mins for c in cols]
        out[k] = dict(zip(cols, row))
        print(f'{NAMES[k] + " 次/分钟":<26}' + ''.join(f'{x:>10.1f}' for x in row))
    for lab, f in (('交界±0.5s 踩地脚滑 平均 cm/s', lambda c: np.mean(np.concatenate([r[c]['slide'] for r in res]))),
                   ('交界±0.5s 踩地脚滑 95% cm/s', lambda c: np.percentile(np.concatenate([r[c]['slide'] for r in res]), 95)),
                   ('交界±0.3s 膝角跳 平均 °/帧', lambda c: np.mean([x for r in res for x in r[c]['knee_bnd']])),
                   ('交界±0.3s 膝角跳>阈值 比例%', lambda c: 100 * np.mean([x > THR_KJ for r in res for x in r[c]['knee_bnd']])),
                   ('同一只脚连迈 次', lambda c: sum(r[c]['same_foot'] for r in res)),
                   ('看得见的帧 踝离真腿 cm', lambda c: 100 * np.nanmedian([r[c]['err_vis'] for r in res])),
                   ('藏掉的帧 踝离真腿 cm', lambda c: 100 * np.median([r[c]['err_hid'] for r in res]))):
        row = [float(f(c)) for c in cols]
        out[lab] = dict(zip(cols, row))
        print(f'{lab:<26}' + ''.join(f'{x:>10.1f}' for x in row))
    return out


def main():
    global THR_KJ
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=int(os.environ.get('BENCH_WORKERS', 48)))
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    import bench
    from mocap_stats import pick_clips
    from qa_legs import load_thr
    THR_KJ = load_thr()['thr']['knee_jump']
    names = pick_clips(a.limit)
    tic = time.time()
    with Pool(a.workers, initializer=bench.init) as pool:
        res = [r for r in pool.map(work, [(n, i) for i, n in enumerate(names)], chunksize=1) if r is not None]
    print(f'{len(res)} 段, 用时 {time.time() - tic:.0f} s; 膝角跳阈值 {THR_KJ:.1f}°/帧')
    S = dict(all=summarize(res, '全部'))
    for g in ('像office', '正常走路'):
        S[g] = summarize([r for r in res if r['group'] == g], g)
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(summary=S, per_clip=[{k: v for k, v in r.items()} for r in res]), open(f'{OUT}/summary.json', 'w'), ensure_ascii=False, indent=1,
              default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print('->', OUT)


THR_KJ = 0.0
if __name__ == '__main__':
    main()
