"""office 关卡(rohm 环境): 用某个版本的补腿程序跑 office p4, 算一组分数, 和 v10(当前在用)比, 有一项明显变差就不通过。
分数(都是越小越好):
  qa_<项目>      qa_legs 体检的问题次数(膝盖跳变、腿抖、整体顿挫、骨盆猛升降、骨盆偏一侧、叉太开)
  sink_cm       骨盆比(原版高度, 站直)取低的 再被压低多少, 最大
  toe_in_L/R    脚尖内八 > 10° 的帧 %
  same_foot     同一只脚连迈两次的次数
  miss_cm       腿够不着 最大
  knee_v99      膝角变化速度 99% 分位 °/s
用法:
  python office_gate.py --impl versions/proclegs_v10.py --tag _v10 --save-base   # 存基准
  python office_gate.py --tag _v23                                               # 考当前 proclegs.py
"""
import argparse, contextlib, io, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = f'{HERE}/outputs/office_gate_base.json'
# 允许的误差: 次数多 1 次以内、厘米多 0.5、百分比多 2 个点、°/s 多 5% 算没变差
TOL = dict(qa=1, sink_cm=0.5, toe_in_L=2.0, toe_in_R=2.0, same_foot=0, miss_cm=0.3, knee_v99=0.05)


def load_impl(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('impl', path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def scores(o):
    from qa_legs import signals, events, load_thr, SMPLX, NAMES
    t = o['time_s']
    J = {n: o['joints'][:, j].astype(np.float64) for n, j in SMPLX.items()}
    ev = events(signals(J), load_thr()['thr'], t)
    sc = {f'qa_{k}': sum(e[0] == k for e in ev) for k in NAMES}
    ref = np.minimum(o['pelvis_z_in'], o['pelvis_z_out'].max())
    sink = ref - o['pelvis_z_out']
    sc['sink_cm'] = float(100 * sink.max()); sc['sink_t'] = float(t[sink.argmax()])
    f = Rot.from_rotvec(o['global_orient']).as_matrix() @ np.array([0, 0, 1.0])
    yaw = np.arctan2(f[:, 1], f[:, 0])
    out = np.degrees((o['foot_yaw'] - yaw[:, None] + np.pi) % (2 * np.pi) - np.pi) * np.array([1, -1])   # > 0 外八
    sc['toe_in_L'], sc['toe_in_R'] = float(100 * np.mean(out[:, 0] < -10)), float(100 * np.mean(out[:, 1] < -10))
    st = o['steps']
    sc['same_foot'] = int(np.sum(st[1:, 0] == st[:-1, 0])); sc['steps'] = len(st)
    sc['miss_cm'] = float(100 * o['ik_miss'].max())
    kv = []
    for h, k, a in ((1, 4, 7), (2, 5, 8)):
        A = o['joints'][:, h] - o['joints'][:, k]; B = o['joints'][:, a] - o['joints'][:, k]
        ang = np.degrees(np.arccos(np.clip((A * B).sum(1) / np.linalg.norm(A, axis=1) / np.linalg.norm(B, axis=1), -1, 1)))
        kv.append(np.abs(np.diff(ang)) * 30)
    sc['knee_v99'] = float(np.percentile(np.concatenate(kv), 99))
    sc['events'] = [(e[0], round(e[4], 2), round(e[3], 2)) for e in ev]
    return sc


def compare(sc, base):
    """-> [(项目, 这版, 基准, 是否变差)]"""
    rows = []
    for k, b in base.items():
        if k in ('events', 'sink_t', 'steps'):
            continue
        tol = TOL['qa'] if k.startswith('qa_') else TOL[k]
        worse = sc[k] > (b * (1 + tol) if k == 'knee_v99' else b + tol)
        rows.append((k, sc[k], b, worse))
    return rows


def run_office(impl_path, tag):
    impl = load_impl(impl_path)
    d = np.load('/home/ubuntu/sam3d/RoHM/ours/inputs/p4_rohm_input.npz')
    with contextlib.redirect_stdout(io.StringIO()):
        o = impl.run(d['smplx_world'].astype(np.float64), d['time_s'], log=lambda *a, **k: None)
    np.savez(f'{HERE}/outputs/p4_proclegs{tag}_W.npz', **o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', default=f'{HERE}/proclegs.py')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--save-base', action='store_true')
    a = ap.parse_args()
    sc = scores(run_office(a.impl, a.tag))
    if a.save_base:
        json.dump(sc, open(BASE, 'w'), indent=1, ensure_ascii=False); print('基准 ->', BASE)
    base = json.load(open(BASE))
    rows = compare(sc, base)
    print(f'office 关卡 {a.tag}: {"通过" if not any(r[3] for r in rows) else "不通过"}  (骨盆最沉在 {sc["sink_t"]:.1f} s, 步数 {sc["steps"]})')
    for k, x, b, w in rows:
        print(f'  {k:<16}{x:>8.1f}  v10 {b:>6.1f}  {"变差" if w else ""}')
    json.dump(sc, open(f'{HERE}/outputs/office_gate{a.tag}.json', 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
