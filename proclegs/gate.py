"""一条命令考一个版本(rohm 环境): 动捕考场 + office 关卡, 最后给 通过/不通过。
  1. 动捕考场(bench.py), 分三组看问题次数/分钟:
     像 office  慢、倒着/横着、边走边看(名字含 side_step/backward/lookfor/look_around/look_at/thinking/call/stand)
     正常走路   其余不超纲的
  2. office 关卡(office_gate.py): 和 v10 比, 不许变差
通过条件: office 关卡通过; 像 office 组 每项不比 v10 多 10%(或 0.5 次/分钟)以上; 正常走路组 腿抖/膝盖跳变 不比 v22 多 10% 以上。
用法: python gate.py --tag v23 [--impl versions/proclegs_v10.py] [--skip-bench]
基准: outputs/bench/base_v10、base_v22(第一次自动跑)
"""
import argparse, json, os, re, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
HARD = re.compile(r'turn-05-turn|MartialArts|leap|jump|run', re.I)
OFFICE = re.compile(r'side_step|backward|lookfor|look_around|look_at|thinking|call|stand', re.I)
GROUPS = {'像office': lambda n: not HARD.search(n) and bool(OFFICE.search(n)),
          '正常走路': lambda n: not HARD.search(n) and not OFFICE.search(n)}
CATS = ['knee_jump', 'leg_jitter', 'body_jerk', 'pelvis_zrate', 'off_center', 'wide_stance']
SHORT = dict(knee_jump='膝盖跳变', leg_jitter='腿抖', body_jerk='整体顿挫', pelvis_zrate='骨盆猛升降', off_center='骨盆偏一侧', wide_stance='叉太开')


def bench(tag, impl=None):
    if not os.path.exists(f'{HERE}/outputs/bench/{tag}/summary.json'):
        cmd = [PY, f'{HERE}/bench.py', '--tag', tag] + (['--impl', impl] if impl else [])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, cwd=HERE)
    return json.load(open(f'{HERE}/outputs/bench/{tag}/summary.json'))


def group_rates(s, keep):
    per = [p for p in s['per_clip'] if keep(p['name'])]
    mins = sum(p['dur'] for p in per) / 60
    r = {side: {k: sum(sum(e[0] == k for e in p['ev_' + side]) for p in per) / mins for k in CATS} for side in ('real', 'ours')}
    r['steps'] = float(np.median([p['steps_ours'] / max(p['steps_real'], 1) for p in per]))
    r['sink'] = float(100 * np.median([p['z_sink'] for p in per]))
    return r, mins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--impl', default=None)
    ap.add_argument('--skip-bench', action='store_true', help='考场结果已有就不重跑')
    ap.add_argument('--opt', default=None, help='覆盖 proclegs 常量, JSON, 如 \'{"EDGE_GATE": 0.7}\'')
    a = ap.parse_args()
    if a.opt:
        os.environ['PROCLEGS_OPT'] = a.opt
        json.dump(json.loads(a.opt), open(f'{HERE}/outputs/opt_{a.tag}.json', 'w'))
    b10 = bench('base_v10', f'{HERE}/versions/proclegs_v10.py'); b22 = bench('base_v22', f'{HERE}/versions/proclegs_v22.py')
    out = f'{HERE}/outputs/bench/{a.tag}/summary.json'
    if os.path.exists(out) and not a.skip_bench:
        os.remove(out)
    me = bench(a.tag, a.impl)
    ok = True; res = dict(tag=a.tag, opt=a.opt)
    for g, keep in GROUPS.items():
        (r10, mins), (r22, _), (rme, _) = group_rates(b10, keep), group_rates(b22, keep), group_rates(me, keep)
        print(f'\n{g}({mins:.0f} 分钟) 次/分钟   真人   v10   v22  {a.tag}')
        for k in CATS:
            ref = r10 if g == '像office' else r22
            bad = rme['ours'][k] > max(ref['ours'][k] * 1.1, ref['ours'][k] + 0.5)
            if g == '正常走路' and k not in ('leg_jitter', 'knee_jump'):
                bad = False
            ok &= not bad
            print(f'  {SHORT[k]:<10}{rme["real"][k]:>6.1f}{r10["ours"][k]:>6.1f}{r22["ours"][k]:>6.1f}{rme["ours"][k]:>6.1f}  {"变差" if bad else ""}')
        res[g] = dict(ours=rme['ours'], steps=rme['steps'], sink=rme['sink'])
        print(f'  {"步数/真人":<10}{1:>6.2f}{r10["steps"]:>6.2f}{r22["steps"]:>6.2f}{rme["steps"]:>6.2f}')
        print(f'  {"骨盆多沉cm":<10}{"":>6}{r10["sink"]:>6.1f}{r22["sink"]:>6.1f}{rme["sink"]:>6.1f}')
    p = subprocess.run([PY, f'{HERE}/office_gate.py', '--tag', '_' + a.tag] + (['--impl', a.impl] if a.impl else []),
                       capture_output=True, text=True, cwd=HERE)
    print('\n' + p.stdout.strip())
    ok &= '不通过' not in p.stdout
    res['office'] = json.load(open(f'{HERE}/outputs/office_gate_{a.tag}.json')); res['pass'] = bool(ok)
    json.dump(res, open(f'{HERE}/outputs/gate_{a.tag}.json', 'w'), indent=1, ensure_ascii=False)
    print(f'\n== {a.tag}: {"通过" if ok else "不通过"} ==')


if __name__ == '__main__':
    main()
