"""比较几次 bench.py 的结果(按 正常走路 / 超纲动作 分组)。用法: python bench_report.py v10 v10_degrade [v11 ...]"""
import json, os, re, sys
import numpy as np
from qa_legs import NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
HARD = re.compile(r'turn-05-turn|MartialArts|leap|jump|run', re.I)      # 超纲: 原地转圈、武术快冲快退、跳


def rates(per, side):
    mins = sum(p['dur'] for p in per) / 60
    return {k: sum(sum(e[0] == k for e in p['ev_' + side]) for p in per) / mins for k in NAMES}, mins


def main():
    runs = [(tag, json.load(open(f'{HERE}/outputs/bench/{tag}/summary.json'))) for tag in sys.argv[1:]]
    for grp, keep in [('正常走路', lambda n: not HARD.search(n)), ('超纲动作(转圈/武术/跳)', lambda n: bool(HARD.search(n)))]:
        cols = []
        for tag, s in runs:
            per = [p for p in s['per_clip'] if keep(p['name'])]
            ro, mins = rates(per, 'ours'); rr, _ = rates(per, 'real')
            sr = np.median([p['steps_ours'] / max(p['steps_real'], 1) for p in per])
            zs = [p['z_sink'] for p in per]
            cols.append((tag, ro, rr, sr, np.median(zs), np.percentile(zs, 90), len(per), mins))
        print(f'\n== {grp}: {cols[0][6]} 段, {cols[0][7]:.1f} 分钟; 次/分钟(越接近真人越好) ==')
        print(f'{"项目":<24}{"真人":>8}' + ''.join(f'{c[0]:>16}' for c in cols))
        for k in NAMES:
            print(f'{NAMES[k]:<24}{cols[0][2][k]:>8.1f}' + ''.join(f'{c[1][k]:>16.1f}' for c in cols))
        print(f'{"步数 补腿/真人(中位)":<24}{1:>8.2f}' + ''.join(f'{c[3]:>16.2f}' for c in cols))
        print(f'{"骨盆多压低 每段最大 中位/90% cm":<24}{"":>8}' + ''.join(f'{100 * c[4]:>9.1f}/{100 * c[5]:<6.1f}' for c in cols))
        if grp == '正常走路':
            gait_table(runs, keep)


def gait_table(runs, keep):
    """按速度档(身高/s)和方向: 一次挪(身高)、迈步时长(s)、抬踝(cm), 真人 vs 各版本(同一个检测方法量)。"""
    bins = [(0.0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.9)]
    cat = lambda a: '往前' if a < 45 else ('倒着' if a > 135 else '横着')
    def pool(s, side):
        return np.array([g for p in s['per_clip'] if keep(p['name']) for g in p['gait_' + side]]).reshape(-1, 6)
    R = pool(runs[0][1], 'r')
    print(f'\n{"方向 速度档":<14}{"真人 步数":>9}{"真人 挪/时长":>16}' + ''.join(f'{tag + " 步数":>12}{"挪/时长":>14}' for tag, _ in runs))
    for c in ['往前', '横着', '倒着']:
        for lo, hi in bins:
            mr = np.array([cat(a) == c for a in R[:, 3]]) & (R[:, 0] >= lo) & (R[:, 0] < hi)
            if mr.sum() < 10:
                continue
            row = f'{c} {lo:.2f}-{hi:.2f}  {mr.sum():>8} {np.median(R[mr, 1]):>8.2f}/{np.median(R[mr, 2]):.2f}s'
            for tag, s in runs:
                O = pool(s, 'o'); mo = np.array([cat(a) == c for a in O[:, 3]]) & (O[:, 0] >= lo) & (O[:, 0] < hi)
                row += f'{mo.sum():>12}' + (f'{np.median(O[mo, 1]):>8.2f}/{np.median(O[mo, 2]):.2f}s' if mo.sum() else f'{"—":>14}')
            print(row)


if __name__ == '__main__':
    main()
