"""按考场结果自动校准步长/迈步时长表(rohm 环境)。
读 outputs/bench/<tag>/summary.json 里 正常走路 的步(真人、补腿用同一个检测方法量的), 每个 方向×速度档:
  规划步长 *= (真人/补腿)^0.5, 规划迈步时长 += 0.5 * (真人 - 补腿)  (只调样本 >= 20 步的档, 每次最多改 25%/0.08 s)
写 gait_tables.json(proclegs.py 启动时读)。用法: python calib_gait.py --tag v11
"""
import argparse, json, os, re
import numpy as np
import proclegs as P

HERE = os.path.dirname(os.path.abspath(__file__))
HARD = re.compile(r'turn-05-turn|MartialArts|leap|jump|run', re.I)
EDGES = [0.0, 0.15, 0.3, 0.45, 0.6, 0.9]
CAL_MAX = 1.25                 # 校准后最多是真人量出来的这么多倍


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    s = json.load(open(f'{HERE}/outputs/bench/{a.tag}/summary.json'))
    pool = lambda side: np.array([g for p in s['per_clip'] if not HARD.search(p['name']) for g in p['gait_' + side]]).reshape(-1, 6)
    R, O = pool('r'), pool('o')
    cat = lambda X: np.where(X[:, 3] < 45, 0, np.where(X[:, 3] > 135, 2, 1))
    cr, co = cat(R), cat(O)
    stride = [list(x) for x in P.STRIDE_TAB]; ts = [list(x) for x in P.TS_TAB]
    if not os.path.exists(P.GAIT_JSON):                      # 第一次: 从"量出来的"表 + 原来的换算开始
        stride = [[v * P.K_STRIDE for v in x] for x in stride]; ts = [[v + P.TS_ADD for v in x] for x in ts]
    names = ['往前', '横着', '倒着']
    for c in range(3):
        for b in range(len(EDGES) - 1):
            mr = (cr == c) & (R[:, 0] >= EDGES[b]) & (R[:, 0] < EDGES[b + 1]); mo = (co == c) & (O[:, 0] >= EDGES[b]) & (O[:, 0] < EDGES[b + 1])
            if mr.sum() < 20 or mo.sum() < 20:
                continue
            fs = float(np.clip(np.sqrt(np.median(R[mr, 1]) / np.median(O[mo, 1])), 0.75, 1.25))
            dt = float(np.clip(0.5 * (np.median(R[mr, 2]) - np.median(O[mo, 2])), -0.08, 0.08))
            print(f'{names[c]} {EDGES[b]:.2f}-{EDGES[b + 1]:.2f}: 挪 真人 {np.median(R[mr, 1]):.2f} 补腿 {np.median(O[mo, 1]):.2f} -> 规划步长 x{fs:.2f}; '
                  f'时长 真人 {np.median(R[mr, 2]):.2f} 补腿 {np.median(O[mo, 2]):.2f} -> {dt:+.2f} s')
            stride[c][b] *= fs                               # 迈步时长不校准(proclegs 里按 迈步占比 x 周期 算)
            # 防"越调越大": 规划步长迈不出来时(被腿长/其他规则卡住), 量出来的永远偏小, 校准会一直往上加。
            # v22 慢速档被调到真人的 2.7 倍(往前 0.36 -> 0.97 身高), 正常抬脚规则几乎不触发, 脚总被拖到够不着才迈
            stride[c][b] = float(np.clip(stride[c][b], 0.8 * P.STRIDE_MEAS[c][b], CAL_MAX * P.STRIDE_MEAS[c][b]))
    json.dump(dict(stride=stride, ts=ts, from_bench=a.tag), open(P.GAIT_JSON, 'w'), indent=1)
    print('->', P.GAIT_JSON)


if __name__ == '__main__':
    main()
