"""补腿自动体检(rohm 环境): 一条命令扫完整段, 列出所有超出真人范围的时刻和原因, 不用人盯着画面找。
每一项是一条逐帧的"严重程度"曲线; 阈值 = 真人动捕(AMASS 走路, bench.py 里同一套算法)的 99.5% 分位,
超过就算一次问题(相邻几帧合成一次)。真人自己按定义每 200 帧也会有 ~1 帧超, 所以要和真人的"次/分钟"比。

项目(都按腿长缩放到 SMPL-X 这个人的腿长再比):
  膝盖跳变   knee_jump   膝角一帧变化 °
  腿抖       leg_jitter  膝/踝位置去掉 0.07 s 平滑后剩下的 mm
  整体顿挫   body_jerk   骨盆位置去掉 0.07 s 平滑后剩下的 mm(整个人一跳)
  骨盆猛升降 pelvis_zrate 骨盆高度变化速度 cm/s
  骨盆偏一侧 off_center  两脚着地时骨盆离两脚中点(左右方向, 0 正中, 1 在一只脚上, >1 两脚在骨盆同一侧)
  叉太开     wide_stance 两脚着地时两踝距离 / 腿长
用法:
  python qa_legs.py --pid 4 --tag _v10      # 体检 office 结果(阈值文件由 bench.py 生成)
"""
import argparse, json, os, warnings
import numpy as np
from scipy.ndimage import gaussian_filter1d
from mocap_stats import analyze

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
THR_FILE = f'{HERE}/outputs/bench/qa_thresholds.json'
SMPLX = dict(pelvis=0, thigh_l=1, thigh_r=2, calf_l=4, calf_r=5, foot_l=7, foot_r=8, ball_l=10, ball_r=11, head=15)
NAMES = dict(knee_jump='膝盖跳变 °/帧', leg_jitter='腿抖 mm', body_jerk='整体顿挫 mm', pelvis_zrate='骨盆猛升降 cm/s',
             off_center='骨盆偏一侧(两脚着地)', wide_stance='叉太开(两脚着地, /腿长)')
REF_LEG = 0.84                       # 阈值按这个腿长(office p4 的 SMPL-X 髋高)


def leg_len(J):
    """腿长 = 髋高(两髋中点离地, 90% 分位; 地面 z=0)。按关节算的 大腿+小腿 在不同骨架间不可比(动捕的踝关节离地 12 cm, SMPL-X 7 cm)。"""
    return float(np.percentile((J['thigh_l'][:, 2] + J['thigh_r'][:, 2]) / 2, 90))


def signals(J, fps=30.0):
    """J: 关节 dict(米, z 朝上, 30 fps)。返回 {项目: 逐帧值}, 已按腿长缩放到 REF_LEG; 以及两脚着地。"""
    sc = REF_LEG / leg_len(J)
    hf = lambda x: 1000 * sc * np.linalg.norm(x - gaussian_filter1d(x, 2, axis=0), axis=-1)
    out = {}
    kj = []
    for s in 'lr':
        a = J['thigh_' + s] - J['calf_' + s]; b = J['foot_' + s] - J['calf_' + s]
        ang = np.degrees(np.arccos(np.clip((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1)))
        kj.append(np.abs(np.diff(ang, prepend=ang[0])))
    out['knee_jump'] = np.max(kj, 0)
    out['leg_jitter'] = np.max([hf(J[n]) for n in ('calf_l', 'calf_r', 'foot_l', 'foot_r')], 0)
    out['body_jerk'] = hf(J['pelvis'])
    z = gaussian_filter1d(J['pelvis'][:, 2], 1.0)
    out['pelvis_zrate'] = 100 * sc * np.abs(np.gradient(z)) * fps
    _, o = analyze(J, fps); c = o['contact'].all(1)
    l = J['thigh_l'] - J['thigh_r']; l[:, 2] = 0; l /= np.linalg.norm(l, axis=1, keepdims=True)
    p = (J['pelvis'][:, :2] * l[:, :2]).sum(1)
    aL = (J['foot_l'][:, :2] * l[:, :2]).sum(1) - p; aR = (J['foot_r'][:, :2] * l[:, :2]).sum(1) - p
    out['off_center'] = np.where(c, np.abs((aL + aR) / 2) / np.maximum(np.abs(aL - aR) / 2, 0.03), 0.0)
    out['wide_stance'] = np.where(c, np.linalg.norm(J['foot_l'][:, :2] - J['foot_r'][:, :2], axis=1) * sc / REF_LEG, 0.0)
    return out


def events(sig, thr, t, gap=3):
    """超阈值的帧 -> 事件列表 [(项目, 开始 s, 结束 s, 峰值, 峰值时刻 s)]; 相隔 <= gap 帧的合并。"""
    ev = []
    for k, x in sig.items():
        bad = np.flatnonzero(x > thr[k])
        if not len(bad):
            continue
        groups = np.split(bad, np.flatnonzero(np.diff(bad) > gap) + 1)
        for g in groups:
            j = g[np.argmax(x[g])]
            ev.append((k, float(t[g[0]]), float(t[g[-1]]), float(x[j]), float(t[j])))
    return sorted(ev, key=lambda e: e[1])


def load_thr():
    return json.load(open(THR_FILE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, default=4)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    thr = load_thr()
    o = np.load(f'{HERE}/outputs/p{a.pid}_proclegs{a.tag}_W.npz')
    J = {n: o['joints'][:, j].astype(np.float64) for n, j in SMPLX.items()}
    t = o['time_s']
    sig = signals(J)
    ev = events(sig, thr['thr'], t)
    mins = len(t) / 30 / 60
    print(f'体检 p{a.pid}{a.tag}: {len(t) / 30:.1f} s, 阈值 = 真人 {thr["pct"]}% 分位(真人自己约 {thr["real_rate_total"]:.1f} 次/分钟)')
    print(f'{"项目":<22}{"阈值":>8}{"次数":>6}{"次/分钟":>9}{"真人 次/分钟":>13}')
    for k in NAMES:
        n = sum(e[0] == k for e in ev)
        print(f'{NAMES[k]:<22}{thr["thr"][k]:>8.2f}{n:>6}{n / mins:>9.1f}{thr["real_rate"][k]:>13.1f}')
    print('问题时刻(峰值):')
    for k, t0, t1, pk, tp in ev:
        print(f'  {t0:6.2f}–{t1:6.2f} s  {NAMES[k]:<20} 峰值 {pk:.2f} @ {tp:.2f} s')


if __name__ == '__main__':
    main()
