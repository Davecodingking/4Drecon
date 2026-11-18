"""汇总 mocap_stats.py 的结果: 往前/横着/倒着 × 速度档(速度按身高归一: 身高/秒)。用法: python mocap_summary.py"""
import numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(f'{HERE}/outputs/mocap_steps.npz')
cat = np.where(d['ang'] < 45, '往前', np.where(d['ang'] > 135, '倒着', '横着'))
v = d['v']; cyc = d['Ts'] + d['stance']
bins = [(0.0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 2)]
print('速度、步长都按身高归一(身高 1.67 m 的人: 速度 0.22 = 0.36 m/s)。步长 = 一只脚一次挪的距离(=步态里的 stride)')
print('%-4s %-11s %4s | %-15s %-13s %-9s %-9s | %-22s %-13s | %-14s %-7s' % ('方向', '速度档', '步数', '一次挪 中位(25-75%)', '迈步时长 s', '迈步占比', '步频/分', '落脚 跟/尖/平 %', '落地俯仰°', '抬踝 cm(1.67m)', '最高点'))
for c in ['往前', '横着', '倒着']:
    for lo, hi in bins:
        m = (cat == c) & (v >= lo) & (v < hi)
        if m.sum() < 8:
            continue
        L = d['land'][m]
        print('%-4s %.2f–%.2f /s %5d | %.2f (%.2f–%.2f)    %.2f (%.2f–%.2f) %.2f      %4.0f     | %3.0f / %3.0f / %3.0f          %+5.1f        | %4.1f          %.2f' % (
            c, lo, hi, m.sum(), np.median(d['dist'][m]), *np.percentile(d['dist'][m], [25, 75]),
            np.median(d['Ts'][m]), *np.percentile(d['Ts'][m], [25, 75]), np.median(d['Ts'][m] / cyc[m]), 120 / np.median(cyc[m]),
            100 * np.mean(L == 'heel'), 100 * np.mean(L == 'toe'), 100 * np.mean(L == 'flat'), np.median(d['pitch_land'][m]),
            100 * 1.67 * np.median(d['lift'][m]) / 1.0 / 1.0 * 1.0 / 1.67 * 1.67 / 1.67, np.median(d['lift_t'][m])))
# 骨盆: 单脚着地中点 vs 两脚着地
o = {k[4:]: d[k] for k in d.files if k.startswith('osc_')}
print('\n骨盆(去掉 1 s 慢变化后): 单脚着地中点 vs 两脚着地中点; 左右偏 >0 = 偏向踩地的脚; 倾斜 >0 = 踩地那边髋高(迈步那边掉)')
for lo, hi in bins:
    s1 = (o['foot'] >= 0) & (o['v'] >= lo) & (o['v'] < hi); s2 = (o['foot'] < 0) & (o['v'] >= lo) & (o['v'] < hi)
    if s1.sum() < 8 or s2.sum() < 4:
        continue
    print('速度 %.2f–%.2f: 高度 单脚 %+.1f cm / 两脚 %+.1f cm (差 %.1f cm, 按 1.67 m 身高); 左右偏 %+.1f cm; 倾斜 %+.1f°; 单脚段 %.2f s' % (
        lo, hi, 100 * np.median(o['z_mid'][s1]), 100 * np.median(o['z_mid'][s2]), 100 * (np.median(o['z_mid'][s1]) - np.median(o['z_mid'][s2])),
        100 * np.median(o['lat_mid'][s1]), np.median(o['obl_mid'][s1]), np.median(o['dur'][s1])))
