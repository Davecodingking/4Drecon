"""body_motion.focal_fix_cut: 切点后开头几帧沿用了上个镜头焦距的, 换到本镜头自己的焦距;
换完用新焦距投回去的根像素和模型自己的一样; 其余帧(上个镜头、本镜头后面的、整镜头焦距都一样的)一个数都不动。
    python tools/tests/test_focal_fix_cut.py"""
import sys
import numpy as np
sys.path.insert(0, "/home/ubuntu/sam3d/pipeline")
import body_motion as BM


def proj(X, f):
    return X[:2] / X[2] * f


rng = np.random.default_rng(0)
# 镜头 0: 0~4 帧 3296;  镜头 1: 5~7 帧沿用 3296, 8~12 帧自己的 4794;  镜头 2: 13~16 帧和镜头 1 最后一帧一样(没换焦距的切点)
focal = {**{b: 3296.5 for b in range(0, 8)}, **{b: 4794.0 + b for b in range(8, 13)}, **{b: 4806.0 for b in range(13, 17)}}
focal[12] = 4806.0
shot = {**{b: 0 for b in range(0, 5)}, **{b: 1 for b in range(5, 13)}, **{b: 2 for b in range(13, 17)}}
rt = {b: np.array([rng.normal(0, 0.3), rng.normal(0, 0.3), 2.5 + 0.1 * b]) for b in range(17)}
rt2, fo2, k_, nfix = BM.focal_fix_cut(rt, focal, shot, vf={b: 100 + b for b in range(17)})

assert nfix == [105, 106, 107], nfix
f_own = float(np.median([focal[b] for b in range(8, 13)]))
for b in (5, 6, 7):
    assert abs(k_[b] - f_own / 3296.5) < 1e-12 and fo2[b] == f_own
    assert np.abs(proj(rt[b], focal[b]) - proj(rt2[b], fo2[b])).max() < 1e-9      # 根投影不变
    assert abs(rt2[b][2] / rt[b][2] - f_own / 3296.5) < 1e-12 and np.allclose(rt2[b][:2], rt[b][:2])
for b in list(range(0, 5)) + list(range(8, 17)):
    assert np.array_equal(rt2[b], rt[b]) and fo2[b] == focal[b] and b not in k_, b
# 整个镜头都和上个镜头焦距一样(没有"自己的"): 不动
rt3, fo3, k3, n3 = BM.focal_fix_cut(rt, {b: 3296.5 for b in range(17)}, shot)
assert n3 == [] and all(np.array_equal(rt3[b], rt[b]) for b in rt)
# 差不到 tol 的: 不动
rt4, fo4, k4, n4 = BM.focal_fix_cut(rt, {**{b: 4700.0 for b in range(0, 8)}, **{b: focal[b] for b in range(8, 17)}}, shot)   # 沿用的 4700 和本镜头 ~4804 只差 2%
assert n4 == [], n4
print(f"OK  换了 {nfix}, 深度 × {k_[5]:.3f}; 其余帧不动; 整镜头同焦距 / 差不到 5% 不换")
