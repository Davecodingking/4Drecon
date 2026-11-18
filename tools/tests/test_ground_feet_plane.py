#!/usr/bin/env python
"""ground_from_body.fit_feet_plane: 看不到地板时拿人的脚底点定地面(方向 + 高度)。
goodfellas 2026-11-09: 相机边走边拍又没端平, "相机的上"歪 6°, 水平面让起点附近的人被抬高 0.3 m。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../pipeline"))
from ground_from_body import fit_feet_plane, TILT_MAX_DEG  # noqa: E402

rng = np.random.default_rng(0)
UP = np.array([0.0, 1.0, 0.0])


def feet(n, sx, sz, slope_x=0.0, noise=0.03, h0=-1.2):
    x = rng.uniform(-sx, sx, n); z = rng.uniform(-sz, sz, n)
    return np.c_[x, h0 + slope_x * x + rng.normal(0, noise, n), z]


# 1) 地面是平的、人走得开: 不该斜
t = fit_feet_plane(feet(150, 2.0, 1.5), UP)
assert t["tilt_deg"] < 1.0, t
assert abs(t["h"] - (-1.2)) < 0.03, t

# 2) 只沿一条走廊走(横向几乎没铺开)、沿走的方向斜 6°: 只放开一个方向, 斜对、横向不乱歪
P = feet(150, 1.8, 0.1, slope_x=np.tan(np.radians(6)))
t = fit_feet_plane(P, UP)
assert t["放开斜的方向"] == 1, t
assert abs(t["tilt_deg"] - 6.0) < 0.6, t
assert abs(t["up"][2]) < 0.01, t           # 横向(z)没被带歪
assert t["mad"] < 0.04, t
r = (P @ t["up"]) - t["h"]                 # 脚底点按新的天量, 都该落在地面上
assert np.median(np.abs(r)) < 0.04, np.median(np.abs(r))

# 3) 两成离群(没站在地上 / 量错的帧, 高出 0.5 m): 不该被带偏
P = np.r_[feet(120, 1.8, 0.1, slope_x=np.tan(np.radians(6))), feet(30, 1.8, 0.1, slope_x=0.0, h0=-0.7)]
t = fit_feet_plane(P, UP)
assert abs(t["tilt_deg"] - 6.0) < 1.0, t

# 4) 相机不动、脚点挤在一处(squat): 两个方向都不放开, 等于水平面
t = fit_feet_plane(feet(60, 0.3, 0.3, slope_x=0.2), UP)
assert t["放开斜的方向"] == 0 and t["tilt_deg"] < 1e-6, t

# 5) 点太少不拟合
assert fit_feet_plane(feet(5, 2, 2), UP) is None
assert TILT_MAX_DEG >= 6.0
print("test_ground_feet_plane OK")
