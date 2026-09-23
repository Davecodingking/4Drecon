#!/usr/bin/env python
"""闸口的提亮和整帧定位图: 纯函数, 合成图, 不碰盘上的轮次。跑: $PY tools/tests/test_gate_locate.py"""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gate_ui as G

def dark_scene():
    img = np.full((360, 640, 3), 30, np.uint8)      # 暗背景
    img[100:200, 300:400] = 45                      # 更暗一点的物体
    m = np.zeros((360, 640), np.uint8); m[100:200, 300:400] = 255
    return img, m

def test_gamma_only_brightens_dark():
    img, m = dark_scene()
    g = G._gamma(img, m)
    assert 0.4 <= g < 1.0, g
    bright = np.full_like(img, 180)
    assert G._gamma(bright, m) == 1.0
    out = G._apply_gamma(img, g)
    med = int(np.median(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[m > 127]))
    assert 118 <= med <= 138, med                    # 中位数被拉到 128 上下

def test_cutout_and_locate_share_gamma():
    img, m = dark_scene()
    g = G._gamma(img, m)
    want = int(G._apply_gamma(np.full((1, 1, 3), 45, np.uint8), g)[0, 0, 0])
    c = G.cutout_rgb(img, m, 120)
    l = G.locate_rgb(img, m, 420)
    assert c.shape == (120, 120, 3)
    assert l.shape[1] == 420 and l.shape[0] == int(360 * 420 / 640), l.shape
    assert abs(int(c[60, 60, 0]) - want) <= 2, (c[60, 60, 0], want)          # 特写中心 = 物体
    assert abs(int(l[98, 229, 0]) - want) <= 2, (l[98, 229, 0], want)        # 定位图里物体中心, 同样亮
    bg = int(G._apply_gamma(np.full((1, 1, 3), 30, np.uint8), g)[0, 0, 0])
    assert abs(int(l[10, 10, 0]) - int(bg * 0.45)) <= 2, (l[10, 10, 0], bg)  # mask 外压暗
    red = (l[:, :, 2] >= 200) & (l[:, :, 1] < 60) & (l[:, :, 0] < 60)
    assert red.sum() > 100, red.sum()                                        # 有红框
    assert G.locate_rgb(img, np.zeros_like(m), 420) is None                  # 空 mask

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f(); print("ok", k)
