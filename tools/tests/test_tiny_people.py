#!/usr/bin/env python
"""pipeline/body_into_scene.tiny_people: 核对完"几帧、很小"的人是误检(joker r2 人 2 = 墙头的植物, 4 帧, 掩码占画面 0.32%)。
真人里帧少的掩码都不小(nfl1 人 8 片尾 3 帧, 0.67%), 掩码小的都出现很多帧(pedestrian 远处的人 0.1~0.5%, 90 帧以上)。
跑: $PY tools/tests/test_tiny_people.py"""
import os, sys, tempfile
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_into_scene as B

H, W = 100, 200                 # 2 万像素: 0.32% = 64 像素, 0.67% = 134 像素


def _run(T, boxes):
    """boxes: {帧: {编号: 像素数}} -> <T>/masks/*.png(像素值 = 编号)"""
    os.makedirs(f"{T}/masks")
    for b, d in boxes.items():
        m = np.zeros((H, W), np.uint8); x = 0
        for rid, n in d.items():
            m.flat[x:x + n] = int(rid); x += n
        Image.fromarray(m).save(f"{T}/masks/{b:08d}.png")


def test_a_few_tiny_frames_are_not_a_person_but_short_or_small_real_people_stay():
    kept = list(range(0, 600, 6)); fps = 30.0            # 每帧 0.2 秒
    plant = {b: {"2": 64} for b in range(68, 72)}          # 4 帧 0.32%: 墙头的植物
    late = {b: {"8": 134} for b in range(90, 93)}          # 3 帧 0.67%: 片尾才进画面的真人
    far = {b: {"4": 30} for b in range(0, 40)}             # 40 帧 0.15%: 远处的小人
    boxes = {}
    for d in (plant, late, far):
        for b, v in d.items():
            boxes.setdefault(b, {}).update(v)
    who = {"2": [("2", b) for b in plant], "8": [("8", b) for b in late], "4": [("4", b) for b in far]}
    with tempfile.TemporaryDirectory() as T:
        _run(T, boxes)
        out = B.tiny_people(T, who, kept, fps)
    assert set(out) == {"2"}, out
    assert out["2"]["帧"] == 4 and abs(out["2"]["掩码面积中位_pct"] - 0.32) < 0.01, out


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f(); print("ok", name)
