#!/usr/bin/env python
"""算尺度时谁能当尺子(body_into_scene.ruler_people): 离全体中位超过 RATIO_TOL 的人踢掉。
起因(office r12, 2026-10-09): 搭在椅背上的一件外套被当成人 7, 它的比例 0.466 混在 8 个人里, 中位从 0.978 拉到 0.945,
整个场景多放大 3.5%。它在第 5 段闸口上被点掉, 但尺度在第 3 段已经乘进深度, 不重算。
跑: $PY tools/tests/test_people_ruler.py"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_into_scene as B

OFFICE_R12 = {"1": 0.890, "2": 1.023, "4": 0.978, "5": 0.912, "7": 0.466, "8": 1.116, "9": 0.863, "10": 1.054}


def test_office_jacket_is_dropped_from_ruler():
    ok, out = B.ruler_people(OFFICE_R12)
    assert set(out) == {"7"}, out
    assert set(ok) == set(OFFICE_R12) - {"7"}
    assert "0.94" in out["7"] and "25%" in out["7"], out["7"]          # 原因里写明全体中位和门槛


def test_same_rule_as_placement():
    """和摆人那步判"自己量的深度离群"是同一条线(RATIO_TOL): 刚好压线的留下, 过线的踢掉。"""
    med = 1.0                                                           # 三个 1.0 把中位钉死在 1.0
    per = {"a": med, "b": med, "e": med, "c": med * (1 + B.RATIO_TOL), "d": med * (1 + B.RATIO_TOL) + 0.01}
    ok, out = B.ruler_people(per)
    assert set(ok) == {"a", "b", "e", "c"} and set(out) == {"d"}, (ok, out)


def test_two_people_disagreeing_keeps_both():
    """两个人各执一词: 中位在中间, 两个都"离群" —— 分不出谁对, 一个都不踢。"""
    ok, out = B.ruler_people({"1": 0.5, "2": 1.0})
    assert ok == {"1": 0.5, "2": 1.0} and out == {}


def test_single_person_and_empty():
    assert B.ruler_people({"1": 0.7}) == ({"1": 0.7}, {})
    assert B.ruler_people({}) == ({}, {})


def test_input_not_mutated():
    per = dict(OFFICE_R12)
    B.ruler_people(per)
    assert per == OFFICE_R12


if __name__ == "__main__":
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f(); print(f"  ok  {k}")
    print("== test_people_ruler 全过 ==")
