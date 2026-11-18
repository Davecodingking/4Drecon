#!/usr/bin/env python
"""本轮页"会生成几个 mesh"必须和逐实例生成那步(gen_per_instance.py)真正生成的一致。

★ 11-10 hitch_cuts_indoor r1: 本轮页只减了整词作废, 说会生成 28 个; 生成那步还挡了
  VLM 否决 6 个、太矮(<0.15 m) 3 个, 实际 19 个。候选页(gate_ui)说 19, 两个页面对不上。
  现在两边都调 data/outdoor/scripts/solo_plan.py, 规则只有一份。
要 hitch_cuts_indoor/r1 在盘上(停在闸 3 那一版); 不在就跳过。
"""
import os, sys, json, subprocess
SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/tools")
import run_ctl

R1 = f"{SAM3D}/data/indoorOutput/hitch_cuts_indoor/r1"


def test_hitch_run_page_matches_generation():
    if not os.path.exists(f"{R1}/06_all_boxes.json"):
        print("[跳过] hitch_cuts_indoor/r1 不在盘上"); return
    c = run_ctl._gate_extra("check_picks", "indoor", "hitch_cuts_indoor", "r1")["候选决定"]
    # 19 是 11-10 按 gen_per_instance.py 的规则逐个核出来的(VLM 否决 book#0,2,3 bookshelf#2 floor lamp#1 vase#2;
    # 太矮 book#1 0.04 m, vase#3 0.15 m, vase#7 0.10 m)
    assert c["会生成"] == 19, c
    assert c["按词"] == {"bookshelf": 2, "chair": 6, "floor lamp": 1, "sculpture": 1, "table": 1, "vase": 8}, c["按词"]
    assert c["不生成"] == {"VLM 判不是": 6, "太矮或太大(当成量错了)": 3}, c["不生成"]


def test_hitch_generation_uses_same_plan():
    """生成那步本身(不写图, 只列计划)也得出 19 个, 而且名单和本轮页一致。"""
    if not os.path.exists(f"{R1}/06_all_boxes.json"):
        print("[跳过] hitch_cuts_indoor/r1 不在盘上"); return
    sys.path.insert(0, f"{SAM3D}/data/outdoor/scripts")
    import solo_plan
    cfg = run_ctl._plan_cfg("indoor", "hitch_cuts_indoor", "r1")
    rows = solo_plan.plan_from_cfg(R1, cfg)
    names = sorted(r["生成名"] for r in rows if r["原因"] is None)
    assert len(names) == 19, names
    c = run_ctl._gate_extra("check_picks", "indoor", "hitch_cuts_indoor", "r1")["候选决定"]
    assert names == sorted(c["生成名单"]), (names, c["生成名单"])


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f(); print("ok", k)
