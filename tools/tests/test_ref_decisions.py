#!/usr/bin/env python
"""把参考轮次落地的闸口决定照抄到新轮次的 05_review.json(tools/apply_ref_decisions.py)。验证脚本自动过闸用。
跑: $PY tools/tests/test_ref_decisions.py"""
import json, os, subprocess, sys, tempfile

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/tools")
import apply_ref_decisions as A

ROWS = [("box", 1), ("box", 5), ("desk chair", 1), ("desk chair", 2), ("trash can", 0), ("door", 0), ("table", 3)]
APPLIED = {"排除词": ["trash can", "pipe"], "逐实例词": ["box", "table"], "共用词": ["desk chair"],
           "不生成": ["off_r1_box_5", "off_r1_door_0", "off_r1_aed_unit_10044"], "指定样本": {}, "改类别": {}, "VLM放行": []}


def _run():
    T = tempfile.mkdtemp()
    json.dump({"候选": [{"词": w, "实例": i, "生成名": f"off_r7_{w.replace(' ', '_')}_{i}", "决定": "keep"} for w, i in ROWS],
               "说明": "x"}, open(f"{T}/05_review.json", "w"), ensure_ascii=False)
    json.dump(APPLIED, open(f"{T}/ref.json", "w"), ensure_ascii=False)
    rep = A.apply(T, f"{T}/ref.json", "off_r1")
    d = json.load(open(f"{T}/05_review.json"))
    return {(r["词"], r["实例"]): r["决定"] for r in d["候选"]}, rep, d


def test_decisions_are_copied_by_kind():
    got, rep, d = _run()
    assert got[("trash can", 0)] == "drop-word"                       # 整词作废
    assert got[("box", 5)] == "drop" and got[("door", 0)] == "drop"   # 单个不生成: 按"词_实例"对, 不看标签
    assert got[("box", 1)] == "solo" and got[("table", 3)] == "solo"  # 逐实例词
    assert {got[("desk chair", 1)], got[("desk chair", 2)]} == {"shared"}
    assert d["说明"] == "x"                                           # 别的内容不动


def test_what_could_not_be_matched_is_reported():
    got, rep, d = _run()
    assert rep["没对上的排除词"] == ["pipe"], rep                       # 新轮次里没有这个词
    assert rep["没对上的不生成"] == ["off_r1_aed_unit_10044"], rep       # 召回才有的实例, 这一轮没开召回


def test_command_line():
    T = tempfile.mkdtemp()
    json.dump({"候选": [{"词": "pipe", "实例": 0, "生成名": "a_pipe_0", "决定": "keep"}]}, open(f"{T}/05_review.json", "w"))
    json.dump({"排除词": ["pipe"]}, open(f"{T}/ref.json", "w"))
    r = subprocess.run([sys.executable, f"{SAM3D}/tools/apply_ref_decisions.py", T, f"{T}/ref.json", "off_r1"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and '"drop-word": 1' in r.stdout, r.stdout + r.stderr
    assert json.load(open(f"{T}/05_review.json"))["候选"][0]["决定"] == "drop-word"


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
