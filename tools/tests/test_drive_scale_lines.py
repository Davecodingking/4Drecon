#!/usr/bin/env python
"""闸 3(尺度)的报告(drive.check_scale): 人当尺子时每人的比例要列出来, 被踢出尺子的人要点名带原因。
起因(office r12, 2026-10-09): 外套被当成人 7 混进尺子, 闸口上只看得到 k, 看不到它。跑: $PY tools/tests/test_drive_scale_lines.py"""
import importlib.util, json, os, tempfile
p = os.path.expanduser("~/sam3d/.claude/skills/run-recon-pipeline/drive.py")
spec = importlib.util.spec_from_file_location("drive", p); drive = importlib.util.module_from_spec(spec); spec.loader.exec_module(drive)

BASE = dict(k=1.023, 一致率=0.89, 采用类数=12, 候选类数=12, 施加=True, 逐类=[], 来源="人")


def _lines(people):
    with tempfile.TemporaryDirectory() as R1:
        os.makedirs(f"{R1}/02_da3")
        json.dump(dict(BASE, 人=people), open(f"{R1}/02_da3/world_scale.json", "w"), ensure_ascii=False)
        lines, _ = drive.check_scale(R1, "off_r1", "office", kind="indoor", run="r1")
    return lines


def test_excluded_person_is_named_with_reason():
    lines = _lines(dict(人数=7, 比例中位=0.978, 每人={"1": {"场景深度比模型": 0.89, "帧": 110}, "7": {"场景深度比模型": 0.466, "帧": 14}},
                        踢掉={"7": "离全体中位 0.945 差 51%, 超过 25%"}))
    per = [l for l in lines if "人当尺子" in l]
    assert len(per) == 1 and "7 个" in per[0] and "#1 0.89" in per[0] and "#7 0.466<-踢掉" in per[0], lines
    star = [l for l in lines if "没当尺子" in l]
    assert len(star) == 1 and "#7" in star[0] and "离全体中位 0.945 差 51%" in star[0] and "点掉" in star[0], lines


def test_nobody_excluded_prints_no_star():
    lines = _lines(dict(人数=2, 比例中位=0.95, 每人={"1": {"场景深度比模型": 0.9, "帧": 50}, "2": {"场景深度比模型": 1.0, "帧": 40}}, 踢掉={}))
    assert any("人当尺子 2 个" in l for l in lines), lines
    assert not any("没当尺子" in l for l in lines), lines


def test_old_record_without_people_still_works():
    with tempfile.TemporaryDirectory() as R1:
        os.makedirs(f"{R1}/02_da3")
        json.dump(dict(BASE, 来源="家具"), open(f"{R1}/02_da3/world_scale.json", "w"), ensure_ascii=False)
        lines, _ = drive.check_scale(R1, "off_r1", "office", kind="indoor", run="r1")
    assert lines and lines[0].startswith("标定倍数 k = 1.023") and not any("尺子" in l for l in lines), lines


if __name__ == "__main__":
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f(); print(f"  ok  {k}")
    print("== test_drive_scale_lines 全过 ==")
