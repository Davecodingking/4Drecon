#!/usr/bin/env python
"""召回开关 RECALL: 默认关 -> 召回五步跳过、组装/房间壳用基础盒子、闸 4 不停。
都是 dry 调用, 不起 GPU、不碰盘上的轮次。跑: $PY tools/tests/test_recall_switch.py"""
import contextlib, importlib.util, io, os, tempfile, time

SAM3D = os.path.expanduser("~/sam3d")
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
out_p = _load("outdoor_pipeline", f"{SAM3D}/pipeline/outdoor_pipeline.py")
in_p  = _load("indoor_pipeline",  f"{SAM3D}/pipeline/indoor_pipeline.py")
drive = _load("drive", f"{SAM3D}/.claude/skills/run-recon-pipeline/drive.py")

def _run(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a)
    return buf.getvalue()

def _ctx(mod, scene, R1):
    c = mod.Ctx(scene, "_test_recall")
    c.R1, c.LOG = R1, f"{R1}/logs"
    return c

def _boxes(R1):
    """基础盒子 + 比它新的合并盒子(召回开着时该用 merged 的形状)。"""
    b, m = f"{R1}/06_all_boxes.json", f"{R1}/06_all_boxes_merged.json"
    open(b, "w").write("{}"); open(m, "w").write("{}")
    os.utime(m, (time.time() + 10,) * 2)
    return b, m

def test_outdoor_steps_skip_by_default():
    with tempfile.TemporaryDirectory() as R1:
        c = _ctx(out_p, "soccer", R1)
        assert c.cfg["RECALL"] == "0"
        for fn in (out_p.s_detect, out_p.s_recall, out_p.s_recall_mask,
                   out_p.s_recall_gen, out_p.s_recall_merge):
            o = _run(fn, c, True)
            assert "召回关" in o and "$ " not in o, (fn.__name__, o)
        c.cfg["RECALL"] = "1"
        assert "$ " in _run(out_p.s_detect, c, True)

def test_outdoor_assemble_ignores_merged_when_off():
    with tempfile.TemporaryDirectory() as R1:
        c = _ctx(out_p, "soccer", R1); b, m = _boxes(R1)
        assert out_p._boxes_arg(c) == {}
        c.cfg["RECALL"] = "1"
        assert out_p._boxes_arg(c) == {"BOXES": m}

def test_indoor_steps_skip_by_default():
    with tempfile.TemporaryDirectory() as R1:
        c = _ctx(in_p, "office", R1)
        assert c.cfg["RECALL"] == "0"
        for fn in (in_p.o_detect, in_p.o_recall, in_p.o_recall_mask,
                   in_p.o_recall_gen, in_p.o_recall_merge):
            o = _run(fn, c, True)
            assert "召回关" in o and "$ " not in o, (fn.__name__, o)

def test_indoor_assemble_and_shell_use_base_boxes_when_off():
    with tempfile.TemporaryDirectory() as R1:
        c = _ctx(in_p, "office", R1); b, m = _boxes(R1)
        for fn in (in_p.o_assemble, in_p.o_shell):
            o = _run(fn, c, True)
            assert b in o and m not in o, (fn.__name__, o)
        c.cfg["RECALL"] = "1"
        for fn in (in_p.o_assemble, in_p.o_shell):
            assert m in _run(fn, c, True), fn.__name__

def test_drive_gate4_follows_switch():
    seg4 = [s[2] for s in drive.SEGMENTS["outdoor"]].index("check_recall")
    assert drive.gate_stops("outdoor", seg4, "soccer", "r1") is False
    assert drive.gate_stops("outdoor", seg4 - 1, "soccer", "r1") is True
    os.environ["RECALL"] = "1"
    try:
        assert drive.gate_stops("outdoor", seg4, "soccer", "r1") is True
    finally:
        del os.environ["RECALL"]

def test_drive_skips_gate4_decisions_when_off():
    seg4 = [s[2] for s in drive.SEGMENTS["outdoor"]].index("check_recall")
    with tempfile.TemporaryDirectory() as R1:
        open(f"{R1}/05_review.json", "w").write("{}")      # 有决策文件, 不跳过就会去 --apply
        o = _run(drive.apply_decisions, R1, "soc_r1", "soccer", "outdoor", "r1", seg4, True)
        assert "召回关" in o and "--apply" not in o, o

if __name__ == "__main__":
    for t in (test_outdoor_steps_skip_by_default, test_outdoor_assemble_ignores_merged_when_off,
              test_indoor_steps_skip_by_default, test_indoor_assemble_and_shell_use_base_boxes_when_off,
              test_drive_gate4_follows_switch, test_drive_skips_gate4_decisions_when_off):
        t(); print("ok", t.__name__)
