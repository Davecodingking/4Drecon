#!/usr/bin/env python
"""按依赖表并行跑步骤(pipeline/sched.py)。不碰显卡: 步骤是假的。
跑: $PY tools/tests/test_sched.py"""
import json, os, subprocess, sys, tempfile, time

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import sched

KEYS = ["frames", "vlm", "da3", "up", "track", "boxes", "scale", "solo"]
DEPS = {"vlm": ["frames"], "da3": ["frames"], "up": ["da3"], "track": ["frames", "vlm"],
        "boxes": ["track", "up"], "scale": ["boxes"], "solo": ["scale"]}


def _raises(fn, *a):
    try:
        fn(*a)
    except ValueError as e:
        return str(e)
    raise AssertionError("该报错没报错")


def test_check_graph_accepts_a_good_table():
    sched.check_graph(KEYS, DEPS)


def test_check_graph_rejects_unknown_names():
    assert "walls" in _raises(sched.check_graph, KEYS, dict(DEPS, boxes=["walls"]))
    assert "walls" in _raises(sched.check_graph, KEYS, dict(DEPS, walls=["frames"]))


def test_check_graph_rejects_backward_edges():
    """依赖排在后面 = 顺序错或者成环。步骤表的顺序就是 PARALLEL=0 时的执行顺序。"""
    assert "solo" in _raises(sched.check_graph, KEYS, dict(DEPS, up=["solo"]))
    assert "up" in _raises(sched.check_graph, KEYS, dict(DEPS, up=["up"]))


def test_ancestors_are_transitive():
    a = sched.ancestors(DEPS)
    assert a["solo"] == {"scale", "boxes", "track", "up", "da3", "vlm", "frames"}, a["solo"]
    assert a["frames"] == set()


def test_real_tables_pass():
    """三条流水线自己的步骤表和依赖表。依赖表以前只用来判过期, 现在还决定谁先谁后, 不能有错。"""
    PY = sys.executable
    for cmd in ([f"{SAM3D}/pipeline/outdoor_pipeline.py"],
                [f"{SAM3D}/pipeline/indoor_pipeline.py", "--chain", "obj"],
                [f"{SAM3D}/pipeline/indoor_pipeline.py", "--chain", "lm"],
                [f"{SAM3D}/pipeline/body4d_pipeline.py"]):
        r = subprocess.run([PY] + cmd + ["--dump-deps"], capture_output=True, text=True)
        assert r.returncode == 0, (cmd, r.stderr[-400:])
        d = json.loads(r.stdout.strip().splitlines()[-1])
        sched.check_graph(d["steps"], d["deps"])
        assert set(d["gpu"]) <= set(d["steps"]), (cmd, set(d["gpu"]) - set(d["steps"]))


def test_track_depends_on_vlm():
    """track 读 vlm 出的词表(01_vlm_vocab.json)。以前依赖表里没写, 因为两步不在同一段里没出过事。"""
    for cmd in ([f"{SAM3D}/pipeline/outdoor_pipeline.py"], [f"{SAM3D}/pipeline/indoor_pipeline.py", "--chain", "obj"]):
        r = subprocess.run([sys.executable] + cmd + ["--dump-deps"], capture_output=True, text=True)
        d = json.loads(r.stdout.strip().splitlines()[-1])
        assert "vlm" in d["deps"]["track"], (cmd, d["deps"]["track"])


def test_indoor_shell_waits_for_recall_merge():
    """室内的房间壳: 开着召回时读的是 06_all_boxes_merged.json(recall_merge 写的)。两步在同一段里,
    依赖表里不写的话并行时会同时跑, 壳读到的是上一次留下的盒子。"""
    r = subprocess.run([sys.executable, f"{SAM3D}/pipeline/indoor_pipeline.py", "--chain", "obj", "--dump-deps"],
                       capture_output=True, text=True)
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert "recall_merge" in d["deps"]["shell"], d["deps"]["shell"]


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
