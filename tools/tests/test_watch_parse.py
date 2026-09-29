#!/usr/bin/env python
"""看板读驱动日志(watch_run.parse_drive): 并行之后一段里有好几步同时在跑。
跑: $PY tools/tests/test_watch_parse.py"""
import os, sys, tempfile

sys.path.insert(0, os.path.expanduser("~/sam3d/tools"))
import watch_run as W


def _parse(txt):
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(txt)
    try:
        return W.parse_drive(f.name)
    finally:
        os.remove(f.name)


def test_parallel_log():
    d = _parse("[开始] 2026-09-29 10:00:00\n[并行] 开始 vlm\n[并行] 开始 da3\n"
               "== vlm  [trellis2]  列词表\n    $ 命令\n   34.2s\n\n[并行] 开始 up\n")
    assert d["steps"] == [("vlm", 34.2)], d["steps"]
    assert d["running_all"] == ["da3", "up"] and d["running"] == "da3", d


def test_failed_step_is_not_running():
    d = _parse("[开始] 2026-09-29 10:00:00\n[并行] 开始 vlm\n[并行] 开始 da3\n"
               "== vlm  [trellis2]  列词表\n[失败] 退出码 1: python x.py\n[失败] 步骤 vlm 退出码 1\n")
    assert d["running_all"] == ["da3"] and d["fail"], d


def test_old_log_unchanged():
    d = _parse("[开始] 2026-09-24 15:43:35\n== vlm  [trellis2]  x\n   3.0s\n\n== da3  [spatiallm]  y\n")
    assert d["steps"] == [("vlm", 3.0)] and d["running"] == "da3" and d["running_all"] == ["da3"], d


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
