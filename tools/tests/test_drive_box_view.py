#!/usr/bin/env python
"""drive.ensure_box_view 的判"要不要重算": 临时目录 + dry, 不真跑 box_fit_view。跑: $PY tools/tests/test_drive_box_view.py"""
import importlib.util, os, tempfile, time
p = os.path.expanduser("~/sam3d/.claude/skills/run-recon-pipeline/drive.py")
spec = importlib.util.spec_from_file_location("drive", p); drive = importlib.util.module_from_spec(spec); spec.loader.exec_module(drive)

def test_missing_stale_fresh():
    with tempfile.TemporaryDirectory() as R1:
        assert drive.ensure_box_view(R1, dry=True) == "干跑"                 # 没有 -> 要算
        open(f"{R1}/10_box_fit_view.json", "w").write("{}")
        assert drive.ensure_box_view(R1, dry=True) == "新鲜"                 # 有且不旧 -> 跳过
        g = f"{R1}/07_scene_phy.glb"; open(g, "wb").write(b"x"); os.utime(g, (time.time() + 10,) * 2)
        assert drive.ensure_box_view(R1, dry=True) == "干跑"                 # 产物比它新 -> 要算

def test_hung_view_does_not_block_gate():
    """box_fit_view 卡住(2026-09-24 nfl1: 找地面吃了 194 GB、近 5 分钟), 驱动不许跟着卡在闸口"运行中"。"""
    with tempfile.TemporaryDirectory() as R1:
        fake = f"{R1}/fake_py"; open(fake, "w").write("#!/bin/bash\nexec sleep 30\n"); os.chmod(fake, 0o755)
        old = drive.PY, getattr(drive, "BOX_VIEW_TIMEOUT", None)
        drive.PY, drive.BOX_VIEW_TIMEOUT = fake, 1
        try:
            t = time.time(); r = drive.ensure_box_view(R1); took = time.time() - t
        finally:
            drive.PY, drive.BOX_VIEW_TIMEOUT = old
        assert r == "超时" and took < 10, (r, took)

if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
