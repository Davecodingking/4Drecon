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

if __name__ == "__main__":
    test_missing_stale_fresh(); print("ok test_missing_stale_fresh")
