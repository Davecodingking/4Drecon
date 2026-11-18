#!/usr/bin/env python
"""A partly observed bed rail must not shrink the entire bed frame."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import trimesh


SCRIPT = Path(__file__).resolve().parents[2] / "data/outdoor/scripts/assemble_scene.py"


def test_bed_frame_uses_footprint_when_height_is_truncated():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "r1"
        (root / "02_da3").mkdir(parents=True)
        (root / "04_picks").mkdir()
        outputs = Path(tmp) / "outputs"
        name = "test_r1_bed_frame_0"
        (outputs / name).mkdir(parents=True)
        trimesh.creation.box(extents=[0.76, 0.65, 1.0]).export(outputs / name / f"{name}.glb")
        (root / "02_da3/world_up.json").write_text(json.dumps({"up": [0, -1, 0], "ground_h": -1.5}))
        (root / "02_da3/cameras.json").write_text(json.dumps({"metric_scale_applied": 1.0}))
        (root / "04_picks/picks.json").write_text("{}")
        (root / "06_all_boxes.json").write_text(json.dumps({"bed frame": [{
            "中心": [0, 0, 2.5], "长轴": [1, 0, 0], "尺寸": [1.56, 0.15, 0.23],
            "底高": -0.2, "obj_id": 0, "点数": 72000, "用帧": 8, "总帧": 120,
        }]}))
        env = dict(os.environ, R1=str(root), TAG="test_r1", GEN_OUT=str(outputs),
                   EXCLUDE="", SNAP_GROUND="0")
        result = subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        scene = trimesh.load(root / "07_scene.glb", force="scene")
        name = next(n for n in scene.graph.nodes_geometry if "bed_frame" in n)
        transform, geometry = scene.graph[name]
        mesh = scene.geometry[geometry].copy()
        mesh.apply_transform(transform)
        footprint = mesh.extents[[0, 2]]
        # The measured 1.56 m rail is the bed's width. Matching it to the
        # generated bed's longest side makes both footprint axes too small.
        assert min(footprint) >= 1.45, mesh.extents
        assert max(footprint) >= 1.95, mesh.extents


if __name__ == "__main__":
    test_bed_frame_uses_footprint_when_height_is_truncated()
    print("test_assemble_bed_frame_scale: PASS")
