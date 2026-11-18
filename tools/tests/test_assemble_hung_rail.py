#!/usr/bin/env python
"""Floor contact must not turn a wall-mounted rail into a tall stand."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import trimesh


SCRIPT = Path(__file__).resolve().parents[2] / "data/outdoor/scripts/assemble_scene.py"


def assemble(rest, height):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "r1"
        (root / "02_da3").mkdir(parents=True)
        (root / "04_picks").mkdir()
        outputs = Path(tmp) / "outputs"
        name = "test_r1_coat_rack_0"
        (outputs / name).mkdir(parents=True)
        trimesh.creation.box(extents=[1.0, 0.211, 0.142]).export(outputs / name / f"{name}.glb")
        (root / "02_da3/world_up.json").write_text(json.dumps({"up": [0, -1, 0], "ground_h": -1.1}))
        (root / "02_da3/cameras.json").write_text(json.dumps({"metric_scale_applied": 1.0}))
        (root / "04_picks/picks.json").write_text("{}")
        (root / "01_vlm_rest.json").write_text(json.dumps({"coat rack": rest}))
        (root / "01_vlm_sizes.json").write_text(json.dumps({"coat rack": 1.8}))
        (root / "06_all_boxes.json").write_text(json.dumps({"coat rack": [{
            "中心": [0, -0.2, 2.0], "长轴": [1, 0, 0], "尺寸": [0.78, 0.081, height],
            "底高": 0.2 - height / 2, "obj_id": 0, "点数": 25000, "用帧": 8, "总帧": 120,
        }]}))
        env = dict(os.environ, R1=str(root), TAG="test_r1", GEN_OUT=str(outputs),
                   EXCLUDE="", REST=str(root / "01_vlm_rest.json"), FLOOR_EXTEND="1",
                   SNAP_GROUND="0", MIN_H="0.15")
        result = subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        scene = trimesh.load(root / "07_scene.glb", force="scene")
        names = [n for n in scene.graph.nodes_geometry if n == "coat_rack_0"]
        assert names, result.stdout
        transform, geometry = scene.graph[names[0]]
        mesh = scene.geometry[geometry].copy()
        mesh.apply_transform(transform)
        return mesh.extents, mesh.bounds, result.stdout


def test_hung_rail_keeps_its_measured_width_and_wall_height():
    ext, bounds, _ = assemble("hung", 0.103)
    assert 0.72 <= max(ext[0], ext[2]) <= 0.84, ext
    assert 0.12 <= ext[1] <= 0.25, ext
    assert bounds[0, 1] > -0.5, bounds


def test_floor_label_alone_does_not_stretch_a_slim_rail():
    ext, _, _ = assemble("floor", 0.20)
    assert ext[1] < 0.4, ext


if __name__ == "__main__":
    test_hung_rail_keeps_its_measured_width_and_wall_height()
    test_floor_label_alone_does_not_stretch_a_slim_rail()
    print("test_assemble_hung_rail: PASS")
