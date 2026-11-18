#!/usr/bin/env python
"""A scene with tracked people but no object candidates can still be assembled."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import trimesh


SCRIPT = Path(__file__).resolve().parents[2] / "data/outdoor/scripts/assemble_scene.py"


def run_empty_scene(*, indoor):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "r1"
        (root / "02_da3").mkdir(parents=True)
        (root / "04_picks").mkdir()
        (Path(tmp) / "outputs").mkdir()
        (root / "02_da3/world_up.json").write_text(json.dumps({"up": [0, -1, 0], "ground_h": -1.5}))
        (root / "02_da3/cameras.json").write_text(json.dumps({
            "metric_scale_applied": 1.0,
            "cameras": {
                "0000.jpg": {"camera_center_world": [0, 0, 0]},
                "0001.jpg": {"camera_center_world": [2, 0, 0]},
            },
        }))
        (root / "04_picks/picks.json").write_text("{}")
        (root / "06_all_boxes.json").write_text("{}")
        env = dict(os.environ, R1=str(root), TAG="empty_r1", GEN_OUT=str(Path(tmp) / "outputs"))
        if indoor:
            shell = trimesh.Scene()
            floor = trimesh.Trimesh(
                vertices=[[-3, 0, -3], [-3, 0, 3], [3, 0, -3], [3, 0, 3]],
                faces=[[0, 1, 3], [0, 3, 2]],
            )
            shell.add_geometry(floor, node_name="floor")
            path = Path(tmp) / "shell.glb"
            shell.export(path)
            env["ROOM_SHELL"] = str(path)
        result = subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        scene = trimesh.load(root / "07_scene.glb", force="scene")
        assert np.all(np.isfinite(scene.bounds)), scene.bounds
        assert ("floor" if indoor else "ground") in scene.graph.nodes_geometry
        assert "[组装] 0 个物体" in result.stdout


def test_ground_only_outdoor():
    run_empty_scene(indoor=False)


def test_shell_only_indoor():
    run_empty_scene(indoor=True)


if __name__ == "__main__":
    test_ground_only_outdoor()
    test_shell_only_indoor()
    print("test_assemble_empty_boxes: PASS")
