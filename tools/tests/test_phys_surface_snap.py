#!/usr/bin/env python
"""A supported table lamp can settle across a small depth error."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import trimesh


SCRIPT = Path(__file__).resolve().parents[2] / "data/outdoor/scripts/scene_phys.py"


def test_surface_lamp_settles_on_nightstand():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scene = trimesh.Scene()
        floor = trimesh.creation.box(extents=[4, 0.01, 4])
        floor.apply_translation([0, -1.105, 0])
        scene.add_geometry(floor, node_name="floor")
        table = trimesh.creation.box(extents=[0.6, 0.35, 0.5])
        table.apply_translation([0, -0.93, 0])  # top -0.755, bottom -1.105
        scene.add_geometry(table, node_name="nightstand_0")
        lamp = trimesh.creation.box(extents=[0.2, 0.3, 0.2])
        lamp.apply_translation([0, -0.415, 0])  # bottom -0.565, 19 cm above table
        scene.add_geometry(lamp, node_name="lamp_0")
        scene.export(root / "07_scene.glb")
        (root / "rest.json").write_text(json.dumps({"nightstand": "floor", "lamp": "surface"}))
        env = dict(os.environ, R1=str(root), REST=str(root / "rest.json"),
                   SUPPORT_LABELS="nightstand", LAND_MAX="0.15", SNAP="0.15")
        result = subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        output = trimesh.load(root / "07_scene_phy.glb", force="scene")
        bounds = {}
        for name in output.graph.nodes_geometry:
            T, g = output.graph[name]
            m = output.geometry[g].copy()
            m.apply_transform(T)
            bounds[name] = m.bounds
        assert abs(bounds["lamp_0"][0, 1] - bounds["nightstand_0"][1, 1]) < 0.03, bounds


if __name__ == "__main__":
    test_surface_lamp_settles_on_nightstand()
    print("test_phys_surface_snap: PASS")
