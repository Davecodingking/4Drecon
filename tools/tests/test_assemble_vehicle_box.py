"""Assemble two rotated vehicle meshes into separated measured boxes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import trimesh

SCRIPTS = Path(__file__).resolve().parents[2] / "data/outdoor/scripts"
sys.path.insert(0, str(SCRIPTS))
from world_frame import inverse_T


def test_rotated_vans_fit_their_individual_boxes():
    up = np.array([0., -1., 0.])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        da = root / "02_da3"
        da.mkdir()
        (da / "world_up.json").write_text(json.dumps({"up": up.tolist(), "ground_h": 0}))
        (da / "cameras.json").write_text('{"metric_scale_applied": 1}')
        (root / "04_picks").mkdir()
        (root / "04_picks/picks.json").write_text("{}")
        boxes = []
        for i, (length, width) in enumerate([(3., 2.), (2.7, 1.6)]):
            name = f"trial_van_{i}"
            mesh = trimesh.creation.box(extents=[1.2, .35, 2.8])
            mesh.apply_transform(trimesh.transformations.rotation_matrix(.6, [0, 1, 0]))
            folder = root / "assets" / name
            folder.mkdir(parents=True)
            mesh.export(folder / f"{name}.glb")
            boxes.append({"中心": [i * 4., 0., 2.], "长轴": [.8, 0., .6],
                          "尺寸": [length, width, 1.4], "底高": -.7,
                          "obj_id": i, "点数": 1500, "用帧": 8, "总帧": 20})
        (root / "06_all_boxes.json").write_text(json.dumps({"van": boxes}))
        env = dict(os.environ, R1=str(root), TAG="trial", GEN_OUT=str(root / "assets"),
                   EXCLUDE="", SNAP_GROUND="0")
        subprocess.run([sys.executable, str(SCRIPTS / "assemble_scene.py")], env=env, check=True)
        scene = trimesh.load(root / "07_scene.glb", force="scene")
        for i, box in enumerate(boxes):
            matrix, key = scene.graph[f"van_{i}"]
            points = trimesh.transform_points(scene.geometry[key].vertices, matrix)
            world = points @ inverse_T(up)[:3, :3].T
            axis = box["长轴"]
            projected = (world - box["中心"]) @ np.array([axis, np.cross(up, axis), up]).T
            np.testing.assert_allclose(np.ptp(projected, axis=0), box["尺寸"], atol=1e-5)
            np.testing.assert_allclose(projected.min(0) + projected.max(0), 0., atol=1e-5)


if __name__ == "__main__":
    test_rotated_vans_fit_their_individual_boxes()
