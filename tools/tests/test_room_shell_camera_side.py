"""The camera-side extension must cover the whole visible wall span."""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
PY = "/home/ubuntu/miniforge3/envs/sam3d/bin/python"


def test_camera_extension_fills_both_shoulders_of_t_footprint(tmp_path):
    # A broad back wall and a narrow foreground person form a T in the point cloud.
    wall = np.array([(x, y, z) for x in np.arange(0, .45, .1)
                     for y in np.arange(-1.5, 1.51, .1) for z in (.4, 1.0, 2.0, 3.0)])
    person = np.array([(x, y, z) for x in np.arange(.4, 2.51, .1)
                       for y in np.arange(-.5, .51, .1) for z in (.4, 1.0, 2.0, 3.0)])
    wing = np.array([(x, y, z) for x in np.arange(.3, .71, .1)
                     for y in np.arange(-2.0, -1.49, .1) for z in (.4, 1.0, 2.0, 3.0)])
    pcd = tmp_path / "points.ply"
    trimesh.PointCloud(np.concatenate([wall, person, wing])).export(pcd)
    zup = tmp_path / "zup.json"
    zup.write_text(json.dumps(dict(R=np.eye(3).tolist(), t=[0, 0, 0], s=1,
                                    camera_centers_new=[[4, 0, 1.5]])))
    up = tmp_path / "world_up.json"
    up.write_text(json.dumps(dict(up=[0, 0, 1], ground_h=0)))
    boxes = tmp_path / "boxes.json"
    boxes.write_text("{}")
    out = tmp_path / "shell.glb"
    seg = tmp_path / "walls.json"
    subprocess.run([PY, str(ROOT / "pipeline/room_shell.py"), "--pcd", str(pcd),
                    "--out", str(out), "--seg-out", str(seg), "--zup", str(zup),
                    "--world-up", str(up), "--cam-margin", ".5", "--close", "9",
                    "--simplify", ".1", "--boxes", str(boxes)],
                   check=True, capture_output=True, text=True)
    outline = np.asarray(json.loads(seg.read_text())["轮廓"])
    # At both ends of the real wall, the camera-side boundary reaches beyond the camera.
    front = outline[outline[:, 0] > 4.3]
    assert len(front) >= 2 and front[:, 1].min() < -1.2 and front[:, 1].max() > 1.2, outline
    assert len(outline) == 4, outline
    assert -1.7 < outline[:, 1].min() < -1.3, outline


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        test_camera_extension_fills_both_shoulders_of_t_footprint(Path(d))
