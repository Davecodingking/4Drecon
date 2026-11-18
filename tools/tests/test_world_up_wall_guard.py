#!/usr/bin/env python
"""A wall filling the view must not be silently accepted as ground."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "data/outdoor/scripts/world_up.py"


def make_wall(da3, tilt=0):
    (da3 / "depth").mkdir(parents=True)
    rows = np.arange(45)[:, None]
    depth = 2.5 + tilt * (rows - 22) / 40 + np.random.default_rng(2).normal(0, 0.005, (45, 45))
    np.save(da3 / "depth/0000_depth.npy", depth.astype(np.float32))
    np.save(da3 / "depth/0000_conf.npy", np.ones((45, 45), np.float32))
    (da3 / "cameras.json").write_text(json.dumps({"cameras": {"0000.jpg": {
        "intrinsic_3x3": [[40, 0, 22], [0, 40, 22], [0, 0, 1]],
        "extrinsic_w2c_4x4": np.eye(4).tolist(),
    }}}))
    (da3 / "bad_frames.json").write_text('{"坏帧": []}')


def run_world_up(root, fallback=False):
    return subprocess.run([sys.executable, str(SCRIPT)],
                          env=dict(os.environ, R1=str(root), UP_WALL_BODY="1" if fallback else "0"),
                          text=True, capture_output=True)


def test_wall_plane_uses_camera_wall_fallback_pending_body():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        da3 = root / "02_da3"
        make_wall(da3)
        result = run_world_up(root, fallback=True)
        assert result.returncode == 0, result.stdout + result.stderr
        actual = json.loads((da3 / "world_up.json").read_text())
        assert np.allclose(actual["up"], [0, -1, 0], atol=.05)
        assert actual["来源"] == "单墙约束+相机上（待Body脚底定高度）"
        assert actual["ground_h"] == -1.5


def test_wall_plane_uses_explicit_override():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        da3 = root / "02_da3"
        make_wall(da3)
        (da3 / "up_override.json").write_text(json.dumps({"up": "camera", "ground_h": -1.1}))
        result = run_world_up(root)
        assert result.returncode == 0, result.stdout + result.stderr
        actual = json.loads((da3 / "world_up.json").read_text())
        assert np.allclose(actual["up"], [0, -1, 0])
        assert actual["ground_h"] == -1.1
        assert actual["相机朝上点乘"] == 1.0


def test_wall_without_body_fallback_still_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        da3 = root / "02_da3"
        make_wall(da3)
        result = run_world_up(root)
        assert result.returncode != 0
        assert "不是可靠地面" in result.stdout + result.stderr


def test_warped_wall_does_not_tilt_floor():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        da3 = root / "02_da3"
        make_wall(da3, tilt=.6)
        result = run_world_up(root, fallback=True)
        assert result.returncode == 0, result.stdout + result.stderr
        actual = json.loads((da3 / "world_up.json").read_text())
        assert np.allclose(actual["up"], [0, -1, 0], atol=.05), actual
        assert "单墙检验" in actual["来源"]



def test_wall_dominated_with_tilted_ground_strip():
    """85% vertical wall + 15% bottom ground strip under a 15-deg upward camera pitch."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        da3 = root / "02_da3"
        (da3 / "depth").mkdir(parents=True)
        H, W, fx, cx, cy = 80, 100, 60.0, 50.0, 40.0
        ys, xs = np.mgrid[0:H, 0:W]
        # True gravity up in camera coords: pitched up 15 deg -> u_true = [0, -cos(15), sin(15)]
        th = np.radians(15.0)
        u_true = np.array([0.0, -np.cos(th), np.sin(th)])
        # Ground plane: P . u_true = -0.50; Wall plane: z_true = P_y*sin(th) + P_z*cos(th) = 3.5
        rx, ry, rz = (xs - cx) / fx, (ys - cy) / fx, np.ones_like(xs, float)
        d_wall = np.where(xs < 35, 1.8, 3.8) / (ry * np.sin(th) + rz * np.cos(th))
        denom_g = rx * u_true[0] + ry * u_true[1] + rz * u_true[2]
        d_gnd = np.where(denom_g < -1e-3, -0.50 / denom_g, 99.0)
        # Ray hits whichever positive surface is closer (bottom ~15% hits ground, top ~85% hits wall!)
        depth = np.minimum(d_wall, np.where(d_gnd > 0, d_gnd, 99.0))
        np.save(da3 / "depth/0000_depth.npy", depth.astype(np.float32))
        np.save(da3 / "depth/0000_conf.npy", np.ones((H, W), np.float32))
        (da3 / "cameras.json").write_text(json.dumps({"cameras": {"0000.jpg": {
            "intrinsic_3x3": [[fx, 0, cx], [0, fx, cy], [0, 0, 1]],
            "extrinsic_w2c_4x4": np.eye(4).tolist(),
        }}}))
        (da3 / "bad_frames.json").write_text('{"坏帧": []}')
        result = run_world_up(root, fallback=True)
        assert result.returncode == 0, result.stdout + result.stderr
        actual = json.loads((da3 / "world_up.json").read_text())
        ang_err = np.degrees(np.arccos(np.clip(np.dot(actual["up"], u_true), -1, 1)))
        assert ang_err < 2.0, f"Expected ~15 deg pitch recovery, got err={ang_err:.2f} deg: {actual}"
        assert abs(actual["ground_h"] - (-0.50)) < 0.05, actual
        assert ("朝上局部平面" in actual["来源"] or "地面平面拟合" in actual["来源"])


if __name__ == "__main__":
    test_wall_plane_uses_camera_wall_fallback_pending_body()
    test_wall_plane_uses_explicit_override()
    test_wall_without_body_fallback_still_fails()
    test_warped_wall_does_not_tilt_floor()
    test_wall_dominated_with_tilted_ground_strip()
    print("test_world_up_wall_guard: PASS")
