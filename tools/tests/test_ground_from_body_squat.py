"""The ground fallback should recover squat's foot height before snapping people."""
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "data/indoorOutput/mansquatindoor/r1"
PY = "/home/ubuntu/miniforge3/envs/sam3d/bin/python"


def test_body_ground_on_squat():
    if not (R1 / "body4d/04_mesh_occ0/mhr_4d_individual/1").exists():
        return
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "world_up.json"
        p = subprocess.run([PY, str(ROOT / "pipeline/ground_from_body.py"),
                            "--r1", str(R1), "--body", str(R1 / "body4d"),
                            "--out", str(out), "--samples", "12", "--force"],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        result = json.loads(out.read_text())
        assert -1.3 < result["ground_h"] < -1.0, result
        assert "Body脚底高度" in result["来源"]
        assert result["Body脚底帧数"] >= 10


if __name__ == "__main__":
    test_body_ground_on_squat()
    print("test_ground_from_body_squat: PASS")
