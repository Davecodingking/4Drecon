#!/usr/bin/env python
"""每张卡一把 GPU 锁(文件名规则) + 网页开跑不把自己的卡号传下去。领卡见 test_gpu_slot.py。
跑: $PY tools/tests/test_gpu_alloc.py"""
import fcntl, os, subprocess, sys, tempfile

SAM3D = os.path.expanduser("~/sam3d")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, f"{SAM3D}/tools"); sys.path.insert(0, f"{SAM3D}/pipeline")
import _common, run_ctl
from test_run_ctl import _sandbox


def _with_cvd(v, fn):
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else: os.environ["CUDA_VISIBLE_DEVICES"] = v
    try:
        return fn()
    finally:
        if old is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else: os.environ["CUDA_VISIBLE_DEVICES"] = old


def test_lock_path_per_gpu():
    assert _with_cvd("3", _common.lock_path) == "/tmp/outdoor_pipeline_gpu3.lock"
    assert _with_cvd("5,6", _common.lock_path) == "/tmp/outdoor_pipeline_gpu5.lock"
    assert _with_cvd(None, _common.lock_path) == "/tmp/outdoor_pipeline_gpu0.lock"     # 不设 = torch 的 cuda:0
    assert _with_cvd("", _common.lock_path) == "/tmp/outdoor_pipeline_gpu0.lock"


def test_shell_lock_matches_python():
    """gpulock.sh 和 Python 必须算出同一个文件, 否则同一张卡又分成两个队。"""
    for v in ("3", "5,6", ""):
        sh = subprocess.run(["bash", "-c", f'CUDA_VISIBLE_DEVICES="{v}"; . {SAM3D}/pipeline/gpulock.sh; echo $GPU_LOCK'],
                            capture_output=True, text=True).stdout.strip()
        assert sh == _with_cvd(v, _common.lock_path), (v, sh)


def test_different_gpus_do_not_block():
    held = open(_with_cvd("7", _common.lock_path), "a+")
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        probe = "import fcntl,sys; sys.path.insert(0,%r); import _common; f=open(_common.lock_path(),'a+');" \
                "fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)" % f"{SAM3D}/pipeline"
        run = lambda v: subprocess.run([sys.executable, "-c", probe], env=dict(os.environ, CUDA_VISIBLE_DEVICES=v),
                                        stderr=subprocess.DEVNULL).returncode
        assert run("6") == 0            # 别的卡: 拿得到
        assert run("7") != 0            # 同一张卡: 被占着
    finally:
        fcntl.flock(held, fcntl.LOCK_UN); held.close()


def test_go_lets_drive_go_pick_the_gpu():
    """网页自己的 CUDA_VISIBLE_DEVICES 不许传给 drive_go.sh —— 否则所有轮次又挤在同一张卡上。"""
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        open(run_ctl.DRIVE_GO, "w").write(f'#!/bin/bash\necho "[${{CUDA_VISIBLE_DEVICES:-unset}}]" > {T}/go_env\n')
        os.makedirs(f"{T}/or/soccer/r1")
        _with_cvd("4", lambda: run_ctl.go("outdoor", "soccer", "r1"))
        assert open(f"{T}/go_env").read().strip() == "[unset]"


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
