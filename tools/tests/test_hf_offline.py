#!/usr/bin/env python
"""流水线每一步(都经 _common.sh 起)默认 HF_HUB_OFFLINE=1: 权重全在 ~/.cache/huggingface, 不许再上网核对。
2026-09-24 nfl1 的 verify: 本机直连不了 huggingface.co, 只能走 ~/.bashrc 的代理(借用户电脑的网),
隧道一卡, transformers 每个文件 HEAD 一次、10 s 超时重试 5 次 —— 验 1 个实例用了 18 分钟。
要下新模型时显式给 HF_HUB_OFFLINE=0(环境变量或 sh 的 env=)就会照常联网。
跑: $PY tools/tests/test_hf_offline.py"""
import os, sys, tempfile

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import _common


def _child_sees(env=None, outer=None):
    old = os.environ.pop("HF_HUB_OFFLINE", None)
    if outer is not None: os.environ["HF_HUB_OFFLINE"] = outer
    try:
        with tempfile.TemporaryDirectory() as T:
            _common.sh('echo "[${HF_HUB_OFFLINE:-unset}]"', log=f"{T}/x.log", env=env)
            return [l for l in open(f"{T}/x.log") if l.startswith("[")][-1].strip()
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
        if old is not None: os.environ["HF_HUB_OFFLINE"] = old


def test_steps_are_offline_by_default():
    assert _child_sees() == "[1]"
    assert _child_sees(env={"CUDA_HOME": "/nonexistent"}) == "[1]"     # 室内给了 env 的那条路也一样


def test_explicit_online_wins():
    assert _child_sees(outer="0") == "[0]"
    assert _child_sees(env={"HF_HUB_OFFLINE": "0"}) == "[0]"


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
