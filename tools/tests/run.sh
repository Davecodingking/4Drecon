#!/bin/bash
# 跑 tools/ 的全部测试(没有 pytest, 都是带 assert 的脚本)。快的先跑; 后两个要 office r6 在盘上, 各 ~40 s。
set -e; cd "$(dirname "$0")"; PY=~/miniforge3/envs/sam3d/bin/python
$PY test_gate_locate.py; $PY test_scene_view.py; $PY test_drive_box_view.py
bash test_box_fit_view_data.sh; bash test_gate_3d.sh
echo "== tools/tests 全过 =="
