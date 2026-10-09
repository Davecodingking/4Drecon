#!/bin/bash
# 跑 tools/ 的全部测试(没有 pytest, 都是带 assert 的脚本)。快的先跑; 后两个要 office r6 在盘上, 各 ~40 s。
set -e; cd "$(dirname "$0")"; PY=~/miniforge3/envs/sam3d/bin/python
$PY test_gate_locate.py; $PY test_scene_view.py; $PY test_drive_box_view.py; $PY test_recall_switch.py; $PY test_people_gate.py
$PY test_run_ctl.py; $PY test_run_page.py; $PY test_gpu_alloc.py; $PY test_body_distance.py; $PY test_body_occlusion.py; $PY test_body_pose_smooth.py; $PY test_body_track_smooth.py; $PY test_body_tidy.py; $PY test_sched.py; $PY test_gpu_slot.py; $PY test_watch_parse.py; $PY test_fanout.py; $PY test_track_clean.py; $PY test_people_identity.py; $PY test_cam_track.py; $PY test_mesh_merge.py; $PY test_ref_decisions.py
$PY test_box_fit_floor.py; $PY test_hf_offline.py
$PY test_motion_export.py; $PY test_mhr_body.py; $PY test_body_motion.py     # 要 MHR 模型文件在盘上(sam-body4d/checkpoints), 两个加起来约 1 分钟
bash test_box_fit_view_data.sh; bash test_gate_3d.sh
echo "== tools/tests 全过 =="
