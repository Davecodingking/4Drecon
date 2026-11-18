#!/usr/bin/env bash
# carryfit 测试场景用: 在 tmux 里跑一段流水线驱动, 输出写进 <R1>/logs/drive_*.log(和 tools/drive_go.sh 同一套日志格式)
# 用法: drv.sh <scene> <pool> [--indoor] [--go]
SC=$1; POOL=$2; shift 2
cd /home/ubuntu/sam3d
source ~/.bashrc >/dev/null 2>&1
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 GPU_POOL=$POOL MAX_GPUS=4
if [[ " $* " == *" --indoor "* ]]; then R1=data/indoorOutput/$SC/r1; else R1=data/outdoor/result/$SC/r1; fi
LOG=$R1/logs/drive_$(date +%m%d_%H%M%S).log
echo "[开始] $(date '+%F %T')" > $LOG
~/miniforge3/envs/sam3d/bin/python -u .claude/skills/run-recon-pipeline/drive.py --scene $SC "$@" >> $LOG 2>&1
echo "[结束] $(date '+%F %T') 退出码 $?" >> $LOG
