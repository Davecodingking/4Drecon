#!/usr/bin/env bash
# 领一张卡(和流水线同一套锁)跑 carryfit 的脚本, 参数原样传过去。
# 用法: carryfit/run.sh --r1 ... --word box --oid 1 --mesh ... --out carryfit/runs/<名字> [--ref ...]   (fit.py, fpose 环境)
#       CF_ENV=sam3 carryfit/run.sh carryfit/retrack.py --r1 ... --word ... --oid ... --out ...      (别的脚本 + 别的环境)
set -eo pipefail
NEED_MB=${NEED_MB:-12000}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CARD=""
until [ -n "$CARD" ]; do
  for row in $(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -n); do
    IFS=, read -r idx used total <<< "$row"
    if [ -n "${CF_CARDS:-}" ] && [[ ",$CF_CARDS," != *",$idx,"* ]]; then continue; fi   # CF_CARDS=7 只用这几张卡
    exec 9>>"/tmp/outdoor_pipeline_gpu${idx}.lock"
    if flock -n 9 && [ $(( total - used )) -ge "$NEED_MB" ]; then CARD=$idx; break; fi
    exec 9>&-
  done
  [ -n "$CARD" ] || sleep 10
done
export CUDA_VISIBLE_DEVICES=$CARD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYOPENGL_PLATFORM=egl
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate ${CF_ENV:-fpose}
cd /home/ubuntu/sam3d
echo "[carryfit] card $CARD $(date '+%H:%M:%S')"
if [[ "${1:-}" == *.py ]]; then SCRIPT=$1; shift; else SCRIPT=carryfit/fit.py; fi
python -u "$SCRIPT" "$@"
