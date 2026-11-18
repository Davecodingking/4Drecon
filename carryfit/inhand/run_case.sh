#!/usr/bin/env bash
# 一个物体跑全套: 挂手拟合 -> 手臂 IK -> 指标 + 四格视频。  run_case.sh <r1> <词> <实例号> <网格 glb> <输出目录> [fit_attach.py 的其它参数]
set -eo pipefail
cd /home/ubuntu/sam3d
R1=$1; WORD=$2; OID=$3; MESH=$4; OUT=$5; shift 5
SUB=${WORD// /_}_$OID
mkdir -p "$OUT"
INIT=""; [ -f "$R1/13_carry/$SUB/carry_poses.npz" ] && INIT="--init_poses $R1/13_carry/$SUB/carry_poses.npz"
MASK=""; [ -f "$R1/13_carry/$SUB/mask.npz" ] && MASK="--masks $R1/13_carry/$SUB/mask.npz"
carryfit/run.sh carryfit/inhand/fit_attach.py --r1 "$R1" --word "$WORD" --oid "$OID" --mesh "$MESH" $MASK $INIT --out "$OUT" "$@" 2>&1 | grep --line-buffered -v -i "warn\|pynvml"
carryfit/run.sh carryfit/inhand/arm_ik.py --r1 "$R1" --targets "$OUT/wrist_targets.npz" --out "$OUT" 2>&1 | grep --line-buffered -v -i "warn\|pynvml"
carryfit/run.sh carryfit/inhand/metrics.py --r1 "$R1" --poses "$OUT/carry_poses.npz" --people_params "$OUT/people_refined.npz" --out "$OUT" --video 2>&1 | grep --line-buffered -v -i "warn\|pynvml" | cut -c1-600
