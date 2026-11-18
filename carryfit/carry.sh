#!/usr/bin/env bash
# carryfit 一条命令: 找物体网格 -> 重跟这个物体的 mask(SAM3) -> 拟合每帧位姿 -> 加进流程导出的 blend(另存)。
#
#   carryfit/carry.sh <r1 目录> <词> <实例号> [fit.py 的其它参数...]
#   例: carryfit/carry.sh data/indoorOutput/goodfellas_tray_indoor/r1 tray 3
#
# 网格: 默认用流水线给这个实例生成的(05_review.json 里这一条的"生成名" -> sam-3d-objects/outputs/<生成名>/*.glb);
#       没生成过(比如闸口点了不要)就用环境变量 MESH=<glb> 指定。
# 输出: carryfit/runs/<场景>_<词><实例号>/  (NAME=... 可改), 里面 preview.mp4 是要看的视频。
set -eo pipefail
cd /home/ubuntu/sam3d
R1=${1:?r1 目录}; WORD=${2:?词}; OID=${3:?实例号}; shift 3
SCENE=$(basename "$(dirname "$(realpath "$R1")")")
W_=${WORD// /_}
NAME=${NAME:-${SCENE}_${W_}${OID}}
OUT=carryfit/runs/$NAME
mkdir -p "$OUT"
if [ -z "${MESH:-}" ]; then
  GEN=$(python3 -c "
import json, sys
d = json.load(open('$R1/05_review.json'))
m = [x['生成名'] for x in d['候选'] if x['词'] == '$WORD' and int(x.get('实例', -1)) == $OID]
print(m[0] if m else '')")
  MESH=$(ls sam-3d-objects/outputs/"$GEN"/*.glb 2>/dev/null | head -1 || true)
  [ -n "$MESH" ] || { echo "[失败] 没找到 $WORD #$OID 的网格(生成名 '$GEN'); 用 MESH=<glb> 指定"; exit 1; }
fi
echo "[carry] $SCENE  $WORD #$OID  网格 $MESH  -> $OUT"
T0=$(date +%s)
if [ -f "$OUT/mask.npz" ] && [ -z "${RETRACK:-}" ]; then
  echo "[carry] 已有重跟的 mask, 不重跟(RETRACK=1 强制重跟)"
else
  CF_ENV=sam3 NEED_MB=10000 carryfit/run.sh carryfit/retrack.py --r1 "$R1" --word "$WORD" --oid "$OID" --out "$OUT/mask.npz" \
    2>&1 | grep --line-buffered "种子\|重跟\|失败\|Error"
fi
carryfit/run.sh --r1 "$R1" --word "$WORD" --oid "$OID" --mesh "$MESH" --masks "$OUT/mask.npz" --out "$OUT" "$@" \
  2>&1 | grep --line-buffered -v "Warning\|pynvml\|warn" | tee "$OUT/log.txt" | grep --line-buffered "^\[\|总用时\|Error\|Traceback"
BLEND=$(ls "$R1"/12_motion/*_scene_people.blend 2>/dev/null | head -1 || true)
if [ -n "$BLEND" ]; then
  ~/venvs/bpy/bin/python carryfit/to_blend.py --r1 "$R1" --poses "$OUT/carry_poses.npz" --name "carried_${W_}_${OID}" \
    --out "$OUT/${SCENE}_scene_people_carry.blend" 2>&1 | grep "自检\|saved\|Error"
else
  echo "[carry] 这一轮还没导出 blend(12_motion), 跳过写 blend"
fi
echo "[carry] 完成 $(( $(date +%s) - T0 )) s:  $OUT/preview.mp4"
