#!/usr/bin/env bash
# 把一轮从头跑到最后一道闸, 闸口自动过(验证并行调度用, 不是日常用法 —— 日常是人在网页上看闸口)。
#   tools/run_to_end.sh <场景> <轮次> [--indoor] [环境变量=值 ...]      例: tools/run_to_end.sh nfl1 r2 PARALLEL=0
# 第 2 道闸按 DROP_WORDS 把词整词作废(默认是 nfl1 r1 当时作废的那五个), 其他闸不改决定。
set -eu
SC=$1; RUN=$2; shift 2
KIND=outdoor; IN=""
if [ "${1:-}" = "--indoor" ]; then KIND=indoor; IN="--indoor"; shift; fi
for kv in "$@"; do export "$kv"; done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=$HOME/miniforge3/envs/sam3d/bin/python
if [ "$KIND" = indoor ]; then R1=$HOME/sam3d/data/indoorOutput/$SC/$RUN; else R1=$HOME/sam3d/data/outdoor/result/$SC/$RUN; fi
DROP_WORDS=${DROP_WORDS-football,gloves,helmet,jersey,referee}      # 传空字符串 = 一个词都不作废
[ -e "$R1" ] && { echo "轮次已经存在, 不覆盖: $R1"; exit 1; }
mkdir -p "$R1/logs"
$PY -c "import sys; sys.path.insert(0, '$HOME/sam3d/pipeline'); import _common; _common.update_run_config('$R1', '$KIND', {'PEOPLE': '1', 'RECALL': '0'}, 0)"
T0=$(date +%s)
while true; do
  LOG=$(bash "$HERE/drive_go.sh" "$SC" "$RUN" $IN | sed -n 's/^已继续: .* 日志 //p')
  [ -n "$LOG" ] || { echo "★ drive_go.sh 没开跑"; exit 1; }
  # 等这一段跑完: 看日志里的 [结束]。不靠 pgrep —— 驱动还没起来时 pgrep 查不到, 会当成已经跑完
  until [ -e "$LOG" ] && grep -q "^\[结束\]" "$LOG"; do sleep 5; done
  grep -q "^\[结束\] .* 退出码 0$" "$LOG" || { echo "★ 失败, 看 $LOG"; tail -5 "$LOG"; exit 1; }
  read -r SEG STOP < <($PY -c "import json; s=json.load(open('$R1/logs/.drive_state.json')); print(s['段'], int(s['停在闸口']))")
  echo "[$(( $(date +%s) - T0 ))s] 停在第 $((SEG + 1)) 道闸"
  grep -q "全部跑完" "$LOG" && break
  [ "$STOP" = 1 ] || { echo "★ 没停在闸口也没跑完, 看 $LOG"; exit 1; }
  if [ "$SEG" = 1 ]; then
    $PY - "$R1" "$DROP_WORDS" <<'EOF'
import json, sys
R1, words = sys.argv[1], set(sys.argv[2].split(","))
p = f"{R1}/05_review.json"; d = json.load(open(p)); seen = set()
for c in d["候选"]:
    if c["词"] in words and c["词"] not in seen:
        c["决定"] = "drop-word"; seen.add(c["词"])
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
print(f"    闸 2: 整词作废 {sorted(seen)}; 词表里没有的 {sorted(words - seen)}")
EOF
  fi
  LAST=$($PY -c "import sys; sys.path.insert(0, '$HOME/sam3d/.claude/skills/run-recon-pipeline'); import drive; print(len(drive.SEGMENTS['$KIND']) - 1)")
  [ "$SEG" = "$LAST" ] && break
done
echo "总用时 $(( $(date +%s) - T0 )) 秒(含闸口之间的切换, 不含人看的时间)"
column -t -s$'\t' "$R1/logs/timing.tsv"
