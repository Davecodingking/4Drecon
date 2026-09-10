#!/usr/bin/env bash
# 实时看流水线在跑什么。用法:  tools/watch_run.sh [场景] [轮次]
#
# ★ 为什么需要它: drive.py 的输出一旦被管道接走(| tail、> 文件)就会被 stdio 缓冲吃掉,
#   等它吐出来的时候活早干完了。所以别去追进程的 stdout —— 追**每一步自己写的日志文件**,
#   那些是无缓冲落盘的, 而且步骤换了文件名就换, 天然告诉你"现在在跑哪一步"。
set -u
SC=${1:-pedestrian}; RUN=${2:-r2}
R1=$HOME/sam3d/data/outdoor/result/$SC/$RUN
[ "${3:-}" = "--once" ] || exec watch -n2 -t "$0" "$SC" "$RUN" --once

hr(){ printf '%.0s─' $(seq 1 72); echo; }

# --- 驱动器还在不在 ---
if pgrep -f "[d]rive.py.*--scene $SC" >/dev/null; then
  ST="running"
else
  ST="not running  (结束了, 或者根本没起来)"
fi
S=$R1/logs/.drive_state.json
SEG=$( [ -f "$S" ] && tr -d ' "' < "$S" | grep -E '段|停在闸口' | tr '\n' ' ' )
echo "$SC/$RUN   driver: $ST"
echo "state: $SEG"
hr

# --- 逐实例生成的进度条(solo 那一步最慢, 值得单独算) ---
JOBS=$R1/logs/solo_jobs.txt
if [ -f "$JOBS" ]; then
  N=$(grep -c . "$JOBS")
  D=$(cd "$HOME/sam3d/sam-3d-objects/outputs" 2>/dev/null &&
      grep -c . <(for j in $(cat "$JOBS"); do [ -d "$j" ] && echo "$j"; done))
  W=$(( D * 40 / (N>0?N:1) ))
  printf 'solo mesh  '
  printf '%.0s#' $(seq 0 $W); printf '%.0s.' $(seq $W 39)
  printf '  %d/%d\n' "$D" "$N"
  hr
fi

# --- 最新的那个日志 = 正在跑的那一步 ---
L=$(ls -t "$R1"/logs/*.log 2>/dev/null | head -1)
if [ -n "$L" ]; then
  echo "step: $(basename "$L")   ($(( $(date +%s) - $(stat -c %Y "$L") ))s 前写过)"
  hr
  # 只留有信息的行: 进度百分比和 kaolin 的 warning 刷屏没意义
  grep -vE '% done|UserWarning|warnings.warn|^\s*$' "$L" | tail -14
fi
