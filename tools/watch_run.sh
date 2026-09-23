#!/usr/bin/env bash
# 实时看流水线在跑什么。
#
#   tools/watch_run.sh                      <- 看"当前正在跑的那一轮"(常用)
#   tools/watch_run.sh <场景> <轮次> [--indoor]   <- 指定看哪一轮
#
# 真正的实现在 watch_run.py(rich 看板): 段 / 步骤耗时 / 当前物体进度 / 显卡 / 闸口。只读, 随时开关。
#
# ★ 不带参数时看哪一轮: 读 tools/.last_run —— drive.py 每次 --go 会把
#   "场景 轮次 室内还是室外" 写进去。以前这里写死默认 pedestrian r2, 结果人跑着 office
#   却看到 pedestrian 的板子, 而且不会报错, 只会显示一堆"earlier"。
#
# ★ 自己挂进 tmux(会话名 watch)。为什么由这里做而不是 drive.py:
#   drive.py 每跑完一段就退出, 看板的命比它长 —— 塞进驱动会变成"跑一段重启一次看板"。
#   会话已存在就换掉里面的窗口(不新开, 也不会留着盯旧场景)。
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${WATCH_SESSION:-watch}"
PY="$HOME/miniforge3/envs/sam3d/bin/python"
APP="$HERE/watch_run.py"
LAST="$HERE/.last_run"

if [ $# -ge 1 ]; then
  SCENE="$1"; RUN="${2:-r1}"; EXTRA="${3:-}"
elif [ -r "$LAST" ]; then
  IFS=$'\t' read -r SCENE RUN KIND < "$LAST"
  EXTRA=""; [ "${KIND:-}" = "indoor" ] && EXTRA="--indoor"
  echo "看板: 当前这轮 = $SCENE $RUN ${EXTRA:-(室外)}   (来自 tools/.last_run)"
else
  echo "还没有跑过任何一轮 —— 先 drive.py --go, 或者自己指定:"
  echo "  tools/watch_run.sh <场景> <轮次> [--indoor]"
  exit 1
fi

run_it() { exec "$PY" "$APP" --scene "$SCENE" --run "$RUN" ${EXTRA:+"$EXTRA"}; }

# 已经在 tmux 里(或者没装 tmux / 不是交互终端)就直接跑, 不套娃
if [ -n "${TMUX:-}" ] || ! command -v tmux >/dev/null 2>&1 || [ ! -t 1 ]; then
  run_it
fi

CMD="$(printf '%q ' "$PY" "$APP" --scene "$SCENE" --run "$RUN")"
[ -n "$EXTRA" ] && CMD="$CMD$(printf '%q' "$EXTRA")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux respawn-window -k -t "$SESSION" "$CMD" 2>/dev/null || tmux new-window -t "$SESSION" "$CMD"
else
  tmux new-session -d -s "$SESSION" "$CMD"
fi
echo "已挂进 tmux 会话 '$SESSION'   重新连: tmux attach -t $SESSION   脱离: Ctrl-b d"
exec tmux attach -t "$SESSION"
