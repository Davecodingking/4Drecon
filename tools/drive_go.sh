#!/usr/bin/env bash
# 人看完闸口, 继续跑下一段 —— 放进后台 tmux, 关掉终端也不断; 输出写进 logs/drive_<时间>.log,
# watch_run.py 看板靠这份日志的 [开始]/[结束] 和步骤行画进度, 每步耗时另记在 logs/timing.tsv。
#
#   tools/drive_go.sh                       <- 继续"上一次跑的那一轮"(常用)
#   tools/drive_go.sh <场景> <轮次> [--indoor]
#
# 跑起来之后:
#   bash tools/watch_run.sh       看板(不带参数 = 当前这轮)
#   tmux attach -t run_<场景>_<轮次>   驱动本身, 一般不用看, 输出都在日志里
#
# ★ 会话名: 看板统一叫 watch(见 watch_run.sh), 不再按场景起名 watch_<场景> ——
#   按场景起名的后果是同时存在好几个板子, 人 attach 过去看到的是上一个场景的, 而且不报错。
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAST="$HERE/.last_run"
if [ $# -ge 1 ]; then
  SC=$1; RUN=${2:?轮次}; IN=${3:-}
elif [ -r "$LAST" ]; then
  IFS=$'\t' read -r SC RUN KIND < "$LAST"
  IN=""; [ "${KIND:-}" = "indoor" ] && IN="--indoor"
  echo "继续上一次那轮: $SC $RUN ${IN:-(室外)}"
else
  echo "还没跑过任何一轮。用法: tools/drive_go.sh <场景> <轮次> [--indoor]"; exit 1
fi
PY=$HOME/miniforge3/envs/sam3d/bin/python
if [ "$IN" = "--indoor" ]; then R1=$HOME/sam3d/data/indoorOutput/$SC/$RUN; else R1=$HOME/sam3d/data/outdoor/result/$SC/$RUN; fi
[ -d "$R1" ] || { echo "没有这个轮次: $R1"; exit 1; }
if pgrep -f "drive.py.*--scene $SC( |$).*--run $RUN" >/dev/null; then echo "$SC/$RUN 的驱动已经在跑了"; exit 1; fi
LOG=$R1/logs/drive_$(date +%m%d_%H%M%S).log
# ★ 会话名带轮次: 原来只按场景起名、开跑前先杀同名会话 —— 本轮页能新开轮次之后,
#   同一场景开第二轮会把第一轮正在跑的驱动杀掉(上面的 pgrep 只查了这一轮)。
S="run_${SC}_${RUN}"
tmux kill-session -t "$S" 2>/dev/null || true
# ★ 挑卡: 调用方没给 CUDA_VISIBLE_DEVICES 就挑一张空的(没有别的驱动在用、没人占着显存),
#   每张卡一把 GPU 锁, 这样各轮次才真的并行。给了就照用(想钉某张卡: CUDA_VISIBLE_DEVICES=3 drive_go.sh ...)
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$($PY -c "import sys; sys.path.insert(0, '$HERE'); import run_ctl; print(run_ctl.pick_gpu())") \
    || { echo "挑不到空闲的卡, 没开跑"; exit 1; }
  export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID
fi
echo "用 GPU $CUDA_VISIBLE_DEVICES"
# ★ GPU 选择显式带进去: tmux 会话继承的是 tmux 服务端的环境, 不是调这个脚本的人的 ——
#   不带的话会落到默认的 0 号卡(见 _common.wait_gpu 的注释)
GPU=""
for v in CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER; do [ -n "${!v:-}" ] && GPU="$GPU $v=${!v}"; done
tmux new-session -d -s "$S" \
  "{ echo \"[开始] \$(date '+%F %T')\"; env $GPU $PY -u $HOME/sam3d/.claude/skills/run-recon-pipeline/drive.py --scene $SC --run $RUN $IN --go; echo \"[结束] \$(date '+%F %T') 退出码 \$?\"; } > $LOG 2>&1"
echo "已继续: $SC/$RUN  日志 $LOG"
echo "看进度: bash $HERE/watch_run.sh        (Ctrl-b d 退出看板, 不影响运行)"
