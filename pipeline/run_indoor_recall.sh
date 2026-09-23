#!/usr/bin/env bash
# ★★ 已被 pipeline/indoor_pipeline.py --chain obj 取代, 不要再用这个跑。★★
# 保留只为查阅当时用了哪些阈值。它和新驱动有一处**行为差异**, 照着跑会得到坏结果:
#   这里的组装用 hybrid_place_5090.py (逐轴拉伸填满盒子), 新驱动用 outdoor 的
#   assemble_scene.py (按高度算标量、等比缩放)。物体链的盒子长宽不可信, 逐轴拉伸会把
#   吧凳压成 0.94x0.31 的板子 —— 这就是 o1 那一版椅子畸形的原因。
# 另外这里整个漏了 world_up 那一步, 下游只能退回硬编码的"首帧相机 -y"。
# cyberpunk 室内召回支路: WildDet3D 直接看原图找 SAM3 漏掉的东西。
#   detect -> recall(聚类+和SAM3比对) -> recall_mask(框喂回SAM3补抠) -> gen -> merge
# 主链 run_indoor_objects.sh 只跑到 render, 这 5 步当初没接进去 —— 高背椅/卡座/吧台
# 全部丢失就是因为 SAM3 对 armchair/sofa/booth seat/couch/bench/bar counter 返回 0 实例,
# 而 geometric 那条(boxes_from_wilddet.py)吃的是 SAM3 的框, 看不见漏检。
#
# 室内和街景的差别 (改了两处阈值):
#   DIST_F  0.50 -> 0.35   街上车距 2m, 酒吧椅子间距 0.6m, 0.5 会把相邻两把并成一个
#   DET_STRIDE 5 -> 3      只有 80 帧, 抽稀了单个物体凑不够 MIN_VIEWS
# 没改的: UP 在 detect/merge_recall 里硬编码 [0,-1,0], 实测真实 up 只偏 4.6°
#         (outdoor 的报警线是 26°), 不值得为此动 outdoor 的脚本。
set -u
cd ~/sam3d
S=data/outdoor/scripts
R1=$PWD/data/indoorOutput/cyberpunk/o1
FR=$PWD/data/indoorOutput/cyberpunk/frames_o1
GEN=$PWD/sam-3d-objects
TAG=cyb_o1
L=$R1/logs; mkdir -p "$L"
P3=~/miniforge3/envs/sam3;  P3D=~/miniforge3/envs/sam3d;  PWD3=~/miniforge3/envs/wilddet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=8.6

export DET_STRIDE=3 DET_SCORE=0.35
export MIN_VIEWS=3 DIST_F=0.35 MATCH_F=0.7
export REC_MIN_PCT=0.15 REC_MIN_IOU=0.5

say () { echo; echo "########## $* ($(date '+%m-%d %H:%M:%S')) ##########"; }
gpu () {
  local tries=0
  while :; do
    ( flock 9; "$@" ) 9>>/tmp/outdoor_pipeline_gpu.lock && return 0
    tries=$((tries+1))
    [ $tries -ge 3 ] && { echo "!! 连续 $tries 次失败, 放弃: $*"; return 1; }
    echo "!! 失败(多半是 OOM/抢卡), 5 分钟后第 $((tries+1)) 次重试"; sleep 300
  done
}

say "1 detect: WildDet3D 全图按词检测 (不吃 SAM3 的框)"
gpu env R1="$R1" FR="$FR" WD3="$PWD/WildDet3D" DET_STRIDE=$DET_STRIDE DET_SCORE=$DET_SCORE \
    $PWD3/bin/python $S/detect_wilddet.py > "$L/12_detect.log" 2>&1 || exit 1
tail -3 "$L/12_detect.log"

say "2 recall: 逐帧检测聚成实例 + 和 SAM3 已有实例比对"
env R1="$R1" MIN_VIEWS=$MIN_VIEWS DIST_F=$DIST_F MATCH_F=$MATCH_F \
    $P3D/bin/python $S/merge_recall.py > "$L/13_recall.log" 2>&1 || exit 1
tail -12 "$L/13_recall.log"

say "3 recall_mask: 漏检的框喂回 SAM3 补抠 mask"
gpu env R1="$R1" FR="$FR" TAG=$TAG REC_MIN_PCT=$REC_MIN_PCT REC_MIN_IOU=$REC_MIN_IOU \
    $P3/bin/python $S/mask_recall.py > "$L/14_recall_mask.log" 2>&1 || exit 1
tail -8 "$L/14_recall_mask.log"

say "4 recall_gen: 给漏检实例生成 mesh"
JOBS=$(cat "$L/recall_jobs.txt" 2>/dev/null || true)
if [ -z "$JOBS" ]; then echo "[跳过] 没有待生成的漏检实例"; else
  for n in $JOBS; do
    [ -f "$GEN/outputs/$n/$n.glb" ] && { echo "  $n 已有, 跳过"; continue; }
    echo "  生成 $n"
    gpu env CONDA_PREFIX=$P3D HF_HUB_OFFLINE=1 bash -c \
      "cd $GEN && $P3D/bin/python run.py --image my_images/$n/frame.png --mask my_images/$n/mask.png --out $n --hq" \
      >> "$L/14_recall_gen.log" 2>&1 || echo "  !! $n 生成失败"
  done
fi

say "5 recall_merge: 盒子并进 06_all_boxes_merged.json"
env R1="$R1" TAG=$TAG $P3D/bin/python $S/merge_boxes_recall.py > "$L/15_merge_boxes.log" 2>&1 || exit 1
tail -5 "$L/15_merge_boxes.log"

say "召回支路完成"
