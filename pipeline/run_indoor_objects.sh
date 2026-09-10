#!/usr/bin/env bash
# ★★ 已被 pipeline/indoor_pipeline.py --chain obj 取代, 不要再用这个跑。★★
# 保留只为查阅当时用了哪些阈值。它和新驱动有一处**行为差异**, 照着跑会得到坏结果:
#   这里的组装用 hybrid_place_5090.py (逐轴拉伸填满盒子), 新驱动用 outdoor 的
#   assemble_scene.py (按高度算标量、等比缩放)。物体链的盒子长宽不可信, 逐轴拉伸会把
#   吧凳压成 0.94x0.31 的板子 —— 这就是 o1 那一版椅子畸形的原因。
# 另外这里整个漏了 world_up 那一步, 下游只能退回硬编码的"首帧相机 -y"。
# cyberpunk 室内物体链: VLM词表(已有) -> SAM3跟踪 -> 挑图 -> VLM验图 -> SAM3D生成
#   -> 量mesh比例 -> 反投影量盒子 -> 逐实例生成 -> 转Hybrid布局 -> 归位组装 -> 渲染回图 -> 墙
# 吃显存的步骤全部走 /tmp/outdoor_pipeline_gpu.lock (和 outdoor_pipeline.py 同一把锁)。
set -u
cd ~/sam3d
S=data/outdoor/scripts
R1=$PWD/data/indoorOutput/cyberpunk/o1
FR=$PWD/data/indoorOutput/cyberpunk/frames_o1
GEN=$PWD/sam-3d-objects
TAG=cyb_o1
L=$R1/logs; mkdir -p "$L"
P3=~/miniforge3/envs/sam3;      P3D=~/miniforge3/envs/sam3d
PT2=~/miniforge3/envs/trellis2; PLM=~/miniforge3/envs/spatiallm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0

# ---- 阈值 (室内, 依据见 plan; 街景那套在密集摆放下会把一桌四椅并成一个) ----
export MIN_POINTS=120 CONF_MIN=1.2 VOX=0.06 DUP_TH=0.70 PART_TH=0.70 SAMPLE_FRAMES=8
export MIN_AREA_PCT=0.05 MIN_FRAMES=2 TOPK=8
export SOLO="grand piano,bar counter,sofa,potted palm,shelf" SOLO_KEEP="" TH=0.1
export SCENE_DESC="photo of an indoor bar lounge"
LAYOUT_EXCLUDE="screen"     # 贴墙的平面, 当独立物体摆必错 (同 outdoor 排 sign/billboard)

say () { echo; echo "########## $* ($(date '+%m-%d %H:%M:%S')) ##########"; }
# GPU 步骤: 拿锁再跑; OOM 了等 5 分钟重试, 最多 3 次
gpu () {
  local tries=0
  while :; do
    ( flock 9; "$@" ) 9>>/tmp/outdoor_pipeline_gpu.lock && return 0
    tries=$((tries+1))
    [ $tries -ge 3 ] && { echo "!! 连续 $tries 次失败, 放弃: $*"; return 1; }
    echo "!! 失败(多半是 OOM/抢卡), 5 分钟后第 $((tries+1)) 次重试"; sleep 300
  done
}

say "1 SAM3 视频跟踪 (13 个词)"
gpu env R1="$R1" FR="$FR" $P3/bin/python $S/track_all.py > "$L/03_track.log" 2>&1
~/miniforge3/envs/sam3d/bin/python - <<PY
import json;d=json.load(open("$R1/03_track_report.json"))
w=d if isinstance(d,list) else d.get("words",d)
print("[跟踪结果]");  print(json.dumps(w,ensure_ascii=False)[:1500])
PY

say "2 每类挑生成样本"
env TAG=$TAG R1="$R1" FR="$FR" $P3D/bin/python $S/pick_from_masks.py > "$L/04_pick.log" 2>&1; tail -3 "$L/04_pick.log"

say "3 VLM 验图 (抠出来的到底是不是那个东西)"
gpu env TAG=$TAG R1="$R1" FR="$FR" $PT2/bin/python $S/verify_picks.py > "$L/04_verify.log" 2>&1; tail -5 "$L/04_verify.log"

say "4 SAM3D 每类生成一个 mesh"
for d in $GEN/my_images/${TAG}_*/; do
  n=$(basename "$d"); [ -f "$GEN/outputs/$n/$n.glb" ] && { echo "  $n 已有, 跳过"; continue; }
  echo "  生成 $n"
  gpu env CONDA_PREFIX=$P3D HF_HUB_OFFLINE=1 bash -c "cd $GEN && $P3D/bin/python run.py --image my_images/$n/frame.png --mask my_images/$n/mask.png --out $n --hq" >> "$L/05_gen.log" 2>&1 || echo "  !! $n 生成失败"
done
ls -d $GEN/outputs/${TAG}_*/ 2>/dev/null | wc -l | xargs echo "  已有 mesh 数:"

say "5 量 mesh 长宽高比例 (量盒子跨词去重要用)"
$P3D/bin/python - <<PY > "$L/05_ratio.log" 2>&1
import os,glob,json,trimesh,numpy as np
r={}
for p in sorted(glob.glob("$GEN/outputs/${TAG}_*/*.glb")):
    n=os.path.basename(os.path.dirname(p))[len("${TAG}")+1:].replace("_"," ")
    if n.split(" ")[-1].isdigit(): continue          # 实例专属 mesh 不参与
    s=trimesh.load(p); m=s.to_geometry() if hasattr(s,"to_geometry") else s
    e=np.sort(np.array(m.extents))[::-1]
    r[n]=(e/max(e[0],1e-9)).tolist()
json.dump(r,open("$R1/05_mesh_ratio.json","w"),indent=1); print(f"[done] {len(r)} 类")
PY
tail -1 "$L/05_ratio.log"

say "6 反投影量盒子 + 跨词去重 (纯 CPU)"
env R1="$R1" $P3D/bin/python $S/boxes_from_masks.py > "$L/06_boxes.log" 2>&1; tail -6 "$L/06_boxes.log"

say "7 主体逐实例生成"
env TAG=$TAG R1="$R1" FR="$FR" $P3D/bin/python $S/gen_per_instance.py > "$L/06_solo.log" 2>&1; tail -3 "$L/06_solo.log"
if [ -s "$L/solo_jobs.txt" ]; then
  for n in $(cat "$L/solo_jobs.txt"); do
    [ -f "$GEN/outputs/$n/$n.glb" ] && continue
    echo "  生成 $n"
    gpu env CONDA_PREFIX=$P3D HF_HUB_OFFLINE=1 bash -c "cd $GEN && $P3D/bin/python run.py --image my_images/$n/frame.png --mask my_images/$n/mask.png --out $n --hq" >> "$L/06_gen_solo.log" 2>&1 || echo "  !! $n 失败"
  done
fi

say "8 盒子转 Hybrid 布局 + 体检"
$P3D/bin/python pipeline/boxes_to_layout.py --boxes "$R1/06_all_boxes.json" \
   --zup "$R1/03_点云_z朝上.json" --out "$R1/06_布局.txt" --exclude "$LAYOUT_EXCLUDE" \
   --map-out "$R1/06_盒子映射.json"
# box_health.py 2026-09-09 搬到了 tools/, 看结果的脚本一律集中在那边
$PLM/bin/python tools/box_health.py --layout "$R1/06_布局.txt" --pcd "$R1/03_点云_z朝上.ply" \
   --csv "$R1/06_盒子体检.csv" | tail -25

say "9 归位 + 组装整屋"
$P3D/bin/python pipeline/place_map_from_boxes.py --map-json "$R1/06_盒子映射.json" \
   --tag $TAG --outputs "$GEN/outputs" --out "$L/place_map.txt"
$P3D/bin/python hybrid_place_5090.py --layout "$R1/06_布局.txt" --objdir "$GEN/outputs" \
   --map "$(cat $L/place_map.txt)" --out "$R1/07_整屋" --settle \
   --hung-labels "screen,shelf" \
   --support-labels "dining_table,side_table,bar_counter,sofa,booth_seat,shelf" 2>&1 | tail -20

say "10 渲染回图验收"
$P3D/bin/python pipeline/render_check_indoor.py --glb "$R1/07_整屋/00_整屋.glb" \
   --da3 "$R1/02_DA3点云" --zup "$R1/03_点云_z朝上.json" --frames "$FR" \
   --out "$R1/08_照片对比.jpg" --n 4 2>&1 | tail -3

say "11 墙 (SpatialLM arch 模式, 只标墙门窗) + 灰墙壳"
gpu bash -c "cd ~/sam3d/SpatialLM && $PLM/bin/python inference.py --point_cloud '$R1/03_点云_z朝上.ply' \
   --output '$R1/09_墙布局.txt' --detect_type arch --model_path manycore-research/SpatialLM1.1-Qwen-0.5B --seed 0" \
   > "$L/09_walls.log" 2>&1 && echo "  墙 $(grep -c '=Wall' $R1/09_墙布局.txt) 门 $(grep -c '=Door' $R1/09_墙布局.txt) 窗 $(grep -c '=Window' $R1/09_墙布局.txt)"
if [ -f "$R1/09_墙布局.txt" ] && [ "$(grep -c '=Wall' $R1/09_墙布局.txt)" -gt 0 ]; then
  # ★ SpatialLM 的输出末尾没有换行符, 直接 cat 会把最后一行窗户和 bbox_0 粘成一行,
  #   正则匹配不上 -> hybrid_place 报 KeyError: 0。awk 1 保证每个文件都以换行结尾。
  awk 1 "$R1/09_墙布局.txt" "$R1/06_布局.txt" > "$R1/09_布局_物体加墙.txt"
  $P3D/bin/python hybrid_place_5090.py --layout "$R1/09_布局_物体加墙.txt" --objdir "$GEN/outputs" \
     --map "$(cat $L/place_map.txt)" --out "$R1/10_整屋含墙" --settle --wall-plain \
     --hung-labels "screen,shelf" \
     --support-labels "dining_table,side_table,bar_counter,sofa,booth_seat,shelf" 2>&1 | tail -6
  $P3D/bin/python pipeline/render_check_indoor.py --glb "$R1/10_整屋含墙/00_整屋.glb" \
     --da3 "$R1/02_DA3点云" --zup "$R1/03_点云_z朝上.json" --frames "$FR" \
     --out "$R1/10_照片对比_含墙.jpg" --n 4 2>&1 | tail -2

  say "12 物理: 拟合地面 + 落地 + 推开穿模 (物体+墙+地面 一个文件)"
  # ★ 必须跑在含墙那版上。hybrid_place 的 --settle 只做落地, 而且场景里没有地面可参照;
  #   穿模那步室内这边压根没接过。脚本会把 wall_* 节点原样带过去, 只对物体做物理。
  $P3D/bin/python pipeline/scene_phys_indoor.py --glb "$R1/10_整屋含墙/00_整屋.glb" \
     --out "$R1/11_整屋_物理.glb" 2>&1 | tail -6
  $P3D/bin/python pipeline/render_check_indoor.py --glb "$R1/11_整屋_物理.glb" \
     --da3 "$R1/02_DA3点云" --zup "$R1/03_点云_z朝上.json" --frames "$FR" \
     --out "$R1/11_照片对比_物理含墙.jpg" --n 4 2>&1 | tail -2
fi

say "全部完成"
ls -la "$R1" | grep -vE "^total|logs|02_da3$"
