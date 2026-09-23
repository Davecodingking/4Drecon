#!/usr/bin/env bash
# =====================================================================================
# 换机器后重新下载模型权重 —— 配套 ~/sam3d/docs/setup/SETUP.md 第 8 节
#
# ★ 只写不跑: 这份脚本是 2026-09-21 在旧机器 ES-MAL-GPU-011 上照着现有权重目录整理的,
#   旧机器上没有执行过。第一次用先 `bash -n download_weights.sh` 查语法, 再按组逐个跑。
#
# 范围: 新机器只支撑 pipeline/outdoor_pipeline.py 和 pipeline/indoor_pipeline.py 的"物体链",
#       外加 body4d。SpatialLM / SpatialGen / InvSR 墙面贴图线 / LayerPano3D / Pano2Room 都不要了,
#       它们的权重在文件末尾整段注释掉; FIRE3D 待用户确认, 放在可选组里不默认下。
#
# 用法:
#   0) 需要 hf 命令(huggingface_hub>=0.34)。环境建好后用环境里的:
#        HF=~/miniforge3/envs/sam3d/bin/hf bash download_weights.sh ...
#      或者随便找个 python 装一个: pip install -U "huggingface_hub[cli]"
#   1) 登录 HF(token 只交给 hf auth login 去存, 不要写进任何脚本/笔记):
#        hf auth login
#      下面 4 个 gated 仓库要先在网页上申请并获批(旧账号 DaveHJC 都批过, 换账号要重新申请):
#        facebook/sam-3d-objects   facebook/sam3   facebook/sam-3d-body-dinov3   facebook/VGGT-Omega
#      没批下来时 hf download 会报 403 GatedRepo。
#   2) 路径变量默认 = 旧机器布局。新机器不一样就用环境变量覆盖, 例如:
#        NVME=/data/jincheng bash download_weights.sh
#      (但代码里有几处写死了 /mnt/nvme/jincheng, 见 SETUP.md 第 2 节; 最省事是让这个路径在新机器上也存在)
#   3) 跑:
#        bash download_weights.sh                 # 下全部默认组
#        bash download_weights.sh sam3d vlm       # 只下列出的组(可选组也要这样点名)
#        bash download_weights.sh --list          # 列出组名
#      默认组: sam3d sam3 vlm da3 wilddet vggt_omega metric3d body4d
#      可选组(不点名不下): body4d_vas(body4d 开补全才用) eval(米制评测脚本用) fire3d(待用户确认)
#
# 已经下完的会跳过(hf download 自己比对; wget 的按目标文件是否存在), 可以反复跑。
# =====================================================================================
set -euo pipefail

SAM3D=${SAM3D:-$HOME/sam3d}
NVME=${NVME:-/mnt/nvme/jincheng}
# 旧机器上 ~/.cache/huggingface 是指向 $NVME/hf 的软链, 没设 HF_HOME。新机器照做(SETUP.md 第 2、7 节)。
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
TORCH_HOME_DEF=${TORCH_HOME_DEF:-$HOME/.cache/torch}          # 默认 TORCH_HOME: SAM3D 的 DINOv2 / Fire3D 的 AnyUp
METRIC3D_TORCH_HOME=${METRIC3D_TORCH_HOME:-$NVME/torch_cache}  # vggt-omega/tools/metric3d_scale.py 里写死
VGGT_OMEGA_DIR=${VGGT_OMEGA_DIR:-$NVME/weights/vggt_omega}     # vggt-omega/tools/vggt_omega_geom.py 里写死
FIRE3D_CKPT=${FIRE3D_CKPT:-$NVME/fire3d_ckpt}                  # ~/sam3d/Fire3D/checkpoints 软链指向这里
HF=${HF:-hf}
PY=${PY:-$HOME/miniforge3/bin/python}                          # 只用来解 zip(机器上没有 unzip)

DEFAULT_GROUPS="sam3d sam3 vlm da3 wilddet vggt_omega metric3d body4d"
OPTIONAL_GROUPS="body4d_vas eval fire3d"
if [ "${1:-}" = "--list" ]; then echo "默认: $DEFAULT_GROUPS"; echo "可选: $OPTIONAL_GROUPS"; exit 0; fi
SEL=" ${*:-$DEFAULT_GROUPS} "
want() { [[ "$SEL" == *" $1 "* ]]; }

command -v "$HF" >/dev/null 2>&1 || {
  echo "找不到 hf 命令。用 HF=<某环境>/bin/hf 指定, 或 pip install -U 'huggingface_hub[cli]'"; exit 1; }

log()  { echo; echo "######## $*"; }
hfdl() { echo "+ $HF download $*"; "$HF" download "$@"; }
fetch() {  # fetch <url> <输出文件>
  local url=$1 out=$2
  if [ -s "$out" ]; then echo "已有, 跳过: $out"; return 0; fi
  mkdir -p "$(dirname "$out")"
  echo "+ wget $url -> $out"
  wget -q --show-progress -c -O "$out.part" "$url" && mv "$out.part" "$out"
}
hub_repo() {  # hub_repo <缓存目录名> <GitHub owner/repo> <分支> <TORCH_HOME>  —— 仿 torch.hub 的缓存布局
  local name=$1 gh=$2 br=$3 th=$4
  local dest="$th/hub/$name"
  if [ -d "$dest" ]; then echo "已有, 跳过: $dest"; return 0; fi
  local zip="$th/hub/$name.zip"
  fetch "https://github.com/$gh/archive/refs/heads/$br.zip" "$zip"
  "$PY" - "$zip" "$th/hub" "$dest" <<'EOF'
import sys, zipfile, os, shutil
z, root, dest = sys.argv[1:]
with zipfile.ZipFile(z) as f:
    top = f.namelist()[0].split('/')[0]
    f.extractall(root)
shutil.move(os.path.join(root, top), dest)
EOF
  rm -f "$zip"
  local trusted=${name%_"$br"}          # torch.hub 的 trusted_list 记 owner_repo
  grep -qx "$trusted" "$th/hub/trusted_list" 2>/dev/null || echo "$trusted" >> "$th/hub/trusted_list"
}

# ------------------------------------------------------------------------------ sam3d
if want sam3d; then
  log "sam3d: SAM 3D Objects 权重 (gated, 13G) -> $SAM3D/sam-3d-objects/checkpoints/hf"
  D=$SAM3D/sam-3d-objects/checkpoints
  if [ ! -f "$D/hf/pipeline.yaml" ]; then
    # 官方 doc/setup.md 的做法: 整个仓库下到 hf-download, 再把里面的 checkpoints/ 挪成 hf/
    hfdl facebook/sam-3d-objects --repo-type model --local-dir "$D/hf-download" --max-workers 1
    rmdir "$D/hf" 2>/dev/null || true
    mv "$D/hf-download/checkpoints" "$D/hf" && rm -rf "$D/hf-download"
  else
    echo "已有, 跳过: $D/hf"
  fi

  log "sam3d: MoGe v1 (checkpoints/hf/pipeline.yaml 里的深度模型, 1.2G, HF 缓存)"
  hfdl Ruicheng/moge-vitl

  log "sam3d: DINOv2 主干 (torch.hub facebookresearch/dinov2 dinov2_vitl14_reg, 1.2G)"
  # 生成步骤带 HF_HUB_OFFLINE=1 跑, 这两样不预先放好第一次会去 GitHub/fbaipublicfiles 拉
  hub_repo facebookresearch_dinov2_main facebookresearch/dinov2 main "$TORCH_HOME_DEF"
  fetch https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth \
        "$TORCH_HOME_DEF/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
fi

# ------------------------------------------------------------------------------ sam3
if want sam3; then
  log "sam3: SAM 3 (gated, 3.3G) -> $SAM3D/sam3/checkpoints/sam3.pt  (旧机器快照 3c879f3)"
  hfdl facebook/sam3 sam3.pt --local-dir "$SAM3D/sam3/checkpoints"
fi

# ------------------------------------------------------------------------------ vlm
if want vlm; then
  log "vlm: Qwen3-VL-8B-Instruct (17G, HF 缓存; 流水线 vlm/verify/vlm_sizes 三步, 跑在 sam3d 环境)"
  hfdl Qwen/Qwen3-VL-8B-Instruct
fi

# ------------------------------------------------------------------------------ da3
if want da3; then
  log "da3: Depth Anything 3 nested 1.1 (6.3G, CC BY-NC, HF 缓存; da3_to_ply.py 的默认模型, spatiallm 环境)"
  hfdl depth-anything/DA3NESTED-GIANT-LARGE-1.1
fi

# ------------------------------------------------------------------------------ wilddet
if want wilddet; then
  log "wilddet: WildDet3D 权重 (4.5G) -> $SAM3D/WildDet3D/ckpt"
  hfdl allenai/WildDet3D wilddet3d_alldata_all_prompt_v1.0.pt --local-dir "$SAM3D/WildDet3D/ckpt"
  log "wilddet: LingBot-Depth 深度后端 (1.2G, HF 缓存; 运行时 hf_hub_download 会自动取, 预下好才能离线)"
  hfdl robbyant/lingbot-depth-postrain-dc-vitl14
fi

# ------------------------------------------------------------------------------ vggt_omega
if want vggt_omega; then
  log "vggt_omega: VGGT-Ω 两个权重 (gated, 8.6G) -> $VGGT_OMEGA_DIR"
  hfdl facebook/VGGT-Omega vggt_omega_1b_512.pt vggt_omega_1b_416_reproduce.pt --local-dir "$VGGT_OMEGA_DIR"
  log "vggt_omega: 天空分割 skyseg.onnx (168M)"
  fetch https://hf-mirror.com/JianyuanWang/skyseg/resolve/main/skyseg.onnx "$VGGT_OMEGA_DIR/skyseg.onnx"
fi

# ------------------------------------------------------------------------------ metric3d
if want metric3d; then
  log "metric3d: Metric3D v2 torch.hub 代码 + Large/Giant2 权重 (6.7G) -> $METRIC3D_TORCH_HOME"
  hub_repo yvanyin_metric3d_main YvanYin/Metric3D main "$METRIC3D_TORCH_HOME"
  fetch https://hf-mirror.com/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth \
        "$METRIC3D_TORCH_HOME/hub/checkpoints/metric_depth_vit_large_800k.pth"     # GEOM=vggt_l(室外默认)
  fetch https://hf-mirror.com/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth \
        "$METRIC3D_TORCH_HOME/hub/checkpoints/metric_depth_vit_giant2_800k.pth"    # GEOM=vggt_g2(对照)
fi

# ------------------------------------------------------------------------------ body4d
if want body4d; then
  B=$SAM3D/sam-body4d/checkpoints       # configs/body4d.yaml 的 ckpt_root 写死成 /home/jincheng/sam3d/sam-body4d/checkpoints
  log "body4d: SAM 3D Body (gated manual, 2.7G)"
  hfdl facebook/sam-3d-body-dinov3 --local-dir "$B/sam-3d-body-dinov3"
  log "body4d: MoGe-2 (估焦距; 必须是 v2, 别拿 moge-vitl 顶, 1.3G)"
  hfdl Ruicheng/moge-2-vitl-normal --local-dir "$B/moge-2-vitl-normal"
  log "body4d: Depth-Anything-V2-Large (给 Diffusion-VAS 的深度条件, 1.3G)"
  hfdl depth-anything/Depth-Anything-V2-Large depth_anything_v2_vitl.pth --local-dir "$B"
  log "body4d: SAM 3 软链到 sam3 那份, 不重复下 (字节数一致)"
  mkdir -p "$B/sam3" && ln -sfn "$SAM3D/sam3/checkpoints/sam3.pt" "$B/sam3/sam3.pt"
  # ViTDet-H 人体检测(2.6G), 只有官方 scripts/offline_app.py 用, 我们的 body4d_*_5090.py 不用; 运行时 detectron2 会自动下:
  # fetch https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl \
  #       "$HOME/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
fi

# ------------------------------------------------------------------------------ body4d_vas (可选)
if want body4d_vas; then
  B=$SAM3D/sam-body4d/checkpoints
  log "body4d_vas: Diffusion-VAS x2 (只有开补全 --completion 1 才用; 32G 卡上开补全本来就 OOM; 共 ~14G)"
  hfdl kaihuac/diffusion-vas-amodal-segmentation --local-dir "$B/diffusion-vas-amodal-segmentation"
  hfdl kaihuac/diffusion-vas-content-completion  --local-dir "$B/diffusion-vas-content-completion"
fi

# ------------------------------------------------------------------------------ eval (可选)
if want eval; then
  # 只给 09-17 的米制尺度评测用(data/indoorOutput/ithor312/analysis_9.17_vggt_omega/metric_eval/), 流水线不用。
  # 旧机器上是 HF_HOME=$NVME/hf_cache 下的; 这里统一放默认 HF 缓存, 跑评测脚本时别再设 HF_HOME 就行。
  log "eval: DA3METRIC-LARGE / UniDepthV2-L / MoGe-2 (共 ~4G, HF 缓存)"
  hfdl depth-anything/DA3METRIC-LARGE
  hfdl lpiccinelli/unidepth-v2-vitl14
  hfdl Ruicheng/moge-2-vitl-normal
fi

# ------------------------------------------------------------------------------ fire3d (可选, 待用户确认)
if want fire3d; then
  # FIRE3D 不在流水线里(09-15 的验证性实验), 要不要保留待用户确认。63G。
  log "fire3d: FIRE3D 模型包 (63G, 公开不 gated) -> $FIRE3D_CKPT/Fire3D  (= fire3d download --models)"
  hfdl hongchi/Fire3D --revision 84d4246ba2a0f47c8b93fd9dd65a67d354afd668 --local-dir "$FIRE3D_CKPT/Fire3D"
  if [ ! -e "$SAM3D/Fire3D/checkpoints" ] && [ ! -L "$SAM3D/Fire3D/checkpoints" ]; then
    ln -s "$FIRE3D_CKPT" "$SAM3D/Fire3D/checkpoints"
  fi
  ( cd "$FIRE3D_CKPT/Fire3D" && sha256sum -c checksums.sha256 )      # 旧机器上 26 个文件全过
  log "fire3d: AnyUp 上采样器 (3.5M, torch.hub 运行时也会自己下)"
  fetch https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/anyup_multi_backbone.pth \
        "$TORCH_HOME_DEF/hub/checkpoints/anyup_multi_backbone.pth"
  # iTHOR 示例数据(12M)在仓库 Fire3D/data/ 里, 随代码走; 要重下: cd ~/sam3d/Fire3D && fire3d download --data --dataset ithor --scene-id iTHOR_FloorPlan312_physics
fi

echo; echo "全部完成: $SEL"

# =====================================================================================
# 以下是弃用路线的权重, 默认不下。已确认 pipeline/ tools/ data/outdoor/scripts/ 这些在用代码
# (outdoor 链 + indoor 物体链 + body4d)都不引用它们。真要复现时去掉行首 # 再跑。
# =====================================================================================
# ---- SpatialLM / SpatialGen 线(用户 09-21 决定不要了; 环境 spatiallm 只保留 DA3, spatialgen 弃用) ----
# hfdl manycore-research/SpatialLM1.1-Qwen-0.5B                                          # 2.3G, CC BY-NC
# hfdl manycore-research/SpatialGen-1.0 --revision 6c8c2d1c72ad6bf3c5a3440697c9767f4dceaec0  # 2.9G; 必须钉, 推理脚本写死了快照目录
# hfdl manycore-research/SpatialLM-Testset  --repo-type dataset --local-dir "$SAM3D/SpatialLM/SpatialLM-Testset"   # 旧机器只放了 2 个场景(59M)
# hfdl manycore-research/SpatialGen-Testset --repo-type dataset --local-dir "$SAM3D/SpatialGen/SpatialGen-Testset" # 1.3G
# hfdl facebook/VGGT-1B config.json model.safetensors README.md     # 4.7G(旧机器 9.4G, 多一份 model.pt); 原版 VGGT 只是对照工具, 流水线用的是 VGGT-Ω
# hfdl depth-anything/DA3NESTED-GIANT-LARGE                          # 6.3G, DA3 老版(作者标 deprecated), da3_to_ply.py 默认用 -1.1
#
# ---- 墙面贴图 / 地板贴图线(属于 LM/Gen 线; 环境 invsr、spatialgen 弃用) ----
# fetch https://hf-mirror.com/OAOA/InvSR/resolve/main/noise_predictor_sd_turbo_v5.pth "$SAM3D/InvSR/weights/noise_predictor_sd_turbo_v5.pth"
# hfdl stabilityai/sd-turbo --cache-dir "$SAM3D/InvSR/weights" \
#      model_index.json scheduler/scheduler_config.json text_encoder/config.json text_encoder/model.safetensors \
#      tokenizer/merges.txt tokenizer/special_tokens_map.json tokenizer/tokenizer_config.json tokenizer/vocab.json \
#      unet/config.json unet/diffusion_pytorch_model.safetensors vae/config.json vae/diffusion_pytorch_model.safetensors
# fetch https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt "$TORCH_HOME_DEF/hub/checkpoints/big-lama.pt"  # LaMa, spatialgen 环境
#
# ---- LayerPano3D (环境 layerpano3d; 旧机器 /mnt/nvme/jincheng/layerpano3d_ckpt 25G + HF 缓存 ~50G) ----
# L=$NVME/layerpano3d_ckpt
# hfdl black-forest-labs/FLUX.1-Fill-dev --exclude "flux1-fill-dev.safetensors"      # gated, 非商用; 只要 diffusers 分片 ~32G
# hfdl ysmikey/Layerpano3D-FLUX-Panorama-LoRA --local-dir "$L"                         # pano_lora_720*1440_v1.safetensors
# hfdl lllyasviel/Annotators ControlNetLama.pth --local-dir "$L"
# hfdl depth-anything/Depth-Anything-V2-Large depth_anything_v2_vitl.pth --local-dir "$L"
# fetch https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth "$L/sam_vit_h_4b8939.pth"
# hfdl Johanan0528/Infusion --local-dir "$L/Infusion"
# hfdl stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir "$L/pasd/stable-diffusion-v1-5" \
#      model_index.json scheduler/scheduler_config.json feature_extractor/preprocessor_config.json \
#      text_encoder/config.json text_encoder/model.safetensors \
#      tokenizer/merges.txt tokenizer/special_tokens_map.json tokenizer/tokenizer_config.json tokenizer/vocab.json \
#      unet/config.json unet/diffusion_pytorch_model.safetensors vae/config.json vae/diffusion_pytorch_model.safetensors
# # PASD checkpoint-100000: 来自阿里云 OSS 的 pasd.zip(5.45G), 下载 URL 旧笔记没记, 未确认; 见 LayerPano3D 仓库 README
# hfdl llava-hf/llava-1.5-7b-hf                                                          # 打标, 14G
# hfdl shi-labs/oneformer_coco_swin_large                                                # 语义分割, 1.8G
# hfdl shi-labs/oneformer_demo --repo-type dataset
# hfdl laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k    # 2.4G; 归属未确认(下载时间 09-11 06:09 与 LayerPano3D 运行吻合), 在用代码不引用
#
# ---- Pano2Room (环境 pano2room; 旧机器 ~/sam3d/Pano2Room/checkpoints 4.1G + HF 缓存 6.6G) ----
# P2R=$SAM3D/Pano2Room/checkpoints
# hfdl clay3d/omnidata omnidata_dpt_depth_v2.ckpt omnidata_dpt_normal_v2.ckpt --local-dir "$P2R"
# hfdl smartywu/big-lama big-lama.zip --local-dir /tmp/big-lama   # 解开后 best.ckpt -> $P2R/big-lama.ckpt, config.yaml -> $P2R/big-lama-config.yaml (zip 内部结构未确认)
# hfdl sd2-community/stable-diffusion-2-inpainting               # 官方 stabilityai 仓库已下架, 代码已改成这个镜像
#
# ---- 已删环境的遗留, 不需要 ----
# hfdl facebook/dinov3-vitl16-pretrain-lvd1689m     # TRELLIS.2 用的(环境 09-10 已删), gated; body4d 的 DINOv3 权重在 model.ckpt 里, Fire3D 模型包自带一份
# hfdl facebook/dinov2-with-registers-small         # PixARMesh 辅助模型(环境已删)
# hfdl facebook/dpt-dinov2-small-nyu                # 同上
