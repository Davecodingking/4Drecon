#!/bin/bash
# SETUP.md 第 9 节验收 + 流水线自检。每条独立, 失败不挡后面。
E=~/miniforge3/envs; V=/mnt/nvme/jincheng/envs
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.6
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
PASS=0; FAIL=0
t() { printf "%-14s " "$1"; shift; if out=$("$@" 2>&1); then echo "✔ ${out##*$'\n'}"; PASS=$((PASS+1)); else echo "✘ $(echo "$out" | tail -1 | cut -c1-110)"; FAIL=$((FAIL+1)); fi; }

echo "===== 1. 七个环境 import ====="
t sam3d       $E/sam3d/bin/python -c "import torch,pytorch3d,gsplat,kaolin,nvdiffrast.torch,spconv.pytorch,moge,transformers; print(torch.__version__, torch.cuda.get_device_name(0), transformers.__version__)"
t sam3         $E/sam3/bin/python -c "import torch,sam3,numpy,cv2; print(torch.cuda.is_available(), numpy.__version__, cv2.__version__)"
t spatiallm    $E/spatiallm/bin/python -c "import torch,depth_anything_3,trimesh,scipy,matplotlib,imageio_ffmpeg,numpy; print('DA3 ok', numpy.__version__)"
t wilddet      bash -c "cd ~/sam3d/WildDet3D && $E/wilddet/bin/python -c \"import sys; sys.path.insert(0,'.'); import torch, vis4d_cuda_ops, wilddet3d; print('ok')\""
t vggt_omega   $V/vggt_omega/bin/python -c "import vggt_omega,onnxruntime; print('ok')"
t metricdepth  $V/metricdepth/bin/python -c "import torch,mmengine,moge,unidepth; print('ok')"
t body4d       $E/body4d/bin/python -c "import torch,detectron2._C,sam3,moge; print('detectron2 ok')"

echo
echo "===== 2. 看得见 8 张卡吗 ====="
t GPU          $E/sam3d/bin/python -c "import torch; print(torch.cuda.device_count(), '张', torch.cuda.get_device_name(0))"

echo
echo "===== 3. 流水线能不能起 ====="
t 室外链        $E/sam3d/bin/python ~/sam3d/pipeline/outdoor_pipeline.py --scene soccer --run r1 --list
t 室内链        $E/sam3d/bin/python ~/sam3d/pipeline/indoor_pipeline.py --scene office --run r6_pillarfix --chain obj --list
t 驱动器        $E/sam3d/bin/python ~/sam3d/.claude/skills/run-recon-pipeline/drive.py --scene soccer --run r1 --status
t body4d链      $E/sam3d/bin/python ~/sam3d/pipeline/body4d_pipeline.py --scene soccer --list

echo
echo "===== 4. 工具 ====="
t 闸口网页      bash -c "$E/sam3d/bin/python -c \"import sys; sys.path.insert(0,'$HOME/sam3d/tools'); import gate_ui, scene_view; print('gate_ui + scene_view ok')\""
t 测试套件      bash -c "cd ~/sam3d && PY=$E/sam3d/bin/python bash tools/tests/run.sh 2>&1 | tail -1"

echo
echo "===== 5. 数据在不在 ====="
t 权重          bash -c "du -sc /mnt/nvme/jincheng/weights ~/sam3d/*/checkpoints 2>/dev/null | tail -1 | awk '{printf \"%.1f GB\n\", \$1/1048576}'"
t 结果目录      bash -c "ls ~/sam3d/data/indoorOutput/office | tr '\n' ' '"
t soccer轮次    bash -c "ls ~/sam3d/data/outdoor/result/soccer/r1 | head -3 | tr '\n' ' '"

echo
echo "===== 总计: 通过 $PASS, 失败 $FAIL ====="
