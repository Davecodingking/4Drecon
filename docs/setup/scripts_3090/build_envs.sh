#!/bin/bash
# 按 SETUP.md 第 6 节建 7 个环境。一个失败不挡后面的, 最后汇总。
source ~/envfn.sh; cuda_env
E=~/miniforge3/envs
LOG=~/envlogs; mkdir -p $LOG
step() { echo "=== [$(date +%H:%M:%S)] $* ==="; }
ok()   { echo "OK $1" >> $LOG/status.txt; }
bad()  { echo "FAIL $1" >> $LOG/status.txt; }

# ---------- 6.1 sam3d 的收尾(基础部分已在 env_sam3d.log 里跑过) ----------
build_sam3d_extra() {
  P=$E/sam3d
  FORCE_CUDA=1 $P/bin/pip install --no-build-isolation --no-deps \
    "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
  $P/bin/pip install --no-build-isolation --no-deps \
    "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"
  $P/bin/pip install --no-build-isolation --no-deps \
    "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"
  $P/bin/pip install --no-deps kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
  $P/bin/pip install --no-deps \
    "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" \
    "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"
  $P/bin/pip install --no-deps -e ~/sam3d/sam-3d-objects
  ( cd ~/sam3d/sam-3d-objects && $P/bin/python patching/hydra )
  $P/bin/python -c "import numpy; assert numpy.__version__=='1.26.4', numpy.__version__"
}

# ---------- 6.2 sam3 ----------
build_sam3() {
  N=sam3; P=$E/sam3
  mkenv $N $P
  $P/bin/pip install -q setuptools==80.10.2
  $P/bin/pip install -q torch==2.8.0 torchvision==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
  pipreq $N $P -q
  $P/bin/pip install --no-deps -e ~/sam3d/sam3
}

# ---------- 6.3 spatiallm (只有 DA3) ----------
build_spatiallm() {
  N=spatiallm; P=$E/spatiallm
  mkenv $N $P
  $P/bin/pip install -q torch==2.8.0 torchvision==0.23.0 xformers==0.0.32.post2 -i https://pypi.tuna.tsinghua.edu.cn/simple
  pipreq spatiallm.da3only $P -q
  $P/bin/pip install --no-deps -e ~/sam3d/Depth-Anything-3
}

# ---------- 6.4 wilddet ----------
build_wilddet() {
  N=wilddet; P=$E/wilddet
  mkenv $N $P
  $P/bin/pip install -q setuptools==79.0.1
  $P/bin/pip install -q torch==2.8.0 torchvision==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
  pipreq $N $P -q
  $P/bin/pip install --no-build-isolation --no-deps \
    "git+https://github.com/SysCV/vis4d_cuda_ops.git@d2478d80045b62f85be19235981a63ea9bc8db66"
  $P/bin/pip install --no-deps "git+https://github.com/EasternJournalist/utils3d.git@ca3e9ac015ab3473a37ea60f2f55063696638416"
  ( cd ~/sam3d/WildDet3D && git submodule update --init --recursive )
}

# ---------- 6.5 vggt_omega ----------
build_vggt_omega() {
  N=vggt_omega; P=/mnt/nvme/jincheng/envs/vggt_omega
  mkenv $N $P
  $P/bin/pip install -q torch==2.8.0 torchvision==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
  pipreq $N $P -q
  $P/bin/pip install --no-deps -e ~/sam3d/vggt-omega
}

# ---------- 6.6 metricdepth ----------
build_metricdepth() {
  N=metricdepth; P=/mnt/nvme/jincheng/envs/metricdepth
  mkenv $N $P
  $P/bin/pip install -q torch==2.8.0 torchvision==0.23.0 xformers==0.0.33 -i https://pypi.tuna.tsinghua.edu.cn/simple
  pipreq $N $P -q
  $P/bin/pip install --no-deps \
    "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a" \
    "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
    "git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4" \
    "git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf"
  [ -d /mnt/nvme/jincheng/UniDepth ] || { git clone -q https://github.com/lpiccinelli-eth/UniDepth.git /mnt/nvme/jincheng/UniDepth && git -C /mnt/nvme/jincheng/UniDepth checkout -q 8d8cfe4c7ee15297099983607febf0d4f32eb3d6; }
  $P/bin/pip install --no-deps -e /mnt/nvme/jincheng/UniDepth
}

# ---------- 6.7 body4d ----------
build_body4d() {
  N=body4d; P=$E/body4d
  mkenv $N $P
  $P/bin/pip install -q setuptools==80.10.2
  $P/bin/pip install -q torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
  pipreq $N $P -q
  $P/bin/pip install --no-build-isolation --no-deps \
    "git+https://github.com/facebookresearch/detectron2.git@a1ce2f956a1d2212ad672e3c47d53405c2fe4312"
  $P/bin/pip install --no-deps \
    "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a" \
    "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
    "git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4" \
    "git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf"
}

for f in sam3d_extra sam3 spatiallm wilddet vggt_omega metricdepth body4d; do
  step "开始 $f"
  if build_$f > $LOG/$f.log 2>&1; then ok $f; step "$f 完成"; else bad $f; step "$f 失败(继续下一个), 日志 $LOG/$f.log"; fi
done
echo "ALL_ENVS_DONE"
cat $LOG/status.txt
