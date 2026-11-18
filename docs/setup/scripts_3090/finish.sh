#!/bin/bash
# 用修好的 pipreq 把 7 个环境补齐。已建的补 pip 包, 没建的从头建。
source ~/envfn.sh; cuda_env
E=~/miniforge3/envs; V=/mnt/nvme/jincheng/envs
LOG=~/envlogs; mkdir -p $LOG; : > $LOG/status2.txt
say() { echo "=== [$(date +%H:%M:%S)] $* ==="; }
mark() { echo "$1 $2" >> $LOG/status2.txt; }

run() {  # run <名字> <函数>
  say "开始 $1"
  if $2 > $LOG/$1.2.log 2>&1; then mark OK "$1"; say "$1 完成"; else mark FAIL "$1"; say "$1 失败, 看 $LOG/$1.2.log"; fi
}

f_sam3d() {
  P=$E/sam3d
  pipreq sam3d $P -q
  $P/bin/pip install -q --no-deps -e ~/sam3d/sam-3d-objects
  ( cd ~/sam3d/sam-3d-objects && $P/bin/python patching/hydra )
  $P/bin/pip install -q --no-deps "numpy==1.26.4"      # kaolin/nvdiffrast 会把它顶成 2.x
  $P/bin/python -c "import numpy; assert numpy.__version__=='1.26.4', numpy.__version__"
}
f_sam3() {
  P=$E/sam3
  pipreq sam3 $P -q
  $P/bin/pip install -q --no-deps -e ~/sam3d/sam3
}
f_spatiallm() {
  P=$E/spatiallm
  $P/bin/pip install -q xformers==0.0.32.post2 -i https://pypi.tuna.tsinghua.edu.cn/simple
  pipreq spatiallm.da3only $P -q
  $P/bin/pip install -q --no-deps -e ~/sam3d/Depth-Anything-3
}
f_wilddet() {
  P=$E/wilddet
  pipreq wilddet $P -q
  $P/bin/pip install --no-build-isolation --no-deps "git+https://github.com/SysCV/vis4d_cuda_ops.git@d2478d80045b62f85be19235981a63ea9bc8db66"
  $P/bin/pip install -q --no-deps "git+https://github.com/EasternJournalist/utils3d.git@ca3e9ac015ab3473a37ea60f2f55063696638416"
  ( cd ~/sam3d/WildDet3D && git submodule update --init --recursive )
}
f_vggt_omega() {
  P=$V/vggt_omega
  [ -x $P/bin/python ] || mkenv vggt_omega $P
  $P/bin/pip install -q torch==2.8.0 torchvision==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
  pipreq vggt_omega $P -q
  $P/bin/pip install -q --no-deps -e ~/sam3d/vggt-omega
}
f_metricdepth() {
  P=$V/metricdepth
  [ -x $P/bin/python ] || mkenv metricdepth $P
  $P/bin/pip install -q torch==2.8.0 torchvision==0.23.0 xformers==0.0.33 -i https://pypi.tuna.tsinghua.edu.cn/simple
  pipreq metricdepth $P -q
  $P/bin/pip install -q --no-deps \
    "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a" \
    "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
    "git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4" \
    "git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf"
  [ -d /mnt/nvme/jincheng/UniDepth ] || { git clone -q https://github.com/lpiccinelli-eth/UniDepth.git /mnt/nvme/jincheng/UniDepth && git -C /mnt/nvme/jincheng/UniDepth checkout -q 8d8cfe4c7ee15297099983607febf0d4f32eb3d6; }
  $P/bin/pip install -q --no-deps -e /mnt/nvme/jincheng/UniDepth
}
f_body4d() {
  P=$E/body4d
  [ -x $P/bin/python ] || mkenv body4d $P
  $P/bin/pip install -q setuptools==80.10.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
  $P/bin/pip install -q torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
  pipreq body4d $P -q
  $P/bin/pip install --no-build-isolation --no-deps "git+https://github.com/facebookresearch/detectron2.git@a1ce2f956a1d2212ad672e3c47d53405c2fe4312"
  $P/bin/pip install -q --no-deps \
    "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a" \
    "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
    "git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4" \
    "git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf"
}

for n in sam3d sam3 spatiallm wilddet vggt_omega metricdepth body4d; do run $n f_$n; done
echo "FINISH_DONE"
echo "== 各环境包数 =="
for n in sam3d sam3 spatiallm wilddet body4d; do printf "%-12s %s\n" "$n" "$($E/$n/bin/pip list 2>/dev/null | wc -l)"; done
for n in vggt_omega metricdepth; do printf "%-12s %s\n" "$n" "$($V/$n/bin/pip list 2>/dev/null | wc -l)"; done
cat $LOG/status2.txt
