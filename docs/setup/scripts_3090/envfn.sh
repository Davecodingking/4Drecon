# SETUP.md 第 5 节的三个函数, 改走国内镜像
SETUP=~/sam3d/docs/setup
export CONDA_PKGS_DIRS=/mnt/nvme/jincheng/cache/conda_pkgs
export PIP_CACHE_DIR=/mnt/nvme/jincheng/cache/pip
export TMPDIR=/mnt/nvme/jincheng/cache/tmp
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_ENDPOINT=https://hf-mirror.com

mkenv() {
  local n=$1 p=$2
  sed -e '/^  - pip:/,$d' -e '/^prefix:/d' -e '/^name:/d' "$SETUP/envs/$n.yml" > "/tmp/$n.conda.yml"
  ~/miniforge3/bin/conda env create -p "$p" -f "/tmp/$n.conda.yml" -y
}

# ★ 清单里 torch/torchvision/torchaudio/xformers/torch_scatter 带 "+cu128" 这类本地版本号,
#   只有 download.pytorch.org 有; 国内镜像没有 -> 一行解不出来, 整批 235 个包全部回滚,
#   环境看着"装完了"其实是空壳(实测 sam3d 只剩 41 个包)。所以剥掉本地版本号,
#   并跳过已单独装好的 torch 三件套。整批失败时逐个装, 把真装不上的列出来。
pipreq() {
  local n=$1 p=$2; shift 2
  grep -v -E '^#|@ file://|^-e |@ git\+' "$SETUP/envs/$n.pip.txt" | grep -v -E "${SKIP:-^$}" \
    | grep -v -E '^(torch|torchvision|torchaudio)==' \
    | sed -E 's/\+[A-Za-z0-9][A-Za-z0-9._]*$//' > "/tmp/$n.req.txt"
  if ! "$p/bin/python" -m pip install --no-deps -r "/tmp/$n.req.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple "$@"; then
    echo "[整批失败, 改逐个装]"
    : > "/tmp/$n.failed.txt"
    while read -r line; do
      [ -z "$line" ] && continue
      "$p/bin/python" -m pip install --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple "$@" "$line" \
        || echo "$line" >> "/tmp/$n.failed.txt"
    done < "/tmp/$n.req.txt"
    if [ -s "/tmp/$n.failed.txt" ]; then echo "[装不上的]"; cat "/tmp/$n.failed.txt"; fi
  fi
}

cuda_env() {
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
  export TORCH_CUDA_ARCH_LIST=${ARCH:-8.6}
  export MAX_JOBS=${MAX_JOBS:-64}
}
