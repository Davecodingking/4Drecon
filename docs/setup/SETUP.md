# 换机器重装说明（ES-MAL-GPU-011 → 新机器）

2026-09-21 在旧机器上整理。旧机器要回收：代码 `~/sam3d` 带走，conda 环境（约 123G）和模型权重（约 250G）不带，到新机器上照这份文档重装。

**范围（用户 09-21 拍板）**：新机器只需要撑住两条整理后的流水线：`pipeline/outdoor_pipeline.py`，以及 `pipeline/indoor_pipeline.py` 的**物体链**（`STEPS_OBJ`；1172 行往下的 `STEPS_LM`，也就是 SpatialLM 链，不要了）。另外保留 body4d（`pipeline/body4d_pipeline.py`）。为此要装的环境只有 7 个：

> **sam3d、sam3、spatiallm（名字是历史遗留，里面只装 DA3）、vggt_omega、metricdepth、wilddet、body4d**

SpatialLM、SpatialGen 彻底不要了。spatialgen、invsr（墙面贴图线）、layerpano3d、pano2room 放进最后一节"已弃用（默认不装）"；fire3d 不在流水线里，也先放在那一节，**要不要保留待用户确认**。

同目录文件：

| 文件 | 是什么 |
|---|---|
| `SETUP.md` | 本文 |
| `envs/<环境名>.yml` | `conda env export -p <路径> --no-builds` 原样输出（13 个：12 个环境 + base，弃用的也导了） |
| `envs/<环境名>.pip.txt` | 该环境 `python -m pip freeze` 原样输出 |
| `envs/spatiallm.da3only.pip.txt` | spatiallm 的**精简版**清单：从完整版里只留 DA3 要的 148 个包（怎么算出来的写在文件头）。新机器按这份装；完整版 `spatiallm.pip.txt` 留作参考 |
| `download_weights.sh` | 权重下载脚本（只写没跑过），用法见文件开头注释 |

**本地改动见 `~/sam3d/patches/`**（另外导出，本文不重复）。本文只管"环境 + 权重"。

**推荐顺序**：装驱动和 CUDA 12.8（第 1 节）→ 装 miniforge、放代码、建好路径和软链（第 2 节）→ 核对仓库版本（第 3 节）→ 建 7 个环境（第 5、6 节）→ 配环境变量（第 7 节）→ `hf auth login` 后跑 `download_weights.sh`（第 8 节）→ 跑一遍验收（第 9 节）。

---

## 1. 旧机器基线；新卡不是 5090 时要改什么

| 项 | 旧机器 |
|---|---|
| GPU | 1 张 NVIDIA GeForce RTX 5090，32 GB，Blackwell，**sm_120**（算力 12.0） |
| 驱动 | 580.159.03（`nvidia-smi` 显示 `CUDA Version: 13.0`，意思是驱动最高支持到 13.0） |
| CUDA 工具包 | `/usr/local/cuda-12.8`（`/usr/local/cuda`、`/usr/local/cuda-12` 都链到它），`nvcc --version` = **12.8, V12.8.93**（CUDA 12.8 Update 1）。**nvcc 不在默认 PATH 里**，是各激活脚本自己加的 |
| 系统 | Ubuntu 22.04.5 LTS，内核 5.15.0-187，gcc/g++ 11.4.0，git 2.34 |
| 系统里没有 | `cmake`、`ffmpeg`、`unzip`、`git-lfs`（cmake 从 conda 装；视频用 `imageio-ffmpeg` 自带的二进制；解 zip 用 python `zipfile`） |
| 机器 | 192 核、503 GB 内存（下面编译命令的 `MAX_JOBS` 按这个定，小机器要调低） |
| conda | `~/miniforge3`（conda 26.5.3，base 是 python 3.14.6）。`~/.bashrc` 里**没有** `conda init`，所有激活脚本都用 `source ~/miniforge3/etc/profile.d/conda.sh` |
| `.condarc` | `~/miniforge3/.condarc`：channels 只有 `conda-forge`，另配了 `prefix.dev` 镜像（内容见第 2 节） |
| 磁盘 | 根分区 1.8T 已 99% 满；大件都在 `/mnt/nvme`（7.3T）：prefix 环境、权重、HF 缓存 |
| torch | 7 个在用环境全部 **torch 2.8.0+cu128**。cu128 官方轮子本身含 sm_70…sm_120，换卡**不用**换 torch |

base 环境里没有项目相关的东西（只有 conda 自身和 google-colab-cli、jupyter_client 这类无关工具），新机器装个干净的 miniforge 就行，`envs/base.*` 只是存档。

### 新卡不是 5090 / 不是 Blackwell 时

总原则：**预编译轮子不用管，自己编的 CUDA 扩展要按新卡的算力重编**。把下面这些地方的 `12.0` 换成新卡的算力：

| 新卡 | TORCH_CUDA_ARCH_LIST |
|---|---|
| RTX 5090 / RTX PRO 6000 Blackwell（现在） | `12.0` |
| B200 / GB200 | `10.0` |
| H100 / H200 | `9.0` |
| RTX 4090 / L40S / L4 | `8.9` |
| RTX 3090 / A6000 / A10 | `8.6` |
| A100 | `8.0` |

要改的位置（全是硬编码的 `12.0`）：

- 激活脚本：`~/sam3d/env.sh`、`env_spatiallm.sh`、`env_body4d.sh`
- 流水线里自己塞的环境变量：`pipeline/indoor_pipeline.py` 和 `pipeline/body4d_pipeline.py` 的 `GPUENV`；`pipeline/run_indoor_objects.sh`、`pipeline/run_indoor_recall.sh`
- 本文第 5 节的 `cuda_env` 函数（用 `ARCH=8.9 cuda_env` 这样传进去即可）
- （弃用的 `env_spatialgen.sh`、`env_pixar.sh`、`Fire3D/_5090/install_5090.sh` 里也有，不装就不用管）

在用的 7 个环境里自己编的 CUDA 扩展有：sam3d 的 pytorch3d、gsplat、nvdiffrast（运行时即时编译）；wilddet 的 vis4d_cuda_ops；body4d 的 detectron2。其余全是预编译轮子。

另外几条：

- **驱动**：cu128 的 torch 要求驱动 ≥ 570（Blackwell 必须；旧卡上跑 cu128 也建议 ≥ 570）。
- ★ **国内机器（2026-09-23 迁 render04 实测）**：`download.pytorch.org` 连得上但不传数据，pip 会静默挂死十几分钟。
  一律改走 `-i https://pypi.tuna.tsinghua.edu.cn/simple` —— torch 2.8.0 在 PyPI 上的默认构建就是 cu128，和这里要的一致。
  **但清单里 `torch==2.8.0+cu128` 这类带"本地版本号"的行在镜像上不存在，一行解不出来 pip 会把整批 235 个包全部回滚**，
  环境看着"装完了"其实只有 40 来个包。所以 `pipreq` 要剥掉 `+xxx` 后缀并跳过已单独装好的 torch 三件套，装完务必
  `pip list | wc -l` 对照清单行数。权重同理：`HF_ENDPOINT=https://hf-mirror.com`，而且 `download_weights.sh` 里有 4 处
  **直接 wget huggingface.co**（skyseg、两个 Metric3D），那个环境变量对它们无效，要一起换成镜像域名。
  conda 也用清华镜像，**但别在 conda 跑到一半改 `.condarc`**：它会重读配置然后卡死。
- **CUDA 工具包**：Blackwell 必须 nvcc ≥ 12.8。旧卡用 12.1 以上一般也能编，但脚本都写死了 `CUDA_HOME=/usr/local/cuda-12.8`，最省事还是装 12.8 到同一路径。没有 root 时可以用 conda 的 `cuda-toolkit=12.8` 装进环境，再把 `CUDA_HOME` 指过去。
- `spconv-cu121`（sam3d）换旧卡也照用，不用换。
- **显存 < 32 GB**：SAM 3D Objects 官方要求 32 GB；DA3 60 帧约 15.5 GB、134 帧约 21 GB；VGGT-Ω 60 帧约 11 GB；body4d 出网格峰值约 12 GB。

---

## 2. 目录与路径约定（最省事 = 新机器上保持同样的路径）

代码里写死了不少绝对路径。**最省事的做法：新机器上用户名还叫 `jincheng`，保留 `~/miniforge3`、`~/sam3d`、`/mnt/nvme/jincheng/...` 这几个路径**。大盘不在 `/mnt/nvme` 的话，建一个软链让这个路径存在（要 root）：

```bash
sudo mkdir -p /mnt/nvme && sudo ln -s /你的大盘/jincheng /mnt/nvme/jincheng
```

写死路径的地方（换路径时要改的清单，只列在用部分）：

| 写死的路径 | 在哪里 |
|---|---|
| `/mnt/nvme/jincheng/envs/{vggt_omega,metricdepth}` | `pipeline/outdoor_pipeline.py`、`pipeline/indoor_pipeline.py` 的 `PY`；`vggt-omega/tools/vggt_omega_geom.py` 的 `METRIC_PY` |
| `/mnt/nvme/jincheng/weights/vggt_omega` | `vggt-omega/tools/vggt_omega_geom.py` 的 `WEIGHTS` |
| `/mnt/nvme/jincheng/torch_cache`（Metric3D 的 TORCH_HOME 和 hub 目录） | `vggt-omega/tools/metric3d_scale.py` |
| `/mnt/nvme/jincheng/UniDepth` | metricdepth 环境的可编辑安装 |
| `/home/jincheng/sam3d/sam-body4d/checkpoints` | `sam-body4d/configs/body4d.yaml` 的 `ckpt_root` |
| `~/miniforge3/envs/<名>/bin/python` | 流水线各 `PY` 字典（用 `HOME` 展开，换用户名没关系，换 conda 位置要改） |
| `/usr/local/cuda-12.8` | `env.sh` `env_sam3.sh` `env_spatiallm.sh` `env_body4d.sh`、流水线 `GPUENV`、`run_indoor_*.sh` |
| 可编辑安装（`pip install -e`）记下的源码路径 | 装的时候写进各环境，代码位置变了重跑一次 `pip install --no-deps -e` 就行 |

### 新机器上先做的几件事

```bash
# 1) miniforge（旧机器 conda 26.5.3）
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
cat > ~/miniforge3/.condarc <<'EOF'
channels:
  - conda-forge
mirrored_channels:
  conda-forge:
    - https://conda.anaconda.org/conda-forge
    - https://prefix.dev/conda-forge
EOF

# 2) 大盘上的目录
mkdir -p /mnt/nvme/jincheng/{envs,weights,torch_cache,hf,cache/{conda_pkgs,pip,tmp}}

# 3) HF 缓存放大盘：旧机器上就是这个软链，没设 HF_HOME
[ -d ~/.cache/huggingface ] && [ ! -L ~/.cache/huggingface ] && mv ~/.cache/huggingface ~/.cache/huggingface.bak
mkdir -p ~/.cache && ln -sfn /mnt/nvme/jincheng/hf ~/.cache/huggingface

# 4) 根分区小的话，conda 包缓存 / pip 缓存 / 编译临时目录也放大盘（Fire3D 安装脚本就是这么做的）
export CONDA_PKGS_DIRS=/mnt/nvme/jincheng/cache/conda_pkgs PIP_CACHE_DIR=/mnt/nvme/jincheng/cache/pip TMPDIR=/mnt/nvme/jincheng/cache/tmp
```

---

### 系统包（要 root，装一次）
渲染回图那一步（pyrender 离屏渲染，代码里设了 `PYOPENGL_PLATFORM=egl`）要系统的 EGL/OpenGL 库，旧机器上装着下面这些：
```bash
sudo apt-get install -y build-essential libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libglvnd0 libsm6 libxext6 libxrender1
```
NVIDIA 驱动自带的 `libEGL_nvidia` 随驱动装好，不用单独装。缺这些库时渲染步骤会报 EGL 相关的错。

## 3. 上游仓库版本表

`~/sam3d` 本身也是一个 git 仓库（本地 `master` @ `41d5fd4a`，没有 remote），下面这些上游仓库是嵌在里面的独立 clone。**代码整目录（带 `.git`）拷过去的话不用重新 clone**，用 `git -C <目录> rev-parse HEAD` 核对一下 commit 即可。本地改动见 `~/sam3d/patches/`。

**在用的**：

| 目录 | 上游 | commit | 用在哪个环境 / 备注 |
|---|---|---|---|
| `sam-3d-objects` | https://github.com/facebookresearch/sam-3d-objects.git | `c27ae03f07f92a747d3efa427a0caa644c95e1e3` | sam3d；本地改 6 个文件 |
| `sam3` | https://github.com/facebookresearch/sam3.git | `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da` | sam3；只加了文件，没改上游文件 |
| `Depth-Anything-3` | https://github.com/ByteDance-Seed/Depth-Anything-3 | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | spatiallm；改了 `src/depth_anything_3/model/da3.py`；子模块 `da3_streaming/loop_utils/salad` 没初始化，用不到 |
| `vggt-omega` | https://github.com/facebookresearch/vggt-omega.git | `b2c61f6631d9f344a2d914bfba5d9529d6fc1d35` | vggt_omega；只加了 `tools/`（含 metricdepth 用的 `tools/shims/mmcv`） |
| `WildDet3D` | https://github.com/allenai/WildDet3D.git | `1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a` | wilddet；子模块 `third_party/lingbot_depth` = Robbyant/lingbot-depth @ `ff29de5b…`，`third_party/sam3` = weikaih04/sam3_for_wilddet3d @ `159490bb…` |
| `sam-body4d` | https://github.com/gaomingqi/sam-body4d | `21af1020979ef32ddf6be3597ef59a68bad2f1bf` | body4d；本地改 4 个文件 |
| `/mnt/nvme/jincheng/UniDepth`（**不在 ~/sam3d 下，要重新 clone**） | https://github.com/lpiccinelli-eth/UniDepth.git | `8d8cfe4c7ee15297099983607febf0d4f32eb3d6` | metricdepth 的可编辑安装 |
| `/mnt/nvme/jincheng/torch_cache/hub/yvanyin_metric3d_main`（**不在 ~/sam3d 下**） | https://github.com/YvanYin/Metric3D （torch.hub，main 分支） | 未记录（09-17 下的 main） | metricdepth；`download_weights.sh metric3d` 会重新拉 |
| `~/.cache/torch/hub/facebookresearch_dinov2_main` | https://github.com/facebookresearch/dinov2 （torch.hub，main 分支） | 未记录 | sam3d；`download_weights.sh sam3d` 会重新拉 |

**不再需要的**（代码可以留着，不用装环境）：

| 目录 | 上游 | commit | 为什么不要 |
|---|---|---|---|
| `SpatialLM` | https://github.com/manycore-research/SpatialLM | `8913c44d84a450c53e9340b13317f8cf7144a738` | SpatialLM 线不要了 |
| `SpatialGen` | https://github.com/manycore-research/SpatialGen | `677c1168b359b00ae4a64a58783bc71048aabcdd` | 同上。子模块 `src/recons/Sparse-RaDeGS` = fangchuan/Sparse-RaDeGS @ `ee7b413d…`，子模块里也有本地改动 |
| `vggt` | https://github.com/facebookresearch/vggt | `a288dd0f14786c93483e45524328726ab7b1b4ce` | 原版 VGGT 只是对照工具，流水线用的是 VGGT-Ω |
| `InvSR` | https://github.com/zsyOAOA/InvSR.git | `609788cd076f1990e38d3912092ca67286e1e0bc` | 墙面贴图线（属 LM/Gen 线） |
| `Fire3D` | https://github.com/xiahongchi/Fire3D.git | `2368dd2f3909120cf90bbf8a17807abe9c41e600` | 不在流水线里，**待用户确认** |
| `Pano2Room` | https://github.com/TrickyGo/Pano2Room.git | `bbf93ae57086ed700edc6ee445852d4457a9d704` | 已排除的路线 |
| `PiSA-SR` | https://github.com/csslc/PiSA-SR.git | `9c76e12e261d1980148738a0cff41e5e27d18c9f` | 钉 torch 2.0.1，5090 上跑不了，从没建过环境 |
| `LayerPano3D`（**旧机器上已删**） | https://github.com/3DTopia/LayerPano3D | 未记录 | 09-15 判定没用 |

需要重新 clone 时（只列在用的）：

```bash
cd ~/sam3d
while read dir url commit; do
  [ -d "$dir/.git" ] && { echo "已有 $dir"; continue; }
  git clone "$url" "$dir" && git -C "$dir" checkout "$commit"
done <<'EOF'
sam-3d-objects   https://github.com/facebookresearch/sam-3d-objects.git   c27ae03f07f92a747d3efa427a0caa644c95e1e3
sam3             https://github.com/facebookresearch/sam3.git             8f0b7f4d4e7eda2ed606ebde6702c93359ad01da
Depth-Anything-3 https://github.com/ByteDance-Seed/Depth-Anything-3       3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
vggt-omega       https://github.com/facebookresearch/vggt-omega.git       b2c61f6631d9f344a2d914bfba5d9529d6fc1d35
WildDet3D        https://github.com/allenai/WildDet3D.git                 1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a
sam-body4d       https://github.com/gaomingqi/sam-body4d                  21af1020979ef32ddf6be3597ef59a68bad2f1bf
EOF
git -C WildDet3D submodule update --init --recursive     # third_party/lingbot_depth、third_party/sam3
git clone https://github.com/lpiccinelli-eth/UniDepth.git /mnt/nvme/jincheng/UniDepth \
  && git -C /mnt/nvme/jincheng/UniDepth checkout 8d8cfe4c7ee15297099983607febf0d4f32eb3d6
# 然后按 ~/sam3d/patches/ 把本地改动打回去
```

---

## 4. 环境总表

大小是旧机器上 `du -sh` 的结果。"用途"里的反引号是流水线步骤名（`outdoor_pipeline.py --list` 能打出来，或见 `docs/WORKFLOW.md` 第二节）。

**在用（要装）的 7 个：**

| 环境 | 路径 | Python | torch | 服务于 | 大小 |
|---|---|---|---|---|---|
| sam3d | `~/miniforge3/envs/sam3d` | 3.11.16 | 2.8.0+cu128 | SAM 3D Objects 生成（`gen` `solo` `recall_gen`）；几何、组装、物理、渲染（`frames` `up` `dedup` `pick` `boxes` `scale` `shell` `assemble` `phys` `render` 等大部分步骤）；Qwen3-VL（`vlm` `verify` `vlm_sizes`，步骤表里的 `trellis2` 键也指向这里）；`drive.py`、`tools/gate_ui.py`；body4d 链的抽帧 | 12G |
| sam3 | `~/miniforge3/envs/sam3` | 3.12.14 | 2.8.0+cu128 | SAM 3 视频跟踪/抠图（`track` `recall_mask`）；点选网页 `app.py` | 7.8G |
| spatiallm | `~/miniforge3/envs/spatiallm` | 3.11.16 | 2.8.0+cu128 | **名字是历史遗留，新机器上里面只装 DA3**：两条链的 `da3`（`Depth-Anything-3/da3_to_ply.py`；室内默认 `GEOM=vggt_da3` 时 DA3 当置信度掩码和米制尺子）和室内的 `zup`（`da3_to_spatiallm.py`、`tools/pcd_layers.py`，只用 numpy/trimesh/scipy/matplotlib）。代码按 `spatiallm` 这个名字调，所以名字不改 | 旧环境 12G，精简后未实测 |
| wilddet | `~/miniforge3/envs/wilddet` | 3.11.16 | 2.8.0+cu128 | WildDet3D（`boxes_wd` `detect`）。vis4d 钉 pydantic<2，必须单独一个环境 | 7.9G |
| vggt_omega | `/mnt/nvme/jincheng/envs/vggt_omega` | 3.11.16 | 2.8.0+cu128 | VGGT-Ω 几何后端（`da3` 步在 `GEOM=vggt_l/vggt_g2/vggt_da3` 时走它；室外默认 `vggt_l`、室内默认 `vggt_da3`） | 7.0G |
| metricdepth | `/mnt/nvme/jincheng/envs/metricdepth` | 3.11.16 | 2.8.0+cu128 | Metric3D v2 定米制尺度（`vggt_omega_geom.py` 内部调 `metric3d_scale.py`，`GEOM=vggt_l/vggt_g2` 时用）；另装了 MoGe-2 / UniDepthV2 做过米制评测 | 8.5G |
| body4d | `~/miniforge3/envs/body4d` | 3.12.14 | 2.8.0+cu128 | SAM-Body4D 人体网格（`pipeline/body4d_pipeline.py`）。**必须和 sam3 分开** | 7.8G |

**不装的：**

| 环境 | 路径 | Python | torch | 原来干什么 | 大小 |
|---|---|---|---|---|---|
| base | `~/miniforge3` | 3.14.6 | 无 | 只有 conda 本身，装 miniforge 即可 | — |
| spatialgen（弃用） | `~/miniforge3/envs/spatialgen` | 3.10.21 | 2.8.0+cu128 | SpatialGen 生成；LaMa（室内 LM 链的地板补洞、墙贴图线） | 9.3G |
| invsr（弃用） | `~/miniforge3/envs/invsr` | 3.10.21 | 2.8.0+cu128 | 墙贴图 InvSR 超分 | 7.1G |
| fire3d（**待用户确认**） | `/mnt/nvme/jincheng/envs/fire3d` | 3.10.20 | 2.7.1+cu128 | FIRE3D 前馈室内重建，验证性实验，不在流水线里 | 16G |
| pano2room（弃用） | `~/miniforge3/envs/pano2room` | 3.10.21 | 2.8.0+cu128 | Pano2Room 全景→3DGS，已排除 | 9.5G |
| layerpano3d（弃用） | `~/miniforge3/envs/layerpano3d` | 3.9.23 | 2.8.0+cu128 | LayerPano3D，09-15 判定没用、仓库已删 | 11G |

在用的 7 个环境在旧机器上合计约 63G（spatiallm 精简后会更小）。

---

## 5. 通用建环境方法

旧机器上的环境都是"conda 只装 Python 和几个底层库，其余全用 pip"。所以重建分两半：conda 那一半按 `.yml` 的 conda 段重建，pip 那一半按 `.pip.txt` 装。

**为什么 pip 要加 `--no-deps`**：`pip freeze` 记下的是最终状态，里面本来就有依赖冲突（比如 unidepth 声明要 numpy≥2，实际装的是 1.26.4，照样能跑）。不加 `--no-deps` 的话 pip 会去解依赖，要么报解不出来，要么把 numpy 之类的顶掉。

**为什么要先把本地路径/git/可编辑的包滤掉**：`.pip.txt` 里 `@ file:///…`（旧机器上的本地源码或临时目录里的轮子）、`-e …`（可编辑安装）、`@ git+…`（从 git 装的，有几个要带特殊参数编译）这三类没法直接装。滤掉之后，在每个环境一节的"清单装不回来的"里逐个装。`packaging @ file:///home/conda/feedstock_root/…` 这种是 conda 装的包，conda 那一半已经带了。

**`pip freeze` 不记 setuptools / pip / wheel**，而好几个环境必须钉 setuptools 旧版本（新版删了 `pkg_resources`），所以每节都单独装一次 setuptools。

先把这几个函数粘进 shell（每开一个新终端粘一次）：

```bash
SETUP=~/sam3d/docs/setup

# 按导出的 yml 重建 conda 那一半（去掉 pip 段、name、prefix）
mkenv() {   # 用法: mkenv <环境名> <环境路径>
  local n=$1 p=$2
  sed -e '/^  - pip:/,$d' -e '/^prefix:/d' -e '/^name:/d' "$SETUP/envs/$n.yml" > "/tmp/$n.conda.yml"
  ~/miniforge3/bin/conda env create -p "$p" -f "/tmp/$n.conda.yml"
}
# conda 精确版本解不出来时的退路: ~/miniforge3/bin/conda create -y -p <路径> python=<版本>

# 按 pip 清单装 pip 那一半（去掉本地路径、-e、git 源，这些在各节里单独装）
pipreq() {  # 用法: [SKIP='正则'] pipreq <清单名> <环境路径> [额外 pip 参数...]   清单 = envs/<清单名>.pip.txt
  local n=$1 p=$2; shift 2
  grep -v -E '^#|@ file://|^-e |@ git\+' "$SETUP/envs/$n.pip.txt" | grep -v -E "${SKIP:-^$}" > "/tmp/$n.req.txt"
  "$p/bin/python" -m pip install --no-deps -r "/tmp/$n.req.txt" \
      --extra-index-url https://download.pytorch.org/whl/cu128 "$@"
}

# 编译 CUDA 扩展要的环境变量（和 env_*.sh 一致）。非 5090 用 ARCH=8.9 cuda_env 这样传
cuda_env() {
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
  export TORCH_CUDA_ARCH_LIST=${ARCH:-12.0}
  export MAX_JOBS=${MAX_JOBS:-32}
}
```

装的过程中如果某个 sdist 包报"找不到 torch"（隔离构建环境里没有 torch），给那一个包单独加 `--no-build-isolation` 重装。

每节末尾列了"pip check 预期报警"：这些在旧机器上 `pip check` 就有，环境照样能用，新机器上看到同样的报警不用管。

---

## 6. 在用的 7 个环境

### 6.1 sam3d —— SAM 3D Objects + 几何/组装 + Qwen3-VL

激活：`source ~/sam3d/env.sh`。流水线直接调 `~/miniforge3/envs/sam3d/bin/python`。

```bash
N=sam3d; P=~/miniforge3/envs/sam3d
mkenv $N $P && cuda_env
$P/bin/pip install "setuptools==80.10.2"
$P/bin/pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pipreq $N $P

# --- 清单装不回来的 ---
# 源码编译的 CUDA 扩展（原来各用几分钟，MAX_JOBS=96）
FORCE_CUDA=1 $P/bin/pip install --no-build-isolation --no-deps \
  "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
$P/bin/pip install --no-build-isolation --no-deps \
  "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"
$P/bin/pip install --no-build-isolation --no-deps \
  "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"
# kaolin 0.18.0：NVIDIA S3 上 torch-2.8.0_cu128 的 cp311 轮子（旧机器上是先下到临时目录再装的）
$P/bin/pip install --no-deps kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
# 从 git 装的纯 Python 包
$P/bin/pip install --no-deps \
  "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" \
  "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"
# 本地仓库（可编辑）
$P/bin/pip install --no-deps -e ~/sam3d/sam-3d-objects
# 官方要求的 hydra 补丁（从 GitHub 下一个 hydra/core/utils.py 覆盖进环境；旧机器上打过）
cd ~/sam3d/sam-3d-objects && $P/bin/python patching/hydra
# 最后确认 numpy 没被顶掉
$P/bin/python -c "import numpy; assert numpy.__version__ == '1.26.4', numpy.__version__"
```

- 原来 pytorch3d / gsplat 是 clone 到 `~/sam3d/pytorch3d`、`~/sam3d/gsplat` 编的（目录后来删了），commit 就是上面这两个（官方 `requirements.p3d.txt` / `requirements.inference.txt` 里的 pin）。
- nvdiffrast 在这个环境里没有预编译的 `.so`，第一次用时即时编译到 `~/.cache/torch_extensions`，所以**运行时也要有 `CUDA_HOME` 和正确的 `TORCH_CUDA_ARCH_LIST`**（`env.sh` 和流水线 `GPUENV` 里有）。gsplat 也会在那里放一份即时编译的缓存。
- flash_attn / xformers **没装**：非 A100/H100/H200 时代码默认走 sdpa。
- 裁剪过的依赖清单在 `~/sam3d/requirements-runtime.txt`（官方 requirements 去掉训练/云端包），只作参考，以 `.pip.txt` 为准。

钉版本的原因：

- torch 2.5.1+cu121（官方）→ 2.8.0+cu128：Blackwell 必须。
- `spconv-cu121==2.3.8` 原样能用（PTX 即时编译向前兼容 sm_120，实测过）。
- `numpy==1.26.4`：opencv-python 4.9 的 pin 和 numpy 2 冲突。**装任何新包之后都复查一次**，kaolin、rembg、nvdiffrast 都曾把 numpy 拉到 2.x。
- `setuptools<81`：lightning 2.3.3 要 `pkg_resources`。
- `transformers==4.57.1` + `accelerate`：给 Qwen3-VL 用（原来在已删的 trellis2 环境里，09-10 挪进来，净加 4 个包）。**不要动 `huggingface_hub==0.36.2`**：不钉会被推到 1.x，gradio/moge/timm 会连锁崩。
- 本地代码改动（见 patches）：`notebook/inference.py` 保留已有的 `CUDA_HOME`，不覆盖成 conda 前缀（同上游未合并的 PR #11）；渲染后端 `"inria"` → `"gsplat"`（没装 diff_gaussian_rasterization，贴图烘焙要用）。

运行时注意：流水线直接调这个环境的 python（不经过 `conda activate`），而 `run.py` 里的 `inference.py` 要读 `CONDA_PREFIX`，所以流水线会手动传 `CONDA_PREFIX` 和 `HF_HUB_OFFLINE=1`。**离线跑意味着 MoGe v1 和 DINOv2 必须提前下好**（`download_weights.sh sam3d`）。

pip check 预期报警：gsplat 缺 jaxtyping；kaolin 缺 ipycanvas / ipyevents / jupyter-client；sam3d-objects 缺一长串训练/开发依赖（auto-gptq、bpy、black 等）。

### 6.2 sam3 —— SAM 3 分割/跟踪

激活：`source ~/sam3d/env_sam3.sh`。

```bash
N=sam3; P=~/miniforge3/envs/sam3
mkenv $N $P && cuda_env
$P/bin/pip install "setuptools==80.10.2"
$P/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pipreq $N $P
$P/bin/pip install --no-deps -e ~/sam3d/sam3
```

- 当时手写的依赖清单在 `~/sam3d/sam3/requirements-sam3.txt`（numpy 1.26.4、opencv-python 4.10.0.84、gradio 5.49.0、setuptools<81）。detectron2 / openai / tidecv 这些评测依赖不装。
- 这个环境里**没有** trimesh / pyrender，几何类的步骤都在 sam3d 环境里跑。
- 视频跟踪（`build_sam3_video_model`）和图片分割用同一份 `sam3.pt`。

pip check 预期报警：tifffile、scipy 要 numpy≥2（实际 1.26.4）；decord "not supported on this platform"。

### 6.3 spatiallm —— 名字是历史遗留，里面只有 Depth Anything 3

没有 SpatialLM 了，但**环境名必须还叫 `spatiallm`**：`outdoor_pipeline.py` / `indoor_pipeline.py` 的 `PY["spatiallm"]` 就是按这个名字找 python 的，`da3` 和 `zup` 两步跑在这里。激活脚本 `~/sam3d/env_spatiallm.sh` 照用（里面的注释还写着 SpatialLM 用法，别管）。

新机器按**精简版清单** `envs/spatiallm.da3only.pip.txt` 装，只含 DA3 要的包；SpatialLM 专用的东西（transformers 4.46.1、torch-scatter、spconv-cu126、flash-attn、rerun-sdk、shapely 等）和原版 VGGT 都不装。

```bash
N=spatiallm; P=~/miniforge3/envs/spatiallm
mkenv $N $P && cuda_env                      # conda 那一半和原来一样（python 3.11.16）
$P/bin/pip install torch==2.8.0 torchvision==0.23.0 xformers==0.0.32.post2 \
  --index-url https://download.pytorch.org/whl/cu128
pipreq spatiallm.da3only $P                   # 精简清单
$P/bin/pip install --no-deps -e ~/sam3d/Depth-Anything-3
```

- 这个环境**不编译**任何东西。
- 精简清单是按"DA3 的声明依赖 + 流水线里在这个环境跑的脚本（`da3_to_ply.py`、`da3_to_spatiallm.py`、`topview_da3.py`、`tools/pcd_layers.py`、`data/outdoor/scripts/pose_check.py`）的 import"在旧环境里递归求依赖闭包得到的。**精简后的环境旧机器上没有实际建过**，装完请跑第 9 节的 DA3 端到端验收。如果缺了什么，对照完整版 `spatiallm.pip.txt` 补装那一个包（`--no-deps`）。
- 本地代码改动（见 patches）：`Depth-Anything-3/src/depth_anything_3/model/da3.py` 一处（传了已知内参就用已知焦距算米制尺度）。
- 完整版（带 SpatialLM + VGGT）怎么装，写在第 11.1 节，只作存档。

钉版本的原因：

- `numpy==1.26.4`：DA3 要 numpy<2（当年装 DA3 时从 2.4.6 降下来的）。
- `xformers==0.0.32.post2`（cu128 源）：DA3 只用它的 SwiGLU，注意力走 PyTorch SDPA，所以没有 SpatialGen 那个 FA3 问题。
- `imageio-ffmpeg`：机器上没有系统 ffmpeg，`--video` 抽帧用它自带的二进制。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 必须有（激活脚本和 `GPUENV` 里有），否则 DA3 会因显存碎片在 `custom_interpolate` 里 OOM。
- 旧环境里还有 yt-dlp（流水线不用），要下视频自己装。

pip check：精简环境没在旧机器上建过，预期报警未知（完整版的唯一报警是 rerun-sdk 要 numpy≥2，精简版里没有 rerun）。

### 6.4 wilddet —— WildDet3D

没有激活脚本，流水线直接调 `~/miniforge3/envs/wilddet/bin/python`。WildDet3D 本身**不 pip 安装**，脚本里 `sys.path.insert` 仓库目录（`third_party/sam3` 也是它自己在 `wilddet3d/__init__.py` 里加进 sys.path 的）。

```bash
N=wilddet; P=~/miniforge3/envs/wilddet
mkenv $N $P && cuda_env
$P/bin/pip install "setuptools==79.0.1"
$P/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pipreq $N $P

# --- 清单装不回来的 ---
# vis4d 的 CUDA 算子，源码编
$P/bin/pip install --no-build-isolation --no-deps \
  "git+https://github.com/SysCV/vis4d_cuda_ops.git@d2478d80045b62f85be19235981a63ea9bc8db66"
# PyPI 上的 utils3d==0.1.0 是同名的无关包，真正的只在 GitHub
$P/bin/pip install --no-deps "git+https://github.com/EasternJournalist/utils3d.git@ca3e9ac015ab3473a37ea60f2f55063696638416"
cd ~/sam3d/WildDet3D && git submodule update --init --recursive     # third_party/sam3、third_party/lingbot_depth
```

钉版本的原因：

- torch 2.5.1+cu121（官方 README）→ 2.8.0+cu128。
- `vis4d==1.0.0` 连带钉 `pydantic==1.10.26` 和 `numpy<2`（实际 1.26.4）：塞进其他任何环境都会打架，所以单独一个环境。
- `setuptools<80`：`third_party/sam3/model_builder.py` 还 import `pkg_resources`（官方 requirements 写的就是 <80）。
- `transformers==5.16.1`：只有这个环境是 5.x，别拿去对照其他环境。

pip check：旧机器上没有报警。

### 6.5 vggt_omega —— VGGT-Ω 几何后端

prefix 环境，不在 `~/miniforge3/envs`，流水线直接调 `/mnt/nvme/jincheng/envs/vggt_omega/bin/python`。

```bash
N=vggt_omega; P=/mnt/nvme/jincheng/envs/vggt_omega
mkenv $N $P
$P/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pipreq $N $P
$P/bin/pip install --no-deps -e ~/sam3d/vggt-omega
```

- 这个环境很小：torch + vggt-omega + opencv + onnxruntime（天空分割 `skyseg.onnx` 走 CPU）。没有 huggingface_hub，也没有 `hf` 命令。
- 权重路径写死在 `vggt-omega/tools/vggt_omega_geom.py`：`/mnt/nvme/jincheng/weights/vggt_omega`。
- VGGT-Ω 自己不是米制（ithor312 上小 2.2–2.4 倍），所以 `GEOM=vggt_l/vggt_g2` 还要 metricdepth 环境跑 Metric3D 定尺度；`vggt_da3` 则用 DA3 当尺子（spatiallm 环境）。

pip check：旧机器上没有报警。

### 6.6 metricdepth —— Metric3D v2（定尺度）+ MoGe-2 / UniDepthV2（评测）

prefix 环境，由 `vggt_omega_geom.py` 内部调用。

```bash
N=metricdepth; P=/mnt/nvme/jincheng/envs/metricdepth
mkenv $N $P
# 旧机器上 pip 从 cu128 源解出来的 xformers 是这个带本地标签的版本
$P/bin/pip install torch==2.8.0 torchvision==0.23.0 "xformers==0.0.33+5d4b92a5.d20251029" \
  --index-url https://download.pytorch.org/whl/cu128
pipreq $N $P

# --- 清单装不回来的 ---
# MoGe-2 和它从 git 装的依赖（只有评测用，Metric3D 不需要；想省事可以跳过这两步，但环境就和旧机器不一样了）
$P/bin/pip install --no-deps \
  "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a" \
  "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
  "git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4" \
  "git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf"
# UniDepth：源码不在 ~/sam3d 下，先按第 3 节 clone 到 /mnt/nvme/jincheng/UniDepth
$P/bin/pip install --no-deps -e /mnt/nvme/jincheng/UniDepth
```

- **Metric3D 不用 pip 装**：`metric3d_scale.py` 用 `torch.hub.load(<本地目录>, source="local")` 加载 `/mnt/nvme/jincheng/torch_cache/hub/yvanyin_metric3d_main`，权重在同一个 TORCH_HOME 的 `hub/checkpoints/` 下，都由 `download_weights.sh metric3d` 放好。
- Metric3D 的 hub 代码 import 了 `mmcv.utils` 里几个配置工具，**不编 mmcv**，用仓库里的 `vggt-omega/tools/shims/mmcv`（转发到 `mmengine==0.10.7`）顶上。这个 shim 随代码走，不用另装。
- 安装记录 `~/sam3d/_deploy_logs/metricdepth_install.log`。

pip check 预期报警：unidepth 缺 h5py / imageio / tables / tabulate / torchaudio，并要 numpy≥2；moge 要 numpy≥2（实际 1.26.4，都能跑）。

### 6.7 body4d —— SAM-Body4D 人体网格

激活：`source ~/sam3d/env_body4d.sh`。详见 `sam-body4d/NOTES_5090.md`。**必须是独立环境**：仓库里 `models/sam3/` 是内嵌的另一份 SAM 3，包名同样叫 `sam3`，装进 sam3 环境会把室外链 `track_all.py` 依赖的那份静默替换掉（两份的 `propagate_in_video` 一个 yield 5 个值、一个 yield 6 个）。

```bash
N=body4d; P=~/miniforge3/envs/body4d
mkenv $N $P && cuda_env
$P/bin/pip install "setuptools==80.10.2"
$P/bin/pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pipreq $N $P

# --- 清单装不回来的 ---
# ★ 本地可编辑安装（2026-09-23 迁 render04 时发现原来漏写了这两行，不装 `import sam3` 就失败）
$P/bin/pip install --no-deps -e ~/sam3d/sam-body4d
$P/bin/pip install --no-deps -e ~/sam3d/sam-body4d/models/sam3     # 仓库内嵌的那份 SAM 3
# detectron2 @ a1ce2f9 源码编（早于 Blackwell 的 pin，在 torch 2.8 + sm_120 上实测能编过）
$P/bin/pip install --no-build-isolation --no-deps \
  "git+https://github.com/facebookresearch/detectron2.git@a1ce2f956a1d2212ad672e3c47d53405c2fe4312"
# MoGe（sam-body4d 的 pyproject 里写的 git 依赖）和它的 git 依赖
$P/bin/pip install --no-deps \
  "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a" \
  "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
  "git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4" \
  "git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf"
# 两个本地可编辑包：内嵌的 sam3 和 sam-body4d 本身
cd ~/sam3d/sam-body4d && $P/bin/pip install --no-deps -e models/sam3 && $P/bin/pip install --no-deps -e .
# 验收：这一句过了，detectron2 的 CUDA 扩展就是好的
$P/bin/python -c "import torch,detectron2,detectron2._C; print(torch.ops.detectron2.box_iou_rotated)"
```

钉版本的原因：

- README 写的 cu118 + torch 2.7.1 → 2.8.0+cu128（cu118 的轮子里没有 sm_120）。这是唯一必须偏离 README 的地方。
- `setuptools<81`：detectron2 的 model_zoo 还在 import `pkg_resources`。
- `numpy==2.2.6`：内嵌 sam3 的 metadata 要 `numpy==1.26`，只是报警，实测不影响。
- `imageio-ffmpeg` 只在这个环境里有，`body4d_check_5090.py` 必须用 body4d 环境跑。
- 本地代码改动 4 处（见 patches 和 NOTES）：文字提示路径的解包个数、两处"位置序号冒充身份 ID"、`configs/body4d.yaml` 的 `ckpt_root`。

pip check 预期报警：detectron2 缺 black / hydra-core / tensorboard，要 iopath<0.1.10（实际 0.1.10）；sam3 要 numpy==1.26（实际 2.2.6）；decord "not supported on this platform"。

---

## 7. 环境变量

旧机器上 `~/.bashrc` / `~/.profile` 里**没有**任何项目相关的设置（只有 `PATH=$HOME/.local/bin:$PATH` 和 gcloud），没有 `conda init`，没有设 `HF_HOME` / `TORCH_HOME`。所有东西都在激活脚本和流水线里。新机器上照旧即可，只需保证下面几条成立：

| 变量 / 位置 | 旧机器上的值 | 谁设的 | 新机器怎么办 |
|---|---|---|---|
| HF 缓存 | `~/.cache/huggingface` → 软链到 `/mnt/nvme/jincheng/hf` | 手动软链，没设 `HF_HOME` | 照做软链（第 2 节）。也可以改设 `HF_HOME=/大盘/hf`，在用的代码里没有写死 HF 缓存路径（写死的只有已弃用的 SpatialGen 脚本） |
| `TORCH_HOME` | 默认 `~/.cache/torch`（SAM3D 的 DINOv2 在 `hub/` 下） | 默认值 | 不用设 |
| `TORCH_HOME`（Metric3D 专用） | `/mnt/nvme/jincheng/torch_cache` | `metric3d_scale.py` 里 `os.environ.setdefault`，hub 目录也写死 | 让这个路径存在，或改脚本 |
| `CUDA_HOME` | `/usr/local/cuda-12.8` | `env.sh` `env_sam3.sh` `env_spatiallm.sh` `env_body4d.sh`，流水线 `GPUENV`，`run_indoor_*.sh` | 装 CUDA 12.8 到同一路径，或全部改掉 |
| `PATH` | 前面加 `$CUDA_HOME/bin` | 同上（`pipeline/_common.py` 在带 `CUDA_HOME` 时也会补 PATH） | 同上 |
| `LD_LIBRARY_PATH` | 前面加 `$CUDA_HOME/lib64` | `env.sh` `env_spatiallm.sh` | 同上 |
| `TORCH_CUDA_ARCH_LIST` | `12.0` | 激活脚本和流水线 `GPUENV`（运行时即时编译 nvdiffrast / gsplat 要用） | 换卡改成新算力（第 1 节） |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 激活脚本、`GPUENV` | 不变。少了它 DA3 会因显存碎片 OOM |
| `HF_HUB_OFFLINE` | `1` | 流水线调 SAM3D 生成时 | 不变。所以权重必须提前下好 |
| `CONDA_PREFIX` | sam3d 环境路径 | 流水线调 `run.py` 时手动传（`inference.py` 直接读它） | 不变 |
| `CONDA_PKGS_DIRS` / `PIP_CACHE_DIR` / `TMPDIR` | `/mnt/nvme/jincheng/cache/{conda_pkgs,pip,tmp}` | 只在 Fire3D 的安装脚本里 | 根分区小就照设（第 2 节） |
| HF token | `~/.cache/huggingface/token`（即 nvme 上的 `hf/token`） | `hf auth login` | 新机器重新 `hf auth login`。**不要拷旧 token 文件，也不要把 token 写进任何脚本/文档** |

激活脚本（都在 `~/sam3d/`，随代码走）：在用的是 `env.sh`（sam3d）、`env_sam3.sh`、`env_spatiallm.sh`（里面只有 DA3 了）、`env_body4d.sh`。wilddet / vggt_omega / metricdepth 没有激活脚本，流水线直接用环境里的 python 绝对路径。`env_spatialgen.sh` 对应弃用的环境；`env_pixar.sh` 指向早就删掉的 `pixar` 环境，都别用。

---

## 8. 权重

### 8.1 先登录 HF

```bash
~/miniforge3/envs/sam3d/bin/hf auth login      # 任何带 huggingface_hub 的环境都行；vggt_omega 环境里没有 hf
```

gated 仓库要先在网页上申请并获批。旧机器用的账号 DaveHJC 下面 4 个都批过，**换账号就要重新申请**，其中 sam-3d-body 是人工审批、要等：

- `facebook/sam-3d-objects`、`facebook/sam3`、`facebook/sam-3d-body-dinov3`、`facebook/VGGT-Omega`

（弃用线里还有 `black-forest-labs/FLUX.1-Fill-dev`、`facebook/dinov3-vitl16-pretrain-lvd1689m` 两个 gated，不用管。）

验证有没有批下来：用 `hf download` 真去下一个文件，报 403 `GatedRepo` 就是没批。**别拿 `HfApi().model_info()` 成功当证据**，元数据接口是公开的（body4d 那次踩过）。

### 8.2 下载

```bash
cd ~/sam3d/docs/setup
bash -n download_weights.sh                                     # 先查语法（没在旧机器上跑过）
bash download_weights.sh --list                                 # 默认组 / 可选组
HF=~/miniforge3/envs/sam3d/bin/hf bash download_weights.sh sam3d sam3 vlm da3   # 分组下
HF=~/miniforge3/envs/sam3d/bin/hf bash download_weights.sh                      # 或者一次下全部默认组（约 69G）
```

- 默认组：`sam3d sam3 vlm da3 wilddet vggt_omega metric3d body4d`。可选组（要点名才下）：`body4d_vas`（body4d 开补全才用，14G）、`eval`（米制评测脚本用，4G）、`fire3d`（待用户确认，63G）。
- 弃用线的权重在脚本末尾，整段注释掉。
- 下载一律按原来的位置放（HF 缓存 / `--local-dir` 到仓库目录）。body4d 那几个必须 `--local-dir`，不要"先进缓存再复制"，会占两份空间。

### 8.3 权重清单：在用（默认下）

"HF 缓存"指 `~/.cache/huggingface/hub/models--组织--名字`（旧机器上实际在 `/mnt/nvme/jincheng/hf/hub/`）。大小是旧机器上的实际占用。最后一列是 `download_weights.sh` 的组名。

| 本地路径 | 大小 | 来源 | gated | 谁用 / 组 |
|---|---|---|---|---|
| `~/sam3d/sam-3d-objects/checkpoints/hf/` | 13G | HF `facebook/sam-3d-objects`（取仓库里的 `checkpoints/` 子目录） | **是** | sam3d 生成 / sam3d |
| HF 缓存 `Ruicheng/moge-vitl` | 1.2G | HF（本机快照 `ad326bf`） | 否 | sam3d：`checkpoints/hf/pipeline.yaml` 里的深度模型（MoGe v1）/ sam3d |
| `~/.cache/torch/hub/facebookresearch_dinov2_main` + `hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth` | 1.2G | torch.hub `facebookresearch/dinov2`；权重 `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth` | 否 | sam3d：DINOv2 主干 / sam3d |
| `~/sam3d/sam3/checkpoints/sam3.pt` | 3.3G | HF `facebook/sam3` 文件 `sam3.pt`（本机快照 `3c879f3`） | **是** | sam3；body4d 软链复用 / sam3 |
| HF 缓存 `Qwen/Qwen3-VL-8B-Instruct` | 17G | HF（本机快照 `0c351dd`） | 否 | sam3d：`vlm` `verify` `vlm_sizes`（`data/outdoor/scripts/vlm_*.py`、`verify_picks.py`、`pipeline/vlm_list_objects_indoor.py`）/ vlm |
| HF 缓存 `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | 6.3G | HF（本机快照 `b2359bd`），CC BY-NC 4.0 非商用 | 否 | spatiallm：`da3_to_ply.py` 的默认模型 / da3 |
| `~/sam3d/WildDet3D/ckpt/wilddet3d_alldata_all_prompt_v1.0.pt` | 4.5G | HF `allenai/WildDet3D` | 未确认 | wilddet / wilddet |
| HF 缓存 `robbyant/lingbot-depth-postrain-dc-vitl14` | 1.2G | HF（本机快照 `d1b06f7`；WildDet3D 运行时 `hf_hub_download` 自动取） | 未确认 | wilddet：WildDet3D 的深度后端 / wilddet |
| `/mnt/nvme/jincheng/weights/vggt_omega/vggt_omega_1b_512.pt`、`vggt_omega_1b_416_reproduce.pt` | 8.6G | HF `facebook/VGGT-Omega`，FAIR 非商用许可 | **是** | vggt_omega / vggt_omega |
| `/mnt/nvme/jincheng/weights/vggt_omega/skyseg.onnx` | 168M | `https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx` | 否 | vggt_omega：天空分割 / vggt_omega |
| `/mnt/nvme/jincheng/torch_cache/hub/yvanyin_metric3d_main` + `hub/checkpoints/metric_depth_vit_{large,giant2}_800k.pth` | 64M + 1.6G + 5.1G | torch.hub `YvanYin/Metric3D`；权重 `https://huggingface.co/JUGGHM/Metric3D/resolve/main/...` | 否 | metricdepth：`GEOM=vggt_l` 用 Large，`vggt_g2` 用 Giant2 / metric3d |
| `~/sam3d/sam-body4d/checkpoints/sam-3d-body-dinov3/`（`model.ckpt` + `assets/mhr_model.pt`） | 2.7G | HF `facebook/sam-3d-body-dinov3`（本机快照 `11aaa34`） | **是**（人工审批） | body4d / body4d |
| `~/sam3d/sam-body4d/checkpoints/moge-2-vitl-normal/model.pt` | 1.3G | HF `Ruicheng/moge-2-vitl-normal`（必须 v2） | 否 | body4d：估焦距 / body4d |
| `~/sam3d/sam-body4d/checkpoints/depth_anything_v2_vitl.pth` | 1.3G | HF `depth-anything/Depth-Anything-V2-Large` | 否 | body4d：Diffusion-VAS 的深度条件 / body4d |
| `~/sam3d/sam-body4d/checkpoints/sam3/sam3.pt` | 软链 | → `~/sam3d/sam3/checkpoints/sam3.pt`（和 HF 上那份字节数一致） | — | body4d / body4d |

合计约 69G。

### 8.4 权重清单：可选（点名才下）

| 本地路径 | 大小 | 来源 | gated | 说明 / 组 |
|---|---|---|---|---|
| `~/sam3d/sam-body4d/checkpoints/diffusion-vas-{amodal-segmentation,content-completion}/` | 约 7G ×2 | HF `kaihuac/diffusion-vas-*` | 否 | body4d 开补全（`--completion 1`）才用；32 GB 卡上开补全本来就 OOM / body4d_vas |
| HF 缓存 `depth-anything/DA3METRIC-LARGE`、`lpiccinelli/unidepth-v2-vitl14`、`Ruicheng/moge-2-vitl-normal` | 1.3G + 1.4G + 1.3G | HF；旧机器上放在 `HF_HOME=/mnt/nvme/jincheng/hf_cache` 下 | 否 | 只给 09-17 米制评测脚本（`data/indoorOutput/ithor312/analysis_9.17_vggt_omega/metric_eval/`）用，流水线不用 / eval |
| `/mnt/nvme/jincheng/fire3d_ckpt/Fire3D/`（`~/sam3d/Fire3D/checkpoints` 软链到它的上一级） | 63G | HF `hongchi/Fire3D`（本机快照 `84d4246`，= `fire3d download --models`） | 否 | **待用户确认**；不在流水线里 / fire3d |
| `~/.cache/torch/hub/checkpoints/anyup_multi_backbone.pth` | 3.5M | `https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/anyup_multi_backbone.pth`（运行时也会自动下） | 否 | Fire3D 用，**待用户确认** / fire3d |
| `~/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl` | 2.6G | `https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl`（运行时自动下） | 否 | 只有 body4d 官方 `scripts/offline_app.py` 用，我们的 `body4d_*_5090.py` 不用；脚本里注释着 |

### 8.5 权重清单：弃用（默认不下）

已确认 `pipeline/`、`tools/`、`data/outdoor/scripts/` 这些在用代码（outdoor 链 + indoor 物体链 + body4d）都不引用下面这些。`download_weights.sh` 末尾对应的行默认是注释掉的。

| 本地路径 | 大小 | 来源 | gated | 属于 |
|---|---|---|---|---|
| HF 缓存 `manycore-research/SpatialLM1.1-Qwen-0.5B` | 2.3G | HF | 否 | SpatialLM |
| HF 缓存 `manycore-research/SpatialGen-1.0` | 2.9G | HF（推理脚本写死了快照 `6c8c2d1…`，要重下必须钉 `--revision`） | 否 | SpatialGen |
| `~/sam3d/SpatialLM/SpatialLM-Testset`、`~/sam3d/SpatialGen/SpatialGen-Testset` | 59M、1.3G | HF 数据集 `manycore-research/SpatialLM-Testset`、`SpatialGen-Testset`（在仓库目录里，随代码走） | 否 | SpatialLM / SpatialGen |
| HF 缓存 `facebook/VGGT-1B` | 9.4G | HF，CC BY-NC | 否 | 原版 VGGT 对照工具（spatiallm 完整版里的 `vggt`），流水线用的是 VGGT-Ω |
| HF 缓存 `depth-anything/DA3NESTED-GIANT-LARGE` | 6.3G | HF，DA3 老版（作者标了 deprecated） | 否 | 没人用（`da3_to_ply.py` 默认 -1.1） |
| `~/sam3d/InvSR/weights/noise_predictor_sd_turbo_v5.pth` + `models--stabilityai--sd-turbo`（HF 缓存布局） | 5.0G | `OAOA/InvSR`、`stabilityai/sd-turbo` | 否（sd-turbo 未确认） | 墙面贴图线（invsr） |
| `~/.cache/torch/hub/checkpoints/big-lama.pt` | 196M | GitHub `enesmsahin/simple-lama-inpainting` release | 否 | LaMa（spatialgen 环境；室内 LM 链地板、墙贴图线） |
| `/mnt/nvme/jincheng/layerpano3d_ckpt/`：`ControlNetLama.pth`、`depth_anything_v2_vitl.pth`、`Infusion/`、`pano_lora_720*1440_v1.safetensors`、`pasd/`（SD1.5 diffusers 子目录 + `checkpoint-100000` + `pasd.zip`）、`sam_vit_h_4b8939.pth` | 25G | `lllyasviel/Annotators`、`depth-anything/Depth-Anything-V2-Large`、`Johanan0528/Infusion`、`ysmikey/Layerpano3D-FLUX-Panorama-LoRA`、`stable-diffusion-v1-5/stable-diffusion-v1-5`、阿里云 OSS 的 `pasd.zip`（**URL 未确认**）、`https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth` | 否 | LayerPano3D |
| HF 缓存 `black-forest-labs/FLUX.1-Fill-dev`（只有 diffusers 分片） | 32G | HF，非商用 | **是** | LayerPano3D |
| HF 缓存 `llava-hf/llava-1.5-7b-hf` | 14G | HF | 否 | LayerPano3D（打标） |
| HF 缓存 `shi-labs/oneformer_coco_swin_large` + 数据集 `shi-labs/oneformer_demo` | 1.8G | HF | 否 | LayerPano3D（语义分割） |
| HF 缓存 `ysmikey/Layerpano3D-FLUX-Panorama-LoRA`、`lllyasviel/Annotators`、`depth-anything/Depth-Anything-V2-Large` | 26M + 196M + 1.3G | HF（缓存里这几份是 LayerPano3D 装的；body4d 有自己 `--local-dir` 的那份 DA-V2） | 否 | LayerPano3D |
| HF 缓存 `laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k` | 2.4G | HF | 否 | **未确认**（在用代码不引用；下载时间 09-11 06:09 和 LayerPano3D 运行时间吻合） |
| `~/sam3d/Pano2Room/checkpoints/`：`big-lama.ckpt` + `big-lama-config.yaml`、`omnidata_dpt_{depth,normal}_v2.ckpt` | 4.1G | `smartywu/big-lama`（`big-lama.zip`）、`clay3d/omnidata`（仓库 README 给的是 Dropbox 链接） | 否 | Pano2Room |
| HF 缓存 `sd2-community/stable-diffusion-2-inpainting`、`clay3d/omnidata`、`smartywu/big-lama` | 2.5G + 3.7G + 364M | HF（官方 `stabilityai/stable-diffusion-2-inpainting` 已下架，代码改成这个镜像） | 否 | Pano2Room |
| HF 缓存 `facebook/dinov3-vitl16-pretrain-lvd1689m` | 1.2G | HF | **是** | TRELLIS.2（环境 09-10 已删） |
| HF 缓存 `facebook/dinov2-with-registers-small`、`facebook/dpt-dinov2-small-nyu` | 85M + 142M | HF | 否 | PixARMesh（环境已删） |

旧机器上还有几样运行时自动下的缓存，都不用专门下：`~/.cache/torch/hub/checkpoints/` 里的 `jx_vit_base_resnet50_384-9fd3c705.pth`（timm，Pano2Room 的 omnidata DPT 用）、`fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth`（torchvision，09-20 下的，用途未确认）、`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`（Fire3D 模型包的 `external/` 里已有同名文件）。

### 8.6 几个 HF 模型的归属判定（依据）

| 模型 | 判定 | 依据 |
|---|---|---|
| `Qwen/Qwen3-VL-8B-Instruct` | 在用（sam3d） | `data/outdoor/scripts/vlm_list_objects.py`、`vlm_sizes.py`、`verify_picks.py`、`pipeline/vlm_list_objects_indoor.py` 写死 `MODEL="Qwen/Qwen3-VL-8B-Instruct"` |
| `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | 在用（spatiallm） | `Depth-Anything-3/da3_to_ply.py` `--model` 默认值；流水线调它时不传 `--model` |
| `depth-anything/DA3NESTED-GIANT-LARGE`（老版） | 弃用 | 没有任何在用代码引用；README 说 -1.1 修了训练 bug |
| `Ruicheng/moge-vitl`（MoGe v1） | 在用（sam3d） | `sam-3d-objects/checkpoints/hf/pipeline.yaml` 的深度模型 `moge.model.v1.MoGeModel.from_pretrained` |
| `Ruicheng/moge-2-vitl-normal`（MoGe-2） | 在用（body4d 的 `--local-dir` 那份）；HF 缓存那份只给评测 | `sam-body4d/configs/body4d.yaml` 的 `fov_path`；`metric_eval/eval_mono_metric.py` |
| `robbyant/lingbot-depth-postrain-dc-vitl14` | 在用（wilddet） | `WildDet3D/wilddet3d/inference.py` 在 `skip_pretrained=True` 时用 `hf_hub_download` 取它的配置和权重 |
| `facebook/VGGT-1B` | 弃用 | 只有 `vggt/` 仓库里的脚本（`vggt_to_ply.py`、官方 demo）用；流水线的 `GEOM=vggt_*` 用的是 VGGT-Ω |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | 弃用（TRELLIS.2 遗留） | 09-02 下载，TRELLIS.2 笔记里写着它是 gated、用于 trellis2；body4d 的 DINOv3 是 `torch.hub.load(本地目录, pretrained=False)`，权重在 `model.ckpt` 里；Fire3D 模型包自带一份 |
| `llava-hf/llava-1.5-7b-hf` | 弃用（LayerPano3D） | LayerPano3D 存档笔记的权重表；在用代码不引用 |
| `shi-labs/oneformer_coco_swin_large` | 弃用（LayerPano3D） | 同上 |
| `laion/mscoco_finetuned_CoCa-…` | 弃用，**归属未确认** | 在用代码不引用；任何笔记都没记；只能从下载时间推测是 LayerPano3D |
| `clay3d/omnidata` | 弃用（Pano2Room） | `Pano2Room/modules/geo_predictors/omnidata/*` |
| `sd2-community/stable-diffusion-2-inpainting` | 弃用（Pano2Room） | `Pano2Room/modules/inpainters/SDFT_inpainter.py`、`train_SDFT.py` |

---

## 9. 验收

```bash
E=~/miniforge3/envs
$E/sam3d/bin/python -c "import torch,pytorch3d,gsplat,kaolin,nvdiffrast.torch,spconv.pytorch,moge,sam3d_objects,transformers; print(torch.__version__, torch.cuda.get_device_name(0), transformers.__version__)"
$E/sam3/bin/python -c "import torch,sam3,numpy,cv2; print(torch.cuda.is_available(), numpy.__version__, cv2.__version__)"
$E/spatiallm/bin/python -c "import torch,depth_anything_3,trimesh,scipy,matplotlib,imageio_ffmpeg,numpy; print('DA3 ok', numpy.__version__)"
( cd ~/sam3d/WildDet3D && $E/wilddet/bin/python -c "import sys; sys.path.insert(0,'.'); import vis4d_cuda_ops, wilddet3d; print('ok')" )
/mnt/nvme/jincheng/envs/vggt_omega/bin/python -c "import vggt_omega,onnxruntime; print('ok')"
/mnt/nvme/jincheng/envs/metricdepth/bin/python -c "import torch,mmengine,moge,unidepth; print('ok')"
$E/body4d/bin/python -c "import torch,detectron2._C,sam3,moge; print(torch.ops.detectron2.box_iou_rotated)"
```

端到端（权重下好之后）：

- SAM 3D Objects：`source ~/sam3d/env.sh && cd ~/sam3d/sam-3d-objects && python demo.py`（旧机器约 3 分钟）
- DA3（精简后的 spatiallm 一定要跑这个）：`source ~/sam3d/env_spatiallm.sh && cd ~/sam3d/Depth-Anything-3 && python da3_to_ply.py --images ~/sam3d/vggt/examples/kitchen/images --out /tmp/da3_test --use-ray-pose && python da3_to_spatiallm.py --pred /tmp/da3_test --out /tmp/da3_test_zup.ply`
- 整条流水线：`~/miniforge3/envs/sam3d/bin/python ~/sam3d/pipeline/outdoor_pipeline.py --list` 看步骤，`-n` 干跑打印命令；驱动用法见 `docs/WORKFLOW.md`（`drive.py --scene <场景> --run <轮次>`，室内加 `--indoor`）

---

## 10. 已知坑（和安装有关的）

1. **numpy 被顶到 2.x**：sam3d、sam3、spatiallm（DA3）、wilddet 都必须是 1.26.4。kaolin、rembg、nvdiffrast 这类包装完都可能把 numpy 拉高，**装完任何新包都复查**。
2. **setuptools ≥ 81 删了 `pkg_resources`**：lightning 2.3.3（sam3d）、detectron2 的 model_zoo（body4d）、SAM 3（sam3）、WildDet3D 的 sam3（要 <80）都会炸。而 `pip freeze` 不记 setuptools，所以每节都单独钉。
3. **setuptools 80 下的 `pip install -e` 会走 PEP 517 隔离构建**，隔离环境里没有 torch，CUDA 扩展编不了。所以 CUDA 扩展一律 `--no-build-isolation`。
4. **spconv**：cu118 版没有 sm_120；sam3d 用 cu121 2.3.8 预编译轮子直接能跑。
5. **body4d 必须和 sam3 分开装**（内嵌的 sam3 包名相同，装进去不报错但会静默替换，室外链的跟踪步骤就坏了）。
6. **transformers 版本各环境不一样，别统一**：sam3d 4.57.1（Qwen3-VL）、body4d 4.57.6、wilddet 5.16.1；精简后的 spatiallm 里没有 transformers。给 sam3d 加包时不要动 huggingface_hub 0.36.2。
7. **SAM3D 生成步骤是离线跑的**（`HF_HUB_OFFLINE=1`，不然 HF 一抽风就 504、批量生成中途断），所以 MoGe v1 和 torch.hub 的 DINOv2 必须提前下好；`run.py` 还要 `CONDA_PREFIX`，直接调环境里的 python 时要手动给。
8. **nvdiffrast / gsplat 在 sam3d 里是运行时即时编译的**，运行环境也得有 `CUDA_HOME` 和对的 `TORCH_CUDA_ARCH_LIST`；换卡后清一下 `~/.cache/torch_extensions`。
9. **gated 权限要真下一个文件来验证**，`model_info()` 成功不算数。
10. **body4d 的权重一律 `--local-dir`**，先进缓存再复制会让 15G 变 30G。
11. **系统没有 ffmpeg / unzip / cmake**：视频用 `imageio-ffmpeg`（`cv2.VideoWriter` + `mp4v` 不报错但浏览器不认，要 `libx264` + `yuv420p` + 宽高偶数）；解 zip 用 python `zipfile`；cmake 从 conda 装（sam3d 环境里有）。
12. **根分区满**：旧机器根分区 97–99%，大件都放 nvme，conda 包缓存 / pip 缓存 / `TMPDIR` 也指到 nvme。新机器根分区小的话照做。
13. **可编辑安装记的是绝对路径**：代码放的位置一变，sam3d / sam3 / spatiallm / body4d / vggt_omega / metricdepth 这几个环境都要重跑对应的 `pip install --no-deps -e`。
14. **UniDepth 和 Metric3D 的 hub 代码不在 `~/sam3d` 下**，代码带走时不会跟着走，要重新拉（第 3 节、`download_weights.sh metric3d`）。
15. **spatiallm 这个名字不能改**，流水线按名字找环境；里面只装 DA3。
16. **机器是多人共享的**（旧机器上有 kevin 等人），GPU 要错峰；流水线用 `/tmp/outdoor_pipeline_gpu.lock` 文件锁串行跑吃显存的步骤。
17. **`env_pixar.sh` 指向早已删掉的 `pixar` 环境**（PixARMesh），别 source 它。

弃用环境的坑（xformers FA3、`pidfd_getfd`、`simple-knn` 缺 `<cfloat>`、tinycudann 二阶反传、Fire3D 的 flash-attn 强制编译等）写在第 11 节各自下面。

---

## 11. 已弃用 / 待确认（默认不装）

这些环境照常导出了 `.yml` / `.pip.txt`，但**默认不装**；它们独有的权重在第 8.5 节（fire3d 在 8.4 可选组），`download_weights.sh` 里默认不下。下面的步骤只作存档，真要复现时参考。

### 11.1 spatiallm 完整版（SpatialLM 1.1 + VGGT + DA3，已不需要）

新机器上的 spatiallm 只装 DA3（6.3 节）。旧机器上的完整版多了 SpatialLM 和原版 VGGT，清单是 `envs/spatiallm.pip.txt`，详见 `SpatialLM/NOTES_5090.md`、`vggt/NOTES_5090.md`。和精简版相比多出的步骤：

```bash
P=~/miniforge3/envs/spatiallm
$P/bin/pip install torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pipreq spatiallm $P -f https://data.pyg.org/whl/torch-2.8.0+cu128.html      # torch_scatter==2.1.2+pt28cu128 只在 PyG 源里
# flash-attn 2.8.3 官方 release 轮子（已含 sm_120，不用编）
$P/bin/pip install --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
$P/bin/pip install --no-deps -e ~/sam3d/vggt
# SpatialLM 不能 pip install -e（pyproject 里 torch ^2.4.1+cu124 报 invalid metadata），用 .pth 把仓库加进 sys.path
echo /home/jincheng/sam3d/SpatialLM > "$($P/bin/python -c 'import site; print(site.getsitepackages()[0])')/spatiallm_repo.pth"
```

当年钉的：transformers 4.46.1 / tokenizers 0.20.3（SpatialLM 上限）、rerun-sdk 0.36.0（和用户本机查看器版本绑定）、spconv-cu126 2.3.8、torch-scatter 用 PyG 的 torch-2.8.0+cu128 轮子。SpatialLM 的 `visualize.py` 改了两处（rerun 新接口）。pip check 预期报警：rerun-sdk 0.36.0 要 numpy≥2。

### 11.2 spatialgen —— SpatialGen 1.0 + LaMa（弃用）

py3.10.21 / torch 2.8.0+cu128 / 9.3G。详见 `SpatialGen/NOTES_5090.md`。注意室内 LM 链的地板补洞（`floor_texture_5090.py`）和墙贴图线（`wall_pipeline/`）用的 LaMa（simple-lama-inpainting）也只装在这个环境里，一起弃用。

```bash
N=spatialgen; P=~/miniforge3/envs/spatialgen
mkenv $N $P && cuda_env
$P/bin/pip install torch==2.8.0 torchvision==0.23.0 xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128
pipreq $N $P
$P/bin/pip install --no-deps "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
# pytorch3d 0.7.9 源码编（约 3 分钟）。旧机器上是 08-31 clone 的 main 分支 HEAD（--depth 1），commit 没记下，未确认
git clone https://github.com/facebookresearch/pytorch3d.git /tmp/_build_pytorch3d
( cd /tmp/_build_pytorch3d && FORCE_CUDA=1 MAX_JOBS=64 $P/bin/pip install --no-build-isolation --no-deps . )
cd ~/sam3d/SpatialGen && git submodule update --init --recursive     # Sparse-RaDeGS @ ee7b413
$P/bin/pip install --no-build-isolation --no-deps src/recons/Sparse-RaDeGS/submodules/diff-gaussian-rasterization
$P/bin/pip install --no-build-isolation --no-deps src/recons/Sparse-RaDeGS/submodules/simple-knn
```

- `simple-knn` 要先在 `simple_knn.cu` 顶部加 `#include <cfloat>`（CUDA 12.8 + 新 GCC 下 `FLT_MAX` 未定义）。这个改动在 Sparse-RaDeGS 子模块里。
- 仓库打了 7 处补丁：**xformers 0.0.32 在 sm_120 上错选 Hopper 专用的 FA3 内核**（只看算力 ≥ 9.0），注意力一跑就 `CUDA error: invalid argument`，补丁是 `xformers.ops.fmha.dispatch._get_use_fa3 = lambda: False`；推理 DataLoader `num_workers=0`（旧机器不允许跨进程拿 fd，`pidfd_getfd: Operation not permitted`）；Sparse-RaDeGS 缺 `guidance/` 模块和 `lambda_diffusion` 参数两处兜底；spatiallm 数据分支；`SKIP_GS_RECON` 开关。
- 钉的：transformers 4.49.0（<4.50，配 diffusers 0.32.0）、numpy 1.26.4、typeguard 2.13.3。
- 推理脚本 `scripts/infer_spatialgen_*_5090.sh` 写死了 HF 缓存快照路径 `/home/jincheng/.cache/huggingface/hub/models--manycore-research--SpatialGen-1.0/snapshots/6c8c2d1…`。
- pip check 预期报警：simple-lama-inpainting 要 pillow<10。

### 11.3 invsr —— InvSR 超分（弃用，墙面贴图线）

py3.10.21 / torch 2.8.0+cu128 / 7.1G。`wall_pipeline/run.sh` 第 8 步用。

```bash
N=invsr; P=~/miniforge3/envs/invsr
mkenv $N $P
$P/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pipreq $N $P
# InvSR 自带一个 diffusers 分叉（包名 diffusers 0.30.0.dev0，源码在 InvSR/src），可编辑安装的就是它
$P/bin/pip install --no-deps -e ~/sam3d/InvSR
```

钉的：**transformers 4.46.3（5.x 会炸）**；diffusers 必须是仓库自带的分叉；官方的 torch 2.4.0+cu121 / xformers 0.0.27 换成 torch 2.8.0+cu128、不装 xformers。PiSA-SR 钉 torch 2.0.1，5090 上跑不了，所以当时超分选了 InvSR。pip check 没有报警。

### 11.4 fire3d —— FIRE3D（**不在流水线里，待用户确认**）

prefix 环境 `/mnt/nvme/jincheng/envs/fire3d`，py3.10.20 / **torch 2.7.1+cu128** / 16G；权重 63G（第 8.4 节可选组 `fire3d`）。09-15 装的验证性实验，iTHOR 示例跑通过（22/22 物体、363 秒、峰值 20.9G）。详见 `Fire3D/_5090/NOTES_5090.md`。要装的话**直接用验证过的安装脚本**（约 17 分钟，其中 flash-attn 编译 8 分钟）：

```bash
P=/mnt/nvme/jincheng/envs/fire3d
mkdir -p /mnt/nvme/jincheng/cache/{conda_pkgs,pip,tmp}
# conda 那一半：官方 environment.yml（nvidia + conda-forge 源，含 cuda-toolkit 12.8、cmake、ninja、ffmpeg、gcc 13）
CONDA_PKGS_DIRS=/mnt/nvme/jincheng/cache/conda_pkgs \
  ~/miniforge3/bin/conda env create -p $P -f ~/sam3d/Fire3D/environment.yml
# （想要和旧机器完全一样的 conda 版本就改用: mkenv fire3d $P）
bash ~/sam3d/Fire3D/_5090/install_5090.sh 2>&1 | tee /tmp/fire3d_install.log
[ -e ~/sam3d/Fire3D/checkpoints ] || ln -s /mnt/nvme/jincheng/fire3d_ckpt ~/sam3d/Fire3D/checkpoints
```

`install_5090.sh` 做的事（也就是清单装不回来的部分）：

- `CUDA_HOME` 指向环境本身（conda 装的 cuda-toolkit 12.8），`TORCH_CUDA_ARCH_LIST=12.0`，`MAX_JOBS=32`；conda 包缓存、pip 缓存、`TMPDIR` 都指到 nvme。
- torch 2.7.1+cu128（官方版本不变）；`pip install -e ~/sam3d/Fire3D[dev]`。
- spconv-cu118 → **spconv-cu121==2.3.8**（cu118 没有 sm_120）。
- flash-attn 2.7.3 **强制源码编译，只编 sm_120**：`FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=120`（官方预编译轮子早于 5090，没有 sm_120 内核；换卡要改这个数，如 H100 用 `90`、A100/3090 用 `80`）。
- `--no-build-isolation` 装 nvdiffrast @ `253ac4f`、pytorch3d @ `75ebeea`、utils3d @ `9a4eb15`、FlexGEMM @ `6dd94a8`。
- clone DINOv3 到 `third_party/dinov3` 并 checkout `31703e4`；clone Eigen 3.4.0 到 CuMesh 和 o-voxel 的 third_party。
- `--no-build-isolation --no-deps` 装 `trellis2_x2/CuMesh`、`trellis2_x2/o-voxel`。

装完如果某个依赖版本和旧机器不一样导致出问题，用清单对齐（flash-attn 要排除，否则会去 PyPI 重新下/编）：`SKIP='^flash_attn==' pipreq fire3d /mnt/nvme/jincheng/envs/fire3d`。

注意：目录挪过就要重跑 `pip install --no-deps -e ~/sam3d/Fire3D`（`fire3d.pip.txt` 里 cumesh / o_voxel 记的还是更早的 `/mnt/nvme/jincheng/Fire3D/...`，那是挪目录之前装的，不影响使用）；32 GB 卡必须用 `--protocol _5090/fire3d_video_v1_32g.json`（解码批量 16→4、网格后处理 8→2，生成批量 16 不动），80/96 GB 卡可以换回官方协议；CuMesh 自己申请显存、不归 PyTorch 管；没装 Blender，跑的时候加 `--skip-render`。跑法：`cd ~/sam3d/Fire3D && HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True fire3d infer --dataset ithor --scene-id iTHOR_FloorPlan312_physics --gpu 0 --skip-render --protocol _5090/fire3d_video_v1_32g.json`。pip check 没有报警。

### 11.5 pano2room —— Pano2Room（已排除的路线）

全景 → 3DGS。36 分钟一张、提分辨率深度就崩，已排除（见 `Pano2Room/NOTES_5090.md`）。仓库 `~/sam3d/Pano2Room` 还在。环境是从 spatialgen 克隆出来再加东西的，py3.10.21 / torch 2.8.0+cu128，9.5G。

```bash
N=pano2room; P=~/miniforge3/envs/pano2room
mkenv $N $P && cuda_env
$P/bin/pip install "setuptools==80.10.2"
$P/bin/pip install torch==2.8.0 torchvision==0.23.0 xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128
pipreq $N $P
$P/bin/pip install --no-deps "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
# pytorch3d 0.7.9：同 11.2（main 分支源码编，commit 未确认）
( cd /tmp/_build_pytorch3d && FORCE_CUDA=1 $P/bin/pip install --no-build-isolation --no-deps . )
# 3DGS 扩展：和 spatialgen 共用 SpatialGen 子模块里的源码
$P/bin/pip install --no-build-isolation --no-deps \
  ~/sam3d/SpatialGen/src/recons/Sparse-RaDeGS/submodules/diff-gaussian-rasterization \
  ~/sam3d/SpatialGen/src/recons/Sparse-RaDeGS/submodules/simple-knn
# 代码 import 的是 LucidDreamer 里内嵌的 depth-diff-gaussian-rasterization-min（不是 README 说的那个包），glm 已在它的 third_party/ 里
$P/bin/pip install --no-build-isolation --no-deps \
  ~/sam3d/Pano2Room/_ext_src/LucidDreamer/submodules/depth-diff-gaussian-rasterization-min
# tiny-cuda-nn（它读 TCNN_CUDA_ARCHITECTURES 选算力；旧机器上的编译参数没记，未确认）
TCNN_CUDA_ARCHITECTURES=120 $P/bin/pip install --no-build-isolation --no-deps \
  "git+https://github.com/NVlabs/tiny-cuda-nn/@749dd70c5afc5a9dadb85e5652ed65d55e0ba187#subdirectory=bindings/torch"
```

仓库里的 13 处改动见 patches 和 NOTES：SD2 inpainting 官方仓库下架换成 `sd2-community` 镜像；albumentations 2.x 兼容层；PyTorch ≥ 2.6 的 `torch.load(weights_only=False)` 9 处；**tinycudann 的 HashGrid 二阶反传在 sm_120 上 illegal memory access**，换成纯 PyTorch 哈希编码（任何做 SDF / eikonal 约束的项目在 Blackwell 上都会撞到）；分块求值防 OOM；分辨率常量联动。pip check 预期报警：opencv-python-headless 5.0 要 numpy≥2；simple-lama-inpainting 要 pillow<10。

### 11.6 layerpano3d —— LayerPano3D（09-15 判定没用，仓库已删）

`~/sam3d/LayerPano3D` 在旧机器上已经删了，环境还在（py3.9.23 / torch 2.8.0+cu128，11G），`.pip.txt` 里引用的本地源码目录都不存在了，**没法按本文直接重建**。真要复现只能重新 clone 上游（https://github.com/3DTopia/LayerPano3D ，commit 未记录），照 `docs/archived_notes/LayerPano3D_NOTES_5090.md` 重做，要点：

- conda-forge 装编译 360monodepth 的依赖：`cmake "ceres-solver=2.2.0=*cpu*" glog gflags eigen suitesparse libopencv libboost-devel gtest pybind11`（不指定 cpu 会解到要 CUDA ≥ 12.9 的 GPU 版 Ceres）。这些已经在 `envs/layerpano3d.yml` 的 conda 段里，`mkenv layerpano3d <路径>` 会装上。
- requirements 过滤后安装（用 `-c` 锁住 torch），跳过 natten / mmcv / detectron2 等；补上 omegaconf / accelerate / timm；`setuptools<81`（旧机器 80.9.0）。
- CUDA 扩展（diff-gaussian-rasterization、simple-knn）用 `pip install --no-build-isolation .`，不用 `-e`（setuptools 80 的 `-e` 会转去隔离构建）；simple-knn 加 `#include <cfloat>`。
- 360monodepth 用 cmake 编 `depth_stitch` 和 `EigenSolvers`（完整命令在存档笔记里），再 `pip install --no-build-isolation .` 装 `code/cpp/python`（包名 instaOmniDepth）。
- `segment_anything` 从 git 装：`git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf`。
- 不装 xformers；PASD 两处要改（`enable_xformers_memory_efficient_attention=False`，`is_xformers_available()` 漏括号）。
- 权重见第 8.5 节，大头是 FLUX.1-Fill-dev（gated，32G）和 `/mnt/nvme/jincheng/layerpano3d_ckpt`（25G）。
