# 在新机器上部署 4Drecon（从 render04 直接拷）

2026-11-18 整理。给能 SSH 到 render04 的同事用：代码、模型权重、conda 环境都从 render04 拷，不用去 HuggingFace 申请权限，也不用重新下载 70 G 权重。

**先说清楚两件事：**

- 这份流程**没有人完整走过一遍**。render04 自己是 9 月照 `SETUP.md` 从头装的，不是拷来的。下面"拷环境"那一步在别的机器上没验证过，所以第 4 步的验收一定要跑。
- 拷环境只在新机器和 render04 **同构**时才有希望直接能用（见第 0 步）。不同构就只拷代码和权重，环境照 `SETUP.md` 重装（第 3 步的 B 路）。

---

## 0. 新机器要满足什么

| 项 | render04 上是什么 | 新机器 |
|---|---|---|
| 系统 | Ubuntu 22.04 | 一样最好 |
| 用户名 | `ubuntu`（家目录 `/home/ubuntu`） | **必须一样**，否则环境不能直接拷 |
| 显卡 | 8 × RTX 3090（24 G） | 单卡 24 G 显存起步。不是 3090 / A6000 / A10 这一代的，环境要重装（看 `SETUP.md` 第 1 节） |
| 驱动 / CUDA | 驱动 580，CUDA 12.8 装在 `/usr/local/cuda-12.8` | CUDA 12.8 **必须装在同一个路径**（有几个包是运行时现编译的） |
| 磁盘 | — | 至少 250 G 空闲（代码 45 G + 环境 130 G + 权重 45 G），另外每跑一个视频约 1～2 G |
| 其他 | `tmux`、`rsync` | 要有。后台跑靠 tmux |

系统包（要 root，装一次）：

```bash
sudo apt-get install -y build-essential tmux rsync libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libglvnd0 libsm6 libxext6 libxrender1
```

## 1. 建好路径

代码里有几处写死的路径，在新机器上让它们都存在：

```bash
# 大盘上的目录。大盘不在 /mnt/nvme 就把第一行换成软链: sudo ln -s /你的大盘/jincheng /mnt/nvme/jincheng
sudo mkdir -p /mnt/nvme/jincheng && sudo chown ubuntu:ubuntu /mnt/nvme/jincheng
mkdir -p /mnt/nvme/jincheng/{envs,weights,torch_cache,hf}

# HuggingFace 缓存放大盘(render04 上就是这个软链)
mkdir -p ~/.cache && ln -sfn /mnt/nvme/jincheng/hf ~/.cache/huggingface

# 人体模型的配置里写的是 /home/jincheng/sam3d/..., render04 上靠这个软链让它成立
sudo ln -sfn /home/ubuntu /home/jincheng
```

## 2. 拷代码和权重（两条路都要做）

下面的命令都**在新机器上**执行，从 render04 往回拉。把 `render04` 换成你 SSH 用的主机名。可以反复跑，断了接着传。

```bash
# 代码 + 上游模型仓库 + 仓库里的权重, 约 45 G。跑出来的结果、试验目录、别人的数据不拷
rsync -a --info=progress2 \
  --include='data/' --include='data/outdoor/' --include='data/outdoor/scripts/***' --include='data/outdoor/ISSUES.md' \
  --exclude='data/outdoor/*' --exclude='data/*' \
  --exclude='.worktrees/' --exclude='redir/' --exclude='reports/' --exclude='input/' \
  --exclude='DuoMo/' --exclude='CARI4D/' --exclude='FlowHMR/' --exclude='OmniEgoCap/' \
  --exclude='carryfit/runs/' --exclude='proclegs/outputs/' --exclude='sam-3d-objects/my_images/*' \
  --exclude='*.zip' --exclude='*.tar.xz' --exclude='__pycache__/' --exclude='*.log' \
  render04:~/sam3d/ ~/sam3d/

# 放视频和结果的目录(上面没拷, 这里建空的)
mkdir -p ~/sam3d/data/outdoor/{dataset,result} ~/sam3d/data/indoor/dataset ~/sam3d/data/indoorOutput ~/sam3d/sam-3d-objects/my_images

# 仓库外面的权重和缓存, 约 45 G
rsync -a --info=progress2 render04:/mnt/nvme/jincheng/{weights,torch_cache,hf,UniDepth} /mnt/nvme/jincheng/
rsync -a --info=progress2 render04:~/.cache/torch/ ~/.cache/torch/
```

## 3. 环境

### A 路：直接拷（新机器和 render04 同构时）

```bash
# 约 130 G
rsync -a --info=progress2 render04:~/miniforge3/ ~/miniforge3/
rsync -a --info=progress2 render04:/mnt/nvme/jincheng/envs/ /mnt/nvme/jincheng/envs/
rsync -a --info=progress2 render04:~/venvs/ ~/venvs/
rm -rf ~/.cache/torch_extensions        # 运行时现编译的缓存, 换机器清掉让它重编
```

不用的环境可以不拷，省 50 G 左右：在第一条 rsync 里加
`--exclude='envs/duomo/' --exclude='envs/rohm/' --exclude='envs/flowhmr/' --exclude='envs/omniegocap/' --exclude='envs/pixie/'`。
`envs/cari4d` 要留着（判切镜头用它里面的 ffmpeg）。

### B 路：重装（用户名、系统或显卡不一样时）

照 `docs/setup/SETUP.md` 做：第 1、5、6 节建 7 个基础环境，第 12 节装后来新增的（FoundationPose 环境、Blender 的 Python、ffmpeg）。
`docs/setup/scripts_3090/finish.sh` 是 render04 上实际跑通的安装脚本，国内网络可以直接改着用。
权重和上游代码第 2 步已经拷好了，文档里"下载权重""clone 上游仓库""打补丁"这几步**都跳过**。大约要一天。

## 4. 验收（必做）

```bash
# 1) 各环境能不能 import、看不看得见显卡、流水线能不能起
bash ~/sam3d/docs/setup/scripts_3090/verify.sh
```

这个脚本里第 3、5 段写的是 render04 上的场景名（soccer、office），新机器上没有，那几行失败是正常的；第 1、2、4 段要全过。

```bash
# 2) 后来新增的几样
~/miniforge3/envs/fpose/bin/python -c "import torch,kornia,nvdiffrast.torch,pytorch3d; print('fpose ok')"
~/venvs/bpy/bin/python -c "import bpy; print('bpy', bpy.app.version_string)"
~/miniforge3/envs/cari4d/bin/ffmpeg -version | head -1
ls -la ~/sam3d/sam3/checkpoints/sam3.1_multiplex.pt ~/sam3d/FoundationPose/weights/*/model_best.pth
```

```bash
# 3) 真跑一个短视频。-n 干跑只打印命令, 发现不了真问题
```

见下一节。第一个视频建议拿 render04 上跑通过的：`rsync -a render04:~/sam3d/data/outdoor/dataset/girloutdoor ~/sam3d/data/outdoor/dataset/`，在 render04 上它全程各步耗时合计约 10 分钟。

## 5. 怎么跑

**放视频**（文件名必须是 `video.mp4`）：

- 室外：`~/sam3d/data/outdoor/dataset/<场景名>/video.mp4`
- 室内：`~/sam3d/data/indoor/dataset/<场景名>/video.mp4`

场景名用英文小写加下划线。**前 3 个字母加轮次不要和别的场景重复**（比如 `nfl` 和 `nfl1` 的 `r1` 会撞）：生成的模型按这个标签存目录，撞了会互相覆盖，而且不报错。

### 全自动跑到底（批量测试用）

一次一个视频，命令会一直等到跑完。五道闸口全部自动通过，没有人把关。

```bash
cd ~/sam3d && bash tools/run_to_end.sh <场景名> r1 DROP_WORDS=
```

室内：

```bash
cd ~/sam3d && bash tools/run_to_end.sh <场景名> r1 --indoor DROP_WORDS=
```

- `DROP_WORDS=`（等号后面空着）一定要写。不写的话它会默认作废五个橄榄球相关的词。
- 这个脚本固定按"画面里有人"跑。
- 轮次目录已经存在会拒绝，换个轮次名（`r2`）。
- 限制用哪些卡：在命令最后加 `MAX_GPUS=4 GPU_POOL=0,1,2,3`。想两个视频同时跑，开两个终端，各给一组卡。
- 一批视频就写个循环挨个调它。

### 人看闸口（正常用法，结果更好）

```bash
cd ~/sam3d && ~/miniforge3/envs/sam3d/bin/python tools/gate_ui.py --scene <场景名> --run r1 --port 8765
```

网页只监听本机。在自己电脑上 `ssh -L 8765:localhost:8765 ubuntu@<那台机器>`，浏览器开 `http://localhost:8765/new` 新建一轮，之后每道闸在网页上看图、点继续。室内加 `--indoor`。

命令行等价：

```bash
PY=~/miniforge3/envs/sam3d/bin/python; D=~/sam3d/.claude/skills/run-recon-pipeline/drive.py
$PY $D --scene <场景名> --run r1            # 跑到下一道闸就停
$PY $D --scene <场景名> --run r1 --go       # 看过了, 继续
$PY $D --scene <场景名> --run r1 --status   # 现在停在哪
```

### 看进度、拿结果

- 看板：`bash ~/sam3d/tools/watch_run.sh`
- 日志：`<结果目录>/logs/drive_<时间>.log`；每步耗时：`<结果目录>/logs/timing.tsv`；每一步自己的日志也在 `logs/` 下
- 室外结果：`~/sam3d/data/outdoor/result/<场景名>/<轮次>/`
- 室内结果：`~/sam3d/data/indoorOutput/<场景名>/<轮次>/`
- 交付物在结果目录的 `12_motion/` 下：
  - `<场景名>_scene.fbx`：只有场景
  - `<场景名>_scene_people.fbx`：场景 + 人（带骨骼动画和原视频的相机）
  - `<场景名>_scene_people.blend`：Blender 工程
  - `fbx_per_person/`：每人一个 FBX
- 验收视频：结果目录下的 `09_people_check.mp4`（原片 | 重建 + 人 | 叠加）

## 6. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 第 1 段 `vlm` 退出码 1 | 多半是没抽到帧：视频不叫 `video.mp4` |
| `[失败] 没有这个场景` | 场景名打错，或视频放错了室内 / 室外的目录 |
| CUDA out of memory | 卡上有别的进程。流水线自己会排队，但管不了外面的进程；用 `GPU_POOL` 避开 |
| 找不到 `/mnt/nvme/jincheng/...` 或 `/home/jincheng/...` | 第 1 步的目录和软链没建 |
| `import` 报错、提示找不到 `.so` 或路径里有别的用户名 | 环境是拷来的但机器不同构，改走第 3 步的 B 路重装 |
| nvdiffrast / gsplat 编译报错 | 没有 `/usr/local/cuda-12.8`，或没清 `~/.cache/torch_extensions` |
| 导出 FBX 那步失败 | `~/venvs/bpy` 不能用。它依赖 `~/miniforge3/envs/sam3d` 里的 Python 3.11，两个要一起拷 |
| 有切镜头的片子，人的位置在切点附近乱跳 | `~/miniforge3/envs/cari4d/bin/ffmpeg` 不存在，切镜头没检测出来（这种情况不报错） |
| 跟踪那步找不到 `sam3.1_multiplex.pt` | 权重没拷全。临时绕开：在 `<结果目录>/00_run_config.json` 的"覆盖"里写 `"SAM3_VERSION": "sam3"` |

## 7. 其他要知道的

- **模型许可**：有几个模型是非商用许可或要申请才能下的（SAM 3、SAM 3D Objects、SAM 3D Body、VGGT-Ω、Depth Anything 3 等）。从 render04 拷只是省了下载，使用范围还是受各自许可约束。
- **代码也在 GitHub**：`https://github.com/Davecodingking/4Drecon`（`master`）。那里只有我们自己写的代码，没有上游模型和权重；从 render04 拷的 `~/sam3d` 是完整的，以它为准。
- **更细的文档**：流程和每道闸看什么见 `docs/WORKFLOW.md` 和 `.claude/skills/run-recon-pipeline/SKILL.md`；装机细节和坑见 `docs/setup/SETUP.md`；已知问题见 `data/outdoor/ISSUES.md`。
- **已知的局限**：物体朝向可能差 180°；两个人先后站到同一位置会被并成一个人；被跟踪成"人"的非人物体（比如椅背上的外套）自动跑时不会去掉。
