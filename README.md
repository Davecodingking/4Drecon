# 4Drecon

从一段实拍视频重建出一个可编辑的三维场景：每个物体是独立的模型，位置、朝向、尺寸按真实的米摆好；画面里的人带骨骼动画，放在同一个场景里。室内、室外都能跑。

最后交付的是 FBX：场景、场景 + 人（人带骨骼动画和原视频的相机）、每人一个单独的文件，另有一份 Blender 工程。

## 怎么做的

没有哪个模型能单独做完，每个只管它擅长的一环：

| 环 | 模型 | 做什么 |
|---|---|---|
| 1 | Qwen3-VL | 看几帧，列出画面里有什么 |
| 2 | SAM3 | 按词在整段视频里跟踪，逐帧抠出每个物体 |
| 3 | Depth Anything 3 / VGGT | 每帧深度和相机位姿，把抠出的物体反投影成三维盒子 |
| 4 | SAM 3D Objects | 一张抠图生成完整的模型，看不见的背面一起补 |
| 5 | Qwen3-VL | 回答"这类东西现实里一般多高"，用来定全局尺度 |
| 6 | 组装 + 物理 | 按盒子摆回去、落地、推开穿模 |
| 7 | SAM-Body4D（SAM 3D Body / MHR） | 人体网格和关节参数，修好后放进场景，导出骨骼动画 |

分工的原则：盒子是量出来的，管位置和朝向；模型是生成出来的，管形状；常识管绝对尺寸。

流水线分五段跑，每段结束停在一道人工闸，把该看的数字和图摆在网页上，人看过再继续。机器判不了的事（比如一张抠图抠得全不全、跟踪到的是不是真人）留给人在闸口决定。

完整的流程和每一环的短板见 [docs/WORKFLOW.md](docs/WORKFLOW.md)。

## 仓库里有什么

这个仓库只放我们自己写的代码和文档。上游模型的仓库、模型权重、数据和跑出来的结果都不在里面。

| 目录 | 内容 |
|---|---|
| `pipeline/` | 流水线本体：`outdoor_pipeline.py`、`indoor_pipeline.py`、`body4d_pipeline.py`，以及人体那几步（摆人、身份核对、动作导出） |
| `data/outdoor/scripts/` | 物体链每一步的算法脚本，室内室外共用 |
| `.claude/skills/run-recon-pipeline/` | 驱动 `drive.py`：分段跑、停在闸口 |
| `tools/` | 闸口网页（`gate_ui.py`）、各种核对图、运行监控；`tools/mocap/` 是动作导出，`tools/tests/` 是测试 |
| `patches/` | 我们对上游仓库的改动，存成补丁 |
| `docs/` | 流程说明、装机说明、设计文档 |

## 装环境

需要 NVIDIA 显卡（单卡 24 G 显存可以跑）和 CUDA 12.8。要建 8 个 conda 环境外加一个 Blender 的 Python 环境，并下载约 70 G 在用的模型权重。步骤见 [docs/setup/SETUP.md](docs/setup/SETUP.md)：正文是 7 个基础环境，第 12 节是之后新增的（FoundationPose、Blender、SAM 3.1、ffmpeg），两部分都要做。

上游仓库（SAM3、SAM 3D Objects、SAM-Body4D、Depth Anything 3 等）要自己 clone 到指定的 commit，再打上 `patches/` 里的补丁，说明见 [patches/README.md](patches/README.md)。

代码里的路径默认仓库放在 `~/sam3d`。

## 怎么跑

```bash
# 跑到下一道闸就停
python .claude/skills/run-recon-pipeline/drive.py --scene <场景> --run <轮次>

# 人看过了，继续
python .claude/skills/run-recon-pipeline/drive.py --scene <场景> --run <轮次> --go

# 现在停在哪
python .claude/skills/run-recon-pipeline/drive.py --scene <场景> --run <轮次> --status
```

室内场景加 `--indoor`。加 `-n` 只打印要执行的命令，不真跑。

在网页上看闸口：

```bash
python tools/gate_ui.py --scene <场景> --run <轮次> --port 8765      # 室内加 --indoor
```

室外的输入视频放在 `data/outdoor/dataset/<场景>/video.mp4`，结果在 `data/outdoor/result/<场景>/<轮次>/`；室内的结果在 `data/indoorOutput/<场景>/<轮次>/`。人体动作的交付物在结果目录的 `12_motion/` 下。

## 测试

```bash
bash tools/tests/run.sh
```

都是带 assert 的脚本，不用 pytest。其中几个要模型文件和一轮现成的结果在盘上。

## 文档

- [docs/WORKFLOW.md](docs/WORKFLOW.md)：五段五闸的完整流程
- [docs/setup/SETUP.md](docs/setup/SETUP.md)：换机器重装
- [tools/README.md](tools/README.md)：看结果的工具
- [tools/mocap/README.md](tools/mocap/README.md)：人体动作导出
- [docs/superpowers/specs/](docs/superpowers/specs/)：各项改动的设计和实测结果
- [ROADMAP.md](ROADMAP.md)：目标和路线图
- [data/outdoor/ISSUES.md](data/outdoor/ISSUES.md)：已知问题

## 已知的局限

- 物体的朝向只判得出轴，判不出正反，可能差 180°。
- 人体的身份核对靠位置：甲走了、乙站到同一个位置，会被并成一个人。
- 被跟踪成"人"的非人物体（比如搭在椅背上的外套）不会自动去掉，要在闸口上点掉。

## 许可

Apache License 2.0。上游模型各有自己的许可，使用前请分别确认。
