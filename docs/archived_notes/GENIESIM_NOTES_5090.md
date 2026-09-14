# geniesim_world (Genie Sim PanoRecon) 在 RTX 5090 上的笔记

单张 ERP 全景 → 3DGS。官方 README 标注在 RTX 5090 / CUDA 12.8 / torch 2.8.0 测过，照 README 装即可，**仓库代码零改动**。

## 用法

```bash
conda activate geniesim_world
export OPENCV_IO_ENABLE_OPENEXR=1
cd ~/sam3d/genie_sim/_work
geniesim_world create -p test.png -o .                      # 默认
geniesim_world create -p test.png -o . --pano-depth-engine fuse
# 超分还需要：
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export GENIESIM_REALESRGAN_BIN=~/sam3d/genie_sim/source/external/realesrgan-ncnn-vulkan-20220424-ubuntu/realesrgan-ncnn-vulkan
geniesim_world create -p test.png -o . --super-sample realesrgan-x4plus
```

产物：`<stem>/merged_gaussians.ply`（成品，约 360 MB / 670 万高斯），`cubes/`（6 面图+深度+内外参），`sharp/`（每面 ply）。

## 相对 README 的环境差异（都不是代码改动）

| 问题 | 处理 |
|---|---|
| requirements 没钉 opencv，pip 装到 5.0.0，**5.0 没编 EXR 写入器**（`could not find a writer`） | `pip install opencv-python-headless==4.11.0.86` |
| OpenCV 4.x 默认禁用 EXR | `export OPENCV_IO_ENABLE_OPENEXR=1`（README troubleshooting 第 4 条） |
| DA360 权重官方走 Google Drive（gdown） | 用 HF 镜像 `hljiang/DA360` 的 `DA360_large.pth`（Apache-2.0）放到 `external/DA360/` |
| Real-ESRGAN ncnn 二进制缺 `libvulkan.so.1`，系统没有、无 sudo | `conda install -c conda-forge libvulkan-loader`（包名不是 vulkan-loader）+ `LD_LIBRARY_PATH` |

SHARP 权重首次运行自动下载到 `~/.cache/torch/hub/checkpoints/`。

## 实测（test.png，4608×2304）

| 变体 | 耗时 | 与默认 PSNR | 近处锐度 | 远处锐度 |
|---|---|---|---|---|
| 默认（DA360 深度） | 2m40s（含首次下载） | — | 3.72 | 2.48 |
| `--pano-depth-engine fuse` | 1m26s | 25.2 dB | 3.72 | 2.49 |
| `--super-sample realesrgan-x4plus` | 3m04s | 31.8 dB | 3.97 | 2.61 |
| fuse + 超分 | 5m14s | 26.5 dB | 3.97 | 2.61 |

- fuse 改的是几何（PSNR 25 dB = 像素位置明显变了），锐度不变；超分只让锐度 +5~7%。
- DA360 全景深度 p01/p50/p99 = 1.00/1.50/12.6，max 24.2 —— 对比把 ERP 直接喂 DA3 只有 1.0–2.1，全景专用深度才对。

## 能力边界

- **清晰度天花板**：SHARP `output_resolution = internal_resolution // 2`，每面高斯网格 768×768×2（=1,179,648，与日志一致）。等效一张 3072 宽全景；输入再清晰也被这里截断。`--cube-face-size` 必须等于 SHARP 内部尺寸 1536，不能调大。
- **没有轨迹输入、没有扩散补全**（源码 trajectory/navmesh/diffusion/inpaint 均 0 命中）。纯前馈，从全景中心看不到的地方不会被补，相机离原点越远越拉伸。
- 同组能"按轨迹生成全景视频 + 补遮挡"的是 Genie Sim PanoWorld（arXiv 2607.26646），genie_sim 仓库里**没有**它的代码。

## 渲染脚本

`_work/render_compare.py`（gsplat）。坑：OpenCV 相机约定 x 右 / y 下 / z 前，c2w 旋转列应为 `[-right, -true_up, fwd]`（写成 `[right, true_up, fwd]` 画面会转 180°）。
