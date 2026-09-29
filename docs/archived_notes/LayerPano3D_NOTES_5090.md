# LayerPano3D 在 RTX 5090 (sm_120) 上的配置笔记

环境：conda `layerpano3d`（python 3.9.23，torch 2.8.0+cu128）。
权重在 nvme `/mnt/nvme/jincheng/layerpano3d_ckpt`，仓库 `checkpoints/` 是指向它的软链（原占位目录挪到 `checkpoints_repo_placeholder/`）。FLUX.1-Fill-dev / LLaVA / OneFormer 走 HF 缓存（也在 nvme）。

## 权重

| 用途 | 位置 | 来源 |
|---|---|---|
| 分层补全 | HF 缓存 | `black-forest-labs/FLUX.1-Fill-dev`（gated，非商用；只下 diffusers 分片 ~34GB，跳过重复的单文件版） |
| 全景 LoRA | `checkpoints/pano_lora_720*1440_v1.safetensors` | `ysmikey/Layerpano3D-FLUX-Panorama-LoRA` |
| LaMa | `checkpoints/ControlNetLama.pth` | `lllyasviel/Annotators` |
| 深度 | `checkpoints/depth_anything_v2_vitl.pth` | `depth-anything/Depth-Anything-V2-Large` |
| 分割 | `checkpoints/sam_vit_h_4b8939.pth` | dl.fbaipublicfiles.com |
| 深度补全 | `checkpoints/Infusion/` | `Johanan0528/Infusion` |
| 超分 | `checkpoints/pasd/stable-diffusion-v1-5/`（只 diffusers 分目录）+ `checkpoints/pasd/checkpoint-100000/` | SD1.5 + aliyuncs `pasd.zip`（5.45GB） |
| 打标 / 语义分割 | HF 缓存 | `llava-hf/llava-1.5-7b-hf`、`shi-labs/oneformer_coco_swin_large` |

不下：FLUX.1-dev（只用于文字生成全景，我们跳过第一步）；PASD 的 lcm-lora / personalized（默认不启用）。

## 相对 README 的环境差异

- torch 2.8.0+cu128 代替 README 的 2.4.0+cu118。
- requirements 过滤后安装（`-c` 锁住 torch）：跳过 natten（只有 cu118 轮子）、mmcv、detectron2、hydra、albumentations、shapely、submitit、diffdist、inflect、wldhx、icecream、ultralytics —— grep 确认流程不 import；补上代码实际用到但没列的 omegaconf / accelerate / timm。
- `setuptools<81`（保留 pkg_resources）。CUDA 扩展用 `pip install --no-build-isolation .`（非可编辑）：setuptools 80 的 `-e` 会转调 PEP517 隔离构建，隔离环境里没有 torch。

## 源码 / 配置改动（都留了 `.orig`）

1. `submodules/simple-knn/simple_knn.cu`：加 `#include <cfloat>`（FLT_MAX 未定义）。
2. `360monodepth/code/cpp/CMakeConfig.txt`：numpy include 路径（README 要求改的就是这行）。
3. `360monodepth/code/cpp/python/setup.py`：Linux 分支写死的 OpenCV include/lib → conda 环境（OpenCV 5 头文件在 `include/opencv5/`）。
4. `360monodepth/code/cpp/3rd_party/pybind11/build/mock_install` → 软链到 `$CONDA_PREFIX`（CMakeLists 硬编码此路径，不改 CMakeLists）。
5. 作者旧编译产物 `cpp/python/build`、`dist` 挪成 `*_author_prebuilt`（依赖 OpenCV 4.2，conda-forge 只有 py38 版，py39 用不了）。
6. `gen_layerdata.py` 的 `generate_caption()` 加 2 行：打标前 `model.to(0)`，打完 `model.to("cpu"); torch.cuda.empty_cache()`。原代码 LLaVA 常驻 GPU（~14G），FLUX Fill 主干上 GPU 时（~24G）合计超 32G。实测峰值 27.25G。结果不变，只是每层多一次模型搬运。
7. `gen_layerdata.py` 补完后加 2 行"贴回"：用平滑掩码当透明度，洞外换回本层输入。原代码 FLUX Fill 输出直接保存，整张全景每层都被重画一遍且逐层累积（实测洞外 PSNR：layer2 33.0 dB → layer1 27.3 dB），而 `gen_traindata.rgb_alignment` 洞外取的正是这些内层图，最终场景并不是原图像素。
8. `gen_layerdata.py` 启用作者注释掉的 `prompt = "Empty Scene, Nothing."`（覆盖 LLaVA 描述）。原提示词只描述房间，FLUX 常在挖掉的物体处又画出新物体（沙发→沙发、岛台→岛台、餐椅→餐椅），走动时会出现重影。第一版结果备份在 `_logs/run1_layering/`，对比脚本 `_logs/compare_A.py`。

**7、8 的实测结果（09-11，seed 16806，strength 0.6）**：
- 贴回有效：按实际喂给 FLUX 的掩码（layer2 `mask_smooth`，内层 `mask_smooth_new`）计算，三层洞外 100% 像素与原图相同、最大差 0。副作用：部分洞边缘有一块亮度略高的方形接缝（FLUX 洞内颜色与原图有偏差，平滑掩码羽化不够）。
- 空场景提示词无效：沙发、岛台、吊灯前绿植、餐桌+餐椅处仍画出新家具；窗边扶手椅处第一版干净，改后反而多出一个怪凳子。靠墙的盆栽、扶手椅等约一半的物体能补出干净背景。补出新物体更可能来自 LaMa 模糊底图 + strength 0.6 让 FLUX 按上下文"补全成物体"，不是提示词问题。
- 改回原提示词、`--strength 0.35`：更差。仍画出新物体，且 LaMa 的模糊透出来成大片糊斑（餐桌处一整团灰），原本干净的扶手椅处出现断腿怪椅。三版里内容最好的是 0.6 + LLaVA 原提示词，因此定版 = 0.6 + 原提示词 + 洞外贴回（第 7 条）。0.35 版备份在 `_logs/run3_layering/`，对比图 `_logs/cmp_R3_layer*.png`。

9. `pasd/run_layers_pasd.py` 末尾 `main(args)` → `main(args, enable_xformers_memory_efficient_attention=False)`。该开关是函数默认参数写死为 True、没有命令行入口；环境不装 xformers（SpatialGen 那边 xformers 在 sm_120 会误选 FA3 内核），改用 PyTorch 自带注意力。
10. `pasd/pasd/myutils/vaehook.py` 第 366 行 `if is_xformers_available:` → `if is_xformers_available():`。原代码漏了括号，函数对象恒为真，Tiled VAE 编码时不管装没装都走 xformers 分支，报 `NameError: name 'xformers' is not defined`（第 9 条关掉的只是 UNet 的开关，管不到这里）。修好后自动落到 `F.scaled_dot_product_attention` 分支。

**第 3 步定版实测（0.6 + 原提示词 + 贴回，09-11 06:14，对比图 `_logs/cmp_F_layer*.png`，脚本 `_logs/compare_F.py`）**：
- layer2 与第一版补的内容几乎一样（输入同为原图、种子相同，只差贴回）。
- 贴回的额外好处：第一版整图重画时把扶手椅旁的原茶几抹掉了，定版保住了。
- layer0（最里层）因输入变了，补的内容和第一版不同：餐桌+餐椅区第一版画怪长桌，定版画了一张米色大沙发；从原点被餐桌挡住，走动时可能露出。候选修法：只重跑 layer0、换种子。
- 亮度接缝仍在（墙面最明显）。候选修法：贴回前按洞周一圈的原图颜色对洞内做均值/方差匹配，或泊松融合。

## 360monodepth 编译

conda-forge：`cmake "ceres-solver=2.2.0=*cpu*" glog gflags eigen suitesparse libopencv libboost-devel gtest pybind11`
（不指定 cpu 会解析到 GPU 版 Ceres，要求 CUDA ≥ 12.9，系统是 12.8。）实际版本：Ceres 2.2.0 cpu、Eigen 5.0.1、OpenCV 5.0.0、glog 0.7.1 —— 新大版本对这份代码没问题。

```bash
cd submodules/360monodepth/code/cpp && mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python) \
  -DEIGEN3_INCLUDE_DIRS=$CONDA_PREFIX/include/eigen3 \
  -DGLOG_LIBRARIES=$CONDA_PREFIX/lib/libglog.so \
  "-DCMAKE_CXX_FLAGS=-DGLOG_USE_GLOG_EXPORT -DGLOG_USE_GFLAGS"
cmake --build . --target depth_stitch EigenSolvers -j16
cd ../python && pip install --no-build-isolation .
```

- `EIGEN3_INCLUDE_DIRS`：Eigen 5 不再导出这个老变量，`EigenSolvers` 目标靠它拿头文件。
- glog 两处：glog 0.7 要求通过 `glog::glog` 目标拿宏定义，CMakeLists 把那行注释掉了，改用空的 `${GLOG_LIBRARIES}`。
- 产物 `cpp/lib/{libdepth_stitch.so, EigenSolvers*.so}`；`utility/__init__.py` 会把 `cpp/lib` 加进 sys.path，`blending.py` 在顶层 `from EigenSolvers import LinearSolver`，所以必须编。

## 已验证

- diff-gaussian-rasterization、simple-knn：GPU 前向+反向通过。
- import 链全部通过：`depthmapAlign.init('group')`、`utils.depth_alignment`、FluxFillPipeline（仓库自带 `src/pipeline_flux_fill.py`）、LaMa、SAM、DepthAnythingV2、OneFormer/LLaVA、gaussian_renderer/scene。
- **模型加载和实际运行尚未测试。**

## 已知风险

- `gen_layerdata.py` 把 LLaVA-7B fp16（约 14GB）在 import 时就 `.to(0)` 常驻 GPU，同时 FLUX Fill 用 `enable_model_cpu_offload()`，transformer（约 24GB bf16）上 GPU 时合计可能超过 32GB。

## 高分辨率前景方案（未验证）

`outputs/rgb.png` = 原图缩到 2048×1024（作者测试尺寸），跑到 PASD 超分之后，把 `outputs/layering/rgb_sr.png` 换成 `_input/test_4608.png` 缩放到 `layer0/layer0_inpaint_sr.png` 的**实际尺寸**，再跑 `gen_traindata.py --sr`。依据：`gen_traindata.rgb_alignment` 以最里层尺寸为准缩放各层；`layer_rgb_raw` 取 `rgb_sr.png` 尺寸，深度和掩码都按它缩放，尺寸不一致会在生成点云时对不上。

## 第一版全流程结果（09-11 06:39，`_logs/v1_scene/`）

- 耗时：第 4 步超分 10 分钟，第 5 步几秒，第 6 步 5 分钟，第 7 步训练 4 层共 7 分钟，第 8 步渲染 2 分钟（第 3 步 5 分钟）。
- 产出：`gsplat_layer0~3.ply`（每层约 0.7GB，第 1 层约 287 万个高斯），`renders_zigzag.mp4`（1024²，600 帧）、`renders_0deg.mp4`（2048²，160 帧，34MB）。
- zigzag 轨迹：`rendering/render_video_zigzag.py` 写死三个方向的折线，平移乘 0.3。
- 效果：整体清晰、无拉丝，走到阳台门前、餐厅墙角都干净；问题 ① 餐桌后露出第 3 步 layer0 补出来的米色沙发（白色一团）；② 桌椅边缘发毛（层与层交界）；③ 天花板有一块浅色污迹。
- 修 ①：`_logs/reroll_layer0.py` 只重画 layer0（同掩码、同底图、同提示词，只换种子），挑干净的一版后 `_logs/run_v2.sh <种子>` 从第 4 步重跑。

## 能走多远 / 为什么像贴图（09-11 06:55 实测）

- 训练相机：`outputs/traindata/layer*/frames/transform_matrix_*.npy` 共 24 个，**位置全是 (0,0,0)**，只转方向。场景从头到尾只在全景中心被监督过。
- 尺度：空房间点云（2%–98%）约 294×141×291 单位，按层高约 2.6m 折算 1 单位≈1.8cm。原 zigzag（平移×0.3）最远只走约 15 单位≈30cm。
- 加大走动：`rendering/render_video_zigzag_far.py --scale s`（原脚本的复制，只把 0.3 改成参数、输出名加 `_s{s}`）。0.6（≈60cm）干净；1.0（≈1m）门框/墙开始弯、扶手椅腿糊，无破洞无拉丝；2.0（≈2m）崩：脚下地板拉丝、门和墙严重弯折、边柜和镜子融化。**可用半径约 1m。**
- 贴图感原因：每个物体只有从中心看到的那一面（单张深度估计出的薄壳），光照和反光烘焙在颜色里，侧着看就是贴在薄片上的照片。
- 餐桌处 layer0：7 个种子（16806/1/7/42/123/2024/31337）全部画出家具（沙发、条桌、长凳、白柜），盆栽→墙、扶手椅→地板这两处每个种子都干净 → 洞形状像物体时 FLUX 就补物体，换种子无效。改用 LaMa 在 1024×2048 直出（`_logs/layer0_lama_patch.py`）：一大片灰色糊斑并漫到墙和窗，更差。未解决。
