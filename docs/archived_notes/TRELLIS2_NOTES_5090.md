# TRELLIS.2 在 5090 (Blackwell) 上的部署笔记 (2026-09-02)

环境: conda `trellis2` (python 3.10), torch 2.8.0+cu128, CUDA_HOME=/usr/local/cuda-12.8, TORCH_CUDA_ARCH_LIST="12.0"

## 与官方 README 的差异 (全是踩过的坑)
1. **torch**: 官方 2.6.0+cu124 不支持 sm_120, 换 2.8.0+cu128 (对齐 sam3d 环境)。
2. **flash-attn 不装**: 用 `ATTN_BACKEND=xformers`; xformers 必须 `0.0.32.post2` (cu128 源), 更新版本与 torch 2.8 ABI 不符 (undefined symbol c10::SymInt)。
3. **FA3 补丁 (必须)**: xformers 只按"算力≥9.0"选 Hopper 专用 FA3 内核, 5090 算力 12.0 会 `CUDA error: invalid argument`。运行前:
   `import xformers.ops.fmha.dispatch as _xd; _xd._get_use_fa3 = lambda: False`
   (同 SpatialGen/NOTES_5090.md 的坑)
4. **transformers 钉 4.57.1**: v5 里 DINOv3 的 `.layer` 挪进 `encoder` 子模块, trellis2 的特征提取器直接取 `model.layer` 会崩。
5. **gated 模型**: `facebook/dinov3-vitl16-pretrain-lvd1689m` 需要 HF 申请+登录 (已用 DaveHJC 账号批了)。`briaai/RMBG-2.0` 也是 gated——不用申请: 参考图带 alpha 通道(且不全为255)就不走抠图, 用空壳类替换 `trellis2.pipelines.rembg` 里的类即可 (见 test_wall/run_wall_test.py)。
6. **跑法**: `PYTHONPATH=~/sam3d/trellis.2` 且 cwd 无所谓; 扩展 nvdiffrast/nvdiffrec(包名 nvdiffrec_render)/CuMesh/FlexGEMM(包名 flex_gemm)/o-voxel 均按 12.0 架构编译通过。
7. 显存: 贴图管线 low_vram 模式实测峰值 ~20GB 内 (与 app.py 6.9GB 共存 OK)。

## 冒烟测试 (mesh+参考图 -> PBR 贴图)
`test_wall/`: 3.0x2.84m 墙板 + UE 视频帧电视墙裁剪 -> `wall_textured.glb` (~30s 推理)。
结果: 布局结构还原对 (白板/木边/电视位置), 材质偏脏、水彩感, 电视糊成棕色块。结论见聊天记录/REVIEW_background.md。
