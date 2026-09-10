# GEN3C 在 RTX 5090 (32GB, sm_120) 的部署记录 — 2026-09-06

## 结论(2026-09-06 更新: 已跑通)
**能跑。** 之前判断"显存不够"是错的 —— 真因是官方流程"先把 7B 主干搬进显存, 再做用不到主干的 VAE 编码",
把编码提前即白省 14GB。结果与配置见 `~/sam3d/hyworld_test/r4_crop/gen3c_out/README.md`。

必须同时打开的三个开关:
```
GEN3C_PREENCODE=1        # VAE 编码提到加载主干之前 (省 14GB, 关键)
COSMOS_ATTN_BACKEND=torch # TE 无 sm_120 注意力内核, 切 PyTorch 原生
GEN3C_RENDER_CHUNK=0      # 不要开分块渲染, 会破坏缓冲 top-k 选择导致画面变黑
```
关键帧 8 张是显存上限(12/16 张 OOM)。

## 已完成（换卡即可复用）
- conda 环境 `cosmos-predict1`，torch **2.7.1+cu128**（原 yaml 装的 2.6.0+cu124 无 sm_120，必须升）
- 权重 113GB 在 `/mnt/nvme/jincheng/gen3c_ckpt`（`checkpoints` 是指向它的软链）
- 数据转换脚本 `~/sam3d/hyworld_test/make_gen3c_npz.py`：
  WorldMirror 输出（帧+深度+相机）→ GEN3C multiview npz
- 测试数据：`~/sam3d/hyworld_test/r2_outdoor/gen3c_mv8.npz`（8 关键帧 + 121 帧轨迹）

## 5090 适配补丁（4 处，已应用）
1. `torch` 升 2.7.1+cu128
2. `transformer-engine` 1.12.0 → **2.3.0**（1.12 的 .so 对不上新 torch ABI）
   需 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib`
3. TE 2.x 改名，改导入路径：
   - `apply_rotary_pos_emb` → `transformer_engine.pytorch.attention.rope`
   - `check_set_window_size` → `...attention.dot_product_attention.utils`
   - `_SplitAlongDim` 已删除 → 用 torch.split 写 shim
   - `_RMSNorm` 已删除 → 回退到公开 `RMSNorm`
4. `apex` / `amp_C` 缺失 → 写空壳（仅训练用到）
   文件：`~/sam3d/GEN3C/amp_C.py`、`~/sam3d/GEN3C/apex/`

## 省显存尝试（全部失败，记录以免重走）
| 尝试 | 结果 |
|---|---|
| `--num_video_frames` 降到 61 | 断言拒绝：必须 N*120+1，121 已是最小 |
| `--height/--width` 降到 480×848 | 张量维度对不上：模型只接受 704×1280 |
| 关键帧 16 → 8 | 只省 ~2.5GB，仍差 2.3GB |
| `render_cache` 按目标帧分块 | 能省 6GB, **但有副作用**: top-k 缓冲选择由全局变块内, 关键帧多时画面逐渐变黑发蓝。最终弃用 |
| VAE 按时间分块 | 失败：`pixel_chunk_duration=121` 写死，121 无法拆成 8k+1 的块 |
| VAE 挪 CPU | 失败：tokenizer 是 JIT 图，dtype 烘死；CPU 上 bf16 缺 avg_pool3d |
| encode 期间把 DiT 挪 CPU | 失败：TransformerEngine 模块搬回后 "no kernel image" |

## 有用的中间产物
`render_cache` 分块那处补丁是真优化（原实现一次渲 [1,121,8,3,704,1280] 再挑 top-2），
换大卡后也值得保留，可用环境变量 `GEN3C_RENDER_CHUNK=16` 开启。已回退，需要时从本文档重做。
