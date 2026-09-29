# 换机器备份清单（2026-09-21，ES-MAL-GPU-011 回收前）

范围（用户拍板）：代码全带；实验结果只带**整理后的室外 / 室内流水线**的产物；Gen（SpatialGen）和 LM（SpatialLM）的实验数据、环境、权重都不要。
环境和权重不拷，照 `SETUP.md` 在新机器上重装。

**合计约 17 GB（压缩后 14 GB，四个包）。** 逐文件清单由脚本生成，打包就照它来。

---

## 带走的

### A. 代码与文档（3.3 GB：代码 2.0 GB + Claude 会话 1.3 GB）

| 内容 | 大小 | 说明 |
|---|---|---|
| `~/sam3d` 下所有代码目录，**整目录带走** | 2.0 GB | 包括我们自己写的 pipeline/tools/docs，和改过的各上游仓库（sam-3d-objects、DA3、sam3、WildDet3D…）连同它们的 `.git` 历史；只去掉权重（checkpoints/weights/ckpt）和实验输出 |
| `~/sam3d/patches/` | 70 KB | 各上游仓库改动的补丁，只当"我们改了什么"的说明；整目录带走后不需要打补丁 |
| `~/.claude`（记忆 + 会话记录） | 0.33 GB | 不含 plugins、cache |
| `/mnt/data/jincheng/claude-session-archive` | 0.96 GB | 会话存档 |
| `~/paper`、`.bashrc`、`.profile` | 29 MB | |

整目录带走的好处：到新机器直接能用，不用重新 clone、不用打补丁；想看哪些是我们改的，进仓库 `git diff` 即可。

### B. 流水线实验结果（9.6 GB）

| 内容 | 大小 | 文件数 |
|---|---|---|
| 室外 `data/outdoor/result`：beach r1/r2、carpark r1、pedestrian r1/r2/r2sim/r2sim_vggt/r_vggt_g2/r_vggt_l、realstreet r1、timesquare r0/r1 | 1.92 GB | 2242 |
| 室外原始素材 `data/outdoor/dataset`：beach、carpark、mayday、pedestrian、realstreet、timesquare、warpengine | 1.11 GB | 1387 |
| 室外成品归档 `data/outdoorOutput` | 1.05 GB | 86 |
| 室外脚本和说明 `data/outdoor/scripts`、README、ISSUES | 1 MB | 62 |
| 室内 `indoorOutput/office`：r1 + 三套帧 | 0.70 GB | |
| 室内 `indoorOutput/cyberpunk`：o1、r_vggt_da3/g2/l + 帧 | 0.61 GB | |
| 室内 `indoorOutput/ithor312`：r1、r_vggt_da3、analysis_9.16、analysis_9.17 + 帧 | 0.39 GB | |
| 室内原始素材 `indoor/dataset`：cyberpunk、ithor312、office、clips（21 段候选视频） | 0.43 GB | 73 |
| 物体网格 `.glb`（两条流水线生成的 589 个） | 1.12 GB | 589 |
| 生成用的输入抠图 `my_images`（同一批物体） | 1.78 GB | 1366 |
| 行人 4D 人体 `sam-body4d/outputs`（pedestrian、mayday） | 0.46 GB | 1873 |
| 各轮深度图 `02_da3/depth` 等 20 个目录（单独成包 `depth.tar.zst`，09-21 补打） | 3.94 GB | 4760 |

**深度图原先没带，09-21 补上了**：闸口网页（`gate_ui.py`）和盒子查看页（`box_fit_view.py`）都要读它；而且尺度标定会就地改写它，
重跑要把 DA3 之后好几步都跑一遍，不是"几分钟"。已用还原出来的 ithor312/r1 实跑过 `box_fit_view.py`，能出页面。

**每一轮里仍然不带**（待定，见下方"待确认"）：
- `_` 开头的目录，如 `_prescale`、`_da3_orig`（合计 3.3 GB）：尺度标定 / 换 VGGT 位姿之前的原始数据。**重新标定尺度时要从 `_prescale` 还原**（`world_scale.py` 写明"只有一条路"）
- `07_scene.glb`、`17_scene_wd.glb`（合计 1.1 GB）：物理落稳前的版本，**是物理那一步的输入**；只想重跑物理时要它，否则得从组装那步重来

---

## 不带的

| 内容 | 大小 | 原因 |
|---|---|---|
| 物体的 `.ply` 高斯点云 | 17.1 GB | 流水线只读 .glb，没有代码读 .ply |
| `SpatialGen/out`、`SpatialGen-Testset` | 5.1 GB | Gen，不要了 |
| `data/wall_Inpaint`、`wall_gen` 实验输出 | 1.1 GB | 墙面贴图 / 生成墙体，建在 LM 布局 + Gen 上 |
| `data/Hybrid`、`data/ueindoor_out`、`data/spatialLM`、ueindoor 原始视频、Hybrid 物体（hyb_*） | 0.7 GB | LM 路线 |
| `indoorOutput/ithor312/r2` | 0.36 GB | LM 链跑的一轮 |
| `indoorOutput/ithor312/r3_spatiallmwall`（及 ith_r3 物体） | 0.9 GB | 房间墙用的是 SpatialLM 出的墙 |
| `ith_r2` 物体、`outputs_bak_0917_solo`、warehouse、08-26/27 早期试跑 | 1.4 GB | 旧备份 / 早期测试 |
| Fire3D 结果、`data/fire3d_examples` | 0.5 GB | 不在流水线里 |
| Pano2Room 输出、vggt/DA3 早期测试输出、RealEstate10K 数据 | 0.7 GB | 已放弃路线 / 早期测试 |
| conda 环境 | ~123 GB | 照 SETUP.md 重装 |
| 模型权重 | ~250 GB | 照 SETUP.md / download_weights.sh 重下 |
| 各种缓存（.cache、pip/conda 包缓存、torch_cache） | ~30 GB | 没用 |

---

## 待确认

1. FIRE3D 的环境（16 GB）和权重（63 GB）：不在流水线里，默认不装。
2. Gen / LM 相关的代码目录（SpatialGen、SpatialLM 整目录约 135 MB，另有墙面贴图脚本）：默认照样带走。
3. `_prescale` 等原始数据（3.3 GB）和物理前的场景 glb（1.1 GB）要不要带：带上才能在旧轮次上单独重跑尺度标定 / 物理。
4. `indoorOutput/office/r1` 打包后又有改动，回收前打补丁包带上。

## 打包时要注意

- 结果目录里有约 800 个软链接（轮次目录链到 dataset 的帧、outdoorOutput/timesquare 链到 result），全部指向本来就要带的文件；但它们写的是绝对路径 `/home/jincheng/sam3d/...`，新机器路径不同就得改成相对链接，或者还原到同样的路径。
- 3.9 万个小文件先打成 tar 再传。
