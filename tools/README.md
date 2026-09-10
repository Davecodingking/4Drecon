# tools/ —— 看结果的工具

所有"看结果"的脚本都集中在这一处。**常用、以后要改,所以不许散在三个目录里。**

分两类,都在这里:

- **流程会自动调的**(人工闸的图、链上的一览图)—— 引用方写死 `TOOLS` 路径指过来
- **手动排查用的** —— 你自己敲命令跑

`pipeline/` 那边是流水线本体(调度 + 算法步骤),改了会影响产出;这里的改了只影响你看到的图。

叫 `tools` 不叫 `utils`:`utils` 一般指被 import 的库代码。

★ 2026-09-09 把散在 `.claude/skills/run-recon-pipeline/` 和 `data/outdoor/scripts/`
的 8 个脚本全搬进来了,**不留软链接** —— 软链接会让 `__file__` 解析到别处,那几个
import 算法模块的脚本会找不到路。引用方改代码指过来,一份实现。

## 流程自动调的

| 脚本 | 谁调 | 出什么 |
|---|---|---|
| `review_gate.py` | `drive.py` 闸 2 / 闸 4 | 联络图 `05_review_sheet.jpg` + 决策文件。**人工干预的主入口** |
| `reuse_table.py` | `drive.py` 闸 2 / 闸 4 | 复用表 `05_reuse_table.jpg` —— 一行 = 一个会生成的 mesh,左边代表右边成员 |
| `viz_ground.py` | `drive.py` 闸 1 | 地基核对图:抽帧一览 + 点云俯视(带相机轨迹) + 侧视(带拟合的地面线) |
| `viz_recall_2d.py` | `drive.py` 闸 4 | 补漏核对图:黄框=补漏检到的,彩色=SAM3 已有的,右上角标 IoU |
| `render_meshes.py` | pipeline 的 `meshes` 步 | mesh 一览图 + 网格健康度 |
| `topview.py` | pipeline 的 `merge` 步 | 俯视图:相机轨迹 + 楼 + 物体 |
| `pcd_layers.py` | `indoor_pipeline.py:408`(`zup` 步) | 点云按高度分层的俯视图(桌面层 / 椅背层 / 墙层),看点云成不成团 |
| `box_health.py` | `indoor_pipeline.py:454`(`boxfix` 步);`run_indoor_objects.sh`(已弃用) | 每个盒子的点密度和长轴铺满度,判真/存疑/空。**闸口要看的那张体检表** |

★ `render_meshes.py` 和 `topview.py` **在步骤表里注册着**(`meshes` / `merge` 两步),
所以它们既是"可视化"也是"链上的一步"。放这里是因为它们只出图不影响下游产物 ——
删了链不断,只是少一张图。

## 手动排查用的

| 脚本 | 看什么 |
|---|---|
| `viz_word_candidates.py` | 某个词的全部候选 + 最终选中,轮廓画在原帧上、局部放大。**判断"抠的是不是这个东西"用这个** |
| `viz_recall_candidates.py` | 召回实例的抠图,绿框=被派去生成 mesh 的代表 |
| `viz_masks_frame.py` | 某一帧上所有 SAM3 跟到的 mask,按词着色 |
| `viz_recall_frame.py` | 三栏:原图 \| SAM3 的 mask \| 加上召回的 3D 盒子投影 |
| `viz_word_boxes.py` | 同一个词,PCA 盒子 vs 召回盒子,并排画在帧上(框上标高度) |
| `viz_box_compare.py` | 同一个物体的几种盒子(PCA / WildDet3D geometric / WildDet3D 文本模式)画在一起 |
| `viz_boxes.py` | PCA 的盒子和 WildDet3D 的盒子直接画出来(画盒子本身,不是摆好的场景) |
| `mask_compare.py` | 两版场景的轮廓叠图,逐像素比 |
| `viz_boxes_used.py` | **组装真正用的那批盒子**(默认 `06_all_boxes_merged.json`):上排投到 4 帧原图,下排俯视图 |
| `viz_mesh_gallery.py` | 某个 tag 下所有 glb 排成一张图,格子标题=复用单位,用来查同类有没有复用同一个 mesh。★ 和 `render_meshes.py` 功能重叠,是旧版遗留;链上跑的是 `render_meshes.py`,这个只手动用 |

## 写新工具要注意

**算法模块住在 `data/outdoor/scripts/`**(`world_frame.py` 定天方向和导出旋转、
`word_rank.py` 词优先级表和母词回退、`review_decisions.py` 人工决策)。
这里的脚本要用它们,`sys.path` 得这么拼:

```python
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "data", "outdoor", "scripts"))
```

用 `realpath` 不用 `abspath` —— 万一哪天又有人建软链接,`abspath` 会往上数到错的目录去。

**画到图上的文字一律 ASCII。** cv2 的默认字体画不出中文,会变成一排问号而且不报错。
`review_gate.py` 的 `_SRC`、`reuse_table.py` 的 `_TY` 两张映射表负责中文→英文,
加新标签必须同步加进去 —— 这个坑踩过三次。
