# 人体动作导出(骨骼动画 FBX / glb)

**正式的路(2026-10-08 起): 用模型的一手数据, 不从网格反推。** 设计和验证见 `docs/superpowers/specs/2026-10-08-people-firsthand-motion-design.md`。

```
出网格(存 mhr_4d_individual)  ->  pipeline/body_motion.py(在参数上修)  ->  from_motion.py(参数 -> glb)  ->  to_blender.py(FBX + .blend)
```

**平时不用手敲**: 流水线的 `motion` 步(验收那一段, 合成后面)调 `pipeline/motion_export.py --r1 <R1> --body <人体目录>`, 把这三段串起来,
产物在 `<R1>/12_motion/`。单独重出一轮: `outdoor_pipeline.py --scene <场景> --run <轮次> --only motion`(室内用 indoor_pipeline.py)。

| 脚本 | 环境 | 做什么 |
|---|---|---|
| ../../pipeline/body_motion.py `--r1 <R1> --mesh-run <出网格目录> --out <目录>` | sam3d, GPU | 修复做在关节数据上 -> motion.npz(世界系参数) + people_world.npz(网格, 给验收视频) |
| from_motion.py `<motion.npz> <输出目录> <名字>` | sam3d, GPU | 每人一个带骨骼的 glb + 原始参数 npz。几秒钟, 没有拟合 |
| to_blender.py `<glb,glb,...> <场景glb> <目标网格npz> <输出目录> <名字> [fps]` | ~/venvs/bpy | `_scene_people.fbx`(场景 + 人 + 相机) + `_scene.fbx`(只有场景) + 每人一个 FBX + 一份给人看的 .blend + 预览图; 自检不过就失败 |
| render_video.py `<.blend> <输出.mp4> [--jobs 4]` | sam3d(它自己起 bpy 进程) | 用 .blend 里的相机, 从原视频的视角把"场景 + 人"渲成视频, 和原视频一帧对一帧。`motion_export.py --video` 会调它 |
| ../../pipeline/cam_track.py | 两边都能 import | 相机的换算: 流水线的相机(OpenCV 约定) -> Blender 相机的位姿 / 焦距 |
| tidy_bones.py | ~/venvs/bpy | 骨头怎么显示: 去掉 glTF 导入套的小圆球; `view_bones()` 出给人看的样子(to_blender 都调了) |
| compare_video.py `<老.mp4> <新.mp4> <输出.mp4>` | sam3d | 原片 / 老 / 新 三栏对比视频 |

**相机(2026-10-09 起)**: `_scene_people.fbx` 和 .blend 里有一个叫 `camera` 的相机, 每个场景帧一个关键帧(位置、朝向、焦距),
帧号就是原视频帧号。Blender 里按小键盘 0 = 原视频的视角。原始数据在 `<名字>_motion_params.npz` 的 `cam_*` 里。
- Blender / FBX 的相机只有一个焦距; 流水线的横、竖焦距各估各的(差 0.4%~0.8%), 取了平均 —— 画面最边上和流水线自己的投影差 1~3 个像素。
- 画面比例跟内参那张图(688:384), 不是原视频的 16:9: office 是 1920x1072, nfl1 / soccer 是 1280x714。
- 渲视频用 EEVEE。它用哪张卡不归 CUDA_VISIBLE_DEVICES 管(这台机器上总是 3 号卡)。实测: office 432 帧 118 秒, nfl1 114 帧 11 秒, soccer 205 帧 14 秒。

**骨头长度(2026-10-09 起)**: 每人全程一个值(`body_motion.py` 的 `fix_bone_lengths`), FBX 里骨头不再每帧伸缩。
网页上的验收视频走的是网格那条路, 那上面没有锁。

FBX 和 .blend 动作一样, 差在骨头怎么画:
- FBX: 关节朝向是 MHR 原样的。Blender 沿关节的 Y 轴画骨头, MHR 的 Y 轴垂直于肢体, 所以每个关节是一根横着的短棍;
  四肢上每段还有 4~5 个辅助关节(`*_twist*_proc`, 全身 38 个), 排成一排像梯子。数据没错, 只是不好看。
- `<名字>_scene_people.blend`(只出这一份): 给人看的。导入时让 Blender 把骨头转到朝着子关节(Bone Dir = Fortune, 动作它自己换算),
  再藏掉辅助关节和钉在原点的 `body_world`, 骨头拉长到下一个关节, 显示成四棱锥。**关节的本地坐标轴和 MHR 不一样了, 数据用 FBX。**

---

## 老的路: 从网格反推(留作对照, 不进正式流程)

把 body_into_scene 修正好的人(和验收视频里的一样; 当时还有一个顶点动画的 09_scene_4d.glb, 2026-10-08 起不出了)转成**带骨骼的动作文件**, Blender 能直接打开、能套到别的角色上。
09_scene_4d.glb 里的人是逐帧顶点动画(morph target), 没有骨骼; 这里用 MHR 模型本身拟合每一帧网格, 反推出 127 个关节的旋转。

四步, 各用各的环境:

| 步 | 脚本 | 环境 | 做什么 |
|---|---|---|---|
| 1 | dump_people.py `<R1> <out.npz> [人id,...]` | sam3d | 修正好的网格按 glTF 世界系(天 +y, 米)存下来 |
| 2 | fit_mhr.py `<people.npz> <拟合目录> [人id]` | body4d, GPU | 拟合: 体型/骨骼缩放每人一份, 姿势逐帧。一人 114 帧约 7 分钟, 按人分卡并行 |
| 3 | export_glb.py `<拟合目录> <out.glb> [人id,...]` | sam3d | 骨架 + 蒙皮网格 + 关节动画 -> glb |
| 4 | to_blender.py `<glb> <场景glb> <people.npz> <输出目录> <名字> <fps>` | ~/venvs/bpy (bpy 4.2) | glb -> FBX(全体 + 每人) + .blend(场景 + 人); 自检 + 预览图 |

要点(都验证过):
- MHR 的线性蒙皮 = 关节全局变换 × 自带逆绑定矩阵 × 静止网格, 和 glTF/FBX 的蒙皮逐项一致(差 0.0002 cm)。
- 拟合时关掉姿势修正项(pose correctives): glTF/FBX 放不了它, 这样拟合误差 = 播放误差。平均 0.4~0.9 cm/顶点。
- 时间戳按整数帧率打(21.07 fps -> 21): Blender 的 glTF 导入只认整数帧率, 带小数会把帧号放大上万倍、FBX 导出卡死。
- Blender 导入 FBX 默认把动画往后挪 1 帧(Animation Offset = 1), 要和视频帧号对齐就设成 0。
- 数据本身乱跳的人(nfl1 人 4, 深度离群、一帧跳 1 m)拟合不上, 不导。
