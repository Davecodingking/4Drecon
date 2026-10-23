# FoundationPose 接入前的坑(2026-10-23 调研, 还没装)

用途: 只让 FP 判断物体前后(我们摆好的朝向 vs 转 180°), 位置和大小仍用我们量盒子的结果。
许可证只许研究和评估用(NVIDIA Source Code License), 进产品要另想办法(比如 Orient Anything V2, CC-BY-4.0)。

## 论文里测出来的

| 坑 | 出处 | 我们怎么办 |
|---|---|---|
| 初始大小、中心偏多了, FP 收敛到错的姿态或失败 | SimuScene (arXiv 2606.03994) 附录 A.2 | 用我们量盒子的大小和位置当初值 |
| 对称的桌子在两个朝向之间来回跳 | SimuScene 附录 A.2: FP 之后给桌子另加一步, 带"别离初值太远"的约束 | 桌子类单独看, 必要时不让 FP 改 |
| 估出来的深度(不是传感器)下, 渲染比对类方法会变差: SAM3D+渲染优化 1.31 -> 1.92, Any6D 1.04 -> 2.87(倒角距离 x100) | ScenePlacer (arXiv 2610.15734) 表 1 | 我们是 DA3 深度; 只让 FP 定前后, 大小和位置不交给它 |
| 生成网格 + FP 的整套组合(SceneComplete)比 SAM 3D 自己摆还差 | RecGen (arXiv 2604.27106) 表 1 | 那套用的是 InstantMesh, 不全是 FP 的问题, 当提醒 |
| SimFoundry 只在"精确 CAD + 物体没被挡"时把 FP 结果当真值 | SimFoundry (arXiv 2606.28276) 附录 L.1 | 遮挡重的物体别太信它 |

## 代码里看出来的(NVlabs/FoundationPose 的 estimater.py)

1. `register()` 不吃我们给的朝向: 固定自己生成约 240 个候选(40 个视角 x 6 个转角, 再聚类),
   位置用"抠图外框中心 + 深度中位数"猜, 遮挡时外框中心会偏。
   做法: 不走 `register()`, 直接调 `refiner.predict` / `scorer.predict`, 候选只给两个(原朝向、转 180°)。
2. 网格和深度都必须是米(官方 issue #44 第一条)。我们的深度要先乘尺度倍数 k。
3. 它会先腐蚀 + 双边滤波深度, 是按传感器深度设计的, 用在 DA3 深度上没人测过。
4. 没人测过的: 训练数据多是桌面大小的物体; 办公桌、门这种大件, 和被画面边缘截断的物体, 效果不确定。
5. 背面贴图: SAM 3D 只用一帧生成, 那一帧没看到的面颜色是猜的。FP 要用生成时那一帧, 别用拍到背面的帧。

## 部署要点

- 本机 sam3d 环境已有 torch 2.8+cu128、pytorch3d 0.7.8、nvdiffrast 0.4.0、open3d、pyrender: 复制成 `fpose` 环境(约 12 GB),
  补 kornia 等小包; conda-forge 装 eigen、boost-cpp、cmake、cxx-compiler, 跑 `build_all_conda.sh` 编 mycpp。
  numpy 保持 1.26(官方 requirements 写 numpy>=2, 升了可能弄坏现有编译扩展)。
- 权重(Google Drive, 本机能连): `weights/2023-10-28-18-33-37/model_best.pth` 修正网络 68 MB,
  `weights/2024-01-11-20-02-45/model_best.pth` 打分网络 190 MB。
- 不用 Docker: 官方镜像是旧 CUDA, 和我们的 conda 流程不一致。
