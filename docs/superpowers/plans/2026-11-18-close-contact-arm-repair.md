# 双人贴身手臂防穿模与去抽动 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 升级 `pipeline/pair_arm_repair.py` 与 `pipeline/body_motion.py` 调用接口，通过“遮挡手臂先去抖 + 全肢体光滑胶囊体双人联立防撞（躯干/双臂/自身胯部）+ 原视频前景掩码防飞出与贴身稳固”，消除 `demo1_multi_person` 中双人小臂互穿、单帧 38 cm 抽动、自穿模及飞出人影轮廓问题，同时保证单人及非碰撞场景逐位不变。

**Architecture:** 在 `pipeline/pair_arm_repair.py` 中分三阶段处理重叠双人帧：(1) 利用分割掩码与 3D 遮挡判据识别被对方挡住的手臂段，在不跨镜头切点/缺帧边界的前提下做锚定中值+高斯去抖；(2) 从每人静止 MHR 网格自动拟合躯干、骨盆大腿、上臂、小臂手掌的光滑胶囊体半径，在近身碰撞窗口内**同时联立优化双人手臂关节角**，联合约束 `手臂<->对方躯干`、`手臂<->对方手臂`、`手臂<->自身躯干胯部` 胶囊体与表面间隙、2D 前景并集掩码距离场（防飞出轮廓）、贴腰段局部零滑移及**合成总关节角/手腕 3D 加速度平滑**；(3) 在 `pipeline/body_motion.py` 将已有 `cams / vid2key / kept / mesh_run / cuts` 透传给修复器（全部参数默认 `None` 以兼容旧调用）。

**Tech Stack:** Python 3.11 (`~/miniforge3/envs/sam3d/bin/python`), PyTorch (CUDA/CPU 可微 MHR 前向与 `grid_sample`), NumPy, SciPy (`cKDTree`, `distance_transform_edt`, `median_filter`, `gaussian_filter1d`), OpenCV / Pillow, Trimesh.

**Spec:** `/home/ubuntu/sam3d/docs/superpowers/specs/2026-11-18-close-contact-arm-repair-design.md`

## Global Constraints

- 手不穿进对方躯干（允许 `<= 3 mm` 正常贴身接触，`CONTACT_TOL_M = 0.003`）；
- 两条胳膊之间互不穿插，手也不穿进自己的胯部或拧到自己背后；
- 调完的手臂留在原视频人影轮廓内，帧与帧之间动作平顺，不再来回抽动；
- 两个人的脚、腿、骨盆位置、身体朝向、相机一个数都不动；
- 单人视频，或虽然有多人但手臂没发生近身穿插的视频，保持逐位不变；
- 绝不允许按视频名字写 `if/else` 特判；
- 只在检测到近身穿插的时间窗口上运行，整条 `demo1_multi_person` 增加耗时控制在 `20 秒` 以内。

## Review Focus

1. **缺失相机或分割图文件（如单元测试或老数据 `cams=None` / `mesh_run=None`）**：不报错崩溃，自动跳过 2D 掩码项并退化为纯 3D 伙伴遮挡判据与 3D 胶囊体+表面联立防撞（在 Task 2 & Task 3 测试）。
2. **跨切镜（`cuts`）或中间缺帧（`np.diff(vf) != 1`）的双人片段**：遮挡去抖和时序速度/加速度差分绝不跨越镜头切点或帧号断点（在 Task 2 测试）。
3. **贴身但手臂在画面最前方清晰可见且快速挥动**：自身分割掩码覆盖率高（`>= 0.50`）的手臂帧绝不被遮挡去抖抹平（在 Task 2 测试）。
4. **三人及以上场景，其中两人紧贴拥抱、第三人在 1.5 米外路过**：第三人的关节参数逐位零改动（在 Task 3 测试）。
5. **两条小臂在同一高度狭缝正面对穿 + 靠近自身胯部**：双人联立求解必须让两条小臂上下/内外自然错开，既消除双臂互穿，又不把手顶进自身大腿胯部或顶出 2D 人影轮廓外（在 Task 3 测试）。

---

### Task 1: 光滑骨骼胶囊体几何与全肢体（躯干/双臂/自身）碰撞度量

**Files:**
- Modify: `/home/ubuntu/sam3d/pipeline/pair_arm_repair.py`
- Create: `/home/ubuntu/sam3d/tools/tests/test_pair_arm_repair.py`

**Interfaces:**
- Produces in `pair_arm_repair.py`:
  - `capsule_seg_dist_torch(a0: torch.Tensor, a1: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor) -> torch.Tensor` — 输入形状 `(..., 3)` 线段端点，返回处处可微的线段间最短欧氏距离 `(...)`（米）。
  - `fit_capsule_radii(rest_v: np.ndarray, skel_rest: np.ndarray, dev: str = "cpu") -> dict` — 从单人静止网格（米）和静止骨架（米）自动量出各胶囊体段（躯干脊柱链、左右大腿根、左右上臂、左右小臂、左右手掌）的真实皮肤包络半径（米）。
  - `actor_capsules_torch(S_m: torch.Tensor, radii: dict) -> dict` — 从可微骨架关节坐标 `S_m: (T, 127, 3)`（米）生成 `{"body": (T, N_b, 2, 3, r_b), "arm_r": (T, 3, 2, 3, r_ar), "arm_l": (T, 3, 2, 3, r_al)}`。
  - `measure_pair_and_self_arm_penetration(V_a: np.ndarray, S_a: np.ndarray, radii_a: dict, V_b: np.ndarray, S_b: np.ndarray, radii_b: dict, faces: np.ndarray, dev: str = "cpu") -> dict` — 返回单帧或多帧下的 `torso_pen_m`, `arm_arm_pen_m`, `self_pen_m`（米）。

- [ ] **Step 1: 在 `tools/tests/test_pair_arm_repair.py` 编写胶囊体距离与双臂/自身碰撞检测的必挂测试**

```python
#!/usr/bin/env python
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/sam3d/pipeline")
sys.path.insert(0, "/home/ubuntu/sam3d/data/outdoor/scripts")
import mhr_body as M
import pair_arm_repair as PAR

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SHAPE = np.zeros(M.NSHAPE)


def test_capsule_seg_dist_and_gradients():
    a0 = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    a1 = torch.tensor([[0.3, 0.0, 0.0]], requires_grad=True)
    # 十字交叉线段，垂直间距 0.04 m
    b0 = torch.tensor([[0.15, -0.2, 0.04]])
    b1 = torch.tensor([[0.15, 0.2, 0.04]])
    d = PAR.capsule_seg_dist_torch(a0, a1, b0, b1)
    assert torch.allclose(d, torch.tensor([0.04]), atol=1e-5)
    d.sum().backward()
    assert torch.isfinite(a0.grad).all() and a0.grad[0, 2] < 0.0


def test_detects_arm_to_arm_and_self_pelvis_penetration():
    P1 = np.zeros((1, 204))
    P2 = np.zeros((1, 204))
    # 让两人面对面站近，右小臂在胸前交叉相撞
    P2 = M.set_root(P2, np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, -1.0]]]), M.root_pos(P1) + np.array([[0.0, 0.0, 0.36]]))
    P1[:, PAR.ARM_COLS["r"][3:7]] = np.radians([10.0, -55.0, 20.0, 55.0])
    P2[:, PAR.ARM_COLS["r"][3:7]] = np.radians([10.0, -55.0, 20.0, 55.0])
    V1, S1 = M.forward(SHAPE, P1, True, DEV)
    V2, S2 = M.forward(SHAPE, P2, True, DEV)
    r1 = PAR.fit_capsule_radii(V1[0], S1[0, :, :3] / 100.0, DEV)
    r2 = PAR.fit_capsule_radii(V2[0], S2[0, :, :3] / 100.0, DEV)
    faces = M.model(DEV).character_torch.mesh.faces.cpu().numpy()
    m = PAR.measure_pair_and_self_arm_penetration(V1, S1[:, :, :3] / 100.0, r1, V2, S2[:, :, :3] / 100.0, r2, faces, DEV)
    assert m["arm_arm_pen_m"].max() > 0.01
```

- [ ] **Step 2: 运行测试确认因函数未定义而失败（Verify RED）**

Run: `~/miniforge3/envs/sam3d/bin/python /home/ubuntu/sam3d/tools/tests/test_pair_arm_repair.py`
Expected: FAIL (`AttributeError: module 'pair_arm_repair' has no attribute 'capsule_seg_dist_torch'`)

- [ ] **Step 3: 在 `pipeline/pair_arm_repair.py` 实现 `capsule_seg_dist_torch`、`fit_capsule_radii`、`actor_capsules_torch` 与 `measure_pair_and_self_arm_penetration`**

关键实现约束：
- `capsule_seg_dist_torch` 对平行退化情况用 `clamp` + `1e-8` 正则项、`sqrt(dist_sq + 1e-8)` 保证在距离为 `0`（两轴完全相交）时梯度依然有限不出现 `NaN`；
- 身体胶囊体包含：`root—c_spine1`、`c_spine1—c_spine3`、`l_clavicle—r_clavicle`、`root—l_upleg`、`root—r_upleg`、`l_upleg—l_lowleg`、`r_upleg—r_lowleg`（自身碰撞检查排除同侧肩关节相连的 `clavicle—uparm` 邻接段，只检查小臂/手腕对自身躯干、骨盆与双腿胶囊体）；
- 每侧手臂包含 3 段胶囊体：`uparm—lowarm`（大臂）、`lowarm—wrist`（小臂）、`wrist—middle3`（手掌）。

- [ ] **Step 4: 重新运行测试确认通过（Verify GREEN）**

Run: `~/miniforge3/envs/sam3d/bin/python /home/ubuntu/sam3d/tools/tests/test_pair_arm_repair.py`
Expected: PASS (exit code 0)

---

### Task 2: 遮挡手臂可见度去抖（`smooth_occluded_arms`）与 2D 前景人影距离场（`build_foreground_edt`）

**Files:**
- Modify: `/home/ubuntu/sam3d/pipeline/pair_arm_repair.py`
- Modify: `/home/ubuntu/sam3d/tools/tests/test_pair_arm_repair.py`

**Interfaces:**
- Consumes: Task 1 `actor_capsules_torch`, `fit_capsule_radii`
- Produces in `pair_arm_repair.py`:
  - `detect_arm_occlusion(pid: str, sd: str, V_self: np.ndarray, S_self: np.ndarray, partner_list: list, vf: np.ndarray, frames: np.ndarray | None = None, cams: dict | None = None, vid2key: dict | None = None, kept: list | None = None, mesh_run: str | None = None, dev: str = "cpu") -> np.ndarray` — 返回 `(T,)` 布尔数组 `occluded`（当手臂投影落在本人掩码比例 `< 0.35` 且落在伙伴掩码或伙伴 3D 遮挡体后方比例 `> 0.25` 时为 `True`；若无掩码文件则用相机视线或伙伴躯干 3D 前后遮挡判定）。
  - `smooth_occluded_arms(P: np.ndarray, occluded: np.ndarray, cols: list[int], vf: np.ndarray, fps: float, cuts: np.ndarray | None = None) -> tuple[np.ndarray, int]` — 只改 `occluded` 连续段内的 `cols` 列（以前后可见锚点帧做中值滤波 `0.32s` + 高斯平滑 `0.16s`，绝不跨 `np.diff(vf) != 1` 或 `cuts` 切点），可见帧逐位不变，返回 `(P_new, n_smoothed_frames)`。
  - `build_foreground_edt(frames_bi: list[int], cams: dict | None, vid2key: dict | None, kept: list | None, mesh_run: str | None, scale_hw: tuple[int, int] = (270, 480)) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]` — 对存在掩码的帧读取 `{mesh_run}/masks/{bi:08d}.png`，对前景并集 `mask > 0` 外扩 `1.5%` 容差后计算外部欧氏距离场（单位：归一化图像宽度比例），返回 `{vf_int: (edt_2d, w2c_4x4, K_norm_3x3)}`。

- [ ] **Step 1: 在 `tools/tests/test_pair_arm_repair.py` 添加遮挡去抖、可见快动作保留、跨切镜隔离与无掩码退化的必挂测试**

```python
def test_smooth_occluded_arms_removes_hidden_spikes_and_preserves_visible_and_cuts():
    T, fps = 30, 25.0
    vf = np.arange(T)
    P = np.zeros((T, 204))
    cols = PAR.ARM_COLS["r"]
    # 0..9 可见快速挥臂（正弦大动作），10..20 被遮挡且第 14, 15 帧有 33 度单帧乱跳，21..29 切镜后新镜头
    P[:10, cols[3]] = np.radians(40.0 * np.sin(np.linspace(0, np.pi, 10)))
    P[10:21, cols[3]] = np.radians(15.0)
    P[14, cols[3]] = np.radians(48.0)   # 33 deg 遮挡乱跳（低于旧版 45 deg 毛刺线）
    P[15, cols[3]] = np.radians(-18.0)
    P[21:, cols[3]] = np.radians(-50.0)  # 21 帧切镜突跳

    occ = np.zeros(T, bool)
    occ[11:20] = True
    cuts = np.array([20.5 / fps])  # 20 -> 21 帧之间有切镜

    P_out, n_sm = PAR.smooth_occluded_arms(P, occ, cols, vf, fps, cuts=cuts)
    assert n_sm == 9
    # 可见帧 0..10 与切镜后 21..29 逐位不变
    np.testing.assert_array_equal(P_out[:11], P[:11])
    np.testing.assert_array_equal(P_out[21:], P[21:])
    # 遮挡段内原本 66 度的帧间乱跳被压平到 8 度以内
    max_jump_deg = np.degrees(np.abs(np.diff(P_out[10:21, cols[3]]))).max()
    assert max_jump_deg < 8.0
```

- [ ] **Step 2: 运行测试确认失败（Verify RED）**

Run: `~/miniforge3/envs/sam3d/bin/python /home/ubuntu/sam3d/tools/tests/test_pair_arm_repair.py`
Expected: FAIL (`AttributeError: module 'pair_arm_repair' has no attribute 'smooth_occluded_arms'`)

- [ ] **Step 3: 在 `pipeline/pair_arm_repair.py` 实现 `detect_arm_occlusion`、`smooth_occluded_arms` 与 `build_foreground_edt`**

关键实现约束：
- `smooth_occluded_arms` 按连续同镜头片段（`np.diff(vf) == 1` 且未跨 `cuts`）切分，每个 `occ` 子区间仅取同段内的前后 `1` 个可见锚点帧参与 `median_filter` + `gaussian_filter1d`，写回时只覆盖 `occ == True` 的帧；
- `build_foreground_edt` 将掩码下采样到 `(270, 480)` 加速计算，`cams is None` 或 `mesh_run is None` 或掩码文件缺失时直接返回空字典 `{}`，绝不抛异常。

- [ ] **Step 4: 运行测试确认通过（Verify GREEN）**

Run: `~/miniforge3/envs/sam3d/bin/python /home/ubuntu/sam3d/tools/tests/test_pair_arm_repair.py`
Expected: PASS

---

### Task 3: 双人全肢体联立轨迹优化（`repair_pair_arm_penetration`）与 `body_motion.py` 接入

**Files:**
- Modify: `/home/ubuntu/sam3d/pipeline/pair_arm_repair.py`
- Modify: `/home/ubuntu/sam3d/pipeline/body_motion.py:1087-1095`
- Modify: `/home/ubuntu/sam3d/tools/tests/test_pair_arm_repair.py`
- Modify: `/home/ubuntu/sam3d/tools/tests/run.sh`

**Interfaces:**
- Consumes: Task 1 & Task 2 全部函数
- Produces in `pair_arm_repair.py`:
  - `repair_pair_arm_penetration(res: dict, fps: float, dev: str = "cuda", cams: dict | None = None, vid2key: dict | None = None, kept: list | None = None, mesh_run: str | None = None, cuts: np.ndarray | None = None) -> dict`
- Modifies in `body_motion.py`:
  - 在 `solve()` 末尾调用 `repair_pair_arm_penetration(out, fps, dev=dev, cams=cams, vid2key=vid2key, kept=kept, mesh_run=mesh_run, cuts=_cuts)`。

- [ ] **Step 1: 在 `tools/tests/test_pair_arm_repair.py` 编写双人贴身双臂互穿 + 自穿模 + 2D 轮廓越界 + 远距第三人零改动的必挂集成测试**

```python
def test_joint_repair_resolves_torso_and_arm_arm_without_self_pen_or_jitter():
    T, fps = 16, 25.0
    P1 = np.zeros((T, 204))
    P2 = np.zeros((T, 204))
    P3 = np.zeros((T, 204))
    # P1 与 P2 紧贴面对面，双右臂同时穿入对方腰肋且两条右小臂在中间交叉互穿；并在第 8 帧注入 30 度遮挡跳变
    P2 = M.set_root(P2, np.repeat(np.array([[[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]]), T, 0),
                    M.root_pos(P1) + np.array([0.04, 0.0, 0.34]))
    P3 = M.set_root(P3, None, M.root_pos(P1) + np.array([1.8, 0.0, 0.0]))  # 第三人在 1.8m 外
    P1[:, PAR.ARM_COLS["r"][3:7]] = np.radians([12.0, -48.0, 28.0, 62.0])
    P2[:, PAR.ARM_COLS["r"][3:7]] = np.radians([12.0, -48.0, 28.0, 62.0])
    P1[8, PAR.ARM_COLS["r"][4]] += np.radians(-30.0)  # 单帧乱跳

    P1_orig_legs = P1[:, :30].copy()
    P3_orig = P3.copy()
    res = {
        "1": dict(vf=np.arange(T), frames=np.arange(T), params=P1.copy(), shape=SHAPE.copy()),
        "2": dict(vf=np.arange(T), frames=np.arange(T), params=P2.copy(), shape=SHAPE.copy()),
        "3": dict(vf=np.arange(T), frames=np.arange(T), params=P3.copy(), shape=SHAPE.copy()),
    }
    rep = PAR.repair_pair_arm_penetration(res, fps, dev=DEV)
    assert rep, "应当检测到双人穿模并执行联合修复"
    # 1) 第三人逐位不变，P1/P2 根与下肢逐位不变
    np.testing.assert_array_equal(res["3"]["params"], P3_orig)
    np.testing.assert_array_equal(res["1"]["params"][:, :30], P1_orig_legs)

    # 2) 验证躯干穿模、双臂胶囊体互穿、自身胯部穿模全部消除或压到 <= 1.0 cm
    V1, S1 = M.forward(SHAPE, res["1"]["params"], True, DEV)
    V2, S2 = M.forward(SHAPE, res["2"]["params"], True, DEV)
    r1 = PAR.fit_capsule_radii(V1[0], S1[0, :, :3] / 100.0, DEV)
    r2 = PAR.fit_capsule_radii(V2[0], S2[0, :, :3] / 100.0, DEV)
    faces = M.model(DEV).character_torch.mesh.faces.cpu().numpy()
    m12 = PAR.measure_pair_and_self_arm_penetration(V1, S1[:, :, :3] / 100.0, r1, V2, S2[:, :, :3] / 100.0, r2, faces, DEV)
    m21 = PAR.measure_pair_and_self_arm_penetration(V2, S2[:, :, :3] / 100.0, r2, V1, S1[:, :, :3] / 100.0, r1, faces, DEV)
    assert m12["torso_pen_m"].max() <= 0.015 and m21["torso_pen_m"].max() <= 0.015
    assert m12["arm_arm_pen_m"].max() <= 0.010
    assert m12["self_pen_m"].max() <= 0.008 and m21["self_pen_m"].max() <= 0.008

    # 3) 验证手腕单帧最大跳变被显著抑制（< 12 cm/帧）
    w1 = S1[:, 42, :3] / 100.0
    assert np.linalg.norm(np.diff(w1, axis=0), axis=1).max() < 0.12
```

- [ ] **Step 2: 运行测试确认失败（Verify RED）**

Run: `~/miniforge3/envs/sam3d/bin/python /home/ubuntu/sam3d/tools/tests/test_pair_arm_repair.py`
Expected: FAIL（旧版 `repair_pair_arm_penetration` 不防 `arm_arm_pen_m`、不防 `self_pen_m`、也不压制 `P1[8]` 的原始跳变）

- [ ] **Step 3: 实现新版联立 `repair_pair_arm_penetration`、更新 `body_motion.py` 调用并注册到 `tools/tests/run.sh`**

优化目标各项精确定义（在同一近身时间窗 `[lo, hi)` 上同时优化所有触发碰撞的 `(pid, sd)` 轨迹参数 `d_theta`）：
1. **先跑 `smooth_occluded_arms`**：仅对窗口内与碰撞伙伴发生互遮挡的手臂去抖，更新窗口基准角 `theta0`；
2. **光滑胶囊体 + 躯干表面防穿项 `loss_col`**：
   - `Arm <-> Partner Body Capsules` + 稠密躯干外表面有向距离（`CONTACT_TOL_M = 0.003` 米内零惩罚）；
   - `Arm <-> Partner Arm Capsules`（异体双臂胶囊体间距 `< r_a + r_b` 时平方+障碍惩罚，促使双臂自然错开）；
   - `Forearm/Hand <-> Self Body/Pelvis/Thigh Capsules`（防挤入自身胯部或反折穿背）；
3. **2D 前景人影框死项 `loss_sil`**（当 `edt` 存在时）：将手臂采样点投影至归一化图像坐标，用 `F.grid_sample` 读取 `edt`，落在前景人影并集外部的距离平方乘高权重（`400.0`）拉回人影内；
4. **贴身接触零滑移 + 姿态正则 + 总轨迹平滑**：
   - 当手掌中心与伙伴腰背距离连续 `>= 4` 帧 `< 0.045 m` 时，约束手掌相对伙伴最近腰部骨骼局部坐标系的帧间切向滑移（`loss_slip`）；
   - 先验项约束 `d_theta` 幅度（锁骨权重 `5.0`，大臂 `1.0`，肘 `0.8`，腕 `1.4`）及窗口两端余弦渐隐；
   - **总平滑项 `loss_total_sm`**：直接惩罚合成总角度 `theta = theta0 + d_theta` 的一阶角速度与二阶角加速度，以及 3D 手腕/手肘世界系二阶加速度！
5. **移除旧版会导致相邻帧单帧跳变的逐帧离散 `alpha in (0.75, 0.5, 0.25, 0)` 硬切换**，改为整窗平滑余弦过渡 + 若残余极小躯干擦边则平滑微调，保证时间连续性。
6. 在 `/home/ubuntu/sam3d/tools/tests/run.sh` 的 `test_unified_rules.py` 后追加 `$PY test_pair_arm_repair.py`。

- [ ] **Step 4: 运行新测试与 `test_unified_rules.py` 确认全部通过（Verify GREEN）**

Run: `~/miniforge3/envs/sam3d/bin/python /home/ubuntu/sam3d/tools/tests/test_pair_arm_repair.py && ~/miniforge3/envs/sam3d/bin/python /home/ubuntu/sam3d/tools/tests/test_unified_rules.py`
Expected: PASS

---

### Task 4: 全量回归测试与 `demo1_multi_person` 真机重跑及同屏对比视频验收

**Files:**
- Verify on `render04`: `/home/ubuntu/sam3d/data/outdoor/result/demo1_multi_person/r1`
- Deliver locally: `C:\Users\13670\.gemini\antigravity\brain\7c845d58-6215-455c-a0b6-d99952d1e3ef\demo1_arm_repair_compare.mp4` 及关键帧截图

**Interfaces:**
- Consumes: Task 3 已接入 `body_motion.py` 的新版 `repair_pair_arm_penetration`
- Produces: 全套自动化测试通过日志、`demo1_multi_person` 六维量化前后对比表、同屏对比验收视频与 Blender 文件更新。

- [ ] **Step 1: 运行项目完整测试套件**

Run: `bash /home/ubuntu/sam3d/tools/tests/run.sh`
Expected: 输出 `== tools/tests 全过 ==`，退出码 `0`。

- [ ] **Step 2: 在 `render04` 空闲 GPU 上重跑 `demo1_multi_person` 通用主线（`people,people_glb,motion,render`）及检查 `irish_tapdance_outdoor` 零退化**

Run: `CUDA_VISIBLE_DEVICES=4 ~/miniforge3/envs/sam3d/bin/python /home/ubuntu/sam3d/pipeline/outdoor_pipeline.py --scene demo1_multi_person --run r1 --only people,people_glb,motion,render`
Expected: 退出码 `0`，`pair_arm_repair` 耗时 `< 20s`。

- [ ] **Step 3: 运行六维量化审计脚本并生成三栏上下同屏对比视频（旧基线 vs 旧防穿模规则 vs 新三合一方案）**

核验指标：
1. `P1/P2 手臂 -> 对方躯干` 穿入帧数与最大深度；
2. `P1 手臂 <-> P2 手臂` 互穿帧数与最大深度；
3. `手臂 -> 自身胯部/躯干` 自穿模深度；
4. `P1/P2 右手腕` 70–185 帧单帧最大跳变（`cm/帧`）与 95 分位二阶抖动（Jerk）；
5. 手臂顶点投回原片超出 `Mask_P1 ∪ Mask_P2` 轮廓的比例与像素距离；
6. 导出对比视频并逐帧抽查 `97–108`、`130–153`、`164–175` 帧确认小臂不再互相抽插跳动（若发现极长遮挡段手势仍不自然，按设计文档启动备选方案 B `DuoMo/FlowHMR` 对比）。

