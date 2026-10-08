#!/usr/bin/env python
"""MHR 人体模型的薄封装: SAM 3D Body 的一手参数 <-> 网格 / 骨架, 相机系 <-> 世界系。

SAM 3D Body 看一张图, 先算出一组数, 再喂给 MHR 生成网格。那组数才是一手资料(2026-10-08 起存在
<出网格目录>/mhr_4d_individual/<人>/<帧>.npz, 见 sam-body4d 的 save_mesh_results):
  shape_params (45)        体型, 同一个人各帧一样
  mhr_model_params (204)   [0:3] 根平移  [3:6] 根旋转  [6:136] 各关节(含手)  [136:204] 骨骼缩放
  pred_cam_t (3)           人在相机前面哪儿(OpenCV 相机系, 米)
  网格(相机系, 米) = F · MHR(shape, params) / 100 + pred_cam_t,   F = diag(1,-1,-1)

MHR 的根(都用数值实验核过, 见 tools/tests/test_mhr_body.py):
  - 根旋转是 xyz 外旋欧拉角(= scipy 'xyz' = roma "xyz"), 绕**根关节**转, 根关节固定在 (0, 92.4, 0) cm, 不随体型变;
  - 根平移 1 个单位 = 10 cm, 沿世界轴, 在旋转之后。
  所以  MHR 顶点(cm) = R_根 · (u - c) + c + 10 · 根平移,   u = 只摆了关节、根不动时的身体, c = 根关节。

世界系的表示: 直接用 MHR 自己的根平移/根旋转放世界里的位置和朝向 —— 修好的参数喂给 MHR 就是世界系的网格和骨架,
不需要相机, 也不需要从网格反推。
"""
import os
import numpy as np

MHR_PT = os.path.expanduser("~/sam3d/sam-body4d/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
ROOT_C = np.array([0.0, 92.4, 0.0])        # 根关节, cm
T_UNIT = 10.0                              # 根平移 1 个单位 = 10 cm
F = np.diag([1.0, -1.0, -1.0])             # MHR 输出 -> OpenCV 相机系(y、z 取反)
NPOSE, NSCALE, NSHAPE, NEXPR = 136, 68, 45, 72
_M = {}


def model(dev="cpu"):
    import torch
    if dev not in _M:
        _M[dev] = torch.jit.load(MHR_PT, map_location=dev).eval()
    return _M[dev]


def forward(shape, params, correctives=True, dev="cpu", batch=64):
    """shape (45,), params (T,204) -> 顶点 (T,N,3) 米, 骨架状态 (T,127,8) [平移 cm, 四元数 xyzw, 缩放]。
    correctives: 姿势修正项(关节附近的细微形变)。验收视频里开着; glTF/FBX 放不了它, 导动作文件时关掉。"""
    import torch
    m = model(dev)
    P = torch.as_tensor(np.asarray(params, np.float32), device=dev)
    s = torch.as_tensor(np.asarray(shape, np.float32), device=dev)
    V, S = [], []
    with torch.no_grad():
        for i in range(0, len(P), batch):
            p = P[i:i + batch]
            v, k = m(s.expand(len(p), -1), p, torch.zeros(len(p), NEXPR, device=dev), correctives)
            V.append((v / 100.0).cpu().numpy()); S.append(k.cpu().numpy())
    return np.concatenate(V), np.concatenate(S)


def rest_verts(shape, dev="cpu"):
    """这个人的静止网格(模板 + 体型), cm —— 蒙皮的绑定姿势。"""
    import torch
    bs = model(dev).character_torch.blend_shape
    return (bs.base_shape + torch.einsum("k,kvd->vd", torch.as_tensor(np.asarray(shape, np.float32), device=dev),
                                         bs.shape_vectors)).cpu().numpy()


def skeleton(dev="cpu"):
    """骨架和蒙皮(和人无关): 关节名、父子、逆绑定(127,8)、蒙皮(顶点号, 关节号, 权重)。"""
    ch = model(dev).character_torch
    lbs = ch.linear_blend_skinning
    return dict(names=np.array(list(ch.skeleton.joint_names)), parents=ch.skeleton.joint_parents.cpu().numpy(),
                inv_bind=lbs.inverse_bind_pose.cpu().numpy(), sk_v=lbs.vert_indices_flattened.cpu().numpy(),
                sk_j=lbs.skin_indices_flattened.cpu().numpy(), sk_w=lbs.skin_weights_flattened.cpu().numpy())


def limb_speed(sp_v, dev="cpu"):
    """每个姿势参数"带动的那一段肢体"里最快的顶点有多快。sp_v (T,N) 各顶点的速度 -> (T,136)。
    一个参数带动的肢体 = 它驱动的关节(查 MHR 的参数变换表, 一个参数可以带 1~20 个关节)及其下游关节上的顶点
    (顶点归蒙皮权重最大的那个关节)。肩膀的参数管整条手臂: 手在快速挥, 肩、肘、腕都算"快"。
    ★ 躯干(根 + 脊柱)只看躯干自己那块, 不把挂在上面的手臂/腿/脖子并上来: 手挥得快是肩和肘转出来的,
      脊柱这时转得很慢 —— 把它也当成"快"的话, 一挥手躯干的抖动就回来了(2026-10-08 nfl1 姿势抖动多了 10%)。"""
    k = ("drive", dev)
    if k not in _M:
        ch = model(dev).character_torch
        sk = skeleton(dev); nj = len(sk["parents"])
        dom, bw = np.zeros(int(sk["sk_v"].max()) + 1, int), np.zeros(int(sk["sk_v"].max()) + 1)
        for v, j, w in zip(sk["sk_v"], sk["sk_j"], sk["sk_w"]):
            if w > bw[v]:
                bw[v], dom[v] = w, j
        PT = ch.parameter_transform.parameter_transform.cpu().numpy()[:, :NPOSE]
        trunk = np.array([n in ("body_world", "root") or n.startswith("c_spine") for n in map(str, sk["names"])])
        _M[k] = (dom, sk["parents"].astype(int), [np.unique(np.nonzero(PT[:, c])[0] // 7) for c in range(NPOSE)], nj, trunk)
    dom, parents, pj, nj, trunk = _M[k]
    sub = np.zeros((len(sp_v), nj))
    np.maximum.at(sub, (slice(None), dom), sp_v)                # 每个关节自己那块顶点里最快的
    for j in range(nj - 1, 0, -1):                              # 父关节的编号都比子关节小: 从后往前把下游并上来
        if trunk[parents[j]] and not trunk[j]:
            continue                                            # 四肢/脖子的根: 不往躯干上并
        sub[:, parents[j]] = np.maximum(sub[:, parents[j]], sub[:, j])
    return np.stack([sub[:, j].max(1) if len(j) else np.zeros(len(sp_v)) for j in pj], 1)


def limb_groups(dev="cpu"):
    """姿势参数按肢体分组 {肢体: 参数下标(0~135)}: 躯干 / 脖子和头 / 左右手臂(到手腕) / 左右腿。
    只收管旋转的参数; 手指、管骨头长度的那 6 个(*_length / *_width_flexible)不在里面。
    肢体 = 这个参数驱动的关节往上找, 第一个挂在躯干(根 + 脊柱)上的关节; 用它的名字当肢体名。"""
    k = ("groups", dev)
    if k not in _M:
        ch = model(dev).character_torch
        sk = skeleton(dev); names = [str(x) for x in sk["names"]]; par = sk["parents"].astype(int)
        PT = ch.parameter_transform.parameter_transform.cpu().numpy()[:, :NPOSE]
        trunk = [n in ("body_world", "root") or n.startswith("c_spine") for n in names]
        finger = [any(f in n for f in ("thumb", "index", "middle", "ring", "pinky")) for n in names]

        def limb(j):
            while not trunk[j] and not trunk[par[j]]:
                j = par[j]
            return "trunk" if trunk[j] else names[j]
        g = {}
        for c in range(6, NPOSE):
            rows = np.nonzero(PT[:, c])[0]
            js = np.unique(rows // 7)
            if not len(rows) or (rows % 7 < 3).all() or any(finger[j] for j in js):
                continue
            ls = {limb(j) for j in js}
            if len(ls) == 1:
                g.setdefault(ls.pop(), []).append(c)
        _M[k] = {n: np.array(v) for n, v in g.items()}
    return _M[k]


def length_params(dev="cpu"):
    """管骨头长度/宽度的那几个姿势参数的下标(0~135): spine/neck/arm/leg_length、shoulder/hip_width(都带 _flexible)。
    它们只驱动关节的平移。模型逐帧估, 同一个人每帧都不一样(nfl1 人 6 的前臂差到近 10 cm)。"""
    k = ("lengths", dev)
    if k not in _M:
        PT = model(dev).character_torch.parameter_transform.parameter_transform.cpu().numpy()[:, :NPOSE]
        _M[k] = np.array([c for c in range(6, NPOSE) if len(np.nonzero(PT[:, c])[0]) and (np.nonzero(PT[:, c])[0] % 7 < 3).all()])
    return _M[k]


def root_rot(params):
    """(T,204) -> 根旋转矩阵 (T,3,3)"""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("xyz", np.asarray(params)[..., 3:6].reshape(-1, 3)).as_matrix().reshape(
        np.shape(params)[:-1] + (3, 3))


def root_pos(params):
    """(T,204) -> 根关节的位置 (T,3), 米"""
    return (ROOT_C + T_UNIT * np.asarray(params)[..., :3]) / 100.0


def _continuous_euler(e):
    """(T,3) xyz 欧拉角按时间接成连续的。同一个朝向有两套写法: (x, y, z) 和 (x+π, π-y, z+π), 各自还能加 2π 的整数倍;
    scipy 总给 y 在 ±90° 以内的那套, 人转过 90° 时 x、z 会一起跳 180°(只做 2π 展开接不上)。
    逐帧在所有等价写法里挑离上一帧最近的。"""
    e = np.array(e, np.float64, copy=True)
    for i in range(1, len(e)):
        best = None
        for c in (e[i], np.array([e[i, 0] + np.pi, np.pi - e[i, 1], e[i, 2] + np.pi])):
            c = c - 2 * np.pi * np.round((c - e[i - 1]) / (2 * np.pi))
            d = np.abs(c - e[i - 1]).sum()
            if best is None or d < best[0]:
                best = (d, c)
        e[i] = best[1]
    return e


def set_root(params, Q=None, p=None):
    """换掉根的朝向 Q (T,3,3) 和/或 根关节位置 p (T,3) 米。返回新的参数, 关节和骨骼缩放不动。
    欧拉角按时间接成连续的(不影响网格, 但谁要是直接对这组参数做插值/平滑, 跳变会被当成猛转)。"""
    from scipy.spatial.transform import Rotation
    out = np.array(params, np.float64, copy=True)
    if Q is not None:
        e = Rotation.from_matrix(np.asarray(Q).reshape(-1, 3, 3)).as_euler("xyz").reshape(out[..., 3:6].shape)
        out[..., 3:6] = _continuous_euler(e) if e.ndim == 2 else e
    if p is not None:
        out[..., :3] = (100.0 * np.asarray(p) - ROOT_C) / T_UNIT
    return out


def cam_verts(shape, params, cam_t, correctives=True, dev="cpu"):
    """存盘时的相机系网格(OpenCV, 米): 用来核"存下来的数重新生成的网格 == 存的 ply"。"""
    v, _ = forward(shape, params, correctives, dev)
    return v @ F.T + np.asarray(cam_t)[:, None, :]


def to_world(params, cam_t, c2w):
    """相机系的一手参数 -> 世界系的参数。params (T,204) 根平移为 0(模型给的就是 0); cam_t (T,3) 人在相机系的位置
    (已经乘过远近比例); c2w (T,4,4)。
      世界顶点 = R (F·MHR/100 + cam_t) + t = [R F R_根](u - c)/100 + R (F c/100 + cam_t) + t
      -> 世界里的根朝向 = R F R_根, 根关节位置 = R (F c/100 + cam_t) + t"""
    params = np.asarray(params, np.float64); c2w = np.asarray(c2w, np.float64)
    assert np.abs(params[..., :3]).max() < 1e-6, "相机系参数的根平移应该是 0"
    R, t = c2w[:, :3, :3], c2w[:, :3, 3]
    Q = R @ F @ root_rot(params)
    p = np.einsum("tij,tj->ti", R, (F @ ROOT_C) / 100.0 + np.asarray(cam_t)) + t
    return set_root(params, Q, p)


def transform(params, T4):
    """世界系参数整体再做一次刚体变换 T4 (4,4), 比如 DA3 世界 -> glTF 世界(天 +y)。"""
    T4 = np.asarray(T4, np.float64)
    return set_root(params, T4[:3, :3] @ root_rot(params), root_pos(params) @ T4[:3, :3].T + T4[:3, 3])
