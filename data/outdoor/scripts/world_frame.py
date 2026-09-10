#!/usr/bin/env python
"""世界坐标约定的唯一一份实现: "天"在哪个方向, 以及导出到 glTF 时怎么转。

为什么要有这个文件 —— 这个约定以前散在十几个文件里, 每个消费者各抄一份, 而且已经开始漂移:

  * 兜底 `return np.array([0,-1,0.])` 抄了 6 份, 全部静默。于是"天没算出来"这件事
    永远不会有人知道。室内 cyberpunk 就这么用偏了 4.6° 的方向跑了很久。
  * 导出旋转抄了好几份, 2026-09-08 改成按实际天方向算之后只修了 render_back 和
    mask_compare 两处, topview.py 和 assemble_buildings.py 还写着老的 diag(1,-1,-1) ——
    俯视图里相机轨迹和物体差 29.2°, 楼和物体拼在一个 glb 里楼是歪的。
  * 另有 5 个在链上的脚本干脆写死 UP=[0,-1,0] 不读文件, 而依赖表里却声明它们依赖 up。

约定本身(由 world_up.py 产生, 存在 02_da3/world_up.json):
  up        DA3 世界系里指向"天"的单位向量。取高度一律用 P @ UP, 值越大越高。
  ground_h  地面沿这个方向的高度(中位数)。

导出约定:
  assemble_scene.py 导出前把 up 用罗德里格斯转到 glTF 的 +Y。要把结果投影回原相机,
  必须乘这个旋转的**转置**。两边都从这里拿, 不许各算各的。
"""
import os, json
import numpy as np

FALLBACK = np.array([0., -1., 0.])      # 老假设: 首帧相机的 -y
GLTF_UP  = np.array([0., 1., 0.])       # glTF 里的天

# 缺 world_up.json 时的行为。默认硬停 —— 静默猜一个方向是这条链上代价最大的一类失败:
# 它不报错、不崩、渲染回图也看不出来, 只是所有高度和位置都沿着歪掉的方向量出来。
# 确实想在没有 world_up.json 的情况下跑(比如临时看一眼中间产物), 显式设
# ALLOW_UP_FALLBACK=1, 那时会打一行醒目的警告。
def _allow_fallback():
    """每次调用时读 —— import 时定死的话, 进程内改 os.environ 就无效了。"""
    return os.environ.get("ALLOW_UP_FALLBACK", "0") == "1"


def up_path(da3):
    return os.path.join(da3, "world_up.json")


def load_up(da3, who=None):
    """读"天"。返回 (UP 单位向量, ground_h 或 None, 是否走了兜底)。

    who 只用来把报错说清楚是谁在读, 不影响行为。
    """
    p = up_path(da3)
    tag = f"[{who}] " if who else ""
    if not os.path.exists(p):
        if not _allow_fallback():
            raise SystemExit(
                f"{tag}[失败] 没有 {p}\n"
                f"       这一步要沿'天'的方向量高度, 而'天'还没算出来。\n"
                f"       先跑 up 那一步。真要拿老假设 [0,-1,0] 凑合跑, 显式设 ALLOW_UP_FALLBACK=1。\n"
                f"       (为什么不默认凑合: 猜错了不报错也不崩, 只是所有高度和位置一起歪掉 ——\n"
                f"        室内就这么用偏 4.6° 的方向跑了很久没人发现。)")
        print(f"{tag}★★ 没有 world_up.json, 退回硬编码的 -y。所有高度都可能是歪的。", flush=True)
        return FALLBACK.copy(), None, True
    d = json.load(open(p))
    u = np.asarray(d["up"], float)
    n = np.linalg.norm(u)
    if not np.isfinite(n) or n < 1e-8:
        raise SystemExit(f"{tag}[失败] {p} 里的 up 不是个合法方向: {d.get('up')!r}")
    gh = d.get("ground_h")
    return u / n, (float(gh) if gh is not None else None), False


def export_R(UP):
    """导出用的旋转矩阵: 把 UP 转到 glTF 的 +Y。assemble_scene 用它, 其它地方用它的转置。"""
    u = np.asarray(UP, float); u = u / np.linalg.norm(u)
    v = np.cross(u, GLTF_UP); c = float(u @ GLTF_UP)
    # ★ 判据看 1+c 不看 |v|。用 |v|<1e-8 会有一段坑: UP 落在离 -Y 约 1e-6°~6e-4° 的窄带里,
    #   |v| 已经不小于 1e-8 走进通用分支, 而 1/(1+c) 分母是 0 或接近 0 ——
    #   要么 ZeroDivisionError, 要么悄悄返回一个行列式不为 1 的非正交矩阵(实测 det=0.64)。
    #   1+c 才是那个分母, 判它才对得上。
    if 1.0 + c < 1e-12:                                # 正好反向
        return np.diag([1., -1., -1.])
    if 1.0 - c < 1e-12:                                # 已经就是 +Y
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


def export_T(UP):
    """export_R 的 4x4 版, 给 trimesh 的 apply_transform 用。"""
    T = np.eye(4); T[:3, :3] = export_R(UP); return T


def inverse_T(UP):
    """把已经导出的场景转回 DA3 世界系。渲染回图、俯视图、任何"拿 glb 去比原图"的地方用它。
    ★ 就是 export_R 的转置。分开写死过一次, 代价是四张验收图全废还被当依据看了好几天。"""
    T = np.eye(4); T[:3, :3] = export_R(UP).T; return T


def deviation_deg(UP):
    """这个天离老假设差多少度。只用来打印, 不参与计算。
    ★ 不取绝对值 —— world_up.py:62 那个用了 abs, 两边会给出 26.2 vs 153.8 这种一对数。
      翻转本身就该被看见, 所以这里不 abs。"""
    u = np.asarray(UP, float); u = u / np.linalg.norm(u)
    return float(np.degrees(np.arccos(np.clip(u @ FALLBACK, -1, 1))))


if __name__ == "__main__":
    da3 = os.environ.get("DA3") or f"{os.environ['R1']}/02_da3"
    UP, gh, fb = load_up(da3, "自检")
    print(f"UP = {np.round(UP, 5).tolist()}   ground_h = {gh}   兜底 = {fb}")
    print(f"偏离老假设 {deviation_deg(UP):.2f}°")
    R = export_R(UP)
    assert np.allclose(R @ UP, GLTF_UP, atol=1e-9), "export_R 没把 UP 转到 +Y"
    assert np.allclose(inverse_T(UP)[:3, :3] @ R, np.eye(3), atol=1e-9), "inverse_T 不是 export_R 的逆"
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-9), "export_R 不是纯旋转"
    print("[自检通过] export_R 把天转到 +Y, inverse_T 是它的逆, 行列式为 1")
