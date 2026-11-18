#!/usr/bin/env python
"""从多帧里静止的竖杆(路灯/矮柱/栏杆柱...)解真正的竖直方向。只用相机朝向和内参, 不用深度。

为什么要有它 (2026-11-05, lalaland):
  world_up.py 拿地面法线当"天"。路面要是本身有坡, 这个"天"就跟着歪: lalaland 的观景台路面
  比真竖直斜约 3°, 结果生成的路灯、矮柱、长椅全跟着歪 2~3°, 导出的 FBX 里人也是斜着站的。
  反过来直接把"天"换成真竖直也不行: 链上的地面是"一个高度"的水平面, 斜坡一头高一头低,
  codex 那版 r2 这么换完, 长椅矮柱被压低约 0.3 m。
  所以链上照旧在"地面系"里做(地面是平的, 贴地全对), 只用这里的竖直方向做两件事:
    assemble_scene.py  物体沿真竖直立起来(而不是垂直于斜坡)
    motion_export.py   导出 FBX 前整体转到真竖直朝上 —— 这时地面变成斜面, 竖直的东西是直的
  3D 页(tools/box_fit_view.py)也按它显示。

怎么解: 一根竖杆在一帧画面里是一条线, 它和相机中心张成一个平面, 真竖直一定在这个平面里。
  多帧、多根杆子的平面求交(最小奇异向量)就是竖直方向。
  ★ 一根杆子只管"左右歪": 杆子总在画面中间时, 前后歪多少几度在画面里几乎看不出来。
    lalaland 只用路灯解出来前后偏了 10°; 加上十几根分布在画面各处的矮柱才定住。
    所以要求 ≥3 根杆子, 并且逐根去掉重解看稳不稳。

什么时候用: 杆子够(≥3 根)、稳(去掉任何一根重解, 变化 ≤1°)、和地面法线差 1°~8°。
  差不到 1°: 地面本来就平, 不动。超过 8°: 真坡少见, 多半是杆子或位姿有问题, 不用并在日志里说。
  没有竖杆: 不用(和以前一样拿地面法线当天)。

用法: python gravity_from_poles.py <R1>      # 打印并写 <R1>/06_gravity.json
"""
import json, os, sys
import numpy as np
from pycocotools import mask as mu

OUT_NAME = "06_gravity.json"
# 只要本身是竖直细杆的词。goal post 有横梁、traffic light 常常只抠到灯箱 —— 不要。
POLE_WORDS = {"street lamp", "lamp post", "lamppost", "light pole", "utility pole", "pole", "post",
              "bollard", "fence post", "sign post", "signpost", "flagpole", "parking meter"}
MIN_INST, MAX_SPREAD, MIN_TILT, MAX_TILT = 3, 1.0, 1.0, 8.0
MAX_FRAMES = 40                     # 每根杆子最多用几帧, 免得一根杆子的帧数压过别的杆子


def _is_pole(word):
    return word.replace("_", " ").strip().lower() in POLE_WORDS


def _line(m, min_rows):
    """mask -> (上点, 下点) 齐次像素坐标, 不像竖杆就 None。逐行取 mask 中心, 稳健拟 x = a*y + b。"""
    ys = np.where(m.any(1))[0]
    if len(ys) < min_rows: return None
    xs = np.array([np.mean(np.where(m[y])[0]) for y in ys], float)
    keep = np.ones(len(ys), bool)
    for _ in range(3):                                   # 去掉灯头/被人挡住的行
        a, b = np.polyfit(ys[keep], xs[keep], 1)
        r = np.abs(xs - (a * ys + b)); keep = r < max(1.5, 2.5 * np.median(r[keep]))
    if np.median(r[keep]) > 1.5 or abs(a) > np.tan(np.radians(20)): return None     # 不直 / 在画面里太斜
    if (ys[keep].max() - ys[keep].min()) / max(np.median(m[ys].sum(1)), 1) < 2.0: return None   # 太胖, 不像杆子
    y0, y1 = ys[keep].min(), ys[keep].max()
    return np.array([a * y0 + b, y0, 1.0]), np.array([a * y1 + b, y1, 1.0]), float(y1 - y0)


def _plane_normal(cam, hw, p0, p1):
    K = np.array(cam["intrinsic_3x3"], float); h, w = cam["input_hw"]
    K = np.diag([hw[1] / w, hw[0] / h, 1.0]) @ K        # 内参是 DA3 输入分辨率的, mask 是原图分辨率
    Ki = np.linalg.inv(K); m = np.cross(Ki @ p0, Ki @ p1); m /= np.linalg.norm(m)
    return np.linalg.inv(np.array(cam["extrinsic_w2c_4x4"], float))[:3, :3] @ m


def _solve(N, w):
    return np.linalg.svd(N * w[:, None])[2][-1]


def estimate(r1, UP):
    """返回写进 06_gravity.json 的那个 dict。UP 是 world_up.json 的天(地面法线)。"""
    UP = np.asarray(UP, float); UP = UP / np.linalg.norm(UP)
    _cj = f"{r1}/02_da3/cameras.json"
    cams = json.load(open(_cj)).get("cameras", {}) if os.path.exists(_cj) else {}   # 没有相机 = 没有竖杆可用
    rows, inst, used, lens = [], [], {}, []
    mk = f"{r1}/03_masks"
    for fn in sorted(os.listdir(mk)) if os.path.isdir(mk) else []:
        z = np.load(f"{mk}/{fn}", allow_pickle=True); word = str(z["word"])
        if not _is_pole(word) or len(z["meta"]) == 0: continue
        H, W = (int(x) for x in z["hw"]); min_rows = max(40, int(0.035 * H))
        by = {}
        for i, (oid, fi, area, edge) in enumerate(z["meta"]):
            if area > 0 and edge <= 0 and f"{int(fi):04d}.jpg" in cams: by.setdefault(int(oid), []).append(i)
        for oid, idx in by.items():
            idx = [idx[j] for j in np.linspace(0, len(idx) - 1, min(len(idx), MAX_FRAMES)).astype(int)]
            n0 = len(rows)
            for i in idx:
                L = _line(mu.decode({"counts": z["counts"][i], "size": [H, W]}).astype(bool), min_rows)
                if L is None: continue
                rows.append(_plane_normal(cams[f"{int(z['meta'][i]['frame']):04d}.jpg"], (H, W), L[0], L[1]))
                lens.append(L[2])
                inst.append(f"{word}#{oid}")
            if len(rows) > n0: used[f"{word}#{oid}"] = len(rows) - n0
    out = dict(up=UP.tolist(), gravity=UP.tolist(), 用了=False, 杆子=used, 观测数=len(rows))
    if len(used) < MIN_INST:
        out["原因"] = f"竖杆只有 {len(used)} 根(要 ≥{MIN_INST}), 照旧拿地面法线当天"; return out
    N = np.array(rows); inst = np.array(inst); names = sorted(used)
    # 权重: 线越长方向越准(画面里的角度误差 ~ 像素误差/线长); 再除 sqrt(帧数), 每根杆子分量一样
    cnt = np.array([used[k] for k in inst]); w = np.array(lens) / np.median(lens) / np.sqrt(cnt)
    g = _solve(N, w)
    for _ in range(2):                                    # 剔掉明显对不上的观测(抠歪了、被人挡了一半)再解
        r = np.abs(N @ g); bad = r > max(3 * 1.4826 * np.median(r), np.radians(0.5))
        w = np.where(bad, 0.0, w); g = _solve(N, w)
    g *= np.sign(g @ UP) or 1
    # 稳不稳: 依次去掉每一根杆子重解, 看最多变多少。
    # ★ 不用"按杆子有放回重抽样": 权重很不均(一根长路灯顶十几根矮柱)时, 它会同时抽掉路灯和几根好的矮柱
    #   再重复抽进噪声大的, lalaland 上报 1.66°, 而实际去掉任何一根都只变 ≤0.61°。
    spread = []
    for k in names:
        b = _solve(N, np.where(inst == k, 0.0, w)); spread.append(np.degrees(np.arccos(np.clip(abs(b @ g), 0, 1))))
    spread = float(max(spread)); tilt = float(np.degrees(np.arccos(np.clip(g @ UP, -1, 1))))
    out.update(竖杆竖直=g.tolist(), 和地面法线差=round(tilt, 2), 去掉一根最多变=round(spread, 2))
    if spread > MAX_SPREAD:
        out["原因"] = f"杆子之间不一致(去掉一根最多变 {spread:.2f}° > {MAX_SPREAD}°), 不用"
    elif tilt < MIN_TILT:
        out["原因"] = f"和地面法线只差 {tilt:.2f}°, 地面本来就平, 不动"
    elif tilt > MAX_TILT:
        out["原因"] = f"和地面法线差 {tilt:.2f}° > {MAX_TILT}°, 真坡少见, 多半是杆子或位姿有问题, 不用"
    else:
        out.update(gravity=g.tolist(), 用了=True, 原因=f"地面有约 {tilt:.1f}° 的坡: 物体沿竖杆的竖直立, 导出时转到竖直朝上")
    return out


def load(r1):
    """读 06_gravity.json; 没有或没启用 -> None, 否则返回竖直方向(DA3 世界系, 单位向量)。"""
    p = f"{r1}/{OUT_NAME}"
    if not os.path.exists(p): return None
    d = json.load(open(p))
    return np.asarray(d["gravity"], float) if d.get("用了") else None


def rot(u, v):
    """把单位向量 u 转到 v 的最小旋转(罗德里格斯)。"""
    u = np.asarray(u, float) / np.linalg.norm(u); v = np.asarray(v, float) / np.linalg.norm(v)
    w = np.cross(u, v); c = float(u @ v)
    if np.linalg.norm(w) < 1e-12: return np.eye(3)
    k = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def level_T(r1, UP):
    """地面系(glTF, 地面法线 +y) -> 竖直朝上的 4x4 旋转; 没有坡就是单位阵。导出 FBX 和 3D 页用。"""
    from world_frame import export_R
    g = load(r1); T = np.eye(4)
    if g is not None: T[:3, :3] = rot(export_R(UP) @ g, [0, 1.0, 0])
    return T


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from world_frame import load_up
    r1 = os.path.abspath(sys.argv[1]); UP, _, _ = load_up(f"{r1}/02_da3", "竖杆定竖直")
    d = estimate(r1, UP)
    json.dump(d, open(f"{r1}/{OUT_NAME}", "w"), ensure_ascii=False, indent=1)
    print(f"[竖直] 杆子 {len(d['杆子'])} 根, 观测 {d['观测数']}, 和地面法线差 {d.get('和地面法线差')}°, "
          f"去掉一根最多变 {d.get('去掉一根最多变')}° -> {'用' if d['用了'] else '不用'}: {d['原因']}")


def tilt_glb(src, dst, R):
    """在 glb 最外层套一个带旋转 R(3x3, glTF 系)的父节点 "level"。只改 JSON, 网格和贴图原样拷过去。"""
    import struct
    b = open(src, "rb").read()
    assert b[:4] == b"glTF", f"{src} 不是 glb"
    n0 = struct.unpack_from("<I", b, 12)[0]; js = json.loads(b[20:20 + n0]); rest = b[20 + n0:]
    R = np.asarray(R, float); w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2           # 旋转矩阵 -> 四元数(x,y,z,w)
    q = [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w]
    sc = js["scenes"][js.get("scene", 0)]
    js.setdefault("nodes", []).append({"name": "level", "rotation": [float(x) for x in q], "children": sc["nodes"]})
    sc["nodes"] = [len(js["nodes"]) - 1]
    j = json.dumps(js, separators=(",", ":")).encode(); j += b" " * (-len(j) % 4)
    with open(dst, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(j) + len(rest)))
        f.write(struct.pack("<I4s", len(j), b"JSON")); f.write(j); f.write(rest)
