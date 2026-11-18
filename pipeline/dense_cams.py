"""人体全帧(2026-11-12): 场景只在抽出来的帧(室内 120 帧)上算了相机; 人体按全帧出的时候, 中间帧的相机由前后两个场景相机插出来。

为什么: 人体那条线以前直接用场景那 120 帧(jw4 592 帧里每 5 帧 1 帧, 约 6 fps), 只是因为每一帧要有相机 —— 动作被抽稀了,
  抹鼻子这种不到 1 秒的动作只剩两三帧。场景要抽帧(DA3 / VGGT 显存, 帧多了相机反而漂), 人体不用。

产物 <R1>/02_da3/dense/(和 02_da3 同一套格式, 出网格的 --cam-json 和 body_into_scene.load_scene(dense=True) 读它):
  cameras.json    {"cameras": 场景的键原样 + 插出来的 dNNNN.jpg(NNNN = 原视频帧号)}
  frame_map.json  {"frames": 键 -> 原视频帧号(全部), "fps": ...}
  bad_frames.json {"坏帧": 场景的坏帧 + 挨着坏相机插出来的帧}
规则:
  - 只在同一个镜头里插(切点 = body_into_scene.shot_cuts): 隔着切点插 = 两个机位中间的一个不存在的相机。
  - 镜头里第一个场景帧之前 / 最后一个之后的帧: 用最近的那个场景相机(不外推)。
  - 旋转球面插值, 相机中心 / 内参线性插值。
  - 用到的两个场景相机任一个是坏相机 -> 这一帧也算坏(摆人时坏相机的帧用模型自己的位置, 见 body_motion)。
  - 这个镜头一个场景帧都没有 -> 借最近的、算坏。
  - 深度没有: 只有场景帧有 depth/*.npy; 量远近的地方(depth_ratio)对没有深度的帧本来就跳过, 由前后帧的距离接上(frame_ratios)。
"""
import json, os
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

SUB = "dense"


def _interp(ca, cb, w):
    if w <= 0.0:
        return dict(ca)
    if w >= 1.0:
        return dict(cb)
    A = np.linalg.inv(np.asarray(ca["extrinsic_w2c_4x4"], float)); Bm = np.linalg.inv(np.asarray(cb["extrinsic_w2c_4x4"], float))
    R = Slerp([0.0, 1.0], Rot.from_matrix(np.stack([A[:3, :3], Bm[:3, :3]])))([w]).as_matrix()[0]
    C = (1 - w) * A[:3, 3] + w * Bm[:3, 3]
    c2w = np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = C
    K = (1 - w) * np.asarray(ca["intrinsic_3x3"], float) + w * np.asarray(cb["intrinsic_3x3"], float)
    out = dict(ca)
    out.update(extrinsic_w2c_4x4=np.linalg.inv(c2w).tolist(), intrinsic_3x3=K.tolist(), camera_center_world=C.tolist(),
               插值=dict(前=None, 后=None, w=round(float(w), 4)))
    if "mean_conf" in ca and "mean_conf" in cb:
        out["mean_conf"] = float(min(ca["mean_conf"], cb["mean_conf"]))
    return out


def build(cams, fmap, bad, cuts, fps, vfs):
    """cams {键: 相机}; fmap {键: 原视频帧号}; bad 坏帧的键; cuts 切点(秒); fps; vfs 要的原视频帧号。
    返回 (cams2, fmap2, bad2): 场景的键原样保留, 缺的帧插出来(键 dNNNN.jpg)。"""
    keys = sorted((k for k in fmap if k in cams), key=lambda k: fmap[k])
    kv = np.array([int(fmap[k]) for k in keys])
    cuts = np.asarray(sorted(cuts), float)
    shot = lambda v: int(np.searchsorted(cuts, v / fps + 1e-3, side="right"))
    ks = np.array([shot(v) for v in kv])
    v2k = {int(fmap[k]): k for k in keys}
    cams2 = {k: cams[k] for k in keys}; fmap2 = {k: int(fmap[k]) for k in keys}
    bad = set(bad); bad2 = {k for k in bad if k in cams2}
    for v in sorted({int(x) for x in vfs}):
        if v in v2k:
            continue
        idx = np.where(ks == shot(v))[0]
        orphan = not len(idx)
        if orphan:
            idx = np.arange(len(keys))
        j = int(np.searchsorted(kv[idx], v))
        if orphan:
            a = b = int(idx[np.argmin(np.abs(kv[idx] - v))])
        elif j == 0:
            a = b = idx[0]
        elif j >= len(idx):
            a = b = idx[-1]
        else:
            a, b = idx[j - 1], idx[j]
        w = 0.0 if a == b else (v - kv[a]) / (kv[b] - kv[a])
        key = f"d{v:04d}.jpg"
        c = _interp(cams[keys[a]], cams[keys[b]], w)
        c["插值"] = dict(前=keys[a], 后=keys[b], w=round(float(w), 4), 镜头外借来的=bool(orphan))
        cams2[key] = c; fmap2[key] = v
        if orphan or keys[a] in bad or keys[b] in bad:
            bad2.add(key)
    return cams2, fmap2, bad2


def write(r1, vfs):
    """按场景 02_da3 的相机写 <r1>/02_da3/dense/。返回目录。"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import body_into_scene as B
    from _common import load_frame_map
    da3 = f"{r1}/02_da3"
    cj = json.load(open(f"{da3}/cameras.json"))
    fm = load_frame_map(da3)
    fmj = json.load(open(f"{da3}/frame_map.json")) if os.path.exists(f"{da3}/frame_map.json") else {}
    fps = float(fmj.get("fps") or cj.get("fps") or 30.0)
    cams2, fmap2, bad2 = build(cj["cameras"], fm, B.load_badcam(da3), B.shot_cuts(da3), fps, vfs)
    out = f"{da3}/{SUB}"; os.makedirs(out, exist_ok=True)
    meta = {k: v for k, v in cj.items() if k != "cameras"}
    json.dump(dict(meta, cameras=cams2, 说明="场景相机 + 插出来的中间帧(dense_cams.py)"), open(f"{out}/cameras.json", "w"))
    json.dump(dict({k: v for k, v in fmj.items() if k != "frames"}, frames=fmap2, identity=False, fps=fps), open(f"{out}/frame_map.json", "w"))
    json.dump({"坏帧": sorted(bad2), "说明": "场景的坏帧 + 用到坏相机插出来的帧"}, open(f"{out}/bad_frames.json", "w"), ensure_ascii=False)
    n_new = len(cams2) - len([k for k in cams2 if not k.startswith("d")])
    print(f"[全帧相机] 场景 {len(cams2) - n_new} 个 + 插出来 {n_new} 个 -> {out} (坏 {len(bad2)})")
    return out
