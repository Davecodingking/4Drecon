#!/usr/bin/env python
"""每个镜头按画面里的竖直线扶正。2026-11-12 加, joker 起头。

为什么: 一个片子的所有镜头在 VGGT-Ω 里一起重建、落在同一个世界系, 但各镜头的"竖直"不一样。全片只用一个 up
  (world_up.json, joker 是"相机的上"), 各镜头相对真竖直各歪各的。joker 实测(画面竖线的灭点, 每个镜头 85~2098 条线):
  1 号 +16.4°、3 号 +5.7°、4 号 -15.4°、5 号 +13.6°(沿楼梯上坡方向)。后果:
    - 楼梯被量成 11°(往上看的镜头歪了 +14~16°, 数台阶推出来约 30°);
    - 往下看的镜头(4 号)街面斜 -12.6°, 扶正后剩 +2.8°(这条是独立核对: 街面本该水平);
    - 几个镜头的地面互相对不上, 拼统一场景时人埋进地里 / 街被楼梯盖住。
  另外几条线索: 路灯杆(画面方向)1 号 +15.2°、5 号 +25.3°、4 号 -8.1°, 和竖线一致; GeoCalib、人体模型的身体竖轴
  在往上看的镜头上给出相反的方向(学出来的, 被楼梯带偏), 不用。
做法(只在 GROUND_MODE=terrain 时由 boxes 那步调用, 在 shot_register 之前):
  1. 每个镜头的好相机帧: LSD 检测线段, 留画面里接近竖直(±31°)、长度 > 画面高 6% 的; 每条线段和相机中心组成一个平面,
     平面法向转到世界系。真竖直方向和这些平面都平行 -> RANSAC(内点: 法向和方向夹角 89~91°)+ 内点最小特征向量。
  2. 可信才扶: 内点 >= MIN_LINES 条、条件数(次小 / 最小特征值) >= MIN_COND、偏差在 MIN_DEG~MAX_DEG 之间。
  3. 扶正 = 绕这个镜头相机中心的平均位置, 把它的真竖直转到全片的 up(最小转角)。这个镜头的所有相机(含坏相机帧)一起转;
     深度在相机系里, 跟着相机走, 镜头内部一点不变, 人投回原图也不变。
  原件存 02_da3/cameras_before_level.json(第一次); 结果写进 cameras.json 的"镜头扶正"。
  再跑一次: 已经扶正, 偏差 < MIN_DEG, 不再转。cam_smooth --redo 之后要重跑。
用法: python shot_level.py --r1 <场景结果目录> [--frames <原图目录>]
"""
import argparse, json, os, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data", "outdoor", "scripts"))

MIN_LINES = 40      # 至少这么多条竖线支持(joker 3 号镜头 43 条, 条件数 155)
MIN_COND = 50       # 条件数: 线段要来自不同方向(只有一侧的竖线定不出前后倾)
MIN_DEG = 2.0       # 偏不到这么多不转
MAX_DEG = 35.0      # 偏这么多以上多半是检测错了, 不转
LINE_MIN = 0.06     # 线段至少画面高的这么多
INLIER_DEG = 1.0
EVERY = 2           # 每隔几个好相机帧取一帧检测


def vertical_of(frames, keys, cams, rng, up):
    """一组帧 -> (世界系真竖直方向 v, 内点数, 线段数, 条件数) 或 None。v 的方向未定号, 调用方按 up 定。"""
    import cv2
    lsd = cv2.createLineSegmentDetector()
    N = []
    for k in keys:
        img = cv2.imread(f"{frames}/{k}", cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H_, W_ = img.shape; segs = lsd.detect(img)[0]
        if segs is None:
            continue
        cm = cams[k]; K = np.array(cm["intrinsic_3x3"], float); h0, w0 = cm.get("input_hw", (H_, W_)); Ki = np.linalg.inv(K)
        R = np.array(cm["extrinsic_w2c_4x4"], float)[:3, :3]
        for x1, y1, x2, y2 in segs.reshape(-1, 4):
            if np.hypot(x2 - x1, y2 - y1) < LINE_MIN * H_ or abs(x2 - x1) > 0.6 * abs(y2 - y1):
                continue
            p1 = Ki @ np.r_[x1 * w0 / W_, y1 * h0 / H_, 1]; p2 = Ki @ np.r_[x2 * w0 / W_, y2 * h0 / H_, 1]
            n = np.cross(p1, p2); n /= np.linalg.norm(n); N.append(R.T @ n)
    if len(N) < MIN_LINES:
        return None
    N = np.array(N); best = None; s_in = np.sin(np.radians(INLIER_DEG))
    for _ in range(3000):
        a, b = N[rng.choice(len(N), 2, replace=False)]; v = np.cross(a, b)
        if np.linalg.norm(v) < 1e-6:
            continue
        v /= np.linalg.norm(v)
        if abs(v @ up) < np.cos(np.radians(MAX_DEG)):     # 只在 up 附近找: 往下拍的镜头里远去的街道线在画面里也近乎竖直(joker 4 号镜头被它带到 68.9°)
            continue
        inl = np.abs(N @ v) < s_in
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    if best is None:
        return None
    Ni = N[best[1]]; ev, V = np.linalg.eigh(Ni.T @ Ni)
    return V[:, 0], best[0], len(N), float(ev[1] / max(ev[0], 1e-12))


def rot_between(a, b):
    """把单位向量 a 转到 b 的最小旋转矩阵。"""
    v = np.cross(a, b); c = float(a @ b); s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def main(r1, frames):
    import terrain
    D = f"{r1}/02_da3"; p = f"{D}/cameras.json"; doc = json.load(open(p)); cams = doc["cameras"]
    fm = json.load(open(f"{D}/frame_map.json")); vfm = fm["frames"]; fps = float(fm.get("fps") or 25)
    up = np.array(json.load(open(f"{D}/world_up.json"))["up"], float); up /= np.linalg.norm(up)
    cuts = json.load(open(f"{D}/shot_cuts.json"))["cuts"] if os.path.exists(f"{D}/shot_cuts.json") else []
    bad = terrain._bad(D)
    if frames is None:
        frames = fm.get("frames_dir") or doc.get("source")
    shot = {k: int(np.searchsorted(cuts, vfm[k] / fps + 1e-3, side="right")) for k in cams}
    rng = np.random.default_rng(0); rep, moved = [], False
    for s in sorted(set(shot.values())):
        ks = sorted([k for k in cams if shot[k] == s], key=lambda k: vfm[k])
        good = [k for k in ks if k not in bad][::EVERY]
        r = dict(镜头=s + 1, 帧=len(ks))
        res = vertical_of(frames, good, cams, rng, up) if good else None
        if res is None:
            r["结论"] = "竖线不够, 不扶"; rep.append(r); continue
        v, ninl, nall, cond = res
        v = v if v @ up > 0 else -v
        ang = float(np.degrees(np.arccos(np.clip(v @ up, -1, 1))))
        r.update(竖线=nall, 内点=ninl, 条件数=round(cond), 偏_度=round(ang, 1))
        if ninl < MIN_LINES or cond < MIN_COND or not (MIN_DEG <= ang <= MAX_DEG):
            r["结论"] = "不扶(" + ("竖线不够" if ninl < MIN_LINES else "条件数低" if cond < MIN_COND else "偏得太小" if ang < MIN_DEG else "偏得太大, 多半检测错") + ")"
            rep.append(r); continue
        Rc = rot_between(v, up)
        P0 = np.mean([-np.array(cams[k]["extrinsic_w2c_4x4"], float)[:3, :3].T @ np.array(cams[k]["extrinsic_w2c_4x4"], float)[:3, 3] for k in ks], 0)
        for k in ks:
            E = np.array(cams[k]["extrinsic_w2c_4x4"], float); R = E[:3, :3]; C = -R.T @ E[:3, 3]
            C2 = Rc @ (C - P0) + P0; R2 = R @ Rc.T
            E[:3, :3] = R2; E[:3, 3] = -R2 @ C2; cams[k]["extrinsic_w2c_4x4"] = E.tolist()
            if cams[k].get("camera_center_world") is not None:
                cams[k]["camera_center_world"] = C2.tolist()
        r["结论"] = f"扶正 {ang:.1f}°"; moved = True; rep.append(r)
    out = dict(门槛=dict(竖线=MIN_LINES, 条件数=MIN_COND, 最小_度=MIN_DEG, 最大_度=MAX_DEG), 各镜头=rep)
    if moved:
        bak = f"{D}/cameras_before_level.json"
        if not os.path.exists(bak):
            shutil.copy(p, bak)
        doc["镜头扶正"] = out
        json.dump(doc, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"[镜头扶正] {json.dumps(out, ensure_ascii=False)}" + ("" if moved else " (没有要扶的)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--frames", default=None)
    a = ap.parse_args()
    main(a.r1, a.frames)
