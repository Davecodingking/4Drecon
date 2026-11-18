#!/usr/bin/env python
"""坏相机的帧: 相机沿天竖直挪, 让画面里的人脚踩在地形上。2026-11-12 加, joker 起头。

为什么: 坏相机的帧(bad_frames.json)位置本来就不可信。摆人那步的"地面硬约束"(body_motion.ground_lift)
  对这些帧"上下都贴地": 只挪人, 相机不动 —— 人在世界里站到地上了, 但用同一个相机投回原图就偏了。
  joker 2 号镜头开头 39 帧坏相机(视频 110~205 帧)贴地挪了 -57~+7 cm, 这段对齐 7~28%(同镜头好相机的帧 0.2~6.5%),
  镜头中位 8.5%。人和相机一起挪, 人相对相机不变(和原图对得上), 人也站在地上。
做法(只在 GROUND_MODE=terrain 时由 assemble 那步在 terrain.py 之后调用; 地形只用好相机的帧建, 挪坏相机不影响地形):
  1. 坏相机的每帧: terrain.people_feet 算每个人的脚点(人体网格按掩码里的深度定远近, 和摆人那步同一个比例)。
  2. 这帧要挪 = 各人(脚下地形高 - 脚高)的中位。没有人的帧在同一段坏相机里按时间插值, 整段都没有就不挪。
  3. 只挪连续 MIN_RUN 帧以上的坏段(更短的 cam_smooth 已用前后好帧插值), 整段挪同一个量(各帧要挪的量取中位),
     不逐帧挪(逐帧会把脚点 ±10 cm 的噪声加回相机); 剩下的残差由摆人那步的贴地兜。坏段不跨切点。
  4. 相机中心 += 挪 * up(只挪位置不转)。超过 MAX_SHIFT 的不挪(坏得离谱的相机, 挪了也救不回来), 记进报告。
  原件存 02_da3/cameras_before_badcam.json(第一次); 结果写进 cameras.json 的"坏相机贴地"。
  再跑一次: 脚已经在地上, 要挪的量接近 0, 不会越挪越多。cam_smooth --redo 从平滑前的原件重来, 之后要重跑这一步。
用法: python badcam_ground.py --r1 <场景结果目录>
"""
import argparse, json, os, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "data", "outdoor", "scripts")
sys.path.insert(0, SCRIPTS); sys.path.insert(0, HERE)

MIN_RUN = 4           # 和 cam_smooth.LONG_MIN 一样: 连续这么多帧以上的坏段才挪(更短的 cam_smooth 已插值)
MAX_SHIFT = 2.0       # 米: 要挪超过这么多的帧不挪
MIN_SHIFT = 0.02      # 米: 整段都小于这个就不改文件


def runs(idx):
    """按先后排好的帧号 -> 连续段(相邻两帧在原视频里中间没有好相机帧)的下标列表。idx: 每帧在全部相机里的序号。"""
    out, cur = [], [0]
    for i in range(1, len(idx)):
        if idx[i] == idx[i - 1] + 1:
            cur.append(i)
        else:
            out.append(cur); cur = [i]
    out.append(cur)
    return out


def main(r1):
    import terrain
    from scipy.ndimage import gaussian_filter1d
    terrain.HERE = SCRIPTS
    da3 = f"{r1}/02_da3"; p = f"{da3}/cameras.json"; tp = f"{r1}/07_terrain.npz"
    # ★ 11-12 试过没有地形(GROUND_MODE=plane)也做, 已撤: 平面地面是全片一块, 和某个镜头的脚差一截时, 好相机帧和坏相机帧
    #   差得一样多(irish 镜头 2: 好帧脚高出地面 11~38 cm, 坏帧 13~33 cm), 只挪坏相机段 = 段两头相机各跳一下
    #   (328~341 帧整段 -23 cm, 这个镜头位置抖动 95% 分位 10 -> 24 cm)。没有地形就不做。
    if not os.path.exists(tp):
        print("[坏相机贴地] 没有地形(07_terrain.npz), 不做"); return
    doc = json.load(open(p)); cams = doc["cameras"]; bad = terrain._bad(da3)
    fm_ = json.load(open(f"{da3}/frame_map.json")); vfm = fm_["frames"]; fps_ = float(fm_.get("fps") or 25)
    allk = sorted(cams, key=lambda k: int(vfm[k])); pos = {k: i for i, k in enumerate(allk)}
    keys = [k for k in allk if k in bad and os.path.exists(f"{da3}/depth/{k[:-4]}_depth.npy")]
    if not keys:
        print("[坏相机贴地] 没有坏相机的帧"); return
    if os.path.exists(tp):
        T = terrain.Terrain(tp); up = T.up; gh = lambda q, vf: float(T.height_at(q, vf=vf)[0]); shot_of = T.shot_of
    else:
        W_ = json.load(open(f"{da3}/world_up.json")); up = np.array(W_["up"], float); up /= np.linalg.norm(up); GH_ = float(W_["ground_h"])
        cuts_ = json.load(open(f"{da3}/shot_cuts.json"))["cuts"] if os.path.exists(f"{da3}/shot_cuts.json") else []
        gh = lambda q, vf: GH_; shot_of = lambda vf: int(np.searchsorted(cuts_, vf / fps_ + 1e-3, side="right"))
    PF = terrain.people_feet(r1, da3, keys, cams, vfm, up)
    raw = np.full(len(keys), np.nan)
    for i, k in enumerate(keys):
        vf = int(vfm[k]); P = PF.get(vf)
        if P is None or not len(P):
            continue
        d = np.array([gh(q, vf) - float(q @ up) for q in P])
        d = d[np.isfinite(d)]
        if len(d):
            raw[i] = float(np.median(d))
    shift = np.zeros(len(keys)); segs = []; handled = []
    sh_ = [shot_of(int(vfm[k])) for k in keys]
    rs = [x for r0 in runs([pos[k] for k in keys]) for x in np.split(np.array(r0), np.nonzero(np.diff([sh_[i] for i in r0]))[0] + 1)]
    for r in rs:   # 坏段不跨切点(joker 110~205 帧那串跨 6.71 秒切点, 后两帧是下一个镜头的开头)
        r = np.array(r); v = raw[r]; ok = np.isfinite(v)
        seg = dict(帧=[int(vfm[keys[r[0]]]), int(vfm[keys[r[-1]]])], 帧数=len(r), 有脚点的帧=int(ok.sum()))
        # ★ 11-12 第二版: 短坏段(< MIN_RUN 帧)cam_smooth 已经用前后好帧插值过, 和地形是一致的, 不挪。
        #   第一版逐帧挪, joker 3 号镜头里 298 / 305 / 318 三个零散坏帧被上下挪 +19 / -33 / -11 cm, 镜头里一跳一跳。
        #   长坏段整段挪同一个量(各帧要挪的量取中位): cam_smooth 已把这段拟合成匀速运动, 逐帧挪会把脚点的噪声(±10 cm)加回相机。
        if len(r) < MIN_RUN:
            seg["挪_m"] = "短坏段, 已插值, 不挪"; segs.append(seg); continue
        if not ok.any():
            seg["挪_m"] = None; segs.append(seg); continue
        v0 = float(np.median(v[ok]))
        if abs(v0) > MAX_SHIFT:
            seg.update(挪_m=None, 太大不挪=round(v0, 2)); segs.append(seg); continue
        shift[r] = v0; handled += [int(vfm[keys[i]]) for i in r]
        seg.update(挪_m=round(v0, 2), 各帧要挪_m=[round(float(np.min(v[ok])), 2), round(float(np.max(v[ok])), 2)])
        segs.append(seg)
    # 处理过的帧: 这些帧的高低已经连相机一起定了, 摆人那步的地面约束不再对它们"上下都贴地"只挪人(见 body_motion)
    out = dict(最短段=MIN_RUN, 上限_m=MAX_SHIFT, 段=segs, 处理过的帧=sorted(handled),
               逐帧_m={str(int(vfm[k])): round(float(s), 3) for k, s in zip(keys, shift)})
    if not handled:
        print(f"[坏相机贴地] {json.dumps(dict(段=segs), ensure_ascii=False)} (没有要处理的长坏段, 不改)"); return
    for k, s in zip(keys, shift):
        if abs(s) < MIN_SHIFT:
            continue
        E = np.array(cams[k]["extrinsic_w2c_4x4"], float); R = E[:3, :3]; C = -R.T @ E[:3, 3] + s * up
        E[:3, 3] = -R @ C; cams[k]["extrinsic_w2c_4x4"] = E.tolist()
        if cams[k].get("camera_center_world") is not None:
            cams[k]["camera_center_world"] = C.tolist()
    bak = f"{da3}/cameras_before_badcam.json"
    if not os.path.exists(bak):
        shutil.copy(p, bak)
    prev = doc.get("坏相机贴地")
    if prev:   # 再跑: 累计挪的量记在一起, 方便核对
        out["上一次"] = {kk: prev[kk] for kk in ("段",) if kk in prev}
    doc["坏相机贴地"] = out
    json.dump(doc, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"[坏相机贴地] {json.dumps(dict(段=segs), ensure_ascii=False)} -> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    main(ap.parse_args().r1)
