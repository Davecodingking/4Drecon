#!/usr/bin/env python
"""切镜头处拆轨迹: 一个跟踪编号跨过切点, 切点两边在三维里不在一个地方 -> 拆成两个实例。(2026-11-10)

起因: hitch_cuts_indoor r1 的 floor lamp#0 跨了 6 个镜头 —— 远景(镜头 1/3/5/7)里是右边那盏白色三脚落地灯,
反打(镜头 2/4)里是窗边那个米色鼓形灯罩, 两盏灯隔着好几米。SAM3 视频跟踪靠"上一帧在哪、长什么样"往下传编号,
不知道切了镜头, 新镜头里又出现一个"落地灯"就续进了老编号。下游把这个编号所有帧的点云堆在一起量盒子 ->
6.7 m 长、离地 1.3~1.9 m 的盒子, mesh 又是从第 60 帧(灯罩近景, 没有杆)生成的 -> 一个灯罩飘在两盏灯中间(桌子上方)。

做法: 每个编号的帧按镜头分段(切点 = body_into_scene.shot_cuts, ffmpeg scdet, 和摆人那步同一份);
每段取相机可信的帧(不在 bad_frames.json 里), 把 mask 反投影成三维点取中位 -> 这一段的位置。
  - 和这个编号第一段(帧最多的那组)的位置差不超过 TOL 米 -> 同一个东西, 留在原编号;
  - 差得远 -> 拆出去, 位置相近的几段并成一个新编号;
  - 这一段一帧可信相机都没有 -> 判不了, 单独拆成一个新编号(相机错的帧本来就量不了盒子, 留在原编号只会污染挑图)。
只在跨了切点的编号上动; 一镜到底的片子(没有切点)什么都不改。
改的是 03_masks/<词>.npz 的 meta["oid"](mask 本身不动), 原件留在 03_masks_before_split/,
账在 03_split_cuts.json。重跑幂等: 原件在就从原件拆。
环境变量: R1(必填), SPLIT_TOL(米, 默认 1.0; 尺度这一步之前是 DA3 自己的米)。hitch 上同一件东西跨镜头差 0.7~0.8 m(各镜头相机 / 深度的误差), 真不是一个的差 1.4 m 以上。
"""
import os, sys, json, glob, shutil
import numpy as np, cv2
from pycocotools import mask as mask_util
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_into_scene as B

R1 = os.environ["R1"]; MK = f"{R1}/03_masks"; BAK = f"{R1}/03_masks_before_split"   # 不放进 03_masks: pick_from_masks 按目录列文件, 会把子目录当 npz 读而崩
DA3 = f"{R1}/02_da3"
TOL = float(os.environ.get("SPLIT_TOL", "1.0"))
OUT = f"{R1}/03_split_cuts.json"

cuts = B.shot_cuts(DA3)
if not cuts:
    json.dump(dict(切点=[], 说明="一镜到底, 不拆"), open(OUT, "w"), ensure_ascii=False, indent=1)
    print("[切镜头拆轨迹] 没有切点, 不拆"); sys.exit(0)
fm = json.load(open(f"{DA3}/frame_map.json")); fps = fm.get("fps") or 25.0; vf = fm["frames"]
cams = json.load(open(f"{DA3}/cameras.json"))["cameras"]
bad = B.load_badcam(DA3)


IDENT = bool(fm.get("identity"))      # 室外 True: 帧文件名就是视频帧号; DA3 只抽了一部分帧, 其余帧查不到 frame_map


def shot_of(f):
    k = f"{f:04d}.jpg"
    t = vf[k] if k in vf or not IDENT else f                             # 室内(identity=False)查不到照旧 KeyError
    return int(np.searchsorted(cuts, t / fps + 1e-3, side="right"))


def pos3d(f, m):
    """这一帧 mask 的三维位置: mask 里深度的中位 × mask 像素中心那条视线, 转到世界系。"""
    k = f"{f:04d}.jpg"; c = cams.get(k)
    if c is None or k in bad:
        return None
    dp = f"{DA3}/depth/{f:04d}_depth.npy"
    if not os.path.exists(dp):
        return None
    d = np.load(dp); H, W = d.shape
    mm = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    if mm.sum() < 20:
        return None
    z = float(np.median(d[mm]))
    ys, xs = np.nonzero(mm)
    K = np.array(c["intrinsic_3x3"], float); h0, w0 = c["input_hw"]
    sx, sy = W / w0, H / h0
    u, v = xs.mean() / sx, ys.mean() / sy
    pc = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    c2w = np.linalg.inv(np.array(c["extrinsic_w2c_4x4"], float))
    return c2w[:3, :3] @ pc + c2w[:3, 3]


os.makedirs(BAK, exist_ok=True)
report = dict(切点=cuts, 门槛_米=TOL, 拆了={}, 没拆的跨切点编号={})
for p in sorted(glob.glob(f"{MK}/*.npz")):
    name = os.path.basename(p)
    src = f"{BAK}/{name}"
    if not os.path.exists(src):
        shutil.copyfile(p, src)
    z = np.load(src, allow_pickle=True)
    meta = z["meta"].copy(); counts = z["counts"]; H, W = z["hw"]
    word = str(z["word"]) if "word" in z.files else name[:-4].replace("_", " ")
    nxt = int(meta["oid"].max()) + 1 if len(meta) else 0
    changed = False
    for oid in sorted(set(int(x) for x in meta["oid"])):
        rows = np.where(meta["oid"] == oid)[0]
        segs = {}
        for i in rows:
            segs.setdefault(shot_of(int(meta["frame"][i])), []).append(i)
        if len(segs) < 2:
            continue
        P = {}
        for s, ii in segs.items():
            jj = [i for i in ii if f"{int(meta['frame'][i]):04d}.jpg" in cams] or ii   # 只有 DA3 抽到的帧有深度, 先在这些里抽
            pts = [pos3d(int(meta["frame"][i]), mask_util.decode({"counts": counts[i], "size": [int(H), int(W)]}))
                   for i in jj[:: max(1, len(jj) // 6)]]                      # 每段最多抽 6 帧
            pts = [q for q in pts if q is not None]
            P[s] = np.median(np.array(pts), 0) if pts else None
        # 第一组: 帧最多、而且有位置的那段
        order = sorted(segs, key=lambda s: (P[s] is None, -len(segs[s])))
        groups = []                                                           # [(位置, [段])]
        for s in order:
            if P[s] is None:
                groups.append((None, [s])); continue
            for g in groups:
                if g[0] is not None and np.linalg.norm(P[s] - g[0]) <= TOL:
                    g[1].append(s); break
            else:
                groups.append((P[s], [s]))
        if len(groups) == 1:
            report["没拆的跨切点编号"][f"{word}#{oid}"] = sorted(int(s) + 1 for s in segs)
            continue
        info = []
        for gi, (pos, ss) in enumerate(groups):
            new = oid if gi == 0 else nxt
            if gi:
                nxt += 1
                for s in ss:
                    meta["oid"][segs[s]] = new
            d0 = None if pos is None or groups[0][0] is None else round(float(np.linalg.norm(pos - groups[0][0])), 2)
            info.append(dict(编号=new, 镜头=sorted(int(s) + 1 for s in ss), 帧数=int(sum(len(segs[s]) for s in ss)),
                             离第一组_米=d0, 判不了=pos is None))
        report["拆了"][f"{word}#{oid}"] = info
        changed = True
    if changed:
        np.savez(p, **{k: (meta if k == "meta" else z[k]) for k in z.files})
    elif not os.path.exists(p) or os.path.getsize(p) != os.path.getsize(src):
        shutil.copyfile(src, p)                                               # 上次拆过、这次不拆了: 还原
json.dump(report, open(OUT, "w"), ensure_ascii=False, indent=1)
n = sum(len(v) - 1 for v in report["拆了"].values())
print(f"[切镜头拆轨迹] {len(cuts)} 个切点; 拆了 {len(report['拆了'])} 个编号, 多出 {n} 个实例; "
      f"跨切点但位置对得上没拆的 {len(report['没拆的跨切点编号'])} 个 -> {OUT}")
for k, v in report["拆了"].items():
    print(f"    {k}: " + "; ".join(f"#{x['编号']} 镜头{x['镜头']} {x['帧数']}帧" +
                                  (" (相机都不可信, 判不了)" if x["判不了"] else
                                   f" 离第一组 {x['离第一组_米']} m" if x["离第一组_米"] is not None else "") for x in v))
