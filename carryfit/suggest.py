"""闸 2 的"可能被拿着"建议。只是提示, 由人在候选页上点 held 确认(不自动勾)。
每个物体实例看四条, 都满足才建议:
  1. 动得远: 每帧取 mask 中心 + DA3 深度 -> 世界坐标; 位置散布(到中位点距离的 90 分位)至少是物体大小的 MOVE_X 倍。
     ★ 要用倍数, 不能用米数: 逐帧深度有噪声, 栏杆、台阶这种静物算出来也"动了"一米多, 但它们本身也有一米大;
       不能只看画面上的偏移: couplelift 的箱子主要朝镜头走, 画面上几乎不挪。
  2. 挨着人: 物体 mask 往外扩一圈碰到人的 mask, 而且碰到的那块人和物体差不多远(远处的树在画面上也会碰到人),
     这样的帧占 TOUCH_MIN 以上。
  3. 不是人身上的东西: 物体落在人 mask 里的比例(中位)不超过 INSIDE_MAX。衣服帽子是 ~1;
     人的 mask 有时会把手里的托盘也包进去(goodfellas tray #3 是 0.86), 所以门槛放得很高。
  4. 不瞬移: 跟踪编号跳到别的东西上时, mask 中心会一步跨好几个物体大小(stepup 的台阶、花盆)。
     相邻关键帧一步超过 JUMP_STEP 倍大小, 或最大的一步占总路程 JUMP_SHARE 以上, 就当编号跳了, 不建议。
门槛按 couplelift / goodfellas / stepup 三个场景定的: 箱子、托盘、两辆滑板车都标出来, 静物和衣服都不标。
只用闸 2 时盘上已有的东西(物体跟踪、人跟踪、DA3 深度和相机), 纯 CPU, 一个场景半分钟左右。

sam3d 环境:  python carryfit/suggest.py --r1 <r1> [--out <json>]
输出: <r1>/05_carry_suggest.json  {生成名: {词, 实例, 建议, 移动倍数, 挨着人, 在人里面, 最大一步}}, 并打印建议的几条
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
from pycocotools import mask as mu

MOVE_X = 5.0        # 位置散布 / 物体大小
TOUCH_MIN = 0.5     # 挨着人的帧占比
INSIDE_MAX = 0.9    # 落在人 mask 里的比例(中位)
NEAR_M, NEAR_REL = 0.5, 0.2   # 挨着人: 物体和旁边那块人的深度差在 0.5 m 或 20% 以内
JUMP_STEP = 2.5     # 相邻关键帧(按帧距折算到每帧)中心挪动 / 物体像素大小
JUMP_SHARE = 0.3    # 最大的一步 / 画面上的总路程
MIN_FRAMES = 8      # 有 mask 的帧少于这个数不判
OUT = "05_carry_suggest.json"


def person_masks(r1, n, H, W, step):
    """所有人的 mask 并起来, 按关键帧序号, 缩到 1/step。没跑人体线的轮次返回 None。"""
    d = f"{r1}/body4d/02_body4d/masks"
    if not os.path.isdir(d):
        return None
    out = np.zeros((n, H // step, W // step), bool)
    for i in range(n):
        p = f"{d}/{i:08d}.png"
        if os.path.exists(p):
            m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            m = m.reshape(m.shape[0], m.shape[1], -1).max(-1) > 0
            out[i] = cv2.resize(m.astype(np.uint8), (W // step, H // step), interpolation=cv2.INTER_NEAREST) > 0
    return out


def tag_of(r1, rows):
    """生成名前缀(<场景缩写>_<轮次>): 从闸口表的生成名里反推, 表没有就按 pipeline 的默认规则拼。"""
    for r in rows:
        tail = f"_{r['词'].replace(' ', '_')}_{r['实例']}"
        if str(r["生成名"]).endswith(tail):
            return str(r["生成名"])[: -len(tail)]
    return os.path.basename(os.path.dirname(os.path.abspath(r1)))[:3] + "_" + os.path.basename(os.path.abspath(r1))


def judge(word, md, oid, H, W, kidx, PM, step, Ks, w2c, depth_of):
    """一个实例的四个数。帧太少返回 None。"""
    meta = md["meta"]
    idx = np.array([i for i in np.where(meta["oid"] == oid)[0] if int(meta["frame"][i]) in kidx], int)
    if len(idx) < MIN_FRAMES:
        return None
    idx = idx[:: max(1, len(idx) // 30)]                 # 最多看 30 帧左右
    ker = np.ones((5, 5), np.uint8)
    P, size, C, R, F, touch, inside, dz = [], [], [], [], [], [], [], []
    for i in idx:
        f = kidx[int(meta["frame"][i])]
        m = mu.decode({"counts": md["counts"][i], "size": [H, W]})
        Ds = depth_of(f)
        if m.sum() < 50 or Ds is None:
            continue
        ms = cv2.resize(m, (W // step, H // step), interpolation=cv2.INTER_NEAREST) > 0
        z = Ds[ms]
        z = z[np.isfinite(z) & (z > 0)]
        if len(z) < 3:
            continue
        z = float(np.median(z))
        inside.append((ms & PM[f]).sum() / ms.sum())
        # 挨着 = 画面上碰到人, 而且碰到的那块人和物体差不多远(远处的树在画面上也会"碰到"人)
        ring = cv2.dilate(ms.astype(np.uint8), ker).astype(bool) & ~ms & PM[f]
        zp = Ds[ring]
        zp = zp[np.isfinite(zp) & (zp > 0)]
        if len(zp):
            dz.append(abs(float(np.median(zp)) - z) / z)
        touch.append(bool(len(zp) and abs(float(np.median(zp)) - z) < max(NEAR_M, NEAR_REL * z)))
        ys, xs = np.nonzero(m)
        c = np.array([xs.mean(), ys.mean()])
        ray = np.linalg.inv(Ks[f]) @ np.r_[c, 1.0]
        P.append((np.linalg.inv(w2c[f]) @ np.r_[ray * z, 1.0])[:3])
        r = float(np.sqrt(m.sum()))
        size.append(r * z / Ks[f][0, 0])
        C.append(c); R.append(r); F.append(f)
    if len(P) < MIN_FRAMES:
        return None
    P = np.array(P)
    move = float(np.percentile(np.linalg.norm(P - np.median(P, 0), axis=1), 90)) / max(float(np.median(size)), 1e-3)
    d = [float(np.linalg.norm(C[j + 1] - C[j])) for j in range(len(C) - 1)]
    steps = [d[j] / max(R[j], R[j + 1]) / (F[j + 1] - F[j]) for j in range(len(d)) if 0 < F[j + 1] - F[j] <= 3]
    jump = max(steps) if steps else 0.0
    share = max(d) / max(sum(d), 1e-6) if d else 0.0
    t = float(np.mean(touch)) if touch else 0.0
    ins = float(np.median(inside)) if inside else 0.0
    ok = move >= MOVE_X and t >= TOUCH_MIN and ins <= INSIDE_MAX and jump <= JUMP_STEP and share <= JUMP_SHARE
    why = [] if ok else [s for s, bad in (("动得不够远", move < MOVE_X), ("很少挨着人", t < TOUCH_MIN),
                                          ("在人身上", ins > INSIDE_MAX), ("编号跳了", jump > JUMP_STEP or share > JUMP_SHARE)) if bad]
    return dict(词=word, 实例=int(oid), 建议=bool(ok), 移动倍数=round(move, 1), 挨着人=round(t, 2),
                在人里面=round(ins, 2), 最大一步=round(jump, 2), 和人的深度差=round(float(np.median(dz)), 2) if dz else None,
                不建议的原因=why)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--out", default=None, help="写到哪(默认 <r1>/05_carry_suggest.json)")
    a = ap.parse_args()
    r1 = a.r1
    dst = a.out or f"{r1}/{OUT}"
    cams = json.load(open(f"{r1}/02_da3/cameras.json"))["cameras"]
    keys = sorted(cams)
    n = len(keys)
    rows = (json.load(open(f"{r1}/05_review.json")).get("候选") or []) if os.path.exists(f"{r1}/05_review.json") else []
    tag = tag_of(r1, rows)
    dead = {r["词"] for r in rows if r.get("决定") == "drop-word"}     # 人已经整词作废的不看
    live = {r["词"] for r in rows}            # 候选表里没有的词(排除表里的、去重判掉的)不看; 没有表就全看
    files = sorted(glob.glob(f"{r1}/03_masks/*.npz"))
    if not files:
        raise SystemExit(f"[失败] {r1}/03_masks 下没有物体 mask")
    H, W = [int(x) for x in np.load(files[0], allow_pickle=True)["hw"]]
    step = 4 if W > 1000 else 2
    PM = person_masks(r1, n, H, W, step)
    if PM is None:
        json.dump({}, open(dst, "w"))
        print("[拿着建议] 这一轮没有人, 不建议")
        return
    Ks, w2c = [], []
    for k in keys:
        c = cams[k]
        h, w = c["input_hw"]
        Ks.append(np.diag([W / w, H / h, 1.0]) @ np.array(c["intrinsic_3x3"], float))
        w2c.append(np.array(c["extrinsic_w2c_4x4"], float).reshape(4, 4))
    # ★ mask 的 frame = 帧文件名里的号(0004.jpg -> 4), 不是第几个关键帧; 相机隔帧取的场景两者不同。人的 mask 按第几个关键帧存。
    kidx = {int(os.path.splitext(k)[0]): i for i, k in enumerate(keys)}
    cache = {}

    def depth_of(f):
        if f not in cache:
            p = f"{r1}/02_da3/depth/{os.path.splitext(keys[f])[0]}_depth.npy"
            cache[f] = (cv2.resize(np.load(p).astype(np.float32), (W // step, H // step), interpolation=cv2.INTER_NEAREST)
                        if os.path.exists(p) else None)
        return cache[f]

    out = {}
    for fp in files:
        md = np.load(fp, allow_pickle=True)
        word = str(md["word"]) if "word" in md.files else os.path.basename(fp)[:-4].replace("_", " ")
        if word in dead or (live and word not in live):
            continue
        for oid in sorted(set(int(x) for x in md["meta"]["oid"])):
            v = judge(word, md, oid, H, W, kidx, PM, step, Ks, w2c, depth_of)
            if v:
                out[f"{tag}_{word.replace(' ', '_')}_{oid}"] = v
    json.dump(out, open(dst, "w"), ensure_ascii=False, indent=1)
    hits = sorted(((k, v) for k, v in out.items() if v["建议"]), key=lambda kv: -kv[1]["移动倍数"])
    for k, v in hits:
        print(f"[可能被拿着] {v['词']} #{v['实例']}: 移动了自身大小的 {v['移动倍数']} 倍, "
              f"{round(v['挨着人'] * 100)}% 的帧挨着人 —— 是的话在候选页上点 held")
    if not hits:
        print(f"[拿着建议] 看了 {len(out)} 个物体实例, 没有又在动、又挨着人的")


if __name__ == "__main__":
    main()
