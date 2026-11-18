"""给 carryfit 补物体 mask: 流水线的 SAM3 跟踪是按词检测再往后跟的, 被拿着的东西常常只跟上一小段
(goodfellas 的托盘只有最后 10 帧, 厨师端着走过来的 20 多帧没有)。
这里拿这个实例已有的 mask 里露得最好的几帧当种子(每帧取 mask 内部的几个点), 用 SAM3 的跟踪器前后双向重跟一遍。

sam3 环境(carryfit/run.sh 不管这个, 用 pipeline 的 gpulock):
  python carryfit/retrack.py --r1 <r1> --word tray --oid 3 --out carryfit/runs/<名字>/mask.npz
输出和 03_masks/<词>.npz 同格式(RLE, meta 有 frame / oid), 只有这一个实例, oid 不变; 另出一张对比图 mask.jpg。
"""
import argparse
import json
import os

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

CKPT = os.path.expanduser("~/sam3d/sam3/checkpoints/sam3.pt")


def interior_points(m, k=4):
    """mask 内部离边最远的点 + 沿主轴两侧各一个, 共 k 个(像素坐标)"""
    dt = ndimage.distance_transform_edt(m)
    pts = []
    d = dt.copy()
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(d), d.shape)
        if d[y, x] <= 1:
            break
        pts.append((x, y))
        r = max(3, int(dt[y, x] * 1.5))
        d[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = 0
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--word", required=True)
    ap.add_argument("--oid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=4, help="最多几帧当种子")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fm = json.load(open(f"{a.r1}/02_da3/frame_map.json"))
    keys = sorted(fm["frames"])
    n = len(keys)
    md = np.load(f"{a.r1}/03_masks/{a.word.replace(' ', '_')}.npz", allow_pickle=True)
    H, W = [int(x) for x in md["hw"]]
    meta = md["meta"]
    # ★ 03_masks 的 frame 是帧文件名里的号(0004.jpg -> 4), 不是第几个关键帧: 相机隔几帧取一帧的场景(coupledance)
    #   两者不一样。这里内部按第几个关键帧算(SAM3 的帧目录只放关键帧), 写出时换回文件名号, 和 03_masks 同一个约定。
    kidx = {int(os.path.splitext(k)[0]): i for i, k in enumerate(keys)}
    old = {}
    for i in np.where(meta["oid"] == a.oid)[0]:
        if int(meta["frame"][i]) in kidx:
            old[kidx[int(meta["frame"][i])]] = mu.decode({"counts": md["counts"][i], "size": [H, W]}).astype(bool)
    area = {f: m.sum() for f, m in old.items()}
    # 种子: 面积最大的帧, 再挑面积 >= 一半、彼此隔开的几帧
    cand = sorted(area, key=lambda f: -area[f])
    seeds = []
    for f in cand:
        if area[f] < 0.5 * area[cand[0]] or len(seeds) >= a.seeds:
            break
        if all(abs(f - s) >= max(3, n // 20) for s in seeds):
            seeds.append(f)
    print(f"[种子] 原来跟上 {len(old)}/{n} 帧, 种子帧 {seeds}")

    # 帧目录: SAM3 要一个只放这些帧的目录(文件名按序), 关键帧的图片就是
    frames_dir = fm["frames_dir"]
    files = sorted(os.listdir(frames_dir))
    if files != keys:
        tmp = os.path.join(os.path.dirname(os.path.abspath(a.out)), "_frames")
        os.makedirs(tmp, exist_ok=True)
        for k in keys:
            dst = f"{tmp}/{k}"
            if not os.path.exists(dst):
                os.symlink(os.path.join(frames_dir, k), dst)
        frames_dir = tmp

    from sam3.model_builder import build_sam3_video_predictor
    pred = build_sam3_video_predictor(checkpoint_path=CKPT)
    sid = pred.handle_request(request=dict(type="start_session", resource_path=frames_dir,
                                           offload_video_to_cpu=True, offload_state_to_cpu=True))["session_id"]
    for f in seeds:
        pts = interior_points(old[f])
        pred.handle_request(request=dict(type="add_prompt", session_id=sid, frame_index=f, obj_id=1,
                                         points=[[x / W, y / H] for x, y in pts], point_labels=[1] * len(pts)))
    new = {}
    for r in pred.handle_stream_request(request=dict(type="propagate_in_video", session_id=sid,
                                                     propagation_direction="both", start_frame_index=seeds[0])):
        o = r["outputs"]
        ids = o["out_obj_ids"].tolist()
        if 1 in ids:
            m = o["out_binary_masks"][ids.index(1)].astype(bool)
            if m.any():
                new[r["frame_index"]] = m
    pred.handle_request(request=dict(type="close_session", session_id=sid))

    ious = [np.logical_and(old[f], new[f]).sum() / max(np.logical_or(old[f], new[f]).sum(), 1) for f in old if f in new]
    print(f"[重跟] 跟上 {len(new)}/{n} 帧; 和原来都有的 {len(ious)} 帧 IoU 中位 {np.median(ious) if ious else float('nan'):.3f}")
    fr = sorted(new)
    enc = [mu.encode(np.asfortranarray(new[f].astype(np.uint8))) for f in fr]
    meta_out = np.zeros(len(fr), dtype=[("frame", "i4"), ("oid", "i4")])
    meta_out["frame"] = [int(os.path.splitext(keys[f])[0]) for f in fr]
    meta_out["oid"] = a.oid
    np.savez_compressed(a.out, hw=np.array([H, W]), counts=np.array([e["counts"] for e in enc], dtype=object),
                        meta=meta_out, seeds=np.array(seeds), old_frames=np.array(sorted(old)))
    # 对比图: 12 帧, 原来的(红) / 重跟的(绿)
    pick = np.linspace(0, n - 1, 12).astype(int)
    tiles = []
    for f in pick:
        im = cv2.imread(os.path.join(frames_dir, keys[f]))
        for mm, c in ((old.get(f), (0, 0, 255)), (new.get(f), (0, 255, 0))):
            if mm is not None:
                cs, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, cs, -1, c, max(2, W // 300))
        cv2.putText(im, f"kf{f}", (20, 60), 0, 2, (0, 255, 255), 4)
        tiles.append(cv2.resize(im, (480, int(480 * H / W))))
    rows = [np.hstack(tiles[k:k + 4]) for k in range(0, 12, 4)]
    cv2.imwrite(os.path.splitext(a.out)[0] + ".jpg", np.vstack(rows))


if __name__ == "__main__":
    main()
