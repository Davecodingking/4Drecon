"""按词单独跟踪一个东西: 流水线的词表里没有它(杀破狼的小刀没被 VLM 列出来, 有几帧被"警棍"吞了), 又不想重跑整轮的跟踪。
不动 <r1>/03_masks, 结果写到 --out(和 03_masks/<词>.npz 同格式, fit_attach.py 的 --masks 直接能读)。

按镜头分别跑: SAM3 的文本提示在一段的第 0 帧给, 往后跟; 跨了硬切它会把编号接到别的东西上。切点读 <r1>/03_split_cuts.json
(没有这个文件 = 整段一个镜头)。每个镜头里的实例号 = 镜头号 * 1000 + SAM3 给的号, 所以跨镜头是不同的实例号 ——
是不是同一个物体由人看联络图定, 然后 fit_attach.py --oid a,b,c 合起来。

sam3 环境, 一张卡:
  CF_ENV=sam3 carryfit/run.sh carryfit/inhand/track_word.py --r1 <r1> --word knife --out carryfit/runs/<名字>/knife.npz
输出: <out>.npz, <out>.jpg(每个实例面积最大的那帧, 绿框 + 实例号 + 跟上几帧), 终端列出每个实例的帧范围。
"""
import argparse
import json
import os
import shutil

import cv2
import numpy as np
from pycocotools import mask as mu

CKPT = os.path.expanduser("~/sam3d/sam3/checkpoints/sam3.pt")


def shots_of(r1, keys, fps):
    """每个关键帧在第几个镜头。帧号 = 帧文件名里的号。"""
    fn = np.array([int(os.path.splitext(k)[0]) for k in keys])
    p = f"{r1}/03_split_cuts.json"
    cuts = json.load(open(p)).get("切点", []) if os.path.exists(p) else []
    cf = np.array([round(c * fps) for c in cuts])
    return np.array([int((cf <= f).sum()) for f in fn]), fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--word", required=True, help="文本提示; 几个说法用逗号隔开(都跑, 实例号再错开)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_px", type=int, default=150, help="比这还小的 mask 不要")
    a = ap.parse_args()
    out = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fm = json.load(open(f"{a.r1}/02_da3/frame_map.json"))
    keys = sorted(fm["frames"])
    shot, fn = shots_of(a.r1, keys, float(fm.get("fps", 30.0)))
    from sam3.model_builder import build_sam3_video_predictor
    pred = build_sam3_video_predictor(checkpoint_path=CKPT)
    recs, rles, best = [], [], {}
    H = W = None
    for wi, word in enumerate([w.strip() for w in a.word.split(",") if w.strip()]):
        for s in sorted(set(shot.tolist())):
            idx = np.where(shot == s)[0]
            if len(idx) < 2:
                continue
            tmp = f"{out}_frames_{s}"
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            for i in idx:
                os.symlink(os.path.join(fm["frames_dir"], keys[i]), f"{tmp}/{keys[i]}")
            sid = pred.handle_request(request=dict(type="start_session", resource_path=tmp, offload_video_to_cpu=True,
                                                   offload_state_to_cpu=True))["session_id"]
            try:
                pred.handle_request(request=dict(type="add_prompt", session_id=sid, frame_index=0, text=word))
                for r in pred.handle_stream_request(request=dict(type="propagate_in_video", session_id=sid)):
                    fi, o = int(r["frame_index"]), r["outputs"]
                    for k, oid in enumerate(o["out_obj_ids"].tolist()):
                        m = o["out_binary_masks"][k].astype(bool)
                        if m.sum() < a.min_px:
                            continue
                        H, W = m.shape
                        g = wi * 100000 + s * 1000 + int(oid)
                        recs.append((g, int(fn[idx[fi]]), int(m.sum())))
                        rles.append(mu.encode(np.asfortranarray(m.astype(np.uint8)))["counts"])
                        if g not in best or m.sum() > best[g][0]:
                            best[g] = (int(m.sum()), int(idx[fi]), m.copy())
            finally:
                pred.handle_request(request=dict(type="close_session", session_id=sid))
                shutil.rmtree(tmp, ignore_errors=True)
            print(f"[{word}] 镜头 {s}: 关键帧 {idx[0]}-{idx[-1]}, 实例 {sorted({g for g, _, _ in recs if g // 1000 % 100 == s and g // 100000 == wi})}", flush=True)
    if not recs:
        raise SystemExit(f"[失败] 整段没有跟上任何 '{a.word}'")
    meta = np.array(recs, dtype=[("oid", "i4"), ("frame", "i4"), ("area", "i8")])
    np.savez_compressed(out + ".npz", meta=meta, counts=np.array(rles, dtype=object), hw=np.array([H, W]), word=a.word)
    tiles = []
    for g in sorted(best):
        area, ki, m = best[g]
        fr = meta["frame"][meta["oid"] == g]
        im = cv2.imread(os.path.join(fm["frames_dir"], keys[ki]))
        im = cv2.resize(im, (W, H))
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, (0, 255, 0), max(2, W // 400))
        im = cv2.resize(im, (480, int(480 * H / W)))
        cv2.putText(im, f"#{g}  {len(fr)}f  {fr.min()}-{fr.max()}", (6, 22), 0, 0.6, (0, 255, 255), 2)
        tiles.append(im)
        print(f"  实例 {g}: {len(fr)} 帧, 帧号 {fr.min()}-{fr.max()}, 最大面积 {area} px", flush=True)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out + ".jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print(f"[done] {out}.npz  {out}.jpg")


if __name__ == "__main__":
    main()
