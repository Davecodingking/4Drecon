#!/usr/bin/env python
"""用站直的人定"天"(2026-11-11, joker 起头): 楼梯场景里地面拟合会拟到楼梯斜面上, 相机的上又跟着仰角歪 ——
站着的人身体主轴平均是竖直的, 是这类场景里最可靠的重力线索。
做法: 10_people_world.npz 每人每帧的网格 -> 主轴(够细长才算站着: 第一奇异值 >= STAND_RATIO x 第二) -> 朝上的那头 ->
所有站着的帧取中位方向(先粗后精, 只留和中位差 OUTLIER_DEG 以内的再取一次)。人在跳舞 / 弯腰的帧被细长判据和离群判据挡掉。
--write: 改写 world_up.json 的 up(原件 world_up_before_people.json), ground_h 按"相机离地中位不变"重算; 默认只出报告。
"""
import argparse, json, os, sys
import numpy as np

STAND_RATIO = 3.0
OUTLIER_DEG = 12.0
MIN_FRAMES = 20

ap = argparse.ArgumentParser()
ap.add_argument("--r1", required=True)
ap.add_argument("--shots", default="", help="只用这些机位(1 起, 逗号分隔); 空 = 全部")
ap.add_argument("--write", action="store_true")
a = ap.parse_args()
D = f"{a.r1}/02_da3"
W = json.load(open(f"{D}/world_up.json")); up0 = np.array(W["up"], float); up0 /= np.linalg.norm(up0)
fm = json.load(open(f"{D}/frame_map.json")); fps = float(fm.get("fps") or 25)
cuts = json.load(open(f"{D}/shot_cuts.json"))["cuts"] if os.path.exists(f"{D}/shot_cuts.json") else []
use = {int(x) for x in a.shots.split(",") if x.strip()}
z = np.load(f"{a.r1}/10_people_world.npz", allow_pickle=True)
A, by_shot = [], {}
for k in z.files:
    if not k.endswith("_V"): continue
    for V, f in zip(z[k], z[k[:-2] + "_vf"]):
        s = int(np.searchsorted(cuts, int(f) / fps + 1e-3, side="right")) + 1
        if use and s not in use: continue
        X = V[::20] - V[::20].mean(0); _, S_, Vt = np.linalg.svd(X, full_matrices=False)
        if S_[0] < STAND_RATIO * S_[1]: continue
        ax = Vt[0] * np.sign(Vt[0] @ up0); A.append(ax); by_shot.setdefault(s, []).append(ax)
A = np.array(A)
if len(A) < MIN_FRAMES:
    raise SystemExit(f"[失败] 站着的帧只有 {len(A)} 个(< {MIN_FRAMES}), 定不了天")
g = np.median(A, 0); g /= np.linalg.norm(g)
for _ in range(3):
    keep = np.degrees(np.arccos(np.clip(A @ g, -1, 1))) < OUTLIER_DEG
    g = np.median(A[keep], 0); g /= np.linalg.norm(g)
ang = float(np.degrees(np.arccos(np.clip(g @ up0, -1, 1))))
rep = dict(站着的帧=len(A), 用上的帧=int(keep.sum()), 人的竖直和现在的天差_度=round(ang, 2),
           各机位={s: dict(帧=len(v), 和人的竖直差_度_中位=round(float(np.median(np.degrees(np.arccos(np.clip(np.array(v) @ g, -1, 1))))), 1))
                  for s, v in sorted(by_shot.items())})
print(f"[人定天] {json.dumps(rep, ensure_ascii=False)}")
if a.write:
    bak = f"{D}/world_up_before_people.json"
    if not os.path.exists(bak): json.dump(W, open(bak, "w"), ensure_ascii=False, indent=1)
    cams = json.load(open(f"{D}/cameras.json"))["cameras"]
    Cs = np.array([np.linalg.inv(np.array(c["extrinsic_w2c_4x4"], float))[:3, 3] for c in cams.values()])
    cam_h = float(np.median(Cs @ up0) - W["ground_h"])
    W.update(up=g.tolist(), ground_h=float(np.median(Cs @ g) - cam_h), 相机离地=round(cam_h, 3),
             来源=f"站直的人的身体主轴(up_from_people.py), 原来的: {W.get('来源')}", 人定天=rep)
    json.dump(W, open(f"{D}/world_up.json", "w"), ensure_ascii=False, indent=1)
    print(f"[人定天] 写进 {D}/world_up.json(原件 {bak}), 天转了 {ang:.2f}°, 相机离地照旧 {cam_h:.2f} m")
