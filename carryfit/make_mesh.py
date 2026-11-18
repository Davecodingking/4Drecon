"""给被拿着的物体生成网格(流水线不一定生成它: 室外会把滑板车这类会动的东西排除掉; 闸口点了不要的也没有)。
从这个物体的 mask(最好是 retrack.py 重跟过的)里挑一帧: 面积大、不贴画面边、和人的 mask 重叠少; 存成 SAM 3D Objects
要的输入(frame.png + mask.png), 再调它的 run.py 生成。

sam3d 环境(要一张卡):
  python carryfit/make_mesh.py --r1 <r1> --masks carryfit/runs/<名字>/mask.npz --name <输出名>
输出: sam-3d-objects/outputs/<输出名>/<输出名>.glb; 挑的帧和打分写在 carryfit/runs/<名字>/mesh_pick.json
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from pycocotools import mask as mu

SAM3D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # 原来写死 /home/ubuntu/sam3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--frame", type=int, default=None, help="指定关键帧(不自动挑)")
    a = ap.parse_args()
    fm = json.load(open(f"{a.r1}/02_da3/frame_map.json"))
    keys = sorted(fm["frames"])
    md = np.load(a.masks, allow_pickle=True)
    H, W = [int(x) for x in md["hw"]]
    pm_dir = f"{a.r1}/body4d/02_body4d/masks"
    kidx = {int(os.path.splitext(k)[0]): i for i, k in enumerate(keys)}   # mask 的 frame = 帧文件名里的号
    rows = []
    for j, fn in enumerate(md["meta"]["frame"]):
        f = kidx.get(int(fn))
        if f is None:
            continue
        m = mu.decode({"counts": md["counts"][j], "size": [H, W]}).astype(bool)
        area = m.sum()
        if area < 200:
            continue
        ys, xs = np.nonzero(m)
        edge = min(xs.min(), ys.min(), W - 1 - xs.max(), H - 1 - ys.max())
        border = 0.0 if edge > 5 else 1.0
        occ = 0.0
        p = f"{pm_dir}/{int(f):08d}.png"
        if os.path.exists(p):
            pm = cv2.imread(p, cv2.IMREAD_UNCHANGED).reshape(H, W, -1).max(-1) > 0
            ring = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~m
            occ = (ring & pm).sum() / max(ring.sum(), 1)            # 物体四周有多少挨着人: 挨得多 = 多半被手 / 身体挡了一块
        rows.append(dict(帧=int(f), 帧号=int(fn), 行=j, 面积=int(area), 贴边=border, 挨着人=round(float(occ), 3)))
    amax = max(r["面积"] for r in rows)
    for r in rows:
        r["打分"] = round(r["面积"] / amax * (1 - 0.7 * r["贴边"]) * (1 - 0.6 * r["挨着人"]), 4)
    best = max(rows, key=lambda r: r["打分"]) if a.frame is None else next(r for r in rows if r["帧"] == a.frame)
    f, j = best["帧"], best["行"]
    m = mu.decode({"counts": md["counts"][j], "size": [H, W]}).astype(np.uint8) * 255
    img = cv2.imread(os.path.join(fm["frames_dir"], keys[f]))
    d = f"{SAM3D}/sam-3d-objects/my_images/{a.name}"
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(f"{d}/frame.png", img)
    cv2.imwrite(f"{d}/mask.png", m)
    json.dump(dict(挑的帧=best, 前五=sorted(rows, key=lambda r: -r["打分"])[:5]), open(
        os.path.join(os.path.dirname(os.path.abspath(a.masks)), "mesh_pick.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[挑帧] 关键帧 {f} ({keys[f]}), {best}")
    env = dict(os.environ, HF_HUB_OFFLINE="1", CONDA_PREFIX=os.path.dirname(os.path.dirname(sys.executable)))
    subprocess.run([sys.executable, f"{SAM3D}/sam-3d-objects/run.py", "--dirs", d, "--hq", "--bake-workers", "1"],
                   cwd=f"{SAM3D}/sam-3d-objects", env=env, check=True)
    out = f"{SAM3D}/sam-3d-objects/outputs/{a.name}/{a.name}.glb"
    assert os.path.exists(out), f"没生成出来: {out}"
    print(f"[网格] {out}")


if __name__ == "__main__":
    main()
