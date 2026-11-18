"""调试图: 原图上叠 物体 mask(蓝)、拟合的物体轮廓(绿, 凸包)、碰着的手 / 脚(黄点, 标人号), 外加物体中心离相机的远近和朝向。
  python carryfit/debug_overlay.py --r1 <r1> --run carryfit/runs/<名字> --frames 56,86,97 [--masks <npz>] --out x.jpg
"""
import argparse, json
import cv2, numpy as np, trimesh
from pycocotools import mask as mu

ap = argparse.ArgumentParser()
ap.add_argument("--r1", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--frames", required=True); ap.add_argument("--masks", default=None); ap.add_argument("--out", required=True)
a = ap.parse_args()
P = np.load(f"{a.run}/carry_poses.npz", allow_pickle=True)
md = np.load(a.masks or f"{a.run}/mask.npz", allow_pickle=True)
cams = json.load(open(f"{a.r1}/02_da3/cameras.json"))["cameras"]; fm = json.load(open(f"{a.r1}/02_da3/frame_map.json"))
keys = sorted(cams); H, W = [int(x) for x in md["hw"]]
pw = np.load(f"{a.r1}/10_people_world.npz"); parts = [str(x) for x in P["parts"]]
import os
PT = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "parts.npz"))
tiles = []
for f in [int(x) for x in a.frames.split(",")]:
    k = keys[f]; c = cams[k]; h, w = c["input_hw"]
    K = np.diag([W / w, H / h, 1]) @ np.array(c["intrinsic_3x3"]); w2c = np.array(c["extrinsic_w2c_4x4"]).reshape(4, 4)
    im = cv2.imread(f"{fm['frames_dir']}/{k}")
    j = np.where(md["meta"]["frame"] == f)[0]
    if len(j):
        m = mu.decode({"counts": md["counts"][j[0]], "size": [H, W]})
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cs, -1, (255, 0, 0), 4)
    V = trimesh.transform_points(P["box_V"], P["local_to_world"][f]); Xc = V @ w2c[:3, :3].T + w2c[:3, 3]
    uv = Xc @ K.T; uv = uv[:, :2] / uv[:, 2:]
    cv2.polylines(im, [cv2.convexHull(uv.astype(np.int32))], True, (0, 255, 0), 4)
    vf = int(fm["frames"][k])
    for pi in np.where(P["contact"][f])[0]:
        p, kind = parts[pi].split(":")
        vv = [int(x) for x in pw[p + "_vf"]]
        if vf not in vv: continue
        X = pw[p + "_V"][vv.index(vf)][PT[kind]].mean(0); x = K @ (w2c[:3, :3] @ X + w2c[:3, 3]); x = x[:2] / x[2]
        cv2.circle(im, (int(x[0]), int(x[1])), 14, (0, 255, 255), -1); cv2.putText(im, p, (int(x[0]) + 15, int(x[1])), 0, 1.2, (0, 255, 255), 3)
    cv2.putText(im, f"kf{f} z={Xc[:, 2].mean():.2f}m IoU={P['iou'][f]:.2f} {P['phase'][f]}", (20, 60), 0, 1.6, (0, 0, 255), 4)
    tiles.append(cv2.resize(im, (640, int(640 * H / W))))
while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(a.out, np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
