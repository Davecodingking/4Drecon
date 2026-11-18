"""同一段视频里几个被拿着的物体(各自用 fit.py 拟合过)画进一个预览视频: 原图 | 渲染(人 + 所有物体) | 叠加。

fpose 环境:
  python carryfit/preview_multi.py --r1 <r1> --poses runs/a/carry_poses.npz runs/b/carry_poses.npz --out runs/<名字>.mp4
"""
import argparse
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2
import imageio
import numpy as np
import pyrender
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit import frame_image  # noqa: E402

COLS = [(205, 170, 120), (230, 80, 80), (80, 200, 120), (90, 120, 230)]
PAL = [(230, 120, 40), (40, 160, 230), (120, 200, 90), (200, 90, 200), (220, 200, 60), (90, 200, 200), (160, 160, 160)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--poses", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import json
    cams = json.load(open(f"{a.r1}/02_da3/cameras.json"))["cameras"]
    fm = json.load(open(f"{a.r1}/02_da3/frame_map.json"))
    keys = sorted(cams)
    P = [np.load(p, allow_pickle=True) for p in a.poses]
    pw = np.load(f"{a.r1}/10_people_world.npz")
    faces = pw["faces"]
    W_, H_ = cv2.imread(os.path.join(fm["frames_dir"], keys[0])).shape[1::-1] if os.path.exists(os.path.join(fm["frames_dir"], keys[0])) else (1920, 1080)
    D = dict(frames_dir=fm["frames_dir"], video=fm.get("video"), vf=np.array([fm["frames"][k] for k in keys]), H=H_, W=W_)
    TW = 640
    TH = int(round(TW * H_ / W_))
    TH -= TH % 2
    r = pyrender.OffscreenRenderer(TW, TH)
    people = {}
    for f in pw.files:
        if f.endswith("_V"):
            p = f[:-2]
            people[p] = {int(v): j for j, v in enumerate(pw[f"{p}_vf"])}
    pids = sorted(people)
    fps = int(np.clip(round(fm.get("fps", 30) * len(keys) / max(1, D["vf"][-1] - D["vf"][0] + 1)), 4, 30))
    wr = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    meshes = [trimesh.Trimesh(p["box_V"], p["box_F"], process=False) for p in P]
    for i, k in enumerate(keys):
        c = cams[k]
        h, w = c["input_hw"]
        K = np.diag([TW / w, TH / h, 1.0]) @ np.array(c["intrinsic_3x3"], float)
        w2c = np.array(c["extrinsic_w2c_4x4"], float).reshape(4, 4)
        sc = pyrender.Scene(bg_color=[0.96, 0.96, 0.96, 1.0], ambient_light=[0.35, 0.35, 0.35])
        pose = np.linalg.inv(w2c) @ np.diag([1, -1, -1, 1])
        sc.add(pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=0.05, zfar=80), pose=pose)
        sc.add(pyrender.DirectionalLight(intensity=4.0), pose=pose)
        v = int(D["vf"][i])
        for j, p in enumerate(pids):
            if v in people[p]:
                pm = trimesh.Trimesh(pw[f"{p}_V"][people[p][v]], faces, process=False)
                pm.visual.vertex_colors = np.array(list(PAL[j % len(PAL)]) + [255], np.uint8)
                sc.add(pyrender.Mesh.from_trimesh(pm, smooth=True))
        tags = []
        for o, (p, m) in enumerate(zip(P, meshes)):
            row = int(np.where(p["keys"] == k)[0][0])
            bm = trimesh.Trimesh(trimesh.transform_points(m.vertices, p["local_to_world"][row]), m.faces, process=False)
            bm.visual.vertex_colors = np.array(list(COLS[o % len(COLS)]) + [255], np.uint8)
            sc.add(pyrender.Mesh.from_trimesh(bm, smooth=False))
            ph = {"拿着": "held", "空中": "in air", "放着": "resting"}.get(str(p["phase"][row]), "")
            tags.append(f"#{o}:{ph}")
        col, _ = r.render(sc)
        real = cv2.resize(frame_image(D, i, k), (TW, TH))
        rowimg = np.hstack([real, col, (0.5 * real + 0.5 * col).astype(np.uint8)])
        cv2.putText(rowimg, f"{k}  frame {v}  " + "  ".join(tags), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        wr.append_data(rowimg)
    wr.close()
    print("saved", a.out)


if __name__ == "__main__":
    main()
