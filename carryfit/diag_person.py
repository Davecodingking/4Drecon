"""查一个人(跟踪编号)每帧对应画面里的谁: 原图上画这个编号的 mask(紫)和摆进场景的网格投影(黄点), 打印 mask 面积 / 脚离地 / 远近。
  python carryfit/diag_person.py <r1> <编号> <隔几帧> <输出 jpg>
"""
import sys, json, numpy as np, cv2
from PIL import Image
R1, pid, step = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
fm = json.load(open(f"{R1}/02_da3/frame_map.json")); keys = sorted(fm["frames"]); fd = fm["frames_dir"]
wu = json.load(open(f"{R1}/02_da3/world_up.json")); up = np.array(wu["up"]); up /= np.linalg.norm(up); gh = wu["ground_h"]
cams = json.load(open(f"{R1}/02_da3/cameras.json"))["cameras"]
pw = np.load(f"{R1}/10_people_world.npz"); P = np.load("carryfit/parts.npz")
V, vf = pw[f"p{pid}_V"], [int(x) for x in pw[f"p{pid}_vf"]]
tiles = []
for i in range(0, len(keys), step):
    k = keys[i]; v = int(fm["frames"][k])
    m = np.array(Image.open(f"{R1}/body4d/02_body4d/masks/{i:08d}.png")) == pid
    im = cv2.imread(f"{fd}/{k}"); H, W = im.shape[:2]
    line = f"kf{i:3d}"
    if m.any():
        ys, xs = np.nonzero(m); line += f" mask 面积{m.mean()*100:5.1f}% x{xs.min():4d}-{xs.max():4d} y{ys.min():4d}-{ys.max():4d}"
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cs, -1, (255, 0, 255), 6)
    else:
        line += " mask 无" + " " * 34
    if v in vf:
        X = V[vf.index(v)]; foot = (X[np.r_[P["foot_L"], P["foot_R"]]] @ up).min() - gh
        c = cams[k]; h, w = c["input_hw"]; K = np.diag([W / w, H / h, 1]) @ np.array(c["intrinsic_3x3"]); w2c = np.array(c["extrinsic_w2c_4x4"]).reshape(4, 4)
        Xc = X @ w2c[:3, :3].T + w2c[:3, 3]; uv = Xc @ K.T; uv = uv[:, :2] / uv[:, 2:]
        for p in uv[::40]:
            if 0 <= p[0] < W and 0 <= p[1] < H: cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)
        line += f" | 网格 脚离地 {foot:+.2f}m 远近 {Xc[:,2].mean():.1f}m"
    else:
        line += " | 网格 无"
    print(line)
    cv2.putText(im, f"kf{i}", (20, 60), 0, 2, (0, 0, 255), 4)
    tiles.append(cv2.resize(im, (480, int(480 * H / W))))
while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[4], np.vstack([np.hstack(tiles[j:j + 4]) for j in range(0, len(tiles), 4)]))
