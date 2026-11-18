"""接观测实验看效果(sam3d 环境): 动捕考场片段, 三栏火柴人 真人 | B 硬切 | C 接上(bench_join.py 存的关节)。
相机在人侧面(垂直于整段平均移动方向)、离骨盆 3 m、高 0.9 m, 只跟着平滑后的骨盆平移; 地上 0.5 m 格子不动, 看脚滑看脚和格子。
腿藏掉的那段(只给骨盆, 补腿)画面上方红条 "legs hidden"。左腿蓝, 右腿红。
用法: python video_join.py 14 4 6   -> outputs/bench/join/video_<片段号>.mp4, 和 outputs/bench/join/video_join.mp4(几段接起来)
"""
import os, sys
import numpy as np, cv2, imageio_ffmpeg
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f'{HERE}/outputs/bench/join'
PW, PH, F = 480, 480, 420.0
BONES = [('pelvis', 'thigh_l'), ('thigh_l', 'calf_l'), ('calf_l', 'foot_l'), ('foot_l', 'ball_l'),
         ('pelvis', 'thigh_r'), ('thigh_r', 'calf_r'), ('calf_r', 'foot_r'), ('foot_r', 'ball_r'), ('pelvis', 'head')]
COL = lambda a, b: (220, 120, 40) if a.endswith('_l') or b.endswith('_l') else ((60, 60, 230) if a.endswith('_r') or b.endswith('_r') else (60, 60, 60))


def render(idx):
    d = np.load(f'{OUT}/clips/{idx:03d}.npz'); t, vis = d['t'], d['vis']; T = len(t)
    cols = [('real', 'real legs'), ('B', 'B: hard cut'), ('C', 'C: join')]
    Js = {m: {n[len(m) + 1:]: d[n].astype(float) for n in d.files if n.startswith(m + '_')} for m, _ in cols}
    pel = gaussian_filter1d(Js['real']['pelvis'][:, :2], 15, axis=0)
    mv = pel[-1] - pel[0]; mv = mv / max(np.linalg.norm(mv), 1e-6)
    side = np.array([-mv[1], mv[0]])                                    # 相机在移动方向的左边
    fwd3 = np.array([-side[0], -side[1], 0.0]); up = np.array([0, 0, 1.0]); right = np.cross(fwd3, up)
    lo, hi = np.floor(np.min(pel, 0) - 2), np.ceil(np.max(pel, 0) + 2)
    frames = []
    for k in range(T):
        C = np.array([*(pel[k] + 3.0 * side), 0.9])
        def proj(P):
            q = P - C; z = q @ fwd3
            return np.stack([PW / 2 + F * (q @ right) / z, PH * 0.45 - F * (q @ up) / z], -1), z
        tiles = []
        for m, lab in cols:
            im = np.full((PH, PW, 3), 245, np.uint8)
            for x in np.arange(lo[0], hi[0] + 0.01, 0.5):                 # 地面格子(世界里不动)
                for y0, y1 in ((lo[1], hi[1]),):
                    seg = np.array([[x, y0, 0], [x, y1, 0]]); uv, z = proj(np.linspace(seg[0], seg[1], 60))
                    pts = uv[z > 0.3].astype(int)
                    if len(pts) > 1: cv2.polylines(im, [pts], False, (200, 200, 200), 1)
            for y in np.arange(lo[1], hi[1] + 0.01, 0.5):
                uv, z = proj(np.linspace([lo[0], y, 0], [hi[0], y, 0], 60)); pts = uv[z > 0.3].astype(int)
                if len(pts) > 1: cv2.polylines(im, [pts], False, (200, 200, 200), 1)
            J = Js[m]
            for a, b in BONES:
                (pa, za), (pb, zb) = proj(J[a][k]), proj(J[b][k])
                if za > 0.3 and zb > 0.3:
                    cv2.line(im, tuple(pa.astype(int)), tuple(pb.astype(int)), COL(a, b), 4, cv2.LINE_AA)
            hp, hz = proj(J['head'][k])
            if hz > 0.3: cv2.circle(im, tuple(hp.astype(int)), 14, (60, 60, 60), 2, cv2.LINE_AA)
            if not vis[k]:
                cv2.rectangle(im, (0, 0), (PW, 34), (60, 60, 220), -1)
                cv2.putText(im, 'legs hidden (only pelvis given)' if m != 'real' else 'legs hidden from method', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            cv2.putText(im, lab, (10, PH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
            cv2.putText(im, f'{t[k]:.2f}s', (PW - 100, PH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1)
            tiles.append(im)
        frames.append(np.hstack(tiles))
    name = str(d['name']).replace('_stageii.npz', '').split('__')[-1]
    title = np.full((40, PW * 3, 3), 255, np.uint8); cv2.putText(title, f'clip {idx}: {name}', (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    frames = [np.vstack([title, f]) for f in frames]
    path = f'{OUT}/video_{idx:03d}.mp4'
    w = imageio_ffmpeg.write_frames(path, (PW * 3, PH + 40), fps=30, quality=8, macro_block_size=1); w.send(None)
    for f in frames:
        w.send(np.ascontiguousarray(f[:, :, ::-1]))
    w.close()
    return path, frames


def main():
    ids = [int(x) for x in sys.argv[1:]]
    allf = []
    for i in ids:
        p, fr = render(i); print('->', p); allf += fr + [fr[-1]] * 15
    w = imageio_ffmpeg.write_frames(f'{OUT}/video_join.mp4', (PW * 3, PH + 40), fps=30, quality=8, macro_block_size=1); w.send(None)
    for f in allf:
        w.send(np.ascontiguousarray(f[:, :, ::-1]))
    w.close(); print('->', f'{OUT}/video_join.mp4')


if __name__ == '__main__':
    main()
