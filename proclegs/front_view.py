"""正面看腿(sam3d 环境): 相机在人正前方(按骨盆朝向), 看膝盖内扣/外翻。每个时间点一列, 每个版本一行。
用法: python front_view.py --pid 4 --tags _v6 --times 3 6.2 11.5 12.4 13.2
"""
import argparse, os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np, trimesh, pyrender, cv2
from scipy.spatial.transform import Rotation as Rot
from video_legs import floor_mesh, cam_pose, UP

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 300, 420


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--times', nargs='+', type=float, required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    srcs = [('original', np.load(f'/home/ubuntu/sam3d/RoHM/ours/outputs/p{a.pid}_input_W_verts.npz'))]
    srcs += [(f'proclegs{t}', np.load(f'{HERE}/outputs/p{a.pid}_proclegs{t}_W.npz')) for t in a.tags]
    ref = srcs[-1][1]
    f = Rot.from_rotvec(ref['global_orient']).as_matrix() @ np.array([0, 0, 1.0])
    r = pyrender.OffscreenRenderer(W, H)
    rows = []
    for name, d in srcs:
        cols = []
        for t0 in a.times:
            k = int(round(t0 * 30))
            pel = ref['joints'][k, 0].copy(); pel[2] = 0
            fw = np.array([f[k, 0], f[k, 1], 0]); fw /= np.linalg.norm(fw)
            floor, grid = floor_mesh(pel[:2] - 2, pel[:2] + 2)
            s = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[.45] * 3)
            s.add(floor); s.add(grid)
            m = trimesh.Trimesh(d['vertices'][k], d['faces'], process=False); m.visual.vertex_colors = [230, 120, 200, 255]
            s.add(pyrender.Mesh.from_trimesh(m, smooth=True))
            pose = cam_pose(pel - 0.15 * UP, fw, dist=2.2, height=0.2)            # 正前方、膝盖高度附近
            s.add(pyrender.IntrinsicsCamera(fx=1.1 * W, fy=1.1 * W, cx=W / 2, cy=H / 2), pose=pose)
            s.add(pyrender.DirectionalLight(intensity=3.5), pose=pose)
            img = r.render(s)[0].copy()
            cv2.putText(img, f'{name} {t0:.1f}s', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cols.append(img)
        rows.append(np.hstack(cols))
    out = a.out or f'{HERE}/outputs/front_p{a.pid}{"".join(a.tags)}.jpg'
    cv2.imwrite(out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print('->', out)


if __name__ == '__main__':
    main()
