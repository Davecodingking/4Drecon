"""补腿结果看效果(sam3d 环境, 有 pyrender):
  1) 对比视频: 原版 | 补腿各版本, 并排。相机在人的侧面、离骨盆 3 m, 朝向全程不变, 只跟着(平滑后的)骨盆平移,
     所以人一直在画面中间; 地上 0.5 m 格子跟着世界不动, 脚滑不滑看脚和格子的相对位置。
  2) 俯视脚印图: 骨盆路线 + 每一步落脚的位置和脚尖朝向。
  3) 连拍图(给我自己看动作用): 几个时间段, 每 0.2 s 一张。
用法: python video_legs.py --pid 4 --tags _v3 [_v4 ...]
"""
import argparse, os, subprocess, tempfile, shutil
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np, trimesh, pyrender, cv2, imageio_ffmpeg
from scipy.ndimage import gaussian_filter1d
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
UP = np.array([0, 0, 1.0])
W, H = 560, 480


def floor_mesh(lo, hi):
    lines = []
    for x in np.arange(np.floor(lo[0] * 2) / 2, hi[0] + 0.01, 0.5):
        lines.append(trimesh.creation.cylinder(radius=0.005, segment=[[x, lo[1], 0.002], [x, hi[1], 0.002]]))
    for y in np.arange(np.floor(lo[1] * 2) / 2, hi[1] + 0.01, 0.5):
        lines.append(trimesh.creation.cylinder(radius=0.005, segment=[[lo[0], y, 0.002], [hi[0], y, 0.002]]))
    grid = trimesh.util.concatenate(lines); grid.visual.vertex_colors = [90, 90, 90, 255]
    q = np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0], [hi[0], hi[1], 0], [lo[0], hi[1], 0]])
    floor = trimesh.Trimesh(q, [[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], process=False)
    floor.visual.vertex_colors = [235, 235, 228, 255]
    return pyrender.Mesh.from_trimesh(floor, smooth=False), pyrender.Mesh.from_trimesh(grid, smooth=False)


def cam_pose(target, side, dist=3.0, height=0.9):
    eye = target + dist * side + height * UP
    z = (target + 0.55 * UP) - eye; z /= np.linalg.norm(z)
    x = np.cross(-UP, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w @ np.diag([1, -1, -1, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--rohm', action='store_true', help='加一栏 RoHM A 套的原始输出(同一个 W, 30 fps)')
    a = ap.parse_args()
    srcs = [('original', np.load(f'/home/ubuntu/sam3d/RoHM/ours/outputs/p{a.pid}_input_W_verts.npz'))]
    if a.rohm:
        srcs.append(('RoHM A', np.load(f'/home/ubuntu/sam3d/RoHM/ours/outputs/p{a.pid}_rohm_W.npz')))
    srcs += [(f'proclegs{t}', np.load(f'{HERE}/outputs/p{a.pid}_proclegs{t}_W.npz')) for t in a.tags]
    n = min(len(d['vertices']) for _, d in srcs)
    ref = srcs[-1][1]
    pel = ref['joints'][:n, 0].copy(); pel[:, 2] = 0
    track = gaussian_filter1d(pel, 15, axis=0)
    P = pel - pel.mean(0)
    walk = np.linalg.svd(P[:, :2], full_matrices=False)[2][0]
    side = np.array([-walk[1], walk[0], 0])
    lo, hi = pel[:, :2].min(0) - 3, pel[:, :2].max(0) + 3
    floor, grid = floor_mesh(lo, hi)
    r = pyrender.OffscreenRenderer(W, H)
    tmp = tempfile.mkdtemp(); frames = []
    for i in range(n):
        pose = cam_pose(track[i], side)
        cols = []
        for (name, d) in srcs:
            s = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[.45] * 3)
            s.add(floor); s.add(grid)
            m = trimesh.Trimesh(d['vertices'][i], d['faces'], process=False)
            m.visual.vertex_colors = [230, 120, 200, 255]
            s.add(pyrender.Mesh.from_trimesh(m, smooth=True))
            s.add(pyrender.IntrinsicsCamera(fx=0.9 * W, fy=0.9 * W, cx=W / 2, cy=H / 2), pose=pose)
            s.add(pyrender.DirectionalLight(intensity=3.5), pose=pose)
            img = r.render(s)[0].copy()
            if name.startswith('proclegs'):                                    # 绿点 = 这只脚踩地(左, 右)
                for j, c in enumerate(d['contact'][i]):
                    cv2.circle(img, (W - 40 + 18 * j, 20), 6, (40, 160, 40) if c else (200, 200, 200), -1)
            cv2.putText(img, f'{name}  {i / 30:.1f}s', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cols.append(img)
        frame = np.hstack(cols); frames.append(frame)
        cv2.imwrite(f'{tmp}/{i:05d}.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    tag = ''.join(a.tags)
    out = f'{HERE}/outputs/compare_p{a.pid}' + ('_rohm' if a.rohm else '') + f'{tag}_30fps.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', '30', '-i', f'{tmp}/%05d.jpg',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22', out], check=True)
    shutil.rmtree(tmp)
    print(n, '帧 @30fps ->', out)

    # 连拍: 只拿最后一个版本那一栏, 下半身
    for t0 in (1.0, 9.0, 14.5):
        ks = [int((t0 + 0.2 * j) * 30) for j in range(12)]
        strip = np.hstack([frames[k][H // 3:, -W:][:, W // 5: -W // 5] for k in ks if k < n])
        cv2.imwrite(f'{HERE}/outputs/strip_p{a.pid}{a.tags[-1]}_{t0:.1f}s.jpg', cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    # 俯视脚印
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(pel[:, 0], pel[:, 1], color='0.6', lw=1, label='pelvis')
    st = ref['steps']
    from matplotlib.patches import Polygon
    for j, (foot, t0, t1, ax_, ay, bx, by, yb, v) in enumerate(st):
        c, s_ = np.cos(yb), np.sin(yb)
        outline = np.array([[-0.08, -0.045], [0.17, -0.045], [0.17, 0.045], [-0.08, 0.045]])
        pts = outline @ np.array([[c, s_], [-s_, c]]) + [bx, by]
        ax.add_patch(Polygon(pts, closed=True, fc=('tab:blue' if foot == 0 else 'tab:red'), alpha=0.35, ec='k', lw=0.5))
        ax.text(bx, by, str(j), fontsize=7, ha='center', va='center')
    gor = ref['global_orient']
    from scipy.spatial.transform import Rotation as Rot
    f = Rot.from_rotvec(gor).as_matrix() @ np.array([0, 0, 1.0])
    for k in range(0, n, 30):
        ax.arrow(pel[k, 0], pel[k, 1], 0.25 * f[k, 0], 0.25 * f[k, 1], head_width=0.05, color='k')
        ax.text(pel[k, 0] + 0.05, pel[k, 1] + 0.05, f'{k / 30:.0f}s', fontsize=7, color='0.3')
    ax.set_aspect('equal'); ax.set_title('footprints (blue L, red R; number = step order; arrow = facing every 1 s)')
    fig.savefig(f'{HERE}/outputs/footprints_p{a.pid}{a.tags[-1]}.png', dpi=110, bbox_inches='tight')
    print('脚印图、连拍图 ->', f'{HERE}/outputs/')


if __name__ == '__main__':
    main()
