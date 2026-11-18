"""接观测实验 看得懂版(sam3d 环境): 只放每段两个交界(腿消失 / 腿回来)前后 1.5 s, 慢放 0.5 倍。
两栏: B 硬切 | C 接上; 每栏上面侧视(彩色 = 这个做法的腿, 浅灰 = 真腿叠在一起), 下面俯视(脚走过的路线 + 踩地的脚印)。
颜色: 左腿蓝, 右腿红; 踩着地却在滑(前掌贴地、支点 > 20 cm/s、连着 2 帧以上)的那只脚画成黄色大圈。俯视图里 空心灰圈 = 真脚印, 实心彩色点 = 这个做法的脚印。
用法: python video_join2.py 6 14 93   -> outputs/bench/join/video_join2.mp4
"""
import os, sys
import numpy as np, cv2, imageio_ffmpeg
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f'{HERE}/outputs/bench/join'
sys.path.insert(0, HERE)
PW, PH, PT, F = 520, 400, 320, 420.0
BONES = [('pelvis', 'thigh_l'), ('thigh_l', 'calf_l'), ('calf_l', 'foot_l'), ('foot_l', 'ball_l'),
         ('pelvis', 'thigh_r'), ('thigh_r', 'calf_r'), ('calf_r', 'foot_r'), ('foot_r', 'ball_r'), ('pelvis', 'head')]
LC, RC, GREY = (220, 120, 40), (60, 60, 230), (205, 205, 205)


def contact_slide(J):
    from mocap_stats import analyze
    _, o = analyze(J, 30.0); c = o['contact']
    sl = np.zeros_like(c)
    floor = min(np.percentile(J['ball_l'][:, 2], 3), np.percentile(J['ball_r'][:, 2], 3))
    for i, s in enumerate('lr'):
        v = np.minimum(*[np.linalg.norm(np.gradient(J[n + s][:, :2], axis=0), axis=1) * 30 for n in ('foot_', 'ball_')])
        x = c[:, i] & (v > 0.20) & (J['ball_' + s][:, 2] < floor + 0.03)                # 踩着、前掌贴地、支点还在动
        sl[:, i] = x & (np.roll(x, 1) | np.roll(x, -1))                                   # 连着 2 帧以上才算(滚脚那一两帧不算)
    return c, sl


def clip_frames(idx):
    d = np.load(f'{OUT}/clips/{idx:03d}.npz'); t, vis = d['t'], d['vis']; T = len(t)
    Js = {m: {n[len(m) + 1:]: d[n].astype(float) for n in d.files if n.startswith(m + '_')} for m in ('real', 'B', 'C')}
    cs = {m: contact_slide(Js[m]) for m in Js}
    pel = gaussian_filter1d(Js['real']['pelvis'][:, :2], 15, axis=0)
    mv = pel[-1] - pel[0]; mv = mv / max(np.linalg.norm(mv), 1e-6); side = np.array([-mv[1], mv[0]])
    fwd3 = np.array([-side[0], -side[1], 0.0]); up = np.array([0, 0, 1.0]); right = np.cross(fwd3, up)
    a = int(np.argmin(vis)); b = a + int((~vis).sum())
    name = str(d['name']).replace('_stageii.npz', '').split('__')[-1]
    out = []
    for kb, what in ((a, 'legs DISAPPEAR'), (b, 'legs COME BACK')):
        for k in range(max(kb - 45, 0), min(kb + 45, T)):
            C = np.array([*(pel[k] + 3.0 * side), 0.9])
            proj = lambda P: (np.stack([PW / 2 + F * ((P - C) @ right) / ((P - C) @ fwd3), PH * 0.45 - F * ((P - C) @ up) / ((P - C) @ fwd3)], -1))
            top = lambda P: np.stack([PW / 2 + 230 * ((P[..., :2] - pel[k]) @ mv), PT / 2 - 230 * ((P[..., :2] - pel[k]) @ side)], -1)
            cols = []
            for m, lab in (('B', 'B: hard cut'), ('C', 'C: join (new)')):
                sv = np.full((PH, PW, 3), 246, np.uint8); tv = np.full((PT, PW, 3), 252, np.uint8)
                # 侧视: 先画真腿(浅灰), 再画这个做法的
                for JJ, thick, colf in ((Js['real'], 7, lambda a_, b_: GREY), (Js[m], 4, lambda a_, b_: LC if '_l' in a_ + b_ else (RC if '_r' in a_ + b_ else (70, 70, 70)))):
                    for a_, b_ in BONES:
                        cv2.line(sv, tuple(proj(JJ[a_][k]).astype(int)), tuple(proj(JJ[b_][k]).astype(int)), colf(a_, b_), thick, cv2.LINE_AA)
                c, sl = cs[m]
                for i, s in enumerate('lr'):
                    if sl[k, i]:
                        cv2.circle(sv, tuple(proj(Js[m]['foot_' + s][k]).astype(int)), 16, (0, 200, 255), 3)
                gy = int(PH * 0.45 + F * 0.9 / 3.0); cv2.line(sv, (0, gy), (PW, gy), (190, 190, 190), 1)
                # 俯视: 1 s 内踝走过的路线 + 踩地的脚印
                k0 = max(k - 30, 0)
                for JJ, cc, col_l, col_r, real in ((Js['real'], cs['real'][0], GREY, GREY, True), (Js[m], c, LC, RC, False)):
                    for i, (s, col) in enumerate((('l', col_l), ('r', col_r))):
                        tr = top(JJ['foot_' + s][k0:k + 1]).astype(int)
                        if len(tr) > 1: cv2.polylines(tv, [tr], False, col, 1 if real else 2, cv2.LINE_AA)
                        for kk in range(max(k - 60, 0), k + 1, 3):
                            if cc[kk, i]:
                                p = tuple(top(JJ['foot_' + s][kk]).astype(int))
                                cv2.circle(tv, p, 12, col, 2) if real else cv2.circle(tv, p, 5, col, -1)
                    if not real:
                        for i, s in enumerate('lr'):
                            p = tuple(top(JJ['foot_' + s][k]).astype(int)); cv2.circle(tv, p, 7, (40, 40, 40), 1)
                cv2.circle(tv, (PW // 2, PT // 2), 5, (80, 80, 80), -1)
                hid = not vis[k]
                cv2.rectangle(sv, (0, 0), (PW, 30), (60, 60, 220) if hid else (90, 170, 90), -1)
                cv2.putText(sv, 'legs HIDDEN - method fills them in' if hid else 'legs VISIBLE - real legs given', (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(sv, lab, (10, PH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
                cv2.putText(tv, 'top view: grey rings = real footprints, dots = method', (8, PT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
                cols.append(np.vstack([sv, tv]))
            fr = np.hstack([cols[0], np.full((PH + PT, 6, 3), 0, np.uint8), cols[1]])
            title = np.full((56, fr.shape[1], 3), 255, np.uint8)
            cv2.putText(title, f'clip {idx} {name}   |   {what} at {t[kb]:.2f}s   now {t[k]:.2f}s ({(k - kb) / 30:+.2f}s)', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
            cv2.putText(title, 'side: color = method, light grey = real legs | yellow ring = planted foot sliding | slowed 0.5x', (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
            out.append(np.vstack([title, fr]))
        out += [out[-1]] * 20
    return out


def main():
    frames = []
    for i in [int(x) for x in sys.argv[1:]]:
        frames += clip_frames(i)
    h, w = frames[0].shape[:2]
    wr = imageio_ffmpeg.write_frames(f'{OUT}/video_join2.mp4', (w, h), fps=15, quality=8, macro_block_size=1); wr.send(None)
    for f in frames:
        wr.send(np.ascontiguousarray(f[:, :, ::-1]))
    wr.close(); print('->', f'{OUT}/video_join2.mp4', len(frames), 'frames')


if __name__ == '__main__':
    main()
