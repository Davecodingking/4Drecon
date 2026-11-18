"""看考场某一段每一步落在哪、被哪条规则推过(rohm 环境)。
落点按"落地那一刻骨盆的朝向"分解: 左右(>0 = 这只脚该在的那一侧, 标准 = body.lat)、前后(>0 = 骨盆前面)。
列: 落地时刻 脚 原因 | 标准位置(看后面路线) -> 对称点规则后 -> 够得着限制后 -> 最终 | 真人这只脚在那一刻的位置
用法: python diag_steps.py 131 [--t0 2 --t1 7]
"""
import argparse, contextlib, io, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clip', type=int, help='考场片段号; -1 = office p4')
    ap.add_argument('--tag', default='v22chk', help='用哪次考场的片段(只取真人关节)')
    ap.add_argument('--t0', type=float, default=0); ap.add_argument('--t1', type=float, default=1e9)
    a = ap.parse_args()
    import bench
    bench.init(); G = bench.G; impl = G['impl']
    body = G['body']
    if a.clip < 0:
        d = dict(np.load('/home/ubuntu/sam3d/RoHM/ours/inputs/p4_rohm_input.npz')); d['name'] = 'office p4'
        s = d['smplx_world'].astype(np.float64); t = d['time_s']
        J = None
    else:
        d = np.load(f'{HERE}/outputs/bench/{a.tag}/clips/{a.clip:03d}.npz')
        J = {n[5:]: d[n].astype(np.float64) for n in d.files if n.startswith('real_')}
        t = d['t']
        s = bench.make_input(J, body, G['betas'], False, 0)
    pel = s[:, 3:6] + body.J[0]
    f = Rot.from_rotvec(s[:, 0:3]).as_matrix() @ np.array([0, 0, 1.0])
    yaw = gaussian_filter1d(np.unwrap(np.arctan2(f[:, 1], f[:, 0])), impl.PELVIS_YAW_SIGMA * impl.FPS)
    pxy = gaussian_filter1d(pel[:, :2], 0.25 * impl.FPS, axis=0)
    with contextlib.redirect_stdout(io.StringIO()):
        fxy, fyaw, contact, swings, speed = impl.plan(t, pxy, yaw, body)
    print(f'{d["name"]}  body.lat {body.lat:.3f}')
    print(f'{"落地 s":>6} 脚 {"原因":<10}{"速度":>5}  {"左右: 标准 -> 对称 -> 限制 -> 最终":<34}{"前后: 标准 -> 对称 -> 限制 -> 最终":<34}{"真人 左右/前后":>14}')
    for sw in swings:
        db = sw.get('dbg')
        if db is None:
            continue
        kl = db['kl']
        if not a.t0 <= t[kl] <= a.t1:
            continue
        i = sw['i']; c, sn = np.cos(yaw[kl]), np.sin(yaw[kl])
        side = np.array([-sn, c]) * impl.SIDE[i]; fw = np.array([c, sn])
        pts = [db['B0'], db['Bmir'], db['Bclamp'], sw['B']]
        lat = [(p - pxy[kl]) @ side for p in pts]; fwd = [(p - pxy[kl]) @ fw for p in pts]
        nm = 'foot_l' if i == 0 else 'foot_r'
        rl, rf = ((J[nm][kl, :2] - pxy[kl]) @ side, (J[nm][kl, :2] - pxy[kl]) @ fw) if J is not None else (np.nan, np.nan)
        ang = '' if db['u'] is None else f'{np.degrees(np.arctan2(db["u"][1], db["u"][0]) - yaw[kl]) % 360:5.0f}°'
        print(f'{t[kl]:6.2f} {"LR"[i]}  {sw["why"][:10]:<10}{sw["v"]:5.2f}  ' + ' '.join(f'{x:+.2f}' for x in lat) + '    '
              + ' '.join(f'{x:+.2f}' for x in fwd) + f'   {rl:+.2f}/{rf:+.2f}  u-朝向{ang}')


if __name__ == '__main__':
    main()
