"""自动判断谁的腿看得见、哪几段看不见(rohm 环境), 用 office 的流水线结果试。
每个人每个关键帧: 两膝、两踝、两脚尖投到原视频相机里, 在画面内(留 2% 边)且在相机前 = 看得见。
腿看得见 = 两踝和两脚尖都看得见(膝盖在画里、脚在画外 = 看不见: 这时流水线猜的脚不可信)。
关键帧 -> 每个视频帧(就近), 短于 MIN_RUN 秒的进出画并到前后段(人在画面边上来回晃时不来回切)。
分类: A 全程看得见 / B 全程看不见 / C 一会儿看得见一会儿看不见。
被桌椅挡住: 再看投影点落没落在这个人自己的分割(流水线 body4d/06_identity/masks, 调色板编码 一种颜色一个人; 哪种颜色是谁按头和骨盆的投影点落在哪种颜色上最多来认),
两踝两脚尖都落在(膨胀 MASK_DIL 像素的)自己的分割里才算看得见。
数据: 人 = ue/out/smplx/office_P*.npz 的关键帧关节(glTF 世界); 相机 = RoHM/ours/inputs/p4_smplx_Wy.npz(所有人共用同一套关键帧);
glTF -> 补腿的世界 W 用 to_ue.py 求出的对齐(outputs/to_ue_report.json)。
用法: python detect_legs.py   -> outputs/office_leg_visibility.json
"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAM = '/home/ubuntu/sam3d/RoHM/ours/inputs/p4_smplx_Wy.npz'
LEG = dict(knee=(4, 5), ankle=(7, 8), foot=(10, 11))
MARGIN = 0.02
MIN_RUN = 0.5
MASKS = '/home/ubuntu/sam3d/data/indoorOutput/office/r1/body4d/06_identity/masks'
FRAME_MAP = '/home/ubuntu/sam3d/data/indoorOutput/office/r1/02_da3/frame_map.json'
MASK_DIL = 20                                   # 像素(1920 宽): 投影点和分割边缘差一点也算在里面


def runs_of(mask):
    m = np.concatenate([[0], np.asarray(mask, int), [0]]); e = np.flatnonzero(np.diff(m))
    return list(zip(e[::2], e[1::2]))


def clean(vis, n_min):
    """短于 n_min 帧的段改成和前一段一样(开头的改成和后一段一样), 直到没有短段。"""
    vis = vis.copy()
    for _ in range(10):
        segs = runs_of(vis) + runs_of(~vis); segs.sort()
        short = [(a, b) for a, b in segs if b - a < n_min and not (a == 0 and b == len(vis))]
        if not short:
            break
        a, b = short[0]
        vis[a:b] = vis[a - 1] if a > 0 else vis[b]
    return vis


def load_masks():
    """{视频帧: 每像素人编号(0 = 没人)}; 调色板(VOC): 编号的第 0/1/2 位 -> R/G/B 的 128。"""
    import cv2
    fm = json.load(open(FRAME_MAP))['frames']
    out = {}
    for name, vf in fm.items():
        f = f'{MASKS}/{int(name[:-4]):08d}.png'
        if not os.path.exists(f):
            continue
        im = cv2.imread(f)[:, :, ::-1].astype(int)                              # -> RGB
        out[int(vf)] = sum(((im[..., ch] >> (7 - lv)) & 1) << (3 * lv + ch) for lv in range(8) for ch in range(3))
    return out


def main():
    import cv2
    c = np.load(CAM); H, W = c['input_hw']; fps = float(c['fps'])
    masks = load_masks() if os.path.isdir(MASKS) else {}
    ker = np.ones((2 * MASK_DIL + 1, 2 * MASK_DIL + 1), np.uint8)
    al = json.load(open(f'{HERE}/outputs/to_ue_report.json'))['align']; R, tt = np.array(al['R']), np.array(al['t'])
    NF = int(c['vf'].max()) + 1
    out = {}; dbg = {}
    print(f'{"人":<5}{"关键帧":>6}  {"时间段":<14}{"膝在画里":>8}{"踝在画里":>8}{"脚尖在画里":>9}{"被挡住":>7}{"腿看得见":>8}  类别  分割编号  看不见的段(秒)')
    for f in sorted(glob.glob('/home/ubuntu/ue/out/smplx/office_P[0-9]*.npz')):
        pid = os.path.basename(f)[8:-4]
        if not pid.isdigit():
            continue
        u = np.load(f); kv = u['key_vf']; Jg = u['key_joints'].astype(np.float64)
        Jw = (Jg - tt) @ R                                                      # glTF -> W
        pd = (Jw - c['t_W_da3']) @ c['R_W_da3']                                 # W -> DA3
        ks = [int(np.abs(c['vf'] - v).argmin()) for v in kv]                  # 相机关键帧是 p4 的, 个别帧没有: 用最近的(差 <= 几帧)
        w2c = c['w2c_da3'][ks]; K = c['K'][ks]
        pc = np.einsum('kab,kjb->kja', w2c[:, :3, :3], pd) + w2c[:, None, :3, 3]
        uv = pc[..., :2] / np.maximum(pc[..., 2:], 1e-6) * K[:, None, [0, 1], [0, 1]] + K[:, None, :2, 2]
        inside = (pc[..., 2] > 0.1) & (uv[..., 0] > MARGIN * W) & (uv[..., 0] < (1 - MARGIN) * W) & (uv[..., 1] > MARGIN * H) & (uv[..., 1] < (1 - MARGIN) * H)
        part = {n: inside[:, list(j)].all(1) for n, j in LEG.items()}
        vis_k = part['ankle'] & part['foot']
        occl = np.zeros(len(kv), bool); mid = None
        if masks:
            mh, mw = next(iter(masks.values())).shape
            px = np.round(uv * [mw / W, mh / H]).astype(int)
            votes = [masks[int(v)][y, x] for kk, v in enumerate(kv) if int(v) in masks for j in (0, 15)
                     for x, y in [px[kk, j]] if inside[kk, j] and 0 <= x < mw and 0 <= y < mh]
            votes = [v for v in votes if v > 0]
            mid = int(np.bincount(votes).argmax()) if votes else None
            if mid is not None:
                for kk, v in enumerate(kv):
                    if not vis_k[kk] or int(v) not in masks:
                        continue
                    m = cv2.dilate((masks[int(v)] == mid).astype(np.uint8), ker)
                    if not all(m[min(max(px[kk, j, 1], 0), mh - 1), min(max(px[kk, j, 0], 0), mw - 1)] for j in (7, 8, 10, 11)):
                        occl[kk] = True
        vis_k = vis_k & ~occl
        frames = np.arange(kv[0], kv[-1] + 1)
        near = np.abs(frames[:, None] - kv[None]).argmin(1)
        vis = clean(vis_k[near], int(round(MIN_RUN * fps)))
        # 每条腿分开: 这条腿的踝和脚尖都在画里且在自己的分割上 = 这条腿看得见; 膝盖看得见但脚看不见 = 半条腿
        ok = inside.copy()
        if masks and mid is not None:
            for kk, v in enumerate(kv):
                if int(v) not in masks:
                    continue
                m = cv2.dilate((masks[int(v)] == mid).astype(np.uint8), ker)
                for j in (4, 5, 7, 8, 10, 11):
                    ok[kk, j] &= bool(m[min(max(px[kk, j, 1], 0), mh - 1), min(max(px[kk, j, 0], 0), mw - 1)])
        n_min = int(round(MIN_RUN * fps))
        leg = {s: clean(ok[:, [a, f]].all(1)[near], n_min) for s, (a, f) in (('l', (7, 10)), ('r', (8, 11)))}
        knee_only = {s: (ok[:, kn] & ~ok[:, [a, f]].all(1))[near] for s, (kn, a, f) in (('l', (4, 7, 10)), ('r', (5, 8, 11)))}
        one = leg['l'] ^ leg['r']
        sec = lambda r: [(round(float(frames[a] / fps), 2), round(float((frames[b - 1] + 1) / fps), 2)) for a, b in r]
        per_leg = dict(both=float((leg['l'] & leg['r']).mean()), only_l=float((leg['l'] & ~leg['r']).mean()), only_r=float((~leg['l'] & leg['r']).mean()),
                       neither=float((~leg['l'] & ~leg['r']).mean()), one_leg_s=sec(runs_of(one)),
                       knee_only_l=float(knee_only['l'].mean()), knee_only_r=float(knee_only['r'].mean()))
        hid = [(round(float((frames[a]) / fps), 2), round(float((frames[b - 1] + 1) / fps), 2)) for a, b in runs_of(~vis)]
        cls = 'A' if vis.all() else ('B' if not vis.any() else 'C')
        out[pid] = dict(key_frames=len(kv), span_s=[round(kv[0] / fps, 2), round((kv[-1] + 1) / fps, 2)], frac_vis=float(vis.mean()), mask_id=mid,
                        knee=float(part['knee'].mean()), ankle=float(part['ankle'].mean()), foot=float(part['foot'].mean()), occluded=float(occl.mean()),
                        cls=cls, hidden_s=hid, per_leg=per_leg)
        dbg.update({f'P{pid}_kv': kv, f'P{pid}_uv': uv, f'P{pid}_ok': ok})
        print(f'P{pid:<4}{len(kv):>6}  {kv[0] / fps:5.1f}-{(kv[-1] + 1) / fps:5.1f} s {100 * part["knee"].mean():>7.0f}%{100 * part["ankle"].mean():>7.0f}%'
              f'{100 * part["foot"].mean():>8.0f}%{100 * occl.mean():>6.0f}%{100 * vis.mean():>7.0f}%   {cls}    {mid}    {hid}')
        print(f'      每条腿分开: 两条都见 {100 * per_leg["both"]:.0f}%  只见左 {100 * per_leg["only_l"]:.0f}%  只见右 {100 * per_leg["only_r"]:.0f}%  都不见 {100 * per_leg["neither"]:.0f}%'
              f'  只见一条的段 {per_leg["one_leg_s"]}  膝盖见脚不见 左 {100 * per_leg["knee_only_l"]:.0f}% 右 {100 * per_leg["knee_only_r"]:.0f}%')
    json.dump(out, open(f'{HERE}/outputs/office_leg_visibility.json', 'w'), indent=1, ensure_ascii=False)
    np.savez(f'{HERE}/outputs/office_leg_visibility_debug.npz', mask_ids=json.dumps({p: o['mask_id'] for p, o in out.items()}), **dbg)   # 核对图用: 关键帧/投影像素/每关节看不看得见
    print('->', f'{HERE}/outputs/office_leg_visibility.json')


if __name__ == '__main__':
    main()
