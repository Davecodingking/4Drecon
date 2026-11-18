"""核对"每条腿分开判断"(rohm 环境): 在原视频帧上画某人的膝/踝/脚尖投影, 实心 = 判成看得见, 空心叉 = 看不见; 再描这个人的分割边。
读 detect_legs.py 存的 outputs/office_leg_visibility_debug.npz。
用法: python check_leg_parts.py P2:5.0 P2:7.0 P2:12.0 P8:15.0   -> outputs/office_leg_parts_check.jpg
"""
import json, os, sys
import numpy as np, cv2
from detect_legs import CAM, FRAME_MAP, load_masks

HERE = os.path.dirname(os.path.abspath(__file__))
JOINTS = {4: ('knee', 'l'), 5: ('knee', 'r'), 7: ('ankle', 'l'), 8: ('ankle', 'r'), 10: ('toe', 'l'), 11: ('toe', 'r')}
COL = {'l': (220, 120, 40), 'r': (60, 60, 230)}


def main():
    d = np.load(f'{HERE}/outputs/office_leg_visibility_debug.npz'); mids = json.loads(str(d['mask_ids']))
    fm = json.load(open(FRAME_MAP)); fps = float(fm['fps']); name_of = {int(v): k for k, v in fm['frames'].items()}
    masks = load_masks(); tiles = []; hw = np.load(CAM)["input_hw"]
    for arg in sys.argv[1:]:
        pid, ts = arg.split(':'); kv, uv, ok = d[f'{pid}_kv'], d[f'{pid}_uv'], d[f'{pid}_ok']
        kk = int(np.abs(kv / fps - float(ts)).argmin()); v = int(kv[kk])
        im = cv2.imread(f"{fm['frames_dir']}/{name_of[v]}"); H, W = im.shape[:2]
        mid = mids[pid[1:]]
        if mid is not None and v in masks:
            m = cv2.resize((masks[v] == mid).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cs, -1, (0, 230, 0), 3)
        pts = []
        for j, (part, s) in JOINTS.items():
            p = uv[kk, j] * [W / hw[1], H / hw[0]]                                 # 投影是按相机的输入尺寸算的
            x, y = int(p[0]), int(p[1]); pts.append((x, y))
            if ok[kk, j]:
                cv2.circle(im, (x, y), 14, COL[s], -1)
            else:
                cv2.circle(im, (x, y), 16, COL[s], 3); cv2.line(im, (x - 11, y - 11), (x + 11, y + 11), COL[s], 3); cv2.line(im, (x - 11, y + 11), (x + 11, y - 11), COL[s], 3)
        lab = ' '.join(f"{s.upper()}-{part}:{'Y' if ok[kk, j] else 'n'}" for j, (part, s) in JOINTS.items())
        # 以腿为中心裁一块(点可能在画外: 裁剪框夹在画内)
        P = np.array(pts); cx, cy = np.clip(P.mean(0), [450, 450], [W - 450, H - 450]).astype(int)
        crop = im[cy - 450:cy + 450, cx - 450:cx + 450].copy()
        crop = cv2.resize(crop, (540, 540))
        cv2.rectangle(crop, (0, 0), (540, 54), (255, 255, 255), -1)
        cv2.putText(crop, f'{pid} {kv[kk] / fps:.2f}s', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(crop, lab, (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        tiles.append(crop)
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(f'{HERE}/outputs/office_leg_parts_check.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('->', f'{HERE}/outputs/office_leg_parts_check.jpg')


if __name__ == '__main__':
    main()
