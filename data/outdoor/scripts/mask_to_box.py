#!/usr/bin/env python
"""把 SAM3 的 mask 通过 DA3 深度反投影成三维点, 量出物体盒子.
关键检查: 点是聚成一团(深度可信) 还是拖成一条(深度错/边缘穿透)."""
import os, json, numpy as np, cv2

ROOT = "/home/jincheng/sam3d/data/outdoor/result/timesquare"
DA3  = f"{ROOT}/da3"
PICK = json.load(open(f"{ROOT}/picks/picks.json"))
cams = json.load(open(f"{DA3}/cameras.json"))["cameras"]

def backproject(mask_full, fname, conf_min=2.0):
    """mask_full: 原图分辨率 bool; 返回世界系点 (N,3)"""
    dep = np.load(f"{DA3}/depth/{fname[:-4]}_depth.npy")
    cnf = np.load(f"{DA3}/depth/{fname[:-4]}_conf.npy")
    h, w = dep.shape
    # 腐蚀半径随物体大小自适应: 小物体腐蚀太狠会整个消失
    r = int(np.clip(np.sqrt(mask_full.sum())/25, 1, 6))
    m = cv2.erode(mask_full.astype(np.uint8), np.ones((r*2+1,)*2, np.uint8))
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    m &= (cnf >= conf_min) & np.isfinite(dep) & (dep > 0)
    if m.sum() < 12: return None, m.sum()
    d = dep[m]
    med = np.median(d); mad = np.median(np.abs(d-med)) + 1e-6
    keep = np.abs(d-med) < 4*mad                       # 剔穿透到背景的像素
    v, u = np.nonzero(m); v, u, d = v[keep], u[keep], d[keep]
    K = np.array(cams[fname]["intrinsic_3x3"])
    X = (u-K[0,2])*d/K[0,0]; Y = (v-K[1,2])*d/K[1,1]
    P = np.stack([X, Y, d], 1)
    c2w = np.linalg.inv(np.array(cams[fname]["extrinsic_w2c_4x4"]).reshape(4,4))
    return (c2w[:3,:3] @ P.T).T + c2w[:3,3], m.sum()

def shape_stats(P):
    c = P.mean(0); Q = P - c
    ev, V = np.linalg.eigh(Q.T @ Q / len(Q))
    V = V[:,::-1]                                      # 主轴 大->小
    proj = Q @ V                                       # 点在主轴上的坐标
    ext = proj.max(0) - proj.min(0)                    # 真实的定向包围盒边长
    return c, ext, V

print(f"{'类别':14s} {'像素':>6s} {'点数':>7s} {'离相机':>7s} {'主轴伸展(m)':>22s} {'扁平度':>7s}")
out={}
for w, info in PICK.items():
    fn = f"{info['frame']:04d}.jpg"
    mk = cv2.imread(os.path.expanduser(f"~/sam3d/sam-3d-objects/my_images/ts_{w.replace(' ','_')}/mask.png"), 0) > 127
    P, npx = backproject(mk, fn)
    if P is None: print(f"{w:14s} {npx:6d}  点太少"); continue
    c, ext, V = shape_stats(P)
    cc = np.array(cams[fn]["camera_center_world"])
    flat = ext[2]/max(ext[0],1e-6)
    print(f"{w:14s} {npx:6d} {len(P):7d} {np.linalg.norm(c-cc):7.1f} "
          f"{ext[0]:6.2f} x{ext[1]:6.2f} x{ext[2]:6.2f}  {flat:6.3f}")
    out[w]=dict(frame=info['frame'], 点数=len(P), 距相机=round(float(np.linalg.norm(c-cc)),2),
                中心=np.round(c,3).tolist(), 主轴伸展=np.round(ext,3).tolist(), 扁平度=round(float(flat),4))
json.dump(out, open(f"{ROOT}/04_boxes_singleframe.json","w"), ensure_ascii=False, indent=1)
print(f"\n扁平度 = 最短主轴/最长主轴。接近 0 = 塌成一个面(只看到一层壳), 越大越像个立体的团。")
