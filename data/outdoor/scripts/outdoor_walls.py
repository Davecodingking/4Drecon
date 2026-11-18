#!/usr/bin/env python
"""室外的墙(OUTDOOR_SHELL=on, spl 巷子起头, 2026-11-11): 只在真拍到墙面的地方盖墙, 灰色半透明, 和室内房间壳同一个样子。

★ 为什么不直接用室内的房间壳(room_shell.py): 它圈的是"地面以上有点的区域"的外轮廓, 室内这个轮廓就是墙;
  巷子里相机一侧的墙几乎没拍到, 轮廓那一边沿着打斗的人和杂物走 —— spl 上相机全落在轮廓外, 盖出来的墙一半是假的,
  而且沿线有点, 按"沿线有没有点云"也筛不掉。
做法: 深度图逐像素法向(terrain.frame_points), 法向水平的点 = 竖直的面; 离地 WALL_LO~WALL_HI 之间、相机轨迹 RADIUS 米以内的
投到地面上 -> 5 cm 栅格 -> 直线检测(HoughLinesP)出墙线 -> 共线的并起来 -> 每段盖一个薄盒子(高 = 那段墙面点的高处)。
★ 人的像素先去掉(03_masks 里表示人的词的掩码, 外扩 PEOPLE_DILATE 像素): 打斗戏画面大半是人, 人的身体也是竖直的面,
  不去掉的话墙线顺着人站的地方检出来 —— spl 第一版墙盖到了人身上, 9 个镜头里人几乎每帧都"穿墙" 13~42 cm。

产物(和 room_shell.py 同格式, 组装那步按 ROOM_SHELL 并进来, ROOM_SHELL_FLOOR=0 只要墙):
  <r1>/09_房间壳.glb        wall_0..N, 已是最终 glTF 系(export_T(UP), 和 assemble 转正同一份)
  <r1>/09_房间壳_墙线.json   每段 {a, b, z0, h, thick}, 在 (e1, e2, 离地高) 系 + 用到的 e1/e2/up
  <r1>/09_房间壳_俯视.jpg    灰 = 竖直面的点, 红 = 墙, 蓝 = 相机
"""
import argparse, json, os, sys
import numpy as np, cv2, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terrain import iter_frames, basis
from world_frame import export_T
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "pipeline"))
from _common import PEOPLE_WORDS
PEOPLE_DILATE = 25     # 人的掩码外扩(原图像素): 掩码边缘常漏掉衣角、甩出去的手

RES = 0.05
VERT_COS = 0.25        # |法向·天| 小于这个 = 竖直的面
WALL_LO, WALL_HI = 0.3, 4.0   # 离地这个高度区间的竖直面才算墙(躲开地面边缘和远处楼顶)
RADIUS = 15.0          # 相机轨迹水平这么远以内
MIN_HITS = 4           # 一格至少这么多点
MIN_LEN = 2.0          # 墙线至少多长(m)。1.5 时 spl 上人 / 杂物的侧面能连出几小段
MERGE_DEG, MERGE_OFF, MERGE_GAP = 8.0, 0.6, 1.0    # 共线合并: 夹角 / 横向偏移 / 首尾间隙。偏移 0.25 时 spl 同一面墙检出 3~4 条平行线(各帧深度差约半米)
THICK, ALPHA = 0.10, 140
SUPPORT = 0.6          # 墙段沿线至少这么大比例的长度上真有墙面点
MANH_DEG = 15.0        # 和主方向(或它的垂直方向)差这么多以上的墙段不盖: singin 橱窗里的斜面会被检成交叉的斜墙

from facade_fit import fit_facade, facade_explains_walls

ap = argparse.ArgumentParser()
ap.add_argument("--r1", required=True)
ap.add_argument("--mode", choices=("auto", "walls", "facade"), default="auto")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--shot", type=int, default=0, help="只用第几个镜头(1 起)的帧; 0 = 全部")
ap.add_argument("--out-dir", default="", help="产物写到哪, 默认 r1")
a = ap.parse_args()
OUT = a.out_dir or a.r1; os.makedirs(OUT, exist_ok=True)
da3 = f"{a.r1}/02_da3"
W = json.load(open(f"{da3}/world_up.json")); up = np.array(W["up"], float); up /= np.linalg.norm(up); gh = float(W["ground_h"])
e1, e2 = basis(up)

_fm = json.load(open(f"{da3}/frame_map.json")); _fps = float(_fm.get("fps") or 25.0); _vf = _fm["frames"]
_cuts = json.load(open(f"{da3}/shot_cuts.json"))["cuts"] if os.path.exists(f"{da3}/shot_cuts.json") else []
# 人的掩码: {原视频帧号: [(RLE, (h, w))]}
from pycocotools import mask as mask_util
_PM = {}
for _w in PEOPLE_WORDS:
    _p = f"{a.r1}/03_masks/{_w.replace(' ', '_')}.npz"
    if not os.path.exists(_p): continue
    _z = np.load(_p, allow_pickle=True); _hw = [int(x) for x in _z["hw"]]
    for _r, _c in zip(_z["meta"], _z["counts"]):
        _PM.setdefault(int(_r["frame"]), []).append((_c, _hw))
# ★ 被人拿着的东西(闸 2 标了"拿着"的实例)也去掉: 它跟着人走, 竖直的面会沿人的路线连成一条假墙(singin 的伞)
_ap = f"{a.r1}/05_review_applied.json"
_held = json.load(open(_ap)).get("拿着的", []) if os.path.exists(_ap) else []
for _h in _held:
    _p = f"{a.r1}/03_masks/{_h['词'].replace(' ', '_')}.npz"
    if not os.path.exists(_p): continue
    _z = np.load(_p, allow_pickle=True); _hw = [int(x) for x in _z["hw"]]
    for _r, _c in zip(_z["meta"], _z["counts"]):
        if int(_r["oid"]) == int(_h["实例"]):
            _PM.setdefault(int(_r["frame"]), []).append((_c, _hw))
if _held:
    print(f"[室外墙] 手持物的像素也去掉: {[h['生成名'] for h in _held]}")
def people_mask(k):
    rows = _PM.get(int(_vf[k]))
    if not rows: return None
    m = np.zeros(rows[0][1], bool)
    for c, hw in rows: m |= mask_util.decode({"counts": c, "size": hw}).astype(bool)
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * PEOPLE_DILATE + 1,) * 2, np.uint8)).astype(bool)
from terrain import frame_points as _fp
_cams = json.load(open(f"{da3}/cameras.json"))["cameras"]
_bad = set(json.load(open(f"{da3}/bad_frames.json")).get("坏帧", [])) if os.path.exists(f"{da3}/bad_frames.json") else set()
_keys = [k for k in sorted(_cams) if k not in _bad and os.path.exists(f"{da3}/depth/{k[:-4]}_depth.npy")]
if a.shot:
    _keys = [k for k in _keys if int(np.searchsorted(_cuts, _vf[k] / _fps + 1e-3, side="right")) + 1 == a.shot]
_keys = [_keys[i] for i in np.unique(np.linspace(0, len(_keys) - 1, min(a.frames if not a.shot else 400, len(_keys))).astype(int))]
U, Hh, CS, _npm = [], [], [], 0
frame_ids = []
for k in _keys:
    _m = people_mask(k); _npm += _m is not None
    P, N, C = _fp(da3, k, _cams[k], skip=_m)
    CS.append(C)
    h = P @ up - gh
    s = (np.abs(N @ up) < VERT_COS) & (h > WALL_LO) & (h < WALL_HI)
    if a.mode != "walls":
        frame_ids.append(np.full(np.count_nonzero(s), len(CS)-1, dtype=np.int32))
    U.append(np.c_[P[s] @ e1, P[s] @ e2]); Hh.append(h[s])
U = np.concatenate(U); Hh = np.concatenate(Hh); C2 = np.array([[c @ e1, c @ e2] for c in CS])
d = np.concatenate([np.min(np.linalg.norm(U[i:i + 200_000, None] - C2[None, ::max(1, len(C2) // 60)], axis=2), axis=1)
                    for i in range(0, len(U), 200_000)])
U, Hh = U[d < RADIUS], Hh[d < RADIUS]
lo = np.percentile(U, 0.5, 0) - 1.0; hi = np.percentile(U, 99.5, 0) + 1.0
nx, ny = (np.ceil((hi - lo) / RES)).astype(int) + 1
ij = np.floor((U - lo) / RES).astype(int)
m = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
cnt = np.zeros((ny, nx), np.int32); np.add.at(cnt, (ij[m, 1], ij[m, 0]), 1)
occ = (cnt >= MIN_HITS).astype(np.uint8) * 255
occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
def _dir(s):
    v = s[1] - s[0]; return v / np.linalg.norm(v)
mode = a.mode
reason = "显式选择"
facade = None
if mode != "walls":
    box_file = next((os.path.join(a.r1, p) for p in ("06_all_boxes_merged.json", "06_all_boxes.json")
                     if os.path.isfile(os.path.join(a.r1, p))), None)
    boxes = json.load(open(box_file)) if box_file else {}
    try:
        facade = fit_facade(boxes, e1, e2, U, Hh, np.concatenate(frame_ids)[d < RADIUS], C2, THICK)[0]
    except ValueError as exc:
        if mode == "facade":
            raise
        reason = str(exc)
if mode == "facade":
    segs = [facade]
else:
    L = cv2.HoughLinesP(occ, 1, np.pi / 360, threshold=int(1.5 / RES * 0.5),
                        minLineLength=int(1.5 / RES), maxLineGap=int(0.5 / RES))
    segs = [] if L is None else [(np.array(l[:2], float) * RES + lo, np.array(l[2:], float) * RES + lo) for l in L[:, 0]]

    # 共线合并: 同一面墙常被检出好几条平行近邻的短线
    merged = True
    while merged and len(segs) > 1:
        merged = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                di, dj = _dir(segs[i]), _dir(segs[j])
                if abs(di @ dj) < np.cos(np.radians(MERGE_DEG)): continue
                nrm = np.array([-di[1], di[0]])
                if abs((segs[j][0] - segs[i][0]) @ nrm) > MERGE_OFF or abs((segs[j][1] - segs[i][0]) @ nrm) > MERGE_OFF: continue
                t = [(p - segs[i][0]) @ di for p in (segs[i][0], segs[i][1], segs[j][0], segs[j][1])]
                if max(min(t[2], t[3]) - max(t[0], t[1]), min(t[0], t[1]) - max(t[2], t[3])) > MERGE_GAP: continue
                segs[i] = (segs[i][0] + min(t) * di, segs[i][0] + max(t) * di); segs.pop(j); merged = True; break
            if merged: break
    segs = [s for s in segs if np.linalg.norm(s[1] - s[0]) >= MIN_LEN]
    # 沿线真有墙面点的比例: Hough 会把零散的点连成斜线(spl 去掉人以后还剩两三条斜穿巷子的), 沿线 0.15 m 内有点的长度不到 SUPPORT 就不盖
    _sup = cv2.dilate((cnt >= MIN_HITS).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    def _support(p, q):
        n = max(2, int(np.linalg.norm(q - p) / 0.1)); pts = p + np.outer(np.linspace(0, 1, n), q - p)
        ij = np.floor((pts - lo) / RES).astype(int); ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
        v = np.zeros(n, bool); v[ok] = _sup[ij[ok, 1], ij[ok, 0]]; return float(v.mean())
    _n0 = len(segs); segs = [s for s in segs if _support(*s) >= SUPPORT]
    print(f"[室外墙] 沿线有墙面点不到 {SUPPORT:.0%} 的去掉 {_n0 - len(segs)} 段")
    # 建筑的墙基本只有两个互相垂直的方向: 主方向 = 墙段方向按长度加权的众数(角度对 90° 取模), 差 MANH_DEG 以上的斜段不盖
    supported_lines = list(segs)
    if segs:
        _ang = np.array([np.degrees(np.arctan2(*(q - p)[::-1])) % 90 for p, q in segs])
        _len = np.array([np.linalg.norm(q - p) for p, q in segs])
        _hist = np.array([(_len * (np.abs(((_ang - c) + 45) % 90 - 45) < 5)).sum() for c in range(90)])
        _dom = float(np.argmax(_hist))
        _off = np.abs(((_ang - _dom) + 45) % 90 - 45)
        _n0 = len(segs); segs = [sg for sg, o in zip(segs, _off) if o <= MANH_DEG]
        print(f"[室外墙] 主方向 {_dom:.0f}°(对 90° 取模), 斜着的(差 {MANH_DEG:.0f}° 以上)去掉 {_n0 - len(segs)} 段")

    # ★ 2026-11-11 看穿检查: 墙面切成 0.5 m 宽 x 0.3 m 高的格子, 格子上的点投到各帧, 深度图在那里比墙远(隔着墙看到了后面 = 那里没墙)
    #   的比例超过 SEE_MAX 的格子算"空"。每一竖条的墙顶 = 从地面往上第一个空格子的下沿; 墙顶不到 SEE_MIN_H 的竖条去掉。
    #   连续的竖条(总长够 MIN_LEN)按墙顶分组(组内差 <= 0.3 m), 每组一段, 高取组内最低。看不到的格子判不了, 当实。
    #   singin: 一个街面检出 5 段前后错开的墙(25 s 长镜头深度有漂移), 一段 91% 被看穿, 整段去掉; 另一段是橱窗下的白台座,
    #   1.2 m 以下 0% 被看穿, 往上 72~97%(后面是海报)—— 只留到 1.2 m。用户: "墙体应该在海报后面啊"。门窗洞同理被挖掉。
    SEE_COL, SEE_ROW, SEE_MAX, SEE_MIN_VIS, SEE_MIN_H = 0.5, 0.3, 0.5, 10, 0.6
    _D = {k: np.load(f"{da3}/depth/{k[:-4]}_depth.npy") for k in _keys}
    def _seegrid(p, q, top):
        di = _dir((p, q)); Lg = float(np.linalg.norm(q - p)); nc = max(1, int(np.ceil(Lg / SEE_COL))); nr = max(1, int(np.ceil(top / SEE_ROW)))
        t = np.arange(0.05, Lg, 0.1); z = (np.arange(nr) + 0.5) * SEE_ROW
        uv = p[None] + t[:, None] * di[None]
        P = (uv[:, None, 0, None] * e1 + uv[:, None, 1, None] * e2 + (gh + z)[None, :, None] * up).reshape(-1, 3)
        cid = (np.minimum((t / SEE_COL).astype(int), nc - 1)[:, None] * nr + np.arange(nr)[None]).reshape(-1)
        see, vis = np.zeros(nc * nr), np.zeros(nc * nr)
        for k in _keys:
            c = _cams[k]; E = np.array(c["extrinsic_w2c_4x4"], float); K = np.array(c["intrinsic_3x3"], float)
            h_, w_ = c["input_hw"] if "input_hw" in c else (2 * K[1, 2], 2 * K[0, 2])
            Pc = P @ E[:3, :3].T + E[:3, 3]; zc = Pc[:, 2]; ok = zc > 0.3
            x = K[0, 0] * Pc[:, 0] / np.maximum(zc, 1e-6) + K[0, 2]; y = K[1, 1] * Pc[:, 1] / np.maximum(zc, 1e-6) + K[1, 2]
            ok &= (x >= 0) & (x < w_) & (y >= 0) & (y < h_)
            if not ok.any(): continue
            d = _D[k]; sx, sy = d.shape[1] / w_, d.shape[0] / h_
            do = d[(y[ok] * sy).astype(int).clip(0, d.shape[0] - 1), (x[ok] * sx).astype(int).clip(0, d.shape[1] - 1)]
            np.add.at(see, cid[ok], (do > zc[ok] * 1.05 + 0.2).astype(float)); np.add.at(vis, cid[ok], 1.0)
        empty = ((see / np.maximum(vis, 1)) > SEE_MAX) & (vis >= SEE_MIN_VIS)
        return empty.reshape(nc, nr), (see.sum() / max(vis.sum(), 1)), Lg, di
    def _height(p, q):
        di = _dir((p, q)); nrm = np.array([-di[1], di[0]]); Lg = float(np.linalg.norm(q - p))
        rel = U - p; t = rel @ di; off = np.abs(rel @ nrm); near = (off < 0.2) & (t > -0.1) & (t < Lg + 0.1)
        return (float(np.percentile(Hh[near], 95)) if near.sum() > 20 else 2.5), int(near.sum())
    _new, _rep = [], []
    for p, q in segs:
        top0, _ = _height(p, q)
        emp, ratio, Lg, di = _seegrid(p, q, top0)
        tops = np.array([top0 if not e.any() else int(np.argmax(e)) * SEE_ROW for e in emp])     # 每竖条: 第一个空格子的下沿
        keep = tops >= SEE_MIN_H
        _rep.append((round(Lg, 1), int(round(100 * ratio)), round(float((~keep).sum() * SEE_COL), 1), round(float(top0 - np.where(keep, tops, top0).min()), 1) if keep.any() else None))
        i = 0
        while i < len(keep):
            if not keep[i]: i += 1; continue
            j = i
            while j + 1 < len(keep) and keep[j + 1]: j += 1
            if min((j + 1) * SEE_COL, Lg) - i * SEE_COL >= MIN_LEN:
                g0 = i
                for g in range(i, j + 2):                                  # 按墙顶分组
                    if g == j + 1 or abs(tops[g] - tops[g0:g].min() if g > g0 else 0) > 0.3:
                        a0, b0 = g0 * SEE_COL, min(g * SEE_COL, Lg)
                        if b0 - a0 > 0.05: _new.append((p + a0 * di, p + b0 * di, float(tops[g0:g].min())))
                        g0 = g
            i = j + 1
    print(f"[室外墙] 看穿检查(每段: 长 m / 被看穿 % / 整条去掉 m / 最多削低 m): {_rep} -> {len(segs)} 段变 {len(_new)} 段")
    segs = _new
    if mode == "auto":
        mode = "walls"
        if facade is not None:
            reason = "另有墙向、偏移或范围不能由同一立面解释"
            if facade_explains_walls(facade, segs, supported_lines):
                p, q, top = facade
                empty, _, _, _ = _seegrid(p, q, top)
                if np.any(empty[:, :min(3, empty.shape[1])].all(axis=1)):
                    reason = "多视角深度显示立面底部有开口"
                else:
                    segs, mode = [facade], "facade"
                    reason = "锚点共面，深度连续，候选墙段属于同一立面"
        print(f"[自动选墙] {mode}: {reason}")
    del _D

# 每段的高: 离这段 0.2 m 内的竖直面点的 95 分位
scene, rec = trimesh.Scene(), []
for i, (p, q, htop) in enumerate(segs):
    di = _dir((p, q)); nrm = np.array([-di[1], di[0]]); Lg = float(np.linalg.norm(q - p))
    rel = U - p; t = rel @ di; off = np.abs(rel @ nrm)
    near = (off < 0.2) & (t > -0.1) & (t < Lg + 0.1)
    hh = float(np.percentile(Hh[near], 95)) if near.sum() > 20 else 2.5
    hh = htop if mode == "facade" else min(hh, htop)                                       # 看穿检查削过的墙顶
    box = trimesh.creation.box(extents=[Lg + THICK, THICK, hh])
    yaw = float(np.arctan2(di[1], di[0]))
    T = np.eye(4); T[:3, :3] = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.]])
    T[:3, 3] = np.r_[(p + q) / 2, hh / 2]
    box.apply_transform(T)
    # (e1, e2, 离地高) -> DA3 世界系
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2] = e1, e2, up; M[:3, 3] = gh * up
    box.apply_transform(M)
    box.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=[150 / 255, 150 / 255, 155 / 255, ALPHA / 255], metallicFactor=0.0, roughnessFactor=0.9,
        alphaMode="BLEND", doubleSided=True))
    scene.add_geometry(box, node_name=f"wall_{i}")
    rec.append(dict(a=p.tolist(), b=q.tolist(), z0=0.0, h=hh, thick=THICK, 长=round(Lg, 2), 墙面点=int(near.sum())))
if not len(scene.geometry):
    raise SystemExit("[失败] 一段墙都没检出来 —— 这个场景可能没有成片的竖直墙面, 别开 OUTDOOR_SHELL")
scene.apply_transform(export_T(up))                      # 和 assemble_scene 的转正同一份: 壳在 FLIP 之后并进去
scene.export(f"{OUT}/09_房间壳.glb")
json.dump(dict(请求模式=a.mode, 模式=mode, 选择依据=reason, 来源="多视角立面拟合" if mode == "facade" else "outdoor_walls.py: 竖直面的点 + 直线检测", 系="(e1, e2, 离地高)", e1=e1.tolist(), e2=e2.tolist(),
               up=up.tolist(), ground_h=gh, 墙线=rec), open(f"{OUT}/09_房间壳_墙线.json", "w"), ensure_ascii=False, indent=1)

sc = 900 / max(hi - lo); img = np.full((int((hi - lo)[1] * sc) + 1, int((hi - lo)[0] * sc) + 1, 3), 255, np.uint8)
pij = ((U - lo) * sc).astype(int); ok = (pij[:, 0] < img.shape[1]) & (pij[:, 1] < img.shape[0]) & (pij >= 0).all(1)
img[pij[ok, 1], pij[ok, 0]] = (170, 170, 170)
for r in rec:
    cv2.line(img, tuple(((np.array(r["a"]) - lo) * sc).astype(int)), tuple(((np.array(r["b"]) - lo) * sc).astype(int)), (0, 0, 255), 3)
for c in C2:
    cv2.circle(img, tuple(((c - lo) * sc).astype(int)), 2, (255, 0, 0), -1)
img = np.ascontiguousarray(img[::-1])
cv2.putText(img, f"gray=vertical surfaces red=walls({len(rec)}) blue=cameras", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
cv2.imwrite(f"{OUT}/09_房间壳_俯视.jpg", img)
print(f"[室外墙] 去掉人的像素的帧 {_npm}/{len(_keys)}, 竖直面点 {len(U)}, 检出 {len(rec)} 段墙, 总长 {sum(r['长'] for r in rec):.1f} m, "
      f"高 {min(r['h'] for r in rec):.2f}~{max(r['h'] for r in rec):.2f} m -> {OUT}/09_房间壳.glb")
