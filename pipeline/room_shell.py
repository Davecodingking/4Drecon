#!/usr/bin/env python
"""房间壳: 从点云算出闭合的房间轮廓, 建灰色半透明墙 + 地板。两条室内链共用。

为什么不用 SpatialLM 的墙 —— 在 cyberpunk 上把它的 arch 输出投回画面核对过:
117 面墙里 108 面是短段, 首尾相接组成 5 条弧, 而这些弧**贴在弧形栏杆和演奏台边缘上**,
不是房间边界; 剩下 9 面长墙也对不上画面里那面带垂挂灯管的大墙。真正的玻璃幕墙一面都没认出来
(它训练集是住宅, 玻璃在点云里几乎没点)。所以拿它的墙线当房间轮廓, 围出来的不是房间。

改成从点云自己算: 地板以上、天花板以下的点栅格化 -> 形态学闭运算填掉窗洞和遮挡缺口 ->
取最大连通域的外轮廓 -> 折线简化。这个判据不依赖语义, 玻璃幕墙那侧只要有一点点点云
(窗框、栏杆、外面的楼) 就能围上, 而且天然闭合。

产物:
  房间壳.glb        floor(多边形地板) + wall_0..N(有厚度的实体盒子, 灰色半透明)
  房间壳_墙线.json  每面墙的 {a, b, z0, h, thick}, 给物理那步做 2D 线段碰撞用
                    —— 从 mesh 反推有向包围盒不可靠, 建的时候存下来最省事

坐标系: 点云和布局都在 "z 朝上、地板 z=0" 系。--up y 时按用户拍板转成标准 glTF 的 y 朝上,
变换和 assemble_scene.py:225-229 的罗德里格斯一致 (室内的天恒为 +z, 所以退化成一个常量矩阵)。

用法:
  python room_shell.py --pcd 03_点云_z朝上.ply --out 房间壳.glb --seg-out 房间壳_墙线.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "outdoor", "scripts"))
from world_frame import load_up, export_R
import numpy as np, cv2, trimesh

ap = argparse.ArgumentParser()
ap.add_argument("--pcd", required=True, help="03_点云_z朝上.ply")
ap.add_argument("--out", required=True)
ap.add_argument("--seg-out", default="", help="墙线 json, 空=和 --out 同名换后缀")
ap.add_argument("--floor-glb", default="", help="有贴图的地板 glb, 给了就用它替换纯色地板")
ap.add_argument("--res", type=float, default=0.05, help="栅格分辨率(m)")
ap.add_argument("--close", type=int, default=25, help="形态学闭运算核(格). 越大越能填窗洞")
ap.add_argument("--simplify", type=float, default=0.25, help="轮廓折线简化容差(m)")
ap.add_argument("--z-lo", type=float, default=0.05, help="取点的下界, 躲开地板本身")
ap.add_argument("--z-hi", type=float, default=0.0, help="取点的上界, 0=自动取 97 分位")
ap.add_argument("--wall-h", type=float, default=0.0, help="墙高, 0=自动取天花板")
ap.add_argument("--thick", type=float, default=0.10, help="墙厚。SpatialLM 给的恒为 0")
ap.add_argument("--alpha", type=int, default=140, help="墙的不透明度 0-255。255=不透明")
ap.add_argument("--up", choices=["y", "z"], default="y", help="输出坐标系的竖直轴")
ap.add_argument("--zup", default="", help="03_点云_z朝上.json。给了就把壳变回 DA3 世界系, "
                                          "好和 assemble_scene 的输出对齐")
ap.add_argument("--world-up", default="", help="02_da3/world_up.json。--zup 给了才用")
ap.add_argument("--walls-txt", default="", help="SpatialLM 布局 txt (wall_i=Wall(...))。给了就用它的墙段"
                                               "代替点云轮廓的墙; 地板仍按点云轮廓铺")
ap.add_argument("--boxes", default="", help="06_all_boxes_merged.json。给了就保证壳把每个量到的物体都围进去"
                                           " —— 认出来、生成了的东西一定在房间里, 壳不许把它关在外面")
ap.add_argument("--exclude", default="", help="这些词的盒子不参与围壳(逗号分隔), 和 assemble 的 EXCLUDE 同一份")
ap.add_argument("--box-out", type=float, default=0.5, help="盒子中心在点云足迹外超过这么多米才撑壳去围它; "
                                                       "贴墙的窗/门中心本来就常落在墙线外一点, 不该因为它们鼓包")
ap.add_argument("--box-margin", type=float, default=1.0, help="撑过去时物体周围至少留这么多米。盒子是点云量的, "
                                                          "组装时 mesh 按自己的长宽比摆, 足迹可能比盒子宽")
ap.add_argument("--cam-margin", type=float, default=0.5, help="拍摄的相机离墙至少留这么多米, 不够就把那面墙整体外推 "
                                                          "(拿相机的人站在相机后面, 大约半米)。<0 = 不管相机。要配 --zup")
ap.add_argument("--columns", default="", help="columns_fit.py 出的 09_柱子.json。给了就把柱子盖成地板到天花板的灰色方柱"
                                             " (节点名 pillar_N, 物理那步当钉死的东西)")
ap.add_argument("--column-wall-gap", type=float, default=0.15, help="柱子截面离墙不到这么多米就不盖 —— 墙已经挡在那儿了, "
                                                                  "再盖一根只会和墙上的窗/门撞 (实测 office: 那根其实是两扇窗之间的一段墙)")
a = ap.parse_args()

V = np.asarray(trimesh.load(a.pcd).vertices)
z_hi = a.z_hi if a.z_hi > 0 else float(np.percentile(V[:, 2], 97))
wall_h = a.wall_h if a.wall_h > 0 else z_hi
print(f"[点云] {len(V)} 点, 天花板约 {z_hi:.2f}m -> 墙高 {wall_h:.2f}m")

# ---------- 1. 栅格化 + 闭运算 + 最大连通域外轮廓 ----------
P = V[(V[:, 2] > a.z_lo) & (V[:, 2] < z_hi)][:, :2]
lo = np.percentile(P, 0.5, axis=0) - 0.5
hi = np.percentile(P, 99.5, axis=0) + 0.5
W = int((hi[0] - lo[0]) / a.res) + 1
H = int((hi[1] - lo[1]) / a.res) + 1
ij = ((P - lo) / a.res).astype(int)
ij = ij[(ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)]
occ = np.zeros((H, W), np.uint8)
occ[ij[:, 1], ij[:, 0]] = 255
# 两级闭运算: 小核先把点云的稀疏空隙连上, 大核再填掉窗洞和被家具遮挡的缺口
occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((a.close, a.close), np.uint8))

# ---------- 1b. 壳必须把量到的物体都围进去 ----------
# ★ 上面按 0.5/99.5 分位切点, 是为了扔掉远处乱飞的深度噪声。但远处真实的东西也在那一撮里:
#   office 最远那头离镜头二十多米, 红柜子(shelf unit#2)被识别、被生成了, 却和噪声一起被切到
#   壳外, 场景里它就立在墙外面。认出来的东西一定在房间里 —— 所以对落在足迹外的物体,
#   把它那一侧的整面墙平移出去围住它(房间保持方方正正, 不单独鼓包, 见下面 SIDE_TOL 那段)。
# ★ 只动真在外面的: 足迹本身照旧只由点云决定, 所有物体都在里面时壳逐位不变。试过把全部盒子
#   都涂进占用图, 窗/门的盒子常凸出墙线(透过玻璃量到了外面), 壳在每扇窗那里都鼓包,
#   墙从 38 面变 60 面, 窗户反而插进了新墙里。
if a.boxes:
    if not (a.zup and a.world_up):
        raise SystemExit("[失败] --boxes 要配 --zup 和 --world-up: 盒子在 DA3 世界系, 得先转到 z 朝上系")
    _z = json.load(open(a.zup)); _R = np.array(_z["R"]); _t = np.array(_z["t"]); _s = float(_z["s"])
    _up = np.array(json.load(open(a.world_up))["up"], float); _up /= np.linalg.norm(_up)
    _cam = np.asarray(_z.get("camera_centers_new") or [[0, 0, 0]], float)[:, :2]
    _exc = {w.strip() for w in a.exclude.split(",") if w.strip()}
    BOX = []
    for _w, _bs in json.load(open(a.boxes)).items():
        if _w in _exc: continue
        for _b in _bs:                                  # 盒子约定见 boxes_from_masks.fit_box: 尺寸 = [长轴, 横轴, 高]
            _c = np.array(_b["中心"]); _a1 = np.array(_b["长轴"]); _a2 = np.cross(_up, _a1)
            _L, _W = _b["尺寸"][0] / 2, _b["尺寸"][1] / 2
            _cs = np.array([_c + sx * _L * _a1 + sy * _W * _a2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            BOX.append((f"{_w}#{_b.get('obj_id')}", (_s * (_R @ _cs.T)).T[:, :2] + _t[:2]))
    _n, _lab = cv2.connectedComponents(occ)
    _main = (_lab == 1 + int(np.argmax(np.bincount(_lab.ravel())[1:]))).astype(np.uint8)
    _dist = cv2.distanceTransform(1 - _main, cv2.DIST_L2, 5) * a.res       # 到主体的距离(m)
    def _gap(q):
        i, j = np.floor((q.mean(0) - lo) / a.res).astype(int)
        return float(_dist[j, i]) if (0 <= i < W and 0 <= j < H) else np.inf  # 画布外 = 远在外面
    _out = [(nm, q) for nm, q in BOX if _gap(q) > a.box_out]
    if _out:
        # 画布放大到盖住它们; 原来的占用图原样贴进去, 足迹其余部分一个像素都不变
        _pad = 0.5 + max(a.box_margin, max(0.5 * float(np.max(np.linalg.norm(q - np.roll(q, 1, 0), axis=1))) for _, q in _out))
        _all = np.concatenate([q for _, q in _out] + [_cam])
        lo2 = np.minimum(lo, np.floor((_all.min(0) - _pad - lo) / a.res) * a.res + lo)
        hi2 = np.maximum(hi, _all.max(0) + _pad)
        W2 = int((hi2[0] - lo2[0]) / a.res) + 1; H2 = int((hi2[1] - lo2[1]) / a.res) + 1
        oi, oj = np.round((lo - lo2) / a.res).astype(int)
        W2, H2 = max(W2, oi + W), max(H2, oj + H)
        big = np.zeros((H2, W2), np.uint8); big[oj:oj + H, oi:oi + W] = _main * 255
        # ★ 整面墙平移出去, 不在一处单独鼓包 —— 房间要保持方方正正(用户拍板)。
        #   z 朝上系做过绕 z 的轴对齐, 墙基本沿 x/y 走, 所以"一面墙"= 朝物体那一侧、
        #   边界离最近点不到 SIDE_TOL 的那一段连续的列(或行)。整段一起推到物体外沿 + 余量。
        #   试过从相机拉楔形: 盒子是细长条, 楔形成了一根细针, 简化后红柜子 0% 在壳里;
        #   加宽成一块之后倒是围住了, 但是单独伸出去一块, 不像房间。
        SIDE_TOL = 1.0
        big = (big > 0).astype(np.uint8)
        for _nm, q in _out:
            qp = (q - lo2) / a.res                                       # 足迹, 像素 (列, 行)
            _r = max(a.box_margin, 0.5 * float(np.max(np.linalg.norm(q - np.roll(q, 1, 0), axis=1)))) / a.res
            ys, xs = np.nonzero(big); c0 = qp.mean(0)
            k = int(np.argmin((xs - c0[0]) ** 2 + (ys - c0[1]) ** 2)); near = np.array([xs[k], ys[k]])
            v = c0 - near; ax = int(np.argmax(np.abs(v))); sg = 1 if v[ax] > 0 else -1
            M = big if ax == 1 else big.T                                # 统一成"沿行方向往外推"
            qq = qp if ax == 1 else qp[:, ::-1]; nn = near if ax == 1 else near[::-1]
            rows = np.arange(M.shape[0])[:, None]
            edge = np.where(M.any(0), (np.where(M > 0, rows, -1).max(0) if sg > 0
                                       else np.where(M > 0, rows, 10 ** 9).min(0)), -1)
            e0 = edge[nn[0]]; tol = SIDE_TOL / a.res
            i0 = i1 = nn[0]                                              # 这面墙: 连续、边界和最近点齐平的列
            while i0 - 1 >= 0 and edge[i0 - 1] >= 0 and abs(edge[i0 - 1] - e0) <= tol: i0 -= 1
            while i1 + 1 < M.shape[1] and edge[i1 + 1] >= 0 and abs(edge[i1 + 1] - e0) <= tol: i1 += 1
            i0 = int(min(i0, np.floor(qq[:, 0].min() - _r))); i1 = int(max(i1, np.ceil(qq[:, 0].max() + _r)))
            i0, i1 = max(i0, 0), min(i1, M.shape[1] - 1)
            tgt = int(np.ceil(qq[:, 1].max() + _r)) if sg > 0 else int(np.floor(qq[:, 1].min() - _r))
            tgt = int(np.clip(tgt, 0, M.shape[0] - 1))
            for i in range(i0, i1 + 1):
                e = edge[i] if edge[i] >= 0 and abs(edge[i] - e0) <= tol else e0
                if sg > 0: M[min(e, tgt):tgt + 1, i] = 1
                else:      M[tgt:max(e, tgt) + 1, i] = 1
            big = M if ax == 1 else M.T
            print(f"[物体] {_nm}: 在{'+-'[sg < 0]}{'xy'[ax]} 那面墙外, 整面墙({(i1 - i0) * a.res:.1f} m 宽)"
                  f"外推 {abs(tgt - e0) * a.res:.1f} m 把它围进来")
        big = big * 255
        occ, lo, W, H = big, lo2, W2, H2
    print(f"[物体] {len(BOX)} 个盒子查过, " + (f"{len(_out)} 个在点云足迹外超过 {a.box_out}m, "
          f"已把那面墙整体外推围住: {[nm for nm, _ in _out]}" if _out else "全在足迹内, 壳不动"))
# ---------- 1c. 拍摄的相机必须在屋里 ----------
# ★ 2026-09-22 office: 开头 23 帧相机在墙外, 最远 1.5 m。足迹只由点云决定, 而相机朝前拍,
#   自己脚下和身后那片从来没拍到、没有点, 墙就画到了相机前面。视频是在屋里拍的, 相机和拿相机的人
#   一定在屋里 —— 和上面"认出来的物体一定在屋里"是同一个道理, 以前相机只拿来算画布大小。
# ★ 办法和物体一样(用户拍板): 相机那一侧的整面墙平移出去, 房间保持方正, 离相机至少 --cam-margin。
#   和物体那段只差一处: 新墙和原来的边之间整段填满。相机身后常是漏斗形的斜墙(只拍到朝前的一个锥),
#   只填到齐平那段, 斜墙那里会留下缺口。所有相机都离墙够远时, 壳逐位不变。
def _sdist(F, pts, lo_):
    """每个点到足迹边界的有符号距离(米), 在里面为正, 画布外当成远在外面。"""
    d_in = cv2.distanceTransform(F, cv2.DIST_L2, 5) * a.res
    d_out = cv2.distanceTransform(1 - F, cv2.DIST_L2, 5) * a.res
    ij = np.floor((pts - lo_) / a.res).astype(int)
    s_ = np.full(len(pts), -np.inf)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < F.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < F.shape[0])
    i_, j_ = ij[ok, 0], ij[ok, 1]
    s_[ok] = np.where(F[j_, i_] > 0, d_in[j_, i_], -d_out[j_, i_])
    return s_
def _push_wall(big, qp, ax, sg, r, tol):
    """把 sg 那一侧的整面墙推到 qp(像素, 列/行) 外 r 格。哪几列算"这面墙"和物体那段同一个判据。"""
    M = big if ax == 1 else big.T
    qq = qp if ax == 1 else qp[:, ::-1]
    rows = np.arange(M.shape[0])[:, None]
    has = M.any(0)
    edge = np.where(has, (np.where(M > 0, rows, -1).max(0) if sg > 0
                          else np.where(M > 0, rows, 10 ** 9).min(0)), -1)
    c0 = int(np.clip(np.round(np.median(qq[:, 0])), 0, M.shape[1] - 1))
    if not has[c0]:
        cols = np.nonzero(has)[0]; c0 = int(cols[np.argmin(np.abs(cols - c0))])
    e0 = int(edge[c0]); i0 = i1 = c0
    while i0 - 1 >= 0 and edge[i0 - 1] >= 0 and abs(edge[i0 - 1] - e0) <= tol: i0 -= 1
    while i1 + 1 < M.shape[1] and edge[i1 + 1] >= 0 and abs(edge[i1 + 1] - e0) <= tol: i1 += 1
    i0 = int(max(0, min(i0, np.floor(qq[:, 0].min() - r))))
    i1 = int(min(M.shape[1] - 1, max(i1, np.ceil(qq[:, 0].max() + r))))
    tgt = int(np.ceil(qq[:, 1].max() + r)) if sg > 0 else int(np.floor(qq[:, 1].min() - r))
    tgt = int(np.clip(tgt, 0, M.shape[0] - 1))
    for i in range(i0, i1 + 1):
        e = int(edge[i]) if edge[i] >= 0 else e0          # 填到这一列原来的边, 中间不留缝
        if sg > 0: M[min(e, tgt):tgt + 1, i] = 1
        else:      M[tgt:max(e, tgt) + 1, i] = 1
    return (M if ax == 1 else M.T), (i1 - i0) * a.res, abs(tgt - e0) * a.res
def _ranges(ix):
    ix = sorted(int(x) for x in ix); out = []; st = pv = ix[0]
    for x in ix[1:] + [None]:
        if x is not None and x == pv + 1: pv = x; continue
        out.append(f"{st}" if st == pv else f"{st}-{pv}")
        if x is not None: st = pv = x
    return ", ".join(out)
if a.cam_margin >= 0 and a.zup:
    _cc = np.asarray(json.load(open(a.zup)).get("camera_centers_new") or [], float).reshape(-1, 3)[:, :2]
    if len(_cc):
        _n, _lab = cv2.connectedComponents((occ > 0).astype(np.uint8))
        _F = (_lab == 1 + int(np.argmax(np.bincount(_lab.ravel())[1:]))).astype(np.uint8)
        _sd = _sdist(_F, _cc, lo)
        _bad = np.nonzero(_sd < a.cam_margin)[0]
        if not len(_bad):
            print(f"[相机] {len(_cc)} 帧全在墙内, 离墙都不少于 {a.cam_margin} m, 壳不动")
        else:
            print(f"[相机] {len(_cc)} 帧里 {int((_sd < 0).sum())} 帧在墙外(最远 {max(0.0, -float(_sd.min())):.2f} m), "
                  f"{int(((_sd >= 0) & (_sd < a.cam_margin)).sum())} 帧离墙不到 {a.cam_margin} m: 第 {_ranges(_bad)} 帧")
            _pad = a.cam_margin + 0.5                        # 画布放大到盖住所有相机, 原来的足迹原样贴进去
            _hi = lo + np.array([W, H]) * a.res
            lo2 = np.minimum(lo, np.floor((_cc.min(0) - _pad - lo) / a.res) * a.res + lo)
            hi2 = np.maximum(_hi, _cc.max(0) + _pad)
            W2 = int((hi2[0] - lo2[0]) / a.res) + 1; H2 = int((hi2[1] - lo2[1]) / a.res) + 1
            oi, oj = np.round((lo - lo2) / a.res).astype(int)
            W2, H2 = max(W2, oi + W), max(H2, oj + H)
            big = np.zeros((H2, W2), np.uint8); big[oj:oj + H, oi:oi + W] = _F
            for _round in range(4):                          # 推一面墙可能让别的相机露出来, 最多再查三轮
                _sd = _sdist(big, _cc, lo2)
                _bad = np.nonzero(_sd < a.cam_margin - a.res)[0]   # 留一格的取整余量
                if not len(_bad): break
                _bd = np.argwhere((big > 0) & (cv2.erode(big, np.ones((3, 3), np.uint8)) == 0))[:, ::-1].astype(float)
                _grp = {}
                for k in _bad:
                    p = (_cc[k] - lo2) / a.res
                    b = _bd[np.argmin(((_bd - p) ** 2).sum(1))]
                    v = (p - b) if _sd[k] < 0 else (b - p)   # 朝外: 墙外的从足迹指向相机, 墙内的从相机指向墙
                    if not np.any(v): v = p - _bd.mean(0)
                    ax = int(np.argmax(np.abs(v))); sg = 1 if v[ax] > 0 else -1
                    _grp.setdefault((ax, sg), []).append(k)
                for (ax, sg), ks in _grp.items():
                    big, wlen, dist = _push_wall(big, (_cc[ks] - lo2) / a.res, ax, sg, a.cam_margin / a.res, 1.0 / a.res)
                    big = np.ascontiguousarray(big)
                    print(f"[相机] 第 {_ranges(ks)} 帧在{'+-'[sg < 0]}{'xy'[ax]} 那面墙外或贴着墙: "
                          f"整面墙({wlen:.1f} m 宽)外推 {dist:.1f} m")
            _sd = _sdist(big, _cc, lo2)
            print(f"[相机] 外推后 {int((_sd < 0).sum())} 帧在墙外, 离墙最近的一帧 {float(_sd.min()):.2f} m")
            occ, lo, W, H = big * 255, lo2, W2, H2
cnts, _ = cv2.findContours(occ, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not cnts:
    raise SystemExit("[失败] 点云里找不到连通的房间区域")
c = max(cnts, key=cv2.contourArea)
poly = cv2.approxPolyDP(c, a.simplify / a.res, True).reshape(-1, 2)
XY = poly * a.res + lo
per = float(sum(np.linalg.norm(XY[i] - XY[(i + 1) % len(XY)]) for i in range(len(XY))))
print(f"[轮廓] 面积 {cv2.contourArea(c)*a.res**2:.1f} m², 周长 {per:.1f}m, {len(XY)} 个角点 (闭合)")

# ---------- 2. 地板: 多边形三角化 ----------
# ★ 必须用耳切(ear clipping), 不能用 "Delaunay + 质心在不在多边形里筛"。
#   后者的边界是筛剩下的三角形拼出来的, 不是轮廓本身: 贴边那一圈三角形又大又扁,
#   质心稍微偏出去整块就被删, 地板铺不到墙根。实测中位差 16~22cm, 最坏一段 2.5m ——
#   这就是"地板和墙没严丝合缝"。补密边界点也治不好, 只是把缝从 22cm 变成 16cm。
#   耳切是对多边形本身做三角化, 边界逐点等于轮廓, 缝为 0。任意凹多边形都成立。
def ear_clip(P):
    n = len(P)
    idx = list(range(n))
    area2 = sum((P[i][0]*P[(i+1) % n][1] - P[(i+1) % n][0]*P[i][1]) for i in range(n))
    if area2 < 0:                                   # 统一成逆时针, 凸性判据才有定义
        idx = idx[::-1]
    def cross(o, a, b):
        return (P[a][0]-P[o][0])*(P[b][1]-P[o][1]) - (P[a][1]-P[o][1])*(P[b][0]-P[o][0])
    def inside(p, a, b, c):
        d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))
    tris, guard = [], 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            a, b, c = idx[k-1], idx[k], idx[(k+1) % len(idx)]
            if cross(a, b, c) <= 0:                 # 凹角, 不是耳朵
                continue
            if any(inside(v, a, b, c) for v in idx if v not in (a, b, c)):
                continue                            # 三角形里含别的顶点, 不是耳朵
            tris.append([a, b, c]); idx.pop(k); break
        else:
            break                                   # 找不到耳朵(多边形自交), 就此打住
    if len(idx) == 3:
        tris.append(list(idx))
    return np.array(tris)

keep = ear_clip(XY)
pts2d = XY
floor = trimesh.Trimesh(np.c_[pts2d, np.zeros(len(pts2d))], keep)
if float(floor.face_normals[0][2]) < 0:          # 法向朝天, 否则从上往下看是隐形的
    floor.faces = floor.faces[:, ::-1]
floor.visual.face_colors = [150, 150, 155, 255]
print(f"[地板] 耳切: {len(pts2d)} 顶点 / {len(keep)} 三角形 (边界逐点等于轮廓)")

# ---------- 3. 墙: 沿轮廓每条边挤出一个实体盒子 ----------
# ★ 必须是实体盒子, 不能是零厚度的面片。面片的世界 AABB 在法向上 extent 恰好为 0,
#   物理推开循环第一句 `if (ov<=0).any(): continue` 永远成立, 一次都推不动。
# 墙段来源二选一:
#   默认      点云轮廓的每条边 (闭合, 不依赖语义; 开放式空间 cyberpunk 只能走这条)
#   --walls-txt  SpatialLM 的墙 (封闭住宅房间是它的训练分布, 墙面平直、门洞位置有语义)
#   SpatialLM 的墙线在同一个 "z 朝上、地板 z=0" 系 (它吃的就是 03_点云_z朝上.ply), 后面的坐标变换一行不用改。
#   SpatialLM 的墙不保证首尾闭合, 所以地板不跟着换, 仍按点云轮廓铺。
def _lm_walls(path):
    import re
    out = []
    for line in open(path):
        m = re.match(r"\s*wall_\d+\s*=\s*Wall\(([^)]*)\)", line)
        if not m:
            continue
        v = [float(x) for x in m.group(1).split(",")]
        ax, ay, az, bx, by, bz, h = v[:7]
        out.append((np.array([ax, ay]), np.array([bx, by]), min(az, bz), h))
    if not out:
        raise SystemExit(f"[失败] {path} 里一面墙都没有 —— SpatialLM 没出墙, 别用空壳往下跑")
    return out

if a.walls_txt:
    SEGS = _lm_walls(a.walls_txt)
    print(f"[墙] 用 SpatialLM 的墙: {len(SEGS)} 面 <- {a.walls_txt}")
else:
    SEGS = [(XY[i], XY[(i + 1) % len(XY)], 0.0, wall_h) for i in range(len(XY))]

walls, segs = [], []
for p, q, z0, seg_h in SEGS:
    L = float(np.linalg.norm(q - p))
    if L < 1e-3:
        continue
    # ★ 两头各伸长半个墙厚: 每面墙的盒子长度本来正好等于线段长度, 相邻两面在共用角点处
    #   刚好各自截断, 外侧就缺一块楔形 —— 32 个角就是 32 道竖缝。伸长后相邻盒子在角上
    #   互相重叠, 缝补上; 代价是内侧多一点点交叠, 看不出来。
    box = trimesh.creation.box(extents=[L + a.thick, a.thick, seg_h])
    yaw = float(np.arctan2(*(q - p)[::-1]))
    T = np.eye(4)
    T[:3, :3] = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.]])
    T[:3, 3] = np.r_[(p + q) / 2, z0 + seg_h / 2]
    box.apply_transform(T)
    # ★ 光把 face_colors 的 alpha 设成 140 是不够的: 导出 glTF 时会变成一个 alphaMode
    #   默认 OPAQUE 的材质, 渲染器(pyrender / Blender / three.js)一律当不透明画, 看起来
    #   就是一堵白墙。必须显式给 PBRMaterial 并写 alphaMode="BLEND"。
    box.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=[150/255, 150/255, 155/255, a.alpha/255],
        metallicFactor=0.0, roughnessFactor=0.9,
        alphaMode="BLEND", doubleSided=True))
    walls.append(box)
    segs.append(dict(a=p.tolist(), b=q.tolist(), z0=float(z0), h=float(seg_h), thick=a.thick))
print(f"[墙] {len(walls)} 面, 厚 {a.thick}m, 不透明度 {a.alpha}/255, 来源 "
      + ("SpatialLM" if a.walls_txt else f"点云轮廓 (高 {wall_h:.2f}m)"))

# ---------- 3b. 柱子: 量到的截面挤出成地板到天花板的方柱 ----------
# ★ 2026-09-22 office: 柱子不走生成(用户拍板), 截面由 columns_fit.py 量, 这里负责补全和盖。
#   只拍到一面的: 按看到的那面宽度补成方柱, 补在相机看不到的一侧; 补到碰墙就停 ——
#   贴墙的柱子"看不到的那一侧"就是墙里面, 不停的话会补进墙上的窗户(试验时 66 升)。
#   墙的位置到这一步才有, 所以补全放在这里而不是 columns_fit。
#   材质和墙一样(灰色半透明)。节点名 pillar_N: scene_phys 按名字把它当钉死的物体, 撞上它只推别人。
pillars, pillar_rec = [], []
if a.columns:
    _cj = json.load(open(a.columns)).get("柱子", [])
    _cnt = XY.astype(np.float32).reshape(-1, 1, 2)
    def _to_wall(p0, d):
        """从 p0 沿单位向量 d 走到轮廓(墙)要多远。"""
        best = np.inf
        for i in range(len(XY)):
            e0, e1 = XY[i], XY[(i + 1) % len(XY)]; e = e1 - e0
            M2 = np.array([[d[0], -e[0]], [d[1], -e[1]]])
            if abs(np.linalg.det(M2)) < 1e-9: continue
            t_, u_ = np.linalg.solve(M2, e0 - p0)
            if t_ > 0 and 0 <= u_ <= 1: best = min(best, t_)
        return best
    for col in _cj:
        lo_c, hi_c = np.array(col["lo"], float), np.array(col["hi"], float); lo2, hi2 = lo_c.copy(), hi_c.copy()
        note = "几面都拍到, 按量到的"
        if not col["both"]:
            th, sg = int(col["thin"]), float(col["unseen"]); ctr = (lo_c + hi_c) / 2
            face = hi_c[th] if sg < 0 else lo_c[th]           # 相机看到的那面
            d = np.zeros(2); d[th] = sg; p0 = ctr.copy(); p0[th] = face
            room = _to_wall(p0, d)
            depth = max(min(col["wide"], room), hi_c[th] - lo_c[th])
            if sg < 0: lo2[th] = face - depth
            else:      hi2[th] = face + depth
            note = f"只拍到一面, 沿 {'xy'[th]} 补到 {depth:.2f} m" + (" (碰到墙停下)" if room < col["wide"] else "")
        # ★ 贴着墙的不盖 (2026-09-22 用户拍板): 墙已经挡在那儿了, 再盖一根只会和墙上的窗/门撞。
        #   office 的 6 号"柱子"其实是两扇窗之间的一段墙, 盖出来和窗户重叠 55 升, 物理还推不开(窗户最多推 15 cm)。
        #   判据用**补全之后**的截面: 量到的往往只有朝屋里那一面, 离墙还有半米, 光看它认不出来。
        _d = min(abs(cv2.pointPolygonTest(_cnt, (float(x), float(y)), True))
                 for x in (lo2[0], hi2[0]) for y in (lo2[1], hi2[1]))
        if _d < a.column_wall_gap:
            print(f"[柱子] {col['ids']} 补完贴着墙(离墙 {_d:.2f} m), 算墙的一部分, 不盖"); continue
        cc = (lo2 + hi2) / 2
        if cv2.pointPolygonTest(_cnt, (float(cc[0]), float(cc[1])), False) < 0:
            print(f"[柱子] {col['ids']} 在房间外, 不盖"); continue
        box = trimesh.creation.box(extents=[hi2[0] - lo2[0], hi2[1] - lo2[1], wall_h])
        box.apply_translation([cc[0], cc[1], wall_h / 2])
        box.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[150/255, 150/255, 155/255, a.alpha/255],
            metallicFactor=0.0, roughnessFactor=0.9, alphaMode="BLEND", doubleSided=True))
        pillars.append(box)
        pillar_rec.append(dict(ids=col["ids"], lo=lo2.tolist(), hi=hi2.tolist(), h=float(wall_h), 量到=[col["lo"], col["hi"]], 补=note))
        print(f"[柱子] pillar_{len(pillars) - 1} <- {col['ids']}: {np.round(hi2 - lo2, 2).tolist()} m, {note}")
    print(f"[柱子] {len(_cj)} 根量到的 -> 盖了 {len(pillars)} 根 (高 {wall_h:.2f} m, 和墙一样)")

# ---------- 4. 组装 + 坐标系 ----------
scene = trimesh.Scene()
if a.floor_glb and os.path.exists(a.floor_glb):
    fs = trimesh.load(a.floor_glb, force="scene")
    for g in fs.geometry.values():
        scene.add_geometry(g, node_name="floor")
    print(f"[地板] 改用有贴图的 {os.path.basename(a.floor_glb)}")
else:
    scene.add_geometry(floor, node_name="floor")
for i, w in enumerate(walls):
    scene.add_geometry(w, node_name=f"wall_{i}")
for i, pl in enumerate(pillars):
    scene.add_geometry(pl, node_name=f"pillar_{i}")

# ★ 两条链的物体在哪个系, 壳就得在哪个系, 否则合起来物体会沉到地板下面 (实测差 0.75m,
# 而且水平方向还差一个相似变换 —— zup 系是 p_zup = s·R·p_da3 + t 变过来的)。
#   LM 链   物体由 hybrid_place 摆在 zup 系   -> 不给 --zup, 壳留在 zup 系
#   obj 链  物体由 assemble_scene 摆在 DA3 系 -> 给 --zup, 壳变回 DA3 系
if a.zup:
    zt = json.load(open(a.zup))
    s_, R_, t_ = float(zt.get("s", 1.0)), np.array(zt["R"], float), np.array(zt["t"], float)
    M = np.eye(4)                       # p_da3 = R.T @ (p_zup - t) / s
    M[:3, :3] = R_.T / s_
    M[:3, 3] = -(R_.T @ t_) / s_
    scene.apply_transform(M)
    print(f"[坐标] zup -> DA3 世界系 (s={s_:.4f})")

if a.up == "y":
    if a.zup:
        # ★ 用 world_frame 的同一份实现。这里原本是第 12 份手抄的罗德里格斯 ——
        #   而房间壳是在 assemble_scene 翻轴**之后**并进整场景的, 两边旋转必须逐位相同。
        #   自己抄一份 + 静默兜底 [0,-1,0] 的话, --world-up 缺失时家具对、墙歪,
        #   和当初"楼歪了"是同一类事故。
        UP, _GH, _FB = load_up(os.path.dirname(a.world_up) if a.world_up
                               else os.path.dirname(a.zup or "."), "房间壳")
        R3 = export_R(UP)
    else:
        R3 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.]])   # z 朝上 -> y 朝上
    M = np.eye(4); M[:3, :3] = R3
    scene.apply_transform(M)

scene.export(a.out)
seg_out = a.seg_out or os.path.splitext(a.out)[0] + "_墙线.json"
json.dump(dict(up=a.up, 墙高=wall_h, 厚=a.thick, 面积=float(cv2.contourArea(c) * a.res ** 2),
               轮廓=XY.tolist(), 墙线=segs, 墙来源=("spatiallm:" + a.walls_txt) if a.walls_txt else "点云轮廓",
               **({"柱子": pillar_rec} if pillar_rec else {})),
          open(seg_out, "w"), ensure_ascii=False, indent=1)
print(f"[done] {a.out}\n       {seg_out}  (墙线在 z 朝上系, 物理那步自己转)")
