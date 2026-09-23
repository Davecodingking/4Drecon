#!/usr/bin/env python
"""量盒子那步的 3D 查看页: 场景点云 + 每个实例的逐帧点 / 逐帧盒子 / 合并盒子。

  python box_fit_view.py --r1 ~/sam3d/data/indoorOutput/office/r1
  python box_fit_view.py --r1 ~/sam3d/data/outdoor/result/pedestrian/r2sim_vggt
出两样: <r1>/10_box_fit_view.json(闸口网页 3D tab 读的数据, 见 scene_view.py; --data off 不出)
        <r1>/10_box_fit_view.html(单文件, 浏览器直接开, 要发给人用它; --out off 不出)
drive.py 在闸 3/4/5 自动跑 --out off(ensure_box_view); 手动要单文件就自己跑。

看什么:
  每个实例取面积最大的 8 帧(和 boxes_from_masks.py 的 SAMPLE_FRAMES 同一个取法), 每帧单独
  反投影量一个盒子(橙), 8 帧合起来量一个(蓝)。静止的东西 8 个橙盒子应该叠在一起;
  叠不上 = 东西在动(人/车), 或者 mask / 深度在帧间漂, 或者同一个 track 跳到了别的物体上。
  spread = 8 帧质心之间最大的水平距离, 列表按它排。

★ 读 06_all_boxes_merged.json(召回并进来之后的), 没有才退回 06_all_boxes.json。
  第一版只读 06_all_boxes.json, 召回补回来的实例一个都不在列表里 —— office 的
  laptop stand 就是这么"消失"的: SAM 3 跟踪 0 个, 只有召回捡回来了。
  召回实例在 03_masks 里没有 mask(只有 14_recall_masks 下的叠图 jpg), 所以只画盒子(虚线),
  没有点、没有逐帧盒子、没有 spread。

★ 反投影 / 量盒子直接执行 boxes_from_masks.py 的前半段拿 backproj / fit_box ——
  不自己再写一份, 否则这页量出来的和流水线量出来的会悄悄不一样。

★ 默认会自动转正(--relevel auto): 流水线的天可能歪(office 歪了 9.45°), 页面先从点云里找真地面再画。

sam3d 环境, 纯 CPU。模板在同目录 box_fit_view_template.html。
"""
import argparse, base64, collections, contextlib, io, json, os, sys
import numpy as np, cv2
from pycocotools import mask as mu

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.expanduser("~/sam3d/data/outdoor/scripts")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--r1", required=True, help="一轮结果目录(底下有 02_da3/ 03_masks/ 06_all_boxes*.json)")
ap.add_argument("--fr", default=None, help="帧目录; 不给就用 cameras.json 里记的 source(就是 da3 当时吃的那套)")
ap.add_argument("--indoor", action="store_true", default=None, help="室内阈值; 不给就看路径里有没有 indoorOutput")
ap.add_argument("--conf-min", type=float, default=None, help="深度置信度下限; 默认 室内 1.2 / 室外 2.0(同两条链的 CONF)")
ap.add_argument("--ctx-depth", type=float, default=None, help="背景点只要这么近的(米); 默认 室内 12 / 室外 35")
ap.add_argument("--ctx-points", type=int, default=140000, help="背景点上限")
ap.add_argument("--per-frame", type=int, default=600, help="每个实例每帧最多留多少点")
ap.add_argument("--out", default=None, help="单文件 html; 默认 <r1>/10_box_fit_view.html, off = 不出")
ap.add_argument("--data", default=None,
                help="数据 json(闸口网页的 3D tab 读它, tools/scene_view.py); 默认 <r1>/10_box_fit_view.json, off = 不出")
ap.add_argument("--title", default="Box Fit View")
ap.add_argument("--scene-glb", default="07_scene_phy.glb", help="用哪个 glb 判断\"最终进场景\"")
ap.add_argument("--relevel", choices=["auto", "off"], default="auto",
                help="auto: 从场景点云里找真地面, 歪了超过 --relevel-min 度就整体转正(见文件头); off: 按流水线的天原样画")
ap.add_argument("--relevel-min", type=float, default=2.0)
ap.add_argument("--body", default=None,
                help="SAM 3D Body 那一轮(sam-body4d/outputs/<场景>/bN); 不给就读 09_body4d_report.json 里记的; 'off' = 不画人")
ap.add_argument("--body-pts", type=int, default=800, help="每人每帧画多少个人体顶点")
ap.add_argument("--frustum", type=float, default=None, help="相机视锥画多深(米); 默认 室内 0.25 / 室外 1.0")
a = ap.parse_args()

R1 = os.path.abspath(os.path.expanduser(a.r1))
indoor = a.indoor if a.indoor is not None else "/indoorOutput/" in R1
conf_min = a.conf_min if a.conf_min is not None else (1.2 if indoor else 2.0)
ctx_depth = a.ctx_depth if a.ctx_depth is not None else (12.0 if indoor else 35.0)
OUT  = None if a.out  == "off" else (a.out  or f"{R1}/10_box_fit_view.html")
DATA = None if a.data == "off" else (a.data or f"{R1}/10_box_fit_view.json")

# ---- 借流水线的反投影和量盒子 ----
os.environ["R1"] = R1
os.environ["CONF_MIN"] = str(conf_min)
src = open(f"{SCRIPTS}/boxes_from_masks.py").read().split("# 前置的二维跨词去重")[0]
g = {"__file__": f"{SCRIPTS}/boxes_from_masks.py", "__name__": "bfm"}
sys.path.insert(0, SCRIPTS)
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "bfm", "exec"), g)
backproj, fit_box, UP, HAVE, cams, DA3 = g["backproj"], g["fit_box"], g["UP"], g["HAVE"], g["cams"], g["DA3"]
FR = a.fr or json.load(open(f"{DA3}/cameras.json")).get("source")
if not FR or not os.path.isdir(FR):
    raise SystemExit(f"[失败] 找不到帧目录 {FR!r}, 用 --fr 指定")

# 显示坐标: 天朝 +y, 地面 y=0, 首帧朝向大致朝 -z
GH = json.load(open(f"{DA3}/world_up.json"))["ground_h"]
F = np.array([0, 0, 1.0]); F = F - UP * (F @ UP); F /= np.linalg.norm(F)
Y = UP; Z = -F; X = np.cross(Y, Z)
def tw(P): return np.stack([P @ X, P @ Y - GH, P @ Z], -1)
def b64(x): return base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()
def corners(b):
    """8 个角, 顺序 = ix*4 + iy*2 + iz(模板里的连线和点选三角形都按这个顺序)。"""
    c = np.array(b["中心"]); a1 = np.array(b["长轴"]); a2 = np.cross(UP, a1); L, W, H = b["尺寸"]
    out = [c + a1 * L * sx + a2 * W * sy + UP * H * sz for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)]
    return np.round(tw(np.array(out)), 3).tolist()
rng = np.random.default_rng(0)
_masks = {}
def load(word):
    fn = f"{R1}/03_masks/{word.replace(' ', '_')}.npz"
    if word not in _masks:
        _masks[word] = None
        if os.path.exists(fn):
            z = np.load(fn, allow_pickle=True); _masks[word] = (z["meta"], z["counts"], z["hw"])
    return _masks[word]

bp = f"{R1}/06_all_boxes_merged.json"
if not os.path.exists(bp):
    bp = f"{R1}/06_all_boxes.json"
boxes = json.load(open(bp))
print(f"[盒子] {os.path.basename(bp)}: {sum(len(v) for v in boxes.values())} 个实例, {len(boxes)} 个词")

insts, skipped = [], collections.Counter()
for word in sorted(boxes, key=lambda w: -len(boxes[w])):
    mk = load(word)
    for b in boxes[word]:
        oid = int(b["obj_id"])
        rec = b.get("来源") == "召回"
        rows = []
        if mk is not None and not rec:
            meta, counts, (H, W) = mk
            rows = [(int(ar), int(fi), i) for i, (o, fi, ar, e) in enumerate(meta)
                    if int(o) == oid and ar > 0 and int(fi) in HAVE]
        if not rows:
            # 召回的(或者 mask 没了的): 只画盒子
            insts.append(dict(word=word, id=oid, recall=True, seen=int(b.get("用帧", 0)),
                              frames=[], pts="", fid="", n=0, single=[], cent=[],
                              box=corners(b), dims=[round(x, 2) for x in b["尺寸"]], sdims=[], spread=None))
            skipped["召回/无 mask"] += 1
            continue
        tracked = sorted(f for _, f, _ in rows)
        rows.sort(key=lambda t: -t[0]); sel = rows[:8]
        frames = []; pts = []; fid = []; single = []; cent = []; full = []
        for ar, fi, i in sorted(sel, key=lambda t: t[1]):
            m = mu.decode({"counts": counts[i], "size": [int(H), int(W)]}).astype(bool)
            P = backproj(m, fi)
            if P is None: continue
            full.append(P); k = len(frames); frames.append(fi)
            sb = fit_box(P); single.append(dict(c=corners(sb), dims=[round(x, 2) for x in sb["尺寸"]]))
            cent.append(np.round(tw(P.mean(0)[None])[0], 3).tolist())
            Q = P[rng.choice(len(P), min(len(P), a.per_frame), replace=False)]
            pts.append(tw(Q)); fid.append(np.full(len(Q), k, np.uint8))
        if not full:
            skipped["反投影全空"] += 1; continue
        ch = np.array(cent)[:, [0, 2]]
        spread = float(max(np.linalg.norm(p - q) for p in ch for q in ch)) if len(ch) > 1 else 0.0
        sd = np.median(np.array([s["dims"][:2] for s in single]), 0)
        P = np.concatenate(pts)
        # ★ 蓝盒子用流水线自己的那个(06 里的), 不在这里重量: 要看的就是"流水线量出来的"
        #   和"逐帧量出来的"差多少, 自己重量一遍就把前者换掉了。
        insts.append(dict(word=word, id=oid, recall=False, frames=frames,
                          tracked=[tracked[0], tracked[-1], len(tracked)],
                          pts=b64(np.round(P * 100).astype(np.int16)), fid=b64(np.concatenate(fid)), n=len(P),
                          single=single, cent=cent, box=corners(b), dims=[round(x, 2) for x in b["尺寸"]],
                          sdims=[round(float(x), 2) for x in sd], spread=round(spread, 2)))
# ---- 哪些最后真的进了场景 ----
# ★ 以 glb 的节点为准, 不靠推算: 组装那步有 EXCLUDE / 人工否掉 / VLM 否决 / 尺寸门槛好几道,
#   自己按规则重推一遍迟早和实际对不上。节点名 = 词(空格换下划线)_obj_id, trimesh 重名会加 .N。
#   没进的原因: 优先读组装那步落的 07_scene_skipped.json(完整名单); 老轮次没有这个文件,
#   就从 07_assemble.log 最后一段里抠 —— 但日志每条原因只打前 5 个, 抠不全的标"原因没记全"。
import ast, re
glb = f"{R1}/{a.scene_glb}"
placed = None
if os.path.exists(glb):
    import trimesh
    from world_frame import inverse_T
    _sc = trimesh.load(glb)
    placed = {n.rsplit(".", 1)[0] for n in _sc.graph.nodes_geometry}
    # 物理之后物体真正在哪: 直接量 glb 里那块网格。盒子用的还是 fit_box, 和量盒子那步同一把尺,
    # 所以"测出来的盒子"和"物理之后的盒子"可以直接比。
    # ★ glb 是 DA3 世界系乘过 export_T(UP) 的, 要先用 inverse_T 转回 DA3, 再走 tw() 进显示坐标,
    #   否则会和点云差一个绕天的旋转。
    _INV = inverse_T(UP)[:3, :3]        # = export_R 的转置; 行向量要右乘它的转置
    _phys = {}
    for node in _sc.graph.nodes_geometry:
        base = node.rsplit(".", 1)[0]
        if base in _phys: continue
        try:
            Tn, gn = _sc.graph[node]
            V = trimesh.transform_points(np.asarray(_sc.geometry[gn].vertices, float), Tn) @ _INV.T
            _phys[base] = fit_box(V)
        except Exception:
            pass
    why = {}; excl = set()
    sk = f"{R1}/{a.scene_glb.replace('_phy', '')[:-4]}_skipped.json"
    if os.path.exists(sk):
        j = json.load(open(sk)); excl = set(j.get("exclude", []))
        for x in j["skipped"]: why[(x["word"], int(x["obj_id"]))] = x["why"]
        src_why = os.path.basename(sk)
    else:
        src_why = "07_assemble.log(日志只列每类前 5 个)"
        lg = f"{R1}/logs/07_assemble.log"
        if os.path.exists(lg):
            txt = open(lg, encoding="utf-8", errors="replace").read()
            m = list(re.finditer(r"排除了 (\[.*?\])", txt))
            if m: excl = set(ast.literal_eval(m[-1].group(1)))
            k = txt.rfind("[跳过] 共")
            for line in (txt[k:].splitlines()[1:] if k >= 0 else []):
                mm = re.match(r"\s+(\S.*?)\s+\d+ 个: (\[.*\])\s*$", line)
                if not mm: break
                try:
                    for it in ast.literal_eval(mm.group(2)): why[(it[0], int(it[1]))] = mm.group(1).strip()
                except Exception:
                    pass
        # 日志抠不全的, 按组装那步自己的判据补: 人工否掉 / VLM 否决 / 没有 mesh, 读的是同一批文件
        try:
            import subprocess as _sp
            _pipe = "indoor_pipeline.py" if indoor else "outdoor_pipeline.py"
            _args = [sys.executable, os.path.expanduser(f"~/sam3d/pipeline/{_pipe}"),
                     "--scene", R1.rstrip("/").split("/")[-2], "--run", R1.rstrip("/").split("/")[-1],
                     "--dump-cfg", "--tag-only"] + (["--chain", "obj"] if indoor else [])
            TAG = _sp.run(_args, capture_output=True, text=True).stdout.strip().splitlines()[-1]
            import review_decisions as _rv
            SKIPN = set(_rv.skip_names(R1)); VREJ = _rv.vlm_rejected(R1); VREL = _rv.vlm_released(R1)
            gname = lambda w, oid=None: (f"{TAG}_{w}" if oid is None else f"{TAG}_{w}_{oid}").replace(" ", "_")
            for w in boxes:
                for b in boxes[w]:
                    k2 = (w, int(b["obj_id"]))
                    if k2 in why: continue
                    if gname(w) in SKIPN or gname(w, b["obj_id"]) in SKIPN: why[k2] = "人工否掉"
                    elif b["obj_id"] in VREJ.get(w, ()) and gname(w, b["obj_id"]) not in VREL: why[k2] = "VLM否决(人没放行)"
            src_why += f" + 闸口决策/VLM 名单(TAG={TAG})"
        except Exception as e:
            print(f"[场景] 补原因失败, 只用日志: {e}")
    WHY_EN = {"人工否掉": "rejected at review", "VLM否决(人没放行)": "rejected by VLM check",
              "绝对": "below the absolute size floor", "类内偏小": "much smaller than others of its word",
              "人工否掉(借的是被否的mesh)": "its borrowed mesh was rejected", "没有 mesh": "no mesh was generated"}
    for x in insts:
        nm_ = f"{x['word'].replace(' ', '_')}_{x['id']}"
        x["inscene"] = nm_ in placed
        pb = _phys.get(nm_)
        if pb is not None:
            x["pbox"] = corners(pb); x["pdims"] = [round(v, 2) for v in pb["尺寸"]]
            x["moved"] = round(float(np.linalg.norm(np.array(x["box"]).mean(0) - np.array(x["pbox"]).mean(0))), 3)
        if x["inscene"]:
            x["why"] = None
        elif x["word"] in excl:
            x["why"] = "word is excluded"
        elif (x["word"], x["id"]) in why:
            w0 = why[(x["word"], x["id"])]; x["why"] = WHY_EN.get(w0, w0)
        else:
            x["why"] = "skipped at assembly (most likely no mesh was generated)"
    _mv = sorted((x["moved"] for x in insts if "moved" in x), reverse=True)
    print(f"[场景] {a.scene_glb}: 进了 {sum(x['inscene'] for x in insts)} 个, 没进 {sum(not x['inscene'] for x in insts)} 个; 原因来自 {src_why}")
    if _mv:
        print(f"[物理] 量到物理后位置的 {len(_mv)} 个, 位移中位 {_mv[len(_mv)//2]:.2f}m, 最大 {_mv[0]:.2f}m")
else:
    print(f"[场景] 没有 {a.scene_glb}, 不分最终版")
nm = sum(not x["recall"] for x in insts); nr = sum(x["recall"] for x in insts)
print(f"[实例] 量过的 {nm} 个, 只有盒子的 {nr} 个" + (f"  {dict(skipped)}" if skipped else ""))

# ---- 背景点云: 抽帧反投影, 抠掉人(人在走, 留着会糊成一道道拖影), 带原色 ----
byf = {}
pk = load("person")
if pk is not None:
    pm, pc, (PH, PW) = pk
    for i, (o, fi, ar, e) in enumerate(pm):
        if ar > 0: byf.setdefault(int(fi), []).append(i)
have = sorted(HAVE); step = max(1, len(have) // 30); ctx = have[::step]
CP, CC = [], []
for fi in ctx:
    fn = f"{fi:04d}.jpg"
    dep = np.load(f"{DA3}/depth/{fi:04d}_depth.npy"); cnf = np.load(f"{DA3}/depth/{fi:04d}_conf.npy")
    h, w = dep.shape
    img = cv2.imread(f"{FR}/{fn}")
    if img is None: continue
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (w, h))
    bad = np.zeros((h, w), bool)
    if byf.get(fi):
        bm = np.zeros((int(PH), int(PW)), np.uint8)
        for i in byf[fi]: bm |= mu.decode({"counts": pc[i], "size": [int(PH), int(PW)]}).astype(np.uint8)
        bad = cv2.resize(cv2.dilate(bm, np.ones((15, 15), np.uint8)), (w, h),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    s = 3 if indoor else 5
    v, u = np.mgrid[0:h:s, 0:w:s]; d = dep[v, u]
    ok = (cnf[v, u] >= conf_min) & (d > 0) & (d < ctx_depth) & (~bad[v, u])
    v, u, d = v[ok], u[ok], d[ok]; K = np.array(cams[fn]["intrinsic_3x3"])
    P = np.stack([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d], 1)
    c2w = np.linalg.inv(np.array(cams[fn]["extrinsic_w2c_4x4"]))
    CP.append(tw((c2w[:3, :3] @ P.T).T + c2w[:3, 3])); CC.append(img[v, u])
CP = np.concatenate(CP); CC = np.concatenate(CC)
keep = rng.choice(len(CP), min(len(CP), a.ctx_points), replace=False); CP, CC = CP[keep], CC[keep]

# ---- 转正: 流水线的"天"可能是错的, 画之前先找真地面 ----
# ★ 2026-09-21 office 上发现的: world_up.py 在这个场景里没找到地板(几乎全被桌子挡住),
#   锁到了家具顶面上, 天歪了 9.45°。按它画出来整个房间是斜的, 盒子也全沿着歪的竖直方向量。
#   另一个 session 先在页面上手工转正过一版; 这里把同一件事做成自动的:
#   在背景点里 RANSAC 找"最低的一大片近水平面"(法向离 +y 不超过 25°, 平面下方几乎没有点),
#   内点再做一次 SVD 精修。office 上和那一版手工转正的结果差 0.15°。
#   找到之后: 整体刚体旋转把地面转平、平移到 y=0; 每个盒子(蓝/橙)再绕自己的中心转回立正
#   (它们是沿旧的天量的, 只做整体旋转会斜着); spread 在真的水平面里重算。
#   盒子尺寸不变 —— 那是流水线量出来的, 这一页只负责把它摆正了看, 不重量。
#   ★ 这只修了这一页的显示。流水线里的高度仍是沿歪的天量的, 要修得去改 world_up.py。
def _rot(u, v):
    """把单位向量 u 转到 v 的最小旋转(罗德里格斯)。"""
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v); w = np.cross(u, v); c = float(u @ v)
    if np.linalg.norm(w) < 1e-9: return np.eye(3)
    k = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)
def find_floor(P, seed=0, iters=4000, cone=25.0, tol=0.04):
    r = np.random.default_rng(seed); best = None; yy = np.array([0, 1.0, 0])
    for _ in range(iters):
        s3 = P[r.choice(len(P), 3, replace=False)]
        n = np.cross(s3[1] - s3[0], s3[2] - s3[0]); L = np.linalg.norm(n)
        if L < 1e-6: continue
        n /= L; n *= np.sign(n[1]) or 1
        if np.degrees(np.arccos(min(1, n @ yy))) > cone: continue
        d = (P - s3[0]) @ n
        if (d < -0.10).mean() > 0.03: continue          # 下面还有一堆点的不是地面
        k = int((np.abs(d) < tol).sum())
        if best is None or k > best[0]: best = (k, n, s3[0])
    if best is None: return None
    Q = P[np.abs((P - best[2]) @ best[1]) < tol]
    n = np.linalg.svd(Q - Q.mean(0))[2][2]; n *= np.sign(n[1])
    return n, Q.mean(0), len(Q)
notice = ""
def T3(x): return np.asarray(x, float)             # 不转正时就是原样
Ru = np.eye(3)
if a.relevel == "auto":
    fl = find_floor(CP)
    if fl is None:
        print("[转正] 没找到像地面的平面, 按流水线的天原样画")
    else:
        n, q0, nin = fl
        tilt = float(np.degrees(np.arccos(min(1, n[1]))))
        print(f"[转正] 找到地面: {nin} 个内点, 和流水线的天差 {tilt:.2f}°")
        if tilt >= a.relevel_min:
            Rf = _rot(n, np.array([0, 1.0, 0]))
            ty = -float((Rf @ q0)[1])
            def T3(x): return (np.asarray(x, float) @ Rf.T) + np.array([0, ty, 0])
            Ru = _rot(Rf @ np.array([0, 1.0, 0]), np.array([0, 1.0, 0]))   # 盒子转回立正
            def upright(c8):
                c8 = T3(c8); cc = c8.mean(0); return np.round((c8 - cc) @ Ru.T + cc, 3).tolist()
            CP = T3(CP)
            for x in insts:
                x["box"] = upright(x["box"])
                if x.get("pbox"): x["pbox"] = upright(x["pbox"])
                if x["recall"]: continue
                x["single"] = [dict(c=upright(sb["c"]), dims=sb["dims"]) for sb in x["single"]]
                x["cent"] = np.round(T3(x["cent"]), 3).tolist()
                P = np.frombuffer(base64.b64decode(x["pts"]), np.int16).reshape(-1, 3) / 100
                x["pts"] = b64(np.round(T3(P) * 100).astype(np.int16))
                ch = np.array(x["cent"])[:, [0, 2]]
                x["spread"] = round(float(max(np.linalg.norm(p - q) for p in ch for q in ch)) if len(ch) > 1 else 0.0, 2)
            notice = (f'      <p class="lede notice"><b>Re-levelled.</b> As exported, this scene stood <b>{tilt:.1f}&deg; off vertical</b>: '
                      "the pipeline's world-up estimate did not lock onto the real floor, so every height was measured along a tilted "
                      "direction. The real floor was found in the points; points, boxes and the camera path were rigidly rotated until "
                      "it lies flat at y&nbsp;=&nbsp;0, and each box was then rolled upright about its own centre. Box sizes are "
                      "the pipeline's own and unchanged; spreads are recomputed in the true horizontal.</p>")
        else:
            print(f"[转正] 偏差不到 {a.relevel_min}°, 不动")

# ---- 相机: 每个相机的位置 + 视锥, 按原视频帧号排 ----
# ★ 按原视频帧号(frame_map)而不是相机键排: 人体那边用的是原视频帧号, 两边要对到同一根时间轴上。
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
from _common import load_frame_map
try:
    fmap = load_frame_map(DA3)                          # {相机键: 原视频帧号}
except Exception as e:
    print(f"[相机] 没有帧号映射({e}), 按相机键的数字当帧号"); fmap = {}
fdep = a.frustum if a.frustum is not None else (0.25 if indoor else 1.0)
camlist = []
for k, cm in cams.items():
    vf = int(fmap.get(k, int(os.path.splitext(k)[0])))
    c2w = np.linalg.inv(np.array(cm["extrinsic_w2c_4x4"], float)); K = np.array(cm["intrinsic_3x3"], float)
    h, w = cm.get("input_hw") or [2 * K[1, 2], 2 * K[0, 2]]
    uv = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    Pc = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(4)] * fdep
    Pw = np.r_[c2w[:3, 3][None], Pc @ c2w[:3, :3].T + c2w[:3, 3]]       # 0 = 光心, 1..4 = 左上 右上 右下 左下
    camlist.append((vf, k, np.round(T3(tw(Pw)), 3).tolist()))
camlist.sort()
camc = [c[2][0] for c in camlist]

# ---- 人: SAM 3D Body 的逐帧人体, 按 body_into_scene 同一套校正摆进来 ----
# ★ 不自己摆: 直接调 body_into_scene 的 load_body / apply_corr, 读 10_people.json 的校正表
#   (远近比例 + 逐帧平移), 和进 glb 的那份人是同一个位置。
# ★ 03_masks 里的 person / pedestrian 实例(量盒子那步当静态物体量的)有人体时就不画了:
#   走动的人被合成一个沿路线拉长的盒子, 那正是要被 SAM 3D Body 换掉的东西。
people = []; body_note = ""
bdir = a.body
if bdir is None and os.path.exists(f"{R1}/09_body4d_report.json"):
    bdir = json.load(open(f"{R1}/09_body4d_report.json")).get("人体")
    if bdir and not os.path.isabs(bdir): bdir = os.path.expanduser(f"~/sam3d/{bdir}")
pj = f"{R1}/10_people.json"
if bdir and bdir != "off":
    if not os.path.isdir(bdir) or not os.path.exists(pj):
        print(f"[人] 跳过: {'没有 ' + bdir if not os.path.isdir(bdir) else '没有 10_people.json(body_into_scene --mode place 还没跑)'}")
    else:
        import body_into_scene as bis
        with contextlib.redirect_stdout(io.StringIO()):
            bpeople, _faces, kept = bis.load_body(bdir)
            _, _, _, _, vid2key = bis.load_scene(R1)
            mesh_run = next(f"{bdir}/{d}" for d in ("04_mesh_occ1", "04_mesh_occ0")
                            if os.path.isdir(f"{bdir}/{d}/mesh_4d_individual"))
            roots = bis.load_roots(mesh_run)
        PJ = json.load(open(pj)); corr = PJ["people"]; rep = PJ.get("报告", {})
        world = bis.apply_corr(bpeople, corr, cams, vid2key, kept, roots)
        nv = len(next(iter(next(iter(world.values())).values())))
        vsel = np.sort(rng.choice(nv, min(nv, a.body_pts), replace=False))   # 每帧同一批顶点, 动起来才连贯
        for pid in sorted(world, key=lambda x: int(x) if x.isdigit() else 1 << 30):
            fr = world[pid]; vfs = sorted(fr)
            bxs, pts, path, dims, tall = [], [], [], [], []
            for vf in vfs:
                V = T3(tw(fr[vf]))                       # 显示坐标: y 朝上, 地面 y=0
                xz = V[:, [0, 2]].astype(np.float32)
                (cx, cz), (rw, rh), ang = cv2.minAreaRect(xz)
                t = np.radians(ang); e1 = np.array([np.cos(t), np.sin(t)]); e2 = np.array([-e1[1], e1[0]])
                if rh > rw: rw, rh, e1, e2 = rh, rw, e2, -e1
                y0, y1 = float(V[:, 1].min()), float(V[:, 1].max())
                c8 = [[cx + e1[0] * rw * sx + e2[0] * rh * sy, (y0 + y1) / 2 + (y1 - y0) * sz, cz + e1[1] * rw * sx + e2[1] * rh * sy]
                      for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)]
                bxs.append(np.round(c8, 3).tolist()); dims.append((rw, rh, y1 - y0))
                pts.append(V[vsel]); path.append([round(float(cx), 3), round(y0, 3), round(float(cz), 3)])
            dims = np.array(dims); P = np.concatenate(pts)
            steps = np.linalg.norm(np.diff(np.array(path)[:, [0, 2]], axis=0), axis=1) if len(path) > 1 else np.zeros(1)
            r = rep.get(pid, {})
            people.append(dict(id=pid, vf=[int(v) for v in vfs], boxes=bxs, path=path,
                               pts=b64(np.round(P * 100).astype(np.int16)), npf=int(len(vsel)),
                               dims=[round(float(x), 2) for x in np.median(dims, 0)],
                               hmax=round(float(dims[:, 2].max()), 2),
                               walk=round(float(steps.sum()), 2), seated=bool(corr[pid].get("seated")),
                               chair=corr[pid].get("chair"), ratio=round(float(corr[pid]["ratio"]), 3)))
        n_old = sum(x["word"] in ("person", "pedestrian") for x in insts)
        insts = [x for x in insts if x["word"] not in ("person", "pedestrian")]
        body_note = f"{len(people)} people from SAM 3D Body" + (f" (replacing {n_old} mask-measured person boxes)" if n_old else "")
        print(f"[人] {bdir}: {len(people)} 人, 帧数 {[len(p['vf']) for p in people]}; 换掉了量盒子那步的 person 实例 {n_old} 个")
elif bdir == "off":
    print("[人] --body off, 不画")

parts = R1.rstrip("/").split("/")
name = f"{parts[-2]}/{parts[-1]}"
data = dict(scene=name, glb=a.scene_glb if placed is not None else None, ctx_frames=len(ctx), total_frames=len(have), frame_max=int(max(have)),
            grid=40 if indoor else 80, ctx_size=0.03 if indoor else 0.045, pt_size=0.04 if indoor else 0.06,
            ctx=dict(pts=b64(np.round(CP * 100).astype(np.int16)), rgb=b64(CC.astype(np.uint8)), n=len(CP)),
            cams=camc, camf=[dict(vf=c[0], key=c[1], p=c[2]) for c in camlist], people=people,
            people_glb=bool(people) and os.path.exists(pj), insts=insts)
nm = sum(not x["recall"] for x in insts)
eyebrow = f"{name} · {nm} measured" + (f" + {nr} recalled" if nr else "") + f" · {len(boxes)} words" + (f" · {len(people)} people" if people else "")
data["页"] = dict(title=a.title, eyebrow=eyebrow, notice=notice)   # 服务端出页时由页面自己填
blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
if DATA:
    open(DATA, "w", encoding="utf-8").write(blob)
    print(f"[数据] {DATA}  {os.path.getsize(DATA)/1e6:.1f} MB   (闸口网页 3D tab 读这个)")
if OUT:
    t = open(f"{HERE}/box_fit_view_template.html", encoding="utf-8").read()
    t = (t.replace("__TITLE__", a.title).replace("__EYEBROW__", eyebrow).replace("__NOTICE__", notice)
          .replace("__DATA__", blob))
    open(OUT, "w", encoding="utf-8").write(t)
    print(f"[输出] {OUT}  {os.path.getsize(OUT)/1e6:.1f} MB")
sp = sorted((x for x in insts if not x["recall"]), key=lambda x: -x["spread"])[:5]
print(f"[背景] {len(CP)} 点, 取自 {len(ctx)} 帧")
print(f"[spread 最大] " + ", ".join(f"{x['word']}#{x['id']} {x['spread']}m" for x in sp))
