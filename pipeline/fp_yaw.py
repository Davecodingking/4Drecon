"""FP 定朝向(流水线 fpose 那一步, 在 fpose 环境里跑): 组装之后, 每个物体只重定水平朝向, 位置和大小不动。

每个物体(和 FoundationPose/ours/flip_eval.py --mode yaw 那次实验同一套做法):
  1. 原摆放(07_scene_摆放.glb)里取出网格, 转回 DA3 世界系; 用生成时那一帧(06_solo.log 里"第 N 帧")的相机、DA3 深度、SAM3 抠图
  2. 对齐: 朝向不动, 按这一帧看得到的深度点把位置和大小贴紧 —— 只给 FP 当起点(FP 裁小块比对, 起点偏了就比错)
  3. 24 个候选 = 绕天每 15° 一个, 位置都用对齐后的; FP 修 5 轮再打分, 取第一名
  4. 只取第一名"绕天转了几度", 在 07_scene.glb 里让这个物体绕自己盒子中心转过去。对齐和 FP 挪的位置都不用
找不到生成帧 / 相机 / 抠图的物体不转; 生成帧里露出不到 --min-vis 的也不转(被挡得多, FP 是在猜)。原样留着, 07_fp_yaw.json 里写原因。
用法: python fp_yaw.py --r1 <轮次目录> --fr <frames_<run> 目录> [--nyaw 24] [--only 词_号,...]"""
import sys, os, re, json, time, argparse, logging, warnings
warnings.filterwarnings("ignore")
SAM3D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{SAM3D}/FoundationPose"); sys.path.insert(0, f"{SAM3D}/data/outdoor/scripts"); sys.path.insert(0, f"{SAM3D}/pipeline")
from estimater import *                   # FoundationPose: trimesh/np/torch/cv2/nvdiffrast 都从这里来
from pycocotools import mask as mask_util
from world_frame import load_up, inverse_T
import fp_yaw_glb as G
import estimater as _E, contextlib
logging.getLogger().setLevel(logging.WARNING)

# ★ 提速, 不改 FP 的代码(2026-10-28 实测一个物体 5.4 秒 -> 见 FoundationPose/NOTES.md 坑 12):
#   1. 建 FoundationPose 时它在 CPU 上算 1 万个点两两距离取最大当直径, 每个物体 3.8 秒。同一批点、改到显卡上算
#   2. 它每个物体往 /tmp 存一份带贴图的 obj(0.7 秒), 用不上也从不删 —— 跑了几天堆了 444 MB。建对象时不让它存
def _diameter_gpu(model_pts=None, mesh=None, n_sample=1000):
    ids = np.random.choice(len(model_pts), size=min(n_sample, len(model_pts)), replace=False)   # 和原版同样抽点
    p = torch.as_tensor(model_pts[ids], dtype=torch.float64, device="cuda")
    return np.float64(max(torch.cdist(p[i:i + 2000], p).max().item() for i in range(0, len(p), 2000)))   # 和原版同一个类型, 下游拿它乘张量
_E.compute_mesh_diameter = _diameter_gpu

@contextlib.contextmanager
def _no_tmp_obj():
    ex = trimesh.Trimesh.export; trimesh.Trimesh.export = lambda self, *a, **k: None
    try: yield
    finally: trimesh.Trimesh.export = ex

ap = argparse.ArgumentParser(); ap.add_argument("--r1", required=True); ap.add_argument("--fr", required=True)
ap.add_argument("--nyaw", type=int, default=24); ap.add_argument("--only", default="")
ap.add_argument("--min-vis", type=float, default=0.5, help="生成帧里露出不到这个比例就不转(FP 看到的太少, 转的角度不可靠)")
a = ap.parse_args(); R1 = a.r1; DA3 = f"{R1}/02_da3"; SHEETS = f"{R1}/07_fp_yaw"; os.makedirs(SHEETS, exist_ok=True)
t_all = time.time()

cams = json.load(open(f"{DA3}/cameras.json"))["cameras"]
UP, _, _ = load_up(DA3, "fp 定朝向"); UP = np.asarray(UP, float); UP /= np.linalg.norm(UP)
INV = inverse_T(UP)
gen = {}     # "[office chair#8] 第59帧 占3.30% (68346px) -> off_r1_office_chair_8"
for l in open(f"{R1}/logs/06_solo.log"):
    m = re.match(r"\[(.+)#(\d+)\] 第(\d+)帧", l)
    if m: gen[(m.group(1), int(m.group(2)))] = int(m.group(3))

def mask_of(word, oid, frame, hw):
    p = f"{R1}/03_masks/{word.replace(' ', '_')}.npz"
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=True); meta, counts, (H, W) = z["meta"], z["counts"], z["hw"]
    i = np.where((meta["oid"] == oid) & (meta["frame"] == frame))[0]
    if not len(i): return None
    m = mask_util.decode({"counts": counts[i[0]], "size": [int(H), int(W)]}).astype(np.uint8)
    return cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

def align_ts(P, verts, dep_r, depth, mask, K, iters=30):
    """朝向不动, 只调位置和大小, 让网格看得到的那部分贴上抠图里的深度点。返回新位姿(不带缩放)。"""
    from scipy.spatial import cKDTree
    H, W = depth.shape
    v_cam = verts @ P[:3, :3].T + P[:3, 3]
    uv = v_cam @ K.T; u = np.round(uv[:, 0] / uv[:, 2]).astype(int); v = np.round(uv[:, 1] / uv[:, 2]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    vis = np.zeros(len(verts), bool); vis[ok] = np.abs(dep_r[v[ok], u[ok]] - v_cam[ok, 2]) < 0.03
    ys, xs = np.nonzero(mask & (depth > 0.01))
    if vis.sum() < 30 or len(ys) < 30: return P
    sel = np.random.default_rng(0).choice(len(ys), min(3000, len(ys)), replace=False)
    z = depth[ys[sel], xs[sel]]; Y = np.stack([(xs[sel] - K[0, 2]) * z / K[0, 0], (ys[sel] - K[1, 2]) * z / K[1, 1], z], 1)
    V0 = verts[vis] @ P[:3, :3].T; t = P[:3, 3].copy(); s_ = 1.0
    for _ in range(iters):
        V = V0 * s_ + t; d, nn = cKDTree(V).query(Y); keep = d < np.percentile(d, 90)
        t = t + (Y[keep] - V[nn[keep]]).mean(0)
        A = V0[nn[keep]]; s_ = float(np.clip(((Y[keep] - t) * A).sum() / max(1e-9, (A * A).sum()), 0.8, 1.25))
    Q = P.copy(); Q[:3, 3] = t          # ★ FP 的网格不能带缩放, 只取位置; 大小差在 0.8~1.25 内由修正网络消化
    return Q

def rgb8(x): return (x.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

src = G.source(R1)                         # 组装的原摆放(重跑也从它转)
sc = trimesh.load(src)
skip = re.compile(r"^(wall|floor|pillar|ground|column)(_\d+)?$")
only = set(x for x in a.only.split(",") if x)
glctx = dr.RasterizeCudaContext(); scorer = ScorePredictor(); refiner = PoseRefinePredictor()
yaws, rows = {}, []
for node in sc.graph.nodes_geometry:
    if skip.match(node) or (only and node not in only): continue
    word, oid = node.rsplit("_", 1)[0].replace("_", " "), int(node.rsplit("_", 1)[1])
    why = None
    if (word, oid) not in gen: why = "06_solo.log 里找不到生成帧"
    else:
        f = gen[(word, oid)]; fn = f"{f:04d}.jpg"; cam = cams.get(fn)
        if cam is None: why = f"第 {f} 帧没有相机"
    if why is None:
        K = np.array(cam["intrinsic_3x3"], float); w2c = np.array(cam["extrinsic_w2c_4x4"], float).reshape(4, 4)
        depth = np.load(f"{DA3}/depth/{f:04d}_depth.npy").astype(np.float32); H, W = depth.shape
        rgb = cv2.resize(cv2.imread(f"{a.fr}/{fn}")[..., ::-1], (W, H), interpolation=cv2.INTER_AREA)
        mask = mask_of(word, oid, f, (H, W))
        if mask is None or mask.sum() < 20: why = f"第 {f} 帧没有抠图"
    if why:
        print(f"[{node}] 不转: {why}"); rows.append(dict(物体=node, 转了=None, 原因=why)); continue

    t0 = time.time()
    T, g = sc.graph[node]; mesh = sc.geometry[g].copy(); mesh.apply_transform(T); mesh.apply_transform(INV)   # -> DA3 世界系
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals) and not hasattr(mesh.visual.material, "image"):
        mesh.visual.material = mesh.visual.material.to_simple()     # ★ FP 只认普通贴图(见 FoundationPose/NOTES.md 坑 6)
    c = mesh.bounds.mean(0); mesh.vertices -= c                     # 物体系: 原点在盒子中心, 轴和世界系一致
    Tc = np.eye(4); Tc[:3, 3] = c; P0 = w2c @ Tc                     # 原摆放(物体 -> 相机)
    # ★ 每个物体从同一个随机数起点开始: FP 算直径是随机抽点, 不固定的话同一个输入连跑两次, 分不出前后的物体
    #   (office 的 chair_8 / table_4 / plant_stand_1)会在两个朝向之间来回换(2026-10-28 实测)
    np.random.seed(0); torch.manual_seed(0)
    with _no_tmp_obj():
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner,
                             glctx=glctx, debug=0, debug_dir=SHEETS)
    _, d0, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=torch.as_tensor(P0[None], dtype=torch.float, device="cuda"),
                                 glctx=glctx, mesh_tensors=est.mesh_tensors)
    vis = G.visible_frac(mask, d0[0].cpu().numpy() > 0.001)
    if vis < a.min_vis:
        why = f"第 {f} 帧只露出 {vis:.0%}(门槛 {a.min_vis:.0%}), FP 看到的太少"
        print(f"[{node}] 不转: {why}"); rows.append(dict(物体=node, 帧=f, 露出=round(vis, 2), 转了=None, 原因=why)); continue
    Pa = align_ts(P0, mesh.vertices, d0[0].cpu().numpy(), depth, mask, K)
    dpp = bilateral_filter_depth(erode_depth(depth, radius=2, device="cuda"), radius=2, device="cuda")   # 和原版 register 同样的预处理
    dpp = dpp.cpu().numpy() if torch.is_tensor(dpp) else dpp
    cands = np.stack([w2c @ Tc @ G.rot_about(UP, 360.0 * k / a.nyaw) for k in range(a.nyaw)]); cands[:, :3, 3] = Pa[:3, 3]
    Gr, _ = refiner.predict(mesh=est.mesh, mesh_tensors=est.mesh_tensors, rgb=rgb, depth=dpp, K=K, ob_in_cams=cands,
                            normal_map=None, xyz_map=depth2xyzmap(dpp, K), glctx=glctx, mesh_diameter=est.diameter, iteration=5)
    S, _ = scorer.predict(mesh=est.mesh, rgb=rgb, depth=dpp, K=K, ob_in_cams=Gr.cpu().numpy(), normal_map=None,
                          mesh_tensors=est.mesh_tensors, glctx=glctx, mesh_diameter=est.diameter)
    S = np.asarray(S.cpu() if torch.is_tensor(S) else S, float); best = Gr.cpu().numpy()[int(np.argmax(S))]
    yaw, tilt = G.yaw_of(P0, best, w2c, UP); dt = time.time() - t0
    yaws[node] = (yaw, c)
    rows.append(dict(物体=node, 帧=f, 露出=round(vis, 2), 转了=round(yaw, 1), 歪了=round(tilt, 1), 用时秒=round(dt, 2)))
    print(f"[{node}] 第 {f} 帧, 绕天转 {yaw:.0f}°(FP 歪了 {tilt:.0f}°, 不用), {dt:.2f} 秒")

    # 对比图: 照片 | 原摆放 | 转过之后(位置是原摆放的, 和场景里一样)
    Pn = w2c @ Tc @ G.rot_about(UP, yaw)
    cr, dr_, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=torch.as_tensor(np.stack([P0, Pn]), dtype=torch.float, device="cuda"),
                                   glctx=glctx, mesh_tensors=est.mesh_tensors)
    sil = [dr_[i].cpu().numpy() > 0.001 for i in (0, 1)]
    ys, xs = np.nonzero(mask | sil[0] | sil[1]); y0, y1 = max(0, ys.min() - 20), min(H, ys.max() + 20); x0, x1 = max(0, xs.min() - 20), min(W, xs.max() + 20)
    tiles = [rgb] + [np.where(sil[i][..., None], rgb8(cr[i]), (rgb * 0.35).astype(np.uint8)) for i in (0, 1)]
    tiles = [cv2.resize(t[y0:y1, x0:x1], (max(1, int((x1 - x0) * 260 / (y1 - y0))), 260)) for t in tiles]
    body = np.hstack([np.pad(t, ((0, 0), (0, 6), (0, 0))) for t in tiles])[..., ::-1].copy()
    bar = np.zeros((30, body.shape[1], 3), np.uint8)
    cv2.putText(bar, f"{node} (frame {f})   photo | assembled | FP turned {yaw:.0f} deg", (6, 21), 0, 0.6, (0, 255, 255), 1)
    cv2.imwrite(f"{SHEETS}/{node}.jpg", np.vstack([bar, body]))

G.apply_yaws(src, f"{R1}/07_scene.glb", yaws, UP)
G.mark_done(R1, dict(说明="FP 只绕天转(每 15° 一个候选), 只取角度; 位置和大小是组装的原摆放", 原摆放=G.BACKUP,
                     总用时秒=round(time.time() - t_all, 1), 物体=rows))
print(f"转了 {len(yaws)} 个, 没转 {len(rows) - len(yaws)} 个, 共 {time.time() - t_all:.0f} 秒 -> 07_scene.glb(原摆放在 {G.BACKUP})")
