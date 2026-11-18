#!/usr/bin/env python
"""人放进场景之后的验收视频: 原视频 | 同一个相机看到的重建场景+人 | 全身追随视角(近景特写也能看清补腿和脚底), 逐帧拼成 mp4。

为什么要它: 4 张静帧看不出人在走动时对不对得上 —— 脚有没有滑、人有没有忽大忽小、
坐着的人是不是一直坐在椅子上; 而特写镜头原机位看不见腿, 第三栏拉远看全身+脚下地面。
人的顶点读摆人那步存下来的 10_people_world.npz(people_store.load_world):
它就是修好的关节参数生成的网格, 交付的 FBX 用的是同一份参数 —— 这里看到的人就是 FBX 里的人。

用法:
  python body_video.py --r1 <场景结果目录> --body <body4d轮次目录> --scene <静态场景 glb> \
                       --fr <场景帧目录> [--out 09_people_check.mp4]
sam3d 环境 (pyrender EGL + trimesh + imageio-ffmpeg)。
"""
import os, sys, json, argparse, subprocess, tempfile, shutil
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np, trimesh, cv2, pyrender
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/sam3d/data/outdoor/scripts"))
import body_into_scene as B
import people_identity, people_store
from world_frame import inverse_T

ap = argparse.ArgumentParser()
ap.add_argument("--r1", required=True); ap.add_argument("--body", required=True)
ap.add_argument("--scene", required=True, help="静态场景 glb (相对 r1), 如 07_scene_phy.glb")
ap.add_argument("--fr", required=True, help="场景那套帧的目录 (文件名和 cameras.json 的键一致)")
ap.add_argument("--out", default="09_people_check.mp4")
ap.add_argument("--tile", type=int, default=640, help="每一栏的宽")
a = ap.parse_args()

_fj = json.load(open(f"{a.body}/00_frames/frames.json"))
da3, cams, UP, GH, vid2key = B.load_scene(a.r1, dense=bool(_fj.get("dense")))
up = np.asarray(UP, float) / np.linalg.norm(UP)
people_store.check_fresh(a.r1, people_identity.source_of(a.body))
world, faces = people_store.load_world(a.r1)                          # DA3 世界系, 和相机同一个系
key2vid = {k: v for v, k in vid2key.items()}
# ★ 2026-11-11 被人拿着的物体(carry 步求的、闸 5 没点不要的)也画上, 红色。以前验收视频只有人和静物, 伞在前端根本看不到。
#   位姿和 motion_export.carry_bundle 同一个约定: DA3 世界 = local_to_world[k] @ mesh_to_local @ glb 顶点。
import carry_decisions
CARRIED = []
for _r in [r for r in carry_decisions.rows(a.r1) if r["生成名"] in set(carry_decisions.want(a.r1))]:
    _P = np.load(f"{a.r1}/{_r['目录']}/carry_poses.npz", allow_pickle=True)
    _m = trimesh.load(str(_P["mesh"]), force="mesh")
    _V = np.c_[_m.vertices, np.ones(len(_m.vertices))] @ np.asarray(_P["mesh_to_local"], float).T
    CARRIED.append(dict(name=_r["生成名"], V=_V, F=np.asarray(_m.faces), M={int(v): np.asarray(M, float) for v, M in zip(_P["vf"], _P["local_to_world"])}))
if CARRIED:
    print(f"[手持物] 画 {len(CARRIED)} 个: {[c['name'] for c in CARRIED]}")

# 静态场景: glb 是 glTF 系(天 +y), 乘 inverse_T 回到 DA3 世界系, 和 render_back 同一套
sc = trimesh.load(f"{a.r1}/{a.scene}")
INV = inverse_T(UP)
scene = pyrender.Scene(bg_color=[0.85, 0.87, 0.9, 1.0], ambient_light=[.5] * 3)
def _alpha(g):
    m = getattr(g.visual, "material", None); f = getattr(m, "baseColorFactor", None)
    return float(f[3]) / (255.0 if max(f) > 1.001 else 1.0) if f is not None and len(f) > 3 else 1.0
items = []
for n in sc.graph.nodes_geometry:
    T, g = sc.graph[n]; m = sc.geometry[g].copy(); m.apply_transform(T); m.apply_transform(INV)
    items.append(m)
for m in sorted(items, key=lambda g: -_alpha(g)):                   # 半透明的墙最后画
    scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))

k0 = sorted(cams)[0]; K0 = np.array(cams[k0]["intrinsic_3x3"])
AR = K0[0, 2] / K0[1, 2]
W = a.tile - a.tile % 2; H = int(round(W / AR)); H -= H % 2
r = pyrender.OffscreenRenderer(W, H)
COL = [(230, 120, 40), (40, 160, 230), (60, 190, 90), (200, 70, 160), (220, 200, 40), (120, 90, 220), (40, 200, 200)]
tmp = tempfile.mkdtemp(prefix="bodyvid_")
if _fj.get("dense"):      # ★ 人体全帧(PEOPLE_FULL=1): 每个人体帧出一帧, 底图用人体那套帧(原视频全帧), 相机用插出来的
    IMG = {int(v): f"{a.body}/00_frames/{i:04d}.jpg" for i, v in enumerate(_fj["kept_indices"])}
    KEYS = [vid2key[int(v)] for v in _fj["kept_indices"] if int(v) in vid2key]
else:
    IMG, KEYS = None, sorted(cams)

# 预计算第三栏"全身追随相机"的平滑轨迹: 始终从原机位侧前方约 22° 拉远包住所有人从头到脚+脚下地面
N_k = len(KEYS)
centers = np.zeros((N_k, 3))
spans = np.full(N_k, 2.0)
view_dirs = np.zeros((N_k, 3))
has_person = np.zeros(N_k, bool)

# 绕 up 旋转 22°(侧前视角: 既能看正面动作, 也能看清脚步前后迈步与台阶接触)
ang = np.radians(22.0)
def _rot_u(v, th):
    c_, s_ = np.cos(th), np.sin(th)
    return v * c_ + np.cross(up, v) * s_ + up * (up @ v) * (1.0 - c_)

for idx_k, key in enumerate(KEYS):
    vf = key2vid.get(key)
    c2w_k = np.linalg.inv(np.array(cams[key]["extrinsic_w2c_4x4"]).reshape(4, 4))
    pts_list = [fr[vf] for _, fr in sorted(world.items(), key=lambda x: int(x[0])) if vf in fr]
    if pts_list:
        all_v = np.concatenate(pts_list, axis=0)
        lo_v, hi_v = all_v.min(0), all_v.max(0)
        ctr = 0.5 * (lo_v + hi_v)
        sp_v = float(np.linalg.norm(hi_v - lo_v))
        has_person[idx_k] = True
    else:
        ctr = c2w_k[:3, 3] + 3.0 * c2w_k[:3, 2]
        sp_v = 2.0
    to_cam = c2w_k[:3, 3] - ctr
    hz = to_cam - (to_cam @ up) * up
    if np.linalg.norm(hz) < 1e-4:
        hz = -c2w_k[:3, 2] + (c2w_k[:3, 2] @ up) * up
    hz = hz / max(np.linalg.norm(hz), 1e-6)
    centers[idx_k] = ctr
    spans[idx_k] = max(sp_v, 1.85)
    view_dirs[idx_k] = _rot_u(hz, ang)

# 若开头/中间缺人帧, 用最近有人帧插值保证第三栏镜头连续
if has_person.any() and not has_person.all():
    good_i = np.flatnonzero(has_person)
    all_i = np.arange(N_k)
    for d_i in range(3):
        centers[:, d_i] = np.interp(all_i, good_i, centers[good_i, d_i])
        view_dirs[:, d_i] = np.interp(all_i, good_i, view_dirs[good_i, d_i])
    spans = np.interp(all_i, good_i, spans[good_i])

# 分镜头平滑追随轨迹(切镜头处不跨段乱甩, 段内稳如滑轨)
cuts_s = B.shot_cuts(da3)
vfs_all = np.array([key2vid.get(k, 0) for k in KEYS], float)
shot_ids = np.searchsorted(np.asarray(cuts_s, float), vfs_all / max(float(_fj["src_fps"]), 1.0) + 1e-3, side="right")
for sid in np.unique(shot_ids):
    m_s = shot_ids == sid
    if m_s.sum() > 3:
        centers[m_s] = gaussian_filter1d(centers[m_s], sigma=5.0, axis=0, mode="nearest")
        spans[m_s] = gaussian_filter1d(spans[m_s], sigma=7.0, mode="nearest")
        vd = gaussian_filter1d(view_dirs[m_s], sigma=6.0, axis=0, mode="nearest")
        vd -= (vd @ up)[:, None] * up
        view_dirs[m_s] = vd / np.maximum(np.linalg.norm(vd, axis=1, keepdims=True), 1e-6)

fy_full = 0.88 * H
fx_full = fy_full

def _look_at_gl(eye, target, up_vec):
    fwd = target - eye
    fwd = fwd / max(np.linalg.norm(fwd), 1e-6)
    right = np.cross(fwd, up_vec)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    pose_gl = np.eye(4)
    pose_gl[:3, 0] = right
    pose_gl[:3, 1] = true_up
    pose_gl[:3, 2] = -fwd
    pose_gl[:3, 3] = eye
    return pose_gl

n = 0
for idx_k, key in enumerate(KEYS):
    vf = key2vid.get(key)
    c = cams[key]; K = np.array(c["intrinsic_3x3"]); h, w = c["input_hw"]
    sx, sy = W / w, H / h
    cam = pyrender.IntrinsicsCamera(fx=K[0, 0] * sx, fy=K[1, 1] * sy, cx=K[0, 2] * sx, cy=K[1, 2] * sy)
    c2w = np.linalg.inv(np.array(c["extrinsic_w2c_4x4"]).reshape(4, 4))
    pose = c2w @ np.diag([1, -1, -1, 1])

    mesh_nodes = []
    for i, (pid, fr) in enumerate(sorted(world.items(), key=lambda x: int(x[0]))):
        if vf in fr:
            pm = trimesh.Trimesh(vertices=fr[vf], faces=faces, process=False)
            pm.visual.vertex_colors = np.array(list(COL[(int(pid) - 1) % len(COL)]) + [255], np.uint8)
            mesh_nodes.append(scene.add(pyrender.Mesh.from_trimesh(pm, smooth=True)))
    for _c in CARRIED:
        if vf in _c["M"]:
            _cm = trimesh.Trimesh(vertices=(_c["V"] @ _c["M"][vf].T)[:, :3], faces=_c["F"], process=False)
            _cm.visual.vertex_colors = np.array([220, 40, 40, 255], np.uint8)
            mesh_nodes.append(scene.add(pyrender.Mesh.from_trimesh(_cm, smooth=False)))

    # 第 2 栏: 原相机视角
    cam_nd = scene.add(cam, pose=pose)
    lt_nd = scene.add(pyrender.DirectionalLight(intensity=4.0), pose=pose)
    col, _ = r.render(scene)
    scene.remove_node(cam_nd)
    scene.remove_node(lt_nd)

    # 第 3 栏: 全身追随视角(包住全身 + 脚下地面/台阶; 近裁面切掉挡在镜头前的远墙)
    tgt = centers[idx_k] - 0.08 * up
    dist = max(2.7, 0.72 * spans[idx_k] / (0.5 * H / fy_full))
    eye = tgt + dist * view_dirs[idx_k] + 0.38 * up
    pose_full = _look_at_gl(eye, tgt, up)
    cam_full = pyrender.IntrinsicsCamera(
        fx=fx_full, fy=fy_full, cx=W / 2.0, cy=H / 2.0,
        znear=max(0.25, dist - max(1.35, 0.65 * spans[idx_k])), zfar=120.0
    )
    cf_nd = scene.add(cam_full, pose=pose_full)
    lf_nd = scene.add(pyrender.DirectionalLight(intensity=4.0), pose=pose_full)
    full_col, _ = r.render(scene)
    scene.remove_node(cf_nd)
    scene.remove_node(lf_nd)

    for nd in mesh_nodes:
        scene.remove_node(nd)

    real = cv2.resize(cv2.cvtColor(cv2.imread(IMG[vf] if IMG else f"{a.fr}/{key}"), cv2.COLOR_BGR2RGB), (W, H))
    row = np.hstack([real, col, full_col])
    cv2.putText(row, "full-body view", (2 * W + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 80, 30), 2)
    cv2.putText(row, f"{key}  video frame {vf}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imwrite(f"{tmp}/{n:05d}.jpg", cv2.cvtColor(row, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    n += 1

fps = (_fj["src_fps"] / max(1.0, float(np.median(np.diff(sorted(IMG)))))) if IMG and len(IMG) > 1 else \
    _fj["src_fps"] * len(cams) / max(1, max(key2vid.values()) + 1)
import imageio_ffmpeg
ff = imageio_ffmpeg.get_ffmpeg_exe()
out = f"{a.r1}/{a.out}"
# ★ 上一版不覆盖: 挪成 <名字>_prev.mp4(只留一份), 网页上和新的并排看改了什么
if os.path.exists(out):
    os.replace(out, out[:-4] + "_prev.mp4")
# ★ H.264 + yuv420p + 宽高偶数, 浏览器才认 (cv2 的 mp4v 不报错但放不了)
subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", f"{fps:.3f}", "-i", f"{tmp}/%05d.jpg",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], check=True)
shutil.rmtree(tmp)
print(f"[done] {n} 帧 @ {fps:.2f}fps -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
