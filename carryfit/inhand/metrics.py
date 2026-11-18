"""尺子: 被拿着的物体到底"在不在手上"。

现有的两个数(轮廓 IoU、手到物体表面的距离)都测不出这件事: 握点可以在物体上滑, 物体可以绕握点转, 两个数照样好看,
从原相机看也照样对。这里量的是物体在"握持系"(hand.py)里稳不稳, 外加换个角度渲染给人看。

对每只手的每一段握持(carry_poses.npz 里的 contact 列, 第二版和新版同格式):
  握点漂移      掌心在物体自己坐标系里的位置, 离这一段中位位置的距离(cm), 报 p95
                细长的东西(rod): 沿杆滑、绕杆转是握法允许的, 只算"掌心离杆轴的距离"变了多少
  手内倾斜      物体的特征轴(rod = 长轴, plate = 法线)在握持系里偏离这一段平均方向的角度, 报 p90
  手内转动      物体整体朝向在握持系里偏离平均朝向的角度, 报 p90 (rod 绕自己的轴转不算错, 看倾斜)
  帧间抖动      相邻关键帧之间, 物体相对手转了多少度(中位)
  掌心到表面    掌心离物体表面多远(cm, 中位) —— 握点滑不滑都骗不了它
"在手上" = 垂直漂移 p95 <= 5 cm 且 倾斜 p90 <= 15 度 且 掌心到表面中位 <= 6 cm (门槛在 PASS 里)。

  python carryfit/inhand/metrics.py --r1 <r1> --poses <carry_poses.npz> --out <目录> [--video]
--video: 出 inhand_view.mp4, 四格: 原片 | 叠加 | 侧面(绕天转 90 度) | 俯视。pyrender, fpose 环境。
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

import hand as hd

PASS = dict(drift_cm=5.0, tilt_deg=15.0, palm_cm=6.0)
MIN_SEG = 5


def runs_of(b):
    out, i = [], 0
    while i < len(b):
        if b[i]:
            j = i
            while j < len(b) and b[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def fill_gaps(b, gap):
    b = b.copy()
    for s, e in runs_of(~b):
        if s > 0 and e < len(b) and e - s <= gap:
            b[s:e] = True
    return b


def mean_dir(D):
    """单位向量的平均方向, 不分正反(轴没有正反)"""
    w, v = np.linalg.eigh(np.einsum("ti,tj->ij", D, D))
    return v[:, -1]


def analyse(r1, poses_path, people=None):
    c = np.load(poses_path, allow_pickle=True)
    if people is None:
        people, _ = hd.load_people(r1)
    L = c["local_to_world"].astype(float)
    n = len(L)
    s = np.linalg.norm(L[:, :3, :3], axis=1).mean(0)                 # 三向缩放
    Ro = L[:, :3, :3] / np.linalg.norm(L[:, :3, :3], axis=1, keepdims=True)
    to = L[:, :3, 3]
    ext = np.ptp(c["box_V"], 0) * s
    cls = hd.shape_class(ext)
    ax = {"rod": 0, "plate": 2}.get(cls)                             # 局部系 x 最长 z 最短
    rng = np.random.default_rng(0)
    sp = c["box_V"][rng.choice(len(c["box_V"]), min(3000, len(c["box_V"])), replace=False)] * s   # 物体表面点(米, 物体系)
    parts = [str(p) for p in c["parts"]]
    cont = c["contact"].astype(bool)
    segs = []
    for j, name in enumerate(parts):
        pid, part = name.split(":")
        if part not in hd.SIDE or pid not in people:
            continue
        row = hd.rows_for(people[pid], c["vf"])
        Rg, tg = people[pid]["grip"][part]
        # 分段: 同一只手连续拿着, 而且是同一个握姿(挂手版存了 hold_run; 换了握法 = 换了一段, 不算漂移)
        ids = np.where(cont[:, j] & (row >= 0), c["hold_run"][:, j] if "hold_run" in c.files else 0, -1)
        spans = [(a_, b_) for g in sorted(set(ids[ids >= 0].tolist())) for a_, b_ in runs_of(fill_gaps(ids == g, 2))]
        for a_, b_ in sorted(spans):
            fi = np.array([i for i in range(a_, b_) if row[i] >= 0 and ids[i] >= 0])
            if len(fi) < MIN_SEG:
                continue
            R_h, t_h = Rg[row[fi]], tg[row[fi]]
            rel = np.einsum("tji,tjk->tik", R_h, Ro[fi])              # 物体朝向, 在握持系里
            p = np.einsum("tji,tj->ti", Ro[fi], t_h - to[fi])          # 掌心, 在物体系里(米)
            d = p - np.median(p, 0)
            dist = np.linalg.norm(d, axis=1)
            if cls == "rod":                                          # 细长: 沿杆滑、绕杆转都是握法允许的 -> 只看掌心离杆轴的距离变没变
                rad = np.linalg.norm(p[:, 1:], axis=1)
                perp = np.abs(rad - np.median(rad))
            else:
                perp = dist
            Rm = Rotation.from_matrix(rel)
            dev = np.degrees((Rm.mean().inv() * Rm).magnitude())
            if ax is not None:
                A = rel[:, :, ax]
                tilt = np.degrees(np.arccos(np.clip(np.abs(A @ mean_dir(A)), 0, 1)))
            else:
                tilt = dev
            step = np.degrees((Rm[:-1].inv() * Rm[1:]).magnitude()) if len(fi) > 1 else np.zeros(1)
            palm = np.array([np.linalg.norm(sp - q, axis=1).min() for q in p])
            m = dict(部位=name, 关键帧=[int(a_), int(b_ - 1)], 帧数=int(len(fi)),
                     握点漂移_cm_p95=round(float(np.percentile(dist, 95)) * 100, 1),
                     握点漂移_垂直_cm_p95=round(float(np.percentile(perp, 95)) * 100, 1),
                     手内倾斜_度_p90=round(float(np.percentile(tilt, 90)), 1),
                     手内转动_度_p90=round(float(np.percentile(dev, 90)), 1),
                     帧间抖动_度_中位=round(float(np.median(step)), 1),
                     掌心到表面_cm_中位=round(float(np.median(palm)) * 100, 1))
            # 细长的东西松握时会绕握点转(伞搭肩上 / 垂着 / 甩起来), 倾斜不算错: 只要求握点钉在杆上同一处、掌心贴着
            m["在手上"] = bool(m["握点漂移_垂直_cm_p95"] <= PASS["drift_cm"] and m["掌心到表面_cm_中位"] <= PASS["palm_cm"]
                             and (cls == "rod" or m["手内倾斜_度_p90"] <= PASS["tilt_deg"]))
            segs.append(m)
    tot = sum(m["帧数"] for m in segs)
    wavg = lambda k: round(sum(m[k] * m["帧数"] for m in segs) / tot, 1) if tot else None
    iou = c["iou"] if "iou" in c.files else np.full(n, np.nan)
    held = cont[:, [j for j, p in enumerate(parts) if p.split(":")[1] in hd.SIDE]].any(1)
    out = dict(位姿文件=os.path.abspath(poses_path), 形状类=cls, 尺寸米=[round(float(x), 3) for x in ext], 关键帧数=int(n),
               手碰着的帧数=int(held.sum()), 没挂在手上的帧数=int((~held).sum()),
               没挂在手上的关键帧段=[[int(s_), int(e_ - 1)] for s_, e_ in runs_of(~held)],
               轮廓IoU_中位=round(float(np.nanmedian(iou)), 3) if np.isfinite(iou).any() else None,
               轮廓IoU_手碰着时_中位=round(float(np.nanmedian(iou[held])), 3) if held.any() and np.isfinite(iou[held]).any() else None,
               握点漂移_垂直_cm_p95=wavg("握点漂移_垂直_cm_p95"), 手内倾斜_度_p90=wavg("手内倾斜_度_p90"),
               手内转动_度_p90=wavg("手内转动_度_p90"), 帧间抖动_度_中位=wavg("帧间抖动_度_中位"),
               掌心到表面_cm_中位=wavg("掌心到表面_cm_中位"),
               在手上的帧占比=round(sum(m["帧数"] for m in segs if m["在手上"]) / tot, 2) if tot else None,
               门槛=PASS, 握持段=segs)
    return out, c, people


def render_video(r1, c, people, faces, out_path, summary, extra=()):
    """extra: 别的物体的位姿文件(已经 np.load 过的), 画进同一个画面(颜色不同)。侧面 / 俯视的相机跟着"这一帧挂在手上的物体"走。"""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import cv2
    import imageio
    import pyrender
    import trimesh
    cams = json.load(open(f"{r1}/02_da3/cameras.json"))["cameras"]
    fm = json.load(open(f"{r1}/02_da3/frame_map.json"))
    up = np.array(json.load(open(f"{r1}/02_da3/world_up.json"))["up"], float)
    up /= np.linalg.norm(up)
    keys = [str(k) for k in c["keys"]]
    n = len(keys)
    L = c["local_to_world"].astype(float)
    cont = c["contact"].astype(bool)
    parts = [str(p) for p in c["parts"]]
    pids = sorted(people)
    rows = {p: hd.rows_for(people[p], c["vf"]) for p in pids}
    TW = 480
    pal = [(230, 120, 40), (40, 160, 230), (120, 200, 90), (200, 90, 200), (220, 200, 60), (90, 200, 200), (160, 160, 160)]
    cap = None
    wr, r = None, None
    vf = c["vf"].astype(int)
    fps = int(np.clip(round(float(fm.get("fps", 30)) * n / max(1, vf[-1] - vf[0] + 1)), 4, 30))
    # 侧面 / 俯视相机看的中心: 物体中心, 按时间平滑(免得画面跟着抖)
    objs = [c] + list(extra)
    OC = [(205, 150, 60), (90, 170, 230), (150, 210, 110), (220, 120, 200)]
    held_o = [o["contact"].astype(bool).any(1) if "contact" in o.files else np.ones(n, bool) for o in objs]
    ctr = L[:, :3, 3].copy()
    for i in range(n):                                   # 相机看的中心: 这一帧挂在手上的那些物体的中心; 都没挂就沿用上一帧
        cs = [o["local_to_world"][i][:3, 3] for o, h in zip(objs, held_o) if h[i]]
        ctr[i] = np.mean(cs, 0) if cs else (ctr[i - 1] if i else ctr[i])
    k_ = max(1, n // 12)
    # 只在同一个镜头里平滑: 换镜头时物体在世界里会跳几米, 跨着切点平滑的话相机会对着两个位置中间的空地(杀破狼前几个镜头侧面 / 俯视全白)
    jump = np.where(np.linalg.norm(np.diff(ctr, axis=0), axis=1) > 1.0)[0] + 1
    for s_, e_ in zip(np.r_[0, jump], np.r_[jump, n]):
        seg = ctr[s_:e_]
        kk = min(k_, max(0, (len(seg) - 1) // 2))
        if kk > 0:
            ctr[s_:e_] = np.stack([np.convolve(np.pad(seg[:, a], kk, mode="edge"), np.ones(2 * kk + 1) / (2 * kk + 1), "valid") for a in range(3)], 1)

    def look_at(eye, tgt, upv):
        z = hd.unit(eye - tgt)
        x = hd.unit(np.cross(upv, z))
        y = np.cross(z, x)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, eye
        return M

    for i, k in enumerate(keys):
        cam = cams[k]
        h_in, w_in = cam["input_hw"]
        TH = int(round(TW * h_in / w_in))
        TH -= TH % 2
        if r is None:
            r = pyrender.OffscreenRenderer(TW, TH)
            wr = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
        K = np.diag([TW / w_in, TH / h_in, 1.0]) @ np.array(cam["intrinsic_3x3"], float)
        w2c = np.array(cam["extrinsic_w2c_4x4"], float).reshape(4, 4)
        c2w = np.linalg.inv(w2c)
        p = os.path.join(fm.get("frames_dir") or "", k)
        if os.path.exists(p):
            real = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        else:
            cap = cap or cv2.VideoCapture(fm.get("video"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(vf[i]))
            ok, im = cap.read()
            real = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if ok else np.zeros((TH, TW, 3), np.uint8)
        real = cv2.resize(real, (TW, TH))
        sc = pyrender.Scene(bg_color=[0.96, 0.96, 0.96, 1.0], ambient_light=[0.4, 0.4, 0.4])
        for j, pid in enumerate(pids):
            if rows[pid][i] < 0:
                continue
            pm = trimesh.Trimesh(people[pid]["V"][rows[pid][i]], faces, process=False)
            pm.visual.vertex_colors = np.array(list(pal[j % len(pal)]) + [255], np.uint8)
            sc.add(pyrender.Mesh.from_trimesh(pm, smooth=True))
        for oi, o in enumerate(objs):
            if oi and not held_o[oi][i]:
                continue                                       # 别的物体: 没挂在手上的帧不画(那时它的位姿只是停在最近一次的地方)
            bm = trimesh.Trimesh(trimesh.transform_points(o["box_V"], o["local_to_world"][i].astype(float)), o["box_F"], process=False)
            bm.visual.vertex_colors = np.array(list(OC[oi % len(OC)]) + [255], np.uint8)
            sc.add(pyrender.Mesh.from_trimesh(bm, smooth=False))
            for j, name in enumerate([str(p_) for p_ in o["parts"]]):   # 碰着的手: 掌心画个小绿球
                pid, part = name.split(":")
                if o["contact"][i, j] and part in hd.SIDE and pid in people and rows[pid][i] >= 0:
                    sp_ = trimesh.creation.icosphere(1, 0.025)
                    sp_.apply_translation(people[pid]["grip"][part][1][rows[pid][i]])
                    sp_.visual.vertex_colors = np.array([0, 220, 0, 255], np.uint8)
                    sc.add(pyrender.Mesh.from_trimesh(sp_, smooth=False))
        eye = c2w[:3, 3]
        dist = float(np.clip(np.linalg.norm(eye - ctr[i]), 1.2, 6.0))
        v = hd.unit(eye - ctr[i])
        side = Rotation.from_rotvec(up * np.pi / 2).apply(v - (v @ up) * up)
        side = hd.unit(side + 0.15 * up)
        fwd_h = hd.unit(v - (v @ up) * up)
        views = [c2w @ np.diag([1, -1, -1, 1.0]), look_at(ctr[i] + side * dist, ctr[i], up), look_at(ctr[i] + up * dist, ctr[i], -fwd_h)]
        cols = []
        for vi, pose in enumerate(views):
            cm = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=0.05, zfar=80) if vi == 0 else \
                pyrender.IntrinsicsCamera(1.4 * TW, 1.4 * TW, TW / 2, TH / 2, znear=0.05, zfar=80)
            nc = sc.add(cm, pose=pose)
            nl = sc.add(pyrender.DirectionalLight(intensity=4.0), pose=pose)
            col, _ = r.render(sc)
            sc.remove_node(nc)
            sc.remove_node(nl)
            cols.append(col)
        ov = (0.5 * real + 0.5 * cols[0]).astype(np.uint8)
        row_ = np.hstack([real, ov, cols[1], cols[2]])
        iou = c["iou"][i] if "iou" in c.files else float("nan")
        held_i = any(h[i] for h in held_o) if extra else any(cont[i, j] for j, nm in enumerate(parts) if nm.split(":")[1] in hd.SIDE)
        cv2.putText(row_, f"{k} IoU {iou:.2f}" + ("" if held_i else "  NOT ATTACHED"), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 0) if held_i else (255, 60, 60), 2)
        for x_, t_ in ((TW, "overlay"), (2 * TW, "side +90"), (3 * TW, "top")):
            cv2.putText(row_, t_, (x_ + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 200), 2)
        cv2.putText(row_, summary, (8, TH - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        wr.append_data(row_)
    if wr is not None:
        wr.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--people_params", default=None, nargs="+", help="arm_ik.py 出的 people_refined.npz: 用修过手臂的人来量、来渲染; 给几个就把各自的改动加在一起")
    ap.add_argument("--also", default=[], nargs="*", help="别的物体的 carry_poses.npz: 画进同一个视频(指标只算 --poses 这一个)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pp_ = a.people_params[0] if a.people_params and len(a.people_params) == 1 else a.people_params
    people, faces = hd.load_people(a.r1, params_path=pp_)
    for pid, d in people.items():
        for part, (th, curl) in d["thumb_side"].items():
            assert th > 0, f"{pid} {part}: 拇指不在该在的一侧({th:.3f}), 握持系的定义和这个骨架对不上"
    out, c, _ = analyse(a.r1, a.poses, people)
    json.dump(out, open(f"{a.out}/inhand_metrics.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "握持段"}, ensure_ascii=False))
    for m in out["握持段"]:
        print("   ", json.dumps(m, ensure_ascii=False))
    if a.video:
        txt = (f"drift {out['握点漂移_垂直_cm_p95']}cm tilt {out['手内倾斜_度_p90']}deg palm {out['掌心到表面_cm_中位']}cm in-hand {out['在手上的帧占比']} "
               f"NOT attached {out['没挂在手上的帧数']}/{out['关键帧数']} frames")
        render_video(a.r1, c, people, faces, f"{a.out}/inhand_view.mp4", txt, extra=[np.load(x, allow_pickle=True) for x in a.also])
        print("[video]", f"{a.out}/inhand_view.mp4")


if __name__ == "__main__":
    main()
