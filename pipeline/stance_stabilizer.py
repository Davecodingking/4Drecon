"""Unified stepped-on support collider + Rule-2 stance stabilization for original & completed legs."""
import json
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.ndimage import gaussian_filter1d

import body_into_scene as B
import legs_proc as L
import mhr_body as M
from landing_motion import skin_vertices, vertex_binding
from observed_support import common_flight
from world_frame import export_T, inverse_T

NON_STEP_WORDS = {
    "ground", "floor", "wall", "chair", "bench", "sofa", "couch", "stool",
    "car", "taxi", "cab", "sedan", "suv", "van", "minivan", "truck", "pickup", "bus",
    "person", "man", "woman", "boy", "girl", "pedestrian", "dancer",
    "tree", "plant", "sign_post", "bollard", "street_lamp", "lamp_post", "pole",
    "window", "window_frame", "door", "door_frame", "house", "building", "facade", "storefront",
}


def _box_sdf_torch(pts, bmin, bmax):
    """Differentiable 3D AABB signed distance (>0 outside, <0 inside)."""
    center = 0.5 * (bmin + bmax)
    half = 0.5 * (bmax - bmin)
    q = torch.abs(pts - center) - half
    outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.clamp(torch.max(q, dim=-1).values, max=0.0)
    return outside + inside


class SupportCollider:
    """Unified surface elevation & 3D box collider across flat ground, terrain, and stepped-on scene objects."""

    def __init__(self, up, gh, terrain=None, boxes_gltf=None, pinned_supports=()):
        self.up = np.asarray(up, float) / np.linalg.norm(up)
        self.gh = float(gh)
        self.terrain = terrain
        self.T_exp = export_T(self.up)
        self.T_inv = inverse_T(self.up)
        self.boxes_gltf = list(boxes_gltf or [])  # list of dict(name, bmin, bmax) in glTF (+Y up)
        self.pinned_supports = list(pinned_supports)
        self.model_mesh = bool((terrain is not None and getattr(terrain, "model_mesh", False)) or len(self.boxes_gltf) > 0)

    def to_gltf(self, pts_da3):
        pts = np.asarray(pts_da3, float)
        return pts @ self.T_exp[:3, :3].T + self.T_exp[:3, 3]

    def base_h(self, pts_da3, vf=None):
        pts = np.asarray(pts_da3, float)
        orig_shape = pts.shape[:-1]
        flat = pts.reshape(-1, 3)
        if self.terrain is not None:
            if getattr(self.terrain, "model_mesh", False):
                h = np.asarray(self.terrain.model_h(flat), float)
            else:
                h = np.asarray(self.terrain.height_at(flat, vf=vf), float)
                h = np.where(np.isfinite(h), h, self.gh)
        else:
            h = np.full(len(flat), self.gh, dtype=float)
        return h.reshape(orig_shape)

    def model_h(self, pts_da3, vf=None):
        pts = np.asarray(pts_da3, float)
        orig_shape = pts.shape[:-1]
        flat = pts.reshape(-1, 3)
        h = self.base_h(flat, vf=vf).ravel()
        if self.boxes_gltf:
            pg = self.to_gltf(flat)
            for bx in self.boxes_gltf:
                lo, hi = bx["bmin"], bx["bmax"]
                inside = (pg[:, 0] >= lo[0]) & (pg[:, 0] <= hi[0]) & (pg[:, 2] >= lo[2]) & (pg[:, 2] <= hi[2])
                if np.any(inside):
                    # Convert glTF y_top to DA3 UP height (same offset from GH since export_T maps UP -> +Y)
                    top_da3 = self.gh + (float(hi[1]) - float(lo[1]))
                    h[inside] = np.maximum(h[inside], top_da3)
        return h.reshape(orig_shape)

    def height_at(self, pts_da3, vf=None):
        pts = np.asarray(pts_da3, float)
        if pts.ndim == 1:
            return np.atleast_1d(self.model_h(pts[None, :], vf=vf))
        return self.model_h(pts, vf=vf)


def build_support_collider(r1_dir, scene_in, up, gh, terrain, raw_actor_tracks, fps, dev="cuda"):
    """Detect stepped-on rigid boxes in 07_scene.glb, CRISP-align their 3D bounds to stance contacts, and build collider.

    raw_actor_tracks: list of dict(W_da3=(T,N,3), shape=(45,), feet_ok=(T,))
    """
    scene_path = os.path.join(r1_dir, scene_in)
    up = np.asarray(up, float) / np.linalg.norm(up)
    T_exp = export_T(up)
    gh_gltf = float((up * gh) @ T_exp[:3, :3].T @ np.array([0.0, 1.0, 0.0]) + T_exp[1, 3])

    if not os.path.isfile(scene_path) or not raw_actor_tracks:
        return SupportCollider(up, gh, terrain)

    # Do not modify multi-shot layered terrain scenes that already model stairs/platforms in 07_terrain.npz
    if terrain is not None and getattr(terrain, "model_mesh", False):
        return SupportCollider(up, gh, terrain)

    try:
        sc = trimesh.load(scene_path, force="scene")
    except Exception:
        return SupportCollider(up, gh, terrain)

    # Also inspect 06_all_boxes.json for raw point-cloud top elevation if available
    boxes_json = {}
    bj_path = os.path.join(r1_dir, "06_all_boxes.json")
    if os.path.isfile(bj_path):
        try:
            boxes_json = json.load(open(bj_path))
        except Exception:
            boxes_json = {}

    # Collect elevated low-velocity visible foot contacts & ground-level contacts in glTF space
    actor_soles = []
    sk = M.skeleton(dev)
    jnames = [str(x) for x in sk["names"]]
    dom, bw = np.zeros(int(sk["sk_v"].max()) + 1, int), np.zeros(int(sk["sk_v"].max()) + 1)
    for v, j, w in zip(sk["sk_v"], sk["sk_j"], sk["sk_w"]):
        if w > bw[v]:
            bw[v], dom[v] = w, j
    l_fj = [i for i, nm in enumerate(jnames) if nm in ("l_talocrural", "l_subtalar", "l_transversetarsal", "l_ball")]
    r_fj = [i for i, nm in enumerate(jnames) if nm in ("r_talocrural", "r_subtalar", "r_transversetarsal", "r_ball")]
    l_fv = np.where(np.isin(dom, l_fj))[0]
    r_fv = np.where(np.isin(dom, r_fj))[0]

    for tr in raw_actor_tracks:
        W_da3 = tr["W_da3"]
        fok = np.asarray(tr["feet_ok"], bool)
        rest = M.rest_verts(tr["shape"], dev)
        l_sole = l_fv[rest[l_fv, 1] < np.percentile(rest[l_fv, 1], 25)]
        r_sole = r_fv[rest[r_fv, 1] < np.percentile(rest[r_fv, 1], 25)]
        Vg = np.einsum("ij,tnj->tni", T_exp[:3, :3], W_da3) + T_exp[:3, 3]
        actor_soles.append((Vg[:, l_sole], Vg[:, r_sole], fok))

    boxes_gltf = []
    pinned = []
    modified = False

    for nd in list(sc.graph.nodes_geometry):
        short = nd.split("/")[-1]
        cls = re.sub(r"_\d+$", "", short)
        if cls in NON_STEP_WORDS or any(w in cls for w in ("chair", "bench", "sofa", "couch", "stool", "wall", "car", "truck", "bus")):
            continue
        T_nd, gname = sc.graph[nd]
        geom = sc.geometry[gname]
        vg = trimesh.transform_points(geom.vertices, T_nd)
        lo0, hi0 = vg.min(0), vg.max(0)
        h_obj = float(hi0[1] - gh_gltf)
        if not (0.18 <= h_obj <= 1.25):
            continue
        if max(hi0[0] - lo0[0], hi0[2] - lo0[2]) > 4.0:
            continue

        # Check raw point-cloud box top height if matching word#id exists in 06_all_boxes.json
        pc_top_y = float(hi0[1])
        m_id = re.search(r"_(\d+)$", short)
        word_sp = cls.replace("_", " ")
        if m_id and word_sp in boxes_json:
            oid = int(m_id.group(1))
            for item in boxes_json[word_sp]:
                if int(item.get("obj_id", -1)) == oid:
                    raw_top_da3 = float(item.get("底高", gh)) + float(item["尺寸"][2])
                    pc_top_y = gh_gltf + (raw_top_da3 - gh)
                    break

        # Gather visible low-velocity foot contacts near this object's top & XZ footprint
        elev_pts = []
        elev_ys = []
        ground_pts = []
        n_stance_frames = 0
        for Vl, Vr, fok in actor_soles:
            for Vfoot in (Vl, Vr):
                f_mean = Vfoot.mean(1)
                f_ymin = Vfoot[:, :, 1].min(1)
                f_vel = np.linalg.norm(np.diff(f_mean, axis=0, prepend=f_mean[:1]), axis=1) * fps
                # Ground-level contacts (used to keep support box strictly clear of street landings)
                g_mask = np.abs(f_ymin - gh_gltf) < 0.11
                if np.any(g_mask):
                    ground_pts.append(Vfoot[g_mask][:, :, [0, 2]].reshape(-1, 2))
                # Elevated stance contacts on this candidate object
                in_xz = ((f_mean[:, 0] >= lo0[0] - 0.25) & (f_mean[:, 0] <= hi0[0] + 0.25) &
                         (f_mean[:, 2] >= lo0[2] - 0.25) & (f_mean[:, 2] <= hi0[2] + 0.25))
                near_top = (np.abs(f_ymin - pc_top_y) < 0.14) | (np.abs(f_ymin - hi0[1]) < 0.14)
                elev_m = fok & in_xz & near_top & (f_vel < 0.65)
                if np.any(elev_m):
                    n_stance_frames += int(elev_m.sum())
                    elev_pts.append(Vfoot[elev_m][:, :, [0, 2]].reshape(-1, 2))
                    elev_ys.append(f_ymin[elev_m])

        if n_stance_frames < 8 or not elev_pts:
            continue

        blk_xz = np.concatenate(elev_pts, axis=0)
        stance_y = float(np.median(np.concatenate(elev_ys)))
        y_top = 0.5 * pc_top_y + 0.5 * stance_y

        x_min_t = float(blk_xz[:, 0].min()) - 0.14
        x_max_t = float(blk_xz[:, 0].max()) + 0.14
        z_min_t = float(blk_xz[:, 1].min()) - 0.14
        z_max_t = float(blk_xz[:, 1].max()) + 0.18

        if ground_pts:
            land_xz = np.concatenate(ground_pts, axis=0)
            # Only compare ground contacts within 2.5 m of the block
            blk_c = np.median(blk_xz, axis=0)
            near_land = land_xz[np.linalg.norm(land_xz - blk_c, axis=1) < 2.5]
            if len(near_land) >= 10:
                land_c = np.median(near_land, axis=0)
                if abs(land_c[1] - blk_c[1]) >= abs(land_c[0] - blk_c[0]):
                    if land_c[1] > blk_c[1]:
                        z_max_t = min(z_max_t, float(np.percentile(near_land[:, 1], 5)) - 0.10)
                    else:
                        z_min_t = max(z_min_t, float(np.percentile(near_land[:, 1], 95)) + 0.10)
                else:
                    if land_c[0] > blk_c[0]:
                        x_max_t = min(x_max_t, float(np.percentile(near_land[:, 0], 5)) - 0.10)
                    else:
                        x_min_t = max(x_min_t, float(np.percentile(near_land[:, 0], 95)) + 0.10)

        # Ensure sensible minimum width/depth (at least 0.35 m)
        if x_max_t - x_min_t < 0.35:
            mid_x = 0.5 * (x_min_t + x_max_t)
            x_min_t, x_max_t = mid_x - 0.18, mid_x + 0.18
        if z_max_t - z_min_t < 0.35:
            mid_z = 0.5 * (z_min_t + z_max_t)
            z_min_t, z_max_t = mid_z - 0.18, mid_z + 0.18

        target_lo = np.array([x_min_t, gh_gltf, z_min_t])
        target_hi = np.array([x_max_t, y_top, z_max_t])
        v_new = (vg - lo0) / np.maximum(hi0 - lo0, 1e-6) * (target_hi - target_lo) + target_lo
        geom.vertices = trimesh.transform_points(v_new, np.linalg.inv(T_nd))
        boxes_gltf.append(dict(name=short, bmin=target_lo, bmax=target_hi))
        pinned.append(short)
        modified = True
        print(f"  [支撑物对齐] {short}: 检测到 {n_stance_frames} 帧脚踩支撑, 顶高 {y_top - gh_gltf:.2f}m, "
              f"范围 {np.round(target_lo, 2)} ~ {np.round(target_hi, 2)}", flush=True)

    if modified:
        sc.export(scene_path)

    return SupportCollider(up, gh, terrain, boxes_gltf=boxes_gltf, pinned_supports=pinned)


def _optimize_box_hsi(Pw_da3, shape, fps, collider, cams_w2c, cams_K, dev="cuda"):
    """MeshMimic TSDF + JOSH anti-slip root refinement when stepped-on rigid 3D boxes are present."""
    T = len(Pw_da3)
    P_gltf = M.transform(Pw_da3, collider.T_exp)
    V_init, _ = M.forward(shape, P_gltf, True, dev)

    rest = M.rest_verts(shape, dev)
    sk = M.skeleton(dev)
    jnames = [str(x) for x in sk["names"]]
    dom, bw = np.zeros(V_init.shape[1], int), np.zeros(V_init.shape[1])
    for v, j, w in zip(sk["sk_v"], sk["sk_j"], sk["sk_w"]):
        if w > bw[v]:
            bw[v], dom[v] = w, j
    l_fj = [i for i, nm in enumerate(jnames) if nm in ("l_talocrural", "l_subtalar", "l_transversetarsal", "l_ball")]
    r_fj = [i for i, nm in enumerate(jnames) if nm in ("r_talocrural", "r_subtalar", "r_transversetarsal", "r_ball")]
    l_fv = np.where(np.isin(dom, l_fj))[0]
    r_fv = np.where(np.isin(dom, r_fj))[0]
    l_sole = l_fv[rest[l_fv, 1] < np.percentile(rest[l_fv, 1], 25)]
    r_sole = r_fv[rest[r_fv, 1] < np.percentile(rest[r_fv, 1], 25)]

    lf_pos = V_init[:, l_sole].mean(1)
    rf_pos = V_init[:, r_sole].mean(1)
    lf_ymin = V_init[:, l_sole, 1].min(1)
    rf_ymin = V_init[:, r_sole, 1].min(1)
    lf_vel = np.linalg.norm(np.diff(lf_pos, axis=0, prepend=lf_pos[:1]), axis=1) * fps
    rf_vel = np.linalg.norm(np.diff(rf_pos, axis=0, prepend=rf_pos[:1]), axis=1) * fps

    gh_gltf = float(collider.boxes_gltf[0]["bmin"][1])
    # Per-frame expected surface height for each foot if near a surface (box top or floor)
    l_tgt_y = np.full(T, np.nan)
    r_tgt_y = np.full(T, np.nan)
    for bx in collider.boxes_gltf:
        lo, hi = bx["bmin"], bx["bmax"]
        y_top = float(hi[1])
        l_on = ((lf_pos[:, 0] >= lo[0] - 0.12) & (lf_pos[:, 0] <= hi[0] + 0.12) &
                (lf_pos[:, 2] >= lo[2] - 0.12) & (lf_pos[:, 2] <= hi[2] + 0.12) &
                (np.abs(lf_ymin - y_top) < 0.11))
        r_on = ((rf_pos[:, 0] >= lo[0] - 0.12) & (rf_pos[:, 0] <= hi[0] + 0.12) &
                (rf_pos[:, 2] >= lo[2] - 0.12) & (rf_pos[:, 2] <= hi[2] + 0.12) &
                (np.abs(rf_ymin - y_top) < 0.11))
        l_tgt_y[l_on] = y_top
        r_tgt_y[r_on] = y_top

    l_str = np.isnan(l_tgt_y) & (np.abs(lf_ymin - gh_gltf) < 0.09)
    r_str = np.isnan(r_tgt_y) & (np.abs(rf_ymin - gh_gltf) < 0.09)
    l_tgt_y[l_str] = gh_gltf
    r_tgt_y[r_str] = gh_gltf

    l_has_c = np.isfinite(l_tgt_y)
    r_has_c = np.isfinite(r_tgt_y)
    l_stat = l_has_c & (lf_vel < 0.35)
    r_stat = r_has_c & (rf_vel < 0.35)

    V_base = torch.tensor(V_init, dtype=torch.float32, device=dev)
    sub_idx = np.unique(np.concatenate([np.arange(0, V_init.shape[1], 6), l_sole, r_sole]))
    V_sub = V_base[:, sub_idx, :]
    l_sole_t = torch.tensor(l_sole, dtype=torch.long, device=dev)
    r_sole_t = torch.tensor(r_sole, dtype=torch.long, device=dev)

    w2c_gltf = np.stack([w2c @ collider.T_inv for w2c in cams_w2c])
    R_w2c = torch.tensor(w2c_gltf[:, :3, :3], dtype=torch.float32, device=dev)
    t_w2c = torch.tensor(w2c_gltf[:, :3, 3], dtype=torch.float32, device=dev)
    K_t = torch.tensor(np.asarray(cams_K, float), dtype=torch.float32, device=dev)
    root_0 = torch.tensor(M.root_pos(P_gltf), dtype=torch.float32, device=dev)

    def proj_2d(pts_3d):
        pc = torch.einsum("tij,tj->ti", R_w2c, pts_3d) + t_w2c
        uv = torch.einsum("tij,tj->ti", K_t, pc)
        return uv[:, :2] / torch.clamp(uv[:, 2:3], min=0.1), pc[:, 2]

    uv0, z0 = proj_2d(root_0)
    l_c_t = torch.tensor(l_has_c, device=dev)
    r_c_t = torch.tensor(r_has_c, device=dev)
    l_ty_t = torch.tensor(np.nan_to_num(l_tgt_y, nan=0.0), dtype=torch.float32, device=dev)
    r_ty_t = torch.tensor(np.nan_to_num(r_tgt_y, nan=0.0), dtype=torch.float32, device=dev)
    l_stat_t = torch.tensor(l_stat[1:] & l_stat[:-1], device=dev)
    r_stat_t = torch.tensor(r_stat[1:] & r_stat[:-1], device=dev)

    box_tensors = [
        (torch.tensor(bx["bmin"], dtype=torch.float32, device=dev),
         torch.tensor(bx["bmax"], dtype=torch.float32, device=dev))
        for bx in collider.boxes_gltf
    ]

    dp = torch.zeros((T, 3), dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([dp], lr=0.015)

    for _ in range(260):
        opt.zero_grad()
        V_cur = V_sub + dp[:, None, :]
        Vl = V_base[:, l_sole_t, :] + dp[:, None, :]
        Vr = V_base[:, r_sole_t, :] + dp[:, None, :]

        uv_cur, z_cur = proj_2d(root_0 + dp)
        loss_align = ((uv_cur - uv0) / 25.0).pow(2).mean() + 0.8 * (z_cur - z0).pow(2).mean()

        ly_min = Vl[:, :, 1].min(dim=1).values
        ry_min = Vr[:, :, 1].min(dim=1).values
        loss_contact = torch.tensor(0.0, device=dev)
        if l_c_t.any():
            loss_contact = loss_contact + 45.0 * (ly_min[l_c_t] - l_ty_t[l_c_t]).pow(2).mean()
        if r_c_t.any():
            loss_contact = loss_contact + 45.0 * (ry_min[r_c_t] - r_ty_t[r_c_t]).pow(2).mean()

        loss_pen = torch.tensor(0.0, device=dev)
        for bmin_t, bmax_t in box_tensors:
            sdf_b = _box_sdf_torch(V_cur, bmin_t, bmax_t - torch.tensor([0.0, 0.003, 0.0], device=dev))
            pen_b = F.relu(-sdf_b)
            loss_pen = loss_pen + 140.0 * F.huber_loss(pen_b, torch.zeros_like(pen_b), delta=0.02, reduction="none").sum(dim=1).mean()
        pen_g = F.relu((gh_gltf - 0.003) - V_cur[:, :, 1])
        loss_pen = loss_pen + 140.0 * F.huber_loss(pen_g, torch.zeros_like(pen_g), delta=0.02, reduction="none").sum(dim=1).mean()

        l_mean = Vl.mean(dim=1)
        r_mean = Vr.mean(dim=1)
        loss_static = torch.tensor(0.0, device=dev)
        if l_stat_t.any():
            loss_static = loss_static + 15.0 * (l_mean[1:][l_stat_t] - l_mean[:-1][l_stat_t]).pow(2).mean()
        if r_stat_t.any():
            loss_static = loss_static + 15.0 * (r_mean[1:][r_stat_t] - r_mean[:-1][r_stat_t]).pow(2).mean()

        vel = (dp[1:] - dp[:-1]) * fps
        acc = (dp[2:] - 2 * dp[1:-1] + dp[:-2]) * fps if T > 2 else torch.zeros_like(vel)
        loss_sm = 0.15 * vel.pow(2).mean() + 0.6 * acc.pow(2).mean()

        loss = loss_align + loss_contact + loss_pen + loss_static + loss_sm
        loss.backward()
        opt.step()

    P_gltf_opt = M.set_root(P_gltf, None, M.root_pos(P_gltf) + dp.detach().cpu().numpy())
    return M.transform(P_gltf_opt, collider.T_inv)


def stabilize_actor_stances(Pw, shape, vfs, shot, badcam, fps, up, collider, dev="cuda",
                            cams_w2c=None, cams_K=None, seated=False):
    """Rule 2: Unified tread-contact & single-leg anti-slip stabilization for original & completed legs.

    - Stepped-on rigid boxes (if any): MeshMimic 3D box SDF + multi-height contact optimization.
    - All standing/moving actors: stabilize load-bearing contact point on its tread by adjusting ONLY that leg's joints
      (allowing forefoot pivot on tiptoe/turns and complete freedom during kicks, jumps, and dives/flips).
    """
    out = np.array(Pw, dtype=np.float64, copy=True)
    T = len(out)
    if T < 4 or seated:
        return out, {}

    up = np.asarray(up, float) / np.linalg.norm(up)
    report = {}

    if collider.boxes_gltf and cams_w2c is not None and cams_K is not None:
        out = _optimize_box_hsi(out, shape, fps, collider, cams_w2c, cams_K, dev=dev)
        report["支撑盒碰撞优化"] = [bx["name"] for bx in collider.boxes_gltf]

    # Evaluate skinned mesh & skeleton to detect load-bearing stance intervals and flight/kicks
    V, S = M.forward(shape, out, True, dev)
    rest = M.rest_verts(shape, dev)
    soles_full = [np.flatnonzero((rest[:, 1] < rest[:, 1].min() + 10.0) & (s * rest[:, 0] > 0)) for s in (1, -1)]
    geo = L.geometry(shape, out[0], dev)
    pts5_ids = np.array([[geo["side"][sd]["vids"][nm] for nm in L.PTS] for sd in L.SIDES])
    sk_names = list(M.skeleton(dev)["names"])
    spine_j = sk_names.index("c_spine3")
    hip_j = [geo["side"][sd]["j"]["hip"] for sd in L.SIDES]
    knee_j = [geo["side"][sd]["j"]["knee"] for sd in L.SIDES]
    ank_j = [geo["side"][sd]["j"]["ank"] for sd in L.SIDES]
    max_reach = [float(np.linalg.norm(geo["side"][sd]["t0"]) + np.linalg.norm(geo["side"][sd]["s0"])) for sd in L.SIDES]

    hips = S[:, hip_j, :3] / 100.0
    pelvis = hips.mean(1)
    chest = S[:, spine_j, :3] / 100.0
    trunk = chest - pelvis
    upright = (trunk @ up) / np.maximum(np.linalg.norm(trunk, axis=1), 1e-6)

    # Gap & tread height per foot
    gap = np.zeros((T, 2))
    tread_h = np.zeros((T, 2))
    fore = np.zeros((T, 2, 3))
    heel = np.zeros((T, 2, 3))
    fyaw = np.zeros((T, 2))
    for s_i in range(2):
        sv = V[:, soles_full[s_i]]
        sh = collider.model_h(sv.reshape(-1, 3)).reshape(T, -1)
        gap[:, s_i] = (sv @ up - sh).min(1)
        p5 = V[:, pts5_ids[s_i]]
        p5_h = collider.model_h(p5.reshape(-1, 3)).reshape(T, -1)
        tread_h[:, s_i] = np.nanmax(p5_h, axis=1)
        heel[:, s_i] = p5[:, 0]
        fore[:, s_i] = 0.5 * (p5[:, 2] + p5[:, 3])
        d_hf = fore[:, s_i] - heel[:, s_i]
        d_hf_p = d_hf - (d_hf @ up)[:, None] * up
        # Orthonormal planar basis
        B3 = L._basis(up)
        fyaw[:, s_i] = np.arctan2(d_hf_p @ B3[1], d_hf_p @ B3[0])

    fyaw = np.unwrap(fyaw, axis=0)
    yaw_speed = np.abs(np.gradient(fyaw, axis=0)) * fps  # rad/s

    # Flight / jump / dive / backflip exemption
    heights_all = np.column_stack([pelvis @ up, (V[:, soles_full[0]] @ up).min(1), (V[:, soles_full[1]] @ up).min(1)])
    airborne = (common_flight(heights_all, shot, fps)
                | (gap > 0.085).all(1)
                | (upright < 0.55))

    vfs = np.asarray(vfs, int)
    badcam = np.asarray(badcam, bool)
    shot = np.asarray(shot, int)
    contig_breaks = (np.diff(shot) != 0) | (np.diff(vfs) != 1)
    seg_id = np.r_[0, np.cumsum(contig_breaks)]

    # Forefoot horizontal velocity
    fore_step = np.diff(fore, axis=0, prepend=fore[:1])
    fore_step_p = fore_step - (fore_step @ up)[:, :, None] * up
    fore_speed = np.linalg.norm(fore_step_p, axis=-1) * fps

    min_run = max(2, round(0.08 * fps))
    pad = max(2, round(0.12 * fps))
    stabilized_runs = {0: 0, 1: 0}
    stabilized_frames = 0

    u_t = torch.tensor(up, dtype=torch.float32, device=dev)

    for sid in np.unique(seg_id):
        seg_rows = np.flatnonzero(seg_id == sid)
        if len(seg_rows) < min_run:
            continue
        s0, s1 = int(seg_rows[0]), int(seg_rows[-1]) + 1

        for s_i, sd in enumerate(L.SIDES):
            other = 1 - s_i
            # Load-bearing candidate: not airborne, good camera, sole on tread, and either single-support (other foot in air),
            # low-to-moderate forefoot speed, or near-zero-gap tread scraping
            on_tread = (~airborne[s0:s1]) & (~badcam[s0:s1]) & (gap[s0:s1, s_i] >= -0.025) & (gap[s0:s1, s_i] <= 0.028)
            single_support = gap[s0:s1, other] > 0.040
            scrape_touch = np.zeros(s1 - s0, bool)
            if s1 - s0 > 1:
                pair_touch = np.maximum(gap[s0:s1 - 1, s_i], gap[s0 + 1:s1, s_i]) <= 0.015
                scrape_touch[:-1] |= pair_touch
                scrape_touch[1:] |= pair_touch
            planted_speed = (fore_speed[s0:s1, s_i] < 0.65) | scrape_touch
            cand = on_tread & (single_support | planted_speed)

            # Split candidate runs whenever tread elevation steps by > 3.5 cm (stair transition!)
            runs = []
            for r_lo, r_hi in L.runs_of(cand):
                a, b = s0 + r_lo, s0 + r_hi
                if b - a < min_run:
                    continue
                # Sub-segment by constant tread height (within 3.5 cm)
                sub_start = a
                ref_h = tread_h[a, s_i]
                for k in range(a + 1, b):
                    if abs(tread_h[k, s_i] - ref_h) > 0.035:
                        if k - sub_start >= min_run:
                            runs.append((sub_start, k))
                        sub_start = k
                        ref_h = tread_h[k, s_i]
                if b - sub_start >= min_run:
                    runs.append((sub_start, b))

            # Precompute actual lowest-sole horizontal step along this shot segment
            low_step = np.zeros(max(T - 1, 1), dtype=float)
            for k_e in range(s0, s1 - 1):
                sv_k = V[k_e, soles_full[s_i]]
                lk = int(np.argmin(sv_k @ up - collider.model_h(sv_k)))
                dd_l = V[k_e + 1, soles_full[s_i][lk]] - V[k_e, soles_full[s_i][lk]]
                low_step[k_e] = float(np.linalg.norm(dd_l - (dd_l @ up) * up))

            for a, b in runs:
                # Trim any >=3-edge rock-solid (< 0.6 cm/frame) prefix/suffix so already-settled stances stay 100% untouched
                settled_left = False
                settled_right = False
                if b - a > 4:
                    n_pre = 0
                    while a + n_pre < b - 1 and low_step[a + n_pre] < 0.006:
                        n_pre += 1
                    if n_pre >= 3:
                        a = a + n_pre + 1
                        settled_left = True
                    n_suf = 0
                    while b - 1 - n_suf > a and low_step[b - 2 - n_suf] < 0.006:
                        n_suf += 1
                    if n_suf >= 3:
                        b = b - 1 - n_suf
                        settled_right = True
                if b - a < min_run:
                    continue

                h_tr = float(np.median(tread_h[a:b, s_i]))
                heel_lift = (heel[a:b, s_i] @ up - h_tr) - (fore[a:b, s_i] @ up - h_tr)
                fore_lift = -heel_lift
                use_heel_piv = bool(np.median(fore_lift) > 0.022 and np.max(yaw_speed[a:b, s_i]) > np.deg2rad(25.0))
                use_fore_piv = bool((not use_heel_piv) and (np.median(heel_lift) > 0.018 or np.max(yaw_speed[a:b, s_i]) > np.deg2rad(40.0)))

                # Measure existing slip of the consistent load-bearing pivot across [eval_a, eval_b)
                eval_a = a - 1 if (settled_left and a > s0) else a
                eval_b = b + 1 if (settled_right and b < s1) else b
                if use_fore_piv:
                    piv_pts = fore[eval_a:eval_b, s_i]
                elif use_heel_piv:
                    piv_pts = heel[eval_a:eval_b, s_i]
                else:
                    piv_pts = 0.5 * (fore[eval_a:eval_b, s_i] + heel[eval_a:eval_b, s_i])
                d_piv = np.diff(piv_pts, axis=0)
                d_piv_p = d_piv - (d_piv @ up)[:, None] * up
                slip_per_frame = np.linalg.norm(d_piv_p, axis=1)
                max_slip = float(slip_per_frame.max()) if len(slip_per_frame) else 0.0
                max_low_slip = float(low_step[eval_a:eval_b - 1].max()) if eval_b - eval_a > 1 else 0.0
                total_drift = float(np.linalg.norm((piv_pts[-1] - piv_pts[0]) - ((piv_pts[-1] - piv_pts[0]) @ up) * up))

                # Non-regression guard: leave already-solid stances (e.g. Joker 305, 425-430 R) 100% untouched!
                if (max_low_slip <= 0.014 or (max_slip <= 0.014 and total_drift <= 0.022)) and gap[a:b, s_i].min() >= -0.008:
                    continue

                # Solve single-leg IK on [lo, hi) within [s0, s1), never bleeding padding into trimmed rock-solid runs
                lo = a if settled_left else max(s0, a - pad)
                hi = b if settled_right else min(s1, b + pad)
                win_len = hi - lo
                act = np.zeros(win_len, bool)
                act[a - lo:b - lo] = True

                cols = [c for c in L.LEG_COLS[sd] if c not in L.BALL_TWIST]
                k_idx = cols.index(L.KNEE_COL[sd])
                sole_sub = np.unique(np.r_[pts5_ids[s_i], soles_full[s_i][::3]])
                bind_sole = vertex_binding(shape, sole_sub, dev)
                bind_p5 = vertex_binding(shape, pts5_ids[s_i], dev)

                # Compute reachable forefoot & heel anchor on tread h_tr (preferring adjacent settled stance if present)
                if settled_left and settled_right and a > s0 and b < s1:
                    if np.linalg.norm(fore[b, s_i] - fore[a - 1, s_i]) <= 0.025:
                        fore_med = 0.5 * (fore[a - 1, s_i] + fore[b, s_i])
                        heel_med = 0.5 * (heel[a - 1, s_i] + heel[b, s_i])
                    else:
                        fore_med = fore[b, s_i].copy()
                        heel_med = heel[b, s_i].copy()
                        settled_left = False
                        lo = max(s0, a - pad)
                        win_len = hi - lo
                        act = np.zeros(win_len, bool)
                        act[a - lo:b - lo] = True
                elif settled_right and b < s1:
                    fore_med = fore[b, s_i].copy()
                    heel_med = heel[b, s_i].copy()
                elif settled_left and a > s0:
                    fore_med = fore[a - 1, s_i].copy()
                    heel_med = heel[a - 1, s_i].copy()
                else:
                    fore_med = np.median(fore[a:b, s_i], axis=0)
                    heel_med = np.median(heel[a:b, s_i], axis=0)
                fore_anchors = np.repeat(fore_med[None, :], win_len, axis=0)
                heel_anchors = np.repeat(heel_med[None, :], win_len, axis=0)
                # Ensure target ankle stays within anatomical hip reach (0.965 * max_reach) so pelvis stays fixed!
                ank_off = S[lo:hi, ank_j[s_i], :3] / 100.0 - fore[lo:hi, s_i]
                hip_win = hips[lo:hi, s_i]
                r_lim = 0.982 * max_reach[s_i]
                for t_w in range(win_len):
                    tgt_ank = fore_anchors[t_w] + (h_tr - (fore_anchors[t_w] @ up)) * up + ank_off[t_w]
                    vec = tgt_ank - hip_win[t_w]
                    dist = float(np.linalg.norm(vec))
                    if dist > r_lim:
                        # Pull horizontal anchor just inside hip reach sphere
                        vec_h = vec - (vec @ up) * up
                        vh_norm = float(np.linalg.norm(vec_h))
                        vert = abs(float(vec @ up))
                        max_h = np.sqrt(max(r_lim ** 2 - vert ** 2, 0.0))
                        if vh_norm > max_h and vh_norm > 1e-6:
                            shift_h = vec_h * (max_h / vh_norm - 1.0)
                            fore_anchors[t_w] += shift_h
                            heel_anchors[t_w] += shift_h

                fore_w_mask = np.zeros(win_len, np.float32)
                heel_w_mask = np.zeros(win_len, np.float32)
                fore_w_mask[a - lo:b - lo] = 0.0 if use_heel_piv else 1.0
                heel_w_mask[a - lo:b - lo] = 0.0 if use_fore_piv else 1.0

                base_t = torch.tensor(out[lo:hi], dtype=torch.float32, device=dev)
                d_leg = torch.zeros((win_len, len(cols)), dtype=torch.float32, device=dev, requires_grad=True)
                act_t = torch.tensor(act.astype(np.float32), device=dev)
                fw_t = torch.tensor(fore_w_mask, device=dev)
                hw_t = torch.tensor(heel_w_mask, device=dev)
                fa_t = torch.tensor(fore_anchors, dtype=torch.float32, device=dev)
                ha_t = torch.tensor(heel_anchors, dtype=torch.float32, device=dev)
                distal_idx, src_foot, foot_slack = L.distal_prior(base_t, cols)

                opt = torch.optim.Adam([d_leg], lr=0.025)
                for _ in range(130):
                    opt.zero_grad()
                    p_cur = base_t.clone()
                    p_cur[:, cols] = base_t[:, cols] + d_leg
                    sk_cur = L.skel_torch(p_cur, dev)
                    v_sole = skin_vertices(sk_cur, bind_sole)
                    v_p5 = skin_vertices(sk_cur, bind_p5)

                    surf_cur = collider.model_h(v_sole.detach().cpu().numpy().reshape(-1, 3)).reshape(win_len, -1)
                    gap_cur = v_sole @ u_t - torch.tensor(surf_cur, dtype=torch.float32, device=dev)
                    min_g = gap_cur.amin(1)

                    fore_cur = 0.5 * (v_p5[:, 2] + v_p5[:, 3])
                    heel_cur = v_p5[:, 0]
                    df = fore_cur - fa_t
                    df_p = df - (df @ u_t)[:, None] * u_t
                    dh = heel_cur - ha_t
                    dh_p = dh - (dh @ u_t)[:, None] * u_t

                    # Contact height & non-penetration
                    loss = 1800.0 * ((min_g - 0.002).square() * act_t).sum() / act_t.sum().clamp(min=1.0)
                    loss = loss + 8000.0 * torch.relu(-min_g - 0.001).square().mean()
                    # Load-bearing forefoot horizontal anchor + pairwise anti-slip
                    if fw_t.sum() > 0:
                        loss = loss + 2400.0 * (df_p.square().sum(-1) * fw_t).sum() / fw_t.sum().clamp(min=1.0)
                        pair_fw = fw_t[1:] * fw_t[:-1]
                        if pair_fw.sum() > 0:
                            sf = fore_cur[1:] - fore_cur[:-1]
                            sf_p = sf - (sf @ u_t)[:, None] * u_t
                            loss = loss + 3600.0 * (sf_p.square().sum(-1) * pair_fw).sum() / pair_fw.sum().clamp(min=1.0)
                    # Load-bearing heel anchor (active when flat or heel-pivoting; off during tiptoe/forefoot turns!)
                    if hw_t.sum() > 0:
                        loss = loss + 1400.0 * (dh_p.square().sum(-1) * hw_t).sum() / hw_t.sum().clamp(min=1.0)
                        pair_hw = hw_t[1:] * hw_t[:-1]
                        if pair_hw.sum() > 0:
                            sh_v = heel_cur[1:] - heel_cur[:-1]
                            sh_p = sh_v - (sh_v @ u_t)[:, None] * u_t
                            loss = loss + 2200.0 * (sh_p.square().sum(-1) * pair_hw).sum() / pair_hw.sum().clamp(min=1.0)

                    # Distal foot prior, knee limits, boundary tether & temporal smoothness
                    dev_f = (p_cur[:, distal_idx] - src_foot).abs()
                    loss = loss + 120.0 * torch.relu(dev_f - foot_slack).square().mean()
                    loss = loss + 0.18 * d_leg.square().mean() + 18.0 * ((1.0 - act_t)[:, None] * d_leg.square()).mean()
                    if win_len > 1:
                        loss = loss + 3.5 * d_leg.diff(dim=0).square().mean()
                    if win_len > 2:
                        loss = loss + 14.0 * d_leg.diff(n=2, dim=0).square().mean()

                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        kn = base_t[:, cols[k_idx]] + d_leg[:, k_idx]
                        d_leg[:, k_idx] = kn.clamp(0.015, L.KNEE_MAX) - base_t[:, cols[k_idx]]

                # Cosine taper on surrounding padding frames
                idx_w = np.arange(win_len, dtype=float)
                left_ramp = np.clip((idx_w + 1) / (a - lo + 1), 0.0, 1.0) if a > lo else np.ones(win_len)
                right_ramp = np.clip((win_len - idx_w) / (hi - b + 1), 0.0, 1.0) if hi > b else np.ones(win_len)
                fade = np.minimum(left_ramp, right_ramp)
                fade = fade * fade * (3.0 - 2.0 * fade)
                fade[act] = 1.0

                cand_pose = out[lo:hi].copy()
                cand_pose[:, cols] += d_leg.detach().cpu().numpy() * fade[:, None]

                # Verify strict improvement on actual full skinned mesh (including padding transitions!) before accepting
                chk_lo = max(s0, lo - 1)
                chk_hi = min(s1, hi + 1)
                pose_chk = out[chk_lo:chk_hi].copy()
                pose_chk[lo - chk_lo:hi - chk_lo] = cand_pose
                V_chk_old = V[chk_lo:chk_hi]
                V_chk_new, _ = M.forward(shape, pose_chk, True, dev)

                V_cand = V_chk_new[a - chk_lo:b - chk_lo]
                V_eval = V_chk_new[eval_a - chk_lo:eval_b - chk_lo]
                p5_new = V_eval[:, pts5_ids[s_i]]
                fore_new = 0.5 * (p5_new[:, 2] + p5_new[:, 3])
                heel_new = p5_new[:, 0]
                if use_fore_piv:
                    piv_new = fore_new
                elif use_heel_piv:
                    piv_new = heel_new
                else:
                    piv_new = 0.5 * (fore_new + heel_new)
                dn = np.diff(piv_new, axis=0)
                dn_p = dn - (dn @ up)[:, None] * up
                new_max_slip = float(np.linalg.norm(dn_p, axis=1).max()) if len(dn_p) else 0.0

                # Also ensure near-ground padding transitions (< 3.5 cm) do not introduce a new slip jump
                f_old_all = 0.5 * (V_chk_old[:, pts5_ids[s_i][2]] + V_chk_old[:, pts5_ids[s_i][3]])
                f_new_all = 0.5 * (V_chk_new[:, pts5_ids[s_i][2]] + V_chk_new[:, pts5_ids[s_i][3]])
                d_old_all = np.diff(f_old_all, axis=0)
                d_new_all = np.diff(f_new_all, axis=0)
                step_old = np.linalg.norm(d_old_all - (d_old_all @ up)[:, None] * up, axis=1)
                step_new = np.linalg.norm(d_new_all - (d_new_all @ up)[:, None] * up, axis=1)
                sv_chk_new = V_chk_new[:, soles_full[s_i]]
                g_chk_new = (sv_chk_new @ up - collider.model_h(sv_chk_new.reshape(-1, 3)).reshape(len(V_chk_new), -1)).min(1)
                grounded_edge = np.maximum(g_chk_new[:-1], g_chk_new[1:]) < 0.035
                max_old_g = float(step_old[grounded_edge].max()) if np.any(grounded_edge) else 0.0
                max_new_g = float(step_new[grounded_edge].max()) if np.any(grounded_edge) else 0.0

                # Never disturb adjacent rock-solid grounded frames (< 0.8 cm/frame) on boundary/padding edges!
                n_edges = len(V_chk_new) - 1
                edge_idx = np.arange(n_edges)
                pad_edge = (edge_idx < (a - chk_lo)) | (edge_idx >= (b - 1 - chk_lo))
                low_chk_old = low_step[chk_lo:chk_hi - 1]
                low_chk_new = np.zeros(n_edges, dtype=float)
                for e_k in range(n_edges):
                    sv_ek = V_chk_new[e_k, soles_full[s_i]]
                    lk_e = int(np.argmin(sv_ek @ up - collider.model_h(sv_ek)))
                    dd_e = V_chk_new[e_k + 1, soles_full[s_i][lk_e]] - V_chk_new[e_k, soles_full[s_i][lk_e]]
                    low_chk_new[e_k] = float(np.linalg.norm(dd_e - (dd_e @ up) * up))
                rock_pad = pad_edge & grounded_edge & (low_chk_old < 0.008)
                rock_ok = bool(np.all(low_chk_new[rock_pad] <= low_chk_old[rock_pad] + 0.004)) if np.any(rock_pad) else True

                sv_new = V_cand[:, soles_full[s_i]]
                new_min_gap = float((sv_new @ up - collider.model_h(sv_new.reshape(-1, 3)).reshape(len(V_cand), -1)).min())

                if (new_max_slip < max_slip - 0.001
                        and rock_ok
                        and max_new_g <= max(max_old_g + 0.012, 0.045)
                        and new_min_gap >= min(-0.010, float(gap[a:b, s_i].min()) - 0.002)):
                    out[lo:hi] = cand_pose
                    # Refresh V, fore, heel, low_step on [lo:hi] so the opposite leg or next run sees updated state
                    V[lo:hi], _ = M.forward(shape, out[lo:hi], True, dev)
                    p5_u = V[lo:hi, pts5_ids[s_i]]
                    heel[lo:hi, s_i] = p5_u[:, 0]
                    fore[lo:hi, s_i] = 0.5 * (p5_u[:, 2] + p5_u[:, 3])
                    sv_u = V[lo:hi, soles_full[s_i]]
                    gap[lo:hi, s_i] = (sv_u @ up - collider.model_h(sv_u.reshape(-1, 3)).reshape(win_len, -1)).min(1)
                    for k_e in range(max(s0, lo - 1), min(s1 - 1, hi)):
                        sv_k = V[k_e, soles_full[s_i]]
                        lk = int(np.argmin(sv_k @ up - collider.model_h(sv_k)))
                        dd_l = V[k_e + 1, soles_full[s_i][lk]] - V[k_e, soles_full[s_i][lk]]
                        low_step[k_e] = float(np.linalg.norm(dd_l - (dd_l @ up) * up))
                    stabilized_runs[s_i] += 1
                    stabilized_frames += (b - a)

    if stabilized_frames > 0:
        report["单脚防滑稳固"] = {
            "左脚段数": stabilized_runs[0],
            "右脚段数": stabilized_runs[1],
            "稳固帧数": int(stabilized_frames),
        }
    return out, report
