#!/usr/bin/env python
"""
Experiment 1: Contact-Guided Planar Support & Kinematic Optimization on demo3_backup_backflip
Synthesizing:
  1. CRISP (ICLR 2026): Upward planar primitive extraction (true gravity UP + multi-height support planes)
     & contact-guided planar primitive refinement (matching concrete_block_1 3D box to stance contacts).
  2. MeshMimic (arXiv:2602.15733): Kinematic consistency optimization (L_align + L_c + L_p [3D Box TSDF]
     + L_sm + L_fs [foot-snapping]) so stance feet rest cleanly on block/street and aerial flip has 0 penetration.
  3. JOSH (ICLR 2026): Stationary contact anti-slip loss (L_static) on planted stance feet.
"""
import os, sys, json, subprocess
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import cv2
from PIL import Image

sys.path.insert(0, "/home/ubuntu/sam3d/pipeline")
sys.path.insert(0, "/home/ubuntu/sam3d/data/outdoor/scripts")
import mhr_body as M
import body_into_scene as B
import body_motion as BM
import people_identity, people_store
from world_frame import export_T, inverse_T

R3 = "/home/ubuntu/sam3d/data/outdoor/result/demo3_backup_backflip/r1"
BODY = f"{R3}/body4d"
PY = os.path.expanduser("~/miniforge3/envs/sam3d/bin/python")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def step1_crisp_planar_up():
    cams_da3 = json.load(open(f"{R3}/02_da3/cameras.json"))["cameras"]
    fm_da3 = json.load(open(f"{R3}/02_da3/frame_map.json"))["frames"]
    pts_all, nrm_all = [], []
    for k in sorted(cams_da3.keys())[::8]:
        dp = f"{R3}/02_da3/depth/{k[:-4]}_depth.npy"
        if not os.path.exists(dp):
            continue
        d = np.load(dp).astype(np.float64)
        H, W = d.shape
        c = cams_da3[k]
        K = np.array(c["intrinsic_3x3"], float)
        h0, w0 = c["input_hw"]
        fx, fy, cx, cy = K[0, 0] * W / w0, K[1, 1] * H / h0, K[0, 2] * W / w0, K[1, 2] * H / h0
        ys, xs = np.mgrid[0:H, 0:W]
        Pc = np.stack([(xs - cx) / fx * d, (ys - cy) / fy * d, d], -1)
        dx = Pc[1:-1, 2:] - Pc[1:-1, :-2]
        dy = Pc[2:, 1:-1] - Pc[:-2, 1:-1]
        n = np.cross(dx, dy)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        n[(n * Pc[1:-1, 1:-1]).sum(-1) > 0] *= -1
        vf = int(fm_da3[k])
        mp = f"{BODY}/06_identity/masks/{vf:08d}.png"
        skip = np.zeros((H, W), bool)
        if os.path.exists(mp):
            pm = np.array(Image.open(mp)) > 0
            skip = cv2.resize(pm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        E = np.array(c["extrinsic_w2c_4x4"], float)
        R_c, t_c = E[:3, :3], E[:3, 3]
        Pw = (Pc[1:-1, 1:-1] - t_c) @ R_c
        Nw = n @ R_c
        ok = (~skip[1:-1, 1:-1]) & (d[1:-1, 1:-1] < 7.5) & (Nw[..., 1] < -0.75)
        pts_all.append(Pw[ok][::2])
        nrm_all.append(Nw[ok][::2])

    P_all = np.concatenate(pts_all)
    N_all = np.concatenate(nrm_all)
    u0 = np.median(N_all, axis=0)
    u0 /= np.linalg.norm(u0)
    inl = np.degrees(np.arccos(np.clip(N_all @ u0, -1, 1))) < 8.0
    u_up = np.median(N_all[inl], axis=0)
    u_up /= np.linalg.norm(u_up)
    hs = P_all[inl] @ u_up
    gh_street = float(np.median(hs[(hs > -0.44) & (hs < -0.32)]))
    h_block_pts = float(np.median(hs[(hs > 0.08) & (hs < 0.18)]))
    up_doc = {
        "up": u_up.tolist(),
        "ground_h": gh_street,
        "偏离老假设度数": round(float(np.degrees(np.arccos(np.clip(u_up @ np.array([0, -1, 0]), -1, 1)))), 2),
        "来源": "CRISP平面基元提取(路面+水泥台顶面法向)",
        "block_top_da3_h": h_block_pts,
    }
    for sub in ["02_da3", "02_da3/dense"]:
        p = f"{R3}/{sub}"
        if os.path.isdir(p):
            json.dump(up_doc, open(f"{p}/world_up.json", "w"), ensure_ascii=False, indent=2)
            json.dump(up_doc, open(f"{p}/up_override.json", "w"), ensure_ascii=False, indent=2)
    print(f"[Step 1 CRISP UP] u_up={np.round(u_up,4)}, pitch={up_doc['偏离老假设度数']} deg, gh_street={gh_street:.3f}m, block_top_pts={h_block_pts:.3f}m")
    return u_up, gh_street, h_block_pts


def box_sdf_torch(pts, bmin, bmax):
    """Exact differentiable 3D AABB signed distance: >0 outside, <0 inside."""
    center = 0.5 * (bmin + bmax)
    half = 0.5 * (bmax - bmin)
    q = torch.abs(pts - center) - half
    outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.clamp(torch.max(q, dim=-1).values, max=0.0)
    return outside + inside


def main():
    u_up, gh_street, h_block_pts = step1_crisp_planar_up()
    subprocess.run([PY, "/home/ubuntu/sam3d/pipeline/outdoor_pipeline.py",
                    "--scene", "demo3_backup_backflip", "--run", "r1",
                    "--only", "boxes,assemble"], check=True)

    da3, cams, UP, GH, vid2key = B.load_scene(R3, dense=True)
    T_exp = export_T(UP)
    T_inv = inverse_T(UP)

    fj = json.load(open(f"{BODY}/00_frames/frames.json"))
    kept = fj["kept_indices"]
    fps = float(fj.get("fps") or fj.get("src_fps") or 24.0)
    mesh_run = f"{BODY}/06_identity"
    par, shape = BM.load_params(mesh_run)

    bis = sorted(par["1"].keys())
    T = len(bis)
    fm_sparse = json.load(open(f"{R3}/02_da3/frame_map.json"))["frames"]
    vid2key_sparse = {int(v): k for k, v in fm_sparse.items()}
    rt = {b: par["1"][b][1] for b in bis}
    ratios = B.depth_ratio("1", bis, rt, mesh_run, f"{R3}/02_da3", vid2key_sparse, kept)
    R_scale = float(np.median(list(ratios.values())))

    c2w = np.stack([np.linalg.inv(np.array(cams[vid2key[kept[b]]]["extrinsic_w2c_4x4"], float)) for b in bis])
    Pc = np.stack([par["1"][b][0] for b in bis])
    ct = np.stack([rt[b] for b in bis])
    Pw0 = M.to_world(Pc, ct * R_scale, c2w)
    tt = np.array([kept[b] / fps for b in bis])
    pose = Pw0[:, BM.POSE].copy()
    Q = M.root_rot(Pw0)
    p0 = M.root_pos(Pw0)

    reliable = np.ones(T, bool)
    pose = BM.fix_bone_lengths(pose, reliable, M.length_params(dev) - BM.POSE.start)
    Pw0[:, M.NPOSE:] = BM.fix_bone_lengths(Pw0[:, M.NPOSE:], reliable, np.arange(M.NSCALE))
    groups = {n: c - BM.POSE.start for n, c in M.limb_groups(dev).items()}
    pose, _ = BM.fix_limb_spikes(pose, tt, groups)
    pose = BM.smooth_joints(pose, tt, BM.limb_speed_fn(shape["1"], Pw0[0, M.NPOSE:], tt, dev))
    Q = BM.smooth_rots(Q, tt)
    Pw_clean = Pw0.copy()
    Pw_clean[:, BM.POSE] = pose
    Pw_clean = M.set_root(Pw_clean, Q, p0)

    P_gltf = M.transform(Pw_clean, T_exp)
    V_init, S_init = M.forward(shape["1"], P_gltf, False, dev)

    rest = M.rest_verts(shape["1"], dev)
    sk = M.skeleton(dev)
    jnames = [str(x) for x in sk["names"]]
    dom, bw = np.zeros(V_init.shape[1], int), np.zeros(V_init.shape[1])
    for v, j, w in zip(sk["sk_v"], sk["sk_j"], sk["sk_w"]):
        if w > bw[v]:
            bw[v], dom[v] = w, j
    l_foot_j = [i for i, nm in enumerate(jnames) if nm in ("l_talocrural", "l_subtalar", "l_transversetarsal", "l_ball")]
    r_foot_j = [i for i, nm in enumerate(jnames) if nm in ("r_talocrural", "r_subtalar", "r_transversetarsal", "r_ball")]
    l_fv = np.where(np.isin(dom, l_foot_j))[0]
    r_fv = np.where(np.isin(dom, r_foot_j))[0]
    l_sole = l_fv[rest[l_fv, 1] < np.percentile(rest[l_fv, 1], 25)]
    r_sole = r_fv[rest[r_fv, 1] < np.percentile(rest[r_fv, 1], 25)]

    lf_pos = V_init[:, l_sole].mean(1)
    rf_pos = V_init[:, r_sole].mean(1)
    lf_ymin = V_init[:, l_sole, 1].min(1)
    rf_ymin = V_init[:, r_sole, 1].min(1)
    lf_vel = np.linalg.norm(np.diff(lf_pos, axis=0, prepend=lf_pos[:1]), axis=1) * fps
    rf_vel = np.linalg.norm(np.diff(rf_pos, axis=0, prepend=rf_pos[:1]), axis=1) * fps

    stance_heights = np.concatenate([lf_ymin[:75], rf_ymin[:60], lf_ymin[125:145]])
    y_block_top = float(0.5 * h_block_pts + 0.5 * np.median(stance_heights))
    print(f"[Step 2 CRISP Contact] Stance foot median y={np.median(stance_heights):.3f}m, DA3 block top={h_block_pts:.3f}m -> Fused y_block_top={y_block_top:.3f}m (height above street: {(y_block_top - GH)*100:.1f} cm)")

    sc = trimesh.load(f"{R3}/07_scene.glb", force="scene")
    blk_foot_pts = []
    for t in range(147):
        if abs(lf_ymin[t] - y_block_top) < 0.08:
            blk_foot_pts.append(V_init[t, l_sole][:, [0, 2]])
        if abs(rf_ymin[t] - y_block_top) < 0.08:
            blk_foot_pts.append(V_init[t, r_sole][:, [0, 2]])
    blk_xz = np.concatenate(blk_foot_pts, axis=0)
    land_xz = np.concatenate([V_init[190:, l_sole, :][:, :, [0, 2]].reshape(-1, 2),
                              V_init[190:, r_sole, :][:, :, [0, 2]].reshape(-1, 2)], axis=0)

    print(f"  Block stance XZ bounds: min={np.round(blk_xz.min(0),3)}, max={np.round(blk_xz.max(0),3)}")
    print(f"  Street landing XZ bounds: min={np.round(land_xz.min(0),3)}, max={np.round(land_xz.max(0),3)}")

    blk_bounds = None
    for nd in list(sc.graph.nodes_geometry):
        if "concrete_block_1" in nd:
            T_nd, gname = sc.graph[nd]
            geom = sc.geometry[gname]
            vg = trimesh.transform_points(geom.vertices, T_nd)
            print(f"  Original concrete_block_1 glTF bounds: min={np.round(vg.min(0),3)}, max={np.round(vg.max(0),3)}")
            # Keep block strictly separated from street landing footprint along whichever axis separates them!
            blk_c = np.median(blk_xz, axis=0)
            land_c = np.median(land_xz, axis=0)
            print(f"  Stance center XZ={np.round(blk_c,3)} vs Landing center XZ={np.round(land_c,3)}")

            x_min_t = float(blk_xz[:, 0].min()) - 0.14
            x_max_t = float(blk_xz[:, 0].max()) + 0.14
            z_min_t = float(blk_xz[:, 1].min()) - 0.14
            z_max_t = float(blk_xz[:, 1].max()) + 0.18
            # Ensure landing XZ at t>=192 is outside [x_min_t, x_max_t] x [z_min_t, z_max_t]
            if abs(land_c[1] - blk_c[1]) >= abs(land_c[0] - blk_c[0]):
                if land_c[1] > blk_c[1]:
                    z_max_t = min(z_max_t, float(np.percentile(land_xz[:, 1], 5)) - 0.10)
                else:
                    z_min_t = max(z_min_t, float(np.percentile(land_xz[:, 1], 95)) + 0.10)
            else:
                if land_c[0] > blk_c[0]:
                    x_max_t = min(x_max_t, float(np.percentile(land_xz[:, 0], 5)) - 0.10)
                else:
                    x_min_t = max(x_min_t, float(np.percentile(land_xz[:, 0], 95)) + 0.10)

            y_min_t = float(GH)
            y_max_t = float(y_block_top)

            v0 = vg.copy()
            lo0, hi0 = v0.min(0), v0.max(0)
            target_lo = np.array([x_min_t, y_min_t, z_min_t])
            target_hi = np.array([x_max_t, y_max_t, z_max_t])
            v_new = (v0 - lo0) / np.maximum(hi0 - lo0, 1e-6) * (target_hi - target_lo) + target_lo
            geom.vertices = trimesh.transform_points(v_new, np.linalg.inv(T_nd))
            blk_bounds = (target_lo, target_hi)
            print(f"  CRISP-Refined concrete_block_1 glTF bounds: min={np.round(target_lo,3)}, max={np.round(target_hi,3)}")

    sc.export(f"{R3}/07_scene.glb")
    sc.export(f"{R3}/07_scene_phy.glb")
    assert blk_bounds is not None, "concrete_block_1 not found in 07_scene.glb!"

    bmin_t = torch.tensor(blk_bounds[0], dtype=torch.float32, device=dev)
    bmax_t = torch.tensor(blk_bounds[1], dtype=torch.float32, device=dev)

    V_base = torch.tensor(V_init, dtype=torch.float32, device=dev)
    sub_idx = np.unique(np.concatenate([np.arange(0, V_init.shape[1], 6), l_sole, r_sole]))
    V_sub = V_base[:, sub_idx, :]
    l_sole_t = torch.tensor(l_sole, dtype=torch.long, device=dev)
    r_sole_t = torch.tensor(r_sole, dtype=torch.long, device=dev)

    w2c_gltf = np.stack([np.array(cams[vid2key[kept[b]]]["extrinsic_w2c_4x4"], float) @ T_inv for b in bis])
    R_w2c = torch.tensor(w2c_gltf[:, :3, :3], dtype=torch.float32, device=dev)
    t_w2c = torch.tensor(w2c_gltf[:, :3, 3], dtype=torch.float32, device=dev)
    K_cam = np.stack([np.array(cams[vid2key[kept[b]]]["intrinsic_3x3"], float) for b in bis])
    K_t = torch.tensor(K_cam, dtype=torch.float32, device=dev)

    root_0 = torch.tensor(M.root_pos(P_gltf), dtype=torch.float32, device=dev)

    def proj_2d(pts_3d):
        pc = torch.einsum("tij,tj->ti", R_w2c, pts_3d) + t_w2c
        uv = torch.einsum("tij,tj->ti", K_t, pc)
        return uv[:, :2] / torch.clamp(uv[:, 2:3], min=0.1), pc[:, 2]

    uv0, z0 = proj_2d(root_0)

    l_blk_c = np.zeros(T, bool)
    r_blk_c = np.zeros(T, bool)
    l_blk_c[:147] = np.abs(lf_ymin[:147] - y_block_top) < 0.09
    r_blk_c[:147] = np.abs(rf_ymin[:147] - y_block_top) < 0.09
    l_str_c = np.zeros(T, bool)
    r_str_c = np.zeros(T, bool)
    l_str_c[192:] = np.abs(lf_ymin[192:] - GH) < 0.09
    r_str_c[192:] = np.abs(rf_ymin[192:] - GH) < 0.09

    l_stat = (l_blk_c | l_str_c) & (lf_vel < 0.35)
    r_stat = (r_blk_c | r_str_c) & (rf_vel < 0.35)

    l_blk_t = torch.tensor(l_blk_c, device=dev)
    r_blk_t = torch.tensor(r_blk_c, device=dev)
    l_str_t = torch.tensor(l_str_c, device=dev)
    r_str_t = torch.tensor(r_str_c, device=dev)
    l_stat_t = torch.tensor(l_stat[1:] & l_stat[:-1], device=dev)
    r_stat_t = torch.tensor(r_stat[1:] & r_stat[:-1], device=dev)

    dp = torch.zeros((T, 3), dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([dp], lr=0.015)

    for it in range(300):
        opt.zero_grad()
        V_cur = V_sub + dp[:, None, :]
        Vl = V_base[:, l_sole_t, :] + dp[:, None, :]
        Vr = V_base[:, r_sole_t, :] + dp[:, None, :]

        uv_cur, z_cur = proj_2d(root_0 + dp)
        loss_align = ((uv_cur - uv0) / 25.0).pow(2).mean() + 0.8 * (z_cur - z0).pow(2).mean()

        ly_min = Vl[:, :, 1].min(dim=1).values
        ry_min = Vr[:, :, 1].min(dim=1).values
        loss_contact = torch.tensor(0.0, device=dev)
        if l_blk_t.any():
            loss_contact = loss_contact + 45.0 * (ly_min[l_blk_t] - y_block_top).pow(2).mean()
        if r_blk_t.any():
            loss_contact = loss_contact + 45.0 * (ry_min[r_blk_t] - y_block_top).pow(2).mean()
        if l_str_t.any():
            loss_contact = loss_contact + 45.0 * (ly_min[l_str_t] - float(GH)).pow(2).mean()
        if r_str_t.any():
            loss_contact = loss_contact + 45.0 * (ry_min[r_str_t] - float(GH)).pow(2).mean()

        sdf_box = box_sdf_torch(V_cur, bmin_t, bmax_t - torch.tensor([0.0, 0.003, 0.0], device=dev))
        pen_box = F.relu(-sdf_box)
        sdf_gnd = V_cur[:, :, 1] - (float(GH) - 0.003)
        pen_gnd = F.relu(-sdf_gnd)
        loss_pen = 140.0 * (F.huber_loss(pen_box, torch.zeros_like(pen_box), delta=0.02, reduction="none").sum(dim=1).mean() +
                            F.huber_loss(pen_gnd, torch.zeros_like(pen_gnd), delta=0.02, reduction="none").sum(dim=1).mean())

        l_mean = Vl.mean(dim=1)
        r_mean = Vr.mean(dim=1)
        loss_static = torch.tensor(0.0, device=dev)
        if l_stat_t.any():
            loss_static = loss_static + 15.0 * (l_mean[1:][l_stat_t] - l_mean[:-1][l_stat_t]).pow(2).mean()
        if r_stat_t.any():
            loss_static = loss_static + 15.0 * (r_mean[1:][r_stat_t] - r_mean[:-1][r_stat_t]).pow(2).mean()

        vel = (dp[1:] - dp[:-1]) * fps
        acc = (dp[2:] - 2 * dp[1:-1] + dp[:-2]) * fps
        loss_sm = 0.15 * vel.pow(2).mean() + 0.6 * acc.pow(2).mean()

        loss = loss_align + loss_contact + loss_pen + loss_static + loss_sm
        loss.backward()
        opt.step()

    dp_np = dp.detach().cpu().numpy()
    P_gltf_opt = M.set_root(P_gltf, None, M.root_pos(P_gltf) + dp_np)
    with torch.no_grad():
        V_fin = V_sub + dp[:, None, :]
        sdf_fin = box_sdf_torch(V_fin, bmin_t, bmax_t)
        max_pen_cm = float(F.relu(-sdf_fin).max().item() * 100.0)
        mean_dp_cm = float(np.linalg.norm(dp_np, axis=1).mean() * 100.0)
        uv_fin, _ = proj_2d(root_0 + dp)
        px_drift = float(torch.linalg.norm(uv_fin - uv0, dim=-1).mean().item())
    print(f"[Step 3 MeshMimic+JOSH Done] Max block penetration: {max_pen_cm:.2f} cm | Mean root adjust: {mean_dp_cm:.1f} cm | Mean 2D root drift: {px_drift:.1f} px")

    P_da3_opt = M.transform(P_gltf_opt, T_inv)
    V_da3_opt, _ = M.forward(shape["1"], P_da3_opt, True, dev)
    res_out = {
        "1": dict(frames=bis, vf=np.array([kept[b] for b in bis], np.int32),
                  params=P_da3_opt, shape=shape["1"], conf=np.ones(T, np.float32))
    }
    body_wall = B.people_body({"1": dict(zip([kept[b] for b in bis], V_da3_opt.astype(np.float64)))}, T_exp)
    meta = json.load(open(f"{R3}/10_people.json"))
    meta["floor_y"] = float(GH)
    meta["hsi_optimization"] = {
        "method": "CRISP_planar_support + MeshMimic_TSDF_kinematic + JOSH_static_contact",
        "block_top_y_m": round(y_block_top, 4),
        "street_y_m": round(float(GH), 4),
        "max_block_penetration_cm": round(max_pen_cm, 2),
        "mean_2d_drift_px": round(px_drift, 2),
    }
    faces = M.model(dev).character_torch.mesh.faces.cpu().numpy()
    people_store.save(R3, meta, res_out, {"1": V_da3_opt}, faces, body_wall, B.BODY_PITCH, fps)
    print("[Saved] Updated 10_people_params.npz & 10_people_world.npz for demo3_backup_backflip!")

    subprocess.run([PY, "/home/ubuntu/sam3d/pipeline/outdoor_pipeline.py",
                    "--scene", "demo3_backup_backflip", "--run", "r1",
                    "--only", "people_glb,motion,render"], check=True)


if __name__ == "__main__":
    main()
