#!/usr/bin/env python
"""
Experiment 2: Anatomical Surface Contact Pinning + Differentiable MHR Arm IK on demo1_multi_person
Solves partner-dance hand-into-waist penetration WITHOUT distorting feet or body silhouettes:
  1. Clean A-class full-body solve (preserving true visible footsteps at frame 149 & 0-2s).
  2. Anatomical Waist-Hold Surface Anchor: detects embrace intervals where P1's right hand holds P2's waist
     and P2's right hand holds P1's waist; binds palm target to a canonical anatomical surface vertex + outward
     normal offset (+1.6 cm) on the partner's moving torso (zero-slip local contact like JOSH L_static).
  3. Differentiable PyTorch MHR Arm-Only Optimization (cols 30:40 = r_clavicle, r_uparm, r_lowarm, r_wrist):
     - L_pin: anchors palm center to partner's waist surface patch
     - L_local_static: locks hand relative to partner's local waist vertex during hold
     - L_sdf: dense signed-distance clearance (>= 5 mm) for ALL upper-arm, forearm, wrist, and finger vertices
     - L_prior + L_smooth: natural, jitter-free joint rotations
"""
import os, sys, json, subprocess
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, "/home/ubuntu/sam3d/pipeline")
sys.path.insert(0, "/home/ubuntu/sam3d/data/outdoor/scripts")
import mhr_body as M
import body_into_scene as B
import body_motion as BM
import people_identity, people_store
from world_frame import export_T
import gravity_from_poles

R1 = "/home/ubuntu/sam3d/data/outdoor/result/demo1_multi_person/r1"
BODY = f"{R1}/body4d"
PY = os.path.expanduser("~/miniforge3/envs/sam3d/bin/python")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def hysteresis_runs(mask, max_gap=4, min_len=6):
    m = mask.copy()
    n = len(m)
    i = 0
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                m[i:j] = True
            i = j
        else:
            i += 1
    out = np.zeros(n, bool)
    runs = []
    i = 0
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if (j - i) >= min_len:
                out[i:j] = True
                runs.append((i, j))
            i = j
        else:
            i += 1
    return out, runs


def main():
    da3, cams, UP, GH, vid2key = B.load_scene(R1, dense=True)
    FL = gravity_from_poles.level_T(R1, UP) @ export_T(UP)
    FL_inv = np.linalg.inv(FL)

    fj = json.load(open(f"{BODY}/00_frames/frames.json"))
    kept = fj["kept_indices"]
    fps = float(fj.get("fps") or fj.get("src_fps") or 25.0)
    mesh_run = f"{BODY}/06_identity"
    par, shape = BM.load_params(mesh_run)
    focals = B.load_focals(mesh_run)

    res, info, rep = BM.solve(par, shape, cams, vid2key, kept, UP, GH, mesh_run, da3, fps, focals,
                              dev=dev, leg_fit=True, dist_keep=True)

    P1 = M.transform(res["1"]["params"], FL)
    P2 = M.transform(res["2"]["params"], FL)
    sh1, sh2 = res["1"]["shape"], res["2"]["shape"]
    vf1, vf2 = res["1"]["vf"], res["2"]["vf"]
    map1 = {int(f): i for i, f in enumerate(vf1)}
    map2 = {int(f): i for i, f in enumerate(vf2)}
    common = sorted(set(map1) & set(map2))
    print(f"Frames: P1={len(vf1)}, P2={len(vf2)}, common={len(common)}")

    mhr = M.model(dev)
    faces = mhr.character_torch.mesh.faces.cpu().numpy()
    sk = M.skeleton(dev)
    names = [str(x) for x in sk["names"]]
    dom, bw = np.zeros(int(sk["sk_v"].max()) + 1, int), np.zeros(int(sk["sk_v"].max()) + 1)
    for v, j, w in zip(sk["sk_v"], sk["sk_j"], sk["sk_w"]):
        if w > bw[v]:
            bw[v], dom[v] = w, j
    part_mask = lambda ss: np.isin(dom, [i for i, nm in enumerate(names) if any(s in nm for s in ss)])

    torso_m = part_mask(["root", "c_spine0", "c_spine1", "c_spine2", "c_spine3", "l_clavicle", "r_clavicle", "l_upleg", "r_upleg"])
    waist_m = part_mask(["root", "c_spine0", "c_spine1", "c_spine2"])
    r_arm_m = part_mask(["r_uparm", "r_lowarm", "r_wrist", "r_thumb", "r_index", "r_middle", "r_ring", "r_pinky"])
    r_palm_m = part_mask(["r_wrist", "r_index", "r_middle", "r_ring"])

    V1_0, S1_0 = M.forward(sh1, P1, False, dev)
    V2_0, S2_0 = M.forward(sh2, P2, False, dev)
    torso_idx = np.where(torso_m)[0][::3]
    sep1 = np.zeros((len(vf1), 3))
    sep2 = np.zeros((len(vf2), 3))
    for f in common:
        i1, i2 = map1[f], map2[f]
        m2 = trimesh.Trimesh(V2_0[i2], faces, process=False)
        pts2 = V2_0[i2, torso_idx]
        nrm2 = m2.vertex_normals[torso_idx]
        pts1 = V1_0[i1, torso_idx]
        d, nn = cKDTree(pts2).query(pts1)
        sd = ((pts1 - pts2[nn]) * nrm2[nn]).sum(1)
        ins = (d < 0.12) & (sd < -0.005)
        if ins.any():
            pen = float(np.percentile(-sd[ins], 90))
            u = V1_0[i1, waist_m].mean(0) - V2_0[i2, waist_m].mean(0)
            u[1] = 0.0
            u /= max(np.linalg.norm(u), 1e-6)
            sh_amt = min(pen + 0.008, 0.07)
            sep1[i1] = 0.5 * sh_amt * u
            sep2[i2] = -0.5 * sh_amt * u
    if np.linalg.norm(sep1, axis=1).max() > 0:
        for d_i in range(3):
            sep1[:, d_i] = gaussian_filter1d(sep1[:, d_i], sigma=3.5)
            sep2[:, d_i] = gaussian_filter1d(sep2[:, d_i], sigma=3.5)
        P1 = M.set_root(P1, None, M.root_pos(P1) + sep1)
        P2 = M.set_root(P2, None, M.root_pos(P2) + sep2)
        V1_0, S1_0 = M.forward(sh1, P1, False, dev)
        V2_0, S2_0 = M.forward(sh2, P2, False, dev)
        print(f"[Torso clearance] Applied mild torso-only clearance: max={np.linalg.norm(sep1,axis=1).max()*100:.1f} cm")

    def measure_arm_pen(Va, map_a, Vb, map_b):
        tb_i = np.where(torso_m)[0]
        max_p, n_fr = 0.0, 0
        for f in common:
            ia, ib = map_a[f], map_b[f]
            mb = trimesh.Trimesh(Vb[ib], faces, process=False)
            pts_b = Vb[ib, tb_i]
            nrm_b = mb.vertex_normals[tb_i]
            pa = Va[ia, r_arm_m]
            d, nn = cKDTree(pts_b).query(pa)
            sd = ((pa - pts_b[nn]) * nrm_b[nn]).sum(1)
            ins = (d < 0.16) & (sd < -0.003)
            if ins.any():
                n_fr += 1
                max_p = max(max_p, float((-sd[ins]).max() * 100.0))
        return n_fr, max_p

    b_n1, b_p1 = measure_arm_pen(V1_0, map1, V2_0, map2)
    b_n2, b_p2 = measure_arm_pen(V2_0, map2, V1_0, map1)
    print(f"[Before Arm IK] P1_R_arm->P2: {b_n1} frames (max {b_p1:.2f} cm) | P2_R_arm->P1: {b_n2} frames (max {b_p2:.2f} cm)")

    R_ARM_COLS = list(range(30, 40))
    sh_j_idx = names.index("r_uparm")
    wr_j_idx = names.index("r_wrist")

    def solve_partner_arm_ik(P_self, vf_self, sh_self, V_self0, S_self0, V_part, map_part, sh_part, label):
        T_s = len(vf_self)
        tb_idx = np.where(torso_m)[0]
        wb_idx = np.where(waist_m)[0]
        rest_part = M.rest_verts(sh_part, dev)

        # Partner torso points aligned to vf_self (for frames without partner, put far away at +100m)
        part_pts = np.full((T_s, len(tb_idx), 3), 100.0, np.float32)
        part_nrm = np.zeros((T_s, len(tb_idx), 3), np.float32)
        part_nrm[:, :, 1] = 1.0
        has_part = np.zeros(T_s, bool)

        raw_contact = np.zeros(T_s, bool)
        proj_vid = np.full(T_s, int(wb_idx[0]), int)
        arm_len = float(np.median(np.linalg.norm(S_self0[:, wr_j_idx, :3] - S_self0[:, sh_j_idx, :3], axis=1)) / 100.0) + 0.08

        for ts, f in enumerate(vf_self):
            f_int = int(f)
            if f_int not in map_part:
                continue
            has_part[ts] = True
            tp = map_part[f_int]
            mb = trimesh.Trimesh(V_part[tp], faces, process=False)
            part_pts[ts] = V_part[tp, tb_idx]
            part_nrm[ts] = mb.vertex_normals[tb_idx]

            palm_c = V_self0[ts, r_palm_m].mean(0)
            arm_pts = V_self0[ts, r_arm_m]
            tree_t = cKDTree(part_pts[ts])
            d_a, nn_a = tree_t.query(arm_pts)
            sd_a = ((arm_pts - part_pts[ts, nn_a]) * part_nrm[ts, nn_a]).sum(1)
            if ((d_a < 0.16) & (sd_a < 0.015)).any() or tree_t.query(palm_c)[0] < 0.07:
                raw_contact[ts] = True
            sh_pos = S_self0[ts, sh_j_idx, :3] / 100.0
            w_pts = V_part[tp, wb_idx]
            reach_d = np.linalg.norm(w_pts - sh_pos, axis=1)
            cand_m = reach_d < arm_len * 1.05
            cand_indices = wb_idx[cand_m] if cand_m.any() else wb_idx
            _, idx_w = cKDTree(V_part[tp, cand_indices]).query(palm_c)
            proj_vid[ts] = int(cand_indices[idx_w])

        c_mask, runs = hysteresis_runs(raw_contact & has_part, max_gap=4, min_len=6)
        runs_vf = [(int(vf_self[s]), int(vf_self[e - 1])) for s, e in runs]
        print(f"  [{label}] Contact intervals (vf): {runs_vf} (total {c_mask.sum()} frames)")

        anchor_vid = proj_vid.copy()
        pin_weight = np.zeros(T_s, np.float32)
        static_pair = np.zeros(T_s - 1, bool)
        for s_i, e_i in runs:
            r_coords = rest_part[proj_vid[s_i:e_i]].copy()
            for d_i in range(3):
                r_coords[:, d_i] = gaussian_filter1d(r_coords[:, d_i], sigma=4.0)
            _, sm_idx = cKDTree(rest_part[wb_idx]).query(r_coords)
            vids_run = wb_idx[sm_idx]
            anchor = vids_run[0]
            for k_i in range(len(vids_run)):
                if np.linalg.norm(rest_part[vids_run[k_i]] - rest_part[anchor]) <= 4.0:
                    vids_run[k_i] = anchor
                else:
                    anchor = vids_run[k_i]
            anchor_vid[s_i:e_i] = vids_run
            w_run = np.ones(e_i - s_i, np.float32)
            ramp = min(3, (e_i - s_i) // 2)
            if ramp > 0:
                w_run[:ramp] = np.linspace(0.3, 1.0, ramp)
                w_run[-ramp:] = np.linspace(1.0, 0.3, ramp)
            pin_weight[s_i:e_i] = w_run * has_part[s_i:e_i].astype(np.float32)
            for ts in range(s_i, e_i - 1):
                if has_part[ts] and has_part[ts + 1] and anchor_vid[ts] == anchor_vid[ts + 1]:
                    static_pair[ts] = True

        anchor_pos = np.zeros((T_s, 3), np.float32)
        tb_map = {int(v): i for i, v in enumerate(tb_idx)}
        for ts, f in enumerate(vf_self):
            f_int = int(f)
            if f_int in map_part:
                tp = map_part[f_int]
                vid = int(anchor_vid[ts])
                n_v = part_nrm[ts, tb_map[vid]]
                anchor_pos[ts] = V_part[tp, vid] + 0.016 * n_v

        P_base_t = torch.tensor(P_self, dtype=torch.float32, device=dev)
        theta0 = P_base_t[:, R_ARM_COLS].clone()
        d_theta = torch.zeros_like(theta0, requires_grad=True)
        sh_t = torch.tensor(sh_self, dtype=torch.float32, device=dev).expand(T_s, -1)
        expr_t = torch.zeros((T_s, M.NEXPR), dtype=torch.float32, device=dev)

        tb_sub = np.linspace(0, len(tb_idx) - 1, min(650, len(tb_idx)), dtype=int)
        part_pts_t = torch.tensor(part_pts[:, tb_sub, :], dtype=torch.float32, device=dev)
        part_nrm_t = torch.tensor(part_nrm[:, tb_sub, :], dtype=torch.float32, device=dev)

        arm_all_idx = np.where(r_arm_m)[0]
        arm_sub_idx = arm_all_idx[np.linspace(0, len(arm_all_idx) - 1, min(420, len(arm_all_idx)), dtype=int)]
        arm_sub_t = torch.tensor(arm_sub_idx, dtype=torch.long, device=dev)
        palm_idx_t = torch.tensor(np.where(r_palm_m)[0][::2], dtype=torch.long, device=dev)

        anchor_pos_t = torch.tensor(anchor_pos, dtype=torch.float32, device=dev)
        pin_w_t = torch.tensor(pin_weight, dtype=torch.float32, device=dev)
        stat_t = torch.tensor(static_pair, dtype=torch.bool, device=dev)

        opt = torch.optim.Adam([d_theta], lr=0.025)
        for it in range(260):
            opt.zero_grad()
            P_cur = P_base_t.clone()
            P_cur[:, R_ARM_COLS] = theta0 + d_theta
            V_out, _ = mhr(sh_t, P_cur, expr_t, False)
            V_cur = V_out / 100.0

            V_arm = V_cur[:, arm_sub_t, :]
            V_palm = V_cur[:, palm_idx_t, :]
            palm_c = V_palm.mean(dim=1)

            dists = torch.cdist(V_arm, part_pts_t)
            min_d, nn_i = dists.min(dim=-1)
            nn_pts = torch.gather(part_pts_t, 1, nn_i.unsqueeze(-1).expand(-1, -1, 3))
            nn_nrm = torch.gather(part_nrm_t, 1, nn_i.unsqueeze(-1).expand(-1, -1, 3))
            sd = ((V_arm - nn_pts) * nn_nrm).sum(dim=-1)
            near_gate = (min_d < 0.16).float()
            pen = F.relu(0.005 - sd) * near_gate
            loss_sdf = 180.0 * pen.pow(2).sum(dim=1).mean()

            loss_pin = (pin_w_t * (palm_c - anchor_pos_t).pow(2).sum(dim=-1)).mean() * 35.0

            rel_palm = palm_c - anchor_pos_t
            loss_stat = torch.tensor(0.0, device=dev)
            if stat_t.any():
                loss_stat = 60.0 * (rel_palm[1:][stat_t] - rel_palm[:-1][stat_t]).pow(2).sum(dim=-1).mean()

            theta_cur = theta0 + d_theta
            vel = (theta_cur[1:] - theta_cur[:-1]) * fps
            acc = (theta_cur[2:] - 2 * theta_cur[1:-1] + theta_cur[:-2]) * fps
            non_c = (1.0 - pin_w_t)[:, None]
            loss_prior = (0.06 * d_theta.pow(2) + 0.35 * non_c * d_theta.pow(2)).mean()
            loss_sm = 0.02 * vel.pow(2).mean() + 0.08 * acc.pow(2).mean()

            loss = loss_sdf + loss_pin + loss_stat + loss_prior + loss_sm
            loss.backward()
            torch.nn.utils.clip_grad_norm_([d_theta], 2.0)
            opt.step()

        P_opt = P_self.copy()
        P_opt[:, R_ARM_COLS] = (theta0 + d_theta).detach().cpu().numpy()
        V_opt, _ = M.forward(sh_self, P_opt, False, dev)
        return P_opt, V_opt, runs_vf

    P1_opt, V1_opt, runs1 = solve_partner_arm_ik(P1, vf1, sh1, V1_0, S1_0, V2_0, map2, sh2, "P1(Man)_R_Arm")
    P2_opt, V2_opt, runs2 = solve_partner_arm_ik(P2, vf2, sh2, V2_0, S2_0, V1_opt, map1, sh1, "P2(Woman)_R_Arm")

    a_n1, a_p1 = measure_arm_pen(V1_opt, map1, V2_opt, map2)
    a_n2, a_p2 = measure_arm_pen(V2_opt, map2, V1_opt, map1)
    print(f"[After Arm IK]  P1_R_arm->P2: {a_n1} frames (max {a_p1:.2f} cm) | P2_R_arm->P1: {a_n2} frames (max {a_p2:.2f} cm)")

    P1_da3 = M.transform(P1_opt, FL_inv)
    P2_da3 = M.transform(P2_opt, FL_inv)
    W1_da3, _ = M.forward(sh1, P1_da3, True, dev)
    W2_da3, _ = M.forward(sh2, P2_da3, True, dev)

    res["1"]["params"] = P1_da3
    res["2"]["params"] = P2_da3
    verts = {"1": W1_da3, "2": W2_da3}
    FLIP = export_T(UP)
    body_wall = B.people_body({pid: dict(zip(res[pid]["vf"].tolist(), verts[pid].astype(np.float64))) for pid in res}, FLIP)
    meta = json.load(open(f"{R1}/10_people.json"))
    meta["pair_arm_ik"] = {
        "before": {"p1_frames": b_n1, "p1_max_cm": round(b_p1, 2), "p2_frames": b_n2, "p2_max_cm": round(b_p2, 2)},
        "after": {"p1_frames": a_n1, "p1_max_cm": round(a_p1, 2), "p2_frames": a_n2, "p2_max_cm": round(a_p2, 2)},
        "p1_pinned_runs": runs1,
        "p2_pinned_runs": runs2,
    }
    people_store.save(R1, meta, res, verts, faces, body_wall, B.BODY_PITCH, fps)
    print("[Saved] Updated 10_people_params.npz & 10_people_world.npz for demo1_multi_person!")

    subprocess.run([PY, "/home/ubuntu/sam3d/pipeline/outdoor_pipeline.py",
                    "--scene", "demo1_multi_person", "--run", "r1",
                    "--only", "people_glb,motion,render"], check=True)


if __name__ == "__main__":
    main()
