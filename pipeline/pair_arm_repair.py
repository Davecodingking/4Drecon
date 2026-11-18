"""Rule 3: Multi-person close contact allowed; repair arm-into-torso penetration locally without worsening any frame."""
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial import cKDTree

import legs_proc as L
import mhr_body as M

ARM_COLS = {
    "r": list(range(30, 40)),  # r_clavicle (3), r_uparm (3), r_lowarm (1), r_wrist (3)
    "l": list(range(40, 50)),  # l_clavicle (3), l_uparm (3), l_lowarm (1), l_wrist (3)
}
# Penalize clavicle shrugs more than shoulder/elbow/wrist flexion so arm clearance looks natural
DOF_WEIGHTS = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 0.8, 1.4, 1.4, 1.4], dtype=np.float32)
CONTACT_TOL_M = 0.003  # 3 mm: surface contact allowed without penalty!


def _part_masks(dev="cuda"):
    sk = M.skeleton(dev)
    names = [str(x) for x in sk["names"]]
    n_v = int(sk["sk_v"].max()) + 1
    dom = np.zeros(n_v, dtype=int)
    bw = np.zeros(n_v, dtype=float)
    for v, j, w in zip(sk["sk_v"], sk["sk_j"], sk["sk_w"]):
        if w > bw[v]:
            bw[v], dom[v] = w, j

    def mask_of(substrings):
        j_ids = [i for i, nm in enumerate(names) if any(s in nm for s in substrings)]
        return np.isin(dom, j_ids)

    torso_m = mask_of(["root", "c_spine0", "c_spine1", "c_spine2", "c_spine3", "l_clavicle", "r_clavicle", "l_upleg", "r_upleg"])
    arms_m = {
        "r": mask_of(["r_uparm", "r_lowarm", "r_wrist", "r_thumb", "r_index", "r_middle", "r_ring", "r_pinky"]),
        "l": mask_of(["l_uparm", "l_lowarm", "l_wrist", "l_thumb", "l_index", "l_middle", "l_ring", "l_pinky"]),
    }
    spine_j = [names.index(nm) for nm in ("root", "c_spine0", "c_spine1", "c_spine2", "c_spine3")]
    uparm_j = {"r": names.index("r_uparm"), "l": names.index("l_uparm")}
    return torso_m, arms_m, spine_j, uparm_j


def _outward_torso_normals(V_frame, S_frame, faces, tb_idx, spine_j):
    """Compute outward-verified normals on partner torso vertices using spine centerline orientation."""
    mesh = trimesh.Trimesh(V_frame, faces, process=False)
    pts = V_frame[tb_idx]
    nrm = mesh.vertex_normals[tb_idx].copy()
    spine = S_frame[spine_j, :3] / 100.0  # (5, 3) metres
    # Closest spine node for radial outward verification
    d_sp = np.linalg.norm(pts[:, None, :] - spine[None, :, :], axis=-1)
    nearest_sp = spine[np.argmin(d_sp, axis=1)]
    rad = pts - nearest_sp
    rad_norm = np.linalg.norm(rad, axis=1, keepdims=True)
    rad_unit = rad / np.maximum(rad_norm, 1e-6)
    flip = (nrm * rad_unit).sum(axis=1) < 0.0
    nrm[flip] *= -1.0
    blended = 0.80 * nrm + 0.20 * rad_unit
    blended /= np.maximum(np.linalg.norm(blended, axis=1, keepdims=True), 1e-6)
    return pts.astype(np.float32), blended.astype(np.float32), spine.astype(np.float32)


def _frame_arm_penetration(arm_pts, sh_pos, tb_pts, tb_nrm, spine_pts, tree=None):
    """Return max penetration depth (metres, >=0) of arm_pts into partner torso."""
    if tree is None:
        tree = cKDTree(tb_pts)
    d, nn = tree.query(arm_pts)
    sd = ((arm_pts - tb_pts[nn]) * tb_nrm[nn]).sum(axis=1)
    # Check vertical bounds around partner spine so points above shoulders/below hips aren't miscounted
    sp_c = spine_pts.mean(axis=0)
    sp_axis = spine_pts[-1] - spine_pts[0]
    sp_len = max(float(np.linalg.norm(sp_axis)), 1e-6)
    sp_u = sp_axis / sp_len
    proj_h = (arm_pts - spine_pts[0]) @ sp_u
    in_span = (proj_h >= -0.14) & (proj_h <= sp_len + 0.12)
    ins = in_span & (d < 0.20) & (sd < -CONTACT_TOL_M)
    if not np.any(ins):
        return 0.0, 0
    return float((-sd[ins]).max()), int(ins.sum())


def repair_pair_arm_penetration(res, fps, dev="cuda"):
    """Locally repair arm-into-torso penetrations across all overlapping actor pairs in `res`.

    Modifies `res[pid]["params"]` in-place ONLY inside penetrating intervals, preserving natural surface contact
    and guaranteeing per-frame penetration never increases on either corrective=True or corrective=False skin.
    """
    pids = sorted(res.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    if len(pids) < 2:
        return {}

    mhr = M.model(dev)
    faces = mhr.character_torch.mesh.faces.cpu().numpy()
    torso_m, arms_m, spine_j, uparm_j = _part_masks(dev)
    tb_idx = np.where(torso_m)[0]
    arm_indices = {sd: np.where(arms_m[sd])[0] for sd in ("r", "l")}
    dof_w_t = torch.tensor(DOF_WEIGHTS, dtype=torch.float32, device=dev)

    # Cache corrective=True and corrective=False meshes + skeleton for each person
    cache = {}
    for pid in pids:
        P = res[pid]["params"]
        sh = res[pid]["shape"]
        vf = res[pid]["vf"]
        V_true, S_true = M.forward(sh, P, True, dev)
        V_false, _ = M.forward(sh, P, False, dev)
        cache[pid] = {
            "vf_map": {int(f): i for i, f in enumerate(vf)},
            "V_true": V_true,
            "V_false": V_false,
            "S": S_true,
        }

    report = {}
    pad = max(2, round(0.12 * fps))

    for pid_a in pids:
        for pid_b in pids:
            if pid_a == pid_b:
                continue
            map_a = cache[pid_a]["vf_map"]
            map_b = cache[pid_b]["vf_map"]
            common = sorted(set(map_a) & set(map_b))
            if len(common) < 2:
                continue

            # Quick AABB proximity filter per frame
            close_frames = []
            for f in common:
                ia, ib = map_a[f], map_b[f]
                va = cache[pid_a]["V_true"][ia]
                vb = cache[pid_b]["V_true"][ib, tb_idx]
                if np.all(va.max(0) >= vb.min(0) - 0.05) and np.all(vb.max(0) >= va.min(0) - 0.05):
                    close_frames.append(f)
            if not close_frames:
                continue

            # Precompute partner B outward torso normals only on close frames
            b_torso = {}
            for f in close_frames:
                ib = map_b[f]
                pts_t, nrm_t, sp_t = _outward_torso_normals(
                    cache[pid_b]["V_true"][ib], cache[pid_b]["S"][ib], faces, tb_idx, spine_j
                )
                pts_f, nrm_f, _ = _outward_torso_normals(
                    cache[pid_b]["V_false"][ib], cache[pid_b]["S"][ib], faces, tb_idx, spine_j
                )
                b_torso[f] = (pts_t, nrm_t, cKDTree(pts_t), pts_f, nrm_f, cKDTree(pts_f), sp_t)

            P_a = res[pid_a]["params"].copy()
            sh_a = res[pid_a]["shape"]
            vf_a = res[pid_a]["vf"]
            T_a = len(vf_a)

            for sd in ("r", "l"):
                a_idx = arm_indices[sd]
                cols = ARM_COLS[sd]
                pen_before = np.zeros(T_a, dtype=float)
                for f in close_frames:
                    ia = map_a[f]
                    pts_t, nrm_t, tree_t, pts_f, nrm_f, tree_f, sp_t = b_torso[f]
                    sh_pos = cache[pid_a]["S"][ia, uparm_j[sd], :3] / 100.0
                    p_t, _ = _frame_arm_penetration(cache[pid_a]["V_true"][ia, a_idx], sh_pos, pts_t, nrm_t, sp_t, tree_t)
                    p_f, _ = _frame_arm_penetration(cache[pid_a]["V_false"][ia, a_idx], sh_pos, pts_f, nrm_f, sp_t, tree_f)
                    pen_before[ia] = max(p_t, p_f)

                pen_mask = pen_before > CONTACT_TOL_M
                if not np.any(pen_mask):
                    continue

                n_before = int(pen_mask.sum())
                max_before_cm = round(float(pen_before.max() * 100.0), 2)

                # Merge penetrating intervals separated by <= 3 frames so adjacent frames solve coherently
                merged = pen_mask.copy()
                for g_lo, g_hi in L.runs_of(~merged):
                    if g_lo > 0 and g_hi < T_a and (g_hi - g_lo) <= 3:
                        if vf_a[g_hi] - vf_a[g_lo - 1] == (g_hi - g_lo + 1):
                            merged[g_lo:g_hi] = True

                # Break windows at frame gaps
                contig_id = np.r_[0, np.cumsum(np.diff(vf_a) != 1)]

                for r_lo, r_hi in L.runs_of(merged):
                    # Ensure single contiguous track segment
                    for cid in np.unique(contig_id[r_lo:r_hi]):
                        sub = np.flatnonzero((contig_id == cid) & (np.arange(T_a) >= r_lo) & (np.arange(T_a) < r_hi))
                        if not len(sub):
                            continue
                        a, b = int(sub[0]), int(sub[-1]) + 1
                        seg_rows = np.flatnonzero(contig_id == cid)
                        s0, s1 = int(seg_rows[0]), int(seg_rows[-1]) + 1
                        lo = max(s0, a - pad)
                        hi = min(s1, b + pad)
                        win_len = hi - lo

                        # Prepare partner torso targets across [lo, hi) (use ALL arm/finger vertices & dense torso!)
                        tb_sub = np.linspace(0, len(tb_idx) - 1, min(900, len(tb_idx)), dtype=int)
                        part_pts_w = np.full((win_len, len(tb_sub), 3), 100.0, np.float32)
                        part_nrm_w = np.zeros((win_len, len(tb_sub), 3), np.float32)
                        part_nrm_w[:, :, 1] = 1.0
                        part_sp0_w = np.zeros((win_len, 3), np.float32)
                        part_spu_w = np.zeros((win_len, 3), np.float32)
                        part_spu_w[:, 1] = 1.0
                        part_spl_w = np.ones(win_len, np.float32)
                        has_b = np.zeros(win_len, np.float32)

                        for tw in range(win_len):
                            f_int = int(vf_a[lo + tw])
                            if f_int in map_b:
                                has_b[tw] = 1.0
                                if f_int in b_torso:
                                    pts_t, nrm_t, _, _, _, _, sp_t = b_torso[f_int]
                                else:
                                    ib = map_b[f_int]
                                    pts_t, nrm_t, sp_t = _outward_torso_normals(
                                        cache[pid_b]["V_true"][ib], cache[pid_b]["S"][ib], faces, tb_idx, spine_j
                                    )
                                part_pts_w[tw] = pts_t[tb_sub]
                                part_nrm_w[tw] = nrm_t[tb_sub]
                                part_sp0_w[tw] = sp_t[0]
                                sp_ax = sp_t[-1] - sp_t[0]
                                sp_l = max(float(np.linalg.norm(sp_ax)), 1e-6)
                                part_spl_w[tw] = sp_l
                                part_spu_w[tw] = sp_ax / sp_l

                        # Fix entry-side hemisphere per frame so deep fingers (> half torso) push back toward A's shoulder!
                        sh_w = cache[pid_a]["S"][lo:hi, uparm_j[sd], :3] / 100.0  # (win_len, 3)
                        sp_mid_w = part_sp0_w + 0.5 * part_spl_w[:, None] * part_spu_w
                        side_dir = sh_w - sp_mid_w
                        side_dir /= np.maximum(np.linalg.norm(side_dir, axis=1, keepdims=True), 1e-6)
                        # Mask out opposite-back-wall torso vertices whose outward normal faces > 110 deg away from A's shoulder
                        facing = (part_nrm_w * side_dir[:, None, :]).sum(axis=-1) >= -0.35
                        part_pts_masked = np.where(facing[:, :, None], part_pts_w, 100.0).astype(np.float32)

                        P_win_t = torch.tensor(P_a[lo:hi], dtype=torch.float32, device=dev)
                        theta0 = P_win_t[:, cols].clone()
                        d_theta = torch.zeros_like(theta0, requires_grad=True)
                        sh_t = torch.tensor(sh_a, dtype=torch.float32, device=dev).expand(win_len, -1)
                        expr_t = torch.zeros((win_len, M.NEXPR), dtype=torch.float32, device=dev)

                        arm_idx_t = torch.tensor(a_idx, dtype=torch.long, device=dev)
                        p_pts_t = torch.tensor(part_pts_masked, dtype=torch.float32, device=dev)
                        p_nrm_t = torch.tensor(part_nrm_w, dtype=torch.float32, device=dev)
                        sp0_t = torch.tensor(part_sp0_w, dtype=torch.float32, device=dev)
                        spu_t = torch.tensor(part_spu_w, dtype=torch.float32, device=dev)
                        spl_t = torch.tensor(part_spl_w, dtype=torch.float32, device=dev)
                        has_b_t = torch.tensor(has_b, dtype=torch.float32, device=dev)

                        act = np.zeros(win_len, bool)
                        act[a - lo:b - lo] = True
                        pad_mask_t = torch.tensor((~act).astype(np.float32), device=dev)

                        opt = torch.optim.Adam([d_theta], lr=0.025)
                        for _ in range(130):
                            opt.zero_grad()
                            P_cur = P_win_t.clone()
                            P_cur[:, cols] = theta0 + d_theta
                            # Optimize against both corrective=False and corrective=True skins!
                            loss_sdf = torch.tensor(0.0, device=dev)
                            for corr in (False, True):
                                V_out, _ = mhr(sh_t, P_cur, expr_t, corr)
                                V_arm = (V_out / 100.0)[:, arm_idx_t, :]
                                dists = torch.cdist(V_arm, p_pts_t)
                                min_d, nn_i = dists.min(dim=-1)
                                nn_pts = torch.gather(p_pts_t, 1, nn_i.unsqueeze(-1).expand(-1, -1, 3))
                                nn_nrm = torch.gather(p_nrm_t, 1, nn_i.unsqueeze(-1).expand(-1, -1, 3))
                                sd_val = ((V_arm - nn_pts) * nn_nrm).sum(dim=-1)

                                proj_h = ((V_arm - sp0_t[:, None, :]) * spu_t[:, None, :]).sum(dim=-1)
                                in_span = ((proj_h >= -0.14) & (proj_h <= spl_t[:, None] + 0.12) & (min_d < 0.22)).float()
                                # Push ONLY penetrating vertices out to surface contact (+1 mm), zero pull on non-penetrating!
                                pen = F.relu(0.001 - sd_val) * in_span * has_b_t[:, None]
                                loss_sdf = loss_sdf + 220.0 * (pen.pow(2).sum(dim=1).mean() + 8.0 * pen.amax(dim=1).pow(2).mean())

                            loss_prior = (0.08 * (d_theta.pow(2) * dof_w_t[None, :])).mean() + 8.0 * (pad_mask_t[:, None] * d_theta.pow(2)).mean()
                            loss_sm = torch.tensor(0.0, device=dev)
                            if win_len > 1:
                                vel = (d_theta[1:] - d_theta[:-1]) * fps
                                loss_sm = loss_sm + 0.015 * vel.pow(2).mean()
                            if win_len > 2:
                                acc = (d_theta[2:] - 2.0 * d_theta[1:-1] + d_theta[:-2]) * fps
                                loss_sm = loss_sm + 0.05 * acc.pow(2).mean()

                            loss = loss_sdf + loss_prior + loss_sm
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_([d_theta], 2.0)
                            opt.step()

                        # Cosine boundary fade on padding frames
                        idx_w = np.arange(win_len, dtype=float)
                        l_ramp = np.clip((idx_w + 1) / (a - lo + 1), 0.0, 1.0) if a > lo else np.ones(win_len)
                        r_ramp = np.clip((win_len - idx_w) / (hi - b + 1), 0.0, 1.0) if hi > b else np.ones(win_len)
                        fade = np.minimum(l_ramp, r_ramp)
                        fade = fade * fade * (3.0 - 2.0 * fade)
                        fade[act] = 1.0
                        dt_np = d_theta.detach().cpu().numpy() * fade[:, None]

                        # Strict per-frame monotonic non-degradation check across [lo, hi)!
                        cand_win = P_a[lo:hi].copy()
                        cand_win[:, cols] += dt_np
                        Vt_c, Sc_c = M.forward(sh_a, cand_win, True, dev)
                        Vf_c, _ = M.forward(sh_a, cand_win, False, dev)

                        for tw in range(win_len):
                            ia = lo + tw
                            f_int = int(vf_a[ia])
                            if f_int not in b_torso:
                                P_a[ia] = cand_win[tw]
                                continue
                            pts_t, nrm_t, tree_t, pts_f, nrm_f, tree_f, sp_t = b_torso[f_int]
                            sh_p = Sc_c[tw, uparm_j[sd], :3] / 100.0
                            p_t, _ = _frame_arm_penetration(Vt_c[tw, a_idx], sh_p, pts_t, nrm_t, sp_t, tree_t)
                            p_f, _ = _frame_arm_penetration(Vf_c[tw, a_idx], sh_p, pts_f, nrm_f, sp_t, tree_f)
                            pen_new = max(p_t, p_f)
                            allow = max(pen_before[ia], CONTACT_TOL_M)
                            if pen_new <= allow + 1e-4:
                                P_a[ia] = cand_win[tw]
                            else:
                                # Search partial blend alpha in (0.75, 0.5, 0.25, 0.0) so worst-frame regression is 0%
                                best_p = P_a[ia].copy()
                                best_val = pen_before[ia]
                                for alpha in (0.75, 0.50, 0.25):
                                    tr_p = P_a[ia:ia + 1].copy()
                                    tr_p[0, cols] += alpha * dt_np[tw]
                                    vt1, sc1 = M.forward(sh_a, tr_p, True, dev)
                                    vf1, _ = M.forward(sh_a, tr_p, False, dev)
                                    pt1, _ = _frame_arm_penetration(vt1[0, a_idx], sc1[0, uparm_j[sd], :3] / 100.0, pts_t, nrm_t, sp_t, tree_t)
                                    pf1, _ = _frame_arm_penetration(vf1[0, a_idx], sc1[0, uparm_j[sd], :3] / 100.0, pts_f, nrm_f, sp_t, tree_f)
                                    if max(pt1, pf1) < best_val:
                                        best_val = max(pt1, pf1)
                                        best_p = tr_p[0]
                                P_a[ia] = best_p

                # Refresh person A cache and measure final penetration
                res[pid_a]["params"] = P_a
                cache[pid_a]["V_true"], cache[pid_a]["S"] = M.forward(sh_a, P_a, True, dev)
                cache[pid_a]["V_false"], _ = M.forward(sh_a, P_a, False, dev)

                pen_after = np.zeros(T_a, dtype=float)
                for f in close_frames:
                    ia = map_a[f]
                    pts_t, nrm_t, tree_t, pts_f, nrm_f, tree_f, sp_t = b_torso[f]
                    sh_pos = cache[pid_a]["S"][ia, uparm_j[sd], :3] / 100.0
                    p_t, _ = _frame_arm_penetration(cache[pid_a]["V_true"][ia, a_idx], sh_pos, pts_t, nrm_t, sp_t, tree_t)
                    p_f, _ = _frame_arm_penetration(cache[pid_a]["V_false"][ia, a_idx], sh_pos, pts_f, nrm_f, sp_t, tree_f)
                    pen_after[ia] = max(p_t, p_f)

                n_after = int((pen_after > CONTACT_TOL_M).sum())
                max_after_cm = round(float(pen_after.max() * 100.0), 2)
                key = f"p{pid_a}_{sd}_arm_to_p{pid_b}"
                report[key] = {
                    "穿入帧_前": n_before,
                    "最深_cm_前": max_before_cm,
                    "穿入帧_后": n_after,
                    "最深_cm_后": max_after_cm,
                }
                print(f"  [双人局部修手] 人 {pid_a} ({sd}臂 -> 人 {pid_b}): 穿入帧 {n_before} -> {n_after}, "
                      f"最深 {max_before_cm:.2f} -> {max_after_cm:.2f} cm", flush=True)

    return report
