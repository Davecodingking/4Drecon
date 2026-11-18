"""Temporal lower-body solve for a continuous actor on a registered landing."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial.transform import Rotation, Slerp
import torch
import mhr_body as M
import legs_proc as L
from planned_swing import contradictory_swing, recovered_contacts, affected_legs, stance_transitions
from fit_exposure import fit_exposure_states, fit_unloaded_exposures
from observed_pose import apply_upper_pose_holds, hold_rotations, source_hold_edges
from sole_clearance import resolve_inferred_sole_clearance, shared_contact_vertices
from observed_support import common_flight, mask_shin_visibility, support_observations
from leg_observations import (inferred_sole_weights, projected_thigh_error, registered_root_prior,
                              root_displacement, sole_inclination, stance_groups)


def smooth_turn_rotations(rotation, turning, boundaries, fps):
    """Filter local rotations on SO(3) inside continuous inferred turns."""
    result = Rotation.from_quat(rotation.as_quat())
    window = max(1, round(.13*fps))
    sigma = max(1., .055*fps)
    edges = np.r_[0,boundaries,len(turning)]
    for start,end in zip(edges[:-1],edges[1:]):
        for i in np.flatnonzero(turning[start:end])+start:
            lo = max(start,i-window)
            hi = min(end,i+window+1)
            distance = np.arange(lo,hi)-i
            weight = np.exp(-.5*(distance/sigma)**2)
            relative = (rotation[i].inv()*rotation[lo:hi]).as_rotvec()
            result[i] = rotation[i]*Rotation.from_rotvec(np.average(relative,axis=0,weights=weight))
    return result


def smooth_pose_impulses(params, fps, boundaries, dev, hold_edges=None):
    """Blend across camera cuts and isolated jumps in observed body rotations."""
    out = params.copy()
    radius = max(1., fps * .07)
    groups = M.limb_groups(dev)
    for name, columns in groups.items():
        if name.endswith("upleg"):
            continue
        angles = params[:, columns]
        velocity = np.diff(angles, axis=0, prepend=angles[:1])
        accel = np.max(np.abs(np.diff(velocity, axis=0, prepend=velocity[:1])), axis=1)
        local = median_filter(accel, size=max(3, round(fps*.4) | 1), mode="nearest")
        sudden = (accel > np.maximum(2.5*local, np.deg2rad(12)))
        # Pose parameters are coupled across joints. A sustained fast rotation
        # benefits from a shorter filter than an isolated reversal at an edit.
        speed = np.max(np.abs(velocity), axis=1)
        strength = np.maximum(sudden.astype(float), np.clip((speed*fps-6.)/5., 0., .8))
        strength[boundaries] = 1.
        weight = gaussian_filter1d(strength, radius, mode="nearest")
        peak = 1 / (np.sqrt(2*np.pi)*radius)
        weight = np.clip(weight / peak, 0, 1)[:,None]
        smooth = gaussian_filter1d(angles, radius, axis=0, mode="nearest")
        out[:,columns] += weight*(smooth-angles)
    apply_upper_pose_holds(out, groups, hold_edges, boundaries)
    return out


def vertex_binding(shape, ids, dev):
    """Precompute weighted bind-space coordinates for a subset of the exported mesh."""
    from scipy.spatial.transform import Rotation
    skeleton = M.skeleton(dev)
    rest = M.rest_verts(shape, dev)
    vertex, joint, weight = skeleton['sk_v'],skeleton['sk_j'],skeleton['sk_w']
    joints = np.unique(joint[np.isin(vertex,ids)])
    bind = skeleton['inv_bind'][joints]
    local = Rotation.from_quat(bind[:,3:7]).as_matrix()
    positions = np.einsum('jik,vk->vji',local,rest[ids])*bind[None,:,7:8]+bind[None,:,:3]
    weights = np.zeros((len(ids),len(joints)))
    for i, vid in enumerate(ids):
        use = vertex == vid
        np.add.at(weights[i],np.searchsorted(joints,joint[use]),weight[use])
    tensor = lambda a: torch.as_tensor(a,dtype=torch.float32,device=dev)
    return (torch.as_tensor(joints,device=dev),tensor(weights),tensor(positions))


def skin_vertices(skeleton, binding):
    joints, weights, local = binding
    state = skeleton[:,joints]
    rotation = L._qmat(state[:,:,3:7])
    placed = torch.einsum('tjab,vjb->tvja',rotation,local)
    placed = placed*state[:,None,:,7:8]+state[:,None,:,:3]
    return (placed*weights[None,:,:,None]).sum(2)/100.


def skinned_soles(skeleton, binding):
    return skin_vertices(skeleton,binding).reshape(-1,2,len(L.PTS),3)


def sole_orientation(points):
    """Orthonormal heel-to-forefoot and across-forefoot axes for MHR sole points."""
    forward = torch.nn.functional.normalize(points[:,:,2:4].mean(2)-points[:,:,0],dim=-1)
    across = points[:,:,3]-points[:,:,2]
    across = across - (across*forward).sum(-1,keepdim=True)*forward
    across = torch.nn.functional.normalize(across,dim=-1)
    return forward, across


def observed_swing_weights(uv, depth, height, image_wh, shot, support, fps, masks, person_id):
    """Return weights for measured knee/ankle positions and damping per leg."""
    from PIL import Image
    from scipy.ndimage import maximum_filter1d

    n = len(uv)
    motion = np.zeros((n, 2, 2))
    if masks is None or person_id is None:
        return motion, motion.max(2)

    width, rows = image_wh
    valid = ((uv >= 0) & (uv < [width, rows])).all(-1) & (depth > .05)
    # The tracker may estimate a cropped or occluded joint. Require the
    # projected point to fall on (or very close to) this person's silhouette.
    for i, path in enumerate(masks):
        if not valid[i].any():
            continue
        try:
            with Image.open(path) as image:
                labels = np.asarray(image)
        except FileNotFoundError:
            valid[i] = False
            continue
        if labels.ndim != 2 or labels.shape != (rows, width):
            valid[i] = False
            continue
        radius = max(2, round(.012*height[i]))
        for side, joint in zip(*np.nonzero(valid[i])):
            x, y = uv[i, side, joint].astype(int)
            patch = labels[max(0,y-radius):y+radius+1, max(0,x-radius):x+radius+1]
            valid[i,side,joint] = np.any(patch == person_id)

    relative = (uv[:,:,1:] - uv[:,:,:1]) / height[:,None,None,None]
    reliable = valid[:,:,1:] & valid[:,:,:1] & ~support[:,:,None]
    edges = np.r_[0,np.flatnonzero(np.diff(shot))+1,n]
    window = 2*max(1,round(.075*fps))+1
    for start, stop in zip(edges[:-1],edges[1:]):
        for side in range(2):
            for joint in range(2):
                for lo, hi in L.runs_of(reliable[start:stop,side,joint]):
                    a, b = start+lo, start+hi
                    if b-a < 3:
                        reliable[a:b,side,joint] = False
                        continue
                    coords = relative[a:b,side,joint]
                    velocity = np.diff(coords,axis=0)*fps
                    speed = np.linalg.norm(velocity,axis=-1)
                    dot = (velocity[1:]*velocity[:-1]).sum(-1)
                    alignment = dot / np.maximum(speed[1:]*speed[:-1],1.e-8)
                    # Confidence needs sustained travel in the same direction.
                    strength = np.zeros(b-a)
                    coherent = np.clip((np.minimum(speed[1:],speed[:-1])-.15)/.50,0,1)
                    strength[1:-1] = coherent*np.clip((alignment-.3)/.5,0,1)
                    motion[a:b,side,joint] = maximum_filter1d(strength,window,mode='nearest')
                    deviation = np.linalg.norm(coords[1:-1]-(coords[:-2]+coords[2:])*.5,axis=-1)
                    neighbors = np.linalg.norm(coords[2:]-coords[:-2],axis=-1)
                    spike = (deviation > .035) & (deviation > 1.6*neighbors)
                    reliable[a+1:b-1,side,joint] &= ~spike
    moving_leg = (motion*reliable).max(2)
    # A visible stationary knee can constrain its leg while the opposite
    # ankle swings out. It keeps normal damping and any planted-foot lock.
    confidence = np.maximum(moving_leg, .40*moving_leg[:,::-1])
    return confidence[:,:,None]*reliable, moving_leg


def foot_targets(plan, inferred, observed, up):
    """Keep the planner's swing arcs and planted heel/forefoot in world space."""
    points = observed.copy()
    contact = inferred.copy()
    planned = np.zeros(len(points), dtype=bool)
    if plan is not None:
        planned = np.asarray(plan["done"], dtype=bool)
        points[planned] = plan["sole_world"][planned]
        contact[planned] = plan["contact"][planned]

    targets = points.copy()
    for side in range(2):
        sole = points[:, side]
        offsets = np.zeros((len(sole), 3))
        for start, end in L.runs_of(contact[:, side]):
            for i in range(start + 1, end):
                # Track the same low landmark across both frames. The heel
                # may lift as the forefoot becomes the pivot of a planted foot.
                pivot = np.argmin(np.maximum(sole[i-1] @ up, sole[i] @ up))
                delta = sole[i-1, pivot] - sole[i, pivot]
                offsets[i] = offsets[i-1] + delta - (delta @ up) * up
        for start, end in L.runs_of(~contact[:, side]):
            if not planned[start:end].any():
                continue
            previous = offsets[max(0, start-1)]
            following = offsets[min(len(sole)-1, end)]
            phase = np.arange(1, end-start+1) / (end-start+1)
            offsets[start:end] = previous + (following-previous) * phase[:, None]
        targets[:, side] += offsets[:, None, :]
    return targets, contact, planned


def turn_support_loss(hips, soles, weight_share, up):
    """Measure pelvis placement against the feet that carry its weight (metres²)."""
    pelvis = hips.mean(1)
    left = hips[:,0]-hips[:,1]
    left = torch.nn.functional.normalize(left-(left@up)[:,None]*up,dim=1)
    forward = torch.linalg.cross(left,up.expand_as(left))
    centers = .5*(soles[:,:,0]+soles[:,:,2:4].mean(2))
    offset = centers-pelvis[:,None]
    sign = hips.new_tensor([1.,-1.])
    lateral = (offset*left[:,None]).sum(-1)*sign
    hip_width = ((hips[:,0]-hips[:,1])*left).sum(-1)
    # Place the swinging foot outside its hip and allow the loaded foot to
    # approach the midline as the pelvis moves over it.
    width = hip_width[:,None]*(.85-.65*weight_share)
    spread_error = torch.relu(width-lateral).square().sum(1)
    support_center = (centers*weight_share[:,:,None]).sum(1)
    delta = pelvis-support_center
    side_error = torch.relu((delta*left).sum(-1).abs()-.035)
    fore_error = torch.relu((delta*forward).sum(-1).abs()-.055)
    return side_error.square()+fore_error.square(), spread_error


def clear_export_tread_edges(pose, shape, up, terrain, sole, dev):
    """Keep the exported skin on the same stair tread as the corrective mesh."""
    exported, _ = M.forward(shape, pose, False, dev)
    feet = exported[:,sole]
    height = feet @ up
    required = (terrain.model_h(feet.reshape(-1,3)).reshape(height.shape)-height).max(1)
    suspect = np.flatnonzero(required > .02)
    if not len(suspect):
        return pose
    corrected, _ = M.forward(shape, pose[suspect], True, dev)
    stair = terrain.ST[terrain.main]
    along = stair[1]*terrain.e1+stair[2]*terrain.e2
    across = -stair[2]*terrain.e1+stair[1]*terrain.e2
    directions = np.array([along,-along,across,-across])
    root = M.root_pos(pose)
    for frame, original in zip(suspect, corrected[:,sole]):
        if np.min(original @ up-terrain.model_h(original)) < -.02:
            continue
        both = np.concatenate([original,feet[frame]])
        # A millimetric move off a vertical tread face can clear a large
        # apparent penetration without introducing a vertical root impulse.
        for radius in (.001,.002,.003,.004,.006,.008):
            candidates = both[None] + radius*directions[:,None]
            ground = terrain.model_h(candidates.reshape(-1,3)).reshape(candidates.shape[:-1])
            free = np.flatnonzero((candidates@up-ground).min(1) >= -.002)
            if len(free):
                root[frame] += radius*directions[free[0]]
                break
    return M.set_root(pose,None,root)


def inferred_turn_frames(active, observed_crossing):
    """A turn is jointly fitted. A real crossed stance protects the whole fitted event."""
    return np.zeros_like(active) if np.any(active & observed_crossing) else active


def open_turn_evidence(hip_width, knee_span, ankle_span, knee_seen, ankle_seen, thigh_confidence):
    """Use an open measured knee stance while protecting visibly crossed ankles."""
    seen = knee_seen.all(1) | (thigh_confidence > .60).all(1)
    return seen & (knee_span > .35*hip_width) & (~ankle_seen.all(1) | (ankle_span > .10*hip_width))


def contact_handoff_weights(contact, inferred, fps):
    """Local swing constraints next to an inferred, load-bearing footprint."""
    n = len(contact)
    weights = np.zeros((max(0, n-1), 2))
    radius = max(2, round(.20 * fps))
    for side in range(2):
        swing = inferred & ~contact[:, side]
        for lo, hi in L.runs_of(swing):
            if lo and inferred[lo-1] and contact[lo-1, side]:
                k = np.arange(lo-1, min(n-1, hi-1, lo-1+radius))
                weights[k, side] = np.maximum(weights[k, side], .5 + .5*np.cos(np.pi*(k-lo+1)/radius))
            if hi < n and inferred[hi] and contact[hi, side]:
                k = np.arange(max(0, lo, hi-radius), hi)
                weights[k, side] = np.maximum(weights[k, side], .5 + .5*np.cos(np.pi*(hi-1-k)/radius))
    return weights


def refine_turn_contact(params, shape, up, terrain, plan, fps, dev):
    """Resolve loaded and swinging soles independently, using the skinned mesh.

    Translation of the whole actor cannot fix an airborne stance foot if the
    other leg has already hit the floor. Fit leg angles and pelvis height
    together. Keep horizontal root position and the forefoot twist fixed.
    """
    if plan is None or 'turning' not in plan:
        return params
    active = np.asarray(plan['turning'],bool) & np.asarray(plan['done'],bool)
    if not active.any():
        return params
    rest = M.rest_verts(shape,dev)
    sides = [np.flatnonzero((rest[:,1]<rest[:,1].min()+10)&(sign*rest[:,0]>0)) for sign in (1,-1)]
    out = params.copy()
    tensor = lambda a: torch.as_tensor(a,dtype=torch.float32,device=dev)
    u = tensor(up)
    columns = [col for side in L.SIDES for col in L.LEG_COLS[side] if col not in L.BALL_TWIST]
    geo = L.geometry(shape,params[0],dev)
    soles = np.array([[geo['side'][sd]['vids'][pt] for pt in L.PTS] for sd in L.SIDES])
    for begin,end in L.runs_of(active):
        padding = max(2,round(.20*fps))
        lo,hi = max(0,begin-padding),min(len(out),end+padding)
        sl = slice(lo,hi)
        original = out[sl].copy()
        world,_ = M.forward(shape,original,False,dev)
        contact = np.asarray(plan['contact'][sl],bool)
        trace = np.asarray(plan['sole_world'][sl])
        status = np.arange(lo,hi)
        active_here = (status>=begin)&(status<end)
        crossed = np.asarray(plan.get('observed_crossing', np.zeros(len(params), bool)))[sl]
        inferred_here = inferred_turn_frames(active_here, crossed)
        ref_gap = np.stack([(world[:,ix]@up-terrain.model_h(world[:,ix].reshape(-1,3)).reshape(len(world),-1)).min(1) for ix in sides],1)
        reference = world[:,soles]
        target_gap = ref_gap.copy()
        swing_target = np.zeros(contact.shape, bool)
        planned_phase = plan.get('swing_phase') if inferred_here.any() else None
        airborne = np.asarray(plan.get('observed_flight', np.zeros(len(params), bool)),bool)[sl]
        target_gap[active_here] = .001
        for j in range(2):
            planned_gap = (trace[:,j]@up-terrain.model_h(trace[:,j].reshape(-1,3)).reshape(len(world),-1)).min(1)
            for a,b in L.runs_of(active_here & ~contact[:,j]):
                # Use the observed/planned lift where it is reliable. Give an
                # inferred moving foot clearance even when its next touchdown
                # lies outside the inferred interval.
                phase = np.arange(1,b-a+1)/(b-a+1)
                travel = np.linalg.norm(np.diff(trace[a:b,j].mean(1),axis=0),axis=1).sum()
                arc = min(.065,max(.025,.22*travel))*np.sin(np.pi*phase)**2
                target_gap[a:b,j] = np.maximum(arc,planned_gap[a:b])
                if planned_phase is not None:
                    # Losing the observed leg does not mean the step ended. Use its
                    # actual landing time while the opposite foot carries the load.
                    angle = planned_phase[sl][a:b,j]
                    valid = np.isfinite(angle) & contact[a:b,1-j] & ~airborne[a:b]
                    rows = np.flatnonzero(valid)+a
                    amplitude = min(.058,max(.025,.22*travel))
                    target_gap[rows,j] = np.maximum(target_gap[rows,j], amplitude*np.sin(np.pi*angle[valid])**2)
                    swing_target[rows,j] = True
        weight = tensor(active_here.astype(float))
        moving_target = tensor(swing_target)
        release_pair = tensor((swing_target[1:] | swing_target[:-1]) & (active_here[1:] & active_here[:-1])[:,None])
        fit_release = bool(swing_target.any())
        target = tensor(target_gap)
        base = tensor(original)
        reference_t = tensor(reference)
        hip_joints = [geo["side"][side]["j"]["hip"] for side in L.SIDES]
        with torch.no_grad():
            hip = L.skel_torch(base,dev)[:,hip_joints,:3]/100
            hip_axis = hip[:,0]-hip[:,1]
            hip_axis -= (hip_axis@u)[:,None]*u
            width = hip_axis.norm(dim=1)
            hip_axis = hip_axis/width[:,None].clamp(min=.01)
        # A material pivot can rotate without translating the sole centre.
        # Keep the observed ordering when the next foot takes the load.
        centers = tensor(np.stack([world[:,ids].mean(1) for ids in sides],axis=1))
        reference_width = ((centers[:,0]-centers[:,1])*hip_axis).sum(1)
        min_width = torch.minimum(reference_width,.60*width)
        double = tensor(contact.all(1) & active_here)
        # Keep the incoming placement of a planted foot across the local
        # solve boundary. Velocity constraints alone leave its location free.
        adjacent = active_here[1:] != active_here[:-1]
        measured_step = reference[1:].mean(2)-reference[:-1].mean(2)
        planar_step = measured_step-(measured_step@up)[:,:,None]*up
        stable = contact[:-1] & contact[1:] & (np.linalg.norm(planar_step,axis=2)*fps<.35)
        anchored = np.zeros_like(contact)
        anchored[1:] |= adjacent[:,None] & stable
        anchored[:-1] |= adjacent[:,None] & stable
        boundary_weight = tensor(anchored)
        skin_ids = np.stack([ids[np.linspace(0,len(ids)-1,48).astype(int)] for ids in sides])
        center_bind = vertex_binding(shape,skin_ids.ravel(),dev)
        sampled_original = tensor(world[:,skin_ids].mean(2))

        transfer_weights = contact_handoff_weights(contact, inferred_here, fps)
        handoff_weight = tensor(transfer_weights)
        # Maintain a continuous world-space stride on both sides of the handoff,
        # including the measured samples at the edge of a trimmed inferred turn.
        from scipy.ndimage import maximum_filter1d
        radius = max(1, round(.20*fps))
        curve_weight = maximum_filter1d(transfer_weights, size=2*radius+1, axis=0, mode='constant')
        curve_weight = tensor(np.maximum(curve_weight[:-1], curve_weight[1:]))
        floor_window = weight * float(transfer_weights.any())
        support_here = tensor(contact & active_here[:,None]) * float(transfer_weights.any())
        path_step = np.diff(trace, axis=0)
        expected_step = tensor(path_step - (path_step@up)[...,None]*up)

        step = torch.zeros((len(world),len(columns)),device=dev,requires_grad=True)
        dz = torch.zeros(len(world),device=dev,requires_grad=True)
        opts = torch.optim.Adam([step,dz],lr=.012)
        # Include the lowest skin vertices seen at each frame in addition to
        # the sole landmarks. Re-evaluate the extremal vertices during IK.
        loaded_pairs = tensor(contact[1:] & contact[:-1] & (active_here[1:] & active_here[:-1])[:,None])
        pair_rows = torch.arange(len(world)-1,device=dev)[:,None]
        selected = [soles[j].copy() for j in range(2)]
        open_stance = np.asarray(plan.get('open_stance', np.zeros(len(params), bool)))[sl] & active_here
        separate = bool(open_stance.any() and transfer_weights.any())
        open_weight = tensor(open_stance)
        contact_cost = 18000 if separate else 3600
        pivot_weight = 30000 if separate else 9000
        forward_axis = torch.linalg.cross(hip_axis, u.expand_as(hip_axis))
        axes = torch.stack([hip_axis, forward_axis, u.expand_as(hip_axis)], -1)
        for stage in range(3):
            solved = original.copy()
            solved[:,columns] += step.detach().cpu().numpy()
            posed,_ = M.forward(shape,solved,False,dev)
            bindings = []
            pivot_indices = []
            for j,ix in enumerate(sides):
                feet = posed[:,ix]
                gaps = feet@up - terrain.model_h(feet.reshape(-1,3)).reshape(len(world),-1)
                if separate:
                    # Update the support function of the actual shoes on all three anatomical axes.
                    local = np.einsum('tnj,tjk->tnk', feet, axes.cpu().numpy())
                    extrema = ix[np.r_[local.argmin(1).ravel(), local.argmax(1).ravel()]]
                    selected[j] = np.unique(np.r_[selected[j], extrema])
                near = shared_contact_vertices(gaps,ix)
                selected[j] = np.unique(np.r_[selected[j],ix[np.argmin(gaps,axis=1)],near.ravel()])
                pivot_indices.append(torch.as_tensor(np.searchsorted(selected[j],near),device=dev))
                bindings.append(vertex_binding(shape,selected[j],dev))
            bind5 = vertex_binding(shape,soles.ravel(),dev)
            handoff = tensor((~active_here).astype(float))
            for it in range(100):
                opts.zero_grad()
                p = base.clone()
                p[:,columns] += step
                sk = L.skel_torch(p,dev)
                gap_list = []
                scrape_cost = 0.
                shoe_bounds = []
                pivot_cost = 0.
                for side_idx, bind in enumerate(bindings):
                    pts = skin_vertices(sk,bind)+dz[:,None,None]*u
                    if separate:
                        local = pts @ axes
                        shoe_bounds.append((local.amin(1), local.amax(1)))
                    surface = terrain.model_h(pts.detach().cpu().numpy().reshape(-1,3)).reshape(pts.shape[:2])
                    vertex_height = pts@u - tensor(surface)
                    gap_list.append(vertex_height.amin(1))
                    if fit_release:
                        material_step = pts.diff(dim=0)
                        planar = material_step - (material_step@u)[...,None]*u
                        contact_height = torch.maximum(vertex_height[1:],vertex_height[:-1])
                        touching = torch.sigmoid((.012-contact_height)/.003)
                        excess = torch.relu(planar.norm(dim=-1)-.18/fps)
                        sliding = (excess.square()*touching).topk(min(5,pts.shape[1]),dim=1).values.mean(1)
                        edges = release_pair[:,side_idx]
                        scrape_cost += (sliding*edges).sum()/edges.sum().clamp(min=1)
                    stride = (pts[1:]-pts[:-1])[pair_rows,pivot_indices[side_idx]]
                    planar_stride = stride-(stride@u)[:,:,None]*u
                    loaded_pair = loaded_pairs[:,side_idx]
                    pivot_cost += (planar_stride.square().sum(-1).mean(1)*loaded_pair).sum()/loaded_pair.sum().clamp(min=1)
                gap = torch.stack(gap_list,1)
                landmarks = skin_vertices(sk,bind5).reshape(len(world),2,len(L.PTS),3)+dz[:,None,None,None]*u
                difference = landmarks-reference_t
                horizontal = difference-(difference@u)[...,None]*u
                loss = 900*(gap-target).square().mean()+1200*torch.relu(-gap).square().mean()
                loss += 40000*(torch.relu(-gap-.001).square()*floor_window[:,None]).mean()
                loss += contact_cost*((gap-target).square()*support_here).mean()
                loss += pivot_weight*pivot_cost
                sampled = skin_vertices(sk,center_bind).reshape(len(world),2,-1,3).mean(2)
                centers_now = centers+(sampled-sampled_original)
                step_width = ((centers_now[:,0]-centers_now[:,1])*hip_axis).sum(1)
                loss += 4000*(((min_width-step_width).clamp(min=0).square())*double).sum()/double.sum().clamp(min=1)
                loss += 80000*(horizontal.square().sum(-1).mean(2)*boundary_weight).sum()/boundary_weight.sum().clamp(min=1)
                stride = landmarks.diff(dim=0)
                planar_stride = stride - (stride@u)[...,None]*u
                tangent_error = (planar_stride-expected_step).square().sum(-1).mean(2)
                loss += 1800*(tangent_error*handoff_weight).sum()/handoff_weight.sum().clamp(min=1)
                curve = planar_stride.diff(dim=0).square().sum(-1).mean(2)
                loss += 16000*(curve*curve_weight).sum()/curve_weight.sum().clamp(min=1)
                if separate:
                    # Any separating axis suffices. A shoe can pass in front of or above its partner.
                    left_low, left_high = shoe_bounds[0]
                    right_low, right_high = shoe_bounds[1]
                    lateral_gap = left_low[:,0]-right_high[:,0]
                    other_gaps = torch.maximum(left_low[:,1:]-right_high[:,1:], right_low[:,1:]-left_high[:,1:])
                    clearance = torch.maximum(lateral_gap, other_gaps.amax(1))
                    collision = torch.relu(.020-clearance).square()
                    side_error = torch.relu(.40*width-step_width).square()
                    loss += ((85000*collision + 40000*side_error)*open_weight).sum()/open_weight.sum().clamp(min=1)
                if fit_release:
                    lift_error = torch.relu(target-gap-.002).square()
                    loss += 8000*scrape_cost + 20000*(lift_error*moving_target).sum()/moving_target.sum().clamp(min=1)
                loss += 280*horizontal.square().sum(-1).mean()
                loss += 130*((difference@u)-(difference@u).mean(2,keepdim=True)).square().mean()
                # Keep the transition pose, angular velocity and planted sole in place.
                drift = horizontal.diff(dim=0)
                pair = tensor(contact[1:]&contact[:-1]) * weight[1:,None]*weight[:-1,None]
                loss += 380*(drift.square().sum(-1).mean(2)*pair).mean()
                loss += .16*step.square().mean()+.4*dz.square().mean()
                loss += 9*step.diff(n=2,dim=0).square().mean()+10*dz.diff(n=2,dim=0).square().mean()
                loss += 20*(handoff*(step.square().mean(1)+dz.square())).mean()
                loss.backward()
                opts.step()
            opts.param_groups[0]['lr'] *= .5
        # Taper only the surrounding observed frames to guarantee a continuous
        # return to their original motion at the boundary of the local solve.
        fade = np.minimum(np.clip((status-lo)/padding,0,1),np.clip((hi-1-status)/padding,0,1))
        fade[active_here] = 1.
        solved = original.copy()
        solved[:,columns] += step.detach().cpu().numpy()*fade[:,None]
        out[sl] = M.set_root(solved,None,M.root_pos(solved)+(dz.detach().cpu().numpy()*fade)[:,None]*up)
    return out


def match_sole_height(params, shape, up, terrain, plan, input_pose, boundaries, fps, dev, fixed=None, confirmed=None):
    """Use skinned soles for final contact, including at the edge of a tread."""
    rest = M.rest_verts(shape, dev)
    sole = np.flatnonzero(rest[:,1] < rest[:,1].min()+10)

    def clearance(pose):
        vertices, _ = M.forward(shape, pose, True, dev)
        feet = vertices[:,sole]
        surface = terrain.model_h(feet.reshape(-1,3)).reshape(feet.shape[:2])
        return (feet @ up - surface).min(1)

    baseline = clearance(input_pose)
    gap = clearance(params)
    near = np.abs(baseline) < .04
    if plan is not None:
        planned = np.asarray(plan["done"], bool)
        near[planned] = np.any(plan["contact"][planned], axis=1)
    # Preserve observed airborne intervals. Spread the vertical correction
    # over nearby frames so crossing one stair edge cannot pop the root.
    offset = np.where(near | (gap < 0), .001-gap, 0)
    radius = max(1., fps*.055)
    offset = gaussian_filter1d(offset, radius, mode="nearest")
    required = np.maximum(0., -gap-.002)
    for distance in range(1, int(np.ceil(3*radius))+1):
        weight = np.exp(-.5*(distance/radius)**2)
        offset[distance:] = np.maximum(offset[distance:], np.where(required[:-distance]>0, required[:-distance]*weight, -np.inf))
        offset[:-distance] = np.maximum(offset[:-distance], np.where(required[distance:]>0, required[distance:]*weight, -np.inf))
    offset = np.maximum(offset, -gap-.002)
    for first in boundaries:
        offset[first-1:first+1] = max(offset[first-1:first+1])
    if fixed is not None:
        # A generated footprint supplies no vertical translation measurement.
        offset[fixed] = np.where(confirmed[fixed],np.clip(offset[fixed],-.025,.025),0.)
    grounded = M.set_root(params,None,M.root_pos(params)+offset[:,None]*up)
    grounded = refine_turn_contact(grounded,shape,up,terrain,plan,fps,dev)
    return clear_export_tread_edges(grounded,shape,up,terrain,sole,dev)


def refine(params, shape, vf, cams, keys, cuts, fps, up, terrain, registration,
           raw_track, body_focals, image_wh, dev, plan=None, masks=None, person_id=None, thighs=None, frame_paths=None, verified_frames=(), calibrated_root=None, prefill_pose=None):
    args = (params,shape,vf,cams,keys,cuts,fps,up,terrain,registration,raw_track,body_focals,image_wh,dev)
    observations = dict(masks=masks,person_id=person_id,thighs=thighs,frame_paths=frame_paths,
                        verified_frames=verified_frames,calibrated_root=calibrated_root)
    recovery = {}
    registered = _solve(*args,plan=plan,prefill_pose=prefill_pose,recovery=recovery,**observations)
    if not recovery:
        return registered
    # The existing camera/terrain registration owns the actor trajectory. Refit the
    # affected legs against that trajectory so a revised footprint cannot move it.
    return _solve(*args,plan=recovery['plan'],registered_pose=registered,recovery=recovery,**observations)


def _solve(params, shape, vf, cams, keys, cuts, fps, up, terrain, registration,
           raw_track, body_focals, image_wh, dev, plan=None, masks=None, person_id=None, thighs=None, frame_paths=None, verified_frames=(), calibrated_root=None, prefill_pose=None, recovery=None, registered_pose=None):
    shot = np.searchsorted(cuts, vf/fps+.001)
    boundaries = np.flatnonzero(np.diff(shot))+1
    if not len(boundaries) or np.any(np.diff(vf)!=1):
        return params
    layer = registration['shot']-1
    idx = np.flatnonzero(shot==layer)
    if not len(idx): return params
    side = registration['support_side']
    mid = idx[0]
    n = len(params)
    cols = [c for side in L.SIDES for c in L.LEG_COLS[side] if c not in L.BALL_TWIST]
    knees = [cols.index(k) for k in (62,53)]
    tensor = lambda a: torch.as_tensor(np.asarray(a),dtype=torch.float32,device=dev)
    hold_edges = source_hold_edges(frame_paths, masks, person_id, shot)
    base = smooth_pose_impulses(params, fps, boundaries, dev, hold_edges)
    # Remove orientation impulses at edits on SO(3), rather than averaging Euler angles.
    turning = np.asarray((plan or {}).get("turning", np.zeros(n,bool)),bool)
    rot = Rotation.from_matrix(M.root_rot(base))
    rot = smooth_turn_rotations(rot,turning,boundaries,fps)
    for cut in boundaries:
        lo=max(0,cut-round(.18*fps)); hi=min(n-1,cut+round(.18*fps))
        if lo<hi:
            rot[lo:hi+1]=Slerp([lo,hi],rot[[lo,hi]])(np.arange(lo,hi+1))
    rot = hold_rotations(rot, hold_edges, boundaries)
    base=M.set_root(base,rot.as_matrix())
    root=M.root_pos(base)
    # Input observed root comes from the body model's own image rays.
    proj=[]; matrices=[]; observed=[]; sizes=[]
    for i,f in enumerate(vf):
        cam=cams[keys[int(f)]]
        E=np.asarray(cam['extrinsic_w2c_4x4']);K=np.asarray(cam['intrinsic_3x3']).copy()
        wh=np.array(image_wh);K[:2]*=(wh/np.array(cam['input_hw'][::-1]))[:,None]
        camera_xyz=(M.F@M.ROOT_C)/100 + raw_track[i][1]
        uv=camera_xyz[:2]/camera_xyz[2]*body_focals[i]+wh/2
        proj.append(K);matrices.append(E);observed.append(uv)
        sizes.append(max(body_focals[i]/camera_xyz[2]*1.7,20))
    K,E,uv,height=map(tensor,(proj,matrices,observed,sizes))
    root0=tensor(root);base_t=tensor(base);u=tensor(up)
    geo=L.geometry(shape,base[0],dev)
    sole_ids = np.array([geo['side'][sd]['vids'][name] for sd in L.SIDES for name in L.PTS])
    binding = vertex_binding(shape,sole_ids,dev)
    with torch.no_grad():
        pts0=skinned_soles(L.skel_torch(base_t,dev),binding).cpu().numpy()
    feet=pts0.mean(2)
    # Robust low-pass rejects one-frame and short-fragment IK flips on both legs.
    smooth=gaussian_filter1d(median_filter(base[:,cols],size=(5,1),mode='nearest'),fps*.08,axis=0,mode='nearest')
    smooth[:,knees]=np.clip(smooth[:,knees],.04,L.KNEE_MAX)
    sy=tensor(smooth)
    rs=gaussian_filter1d(median_filter(root,size=(5,1),mode='nearest'),fps*.10,axis=0,mode='nearest')
    init=tensor(rs-root0.cpu().numpy())
    dx=init.clone().requires_grad_(True);leg=sy.clone().requires_grad_(True)
    landing=terrain.extra[-1];st=terrain.ST[terrain.main]
    along=tensor(st[1]*terrain.e1+st[2]*terrain.e2);across=tensor(-st[2]*terrain.e1+st[1]*terrain.e2)
    b=np.array(landing);active=np.zeros(n,bool);active[idx]=True
    # Establish a stance segment on the landing; keep the visible kicking leg free.
    # Where both feet are inferred, use the anatomical side from cut registration.
    support=np.zeros((n,2),bool)
    speed=np.linalg.norm(np.diff(feet,axis=0,prepend=feet[:1]),axis=-1)*fps
    for i in range(2):
        s=feet[:,i]@up
        support[:,i]=(s <= np.min(feet@up,axis=1)+.08)&(gaussian_filter1d(speed[:,i],2)<.65)
    pre=max(0,mid-round(fps*.6))
    support[pre:mid+1,side]=True
    # In the reverse shot, infer the stance side from the sustained relative
    # foot elevation. Bridge detector gaps; the low foot may pivot, not translate.
    rel=gaussian_filter1d(feet@up, max(1,fps*.12),axis=0)
    stance=side
    for i in idx:
        if rel[i,stance]>rel[i,1-stance]+.12: stance=1-stance
        support[i]=False;support[i,stance]=True
    # Only the registered view has evidence for inferred static contacts.
    # Where the planner succeeded its per-foot contact flags take precedence.
    support[:pre] = False
    support[idx[-1]+1:] = False
    leg_joints = torch.as_tensor([[geo['side'][sd]['j'][name] for name in ('hip','knee','ank')]
                                  for sd in L.SIDES],device=dev)
    with torch.no_grad():
        raw_params = tensor([track[0] for track in raw_track])
        source_skeleton = L.skel_torch(raw_params,dev)
        raw_joints = source_skeleton[:,leg_joints,:3]/100
        raw_camera = raw_joints @ tensor(M.F).T + tensor([track[1] for track in raw_track])[:,None,None,:]
        raw_uv = raw_camera[...,:2]/raw_camera[...,2:].clamp(min=.1)*tensor(body_focals)[:,None,None,None] + tensor(image_wh)/2
        raw_world = torch.einsum('tji,tskj->tski',E[:,:3,:3],raw_camera-E[:,None,None,:3,3])
        raw_pts = skinned_soles(source_skeleton,binding)
        raw_pts_cam = raw_pts @ tensor(M.F).T + tensor([track[1] for track in raw_track])[:,None,None,:]
        raw_pts_world = torch.einsum('tji,tspj->tspi',E[:,:3,:3],raw_pts_cam-E[:,None,None,:3,3])
        raw_low = (raw_pts_world@u).amin(2)
        raw_sole = raw_pts_world.mean(2)+(raw_low-(raw_pts_world.mean(2)@u))[:,:,None]*u
        bone_names = list(M.skeleton(dev)['names'])
        raw_spine = source_skeleton[:,bone_names.index('c_spine3'),:3]/100
        trunk = (raw_spine-raw_joints[:,:,0].mean(1)) @ tensor(M.F).T
        trunk = torch.einsum('tji,tj->ti',E[:,:3,:3],trunk)
        upright = torch.nn.functional.normalize(trunk,dim=-1)@u
        shin_samples = torch.stack([raw_camera[:,:,1],.4*raw_camera[:,:,1]+.6*raw_camera[:,:,2],raw_camera[:,:,2]],2)
        shin_uv = shin_samples[...,:2]/shin_samples[...,2:].clamp(min=.1)*tensor(body_focals)[:,None,None,None]+tensor(image_wh)/2
        joint_seen = mask_shin_visibility(masks,person_id,shin_uv.cpu().numpy(),shin_samples[...,2].cpu().numpy(),image_wh)
        surface0 = terrain.model_h(pts0.reshape(-1,3)).reshape(pts0.shape[:-1])
        gap0 = (pts0@up-surface0).min(2)-.009
        raw_xyz = raw_sole.cpu().numpy()
        raw_gap = raw_xyz@up-terrain.model_h(raw_xyz.reshape(-1,3)).reshape(n,2)
        raised = np.zeros((n,2),bool)
        if plan is not None and thighs is not None and prefill_pose is not None:
            source_state = L.skel_torch(tensor(prefill_pose),dev)
            hips = source_state[:,leg_joints[:,0],:3].cpu().numpy()/100
            lengths = [[np.linalg.norm(geo['side'][sd][bone]) for bone in ('t0','s0')] for sd in L.SIDES]
            ankle_heights = [geo['side'][sd]['ankle_h'] for sd in L.SIDES]
            raised = contradictory_swing(
                raw_world.cpu().numpy(), raw_xyz, raw_gap, upright.cpu().numpy(), joint_seen,
                thighs['confidence'], hips, lengths, ankle_heights, terrain.model_h,
                np.asarray(plan['done'],bool) & ~turning, plan['contact'], shot, fps, up)
            if raised.any():
                # Do not make a synthetically planted foot compete with its visible knee and ankle.
                source_soles = skinned_soles(source_state,binding).cpu().numpy()
                corrected = dict(plan,sole_world=plan['sole_world'].copy())
                source_height = np.column_stack([raw_world[:,:,0].mean(1).cpu().numpy()@up,raw_xyz@up])
                flight = common_flight(source_height,shot,fps)
                source_floor = terrain.model_h(source_soles.reshape(-1,3)).reshape(source_soles.shape[:-1])
                source_clearance = (source_soles@up-source_floor).min(2)
                corrected['contact'] = recovered_contacts(plan['contact'],raised,plan['done'] & ~turning,
                                                      shot,flight,source_clearance,lengths,fps)
                corrected['sole_world'][raised] = source_soles[raised]
                ramp_n = max(2, round(.12 * fps))
                for s_r in range(2):
                    for r_a, r_b in L.runs_of(raised[:, s_r]):
                        for d_k in range(1, ramp_n + 1):
                            k_pre = r_a - d_k
                            if k_pre >= 0 and shot[k_pre] == shot[r_a] and not corrected['contact'][k_pre, s_r]:
                                w_r = 1.0 - d_k / (ramp_n + 1.0)
                                corrected['sole_world'][k_pre, s_r] = (1.0 - w_r) * corrected['sole_world'][k_pre, s_r] + w_r * source_soles[k_pre, s_r]
                            k_post = r_b - 1 + d_k
                            if k_post < n and shot[k_post] == shot[r_b - 1] and not corrected['contact'][k_post, s_r]:
                                w_r = 1.0 - d_k / (ramp_n + 1.0)
                                corrected['sole_world'][k_post, s_r] = (1.0 - w_r) * corrected['sole_world'][k_post, s_r] + w_r * source_soles[k_post, s_r]
                registered_side = registration['support_side']
                join = mid if (mid and corrected['contact'][mid-1:mid+1,registered_side].all()
                               and not raised[mid-1:mid+1,registered_side].any()
                               and not flight[mid-1:mid+1].any()) else -1
                recovery.update(plan=corrected,raised=raised,
                                adjustment=affected_legs(plan['contact'],corrected['contact'],raised,shot,fps,join,registered_side))
                print("Observed swing overriding planned stance, frames per foot:",raised.sum(0).tolist(),flush=True)
        raised = recovery['raised'] if registered_pose is not None else np.zeros((n,2),bool)
        swing_w = raised.astype(np.float32)
        for s_r in range(2):
            for r_a, r_b in L.runs_of(raised[:, s_r]):
                if r_a > 0 and not joint_seen[r_a, s_r, 2]:
                    swing_w[r_a, s_r] = 0.35
                    if r_a + 1 < r_b and not joint_seen[r_a + 1, s_r, 2]:
                        swing_w[r_a + 1, s_r] = 0.70
        swing_points = tensor(swing_w[:, :, None] * joint_seen[:, :, [0, 2]].astype(np.float32))
        held_swing = tensor((raised[1:] | raised[:-1]).any(1) & hold_edges)
        inferred_plant = tensor(raised[:,::-1].copy())
        has_swing = raised.any()
    targets, support, planned = foot_targets(plan, support, pts0, up)
    if thighs is not None:
        th = {k:tensor(value) for k,value in thighs.items()}
        # Observed knees constrain otherwise inferred legs. A turn on a
        # registered support also has measured terrain and contact geometry.
        partial = th["confidence"]*tensor((planned & ~turning)[:,None])
    else:
        partial = tensor(np.zeros((n,2)))
    registered = active.copy()
    registered[pre:mid] = True
    fixed = planned & ~registered
    same_shot = tensor(shot[1:]==shot[:-1])
    fixed_root = torch.as_tensor(fixed,device=dev)
    # The calibrated actor trajectory carries the hidden legs. A measured,
    # registered plant has its own world-space reference at the camera handoff.
    if thighs is not None:
        partial[pre:mid,side] = 0.
    # Preserve clearance over the swing's chord independently of where its
    # footprints are registered. This also works on successive stair treads.
    swing_height = np.zeros((n,2))
    swing_valid = np.zeros((n,2), bool)
    target_low = (targets @ up).min(2)
    for side_index in range(2):
        for lo,hi in L.runs_of(planned & ~support[:,side_index]):
            if lo == 0 or hi == n or not (support[lo-1,side_index] and support[hi,side_index]):
                continue
            fraction = np.arange(1,hi-lo+1)/(hi-lo+1)
            lo_h,hi_h = target_low[[lo-1,hi],side_index]
            swing_height[lo:hi,side_index] = np.maximum(0.,target_low[lo:hi,side_index] - (lo_h + fraction*(hi_h-lo_h)))
            swing_valid[lo:hi,side_index] = True
    swing_t = tensor(swing_valid)
    lift_t = tensor(swing_height)
    # A continuous footfall has an unknown absolute placement in world space.
    # Estimate one rigid horizontal offset per footprint. Interpolate offsets
    # during swings so the image observations can register a whole step.
    pairs = np.zeros((n, 2, 2), dtype=int)
    blend = np.zeros((n, 2), dtype=float)
    count = 1
    for j in range(2):
        for lo, hi in L.runs_of(support[:, j]):
            pairs[lo:hi, j] = count
            count += 1
        for lo, hi in L.runs_of(~support[:, j]):
            left = pairs[lo-1, j, 0] if (lo and shot[lo-1] == shot[lo]) else 0
            right = pairs[hi, j, 0] if (hi < n and shot[hi-1] == shot[hi]) else 0
            if left == 0 and right != 0:
                left = right
            elif right == 0 and left != 0:
                right = left
            pairs[lo:hi, j] = [left, right]
            blend[lo:hi, j] = np.arange(1, hi-lo+1)/(hi-lo+1)
    event_ids = torch.as_tensor(pairs, device=dev)
    mix = tensor(blend)[..., None]
    placements = torch.zeros((count,2), device=dev, requires_grad=True)
    target = tensor(targets)
    direction_target, lateral_target = sole_orientation(target)
    # Angular errors have to be measured in unit-vector space. A wrongly
    # rotated sole can have an almost perfect center and small vertex errors.
    turn_weight = gaussian_filter1d(turning.astype(float),max(1.,fps*.055),mode="nearest")
    orientation_mask = tensor((planned[:,None] | support)*turn_weight[:,None])
    # Transfer weight before liftoff, then ease toward the next footprint.
    # Use only inferred turning steps. The visible kicking leg keeps its pose.
    load_share = support.astype(float)
    load_share /= np.maximum(1.,load_share.sum(1,keepdims=True))
    load_share = gaussian_filter1d(load_share, max(1.,fps*.075),axis=0,mode="nearest")
    load_share /= np.maximum(1e-8,load_share.sum(1,keepdims=True))
    share = tensor(load_share)
    balance_mask = tensor(turning & support.any(1))
    mask = tensor(support)
    planned_t = tensor(planned)
    active_t=torch.as_tensor(active,device=dev)
    cuts_t=torch.as_tensor(boundaries,device=dev)
    # A camera cut can require registration of a new shot. Elsewhere an
    # observed, unplanned walking trajectory already supplies metric depth.
    # A 2-D image residual alone cannot constrain motion along a camera ray.
    observed_path = ~planned & ~active
    observed_support_avail = observed_path.copy()
    for cut in boundaries:
        if planned[cut - 1] or planned[cut] or active[cut - 1] and active[cut]:
            radius = round(.45*fps)
            # Only relax observed_t on planned/active cut transitions; keep fully observed shots anchored.
            mask_cut = np.zeros(n, bool)
            mask_cut[max(0, cut - radius):min(n, cut + radius + 1)] = True
            observed_path[mask_cut & (planned | active)] = False
    observed_t = tensor(observed_path)
    image_weight = torch.where(observed_t.bool(), 1.4, 140. + 55.*tensor(turn_weight))

    # Use ankle and knee order from the source pose, in the pelvis frame.
    # A crossed stance is valid when it is actually visible in both feet.
    with torch.no_grad():
        in_frame = ((raw_uv >= 0) & (raw_uv < tensor(image_wh))).all(-1).all(1)
        raw_span = raw_joints[:,0]-raw_joints[:,1]
        hip_width = raw_span[:,0].norm(dim=1)
        hip_axis = raw_span[:,0] / hip_width[:,None].clamp(min=.01)
        width = (raw_span*hip_axis[:,None]).sum(dim=-1)
        # If the ankles are cropped, retain a modest uncrossed step width
        # when the source agrees or the visible knees indicate an open stance.
        minimum = torch.minimum(width,hip_width[:,None]*.95)
        minimum = torch.where(in_frame,minimum-.015,minimum.clamp(min=.04)-.01)
        side_hints = tensor(planned | active)
        image_span = (raw_uv[:,0]-raw_uv[:,1])/height[:,None,None]
        evidence, movement = observed_swing_weights(
            raw_uv.cpu().numpy(), raw_camera[...,2].cpu().numpy(), height.cpu().numpy(),
            image_wh, shot, support | planned[:,None], fps, masks, person_id)
        observation = tensor(evidence)
        observed_side = tensor(movement)
        ankle_vis = tensor(joint_seen[:,:,2].astype(np.float32))
        eff_move = observed_side * (.35 + .65 * ankle_vis)
        # Use full damping on inferred/support legs. A moving, measured leg keeps
        # some damping while its two image landmarks constrain actual motion.
        column_sides = [0 if c in L.LEG_COLS[L.SIDES[0]] else 1 for c in cols]
        damping = 1. - .92*eff_move[:,column_sides]
        velocity_weight = torch.maximum(damping[1:],damping[:-1])
        accel_weight = torch.maximum(velocity_weight[1:],velocity_weight[:-1])
        relative_target = (raw_uv[:,:,1:]-raw_uv[:,:,:1])/height[:,None,None,None]
        evidence_frames = observed_side.amax(1)
        # Anchor the supporting foot of an observed swing independently from
        # the per-leg image fit. Transform raw observations before ground lift.
        loaded = support_observations(raw_world.cpu().numpy(),up,raw_sole.cpu().numpy(),gap0,
                                      upright.cpu().numpy(),joint_seen[:,:,:2].all(2),raw_gap,movement,observed_support_avail,shot,fps)
        # Use the forefoot on pivots, allowing the heel to turn about it.
        # Each continuous footfall gets its own stationary XY reference.
        fore = pts0[:,:,2:4].mean(2)
        anchor = fore.copy()
        contact_height = np.zeros(loaded.shape)
        ext = E.cpu().numpy()
        intr = K.cpu().numpy()
        ankle_uv = raw_uv[:,:,2].cpu().numpy()
        base_skeleton = L.skel_torch(base_t,dev)
        ankle_offset = base_skeleton[:,leg_joints[:,2],:3].cpu().numpy()/100.-fore
        for side in range(2):
            for lo,hi in L.runs_of(loaded[:,side]):
                event = slice(lo,hi)
                candidates = np.flatnonzero(np.abs(gap0[event,side]) < .045)+lo
                if not len(candidates):
                    candidates = np.arange(lo,hi)
                samples = fore[candidates,side]
                # A drifting world estimate can cross multiple plausible treads.
                # Rank candidate footprints with independent ankle image rays,
                # using only visible ankles. A median footprint breaks ties.
                observed_ankle = ankle_uv[event,side]
                valid = joint_seen[event,side,2]
                center = np.median(samples,axis=0)
                cost = .025*np.linalg.norm(samples-center,axis=1)
                if valid.sum() >= 2:
                    points = samples[None,:,:]+ankle_offset[event,side,None,:]
                    eye = np.einsum('fij,fcj->fci',ext[event,:3,:3],points)+ext[event,None,:3,3]
                    pix = np.einsum('fij,fcj->fci',intr[event],eye)
                    pred = pix[:,:,:2]/np.maximum(.05,pix[:,:,2:3])
                    err = np.linalg.norm(pred-observed_ankle[:,None],axis=-1)/np.asarray(sizes)[event,None]
                    cost += np.median(err[valid],axis=0)
                best = candidates[np.argmin(cost)]
                anchor[event,side] = fore[best,side]
                contact_height[event,side] = terrain.model_h(fore[best,side][None])[0]
        observed_floor = tensor(contact_height)
        observed_anchor = tensor(anchor)
        locked = tensor(loaded)
        adjacent_lock = locked[1:]*locked[:-1]
        # The old trajectory and a stationary tread can disagree in depth.
        # Make the correction continuous when acquiring/releasing the pivot.
        # Track changes of the source movement, not zero velocity after release.
        handoff = np.zeros((n,2))
        root_source = raw_world[:,:,0].mean(1).cpu().numpy().copy()
        foot_source = raw_pts_world[:,:,2:4].mean(2).cpu().numpy()
        airborne = common_flight(np.column_stack([root_source@up,raw_xyz@up]),shot,fps)
        source_root_air = root_source[airborne].copy()
        source_feet_air = foot_source[airborne].copy()
        for sid in np.unique(shot):
            ix = np.flatnonzero(shot==sid)
            local = loaded[ix]
            root_source[ix] = gaussian_filter1d(root_source[ix],max(.5,.055*fps),axis=0,mode='nearest')
            foot_source[ix] = gaussian_filter1d(foot_source[ix],max(.5,.055*fps),axis=0,mode='nearest')
            for side in range(2):
                for edge in np.flatnonzero(np.diff(local[:,side]))+1:
                    weight = np.exp(-.5*((np.arange(len(ix))-edge)/max(1.,.30*fps))**2)
                    handoff[ix,side] = np.maximum(handoff[ix,side],weight)
        root_source[airborne] = source_root_air
        foot_source[airborne] = source_feet_air
        # Do not bias an established contact towards a drifting source.
        transition = tensor(handoff[1:])*(1-adjacent_lock)*tensor((shot[1:]==shot[:-1])[:,None])
        # On an observed swing the other planted leg already anchors the
        # actor. Do not penalize its kick as a transition from an old stance.
        transition *= 1.-observed_side[1:]*locked[1:].flip(1)
        # No new lock or handoff penalty on a recognized flight interval.
        transition *= tensor(~(airborne[1:] | airborne[:-1]))[:,None]
        root_transition = transition.amax(1)
        corner = tensor((shot[2:]==shot[:-2]) & ~(airborne[2:] | airborne[1:-1] | airborne[:-2]))
        foot_corner = transition[1:]*corner[:,None]
        body_corner = root_transition[1:]*corner
        source_foot_step = tensor(np.diff(foot_source,axis=0))
        source_body_step = tensor(np.diff(root_source,axis=0))

        print("Observed support frames per foot:",loaded.sum(0).tolist(),flush=True)

    # Keep a planted midsole on its tread while fitting visible knees. The
    # shared pivot is estimated from all stance frames, allowing a free footfall
    # location. Only the registered support continues through its camera edit.
    registered_side = registration["support_side"]
    join = mid
    if has_swing and mid and (raised[mid-1:mid+1,registered_side].any() or airborne[mid-1:mid+1].any()):
        join = -1
    groups, n_stances = stance_groups(support,shot,join,registered_side)
    g_registered = groups[mid,registration["support_side"]]
    bridge_contact = (groups == g_registered) & (g_registered != 0)
    seams = np.flatnonzero(np.isin(vf, verified_frames))
    prior, seam_strength = registered_root_prior(root, shot, fps, hold_edges, seams, calibrated_root, up, airborne)
    seam_velocity = tensor(np.maximum(seam_strength[1:],seam_strength[:-1]))
    independent_velocity = tensor(prior).diff(dim=0)
    prior_shift = tensor((prior-root)*fixed[:,None])
    tilt_weight = tensor(inferred_sole_weights(fixed,joint_seen[:,:,2],airborne,support))
    planned_tilt = sole_inclination(target,u)
    distal_indices, source_foot, foot_slack = L.distal_prior(raw_params,cols)
    material_contact = tensor(support & fixed[:,None] & ~joint_seen[:,:,2] & ~airborne[:,None])
    stance_weighted = tensor(support & (fixed[:,None] | bridge_contact) & ~airborne[:,None])
    stance_group = torch.as_tensor(groups,device=dev)
    group_flat = stance_group.reshape(-1)
    material_count = torch.zeros(n_stances,device=dev).index_add(0,group_flat,material_contact.reshape(-1))
    group_size = torch.zeros(n_stances,device=dev).index_add(0,group_flat,stance_weighted.reshape(-1))
    if registered_pose is not None:
        fixed_displacement = tensor(M.root_pos(registered_pose)-root)
        registered_leg = tensor(registered_pose[:,cols])
        adjustment = recovery['adjustment'][:,column_sides]
        factor = tensor(adjustment)
        plant = (recovery['adjustment']==1) & support & ~airborne[:,None]
        recovered_group = np.unique(groups[raised[:,::-1]])
        plant &= np.isin(groups,recovered_group[recovered_group!=0])
        whole_plant = tensor(plant)
        stable_pair = tensor(plant[1:] & plant[:-1] & (groups[1:]==groups[:-1]))
        seam_edges = tensor(stance_transitions(support,raised,shot,airborne,fps,join,registered_side))
        with torch.no_grad():
            prior_soles = skinned_soles(L.skel_torch(tensor(registered_pose),dev),binding).mean(2).cpu().numpy()
        reference_contact = tensor(plant & registered[:,None])
        with torch.no_grad():
            registered_soles = skinned_soles(L.skel_torch(tensor(registered_pose),dev),binding)
        # Keep the registered footprint and avoid increasing an existing cut step
        # when only one side of that edit belongs to the affected footfall.
        edit_side = ((recovery['adjustment'][1:]>0) != (recovery['adjustment'][:-1]>0))
        edit_mask = tensor(edit_side & (shot[1:]!=shot[:-1])[:,None])
        baseline_step = registered_soles.mean(2).diff(dim=0)
        baseline_step -= (baseline_step@u)[:,:,None]*u
        # Use the normal swing speed at a plant handoff. A single abrupt IK
        # sample is a poor estimate of the speed at which a foot leaves the tread.
        for first,last in zip(np.r_[0,boundaries],np.r_[boundaries,n]):
            prior_soles[first:last] = gaussian_filter1d(prior_soles[first:last],max(.65,.085*fps),axis=0,mode='nearest')
        release_velocity = tensor(np.diff(prior_soles,axis=0))
    shared_exposure = tensor(hold_edges & registered[1:] & registered[:-1])
    opt=torch.optim.Adam([{'params':[leg],'lr':.017},{'params':[dx, placements],'lr':.014}])
    for it in range(500):
        opt.zero_grad()
        p=base_t.clone()
        p[:,cols] = leg if registered_pose is None else registered_leg+factor*(leg-registered_leg)
        # Root translation commutes with FK; add it in metres.
        sk = L.skel_torch(p,dev)
        delta = root_displacement(dx,prior_shift,fixed_root) if registered_pose is None else fixed_displacement
        pts=skinned_soles(sk,binding)+delta[:,None,None,:];foot=pts.mean(2);rp=root0+delta
        surf=terrain.model_h(pts.detach().cpu().numpy().reshape(-1,3)).reshape(*pts.shape[:-1])
        gap=pts@u-tensor(surf)
        # Sole landmarks sit slightly above the lowest skinned surface.
        ground=gap.amin(2)-.009
        moves = torch.cat([torch.zeros_like(placements[:1]), placements[1:]])
        placement = moves[event_ids[:,:,0]]*(1-mix)+moves[event_ids[:,:,1]]*mix
        shift = placement[:,:,0,None]*along + placement[:,:,1,None]*across
        error = pts-target-shift[:,:,None,:]
        horizontal = error-(error@u)[..., None]*u
        # The planner already supplied the swing height and the heel/toe roll.
        # Retain all five sole landmarks, including while the foot is in the air.
        footprint_weight = 1.-.88*partial
        loss = 140*(horizontal.mean(2).square().sum(-1)*mask*footprint_weight).sum()/(mask.sum()+1)
        loss += 45*((error.mean(2).square().sum(-1)*footprint_weight).mean(1)*planned_t).sum()/(planned_t.sum()+1)
        loss += 2*((error.square().sum(-1).mean(2)*footprint_weight).mean(1)*planned_t).sum()/(planned_t.sum()+1)
        loss += 2300*(ground.square()*stance_weighted).sum()/stance_weighted.sum().clamp(min=1)
        marker = (pts[:,:,0]+pts[:,:,2]+pts[:,:,3])/3.
        total = pts.new_zeros((n_stances,3)).index_add(0,group_flat,(marker*stance_weighted[:,:,None]).reshape(-1,3))
        centers = total/group_size[:,None].clamp(min=1)
        pivot_error = marker-centers[stance_group]
        pivot_error = pivot_error-(pivot_error@u)[:,:,None]*u
        loss += 5000*(pivot_error.square().sum(-1)*stance_weighted).sum()/stance_weighted.sum().clamp(min=1)
        # Estimate a single footprint origin for each loaded footfall. Keep
        # the planned heel roll while preventing the rest of the skin from
        # rotating around an arbitrary fixed midsole marker.
        material_delta = pts-target
        material_sum = pts.new_zeros((n_stances,3)).index_add(
            0,group_flat,(material_delta.mean(2)*material_contact[:,:,None]).reshape(-1,3))
        origin = material_sum/material_count[:,None].clamp(min=1)
        material_delta = material_delta-origin[stance_group][:,:,None]
        planar_delta = material_delta-(material_delta@u)[:,:,:,None]*u
        loss += 1200*(planar_delta.square().sum(-1).mean(2)*material_contact).sum()/material_contact.sum().clamp(min=1)
        deviation = (leg[:,distal_indices]-source_foot).abs()
        loss += 128*torch.relu(deviation-foot_slack).square().mean()
        tilt_residual = (sole_inclination(pts,u)-planned_tilt).square().mean(-1)
        loss += 120*(tilt_residual*tilt_weight).sum()/tilt_weight.sum().clamp(min=1)
        direction, lateral = sole_orientation(pts)
        orientation_error = (direction-direction_target).square().sum(-1)
        orientation_error += (lateral-lateral_target).square().sum(-1)
        loss += 12.0*(orientation_error*orientation_mask).sum()/(orientation_mask.sum()+1)
        lift_error = torch.relu(lift_t - ground)
        loss += 320*(swing_t*lift_error.square()).sum()/(swing_t.sum()+1)
        loss += 100*(ground.square()*mask).sum()/(mask.sum()+1)
        loss += 18*torch.relu(-ground).square().mean()+1.5*ground.amin(1).square().mean()
        # During weight transfer the supporting sole must also be the low
        # sole. A shared root-height adjustment alone can ground the swing leg.
        stance_weight = mask * balance_mask[:,None]
        loss += 850*(stance_weight*ground.square()).sum()/(stance_weight.sum()+1)
        low_support = torch.where(mask.bool(),ground,torch.full_like(ground,10.)).amin(1)
        swing_weight = (1-mask)*balance_mask[:,None]*swing_t
        swing_gap = torch.relu(low_support[:,None]+lift_t-ground)
        loss += 650*(swing_weight*swing_gap.square()).sum()/(swing_weight.sum()+1)
        joints = sk[:,leg_joints,:3]/100 + delta[:,None,None,:]
        if thighs is not None:
            thigh_error = projected_thigh_error(joints[:,:,0],joints[:,:,1],th["projection"],th["size"],th["relative"])
            loss += 2600*(thigh_error*partial).sum()/partial.sum().clamp(min=1)
        support_error, spread_error = turn_support_loss(joints[:,:,0],pts,share,u)
        loss += (balance_mask*(140.*support_error + 100.*spread_error)).sum()/(balance_mask.sum()+1)
        span = joints[:,0]-joints[:,1]
        side_axis = torch.nn.functional.normalize(span[:,0],dim=-1)
        widths = (span*side_axis[:,None]).sum(-1)
        order_error = torch.relu(minimum[:,1:]-widths[:,1:]).square().mean(1)
        loss += 400*(order_error*side_hints).mean()
        # Match the visible relative leg pose independently of the torso placement.
        camera_joints = torch.einsum('tij,tskj->tski', E[:,:3,:3], joints) + E[:,None,None,:3,3]
        pix = torch.einsum('tij,tskj->tski', K, camera_joints)
        uv_joints = pix[...,:2]/pix[...,2:].clamp(min=.2)
        view_span = (uv_joints[:,0]-uv_joints[:,1])/height[:,None,None]
        visible_err = (view_span[:,1:]-image_span[:,1:]).square().sum(-1)*in_frame[:,1:]
        loss += 2*(visible_err.mean(1)*side_hints).mean()
        relative = (uv_joints[:,:,1:]-uv_joints[:,:,:1])/height[:,None,None,None]
        measured = (relative-relative_target).square().sum(-1)
        loss += 460*(measured*observation).sum()/observation.sum().clamp(min=1)
        if has_swing:
            loss += 18000*(ground.square()*whole_plant).sum()/whole_plant.sum().clamp(min=1)
            heel_to_ball = pts[:,:,[0,2,3]].mean(2)
            pivot_step = heel_to_ball.diff(dim=0)
            pivot_step = pivot_step-(pivot_step@u)[:,:,None]*u
            loss += 8000*(pivot_step.square().sum(-1)*stable_pair).sum()/stable_pair.sum().clamp(min=1)
            anchor_residual = (pts-registered_soles).square().sum(-1).mean(2)
            loss += 5000*(anchor_residual*reference_contact).sum()/reference_contact.sum().clamp(min=1)
            jumps = foot.diff(dim=0)
            jumps = jumps-(jumps@u)[:,:,None]*u
            excess = torch.relu(jumps.norm(dim=-1)-baseline_step.norm(dim=-1))
            loss += 8000*(excess.square()*edit_mask).sum()/edit_mask.sum().clamp(min=1)
            tangent = foot.diff(dim=0)-release_velocity
            loss += 2200*(tangent.square().sum(-1)*seam_edges).sum()/seam_edges.sum().clamp(min=1)
            loss += 6500*(ground.square()*inferred_plant).sum()/inferred_plant.sum().clamp(min=1)
            loss += 2200*(measured*swing_points).sum()/swing_points.sum().clamp(min=1)
            # An exposure held by the source should also hold the observed swing.
            foot_step = pts.diff(dim=0).square().sum(-1).mean((1,2))
            joint_step = joints.diff(dim=0).square().sum(-1).mean((1,2))
            loss += 1800*(held_swing*(foot_step+joint_step)).sum()/held_swing.sum().clamp(min=1)
        swing_confidence = observation*locked.flip(1)[:,:,None]*observed_side[:,:,None]
        swing_pixel = ((uv_joints[:,:,1:]-raw_uv[:,:,1:])/height[:,None,None,None]).square().sum(-1)
        loss += 190*(swing_pixel*swing_confidence).sum()/swing_confidence.sum().clamp(min=1)
        # The apparent height of a kick also depends on torso placement. A
        # root-image residual anchors it without inventing depth or a footfall.
        pelvis_delta = (uv_joints[:,:,0]-raw_uv[:,:,0]).mean(1)/height[:,None]
        loss += 260*(pelvis_delta.square().sum(1)*evidence_frames).mean()

        forefoot = pts[:,:,2:4].mean(2)
        drift = forefoot-observed_anchor
        planar = drift-(drift@u)[:,:,None]*u
        loss += 1600*(planar.square().sum(-1)*locked).sum()/locked.sum().clamp(min=1)
        # Target the lowest of the five soles only inside an established
        # contact. During flight this term and its horizontal pivot both vanish.
        contact_gap = (pts@u).amin(2)-observed_floor-.009
        loss += 1200*(contact_gap.square()*locked).sum()/locked.sum().clamp(min=1)
        step = forefoot.diff(dim=0)
        step = step-(step@u)[:,:,None]*u
        loss += 3200*(step.square().sum(-1)*adjacent_lock).sum()/adjacent_lock.sum().clamp(min=1)
        foot_residual = forefoot.diff(dim=0)-source_foot_step
        body_residual = rp.diff(dim=0)-source_body_step
        loss += 1300*(foot_residual.square().sum(-1)*transition).sum()/transition.sum().clamp(min=1)
        loss += 2200*(body_residual.square().sum(-1)*root_transition).sum()/root_transition.sum().clamp(min=1)
        foot_curve = foot_residual.diff(dim=0).square().sum(-1)
        body_curve = body_residual.diff(dim=0).square().sum(-1)
        loss += 5400*(foot_curve*foot_corner).sum()/foot_corner.sum().clamp(min=1)
        loss += 6500*(body_curve*body_corner).sum()/body_corner.sum().clamp(min=1)

        same_contact = mask[1:]*mask[:-1]
        pivot = (torch.maximum(target[1:]@u,target[:-1]@u)).argmin(2)
        track = pivot[:,:,None,None].expand(-1,-1,1,3)
        slide = (pts[1:]-pts[:-1]).gather(2,track).squeeze(2)
        slide = slide-(slide@u)[:,:,None]*u
        loss += 2200*(same_contact*slide.square().sum(-1)).sum()/(same_contact.sum()+1)
        # The planner's pivot and the lowest point on the fitted sole can
        # differ while the heel rolls. Constrain motion at both contacts.
        pair_gap = torch.maximum(gap[1:],gap[:-1]).detach()
        near_ground = torch.softmax(-pair_gap/.006,dim=2)
        vertex_step = pts[1:]-pts[:-1]
        planar_step = vertex_step-(vertex_step@u)[...,None]*u
        ground_slide = (planar_step.square().sum(-1)*near_ground).sum(2)
        turning_pairs = same_contact*balance_mask[1:,None]*balance_mask[:-1,None]
        loss += 220*(turning_pairs*ground_slide).sum()/(turning_pairs.sum()+1)
        min_pair_gap = torch.maximum(gap[1:], gap[:-1]).amin(2)
        touch_gate = torch.sigmoid((.022 - min_pair_gap) / .004)
        non_air = tensor(~(airborne[1:] | airborne[:-1])) * same_shot
        free_kick = (observed_side[1:] > .45) & (ankle_vis[1:] > .5)
        scrape_excess = torch.relu(planar_step.norm(dim=-1) - .36 / fps).square()
        active_scrape = (scrape_excess.detach().amax(2) > 0).float() * non_air[:, None] * (~free_kick).float()
        scrape = (scrape_excess * near_ground).sum(2) * touch_gate * non_air[:, None] * (~free_kick).float()
        loss += 3200 * scrape.sum() / active_scrape.sum().clamp(min=1)
        sx,wy=pts[active_t]@along,pts[active_t]@across
        loss+=5*(torch.relu(b[0]+.015-sx).square()+torch.relu(sx-b[1]+.015).square()+torch.relu(b[2]+.015-wy).square()+torch.relu(wy-b[3]+.015).square()).mean()
        # Keep original image-plane upper-body localization and depth scale.
        pc=(E[:,:3,:3]@rp[:,:,None]).squeeze(-1)+E[:,:3,3]
        hp=(K@pc[:,:,None]).squeeze(-1)
        repro=(hp[:,:2]/hp[:,2:].clamp(min=.2)-uv)/height[:,None]
        loss += (image_weight*repro.square().mean(1)).mean()
        loss += (observed_t*dx.square().sum(1)).sum()/(observed_t.sum()+1)*3
        loss += .002*placements.square().mean()+.025*dx.square().mean()+.025*(leg-sy).square().mean()
        ds=rp[1:]-rp[:-1];df=foot[1:]-foot[:-1]
        loss+=5*(ds[1:]-ds[:-1]).square().mean()+2*(df[1:]-df[:-1]).square().mean()
        # Stabilize corrections to the calibrated path within each continuous shot.
        motion_error = (ds-independent_velocity)*same_shot[:,None]
        loss += 1200*motion_error.diff(dim=0).square().mean()+100*motion_error.square().mean()
        loss += 4000*((ds-independent_velocity).square().sum(1)*seam_velocity).sum()/seam_velocity.sum().clamp(min=1)
        loss+=.65*ds.square().mean()+4*(leg.diff(dim=0).square()*velocity_weight).mean()+65*(leg.diff(n=2,dim=0).square()*accel_weight).mean()
        loss+=95*ds[cuts_t-1].square().mean()+14*df[cuts_t-1].square().mean()
        velocity = leg.diff(dim=0)*fps
        acceleration = velocity.diff(dim=0)*fps
        loss += .2*(torch.relu(velocity.abs()-4.5).square()*velocity_weight).mean()
        loss += .00035*(torch.relu(acceleration.abs()-40).square()*accel_weight).mean()
        if it>320:
            for group in opt.param_groups:group['lr']=.004
        loss += 3200*(ds.square().sum(-1)*shared_exposure).sum()/shared_exposure.sum().clamp(min=1)
        loss.backward();opt.step()
        with torch.no_grad():leg[:,knees]=leg[:,knees].clamp(.015,L.KNEE_MAX)
    if registered_pose is not None:
        result = registered_pose.copy()
        candidate_leg = (registered_leg+factor*(leg-registered_leg)).detach().cpu().numpy()
        result[:,cols] = np.where(adjustment>0,candidate_leg,registered_pose[:,cols])
        eligible = planned & (adjustment>0).any(1)
        result = resolve_inferred_sole_clearance(result,shape,up,terrain,dict(plan,done=eligible,turning=np.zeros_like(planned)),
                                                 np.setdiff1d(boundaries,seams),fps,dev,airborne)
        result[:,cols] = np.where(adjustment>0,result[:,cols],registered_pose[:,cols])
        return result
    delta=root_displacement(dx,prior_shift,fixed_root)
    out=M.set_root(base,None,root+delta.detach().cpu().numpy());out[:,cols]=leg.detach().cpu().numpy()
    confirmed = (joint_seen[:,:,2] & (np.abs(raw_gap)<.05)).any(1) & ~airborne
    if plan is not None and thighs is not None:
        open_stance = open_turn_evidence(hip_width.cpu().numpy(), width[:,1].cpu().numpy(),
                                         width[:,2].cpu().numpy(), joint_seen[:,:,0], joint_seen[:,:,2],
                                         np.asarray(thighs['confidence']))
        observed_crossing = joint_seen[:,:,2].all(1) & (width[:,2].cpu().numpy() < 0.)
        plan = dict(plan, open_stance=open_stance & registered, observed_flight=airborne,
                    observed_crossing=observed_crossing)
    out = match_sole_height(out, shape, up, terrain, plan, params, boundaries, fps, dev, fixed, confirmed)
    out = fit_exposure_states(out, shape, shot, fps, up, terrain, dev, hold_edges,
                              ~(fixed | turning | airborne))
    out = fit_unloaded_exposures(out, shape, shot, fps, up, terrain, dev, hold_edges, turning,
                                planned, support, airborne, registered)
    return resolve_inferred_sole_clearance(out, shape, up, terrain, plan,
                                           np.setdiff1d(boundaries,seams), fps, dev, airborne)
