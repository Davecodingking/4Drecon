"""Fit root and lower-body state to repeated actor exposures on the actual terrain."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation, Slerp

import legs_proc as L
import mhr_body as M


def exposure_clock(held, eligible, shot, fps):
    """Coalesce exposure times; taper the phase adjustment over nearby active samples."""
    n = len(eligible)
    frame = np.arange(n, dtype=float)
    if n < 2:
        return frame
    eligible = np.asarray(eligible, bool)
    same = np.asarray(held, bool) & eligible[:-1] & eligible[1:] & (np.diff(shot) == 0)
    strong = np.r_[False, same] | np.r_[same, False]
    offsets = np.zeros(n)
    for lo, hi in np.flatnonzero(np.diff(np.r_[False, same, False])).reshape(-1, 2):
        offsets[lo:hi+1] = (lo+hi)/2 - frame[lo:hi+1]
    breaks = ~eligible[:-1] | ~eligible[1:] | (np.diff(shot) != 0)
    bounds = np.r_[0, np.flatnonzero(breaks)+1, n]
    radius = max(2, round(.16*fps))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        local = strong[lo:hi]
        if not local.any():
            continue
        pinned = local | (distance_transform_edt(~local) >= radius)
        pinned[[0, -1]] = True
        indices = np.flatnonzero(pinned) + lo
        offsets[lo:hi] = np.interp(frame[lo:hi], frame[indices], offsets[indices])
    # A long editorial freeze supplies one pose. It cannot justify a shift
    # of several source frames on actions either side of it.
    offsets[~strong] = np.clip(offsets[~strong], -.5, .5)
    return frame + offsets


def fit_exposure_states(pose, shape, shot, fps, up, terrain, dev, held, eligible):
    """Refit root and leg time within measured holds and verify both skin surfaces."""
    frame = np.arange(len(pose))
    time = exposure_clock(held, eligible, shot, fps)
    fixed_repeat = np.r_[False, np.diff(time)==0] | np.r_[np.diff(time)==0, False]
    active = np.flatnonzero((time != frame) | fixed_repeat)
    if not len(active):
        return pose
    out = pose.copy()
    rest = M.rest_verts(shape, dev)
    sole = np.flatnonzero(rest[:,1] < rest[:,1].min() + 10)
    columns = [col for side in L.SIDES for col in L.LEG_COLS[side]]
    positions = M.root_pos(pose)
    orientations = Slerp(frame, Rotation.from_matrix(M.root_rot(pose)))
    def clearance(params):
        result = []
        for start in range(0, len(params), 24):
            batch = params[start:start+24]
            distances = []
            for corrective in (False, True):
                verts, _ = M.forward(shape, batch, corrective, dev)
                feet = verts[:,sole]
                floor = terrain.model_h(feet.reshape(-1,3)).reshape(feet.shape[:2])
                distances.append((feet@up - floor).min(1))
            result.extend(np.min(distances, axis=0))
        return np.asarray(result)

    # Only inspect source poses that actually bracket affected frames.
    start = np.floor(time[active]).astype(int)
    stop = np.ceil(time[active]).astype(int)
    needed = np.unique(np.r_[active, start, stop])
    # The minimum mesh clearance of a unique observation is the contact
    # reference, also when neither foot is grounded.
    baseline = np.full(len(pose), np.nan)
    baseline[needed] = clearance(pose[needed])
    for _ in range(7):
        lo = np.floor(time[active]).astype(int)
        hi = np.ceil(time[active]).astype(int)
        phase = (time[active]-lo)[:,None]
        trial = pose[active].copy()
        trial[:,columns] = (1-phase)*pose[lo][:,columns] + phase*pose[hi][:,columns]
        trial = M.set_root(trial,orientations(time[active]).as_matrix(),
                           (1-phase)*positions[lo] + phase*positions[hi])
        target = (1-phase[:,0])*baseline[lo] + phase[:,0]*baseline[hi]
        lift = target - clearance(trial)
        # Crossing a stair riser changes height discontinuously. Move a weak
        # neighbour less, or retain the original held group if it is infeasible.
        continuous_leg = np.max(np.abs(pose[hi][:,columns]-pose[lo][:,columns]), axis=1) <= np.pi
        feasible = (np.abs(lift) <= .012) & continuous_leg
        for stamp in np.unique(time[active][fixed_repeat[active]]):
            group = fixed_repeat[active] & (time[active] == stamp)
            if feasible[group].all():
                lift[group] = lift[group].max()
            else:
                feasible[group] = False
        accept = active[feasible]
        if len(accept):
            out[accept] = M.set_root(trial[feasible], None, M.root_pos(trial[feasible])+lift[feasible,None]*up)
        active = active[~feasible & ~fixed_repeat[active]]
        if not len(active):
            break
        time[active] = (time[active]+frame[active])/2
    return out



def fit_unloaded_exposures(pose, shape, shot, fps, up, terrain, dev, held, turning, planned, support, airborne, registered):
    """Share the pose of an unloaded leg across repeated exposures without moving its support."""
    if not np.any(held):
        return pose
    rest = M.rest_verts(shape, dev)
    sole = [np.flatnonzero((rest[:,1] < rest[:,1].min()+10) & (sign*rest[:,0] > 0)) for sign in (1, -1)]

    def soles(params):
        feet = [[], []]
        for start in range(0, len(params), 32):
            vertices, _ = M.forward(shape, params[start:start+32], False, dev)
            for side, ids in enumerate(sole):
                feet[side].append(vertices[:,ids])
        return [np.concatenate(part) for part in feet]

    def clearance(points):
        floor = terrain.model_h(points.reshape(-1,3)).reshape(points.shape[:2])
        return points @ up - floor

    n = len(pose)
    points = soles(pose)
    gaps = [clearance(part) for part in points]
    lowest = np.column_stack([a.min(1) for a in gaps])
    stable = np.zeros((n,2), bool)
    for side in range(2):
        adjacent = np.maximum(np.abs(gaps[side][1:]), np.abs(gaps[side][:-1])) < .015
        motion = np.linalg.norm(np.diff(points[side], axis=0), axis=-1)
        edge = (np.sum(adjacent & (motion*fps < .35), axis=1) >= 3) & (np.diff(shot)==0)
        for lo, hi in L.runs_of(edge):
            if hi-lo >= max(2, round(.10*fps)):
                stable[lo:hi+1,side] = True

    out = pose.copy()
    frames = np.arange(n)
    for side, name in enumerate(L.SIDES):
        other = 1-side
        inferred = planned & ~support[:,side] & support[:,other] & ~airborne
        material_contact = registered & stable[:,other]
        eligible = ~airborne & ~turning & (inferred | material_contact) & (lowest[:,side]>.025) & (lowest[:,other]<.04)
        time = exposure_clock(held, eligible, shot, fps)
        rows = np.flatnonzero(time != frames)
        if not len(rows):
            continue
        lo, hi = np.floor(time[rows]).astype(int), np.ceil(time[rows]).astype(int)
        columns = L.LEG_COLS[name]
        phase = (time[rows]-lo)[:,None]
        trial = pose[rows].copy()
        start_pose, end_pose = pose[np.ix_(lo,columns)], pose[np.ix_(hi,columns)]
        trial[:,columns] = (1.-phase)*start_pose + phase*end_pose
        gap = clearance(soles(trial)[side]).min(1)
        ok = (gap >= np.maximum(.015, np.minimum(lowest[lo,side],lowest[hi,side])-.015))
        ok &= gap <= np.maximum(lowest[lo,side],lowest[hi,side])+.015
        ok &= np.max(np.abs(start_pose-end_pose), axis=1) <= np.pi
        for stamp in np.unique(time[rows]):
            group = time[rows] == stamp
            if not ok[group].all():
                ok[group] = False
        out[np.ix_(rows[ok],columns)] = trial[ok][:,columns]
    return out
