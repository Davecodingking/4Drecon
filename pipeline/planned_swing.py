"""Evidence that a planned stance foot is visibly raised above its tread."""
import numpy as np
from observed_support import common_flight


def _runs(mask):
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return changes.reshape(-1, 2)



def _contact_runs(contact, shot, join=-1):
    cuts = np.flatnonzero(np.diff(shot))+1
    edges = np.r_[0,cuts[cuts!=join],len(shot)]
    for start,stop in zip(edges[:-1],edges[1:]):
        yield from _runs(contact[start:stop]) + start


def _feasible_stance(raw, sole, hips, lengths, ankle_height, ground_height, up):
    """Can one stationary flat foot fit the calibrated hip and observed knee?"""
    foot = hips + sole - raw[:, 0]
    indices = np.linspace(0, len(foot)-1, min(24, len(foot))).round().astype(int)
    candidates = np.vstack((foot[indices], np.median(foot, axis=0)))
    ground = np.asarray(ground_height(candidates))
    ankle = candidates + (ground + ankle_height - candidates @ up)[:, None] * up
    reach = np.linalg.norm(hips[:, None] - ankle, axis=-1)
    knee = hips + raw[:, 1] - raw[:, 0]
    shin = np.linalg.norm(knee[:, None] - ankle, axis=-1)
    plausible = np.percentile(np.abs(shin-lengths[1]), 75, axis=0) < .10*sum(lengths)
    return np.any((np.max(reach, axis=0) <= sum(lengths)) & plausible)


def contradictory_swing(raw_joints, raw_sole, raw_gap, upright, joint_seen, thigh_confidence,
                        world_hips, bone_lengths, ankle_heights, ground_height, planned,
                        contact, shot, fps, up):
    """Return (time, side) mask for an observed airborne leg assigned a stance.

    raw_gap and ground_height are measured against the scene in metres. The
    support side may have a cropped ankle, but its knee must be visible. The
    fixed-foot reach test returns feasibility, not a measured foot position.
    """
    X, sole = np.asarray(raw_joints), np.asarray(raw_sole)
    available = np.asarray(planned, bool)
    result = np.zeros_like(contact, dtype=bool)
    if thigh_confidence is None or not np.any(available):
        return result
    hip, knee, ankle = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    pelvis = hip.mean(1)
    femur = np.maximum(np.linalg.norm(hip-knee, axis=-1), 1.e-6)
    length = femur + np.linalg.norm(ankle-knee, axis=-1)
    knee_down = ((hip-knee) @ up) / femur
    drop = ((pelvis[:, None]-sole) @ up) / np.maximum(length, 1.e-6)
    height = sole @ up
    flight = common_flight(np.column_stack([pelvis @ up, height]), shot, fps)
    trusted = (available & ~flight & (upright > .75) & joint_seen[:, :, 0].all(1)
               & (np.min(thigh_confidence, axis=1) > .45))
    boundaries = np.r_[0, np.flatnonzero(np.diff(shot))+1, len(X)]
    acquire = max(2, int(np.ceil(.10*fps)))
    max_hole = max(0, int(np.floor(.045*fps)))
    for raised in (0,1):
        stance = 1-raised
        posture = (trusted & (knee_down[:, raised] < .60)
                   & (knee_down[:, stance]-knee_down[:, raised] > .25)
                   & (height[:, raised]-height[:, stance] > .14*length[:, stance])
                   & (raw_gap[:, raised] > .09*length[:, raised])
                   & (np.abs(raw_gap[:, stance]) < .11*length[:, stance])
                   & (drop[:, stance] > .75))
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            # Fill only a brief missing shin measurement inside sustained
            # geometric evidence. A landing, cut or flight ends the event.
            for lo, hi in _runs(posture[begin:end]) + begin:
                seen = np.asarray(joint_seen[lo:hi, raised, 1], bool).copy()
                for a, b in _runs(~seen):
                    if a and b < len(seen) and b-a <= max_hole:
                        seen[a:b] = True
                for a, b in _runs(seen) + lo:
                    if b-a < acquire or not np.any(contact[a:b, raised]):
                        continue
                    close = np.abs(raw_gap[a:b, stance]) < .075*length[a:b, stance]
                    if np.count_nonzero(close) < acquire:
                        continue
                    if _feasible_stance(X[a:b, stance], sole[a:b, stance], world_hips[a:b, stance],
                                        bone_lengths[stance], ankle_heights[stance], ground_height, up):
                        result[a:b, raised] = True
    return result

def recovered_contacts(contact, raised, planned, shot, flight, clearance, leg_length, fps):
    """Join an inferred support to the preceding plant across a short labeling gap."""
    result = np.array(contact, dtype=bool, copy=True)
    result[raised] = False
    result[raised[:, ::-1]] = True
    limit = int(np.floor(.075*fps))
    for side in (0, 1):
        for start, stop in _contact_runs(raised[:, 1-side],shot):
            first = start
            while (first > 0 and start-first < limit and not result[first-1, side]
                   and shot[first-1] == shot[start] and planned[first-1] and not flight[first-1]
                   and not raised[first-1, side] and abs(clearance[first-1,side]) < .10*sum(leg_length[side])):
                first -= 1
            if first > 0 and shot[first-1] == shot[start] and result[first-1,side]:
                result[first:start,side] = True
    return result

def affected_legs(original, corrected, raised, shot, fps, join=-1, registered_side=-1):
    """Limit a contact correction to both implicated footfalls and their swing transitions."""
    weights = np.zeros_like(raised,dtype=float)
    # A stance run selects the contributing poses. Separate camera pivots
    # are still estimated by stance_groups in the solver.
    pad = max(2,round(.22*fps))
    for side in (0,1):
        change = original[:,side] != corrected[:,side]
        core = raised.any(1).copy()
        for contacts in (original[:,side],corrected[:,side]):
            seam = join if side==registered_side else -1
            for start,stop in _contact_runs(contacts,shot,seam):
                if change[start:stop].any():
                    core[start:stop] = True
        for start,stop in _runs(core):
            weights[start:stop,side] = 1.
            for edge,direction in ((start,-1),(stop-1,1)):
                for distance in range(1,pad+1):
                    i = edge+direction*distance
                    if not 0<=i<len(shot) or shot[i]!=shot[edge]:
                        break
                    t = 1.-distance/(pad+1.)
                    weights[i,side] = max(weights[i,side],t*t*(3.-2.*t))
    return weights

def stance_transitions(contact, raised, shot, flight, fps, join=-1, registered_side=-1):
    """Weight free-foot edges next to the recovered plant within the same shot."""
    weights = np.zeros((max(0,len(contact)-1),2))
    radius = max(2,round(.20*fps))
    for side in (0,1):
        seam = join if side==registered_side else -1
        for start,stop in _contact_runs(contact[:,side],shot,seam):
            if not raised[start:stop,1-side].any():
                continue
            for edge,direction in ((start-1,-1),(stop-1,1)):
                for distance in range(radius):
                    i=edge+direction*distance
                    if not 0<=i<len(weights) or shot[i]!=shot[i+1] or flight[i:i+2].any():
                        break
                    if distance and contact[i:i+2,side].any():
                        break
                    weights[i,side]=max(weights[i,side],(1.-distance/radius)**2)
    return weights
