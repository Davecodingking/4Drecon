"""Observation-based stance and flight evidence for human motion on terrain."""
import numpy as np
from scipy.ndimage import median_filter
import legs_proc as L


def common_flight(heights, shot, fps):
    """A brief joint rise above both the approach and the landing is evidence of flight."""
    airborne = np.zeros(len(heights),bool)
    edges = np.r_[0,np.flatnonzero(np.diff(shot))+1,len(shot)]
    radius = max(3,round(.23*fps))
    for a,b in zip(edges[:-1],edges[1:]):
        for i in range(a+2,b-2):
            before = np.median(heights[max(a,i-radius):i],axis=0)
            after = np.median(heights[i+1:min(b,i+radius+1)],axis=0)
            airborne[i] = np.all(heights[i]-np.maximum(before,after) > .015)
    return airborne


def support_observations(raw_joints, up, raw_sole, gap, upright, visible_shin, raw_gap,
                         swing, available, shot, fps):
    """Infer a loaded pivot only during a sustained, observed one-leg gesture.

    gap comes from the scene. The flight veto uses independent, pre-grounding
    motion, so the absence of a long airborne interval is never positive
    evidence of contact. All distances below are metres.
    """
    height = raw_sole @ up
    hips = raw_joints[:,:,0].mean(1)
    hip_height = hips @ up
    shin = raw_joints[:,:,1]-raw_joints[:,:,2]
    vertical = (shin @ up)/np.maximum(.001,np.linalg.norm(shin,axis=-1))
    below_hip = hip_height[:,None]-height
    leg_length = np.linalg.norm(raw_joints[:,:,0]-raw_joints[:,:,1],axis=-1)+np.linalg.norm(shin,axis=-1)
    airborne_hint = common_flight(np.column_stack([hip_height,height]),shot,fps)
    loaded = np.zeros_like(gap,dtype=bool)
    edges = np.r_[0,np.flatnonzero(np.diff(shot))+1,len(shot)]
    for begin,end in zip(edges[:-1],edges[1:]):
        sl = slice(begin,end)
        # Reference the height before ground correction, across independently
        # visible single-leg gestures. A one-frame dip in a noisy depth track
        # must not become the take-off baseline of all subsequent frames.
        for side in range(2):
            other = 1-side
            asymmetry = height[sl,other]-height[sl,side]
            other_unloaded = ((asymmetry > .085) & (swing[sl,other] > .25)) | ((gap[sl,other] > .045) & (asymmetry > .045))
            stance = (available[sl] & visible_shin[sl,side] & (upright[sl] > .70)
                      & (vertical[sl,side] > .78) & (below_hip[sl,side] > .63*leg_length[sl,side])
                      & other_unloaded)
            for lo,hi in L.runs_of(stance):
                a, b = begin+lo, begin+hi
                acquire = max(3,round(.1*fps))
                if b-a < acquire:
                    continue
                start = None
                for j in range(a,b-acquire+1):
                    close = np.abs(gap[j:j+acquire,side]) < .035
                    steady = np.ptp(height[j:j+acquire,side]) < .035
                    approach = max(begin,j-round(.22*fps))
                    raw_near = np.abs(raw_gap[approach:j+acquire,side]) < .065
                    if close.all() and steady and raw_near.sum() >= 2:
                        start = j
                        break
                if start is None:
                    continue
                # A fixed physical tread has constant height. A moving footprint
                # may cross an edge in the noisy input and report the next tread.
                reference = np.median(height[start:start+acquire,side])
                # Confirm a footprint from several stable samples, then recover
                # the earlier part of the same continuous stance in this shot.
                # An isolated downward depth error cannot indicate take-off.
                start = a
                raw_h = height[start:b,side]
                over = raw_h-reference
                # Compare the actor's lowest foot with the approach and landing,
                # including very short arcs that the body's coarse air flag misses.
                flight = airborne_hint[start:b]
                noise = np.median(np.abs(height[a:b,side]-reference))
                lift_tolerance = np.clip(2.5*noise,.012,.03)
                can_hold = (over < lift_tolerance) & (median_filter(over, size=3,mode="nearest") > -.09) & ~flight
                for l,hx in L.runs_of(can_hold):
                    if hx-l >= acquire:
                        loaded[start+l:start+hx,side] = True
    return loaded


def mask_shin_visibility(mask_paths, person, coordinates, depth, image_wh):
    """A cropped ankle can still have a visible lower shin on this actor."""
    from PIL import Image
    valid = np.zeros(coordinates.shape[:-1],bool)
    if mask_paths is None or person is None:
        return valid
    for i,path in enumerate(mask_paths):
        if not np.isfinite(coordinates[i]).all():
            continue
        try:
            with Image.open(path) as img:
                labels = np.asarray(img)
        except FileNotFoundError:
            continue
        if labels.shape != tuple(image_wh[::-1]):
            continue
        for side,points in enumerate(coordinates[i]):
            if depth[i,side].min() < .05:
                continue
            x,y = np.rint(points).astype(int).T
            inside = (x>=0)&(x<image_wh[0])&(y>=0)&(y<image_wh[1])
            valid[i,side,inside] = labels[y[inside],x[inside]]==person
    return valid


