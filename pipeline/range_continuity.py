"""Continuity of metric actor range across a short loss of camera confidence."""
import numpy as np


def bridge_range_fallbacks(frames, shot, bad_camera, distance, scene_distance, measured, fps, dense=None, depth_samples=None):
    """Rejoin both measured ends of a brief fallback on one contiguous shot.

    The scene-depth path supplies relative motion. The good-camera estimates on
    either side still set its absolute range. Unbracketed samples keep their input.
    """
    result = np.array(distance, copy=True)
    if fps <= 0 or len(frames) < 3 or not np.any(bad_camera):
        return result
    bad_camera = np.asarray(bad_camera, bool)
    measured = np.asarray(measured, bool) & ~bad_camera & np.isfinite(scene_distance)
    splits = np.flatnonzero((np.diff(shot) != 0) | (np.diff(frames) != 1)) + 1
    edges = np.r_[0, splits, len(frames)]
    radius = max(1, round(.45 * fps))
    margin = max(1, round(.25 * fps))
    jump = max(.25, 4. / fps)
    for start, stop in zip(edges[:-1], edges[1:]):
        switches = np.flatnonzero(np.diff(np.r_[False, bad_camera[start:stop], False])) + start
        for lo, hi in switches.reshape(-1, 2):
            if dense is not None and not np.any(dense[max(start, lo-radius):min(stop, hi+radius)]):
                continue
            if min(lo-start, stop-hi) < margin or (hi-lo) / fps > .55:
                continue
            if not (measured[max(start, lo-radius):lo].any() and measured[hi:min(stop, hi+radius)].any()):
                continue
            ii = [lo-1, lo, hi-1, hi]
            if not (np.isfinite(distance[ii]).all() and np.isfinite(scene_distance[lo-1:hi+1]).all()):
                continue
            enter, leave = np.diff(distance[ii])[[0, 2]]
            amplitude = max(abs(enter), abs(leave))
            if enter * leave >= 0 or amplitude <= jump:
                continue
            scene_edges = np.diff(scene_distance[ii])[[0, 2]]
            if np.max(np.abs(scene_edges)) > max(.10, .30 * amplitude):
                continue
            ends = [lo-1, hi]
            bias = distance[ends] - scene_distance[ends]
            joined = scene_distance[lo:hi] + np.interp(frames[lo:hi], frames[ends], bias)
            if depth_samples is not None:
                evidence = depth_samples[lo:hi]
                inside = np.isfinite(evidence)
                if inside.any():
                    old_error = np.median(np.abs(distance[lo:hi][inside] - evidence[inside]))
                    new_error = np.median(np.abs(joined[inside] - evidence[inside]))
                    if new_error > old_error + .10:
                        continue
            result[lo:hi] = joined
    return result
