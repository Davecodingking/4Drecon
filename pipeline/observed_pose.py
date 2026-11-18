"""Repeated source exposures and their pose constraints."""
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def _read_actor(frame_path, mask_path, person_id):
    try:
        with Image.open(frame_path) as frame, Image.open(mask_path) as mask:
            if frame.size != mask.size:
                return None
            labels = np.asarray(mask)
            if labels.ndim != 2:
                return None
            width, height = frame.size
            scale = min(1., 512 / max(width, height))
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            rgb = cv2.resize(np.asarray(frame.convert("RGB")), size, interpolation=cv2.INTER_AREA)
            actor = cv2.resize((labels == person_id).astype(np.uint8), size,
                               interpolation=cv2.INTER_NEAREST).astype(bool)
    except (OSError, ValueError):
        return None
    if actor.sum() < 64:
        return None
    # A featureless or tiny patch provides little evidence of a repeated exposure.
    appearance = rgb[actor].mean(axis=1)
    if np.percentile(appearance, 90) - np.percentile(appearance, 10) < 12:
        return None
    return rgb, actor


def _same_exposure(first, second):
    if first is None or second is None:
        return False
    rgb_a, mask_a = first
    rgb_b, mask_b = second
    if rgb_a.shape != rgb_b.shape:
        return False
    region = mask_a | mask_b
    if (mask_a & mask_b).sum() < .85 * region.sum():
        return False
    difference = cv2.absdiff(rgb_a, rgb_b).mean(axis=2, dtype=np.float32)
    if difference[region].mean() > 1.:
        return False
    # A small moving hand may be invisible in a whole-person average.
    return cv2.blur(difference, (5, 5))[region].max() < 16.


def source_hold_edges(frame_paths, masks, person_id, shot):
    """True at i when frames i and i+1 repeat this actor's exposure within a shot."""
    edges = np.zeros(max(0, len(shot) - 1), dtype=bool)
    if frame_paths is None or masks is None or person_id is None or not len(edges):
        return edges
    if len(frame_paths) != len(shot) or len(masks) != len(shot):
        raise ValueError("One source frame and actor mask are required per pose")
    previous = None
    anchor = None
    start = 0
    consistent = True
    for i, (frame, mask) in enumerate(zip(frame_paths, masks)):
        current = _read_actor(frame, mask, person_id)
        if i and shot[i] == shot[i - 1] and _same_exposure(previous, current):
            # Bound the total change from the first exposure. A slow continuous
            # gesture can make every individual pair look almost identical.
            if consistent and start < i - 1:
                consistent = _same_exposure(anchor, current)
        else:
            if i and consistent:
                edges[start:i-1] = True
            start, anchor, consistent = i, current, True
        previous = current
    if consistent:
        edges[start:] = True
    return edges


def held_samples(edges, boundaries=()):
    """Yield a held sample span and its representative samples in the same shot."""
    if edges is None:
        return
    changes = np.flatnonzero(np.diff(np.r_[False, edges, False]))
    cuts = set(boundaries)
    for start, stop in changes.reshape(-1, 2):
        span = slice(start, stop + 1)
        # At an edit the fitted samples nearest the boundary include pose from
        # the adjacent shot. Take the within-shot end of this held exposure.
        at_start, at_end = start in cuts, stop + 1 in cuts
        if at_start and not at_end:
            reference = slice(stop, stop + 1)
        elif at_end and not at_start:
            reference = slice(start, start + 1)
        else:
            reference = span
        yield span, reference


def apply_upper_pose_holds(params, groups, hold_edges, boundaries):
    """Share only upper-body articulation across held exposures, in place."""
    if not np.any(hold_edges):
        return False
    columns = np.concatenate([cols for name, cols in groups.items() if not name.endswith("upleg")])
    for span, reference in held_samples(hold_edges, boundaries):
        params[span, columns] = params[reference, columns].mean(axis=0)
    return True


def hold_tracked_upper_pose(params, groups, frame_paths, masks, person_id, source_frames, segments):
    """Apply reliable holds on consecutive samples in the same track segment."""
    adjacent = (np.diff(source_frames) == 1) & (np.diff(segments) == 0)
    if not np.any(adjacent):
        return False
    boundaries = np.flatnonzero(~adjacent) + 1
    parts = np.r_[0, np.cumsum(~adjacent)]
    hold_edges = source_hold_edges(frame_paths, masks, person_id, parts)
    return apply_upper_pose_holds(params, groups, hold_edges, boundaries)


def hold_rotations(rotation, hold_edges, boundaries=()):
    """Share a root orientation across each held exposure, including the +/-pi seam."""
    if not np.any(hold_edges):
        return rotation
    result = Rotation.from_quat(rotation.as_quat())
    for span, reference in held_samples(hold_edges, boundaries):
        result[span] = rotation[reference].mean()
    return result
