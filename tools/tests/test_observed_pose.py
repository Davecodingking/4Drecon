"""A repeated exposure is one articulated pose, independent of the source frame rate."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.append(str(Path(__file__).resolve().parents[2] / 'pipeline'))
import landing_motion as LM
from observed_pose import held_samples, hold_rotations, hold_tracked_upper_pose, source_hold_edges


def actor():
    y, x = np.indices((64, 96))
    gray = (30 + (x * 3 + y * 7) % 160).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def save_frames(folder, frames):
    filenames = []
    labels = []
    actor_mask = np.zeros(frames[0].shape[:2], np.uint8)
    actor_mask[4:-4, 8:-8] = 2
    for i, rgb in enumerate(frames):
        frame = Path(folder) / f'rgb{i}.png'
        mask = Path(folder) / f'mask{i}.png'
        Image.fromarray(rgb).save(frame)
        Image.fromarray(actor_mask).save(mask)
        filenames.append(frame)
        labels.append(mask)
    return filenames, labels


def test_source_detects_repeats_and_distinguishes_local_gestures():
    base = actor()
    active = base.copy()
    active[23:28, 50:55] += 55  # whole actor mean difference <1 intensity level
    near = active.copy()
    near[::10] += 2           # minor encoding differences within the person
    with TemporaryDirectory() as tmp:
        frames, masks = save_frames(tmp, [base, base, active, near, near])
        edges = source_hold_edges(frames, masks, 2, [0,0,0,0,1])
        np.testing.assert_array_equal(edges, [True,False,True,False])
        np.testing.assert_array_equal(source_hold_edges(frames, masks, 1, [0]*5), [False]*4)
        masks[2] = Path(tmp) / 'missing.png'
        np.testing.assert_array_equal(source_hold_edges(frames, masks, 2, [0]*5), [True,False,False,True])


def test_gradual_change_in_low_contrast_frames_does_not_form_a_hold():
    base = actor()
    with TemporaryDirectory() as tmp:
        frames, masks = save_frames(tmp, [base+i for i in range(7)])
        assert not source_hold_edges(frames, masks, 2, [0]*7).any()


def test_missing_or_uninformative_actor_cannot_confirm_a_hold():
    with TemporaryDirectory() as tmp:
        frames, masks = save_frames(tmp, [np.full((64,96,3), 80, np.uint8)]*2)
        assert not source_hold_edges(frames, masks, 2, [0,0]).any()
        assert not source_hold_edges(None, masks, 2, [0,0]).any()


def test_rotations_share_a_hold_across_the_angle_wrap():
    rotation = Rotation.from_euler('z', np.array([90, 178, -178, 180, 40])[:, None], degrees=True)
    held = np.array([False, True, True, False])
    result = hold_rotations(rotation, held)
    np.testing.assert_allclose(result[[0,4]].as_matrix(), rotation[[0,4]].as_matrix(), atol=1e-12)
    error = result[1:4].inv() * Rotation.from_euler('z', 180, degrees=True)
    np.testing.assert_allclose(error.magnitude(), 0., atol=1e-12)
    assert [(s.start,s.stop) for s, reference in held_samples(held)] == [(1,4)]


def test_cut_adjacent_holds_use_the_pose_further_inside_the_same_shot():
    edges = [False,True,False,True,False,True,False,False]
    # The cut is between frames 2 and 3. Both touching exposures have two
    # copies; frames 1 and 4 are the samples farther inside their shots.
    spans = [(s.start,s.stop,r.start,r.stop) for s,r in held_samples(edges,[3])]
    assert spans == [(1,3,1,2),(3,5,4,5),(5,7,5,7)]
    angle = np.arange(9.)[:,None] * 5
    rotation = Rotation.from_euler('z', angle, degrees=True)
    result = hold_rotations(rotation, edges, [3])
    np.testing.assert_allclose(result[2].as_matrix(),rotation[1].as_matrix(),atol=1e-12)
    np.testing.assert_allclose(result[3].as_matrix(),rotation[4].as_matrix(),atol=1e-12)


def test_smoothing_preserves_deliberate_arm_movement_and_lower_body_at_multiple_rates():
    groups = LM.M.limb_groups('cpu')
    arm = groups['l_clavicle']
    for fps in (15, 30, 60):
        n = fps * 2
        p = np.zeros((n, 136), dtype=np.float32)
        t = np.arange(n) / fps
        p[:,arm[0]] = np.clip((t-.35)*5, 0., 1.9) - np.clip((t-1.05)*4, 0., 1.8)
        i = round(.50*fps)
        p[i+1] = p[i]  # one repeated exposure in an otherwise deliberate movement
        p[:,52] = np.sin(t)
        edges = np.zeros(n-1,bool)
        edges[i] = True
        original = LM.smooth_pose_impulses(p, fps, np.array([],int), 'cpu')
        candidate = LM.smooth_pose_impulses(p, fps, np.array([],int), 'cpu', edges)
        np.testing.assert_allclose(candidate[i,arm], candidate[i+1,arm], atol=1e-6)
        np.testing.assert_array_equal(candidate[:,52], p[:,52])
        neighbors = np.arange(n)
        np.testing.assert_array_equal(candidate[(neighbors!=i)&(neighbors!=i+1)], original[(neighbors!=i)&(neighbors!=i+1)])
        assert np.ptp(candidate[:,arm[0]]) > 1.8
        np.testing.assert_allclose(candidate[i,arm], original[i:i+2,arm].mean(axis=0), atol=1e-6)


def test_generic_holds_do_not_change_root_lower_body_or_cross_missing_frames():
    base = actor()
    groups = LM.M.limb_groups('cpu')
    upper = np.concatenate([cols for name, cols in groups.items() if not name.endswith('upleg')])
    p = np.arange(8 * 204, dtype=np.float32).reshape(8, 204)
    original = p.copy()
    with TemporaryDirectory() as tmp:
        frames, masks = save_frames(tmp, [base]*8)
        # Rows 2/3 span a missing exposure. Rows 5/6 are a shot cut.
        vf = [0,1,2,5,6,7,8,9]
        segments = [0,0,0,0,0,0,1,1]
        assert hold_tracked_upper_pose(p, groups, frames, masks, 2, vf, segments)
        np.testing.assert_array_equal(p[:,np.setdiff1d(np.arange(p.shape[1]),upper)],
                                      original[:,np.setdiff1d(np.arange(p.shape[1]),upper)])
        np.testing.assert_array_equal(p[0,upper],p[2,upper])
        np.testing.assert_array_equal(p[3,upper],p[5,upper])
        np.testing.assert_array_equal(p[6,upper],p[7,upper])
        assert not np.array_equal(p[2,upper],p[3,upper])
        assert not np.array_equal(p[5,upper],p[6,upper])
        sparse = original.copy()
        assert not hold_tracked_upper_pose(sparse, groups, frames, masks, 2, np.arange(8)*3, [0]*8)
        np.testing.assert_array_equal(sparse, original)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
