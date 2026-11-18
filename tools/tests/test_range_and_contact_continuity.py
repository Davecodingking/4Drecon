"""Metric depth gaps and the handoff between a footprint and inferred swing IK."""
import sys
from pathlib import Path
import numpy as np

sam = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(sam / 'pipeline'), str(sam / 'proclegs')]
from range_continuity import bridge_range_fallbacks
from proclegs import constrain_swing_reach
from landing_motion import contact_handoff_weights


def range_fixture():
    frames = np.arange(44)
    shot = np.zeros(44, int)
    bad = (frames >= 18) & (frames < 23)
    scene = 5.0 + .015 * frames
    raw = scene + .2
    raw[bad] += .9
    return frames, shot, bad, raw, scene, ~bad


def test_range_bridge_keeps_measured_motion_and_existing_anchors():
    f, shot, bad, raw, scene, usable = range_fixture()
    fixed = bridge_range_fallbacks(f, shot, bad, raw, scene, usable, 30.)
    np.testing.assert_allclose(fixed, scene + .2)
    np.testing.assert_array_equal(fixed[~bad], raw[~bad])


def test_range_bridge_preserves_motion_supported_by_depth():
    f, shot, bad, raw, scene, usable = range_fixture()
    scene[bad] += .8
    np.testing.assert_array_equal(bridge_range_fallbacks(f, shot, bad, raw, scene, usable, 30.), raw)


def test_range_bridge_needs_continuity_and_observations_on_both_sides():
    f, shot, bad, raw, scene, usable = range_fixture()
    for mode in ('cut', 'gap', 'missing', 'open', 'long', 'stable'):
        fr, sh, bc, seen = f.copy(), shot.copy(), bad.copy(), usable.copy()
        if mode == 'cut': sh[21:] += 1
        if mode == 'gap': fr[21:] += 1
        if mode == 'missing': seen[:23] = False
        if mode == 'open': bc[:23] = True
        if mode == 'long': bc[10:35] = True
        if mode == 'stable': bc[:] = False
        np.testing.assert_array_equal(bridge_range_fallbacks(fr, sh, bc, raw, scene, seen, 30.), raw)


def test_swing_does_not_snap_to_reach_sphere_at_liftoff_or_landing():
    # Same ankle chord is reversed for landing, with a planted endpoint just beyond reach.
    for reverse in (False, True):
        n = 19
        u = np.linspace(0, 1, n)
        x = .72 - .58 * u**3 * (10 - 15*u + 6*u*u)
        if reverse: x = x[::-1]
        ankles = np.zeros((n, 2, 3))
        ankles[:, 0, 0] = x
        hips = np.zeros_like(ankles)
        hips[..., 2] = .94
        contact = np.ones((n, 2), bool)
        contact[1:-1, 0] = False
        original = ankles.copy()
        constrain_swing_reach(ankles, hips, np.array([1., 1.]), contact, np.zeros(n, bool), [dict(i=0, k0=0, k1=n-1)])
        steps = np.abs(np.diff(ankles[:, 0, 0]))
        assert steps.max() < .095
        assert steps[-1 if reverse else 0] < .025
        np.testing.assert_array_equal(ankles[contact], original[contact])
        np.testing.assert_array_equal(ankles[..., 2], original[..., 2])


def test_swing_preserves_visible_samples_and_ordinary_reach_limit():
    n=11
    ankles = np.zeros((n, 2, 3))
    ankles[1:-1, 0, 0] = .8
    hips = np.zeros_like(ankles)
    hips[..., 2] = .8
    c = np.ones((n, 2), bool)
    c[1:-1, 0] = False
    visible = np.zeros(n, bool)
    visible[3:5] = True
    constrain_swing_reach(ankles, hips, np.ones(2), c, visible, [dict(i=0, k0=0, k1=n-1)])
    np.testing.assert_allclose(ankles[1:3,0,0], .6)
    np.testing.assert_allclose(ankles[3:5,0,0], .8)




def test_sparse_shot_and_contradictory_inner_depth_keep_original():
    f, shot, bad, raw, scene, usable = range_fixture()
    dense = f < 4
    np.testing.assert_array_equal(bridge_range_fallbacks(f, shot, bad, raw, scene, usable, 30., dense=dense), raw)
    samples = np.full(len(f), np.nan)
    samples[20] = raw[20] + .03
    np.testing.assert_array_equal(bridge_range_fallbacks(f, shot, bad, raw, scene, usable, 30., depth_samples=samples), raw)


def test_trimmed_inferred_swing_does_not_treat_last_view_as_a_touchdown():
    n = 8
    ankles = np.zeros((n, 2, 3))
    ankles[:, 0, 0] = .7
    hips = np.zeros_like(ankles)
    hips[..., 2] = .94
    contact = np.ones((n, 2), bool)
    contact[2:, 0] = False
    constrain_swing_reach(ankles, hips, np.ones(2), contact, np.zeros(n, bool), [dict(i=0, k0=1, k1=15)])
    assert np.isfinite(ankles).all()
    assert .4 < ankles[-1, 0, 0] < ankles[2, 0, 0]


def test_cropped_swing_uses_its_original_step_clock_and_skips_visible_ankles():
    from legs_proc import planned_swing_samples
    times = np.arange(8)/30
    contact = np.tile([False,True], (8,1))
    visible = np.zeros(8,bool)
    visible[2] = True
    phase = planned_swing_samples(times, contact, visible, [[0, 0., .5]])
    assert np.isnan(phase[0,0]) and np.isnan(phase[2,0]) and np.isnan(phase[:,1]).all()
    np.testing.assert_allclose(phase[3:,0],times[3:]/.5)
    assert phase[-1,0] < .5


def test_visible_crossing_in_a_mixed_turn_protects_the_joint_fit():
    from landing_motion import inferred_turn_frames
    active = np.array([0,1,1,1,1,0],bool)
    crossing = np.array([0,0,1,0,0,0],bool)
    assert not inferred_turn_frames(active,crossing).any()
    crossing = np.array([1,0,0,0,0,0],bool)
    np.testing.assert_array_equal(inferred_turn_frames(active,crossing),active)


def test_open_turn_requires_anatomical_evidence_and_allows_visible_crossing():
    from landing_motion import open_turn_evidence
    knee_seen = np.array([[1,1],[1,1],[0,0],[0,0]],bool)
    ankle_seen = np.array([[1,1],[0,0],[0,0],[0,0]],bool)
    hint = np.array([[1,1],[1,1],[0,0],[.8,.9]])
    hip = np.ones(4)*.15
    knee = np.ones(4)*.2
    ankle = np.array([-.12,.15,.1,.1])
    np.testing.assert_array_equal(open_turn_evidence(hip,knee,ankle,knee_seen,ankle_seen,hint),[False,True,False,True])


def test_footfall_handoff_is_local_to_an_inferred_transition():
    c = np.ones((40,2), bool)
    c[12:27,0] = False
    inferred = np.ones(40,bool)
    weights = contact_handoff_weights(c, inferred, 30)
    assert weights.shape == (39,2)
    np.testing.assert_allclose(weights[:,1], 0.)
    assert weights[11,0] == weights[26,0] == 1.
    assert weights[19,0] == 0.
    inferred[:15] = False
    assert not contact_handoff_weights(c, inferred, 30)[:18].any()




def test_repeated_swing_shares_only_unloaded_leg_and_protects_flight():
    from types import SimpleNamespace
    from unittest.mock import patch
    import fit_exposure as fit
    import legs_proc as legs
    n = 14
    columns = legs.LEG_COLS['l']
    right = legs.LEG_COLS['r']
    q = np.zeros((n, 204))
    q[:,columns[0]] = np.linspace(0., 1., n)
    q[:,right[0]] = np.linspace(.02,.04,n)
    rest = np.array([[1,0,0], [2,0,0], [3,0,0], [-1,0,0], [-2,0,0], [-3,0,0]])
    def vertices(shape, x, corrective, dev):
        floor = .25 if floating else 0.
        xyz = np.zeros((len(x),6,3))
        xyz[:,:3,0] = x[:,columns[0],None]
        xyz[:,:3,2] = .3
        xyz[:,3:,2] = floor
        return xyz, None
    held = np.zeros(n-1, bool)
    held[6] = True
    shots = np.zeros(n, int)
    support = np.tile([False,True], (n,1))
    floor = SimpleNamespace(model_h=lambda xyz: np.zeros(len(xyz)))
    for floating, flight in ((False,False), (True,True), (False,True)):
        with patch.object(fit.M,'rest_verts',return_value=rest), patch.object(fit.M,'forward',side_effect=vertices):
            out = fit.fit_unloaded_exposures(q, None, shots, 30, np.array([0,0,1]), floor, 'cpu',
                                             held, np.zeros(n,bool), np.ones(n,bool), support,
                                             np.full(n,flight), np.ones(n,bool))
        if flight:
            np.testing.assert_array_equal(out, q)
        else:
            assert out[6,columns[0]] == out[7,columns[0]]
            np.testing.assert_array_equal(out[:,np.setdiff1d(np.arange(q.shape[1]),columns)],
                                          q[:,np.setdiff1d(np.arange(q.shape[1]),columns)])
            np.testing.assert_array_equal(out[[0,-1]], q[[0,-1]])


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('ok', name)
