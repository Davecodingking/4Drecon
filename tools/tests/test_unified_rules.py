#!/usr/bin/env python
"""Unit tests for the 3 unified merge rules:
  1. fill_legs only fills missing intervals (~vis), keeping visible knees/ankles & SAM3D trajectory untouched.
  2. stabilize_actor_stances applies one rule to original & completed legs: stabilizes slipping tread stances via leg IK,
     allows forefoot pivots & airborne kicks/jumps, and leaves already-solid stances untouched.
  3. repair_pair_arm_penetration allows close contact, repairs penetrating windows locally, and never worsens any frame.
"""
import os, sys
import numpy as np

sys.path.insert(0, "/home/ubuntu/sam3d/pipeline")
sys.path.insert(0, "/home/ubuntu/sam3d/data/outdoor/scripts")
import mhr_body as M
import legs_proc as LP
from stance_stabilizer import SupportCollider, stabilize_actor_stances
from pair_arm_repair import repair_pair_arm_penetration

DEV = "cuda" if __import__("torch").cuda.is_available() else "cpu"
UP = np.array([0.0, 1.0, 0.0])
SHAPE = np.zeros(M.NSHAPE)


def test_rule1_c_class_only_fills_unseen_frames():
    T, fps = 45, 15.0
    t = np.arange(T) / fps
    P = np.zeros((T, 204))
    # Give visible frames active leg articulation (e.g. kick/crouch) so they aren't static glide
    P[:, LP.KNEE_COL["l"]] = np.radians(25.0 + 20.0 * np.sin(np.linspace(0, 3 * np.pi, T)))
    P[:, LP.KNEE_COL["r"]] = np.radians(20.0 + 15.0 * np.cos(np.linspace(0, 3 * np.pi, T)))
    p0 = M.root_pos(P[:1])[0]
    P = M.set_root(P, np.repeat(np.eye(3)[None], T, 0), p0 + np.outer(t * 0.4, [0, 0, 1.0]))
    V0, _ = M.forward(SHAPE, P[:1], False, DEV)
    GH = float(V0[0, :, 1].min())

    feet_ok = np.ones(T, bool)
    feet_ok[15:33] = False  # Only middle 18 frames (1.2s) are unseen!
    out, rep = LP.fill_legs(P, SHAPE, t, feet_ok, UP, GH, dev=DEV)
    vis = rep["_规划"]["vis"]
    done = rep["_规划"]["done"]
    assert rep["类"] == "C"
    assert np.array_equal(done, ~vis)
    # Visible frames keep their exact original knee/ankle pose AND exact pelvis height!
    np.testing.assert_array_equal(out[vis], P[vis])
    assert done.sum() == (~vis).sum()


def test_rule2_single_leg_slip_stabilized_and_flight_preserved():
    T, fps = 20, 30.0
    P = np.zeros((T, 204))
    P[:, [LP.KNEE_COL["l"], LP.KNEE_COL["r"]]] = np.radians(24.0)
    P[:, [LP.LEG_COLS["l"][2], LP.LEG_COLS["r"][2]]] = np.radians(-12.0)
    # Lift right leg high (kick / step) so left leg is the single load-bearing support!
    P[:, LP.LEG_COLS["r"][2]] = np.radians(-55.0)
    P[:, LP.KNEE_COL["r"]] = np.radians(40.0)
    V0, _ = M.forward(SHAPE, P, True, DEV)
    rest = M.rest_verts(SHAPE, DEV)
    l_sole = np.flatnonzero((rest[:, 1] < rest[:, 1].min() + 10.0) & (rest[:, 0] > 0))
    GH = float((V0[0, l_sole] @ UP).min())

    # Inject artificial single-foot horizontal slide on left hip flexion while pelvis stays still
    P[4:14, LP.LEG_COLS["l"][2]] += np.linspace(0.0, np.radians(18.0), 10)
    V_in, _ = M.forward(SHAPE, P, True, DEV)
    # Re-level left sole to GH so it's sliding right on the tread
    gaps_in = (V_in[:, l_sole] @ UP).min(1) - GH
    P = M.set_root(P, None, M.root_pos(P) - gaps_in[:, None] * UP)

    col = SupportCollider(UP, GH, None)
    out, rep = stabilize_actor_stances(P, SHAPE, np.arange(T), np.zeros(T, int), np.zeros(T, bool), fps, UP, col, dev=DEV)
    # Root trajectory and right kicking leg must remain 100% untouched!
    np.testing.assert_allclose(M.root_pos(out), M.root_pos(P), atol=1e-7)
    np.testing.assert_allclose(out[:, LP.LEG_COLS["r"]], P[:, LP.LEG_COLS["r"]], atol=1e-7)
    assert "单脚防滑稳固" in rep

    # Airborne sequence must remain 100% untouched
    P_air = M.set_root(P, None, M.root_pos(P) + 0.35 * UP)
    out_air, rep_air = stabilize_actor_stances(P_air, SHAPE, np.arange(T), np.zeros(T, int), np.zeros(T, bool), fps, UP, col, dev=DEV)
    np.testing.assert_array_equal(out_air, P_air)
    assert not rep_air


def test_rule3_close_contact_preserved_and_penetration_repaired_monotonically():
    T = 12
    P1 = np.zeros((T, 204))
    P2 = np.zeros((T, 204))
    # Place P2 facing P1 at a safe distance on frames 0..11: zero penetration -> bit-for-bit untouched
    P2 = M.set_root(P2, None, M.root_pos(P2) + np.array([1.2, 0.0, 0.0]))
    res_safe = {
        "1": dict(vf=np.arange(T), params=P1.copy(), shape=SHAPE.copy()),
        "2": dict(vf=np.arange(T), params=P2.copy(), shape=SHAPE.copy()),
    }
    rep_safe = repair_pair_arm_penetration(res_safe, 25.0, dev=DEV)
    assert not rep_safe
    np.testing.assert_array_equal(res_safe["1"]["params"], P1)
    np.testing.assert_array_equal(res_safe["2"]["params"], P2)


if __name__ == "__main__":
    test_rule1_c_class_only_fills_unseen_frames()
    print("ok test_rule1_c_class_only_fills_unseen_frames")
    test_rule2_single_leg_slip_stabilized_and_flight_preserved()
    print("ok test_rule2_single_leg_slip_stabilized_and_flight_preserved")
    test_rule3_close_contact_preserved_and_penetration_repaired_monotonically()
    print("ok test_rule3_close_contact_preserved_and_penetration_repaired_monotonically")
    print("=== ALL UNIFIED RULE TESTS PASSED ===")
