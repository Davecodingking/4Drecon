"""proclegs 收尾规划: 看不见的段开头, 另一只脚正迈着一步(这步在段开头之前就离地) —— 2026-11-12 以前 StopIteration,
整个摆人那步崩掉。数据是 jw4 r4_30fps(人体全帧)人 5 那一段喂给 proclegs.run 的真实输入。"""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import legs_proc as LP


def test_mid_swing_at_start_of_unseen_run_plans():
    PL = LP._proclegs()
    d = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "proclegs_midswing_seg.pkl"), "rb"))
    body = PL.Body.__new__(PL.Body); body.__dict__.update(d["body"])

    class _Dummy:
        faces = np.zeros((1, 3), np.int32)

    def _fake_bj(model, P):
        n = len(P["transl"]); return np.zeros((n, 25, 3)), np.zeros((n, 1, 3))
    oldB, oldbj = PL.Body, PL.body_joints
    PL.Body, PL.body_joints = (lambda model, betas: body), _fake_bj
    try:
        res = PL.run(d["s"], d["t"], model=_Dummy(), log=lambda *a: None, obs=d["obs"], join=d["join"])
    finally:
        PL.Body, PL.body_joints = oldB, oldbj
    assert len(res["steps"]) > 0
    assert np.isfinite(res["feet"]).all()


if __name__ == "__main__":
    test_mid_swing_at_start_of_unseen_run_plans(); print("ok test_mid_swing_at_start_of_unseen_run_plans")
    print("== test_proclegs_midswing 全过 ==")
