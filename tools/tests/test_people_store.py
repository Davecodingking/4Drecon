#!/usr/bin/env python
"""pipeline/people_store.py: 摆人那步的结果怎么存、怎么读。网页上的人(网格)和 FBX 里的人(参数)都从这里读,
两边必须是同一个人 —— 包括"椅子被物理挪了, 人跟着挪"那一下。
要 MHR 模型文件在盘上。跑: $PY tools/tests/test_people_store.py"""
import json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import mhr_body as M
import people_store as S

SRC = dict(来源="04_mesh_occ0", 来源帧数={"1": 4, "2": 4}, 编号对照={"1": 3, "2": 7})


def _save(T, seed=0):
    """两个人(人号 3 和 7), 各 4 帧: 参数随便给, 网格就是参数喂给 MHR 的结果 —— 和摆人那步落盘时一样。"""
    rng = np.random.default_rng(seed)
    res, verts = {}, {}
    for pid in ("3", "7"):
        P = np.zeros((4, 204)); P[:, 6:136] = rng.normal(0, 0.1, (4, 130))
        P = M.set_root(P, None, rng.normal(0, 1.0, (4, 3)) + [0, 1, 4])
        shape = rng.normal(0, 0.5, 45)
        res[pid] = dict(frames=np.arange(4), vf=np.array([0, 3, 7, 10]), params=P, shape=shape, conf=np.ones(4))
        verts[pid] = M.forward(shape, P, True)[0]
    faces = M.model().character_torch.mesh.faces.cpu().numpy()
    meta = dict(floor_y=0.0, pinned_chairs=[], people={p: dict(seated=False, chair=None, footprints=[]) for p in res}, 人体来源=SRC)
    sid = S.save(T, meta, res, verts, faces, {p: np.zeros((2, 3), np.int32) for p in res}, 0.05, 24.0)
    return sid, res, verts


def _fails(fn, *words):
    try:
        fn()
    except SystemExit as e:
        assert all(w in str(e) for w in words), e
        return
    assert False, "应该报错"


def test_what_is_saved_is_what_is_read():
    with tempfile.TemporaryDirectory() as T:
        sid, res, verts = _save(T)
        S.save_shift(T, sid, {})
        world, faces = S.load_world(T)
        people, fps = S.load_params(T)
        assert list(world) == list(people) == ["3", "7"] and fps == 24.0 and faces.shape[1] == 3
        for pid in res:
            assert sorted(world[pid]) == [0, 3, 7, 10]
            assert np.array_equal(people[pid]["params"], res[pid]["params"])           # 参数逐位一样(导出读的就是它)
            assert np.abs(np.stack([world[pid][v] for v in (0, 3, 7, 10)]) - verts[pid]).max() < 1e-6
        assert S.meta(T)["solve_id"] == sid and sorted(S.FILES) == sorted(os.listdir(T))


def test_chair_shift_moves_the_mesh_and_the_parameters_by_the_same_amount():
    """椅子被物理挪了 12 cm: 读出来的网格 == 读出来的参数喂给 MHR 的网格。没被挪的人两边都不动。
    (三个现成的场景里一次位移都没有, 这条只能靠造出来的位移测。)"""
    with tempfile.TemporaryDirectory() as T:
        sid, res, verts = _save(T)
        d = np.array([0.1, 0.0, -0.066])
        S.save_shift(T, sid, {"7": d})
        world, _ = S.load_world(T)
        people, _ = S.load_params(T)
        for pid in ("3", "7"):
            V = M.forward(people[pid]["shape"], people[pid]["params"], True)[0]
            W = np.stack([world[pid][v] for v in (0, 3, 7, 10)])
            assert np.abs(V - W).max() < 2e-5, (pid, np.abs(V - W).max())               # 两边差不到 0.02 mm
            assert np.abs(W - verts[pid] - (d if pid == "7" else 0)).max() < 1e-6       # 挪的就是 d
        raw, _ = S.load_world(T, shifted=False)                                          # 合成自己量位移时读不带位移的
        assert np.abs(np.stack([raw["7"][v] for v in (0, 3, 7, 10)]) - verts["7"]).max() < 1e-6


def test_rerunning_compose_does_not_add_the_shift_twice():
    with tempfile.TemporaryDirectory() as T:
        sid, res, verts = _save(T)
        for _ in range(3):
            S.save_shift(T, sid, {"3": [0.2, 0, 0]})
        W = np.stack(list(S.load_world(T)[0]["3"].values()))
        assert np.abs(W - verts["3"] - [0.2, 0, 0]).max() < 1e-6


def test_shift_from_the_previous_solve_is_refused():
    """摆人重跑了、合成还没重跑: 上一次的位移不能套到这一次的人身上。位移文件不在也一样 —— 都说清楚该重跑哪一步。"""
    with tempfile.TemporaryDirectory() as T:
        sid, _, _ = _save(T)
        _fails(lambda: S.load_world(T), "people_glb")                    # 合成还没跑过
        S.save_shift(T, sid, {"3": [1, 0, 0]})
        json.dump(dict(solve_id="别的一次", shift={"3": [1, 0, 0]}), open(f"{T}/{S.SHIFT}", "w"))
        _fails(lambda: S.load_world(T), "people_glb")
        _fails(lambda: S.load_params(T), "people_glb")
        sid2, _, _ = _save(T, seed=1)                                    # 摆人重跑: 上一次的位移文件跟着删掉
        assert sid2 != sid and not os.path.exists(f"{T}/{S.SHIFT}")


def test_old_runs_are_told_to_rerun_people():
    """10-09 之前的 10_people.json(修在网格上, 没存修好的参数): 不悄悄用, 直说重跑 people。没有文件也一样。"""
    with tempfile.TemporaryDirectory() as T:
        _fails(lambda: S.load_world(T), "people(摆人)")
        json.dump(dict(floor_y=0, people={"1": dict(ratio=1.0, delta={}, frames=[])}), open(f"{T}/{S.META}", "w"))
        _fails(lambda: S.load_world(T), "2026-10-09 之前", "people(摆人)")
        _fails(lambda: S.load_params(T), "people(摆人)")


def test_stale_against_a_remeshed_body_run():
    """人体网格重出过(帧数变了, 或者挑的人变了 -> 编号对照变了): 摆人的结果过期。"""
    with tempfile.TemporaryDirectory() as T:
        _save(T)
        assert S.check_fresh(T, SRC)["人体来源"] == SRC
        _fails(lambda: S.check_fresh(T, dict(SRC, 来源帧数={"1": 4, "2": 5})), "people(摆人)")
        _fails(lambda: S.check_fresh(T, dict(SRC, 编号对照={"1": 3, "2": 8})), "people(摆人)")    # 帧数一样, 但 2 号换了人


def test_half_written_files_are_refused():
    with tempfile.TemporaryDirectory() as T:
        sid, _, _ = _save(T)
        S.save_shift(T, sid, {})
        m = json.load(open(f"{T}/{S.META}")); m["solve_id"] = "另一次"
        json.dump(m, open(f"{T}/{S.META}", "w"))
        S.save_shift(T, "另一次", {})
        _fails(lambda: S.load_world(T), "不是同一次", "people(摆人)")


def test_no_people_clears_what_each_step_left_behind():
    """闸口上把人全点掉: people / people_glb / motion 跳过之前各自清掉上一次的产物; 验收视频挪成 _prev 不删。"""
    with tempfile.TemporaryDirectory() as T:
        sid, _, _ = _save(T)
        S.save_shift(T, sid, {})
        for f in ("10_box_fit_view.json", "09_scene_nopeople.glb", "09_body4d_report.json", "09_people_f0003.glb",
                  "09_render_4d_f0003.jpg", "09_people_check.mp4", "07_scene_phy.glb"):
            open(f"{T}/{f}", "w").write("x")
        os.makedirs(f"{T}/12_motion"); open(f"{T}/12_motion/a.fbx", "w").write("x")
        assert sorted(S.clear_step(T, "people")) == sorted(list(S.FILES) + ["10_box_fit_view.json"])
        assert S.clear_step(T, "people") == []                                           # 本来就没有: 什么都不动
        S.clear_step(T, "people_glb"); S.clear_step(T, "motion")
        assert sorted(os.listdir(T)) == ["07_scene_phy.glb", "09_people_check_prev.mp4"], os.listdir(T)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f(); print("ok", name)
