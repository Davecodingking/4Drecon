#!/usr/bin/env python
"""叠加检查图从库里的网格现画(pipeline/mesh_overlay.py): 把盘上的文件换回上游画图函数要的格式。
起因(2026-10-12): 以前叠加图是出网格时顺手画的, 按人拆开后是几份的图叠起来的 —— 点掉一个人, 图里没法只去掉他。
现在每次从库里读网格现画, 库里有谁图里就有谁。跑: $PY tools/tests/test_mesh_overlay.py"""
import json, os, sys, tempfile
import numpy as np
import trimesh
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import mesh_overlay as O

FACES = np.array([[0, 1, 2], [0, 2, 3]])
RNG = np.random.default_rng(0)


def _save_like_upstream(run, pid, stem, verts, cam_t, focal=1773.3):
    """照上游 save_mesh_results 的写法存一帧: Renderer.vertices_to_trimesh = (顶点 + 相机平移) 再绕 x 轴转 180°。"""
    os.makedirs(f"{run}/mesh_4d_individual/{pid}", exist_ok=True); os.makedirs(f"{run}/focal_4d_individual/{pid}", exist_ok=True)
    m = trimesh.Trimesh(verts.copy() + cam_t, FACES.copy(), process=False)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))
    m.export(f"{run}/mesh_4d_individual/{pid}/{stem}.ply")
    json.dump({"focal_length": focal, "camera": [float(x) for x in cam_t]}, open(f"{run}/focal_4d_individual/{pid}/{stem}.json", "w"))


def _run():
    """8 号两帧都在, 3 号只在第 1 帧。目录名是跟踪编号。"""
    run = tempfile.mkdtemp()
    data = {}
    for pid, stems in ((8, ("00000000", "00000001")), (3, ("00000001",))):
        for stem in stems:
            v = RNG.normal(size=(4, 3)).astype(np.float32)
            t = np.array([0.3 * pid, 1.5, 4.0 + pid], np.float32)
            _save_like_upstream(run, pid, stem, v, t)
            data[pid, stem] = (v, t)
    return run, data


def test_what_upstream_saved_is_turned_back_into_what_its_renderer_wants():
    run, data = _run()
    outs, ids, faces = O.frame_people(run, "00000001", {3: "3", 8: "8"})
    assert ids == [3, 8], ids                                          # 按编号排: 颜色跟编号走, 每次画都一样
    for out, pid in zip(outs, ids):
        v, t = data[pid, "00000001"]
        assert np.abs(out["pred_vertices"] - v).max() < 1e-5, np.abs(out["pred_vertices"] - v).max()
        assert np.abs(out["pred_cam_t"] - t).max() < 1e-6 and abs(float(out["focal_length"]) - 1773.3) < 1e-3
        assert out["pred_vertices"].dtype == np.float32
    assert (faces == FACES).all()


def test_only_people_in_this_frame_are_drawn():
    run, _ = _run()
    outs, ids, _ = O.frame_people(run, "00000000", {3: "3", 8: "8"})
    assert ids == [8] and len(outs) == 1
    assert O.frame_people(run, "00000007", {3: "3", 8: "8"}) == ([], [], None)       # 这一帧谁都不在: 什么都不画


def test_old_dirs_are_found_through_the_relabel_table():
    """10-12 之前出的目录: 跟踪的 8 号放在目录 2。图里的编号(颜色)仍然按跟踪编号。"""
    run, data = _run()
    os.rename(f"{run}/mesh_4d_individual/8", f"{run}/mesh_4d_individual/2")
    os.rename(f"{run}/focal_4d_individual/8", f"{run}/focal_4d_individual/2")
    outs, ids, _ = O.frame_people(run, "00000000", {8: "2"})
    assert ids == [8] and np.abs(outs[0]["pred_vertices"] - data[8, "00000000"][0]).max() < 1e-5


def test_blend_only_touches_pixels_where_a_mesh_was_drawn():
    img = np.full((4, 6, 3), 100, np.uint8)
    ren = np.full((4, 6, 3), 255, np.float32); ren[1:3, 2:4] = (200, 50, 0)       # 白底上一块网格
    out = O.blend(img, ren, 0.6)
    assert out.dtype == np.uint8 and (out[0] == 100).all() and (out[:, 0] == 100).all(), "白底的地方原片不该变"
    assert tuple(out[1, 2]) == (160, 70, 40), out[1, 2]                # 100*0.4 + (200, 50, 0)*0.6
    assert O.blend(img, None, 0.6) is img                              # 这一帧没人: 原片原样


def test_color_slot_stays_inside_the_palette():
    """上游按 color_list[编号 + 4] 取色, 表里只有 81 种: 编号大了要绕回来, 不能越界。"""
    assert O.color_slot(1, 81) == 1 and O.color_slot(7, 81) == 7 and O.color_slot(76, 81) == 76
    assert [O.color_slot(i, 81) + 4 < 81 and O.color_slot(i, 81) >= 1 for i in range(1, 400)] == [True] * 399
    assert O.color_slot(77, 81) == 1


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
    print("== test_mesh_overlay 全过 ==")
