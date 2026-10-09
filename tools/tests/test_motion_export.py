#!/usr/bin/env python
"""人只修一遍(people 步) + 动作导出(motion 步)接在流水线里: 步骤表 / 分段表 / 谁读谁 / 没有人的轮次 / 没有人的场景文件。
跑: $PY tools/tests/test_motion_export.py"""
import importlib.util, json, os, subprocess, sys, tempfile
import numpy as np, trimesh
SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import body_into_scene as B
import motion_export as ME


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_step_is_in_both_pipelines_and_the_drive_table():
    """motion 排在合成(people_glb)后面, 依赖合成(场景用它出的 09_scene_nopeople.glb)和摆人(人的参数是它存的); 驱动的验收段里有它。"""
    out, ind = _load("_o", f"{SAM3D}/pipeline/outdoor_pipeline.py"), _load("_i", f"{SAM3D}/pipeline/indoor_pipeline.py")
    drive = _load("_d", f"{SAM3D}/.claude/skills/run-recon-pipeline/drive.py")
    for kind, steps, deps in (("outdoor", out.STEPS, out.DEPS), ("indoor", ind.STEPS_OBJ, ind.DEPS_OBJ)):
        keys = [s[0] for s in steps]
        assert keys.index("motion") == keys.index("people_glb") + 1, keys
        assert deps["motion"] == ["people_glb", "people"] and deps["people_glb"] == ["phys", "people"]
        assert "people" in deps["phys"]                                    # 物理拿人当空气墙: 摆人在它前面
        seg = [s for s in drive.SEGMENTS[kind] if s[2] == "check_scene"][0][1]
        assert seg.index("people") < seg.index("phys") < seg.index("people_glb") < seg.index("motion"), seg


def test_people_are_fixed_once_and_everything_downstream_reads_that():
    """人只在摆人那步(body_motion.py)修: 合成、验收视频、导出、3D 页都读它存下来的(people_store), 谁也不自己再修、再重放。"""
    src = {f: open(f"{SAM3D}/{f}").read() for f in ("pipeline/outdoor_pipeline.py", "pipeline/indoor_pipeline.py",
           "pipeline/body_into_scene.py", "pipeline/body_video.py", "pipeline/motion_export.py", "tools/box_fit_view.py",
           "tools/mocap/dump_people.py")}
    for f in ("pipeline/outdoor_pipeline.py", "pipeline/indoor_pipeline.py"):
        fn = "s_people" if "outdoor" in f else "o_people"
        body = src[f][src[f].index(f"def {fn}(c, dry):"):src[f].index(f"def {fn}_glb(c, dry):")]
        assert "body_motion.py" in body and "body_into_scene.py" not in body, f
    for f in ("pipeline/body_into_scene.py", "pipeline/body_video.py", "tools/box_fit_view.py", "tools/mocap/dump_people.py"):
        assert "people_store.load_world(" in src[f], f
    assert "people_store.load_params(" in src["pipeline/motion_export.py"]
    for f, txt in src.items():
        if f.startswith("pipeline/") and "pipeline.py" in f:
            continue
        code = "\n".join(l.split("#")[0] for l in txt.split('"""')[0::2] for l in l.split("\n"))    # 去掉文档串和注释再查
        for gone in ("apply_corr(", "solve_people(", "smooth_pose(", "fill_occluded(", "apply_seat_fix(", "BM.solve(", "body_motion.py"):
            assert gone not in code, (f, gone)


def test_no_people_never_waits_for_a_gpu_and_clears_the_old_outputs():
    """没有人的轮次: 摆人 / 导出进了步骤就跳过 —— 不在"先领卡"的表里(那是进步骤之前就排队); 真有人时才在步骤里领。
    跳过之前清掉上一次有人时的产物(见 people_store.clear_step)。"""
    out, ind = _load("_o2", f"{SAM3D}/pipeline/outdoor_pipeline.py"), _load("_i2", f"{SAM3D}/pipeline/indoor_pipeline.py")
    for m, pre in ((out, "s"), (ind, "o")):
        assert not {"people", "people_glb", "motion"} & set(m.GPU_STEPS)
        src = open(m.__file__).read()
        for step in ("people", "people_glb", "motion"):
            a = src.index(f"def {pre}_{step}(c, dry):"); body = src[a:src.index("\ndef ", a + 10)]
            assert f'return _no_people(c, dry, "{step}")' in body, (pre, step)
            assert ("_with_card(" in body) == (step != "people_glb"), (pre, step)

        class C:
            R1 = None; LOG = "/tmp"; cfg = dict(BODY_RUN="")
        with tempfile.TemporaryDirectory() as T:
            C.R1 = T
            for f in ("10_people.json", "10_people_world.npz", "09_scene_nopeople.glb", "09_people_check.mp4", "07_scene_phy.glb"):
                open(f"{T}/{f}", "w").write("x")
            os.makedirs(f"{T}/12_motion")
            for step in ("people", "people_glb", "motion"):
                getattr(m, f"{pre}_{step}")(C, True)                       # 干跑: 什么都不删
            assert len(os.listdir(T)) == 6
            for step in ("people", "people_glb", "motion"):
                getattr(m, f"{pre}_{step}")(C, False)
            assert sorted(os.listdir(T)) == ["07_scene_phy.glb", "09_people_check_prev.mp4"], os.listdir(T)


def test_export_refuses_stale_people_before_touching_the_previous_export():
    """摆人的结果过期 / 是老写法: 导出直接失败, 而且不先把上一次导出的 12_motion/ 删了。"""
    with tempfile.TemporaryDirectory() as T:
        r1 = f"{T}/sc/r1"; os.makedirs(f"{r1}/12_motion"); open(f"{r1}/12_motion/old.fbx", "w").write("x")
        open(f"{r1}/09_scene_nopeople.glb", "w").write("x")
        json.dump(dict(people={}), open(f"{r1}/10_people.json", "w"))                       # 10-09 之前的写法
        os.makedirs(f"{T}/body/04_mesh_occ0/mesh_4d_individual/1")
        r = subprocess.run([sys.executable, f"{SAM3D}/pipeline/motion_export.py", "--r1", r1, "--body", f"{T}/body"],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "重跑 people" in r.stderr, r.stderr[-300:]
        assert os.listdir(f"{r1}/12_motion") == ["old.fbx"]


def test_static_scene_drops_only_the_matched_people():
    """没有人的场景: 只摘掉 4D 顶替了的那几个静态 person, 名字精确匹配(person_1 不连坐 person_10), 别的物体原样留着。"""
    with tempfile.TemporaryDirectory() as T:
        sc = trimesh.Scene()
        for i, n in enumerate(("person_1", "person_10", "pedestrian_3", "car_1")):
            sc.add_geometry(trimesh.creation.box(extents=(1, 1, 1), transform=trimesh.transformations.translation_matrix((3 * i, 0, 0))), node_name=n)
        sc.export(f"{T}/in.glb")
        dropped = B.static_scene(f"{T}/in.glb", f"{T}/out.glb", [1, 3])
        assert sorted(dropped) == ["pedestrian_3", "person_1"], dropped
        got = trimesh.load(f"{T}/out.glb")
        assert sorted(got.graph.nodes_geometry) == ["car_1", "person_10"], got.graph.nodes_geometry
        assert np.allclose(sorted(g.centroid[0] for g in got.dump()), [3, 9])          # 位置没动


def test_render_video_says_what_to_do_when_there_is_no_blend():
    """渲视频吃的是 motion 步出的 .blend: 没有就直说先跑 motion, 不留半个 mp4。"""
    with tempfile.TemporaryDirectory() as T:
        r = subprocess.run([sys.executable, f"{SAM3D}/tools/mocap/render_video.py", f"{T}/x.blend", f"{T}/x.mp4"],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "先跑 motion" in r.stderr, r.stderr
        assert os.listdir(T) == []


def test_video_is_opt_in_and_rendered_from_the_delivered_blend():
    """--video 才渲; 渲的是这一步自己刚出的 .blend(和 FBX 同一份数据), 排在自检过了之后。"""
    src = open(f"{SAM3D}/pipeline/motion_export.py").read()
    assert 'ap.add_argument("--video", action="store_true"' in src
    call = 'render_video.py", f"{out}/{name}_scene_people.blend"'
    assert src.index("to_blender.py") < src.index("if a.video:") < src.index(call)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f(); print("ok", name)
