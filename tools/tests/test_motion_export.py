#!/usr/bin/env python
"""动作导出接进流水线(motion 步): 步骤表 / 分段表 / 找一手数据 / 没有人的场景文件。
跑: $PY tools/tests/test_motion_export.py"""
import importlib.util, os, subprocess, sys, tempfile
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
    """motion 排在合成(people_glb)后面、依赖它(场景用它出的 09_scene_nopeople.glb), 领一张卡; 驱动的验收段里有它。"""
    out, ind = _load("_o", f"{SAM3D}/pipeline/outdoor_pipeline.py"), _load("_i", f"{SAM3D}/pipeline/indoor_pipeline.py")
    drive = _load("_d", f"{SAM3D}/.claude/skills/run-recon-pipeline/drive.py")
    for kind, steps, deps, gpu in (("outdoor", out.STEPS, out.DEPS, out.GPU_STEPS), ("indoor", ind.STEPS_OBJ, ind.DEPS_OBJ, ind.GPU_STEPS)):
        keys = [s[0] for s in steps]
        assert keys.index("motion") == keys.index("people_glb") + 1, keys
        assert deps["motion"] == ["people_glb"] and "motion" in gpu
        seg = [s for s in drive.SEGMENTS[kind] if s[2] == "check_scene"][0][1]
        assert seg.index("motion") > seg.index("people_glb"), seg


def test_mesh_run_needs_first_hand_data():
    """出网格那一轮: 和 body_into_scene 同一个找法(occ1 优先); 没存一手数据的老轮次直接失败, 说清楚要重跑哪一步。"""
    with tempfile.TemporaryDirectory() as T:
        try:
            ME.mesh_run(T); assert False
        except SystemExit as e:
            assert "people_mesh" in str(e)
        os.makedirs(f"{T}/04_mesh_occ0/mesh_4d_individual")
        try:
            ME.mesh_run(T); assert False
        except SystemExit as e:
            assert "一手数据" in str(e) and "重跑 people_mesh" in str(e), e
        os.makedirs(f"{T}/04_mesh_occ0/mhr_4d_individual")
        assert ME.mesh_run(T) == f"{T}/04_mesh_occ0"
        os.makedirs(f"{T}/04_mesh_occ1/mesh_4d_individual"); os.makedirs(f"{T}/04_mesh_occ1/mhr_4d_individual")
        assert ME.mesh_run(T) == f"{T}/04_mesh_occ1"


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
