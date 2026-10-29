#!/usr/bin/env python
"""FP 定朝向那一步(fpose)里不用模型的部分: 改 glb 里物体的朝向、从 FP 的位姿取"绕天转了几度"、重跑时从原摆放读、
步骤表里排在组装和摆人之间。不起 GPU。跑: $PY tools/tests/test_fp_yaw.py"""
import contextlib, importlib.util, io, json, os, sys, tempfile
import numpy as np, trimesh

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline"); sys.path.insert(0, f"{SAM3D}/data/outdoor/scripts")
import fp_yaw_glb as G
from world_frame import export_R, inverse_T

UP = np.array([0.1, -0.98, 0.15]); UP /= np.linalg.norm(UP)      # 歪一点的天, 和真轮次一样不是正 -y

def _scene(path):
    """两个盒子, 顶点在 DA3 世界系里, 节点矩阵 = 导出旋转(和 assemble_scene 导出的一样: 每个节点同一个矩阵)。"""
    sc = trimesh.Scene(); E = np.eye(4); E[:3, :3] = export_R(UP)
    for name, c in (("chair_4", [1.0, 0.2, 3.0]), ("desk_10", [-1.0, 0.0, 4.0])):
        m = trimesh.creation.box([0.6, 0.9, 0.3]); m.apply_translation(c)
        sc.add_geometry(m, node_name=name, geom_name=name, transform=E)
    sc.export(path)

def _verts_da3(path, name):
    sc = trimesh.load(path); T, g = sc.graph[name]; m = sc.geometry[g].copy(); m.apply_transform(T); m.apply_transform(inverse_T(UP))
    return m.vertices

def test_rotate_only_named_object_about_its_center():
    with tempfile.TemporaryDirectory() as d:
        src, dst = f"{d}/in.glb", f"{d}/out.glb"; _scene(src)
        v0 = _verts_da3(src, "chair_4"); c = (v0.min(0) + v0.max(0)) / 2
        G.apply_yaws(src, dst, {"chair_4": (90.0, c)}, UP)
        v1 = _verts_da3(dst, "chair_4")
        want = (v0 - c) @ G.rot_about(UP, 90.0)[:3, :3].T + c
        assert np.abs(np.sort(v1, 0) - np.sort(want, 0)).max() < 1e-5, "chair_4 没绕自己中心、绕天转 90°"
        assert np.abs((v1.min(0) + v1.max(0)) / 2 @ UP - c @ UP) < 1e-6, "转完高度变了"
        assert np.allclose(_verts_da3(dst, "desk_10"), _verts_da3(src, "desk_10"), atol=1e-6), "没点名的 desk_10 也动了"

def test_yaw_read_from_fp_pose():
    """FP 的结果 = 相机系的 4x4。只取绕天转的角度; 它顺带挪的位置、歪的一点点都不要。"""
    rng = np.random.default_rng(0)
    w2c = np.eye(4); w2c[:3, :3] = trimesh.transformations.random_rotation_matrix(rng.random(3))[:3, :3]; w2c[:3, 3] = [0.2, -0.1, 3]
    Tc = np.eye(4); Tc[:3, 3] = [1, 0.2, 3]; Pc = w2c @ Tc
    for deg in (0, 37, 180, 283):
        best = w2c @ Tc @ G.rot_about(UP, deg) @ G.rot_about(np.cross(UP, [1, 0, 0]), 3)   # 再歪 3°
        best[:3, 3] += [0.05, 0.02, -0.04]                                                  # 再挪 7 cm
        y, tilt = G.yaw_of(Pc, best, w2c, UP)
        assert min(abs(y - deg), 360 - abs(y - deg)) < 0.5, (deg, y)
        assert 2 < tilt < 4, tilt

def test_rerun_reads_original_placement():
    """重跑这一步要从组装出的原摆放转, 不能在已经转过的上面再转一遍。组装重跑过(07_scene.glb 换新了)就用新的。"""
    with tempfile.TemporaryDirectory() as R1:
        _scene(f"{R1}/07_scene.glb"); v0 = _verts_da3(f"{R1}/07_scene.glb", "chair_4"); c = (v0.min(0) + v0.max(0)) / 2
        for _ in range(2):
            src = G.source(R1); G.apply_yaws(src, f"{R1}/07_scene.glb", {"chair_4": (90.0, c)}, UP); G.mark_done(R1)
        want = (v0 - c) @ G.rot_about(UP, 90.0)[:3, :3].T + c
        assert np.abs(np.sort(_verts_da3(f"{R1}/07_scene.glb", "chair_4"), 0) - np.sort(want, 0)).max() < 1e-5, "转了两遍"
        _scene(f"{R1}/07_scene.glb"); new = open(f"{R1}/07_scene.glb", "rb").read()   # 组装重跑
        assert open(G.source(R1), "rb").read() == new, "组装重跑后还在读旧的原摆放"

def test_visible_fraction():
    """露出 = 抠图面积 / 按摆放渲染出的整个轮廓面积。office table_7 被胳膊挡得只剩 29%, FP 把它转错了 87°。"""
    sil = np.zeros((100, 100), bool); sil[40:60, 40:60] = True
    half = sil.copy(); half[:, 50:] = False
    assert G.visible_frac(sil, sil) == 1 and G.visible_frac(half, sil) == 0.5
    assert G.visible_frac(sil, np.zeros_like(sil)) == 0          # 物体不在画面里

def test_step_sits_between_assemble_and_people():
    spec = importlib.util.spec_from_file_location("indoor_pipeline", f"{SAM3D}/pipeline/indoor_pipeline.py")
    p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
    names = [s[0] for s in p.STEPS_OBJ]
    assert names.index("fpose") == names.index("assemble") + 1 and names.index("people") == names.index("fpose") + 1
    assert "fpose" in p.DEPS_OBJ["people"] and "fpose" in p.DEPS_OBJ["phys"] and p.DEPS_OBJ["fpose"] == ["assemble"]
    assert "fpose" in p.GPU_STEPS and p.CFG["FP_YAW"] == "1" and p.CFG["FP_MIN_VIS"] == "0.5"
    c = p.Ctx("office", "_test_fp"); c.R1, c.LOG = "/x/r1", "/x/r1/logs"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): p.o_fpose(c, True)
    assert "fp_yaw.py" in buf.getvalue() and p.PY["fpose"] in buf.getvalue() and "--min-vis 0.5" in buf.getvalue(), buf.getvalue()

def test_web_driver_runs_it_and_gate_5_reports_it():
    """网页走的是驱动的分段表, 不是 STEPS_OBJ —— 漏在表外面这一步就永远不跑(people 以前就这么漏过)。"""
    spec = importlib.util.spec_from_file_location("drive", f"{SAM3D}/.claude/skills/run-recon-pipeline/drive.py")
    d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
    seg = [s[1] for s in d.SEGMENTS["indoor"] if s[2] == "check_scene"][0]
    assert seg.index("fpose") == seg.index("assemble") + 1, seg
    with tempfile.TemporaryDirectory() as R1:
        json.dump(dict(物体=[dict(物体="office_chair_1", 转了=283.4), dict(物体="chair_8", 转了=1.2),
                            dict(物体="door_3", 转了=None, 原因="第 5 帧没有抠图")], 总用时秒=9.1), open(f"{R1}/07_fp_yaw.json", "w"))
        txt = "\n".join(d.check_scene(R1, "off_r1", "office", "indoor")[0])
        assert "[FP 朝向]" in txt and "office_chair_1 77°" in txt and "door_3" in txt and "chair_8" not in txt, txt

if __name__ == "__main__":
    for t in (test_rotate_only_named_object_about_its_center, test_yaw_read_from_fp_pose,
              test_rerun_reads_original_placement, test_visible_fraction, test_step_sits_between_assemble_and_people,
              test_web_driver_runs_it_and_gate_5_reports_it):
        t(); print("ok", t.__name__)
    print("== test_fp_yaw 全过 ==")
