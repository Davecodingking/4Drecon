#!/usr/bin/env python
"""每轮配置(00_run_config.json) + 人体线接进主流程(people_track / people_mesh) + 挑人决策(05_people.json)。
都是临时目录 + dry, 不起 GPU、不碰盘上的轮次。跑: $PY tools/tests/test_people_gate.py"""
import contextlib, importlib.util, io, json, os, sys, tempfile

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
import _common, people_decisions as pd
out_p = _load("outdoor_pipeline", f"{SAM3D}/pipeline/outdoor_pipeline.py")
in_p  = _load("indoor_pipeline",  f"{SAM3D}/pipeline/indoor_pipeline.py")
drive = _load("drive", f"{SAM3D}/.claude/skills/run-recon-pipeline/drive.py")

def _run(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a)
    return buf.getvalue()

def _rc(R1, **over):
    json.dump({"类型": "outdoor", "覆盖": over}, open(f"{R1}/00_run_config.json", "w"))

def _track_report(trk):
    """1 号 69 帧最大 3.5%, 3 号只有 3 帧(一闪而过)。画面 100x100 = 10000 像素。"""
    os.makedirs(trk, exist_ok=True)
    pf = {"1": {str(i): {"area": 350 if i == 7 else 100} for i in range(69)},
          "3": {str(i): {"area": 20} for i in range(3)}}
    json.dump(dict(frames=69, wh=[100, 100], kept_ids=[1, 3], coverage={"1": 69, "3": 3},
                   alarms=[{"id": 1, "frame": 10}], per_frame=pf), open(f"{trk}/track_report.json", "w"))

def _ctx(mod, scene, R1):
    """Ctx 在构造时就要读 R1 底下的配置, 所以先把 R1 指到临时目录再重算一遍。"""
    c = mod.Ctx(scene, "_test_people")
    c.R1, c.LOG = R1, f"{R1}/logs"
    c.cfg = mod.resolve_cfg(scene, R1)
    c.body_auto = pd.resolve_body_run(c.cfg, scene, c.run, R1)
    return c

# ---------------------------------------------------------------- 每轮配置
def test_run_config_layer():
    with tempfile.TemporaryDirectory() as R1:
        _rc(R1, PEOPLE="1", RECALL="1")
        cfg = out_p.resolve_cfg("carpark", R1)
        assert cfg["PEOPLE"] == "1" and cfg["RECALL"] == "1"
        os.environ["RECALL"] = "0"                      # 环境变量压得过本轮配置
        try:
            assert out_p.resolve_cfg("carpark", R1)["RECALL"] == "0"
        finally:
            del os.environ["RECALL"]

def test_run_config_rejects_unknown_key():
    with tempfile.TemporaryDirectory() as R1:
        _rc(R1, PEOPEL="1")                             # 拼错的键不许静默忽略
        try:
            out_p.resolve_cfg("carpark", R1)
        except SystemExit as e:
            assert "PEOPEL" in str(e)
        else:
            raise AssertionError("拼错的键应该报错")

# ---------------------------------------------------------------- 挑人决策
def test_build_defaults_and_keep_ids():
    with tempfile.TemporaryDirectory() as R1:
        trk = f"{R1}/trk"; _track_report(trk)
        d = pd.build(R1, trk)
        rows = {r["id"]: r for r in d["人"]}
        assert rows[1]["决定"] == "keep" and rows[3]["决定"] == "drop"      # 帧数不够 5 默认不勾, 但列出来
        assert rows[1]["最大面积%"] == 3.5 and rows[1]["最大面积帧"] == 7 and rows[1]["告警"] == 1
        assert pd.keep_ids(R1) == [1]
        pd.save_decisions(R1, {3: "keep"})
        assert pd.keep_ids(R1) == [1, 3]
        try:
            pd.save_decisions(R1, {1: "maybe"})
        except ValueError:
            pass
        else:
            raise AssertionError("只认 keep / drop")

def test_resolve_body_run():
    with tempfile.TemporaryDirectory() as R1:
        auto = f"{R1}/body4d"                              # 人体产物跟着场景轮次走, 不在 sam-body4d 树里
        cfg = dict(PEOPLE="1", BODY_RUN="")
        assert pd.resolve_body_run(cfg, "sc", "r9", R1) is True and cfg["BODY_RUN"] == auto
        json.dump({"人": [{"id": 1, "决定": "drop"}]}, open(f"{R1}/05_people.json", "w"))
        cfg = dict(PEOPLE="1", BODY_RUN="")
        assert pd.resolve_body_run(cfg, "sc", "r9", R1) is True and cfg["BODY_RUN"] == ""   # 一个都不要 = 没有人
        cfg = dict(PEOPLE="1", BODY_RUN="/x/b2")                                           # 手工指定的老轮次
        assert pd.resolve_body_run(cfg, "sc", "r9", R1) is False and cfg["BODY_RUN"] == "/x/b2"
        cfg = dict(PEOPLE="0", BODY_RUN="")
        assert pd.resolve_body_run(cfg, "sc", "r9", R1) is False and cfg["BODY_RUN"] == ""
        # 有本轮配置的新轮次: SCENES 里写死的 BODY_RUN(soccer 的 b2)不许劫持它, 只认本轮配置
        os.remove(f"{R1}/05_people.json"); _rc(R1, PEOPLE="1")
        cfg = dict(PEOPLE="1", BODY_RUN="/x/b2")
        assert pd.resolve_body_run(cfg, "sc", "r9", R1) is True and cfg["BODY_RUN"] == auto
        _rc(R1, PEOPLE="0")
        cfg = dict(PEOPLE="0", BODY_RUN="/x/b2")
        assert pd.resolve_body_run(cfg, "sc", "r9", R1) is False and cfg["BODY_RUN"] == ""

# ---------------------------------------------------------------- 两条 pipeline 的两步
def _check_steps(mod, scene, track_fn, mesh_fn):
    with tempfile.TemporaryDirectory() as R1:
        c = _ctx(mod, scene, R1)
        assert "没有人" in _run(track_fn, c, True) and "没有人" in _run(mesh_fn, c, True)
        _rc(R1, PEOPLE="1"); c = _ctx(mod, scene, R1)
        o = _run(track_fn, c, True)
        assert "body4d_pipeline.py" in o and "frames,track,check" in o and f"SCENE_R1={R1}" in o, o
        assert f"BODY_OUT={R1}/body4d" in o, o
        json.dump({"人": [{"id": 1, "决定": "keep"}, {"id": 2, "决定": "drop"}, {"id": 3, "决定": "keep"}]},
                  open(f"{R1}/05_people.json", "w"))
        c = _ctx(mod, scene, R1)
        o = _run(mesh_fn, c, True)
        assert "PICK=1,3" in o and "pick,mesh,report" in o, o
        json.dump({"人": [{"id": 1, "决定": "drop"}]}, open(f"{R1}/05_people.json", "w"))
        c = _ctx(mod, scene, R1)
        assert "一个都没勾" in _run(mesh_fn, c, True)

def test_outdoor_people_steps():
    _check_steps(out_p, "carpark", out_p.s_people_track, out_p.s_people_mesh)

def test_indoor_people_steps():
    _check_steps(in_p, "cyberpunk", in_p.o_people_track, in_p.o_people_mesh)

def test_registered_in_order():
    for steps, deps in ((out_p.STEPS, out_p.DEPS), (in_p.STEPS_OBJ, in_p.DEPS_OBJ)):
        ks = [s[0] for s in steps]
        assert ks.index("track") < ks.index("people_track") < ks.index("people_mesh") < ks.index("scale")
        assert "people_mesh" in deps["scale"] and "people_track" in deps["people_mesh"]
    for kind in ("outdoor", "indoor"):
        drive.check_segments(kind)                       # 段表和步骤表对得上, 否则 SystemExit
        segs = {s[2]: s[1] for s in drive.SEGMENTS[kind]}
        assert "people_track" in segs["check_picks"] and "people_mesh" in segs["check_scale"]

def test_body4d_scene_frames_renumbered():
    """室外场景帧按原视频帧号命名、不连续(soccer: 0000 0001 0002 0003 0005 ...)。
    body4d 按位置读帧, 所以软链要按位置重编号, kept_indices 仍记原视频帧号 —— 摆人按它找相机。"""
    from PIL import Image
    b4 = _load("body4d_pipeline", f"{SAM3D}/pipeline/body4d_pipeline.py")
    with tempfile.TemporaryDirectory() as T:
        src, R1 = f"{T}/frames", f"{T}/r1"; os.makedirs(src); os.makedirs(f"{R1}/02_da3")
        for n in ("0000.jpg", "0002.jpg", "0005.jpg"):
            Image.new("RGB", (8, 6)).save(f"{src}/{n}")
        json.dump({"frames_dir": src, "fps": 24.0, "frames": {"0000.jpg": 0, "0002.jpg": 2, "0005.jpg": 5}},
                  open(f"{R1}/02_da3/frame_map.json", "w"))
        c = b4.Ctx("soccer", "_test_frames")
        c.R1, c.FR, c.LOG = f"{T}/body", f"{T}/body/00_frames", f"{T}/body/logs"
        c.cfg["SCENE_R1"] = R1
        _run(b4.s_frames, c, False)
        assert sorted(os.listdir(c.FR)) == ["0000.jpg", "0001.jpg", "0002.jpg", "frames.json"], os.listdir(c.FR)
        assert os.path.realpath(f"{c.FR}/0002.jpg") == os.path.realpath(f"{src}/0005.jpg")
        assert json.load(open(f"{c.FR}/frames.json"))["kept_indices"] == [0, 2, 5]

def test_body4d_out_dir_follows_scene_run():
    """主流程拉起的人体线(BODY_OUT)产物写进 <场景轮次>/body4d; 手工单跑的老轮次(b1/b2)仍在 sam-body4d/outputs。"""
    b4 = _load("body4d_pipeline", f"{SAM3D}/pipeline/body4d_pipeline.py")
    os.environ["BODY_OUT"] = "/x/result/soccer/r9/body4d"
    try:
        c = b4.Ctx("soccer", "r9")
    finally:
        del os.environ["BODY_OUT"]
    assert c.R1 == "/x/result/soccer/r9/body4d" and c.TRK == f"{c.R1}/02_body4d" and c.LOG == f"{c.R1}/logs"
    assert b4.Ctx("soccer", "b2").R1 == f"{SAM3D}/sam-body4d/outputs/soccer/b2"

# ---------------------------------------------------------------- 驱动
def test_drive_init_run_config():
    with tempfile.TemporaryDirectory() as R1:
        drive.init_run_config(R1, "outdoor", people="1", recall=None, done=0)
        d = json.load(open(f"{R1}/00_run_config.json"))
        assert d["类型"] == "outdoor" and d["覆盖"] == {"PEOPLE": "1"}
        for bad in (lambda: drive.init_run_config(R1, "indoor", None, None, 0),     # 类型对不上
                    lambda: drive.init_run_config(R1, "outdoor", "0", None, 2)):    # 过了闸 1 不许改有没有人
            try:
                bad()
            except SystemExit:
                pass
            else:
                raise AssertionError("应该拒绝")

def test_drive_people_changed():
    with tempfile.TemporaryDirectory() as R1:
        body = f"{R1}/body"; os.makedirs(f"{body}/02_body4d")
        json.dump({"keep_ids": [1, 2]}, open(f"{body}/02_body4d/picked.json", "w"))
        json.dump({"人": [{"id": 1, "决定": "keep"}, {"id": 2, "决定": "keep"}]}, open(f"{R1}/05_people.json", "w"))
        assert drive.people_changed(R1, body) is False
        pd.save_decisions(R1, {2: "drop"})
        assert drive.people_changed(R1, body) is True

def test_drive_gate2_lists_people():
    with tempfile.TemporaryDirectory() as R1:
        assert drive.people_lines(R1) == []
        _track_report(f"{R1}/trk"); pd.build(R1, f"{R1}/trk")
        s = "\n".join(drive.people_lines(R1))
        assert "2 个" in s and "勾了 1 个" in s, s

if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
