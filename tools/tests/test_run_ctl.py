#!/usr/bin/env python
"""tools/run_ctl.py —— 启动页 /new 和本轮页 /run 的后端。
数据集/结果根目录指到临时目录, drive_go.sh 换成只记参数的假脚本: 不起驱动、不碰盘上真实轮次。
跑: $PY tools/tests/test_run_ctl.py"""
import json, os, stat, sys, tempfile

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/tools"); sys.path.insert(0, f"{SAM3D}/pipeline")
import run_ctl, people_decisions as pd


def _sandbox(T):
    """临时的 数据集/结果 两个根 + 假 drive_go.sh(把参数写进 T/go_args)。"""
    run_ctl.ROOTS = {"outdoor": (f"{T}/od", f"{T}/or"), "indoor": (f"{T}/id", f"{T}/ir")}
    for k in ("od/soccer/frames", "od/beach", "od/.hidden", "id/office", "or", "ir"):
        os.makedirs(f"{T}/{k}", exist_ok=True)
    open(f"{T}/od/beach/video.mp4", "w").close()
    open(f"{T}/id/office/clip.mp4", "w").close()
    open(f"{T}/od/nfl.mp4", "w").write("video")           # 根目录下散放的视频: 选它时自动建场景文件夹
    go = f"{T}/drive_go.sh"
    open(go, "w").write(f'#!/bin/bash\necho "$@" > {T}/go_args\necho started\n')
    os.chmod(go, os.stat(go).st_mode | stat.S_IEXEC)
    run_ctl.DRIVE_GO = go
    run_ctl.running = lambda scene, run: False


def _state(R1, seg, stop):
    os.makedirs(f"{R1}/logs", exist_ok=True)
    json.dump({"段": seg, "停在闸口": stop}, open(f"{R1}/logs/.drive_state.json", "w"))


def test_datasets_and_next_run():
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        os.makedirs(f"{T}/or/soccer/r1")
        open(f"{T}/od/soccer.mp4", "w").close()                   # 有同名文件夹的散放视频: 文件夹说了算, 不重复列
        d = run_ctl.datasets()
        assert [x["场景"] for x in d["outdoor"]] == ["beach", "nfl", "soccer"], d
        assert [x["场景"] for x in d["indoor"]] == ["office"], d
        assert {x["场景"]: x["下一轮"] for x in d["outdoor"]} == {"beach": "r1", "nfl": "r1", "soccer": "r2"}
        assert [x.get("新视频") for x in d["outdoor"]] == [None, "nfl.mp4", None]


def test_new_run_writes_config_and_starts():
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        R1 = run_ctl.new_run("outdoor", "soccer", "r1", people="1", recall="0")
        assert R1 == f"{T}/or/soccer/r1"
        assert json.load(open(f"{R1}/00_run_config.json")) == {"类型": "outdoor", "覆盖": {"PEOPLE": "1", "RECALL": "0"}}
        assert open(f"{T}/go_args").read().split() == ["soccer", "r1"]
        run_ctl.new_run("indoor", "office", "r7", people="0", recall="0")
        assert open(f"{T}/go_args").read().split() == ["office", "r7", "--indoor"]
        for bad in (("outdoor", "soccer", "r1"),        # 已经有了
                    ("outdoor", "nope", "r1"),          # 数据集里没有
                    ("outdoor", "soccer", "../x"),      # 名字不许带路径
                    ("outdoor", "soc cer", "r1"),
                    ("indoor", "soccer", "r1")):        # 类型和数据集对不上
            try:
                run_ctl.new_run(*bad, people="0", recall="0")
            except ValueError:
                pass
            else:
                raise AssertionError(f"应该拒绝 {bad}")


def test_set_config_follows_segments():
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        R1 = run_ctl.new_run("outdoor", "soccer", "r1", people="0", recall="0")
        _state(R1, 0, True)                                         # 停在闸 1: 有没有人还能改
        run_ctl.set_config("outdoor", "soccer", "r1", {"PEOPLE": "1"})
        assert json.load(open(f"{R1}/00_run_config.json"))["覆盖"]["PEOPLE"] == "1"
        _state(R1, 1, True)                                         # 停在闸 2: 人体跟踪跑过了
        try:
            run_ctl.set_config("outdoor", "soccer", "r1", {"PEOPLE": "0"})
        except ValueError:
            pass
        else:
            raise AssertionError("闸 2 不许改有没有人")
        _state(R1, 2, True)                                         # 停在闸 3: 召回还没跑, 能改
        run_ctl.set_config("outdoor", "soccer", "r1", {"RECALL": "1"})
        run_ctl.running = lambda scene, run: True                   # 驱动在跑时一律不许改
        try:
            run_ctl.set_config("outdoor", "soccer", "r1", {"RECALL": "0"})
        except ValueError:
            pass
        else:
            raise AssertionError("在跑的时候不许改")


def test_state_reports_what_the_page_needs():
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        R1 = run_ctl.new_run("outdoor", "soccer", "r1", people="1", recall="0")
        _state(R1, 1, True)
        json.dump({"人": [{"id": 1, "决定": "keep", "帧数": 9, "最大面积%": 1.0, "最大面积帧": 3, "告警": 0}]},
                  open(pd.path(R1), "w"))
        s = run_ctl.state("outdoor", "soccer", "r1")
        assert s["类型"] == "outdoor" and s["有人"] == "1" and s["召回"] == "0"
        assert s["闸"] == 2 and s["停在闸口"] is True and s["在跑"] is False
        assert s["可改"] == {"PEOPLE": False, "RECALL": True}
        assert s["人"]["人"][0]["id"] == 1 and s["能继续"] is True


def test_set_people_and_go():
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        R1 = run_ctl.new_run("outdoor", "soccer", "r1", people="1", recall="0")
        json.dump({"人": [{"id": 1, "决定": "keep"}, {"id": 2, "决定": "keep"}]}, open(pd.path(R1), "w"))
        run_ctl.set_people("outdoor", "soccer", "r1", {"2": "drop"})
        assert pd.keep_ids(R1) == [1]
        os.remove(f"{T}/go_args")
        run_ctl.go("outdoor", "soccer", "r1")
        assert open(f"{T}/go_args").read().split() == ["soccer", "r1"]
        run_ctl.running = lambda scene, run: True
        try:
            run_ctl.go("outdoor", "soccer", "r1")
        except ValueError:
            pass
        else:
            raise AssertionError("在跑的时候不许再起一个")


def test_steps_view_for_the_wizard():
    """本轮页按段一步步显示: 每段状态、每步状态和耗时、闸口报告原文、这一闸的产出、Continue 之后做什么。"""
    import time
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        R1 = run_ctl.new_run("outdoor", "soccer", "r1", people="1", recall="0")
        _state(R1, 2, True)                                    # 停在第 3 道闸
        now = time.time()
        for k in ("frames", "vlm", "da3", "up", "track", "people_track", "dedup", "relabel", "pick", "verify",
                  "override", "gen", "ratio", "meshes", "boxes", "vlm_sizes", "people_mesh", "scale"):
            open(f"{R1}/logs/.stamp_{k}", "w").write("x")
        open(f"{R1}/logs/timing.tsv", "w").write("步骤\t环境\t开始\t结束\t耗时秒\t等GPU秒\t结果\n"
                                                 "people_mesh\tsam3d\ta\tb\t494.3\t0.0\tok\n")
        open(f"{R1}/logs/drive_0924_154335.log", "w").write(
            "[开始] 2026-09-24 15:43:35\n== scale  [sam3d]  x\n闸 3/6 — 尺度\n" + "=" * 72 + "\n\n"
            "[3D 页] 数据还没有, 重算\n  标定倍数 k = 0.939, 一致率 None\n  3D: open the gate page\n\n  该看的图:\n"
            "    /x/world_scale.json\n\n★ 停在这里等人。\n")
        os.makedirs(f"{R1}/02_da3")
        json.dump({"k": 0.939, "来源": "人", "人": {"每人": {"1": {"场景深度比模型": 1.023, "帧": 180}}}},
                  open(f"{R1}/02_da3/world_scale.json", "w"))
        # 闸 2 的候选决定: 整词作废的词, 哪怕它别的行还写着 keep, 也整词不生成(和 review_gate 落地同一个语义)
        json.dump({"候选": [{"词": "goal post", "实例": 0, "决定": "keep"},
                            {"词": "helmet", "实例": 1, "决定": "drop-word"}, {"词": "helmet", "实例": 2, "决定": "keep"},
                            {"词": "cone", "实例": 1, "决定": "shared"}, {"词": "cone", "实例": 2, "决定": "keep"},
                            {"词": "ball", "实例": 3, "决定": "drop"}]}, open(f"{R1}/05_review.json", "w"))
        s = run_ctl.state("outdoor", "soccer", "r1")
        segs = s["段列表"]
        assert segs[1]["候选决定"] == {"候选": 6, "会生成": 2, "按词": {"goal post": 1, "cone": 1}, "作废词": ["helmet"]}, \
            segs[1]["候选决定"]
        assert [x["状态"] for x in segs] == ["done", "done", "waiting", "pending", "pending", "pending"], segs
        assert segs[2]["名"] == "尺度" and segs[2]["步骤"][-1] == {"步骤": "scale", "说明": segs[2]["步骤"][-1]["说明"],
                                                                "状态": "done", "耗时": None}
        assert [x for x in segs[2]["步骤"] if x["步骤"] == "people_mesh"][0]["耗时"] == 494.3
        assert segs[2]["报告"] == ["标定倍数 k = 0.939, 一致率 None"], segs[2]["报告"]   # 路径/3D 提示不进页面
        assert segs[2]["尺度"]["k"] == 0.939 and segs[2]["尺度"]["每人"] == {"1": 1.023}
        assert segs[3]["闸"] == "跳过"                                                # 召回关, 第 4 道闸不停
        nxt = s["下一步"]
        assert nxt["段"] == 4 and nxt["停在"] == 5, nxt                              # 跑第 4 段, 跳过闸 4, 停在闸 5


def test_new_run_from_loose_video():
    """选数据集根目录下散放的 mp4: 建 <场景>/video.mp4(硬链接, 不复制不移动原文件)再开跑。"""
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        run_ctl.new_run("outdoor", "nfl", "r1", people="1", recall="0")
        v = f"{T}/od/nfl/video.mp4"
        assert os.path.exists(v) and os.path.exists(f"{T}/od/nfl.mp4")
        assert os.stat(v).st_ino == os.stat(f"{T}/od/nfl.mp4").st_ino
        assert open(f"{T}/go_args").read().split() == ["nfl", "r1"]


def test_new_run_refuses_tag_collision():
    """标签默认取场景名前 3 个字母: abcx 和 abcy 的 r1 都是 abc_r1, 生成的 mesh 会互相覆盖。"""
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        for sc in ("abcx", "abcy"):
            os.makedirs(f"{T}/od/{sc}/frames")
        run_ctl.new_run("outdoor", "abcx", "r1", people="0", recall="0")
        try:
            run_ctl.new_run("outdoor", "abcy", "r1", people="0", recall="0")
        except ValueError as e:
            assert "abc_r1" in str(e), e
        else:
            raise AssertionError("标签撞了应该拒绝")
        run_ctl.new_run("outdoor", "abcy", "r2", people="0", recall="0")     # 轮次不同, 标签不同, 可以

def _marker(R1, step=None, card=None, what=None, pid=None):
    os.makedirs(f"{R1}/logs", exist_ok=True)
    pid = os.getpid() if pid is None else pid
    if step:
        open(f"{R1}/logs/.running_{step}", "w").write(str(pid))
    if card:
        json.dump({"卡": card, "步骤": what, "pid": pid, "开始": "x"}, open(f"{R1}/logs/.gpu_{card}.json", "w"))


def _dead_pid():
    import subprocess, sys
    p = subprocess.Popen([sys.executable, "-c", "pass"]); p.wait()
    return p.pid


def test_several_steps_running_at_once():
    """并行之后一段里可以有好几步同时在跑, 各用各的卡。"""
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        R1 = run_ctl.new_run("outdoor", "soccer", "r1", people="1", recall="0")
        run_ctl.running = lambda scene, run: True               # 先建轮次再说它在跑: 在跑的轮次 go() 会拒绝
        _state(R1, 2, False)
        open(f"{R1}/logs/drive_0929_100000.log", "w").write("[开始] 2026-09-29 10:00:00\n[并行] 开始 gen\n")
        for step, card, what in (("gen", "0", "gen"), ("vlm_sizes", "1", "vlm_sizes"), ("people_mesh", "3", "mesh/p0")):
            _marker(R1, step, card, what)
        _marker(R1, card="2", what="mesh/p1")
        s = run_ctl.state("outdoor", "soccer", "r1")
        assert s["GPU"] == "0,1,2,3", s["GPU"]
        assert s["任务"] == [{"卡": "0", "步骤": "gen"}, {"卡": "1", "步骤": "vlm_sizes"},
                             {"卡": "2", "步骤": "mesh/p1"}, {"卡": "3", "步骤": "mesh/p0"}], s["任务"]
        rows = {r["步骤"]: r["状态"] for r in s["段列表"][2]["步骤"]}
        assert [k for k, v in rows.items() if v == "running"] == ["gen", "vlm_sizes", "people_mesh"], rows
        assert rows["scale"] == "pending"


def test_stale_markers_are_ignored():
    """进程被杀的时候来不及删标记。进程已经不在的标记不算数, 不然网页一直显示"在跑"。"""
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        R1 = run_ctl.new_run("outdoor", "soccer", "r1", people="1", recall="0")
        _marker(R1, "gen", "0", "gen", pid=_dead_pid())
        assert run_ctl.running_steps(R1) == [] and run_ctl.gpu_tasks(R1) == []
        open(f"{R1}/logs/.gpu_5.json", "w").write("{坏的")                # 写到一半的文件也不能让网页崩
        assert run_ctl.gpu_tasks(R1) == []


def test_old_style_run_still_shows_its_step():
    """PARALLEL=0 或者老轮次: 没有标记文件, 照旧看驱动日志里最后出现的步骤。"""
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        R1 = run_ctl.new_run("outdoor", "soccer", "r1", people="1", recall="0")
        run_ctl.running = lambda scene, run: True
        _state(R1, 0, False)
        open(f"{R1}/logs/drive_0929_100000.log", "w").write("[开始] 2026-09-29 10:00:00\n== vlm  [trellis2]  x\n   3.0s\n== da3  [spatiallm]  y\n")
        s = run_ctl.state("outdoor", "soccer", "r1")
        rows = {r["步骤"]: r["状态"] for r in s["段列表"][0]["步骤"]}
        assert rows["da3"] == "running" and s["GPU"] is None and s["任务"] == [], (rows, s["GPU"])


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
