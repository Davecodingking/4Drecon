#!/usr/bin/env python
"""按依赖表并行跑步骤(pipeline/sched.py)。不碰显卡: 步骤是假的。
跑: $PY tools/tests/test_sched.py"""
import json, os, subprocess, sys, tempfile, time

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import sched

KEYS = ["frames", "vlm", "da3", "up", "track", "boxes", "scale", "solo"]
DEPS = {"vlm": ["frames"], "da3": ["frames"], "up": ["da3"], "track": ["frames", "vlm"],
        "boxes": ["track", "up"], "scale": ["boxes"], "solo": ["scale"]}


def _raises(fn, *a):
    try:
        fn(*a)
    except ValueError as e:
        return str(e)
    raise AssertionError("该报错没报错")


def test_check_graph_accepts_a_good_table():
    sched.check_graph(KEYS, DEPS)


def test_check_graph_rejects_unknown_names():
    assert "walls" in _raises(sched.check_graph, KEYS, dict(DEPS, boxes=["walls"]))
    assert "walls" in _raises(sched.check_graph, KEYS, dict(DEPS, walls=["frames"]))


def test_check_graph_rejects_backward_edges():
    """依赖排在后面 = 顺序错或者成环。步骤表的顺序就是 PARALLEL=0 时的执行顺序。"""
    assert "solo" in _raises(sched.check_graph, KEYS, dict(DEPS, up=["solo"]))
    assert "up" in _raises(sched.check_graph, KEYS, dict(DEPS, up=["up"]))


def test_ancestors_are_transitive():
    a = sched.ancestors(DEPS)
    assert a["solo"] == {"scale", "boxes", "track", "up", "da3", "vlm", "frames"}, a["solo"]
    assert a["frames"] == set()


def test_real_tables_pass():
    """三条流水线自己的步骤表和依赖表。依赖表以前只用来判过期, 现在还决定谁先谁后, 不能有错。"""
    PY = sys.executable
    for cmd in ([f"{SAM3D}/pipeline/outdoor_pipeline.py"],
                [f"{SAM3D}/pipeline/indoor_pipeline.py", "--chain", "obj"],
                [f"{SAM3D}/pipeline/indoor_pipeline.py", "--chain", "lm"],
                [f"{SAM3D}/pipeline/body4d_pipeline.py"]):
        r = subprocess.run([PY] + cmd + ["--dump-deps"], capture_output=True, text=True)
        assert r.returncode == 0, (cmd, r.stderr[-400:])
        d = json.loads(r.stdout.strip().splitlines()[-1])
        sched.check_graph(d["steps"], d["deps"])
        assert set(d["gpu"]) <= set(d["steps"]), (cmd, set(d["gpu"]) - set(d["steps"]))


def test_track_depends_on_vlm():
    """track 读 vlm 出的词表(01_vlm_vocab.json)。以前依赖表里没写, 因为两步不在同一段里没出过事。"""
    for cmd in ([f"{SAM3D}/pipeline/outdoor_pipeline.py"], [f"{SAM3D}/pipeline/indoor_pipeline.py", "--chain", "obj"]):
        r = subprocess.run([sys.executable] + cmd + ["--dump-deps"], capture_output=True, text=True)
        d = json.loads(r.stdout.strip().splitlines()[-1])
        assert "vlm" in d["deps"]["track"], (cmd, d["deps"]["track"])


def test_indoor_shell_waits_for_recall_merge():
    """室内的房间壳: 开着召回时读的是 06_all_boxes_merged.json(recall_merge 写的)。两步在同一段里,
    依赖表里不写的话并行时会同时跑, 壳读到的是上一次留下的盒子。"""
    r = subprocess.run([sys.executable, f"{SAM3D}/pipeline/indoor_pipeline.py", "--chain", "obj", "--dump-deps"],
                       capture_output=True, text=True)
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert "recall_merge" in d["deps"]["shell"], d["deps"]["shell"]


class _Fake:
    """假步骤: 跑 secs 秒, 然后以 rc 退出。记下起止时刻。"""
    log = []

    def __init__(self, name, secs=0.3, rc=0):
        self.name, self.t0, self.t1, self.rc = name, time.time(), time.time() + secs, rc

    def poll(self):
        if time.time() < self.t1:
            return None
        if self not in _Fake.log:
            _Fake.log.append(self)
        return self.rc


def _run(sel, deps=DEPS, fail=(), **kw):
    _Fake.log = []
    started = []
    def launch(k):
        started.append(k)
        return _Fake(k, rc=1 if k in fail else 0)
    rc = sched.run_graph(sel, deps, launch, poll=0.02, **kw)
    return rc, started, {f.name: (f.t0, f.t1) for f in _Fake.log}


def test_independent_steps_overlap():
    """vlm 和 da3 互不依赖: 同时跑。up 等 da3, track 等 vlm。"""
    rc, started, t = _run(["vlm", "da3", "up", "track"])
    assert set(rc) == {"vlm", "da3", "up", "track"} and not any(rc.values()), rc
    assert t["vlm"][0] < t["da3"][1] and t["da3"][0] < t["vlm"][1], "vlm 和 da3 应该有重叠"
    assert t["up"][0] >= t["da3"][1] and t["track"][0] >= t["vlm"][1]


def test_max_parallel_one_is_the_old_order():
    rc, started, t = _run(["vlm", "da3", "up", "track"], max_parallel=1)
    assert started == ["vlm", "da3", "up", "track"], started
    assert t["da3"][0] >= t["vlm"][1] and t["up"][0] >= t["da3"][1] and t["track"][0] >= t["up"][1]


def test_waits_through_unselected_steps():
    """决策落地后重跑的是 ...,boxes,solo, 中间隔着没选的 scale: solo 仍然要等 boxes。"""
    rc, started, t = _run(["boxes", "solo"])
    assert t["solo"][0] >= t["boxes"][1], t


def test_failure_stops_new_steps_but_not_running_ones():
    rc, started, t = _run(["vlm", "da3", "up", "track"], fail=("vlm",))
    assert rc["vlm"] == 1 and rc["da3"] == 0, rc            # da3 已经在跑, 让它跑完
    assert "up" not in started and "track" not in started, started


def test_steps_without_deps_entry_just_run():
    rc, started, t = _run(["frames", "vlm"], deps={})
    assert set(rc) == {"frames", "vlm"}


def test_run_children_blocks_and_markers():
    """子进程的输出整块写出, 不穿插; 在跑的步骤有 .running_<步骤>, 跑完删掉。"""
    import io
    with tempfile.TemporaryDirectory() as T:
        seen = {}
        def cmd(k):
            code = (f"import time, os; print('== {k}  [env]  说明', flush=True); time.sleep(0.4); "
                    f"print('    $ 命令 {k}'); print('   0.4s')")
            return [sys.executable, "-c", code]
        out = io.StringIO()
        import threading
        def peek():
            time.sleep(0.2); seen["mid"] = sorted(f for f in os.listdir(T) if f.startswith(".running_"))
        th = threading.Thread(target=peek); th.start()
        rc = sched.run_children(["vlm", "da3"], DEPS, cmd, T, out=out)
        th.join()
        assert rc == {"vlm": 0, "da3": 0}, rc
        assert seen["mid"] == [".running_da3", ".running_vlm"], seen
        assert not [f for f in os.listdir(T) if f.startswith(".running_") or f.startswith(".out_")], os.listdir(T)
        txt = out.getvalue()
        assert "[并行] 开始 vlm" in txt and "[并行] 开始 da3" in txt
        for k in ("vlm", "da3"):                             # 每个步骤的三行连在一起
            assert f"== {k}  [env]  说明\n    $ 命令 {k}\n   0.4s\n" in txt, txt


def test_run_children_reports_failure():
    import io
    with tempfile.TemporaryDirectory() as T:
        out = io.StringIO()
        rc = sched.run_children(["vlm"], {}, lambda k: [sys.executable, "-c", "import sys; print('坏了'); sys.exit(3)"],
                                T, out=out)
        assert rc == {"vlm": 3} and "[失败] 步骤 vlm 退出码 3" in out.getvalue() and "坏了" in out.getvalue()


def test_dry_run_is_still_linear_and_prints_commands():
    """干跑不并行: 打印出来的命令要能一条条照着粘去跑, 顺序就是步骤表的顺序。"""
    r = subprocess.run([sys.executable, f"{SAM3D}/pipeline/outdoor_pipeline.py", "--scene", "nfl1", "--run", "r1",
                        "--only", "vlm,da3,up", "-n"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    heads = [l.split()[1] for l in r.stdout.splitlines() if l.startswith("== ")]
    assert heads == ["vlm", "da3", "up"], heads
    assert "[并行]" not in r.stdout


def test_child_mode_prints_no_banner():
    r = subprocess.run([sys.executable, f"{SAM3D}/pipeline/outdoor_pipeline.py", "--scene", "nfl1", "--run", "r1",
                        "--only", "up", "--child", "-n"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    assert "== up  [" in r.stdout, r.stdout[:200]
    assert "[全部完成]" not in r.stdout and "\n场景 " not in "\n" + r.stdout


def _drive_go(T, script=f"{SAM3D}/tools/drive_go.sh", **env):
    """在临时的家目录 T 里用假的 tmux 跑一次 drive_go.sh。返回 (它交给 tmux 的那条命令, 跑它用的环境)。"""
    os.makedirs(f"{T}/sam3d/data/outdoor/result/_t_scene/r1/logs")
    os.makedirs(f"{T}/bin")
    open(f"{T}/bin/tmux", "w").write('#!/bin/bash\n[ "$1" = new-session ] && printf "%s\\n" "${@: -1}" > "$FAKE_OUT"\nexit 0\n')
    os.chmod(f"{T}/bin/tmux", 0o755)
    e = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "PARALLEL", "MAX_GPUS", "GPU_POOL")}
    e.update(HOME=T, PATH=f"{T}/bin:" + os.environ["PATH"], FAKE_OUT=f"{T}/cmd", **env)
    r = subprocess.run(["bash", script, "_t_scene", "r1"], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return open(f"{T}/cmd").read(), e


def _drive_go_cmd(script=f"{SAM3D}/tools/drive_go.sh", **env):
    """用假的 tmux 跑一次 drive_go.sh, 返回它交给 tmux 的那条命令(真的驱动不会起来)。"""
    with tempfile.TemporaryDirectory() as T:
        return _drive_go(T, script, **env)[0]


def test_drive_go_log_records_the_real_exit_code():
    """驱动失败了, 日志结尾的"退出码"要是驱动的退出码。以前 $? 取到的是它前面 $(date) 的, 永远是 0 ——
    看板显示不出失败, 09-29 实跑 r3 第 3 段失败时日志里写的也是"退出码 0"。"""
    import glob
    with tempfile.TemporaryDirectory() as T:
        cmd, e = _drive_go(T)
        os.makedirs(f"{T}/miniforge3/envs/sam3d/bin")
        fake = f"{T}/miniforge3/envs/sam3d/bin/python"
        open(fake, "w").write("#!/bin/bash\necho 假的驱动\nexit 7\n"); os.chmod(fake, 0o755)
        subprocess.run(["bash", "-c", cmd], env=e)
        log = open(glob.glob(f"{T}/sam3d/data/outdoor/result/_t_scene/r1/logs/drive_*.log")[0]).read()
        assert "假的驱动" in log and log.strip().splitlines()[-1].endswith("退出码 7"), log


def test_drive_go_does_not_pin_a_card():
    """不钉卡: 每个任务自己领。要明确去掉 CUDA_VISIBLE_DEVICES —— tmux 服务端的环境里可能留着旧的卡号。"""
    cmd = _drive_go_cmd()
    assert "env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID " in cmd, cmd


def test_drive_go_passes_pin_and_switches():
    """想钉卡、想关并行、想改名额: 环境变量要带进 tmux 里的驱动。"""
    cmd = _drive_go_cmd(CUDA_VISIBLE_DEVICES="3", PARALLEL="0", MAX_GPUS="3", FANOUT="1", CPU_THREADS="16")
    assert "env CUDA_VISIBLE_DEVICES=3 CUDA_DEVICE_ORDER=PCI_BUS_ID PARALLEL=0 MAX_GPUS=3 FANOUT=1 CPU_THREADS=16 " in cmd, cmd


def test_indoor_child_mode_and_dry_run():
    base = [sys.executable, f"{SAM3D}/pipeline/indoor_pipeline.py", "--chain", "obj", "--scene", "office", "--run", "r6_pillarfix"]
    r = subprocess.run(base + ["--only", "vlm,da3,up", "-n"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    assert [l.split()[1] for l in r.stdout.splitlines() if l.startswith("== ")] == ["vlm", "da3", "up"]
    r = subprocess.run(base + ["--only", "up", "--child", "-n"], capture_output=True, text=True)
    assert r.returncode == 0 and "== up  [" in r.stdout and "[全部完成]" not in r.stdout, r.stdout[:200]


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
