#!/usr/bin/env python
"""室内的几何那一步(GEOM=vggt_da3): DA3 和 VGGT-Ω 的推理互不依赖, 有第二张空卡就同时跑, 最后再混合。
起因(office r12, 2026-10-12): 两个模型在同一张卡上一前一后(DA3 约 50 秒, 然后 VGGT-Ω 载模型 26 秒 + 推理 25 秒),
整步 1 分 54 秒, 这时另外几张卡是空的。VGGT-Ω 推理只要图; 只有"混合"(形状用它的, 米数和置信度用 DA3 的)才要 DA3 的结果。
假脚本 + 假卡, 不起模型。跑: $PY tools/tests/test_geom_parallel.py"""
import contextlib, importlib.util, io, json, os, sys, tempfile
import numpy as np

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import _common as C


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# ---------------------------------------------------------------- VGGT-Ω 推理结果的存取(vggt_omega_geom.py)

def _pred(n=3, seed=0):
    r = np.random.default_rng(seed)
    return dict(w2c=r.normal(size=(n, 4, 4)), K=r.normal(size=(n, 3, 3)), depth=r.random((n, 6, 8)).astype(np.float32),
                conf_raw=r.random((n, 6, 8)).astype(np.float32), rgb=r.integers(0, 255, (n, 6, 8, 3)).astype(np.uint8),
                t_load=26.3, t_inf=25.2, peak=16.1)


def test_inference_result_survives_the_trip_through_the_cache_file():
    V = _load("vggt_omega_geom", f"{SAM3D}/vggt-omega/tools/vggt_omega_geom.py")
    with tempfile.TemporaryDirectory() as T:
        names, p = ["0000.jpg", "0001.jpg", "0002.jpg"], _pred()
        V.save_infer(f"{T}/c.npz", names, **p)
        q = V.load_infer(f"{T}/c.npz", names)
        assert q is not None and sorted(q) == sorted(p), sorted(q or {})
        for k in ("w2c", "K", "depth", "conf_raw", "rgb"):
            assert q[k].dtype == p[k].dtype and np.array_equal(q[k], p[k]), k        # 一个比特都不能变: 混合出来的结果要和不存盘时一样
        assert (q["t_load"], q["t_inf"], q["peak"]) == (26.3, 25.2, 16.1)


def test_cache_for_a_different_frame_list_is_not_used():
    """DA3 跳过了某些帧(读不了的图): 它用的帧和 VGGT-Ω 提前算的不是同一批。多视角模型换一批帧结果就不一样,
    所以不凑合 —— 当成没有缓存, 混合那步照老办法现算。文件不在 / 写坏了也一样。"""
    V = _load("vggt_omega_geom", f"{SAM3D}/vggt-omega/tools/vggt_omega_geom.py")
    with tempfile.TemporaryDirectory() as T:
        V.save_infer(f"{T}/c.npz", ["0000.jpg", "0001.jpg", "0002.jpg"], **_pred())
        assert V.load_infer(f"{T}/c.npz", ["0000.jpg", "0002.jpg"]) is None
        assert V.load_infer(f"{T}/c.npz", ["0001.jpg", "0000.jpg", "0002.jpg"]) is None, "顺序不同也不行: 位姿是按顺序对的"
        assert V.load_infer(f"{T}/没有.npz", ["0000.jpg"]) is None
        open(f"{T}/bad.npz", "w").write("写到一半")
        assert V.load_infer(f"{T}/bad.npz", ["0000.jpg"]) is None


# ---------------------------------------------------------------- 流水线: 两个模型同时起

FAKE = r'''
import json, os, sys, time
who = os.path.basename(sys.argv[0])[5:-3]                    # fake_da3.py -> da3
a = sys.argv[1:]
t0 = time.time()
if os.environ.get("FAIL_INFER") and "--infer-only" in a:
    sys.exit(3)
time.sleep(1.0)
if "--infer-only" in a:
    open(a[a.index("--cache") + 1], "w").write("推理结果")
if "--pose-into" in a and "--cache" in a:
    assert os.path.exists(a[a.index("--cache") + 1]), "混合时缓存不在"
open(os.environ["CALLS"], "a").write(json.dumps(dict(who=who, args=a, t0=t0, t1=time.time(),
                                                      card=os.environ.get("CUDA_VISIBLE_DEVICES"))) + "\n")
'''


def _setup(busy=False, **env):
    """假的 DA3 / VGGT-Ω 脚本(各睡 1 秒, 记下起止时间和用的卡) + 两张假卡。返回 (流水线模块, 上下文, DA3 命令, 读调用记录的函数, 还原)。"""
    T = tempfile.mkdtemp()
    keep = {k: os.environ.get(k) for k in ("GPU_POOL", "RUN_DIR", "PARALLEL", "CUDA_VISIBLE_DEVICES", "MAX_GPUS", "CALLS", "FAIL_INFER")}
    for k in keep:
        os.environ.pop(k, None)
    os.environ.update(dict(GPU_POOL="t81,t82", CALLS=f"{T}/calls.jsonl"), **env)
    in_p = _load("indoor_pipeline", f"{SAM3D}/pipeline/indoor_pipeline.py")
    old = C.card_busy
    C.card_busy = lambda card, need_mb=0: busy
    for n in ("da3", "vggt"):
        open(f"{T}/fake_{n}.py", "w").write(FAKE)
    in_p.DA3REPO, in_p.VGGT_GEOM = T, f"{T}/fake_vggt.py"
    in_p.PY = dict(in_p.PY, vggt_omega=sys.executable)
    c = in_p.Ctx("office", "_test_geom"); c.R1, c.LOG = f"{T}/r", f"{T}/r/logs"
    os.makedirs(c.LOG)

    def calls():
        p = f"{T}/calls.jsonl"
        return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

    def undo():
        C.card_busy = old
        for k, v in keep.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    return in_p, c, [sys.executable, "fake_da3.py", "--out", f"{T}/r/02_da3"], calls, undo


def _go(in_p, c, da3, dry=False):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        in_p._da3_with_vggt(c, dry, da3, "/frames", f"{c.R1}/02_da3", "120", "1.2")
    return buf.getvalue()


def _kinds(calls):
    return [("da3" if x["who"] == "da3" else "推理" if "--infer-only" in x["args"] else "混合") for x in calls]


def test_both_models_run_at_the_same_time_on_two_cards():
    in_p, c, da3, calls, undo = _setup()
    try:
        _go(in_p, c, da3)
        got = calls(); by = dict(zip(_kinds(got), got))
        assert sorted(by) == ["da3", "推理", "混合"], _kinds(got)
        assert by["推理"]["t0"] < by["da3"]["t1"] and by["da3"]["t0"] < by["推理"]["t1"], "两个没有同时在跑"
        assert by["推理"]["card"] in ("t81", "t82"), by["推理"]["card"]            # 推理另领了一张卡
        assert by["混合"]["t0"] >= max(by["da3"]["t1"], by["推理"]["t1"]) - 0.05, "混合要等两边都完"
        a = by["混合"]["args"]; inf = by["推理"]["args"]
        assert "--cache" in a and a[a.index("--cache") + 1] == inf[inf.index("--cache") + 1]
        assert inf[inf.index("--max-frames") + 1] == "120" and inf[inf.index("--images") + 1] == "/frames"
        assert not os.path.exists(a[a.index("--cache") + 1]), "缓存有几百兆, 混合完要删"
    finally:
        undo()


def test_second_card_is_found_although_the_step_was_pinned_to_its_own():
    """真机上第一次没并行起来(office r14, 2026-10-12): 主循环替 da3 这一步领卡时把 CUDA_VISIBLE_DEVICES 钉成了那一张,
    之后"还有没有空卡"只看得到自己这张, 答案永远是没有 —— 悄悄退回一前一后, 一秒没省。上面几个测试给了 GPU_POOL,
    绕过了钉卡这一环, 所以没测出来。这里照真机的样子: 不给 GPU_POOL, 先由主循环领一张, 再进这一步。"""
    in_p, c, da3, calls, undo = _setup()
    old_smi = C._smi
    try:
        del os.environ["GPU_POOL"]
        C._smi = lambda query, what="gpu": [["t81"], ["t82"]] if query == "index" else []     # 机器上有两张卡
        with contextlib.redirect_stdout(io.StringIO()):
            slot = C.GpuSlot("da3"); slot.__enter__()                                       # 主循环: with GpuSlot(k): fn(c, dry)
        try:
            assert os.environ["CUDA_VISIBLE_DEVICES"] == slot.card and C.gpu_pool() == [slot.card]   # 钉住了
            assert sorted(C.outer_pool()) == ["t81", "t82"], C.outer_pool()
            _go(in_p, c, da3)
        finally:
            slot.__exit__(None, None, None)
        got = calls(); by = dict(zip(_kinds(got), got))
        assert sorted(by) == ["da3", "推理", "混合"], _kinds(got)
        assert by["da3"]["card"] == slot.card and by["推理"]["card"] not in (None, slot.card), (by["da3"]["card"], by["推理"]["card"])
        assert by["推理"]["t0"] < by["da3"]["t1"] and by["da3"]["t0"] < by["推理"]["t1"], "两个没有同时在跑"
        assert C.outer_pool() == C.gpu_pool(), "出了这一步, 卡池恢复原样"
    finally:
        C._smi = old_smi
        undo()


def test_a_card_pinned_by_the_user_is_respected():
    """人从外面钉了一张卡(CUDA_VISIBLE_DEVICES=3 drive_go.sh ...): 整轮只许用这一张, 不能因为想并行就去占别的卡。"""
    in_p, c, da3, calls, undo = _setup()
    try:
        del os.environ["GPU_POOL"]
        os.environ["CUDA_VISIBLE_DEVICES"] = "t81"
        with contextlib.redirect_stdout(io.StringIO()):
            slot = C.GpuSlot("da3"); slot.__enter__()
        try:
            assert C.outer_pool() == ["t81"]
            _go(in_p, c, da3)
        finally:
            slot.__exit__(None, None, None)
        assert _kinds(calls()) == ["da3", "混合"], _kinds(calls())
    finally:
        undo()


def test_one_after_the_other_when_parallel_is_off():
    in_p, c, da3, calls, undo = _setup(PARALLEL="0")
    try:
        _go(in_p, c, da3)
        got = calls()
        assert _kinds(got) == ["da3", "混合"] and "--cache" not in got[1]["args"], got
    finally:
        undo()


def test_one_after_the_other_when_no_second_card_is_free():
    in_p, c, da3, calls, undo = _setup(busy=True)
    try:
        _go(in_p, c, da3)
        got = calls()
        assert _kinds(got) == ["da3", "混合"] and "--cache" not in got[1]["args"], got
    finally:
        undo()


def test_failed_early_inference_falls_back_instead_of_failing_the_step():
    """提前推理是为了快, 不是必须: 它没跑成(卡被抢了 / 显存不够)就照老办法, 混合那步自己现算。"""
    in_p, c, da3, calls, undo = _setup(FAIL_INFER="1")
    try:
        out = _go(in_p, c, da3)
        got = calls()
        assert _kinds(got) == ["da3", "混合"] and "--cache" not in got[1]["args"], got
        assert "现算" in out, out
    finally:
        undo()


def test_dry_run_prints_what_would_really_run():
    in_p, c, da3, calls, undo = _setup()
    try:
        out = _go(in_p, c, da3, dry=True)
        assert calls() == [], "干跑不能真起命令"
        cmds = [l for l in out.splitlines() if l.strip().startswith("$")]
        assert len(cmds) == 3 and "fake_da3.py" in cmds[0] and "--infer-only" in cmds[1] and "--pose-into" in cmds[2], out
    finally:
        undo()


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
    print("== test_geom_parallel 全过 ==")
