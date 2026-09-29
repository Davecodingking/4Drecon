#!/usr/bin/env python
"""领卡(_common.GpuSlot): 哪张空着用哪张, 用完还。假卡号 + 假的"卡上有没有别人", 不碰真显卡。
跑: $PY tools/tests/test_gpu_slot.py"""
import json, os, sys, tempfile, threading, time

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import _common as C

POOL = ["t91", "t92", "t93"]                    # 假卡号: 锁文件是 /tmp/outdoor_pipeline_gput91.lock, 不和真卡撞
FREE = lambda card, need_mb=0: False


def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
    return old


def _restore(old):
    for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v


def test_lock_file_name_is_unchanged():
    assert C.lock_path_of("3") == "/tmp/outdoor_pipeline_gpu3.lock"
    old = _env(CUDA_VISIBLE_DEVICES="5,6")
    try:
        assert C.lock_path() == "/tmp/outdoor_pipeline_gpu5.lock"
    finally:
        _restore(old)


def test_two_takers_get_different_cards():
    with C.GpuSlot("a", pool=POOL, busy=FREE, set_env=False) as a:
        with C.GpuSlot("b", pool=POOL, busy=FREE, set_env=False) as b:
            assert (a.card, b.card) == ("t91", "t92"), (a.card, b.card)
    with C.GpuSlot("c", pool=POOL, busy=FREE, set_env=False) as c:
        assert c.card == "t91"                                  # 还回去之后又能领到


def test_cards_used_by_others_are_skipped():
    busy = lambda card, need_mb=0: card == "t91"
    with C.GpuSlot("a", pool=POOL, busy=busy, set_env=False) as a:
        assert a.card == "t92"


def test_waits_until_a_card_frees_up():
    """卡全被占着: 等, 不报错; 有卡空出来就拿到。"""
    state = {"n": 0}
    def busy(card, need_mb=0):
        state["n"] += 1
        return state["n"] <= 4                                  # 头两轮(每轮查 2 张)都说有人
    t0 = time.time()
    with C.GpuSlot("a", pool=POOL[:2], busy=busy, set_env=False, poll=0.05) as a:
        assert a.card == "t91" and a.waited >= 0.1, (a.card, a.waited)
    assert time.time() - t0 < 3


def test_one_card_is_shared_in_turn():
    """只有一张卡可用: 第二个等第一个还了再用。不死锁。"""
    order = []
    def work(name, secs):
        with C.GpuSlot(name, pool=["t94"], busy=FREE, set_env=False, poll=0.05):
            order.append((name, "进")); time.sleep(secs); order.append((name, "出"))
    a = threading.Thread(target=work, args=("a", 0.3)); b = threading.Thread(target=work, args=("b", 0.05))
    a.start(); time.sleep(0.1); b.start(); a.join(5); b.join(5)
    assert order == [("a", "进"), ("a", "出"), ("b", "进"), ("b", "出")], order


def test_timeout_is_a_clear_failure():
    try:
        with C.GpuSlot("a", pool=["t95"], busy=lambda c, need_mb=0: True, set_env=False, poll=0.05, timeout=0.2):
            raise AssertionError("不该领到")
    except SystemExit as e:
        assert "t95" in str(e) and "排" in str(e), str(e)


def test_pinned_card_is_the_whole_pool():
    old = _env(CUDA_VISIBLE_DEVICES="t96", GPU_POOL=None)
    try:
        assert C.gpu_pool() == ["t96"]
        os.environ["GPU_POOL"] = "t97,t98"
        assert C.gpu_pool() == ["t97", "t98"]                   # GPU_POOL 说了算
    finally:
        _restore(old)


def test_set_env_and_restore():
    old = _env(CUDA_VISIBLE_DEVICES=None)
    try:
        with C.GpuSlot("a", pool=POOL, busy=FREE) as a:
            assert os.environ["CUDA_VISIBLE_DEVICES"] == a.card and os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
    finally:
        _restore(old)


def test_run_quota_caps_cards_per_run():
    """一轮最多同时用 MAX_GPUS 张: 名额用完, 卡再空也要等。"""
    with tempfile.TemporaryDirectory() as T:
        old = _env(MAX_GPUS="2")
        try:
            assert C.gpus_free_now(run_dir=T, pool=POOL, busy=FREE) == 2
            with C.GpuSlot("a", run_dir=T, pool=POOL, busy=FREE, set_env=False):
                with C.GpuSlot("b", run_dir=T, pool=POOL, busy=FREE, set_env=False):
                    assert C.gpus_free_now(run_dir=T, pool=POOL, busy=FREE) == 0
                    try:
                        with C.GpuSlot("c", run_dir=T, pool=POOL, busy=FREE, set_env=False, poll=0.05, timeout=0.2):
                            raise AssertionError("名额满了不该领到")
                    except SystemExit:
                        pass
            assert C.gpus_free_now(run_dir=T, pool=POOL, busy=FREE) == 2
        finally:
            _restore(old)


def test_marker_file_says_who_uses_which_card():
    with tempfile.TemporaryDirectory() as T:
        with C.GpuSlot("track/w0", run_dir=T, pool=POOL, busy=FREE, set_env=False) as a:
            d = json.load(open(f"{T}/logs/.gpu_{a.card}.json"))
            assert d["步骤"] == "track/w0" and d["卡"] == a.card and d["pid"] == os.getpid(), d
        assert not os.path.exists(f"{T}/logs/.gpu_{a.card}.json")


def test_avoid_prefers_another_card():
    with C.GpuSlot("a", pool=POOL, busy=FREE, set_env=False, avoid=["t91"]) as a:
        assert a.card == "t92"
    with C.GpuSlot("a", pool=["t91"], busy=FREE, set_env=False, avoid=["t91"]) as a:
        assert a.card == "t91"                                  # 只有这一张就还用它


def test_commands_get_a_cpu_thread_cap():
    """255 核的机器上每个进程默认开几百个线程空转: 一个出网格进程占 44~110 个核, 几个一起跑互相拖慢。
    实测限成 8 反而更快(出网格 143s -> 109s), CPU 少用 82 倍。sh() 起的命令默认都带上。"""
    import tempfile
    with tempfile.TemporaryDirectory() as T:
        old = _env(OMP_NUM_THREADS=None, MKL_NUM_THREADS=None, CPU_THREADS=None)
        try:
            show = "echo $OMP_NUM_THREADS $MKL_NUM_THREADS $OPENBLAS_NUM_THREADS $NUMEXPR_NUM_THREADS"
            C.sh(show, f"{T}/a.log")
            assert open(f"{T}/a.log").read().split("\n")[-2] == "8 8 8 8", open(f"{T}/a.log").read()
            os.environ["CPU_THREADS"] = "16"
            C.sh(show, f"{T}/b.log")
            assert open(f"{T}/b.log").read().split("\n")[-2] == "16 16 16 16"
            os.environ["OMP_NUM_THREADS"] = "3"                  # 人明确设了的不动
            C.sh(show, f"{T}/c.log")
            assert open(f"{T}/c.log").read().split("\n")[-2] == "3 16 16 16"
            C.sh(show, f"{T}/d.log", env={"MKL_NUM_THREADS": "5"})   # 步骤自己给的也不动
            assert open(f"{T}/d.log").read().split("\n")[-2] == "3 5 16 16"
        finally:
            _restore(old)


def test_defaults():
    old = _env(MAX_GPUS=None, PARALLEL=None)
    try:
        assert C.max_gpus() == 8 and C.parallel_on() is True and C.cpu_threads() == 8
        os.environ["PARALLEL"] = "0"
        assert C.parallel_on() is False
    finally:
        _restore(old)


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
