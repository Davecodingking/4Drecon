#!/usr/bin/env python
"""步骤内部拆分(pipeline/fanout.py): 同一个脚本起 K 份, 每份算一部分, 各用一张卡。假卡 + 假任务。
跑: $PY tools/tests/test_fanout.py"""
import json, os, sys, tempfile, threading, time

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import fanout as F

FREE = lambda card, need_mb=0: False
KW = dict(pool=["t81", "t82", "t83"], busy=FREE, poll=0.05)
os.environ["FANOUT"] = "1"      # 拆分默认是关的(见 test_split_is_off_by_default), 这里的测试测的是打开以后


def test_split_balances_by_weight():
    """nfl1 的 8 个人, 出现的帧数差很多(114 ~ 12): 按帧数分, 每份的工作量接近。"""
    ids, w = [1, 2, 3, 4, 5, 6, 7, 8], [114, 114, 114, 112, 114, 66, 60, 12]
    g = F.split(ids, w, 3)
    assert sorted(x for grp in g for x in grp) == ids and len(g) == 3
    load = [sum(w[ids.index(x)] for x in grp) for grp in g]
    assert max(load) - min(load) <= 66, load                     # 总共 706, 每份 235 上下
    assert all(grp == sorted(grp) for grp in g)                  # 组内保持原来的顺序


def test_split_small_inputs():
    assert F.split([], [], 3) == []                              # 没有人 / 没有词: 空的, 不崩
    assert F.split([7], [50], 3) == [[7]]                        # 只有 1 个: 1 份
    assert F.split([1, 2], [5, 5], 8) == [[1], [2]]              # 份数不超过个数
    assert F.split([1, 2, 3], [1, 1, 1], 0) == [[1, 2, 3]]       # k 给错了也至少 1 份


def test_how_many():
    assert F.how_many(8, **{k: v for k, v in KW.items() if k != "poll"}) == 3    # 3 张空卡
    assert F.how_many(2, **{k: v for k, v in KW.items() if k != "poll"}) == 2    # 只有 2 个人
    assert F.how_many(5, pool=["t81"], busy=lambda c, need_mb=0: True) == 1      # 一张空卡都没有: 1 份, 排队


def test_workers_run_at_the_same_time_on_different_cards():
    seen, lock = [], threading.Lock()
    def job(name):
        def fn(card):
            t0 = time.time(); time.sleep(0.3)
            with lock: seen.append((name, card, t0, time.time()))
        return fn
    res = F.run_workers([(f"w{i}", job(f"w{i}")) for i in range(3)], "track", **KW)
    assert res == {"w0": None, "w1": None, "w2": None}, res
    assert sorted(c for _, c, _, _ in seen) == ["t81", "t82", "t83"], seen
    assert max(t0 for _, _, t0, _ in seen) < min(t1 for _, _, _, t1 in seen), "三份应该有重叠"


def test_failed_worker_retries_on_another_card_and_others_finish():
    cards, lock = {}, threading.Lock()
    def job(name):
        def fn(card):
            with lock: cards.setdefault(name, []).append(card)
            if name == "w1" and len(cards[name]) == 1:
                raise SystemExit("[失败] 退出码 1: 假的")
        return fn
    res = F.run_workers([(f"w{i}", job(f"w{i}")) for i in range(2)], "track", **KW)
    assert res == {"w0": None, "w1": None}, res                  # 重试成功
    assert len(cards["w1"]) == 2 and cards["w1"][0] != cards["w1"][1], cards


def test_worker_that_keeps_failing_is_reported():
    def bad(card): raise RuntimeError("显存炸了")
    def good(card): pass
    res = F.run_workers([("w0", bad), ("w1", good)], "mesh", **KW)
    assert res["w1"] is None and "显存炸了" in res["w0"], res


def test_track_words_skips_finished_ones():
    with tempfile.TemporaryDirectory() as R1:
        json.dump({"vocab": ["goal post", "helmet", "sign"]}, open(f"{R1}/01_vlm_vocab.json", "w"))
        os.makedirs(f"{R1}/03_masks"); open(f"{R1}/03_masks/goal_post.npz", "w").close()
        assert F.track_words(R1, "building, helmet") == ["helmet", "sign", "building"]


def test_merge_reports():
    with tempfile.TemporaryDirectory() as T:
        json.dump({"helmet": {"实例数": 5}}, open(f"{T}/p0.json", "w"), ensure_ascii=False)
        json.dump({"sign": {"实例数": 2}}, open(f"{T}/p1.json", "w"), ensure_ascii=False)
        d = F.merge_reports([f"{T}/p0.json", f"{T}/p1.json", f"{T}/没跑出来.json"], f"{T}/all.json")
        assert d == {"helmet": {"实例数": 5}, "sign": {"实例数": 2}} and json.load(open(f"{T}/all.json")) == d
        assert sorted(os.listdir(T)) == ["all.json"]


def _fake_track(T):
    """假的 track_all.py: 只认环境变量, 给每个词写一个 npz 和一份报告。FAIL_WORD 里的词第一次跑会失败。"""
    p = f"{T}/fake_track.py"
    open(p, "w").write('''
import json, os, sys
R1 = os.environ["R1"]; only = [w for w in os.environ.get("ONLY_WORDS", "").split(",") if w]
vocab = json.load(open(f"{R1}/01_vlm_vocab.json"))["vocab"]
words = [w for w in vocab if not only or w in only]
rep = {}
os.makedirs(f"{R1}/03_masks", exist_ok=True)
for w in words:
    f = f"{R1}/03_masks/{w.replace(' ', '_')}.npz"
    if os.path.exists(f): continue
    flag = f"{R1}/.failed_once_{w}"
    if w == os.environ.get("FAIL_WORD") and not os.path.exists(flag):
        open(flag, "w").close(); sys.exit(1)
    open(f, "w").write(os.environ.get("CUDA_VISIBLE_DEVICES", "?")); rep[w] = {"实例数": 1}
    g = int(os.environ.get("MW_GROUP", "8")); b = [words[i:i + g] for i in range(0, len(words), g)]
    open(f"{R1}/batch_{w}.txt", "w").write(",".join(next(x for x in b if w in x)))   # 这个词和谁一批(按 track_all 的切法)
    open(f"{R1}/count_{w}.txt", "a").write("1")                                       # 写了几次
    open(f"{R1}/env_{w}.txt", "w").write(os.environ.get("MY_FLAG", "没有") + " " + os.environ.get("PATH", "").split(":")[0])
json.dump(rep, open(os.environ.get("TRACK_REPORT", f"{R1}/03_track_report.json"), "w"), ensure_ascii=False)
''')
    return p


def _track_run(words, **env):
    T = tempfile.mkdtemp()
    json.dump({"vocab": words}, open(f"{T}/01_vlm_vocab.json", "w"))
    F.fan_track(sys.executable, _fake_track(T), dict(R1=T, FR="x", EXTRA_WORDS="", **env), T, f"{T}/logs/03_track.log",
                False, **KW)
    cards = {w: open(f"{T}/03_masks/{w.replace(' ', '_')}.npz").read() for w in words}
    return T, cards, json.load(open(f"{T}/03_track_report.json"))


def test_fan_track_splits_whole_batches_over_cards():
    """6 个词每批 2 个 = 3 批, 3 张空卡 -> 3 份, 每份一整批; 报告并成一份且保持词序。"""
    words = ["goal post", "helmet", "sign", "jersey", "football", "gloves"]
    T, cards, rep = _track_run(words, MW_GROUP="2")
    assert sorted(set(cards.values())) == ["t81", "t82", "t83"], cards          # 三张卡都用上了
    assert cards["goal post"] == cards["helmet"] and cards["sign"] == cards["jersey"], cards   # 同一批在同一张卡
    assert list(rep) == words, rep                                              # 报告并成了一份, 词序和不拆一样
    assert not [f for f in os.listdir(T) if ".part" in f], os.listdir(T)


def test_fan_track_batches_match_the_unsplit_run():
    """拆开以后每份自己切的批必须和不拆时切的一模一样 —— 批的组成一变, 跟踪结果就变(09-29 实测)。
    7 个词每批 3 个: 不拆是 [0,1,2] [3,4,5] [6]; 限 2 份时第二份拿到 [3,4,5,6], 自己再切还是 [3,4,5] [6]。"""
    words = [f"w{i}" for i in range(7)]
    want = {w: ",".join(words[i:i + 3]) for i in range(0, 7, 3) for w in words[i:i + 3]}
    for parts in (None, 2):
        T = tempfile.mkdtemp()
        json.dump({"vocab": words}, open(f"{T}/01_vlm_vocab.json", "w"))
        F.fan_track(sys.executable, _fake_track(T), dict(R1=T, FR="x", EXTRA_WORDS="", MW_GROUP="3"), T,
                    f"{T}/logs/03_track.log", False, max_parts=parts, **KW)
        got = {w: open(f"{T}/batch_{w}.txt").read() for w in words}
        assert got == want, (parts, got)
        cards = {w: open(f"{T}/03_masks/{w}.npz").read() for w in words}
        assert len(set(cards.values())) == (3 if parts is None else 2), (parts, cards)


def test_fan_track_retry_redoes_the_whole_share():
    """一份里第二个词失败: 重试时第一个词也重来(npz 先删掉) —— 否则剩下的词会重新分批, 批一变结果就变。"""
    T, cards, rep = _track_run(["helmet", "sign", "jersey"], FAIL_WORD="sign", MW_GROUP="2")
    assert set(cards) == {"helmet", "sign", "jersey"}, cards
    assert open(f"{T}/count_helmet.txt").read() == "11", open(f"{T}/count_helmet.txt").read()   # 写了两次
    assert cards["helmet"] == cards["sign"], cards                             # 重试那次两个词同一张卡
    assert open(f"{T}/count_jersey.txt").read() == "1"                        # 另一份不受影响


def test_fan_track_names_the_words_that_never_came_out():
    T = tempfile.mkdtemp()
    json.dump({"vocab": ["helmet", "sign"]}, open(f"{T}/01_vlm_vocab.json", "w"))
    bad = f"{T}/bad.py"; open(bad, "w").write("import sys; sys.exit(2)")
    try:
        F.fan_track(sys.executable, bad, dict(R1=T, FR="x", EXTRA_WORDS="", MW_GROUP="1"), T, f"{T}/logs/03_track.log",
                    False, **KW)
    except SystemExit as e:
        assert "helmet" in str(e) and "sign" in str(e), str(e)
    else:
        raise AssertionError("该失败")


def test_fan_track_nothing_to_do():
    T = tempfile.mkdtemp()
    json.dump({"vocab": ["helmet"]}, open(f"{T}/01_vlm_vocab.json", "w"))
    os.makedirs(f"{T}/03_masks"); open(f"{T}/03_masks/helmet.npz", "w").close()
    F.fan_track(sys.executable, _fake_track(T), dict(R1=T, FR="x", EXTRA_WORDS=""), T, f"{T}/logs/03_track.log",
                False, **KW)                                                    # 词都跟踪过了: 照常跑一份, 不崩


def test_fan_track_max_parts_one_means_no_split():
    T = tempfile.mkdtemp()
    json.dump({"vocab": ["helmet", "sign", "jersey"]}, open(f"{T}/01_vlm_vocab.json", "w"))
    F.fan_track(sys.executable, _fake_track(T), dict(R1=T, FR="x", EXTRA_WORDS="", MW_GROUP="1"), T,
                f"{T}/logs/03_track.log", False, max_parts=1, **KW)
    cards = {open(f"{T}/03_masks/{w}.npz").read() for w in ("helmet", "sign", "jersey")}
    assert len(cards) == 1, cards                                               # 一份, 一张卡


def test_fan_track_passes_extra_env_to_every_worker():
    """室内每条命令都要带 CUDA_HOME 那一套环境变量(它自己的 sh 包了一层), 拆不拆都得带到。"""
    for parts in (None, 1):
        T = tempfile.mkdtemp()
        json.dump({"vocab": ["helmet", "sign"]}, open(f"{T}/01_vlm_vocab.json", "w"))
        F.fan_track(sys.executable, _fake_track(T), dict(R1=T, FR="x", EXTRA_WORDS="", MW_GROUP="1"), T,
                    f"{T}/logs/03_track.log", False, max_parts=parts, sh_env={"MY_FLAG": "带到了", "CUDA_HOME": "/假的/cuda"}, **KW)
        for w in ("helmet", "sign"):
            assert open(f"{T}/env_{w}.txt").read() == "带到了 /假的/cuda/bin", (parts, open(f"{T}/env_{w}.txt").read())


def test_track_split_does_not_need_fanout_switch():
    """FANOUT 只管出人网格按人拆(默认关, 会改结果)。track 按批拆结果不变, 不要这个开关也拆;
    一批装得下的词(默认 8 个一批)不拆。"""
    old = os.environ.pop("FANOUT")
    try:
        assert F.on() is False
        T = tempfile.mkdtemp()
        json.dump({"vocab": ["helmet", "sign", "jersey"]}, open(f"{T}/01_vlm_vocab.json", "w"))
        F.fan_track(sys.executable, _fake_track(T), dict(R1=T, FR="x", EXTRA_WORDS="", MW_GROUP="1"), T,
                    f"{T}/logs/03_track.log", False, **KW)
        assert len({open(f"{T}/03_masks/{w}.npz").read() for w in ("helmet", "sign", "jersey")}) == 3
        T = tempfile.mkdtemp()
        json.dump({"vocab": ["helmet", "sign", "jersey"]}, open(f"{T}/01_vlm_vocab.json", "w"))
        F.fan_track(sys.executable, _fake_track(T), dict(R1=T, FR="x", EXTRA_WORDS=""), T,
                    f"{T}/logs/03_track.log", False, **KW)
        assert len({open(f"{T}/03_masks/{w}.npz").read() for w in ("helmet", "sign", "jersey")}) == 1
    finally:
        os.environ["FANOUT"] = old
    assert F.on() is True


def test_track_all_reads_the_two_env_vars():
    txt = open(f"{SAM3D}/data/outdoor/scripts/track_all.py").read()
    assert 'os.environ.get("ONLY_WORDS"' in txt and 'os.environ.get("TRACK_REPORT"' in txt


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
