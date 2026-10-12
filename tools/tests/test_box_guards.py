#!/usr/bin/env python
"""组装时两条"只看盒子"的规则(data/outdoor/scripts/box_guards.py), 以及生成那步(gen_per_instance.py)用它们提前筛。

起因(office r12, 2026-10-12): 47 个物体逐个生成, 其中 8 个生成完到组装时被扔掉 —— 6 个不到 15 厘米高(苹果、鼠标、笔筒...,
组装的规则: 比这矮就当量错了), 2 张桌子是同一张被量了两遍。这两条规则只看量到的盒子, 生成之前盒子就有了,
所以该在生成之前判: 不改结果(它们本来就进不了场景), 每个白生成约 1 分钟。
规则本身这里不动; 这里管的是"两边用同一份规则"。
跑: $PY tools/tests/test_box_guards.py"""
import json, os, subprocess, sys, tempfile
import numpy as np

SAM3D = os.path.expanduser("~/sam3d")
SCRIPTS = f"{SAM3D}/data/outdoor/scripts"
sys.path.insert(0, SCRIPTS)
import box_guards as G

UP = np.array([0.0, -1.0, 0.0])
# office r13 量到的盒子(06_all_boxes.json, 数字四舍五入到 4 位): 桌 5 和桌 10 是同一张, 桌 7 和桌 8 是同一张
OFFICE = [
    ('table', {"中心": [1.4069, 1.0894, 10.6311], "长轴": [-0.984, 0.0019, 0.1782], "尺寸": [1.5657, 1.4721, 0.853], "底高": -1.5125, "obj_id": 5, "点数": 3747}),
    ('table', {"中心": [2.2771, 0.9379, 10.4918], "长轴": [-0.9901, 0.0019, 0.1401], "尺寸": [1.7734, 1.4624, 0.5295], "底高": -1.201, "obj_id": 10, "点数": 9980}),
    ('table', {"中心": [-0.9385, 0.8511, 8.555], "长轴": [-0.0758, -0.0004, -0.9971], "尺寸": [1.2153, 0.4983, 0.9707], "底高": -1.33, "obj_id": 7, "点数": 9397}),
    ('table', {"中心": [-0.9946, 0.7954, 8.2505], "长轴": [-0.2055, -0.0002, -0.9787], "尺寸": [1.3467, 0.6073, 0.9798], "底高": -1.2789, "obj_id": 8, "点数": 6465}),
    ('table', {"中心": [1.4579, 0.8, 6.0601], "长轴": [-0.2644, 0.001, 0.9644], "尺寸": [1.0154, 0.8434, 0.2688], "底高": -0.9336, "obj_id": 0, "点数": 50769}),
    ('green apple', {"中心": [1.3102, 0.6955, 5.8578], "长轴": [-0.9959, 0.0018, -0.0901], "尺寸": [0.0516, 0.0454, 0.033], "底高": -0.7111, "obj_id": 0, "点数": 751}),
    ('office chair', {"中心": [1.5022, 0.8367, 5.2808], "长轴": [-0.9, 0.0019, 0.4358], "尺寸": [0.5699, 0.2793, 0.6274], "底高": -1.1502, "obj_id": 1, "点数": 60662}),
]


def _boxes(rows=OFFICE):
    out = {}
    for w, b in rows:
        out.setdefault(w, []).append(dict(b))
    return out


# ---------------------------------------------------------------- 绝对守卫

def test_height_guard_is_the_assembly_rule():
    """比 MIN_H 矮、或者最长边超过 MAX_D = 当成量错了。边界和组装里原来的写法一样: 正好等于门槛的算过。"""
    assert G.abs_ok([1.0, 1.0, 0.15], 0.15, 15) and not G.abs_ok([1.0, 1.0, 0.1499], 0.15, 15)
    assert G.abs_ok([15.0, 1.0, 1.0], 0.15, 15) and not G.abs_ok([15.01, 1.0, 1.0], 0.15, 15)
    assert not G.abs_ok([1.0, 15.5, 1.0], 0.15, 15), "最长边是三个数里最大的, 不只看第一个"
    assert not G.abs_ok(OFFICE[5][1]["尺寸"], 0.15, 15) and G.abs_ok(OFFICE[6][1]["尺寸"], 0.15, 15)   # 苹果 3 厘米 / 椅子 63 厘米


def test_thresholds_default_to_what_assembly_always_used():
    assert (G.MIN_H, G.MAX_D, G.DUP_OVERLAP, G.DUP_DH) == (0.15, 15.0, 0.5, 0.12), (G.MIN_H, G.MAX_D, G.DUP_OVERLAP, G.DUP_DH)
    assert G.abs_ok([1.0, 1.0, 0.2]) and not G.abs_ok([1.0, 1.0, 0.1])          # 不给门槛 = 用上面这套


# ---------------------------------------------------------------- 同一件量了两遍

def test_office_tables_measured_twice_keep_the_one_with_more_points():
    dup = G.dup_boxes(_boxes(), UP, on=True)
    assert set(dup) == {("table", 5), ("table", 8)}, dup
    assert dup[("table", 5)][0] == "table#10" and dup[("table", 8)][0] == "table#7", dup     # 留点数多的那个
    assert all(ov >= 0.5 for _, ov in dup.values()), dup


def test_no_dedup_when_switched_off():
    """室外默认关着(DEDUP_BOX=0): 一个都不判。"""
    assert G.dup_boxes(_boxes(), UP, on=False) == {}


def test_only_boxy_furniture_is_deduped():
    """只判长宽比本来就千差万别的方盒子家具(桌、柜、架)。椅子叠在一起量到两个盒子, 不当成同一把。"""
    a, b = dict(OFFICE[2][1], obj_id=1), dict(OFFICE[3][1], obj_id=2)
    assert G.dup_boxes({"office chair": [a, b]}, UP, on=True) == {}
    assert set(G.dup_boxes({"table": [a, b]}, UP, on=True)) == {("table", 2)}
    assert set(G.dup_boxes({"table~1": [a, b]}, UP, on=True)) == {("table~1", 2)}, "闸口上分出来的子类按母词算"


def test_different_top_heights_are_two_things():
    """水平位置重合但顶面差得多(桌子上方还有个架子): 两件东西。"""
    lo, hi = dict(OFFICE[2][1], obj_id=1), dict(OFFICE[3][1], obj_id=2)
    hi["底高"] = lo["底高"] + 0.5
    assert G.dup_boxes({"table": [lo, hi]}, UP, on=True) == {}


def test_excluded_words_are_left_alone():
    assert G.dup_boxes(_boxes(), UP, exclude={"table"}, on=True) == {}


def test_reason_says_why_it_would_never_be_placed():
    boxes = _boxes()
    dup = G.dup_boxes(boxes, UP, on=True)
    by = {(w, b["obj_id"]): b for w, bs in boxes.items() for b in bs}
    why = G.never_placed("green apple", by["green apple", 0], dup)
    assert why and "0.03" in why and "0.15" in why, why
    why = G.never_placed("table", by["table", 5], dup)
    assert why and "table#10" in why and "同一件" in why, why
    assert G.never_placed("office chair", by["office chair", 1], dup) is None
    assert G.never_placed("table", by["table", 10], dup) is None


# ---------------------------------------------------------------- 立在地上的盒子: 量到的高度不算数

DESK4 = {"中心": [0.5, 0.3, 4.0], "长轴": [1.0, 0.0, 0.0], "尺寸": [1.2, 0.6, 0.06], "底高": -0.85, "obj_id": 4, "点数": 2000}
REST = {"desk": "floor", "monitor stand": "floor", "green apple": "surface", "table": "floor", "office chair": "floor"}


def test_floor_standing_things_may_be_taller_than_measured():
    """office r12 的 desk#4: 腿被椅子挡住, 只量到 6 厘米厚的桌面。组装会把"立在地上"的类补到地面(0.06 -> 0.74 m),
    补完就过得了绝对守卫、摆进场景。所以这类盒子不能按量到的高度提前判死刑 —— 第一版就这么误杀了它。"""
    assert G.rests_on_floor("desk", REST) and G.rests_on_floor("desk~1", REST), "闸口上分出来的子类按母词算"
    assert not G.rests_on_floor("green apple", REST) and not G.rests_on_floor("没见过的词", REST)
    assert not G.rests_on_floor("desk", REST, hung={"desk"}), "人工改判成挂着的, 不补"
    assert G.never_placed("desk", DESK4, {}, may_grow=True) is None
    assert "0.06" in G.never_placed("desk", DESK4, {}, may_grow=False)


def test_too_big_is_final_even_for_floor_standing_things():
    """补到地面只会让盒子变高, 最长边超过 MAX_D 的补完还是超过: 这条随时可以判。"""
    big = dict(DESK4, 尺寸=[16.0, 0.6, 0.06])
    assert "16.0" in G.never_placed("desk", big, {}, may_grow=True)


# ---------------------------------------------------------------- 组装和生成用的是同一份

def test_assembly_takes_its_rules_from_this_module():
    """组装脚本是顶层脚本(import 就整个执行), 没法在测试里调; 所以看它是不是从 box_guards 拿的规则、自己不再另写一份。"""
    src = open(f"{SCRIPTS}/assemble_scene.py").read()
    assert "import box_guards" in src
    for own in ('os.environ.get("MIN_H"', 'os.environ.get("MAX_D"', 'os.environ.get("DUP_OVERLAP"', 'os.environ.get("DUP_DH"',
                'os.environ.get("DEDUP_BOX"'):
        assert own not in src, f"assemble_scene.py 还在自己读 {own}: 两边的门槛会漂"
    assert "box_guards.abs_ok(" in src and "box_guards.dup_boxes(" in src
    assert "box_guards.rests_on_floor(" in src, "哪些盒子会被补到地面, 也得是同一个判法"


def test_absolute_guard_is_checked_before_looking_for_a_mesh():
    """没生成的物体在"为什么没进场景"里的原因要还是"绝对"(盒子量错了 / 太矮), 不能变成"没有 mesh"。"""
    src = open(f"{SCRIPTS}/assemble_scene.py").read()
    loop = src[src.index("solo_used=0"):]
    assert loop.index('"绝对"') < loop.index("mk=pick_mesh("), "绝对守卫要排在找 mesh 之前"


# ---------------------------------------------------------------- 生成那步的清单

def _scene(T, boxes):
    """假的一轮: 盒子 + 每个实例一帧抠图 + 一张图 + 天的方向。返回 (R1, FR)。"""
    import cv2
    from pycocotools import mask as mask_util
    R1, FR = f"{T}/r1", f"{T}/frames"
    for d in (f"{R1}/03_masks", f"{R1}/02_da3", f"{R1}/logs", FR):
        os.makedirs(d)
    json.dump(boxes, open(f"{R1}/06_all_boxes.json", "w"), ensure_ascii=False)
    json.dump({"up": UP.tolist(), "ground_h": -1.5}, open(f"{R1}/02_da3/world_up.json", "w"))
    json.dump(REST, open(f"{R1}/01_vlm_rest.json", "w"))
    cv2.imwrite(f"{FR}/0000.jpg", np.full((60, 80, 3), 128, np.uint8))
    m = np.zeros((60, 80), np.uint8); m[10:50, 10:70] = 1                       # 2400 像素, 不贴边
    rle = mask_util.encode(np.asfortranarray(m))["counts"]
    for w, bs in boxes.items():
        meta = np.array([(b["obj_id"], 0, int(m.sum()), 0.0) for b in bs],
                        dtype=[("oid", "<i4"), ("frame", "<i4"), ("area", "<i8"), ("edge", "<f4")])
        cs = np.empty(len(bs), object); cs[:] = [rle] * len(bs)
        np.savez(f"{R1}/03_masks/{w.replace(' ', '_')}.npz", meta=meta, counts=cs, hw=np.array([60, 80]), word=w)
    return R1, FR


def _gen_list(boxes, **env):
    """跑一遍 gen_per_instance.py(HOME 指到临时目录, 抠图不会写进真的 my_images/)。返回 (清单, 打印)。"""
    T = tempfile.mkdtemp()
    R1, FR = _scene(T, boxes)
    e = dict(os.environ, HOME=T, TAG="t_r1", R1=R1, FR=FR, SOLO="*", SOLO_MIN_PX="100", REST=f"{R1}/01_vlm_rest.json", **env)
    for k in ("TH", "SOLO_TH", "SOLO_KEEP", "EXCLUDE", "BOXES", "MIN_H", "MAX_D", "FLOOR_EXTEND", "HUNG_LABELS"):
        if k not in env:
            e.pop(k, None)
    r = subprocess.run([sys.executable, f"{SCRIPTS}/gen_per_instance.py"], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return sorted(open(f"{R1}/logs/solo_jobs.txt").read().split()), r.stdout


def test_generation_list_leaves_out_what_assembly_would_throw_away():
    jobs, out = _gen_list(_boxes(), DEDUP_BOX="1")
    assert jobs == ["t_r1_office_chair_1", "t_r1_table_0", "t_r1_table_10", "t_r1_table_7"], jobs
    for name, why in (("green apple#0", "0.15"), ("table#5", "table#10"), ("table#8", "table#7")):
        line = [l for l in out.splitlines() if "不生成" in l and name in l]
        assert len(line) == 1 and why in line[0], (name, out)
    assert "3 个" in [l for l in out.splitlines() if "组装时" in l and "不生成" in l and "共" in l][0], out


def test_generation_list_follows_the_dedup_switch():
    """去重关着(室外默认)时组装不会扔那两张桌子, 所以也得照常生成; 太矮的那条不归这个开关管。"""
    jobs, out = _gen_list(_boxes(), DEDUP_BOX="0")
    assert jobs == ["t_r1_office_chair_1", "t_r1_table_0", "t_r1_table_10", "t_r1_table_5", "t_r1_table_7", "t_r1_table_8"], jobs


def test_generation_list_follows_a_changed_height_threshold():
    """门槛是配置(MIN_H): 调低到 2 厘米, 苹果就该生成了 —— 生成和组装看的是同一个数。"""
    jobs, _ = _gen_list(_boxes(), DEDUP_BOX="1", MIN_H="0.02")
    assert "t_r1_green_apple_0" in jobs, jobs


def test_generation_list_keeps_floor_standing_things_that_assembly_will_extend():
    """室内开着"补到地面"(FLOOR_EXTEND=1): 只量到桌面的 desk#4 组装时会补高、摆进场景, 必须照常生成。苹果是放在台面上的, 照筛。"""
    boxes = _boxes(OFFICE + [("desk", DESK4)])
    jobs, out = _gen_list(boxes, DEDUP_BOX="1", FLOOR_EXTEND="1")
    assert "t_r1_desk_4" in jobs and "t_r1_green_apple_0" not in jobs, jobs
    assert not [l for l in out.splitlines() if "不生成" in l and "desk#4" in l], out


def test_generation_list_drops_them_when_nothing_will_extend_them():
    """室外不补地面(FLOOR_EXTEND 没开): 组装看到的就是量到的 6 厘米, 一定扔, 所以不生成。人工改判成"挂着的"也一样。"""
    boxes = _boxes(OFFICE + [("desk", DESK4)])
    jobs, out = _gen_list(boxes, DEDUP_BOX="1")
    assert "t_r1_desk_4" not in jobs and [l for l in out.splitlines() if "不生成" in l and "desk#4" in l], out
    jobs, _ = _gen_list(boxes, DEDUP_BOX="1", FLOOR_EXTEND="1", HUNG_LABELS="desk")
    assert "t_r1_desk_4" not in jobs, jobs


# ---------------------------------------------------------------- 流水线把开关传给生成那步

def _load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _solo_cmd(mod, fn, scene, recall):
    """干跑 solo 那一步, 返回它打印的 gen_per_instance.py 那条命令。"""
    import contextlib, io
    with tempfile.TemporaryDirectory() as R1:
        c = mod.Ctx(scene, "_test_guards"); c.R1, c.LOG = R1, f"{R1}/logs"
        c.cfg["RECALL"] = recall
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            getattr(mod, fn)(c, True)
    return next(l for l in buf.getvalue().splitlines() if "gen_per_instance.py" in l)


def test_indoor_solo_gets_the_same_dedup_switch_as_assembly():
    in_p = _load("indoor_pipeline", f"{SAM3D}/pipeline/indoor_pipeline.py")
    assert "DEDUP_BOX=1" in _solo_cmd(in_p, "o_solo", "office", "0")


def test_dedup_is_not_prefiltered_while_recall_is_on():
    """开着召回时组装读的是并进了召回物体的盒子文件, 多出来的盒子可能改掉"谁和谁是同一件"的结论。
    生成那步读不到那份文件(召回排在它后面), 所以这时不做去重预筛 —— 宁可白生成, 不能少生成。太矮那条照筛(只看盒子自己)。"""
    in_p = _load("indoor_pipeline", f"{SAM3D}/pipeline/indoor_pipeline.py")
    out_p = _load("outdoor_pipeline", f"{SAM3D}/pipeline/outdoor_pipeline.py")
    assert "DEDUP_BOX=0" in _solo_cmd(in_p, "o_solo", "office", "1")
    assert "DEDUP_BOX=0" in _solo_cmd(out_p, "s_solo", "soccer", "1")
    assert "DEDUP_BOX" not in _solo_cmd(out_p, "s_solo", "soccer", "0"), "室外召回关着: 不另传, 两个脚本读的是同一个环境变量"


def test_indoor_solo_knows_what_assembly_will_extend_to_the_floor():
    """组装补不补地面由三个设置决定(开关 / 哪些类立在地上 / 人工改判成挂着的): 生成那步要拿到同样的三个。"""
    in_p = _load("indoor_pipeline", f"{SAM3D}/pipeline/indoor_pipeline.py")
    cmd = _solo_cmd(in_p, "o_solo", "office", "0")
    assert "FLOOR_EXTEND=1" in cmd and "01_vlm_rest.json" in cmd and "HUNG_LABELS=" in cmd, cmd


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
    print("== test_box_guards 全过 ==")
