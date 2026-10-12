#!/usr/bin/env python
"""人体网格的库(pipeline/mesh_store.py) + 出网格那一步(body4d_pipeline.s_mesh)。假的跟踪结果和假的出网格脚本, 不用卡。

规矩(2026-10-12 起):
  * 每个人的目录名 = 他的跟踪编号(挑人卡片上的数字), 不是"第几个勾的人"。
  * 一个人算过、他的输入(抠图 / 帧 / 设置 / 代码)没变 -> 不重算。点掉一个人 = 删他的目录, 别人一个字节不动。
起因(office r12): 目录名按"第几个勾的人"排, 点掉 7 号后 8、9、10 号全部往前挪一个目录, 只能把 9 个人都重出一遍(3 分半),
而且刚在闸口看过的人也被重算, 网格差几毫米。
跑: $PY tools/tests/test_mesh_store.py"""
import json, os, shutil, sys, tempfile, time
import numpy as np
from PIL import Image

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import mesh_store as M

H, W, NF = 20, 30, 3
WHERE = {1: (slice(0, 5), slice(0, 5)), 2: (slice(5, 10), slice(5, 10)), 4: (slice(10, 15), slice(10, 15)),
         7: (slice(15, 20), slice(20, 30)), 9: (slice(0, 5), slice(20, 30))}
FRAMES = {1: [0, 1, 2], 2: [0, 1], 4: [1, 2], 7: [2], 9: [0]}
EXT = {"mesh_4d_individual": "ply", "focal_4d_individual": "json", "mhr_4d_individual": "npz"}
SET = dict(补全="0", batch="8", max_occ_len="25", 焦距=None)


def _trk(T, frames_of=FRAMES):
    """跟踪目录: images/ + masks/(像素值 = 跟踪编号)。"""
    trk = f"{T}/02_body4d"
    os.makedirs(f"{trk}/images"); os.makedirs(f"{trk}/masks")
    for f in range(NF):
        open(f"{trk}/images/{f:08d}.jpg", "w").write(f"帧{f}")
        m = np.zeros((H, W), np.uint8)
        for o, fs in frames_of.items():
            if f in fs:
                m[WHERE[o]] = o
        q = Image.fromarray(m).convert("P"); q.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * 759)
        q.save(f"{trk}/masks/{f:08d}.png")
    return trk


def _part(run, k, ids, want=None, frames_of=FRAMES, tag=""):
    """照 body4d_mesh_5090.py 的样子造一份: 本份里的人重编号成 1..n。want 给了就盖上"算完了"的章。"""
    p = f"{run}/_part{k}"
    rel = {o: i + 1 for i, o in enumerate(ids)}
    for o, l in rel.items():
        for d, ext in EXT.items():
            os.makedirs(f"{p}/{d}/{l}")
            for f in frames_of[o]:
                open(f"{p}/{d}/{l}/{f:08d}.{ext}", "w").write(f"人{o} 帧{f}{tag}")
    os.makedirs(f"{p}/masks"); os.makedirs(f"{p}/rendered_frames")
    json.dump({"relabel": {str(o): l for o, l in rel.items()}}, open(f"{p}/relabel.json", "w"))
    json.dump({"ids": ids, "frames": NF, "completion": False, "batch_size": 8, "peak_vram_gb": 9.0 + k, "seconds": 100.0},
              open(f"{p}/mesh_report.json", "w"))
    if want is not None:
        M.seal(p, {o: want[o] for o in ids})
    return p


def _setup(ids=(1, 2, 4, 7)):
    """一个算好的库: 两份([1, 7] 和 [2, 4])搬进来、收尾。返回 (库, 跟踪目录, 印记)。"""
    T = tempfile.mkdtemp()
    run, trk = f"{T}/04_mesh_occ0", _trk(T)
    want = {o: {"抠图": f"m{o}"} for o in ids}
    M.begin(run)
    for k, grp in enumerate(([1, 7], [2, 4])):
        M.absorb(run, _part(run, k, [o for o in grp if o in ids], want), want)
    M.finish(run, trk, list(ids), new=list(ids), seconds=123.4, parts=2, settings=SET)
    return run, trk, want


def _snap(run, o):
    """这个人的每个文件现在长什么样: (内容, inode, 改动时间)。"""
    out = {}
    for d in EXT:
        for n in sorted(os.listdir(f"{run}/{d}/{o}")):
            s = os.stat(f"{run}/{d}/{o}/{n}")
            out[f"{d}/{n}"] = (open(f"{run}/{d}/{o}/{n}").read(), s.st_ino, s.st_mtime_ns)
    return out


# ---------------------------------------------------------------- 库

def test_people_are_stored_under_their_tracking_id():
    run, trk, want = _setup()
    for d, ext in EXT.items():
        assert sorted(os.listdir(f"{run}/{d}"), key=int) == ["1", "2", "4", "7"], (d, os.listdir(f"{run}/{d}"))
        for o in (1, 2, 4, 7):
            got = sorted(os.listdir(f"{run}/{d}/{o}"))
            assert got == [f"{f:08d}.{ext}" for f in FRAMES[o]], (d, o, got)
            assert open(f"{run}/{d}/{o}/{got[0]}").read().startswith(f"人{o} "), "人搬错了目录"
    assert not [x for x in os.listdir(run) if x.startswith("_part")], os.listdir(run)
    assert os.path.islink(f"{run}/images") and len(os.listdir(f"{run}/images")) == NF
    assert sorted(M.people(run)) == [1, 2, 4, 7]


def test_masks_keep_tracking_ids_and_only_the_picked_people():
    T = tempfile.mkdtemp()
    run, trk = f"{T}/04_mesh_occ0", _trk(T)                       # 跟踪里有 1、2、4、7、9 五个人
    want = {o: {} for o in (1, 4, 7)}
    M.begin(run); M.absorb(run, _part(run, 0, [1, 4, 7], want), want)
    M.finish(run, trk, [1, 4, 7], new=[1, 4, 7], settings=SET)
    m = np.array(Image.open(f"{run}/masks/00000002.png"))         # 第 2 帧: 人 1、4、7 在
    for o in (1, 4, 7):
        assert m[WHERE[o]].min() == m[WHERE[o]].max() == o, (o, np.unique(m[WHERE[o]]))
    assert sorted(np.unique(m)) == [0, 1, 4, 7]
    m0 = np.array(Image.open(f"{run}/masks/00000000.png"))        # 第 0 帧: 跟踪里有 1、2、9; 没勾的 2、9 不能留
    assert sorted(np.unique(m0)) == [0, 1], np.unique(m0)
    assert Image.open(f"{run}/masks/00000002.png").mode == "P" and len(os.listdir(f"{run}/masks")) == NF


def test_reports_describe_the_store():
    run, trk, want = _setup()
    rel = json.load(open(f"{run}/relabel.json"))
    assert rel["relabel"] == {"1": 1, "2": 2, "4": 4, "7": 7}, rel   # 目录名就是跟踪编号: 对照表两边一样, 留着给读它的地方
    rep = json.load(open(f"{run}/mesh_report.json"))
    assert rep["ids"] == [1, 2, 4, 7] and rep["ids_run"] == [1, 2, 4, 7] and rep["frames"] == NF
    assert rep["meshes_per_id"] == {"1": 3, "2": 2, "4": 2, "7": 1}, rep["meshes_per_id"]
    assert rep["peak_vram_gb"] == 10.0 and rep["seconds"] == 123.4 and rep["parts"] == 2
    assert rep["completion"] is False and rep["batch_size"] == 8
    assert rep["重算"] == [1, 2, 4, 7] and rep["沿用"] == []


def test_dropping_a_person_leaves_everyone_else_untouched():
    """闸口上点掉 7 号: 删他的目录就完了, 1、2、4 号的文件不搬、不改、不重写。"""
    run, trk, want = _setup()
    before = {o: _snap(run, o) for o in (1, 2, 4)}
    keep = {o: want[o] for o in (1, 2, 4)}
    M.begin(run)
    gone = M.prune(run, keep)
    M.finish(run, trk, [1, 2, 4], new=[], settings=SET)
    assert list(gone) == [7] and "没勾" in gone[7], gone
    assert {o: _snap(run, o) for o in (1, 2, 4)} == before, "别人的文件被动过"
    for d in EXT:
        assert sorted(os.listdir(f"{run}/{d}"), key=int) == ["1", "2", "4"], os.listdir(f"{run}/{d}")
    assert sorted(M.people(run)) == [1, 2, 4]
    assert 7 not in np.unique(np.array(Image.open(f"{run}/masks/00000002.png")))
    rep = json.load(open(f"{run}/mesh_report.json"))
    assert rep["ids"] == [1, 2, 4] and rep["meshes_per_id"] == {"1": 3, "2": 2, "4": 2}
    assert rep["重算"] == [] and rep["沿用"] == [1, 2, 4] and rep["seconds"] == 0.0
    assert json.load(open(f"{run}/relabel.json"))["relabel"] == {"1": 1, "2": 2, "4": 4}


def test_a_person_whose_inputs_changed_is_thrown_out():
    """2 号的跟踪结果变了(印记对不上): 他的网格不能再用; 别人照留。"""
    run, trk, want = _setup()
    new = dict(want); new[2] = {"抠图": "变了"}
    M.begin(run)
    gone = M.prune(run, new)
    assert list(gone) == [2] and "变了" in gone[2], gone
    assert sorted(M.people(run)) == [1, 4, 7]
    assert not os.path.exists(f"{run}/mesh_4d_individual/2") and not os.path.exists(f"{run}/mhr_4d_individual/2")


def test_dirs_nobody_registered_are_thrown_out():
    """上一次搬到一半崩了: 目录在、登记簿里没有 —— 不知道是谁、是不是完整的, 不能用。"""
    run, trk, want = _setup()
    os.makedirs(f"{run}/mesh_4d_individual/9"); open(f"{run}/mesh_4d_individual/9/00000000.ply", "w").write("半截")
    M.begin(run)
    gone = M.prune(run, {**want, 9: {"抠图": "m9"}})
    assert list(gone) == [9] and "没登记" in gone[9], gone
    assert not os.path.exists(f"{run}/mesh_4d_individual/9") and sorted(M.people(run)) == [1, 2, 4, 7]


def test_old_numbering_is_wiped_not_reused():
    """2026-10-12 之前出的目录: 名字是"第几个勾的人", 也不知道是哪版代码算的 —— 整个清掉重算, 不猜。"""
    T = tempfile.mkdtemp()
    run = f"{T}/04_mesh_occ0"
    for d in list(EXT) + ["rendered_frames", "masks"]:
        os.makedirs(f"{run}/{d}/3" if d in EXT else f"{run}/{d}")
    open(f"{run}/mesh_4d_individual/3/00000000.ply", "w").write("老的 3 号 = 跟踪的 4 号")
    json.dump({"relabel": {"1": 1, "2": 2, "4": 3}}, open(f"{run}/relabel.json", "w"))
    json.dump({"ids": [1, 2, 4]}, open(f"{run}/mesh_report.json", "w"))
    assert M.finished(run) is None and M.people(run) == {}
    assert M.begin(run) is True
    assert not os.path.exists(f"{run}/mesh_4d_individual") and not os.path.exists(f"{run}/rendered_frames")
    assert not os.path.exists(f"{run}/mesh_report.json"), "老的报告留着会被当成这一次的"
    assert M.begin(run) is False                                   # 已经是新写法了: 不再清


def test_store_is_unfinished_until_finish():
    """改到一半(出网格那步失败了)的库不能往下用: begin 之后、finish 之前都算没完成。"""
    run, trk, want = _setup()
    assert M.finished(run) is True
    M.begin(run)
    assert M.finished(run) is False
    assert not os.path.exists(f"{run}/mesh_report.json"), "上一次的报告留着, 失败的这一步会被当成跑完了"
    M.finish(run, trk, [1, 2, 4, 7], new=[], settings=SET)
    assert M.finished(run) is True and os.path.exists(f"{run}/mesh_report.json")


def test_unsealed_part_is_not_absorbed():
    """没盖章的一份 = 那一份没算完(进程被杀 / 失败): 里面有什么都不收, 清掉。"""
    run, trk, want = _setup(ids=(1, 2, 4))
    w9 = {**want, 9: {"抠图": "m9"}}
    p = _part(run, 0, [9])                                         # 没盖章
    M.begin(run)
    assert M.absorb(run, p, w9) == [] and not os.path.exists(p)
    assert sorted(M.people(run)) == [1, 2, 4] and not os.path.exists(f"{run}/mesh_4d_individual/9")


def test_part_people_no_longer_wanted_are_discarded():
    """上次算好没来得及搬的一份里有 7、9 号; 这次 7 号被点掉了, 9 号的跟踪变了: 两个都不收。4 号还要, 收。"""
    run, trk, want = _setup(ids=(1, 2))
    w = {**want, 4: {"抠图": "m4"}, 7: {"抠图": "m7"}, 9: {"抠图": "m9"}}
    p = _part(run, 0, [4, 7, 9], w)
    now = {1: w[1], 2: w[2], 4: w[4], 9: {"抠图": "m9 变了"}}
    M.begin(run)
    assert M.absorb(run, p, now) == [4] and not os.path.exists(p)
    assert sorted(M.people(run)) == [1, 2, 4]
    assert open(f"{run}/mesh_4d_individual/4/00000001.ply").read().startswith("人4 ")


def test_absorbing_replaces_the_previous_version_of_that_person():
    run, trk, want = _setup()
    w = dict(want); w[2] = {"抠图": "新"}
    M.begin(run)
    p = _part(run, 0, [2], w, frames_of={2: [1]}, tag=" 新")
    assert M.absorb(run, p, w) == [2]
    assert os.listdir(f"{run}/mesh_4d_individual/2") == ["00000001.ply"], "上一版的帧混在新的里面"
    assert open(f"{run}/mesh_4d_individual/2/00000001.ply").read().endswith(" 新")
    assert M.people(run)[2]["印记"] == {"抠图": "新"} and M.people(run)[2]["帧"] == 1


def test_absorb_refuses_empty_meshes():
    """2026-09-29 nfl1 r3: 拆开后人 2/6/7 的网格全是空的(292 字节的 ply)。照收的话到 scale 那步才炸, 报的还是看不懂的错。"""
    run, trk, want = _setup(ids=(1, 4))
    w = {**want, 2: {"抠图": "m2"}}
    p = _part(run, 0, [2], w)
    open(f"{p}/mesh_4d_individual/1/00000000.ply", "wb").write(      # 真实的空网格就长这样
        b"ply\nformat binary_little_endian 1.0\ncomment https://github.com/mikedh/trimesh\nelement vertex 0\n"
        b"property float x\nproperty float y\nproperty float z\nelement face 0\n"
        b"property list uchar int vertex_indices\nend_header\n")
    M.begin(run)
    try:
        M.absorb(run, p, w)
    except SystemExit as e:
        assert "空" in str(e) and "2" in str(e), str(e)
    else:
        raise AssertionError("有空网格该失败")
    assert sorted(M.people(run)) == [1, 4] and not os.path.exists(f"{run}/mesh_4d_individual/2")
    assert not os.path.exists(f"{p}/part.json"), "章要撕掉: 不然下次还是收这份空的, 永远过不去"


# ---------------------------------------------------------------- 印记: 一个人的网格由什么决定

def _b4d(T, text="v1"):
    os.makedirs(f"{T}/b4d", exist_ok=True)
    open(f"{T}/b4d/body4d_mesh_5090.py", "w").write(text)
    return f"{T}/b4d"


def test_stamp_is_per_person_and_follows_his_mask():
    T = tempfile.mkdtemp()
    trk, b4d = _trk(T), _b4d(T)
    a = M.stamps(trk, [1, 2, 4], SET, b4d)
    assert sorted(a) == [1, 2, 4] and a[1] != a[2]
    m = np.array(Image.open(f"{trk}/masks/00000001.png")); m[6, 6] = 0           # 2 号在第 1 帧少了一个像素
    q = Image.fromarray(m).convert("P"); q.putpalette([0] * 768); q.save(f"{trk}/masks/00000001.png")
    b = M.stamps(trk, [1, 2, 4], SET, b4d)
    assert b[2] != a[2], "2 号的抠图变了, 印记没变"
    assert b[1] == a[1] and b[4] == a[4], "别人的抠图没变, 印记不该变"
    assert M.stamps(trk, [4, 1], SET, b4d)[1] == b[1], "勾了谁、按什么顺序, 不影响一个人自己的印记"


def test_stamp_does_not_depend_on_where_the_run_lives():
    """把一轮整个复制到别处(验证时常这么干): 内容一样, 印记就一样, 不重算。"""
    T = tempfile.mkdtemp()
    trk, b4d = _trk(T), _b4d(T)
    a = M.stamps(trk, [1, 2], SET, b4d)
    shutil.copytree(trk, f"{T}/copy/02_body4d")
    assert M.stamps(f"{T}/copy/02_body4d", [1, 2], SET, b4d) == a


def test_stamp_follows_settings_code_and_frames():
    T = tempfile.mkdtemp()
    trk, b4d = _trk(T), _b4d(T)
    a = M.stamps(trk, [1, 2], SET, b4d)
    assert M.stamps(trk, [1, 2], dict(SET, batch="4"), b4d)[1] != a[1]            # 批大小影响批内平滑
    assert M.stamps(trk, [1, 2], dict(SET, 焦距=0.61), b4d)[1] != a[1]            # 焦距决定人离镜头多远
    assert M.stamps(trk, [1, 2], SET, _b4d(T, "v2"))[1] != a[1]                   # 出网格的代码改了
    assert M.stamps(trk, [1, 2], SET, _b4d(T, "v1"))[1] == a[1]
    open(f"{trk}/images/00000001.jpg", "w").write("这一帧换了一张图")
    assert M.stamps(trk, [1, 2], SET, b4d)[1] != a[1]


def test_completion_ties_everyone_together():
    """开了补全, 一个人的结果要看别人挡没挡他: 勾的人一变, 所有人的印记都变。关着的时候互不相干。"""
    T = tempfile.mkdtemp()
    trk, b4d = _trk(T), _b4d(T)
    on = dict(SET, 补全="1")
    assert M.stamps(trk, [1, 2, 4], on, b4d)[1] != M.stamps(trk, [1, 2], on, b4d)[1]
    assert M.stamps(trk, [1, 2, 4], SET, b4d)[1] == M.stamps(trk, [1, 2], SET, b4d)[1]


def test_focal_is_what_the_mesh_script_uses_and_survives_the_scale_step():
    """出网格用的是 cameras.json 里各帧 竖直焦距 / 图高 的中位数。尺度那步会改写 cameras.json(外参乘 k、加标记),
    内参不动 —— 不能因此把所有人算成过期(闸口点人是在尺度之后)。"""
    T = tempfile.mkdtemp()
    cams = {"cameras": {str(i): {"intrinsic_3x3": [[600 + i, 0, 344], [0, 610 + i, 193], [0, 0, 1]], "input_hw": [386, 688],
                                 "extrinsic": [[1, 0, 0, i]]} for i in range(5)}}
    json.dump(cams, open(f"{T}/cameras.json", "w"))
    f = M.focal_of(f"{T}/cameras.json")
    assert abs(f - 612 / 386) < 1e-12, f
    for v in cams["cameras"].values():
        v["extrinsic"] = [[1, 0, 0, 1.058 * v["extrinsic"][0][3]]]
    cams["metric_scale_applied"] = 1.058
    json.dump(cams, open(f"{T}/cameras.json", "w"), indent=1)
    assert M.focal_of(f"{T}/cameras.json") == f
    assert M.focal_of("") is None                                   # 没给场景相机: 模型自己估


# ---------------------------------------------------------------- 出网格那一步(body4d_pipeline.s_mesh)
FAKE_MESH = r'''
import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--dir"); ap.add_argument("--out"); ap.add_argument("--ids", default="")
ap.add_argument("--completion"); ap.add_argument("--batch-size"); ap.add_argument("--max-occ-len")
ap.add_argument("--cam-json", default="")
a = ap.parse_args()
ids = [int(x) for x in a.ids.split(",") if x] or json.load(open(f"{a.dir}/picked.json"))["keep_ids"]
open(f"{a.dir}/calls.txt", "a").write(",".join(map(str, ids)) + "\n")
if int(os.environ.get("FAIL_ALWAYS", "-1")) in ids:
    sys.exit(1)
card = os.environ.get("CUDA_VISIBLE_DEVICES")
n = len(open(f"{a.dir}/calls.txt").read().split())
rel = {o: i + 1 for i, o in enumerate(ids)}
for o, l in rel.items():
    for d, ext in (("mesh_4d_individual", "ply"), ("focal_4d_individual", "json"), ("mhr_4d_individual", "npz")):
        os.makedirs(f"{a.out}/{d}/{l}", exist_ok=True)
        open(f"{a.out}/{d}/{l}/00000000.{ext}", "w").write(f"人{o} 卡{card} 第{n}次")
os.makedirs(f"{a.out}/masks", exist_ok=True)
json.dump({"relabel": {str(o): l for o, l in rel.items()}}, open(f"{a.out}/relabel.json", "w"))
json.dump({"ids": ids, "frames": 1, "completion": False, "batch_size": 8, "peak_vram_gb": 9.0, "seconds": 1.0},
          open(f"{a.out}/mesh_report.json", "w"))
'''
IDS, COVER = [1, 2, 4, 7, 9], {"1": 114, "2": 110, "4": 100, "7": 60, "9": 12}


def _pick(c, ids):
    json.dump({"keep_ids": ids, "coverage": {str(i): COVER[str(i)] for i in ids}}, open(f"{c.TRK}/picked.json", "w"))


def _mesh_setup(**env):
    """临时目录里摆好跟踪结果和假的出网格脚本; 3 张假卡, 都空着。返回 (流水线模块, 上下文, 还原环境的函数)。"""
    import _common as C, body4d_pipeline as B
    T = tempfile.mkdtemp()
    keep = {k: os.environ.get(k) for k in ("BODY_OUT", "GPU_POOL", "RUN_DIR", "OCC", "PARALLEL", "FAIL_ALWAYS",
                                           "CUDA_VISIBLE_DEVICES", "SCENE_R1", "MAX_GPUS", "FANOUT")}
    for k in keep:
        os.environ.pop(k, None)
    os.environ.update(dict(BODY_OUT=f"{T}/body4d", GPU_POOL="t71,t72,t73", FANOUT="1"), **env)
    old = (C.card_busy, B.B4D, B.PY)
    C.card_busy = lambda card, need_mb=0: False
    B.B4D = f"{T}/b4d"; os.makedirs(B.B4D)
    open(f"{B.B4D}/body4d_mesh_5090.py", "w").write(FAKE_MESH)
    B.PY = dict(B.PY, body4d=sys.executable, sam3d=sys.executable)
    c = B.Ctx("nfl1", "_t")
    os.makedirs(os.path.dirname(c.TRK), exist_ok=True)
    shutil.move(_trk(tempfile.mkdtemp()), c.TRK)
    _pick(c, IDS)
    json.dump({}, open(f"{c.TRK}/track_report.json", "w"))

    def undo():
        C.card_busy, B.B4D, B.PY = old
        for k, v in keep.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    return B, c, undo


def _who(c):
    """库里每个目录装的是谁、在哪张卡上、第几次调用算的: {目录名: (人, 卡, 第几次)}。"""
    out = {}
    for g in sorted(os.listdir(f"{c.MESH}/mesh_4d_individual"), key=int):
        o, card, n = open(f"{c.MESH}/mesh_4d_individual/{g}/00000000.ply").read().split()
        out[int(g)] = (int(o[1:]), card[1:], n)
    return out


def _calls(c, clear=True):
    p = f"{c.TRK}/calls.txt"
    out = open(p).read().split() if os.path.exists(p) else []
    if clear and os.path.exists(p):
        os.remove(p)
    return out


def test_s_mesh_splits_people_over_cards_and_names_dirs_by_tracking_id():
    B, c, undo = _mesh_setup()
    try:
        B.s_mesh(c, False)
        who = _who(c)
        assert {g: o for g, (o, _, _) in who.items()} == {1: 1, 2: 2, 4: 4, 7: 7, 9: 9}, who     # 目录名 = 跟踪编号
        assert sorted({card for _, card, _ in who.values()}) == ["t71", "t72", "t73"], who       # 三张卡都用上了
        rep = json.load(open(f"{c.MESH}/mesh_report.json"))
        assert rep["ids"] == IDS and rep["parts"] == 3 and rep["重算"] == IDS, rep
        assert not [x for x in os.listdir(c.MESH) if x.startswith("_part")]
        calls = sorted(_calls(c))
        load = [sum(COVER[i] for i in grp.split(",")) for grp in calls]
        assert len(calls) == 3 and max(load) - min(load) <= 60, (calls, load)                    # 按帧数分, 工作量接近
        assert M.finished(c.MESH) is True
    finally:
        undo()


def test_s_mesh_dropping_people_recomputes_nobody():
    """闸口上点掉 2 号和 7 号再出网格: 一次模型都不起, 1、4、9 号的文件原封不动, 还在原来的目录里。"""
    B, c, undo = _mesh_setup()
    try:
        B.s_mesh(c, False); _calls(c)
        before = {o: _snap(c.MESH, o) for o in (1, 4, 9)}
        _pick(c, [1, 4, 9])
        B.s_mesh(c, False)
        assert _calls(c) == [], "点掉人不该起模型"
        assert {o: _snap(c.MESH, o) for o in (1, 4, 9)} == before, "留下的人被动过"
        for d in EXT:
            assert sorted(os.listdir(f"{c.MESH}/{d}"), key=int) == ["1", "4", "9"], (d, os.listdir(f"{c.MESH}/{d}"))
        rep = json.load(open(f"{c.MESH}/mesh_report.json"))
        assert rep["ids"] == [1, 4, 9] and rep["重算"] == [] and rep["沿用"] == [1, 4, 9], rep
        assert sorted(np.unique(np.array(Image.open(f"{c.MESH}/masks/00000000.png")))) == [0, 1, 9]
    finally:
        undo()


def test_s_mesh_adding_a_person_computes_only_him():
    B, c, undo = _mesh_setup()
    try:
        _pick(c, [1, 4, 9]); B.s_mesh(c, False); _calls(c)
        before = {o: _snap(c.MESH, o) for o in (1, 4, 9)}
        _pick(c, [1, 2, 4, 9])
        B.s_mesh(c, False)
        assert _calls(c) == ["2"]
        assert {o: _snap(c.MESH, o) for o in (1, 4, 9)} == before
        assert {g: o for g, (o, _, _) in _who(c).items()} == {1: 1, 2: 2, 4: 4, 9: 9}
        rep = json.load(open(f"{c.MESH}/mesh_report.json"))
        assert rep["重算"] == [2] and rep["沿用"] == [1, 4, 9] and rep["ids"] == [1, 2, 4, 9], rep
    finally:
        undo()


def test_s_mesh_rerun_with_nothing_changed_does_nothing():
    B, c, undo = _mesh_setup()
    try:
        B.s_mesh(c, False); _calls(c)
        before = {o: _snap(c.MESH, o) for o in IDS}
        B.s_mesh(c, False)
        assert _calls(c) == [] and {o: _snap(c.MESH, o) for o in IDS} == before
    finally:
        undo()


def test_s_mesh_a_person_tracked_differently_is_recomputed_alone():
    """跟踪重跑过, 只有 4 号的抠图变了: 只重算 4 号。"""
    B, c, undo = _mesh_setup()
    try:
        B.s_mesh(c, False); _calls(c)
        before, b4 = {o: _snap(c.MESH, o) for o in (1, 2, 7, 9)}, _snap(c.MESH, 4)
        m = np.array(Image.open(f"{c.TRK}/masks/00000001.png")); m[12, 12] = 0
        q = Image.fromarray(m).convert("P"); q.putpalette([0] * 768); q.save(f"{c.TRK}/masks/00000001.png")
        B.s_mesh(c, False)
        assert _calls(c) == ["4"]
        assert {o: _snap(c.MESH, o) for o in (1, 2, 7, 9)} == before
        assert _snap(c.MESH, 4) != b4 and _who(c)[4][0] == 4, "4 号该是这一次新算的"
    finally:
        undo()


def test_s_mesh_code_change_recomputes_everyone():
    B, c, undo = _mesh_setup()
    try:
        B.s_mesh(c, False); _calls(c)
        open(f"{B.B4D}/body4d_mesh_5090.py", "a").write("\n# 改了一行\n")
        B.s_mesh(c, False)
        assert sorted(int(x) for grp in _calls(c) for x in grp.split(",")) == IDS
    finally:
        undo()


def test_s_mesh_failed_part_is_the_only_thing_redone():
    B, c, undo = _mesh_setup(FAIL_ALWAYS="4")
    try:
        try:
            B.s_mesh(c, False)
        except SystemExit as e:
            assert "4" in str(e) and "没算出来的人" in str(e), str(e)
        else:
            raise AssertionError("有一份一直失败, 这一步该失败")
        assert M.finished(c.MESH) is False, "没全算出来, 库不能算完成"
        assert not os.path.exists(f"{c.MESH}/mesh_report.json")
        first = _calls(c)
        bad = [g for g in first if "4" in g.split(",")]
        assert len(bad) == 2 and len(set(bad)) == 1, first          # 失败的那份试了两次(换卡重试一次)
        lost = sorted(int(x) for x in bad[0].split(","))
        del os.environ["FAIL_ALWAYS"]
        B.s_mesh(c, False)                                          # 重跑: 只补那一份里的人, 别的份算好的不重算
        assert sorted(int(x) for grp in _calls(c) for x in grp.split(",")) == lost
        assert {g: o for g, (o, _, _) in _who(c).items()} == {1: 1, 2: 2, 4: 4, 7: 7, 9: 9}
        assert M.finished(c.MESH) is True
    finally:
        undo()


def test_s_mesh_does_not_split_with_completion_on():
    """补全要看别人的遮挡, 不能按人拆开; 勾的人变了也得全部重算。"""
    B, c, undo = _mesh_setup(OCC="1")
    try:
        B.s_mesh(c, False)
        assert _calls(c) == ["1,2,4,7,9"]
        assert len({card for _, card, _ in _who(c).values()}) == 1
        _pick(c, [1, 2, 4, 9])
        B.s_mesh(c, False)
        assert _calls(c) == ["1,2,4,9"], "开着补全时点掉一个人, 别人的遮挡关系变了, 要全部重算"
    finally:
        undo()


def test_s_mesh_does_not_split_when_fanout_is_off():
    B, c, undo = _mesh_setup(FANOUT="0")
    try:
        B.s_mesh(c, False)
        assert _calls(c) == ["1,2,4,7,9"]
        assert {g: o for g, (o, _, _) in _who(c).items()} == {1: 1, 2: 2, 4: 4, 7: 7, 9: 9}      # 不拆也是跟踪编号
    finally:
        undo()


def test_s_mesh_does_not_split_when_parallel_is_off():
    B, c, undo = _mesh_setup(PARALLEL="0")
    try:
        B.s_mesh(c, False)
        assert _calls(c) == ["1,2,4,7,9"]
    finally:
        undo()


def test_s_mesh_copied_run_reuses_everything():
    """把一轮复制到别处再跑出网格(验证时的标准做法): 什么都不重算。"""
    B, c, undo = _mesh_setup()
    try:
        B.s_mesh(c, False); _calls(c)
        new = os.path.dirname(c.R1) + "/copy"
        shutil.copytree(c.R1, new, symlinks=True)
        os.environ["BODY_OUT"] = new
        c2 = B.Ctx("nfl1", "_t")
        B.s_mesh(c2, False)
        assert _calls(c2) == [] and {g: o for g, (o, _, _) in _who(c2).items()} == {1: 1, 2: 2, 4: 4, 7: 7, 9: 9}
    finally:
        undo()


def test_s_mesh_old_layout_is_recomputed_once():
    """10-12 之前的出网格目录(按"第几个勾的人"命名): 第一次碰到整个重算, 之后就是新写法。"""
    B, c, undo = _mesh_setup()
    try:
        _pick(c, [1, 4, 9])
        for g, o in ((1, 1), (2, 4), (3, 9)):                       # 老写法: 4 号在目录 2, 9 号在目录 3
            for d, ext in EXT.items():
                os.makedirs(f"{c.MESH}/{d}/{g}"); open(f"{c.MESH}/{d}/{g}/00000000.{ext}", "w").write(f"人{o} 卡老 第0次")
        json.dump({"relabel": {"1": 1, "4": 2, "9": 3}}, open(f"{c.MESH}/relabel.json", "w"))
        B.s_mesh(c, False)
        assert sorted(int(x) for grp in _calls(c) for x in grp.split(",")) == [1, 4, 9]
        assert {g: o for g, (o, _, _) in _who(c).items()} == {1: 1, 4: 4, 9: 9}, _who(c)
        assert all(card != "老" for _, card, _ in _who(c).values())
        B.s_mesh(c, False)
        assert _calls(c) == []
    finally:
        undo()


def test_s_mesh_leftovers_of_a_killed_run_do_not_leak_in():
    """上一次被杀在半路: 留下一份没盖章的 _part0, 里面是别的人的帧。这次的结果里不能混进去。
    (10-09 nfl1 r8 就是这一类: 新的 5 号目录里留着上一次 5 号的帧, 对账报"身份被写串"。)"""
    B, c, undo = _mesh_setup(FANOUT="0")
    try:
        for d, ext in EXT.items():
            os.makedirs(f"{c.MESH}/_part0/{d}/1")
            open(f"{c.MESH}/_part0/{d}/1/00000007.{ext}", "w").write("人99 卡旧 第0次")
        B.s_mesh(c, False)
        for d in EXT:
            assert os.listdir(f"{c.MESH}/{d}/1") == [f"00000000.{EXT[d]}"], os.listdir(f"{c.MESH}/{d}/1")
        assert _who(c)[1][0] == 1
    finally:
        undo()


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
    print("== test_mesh_store 全过 ==")
