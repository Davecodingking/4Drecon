#!/usr/bin/env python
"""出人网格按人拆开跑之后并回一个目录(pipeline/mesh_merge.py)。假的几份: 小图 + 空网格文件。
跑: $PY tools/tests/test_mesh_merge.py"""
import json, os, sys, tempfile
import numpy as np
from PIL import Image

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import mesh_merge as M

H, W, NF = 20, 30, 3
WHERE = {1: (slice(0, 5), slice(0, 5)), 2: (slice(5, 10), slice(5, 10)), 4: (slice(10, 15), slice(10, 15)),
         7: (slice(15, 20), slice(20, 30))}


def _part(root, k, ids, frames_of):
    """照 body4d_mesh_5090.py 的样子造一份: 本份里的人重编号成 1..n。"""
    p = f"{root}/_part{k}"
    rel = {o: i + 1 for i, o in enumerate(ids)}
    for o, l in rel.items():
        for d, ext in (("mesh_4d_individual", "ply"), ("focal_4d_individual", "json"), ("mhr_4d_individual", "npz"),
                       ("rendered_frames_individual", "jpg")):
            os.makedirs(f"{p}/{d}/{l}")
            for f in frames_of[o]:
                # 真实输出里每人的渲染图文件名带本份里的编号: rendered_frames_individual/<编号>/00000000_<编号>.jpg
                name = f"{f:08d}_{l}.{ext}" if d == "rendered_frames_individual" else f"{f:08d}.{ext}"
                open(f"{p}/{d}/{l}/{name}", "w").write(f"人{o} 帧{f}")
    os.makedirs(f"{p}/masks"); os.makedirs(f"{p}/rendered_frames")
    for f in range(NF):
        m = np.zeros((H, W), np.uint8)
        img = np.full((H, W, 3), 255, np.uint8)
        for o, l in rel.items():
            if f in frames_of[o]:
                m[WHERE[o]] = l
                img[WHERE[o]] = (10 * o, 20, 30)
        q = Image.fromarray(m).convert("P"); q.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * 759); q.save(f"{p}/masks/{f:08d}.png")
        Image.fromarray(img).save(f"{p}/rendered_frames/{f:08d}.jpg", quality=95)
    json.dump({"relabel": {str(o): l for o, l in rel.items()}}, open(f"{p}/relabel.json", "w"))
    json.dump({"ids": ids, "frames": NF, "completion": False, "batch_size": 8, "peak_vram_gb": 9.0 + k, "seconds": 100.0},
              open(f"{p}/mesh_report.json", "w"))
    return p


def _setup():
    T = tempfile.mkdtemp()
    run, trk = f"{T}/04_mesh_occ0", f"{T}/02_body4d"
    os.makedirs(f"{trk}/images")
    for f in range(NF):
        open(f"{trk}/images/{f:08d}.jpg", "w").close()
    frames_of = {1: [0, 1, 2], 2: [0, 1], 4: [1, 2], 7: [2]}
    _part(run, 0, [1, 7], frames_of); _part(run, 1, [2, 4], frames_of)
    return run, trk, frames_of


def test_merged_layout_matches_an_unsplit_run():
    run, trk, frames_of = _setup()
    M.merge(run, trk, [1, 2, 4, 7], 2, seconds=123.4)
    G = {1: 1, 2: 2, 4: 3, 7: 4}                                 # 全体统一编号: 按挑人的顺序 1..N
    for o, g in G.items():
        for d, ext in (("mesh_4d_individual", "ply"), ("focal_4d_individual", "json"), ("mhr_4d_individual", "npz"),
                       ("rendered_frames_individual", "jpg")):
            got = sorted(os.listdir(f"{run}/{d}/{g}"))
            want = [f"{f:08d}_{g}.{ext}" if d == "rendered_frames_individual" else f"{f:08d}.{ext}" for f in frames_of[o]]
            assert got == want, (d, o, got)                      # 文件名里的编号也换成全体统一的
            assert open(f"{run}/{d}/{g}/{got[0]}").read().startswith(f"人{o} "), "人搬错了目录"
    assert sorted(os.listdir(f"{run}/mesh_4d_individual")) == ["1", "2", "3", "4"]
    assert not [x for x in os.listdir(run) if x.startswith("_part")], os.listdir(run)
    assert os.path.islink(f"{run}/images") and len(os.listdir(f"{run}/images")) == NF


def test_masks_use_the_global_numbering():
    run, trk, frames_of = _setup()
    M.merge(run, trk, [1, 2, 4, 7], 2)
    m = np.array(Image.open(f"{run}/masks/00000002.png"))        # 第 2 帧: 人 1、4、7 在, 人 2 不在
    assert m[WHERE[1]].min() == m[WHERE[1]].max() == 1
    assert m[WHERE[4]].min() == m[WHERE[4]].max() == 3
    assert m[WHERE[7]].min() == m[WHERE[7]].max() == 4
    assert (m[WHERE[2]] == 0).all() and sorted(np.unique(m)) == [0, 1, 3, 4]
    assert Image.open(f"{run}/masks/00000002.png").mode == "P"


def test_rendered_frames_are_composited():
    run, trk, frames_of = _setup()
    M.merge(run, trk, [1, 2, 4, 7], 2)
    img = np.array(Image.open(f"{run}/rendered_frames/00000001.jpg")).astype(int)    # 第 1 帧: 人 1(第 0 份)、2 和 4(第 1 份)
    for o in (1, 2, 4):                                          # JPEG 在小块边上有振铃, 只判"这里有人(暗)"
        assert img[WHERE[o]].mean() < 128, (o, img[WHERE[o]].mean())
    assert img[WHERE[7]].mean() > 230                            # 人 7 这一帧不在: 还是白底


def test_reports_are_merged():
    run, trk, frames_of = _setup()
    M.merge(run, trk, [1, 2, 4, 7], 2, seconds=123.4)
    rel = json.load(open(f"{run}/relabel.json"))
    assert rel["relabel"] == {"1": 1, "2": 2, "4": 3, "7": 4}, rel
    rep = json.load(open(f"{run}/mesh_report.json"))
    assert rep["ids"] == [1, 2, 4, 7] and rep["ids_run"] == [1, 2, 3, 4] and rep["frames"] == NF
    assert rep["meshes_per_id"] == {"1": 3, "2": 2, "3": 2, "4": 1}, rep["meshes_per_id"]
    assert rep["peak_vram_gb"] == 10.0 and rep["seconds"] == 123.4 and rep["parts"] == 2


def test_old_outputs_are_cleared_first():
    """上一次挑了 5 个人, 这次只挑 4 个: 第 5 个人的旧目录不能留着。"""
    run, trk, frames_of = _setup()
    os.makedirs(f"{run}/mesh_4d_individual/5"); open(f"{run}/mesh_4d_individual/5/00000000.ply", "w").close()
    M.merge(run, trk, [1, 2, 4, 7], 2)
    assert sorted(os.listdir(f"{run}/mesh_4d_individual")) == ["1", "2", "3", "4"]


def test_missing_person_is_a_failure():
    run, trk, frames_of = _setup()
    try:
        M.merge(run, trk, [1, 2, 4, 7, 9], 2)
    except SystemExit as e:
        assert "9" in str(e), str(e)
    else:
        raise AssertionError("少了一个人该失败")
    assert os.path.isdir(f"{run}/_part0/mesh_4d_individual/1"), "失败时不能动各份的结果"


# ---------------------------------------------------------------- 出网格那一步的拆分(body4d_pipeline.s_mesh)
FAKE_MESH = r'''
import argparse, json, os, sys
import numpy as np
from PIL import Image
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
rel = {o: i + 1 for i, o in enumerate(ids)}
for o, l in rel.items():
    for d, ext in (("mesh_4d_individual", "ply"), ("focal_4d_individual", "json"), ("rendered_frames_individual", "jpg")):
        os.makedirs(f"{a.out}/{d}/{l}", exist_ok=True)
        name = "00000000" + (f"_{l}" if d.startswith("rendered") else "") + "." + ext
        open(f"{a.out}/{d}/{l}/{name}", "w").write(f"人{o} 卡{card}")
os.makedirs(f"{a.out}/masks", exist_ok=True); os.makedirs(f"{a.out}/rendered_frames", exist_ok=True)
m = np.zeros((16, 16), np.uint8)
for o, l in rel.items():
    m[o, :] = l
Image.fromarray(m).convert("P").save(f"{a.out}/masks/00000000.png")
Image.fromarray(np.full((16, 16, 3), 255, np.uint8)).save(f"{a.out}/rendered_frames/00000000.jpg")
json.dump({"relabel": {str(o): l for o, l in rel.items()}}, open(f"{a.out}/relabel.json", "w"))
json.dump({"ids": ids, "frames": 1, "completion": False, "batch_size": 8, "peak_vram_gb": 9.0, "seconds": 1.0},
          open(f"{a.out}/mesh_report.json", "w"))
'''
IDS, COVER = [1, 2, 4, 7, 9], {"1": 114, "2": 110, "4": 100, "7": 60, "9": 12}


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
    os.makedirs(f"{c.TRK}/images"); open(f"{c.TRK}/images/00000000.jpg", "w").close()
    json.dump({"keep_ids": IDS, "coverage": COVER}, open(f"{c.TRK}/picked.json", "w"))
    json.dump({}, open(f"{c.TRK}/track_report.json", "w"))

    def undo():
        C.card_busy, B.B4D, B.PY = old
        for k, v in keep.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    return B, c, undo


def _who(c):
    """合并后每个目录里是谁、在哪张卡上算的: {统一编号: (原 id, 卡)}。"""
    out = {}
    for g in sorted(os.listdir(f"{c.MESH}/mesh_4d_individual"), key=int):
        o, card = open(f"{c.MESH}/mesh_4d_individual/{g}/00000000.ply").read().split()
        out[int(g)] = (int(o[1:]), card[1:])
    return out


def test_s_mesh_splits_people_over_cards_and_merges():
    B, c, undo = _mesh_setup()
    try:
        B.s_mesh(c, False)
        who = _who(c)
        assert {g: o for g, (o, _) in who.items()} == {1: 1, 2: 2, 3: 4, 4: 7, 5: 9}, who    # 按挑人的顺序 1..N
        assert sorted({card for _, card in who.values()}) == ["t71", "t72", "t73"], who      # 三张卡都用上了
        rep = json.load(open(f"{c.MESH}/mesh_report.json"))
        assert rep["ids"] == IDS and rep["parts"] == 3, rep
        assert not [x for x in os.listdir(c.MESH) if x.startswith("_part")]
        calls = sorted(open(f"{c.TRK}/calls.txt").read().split())
        load = [sum(COVER[i] for i in grp.split(",")) for grp in calls]
        assert len(calls) == 3 and max(load) - min(load) <= 60, (calls, load)                # 按帧数分, 工作量接近
    finally:
        undo()


def test_s_mesh_rerun_only_redoes_the_failed_part():
    B, c, undo = _mesh_setup(FAIL_ALWAYS="4")
    try:
        try:
            B.s_mesh(c, False)
        except SystemExit as e:
            assert "4" in str(e) and "没算出来的人" in str(e), str(e)
        else:
            raise AssertionError("有一份一直失败, 这一步该失败")
        assert not os.path.isdir(f"{c.MESH}/mesh_4d_individual"), "没全算出来就不该合并"
        first = open(f"{c.TRK}/calls.txt").read().split()
        bad = [g for g in first if "4" in g.split(",")]
        assert len(bad) == 2 and len(set(bad)) == 1, first          # 失败的那份试了两次(换卡重试一次)
        os.remove(f"{c.TRK}/calls.txt"); del os.environ["FAIL_ALWAYS"]
        B.s_mesh(c, False)                                          # 重跑: 只补没算出来的那一份
        assert open(f"{c.TRK}/calls.txt").read().split() == [bad[0]], open(f"{c.TRK}/calls.txt").read()
        assert {g: o for g, (o, _) in _who(c).items()} == {1: 1, 2: 2, 3: 4, 4: 7, 5: 9}
    finally:
        undo()


def test_s_mesh_does_not_split_with_completion_on():
    """补全要看别人的遮挡, 不能按人拆开。"""
    B, c, undo = _mesh_setup(OCC="1")
    try:
        B.s_mesh(c, False)
        assert open(f"{c.TRK}/calls.txt").read().split() == ["1,2,4,7,9"]
        assert len({card for _, card in _who(c).values()}) == 1
    finally:
        undo()


def test_s_mesh_does_not_split_by_default():
    """2026-09-29 nfl1 实测按人拆分改变结果(差 1~3 厘米), 默认不拆; FANOUT=1 才拆。"""
    B, c, undo = _mesh_setup(FANOUT="0")
    try:
        B.s_mesh(c, False)
        assert open(f"{c.TRK}/calls.txt").read().split() == ["1,2,4,7,9"]
    finally:
        undo()


def test_remeshing_without_split_does_not_keep_the_previous_people():
    """闸口上点掉一个人再出网格、而且这次没拆(没有空卡 / FANOUT=0): 上一次的产物要先清掉。
    2026-10-09 nfl1 r8: 8 个人重出成 7 个, 新的 5 号(跟踪 6 号, 66 帧)目录里还留着上一次 5 号的 48 帧 —— 别人的网格
    混进了这个人; 第 8 个人的目录也还在。对账那步报"身份被写串"拦住了, 但整个"点掉人再继续"就走不通。
    (拆开跑的那条路合并时本来就清, 见 test_old_outputs_are_cleared_first。)"""
    B, c, undo = _mesh_setup(FANOUT="0")
    try:
        B.s_mesh(c, False)                                          # 第一次: 5 个人
        for d, ext in (("mesh_4d_individual", "ply"), ("focal_4d_individual", "json"), ("mhr_4d_individual", "npz")):
            os.makedirs(f"{c.MESH}/{d}/2", exist_ok=True)
            open(f"{c.MESH}/{d}/2/00000007.{ext}", "w").write("上一次 2 号多出来的一帧")
        open(f"{c.MESH}/masks/00000007.png", "w").write("上一次的抠图")
        json.dump({"keep_ids": [1, 4, 9], "coverage": {"1": 114, "4": 100, "9": 12}}, open(f"{c.TRK}/picked.json", "w"))
        B.s_mesh(c, False)                                          # 点掉 2 号和 7 号, 重出: 3 个人
        assert {g: o for g, (o, _) in _who(c).items()} == {1: 1, 2: 4, 3: 9}, _who(c)
        for d in ("mesh_4d_individual", "focal_4d_individual"):
            assert sorted(os.listdir(f"{c.MESH}/{d}")) == ["1", "2", "3"], (d, os.listdir(f"{c.MESH}/{d}"))
            assert len(os.listdir(f"{c.MESH}/{d}/2")) == 1          # 新的 2 号(跟踪 4 号)只有自己这一次的帧
        assert not os.path.exists(f"{c.MESH}/mhr_4d_individual") and os.listdir(f"{c.MESH}/masks") == ["00000000.png"]
    finally:
        undo()


def test_merge_refuses_empty_meshes():
    """2026-09-29 nfl1 r3: 拆开后人 2/6/7 的网格全是空的(292 字节的 ply) —— 这几个人有"模型拒绝出网格"的帧,
    那一帧里没有别人时上游从此全出空网格。以前合并照收, 到 scale 那步才炸, 报的还是看不懂的错。"""
    run, trk, frames_of = _setup()
    open(f"{run}/_part1/mesh_4d_individual/1/00000000.ply", "wb").write(      # 真实的空网格就长这样
        b"ply\nformat binary_little_endian 1.0\ncomment https://github.com/mikedh/trimesh\nelement vertex 0\n"
        b"property float x\nproperty float y\nproperty float z\nelement face 0\n"
        b"property list uchar int vertex_indices\nend_header\n")
    try:
        M.merge(run, trk, [1, 2, 4, 7], 2)
    except SystemExit as e:
        assert "空" in str(e) and "2" in str(e) and "FANOUT" in str(e), str(e)   # 原 id 2(第 1 份里的 1 号)
    else:
        raise AssertionError("有空网格该失败")
    assert os.path.isdir(f"{run}/_part0/mesh_4d_individual/1"), "失败时不能动各份的结果"


def test_s_mesh_does_not_split_when_parallel_is_off():
    B, c, undo = _mesh_setup(PARALLEL="0")
    try:
        B.s_mesh(c, False)
        assert open(f"{c.TRK}/calls.txt").read().split() == ["1,2,4,7,9"]
    finally:
        undo()


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
