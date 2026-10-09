#!/usr/bin/env python
"""pipeline/people_identity.py: 跟踪给的"编号"不等于"人" —— 放进场景之前按房间里的位置核对一遍身份。
跑: $PY tools/tests/test_people_identity.py"""
import json, os, sys, tempfile
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import people_identity as ID

DT = 0.15          # 秒/帧(office 约 6.7 fps)


def _track(frames, pos, dt=DT):
    """{帧: (秒, 位置)}; pos 给一个点 = 一直站在那儿。"""
    return {int(b): (b * dt, np.asarray(pos, float)) for b in frames}


def _ids(people):
    return {p: sorted({r for r, _ in items}) for p, items in people.items()}


def test_one_person_split_into_three_ids_becomes_one():
    """office r8/r10 的人 2: 走出画面又回来, 跟踪把他拆成了 2 号(前两段)、6 号的一帧(只露了个头)、10 号(最后 5 帧)。
    位置都在同一处、从不同时出现 -> 合成一个人 2。6 号更早的那一帧是画面边上的一只手(离谁都远) -> 不是人, 去掉。"""
    tr = {"2": {**_track(range(0, 14), (5, 0, 5)), **_track(range(37, 58), (5.4, 0, 5))},
          "6": {**_track([19], (11, 0, 5)), **_track([114], (5.2, 0, 5))},
          "10": _track(range(115, 120), (5.25, 0, 5)),
          "4": _track(range(0, 120), (8, 0, 2))}                       # 一直在场的另一个人
    people, log = ID.resolve(tr)
    assert set(people) == {"2", "4"}, _ids(people)
    assert people["2"] == sorted([("2", b) for b in list(range(0, 14)) + list(range(37, 58))] + [("6", 114)]
                                 + [("10", b) for b in range(115, 120)], key=lambda x: x[1])
    assert people["4"] == [("4", b) for b in range(120)]
    kinds = sorted((x["事"], x["编号"]) for x in log)
    assert kinds == [("不算人", "6"), ("并进别人", "10"), ("并进别人", "6")], log


def test_people_piled_on_one_spot_at_the_same_time_stay_separate():
    """nfl1 叠在一起的人: 位置一样, 但同一时刻都在场 -> 各是各的, 一个都不并。"""
    tr = {str(i): _track(range(0, 100), (10 + 0.1 * i, 0, 20), dt=1 / 24) for i in range(1, 5)}
    people, log = ID.resolve(tr)
    assert _ids(people) == {str(i): [str(i)] for i in range(1, 5)} and log == []


def test_seen_together_even_once_means_two_people():
    """两个编号站的地方一样, 只要有一帧同时在场, 就是两个人。"""
    tr = {"1": _track(range(0, 51), (3, 0, 3)), "2": _track(range(50, 100), (3.1, 0, 3))}
    people, log = ID.resolve(tr)
    assert _ids(people) == {"1": ["1"], "2": ["2"]} and log == []


def test_long_tracks_of_one_id_stay_one_person_even_after_moving():
    """同一个编号断开过、回来时换了地方(走开又回来): 两段都够长, 信跟踪的, 还是一个人。"""
    tr = {"3": {**_track(range(0, 20), (2, 0, 2)), **_track(range(60, 80), (7, 0, 2))}}
    people, log = ID.resolve(tr)
    assert people == {"3": [("3", b) for b in list(range(0, 20)) + list(range(60, 80))]} and log == []


def test_a_one_frame_reappearance_somewhere_else_is_not_trusted():
    """同一个编号隔了很久在别处冒出一两帧(office 人 6: 先是一只手, 14 秒后是别人的头): 位置对不上就不认这一两帧。
    位置对得上的(人还坐在原地, 只是中间被挡住了)照样认。"""
    far = {"5": {**_track(range(0, 9), (2, 0, 2)), **_track([98], (6, 0, 2))}}
    people, log = ID.resolve(far)
    assert people == {"5": [("5", b) for b in range(9)]}, people
    assert [(x["事"], x["编号"], x["帧"]) for x in log] == [("不算人", "5", [98])], log
    near = {"5": {**_track(range(0, 9), (2, 0, 2)), **_track([98], (2.2, 0, 2))}}
    people, log = ID.resolve(near)
    assert people == {"5": [("5", b) for b in list(range(9)) + [98]]} and log == []


def test_fewer_than_three_frames_is_not_a_person():
    """核对完只剩一两帧的不算人(一只手、一个头被重建成的"人"); 三帧的留着(nfl1 人 8 片尾才进画面, 只有 3 帧)。"""
    tr = {"7": _track([10, 11], (1, 0, 1)), "8": _track([10, 11, 12], (4, 0, 4))}
    people, log = ID.resolve(tr)
    assert set(people) == {"8"} and [(x["事"], x["编号"]) for x in log] == [("不算人", "7")]


def test_merging_never_joins_two_people_who_were_seen_together():
    """A、B 同时在场(两个人)。后来冒出的 C 离两人都近: 只能并给其中一个(近的那个), 不能把 A、B 连成一个人。"""
    tr = {"1": _track(range(0, 50), (3.0, 0, 3)), "2": _track(range(0, 50), (3.3, 0, 3)), "9": _track(range(70, 80), (3.25, 0, 3))}
    people, log = ID.resolve(tr)
    assert _ids(people) == {"1": ["1"], "2": ["2", "9"]}, _ids(people)


def test_height_is_ignored_when_up_is_given():
    """只露一个头时模型猜的身体可能高一截低一截: 比位置只看水平方向。"""
    tr = {"2": _track(range(0, 30), (5, 0.9, 5)), "6": _track(range(60, 65), (5.1, 1.6, 5))}
    assert _ids(ID.resolve(tr, up=(0, 1, 0))[0]) == {"2": ["2", "6"]}
    assert _ids(ID.resolve(tr)[0]) == {"2": ["2"], "6": ["6"]}


def test_known_limit_someone_else_taking_the_same_spot_later_is_merged():
    """已知的局限(写在这里免得以后当成没想到): 甲走了, 乙后来正好站到同一个位置、两人从没同框 —— 分不出来, 会并成一个人。
    三个场景(office / nfl1 / soccer)里没有这种情况; 并了谁都记在 identity.json 里, 人能看到。"""
    tr = {"1": _track(range(0, 30), (5, 0, 5)), "2": _track(range(80, 110), (5.1, 0, 5))}
    assert _ids(ID.resolve(tr)[0]) == {"1": ["1", "2"]}


def _fake_run(T, frames_by_id):
    run = f"{T}/04_mesh_occ0"
    for sub, ext in (("mesh_4d_individual", "ply"), ("focal_4d_individual", "json"), ("mhr_4d_individual", "npz")):
        for rid, bis in frames_by_id.items():
            os.makedirs(f"{run}/{sub}/{rid}")
            for b in bis:
                open(f"{run}/{sub}/{rid}/{b:08d}.{ext}", "w").write(f"{sub} {rid} {b}")
    os.makedirs(f"{run}/masks")
    for b in sorted({b for bis in frames_by_id.values() for b in bis}):
        m = np.zeros((4, 12), np.uint8)
        for rid, bis in frames_by_id.items():
            if b in bis:
                m[:, int(rid)] = int(rid)                              # 每个编号占自己那一列
        im = Image.fromarray(m).convert("P"); im.putpalette([i % 256 for i in range(768)]); im.save(f"{run}/masks/{b:08d}.png")
    return run


def test_resolved_people_are_written_as_a_mesh_run_everyone_reads():
    """核对完的结果落成一个和出网格目录同样结构的目录(文件是链接, 抠图按新编号重画): 后面的步骤都读它, 用的是同一份人。
    没核对过的老轮次照旧读原来的目录; 人体网格重出过而没重新核对 -> 直接报错, 不拿过期的身份往下走。"""
    with tempfile.TemporaryDirectory() as T:
        run = _fake_run(T, {"2": [0, 1, 2], "6": [5, 9], "10": [10, 11]})
        assert ID.mesh_run_of(T) == run                                # 还没核对: 原来的目录
        people = {"2": [("2", 0), ("2", 1), ("2", 2), ("6", 9), ("10", 10), ("10", 11)]}
        out = ID.write(T, run, people, [dict(事="并进别人", 编号="10", 帧=[10, 11], 并进="2", 原因="x")])
        assert out == f"{T}/{ID.DIR}" == ID.mesh_run_of(T) and ID.mesh_run_of(T, resolved=False) == run
        for sub, ext in (("mesh_4d_individual", "ply"), ("focal_4d_individual", "json"), ("mhr_4d_individual", "npz")):
            assert os.listdir(f"{out}/{sub}") == ["2"]
            assert sorted(os.listdir(f"{out}/{sub}/2")) == [f"{b:08d}.{ext}" for b in (0, 1, 2, 9, 10, 11)]
            assert open(f"{out}/{sub}/2/00000009.{ext}").read() == f"{sub} 6 9"          # 内容是 6 号那一帧的
        m = lambda b: np.array(Image.open(f"{out}/masks/{b:08d}.png"))
        assert (m(9)[:, 6] == 2).all() and (m(10)[:, 10] == 2).all() and (m(0)[:, 2] == 2).all()
        assert not m(5).any()                                          # 6 号第 5 帧(那只手)不算人: 抠图里也没有
        meta = json.load(open(f"{out}/identity.json"))
        assert meta["人"] == {"2": {"2": [0, 1, 2], "6": [9], "10": [10, 11]}} and meta["记录"][0]["并进"] == "2"
        os.remove(f"{run}/mesh_4d_individual/2/00000002.ply")          # 人体网格变了(重出过)
        try:
            ID.mesh_run_of(T); assert False
        except SystemExit as e:
            assert "重跑 people" in str(e)


def test_people_are_named_by_their_track_id_not_by_position_in_the_kept_list():
    """闸口上点掉了 3 号: 出网格那步把剩下的 1/2/5/8 重排成 1/2/3/4。核对完的人要叫回 1/2/5/8(挑人卡片上的数字),
    不然点掉一个人, 后面所有人的号都往前挪 —— FBX 里的 P3 和上一版视频里的 3 号就不是同一个人了。"""
    names = {"1": 1, "2": 2, "3": 5, "4": 8}                           # 出网格目录里的编号 -> 跟踪编号
    tr = {"1": _track(range(0, 60), (1, 0, 5)), "2": _track(range(0, 30), (5, 0, 5)),
          "3": _track(range(0, 60), (9, 0, 2)), "4": _track(range(40, 60), (5.2, 0, 5))}   # 4(跟踪 8 号)是 2 号走开又回来
    people, log = ID.resolve(tr, name=names)
    assert list(people) == ["1", "2", "5"], list(people)
    assert _ids(people) == {"1": ["1"], "2": ["2", "4"], "5": ["3"]}  # 成员还是出网格目录里的编号: 找文件用
    assert [(x["事"], x["编号"], x["并进"]) for x in log] == [("并进别人", "8", "2")], log    # 给人看的记录用跟踪编号
    assert ID.resolve(tr)[0].keys() == {"1", "2", "3"}                 # 不给对照(老的出网格目录): 原样


def test_a_new_person_number_never_reuses_a_dropped_track_id():
    """一个编号被拆成了两个人(极少): 2 号先在甲处站了 10 帧; 后来只有 2 帧出现在很远的乙处(接不上甲处那段),
    和 3 号在乙处的 1 帧接上了 —— 乙处这个人出帧最多的编号也是 2, 可"人 2"已经是甲处那位了, 只能另起一个号。
    另起的号要避开所有跟踪编号, 包括闸口上点掉的(taken) —— 不然网页拿这个号去要截图, 要到的是被点掉的那个人。"""
    tr = {"1": _track(range(0, 60), (0, 0, 9)),
          "2": {**_track(range(0, 10), (0, 0, 0)), **_track(range(30, 32), (9, 0, 0))},
          "3": _track([40], (9.1, 0, 0))}
    people, log = ID.resolve(tr, name={"1": 1, "2": 2, "3": 3}, taken=[1, 2, 3, 4, 7])
    assert list(people) == ["1", "2", "8"], list(people)               # 4、7 是点掉的: 新号从 8 起
    assert _ids(people)["8"] == ["2", "3"] and _ids(people)["2"] == ["2"]
    assert ID.resolve(tr)[0].keys() == {"1", "2", "4"}                 # 不知道有谁被点掉时: 只避开在场的编号


def test_track_ids_come_from_the_mesh_step_relabel_file():
    with tempfile.TemporaryDirectory() as T:
        run = _fake_run(T, {"1": [0, 1, 2], "2": [0, 1, 2]})
        assert ID.track_ids(run) == {"1": 1, "2": 2}                   # 老的出网格目录没有 relabel.json: 没重排过
        json.dump(dict(relabel={"4": 1, "9": 2}, note="左=跟踪出的原始 id, 右=送进 SAM 3D Body 的连续 id"), open(f"{run}/relabel.json", "w"))
        assert ID.track_ids(run) == {"1": 4, "2": 9}
        os.makedirs(f"{T}/02_body4d")
        json.dump(dict(keep_ids=[4, 9], dropped={"7": 30, "12": 3}), open(f"{T}/02_body4d/picked.json", "w"))
        assert ID.all_track_ids(T, run) == [4, 7, 9, 12]               # 连点掉的 7、12 也算占着号


def test_swapping_who_was_picked_is_noticed_even_when_frame_counts_match():
    """点掉甲、留下乙, 两人帧数一样: 出网格目录的帧数表一模一样, 但 1 号已经换了人 —— 核对过的身份必须算过期。"""
    with tempfile.TemporaryDirectory() as T:
        run = _fake_run(T, {"1": [0, 1, 2], "2": [0, 1, 2]})
        json.dump(dict(relabel={"1": 1, "2": 2}), open(f"{run}/relabel.json", "w"))
        ID.write(T, run, {"1": [("1", b) for b in range(3)], "2": [("2", b) for b in range(3)]}, [])
        assert ID.mesh_run_of(T) == f"{T}/{ID.DIR}"
        src = ID.source_of(T)
        assert src == dict(来源="04_mesh_occ0", 来源帧数={"1": 3, "2": 3}, 编号对照={"1": 1, "2": 2})
        json.dump(dict(relabel={"1": 1, "3": 2}), open(f"{run}/relabel.json", "w"))    # 2 号换成了跟踪的 3 号, 帧数没变
        assert ID.source_of(T) != src
        try:
            ID.mesh_run_of(T); assert False
        except SystemExit as e:
            assert "重跑 people" in str(e)


def test_identity_written_before_the_relabel_was_recorded_still_reads():
    """10-09 之前核对的(identity.json 里没有编号对照那一项): 只比有的那两项, 老轮次照旧能读。"""
    with tempfile.TemporaryDirectory() as T:
        run = _fake_run(T, {"1": [0, 1, 2]})
        ID.write(T, run, {"1": [("1", b) for b in range(3)]}, [])
        m = json.load(open(f"{T}/{ID.DIR}/identity.json")); del m["编号对照"]
        json.dump(m, open(f"{T}/{ID.DIR}/identity.json", "w"))
        assert ID.mesh_run_of(T) == f"{T}/{ID.DIR}"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f(); print("ok", name)
