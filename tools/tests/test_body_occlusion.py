#!/usr/bin/env python
"""body_into_scene 的翻转补帧: 人被挡住时 SAM 3D Body 会把挡在前面的人当成他重建。
套得稳(跟着邻居走, soccer r2 人 1)的不动 —— 09-27 按"露出多少"整段换掉, 被补成僵住的动作, 用户否了;
在两个人之间来回认的(守门员扑倒时前面穿过一个近 10 m 的球员)身体朝向一帧翻 30~118°, 这种段换掉:
两头都有好帧 -> 位置按时间插值、动作从前一个好帧过渡到后一个; 只剩一头(片尾) ->
动作定在那个好帧, 位置跟着自己的 mask 走。跑: $PY tools/tests/test_body_occlusion.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_into_scene as B


def _rots(degs):
    """绕竖轴转过 degs[i] 度的一串朝向。"""
    out = []
    for d in degs:
        a = np.radians(d)
        out.append(np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]))
    return out


def test_flip_segments():
    assert B.flip_segments(_rots([0] * 30)) == []
    assert B.flip_segments(_rots([2 * i for i in range(40)])) == []            # 慢慢转身(每帧 2°)不是翻
    assert B.flip_segments(_rots([0] * 20 + [60, 60] + [0] * 20)) == [(20, 21)]  # 中间翻两帧又回来
    # 守门员: 前面稳 20 帧, 之后一路 0/50/0/100... 地乱翻到片尾 -> 从第一次翻开始整段到结尾
    tail = [0] * 20 + [50, 50, 10, 60, 60, 60, 120, 5, 90]
    assert B.flip_segments(_rots(tail)) == [(20, 28)], B.flip_segments(_rots(tail))
    # 两段乱翻中间夹着一段够长的稳定段 -> 分成两段, 稳定段不动
    mid = [0] * 12 + [60, 0, 60] + [0] * 12 + [70, 0] + [0] * 12
    assert B.flip_segments(_rots(mid)) == [(12, 14), (27, 27)], B.flip_segments(_rots(mid))
    assert B.flip_segments(_rots([0, 60, 0, 60, 0, 60])) == []                  # 从头翻到尾: 不知道该信谁, 不动


def _body(n=40):
    """一个 3 顶点的'人', 每帧朝 +x 走 0.1 m, 身体绕竖轴慢慢转。"""
    W, root = {}, {}
    for b in range(n):
        a = 0.02 * b
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        L = np.array([[0, 0.9, 0], [0.3, 0, 0], [0, -0.9, 0.2]]) @ R.T
        root[b] = np.array([0.1 * b, 0, 5.0]); W[b] = L + root[b]
    return W, root


def test_two_sided_gap_interpolates_position_and_pose():
    W, root = _body()
    bad = dict(W); bad[20] = W[20] + [3.0, 0, -2.0]                # 第 20 帧套到了别人身上
    occ = B.occ_plan([b for b in range(40)], [(18, 22)], {b: b / 24 for b in range(40)}, root, None)
    assert sorted(occ) == ["18", "19", "20", "21", "22"] and occ["20"][:2] == [17, 23]
    out = B.fill_occluded(bad, root, occ)
    assert np.allclose(np.asarray(occ["20"][3]), root[20], atol=1e-9)   # 匀速走 -> 插值就是真值
    assert np.abs(out[20] - W[20]).max() < 0.01, np.abs(out[20] - W[20]).max()
    assert out[17] is bad[17] and out[23] is bad[23]               # 好帧不动


def test_one_sided_tail_holds_pose_and_follows_mask():
    W, root = _body()
    t = {b: b / 24 for b in range(40)}
    # 片尾 35~39 不可信; mask 给的位置比真根偏了一个常量(mask 质心不在骨盆上), 补出来要把这个偏差扣掉
    mroot = {b: root[b] + [0, 0.4, 0] for b in range(40)}
    occ = B.occ_plan(list(range(40)), [(35, 39)], t, root, mroot)
    assert occ["39"][:3] == [34, 34, 0.0]
    out = B.fill_occluded(W, root, occ)
    assert np.allclose(np.asarray(occ["39"][3]), root[39])         # 位置跟 mask 走, 偏差按锚点那帧扣掉
    L34 = W[34] - root[34]
    assert np.allclose(out[39] - root[39], L34)                     # 动作定在最后一个好帧
    # mask 那一帧是空的(整个人被挡没了) -> 位置停在上一帧
    mroot[37] = None
    occ = B.occ_plan(list(range(40)), [(35, 39)], t, root, mroot)
    assert occ["37"][3] == occ["36"][3]


def test_whole_clip_bad_is_left_alone():
    W, root = _body(10)
    assert B.occ_plan(list(range(10)), [(0, 9)], {b: b for b in range(10)}, root, None) == {}


def test_apply_corr_replays_occ():
    I = np.eye(4).tolist()
    cams = {f"{i:04d}.jpg": {"extrinsic_w2c_4x4": I} for i in range(3)}
    vid2key, kept = {i: f"{i:04d}.jpg" for i in range(3)}, [0, 1, 2]
    V = np.array([[0.0, 0, 0], [0, 1, 0]])                          # GL 系: 根在原点, 头在 +y
    people = {"1": {0: V + [0, 0, -5], 1: V + [9, 0, -5], 2: V + [2, 0, -5]}}   # 中间那帧套到了别人
    roots = {"1": {b: np.array(people["1"][b][0] * [1, -1, -1]) for b in range(3)}}
    zero = {str(b): [0, 0, 0] for b in range(3)}
    corr = {"1": dict(ratio=1.0, frames=[0, 1, 2], delta=zero,
                      occ={"1": [0, 2, 0.5, [1.0, 0.0, 5.0]]})}
    w = B.apply_corr(people, corr, cams, vid2key, kept, roots)["1"]
    assert np.allclose(w[1][0], [1, 0, 5]) and np.allclose(w[1][1], [1, -1, 5]), w[1]


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
