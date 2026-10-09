#!/usr/bin/env python
"""pipeline/track_clean.py: 人体跟踪里"只剩零星几个像素、又接不上这个人完整的帧"的那些帧不算人。
跑: $PY tools/tests/test_track_clean.py"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import track_clean as TC

W = 1920


def _st(area, bbox, c):
    return dict(area=area, bbox=list(bbox), cx=float(c[0]), cy=float(c[1]))


def _blob(cx, cy, w=100, h=200, fill=0.4):
    """一块实心的人形: 包围盒 w x h, 填了 fill 那么多。"""
    return _st(int(w * h * fill), (cx - w // 2, cy - h // 2, cx + w // 2 - 1, cy + h // 2 - 1), (cx, cy))


def _dots(cx, cy, n=200, w=400, h=400):
    """零星 n 个像素, 散在 w x h 的框里。"""
    return _st(n, (cx - w // 2, cy - h // 2, cx + w // 2 - 1, cy + h // 2 - 1), (cx, cy))


# office r8 人 6 在跟踪原始输出里的样子(2026-10-08 实测): 16~23 帧是画面右边缘的一小块, 114 帧是另一处的一块,
# 其余 14 帧是散在坐着的那个人(人 1)轮廓边上的几十到几百个像素 —— 框和人 1 的差不多大, 里面几乎是空的。
# 出网格那步不看像素数, 照样拿这个框重建出一个完整的人 -> 画面里多一个人, 第 268 帧(这里的 74)闪一下。
OFFICE_P6 = {
    16: _st(1908, (1862, 878, 1919, 1021), (1895, 930)), 17: _st(2369, (1855, 878, 1919, 1022), (1884, 944)),
    18: _st(2486, (1835, 882, 1919, 1029), (1872, 961)), 19: _st(2551, (1822, 883, 1919, 1025), (1867, 970)),
    20: _st(2454, (1830, 887, 1919, 1043), (1876, 985)), 21: _st(2407, (1853, 999, 1919, 1055), (1882, 1024)),
    22: _st(1781, (1884, 1003, 1919, 1062), (1903, 1032)), 23: _st(511, (1906, 1014, 1919, 1063), (1914, 1040)),
    56: _st(451, (1289, 738, 1683, 1002), (1514, 882)), 74: _st(237, (1256, 713, 1562, 967), (1375, 788)),
    85: _st(96, (1169, 657, 1623, 1065), (1474, 911)), 93: _st(199, (1175, 643, 1556, 1079), (1353, 966)),
    94: _st(794, (855, 624, 1576, 1079), (1329, 989)), 95: _st(637, (812, 609, 1491, 1079), (1262, 971)),
    96: _st(76, (776, 691, 1327, 1079), (1222, 842)), 97: _st(531, (769, 560, 1501, 1079), (1131, 1014)),
    98: _st(965, (762, 588, 1455, 1079), (1176, 982)), 99: _st(51, (1064, 669, 1476, 1067), (1298, 701)),
    100: _st(172, (815, 618, 1470, 1079), (1180, 999)), 114: _st(8520, (354, 423, 465, 531), (417, 473)),
    117: _st(558, (809, 432, 1088, 1052), (967, 653)), 118: _st(554, (941, 431, 1221, 1047), (1093, 620)),
    119: _st(760, (1109, 433, 1397, 1077), (1268, 692)),
}


def test_stray_pixels_on_someone_else_are_not_this_person():
    """office 人 6: 凭空冒出来的那 14 帧零星像素摘掉(包括闪的第 74 帧); 它自己实心的 9 帧留着。"""
    assert TC.stray_frames(OFFICE_P6, W) == [56, 74, 85, 93, 94, 95, 96, 97, 98, 99, 100, 117, 118, 119]


def test_a_real_person_who_gets_buried_keeps_every_frame():
    """nfl1 人 3: 前 75 帧是完整的人一路跑过来, 之后被压在别人身下只露出几条边(几十到几百个像素), 一直到片尾。
    这些帧接在他完整的帧后面、位置连着 -> 是被挡住, 一帧都不摘(后面的"翻转补帧"靠这些帧知道他还在)。"""
    fr = {t: _blob(94 + 12 * t, 326 + 2 * t) for t in range(75)}
    for t in range(75, 113):                                  # 被压住: 中心慢慢漂, 偶尔露出稍微多一点
        fr[t] = _dots(990 + 5 * (t - 75), 430 - 2 * (t - 75), n=80 + 10 * (t % 5))
    for t in (81, 87, 89, 105):
        fr[t] = _blob(990 + 5 * (t - 75), 430 - 2 * (t - 75), w=60, h=40, fill=0.1)
    assert TC.stray_frames(fr, W) == []


def test_partly_hidden_is_not_thin():
    """被挡住一半、只露上身: 剩下的还是一整块(填了框的两成), 这条规则不碰。"""
    fr = {t: _blob(500, 500, fill=0.2) for t in range(10)}
    assert TC.stray_frames(fr, W) == []


def test_person_emerging_from_behind_someone_keeps_the_early_frames():
    """先只露出几个像素, 然后整个人走出来: 前面那几帧接在后面完整的帧上, 也留着。"""
    fr = {t: _dots(800 + 3 * t, 400) for t in range(6)}
    fr.update({t: _blob(800 + 3 * t, 400) for t in range(6, 30)})
    assert TC.stray_frames(fr, W) == []


def test_never_solid_means_never_a_person():
    """从头到尾都只是零星像素、从来没完整出现过: 全摘掉(调用方会发现这个编号一帧不剩)。"""
    fr = {t: _dots(800, 400) for t in range(20)}
    assert TC.stray_frames(fr, W) == list(range(20))


def test_thin_frames_must_be_close_in_time_and_place():
    """接得上 = 隔得不久(LINK_GAP 帧以内)而且位置对得上(中心相距不到画面宽的 LINK_DIST)。差一样都不算。"""
    solid = {t: _blob(500, 500) for t in range(10)}
    far = dict(solid); far[11] = _dots(500 + int(TC.LINK_DIST * W) + 20, 500)          # 紧接着, 但跑到别处去了
    late = dict(solid); late[9 + TC.LINK_GAP + 1] = _dots(500, 500)                    # 位置对, 但隔太久
    near = dict(solid); near[9 + TC.LINK_GAP] = _dots(520, 505)
    assert TC.stray_frames(far, W) == [11] and TC.stray_frames(late, W) == [9 + TC.LINK_GAP + 1]
    assert TC.stray_frames(near, W) == []


def test_small_but_solid_person_far_away_is_untouched():
    """远处的真人(soccer 守门员占画面 0.69%, 还有更小的): 小, 但是完整的一块。这条规则看的不是大小。"""
    fr = {t: _blob(1500, 300, w=12, h=30, fill=0.5) for t in range(8)}            # 180 个像素的小人
    assert TC.stray_frames(fr, W) == []


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f(); print("ok", name)
