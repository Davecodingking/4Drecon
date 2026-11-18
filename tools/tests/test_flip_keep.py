# FLIP_KEEP: 翻转段里按配置保留的帧(body_motion.parse_flip_keep / flip_keep_split)
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipeline"))
import body_motion as BM

assert BM.parse_flip_keep("") == {}
assert BM.parse_flip_keep("5:501-517") == {"5": [(501, 517)]}
assert BM.parse_flip_keep(" 5:517-501 ; 3:10-20,40-50") == {"5": [(501, 517)], "3": [(10, 20), (40, 50)]}
# 默认空: 原样
assert BM.flip_keep_split([(2, 5)], list(range(10)), []) == [(2, 5)]
# jw4 人 5 的形状: 翻转段 501..532(下标 0..), 保留 501..517 -> 只剩 518..532, 锚是 517
vfs = list(range(495, 533)); i = vfs.index
assert BM.flip_keep_split([(i(501), i(532))], vfs, [(501, 517)]) == [(i(518), i(532))]
# 保留段在中间: 切成两段
assert BM.flip_keep_split([(0, 9)], list(range(100, 110)), [(103, 105)]) == [(0, 2), (6, 9)]
# 全保留: 没有翻转段
assert BM.flip_keep_split([(0, 3)], [7, 8, 9, 10], [(0, 99)]) == []
# 帧号有空档(模型没出的帧不在 bis 里)
assert BM.flip_keep_split([(0, 4)], [516, 517, 518, 527, 528], [(500, 517)]) == [(2, 4)]
print("test_flip_keep ok")

# flip_keep_mask: FLIP_KEEP 的帧 + 两头锚点都是 FLIP_KEEP 帧的补帧(定在保留段最后一帧上的)
bis = [10, 11, 12, 13, 14, 15]; kept = {10: 500, 11: 501, 12: 517, 13: 518, 14: 519, 15: 520}
occ = {"13": [12, 12, 0.0, [0, 0, 0]], "14": [12, 12, 0.0, [0, 0, 0]], "15": [10, 10, 0.0, [0, 0, 0]]}
m = BM.flip_keep_mask(bis, kept, occ, [(501, 517)])
assert m.tolist() == [False, True, True, True, True, False], m
assert not BM.flip_keep_mask(bis, kept, occ, []).any()
print("test_flip_keep mask ok")
