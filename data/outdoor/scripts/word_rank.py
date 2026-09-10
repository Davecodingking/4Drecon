#!/usr/bin/env python
"""一个物理物体被几个词同时追到时, 该登记成哪个词。

这张表原来长在 boxes_from_masks.py 里(三维体素去重那段用它选组代表)。2026-09-08 二维
去重(dedup_words.py)也开始按实例合并, 两边都要回答"这一组该挂哪个词" —— 抽出来共用, 不
复制两份: 两份迟早会分叉, 而分叉的后果是二维选了 car、三维想选 taxi, 同一批物体在两处
拿到不同的名字。

★ 名字不是标签, 有实际下游用途: 01_vlm_sizes.json 那张"尺子"是按词给真实高度的
  (plant pot 0.3m / tree 10m, 差 33 倍), 全局尺度标定就靠它。名字选错, 整个场景的
  尺度跟着错 —— 所以同一批物体在全场必须叫同一个名字, 不能一半 container 一半 trash can。

数字小的赢。细词赢粗词 —— 能说"厢货"就别只说"车"。这条是为了防止最泛的词
(car/chair/sign)系统性吃掉细词: 泛词实例多、mask 也往往更完整, 只按"谁抠得全"选胜者
它必然占便宜。表里没有的词一律 2(中间档): 未知词既不该压过明确的细词, 也不该输给泛词。

★ 优先级只在"同一类"里有意义, 跨类比是没有意义的 —— 这就是下面按类分块写的原因,
  以前它是平的一张表(靠源码里的空行分块), 踩过一次坑才写进结构:
  timesquare 有个游走的垃圾 mask table#5 和一辆真车(car#3 / taxi#0)进了同一组,
  table 和 taxi 在平表里都是 1, "细词赢粗词"这条规则对 table vs taxi 根本无话可说,
  于是并列后按实例数比 —— 桌子实例多, 一块垃圾 mask 就这么当上了那辆车的代表。
  分了类之后, table 和 taxi 不可比, 谁也压不过谁, 让位给"哪张 mask 更适合送去生成"。
  不在任何类里的词是中性的, 和谁都能比 (container / trailer / dumpster 那批)。
"""

FAMILIES = {
    "车":   {"taxi": 1, "van": 1, "truck": 1, "bus": 1, "car": 3},
    "桌椅": {"folding chair": 1, "bench": 1, "table": 1, "chair": 3},
    "植物": {"potted plant": 1, "planter": 2, "plant pot": 2, "tree": 2},
    "招牌": {"billboard": 1, "sign": 3},
    "人":   {"pedestrian": 1, "person": 2},
}
RANK   = {w: r for f in FAMILIES.values() for w, r in f.items()}
FAMILY = {w: name for name, f in FAMILIES.items() for w in f}

SUB = "~"          # 派生子类的分隔符: umbrella~1 是 umbrella 的一个子类

def base_word(w):
    """派生子类取母词。umbrella~1 -> umbrella。

    ★ 为什么需要: 词内分组复用是靠"把实例挪进一个新词 umbrella~1, 再整词复用"实现的
      —— 这样分组就是"挪类别 + 复用"两个已有机制的组合, 不用第三套。
      代价是派生词对查表的东西是陌生的(常识高度、优先级表、排除表都是精确匹配),
      所以一律先按原词查, 查不到再退回母词。
    """
    w = str(w)
    return w.split(SUB, 1)[0] if SUB in w else w

def rank(w):     return RANK.get(base_word(w), 2)
def family(w):   return FAMILY.get(base_word(w))          # None = 中性词

def comparable(a, b):
    """两个词的优先级能不能直接比。中性词和谁都能比, 两个具名类之间不能比。"""
    fa, fb = family(a), family(b)
    return fa is None or fb is None or fa == fb


# ---- 自检: 分块表和原来 boxes_from_masks.py 里那张平表必须逐项相等 ----
# 分块只是把"哪些词可比"从注释变成结构, 不该改动任何一个词的优先级数值。
# 这条断言是给以后改表的人看的: 加词的时候顺手把平表也加上, 不然这里当场炸。
_FLAT_2026_09_08 = {"taxi":1,"van":1,"truck":1,"bus":1,"car":3,
                    "folding chair":1,"bench":1,"table":1,"chair":3,
                    "potted plant":1,"planter":2,"plant pot":2,"tree":2,
                    "billboard":1,"sign":3,
                    "pedestrian":1,"person":2}
_got = {w: rank(w) for w in _FLAT_2026_09_08}
assert _got == _FLAT_2026_09_08, f"RANK 表和平表对不上: {_got} != {_FLAT_2026_09_08}"

# 上面那条只验了 rank 的数值, 一点没验分块。把词换到别的类里 rank 值不变, 断言照过,
# 可 comparable 的结果已经变了 —— 而分块的全部意义就在这。补几条:
assert comparable("car", "taxi") and comparable("chair", "folding chair")
assert comparable("plant pot", "potted plant") and comparable("sign", "billboard")
assert not comparable("car", "table"), "车和桌椅必须不可比(否则 timesquare 的垃圾 table mask 会当代表)"
assert not comparable("tree", "car") and not comparable("person", "chair")


