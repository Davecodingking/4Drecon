#!/usr/bin/env python
"""跨词去重(二维版): 同一批物体被几个同义词各抠一遍, 在生成之前就把多余的词判掉。

起因: VLM 列词表时会把同一个东西说成好几种说法, SAM3 又对每个说法各跟踪一遍。
beach 上 shop / storefront / restaurant 抠的是同一批 5 栋楼, hot dog cart / hot dog
stand / kiosk 是同一辆热狗车; timesquare 上 plant pot / planter 是同一批花盆,
chair / folding chair 是同一批椅子。三份 mask 一路往下走, 就是三次 SAM3D 生成
(每次 1-2 分钟 GPU)、三个 mesh、三套盒子。

为什么这一步要排在生成之前
--------------------------
原来的跨词去重在 boxes_from_masks.py 里, 用三维体素占用判重叠 —— 但 boxes 排在 gen
之后(它要拿 mesh 的长宽高比例来选词), 于是同义词已经各烧过一次 GPU 才发现是重复的,
去掉的只是"盒子"的重, mesh 白生成了。

为什么改用二维
--------------
判"是不是同一个东西"在同一张照片的像素里比就够了, 不必经过深度和尺度 —— 而深度和尺度
恰恰是这条链上最不可靠的两环(beach 的深度尺度实测差过 6.4 倍, WildDet3D 抬出来的盒子
高度在 carpark 上整体差 2.37 倍)。同一条思路 2026-09-08 刚在 merge_recall.py 上用过
一次: 那里把召回的去重从三维距离改成二维框 IoU, 假漏检从 46 降到 4。
这里比 merge_recall 还稳一档: 那边是"框 vs 框", 这里两个词的 mask 都是 SAM3 在同一帧
同一批像素上画的, 可以直接比 mask 本身。

合并的单位是实例, 不是词
------------------------
2026-09-08 改的。原来是"判定两个词是同义词 -> 输的那个词整个淘汰", 粒度太粗:
car 有 20 个实例、taxi 有 3 个, 三辆出租车确实同时是车 —— 判掉 taxi 这个词就丢了出租车的
形态, 判掉 car 就丢了另外 17 辆车, 两边都不对。真实的关系是"这个 taxi 实例和那个 car 实例
是同一辆车", 所以现在把所有词的所有实例摊平(bin#0、trash can#3 …), 两两比同帧 mask,
判为重复的连边, 并查集连成组, 一组 = 一个物理物体, 组里留一个代表。
逐实例合并之后不存在"整个词消失", 词级那道配对率安全阀(分母取 max 实例数)也就没有把守的
东西了 —— 数字照算, 只当调门槛时的证据写进 json。

不做什么
--------
* 只连"重复"边, 绝不连"部件"边。部件关系(树冠 vs 整盆植物: 单向包含高、小的明显比大的小)
  只记录不合并 —— 把部件边也 union 进来实测过: timesquare 出现一个九元组(4 个 billboard
  + 4 个 sign + 1 栋楼, 楼是所有招牌的父体, 一条条部件边把它们串成了一坨), plant pot
  30 个实例最后一个不剩。
  部件该并进父体还是该单独摆, 得看三维盒子, 这一步看不出来。
* 两个词从来没在同一帧同时出现过的, 这一步看不见 —— 没有可比的像素。beach 上 planter
  只活在第 312 帧、potted plant 只活在第 48~98 帧, 一帧都不重叠。这种死角留给
  boxes_from_masks.py 里那段三维体素去重去捞, 那段因此没有删。
* 不动 03_masks 里的任何 npz。输的词的 mask 原样留着 —— 判错了只要删掉这个 json
  就全恢复了, 而删掉的 mask 要重跑一次 SAM3 跟踪才能拿回来。

输出 $R1/03_dedup_words.json, 下游(pick_from_masks / boxes_from_masks)读它的 "淘汰实例"
(词 -> 该淘汰的 oid 列表); "淘汰词" 只在某个词一个实例都不剩时才出现, 给下游做整词短路。
只读 mask, 不碰 GPU, 也不碰深度 —— 所以调门槛可以秒级重跑。
"""
import os, json
import numpy as np
from pycocotools import mask as mask_util
from word_rank import rank, comparable   # "一组该挂哪个词" 的优先级表, 和三维那步共用同一份

R1  = os.environ.get("R1", "/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
MK  = f"{R1}/03_masks"
OUT = os.environ.get("DW_OUT", f"{R1}/03_dedup_words.json")

# 排除词不参与。它们本来就不往下游走, 让它们参与只有坏处: 万一某个排除词当选了组里的
# 代表, 整组物体就跟着它一起被排除掉了(beach 的 building 就在 EXCLUDE 里)。
EXCLUDE = {w.strip() for w in os.environ.get("EXCLUDE", "").split(",") if w.strip()}

# ---- 门槛。名字统一带 DW_ 前缀: 三维那版用的 DUP_TH/PART_TH 是体素占用的门槛, 含义不同,
#      共用一个名字迟早会被 pipeline 的 CFG 串在一起。
#
# DUP_TH 和 PART_RATIO 默认取同一个值, 这不是偷懒: 一个 mask 完全包在另一个里面时,
# 交集就等于小的那个面积, 于是"较小的那个包含率"恰好等于"小面积/大面积"。两个门槛在这种
# 情形下是同一条线的两侧 —— 取同一个值中间就不留死区(既判不成重复也判不成部件的灰带)。
# 而同义词的典型形态正是"完全包含": 粗词多圈了一点。所以这条线上真正要回答的问题只有一个:
# 多圈了多少还算同一个东西, 多到什么程度才算"父体 vs 部件"。
#
# 0.55 是照三个场景的实测挑的, 两侧都有余量:
#   同一个东西但粗词多圈了一点 —— hot dog cart 完全包在 hot dog stand 里(包含率 1.00),
#     面积比 0.64, 多出来的是头顶那把伞; scaffolding 完全包在 roller coaster 里,
#     面积比 0.65, 多出来的是过山车更外围的钢架。这两个都必须判成重复。
#   真正的同一个东西 —— shop/storefront/restaurant 的 IoU 中位数 0.99, carpark 的
#     trailer/truck 0.99, dumpster/trash can 0.98, 怎么调都过。
# 门槛在 0.50~0.65 之间扫过一遍, beach/carpark/timesquare 三个场景的分组一个字都没变,
# 说明真实的重叠分布在 0.65 以上和 0.4 以下, 中间是空的 —— 这条线放哪儿都行, 取中间。
DUP_TH     = float(os.environ.get("DW_DUP_TH",   "0.55"))  # 双向包含都超过它 = 这一帧算重复
PART_TH    = float(os.environ.get("DW_PART_TH",  "0.70"))  # 单向包含超过它 = 可能是部件
PART_RATIO = float(os.environ.get("DW_PART_RATIO","0.55")) # 且小的面积 < 大的这个比例才算部件
FRAME_FRAC = float(os.environ.get("DW_FRAME_FRAC","0.50")) # 共同帧里有这么多比例投"重复"票
# MIN_COMMON 从 1 改成 3(2026-09-08): 一两帧的证据不够下"是同一个东西"的结论。
# 反例是 timesquare 的 car#39 和 taxi#1, 只有 1 个共同帧(f179), 就凭那一帧 IoU 0.599 判了
# 重复 —— 而那一帧 taxi#1 的 mask 恰好涨大, 把旁边一辆白车也圈进去了。共同帧只有一帧时
# "共同帧里过半判重复"这个分数根本没有分辨力: 分母是 1, 结果只能是 0 或 1。
# 代价实测: 三个场景全场 196 个重复对里, 共同帧 <=2 的一共只有 4 个。
MIN_COMMON = int(  os.environ.get("DW_MIN_COMMON","3"))    # 共同帧少于它就不下结论
# WORD_FRAC 已经不是闸门了, 只用来在 json 里给词对配对率打个"过没过线"的标记, 见文末。
WORD_FRAC  = float(os.environ.get("DW_WORD_FRAC", "0.60")) # 配对率(分母取 max 实例数)
STRIDE     = int(  os.environ.get("DW_STRIDE",    "1"))    # 抽帧比较, 只是速度旋钮
FRAME_SAT  = float(os.environ.get("FRAME_SAT",    "10"))   # 和 pick_from_masks 保持一致

# ------------------------------------------------------------------ 读盘
# 每个词一个 npz: meta 是结构化数组(oid/frame/area/edge), counts 是逐行的 RLE, hw 是画幅。
words = {}          # 词 -> [ (oid, frame, area, edge, rle) ... ]
skipped = {}        # 词 -> 跳过的原因
HW = None
for fn in sorted(os.listdir(MK)):
    if not fn.endswith(".npz"): continue
    z = np.load(f"{MK}/{fn}", allow_pickle=True)
    meta, counts = z["meta"], z["counts"]
    H, W = (int(x) for x in z["hw"]); HW = (H, W)
    word = str(z["word"])
    if word in EXCLUDE:
        skipped[word] = "在 EXCLUDE 里"; continue
    rows = [(int(o), int(f), int(a), float(e), counts[i])
            for i, (o, f, a, e) in enumerate(meta) if a > 0 and int(f) % STRIDE == 0]
    if not rows:
        skipped[word] = "没有有效 mask"; continue
    words[word] = rows
print(f"[输入] {MK}  {len(words)} 个词参与比较"
      + (f", 跳过 {len(skipped)} 个 ({', '.join(sorted(skipped))})" if skipped else ""))

# 每个实例(词#oid)出现在哪些帧。共同帧数是分母, 必须包含"两个都在场但没重叠"的帧 ——
# 只统计重叠过的帧会把"偶尔擦一下边"算成 100% 重复。
frames_of = {}
for w, rows in words.items():
    for oid, fi, a, e, r in rows:
        frames_of.setdefault((w, oid), set()).add(fi)

by_frame = {}
for w, rows in words.items():
    for oid, fi, a, e, r in rows:
        by_frame.setdefault(fi, []).append((w, oid, a, {"counts": r, "size": [HW[0], HW[1]]}))

# ------------------------------------------------------------------ 逐帧比 mask
# 全程在 RLE 里做交集(mask_util.merge intersect=1), 不解码成位图: 1080x1920 解一张要
# 2MB, 一帧五十来个实例解出来就是 100MB; RLE 上求交一次只要几微秒。
def bbox_miss(b1, b2):
    """外接框都不挨着就不用求交了。纯粹是省时间, 不影响结论。"""
    return (b1[0] > b2[0]+b2[2] or b2[0] > b1[0]+b1[2] or
            b1[1] > b2[1]+b2[3] or b2[1] > b1[1]+b1[3])

pair = {}     # (词A,实例A,词B,实例B) -> dict(重复帧, 部件帧, iou 列表)
for fi in sorted(by_frame):
    items = by_frame[fi]
    if len(items) < 2: continue
    bb = [mask_util.toBbox(it[3]) for it in items]
    for i in range(len(items)):
        wi, oi, ai, ri = items[i]
        for j in range(i+1, len(items)):
            wj, oj, aj, rj = items[j]
            if wi == wj: continue                      # 词内的实例分割由 SAM3 负责, 不是这一步的事
            if bbox_miss(bb[i], bb[j]): continue
            inter = float(mask_util.area(mask_util.merge([ri, rj], intersect=1)))
            if inter <= 0: continue
            ci, cj = inter/ai, inter/aj
            small, big = (ai, aj) if ai < aj else (aj, ai)
            k = (wi, oi, wj, oj) if (wi, oi) < (wj, oj) else (wj, oj, wi, oi)
            d = pair.get(k)
            if d is None: d = pair[k] = dict(dup=0, part=0, iou=[])
            if ci > DUP_TH and cj > DUP_TH:
                d["dup"] += 1
                d["iou"].append(inter/(ai+aj-inter))
            elif max(ci, cj) > PART_TH and small < PART_RATIO*big:
                d["part"] += 1
print(f"[逐帧] {len(by_frame)} 帧(步长 {STRIDE}), 有过交集的跨词实例对 {len(pair)} 组")

# ------------------------------------------------------------------ 实例对定性
# 一帧说了不算: 俯视场景里两个不相干的东西前后叠在一起, 某几帧上确实会重得很厉害。
# 要求"共同帧里过半都判重复", 相机一动就会露馅的那种假重叠自然被筛掉。
dup_pairs, part_pairs = [], []
for (wa, oa, wb, ob), d in pair.items():
    n_common = len(frames_of[(wa, oa)] & frames_of[(wb, ob)])
    if n_common < MIN_COMMON: continue
    rec = dict(A=f"{wa}#{oa}", B=f"{wb}#{ob}", 共同帧=n_common,
               判重帧=d["dup"], 判部件帧=d["part"],
               重复占比=round(d["dup"]/n_common, 3),
               中位IoU=round(float(np.median(d["iou"])), 3) if d["iou"] else 0.0)
    if d["dup"]/n_common >= FRAME_FRAC:
        dup_pairs.append(((wa, oa), (wb, ob), rec))
    elif d["part"]/n_common >= FRAME_FRAC:
        part_pairs.append(rec)
print(f"[实例对] 判重复 {len(dup_pairs)} 对, 判部件(不合并, 留给三维那步) {len(part_pairs)} 对")

# ------------------------------------------------------------------ 每个实例的质量
# 一组里要选一个代表送去生成, 得先给每个实例打个分。公式照抄 pick_from_masks
# (面积 x (1-贴边惩罚) x min(帧数/10,1)) —— 赢家马上就要被那一步拿去挑生成样本,
# 与其另发明一个"完整度", 不如直接问"谁能给出更好的生成样本"。
#
# ★ 但面积取活帧的中位数, 不取单帧最大值。这是踩过的坑:
#   timesquare 的 table#5 是个游走的垃圾 mask, f61~f68 涨到 327x142 把一辆卡车和一辆车
#   一起圈进去, 前后帧只剩 6~15 px。它的单帧最大面积 14639 比真车 car#3 的 12346 还大,
#   按最大面积选代表会选中这块垃圾, 真车反被当重复删掉。
#   中位数问的是"这个实例大部分时候多大", 单帧最大值问的是"它最狠的一帧多大" —— 后者
#   正好奖励那种偶尔炸开的坏 mask。贴边同理取中位。
INST = {}
for w, rows in words.items():
    per = {}
    for oid, fi, a, e, r in rows: per.setdefault(oid, []).append((a, e))
    for oid, rs in per.items():
        med_a = float(np.median([a for a, e in rs])); med_e = float(np.median([e for a, e in rs]))
        INST[(w, oid)] = dict(活帧数=len(rs), 中位面积=int(med_a), 最大面积=int(max(a for a, e in rs)),
                              贴边中位=round(med_e, 3),
                              样本分=round(100.0*med_a/(HW[0]*HW[1]) * (1-min(med_e*4, 0.9))
                                        * min(len(rs)/FRAME_SAT, 1.0), 4))
n_inst = {w: len({o for o, *_ in rows}) for w, rows in words.items()}
# 词级统计留着当读日志的参照 —— 分组已经不看它了(以前它是选胜出词的依据)
_by_w = {}
for k in INST: _by_w.setdefault(k[0], []).append(INST[k])
STATS = {w: dict(实例数=n_inst[w],
                 最佳样本分=round(max(q["样本分"] for q in _by_w[w]), 4),
                 总面积=int(sum(a for oid, fi, a, e, r in words[w])),
                 活帧数中位=int(np.median([q["活帧数"] for q in _by_w[w]])))
         for w in words}

# ------------------------------------------------------------------ 实例级并查集
# ★ 合并的单位是实例, 不是词。词级合并("判定两个词是同义词 -> 输的词整个淘汰")粒度太粗:
#   taxi 3 个实例并进 car 时, 判掉 taxi 这个词等于三辆出租车的形态一起没了; 反过来判掉
#   car 就丢了另外 17 辆车。真实的关系是"这个 taxi 实例和那个 car 实例是同一辆车",
#   逐实例连边就没有"整个词消失"这个问题, 词级那道配对率安全阀也就不必要了。
#
# ★ 只连"重复"边, 绝不连"部件"边。部件照旧只记录不合并(见 部件关系 字段)。
#   把部件边也 union 进来的后果实测过: timesquare 出现一个九元组(4 个 billboard +
#   4 个 sign + 1 栋楼 —— 楼是所有招牌的"父体", 一条条部件边把它们串成了一坨),
#   plant pot 30 个实例最后一个不剩。部件该并进父体还是该单独摆, 得看三维盒子。
#
# 两条护栏:
#  1. 同词护栏 —— 一组里不允许出现同一个词的两个实例。词内的实例是 SAM3 自己分开的,
#     这个脚本压根不比它们(上面 wi == wj 就 continue), 所以一组里冒出同词两实例, 只能
#     说明组里必有一条边是错的。加边时按"判重帧数"从多到少排, 会造成同词冲突的边直接丢掉:
#     证据最硬的边先落地, 错边最后才试, 被丢的就是它。按判重帧数排序同时保证结果可复现。
#     已知代价: beach 有一根被 SAM3 断成两截的灯柱(street lamp#9 / #16)接不回来, 会多
#     生成一个 mesh。丢的是 GPU 不是物体, 接受 —— 反过来放开同词, 丢的就是物体。
#  2. MIN_COMMON 默认 3 —— 共同帧少于 3 帧不下重复的结论。timesquare 的 car#39 和
#     taxi#1 只有 1 个共同帧(f179), 就凭那一帧 IoU 0.599 判了重复, 而那一帧 taxi#1 的
#     mask 涨大把旁边一辆白车圈进去了。代价实测: 全场 196 个重复对里共同帧 <=2 的只有 4 个。
NODES = sorted(INST)
par  = {k: k for k in NODES}
wset = {k: {k[0]} for k in NODES}          # 每个组根上挂着"组里已经有哪些词"
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
def union(a, b):
    a, b = find(a), find(b)
    if a == b: return
    par[b] = a; wset[a] |= wset[b]

edge_used, edge_drop = [], []
for a, b, rec in sorted(dup_pairs, key=lambda t: (-t[2]["判重帧"], -t[2]["共同帧"],
                                                  t[2]["A"], t[2]["B"])):
    ra, rb = find(a), find(b)
    if ra == rb: edge_used.append(rec); continue          # 已经在一组里, 冗余边
    clash = wset[ra] & wset[rb]
    if clash:
        edge_drop.append(dict(rec, 丢弃原因="会让 " + "/".join(sorted(clash)) + " 的两个实例进同一组"))
        continue
    union(ra, rb); edge_used.append(rec)
print(f"[连边] 用掉 {len(edge_used)} 条重复边, 同词冲突丢掉 {len(edge_drop)} 条")

# ------------------------------------------------------------------ 每组选代表: 两件事
# (a) 送哪张 mask 去生成 —— 上面那个中位面积的样本分。分最高的那个实例是这一组的"锚":
#     它决定这一组到底是个什么东西。
# (b) 这个物体登记成哪个词 —— 用 word_rank.py 那张 RANK 表, 和三维那步同一份。
#     ★ 不另发明规则: 两套规则会出现二维选 car、三维的 RANK 想选 taxi, 互相打架 ——
#       而二维这步是删实例的, 一旦它把 taxi 那个实例删了, 三维再想选 taxi 也没得选。
#     ★ 名字有实际下游用途, 不是文件名: 01_vlm_sizes.json 那张尺子按词给真实高度
#       (plant pot 0.3m / tree 10m, 差 33 倍), 全局尺度靠它标定。所以同一批物体在全场
#       必须叫同一个名字 —— 不能一半 container 一半 trash can。RANK 是全局固定的表,
#       天然满足这一点; 按"谁抠得全"选就不满足(每组的最全者可能是不同的词)。
#     RANK 分不出高下时退回"实例数多的词", 再平就按词名。
#     ★ 最后这一档只能用全局的量(词名), 不能用 (a) 的样本分 —— 试过, 当场破了"全场同名":
#       timesquare 的 plant pot 和 planter 都是 30 个实例、RANK 里都是 2, 一路平到底,
#       于是每组各自按自己那两张 mask 谁的分高来定名字, 19 个花盆里 6 个叫 plant pot、
#       13 个叫 planter。同一批花盆两个名字, 尺子(01_vlm_sizes.json)就得按两个名字各查
#       一次高度。样本分是逐组的量, 天生做不了全场一致的裁判; 词名是。
#
# ★ 两件事在这里合成一件: 组里留下的那一个实例既是送去生成的那张 mask, 也是登记的那个词
#   (下游只读一张"淘汰实例"表, 没有"换个名字登记"这种表达)。所以 (a) 的作用是划定
#   "哪些词跟这个锚是可比的", (b) 在可比的词里定名字:
#     优先级只在同一类里有意义。timesquare 有个游走的垃圾 mask table#5 和一辆真车
#     (car#3 / taxi#0)进了同一组, 平表里 table 和 taxi 都是 1, "细词赢粗词"这条规则对
#     table vs taxi 无话可说, 并列后按实例数比 —— 桌子实例多, 垃圾 mask 就当上了代表。
#     锚是 car#3(样本分 1.00, table#5 只有 0.62), 车和桌椅不可比, table 出局,
#     剩下 car / taxi 同类可比, RANK 判 taxi 赢 —— 这正是 RANK 表存在的理由。
#   这样就不需要"分数低于组内最高分多少就不许当代表"这种门槛了: 实测三个场景里组内
#   非最高分实例的分数比例是连续分布(0.08 到 1.00 之间没有空档), 门槛放哪儿都要么放过
#   table#5(0.62), 要么误伤(0.65 那个 trash can#16 一被挡掉, 那个桶就成了全场唯一
#   叫 bin 的桶, 全场同名当场破掉)。
# 同词护栏保证一个词在一组里最多一个实例 —— 所以定了词就等于定了实例。
def rep_of(g):
    anchor = max(g, key=lambda k: (INST[k]["样本分"], k[0], k[1]))
    best = {}                                  # 同词护栏下每个词在组里只有一个实例
    for k in g:
        if comparable(k[0], anchor[0]): best[k[0]] = k
    w = min(best, key=lambda w: (rank(w), -n_inst[w], w))
    return best[w], anchor, [k for k in g if k[0] not in best]

ev_of = {}      # 组根 -> 组内的实例对证据
for a, b, rec in dup_pairs:
    if find(a) == find(b): ev_of.setdefault(find(a), []).append(rec)

groups = {}
for k in NODES: groups.setdefault(find(k), []).append(k)

def nm(k): return f"{k[0]}#{k[1]}"
out_groups, dead = [], {}
for root, g in groups.items():
    if len(g) < 2: continue
    g = sorted(g)
    win, anchor, noncmp = rep_of(g)
    lose = [k for k in g if k != win]
    for k in lose: dead.setdefault(k[0], set()).add(k[1])
    # 锚(分最高的实例)和代表不是一个时, 说明 RANK 表让了一步(细词赢粗词) ——
    # 读日志的人有权知道让的是哪一步。
    out_groups.append(dict(实例=[nm(k) for k in g], 代表=nm(win), 代表词=win[0],
                           淘汰=[nm(k) for k in lose],
                           样本分最高=nm(anchor), 代表非最高分=bool(anchor != win),
                           跨类出局=[nm(k) for k in noncmp],
                           各实例={nm(k): INST[k] for k in g},
                           证据=sorted(ev_of.get(root, []), key=lambda r: -r["判重帧"])))
out_groups.sort(key=lambda x: (-len(x["实例"]), x["代表"]))

# 自检: 一组里出现同一个词的两个实例说明护栏漏了, 这种组里必有一条错边 —— 宁可炸也别静默
bad = [g["实例"] for g in out_groups
       if len({s.rsplit("#", 1)[0] for s in g["实例"]}) != len(g["实例"])]
print(f"[自检] {len(out_groups)} 组里同词冲突 {len(bad)} 组" + (f" ★ {bad[:3]}" if bad else " (应为 0)"))

# ------------------------------------------------------------------ 淘汰名单
# 名单按实例发, 不按词发 —— 见上面并查集那段。
inst_of = {w: sorted({o for o, *_ in rows}) for w, rows in words.items()}
dead_out = {w: sorted(v) for w, v in sorted(dead.items()) if v}
# 淘汰词只在"这个词一个实例都不剩"时才留着: 下游拿它做整词短路(不必读 npz, 日志里也一眼
# 看得出是哪个词整个没了)。部分淘汰的词必须照常往下走, 只是少几个实例 —— 所以不能进这个表。
dead_words = sorted(w for w, v in dead.items() if v >= set(inst_of[w]))

# ------------------------------------------------------------------ 词级配对率: 只当证据
# ★ 2026-09-08 起它不再是合并闸门。原来的逻辑是"配对率过线 -> 两个词是同义词 -> 输的词
#   整个淘汰", 分母取 max(实例数) 是那套逻辑唯一的安全阀(防止 taxi 3 个实例把 car 20 个
#   一起带走)。改成逐实例合并之后, 根本不存在"整个词消失", 这道阀门也就没有把守的东西了。
#   数字继续算出来写进 json —— 调门槛时它最有用: 一眼看出哪两个词是成批重合(配对率接近 1,
#   典型的同义词), 哪两个只是零星撞上几个实例(细词是粗词的子集)。
by_word_pair = {}
for (wa, oa), (wb, ob), rec in dup_pairs:
    by_word_pair.setdefault((wa, wb), []).append(((oa, ob), rec))
word_ev = []
for (wa, wb), lst in sorted(by_word_pair.items()):
    ma = len({p[0][0] for p in lst}); mb = len({p[0][1] for p in lst})
    frac = min(ma, mb)/max(n_inst[wa], n_inst[wb])
    word_ev.append(dict(词A=wa, 词B=wb, 实例数=[n_inst[wa], n_inst[wb]],
                        配对实例数=[ma, mb], 配对率=round(frac, 3),
                        过词级门槛=bool(frac >= WORD_FRAC),
                        实例对=[r for _, r in sorted(lst, key=lambda x: -x[1]["共同帧"])][:20]))
word_ev.sort(key=lambda v: -v["配对率"])

json.dump(dict(
    参数=dict(DW_DUP_TH=DUP_TH, DW_PART_TH=PART_TH, DW_PART_RATIO=PART_RATIO,
              DW_FRAME_FRAC=FRAME_FRAC, DW_WORD_FRAC=WORD_FRAC,
              DW_MIN_COMMON=MIN_COMMON, DW_STRIDE=STRIDE,
              EXCLUDE=sorted(EXCLUDE)),
    组=out_groups, 淘汰实例=dead_out, 淘汰词=dead_words,
    未参与=skipped, 各词=STATS,
    # 部件关系只记录不执行 —— 留个凭据, 万一下游三维那步把某个部件误当独立物体摆出来了,
    # 回头能在这里查到"二维早就看出它包在谁里面"。
    部件关系=sorted(part_pairs, key=lambda r: -r["共同帧"])[:200],
    # 被同词护栏丢掉的边。这些是"看起来重复但必有一个是错的"那批, 最值得人眼扫一遍。
    同词冲突丢边=sorted(edge_drop, key=lambda r: -r["判重帧"])[:100],
    词对配对率=word_ev[:80],
), open(OUT, "w"), ensure_ascii=False, indent=1)

print(f"\n[分组] {len(out_groups)} 组(每组 = 一个物理物体), 淘汰 "
      f"{sum(len(v) for v in dead_out.values())} 个实例"
      f"(其中 {len(dead_words)} 个词整个没剩: {', '.join(dead_words) or '无'})")
for g in out_groups:
    print(f"  {g['代表']:22s} <= {', '.join(g['实例'])}"
          + ("   ★代表不是分最高的(RANK 表让了细词)" if g["代表非最高分"] else ""))
    for s in g["实例"]:
        k = (s.rsplit("#", 1)[0], int(s.rsplit("#", 1)[1])); q = INST[k]
        print(f"      {'★' if s == g['代表'] else ' '} {s:22s} 活{q['活帧数']:4d}帧  "
              f"中位面积{q['中位面积']:8d}  最大面积{q['最大面积']:8d}  样本分{q['样本分']:8.4f}"
              + (f"  [和锚 {g['样本分最高']} 不同类, 不许当代表]" if s in g["跨类出局"] else ""))
print(f"\n[done] -> {OUT}")
