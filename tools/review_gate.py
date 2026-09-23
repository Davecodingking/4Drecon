#!/usr/bin/env python
"""人工闸: 把"待生成的全部候选"摆成一张联络图, 停下来等人决策, 再把决策写成配置。

为什么要有这一步 —— 流水线里有三个决定机器判不了, 一直靠改源码里的常量:
  1. 哪个候选拿去生成   (VLM 只判"是不是这个词", 判不了"抠得全不全"。
                        折叠桌自动选中的那张只有一块光桌面, 没有腿)
  2. 哪个词整个作废     (mask 抠出来全是别的东西, 比如 "building" 会把整个餐厅连桌椅一起吃掉)
  3. 哪些类要逐实例生成 (一条街 43 辆车共用一个 mesh 会很假, 但椅子共用没问题)

★ 位置: 必须在 recall_mask 之后、gen 之前。
  之后 —— 召回补抠出来的实例也要一起过目, 否则那一支完全没有人工把关, 垃圾照样烧 GPU。
  之前 —— 生成是最贵的一步, 先筛后生成。

两个模式:
  (默认)   扫出全部候选, 出联络图 05_review_sheet.jpg + 决策文件 05_review.json, 然后停。
  --apply  读回决策文件, 校验, 打印下游该用的配置, 并写 05_review_applied.json。

决策文件是给人改的, 每个候选一行:
  {"编号": 3, "词": "table", "来源": "类", "决定": "keep"}
把 "决定" 改成:
  keep      照常生成 (默认)
  drop      这一条不生成
  drop-word 整个词作废 (等于加进 EXCLUDE)
  solo      这个词逐实例生成
  shared    这个词共用一个 mesh (SOLO="*" 场景把词捞回来的唯一通道)
另有两个字段(不是"决定", 和决定并存):
  "复用组": 1/2/3...   同一个词里填同一个号的实例共用一个 mesh。比 shared 细一档。
  "改类别": "另一个词"  这个实例其实是别的东西, mask 不变只换标签(street lamp -> light fixture)
另有 "改用": [实例号, 帧号] —— 换一个候选, 对应老的 PICK_OVERRIDE。

重复跑不会覆盖已经改过的决定, 只会把新出现的候选追加进来。
"""
import os, json, sys
import numpy as np, cv2

R1   = os.environ["R1"]
TAG  = os.environ["TAG"]
FR   = os.environ["FR"]
IMG  = os.path.expanduser(os.environ.get("MYIMG", "~/sam3d/sam-3d-objects/my_images"))
# ★ 图和决策文件**都**按阶段分两份, 谁也不覆盖谁:
#     闸 2(生成前)          05_review_sheet.jpg            05_review.json
#     闸 4(召回并进来之后)  05_review_sheet_afterrecall.jpg 05_review_afterrecall.json
#   这样"我不想要 recall 了"就是**删掉带后缀的那一份**, 直接回到召回前的状态,
#   不用数版本号, 也不用把召回的决定从一个混在一起的文件里一条条挑出来。
#
# ★ 但闸 4 那份**必须以闸 2 那份为种子**建立(见下面"合并已有决策")。
#   决定是跨阶段累积的: 你在闸 2 标的 shared/分组, 闸 4 还要继续生效。
#   闸 4 从空文件起步的话, 那批词会既不逐实例也没类 mesh, 整个从场景消失且不报错 ——
#   就是"决定自我抹除"的变种, beach 实测 27 个物体。
#
# ★ 阶段由驱动显式告知, 不靠猜。原来是"盘上有没有 14_recall_ready.json"——
#   而那个文件一旦生成就永远在, 于是后面任何一次重跑「生成前」那一段, 闸 2 都会
#   把召回前的清单写进 _afterrecall 那份, 把真正的召回后结果覆盖掉。
_STAGE = (os.environ.get("REVIEW_STAGE")
          or ("post" if os.path.exists(f"{R1}/14_recall_ready.json") else "pre")).lower()
if _STAGE not in ("pre", "post"):
    raise SystemExit(f"[失败] REVIEW_STAGE 只能是 pre 或 post, 拿到 {_STAGE!r}")
_AFTER = _STAGE == "post"
_SFX = "_afterrecall" if _AFTER else ""
SHEET   = f"{R1}/05_review_sheet{_SFX}.jpg"
DEC     = f"{R1}/05_review{_SFX}.json"
DEC_PRE = f"{R1}/05_review.json"          # 闸 2 那份, 闸 4 第一次的种子
COLS = int(os.environ.get("REVIEW_COLS", "6"))
TILE = int(os.environ.get("REVIEW_TILE", "260"))

def cutout(frame_p, mask_p, size):
    """灰底特写: 只留 mask 里的像素, 裁到外接框再留一圈边。"""
    img = cv2.imread(frame_p); m = cv2.imread(mask_p, 0)
    if img is None or m is None: return None
    if m.shape != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    b = m > 127
    if b.sum() < 30: return None
    out = np.full_like(img, 210)                     # 灰底
    out[b] = img[b]
    v, u = np.nonzero(b)
    pad = int(0.12*max(u.max()-u.min(), v.max()-v.min())) + 8
    y0, y1 = max(v.min()-pad, 0), min(v.max()+pad, img.shape[0])
    x0, x1 = max(u.min()-pad, 0), min(u.max()+pad, img.shape[1])
    c = out[y0:y1, x0:x1]
    if c.size == 0: return None
    s = size/max(c.shape[0], c.shape[1])
    c = cv2.resize(c, (max(int(c.shape[1]*s), 1), max(int(c.shape[0]*s), 1)))
    canvas = np.full((size, size, 3), 210, np.uint8)
    oy, ox = (size-c.shape[0])//2, (size-c.shape[1])//2
    canvas[oy:oy+c.shape[0], ox:ox+c.shape[1]] = c
    return canvas

# ---------- 扫出全部待生成候选 ----------
cands = []
picks = json.load(open(f"{R1}/04_picks/picks.json"))
_enum = set()          # 逐实例枚举覆盖到的词。类候选只补它没覆盖的, 免得同一个词两种格子并存

# --- 逐实例候选 (solo 那步会生成的) ---
# gen_per_instance 是从 06_all_boxes.json 拿实例号的, 而盒子要等 gen->ratio->boxes 才有,
# 这道闸在生成之前, 拿不到。但实例号 03_masks 里本来就有, 挑帧的打分公式也照抄得过来,
# 所以这里直接从 mask 枚举一遍 —— 数量可能和最终的差几个(盒子那步还会按点数和跨词去重再滤),
# 但"会生成哪些东西"这件事看得全, 不至于让整支逐实例生成绕过人工。
# 门槛按像素, 定义在 data/outdoor/scripts/solo_gate.py(和 pick / verify / gen_per_instance 同一份)。
# realpath 找仓库根, 同 reuse_table.py。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                "data", "outdoor", "scripts"))
from solo_gate import min_px                                         # noqa: E402
SOLO_MIN_PX = min_px()
SOLO_CFG = os.environ.get("SOLO", "")
KEEP     = {w.strip() for w in os.environ.get("SOLO_KEEP","").split(",") if w.strip()}
EXCL     = {w.strip() for w in os.environ.get("EXCLUDE","").split(",") if w.strip()}
if SOLO_CFG:
    allw = [f[:-4].replace("_", " ") for f in sorted(os.listdir(f"{R1}/03_masks")) if f.endswith(".npz")]
    # ★ 不减 KEEP。人标了"整词复用"之后, 那个词的每一个实例都还要留在表里 ——
    #   决定只改**标签**, 不该让东西从表里消失。
    #   第一版减掉了 KEEP, 于是 10 个 bench 缩成 1 格, 而这正是"决定自我抹除"那个 bug 的
    #   同一个病根: 决定把候选移出表 -> 下次重算时表里没它 -> 决定蒸发 -> 物体全没。
    solo_w = set(allw) if "*" in SOLO_CFG else {w.strip() for w in SOLO_CFG.split(",") if w.strip()}
    solo_w |= KEEP                      # 复用的词也要枚举出来
    solo_w -= EXCL
    # 跨词去重判掉的词也不会被生成 —— 不减掉的话表里会出现三份同一栋楼(实测 beach:
    # shop/storefront/restaurant 各 5 栋), 把去重前的清单当成待生成清单展示是误导。
    # ★ 名单的粒度是实例不是词(2026-09-08 去重改成逐实例合并): 一个词往往只有几个实例是
    #   重复的, 别的实例照样要生成。只减"整个没剩的词"会让表里留着那些注定不生成的实例 ——
    #   timesquare 实测 car 43 个实例里 10 个是 taxi/van 的重复, 词一个也没整个消失。
    _dw = f"{R1}/03_dedup_words.json"; _dead_inst = {}
    if os.path.exists(_dw):
        _d = json.load(open(_dw))
        _lose = set(_d.get("淘汰词", []))
        _dead_inst = {w: set(v) for w, v in _d.get("淘汰实例", {}).items()}
        if _dead_inst:
            print(f"[去重] {sum(len(v) for v in _dead_inst.values())} 个实例被同组的另一个词代表, "
                  f"不进表" + (f"(其中整个没剩的词: {', '.join(sorted(_lose))})" if _lose else ""))
        solo_w -= _lose
    for w in sorted(solo_w):
        _enum.add(w)
        z = np.load(f"{R1}/03_masks/{w.replace(' ','_')}.npz", allow_pickle=True)
        meta, (H, W) = z["meta"], z["hw"]
        per = {}
        for i, r in enumerate(meta):
            if r["area"] <= 0: continue
            sc = int(r["area"]) * (1 - min(float(r["edge"])*4, 0.9))     # 和 gen_per_instance 同一个打分
            o = int(r["oid"])
            if sc > per.get(o, (0,))[0]: per[o] = (sc, int(r["area"]), int(r["frame"]))
        _nfr = {}
        for r in meta:
            if r["area"] > 0: _nfr[int(r["oid"])] = _nfr.get(int(r["oid"]), 0) + 1
        for o, (sc, a, fi) in sorted(per.items()):
            if o in _dead_inst.get(w, ()): continue      # 这个实例由同组另一个词代表
            pct = 100*a/(int(H)*int(W))
            if a < SOLO_MIN_PX: continue
            cands.append(dict(词=w, 来源="逐实例",
                              生成名=f"{TAG}_{w.replace(' ','_')}_{o}",
                              目录=None, 实例=o, 帧=fi, 面积占比=round(pct, 3), 待生成=True,
                              # ★ 把"系统是怎么挑这一帧的"摆出来。人要能改的是规则, 不是逐个纠正结果 ——
                              #   看不见依据就只能凭感觉判, 那样的人工介入不会收敛。
                              挑帧依据=dict(打分=round(sc, 1), 活了几帧=_nfr.get(o, 0),
                                          说明="打分 = 面积 x (1 - 贴边惩罚), 在这个实例活过的帧里取最高")))

# 类候选: 只补逐实例没枚举到的词(比如 SOLO 没开、或者面积全都不过门槛的)。
# 已经逐实例枚举过的词不再放一格 —— 那会让同一个东西在表里出现两次。
for w in sorted(picks):
    if w in _enum: continue
    d = f"{IMG}/{TAG}_{w.replace(' ','_')}"
    if not os.path.exists(f"{d}/frame.png"): continue
    p = picks[w]
    cands.append(dict(词=w, 来源="类", 生成名=f"{TAG}_{w.replace(' ','_')}", 目录=d,
                      实例=p.get("obj_id"), 帧=p.get("frame"),
                      面积占比=p.get("面积占比"), 候选数=p.get("候选数")))

# ---- 召回候选: 排在最后, 单独标黄 ----
# ★ 排最后不排最前: 排最前会让主线所有候选的编号跟着变一遍, 人刚用过的号立刻失效。
#
# 类别继承规则(三条, 见 SKILL.md):
#   1. 这个词前面有决定  -> 继承(复用的继续复用, 逐实例的继续逐实例)
#   2. 这个词前面没决定  -> 默认 solo, 标黄等人判。
#      "没决定"包括"整个词被跨词去重清空了"这种情况 —— 那时这个词从没进过表,
#      没有任何决定可继承。不要拿"当初和它重叠的那个词"的决定去顶: 去重判的是
#      "**那一个**物体是同一个", 不是"这两个词是同义词"。召回的是另一个物体。
#   3. 这个词前面被分了组(word~1/~2/...) -> **硬停**, 等人指定进哪一组。
#      新来一个实例属于哪一组是看外观的视觉判断, 机器猜不了; 猜错的代价是它长成
#      别的组的样子, 而且不报错。
# ★ 只放**真会生成**的召回候选。
#   14_recall_ready.json 是"补抠成功的全部", 里面还混着注定不会生成的:
#     - 词在 EXCLUDE 里(manhole cover)
#     - 同一个词只派一个代表去生成, 其余复用(3 个 barrier 只生成 1 个)
#   真会生成的清单是 mask_recall.py 写的 logs/recall_jobs.txt。
#   读前者的话, 人要逐个判 7 格, 其中 3 格判了也没用 —— 白看。
#   规则: sheet 上只放人判了**有意义**的东西。已经被自动规则判掉的一律不进表。
rp = f"{R1}/14_recall_ready.json"
_jobs_p = f"{R1}/logs/recall_jobs.txt"
_jobs = (set(open(_jobs_p).read().split()) if os.path.exists(_jobs_p) else None)
_rec_pending = []
if os.path.exists(rp):
    _seen_words = {r["词"] for r in cands}
    _subs = {}                     # 母词 -> 有哪些子类, 用来判"这个词被分过组"
    for w in _seen_words:
        if "~" in w: _subs.setdefault(w.split("~", 1)[0], []).append(w)
    for a in json.load(open(rp)):
        d = f"{IMG}/{a['生成名']}"
        if not os.path.exists(f"{d}/frame.png"): continue
        if _jobs is not None and a["生成名"] not in _jobs:
            continue                       # 注定不会生成的, 不摆给人判
        w = a["词"]
        _inherit = None
        if w in _seen_words:                       # 规则 1
            _inherit = next((r.get("决定") for r in cands if r["词"] == w), None)
        elif w in _subs:                           # 规则 3: 分过组, 必须人指定
            _rec_pending.append((a["生成名"], w, sorted(_subs[w])))
        cands.append(dict(词=w, 来源="召回", 生成名=a["生成名"], 目录=d,
                          实例=a.get("obj_id"), 帧=a.get("最佳帧"),
                          面积占比=a.get("面积占比"), 框重合=a.get("框重合"),
                          继承决定=_inherit,
                          待分组=sorted(_subs[w]) if w in _subs else None))

if not cands:
    raise SystemExit(f"[失败] 没找到任何待生成候选。确认 pick 跑过, 且 {IMG}/{TAG}_* 里有 frame.png")

# ---------- 合并已有决策 ----------
old, HIST = {}, {}
# ★ 闸 4 **每次**都要把闸 2 那份读进来当底, 不是只在第一次。
#   只在"自己那份不存在"时才看 pre 的话, 会出这个吞噬:
#     闸 4 跑过 -> afterrecall 建立;  人回头重跑「生成前」, 在 05_review.json 里新标了几个
#     drop-word 并落地 ✓;  再走到闸 4 --apply -> 只读那份**旧的** afterrecall ->
#     覆写同一个 05_review_applied.json -> 刚标的 drop-word 消失, 不报错。
#   两道闸写的是同一份 applied, 所以两份表必须是"叠加"关系: pre 打底, post 覆盖。
_chain = [q for q in (([DEC_PRE] if _AFTER else []) + [DEC]) if os.path.exists(q)]
_prev, _wmap = None, {}
for _q in _chain:
    _prev = json.load(open(_q))
    if _AFTER and _q == DEC_PRE:
        print(f"[打底] 先读闸 2 那份 {os.path.basename(_q)} ({len(_prev['候选'])} 个候选)")
    for r in _prev["候选"]:
        old[r["生成名"]] = {k: r[k] for k in ("决定", "改用", "备注", "复用组", "改类别", "VLM放行") if k in r}
        _wmap[r["生成名"]] = (r["词"], r["实例"])
    HIST.update(_prev.get("历史决定") or {})
# pre 比 post 新 = 人回头改过闸 2 那份。上面的叠加保证"只在 pre 里的行"不丢,
# 但两份都有的行仍然是 post 说了算(post 是当时从 pre 抄的旧值), 那种冲突机器判不了, 只能报出来。
if _AFTER and os.path.exists(DEC) and os.path.exists(DEC_PRE) \
   and os.path.getmtime(DEC_PRE) > os.path.getmtime(DEC) + 1:
    print(f"★ {os.path.basename(DEC_PRE)} 比 {os.path.basename(DEC)} 新 —— 你回头改过闸 2 那份。\n"
          f"  两份都有的行以召回后那份为准。想让闸 2 的新决定全部生效, 删掉 {os.path.basename(DEC)} 再跑一次。")
if _prev:
    # ★ 历史决定: 候选从表里消失了, 决定也要留着。
    #   自吞噬循环实测过: 把一个词标成 shared -> 它整个退出逐实例候选 -> 下次重算 --apply
    #   时表里没有它 -> "共用词"变回空 -> 那个词既不逐实例也没有类 mesh, 整个从场景消失。
    #   所以决定必须独立于候选表存一份, 不能每次从表里重算。
    # 谁是真相: **表是人看到的**, 历史记录只负责"当前不在表里的那些"。
    # 所以在表里出现过的键, 一律以表里的值为准 —— 包括"人看过、就是 keep"这种情况,
    # 那时要把历史记录里的旧决定**删掉**, 否则它会把人改回 keep 的决定复活。
    _w = _wmap
    for k, v in old.items():
        _live = (v.get("决定", "keep") != "keep" or v.get("改用")
                 or v.get("复用组") or v.get("改类别") or v.get("VLM放行"))
        if _live:
            HIST[k] = dict(v, 词=_w.get(k, (None, None))[0], 实例=_w.get(k, (None, None))[1])
        else:
            HIST.pop(k, None)
rows = []
for i, c in enumerate(cands):
    r = dict(编号=i, **c)
    r.pop("目录")
    r.update(dict(决定="keep"))
    # 上一版表里的决定优先; 表里没有就翻历史决定 ——
    # ★ 候选可能因为自己的决定退出过表(标了 shared 就不在逐实例清单里), 回来时
    #   如果不把历史决定贴回去, 它会显示成默认的 keep, 而实际决定还在别处生效, 两边打架。
    _d = old.get(c["生成名"]) or {k: v for k, v in HIST.get(c["生成名"], {}).items()
                                  if k in ("决定", "改用", "备注", "复用组", "改类别", "VLM放行")}
    r.update(_d)
    # ★ 已经落地的"共用词"要反映回表上。派生子类(umbrella~1)是上一次 --apply 展开出来的
    #   新词, 它的行是这一次才第一次出现 —— 不回写的话会显示成默认 keep, 而实际已经在
    #   SOLO_KEEP 里生效, 图和实情两张皮(粉框该有却是绿的)。
    if r["词"] in KEEP and r["决定"] == "keep":
        r["决定"] = "shared"
    # 召回候选继承同词前面的决定(规则 1)。人还没碰过这一行时才继承 —— 人改过就以人为准。
    if r.get("继承决定") and c["生成名"] not in old and r["决定"] == "keep":
        r["决定"] = r["继承决定"]
    rows.append(r)

# ★ 判了 drop / drop-word 的直接从表里删掉, 不占格子。
#   和"标 shared 不许消失"不矛盾: shared 那个东西**还会进场景**(只是共用模型), 要留着让人看;
#   drop 是人判它不要了, 留着只是噪声, 下一轮还得再扫一遍眼。区别在于它还在不在最终场景里。
#   决定本身存在"历史决定"里, 不会因为行没了就失效 —— 所以删格子不影响它继续生效。
_dropped = [r for r in rows if r["决定"] in ("drop", "drop-word")]
if _dropped:
    _dw = {r["词"] for r in rows if r["决定"] == "drop-word"}
    rows = [r for r in rows
            if r["决定"] not in ("drop", "drop-word") and r["词"] not in _dw]
    for i, r in enumerate(rows): r["编号"] = i        # 删完重编号, 别留空号
    print(f"[已判掉] {len(_dropped)} 个不进表(人判了 drop/drop-word): "
          + ", ".join(sorted({f'{r["词"]}#{r["实例"]}' for r in _dropped})))

# 规则 3 的硬停: 放在出图之后, 让人先看到图再报错 —— 报了错却没图可看是最难受的。
def _check_pending(rows_):
    bad = [(g, w, subs) for g, w, subs in _rec_pending
           if not any(r["生成名"] == g and (r.get("复用组") or r.get("改类别")
                                          or r.get("决定") in ("drop", "drop-word"))
                      for r in rows_)]
    if bad:
        print(f"\n★★ 有 {len(bad)} 个召回实例属于**已经分过组**的词, 必须由人指定进哪一组, 不然停在这里:")
        for g, w, subs in bad:
            print(f"     {g}   词 {w!r} 已经分成 {subs}")
        print(f"   在 {DEC} 里给这几条填 \"改类别\": \"{bad[0][2][0]}\" (或别的子类), 再跑 --apply。")
        print(f"   为什么不自动分: 新来一个实例属于哪一组是看外观的判断, 猜错了它会长成别的组的样子而且不报错。")
        return False
    return True

if "--apply" in sys.argv:
    # ---------- 读回决策, 变成下游要的配置 ----------
    if not os.path.exists(DEC):
        raise SystemExit(f"[失败] 还没有决策文件 {DEC}, 先不带 --apply 跑一次")
    # ★ 和出图那边走**同一条链**: pre 打底, post 覆盖。只读 post 的话,
    #   人后来在闸 2 那份里新加的决定会被这一次 --apply 覆写掉(两道闸写同一个 applied)。
    #   ★ 每一份里要先铺"历史决定"再铺"候选行", 两层都参与覆盖。
    #     只铺候选行的话, **后一份**把决定存在历史里(判掉之后那行就从表上消失了)时,
    #     压不过**前一份**表上同名的那行, 前一份的 keep 会赢。
    #     实测 office 2026-09-20: 5 个召回物体在闸 4 判了 drop(所以只剩在闸 4 的历史决定里),
    #     后来闸 2 重出了一次表, 它们又作为新候选回到闸 2 的表上、默认 keep ——
    #     落地时取了那个 keep, 5 个判掉的物体照样摆进了场景, 而且一声不吭。
    #     这和注释里那个"决定自我抹除"是同一个病根, 只是入口换成了"重出表把旧决定顶掉"。
    #   同一份里候选行压历史: 表是人正在看的那个, 它说了算。
    _rows, _hist = {}, {}
    for _q in [q for q in (([DEC_PRE] if _AFTER else []) + [DEC]) if os.path.exists(q)]:
        _d = json.load(open(_q))
        _h = _d.get("历史决定") or {}
        for _k, _v in _h.items(): _rows[_k] = dict(_v, 生成名=_k)
        for r in _d["候选"]: _rows[r["生成名"]] = r
        _hist.update(_h)
    _doc = dict(候选=list(_rows.values()), 历史决定=_hist)
    saved = list(_doc["候选"])
    # 把已经从表里消失、但当初做过决定的那些补回来一起算 —— 否则决定会自己蒸发。
    _now = {r["生成名"] for r in saved}
    _gone = [dict(v, 生成名=k) for k, v in (_doc.get("历史决定") or {}).items() if k not in _now]
    if _gone:
        print(f"[历史决定] {len(_gone)} 个候选已不在表里(多半是被自己的决定移出去的), 决定仍然生效")
        saved += _gone
    # ★ 待分组的硬停要吃 **saved**(含历史决定), 不能只看 DEC 的候选表。
    #   人把那条待分组的召回候选判成 drop 之后, 它已经被从表里删掉了 ——
    #   只看表就永远找不到它, 于是每次 --apply 都硬停, 还提示"去表里给它填改类别",
    #   而它根本不在表里。"这一条不要"是最合理的处理, 不该是唯一走不出去的路。
    if not _check_pending(saved):
        sys.exit(2)
    VALID = {"keep", "drop", "drop-word", "solo", "shared"}
    bad = [r for r in saved if r.get("决定") not in VALID]
    if bad:
        raise SystemExit(f"[失败] {len(bad)} 条的\"决定\"不认识: "
                         f"{[(r['生成名'], r.get('决定')) for r in bad[:5]]}\n       只能是 {sorted(VALID)}")
    drop_word = sorted({r["词"] for r in saved if r["决定"] == "drop-word"})
    solo_word = sorted({r["词"] for r in saved if r["决定"] == "solo"})
    # shared: 反过来 —— 这个词不要逐实例, 共用一个 mesh。
    # ★ SOLO="*" 的场景(beach/carpark 都是)靠这条把词捞回来, 否则人在闸口说不出"这12把
    #   一模一样的椅子共用一个就行", 只能回去改源码 —— 又变回改代码。
    shared_word = sorted({r["词"] for r in saved if r["决定"] == "shared"})
    # 分组复用: 同一个词内部分成几组, 每组共用一个 mesh。
    # ★ 和 shared 的区别: shared 是整词一个 mesh; 这个是词内分组, 组数由人定。
    #   umbrella 14 把有三种花色, 整词共用会全变成同一把, 逐个生成又太贵 —— 分组是中间档。
    relabel = {f'{r["词"]}#{r["实例"]}': r["改类别"] for r in saved if r.get("改类别")}
    relabel_extra = {}
    # ★ "复用组" 只是填写的简写, 不是一套独立机制。
    #   这里把它**展开**成两个已有机制的组合:
    #       把这一组的实例挪进一个派生子类 umbrella~1  (挪类别)
    #       再把 umbrella~1 整词复用                    (共用词)
    #   这样"分组复用"就不需要第三套代码 —— 生成那步、组装那步一行都不用改,
    #   而且 sheet 上天然显示成 umbrella~1/2/3 三行粉框, 和 bench 一模一样。
    groups = {}
    for r in saved:
        g = r.get("复用组")
        if g in (None, "", 0): continue
        groups.setdefault(r["词"], {}).setdefault(str(g), []).append(int(r["实例"]))
    for w, gs in sorted(groups.items()):
        base = w.split("~", 1)[0]                 # 已经是派生词就按母词编号, 别叠成 a~1~2
        # 组号按**数字**排, 不按字符串 —— 字符串排序会把"复用组 10"排到"2"前面,
        # 而子类名 umbrella~1/2/3 是按这个顺序发的, 排错了实例就进错组。
        def _gnum(kv):
            try: return (0, int(kv[0]), "")
            except ValueError: return (1, 0, kv[0])      # 非数字的组名排在后面, 按字面
        for _g, oids in sorted(gs.items(), key=_gnum):
            # ★ 子类名直接用**人填的组号**, 不是重新发的序号。
            #   原来按位置从 1 重数, 而且每个源词各数各的 —— 人在 umbrella~2 的行上填"复用组 9",
            #   它是那个词里唯一一组, 序号重数成 1, 于是被并进**已经存在的 umbrella~1**。
            #   人说 9 就该是 umbrella~9; 人说 1 就该是并进 umbrella~1(这也是他的意思)。
            _gs = str(_g).strip()
            if not _gs or "~" in _gs or " " in _gs:
                raise SystemExit(f"[失败] 复用组的号 {_g!r} 不能带空格或 ~, 用数字最省事")
            # 纯数字先归一 —— 不然人填 "01" 会建出 umbrella~01, 和 umbrella~1 分裂成两个子类、
            # 两份 mesh, 而且那一组常常只有一个实例, 连相对守卫都够不着(实例数不到门槛)。
            if _gs.lstrip("-").isdigit(): _gs = str(int(_gs))
            sub = f"{base}~{_gs}"
            if sub == w: continue                 # 已经在目标子类里了
            for o in sorted(oids):
                relabel_extra[f"{w}#{o}"] = sub
            shared_word.append(sub)
    shared_word = sorted(set(shared_word))
    # 挪类别: 这个实例其实是另一个词。mask 不变, 只换标签。
    drop_one  = sorted(r["生成名"] for r in saved if r["决定"] == "drop")
    # ★ 换候选的键: 逐实例候选按 "词#实例号", 类候选按 "词"。
    #   原来一律按"词"建键, 于是同一个词的两个实例都填"改用"时后者悄悄覆盖前者,
    #   而且逐实例生成那步根本不看这张表 —— SOLO="*" 的场景(beach/carpark)里
    #   联络图 81/83 行都是逐实例, 等于"改用"整体是个摆设。
    #   判据不能只看名字后缀 —— 联络图把**共用词**也一行一个实例地列出来(见上面 KEEP 那段),
    #   所以 bench 这种复用词的行同样长得像逐实例, 名字也带 _1。按后缀判的话键会写成
    #   "bench#1", 而 bench 走的是类 mesh: s_override 收不到(它只认裸词),
    #   gen_per_instance 也不遍历它(它被 SOLO_KEEP 排除) —— 一个消费者都没有。
    #   真正的判据是"这个词最终出的是类 mesh 还是专属 mesh"。
    _shared_set = set(shared_word) | KEEP
    _moved = {**relabel, **relabel_extra}
    def _ov_key(r):
        # ★ 先解析这个实例**最终落进哪个词**。填了"复用组"或"改类别"的行, 生成时它已经
        #   被 relabel 搬进 umbrella~1 了 —— 键还写原词 umbrella 的话,
        #   overrides() 收不到(带 #)、gen_per_instance 也不遍历它(原词已经没实例了), 又是零消费者。
        w = _moved.get(f'{r["词"]}#{r["实例"]}', r["词"])
        if w in _shared_set:
            return w                             # 共用词: 换的是那个类 mesh 的抠图
        return (f'{w}#{r["实例"]}'
                if str(r.get("生成名", "")).endswith(f'_{r["实例"]}') else w)
    # 召回候选的抠图是 mask_recall.py 按检测框喂回 SAM3 现抠的, 不经过挑图那套,
    # 也没有"另一帧"可选 —— 对它填"改用"没有任何消费者, 与其静默丢弃不如当场说清楚。
    _bad_ov = [r["生成名"] for r in saved
               if r.get("改用") and int(r.get("实例") or 0) >= 10000]
    if _bad_ov:
        raise SystemExit(f"[失败] 召回候选不支持\"改用\": {', '.join(_bad_ov)}\n"
                         f"       它的抠图是按检测框现抠的, 没有别的帧可换。\n"
                         f"       这一条要么 keep 要么 drop; 觉得抠得不对就 drop。")
    # ★ 同一行既要搬家又要换抠图 —— 拦住。relabel_words 在目标词已占用该实例号时会重新编号
    #   (实测: street lamp#10/11/12 搬进 light fixture 之后变成 13/14/15), 而这里生成的键
    #   还用着旧号, 会静默作用到目标词里**本来就存在的那个同号实例**上。
    #   分两次做就没有这个问题: 先搬家跑一轮, 看到新号之后再填改用。
    _conf = [r["生成名"] for r in saved if r.get("改类别") and r.get("复用组")]
    if _conf:
        raise SystemExit(f"[失败] 这几行同时填了\"改类别\"和\"复用组\": {', '.join(_conf)}\n"
                         f"       两个都是挪类别, 展开时复用组会把改类别静默压掉。只填一个。")
    # 只拦"改类别 + 改用": 搬进**已有的词**时实例号可能被重编, 换抠图会套到别的物体上。
    # 复用组建的是新子类, 实例号原样保留, 和改用一起填没问题, 不拦。
    _both = [r["生成名"] for r in saved
             if r.get("改用") and r.get("改类别")]
    if _both:
        raise SystemExit(f"[失败] 这几行同时填了\"改用\"和\"改类别/复用组\": {', '.join(_both)}\n"
                         f"       搬家会重新编实例号, 两件事一起做, 换抠图会套到别的物体上。\n"
                         f"       先只填搬家的那一项跑一轮, 在新表上看到新实例号之后再填改用。")
    # ★ 共用词的"改用"不管写在哪一行都塌成同一个键(裸词)。两行各写各的值, 字典里后者静默压掉前者,
    #   谁赢看行的先后 —— 人以为钉的是 #4, 实际生成用的是 #7。前端切模式时会把另一种模式的
    #   改用收起来, 这里是最后一道: 撞了就停, 不替人挑。
    _ovs = {}
    for r in saved:
        if r.get("改用"):
            _ovs.setdefault(_ov_key(r), {})[r["生成名"]] = list(r["改用"])
    _clash = {k: v for k, v in _ovs.items() if len({tuple(x) for x in v.values()}) > 1}
    if _clash:
        raise SystemExit("[失败] 同一个共用 mesh 被指定了不同的样本:\n" + "\n".join(
            f"       {k}: " + ", ".join(f"{g} -> 实例{a}/第{b}帧" for g, (a, b) in v.items())
            for k, v in _clash.items()) + "\n       一个词只有一个整类样本, 在闸口界面上重新选一次再保存。")
    override  = {_ov_key(r): r["改用"] for r in saved if r.get("改用")}
    # VLM 说"不是它"、人看过还是要的。drop 掉的不算(人最后的决定是不要)。
    vlm_release = sorted(r["生成名"] for r in saved
                         if r.get("VLM放行") and r["决定"] not in ("drop", "drop-word"))
    out = dict(排除词=drop_word, 逐实例词=solo_word, 共用词=shared_word,
               不生成=drop_one, 指定样本=override, 改类别={**relabel, **relabel_extra},
               VLM放行=vlm_release)
    json.dump(out, open(f"{R1}/05_review_applied.json", "w"), ensure_ascii=False, indent=1)
    print(f"[决策] 共 {len(saved)} 个候选")
    print(f"  整个词作废 : {drop_word or '无'}")
    print(f"  单个不生成 : {drop_one or '无'}")
    print(f"  逐实例生成 : {solo_word or '无'}")
    print(f"  共用一个mesh: {shared_word or '无'}")
    if relabel_extra:
        print(f"  分组->子类 : {len(relabel_extra)} 个实例挪进派生子类(然后整词复用)")
    if relabel: print(f"  挪类别     : {relabel}")
    print(f"  换候选     : {override or '无'}")
    print(f"  VLM放行    : {vlm_release or '无'}")
    print(f"\n下游用这些环境变量:")
    if drop_word: print(f"  EXCLUDE=\"$EXCLUDE,{','.join(drop_word)}\"")
    if solo_word: print(f"  SOLO=\"{','.join(solo_word)}\"")
    print(f"\n[done] -> {R1}/05_review_applied.json")
    sys.exit(0)

# ---------- 出联络图 ----------
# 每组的代表是谁 —— 摆在格子上, 人才看得出系统挑了谁、挑得对不对。
GRP_REP, REPS = {}, {}
_grp = {}
for r in rows:
    if r.get("复用组"): _grp.setdefault((r["词"], r["复用组"]), []).append(int(r["实例"]))
for (w, g), oids in _grp.items():
    _best, _bs = None, -1
    z = np.load(f"{R1}/03_masks/{w.replace(' ','_')}.npz", allow_pickle=True)
    for rr in z["meta"]:
        o = int(rr["oid"])
        if o not in oids or rr["area"] <= 0: continue
        sc = int(rr["area"]) * (1 - min(float(rr["edge"])*4, 0.9))
        if sc > _bs: _bs, _best = sc, o
    if _best is None: _best = sorted(oids)[0]
    GRP_REP[(w, g)] = _best
    REPS[(w, _best)] = True
    print(f"[分组] {w} 第{g}组 {sorted(oids)} -> 代表 #{_best} (抠得最全, 这一组共用它的 mesh)")

tiles = []
from pycocotools import mask as mask_util
_mk = {}
def cutout_from_mask(word, oid, fi, size):
    """逐实例候选在这一步还没写盘(那是 solo 干的活), 现场从 03_masks 抠一张出来看。"""
    k = word.replace(" ", "_")
    if k not in _mk:
        z = np.load(f"{R1}/03_masks/{k}.npz", allow_pickle=True)
        _mk[k] = (z["meta"], z["counts"], z["hw"])
    meta, counts, (H, W) = _mk[k]
    hit = [i for i, r in enumerate(meta) if int(r["oid"]) == oid and int(r["frame"]) == fi and r["area"] > 0]
    if not hit: return None
    m = mask_util.decode({"counts": counts[hit[0]], "size": [int(H), int(W)]}).astype(np.uint8)*255
    img = cv2.imread(f"{FR}/{fi:04d}.jpg")
    if img is None: return None
    tmp = f"/tmp/_rg_{os.getpid()}"
    cv2.imwrite(tmp+"_f.png", img); cv2.imwrite(tmp+"_m.png", m)
    out = cutout(tmp+"_f.png", tmp+"_m.png", size)
    for x in (tmp+"_f.png", tmp+"_m.png"):
        if os.path.exists(x): os.remove(x)
    return out

for r in rows:
    d = f"{IMG}/{r['生成名']}"
    # ★ 判据是"盘上有没有这个目录", 不是来源标签。
    #   来源标签是给人看的(逐实例/整词复用/类/召回), 它会跟着人的决定变;
    #   而"这张图从哪来"取决于 my_images 里有没有写过盘。用标签去判就会错 ——
    #   把标签改成"整词复用"之后, 那些实例目录并不存在, 整张表变成 mask empty。
    t = (cutout(f"{d}/frame.png", f"{d}/mask.png", TILE)
         if os.path.exists(f"{d}/frame.png")
         else cutout_from_mask(r["词"], r["实例"], r["帧"], TILE))
    if t is None:
        t = np.full((TILE, TILE, 3), 60, np.uint8)
        cv2.putText(t, "mask empty", (10, TILE//2), 0, 0.5, (255,255,255), 1)
    # 决定用颜色标出来, 光靠一行小字扫不过来 —— 一张图 60 多格, 要能一眼看出哪些被判了死刑
    BORDER = {"keep": (90,180,90), "drop": (60,150,240), "drop-word": (60,60,220), "solo": (220,180,60), "shared": (200,120,200)}
    bc = BORDER.get(r["决定"], (90,90,90))
    # 召回的一律标黄边 —— 它是主线漏掉的那一批, 质量最没保证, 要一眼能挑出来。
    # 决定的颜色让给内圈那条细线, 外圈的黄色优先。
    if r["来源"] == "召回": bc = (60, 220, 255)
    t = cv2.copyMakeBorder(t, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=bc)
    t = cv2.resize(t, (TILE, TILE))
    bar = np.full((30, TILE, 3), 30, np.uint8)
    cv2.rectangle(bar, (0,0), (TILE,30), bc, 3)
    mark = "keep" if r["决定"] == "keep" else r["决定"]
    col = (255,255,255) if r["决定"] == "keep" else (120,200,255)
    cv2.putText(bar, f"[{r['编号']}] {r['词'][:16]}", (5, 12), 0, 0.38, col, 1, cv2.LINE_AA)
    # cv2 的字体画不了中文, 会变成一排问号 —— 图上的标签一律用 ASCII
    # ★ 加了新的来源标签就必须同步加进这张表 —— 漏一个, 图上那一格就是一排问号。
    #   实测踩过两次: 加"整词复用"时忘了加映射, 40 格全是 ???。
    # 来源 = 这张图从哪来(出处), 决定 = 人要拿它怎么办。**两者不许用同一个词** ——
    # 第一版把"整词复用"塞进来源, 于是图上 "shared" 既可能是出处也可能是决定, 分不清。
    _SRC = {"类": "class", "召回": "recall", "逐实例": "inst"}
    src = _SRC.get(r["来源"], "".join(ch for ch in str(r["来源"]) if ord(ch) < 128) or "?")
    cv2.putText(bar, f"{src}  #{r['实例']}  f{r['帧']}  {mark}", (5, 25), 0, 0.34, (170,170,170), 1, cv2.LINE_AA)
    tiles.append(np.vstack([bar, t]))

pad = (-len(tiles)) % COLS
tiles += [np.zeros_like(tiles[0])]*pad
grid = np.vstack([np.hstack(tiles[i:i+COLS]) for i in range(0, len(tiles), COLS)])
cv2.imwrite(SHEET, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])

json.dump(dict(历史决定=HIST, 说明="把\"决定\"改成 keep / drop / drop-word / solo / shared; "
                  "换候选加 \"改用\": [实例号, 帧号]; "
                  "词内分组复用加 \"复用组\": 1/2/3; 挪类别加 \"改类别\": \"另一个词\"",
               场景=TAG, 候选=rows), open(DEC, "w"), ensure_ascii=False, indent=1)

from collections import Counter as _C


_n = _C(r["来源"] for r in rows)
print(f"[候选] {len(rows)} 个待生成: " + " + ".join(f"{k} {v} 个" for k, v in _n.most_common()))
print(f"[联络图] {SHEET}   ({COLS} 列, 每格标着编号/词/来源/实例/帧)")
print(f"[决策文件] {DEC}")
print(f"\n★ 停在这里。人看完联络图, 改决策文件, 然后跑 --apply。")
if _rec_pending: _check_pending(rows + _dropped)
