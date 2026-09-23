#!/usr/bin/env python
"""复用表: 每一个"会被生成的 mesh"占一行, 左边是代表(真去生成的那张抠图),
右边是所有共用它的实例。人一眼看清"合并前是哪些东西、合并后长什么样"。

为什么单独一张表, 不塞进联络图 ——
  联络图回答的是"这一个候选要不要生成", 一格一个候选, 逐格判。
  复用表回答的是"这一批东西该不该长成同一个样子", 一行一组, 横着比。
  两个问题的粒度不一样, 摆在一起谁都看不清。

三种来源, 表里都摊平成"一行 = 一个 mesh":
  整词复用 (shared)   一个词全部实例共用一个 -> 一行, 右边是全部实例
  分组复用 (复用组)   词内分几组, 每组一个   -> 每组一行
  逐实例   (solo)     各生成各的           -> 一行只有它自己, 右边空

代表怎么挑: 面积 x (1 - 贴边惩罚), 在这个实例活过的帧里取最高 —— 抠得最全的那个。
共用的只是**形状**, 位置还是各自的盒子。

用法:
  R1=... TAG=... FR=... python reuse_table.py
输出:
  05_reuse_table.jpg   图
  05_reuse_table.json  同样内容的表, 给人改用
"""
import os, sys, json
import numpy as np, cv2

R1   = os.environ["R1"]
TAG  = os.environ["TAG"]
FR   = os.environ["FR"]
MK   = f"{R1}/03_masks"
# ★ 按阶段分两份, 和联络图一个规矩 —— **不许覆盖**。
#   闸 2(生成前) 和 闸 4(召回并进来之后) 的复用清单不一样, 各自占一个文件名,
#   这样"回退到 recall 之前"就是直接看不带后缀的那份。
# 阶段由驱动显式告知, 和 review_gate 一个口径。不靠"盘上有没有 14_recall_ready.json"猜 ——
# 那个文件生成之后就永远在, 于是任何一次重跑「生成前」都会拿召回前的清单
# 覆盖掉召回后的那张表。
_STAGE = (os.environ.get("REVIEW_STAGE")
          or ("post" if os.path.exists(f"{R1}/14_recall_ready.json") else "pre")).lower()
_AFTER = _STAGE == "post"
_SFX   = "_afterrecall" if _AFTER else ""
OUT  = f"{R1}/05_reuse_table{_SFX}.jpg"
OUTJ = f"{R1}/05_reuse_table{_SFX}.json"
TILE = int(os.environ.get("REUSE_TILE", "150"))
MAXN = int(os.environ.get("REUSE_MAXN", "9"))      # 一行最多画几个成员, 多的用数字带过

SOLO_CFG = os.environ.get("SOLO", "")
KEEP     = {w.strip() for w in os.environ.get("SOLO_KEEP", "").split(",") if w.strip()}
EXCL     = {w.strip() for w in os.environ.get("EXCLUDE", "").split(",") if w.strip()}

# 算法模块(world_frame / word_rank / review_decisions)住在 data/outdoor/scripts/,
# 这个脚本在 tools/ —— realpath 解析真身之后往上一层就是仓库根。
# 用 realpath 不用 abspath: 万一哪天又有人建软链接, abspath 会往上数到错的目录去。
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "data", "outdoor", "scripts"))
from review_decisions import load as load_dec                        # noqa: E402



# ★ 编号必须和联络图(05_review.json)一致 —— 两张图指同一个东西时人要能对上号。
#   没有编号的图人根本没法跟你沟通"我要换这个"。
SHEET = f"{R1}/05_review{_SFX}.json"
SHEET_ID = {}
if os.path.exists(SHEET):
    for r in json.load(open(SHEET)).get("候选", []):
        SHEET_ID[r["生成名"]] = r["编号"]


def sid(word, oid):
    """这个实例在联络图上的编号。联络图里没有就返回 None(表里会显示 -)。"""
    return SHEET_ID.get(f"{TAG}_{word.replace(' ', '_')}_{oid}")

_cache = {}


def mask_of(word, oid, fi, name=None):
    """现场抠一张灰底特写。
    召回的实例 mask 不在 npz 里(写在 my_images/<生成名>/), 所以先看盘上有没有现成的。"""
    if name:
        d = os.path.expanduser(f"~/sam3d/sam-3d-objects/my_images/{name}")
        if os.path.exists(f"{d}/frame.png") and os.path.exists(f"{d}/mask.png"):
            img = cv2.imread(f"{d}/frame.png"); mk = cv2.imread(f"{d}/mask.png", 0)
            if img is not None and mk is not None:
                m = mk > 127
                if m.sum() >= 30:
                    out = np.full_like(img, 210); out[m] = img[m]
                    v, u = np.nonzero(m)
                    pad = int(0.12*max(u.max()-u.min(), v.max()-v.min())) + 8
                    y0,y1 = max(v.min()-pad,0), min(v.max()+pad, img.shape[0])
                    x0,x1 = max(u.min()-pad,0), min(u.max()+pad, img.shape[1])
                    c = out[y0:y1, x0:x1]
                    if c.size:
                        sc = TILE/max(c.shape[0], c.shape[1])
                        c = cv2.resize(c, (max(int(c.shape[1]*sc),1), max(int(c.shape[0]*sc),1)))
                        cv_ = np.full((TILE,TILE,3), 210, np.uint8)
                        oy,ox = (TILE-c.shape[0])//2, (TILE-c.shape[1])//2
                        cv_[oy:oy+c.shape[0], ox:ox+c.shape[1]] = c
                        return cv_
    from pycocotools import mask as mask_util
    k = word.replace(" ", "_")
    if k not in _cache:
        z = np.load(f"{MK}/{k}.npz", allow_pickle=True)
        _cache[k] = (z["meta"], z["counts"], z["hw"])
    meta, counts, (H, W) = _cache[k]
    hit = [i for i, r in enumerate(meta)
           if int(r["oid"]) == oid and int(r["frame"]) == fi and r["area"] > 0]
    if not hit:
        return None
    m = mask_util.decode({"counts": counts[hit[0]], "size": [int(H), int(W)]}).astype(bool)
    img = cv2.imread(f"{FR}/{fi:04d}.jpg")
    if img is None:
        return None
    out = np.full_like(img, 210)
    out[m] = img[m]
    v, u = np.nonzero(m)
    if len(u) == 0:
        return None
    pad = int(0.12 * max(u.max() - u.min(), v.max() - v.min())) + 8
    y0, y1 = max(v.min() - pad, 0), min(v.max() + pad, img.shape[0])
    x0, x1 = max(u.min() - pad, 0), min(u.max() + pad, img.shape[1])
    c = out[y0:y1, x0:x1]
    if c.size == 0:
        return None
    sc = TILE / max(c.shape[0], c.shape[1])
    c = cv2.resize(c, (max(int(c.shape[1] * sc), 1), max(int(c.shape[0] * sc), 1)))
    canvas = np.full((TILE, TILE, 3), 210, np.uint8)
    oy, ox = (TILE - c.shape[0]) // 2, (TILE - c.shape[1]) // 2
    canvas[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return canvas


def best_frames(word):
    """每个实例分最高的那一帧: {oid: (打分, 面积, 帧号, 活了几帧)}"""
    z = np.load(f"{MK}/{word.replace(' ', '_')}.npz", allow_pickle=True)
    meta, (H, W) = z["meta"], z["hw"]
    per, nfr = {}, {}
    for r in meta:
        if r["area"] <= 0:
            continue
        o = int(r["oid"])
        nfr[o] = nfr.get(o, 0) + 1
        sc = int(r["area"]) * (1 - min(float(r["edge"]) * 4, 0.9))
        if sc > per.get(o, (0,))[0]:
            per[o] = (sc, int(r["area"]), int(r["frame"]))
    return {o: (*v, nfr[o]) for o, v in per.items()}, int(H), int(W)


def main():
    # ★ 从**联络图的候选清单**构造, 不自己扫 03_masks。
    #   联络图那份已经是"最终会生成什么"的答案: 含召回、排除了 drop、跟着人的决定走。
    #   自己扫 npz 的话两张图必然对不上 —— 召回补抠的 mask 不在 npz 里(它写在
    #   14_recall_masks/ 和 my_images/), 而 drop 只存在决策文件里。
    #   实测踩过: 复用表里没有召回的两个物体, 却还列着人已经 drop 掉的 window。
    # 用**本阶段**那份联络图的候选清单。两处必须是同一个文件, 否则编号会漂
    # (实测 door 11->9、hot dog cart 14->81), 人照着报编号就指错东西。
    sheet = SHEET if os.path.exists(SHEET) else (
        f"{R1}/05_review.json" if os.path.exists(f"{R1}/05_review.json") else None)
    if sheet is None:
        raise SystemExit(f"[失败] 没有联络图的候选清单({R1}/05_review.json), 先跑 review_gate.py")
    cands = json.load(open(sheet))["候选"]

    dec = load_dec(R1)
    shared = {w.strip() for w in (dec.get("共用词") or []) if w.strip()} | KEEP

    # 每个实例的帧和面积直接用联络图里记好的, 不再重算 —— 两边算法一旦分叉就会不一致
    rows, by_word = [], {}
    for c in cands:
        if c.get("决定") in ("drop", "drop-word"): continue
        by_word.setdefault(c["词"], []).append(c)

    for w in sorted(by_word):
        items = sorted(by_word[w], key=lambda c: -(c.get("面积占比") or 0))
        tag = "整词复用" if (w in shared or (c0 := items[0]).get("决定") == "shared") else "逐实例"
        groups = [items] if tag == "整词复用" else [[c] for c in items]
        for g in groups:
            rep = g[0]                      # 已按面积降序, 第一个就是抠得最全的
            rows.append(dict(词=w, 类型=tag, 代表=rep["实例"], 代表编号=rep["编号"],
                             代表生成名=rep.get("生成名"),
                             成员=[c["实例"] for c in g],
                             代表帧=rep["帧"], 代表面积占比=rep.get("面积占比"),
                             来源=rep.get("来源"), 成员数=len(g),
                             各成员={str(c["实例"]): dict(编号=c["编号"], 帧=c["帧"],
                                                       面积占比=c.get("面积占比"),
                                                       生成名=c.get("生成名"),
                                                       来源=c.get("来源")) for c in g}))

    if not rows:
        raise SystemExit("[失败] 一行都算不出来 —— 确认 track/dedup 跑过")

    json.dump(dict(说明="一行 = 一个会被生成的 mesh。代表是真去生成的那张; 成员共用它的形状, 位置各是各的。"
                        "想换代表, 去 05_review.json 调分组或用\"改用\"换帧。",
                   行=rows), open(OUTJ, "w"), ensure_ascii=False, indent=1)

    # ---------- 画 ----------
    W_ROW = TILE * (MAXN + 2) + 40
    strips = []
    for r in rows:
        strip = np.full((TILE + 34, W_ROW, 3), 32, np.uint8)
        # ★ cv2 画不出中文, 会变成一排问号 —— 图上的标签一律 ASCII。
        #   review_gate.py 里早就记着这个坑, 这里第一版又踩了一次。
        _TY = {"整词复用": "word-shared", "第1组": "group1", "第2组": "group2", "第3组": "group3",
               "第4组": "group4", "第5组": "group5", "逐实例": "per-instance", "没分组": "ungrouped"}
        _rn = r.get("代表编号")
        bar = (f"[{_rn if _rn is not None else '-'}] {r['词']}  [{_TY.get(r['类型'], r['类型'])}]  "
               f"rep #{r['代表']} f{r['代表帧']} {r['代表面积占比']}%")
        if r["成员数"] > 1:
            bar += f"   {r['成员数']} instances share this mesh"
        cv2.putText(strip, bar, (8, 22), 0, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        # 代表: 绿框
        t = mask_of(r["词"], r["代表"], r["代表帧"], r.get("代表生成名"))
        if t is None:
            t = np.full((TILE, TILE, 3), 60, np.uint8)
        t = cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(90, 200, 90))
        t = cv2.resize(t, (TILE, TILE))
        strip[30:30 + TILE, 8:8 + TILE] = t
        cv2.putText(strip, f"[{_rn if _rn is not None else '-'}]", (12, 30 + 16), 0, 0.45,
                    (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(strip, f"[{_rn if _rn is not None else '-'}]", (12, 30 + 16), 0, 0.45,
                    (90, 240, 90), 1, cv2.LINE_AA)
        # 成员(不含代表): 灰框
        others = [o for o in r["成员"] if o != r["代表"]][:MAXN]
        for i, o in enumerate(others):
            m = mask_of(r["词"], o, r["各成员"][str(o)]["帧"],
                        r["各成员"][str(o)].get("生成名"))
            if m is None:
                m = np.full((TILE, TILE, 3), 60, np.uint8)
            m = cv2.copyMakeBorder(m, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(120, 120, 120))
            m = cv2.resize(m, (TILE, TILE))
            x = 8 + TILE * (i + 2)
            if x + TILE > W_ROW:
                break
            strip[30:30 + TILE, x:x + TILE] = m
            _mn = r["各成员"][str(o)].get("编号")
            _lab = f"[{_mn if _mn is not None else '-'}] #{o}"
            cv2.putText(strip, _lab, (x + 4, 30 + TILE - 6), 0, 0.4, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.putText(strip, _lab, (x + 4, 30 + TILE - 6), 0, 0.4, (60, 220, 220), 1, cv2.LINE_AA)
        hid = len([o for o in r["成员"] if o != r["代表"]]) - len(others)
        if hid > 0:
            cv2.putText(strip, f"+{hid}", (W_ROW - 60, 30 + TILE // 2), 0, 0.7, (200, 200, 200), 2)
        strips.append(strip)

    grid = np.vstack(strips)
    cv2.imwrite(OUT, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])

    n_mesh = len(rows)
    n_inst = sum(r["成员数"] for r in rows)
    print(f"[复用表] {n_inst} 个实例 -> {n_mesh} 个 mesh (省了 {n_inst - n_mesh} 次生成)")
    for r in rows:
        if r["成员数"] > 1:
            print(f"  {r['词']:16s} [{r['类型']}] 代表 #{r['代表']} <- {r['成员']}")
    print(f"[图]   {OUT}")
    print(f"[表]   {OUTJ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
