"""整类共用一个 mesh 时, "样本从谁身上抠" 的唯一一份规则。

三处在用, 以前各写各的:
  pick_from_masks.py  挑图: 取排第一的
  verify_picks.py     VLM 复核: 取前 TOPK 个给 VLM 做选择题
  tools/gate_ui.py    闸口前端: 预告"不钉的话会用谁"(后端照这里的公式打分)

★ 为什么要抽出来 (2026-09-17):
  pick 早就不许"去重淘汰的 / VLM 否决没放行的 / 人 drop 的"当样本, verify 却从全部实例里重挑 ——
  VLM 选择题一换, 人 drop 掉的、被别的词代表走的实例又回来当了整类样本。
  pick 的帧数下限、帧数饱和读环境变量, verify 写死 2 和 10, 改一边另一边不知道。

只用 numpy 和标准库 —— sam3d / trellis2 两个环境都要 import 它。
"""
import json
import os

MIN_FRAMES = int(os.environ.get("MIN_FRAMES", "2"))       # 硬下限, 防单帧误检
FRAME_SAT = float(os.environ.get("FRAME_SAT", "10"))      # 帧数加分到此饱和

# picks.json 里每个词多记一个字段: 挑这张图时哪些实例被挡在门外(人 drop / VLM 否决没放行)。
# 闸口和 drive.py 拿它和"现在的决定"比, 不一样就说明盘上这张图是按旧决定挑的, 要重挑。
BANNED_KEY = "样本排除"


def frame_score(area, edge):
    """一帧的分: 面积 x (1 - 贴边惩罚)。贴边 0.25 以上罚到只剩一成。"""
    return area * (1 - min(edge * 4, 0.9))


def dedup_dead(R1):
    """跨词去重判掉的实例 {词: {实例号}} —— 它们由同组另一个词代表, 不许再当样本。"""
    p = os.environ.get("DEDUP", f"{R1}/03_dedup_words.json")
    if not (p and os.path.exists(p)):
        return {}
    return {w: {int(x) for x in v} for w, v in (json.load(open(p)).get("淘汰实例") or {}).items()}


def banned_fn(R1, tag):
    """(词, 实例号) -> 这个实例现在能不能当整类样本。人 drop 的、VLM 否决而人没放行的不能。"""
    import review_decisions as review
    rej, rel, skip = review.vlm_rejected(R1), review.vlm_released(R1), set(review.skip_names(R1))

    def banned(word, oid):
        g = f"{tag}_{word.replace(' ', '_')}_{oid}"
        return g in skip or (oid in rej.get(word, ()) and g not in rel)
    return banned


def ranked(meta, have, word, dead=(), banned=None, min_frames=None, frame_sat=None):
    """一个词的整类样本候选, 分数从高到低。

    meta: 03_masks/<词>.npz 的 meta 数组 (oid, frame, area, edge)
    have: 有深度的帧号集合 —— 只在这些帧里挑
    返回 (候选, 排除)
      候选: [(分, 实例号, meta 行号, 帧号, 面积, 贴边, 帧数)], 每个实例一条(它最好的那帧)
      排除: 被 banned 挡掉的实例号集合(不含去重淘汰的) —— 在 meta 里出现过就算, 不看有没有深度帧,
            这样闸口拿全部实例算出来的集合能和它逐个比
    同分时保持 meta 里的先后(和 pick 原来"严格大于才换"的写法一致)。
    """
    min_frames = MIN_FRAMES if min_frames is None else min_frames
    frame_sat = FRAME_SAT if frame_sat is None else frame_sat
    live, out_ban, seen = {}, set(), set()
    for i, (oid, fi, a, e) in enumerate(meta):
        oid = int(oid)
        if a == 0 or oid in dead:
            continue
        if oid not in seen:
            seen.add(oid)
            if banned is not None and banned(word, oid):
                out_ban.add(oid)
        if oid in out_ban or int(fi) not in have:
            continue
        live.setdefault(oid, []).append((i, int(fi), int(a), float(e)))
    cands = []
    for oid, rows in live.items():
        if len(rows) < min_frames:
            continue
        i, fi, a, e = max(rows, key=lambda t: frame_score(t[2], t[3]))
        cands.append((frame_score(a, e) * min(len(rows) / frame_sat, 1.0), oid, i, fi, a, e, len(rows)))
    cands.sort(key=lambda t: -t[0])          # sort 稳定: 同分保持先后
    return cands, out_ban, live


def stale_words(R1, tag):
    """picks.json 里哪些词的样本是按**旧决定**挑的 —— 挑的时候挡掉的实例, 和现在该挡掉的不一样。

    ★ 为什么要有: drive.py 原来只在"排除词/共用词/指定样本/VLM放行..."变了时才重跑挑图,
      人只 drop 了一个实例("不生成")不重跑 —— 可那个实例恰好是整类样本的话, 整类 mesh
      照样拿它抠的图生成, 人 drop 了等于没 drop。反过来, 撤销一个 drop 也可能让样本换人。
    老轮次的 picks.json 没记 样本排除, 这种词跳过不判(没法判, 不瞎报)。
    """
    import numpy as np
    p = f"{R1}/04_picks/picks.json"
    if not os.path.exists(p):
        return []
    picks = json.load(open(p))
    banned = banned_fn(R1, tag)
    dead = dedup_dead(R1)
    out = []
    for word, v in sorted(picks.items()):
        if not isinstance(v, dict) or BANNED_KEY not in v:
            continue
        f = f"{R1}/03_masks/{word.replace(' ', '_')}.npz"
        if not os.path.exists(f):
            continue
        meta = np.load(f, allow_pickle=True)["meta"]
        oids = {int(o) for o, a in zip(meta["oid"], meta["area"]) if a > 0} - dead.get(word, set())
        now = sorted(o for o in oids if banned(word, o))
        if now != sorted(v[BANNED_KEY]):
            out.append(word)
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4 or sys.argv[1] != "stale":
        sys.exit("用法: class_sample.py stale <R1> <TAG>   -> 打印 JSON: 样本按旧决定挑的词")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    print(json.dumps(stale_words(sys.argv[2], sys.argv[3]), ensure_ascii=False))
