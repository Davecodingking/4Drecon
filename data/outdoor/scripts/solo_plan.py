#!/usr/bin/env python
"""逐实例生成: 每个实例生不生成、为什么不生成、用哪一帧 —— **只有这一份规则**。

谁在用:
  gen_per_instance.py   真正生成(按这里的结论裁图、列 solo_jobs.txt)
  tools/run_ctl.py      本轮页"按你现在保存的决定, 会生成 N 个 mesh"

★ 为什么抽出来(2026-11-10): hitch_cuts_indoor r1 上本轮页只减了整词作废, 说会生成 28 个;
  生成那步还挡了 VLM 否决 6 个、太矮 3 个, 实际 19 个, 候选页也说 19 —— 三个页面两个数。
  以后改"哪些不生成"只改这里, 两边自动一致。

不读环境变量(gen_per_instance 自己从环境读好再传进来), 所以网页进程也能直接调。
不管"整词作废 / 单个 drop / 共用词": 那些是候选表那一层的决定, 调用方先筛掉再给 words。
"""
import json, os
import numpy as np

import box_guards

# 不生成的原因分三类, 本轮页按类计数; 原因原文照旧打进生成日志
VLM_REJ = "VLM 判不是"
GUARD = "太矮或太大(当成量错了)"
DUP = "同一件量了两遍"
SMALL = "截图太小"
NO_MASK = "没有 mask"


def vlm_rejected(R1):
    """verify_picks.py 逐实例那一遍否掉的 {词: {实例号}}。文件不在当空 —— 老轮次没有这份。"""
    p = f"{R1}/04_picks/vlm_verify_inst.json"
    if not os.path.exists(p):
        return {}
    return {w: {int(x) for x in v} for w, v in (json.load(open(p)).get("否决") or {}).items()}


def best_frame(meta, oid):
    """实例的代表帧 (mask 行号, 像素, 帧号, 贴边): 面积大且不贴边。和 review_gate / gate_ui 同一个公式。"""
    rows = [(i, int(a), int(f), float(e)) for i, (o, f, a, e) in enumerate(meta) if int(o) == oid and a > 0]
    if not rows:
        return None, []
    return max(rows, key=lambda t: t[1] * (1 - min(t[3] * 4, 0.9))), rows


def plan(R1, boxes, words, *, tag, min_px, dup, rest, floor_extend, hung, vlm_rej, released, ov_inst):
    """逐个实例下结论。返回 [{词, 实例, 生成名, 原因(None = 生成), 类, 行, 提示}], 顺序同 words / boxes。

    boxes: 06_all_boxes.json 的内容。只有量出了盒子的实例才可能生成(和组装同一个前提)。
    dup: box_guards.dup_boxes(...) 的结果({} = 不去重)。
    released: 人在闸口放行的生成名(VLM 说不是也要)。ov_inst: {(词, 实例号): 帧号} 人点名换的帧。
    行: (mask 行号, 像素, 帧号, 贴边), 生成的才有。提示: 要打进日志的人工指定相关的话。
    """
    out = []
    for w in words:
        mp = f"{R1}/03_masks/{w.replace(' ', '_')}.npz"
        meta = np.load(mp, allow_pickle=True)["meta"] if os.path.exists(mp) else []
        for b in boxes.get(w, []):
            oid = b["obj_id"]
            name = f"{tag}_{w.replace(' ', '_')}_{oid}"
            r = dict(词=w, 实例=oid, 生成名=name, 原因=None, 类=None, 行=None, 提示=[])
            out.append(r)
            if oid in vlm_rej.get(w, ()) and name not in released:
                r["原因"], r["类"] = "VLM 说这不是这个词, 人也没放行", VLM_REJ
                continue
            why = box_guards.never_placed(
                w, b, dup,
                may_grow=floor_extend and box_guards.rests_on_floor(w, rest, hung) and box_guards.may_extend_legs(w),
                rest=rest, hung=hung)
            if why:
                r["原因"], r["类"] = why, (DUP if (w, oid) in dup else GUARD)
                continue
            best, rows = best_frame(meta, oid)
            if best is None:
                r["原因"], r["类"] = "没有任何一帧有 mask", NO_MASK
                continue
            want = ov_inst.get((w, oid))
            if want is not None:
                hit = [x for x in rows if x[2] == want]
                if hit:
                    best = hit[0]
                    r["提示"].append(f"[人工指定] {w}#{oid} 改用第 {best[2]} 帧")
                else:
                    r["提示"].append(f"[人工指定] ★ {w}#{oid} 点的第 {want} 帧没有这个实例的 mask, "
                                   f"退回自动挑的第 {best[2]} 帧 —— 换个帧号重填")
            if best[1] < min_px:
                r["原因"], r["类"] = f"第 {best[2]} 帧只有 {best[1]} 像素 < SOLO_MIN_PX={min_px}", SMALL
                if want is not None:
                    r["提示"].append(f"[人工指定] ★ {w}#{oid} 第 {best[2]} 帧只有 {best[1]} 像素 "
                                   f"< SOLO_MIN_PX={min_px}, 仍然不生成")
                continue
            r["行"] = best
    return out


def plan_from_cfg(R1, cfg, released=None, ov_inst=None, words=None):
    """按 pipeline --dump-cfg 的配置算(网页那边用)。参数和 o_solo / s_solo 传给 gen_per_instance 的是同一组:
    召回开着时不去重(那时组装读的盒子文件不是这一份), 立在地上的类可能被补高所以不按量到的高度筛。
    released / ov_inst 不给就读落地过的决策文件(05_review_applied.json), 和生成那步一样。"""
    import review_decisions as review
    from world_frame import load_up
    boxes = json.load(open(f"{R1}/06_all_boxes.json"))
    exclude = {x.strip() for x in str(cfg.get("EXCLUDE", "")).split(",") if x.strip()}
    if words is None:
        solo = {x for x in str(cfg.get("SOLO", "")).split(",") if x}
        keep = {x for x in str(cfg.get("SOLO_KEEP", "")).split(",") if x}
        words = (set(boxes) - keep if "*" in solo else solo) - exclude
    dedup = str(cfg.get("DEDUP_BOX", "0")) == "1" and str(cfg.get("RECALL", "0")) != "1"
    dup = box_guards.dup_boxes(boxes, load_up(f"{R1}/02_da3", "逐实例生成")[0], exclude, on=dedup) if dedup else {}
    rp = f"{R1}/01_vlm_rest.json"
    hung = {x.strip().replace("_", " ") for x in str(cfg.get("HUNG_LABELS", "")).split(",") if x.strip()}
    return plan(R1, boxes, sorted(words),
                tag=cfg.get("TAG") or "", min_px=int(cfg.get("SOLO_MIN_PX", 256)), dup=dup,
                rest=json.load(open(rp)) if os.path.exists(rp) else {},
                floor_extend=str(cfg.get("FLOOR_EXTEND", "0")) == "1", hung=hung,
                vlm_rej=vlm_rejected(R1),
                released=review.vlm_released(R1) if released is None else released,
                ov_inst=review.overrides_inst(R1) if ov_inst is None else ov_inst)
