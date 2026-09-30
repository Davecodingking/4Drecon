#!/usr/bin/env python
"""把参考轮次落地的闸口决定(05_review_applied.json)照抄到新轮次的 05_review.json。
  $PY tools/apply_ref_decisions.py <新轮次目录> <参考轮次的 05_review_applied.json> <参考轮次的标签>
验证用(tools/run_to_end.sh 自动过第 2 道闸时调它) —— 日常是人在网页上看着候选做决定。
对法: 整词的决定(作废 / 逐实例 / 共用)按词对; 单个不生成的按"词_实例"对(生成名去掉标签前缀), 两轮的标签不一样也对得上。
对不上的(新轮次里没有这个词, 或者是开了召回才有的实例)打印出来, 不报错。"""
import json, sys


def apply(R1, ref_applied, ref_tag):
    ref = json.load(open(ref_applied))
    p = f"{R1}/05_review.json"
    d = json.load(open(p))
    rows = d["候选"]
    for r in rows:
        r["决定"] = "keep"
    key = lambda r: f"{r['词'].replace(' ', '_')}_{r['实例']}"
    drop = {n[len(ref_tag) + 1:]: n for n in ref.get("不生成", []) if n.startswith(ref_tag + "_")}
    hit = set()
    for r in rows:                                   # 单个不生成
        if key(r) in drop:
            r["决定"] = "drop"; hit.add(drop[key(r)])
    miss = {}
    for kind, val in (("共用词", "shared"), ("逐实例词", "solo"), ("排除词", "drop-word")):   # 后面的压过前面的
        found = set()
        for r in rows:
            if r["词"] in ref.get(kind, []) and r["决定"] != "drop":
                r["决定"] = val; found.add(r["词"])
        miss[kind] = sorted(set(ref.get(kind, [])) - found)
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
    n = {v: sum(r["决定"] == v for r in rows) for v in ("keep", "drop", "drop-word", "solo", "shared")}
    return {"候选": len(rows), "决定": n, "没对上的排除词": miss["排除词"], "没对上的逐实例词": miss["逐实例词"],
            "没对上的共用词": miss["共用词"], "没对上的不生成": sorted(set(ref.get("不生成", [])) - hit)}


if __name__ == "__main__":
    print("    闸 2 照抄决定:", json.dumps(apply(*sys.argv[1:4]), ensure_ascii=False))
