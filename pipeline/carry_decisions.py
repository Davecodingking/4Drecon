"""闸 5 上"被人拿着的物体"要不要进 FBX(2026-11-09)。和 people_decisions.py(挑人)同一种做法: 网页写决定, 驱动比对后重跑。

  <R1>/13_carry/summary.json   carry 步的结果(每个物体求没求出来、四段轮廓对齐)          carryfit/carry_step.py 写
  <R1>/13_carry/去留.json      人在闸 5 点的: {"不要": [生成名...]}                      网页(run_ctl.set_carry) 写
  <R1>/12_motion/carried.json  motion 步实际导进 FBX 的: {"导出": [生成名...]}           motion_export.py 写

"过期" = 现在该导出的(求出来了、没点不要) != FBX 里实际有的。驱动点继续时看到过期, 只重跑 motion(导出), 别的不动。
★ 只管进不进 FBX。要让它重新当静物摆、或者重新拟合, 得回闸 2 改 held, 那是另一条路。
"""
import json
import os

SUMMARY = "13_carry/summary.json"
DECIDE = "13_carry/去留.json"
USED = "12_motion/carried.json"


def _j(p, d):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return d


def rows(R1):
    """carry 步的结果, 每个物体一行。没跑过 carry(或这一轮没标 held)返回 []。"""
    return list(_j(f"{R1}/{SUMMARY}", {}).get("物体") or [])


def dropped(R1):
    return set(_j(f"{R1}/{DECIDE}", {}).get("不要") or [])


def want(R1):
    """该进 FBX 的: 求出来了、人没点不要。motion_export 按这个导。"""
    no = dropped(R1)
    return sorted(r["生成名"] for r in rows(R1) if r.get("状态") == "ok" and r["生成名"] not in no)


def exported(R1):
    """FBX 里实际有的。motion 没跑过 = None; 跑过但那时还没有这个记录(老轮次) = []。"""
    if not os.path.isdir(f"{R1}/12_motion"):
        return None
    return sorted(_j(f"{R1}/{USED}", {}).get("导出") or [])


def stale(R1):
    """FBX 和现在的决定对不上(要重新导出)。motion 还没跑过不算 —— 那时按现在的决定导就是了。"""
    ex = exported(R1)
    return ex is not None and ex != want(R1)


def write_used(R1, names):
    os.makedirs(f"{R1}/12_motion", exist_ok=True)
    json.dump({"导出": sorted(names)}, open(f"{R1}/{USED}", "w"), ensure_ascii=False, indent=1)


def set_decisions(R1, dec):
    """网页点的: {生成名: "keep" | "drop"}。只认 carry 步结果里有的物体。返回现在不要的那几个。"""
    have = {r["生成名"] for r in rows(R1)}
    bad = [k for k in dec if k not in have]
    if bad:
        raise ValueError(f"没有这些被拿着的物体: {bad}")
    if any(v not in ("keep", "drop") for v in dec.values()):
        raise ValueError("决定只能是 keep / drop")
    no = dropped(R1)
    for k, v in dec.items():
        (no.add if v == "drop" else no.discard)(k)
    p = f"{R1}/{DECIDE}"
    tmp = p + ".tmp"
    json.dump({"不要": sorted(no)}, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    return sorted(no)


def table(R1):
    """闸 5 页面那张表。没有被拿着的物体返回 None。"""
    rs = rows(R1)
    if not rs:
        return None
    no, ex = dropped(R1), exported(R1)
    gate = _j(f"{R1}/{SUMMARY}", {}).get("门槛", 0.5)
    out = []
    for r in rs:
        out.append(dict(生成名=r["生成名"], 词=r["词"], 实例=r["实例"], 状态=r.get("状态"), 原因=r.get("原因"),
                        轮廓IoU=r.get("轮廓IoU"), 四段=r.get("四段") or {}, 低的段=r.get("低的段") or [],
                        网格来源=r.get("网格来源"), 决定="drop" if r["生成名"] in no else "keep",
                        在FBX里=None if ex is None else r["生成名"] in ex))
    return dict(行=out, 门槛=gate, 过期=stale(R1), 导出过=ex is not None)
