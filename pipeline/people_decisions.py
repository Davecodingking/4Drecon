#!/usr/bin/env python
"""挑人闸的决策文件 <R1>/05_people.json —— 室内外两条链、驱动、闸口网页共用这一份实现。

★ 为什么有这个(2026-09-24)
  人体线原来是独立的 body4d_pipeline.py, 挑人只是命令行里一句"看 03_check.jpg 再填 PICK"。
  soccer 上跟踪按"最大面积 < 1% 就整条丢"把守门员(0.7%)静默丢了, 没人看见。
  现在: 跟踪只丢 < PEOPLE_MIN_AREA 的碎片, 其余**全部列出来**, 帧数够的默认勾上,
  人在第 2 道闸取消不要的。宁可多列一个让人去掉, 也不再静默丢人。

文件形状:
  {"说明": ..., "人": [{"id": 1, "帧数": 69, "最大面积%": 3.5, "最大面积帧": 7, "告警": 1, "决定": "keep"}, ...]}
  id 是 body4d 跟踪重编号后的 id(和 03_check.jpg 上的数字一致)。决定只认 keep / drop。

只用标准库(+ 同目录的 _common)。别往里加场景相关的东西。
"""
import json, os

import _common

FILE = "05_people.json"
DECISIONS = ("keep", "drop")


def path(R1):
    return f"{R1}/{FILE}"


def load(R1):
    p = path(R1)
    return json.load(open(p)) if os.path.exists(p) else None


def _save(R1, d):
    tmp = path(R1) + ".tmp"
    json.dump(d, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, path(R1))          # 网页和驱动都会写它, 原子替换免得读到半截


def build(R1, trk, min_frames=5):
    """从 body4d 的 track_report.json 出一份新的决策文件(每次跟踪重跑都重出 —— id 会变, 旧决定不作数)。
    帧数 >= min_frames 的默认 keep, 其余 drop 但照样列出来。"""
    rep = json.load(open(f"{trk}/track_report.json"))
    W, H = rep["wh"]
    alarms = {}
    for a in rep.get("alarms", []):
        alarms[a["id"]] = alarms.get(a["id"], 0) + 1
    rows = []
    for i in rep["kept_ids"]:
        pf = rep["per_frame"].get(str(i), {})
        best = max(pf, key=lambda f: pf[f]["area"]) if pf else None
        n = rep["coverage"].get(str(i), 0)
        rows.append({"id": i, "帧数": n,
                     "最大面积%": round(pf[best]["area"] / (W * H) * 100, 2) if best else 0.0,
                     "最大面积帧": int(best) if best else None,
                     "告警": alarms.get(i, 0),
                     "决定": "keep" if n >= min_frames else "drop"})
    d = {"说明": "决定只认 keep / drop。id 和 03_check.jpg 上的数字一致。改完在第 2 道闸 --go。",
         "跟踪目录": trk, "人": rows}
    _save(R1, d)
    return d


def keep_ids(R1):
    """勾上的 id 列表; 还没有决策文件(人体线没跑过)返回 None —— 和"一个都不要"的 [] 是两回事。"""
    d = load(R1)
    if d is None:
        return None
    return sorted(r["id"] for r in d["人"] if r["决定"] == "keep")


def save_decisions(R1, changes):
    """{id: "keep"/"drop"} 并进决策文件。不认识的值直接抛, 不许写进盘里。"""
    bad = {k: v for k, v in changes.items() if v not in DECISIONS}
    if bad:
        raise ValueError(f"决定只认 {'/'.join(DECISIONS)}: {bad}")
    d = load(R1)
    if d is None:
        raise ValueError(f"{path(R1)} 不存在 —— 人体跟踪还没跑")
    ids = {r["id"] for r in d["人"]}
    miss = [k for k in changes if int(k) not in ids]
    if miss:
        raise ValueError(f"没有这些 id: {miss}, 有的是 {sorted(ids)}")
    for r in d["人"]:
        if r["id"] in changes or str(r["id"]) in changes:
            r["决定"] = changes.get(r["id"], changes.get(str(r["id"])))
    _save(R1, d)
    return d


def auto_dir(R1):
    """人体线自动模式的 body4d 目录: 场景轮次底下的 body4d/(跟踪视频、网格叠回原片都在这)。
    ★ 2026-09-24 前在 sam-body4d/outputs/<场景>/<轮次>, 用户要"一轮的东西都在 outdoor 结果目录里"。"""
    return f"{R1}/body4d"


def resolve_body_run(cfg, scene, run, R1):
    """PEOPLE=1 且没手工指定 BODY_RUN 时, BODY_RUN 自动指向 <R1>/body4d。
    挑人闸一个都没勾 -> BODY_RUN 置空, 下游(尺度/摆人/物理)一律当"没有人"。
    原地改 cfg; 返回"人体线归不归这条流水线管"(people_track / people_mesh 两步看它)。
    手工指定 BODY_RUN 的老轮次(soccer b2、office b1)不归它管: 那是 body4d_pipeline.py 单独跑出来的。"""
    rc = _common.load_run_config(R1)
    if rc is not None and "BODY_RUN" not in rc.get("覆盖", {}) and not os.environ.get("BODY_RUN"):
        # 有本轮配置的新轮次: 有没有人只听本轮配置。SCENES 里写死的 BODY_RUN(soccer 的 b2)
        # 是给老轮次的, 不许劫持新开的轮次 —— 否则网页上勾了"有人", 人体线却一步不跑。
        cfg["BODY_RUN"] = ""
    if cfg.get("PEOPLE") != "1" or cfg.get("BODY_RUN", "").strip():
        return False
    cfg["BODY_RUN"] = "" if keep_ids(R1) == [] else auto_dir(R1)
    return True
