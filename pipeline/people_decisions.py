#!/usr/bin/env python
"""挑人闸的决策文件 <R1>/05_people.json —— 室内外两条链、驱动、闸口网页共用这一份实现。

★ 为什么有这个(2026-09-24)
  人体线原来是独立的 body4d_pipeline.py, 挑人只是命令行里一句"看 03_check.jpg 再填 PICK"。
  soccer 上跟踪按"最大面积 < 1% 就整条丢"把守门员(0.7%)静默丢了, 没人看见。
  现在: 跟踪只丢 < PEOPLE_MIN_AREA 的碎片, 其余**全部列出来**, 帧数够的默认勾上,
  人在第 2 道闸取消不要的。宁可多列一个让人去掉, 也不再静默丢人。

文件形状:
  {"说明": ..., "人": [{"id": 1, "帧数": 69, "最大面积%": 3.5, "最大面积帧": 7, "告警": 1, "决定": "keep"}, ...], "改过": true}
  id 是 body4d 跟踪重编号后的 id(和 03_check.jpg 上的数字一致)。决定只认 keep / drop。
  "改过": 有人保存过决定(网页上点了保存, 或调了 save_decisions)才有; 没有 = 全是自动默认的。

★ 2026-10-09: 这里的 id 也是往后所有地方的人号(10_people.json、验收视频、FBX 里的 P<号>), 见 people_identity.py。
  table() 把"挑人的决定"和"摆人那步核对出来的身份、可疑原因"并成验收那道闸上的一张表。

只用标准库(+ 同目录的 _common)。别往里加场景相关的东西。
"""
import json, os

import _common

FILE = "05_people.json"
DECISIONS = ("keep", "drop")
RERUN_MARK = "logs/.people_rerun_pending"     # 驱动按新挑的人重跑时先放下, 全部跑成了才拿走(见 drive.rerun_people)


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
    d["改过"] = True                    # 出网格那步据此如实记"谁定的"(picked.json 的 decided_by)
    _save(R1, d)
    return d


def decided_by(R1):
    """现在勾的这批人是谁定的: 有人保存过决定 = human; 没人动过 = 自动默认(帧数够的全勾上)。"""
    d = load(R1)
    return "human" if (d or {}).get("改过") else "auto(帧数够的默认勾上, 闸口上没人改过)"


def note_none_picked(R1):
    """挑人闸上把人全点掉了: 出网格那步跳过, 但"上次出网格用的是谁"(body4d 的 picked.json)要跟着改成空。
    驱动靠它判断勾的人变没变(drive.people_changed) —— 不改的话以后每次点继续都当成刚改过, 白白重跑物理和渲染。"""
    p = f"{auto_dir(R1)}/02_body4d/picked.json"
    if not os.path.exists(p):
        return False
    d = json.load(open(p))
    if not d.get("keep_ids"):
        return False
    d.update(keep_ids=[], coverage={}, decided_by=decided_by(R1))
    tmp = p + ".tmp"
    json.dump(d, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    return True


def table(R1):
    """验收那道闸"人"那张表(网页和驱动的闸口报告都用): 每个跟踪编号一行 —— 现在勾没勾、最后算谁、可疑的原因。
    只读两个文件: 05_people.json(挑人的决定)和 10_people.json(摆人那步记下的身份和报告)。
    摆人还没跑过、或者是 2026-10-09 之前的老写法(里面没有身份) -> None。
    返回 {"过期": 现在勾的人和摆人那一次的不一样, 或者上一次按新挑的人重跑没跑完(两种情况点继续都会重跑),
          "重跑没完": 是后一种, "人数", "人号", "行": [
            {"id", "决定": keep/drop/None, "去向": [...见 body_motion.identity_record], "可疑": [原因], "坐着": bool/None,
             "新号": 这个人号不是跟踪编号(一个编号被拆成两个人时另起的)}]}"""
    try:
        m = json.load(open(f"{R1}/10_people.json"))
    except (OSError, ValueError):
        return None
    ident = m.get("身份")
    if not ident:
        return None
    d = load(R1)
    rep, people, fate = m.get("报告", {}), m.get("people", {}), ident["去向"]
    dec = {r["id"]: r["决定"] for r in d["人"]} if d else {i: "keep" for i in ident["勾的人"]}
    rows = []
    for i in sorted(set(dec) | {int(k) for k in fate}):
        k = str(i)
        rows.append({"id": i, "决定": dec.get(i), "去向": fate.get(k, []), "可疑": (rep.get(k) or {}).get("可疑") or [],
                     "坐着": people[k]["seated"] if k in people else None, "新号": False})
    for k in sorted(set(people) - {str(r["id"]) for r in rows}, key=int):       # 另起的人号: 没有对应的跟踪编号
        rows.append({"id": int(k), "决定": None, "去向": [{"事": "人", "人": k, "帧数": (rep.get(k) or {}).get("帧数")}],
                     "可疑": (rep.get(k) or {}).get("可疑") or [], "坐着": people[k]["seated"], "新号": True})
    keep = sorted(i for i, v in dec.items() if v == "keep")
    pending = os.path.exists(f"{R1}/{RERUN_MARK}")
    return {"过期": pending or keep != sorted(ident["勾的人"]), "重跑没完": pending,
            "人数": len(people), "人号": sorted(int(k) for k in people), "行": rows}


def fate_text(row):
    """表里一行的"去向"写成一句话(网页上是同一句, run_page.py 的 fateText)。"""
    if not row["去向"]:
        return "没出网格(闸口上没勾)" if row["决定"] != "keep" else "摆人那一次还没勾上"
    out = []
    for f in row["去向"]:
        span = f"视频第 {f['视频帧'][0]}~{f['视频帧'][1]} 帧" if f.get("视频帧") else ""
        if f["事"] == "人":
            out.append(f"人 {f['人']}" + (f"({f['帧数']} 帧)" if f.get("帧数") else "") + (", 坐着" if row.get("坐着") else ""))
        elif f["事"] == "并进别人":
            out.append(f"{span}({f['帧数']} 帧)并进人 {f['人']}")
        else:
            out.append(f"{span}({f['帧数']} 帧)不算人")
    return "; ".join(out)


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
