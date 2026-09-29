import os, re, sys, glob, datetime as dt
R = sys.argv[1]; L = f"{R}/logs"
H  = re.compile(r"^== (\S+)\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", re.M)
TS = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
P = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
secs = []
for f in sorted(glob.glob(f"{L}/*.log")):
    txt = open(f, errors="ignore").read()
    hs = list(H.finditer(txt))
    # 同一步同一时刻的多个头(逐物体各打一次)合并成一次运行
    groups = []
    for i, m in enumerate(hs):
        key = (m.group(1), m.group(2))
        end = hs[i+1].start() if i+1 < len(hs) else len(txt)
        if groups and groups[-1]["key"] == key: groups[-1]["end"] = end
        else: groups.append(dict(key=key, start=m.start(), end=end))
    for gi, g in enumerate(groups):
        body = txt[g["start"]:g["end"]]
        t0 = P(g["key"][1])
        inner = [P(x) for x in TS.findall(body)]
        inner = [x for x in inner if x >= t0]
        secs.append(dict(log=os.path.basename(f), step=g["key"][0], t0=t0, inner=max(inner) if inner else None,
                         last=(gi == len(groups)-1), mtime=dt.datetime.fromtimestamp(os.path.getmtime(f)),
                         skip=body.count("跳过"), obj=body.count("===== ["), words=body.count("个实例,")))
starts = sorted(s["t0"] for s in secs)
for s in secs:
    nxt = next((t for t in starts if t > s["t0"]), None)
    cands = []
    if s["inner"] and s["inner"] > s["t0"]: cands.append((s["inner"], "内部时间戳"))
    if s["last"]: cands.append((s["mtime"], "日志mtime"))
    if nxt: cands.append((nxt, "下一步起点"))
    # 优先: 内部时间戳 > (最后一段)日志 mtime > 下一步起点; 但都不能晚于下一步起点太多
    end, how = cands[0] if cands else (s["t0"], "?")
    if how == "日志mtime" and nxt and end > nxt and s["inner"] is None: end, how = nxt, "下一步起点"
    s["dur"] = (end - s["t0"]).total_seconds(); s["how"] = how
for s in sorted(secs, key=lambda x: x["t0"]):
    flag = "  <-超1小时,多半含人工等待" if s["dur"] > 3600 else ""
    print(f"{s['t0']:%m-%d %H:%M:%S} {s['step']:15s} {s['dur']:7.0f}s [{s['how']}] 物体={s['obj']:3d} 词={s['words']:3d} 跳过={s['skip']:3d}{flag}")
