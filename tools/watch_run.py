#!/usr/bin/env python
"""流水线实时看板(只读)。开在另一个终端 / tmux 窗口里, 不碰正在跑的流水线。

用法:
  ~/miniforge3/envs/sam3d/bin/python tools/watch_run.py --scene pedestrian --run r2sim
  tmux new -s watch '~/miniforge3/envs/sam3d/bin/python ~/sam3d/tools/watch_run.py --scene pedestrian --run r2sim'
  加 --indoor 看室内链; Ctrl-C 退出。

看的都是盘上的东西, 所以随时开、随时关, 流水线不用停:
  logs/.drive_state.json   驱动停在第几段 / 是不是停在闸口
  logs/drive_*.log         驱动的输出(哪一步开始了、哪一步多少秒、闸口)
  logs/timing.tsv          每一步的开始/结束/耗时/等 GPU(outdoor_pipeline._timing 写的)
  logs/<步骤>.log          正在跑的那一步自己的日志: 第几个物体、采样到哪、贴图烘焙
  nvidia-smi               显卡占用, 以及是不是有别人在用

★ 为什么不追驱动进程的 stdout: 被管道接走以后会被缓冲, 看到的时候活早干完了。
  每一步自己写的日志是直接落盘的, 以它为准(watch_run.sh 原来就是这个思路)。
"""
import argparse
import datetime as dt
import select
import termios
import tty
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

SAM3D = os.path.expanduser("~/sam3d")
DRIVE = f"{SAM3D}/.claude/skills/run-recon-pipeline/drive.py"

SEP = re.compile(rb"^== (\S+)\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", re.M)
# 生成(run.py)开始做一个物体时打的那行: ===== [3/25] .../frame.png -> outputs/<名字>/ =====
# ★ 驱动日志里也有 "[3/25] 名字" —— 那是开跑前的待办清单, 一口气列完, 不是进度, 别拿它算。
ITEM = re.compile(r"===== \[(\d+)/(\d+)\] \S+ -> (?:\S*/)?outputs/([^/\s]+)/?")
ITEM_ANY = re.compile(r"\[(\d+)/(\d+)\]\s*(\S+)?")
QUEUED = re.compile(r"\[批量\] (\d+) 个一起生成")
# 拆开烘焙时(run.py --bake-workers): 采样进程每做完一个打 [采样完], 烘焙进程开工打 [烘焙 wN] 名字 开始,
# 做完由采样进程统一打 完成：/失败：。不拆时也会打 完成：, 所以"做完几个"两种模式一个算法。
# ★ 不锚行首: 几个进程往同一个日志里写, 进度条那种不换行的输出后面会直接接上"完成："这一行
SAMPLED = re.compile(r"\[采样完\]")
BAKE_START = re.compile(r"\[烘焙 w\S+\] (\S+) 开始")
DONE = re.compile(r"(?:完成：outputs/([^/\s]+)/|失败：(\S+) 烘焙)")
# 跟踪(track_all.py): 每跑完一个词打一行 "[词] N 个实例, 最长活 M 帧, 1级, 54s (峰值 17.6G)";
# 开头打 "[多词] 25 个词, ..."。显存不够它会自己退让并打一行 OOM —— 那是退路不是故障, 但要看得见。
TRACK_DONE = re.compile(r"^\[([^\]]+)\] (\d+) 个实例, 最长活 (\d+) 帧[^\n]*?, (\d+)s", re.M)
TRACK_SKIP = re.compile(r"^\[([^\]]+)\] 已有, 跳过", re.M)
TRACK_TOTAL = re.compile(r"\[多词\] (\d+) 个词")
TRACK_OOM = re.compile(r"OOM")
PCT = re.compile(r"(\d+)% done")
STAGE = [  # 日志里认得出的生成阶段, 按出现顺序匹配最后一个
    (re.compile(r"Sampling sparse structure"), "sampling structure"),
    (re.compile(r"Sampling sparse latent"), "sampling latent"),
    (re.compile(r"Decoding sparse latent"), "decoding"),
    (re.compile(r"Postprocessing mesh"), "postprocessing mesh"),
    (re.compile(r"Baking texture"), "baking texture"),
    (re.compile(r"Rendering: \d+it"), "rendering views"),
]


def segments(indoor):
    """段表从 drive.py 读, 不抄一份(抄了就会漂)。"""
    try:
        spec = importlib.util.spec_from_file_location("drive_mod", DRIVE)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.SEGMENTS["indoor" if indoor else "outdoor"]
    except Exception as e:  # 读不到就不画段, 其余照常
        return [("?", [], str(e))]


def fmt_s(s):
    if s is None:
        return "-"
    s = int(round(s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def tail_bytes(p, n=4_000_000):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read()
    except OSError:
        return b""


def read_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def timing_rows(LOG):
    out = []
    try:
        lines = open(f"{LOG}/timing.tsv").read().splitlines()[1:]
    except OSError:
        return out
    for ln in lines:
        f = ln.split("\t")
        if len(f) >= 7:
            out.append(dict(step=f[0], env=f[1], t0=f[2], t1=f[3], dur=float(f[4]), wait=float(f[5]), ok=f[6]))
    return out


def drive_log(LOG):
    ls = sorted(glob.glob(f"{LOG}/drive_*.log"), key=os.path.getmtime)
    return ls[-1] if ls else None


def parse_drive(p):
    """驱动日志: 起止时间、跑过的步骤(含秒数)、正在跑的步骤、最后一道闸。"""
    r = dict(start=None, end=None, steps=[], running=None, running_all=[], gate=None, fail=None)
    if not p:
        return r
    txt = tail_bytes(p).decode("utf-8", "ignore")
    m = re.search(r"\[开始\] (\S+ \S+)", txt)
    r["start"] = m.group(1) if m else None
    m = re.search(r"\[结束\] (\S+ \S+) 退出码 (\d+)", txt)
    if m:
        r["end"] = (m.group(1), int(m.group(2)))
    cur, started = None, []
    for ln in txt.splitlines():
        p = re.match(r"^\[并行\] 开始 (\S+)", ln)
        if p:                                   # 并行: 步骤启动时只有这一行, 它的输出跑完才整块写进来
            started.append(p.group(1))
            continue
        f = re.match(r"^\[失败\] 步骤 (\S+) 退出码", ln)
        if f:                                   # 失败的步骤没有结尾那行秒数: 在这里收尾, 不然它一直算"在跑"
            if f.group(1) in started:
                started.remove(f.group(1))
            if cur == f.group(1):
                cur = None
        s = re.match(r"^== (\S+)\s+\[", ln)
        if s:
            cur = s.group(1)
            continue
        d = re.match(r"^\s+([\d.]+)s$", ln)
        if d and cur:
            r["steps"].append((cur, float(d.group(1))))
            if cur in started:
                started.remove(cur)
            cur = None
            continue
        g = re.search(r"闸 (\d+)/(\d+) — (\S+)", ln)
        if g:
            r["gate"] = (int(g.group(1)), int(g.group(2)), g.group(3))
        if "[失败]" in ln or "Traceback" in ln:
            r["fail"] = ln.strip()[:160]
    r["running_all"] = started + ([cur] if cur and cur not in started else [])
    r["running"] = r["running_all"][0] if r["running_all"] else None
    return r


def step_log(LOG, step):
    """正在跑的那一步写的是哪个日志: 找分隔线 '== <步骤>  <时间>' 最新的那个文件。"""
    best = None
    for f in glob.glob(f"{LOG}/*.log"):
        if os.path.basename(f).startswith("drive_"):
            continue
        b = tail_bytes(f)
        hits = [m for m in SEP.finditer(b) if m.group(1).decode() == step]
        if hits:
            ts = hits[-1].group(2).decode()
            if best is None or ts > best[1]:
                best = (f, ts, b[hits[-1].end():])
    return best


def step_detail(body, drive_txt_tail):
    """这一步现在在干嘛: 第几个物体 / 哪个阶段 / 百分比 / 最后一行有信息的日志。"""
    txt = body.decode("utf-8", "ignore")
    item = None
    for m in ITEM.finditer(txt):
        item = m
    queued = None
    if item is None:
        q = QUEUED.findall(drive_txt_tail)
        queued = int(q[-1]) if q else None
        if queued is None:      # 不是批量生成的步骤: 退回任何 [i/N] 行(这类步骤是边做边打的)
            for m in ITEM_ANY.finditer(txt):
                item = m
    stage = None
    pos = -1
    for rx, name in STAGE:
        for m in rx.finditer(txt):
            if m.start() > pos:
                pos, stage = m.start(), name
    pct = None
    last_item_pos = txt.rfind("=====")
    seg = txt[last_item_pos:] if last_item_pos >= 0 else txt
    ps = PCT.findall(seg[-4000:])
    if ps and stage in ("postprocessing mesh",):
        pct = int(ps[-1])
    lines = [ln for ln in re.split(r"[\r\n]", txt)
             if ln.strip() and not re.search(r"% done|UserWarning|warnings\.warn|it/s\]|^=+$|ipyevents|^Traceback|kaolin", ln)]
    last = lines[-1].strip() if lines else ""
    last = re.sub(r"^\d{4}-\d\d-\d\d [\d:.]+ \| \w+\s+\| [\w.]+:\w+:\d+ - ", "", last)
    tr = None
    tot = TRACK_TOTAL.findall(txt)
    if tot:
        done = TRACK_DONE.findall(txt)
        tr = dict(total=int(tot[-1]), done=len(done) + len(TRACK_SKIP.findall(txt)),
                  last=(f"{done[-1][0]} · {done[-1][1]} instances · {done[-1][3]}s" if done else ""),
                  oom=len(TRACK_OOM.findall(txt)))
    finished = [a or b for a, b in DONE.findall(txt)]
    split = dict(sampled=len(SAMPLED.findall(txt)), done=len(finished),
                 baking=[n for n in BAKE_START.findall(txt) if n not in set(finished)])
    if split["sampled"] == 0:
        split["mode"] = "inline"
    else:
        split["mode"] = "split"
        stage = None           # 几个物体同时在不同阶段, 单一"阶段"没有意义
    return item, stage if item else None, pct, last[:150], queued, split, tr


def gpu():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
        u, used, tot = [int(x) for x in q.stdout.strip().split("\n")[0].split(",")]
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
        apps = []
        for ln in a.stdout.strip().splitlines():
            pid, name, mem = [x.strip() for x in ln.split(",")]
            env = re.search(r"envs/([^/]+)/", name)
            apps.append((pid, env.group(1) if env else os.path.basename(name), int(mem)))
        return u, used, tot, apps
    except Exception:
        return None


def driver_alive(scene, run):
    """驱动或流水线执行器在跑这个轮次吗。单独跑某一步(outdoor_pipeline.py --only ...)也算在跑 ——
    只认 drive.py 的话, 状态文件还写着"停在闸口", 看板会把正在跑的说成在等人。"""
    pat = rf"(drive|outdoor_pipeline|indoor_pipeline)\.py .*--scene {re.escape(scene)} .*--run {re.escape(run)}( |$)"
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    return bool(r.stdout.strip())


VIEW = dict(seg=None)          # 人手选的段(0 基); None = 跟着正在跑的那一段


class Keys:
    """把终端切成不用回车的模式, 一次读一个键。不是 tty(比如输出被重定向)就整个不启用 ——
    看板本来就该能被 `| head` 接走看一眼。"""
    def __init__(self):
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.old = None

    def __enter__(self):
        if self.fd is not None:
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *a):
        if self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def get(self):
        """攒着的键一次读完(按住方向键会连发), 方向键是 ESC [ C 这种三字节。"""
        if self.fd is None:
            return []
        out = []
        while select.select([self.fd], [], [], 0)[0]:
            ch = os.read(self.fd, 1).decode("utf-8", "ignore")
            if ch == "\x1b":
                rest = os.read(self.fd, 2).decode("utf-8", "ignore") if select.select([self.fd], [], [], 0.01)[0] else ""
                out.append({"[C": "right", "[D": "left"}.get(rest, "esc"))
            else:
                out.append(ch)
        return out


def on_key(k, nseg, cur):
    """左右翻段, 数字直接跳, a 回到自动跟随。返回新的 VIEW['seg']。

    cur = 自动跟随算出来的那一段(正在跑的那段)。★ 左右要从**眼下看的这一段**走一步,
    不是从正在跑的那一段走 —— 原来写的是 `cur if VIEW["seg"] is not None else cur`,
    三元两边同一个值, 等于永远按 cur 算。后果: 在第 4 段按一下 → 去第 5 段, 再按还是
    第 5 段(cur+1 不动), 按 ← 掉到第 3 段而不是回第 4 段。人会以为看板卡住了。
    """
    base = cur if VIEW["seg"] is None else VIEW["seg"]
    if k == "left":
        return max(0, base - 1)
    if k == "right":
        return min(nseg - 1, base + 1)
    if k in ("a", "A"):
        return None
    if k.isdigit() and 1 <= int(k) <= nseg:
        return int(k) - 1
    return VIEW["seg"]


def render(a, R1, SEGS):
    LOG = f"{R1}/logs"
    now = dt.datetime.now()
    st = read_json(f"{LOG}/.drive_state.json") or {}
    dlog = drive_log(LOG)
    D = parse_drive(dlog)
    alive = driver_alive(a.scene, a.run)
    T = timing_rows(LOG)
    run_start = D["start"]
    T_run = [t for t in T if run_start and t["t0"] >= run_start]
    # ★ 每按一次"继续"就是一次新运行, 前面几段是上几次跑的。只看本次的话, 翻回去看前面的段
    #   就只剩一个"✓ earlier"、没有秒数 —— 而耗时明明都在 timing.tsv 里。这里按步骤记住
    #   **最近一次**的那条, 翻到哪一段都能看到它上次跑了多久。
    T_last = {}
    for t in T:
        T_last[t["step"]] = t

    # ---- 头 ----
    if alive and D["running"]:
        state = Text("● RUNNING", style="bold green")
    elif D["end"] and D["end"][1] != 0:
        state = Text("✖ FAILED", style="bold red")
    elif st.get("停在闸口") or (D["gate"] and not alive):
        state = Text("⏸ AT GATE — waiting for you", style="bold yellow")
    elif alive:
        state = Text("● RUNNING", style="bold green")
    else:
        state = Text("○ idle", style="dim")
    el = (now - dt.datetime.strptime(run_start, "%Y-%m-%d %H:%M:%S")).total_seconds() if run_start else None
    if D["end"]:
        el = (dt.datetime.strptime(D["end"][0], "%Y-%m-%d %H:%M:%S")
              - dt.datetime.strptime(run_start, "%Y-%m-%d %H:%M:%S")).total_seconds()
    head = Table.grid(expand=True)
    head.add_column(); head.add_column(justify="right")
    head.add_row(Text.assemble((f"{a.scene}/{a.run}", "bold cyan"), "   ", state),
                 Text(f"this run {fmt_s(el)}   ·   {now:%H:%M:%S}", style="dim"))

    # ---- 段 ----
    # 状态文件: 段 是 0 基。停在闸口时它是**刚跑完**的那一段; 在跑时是正在跑的那一段。
    seg_idx = st.get("段", 0)
    at_gate = bool(st.get("停在闸口")) and not (alive and D["running"])
    done_steps = {s for s, _ in D["steps"]} | {t["step"] for t in T_run}
    segline = Text()
    cur_seg = None
    for i, (name, steps, _) in enumerate(SEGS):
        if D["running"] in steps:
            cur_seg = i
    # 看哪一段: 人手翻过就看人翻的, 否则跟着正在跑的那一段(段条上要标出来, 所以先算)
    auto_seg = cur_seg if cur_seg is not None else max(0, min(seg_idx, len(SEGS) - 1))
    show_seg = auto_seg if VIEW["seg"] is None else min(VIEW["seg"], len(SEGS) - 1)
    VIEW["auto"] = auto_seg
    for i, (name, steps, _) in enumerate(SEGS):
        if cur_seg is not None:
            mark = "done" if i < cur_seg else ("run" if i == cur_seg else "todo")
        elif at_gate:
            mark = "done" if i <= seg_idx else "todo"
        else:
            mark = "done" if i < seg_idx else "todo"
        sty = {"done": "green", "run": "bold black on yellow", "todo": "dim"}[mark]
        icon = {"done": "✓", "run": "▶", "todo": "·"}[mark]
        if i == show_seg and mark != "run":
            sty += " underline"          # 人手翻到的那一段(正在跑的那段本来就是黄底)
        segline.append(f" {icon} {i + 1} {name} ", style=sty)
        if i < len(SEGS) - 1:
            segline.append(" ─ ", style="dim")

    # ---- 看的那一段的步骤表 ----
    tbl = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False)
    for c, kw in [("step", {}), ("status", {}), ("started", {}), ("took", {"justify": "right"}),
                  ("gpu wait", {"justify": "right"}), ("now", {"ratio": 3})]:
        tbl.add_column(c, **kw)
    detail = None
    if SEGS and SEGS[0][0] != "?":
        tl = {t["step"]: t for t in T_run}
        for s in SEGS[show_seg][1]:
            t = tl.get(s)
            if s in D["running_all"]:
                sl = step_log(LOG, s)
                t0 = sl[1] if sl else None
                took = (now - dt.datetime.strptime(t0, "%Y-%m-%d %H:%M:%S")).total_seconds() if t0 else None
                drive_tail = tail_bytes(dlog, 20000).decode("utf-8", "ignore") if dlog else ""
                drive_tail = drive_tail[drive_tail.rfind(f"== {s} "):]
                item, stage, pct, last, queued, split, tr = step_detail(sl[2] if sl else b"", drive_tail)
                detail = (s, sl, item, stage, pct, last, queued, split, tr)
                now_txt = ""
                if tr:
                    now_txt = (f"words {tr['done']}/{tr['total']}"
                               + (f" · {tr['last']}" if tr["last"] else "")
                               + (f" · {tr['oom']} OOM retries" if tr["oom"] else ""))
                if item and split["mode"] == "split":
                    now_txt = (f"sampled {split['sampled']}/{item.group(2)} · baked {split['done']}/{item.group(2)}"
                               + (f" · baking {len(split['baking'])}" if split["baking"] else ""))
                elif item:
                    now_txt = f"[{item.group(1)}/{item.group(2)}] {os.path.basename((item.group(3) or '').rstrip('/'))}"
                elif queued:
                    now_txt = f"loading model, {queued} queued"
                if stage:
                    now_txt += f"  · {stage}" + (f" {pct}%" if pct is not None else "")
                tbl.add_row(Text(s, style="bold"), Text("▶ running", style="yellow"),
                            (t0 or "")[11:], fmt_s(took), "", now_txt)
            elif t:
                okst = Text("✓ ok", style="green") if t["ok"] == "ok" else Text("✖ " + t["ok"], style="red")
                tbl.add_row(s, okst, t["t0"][11:], fmt_s(t["dur"]),
                            fmt_s(t["wait"]) if t["wait"] >= 1 else "", "")
            elif s in done_steps:
                sec = dict(D["steps"]).get(s)
                tbl.add_row(s, Text("✓ ok", style="green"), "", fmt_s(sec), "", "")
            elif os.path.exists(f"{LOG}/.stamp_{s}") or s in T_last:
                # 这次没跑, 以前跑过: 完成戳给时间, timing.tsv 给上次的耗时
                old = T_last.get(s)
                try:
                    when = open(f"{LOG}/.stamp_{s}").read().strip()[5:16]
                except OSError:
                    when = old["t0"][5:16] if old else ""
                tbl.add_row(Text(s, style="dim"),
                            Text("✓ earlier" if not old or old["ok"] == "ok" else "✖ " + old["ok"],
                                 style="dim green" if not old or old["ok"] == "ok" else "red"),
                            Text(when, style="dim"),
                            Text(fmt_s(old["dur"]) if old else "", style="dim"),
                            Text(fmt_s(old["wait"]) if old and old["wait"] >= 1 else "", style="dim"), "")
            else:
                tbl.add_row(Text(s, style="dim"), Text("· pending", style="dim"), "", "", "", "")

    parts = [head, segline, Panel(tbl, title=f"segment {show_seg + 1}: {SEGS[show_seg][0]}",
                                  border_style="cyan", padding=(0, 1))]

    # ---- 正在跑的这一步: 物体进度条 + 日志最后一行 ----
    if detail:
        s, sl, item, stage, pct, last, queued, split, tr = detail
        g = Table.grid(padding=(0, 1))
        g.add_column(); g.add_column(ratio=1); g.add_column()
        if tr:
            g.add_row("words", ProgressBar(total=tr["total"], completed=tr["done"], width=None),
                      f"{tr['done']}/{tr['total']}")
            if tr["oom"]:
                g.add_row("", Text(f"{tr['oom']} OOM retries — it lowers the detection threshold and retries; "
                                   f"a fallback, not a failure", style="yellow"), "")
        elif item and split["mode"] == "split":
            n = int(item.group(2))
            g.add_row("sampled", ProgressBar(total=n, completed=split["sampled"], width=None),
                      f"{split['sampled']}/{n}")
            g.add_row("baked", ProgressBar(total=n, completed=split["done"], width=None), f"{split['done']}/{n} done")
            if split["baking"]:
                g.add_row("baking", Text(", ".join(split["baking"]), style="yellow",
                                         overflow="ellipsis", no_wrap=True), f"{len(split['baking'])} at once")
        elif item:
            i, n = int(item.group(1)), int(item.group(2))
            g.add_row("objects", ProgressBar(total=n, completed=split["done"] if split["done"] else i - 1,
                                             width=None), f"{split['done'] if split['done'] else i - 1}/{n} done")
        elif queued:
            g.add_row("objects", ProgressBar(total=queued, completed=0, width=None), f"0/{queued} done")
        if pct is not None:
            g.add_row("this one", ProgressBar(total=100, completed=pct, width=None), f"{pct}%")
        g.add_row("log", Text(os.path.basename(sl[0]) if sl else "?", style="dim"), "")
        g.add_row("last", Text(last, style="dim", overflow="ellipsis", no_wrap=True), "")
        parts.append(Panel(g, title=f"now: {s}", border_style="yellow", padding=(0, 1)))

    # ---- 闸 / 失败 ----
    if D["fail"]:
        parts.append(Panel(Text(D["fail"], style="red"), title="failure", border_style="red"))
    elif D["gate"] and not alive:
        n, tot, name = D["gate"]
        parts.append(Panel(Text(f"Stopped at gate {n}/{tot} — {name}. Review, then drive.py --go.",
                                style="yellow"), border_style="yellow"))

    # ---- 显卡 ----
    G = gpu()
    if G:
        u, used, tot, apps = G
        gg = Table.grid(padding=(0, 1))
        gg.add_column(); gg.add_column(ratio=1); gg.add_column()
        gg.add_row("util", ProgressBar(total=100, completed=u, width=None), f"{u}%")
        gg.add_row("mem", ProgressBar(total=tot, completed=used, width=None), f"{used / 1024:.1f}/{tot / 1024:.0f} GB")
        for pid, env, mem in apps:
            gg.add_row("", Text(f"pid {pid}  {env}", style="dim"), Text(f"{mem / 1024:.1f} GB", style="dim"))
        parts.append(Panel(gg, title="GPU", border_style="magenta", padding=(0, 1)))

    # ---- 这次跑完的步骤耗时(全段) ----
    if T_run:
        tt = Table(box=box.SIMPLE, expand=True, pad_edge=False, show_header=True)
        tt.add_column("finished this run"); tt.add_column("took", justify="right")
        tt.add_column("share", ratio=1)
        tot = sum(t["dur"] for t in T_run) or 1
        for t in sorted(T_run, key=lambda x: -x["dur"])[:8]:
            tt.add_row(t["step"], fmt_s(t["dur"]), ProgressBar(total=tot, completed=t["dur"], width=None))
        tt.add_row(Text("total", style="bold"), Text(fmt_s(sum(t["dur"] for t in T_run)), style="bold"), "")
        parts.append(tt)

    nav = ("← → or 1-6: look at another segment" if VIEW["seg"] is None
           else f"← → 1-6: another segment · a: back to the running one (segment {auto_seg + 1})")
    parts.append(Text(f"{nav}   ·   timing log: {LOG}/timing.tsv   ·   Ctrl-C to quit "
                      f"(the pipeline keeps running)", style="dim"))
    return Group(*parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", default="r1")
    ap.add_argument("--indoor", action="store_true")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--once", action="store_true", help="画一帧就退出(调试用)")
    a = ap.parse_args()
    R1 = (f"{SAM3D}/data/indoorOutput/{a.scene}/{a.run}" if a.indoor
          else f"{SAM3D}/data/outdoor/result/{a.scene}/{a.run}")
    if not os.path.isdir(R1):
        sys.exit(f"没有这个轮次: {R1}")
    SEGS = segments(a.indoor)
    if a.once:
        from rich.console import Console
        Console(width=110).print(render(a, R1, SEGS))
        return
    try:
        with Keys() as keys, Live(render(a, R1, SEGS), refresh_per_second=4, screen=True) as live:
            while True:
                t_end = time.time() + a.interval
                while time.time() < t_end:          # 等的这一秒里也收键, 按下去就有反应
                    ks = keys.get()
                    if ks:
                        for k in ks:
                            VIEW["seg"] = on_key(k, len(SEGS), VIEW.get("auto", 0))
                        live.update(render(a, R1, SEGS))
                    time.sleep(0.05)
                live.update(render(a, R1, SEGS))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
