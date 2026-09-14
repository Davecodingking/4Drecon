#!/usr/bin/env python3
"""室外流水线的流程图 -> docs/pipeline_figure.svg

布局规则(改的时候照着来, 图才不会乱):
  * 主流程沿中轴 CX 一路往下, 不拐弯
  * 人的"不满意 -> 改了再看"回路放左边; 停止 / 回跑这类分支放右边
  * 两道要人填的闸(2、4)在最右边各带一个说明框

事实来源: drive.py 的 SEGMENTS / apply_decisions, tools/review_gate.py 的继承规则。
那几处改了, 这里要跟着改。

    python docs/pipeline_figure.py            # 写出 docs/pipeline_figure.svg
"""
import html, os

W = 1500
CX = 540                  # 主流程中轴
SX0, SX1 = 140, 940       # 段框左右边
LX = 330                  # 左侧人工回路的竖线
RX = 830                  # 回跑框中心
STOP_X = 760              # 停止框左边
CO_X, CO_W = 1030, 430    # 说明框
WD_X = 100                # WildDet3D 那条紫色虚线的竖段
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"

C = dict(ink="#1f2328", sub="#667085", line="#475467",
         stage_fill="#f6f7f9", stage_stroke="#cfd4dc", chip="#98a2b3",
         dec="#c4320a", dec_fill="#fff4ed",
         look="#175cd3", look_fill="#eff8ff",
         stop="#b42318", stop_fill="#fef3f2",
         wd="#6941c6")

out = []
def e(s): out.append(s)
def esc(t): return html.escape(str(t), quote=False)


# ------------------------------------------------------------------ 基本图元
def text(x, y, t, size=13, weight="normal", fill=None, anchor="middle",
         italic=False, ls=None, rotate=None):
    a = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'font-size="{size}"',
         f'fill="{fill or C["ink"]}"', f'text-anchor="{anchor}"']
    if weight != "normal": a.append(f'font-weight="{weight}"')
    if italic: a.append('font-style="italic"')
    if ls: a.append(f'letter-spacing="{ls}"')
    if rotate: a.append(f'transform="rotate({rotate} {x:.1f} {y:.1f})"')
    e(f'<text {" ".join(a)}>{esc(t)}</text>')

def rect(x, y, w, h, fill="#fff", stroke=None, sw=1.4, rx=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    e(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
      f'fill="{fill}" stroke="{stroke or C["line"]}" stroke-width="{sw}"{d}/>')

def path(pts, color=None, sw=1.6, arrow="arr", dash=None):
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    a = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{arrow})"' if arrow else ""
    e(f'<path d="{d}" fill="none" stroke="{color or C["line"]}" stroke-width="{sw}" '
      f'stroke-linejoin="round"{a}{m}/>')

def edge_label(y, t, bold=False, fill=None, left=False):
    """贴着主轴竖线的小字。left=True 放在线左边(右边被占时用)。"""
    text(CX - 10 if left else CX + 10, y, t, 12, "bold" if bold else "normal",
         fill or C["sub"], anchor="end" if left else "start")


# ------------------------------------------------------------------ 节点
def pill(y, label, fill, stroke, color, w=170, h=38):
    rect(CX - w / 2, y - h / 2, w, h, fill=fill, stroke=stroke, sw=1.6, rx=h / 2)
    text(CX, y + 5, label, 13, "bold", color, ls=0.8)
    return y - h / 2, y + h / 2

STAGE_H = 112
def stage(top, num, title, purpose, chips, wd=()):
    """一段 = 一个灰底框, 里面一排步骤小卡片。wd 里的下标画成紫色虚线(WildDet3D 对照路线)。"""
    rect(SX0, top, SX1 - SX0, STAGE_H, fill=C["stage_fill"], stroke=C["stage_stroke"], rx=10)
    e(f'<circle cx="{SX0 + 28}" cy="{top + 26}" r="12" fill="{C["ink"]}"/>')
    text(SX0 + 28, top + 30.5, num, 13, "bold", "#fff")
    text(SX0 + 50, top + 31, title, 13.5, "bold", ls=1.4, anchor="start")
    text(SX1 - 20, top + 31, purpose, 12, fill=C["sub"], anchor="end", italic=True)
    n, pad, gap, ch = len(chips), 20, 28, 50
    cw = (SX1 - SX0 - 2 * pad - (n - 1) * gap) / n
    cy = top + 48
    centers = []
    for i, (name, model) in enumerate(chips):
        x = SX0 + pad + i * (cw + gap)
        is_wd = i in wd
        rect(x, cy, cw, ch, fill="#fff", stroke=C["wd"] if is_wd else C["chip"],
             sw=1.5 if is_wd else 1.2, rx=6, dash="5 3" if is_wd else None)
        text(x + cw / 2, cy + 21, name, 13, "bold")
        text(x + cw / 2, cy + 38, model, 11, fill=C["wd"] if is_wd else C["sub"], italic=True)
        if i < n - 1:
            path([(x + cw + 4, cy + ch / 2), (x + cw + gap - 3, cy + ch / 2)], sw=1.3)
        centers.append(x + cw / 2)
    return top + STAGE_H, centers

def diamond(y, label, w=280, h=62):
    pts = [(CX - w / 2, y), (CX, y - h / 2), (CX + w / 2, y), (CX, y + h / 2)]
    e('<polygon points="' + " ".join(f"{x:.1f},{yy:.1f}" for x, yy in pts) +
      f'" fill="#fff" stroke="{C["line"]}" stroke-width="1.4"/>')
    text(CX, y + 4.5, label, 13)
    return dict(top=y - h / 2, bot=y + h / 2, left=CX - w / 2, right=CX + w / 2, y=y)

def stop_box(y, title, notes=(), w=220):
    h = 38 + 16 * len(notes)
    rect(STOP_X, y - h / 2, w, h, fill=C["stop_fill"], stroke=C["stop"], sw=1.6, rx=6)
    ty = y - h / 2 + 24
    text(STOP_X + w / 2, ty, title, 13, "bold", C["stop"])
    for i, n in enumerate(notes):
        text(STOP_X + w / 2, ty + 17 + 16 * i, n, 11.5, fill=C["stop"], italic=True)

def branch_no(d):
    path([(d["right"], d["y"]), (STOP_X, d["y"])], color=C["stop"], arrow="arr_stop")
    text((d["right"] + STOP_X) / 2, d["y"] - 7, "no", 12, fill=C["stop"])

def gate_look(y, num, note):
    w, h = 220, 40
    rect(CX - w / 2, y - h / 2, w, h, fill=C["look_fill"], stroke=C["look"], sw=1.6, rx=20)
    text(CX, y + 5, f"GATE {num} · LOOK", 13, "bold", C["look"], ls=0.8)
    text(CX + w / 2 + 16, y + 4.5, note, 12, fill=C["look"], anchor="start", italic=True)
    return y - h / 2, y + h / 2


# ------------------------------------------------------------------ 要人填的闸: 一整块
def callout(gy, gate_right, spec):
    title, sub, sections = spec
    h = 58 + sum(26 + 18 * len(items) for _, items in sections) + 8
    top = gy - 28
    rect(CO_X, top, CO_W, h, fill="#fff", stroke=C["dec"], sw=1.3, rx=8)
    e(f'<rect x="{CO_X}" y="{top + 8}" width="4" height="{h - 16}" rx="2" fill="{C["dec"]}"/>')
    text(CO_X + 22, top + 27, title, 14, "bold", C["dec"], anchor="start", ls=0.8)
    text(CO_X + 22, top + 46, sub, 12, fill=C["sub"], anchor="start", italic=True)
    y = top + 58
    for head, items in sections:
        y += 22
        text(CO_X + 22, y, head.upper(), 11, "bold", C["sub"], anchor="start", ls=1.1)
        for it in items:
            y += 18
            text(CO_X + 30, y, it, 12.5, anchor="start")
        y += 4
    path([(gate_right, gy), (CO_X, gy)], color=C["dec"], sw=1.2, arrow=None, dash="3 3")

def gate_decide_block(top, num, subtitle, stop_notes, spec):
    """闸 -> 满意吗(不满意回左边) -> 决定合法吗(不合法停) -> 类级决定变了吗(变了回跑) -> 下一段。
    返回下一段的顶边 y。"""
    gy = top + 48
    gw, gh = 280, 56
    path([(CX, top), (CX, gy - gh / 2)])
    rect(CX - gw / 2, gy - gh / 2, gw, gh, fill=C["dec_fill"], stroke=C["dec"], sw=2.2, rx=8)
    text(CX, gy - 2, f"GATE {num} · DECIDE", 14.5, "bold", C["dec"], ls=0.9)
    text(CX, gy + 16, subtitle, 12, fill=C["dec"], italic=True)

    s = diamond(gy + 95, "satisfied with the sheets?", w=300)
    path([(CX, gy + gh / 2), (CX, s["top"])])
    # 不满意: 左边绕回闸。改决定 + 不加 --go 重跑, 只重出图, 不耗 GPU
    path([(s["left"], s["y"]), (LX, s["y"]), (LX, gy), (CX - gw / 2, gy)],
         color=C["dec"], arrow="arr_dec")
    text(s["left"] - 8, s["y"] - 7, "no", 12, fill=C["dec"], anchor="end")
    for i, t in enumerate(("edit decisions,", "run again without --go", "(sheets redrawn, no GPU)")):
        text(LX - 12, (gy + s["y"]) / 2 - 12 + 16 * i, t, 12, fill=C["dec"], anchor="end", italic=(i == 2))

    v = diamond(s["y"] + 100, "decisions valid?", w=250)
    path([(CX, s["bot"]), (CX, v["top"])])
    edge_label((s["bot"] + v["top"]) / 2 + 4, "yes  ·  --go", bold=True, fill=C["dec"])
    stop_box(v["y"], "STOP · fix the file", stop_notes)
    branch_no(v)

    k = diamond(v["y"] + 100, "class-level choice changed?", w=330, h=64)
    path([(CX, v["bot"]), (CX, k["top"])])
    edge_label((v["bot"] + k["top"]) / 2 + 4, "yes")

    # 类级决定变了: 回跑受影响的步骤(只跑已经跑过的), 然后汇回主轴
    rw, rh = 320, 84
    ry = k["y"] + 90
    path([(k["right"], k["y"]), (RX, k["y"]), (RX, ry - rh / 2)])
    text((k["right"] + RX) / 2, k["y"] - 7, "yes", 12, fill=C["sub"])
    rect(RX - rw / 2, ry - rh / 2, rw, rh, fill="#fff", stroke=C["dec"], sw=1.4, rx=6, dash="6 4")
    text(RX, ry - 20, "RERUN AFFECTED STEPS", 12, "bold", C["dec"], ls=0.8)
    text(RX, ry - 1, "relabel › pick › verify › override ›", 12)
    text(RX, ry + 15, "gen › ratio › boxes › solo", 12)
    text(RX, ry + 32, "already-run steps only; cached cutouts skip", 11, fill=C["sub"], italic=True)
    my = ry + rh / 2 + 34
    path([(RX, ry + rh / 2), (RX, my), (CX + 3, my)])
    e(f'<circle cx="{CX}" cy="{my:.1f}" r="3.2" fill="{C["line"]}"/>')
    edge_label(k["bot"] + 26, "no  (only drop changed)", left=True)
    nxt = my + 44
    path([(CX, k["bot"]), (CX, nxt)])

    callout(gy, CX + gw / 2, spec)
    return nxt


# ------------------------------------------------------------------ 内容
S1 = ("1", "FOUNDATION", "camera, depth, and which way is up",
      [("Frames", "sampled"), ("Vocabulary", "Qwen3-VL"),
       ("Depth + pose", "Depth Anything 3"), ("Sky direction", "ground-plane fit")])
S2 = ("2", "PRE-GENERATION", "decide what gets generated",
      [("Track", "SAM3, per word"), ("Dedup", "cross-word, 2D"), ("Relabel", "move to class"),
       ("Pick", "best cutout"), ("Verify", "Qwen3-VL"), ("Override", "your pick")])
S3 = ("3", "SCALE", "calibrate to real meters",
      [("Mesh gen", "SAM 3D Objects"), ("Mesh ratio", "L : W : H"), ("3D boxes", "back-projection"),
       ("Real heights", "Qwen3-VL"), ("Scale k", "weighted median")])
S4 = ("4", "RECALL", "recover what SAM3 missed",
      [("Per-instance", "SAM 3D Objects"), ("WD boxes", "WildDet3D"), ("Detect", "WildDet3D"),
       ("Match", "vs SAM3, 2D"), ("Re-mask", "box → SAM3")])
S5 = ("5", "ASSEMBLY", "put the scene back together",
      [("Recall gen", "SAM 3D Objects"), ("Merge boxes", "main + recall"), ("Place", "scale + orient"),
       ("Physics", "settle, separate"), ("Render back", "original cameras")])
S6 = ("6", "FINAL", "buildings and a comparison scene",
      [("Buildings", "building+scene mode only"), ("WildDet3D scene", "place · physics · render")])

CALLOUT2 = ("GATE 2 · REVIEW CANDIDATES", "before the most expensive step: generation", [
    ("You look at", ["• Review sheet: one tile per candidate",
                     "• Reuse table: one row per mesh, and who shares it",
                     "• or make the same choices in the web decision desk"]),
    ("You decide, per candidate", ["• keep  /  drop",
                                   "• drop-word: void the whole word",
                                   "• solo / shared: own mesh, or one for the class",
                                   "• reuse group: split a word into look-alike groups",
                                   "• change class: the label moves, the mask stays",
                                   "• change image: use another frame of the object"]),
    ("Default", ["• every instance gets its own mesh (solo)",
                 "• shared only where you say so"]),
])
CALLOUT4 = ("GATE 4 · REVIEW RECALL", "objects SAM3 missed, found again by WildDet3D", [
    ("You look at", ["• Recall check: each recovered object on the photo",
                     "• Verdict table: every candidate, kept or why not",
                     "• After-recall sheet: recalled tiles last, yellow",
                     "• After-recall reuse table"]),
    ("You decide", ["• same fields as Gate 2, except change image"]),
    ("Which class a recalled object joins", ["1  word already decided → inherits that decision",
                                             "2  word never seen before → solo, flagged yellow",
                                             "3  word split into groups → stop until you pick one"]),
    ("Rollback", ["• the pre-recall sheet is kept; delete the",
                  "  after-recall decisions to return to it"]),
])


# ------------------------------------------------------------------ 从上往下排
text(70, 52, "Outdoor Scene Reconstruction — Workflow and Human Gates", 22, "bold", anchor="start")
text(70, 78, "Monocular video → editable, metric-scale 3D scene. Automatic steps run between gates; "
             "the driver stops at every gate and waits for you.", 13, fill=C["sub"], anchor="start")

_, b = pill(128, "INPUT VIDEO", "#fff", C["line"], C["ink"])
top = b + 34
path([(CX, b), (CX, top)])
bot, _ = stage(top, *S1[:4])

t, b = gate_look(bot + 50, 1, "ground fit, camera path")
path([(CX, bot), (CX, t)])
d = diamond(b + 62, "camera moved enough?", w=270)
path([(CX, b), (CX, d["top"])])
stop_box(d["y"], "STOP · reshoot", ("too little parallax for", "depth and scale"))
branch_no(d)
top = d["bot"] + 50
path([(CX, d["bot"]), (CX, top)])
edge_label(d["bot"] + 29, "yes")
bot, _ = stage(top, *S2[:4])

top = gate_decide_block(bot, 2, "review candidates",
                        ("e.g. change class and change", "image on the same row"), CALLOUT2)
bot, _ = stage(top, *S3[:4])

d = diamond(bot + 62, "scale agreement ok?", w=270)
path([(CX, bot), (CX, d["top"])])
stop_box(d["y"], "STOP · scale not applied", ("fix a word's reference height,", "or accept unscaled, rerun"), w=230)
branch_no(d)
t, b = gate_look(d["bot"] + 62, 3, "k, agreement, outlier words")
path([(CX, d["bot"]), (CX, t)])
edge_label((d["bot"] + t) / 2 + 4, "yes  ·  boxes re-measured")
top4 = b + 46
path([(CX, b), (CX, top4)])
bot, c4 = stage(top4, *S4[:4], wd=(1,))

top = gate_decide_block(bot, 4, "review recall",
                        ("e.g. a recalled object of a split", "word has no group yet"), CALLOUT4)
bot, _ = stage(top, *S5[:4])

t, b = gate_look(bot + 50, 5, "render vs photo, skipped objects")
path([(CX, bot), (CX, t)])
top6 = b + 46
path([(CX, b), (CX, top6)])
bot, c6 = stage(top6, *S6[:4], wd=(1,))

t, b = gate_look(bot + 50, 6, "main vs WildDet3D render-back")
path([(CX, bot), (CX, t)])
dt, db = pill(b + 58, "DONE", C["ink"], C["ink"], "#fff", w=130)
path([(CX, b), (CX, dt)])

# WildDet3D 对照路线: 第 4 段的 WD 盒子 -> 第 6 段的 WD 版场景。从卡片底下走, 不穿别的卡片
y4, y6 = top4 + 105, top6 + 105
path([(c4[1], top4 + 98), (c4[1], y4), (WD_X, y4), (WD_X, y6), (c6[1], y6), (c6[1], top6 + 99)],
     color=C["wd"], sw=1.5, arrow="arr_wd", dash="6 4")
text(WD_X - 9, (y4 + y6) / 2, "WildDet3D boxes → parallel scene version", 12,
     fill=C["wd"], italic=True, rotate=-90)

# ------------------------------------------------------------------ 图例 + 通用规则
y = db + 70
e(f'<line x1="70" y1="{y - 34}" x2="{W - 70}" y2="{y - 34}" stroke="{C["stage_stroke"]}" stroke-width="1"/>')
x = 70
def leg(draw, label):
    global x
    draw(x)
    text(x + 44, y + 4.5, label, 12, anchor="start")
    x += 44 + len(label) * 6.9 + 34
leg(lambda x: rect(x, y - 12, 32, 24, stroke=C["chip"], sw=1.2, rx=5), "automatic step")
leg(lambda x: rect(x, y - 12, 32, 24, fill=C["dec_fill"], stroke=C["dec"], sw=2, rx=5), "gate · you decide")
leg(lambda x: rect(x, y - 11, 32, 22, fill=C["look_fill"], stroke=C["look"], sw=1.6, rx=11), "gate · you look")
leg(lambda x: e(f'<polygon points="{x},{y} {x + 16},{y - 11} {x + 32},{y} {x + 16},{y + 11}" '
                f'fill="#fff" stroke="{C["line"]}" stroke-width="1.3"/>'), "check")
leg(lambda x: rect(x, y - 12, 32, 24, fill=C["stop_fill"], stroke=C["stop"], sw=1.6, rx=5), "stop")
leg(lambda x: rect(x, y - 12, 32, 24, stroke=C["dec"], sw=1.4, rx=5, dash="5 3"), "rerun")
leg(lambda x: path([(x, y), (x + 30, y)], color=C["dec"], arrow="arr_dec"), "human loop, no GPU")
leg(lambda x: path([(x, y), (x + 30, y)], color=C["wd"], sw=1.5, arrow="arr_wd", dash="6 4"), "WildDet3D route")

text(70, y + 44, "Any step that fails stops the run in place; the next run restarts that segment "
                 "and skips steps that already finished.", 12.5, fill=C["sub"], anchor="start")
text(70, y + 64, "Driver: drive.py --scene <scene> --run <run>, add --go to pass a gate. "
                 "Sources: drive.py (SEGMENTS, apply_decisions), tools/review_gate.py.",
     12.5, fill=C["sub"], anchor="start")
H = int(y + 96)

# ------------------------------------------------------------------ 写出
defs = ["<defs>"]
for mid, col in (("arr", C["line"]), ("arr_dec", C["dec"]), ("arr_stop", C["stop"]), ("arr_wd", C["wd"])):
    defs.append(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" '
                f'markerHeight="6.5" orient="auto"><path d="M0,0.5 L10,5 L0,9.5 z" fill="{col}"/></marker>')
defs.append("</defs>")
svg = "\n".join([
    '<?xml version="1.0" encoding="UTF-8"?>',
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    f'font-family="{FONT}">',
    *defs,
    f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    *out,
    "</svg>", ""])
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_figure.svg")
open(dst, "w", encoding="utf-8").write(svg)
print(f"wrote {dst}  ({W}x{H})")
