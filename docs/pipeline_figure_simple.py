#!/usr/bin/env python3
"""简化版流程图 -> docs/pipeline_figure_simple.svg

和 pipeline_figure.svg 讲同一件事, 压成一行, 适合放论文 / slide:
  * 六段各一个框, 框里只留两行关键步骤 + 一行模型(按角色上色, 和汇报 slide 同一套)
  * 只看的闸(1/3/5/6)缩成小菱形, 下面一行字写看什么
  * 要人填的闸(2/4)完整保留: 改了再看的小回环、放行时回跑到第 2 段的虚线弧、
    下方说明框(看什么 / 能决定什么 / 默认 / 召回归哪类)

布局规则: 主流程一行从左到右; 回跑弧在上方, 越远的弧越高, 互不交叉; 说明框在下方。
事实来源同 pipeline_figure.py(drive.py 的 SEGMENTS / apply_decisions, review_gate.py)。

    python docs/pipeline_figure_simple.py
"""
import html, os

W = 2000
MX = 40                    # 左右留白
Y = 236                    # 主流程这一行的中线(上方要留给回跑弧和它的说明)
SEG_W, SEG_H = 172, 124
GATE_W, GATE_H = 150, 86
LOOK_D = 34                # 只看的闸: 菱形边长
PILL_IN, PILL_OUT = 84, 100
CO_W = 440                 # 说明框
CO_TOP = Y + 122
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"

C = dict(ink="#1f2328", sub="#667085", line="#475467",
         stage_fill="#f6f7f9", stage_stroke="#cfd4dc",
         dec="#c4320a", dec_fill="#fff4ed", look="#175cd3", look_fill="#eff8ff",
         stop="#b42318",
         meas="#175cd3", gen="#067647", sense="#b54708", wd="#6941c6")

out = []
def e(s): out.append(s)
def esc(t): return html.escape(str(t), quote=False)

def text(x, y, t, size=12, weight="normal", fill=None, anchor="middle", italic=False, ls=None):
    a = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'font-size="{size}"', f'fill="{fill or C["ink"]}"',
         f'text-anchor="{anchor}"']
    if weight != "normal": a.append(f'font-weight="{weight}"')
    if italic: a.append('font-style="italic"')
    if ls: a.append(f'letter-spacing="{ls}"')
    e(f'<text {" ".join(a)}>{esc(t)}</text>')

def models_line(cx, y, models):
    """段框底下那行模型名, 各按角色上色。
    ★ 不用一个 <text> 套几个 <tspan>: 居中对齐时 cairosvg 会把各段各自对齐, 字叠在一起。
      两个名字就一个右对齐在点左边、一个左对齐在点右边, 哪个渲染器都不会错。"""
    if len(models) == 1:
        m, role = models[0]
        text(cx, y, m, 11.5, fill=C[role], italic=True)
        return
    (m1, r1), (m2, r2) = models
    text(cx - 7, y, m1, 11.5, fill=C[r1], anchor="end", italic=True)
    text(cx, y, "·", 11.5, fill=C["sub"])
    text(cx + 7, y, m2, 11.5, fill=C[r2], anchor="start", italic=True)

def rect(x, y, w, h, fill="#fff", stroke=None, sw=1.4, rx=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    e(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
      f'stroke="{stroke or C["line"]}" stroke-width="{sw}"{d}/>')

def path(d, color=None, sw=1.6, arrow="arr", dash=None):
    if isinstance(d, list):
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in d)
    a = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{arrow})"' if arrow else ""
    e(f'<path d="{d}" fill="none" stroke="{color or C["line"]}" stroke-width="{sw}" '
      f'stroke-linejoin="round"{a}{m}/>')

def person(cx, cy, col, k=1.0):
    """人形标记: 头 + 肩。"""
    e(f'<circle cx="{cx:.1f}" cy="{cy - 9 * k:.1f}" r="{6.5 * k:.1f}" fill="{col}"/>')
    e(f'<path d="M{cx - 12 * k:.1f},{cy + 12 * k:.1f} Q{cx:.1f},{cy - 8 * k:.1f} {cx + 12 * k:.1f},'
      f'{cy + 12 * k:.1f} Z" fill="{col}"/>')


# ------------------------------------------------------------------ 内容
SEGS = [("FOUNDATION", ["frames · vocabulary", "depth + pose · sky"], [("Qwen3-VL", "sense"), ("DA3", "meas")]),
        ("PRE-GENERATION", ["track · dedup · relabel", "pick · verify · override"], [("SAM3", "meas"), ("Qwen3-VL", "sense")]),
        ("SCALE", ["mesh gen · 3D boxes", "real heights → k"], [("SAM 3D", "gen"), ("Qwen3-VL", "sense")]),
        ("RECALL", ["per-instance gen", "detect · match · re-mask"], [("WildDet3D", "wd"), ("SAM3", "meas")]),
        ("ASSEMBLY", ["recall gen · merge", "place · physics · render"], [("SAM 3D", "gen")]),
        ("FINAL", ["buildings", "WildDet3D comparison"], [("WildDet3D", "wd")])]
LOOKS = {1: ["ground plane,", "camera path"], 3: ["scale k,", "agreement"],
         5: ["render back", "vs photo"], 6: ["main vs", "WildDet3D"]}
GATES = {2: ["review", "candidates"], 4: ["review", "recall"]}
CALLOUT = {
    2: ("GATE 2 · REVIEW CANDIDATES", [
        ("Look", ["review sheet · reuse table · web decision desk"]),
        ("Decide", ["keep / drop / drop-word / solo / shared /",
                    "reuse group / change class / change image"]),
        ("Default", ["every instance gets its own mesh (solo)"]),
        ("Stop", ["invalid combination → fix the file, --go again"])]),
    4: ("GATE 4 · REVIEW RECALL", [
        ("Look", ["recall check · verdict table · after-recall sheet"]),
        ("Decide", ["same fields as Gate 2, except change image"]),
        ("Class", ["decided word → inherits the decision",
                   "new word → solo, flagged yellow",
                   "split word → stop until you assign a group"]),
        ("Undo", ["pre-recall sheet is kept for rollback"])]),
}

# ------------------------------------------------------------------ 一行排开
ITEMS = [("in", PILL_IN)]
for i in range(6):
    ITEMS.append(("seg", SEG_W, i))
    g = i + 1
    ITEMS.append(("gate", GATE_W, g) if g in GATES else ("look", LOOK_D, g))
ITEMS.append(("out", PILL_OUT))
gap = (W - 2 * MX - sum(it[1] for it in ITEMS)) / (len(ITEMS) - 1)
x = MX
POS = []                     # (种类, 左边 x, 宽, 编号)
for it in ITEMS:
    POS.append((it[0], x, it[1], it[2] if len(it) > 2 else None))
    x += it[1] + gap
def find(kind, n): return next(p for p in POS if p[0] == kind and p[3] == n)

for a, b in zip(POS, POS[1:]):
    path([(a[1] + a[2] + 2, Y), (b[1] - 3, Y)], sw=1.5)

for kind, x, w, n in POS:
    cx = x + w / 2
    if kind in ("in", "out"):
        h = 54
        rect(x, Y - h / 2, w, h, fill="#fff", stroke=C["line"], sw=1.5, rx=h / 2)
        for k, t in enumerate(("Input", "video") if kind == "in" else ("Editable", "3D scene")):
            text(cx, Y - 3 + 15 * k, t, 12.5, "bold")
    elif kind == "seg":
        title, steps, models = SEGS[n]
        top = Y - SEG_H / 2
        rect(x, top, w, SEG_H, fill=C["stage_fill"], stroke=C["stage_stroke"], rx=8)
        e(f'<circle cx="{x + 18:.1f}" cy="{top + 21:.1f}" r="10" fill="{C["ink"]}"/>')
        text(x + 18, top + 25.5, n + 1, 12, "bold", "#fff")
        text(x + 34, top + 25.5, title, 12, "bold", anchor="start", ls=0.6)
        for k, s in enumerate(steps):
            text(cx, top + 56 + 18 * k, s, 11.5)
        models_line(cx, top + 106, models)
    elif kind == "look":
        d = LOOK_D / 2
        e(f'<polygon points="{cx - d:.1f},{Y} {cx:.1f},{Y - d:.1f} {cx + d:.1f},{Y} {cx:.1f},{Y + d:.1f}" '
          f'fill="{C["look_fill"]}" stroke="{C["look"]}" stroke-width="1.6"/>')
        text(cx, Y + 4, f"G{n}", 10.5, "bold", C["look"])
        for k, t in enumerate(LOOKS[n]):
            text(cx, Y + SEG_H / 2 + 20 + 14 * k, t, 11, fill=C["look"], italic=True)
    else:   # 要人填的闸
        top = Y - GATE_H / 2
        rect(x, top, w, GATE_H, fill=C["dec_fill"], stroke=C["dec"], sw=2.4, rx=10)
        person(x + 30, Y - 2, C["dec"])
        text(x + 90, Y - 10, f"GATE {n}", 14, "bold", C["dec"], ls=0.8)
        for k, t in enumerate(GATES[n]):
            text(x + 90, Y + 8 + 14 * k, t, 11.5, fill=C["dec"], italic=True)
        # 小回环: 改决定、不加 --go 再跑, 图重出, 不耗 GPU
        lx0, lx1 = x + w - 22, x + w - 62
        path(f"M{lx0:.1f},{top:.1f} C{lx0:.1f},{top - 46:.1f} {lx1:.1f},{top - 46:.1f} {lx1:.1f},{top - 2:.1f}",
             color=C["dec"], sw=1.6, arrow="arr_dec")
        # 字要整体高过段框顶线(段框比闸高 38 px), 否则会压到右边那个段框的左上角
        text(lx0 + 8, top - 44, "edit → redraw", 11, fill=C["dec"], anchor="start")
        text(lx0 + 8, top - 30, "no GPU", 11, fill=C["dec"], anchor="start", italic=True)

# 放行时回跑: 两道闸都回到第 2 段(relabel 在那里)。越远的弧越高、落点越靠左, 嵌套不交叉
s2 = find("seg", 1)
seg_top = Y - SEG_H / 2
for n, level, entry in ((2, seg_top - 50, s2[1] + 124), (4, seg_top - 96, s2[1] + 52)):
    g = find("gate", n)
    sx = g[1] + 26
    path([(sx, Y - GATE_H / 2), (sx, level), (entry, level), (entry, seg_top - 3)],
         color=C["dec"], sw=1.5, arrow="arr_dec", dash="6 4")
    if n == 2:
        text((sx + entry) / 2, level - 7, "--go: rerun", 11, fill=C["dec"])
    else:
        text((sx + entry) / 2, level - 22, "on --go, a class-level change reruns the affected steps", 12,
             "bold", C["dec"])
        text((sx + entry) / 2, level - 7, "relabel → pick → verify → override → gen → boxes → solo · "
                                         "cached cutouts skip · drop alone reruns nothing", 11, fill=C["dec"])

# 红色的停止条件
g1 = find("look", 1)
text(g1[1] + LOOK_D / 2, Y + SEG_H / 2 + 50, "too short → reshoot", 11, fill=C["stop"])
s3 = find("seg", 2)
text(s3[1] + SEG_W / 2, Y + SEG_H / 2 + 20, "k disagrees → stop", 11, fill=C["stop"])

# 说明框
boxes = []
for n in (2, 4):
    g = find("gate", n)
    gcx = g[1] + GATE_W / 2
    title, rows = CALLOUT[n]
    h = 44 + sum(18 * len(t) + 6 for _, t in rows) + 6
    cx0 = min(max(gcx - CO_W / 2, MX), W - MX - CO_W)
    boxes.append((cx0, cx0 + CO_W))
    path([(gcx, Y + GATE_H / 2), (gcx, CO_TOP)], color=C["dec"], sw=1.2, arrow=None, dash="3 3")
    rect(cx0, CO_TOP, CO_W, h, fill="#fff", stroke=C["dec"], sw=1.3, rx=8)
    e(f'<rect x="{cx0:.1f}" y="{CO_TOP + 8}" width="4" height="{h - 16:.1f}" rx="2" fill="{C["dec"]}"/>')
    text(cx0 + 20, CO_TOP + 26, title, 13, "bold", C["dec"], anchor="start", ls=0.8)
    yy = CO_TOP + 36
    for label, lines in rows:
        yy += 6
        for k, t in enumerate(lines):
            yy += 18
            if k == 0:
                text(cx0 + 20, yy, label.upper(), 10.5, "bold", C["sub"], anchor="start", ls=1)
            text(cx0 + 88, yy, t, 12, anchor="start")
assert boxes[0][1] < boxes[1][0], f"说明框重叠了: {boxes}"
co_bottom = CO_TOP + max(44 + sum(18 * len(t) + 6 for _, t in CALLOUT[n][1]) + 6 for n in (2, 4))

# 图例
ly = co_bottom + 46
e(f'<line x1="{MX}" y1="{ly - 26}" x2="{W - MX}" y2="{ly - 26}" stroke="{C["stage_stroke"]}" stroke-width="1"/>')
lx = MX
def leg(draw, label, w=None):
    global lx
    draw(lx)
    text(lx + 40, ly + 4.5, label, 12, anchor="start")
    lx += 40 + (w or len(label) * 6.9) + 34
leg(lambda x: (rect(x, ly - 12, 30, 24, fill=C["dec_fill"], stroke=C["dec"], sw=2, rx=5),
               person(x + 15, ly + 1, C["dec"], k=0.55)), "human gate · you decide")
leg(lambda x: e(f'<polygon points="{x + 3},{ly} {x + 15},{ly - 12} {x + 27},{ly} {x + 15},{ly + 12}" '
                f'fill="{C["look_fill"]}" stroke="{C["look"]}" stroke-width="1.5"/>'), "gate · you look")
leg(lambda x: path([(x, ly), (x + 28, ly)], color=C["dec"], arrow="arr_dec"), "human loop")
leg(lambda x: path([(x, ly), (x + 28, ly)], color=C["dec"], sw=1.5, arrow="arr_dec", dash="6 4"), "rerun on --go")
leg(lambda x: text(x + 15, ly + 4.5, "stop", 11, "bold", C["stop"]), "stops the run")
for role, name in (("meas", "measure"), ("gen", "generate"), ("sense", "common sense"), ("wd", "detection prior")):
    leg(lambda x, r=role: e(f'<rect x="{x + 7}" y="{ly - 8}" width="16" height="16" rx="3" fill="{C[r]}"/>'),
        name)
H = int(ly + 34)

defs = ["<defs>"]
for mid, col in (("arr", C["line"]), ("arr_dec", C["dec"])):
    defs.append(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" '
                f'markerHeight="6.5" orient="auto"><path d="M0,0.5 L10,5 L0,9.5 z" fill="{col}"/></marker>')
defs.append("</defs>")
svg = "\n".join(['<?xml version="1.0" encoding="UTF-8"?>',
                 f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
                 f'font-family="{FONT}">', *defs, f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
                 *out, "</svg>", ""])
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_figure_simple.svg")
open(dst, "w", encoding="utf-8").write(svg)
print(f"wrote {dst}  ({W}x{H})")
