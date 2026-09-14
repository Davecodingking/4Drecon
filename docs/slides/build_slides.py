#!/usr/bin/env python
"""汇报 slide -> docs/slides/recon_tool.pptx

内容以 docs/slides_outline.md 为准; 图在 docs/slides/assets/(make_assets.py 生成)。
改字就改下面每页那一段, 版式函数不用动。

    <装了 python-pptx + pillow 的 python> docs/slides/build_slides.py

字体: 西文 Arial, 中文 Microsoft YaHei(Mac 的 PowerPoint 会自动换成苹方)。
"""
import os
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
def A(f): return os.path.join(HERE, "assets", f)

# ------------------------------------------------------------------ 配色 (和 pipeline_figure.svg 同一套)
INK, SUB, RULE, BG, GRAY = "1F2328", "667085", "D0D5DD", "F6F7F9", "475467"
DEC, LOOK, STOP = "C4320A", "175CD3", "B42318"
MEAS, GEN, SENSE, WD = "175CD3", "067647", "B54708", "6941C6"      # 量 / 猜 / 常识 / 补漏
MOD = {1: ("找全", "0E7090"), 2: ("摆对", "175CD3"), 3: ("把关", "C4320A")}
LIGHT = {MEAS: "EFF8FF", GEN: "ECFDF3", SENSE: "FFFAEB", WD: "F4F3FF", GRAY: "F2F4F7",
         DEC: "FFF4ED", LOOK: "EFF8FF", "0E7090": "ECFDFF"}
LATIN, EA = "Arial", "Microsoft YaHei"

SW, SH, M = 13.333, 7.5, 0.6
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
BLANK = prs.slide_layouts[6]
def rgb(h): return RGBColor.from_string(h)

def _no_theme_style(shape):
    """去掉形状上的主题样式(p:style)。python-pptx 新建的形状和连线都挂着它,
    会带出阴影 —— 分隔线渲染成粗灰条。填充、线色、字色都已显式给了, 去掉不影响别的。"""
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


# ------------------------------------------------------------------ 文字
def _run_font(run, size, bold, color, italic):
    f = run.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, LATIN
    f.color.rgb = rgb(color)
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = etree.SubElement(rPr, qn("a:ea"))
    ea.set("typeface", EA)

def fill_tf(tf, paras, size=16, color=INK, bold=False, align="l", anchor="t", spacing=1.12, after=6):
    """paras: str | [para]; para: str | dict(t=runs, bullet=bool, size, color, bold, after)
       runs: str | [run]; run: str | (str, dict(size, bold, color, italic))"""
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(paras, (str, dict)):
        paras = [paras]
    for i, p in enumerate(paras):
        p = p if isinstance(p, dict) else dict(t=p)
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        para.line_spacing = spacing
        para.space_after = Pt(p.get("after", after))
        if p.get("bullet"):
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", str(Emu(Inches(0.24))))
            pPr.set("indent", str(-Emu(Inches(0.2))))
            etree.SubElement(pPr, qn("a:buFont")).set("typeface", LATIN)
            etree.SubElement(pPr, qn("a:buChar")).set("char", "•")
        runs = p["t"] if isinstance(p["t"], list) else [p["t"]]
        for r in runs:
            t, o = (r, {}) if isinstance(r, str) else r
            run = para.add_run()
            run.text = t
            _run_font(run, o.get("size", p.get("size", size)), o.get("bold", p.get("bold", bold)),
                      o.get("color", p.get("color", color)), o.get("italic", False))

def text(s, x, y, w, h, paras, **kw):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    fill_tf(tb.text_frame, paras, **kw)
    return tb


# ------------------------------------------------------------------ 形状
def box(s, x, y, w, h, fill=None, line=None, lw=1.0, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        radius=0.08, paras=None, **kw):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    else:
        sh.fill.background()
    if line:
        sh.line.color.rgb = rgb(line); sh.line.width = Pt(lw)
    else:
        sh.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    _no_theme_style(sh)
    if paras is not None:
        fill_tf(sh.text_frame, paras, **kw)
    return sh

def line(s, x1, y1, x2, y2, color=GRAY, w=1.5, arrow=True):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color); c.line.width = Pt(w)
    _no_theme_style(c)
    if arrow:
        ln = c.line._get_or_add_ln()
        t = etree.SubElement(ln, qn("a:tailEnd"))
        t.set("type", "triangle"); t.set("w", "med"); t.set("len", "med")
    return c

def poly(s, pts, color=GRAY, w=1.5):
    """折线, 只在最后一段带箭头。"""
    for i in range(len(pts) - 1):
        line(s, *pts[i], *pts[i + 1], color=color, w=w, arrow=(i == len(pts) - 2))

def pic(s, path, x, y, w, h, valign="m"):
    """等比塞进 (x, y, w, h), 水平居中。返回实际占的矩形。"""
    if not os.path.exists(path):
        box(s, x, y, w, h, fill=BG, line=RULE, paras=f"(图待补: {os.path.basename(path)})",
            color=SUB, align="c", anchor="m", size=14)
        return x, y, w, h
    iw, ih = Image.open(path).size
    k = min(w / iw, h / ih)
    pw, ph = iw * k, ih * k
    px = x + (w - pw) / 2
    py = {"t": y, "m": y + (h - ph) / 2, "b": y + h - ph}[valign]
    s.shapes.add_picture(path, Inches(px), Inches(py), Inches(pw), Inches(ph))
    return px, py, pw, ph

def caption(s, x, y, w, t, align="l"):
    text(s, x, y, w, 0.3, t, size=11, color=SUB, align=align)

def heading(s, x, y, w, t, color=INK, size=13):
    text(s, x, y, w, 0.3, t, size=size, bold=True, color=color, after=0)

def stat(s, x, y, w, big, small, color=INK, big_size=30):
    text(s, x, y, w, 0.6, big, size=big_size, bold=True, color=color, after=0)
    text(s, x, y + 0.62, w, 0.6, small, size=12, color=SUB)


# ------------------------------------------------------------------ 每页的骨架
PAGE = [0]
def slide(tag, title, notes="", module=None, title_size=24):
    s = prs.slides.add_slide(BLANK)
    PAGE[0] += 1
    text(s, M, 0.34, 8.0, 0.3, tag, size=12, bold=True, color=SUB)
    text(s, M, 0.66, SW - 2 * M, 0.75, title, size=title_size, bold=True)
    text(s, M, SH - 0.44, 6.0, 0.3, "场景重建工具 · 2026-09", size=10, color=SUB)
    text(s, SW - M - 1.0, SH - 0.44, 1.0, 0.3, str(PAGE[0]), size=10, color=SUB, align="r")
    if module:
        # 右上角的进度: 讲到第几个模块
        x = SW - M - 3 * 1.02
        for i in (1, 2, 3):
            name, col = MOD[i]
            on = (i == module)
            box(s, x, 0.3, 0.96, 0.32, fill=col if on else None, line=col if on else RULE, lw=1,
                radius=0.5, paras=f"{'①②③'[i - 1]} {name}", size=11, bold=on,
                color="FFFFFF" if on else SUB, align="c", anchor="m")
            x += 1.02
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


# ================================================================== P1 开场
s = slide("场景重建工具 · 汇报", "一段视频进去，出来一个可编辑、真实尺寸的三维场景", title_size=30,
          notes="30 秒讲完输入、输出、为什么难。\n图是 timesquare 渲染回原视角：左实拍、中重建、右叠加。")
text(s, M, 1.42, 9, 0.4, "画面里每个物体都是独立模型，能单独移动、替换", size=16, color=SUB)
px, py, pw, ph = pic(s, A("p1_timesquare.jpg"), M, 2.25, 8.5, 3.4, valign="t")
for i, t in enumerate(("实拍", "重建", "叠加")):
    text(s, px + pw * i / 3, py - 0.34, pw / 3, 0.3, t, size=12, bold=True, color=SUB, align="c")
X = M + 8.9
heading(s, X, 1.95, 3.6, "已跑通的场景 · 独立物体数", color=SUB, size=12)
for i, (n, name, note) in enumerate((("127", "timesquare", "街景，回归基准"),
                                     ("69", "carpark", "已校到真实米制"),
                                     ("57", "beach", "海滩，含人工闸完整一轮"),
                                     ("28", "pedestrian", "行人街道"))):
    y = 2.35 + i * 0.86
    text(s, X, y, 1.3, 0.6, n, size=30, bold=True, color=INK, after=0)
    text(s, X + 1.35, y + 0.06, 2.3, 0.3, name, size=14, bold=True, after=0)
    text(s, X + 1.35, y + 0.38, 2.3, 0.3, note, size=11, color=SUB)
box(s, M, 6.02, 0.07, 0.62, fill=DEC, radius=0)
text(s, M + 0.22, 5.98, 11.5, 0.7,
     [[("难在哪：", dict(bold=True)), "摄像机只看得到物体朝着它的那一面。看不见的那一半，要靠生成补出来。"]],
     size=17, anchor="m")


# ================================================================== P2 技术链
s = slide("第一部分 · 技术链", "没有哪个模型能单独做完：七个环节，各做一件事",
          notes="只讲每一环产出什么，不讲模型内部。\n"
                "备注（要不要讲你定）：DA3 权重许可据 9.4 综述是非商用，产品化前要确认。")
NODES = [("Qwen3-VL", "看懂画面里有什么", "词表", SENSE),
         ("SAM3", "按词逐帧抠出每个物体", "每帧的 mask", MEAS),
         ("Depth Anything 3", "算深度和相机，把 mask 抬成三维", "三维盒子", MEAS),
         ("SAM 3D Objects", "一张抠图生成完整模型", "三维模型", GEN),
         ("Qwen3-VL", "常识：“车一般多高”", "真实尺度", SENSE),
         ("WildDet3D", "靠先验把漏掉的捡回来", "补漏的物体", WD),
         ("组装 + 物理", "摆回去、落地、推开穿模", "场景", GRAY)]
nw, gap = 1.5, 0.272
y0 = 1.85
for i, (name, what, out, col) in enumerate(NODES):
    x = M + i * (nw + gap)
    box(s, x, y0, nw, 0.5, fill=col, radius=0.12, paras=name, size=12, bold=True,
        color="FFFFFF", align="c", anchor="m")
    box(s, x, y0 + 0.5, nw, 1.05, fill=LIGHT[col], radius=0.06,
        paras=what, size=12, align="c", anchor="m")
    line(s, x + nw / 2, y0 + 1.6, x + nw / 2, y0 + 1.98, color=col, w=1.25)
    box(s, x + 0.1, y0 + 2.0, nw - 0.2, 0.42, line=col, lw=1.25, radius=0.5,
        paras=out, size=12, bold=True, color=col, align="c", anchor="m")
    if i < len(NODES) - 1:
        line(s, x + nw + 0.03, y0 + 1.02, x + nw + gap - 0.03, y0 + 1.02, color=GRAY, w=1.5)
text(s, M, y0 + 2.55, 3, 0.3, "↑ 交给下一环的东西", size=11, color=SUB)
# 图例
lx = M
for col, t in ((MEAS, "量：从照片测出来"), (GEN, "猜：生成补出看不见的"),
               (SENSE, "常识：语言模型的先验"), (WD, "补漏：检测模型的先验"), (GRAY, "组装")):
    box(s, lx, 4.97, 0.22, 0.22, fill=col, radius=0.2)
    text(s, lx + 0.3, 4.92, 2.4, 0.3, t, size=12, color=INK)
    lx += 2.45
box(s, M, 5.55, SW - 2 * M, 1.15, fill=BG, radius=0.04)
box(s, M, 5.55, 0.07, 1.15, fill=INK, radius=0)
text(s, M + 0.25, 5.62, SW - 2 * M - 0.4, 1.0, [
    [("每个都是现成模型。", dict(bold=True)), "我们的工作是把它们串成一条链，并在每个接口处加校验：上一环的输出，先检查再交给下一环。"],
    "同一个 Qwen3-VL 出场两次：开头列“有什么”，后面当“尺子”。"], size=15, anchor="m")


# ================================================================== P3 量 / 猜 / 常识
s = slide("第一部分 · 技术链", "测量定位置，生成补形状，常识定尺寸",
          notes="三栏各讲一句。重点是最下面那句：三者各管一件，互不越界。\n"
                "出租车盒子 3.37 m / 真车 4.9 m 来自 9.4 首跑的实测。")
cw, cg, y0 = 3.83, 0.32, 1.6
COLS = [(MEAS, "量 · SAM3 + DA3", "p3_taxi_photo.jpg",
         "位置、朝向可信，但只量得到看得见的那一面", "出租车：盒子量出 3.37 m，真车 4.9 m"),
        (GEN, "猜 · SAM 3D Objects", "p3_taxi_mesh.jpg",
         "照片里只有车尾，生成出完整车身，连车头都有", "形状完整，但不知道该放哪、该多大"),
        (SENSE, "常识 · Qwen3-VL", None,
         "把整个场景从 DA3“以为的米”校准成真米", "停车场：整体放大 6.43 倍")]
for i, (col, head, img, t1, t2) in enumerate(COLS):
    x = M + i * (cw + cg)
    box(s, x, y0, cw, 0.5, fill=col, radius=0.1, paras=head, size=15, bold=True,
        color="FFFFFF", align="c", anchor="m")
    if img:
        box(s, x, y0 + 0.6, cw, 2.2, fill=BG, radius=0.03)
        pic(s, A(img), x + 0.05, y0 + 0.65, cw - 0.1, 2.1)
    else:
        box(s, x, y0 + 0.6, cw, 2.2, fill=LIGHT[col], radius=0.03,
            paras=[dict(t="“轿车一般 1.5 米高”", size=24, bold=True, color=col),
                   dict(t="“集装箱一般 2.5 米高”", size=16, color=col)],
            align="c", anchor="m")
    text(s, x, y0 + 2.95, cw, 1.3, [dict(t=t1, size=14), dict(t=t2, size=14, bold=True, color=col)])
box(s, M, 5.95, SW - 2 * M, 0.78, fill=BG, radius=0.04)
text(s, M + 0.25, 5.97, SW - 2 * M - 0.4, 0.74, [[
    ("三者各管一件：", dict(bold=True)),
    ("mesh 管形状", dict(color=GEN, bold=True)), " · ",
    ("盒子管位置和朝向", dict(color=MEAS, bold=True)), " · ",
    ("尺子管绝对尺度", dict(color=SENSE, bold=True)),
    "。按盒子逐个缩放 mesh，会把车压扁。"]], size=15, anchor="m")


# ================================================================== P4 短板
s = slide("第一部分 · 技术链", "每一环都有短板，由下一环或人来补",
          notes="这一页就是第二部分的目录。讲完，听众已经知道后面每个模块为什么存在。")
COLW = [4.55, 3.35, 2.35, 1.88]
HEAD = ["短板", "后果", "谁来补", "第二部分"]
ROWS = [("VLM 会列同义词：shop / storefront 其实是同一栋楼", "同一个东西抠几遍、生成几遍", "跨词去重", 1),
        ("SAM3 会漏掉被挡住的、小的", "场景里缺东西", "WildDet3D 召回", 1),
        ("DA3 自称真米，停车场却小了 6.4 倍", "整个世界等比缩小，看渲染图发现不了", "VLM 当尺子校准", 2),
        ("VLM 判不了抠图全不全：只有桌面没桌腿，也说是桌子", "残缺的抠图被拿去生成", "人工闸", 3)]
y = 1.75
x = M
for w, h in zip(COLW, HEAD):
    text(s, x, y, w, 0.35, h, size=12, bold=True, color=SUB)
    x += w
y += 0.42
line(s, M, y, SW - M, y, color=INK, w=1.25, arrow=False)
for short, effect, fix, mod in ROWS:
    x = M
    name, col = MOD[mod]
    text(s, x, y + 0.1, COLW[0] - 0.2, 0.8, short, size=15, anchor="m"); x += COLW[0]
    text(s, x, y + 0.1, COLW[1] - 0.2, 0.8, effect, size=14, color=SUB, anchor="m"); x += COLW[1]
    text(s, x, y + 0.1, COLW[2] - 0.1, 0.8, fix, size=15, bold=True, anchor="m"); x += COLW[2]
    box(s, x, y + 0.3, 1.75, 0.42, fill=col, radius=0.5, paras=f"模块 {mod} · {name}",
        size=12, bold=True, color="FFFFFF", align="c", anchor="m")
    y += 1.0
    line(s, M, y, SW - M, y, color=RULE, w=0.75, arrow=False)


# ================================================================== P5 流程总览(原生画, 两行)
s = slide("第二部分 · 逻辑模块", "六段自动跑、六道闸停下来，人只在两道闸动手",
          notes="蓝色闸只看，橙色闸要填。任何一步出错就地停。\n"
                "段框右上角的小标签，是后面哪个模块讲这一段。完整流程图见 docs/pipeline_figure.svg。")
SEGS = [("1  地基", "抽帧 · 列词表\n深度 + 相机 · 定“天”", 2),
        ("2  生成前", "跟踪 · 去重 · 挪类别\n挑图 · 验图 · 换图", 1),
        ("3  尺度", "生成模型 · 量盒子\n问真实高度 · 标定 k", 2),
        ("4  召回", "逐实例生成 · WildDet3D\n再找 · 比对 · 补抠", 1),
        ("5  验收", "召回的也生成 · 合并\n摆回 · 物理 · 渲染回图", 2),
        ("6  收尾", "楼单独出一份\nWildDet3D 对照版", None)]
GATES = [("闸 1 · 看", "看地基", False), ("闸 2 · 填", "审候选", True), ("闸 3 · 看", "看尺度", False),
         ("闸 4 · 填", "审召回", True), ("闸 5 · 看", "看场景", False), ("闸 6 · 看", "看成品", False)]
sw_, gw, gp = 2.42, 1.2, 0.2
row_w = 3 * sw_ + 3 * gw + 5 * gp
x0 = M + (SW - 2 * M - row_w) / 2
ROWY = (1.95, 4.05)
sh_ = 1.3
for r in (0, 1):
    x = x0
    y = ROWY[r]
    for k in range(3):
        i = r * 3 + k
        name, steps, mod = SEGS[i]
        box(s, x, y, sw_, sh_, fill=BG, line=RULE, radius=0.06)
        text(s, x + 0.14, y + 0.1, sw_ - 0.3, 0.35, name, size=15, bold=True)
        text(s, x + 0.14, y + 0.5, sw_ - 0.2, 0.75, steps.split("\n"), size=11.5, color=SUB, after=2)
        if mod:
            mn, mc = MOD[mod]
            box(s, x + sw_ - 0.68, y + 0.12, 0.56, 0.26, fill=mc, radius=0.5, paras=mn,
                size=9.5, bold=True, color="FFFFFF", align="c", anchor="m")
        line(s, x + sw_, y + sh_ / 2, x + sw_ + gp, y + sh_ / 2)
        gx = x + sw_ + gp
        g, gsub, fillme = GATES[i]
        col = DEC if fillme else LOOK
        box(s, gx, y + sh_ / 2 - 0.36, gw, 0.72, fill=col if fillme else LIGHT[LOOK], line=col,
            lw=1.75 if fillme else 1.25, radius=0.5 if not fillme else 0.15,
            paras=[dict(t=g, size=12, bold=True, color="FFFFFF" if fillme else col, after=0),
                   dict(t=gsub, size=10, color="FFFFFF" if fillme else col, after=0)],
            align="c", anchor="m")
        if fillme:
            box(s, gx + 0.08, y + sh_ / 2 + 0.44, gw - 0.16, 0.26, fill=MOD[3][1], radius=0.5,
                paras="把关", size=9.5, bold=True, color="FFFFFF", align="c", anchor="m")
        if k < 2:
            line(s, gx + gw, y + sh_ / 2, gx + gw + gp, y + sh_ / 2)
        x = gx + gw + gp
    if r == 0:     # 第一行末尾的闸 3 拐到第二行开头
        last_gx = x - gp - gw
        ymid = (ROWY[0] + sh_ + ROWY[1]) / 2
        poly(s, [(last_gx + gw / 2, ROWY[0] + sh_ / 2 + 0.36), (last_gx + gw / 2, ymid),
                 (x0 + sw_ / 2, ymid), (x0 + sw_ / 2, ROWY[1])])
box(s, M, 5.85, 0.9, 0.34, fill=LIGHT[LOOK], line=LOOK, radius=0.5, paras="只看", size=11,
    bold=True, color=LOOK, align="c", anchor="m")
text(s, M + 1.0, 5.83, 4.2, 0.4, "看完敲 --go 放行", size=13)
box(s, M + 4.3, 5.85, 0.9, 0.34, fill=DEC, radius=0.15, paras="要填", size=11, bold=True,
    color="FFFFFF", align="c", anchor="m")
text(s, M + 5.3, 5.83, 6.9, 0.4, "改了再看不耗 GPU；放行时自动回跑受影响的步骤", size=13)
text(s, M, 6.33, 9, 0.35, "任何一步出错就地停，重跑时从这一段开头来，已完成的步骤自动跳过。",
     size=12, color=SUB)


# ================================================================== 模块 1
s = slide("第二部分 · 模块 1 · 把物体找全", "同一个东西被几个词各抠一遍：生成之前就按照片合并", module=1,
          notes="人看候选图看不出来：两个词各自都抠得很正常，只有放在同一帧上比才知道。\n"
                "为什么会重：SAM3 是一个词跑一次，词和词之间不互斥。\n"
                "死角：两个词从没在同一帧同时出现的，这一步看不见，留给量盒子时的三维体素去重兜底。\n"
                "数字出处：devLog/devlog_9.8.md；图是 beach r1 第 203 帧。")
LW = 5.7
y = 1.7
for head, body in (("问题", "shop / storefront / restaurant 抠的是同一栋楼 —— 生成三次，只用一次"),
                   ("做法", "所有词的所有实例摊平，在同一帧上比 mask 重叠；重叠高的连成一组，一组挑一个代表"),
                   ("为什么放在生成之前、用照片比",
                    "三维那道去重排在生成之后，发现重复时同义词已经各生成过一次；在同一帧的像素上比就够了，用不到深度")):
    heading(s, M, y, LW, head, color=MOD[1][1])
    text(s, M, y + 0.32, LW, 0.9, body, size=15)
    y += 1.2
px, py, pw, ph = pic(s, A("p6_dedup.jpg"), M + LW + 0.45, 1.65, SW - 2 * M - LW - 0.45, 3.85)
caption(s, px, py + ph + 0.06, pw, "beach：bin#10 和 trash can#10，共同出现 94 帧，重叠 IoU 0.97 —— 同一个垃圾桶")
for i, (big, small) in enumerate((("13 / 13", "bin 的实例全部和 trash can 配上对"),
                                  ("0", "三个场景核过账，没有物体因合并而丢失"))):
    stat(s, M + i * 2.9, 5.45, 2.8, big, small, color=MOD[1][1])

s = slide("第二部分 · 模块 1 · 把物体找全", "召回把漏掉的捡回来，而且只捡真漏的", module=1,
          notes="早期报的“补漏收益”大半是假的：同一辆卡车被算成三个新物体。\n"
                "改成在照片上比重叠之后，剩下的逐个看过都是真漏。")
LW = 6.1
heading(s, M, 1.7, LW, "问题", color=MOD[1][1])
text(s, M, 2.02, LW, 0.8, "SAM3 会漏掉被挡住的、太小的。而早期判的“漏检”大半是假的：同一辆卡车被算成三个新物体",
     size=15)
heading(s, M, 2.95, LW, "做法", color=MOD[1][1])
text(s, M, 3.27, LW, 1.0, "WildDet3D 全图再扫一遍 → 和已有的在照片上比重叠 → 框喂回 SAM3 补抠 → "
                          "四道门槛，每个候选的结局都记进去向表", size=15)
y = 4.45
for j, h in enumerate(("场景", "原来判的“漏检”", "照片比对后")):
    text(s, M + [0, 2.0, 4.1][j], y, 2.0, 0.3, h, size=12, bold=True, color=SUB)
line(s, M, y + 0.36, M + LW, y + 0.36, color=INK, w=1, arrow=False)
for i, (sc, a, b) in enumerate((("timesquare", "48", "11"), ("carpark", "46", "4"), ("beach", "14", "7"))):
    yy = y + 0.42 + i * 0.46
    text(s, M, yy, 2.0, 0.4, sc, size=15)
    text(s, M + 2.0, yy, 2.0, 0.4, a, size=16, color=SUB)
    text(s, M + 4.1, yy, 2.0, 0.4, b, size=16, bold=True, color=MOD[1][1])
text(s, M, 6.3, LW, 0.4, "剩下的逐个看过，都是真漏：深色广告牌、旋转木马、遮阳棚", size=13, color=SUB)
X = M + LW + 0.35
CW = SW - M - X
for y, tag, col, head, img, h, cap in (
        (1.6, "判掉", STOP, "黄框底下已经有 SAM3 抠过的东西", "p7_rejected.jpg", 1.95,
         "timesquare：同一辆卡车被算成 3 个“新物体”，IoU 都是 0.74"),
        (4.42, "召回成功", MOD[1][1], "黄框底下是空的 → 框喂回 SAM3 补抠出 mask", "p7_recalled.jpg", 1.7,
         "beach 热狗车：和已有的 IoU 只有 0.13，是真漏检；右边是补抠出来的 mask")):
    box(s, X, y + 0.02, 0.95, 0.3, fill=col, radius=0.5, paras=tag, size=11, bold=True,
        color="FFFFFF", align="c", anchor="m")
    text(s, X + 1.05, y, CW - 1.05, 0.34, head, size=13, bold=True)
    px, py, pw, ph = pic(s, A(img), X, y + 0.4, CW, h, valign="t")
    caption(s, X, py + ph + 0.04, CW, cap)


# ================================================================== 模块 2
s = slide("第二部分 · 模块 2 · 把物体摆对", "先定“天”：偏 4.5°，离场景中心 8 米就差半米", module=2,
          notes="它在最上游：这里错了，下游任何精修都是在给错的东西抛光。")
LW = 6.6
y = 1.75
for head, body in (("问题", "所有高度、地面都沿“天”的方向量。室内这一步从没跑过，全链一直用一个兜底方向，偏了约 4.5°"),
                   ("做法", "从点云拟合地面，定出“天”朝哪。缺这个结果现在直接停，不再静默猜一个方向"),
                   ("为什么先做它", "它在最上游。这里错了，下游任何精修都是在给错的东西抛光")):
    heading(s, M, y, LW, head, color=MOD[2][1])
    text(s, M, y + 0.32, LW, 1.0, body, size=16)
    y += 1.45
px, py, pw, ph = pic(s, A("../../../data/outdoorOutput/pedestrian/02_ground_check.jpg"),
                     M + LW + 0.4, 1.6, SW - 2 * M - LW - 0.4, 5.0)
caption(s, px, py + ph + 0.04, pw, "pedestrian 地基核对图：抽帧 · 点云俯视带相机轨迹 · 侧视带拟合的地面线")

s = slide("第二部分 · 模块 2 · 把物体摆对", "深度模型自称真米，停车场却小了 6.4 倍：用常识当尺子", module=2,
          notes="更危险的是看渲染图发现不了：整个世界一起缩小，互相自洽。\n"
                "护栏是扁的，高度只量到 6.5 厘米，一除就炸成 15 倍 —— 所以真实高度低于 0.5 米的不当尺子。\n"
                "验证用的是没参与计算的那一维（车长）。数据：data/outdoorOutput/carpark/02_world_scale.json\n"
                "下半页是最新的 pedestrian：k = 1.34，7 类全部采用，一致率 0.895（门槛 0.6）；人的中位高度 1.64 m，常识 1.70 m。\n"
                "如实备注（被问到时用）：同场景车的中位高度 1.78 m，比常识 1.50 m 高约 19%，车的盒子噪声大（0.53~2.62 m）。\n"
                "数据：data/outdoorOutput/pedestrian/02_world_scale.json、06_all_boxes.json")
LW = 7.2
text(s, M, 1.58, LW, 0.4, "DA3 在俯拍素材上判错，车只有 0.23 米，而且看渲染图发现不了", size=15)
text(s, M, 2.04, LW, 0.36, [[("carpark", dict(bold=True)), "　全局倍数 ",
                              ("k = 6.43", dict(bold=True, color=SENSE))]], size=14)
TW = [1.35, 0.9, 1.45, 2.0, 1.35]
y = 2.5
x = M
for w, h in zip(TW, ("类", "实例", "DA3 量到", "校正后", "常识")):
    text(s, x, y, w, 0.3, h, size=12, bold=True, color=SUB); x += w
line(s, M, y + 0.34, M + sum(TW), y + 0.34, color=INK, w=1, arrow=False)
for i, row in enumerate((("轿车", "19", "0.23 m", "1.45 m", "1.50 m"), ("集装箱", "14", "0.40 m", "2.56 m", "2.50 m"),
                         ("厢车", "5", "0.31 m", "2.00 m", "2.00 m"), ("护栏", "5", "0.065 m", "剔除（算出 15 倍）", "1.00 m"))):
    x = M
    yy = y + 0.42 + i * 0.44
    for j, (w, v) in enumerate(zip(TW, row)):
        text(s, x, yy, w, 0.38, v, size=14 if (j != 3 or i < 3) else 12,
             bold=(j == 3 and i < 3), color=(SENSE if j == 3 and i < 3 else (SUB if i == 3 else INK)))
        x += w
X = M + 7.55
CW = SW - M - X
for i, (head, body, col) in enumerate((
        ("做法", "每一类：量到的高度 ÷ 常识高度 = 倍数，取加权中位数，离群的剔掉", SENSE),
        ("验证：用没参与计算的那一维", "只用高度校准；车长量出 4.40 米，真车 4.5 米", GEN),
        ("保险", "各类意见不一致就拒绝校准，只报警 —— 改错比不改更糟", DEC))):
    y = 1.58 + i * 1.08
    box(s, X, y, CW, 0.98, fill=LIGHT[col], radius=0.05)
    box(s, X, y, 0.07, 0.98, fill=col, radius=0)
    heading(s, X + 0.22, y + 0.08, CW - 0.3, head, color=col, size=13)
    text(s, X + 0.22, y + 0.4, CW - 0.35, 0.56, body, size=13)
# 下半页: 最新的 pedestrian —— 校准之后, 人和车摆回画面里的大小关系是对的
line(s, M, 4.9, SW - M, 4.9, color=RULE, w=0.75, arrow=False)
box(s, M, 5.05, 1.6, 0.3, fill=GEN, radius=0.5, paras="最新 · pedestrian", size=11, bold=True,
    color="FFFFFF", align="c", anchor="m")
text(s, M + 1.72, 5.02, 7.2, 0.36, "校准之后，人摆回画面里大小正常", size=14, bold=True)
pic(s, A("p9_pedestrian.jpg"), M, 5.45, 8.9, 1.5, valign="t")
X2 = M + 9.2
stat(s, X2, 5.0, 1.55, "1.64 m", ["人的中位高度", "常识 1.70 m"], color=GEN, big_size=24)
stat(s, X2 + 1.62, 5.0, 1.35, "0.90", ["7 类一致率", "门槛 0.6"], color=GEN, big_size=24)
caption(s, X2, 6.35, SW - M - X2, "第 0088 帧：左实拍、中重建、右叠加。全局 k = 1.34")

s = slide("第二部分 · 模块 2 · 把物体摆对", "摆回去：朝向对齐、落地、推开穿模", module=2,
          notes="朝向：生成的模型没有统一朝向，长轴有的在 X 有的在 Z，摆之前先把 mesh 自己的长轴转正。\n"
                "物理数字来自室内 cyberpunk（devlog_9.8）。")
LW = 5.6
heading(s, M, 1.7, LW, "朝向", color=MOD[2][1])
stat(s, M, 2.02, 2.6, "17 / 51", "carpark 里被横着摆（差 90°）的物体", color=STOP)
stat(s, M + 2.8, 2.02, 2.6, "0°", "现在的中位偏差", color=MOD[2][1])
text(s, M, 3.3, LW, 0.4, "90° 解决了；180° 正反（车头朝哪）还没解", size=13, color=SUB)
heading(s, M, 3.9, LW, "物理（室内 cyberpunk）", color=MOD[2][1])
y = 4.25
for j, h in enumerate(("", "之前", "之后")):
    text(s, M + [0, 2.6, 4.6][j], y, 2.0, 0.3, h, size=12, bold=True, color=SUB)
line(s, M, y + 0.34, M + LW, y + 0.34, color=INK, w=1, arrow=False)
for i, (a, b, c) in enumerate((("该落地却悬空", "32 个，最高 98 cm", "0"),
                               ("物体互相穿模", "最深 25.6 cm", "0"),
                               ("地板铺不到墙根", "平均差 22 cm", "0"))):
    yy = y + 0.42 + i * 0.46
    text(s, M, yy, 2.6, 0.4, a, size=15)
    text(s, M + 2.6, yy, 2.0, 0.4, b, size=15, color=SUB)
    text(s, M + 4.6, yy, 1.0, 0.4, c, size=16, bold=True, color=MOD[2][1])
px, py, pw, ph = pic(s, A("p10_cyberpunk.jpg"), M + LW + 0.4, 2.0, SW - 2 * M - LW - 0.4, 4.2)
for i, t in enumerate(("实拍", "重建", "叠加")):
    text(s, px + pw * i / 3, py - 0.34, pw / 3, 0.3, t, size=12, bold=True, color=SUB, align="c")
caption(s, px, py + ph + 0.05, pw, "室内 cyberpunk，物理处理之后渲染回原视角")


# ================================================================== 模块 3
s = slide("第二部分 · 模块 3 · 人在哪儿把关", "必须停在生成之前：生成最贵，而“该不该生成”机器判不了", module=3,
          notes="VLM 判得了“是不是”，判不了“全不全”。所以停在最贵的那一步之前，让人看一眼。")
cw, cg = 3.83, 0.32
for i, (big, small) in enumerate((("只有桌面没桌腿，VLM 照样说是桌子", "它判得了“是不是”，判不了“全不全”"),
                                  ("让它找沙发，它把最像的报出来", "抠出来的其实是餐桌桌面"),
                                  ("pedestrian 15 个“人”里 3 个不是人", "背包、碎片、半截身子。现在 VLM 先逐个筛，人再看"))):
    x = M + i * (cw + cg)
    box(s, x, 1.65, cw, 2.1, fill=LIGHT[DEC], radius=0.05)
    box(s, x, 1.65, 0.07, 2.1, fill=DEC, radius=0)
    text(s, x + 0.25, 1.8, cw - 0.4, 1.0, big, size=17, bold=True)
    text(s, x + 0.25, 2.85, cw - 0.4, 0.85, small, size=13, color=SUB)
heading(s, M, 4.05, 8, "pedestrian · 闸 2", color=SUB, size=12)
px, py, pw, ph = pic(s, A("p11_person.jpg"), M, 4.4, SW - 2 * M, 1.85)
caption(s, px, py + ph + 0.06, pw, "左：留下的 3 个人。右：闸 2 上被判掉的 5 个 person 候选 —— 碎片、半截身子、一团黑影")

s = slide("第二部分 · 模块 3 · 人在哪儿把关", "闸 4 审召回：主线漏掉的那一批，最该人看一眼", module=3,
          notes="召回捡回来的都是主线漏掉的：多半被挡着、偏小，补抠 mask 天生容易失败。\n"
                "左图 pedestrian r2：召回找到 3 个、一个都没进，每个都写了原因。"
                "盆栽抠出 0.149%，门槛 0.15%，只差 0.001，所以标了 borderline —— 门槛是拍出来的，值得人看一眼。\n"
                "右图 beach r2：召回成功的热狗车和椅子，黄色 recalled 标签，只能 keep / drop"
                "（它们的抠图是按框现抠的，没有别的帧可换）。\n"
                "闸 4 一进来，决策台会自动弹出召回表。")
LW = 5.3
y = 1.65
for head, body in (
        ("为什么要第二道闸", "召回捡回来的都是主线漏掉的：多半被挡着、偏小，补抠最容易失败。质量最没保证，也最容易被忽略"),
        ("看什么", "召回去向表：找到几个、进了几个，没进的为什么没进，一个不落。召回成功的排在候选最后，标黄"),
        ("召回的物体归哪类", None),
        ("回退", "召回前那一版原样保留，不想要召回的结果可以退回去")):
    heading(s, M, y, LW, head, color=DEC)
    if body:
        text(s, M, y + 0.32, LW, 0.9, body, size=14)
        y += 1.1
    else:
        text(s, M, y + 0.32, LW, 1.1, [
            dict(t=[("决定过的词", dict(bold=True)), " → 继承（共用的继续共用）"], bullet=True, after=3),
            dict(t=[("没见过的词", dict(bold=True)), " → 默认单独生成，标黄等人判"], bullet=True, after=3),
            dict(t=[("分过组的词", dict(bold=True)), " → 停下，等人指定进哪组"], bullet=True, after=3)], size=14)
        y += 1.35
X = M + LW + 0.35
px, py, pw, ph = pic(s, A("p12_recall_table_crop.png"), X, 1.6, 3.45, 4.55, valign="t")
caption(s, px, py + ph + 0.05, 3.45, "pedestrian：找到 3 个，一个都没进，每个都写了原因（盆栽只差门槛 0.001）")
X2 = X + 3.6
CW = SW - M - X2
px, py, pw, ph = pic(s, A("p12_recalled_cards_crop.png"), X2, 1.6, CW, 2.6, valign="t")
caption(s, X2, py + ph + 0.05, CW, "beach：召回成功的热狗车和椅子排在候选最后，黄色标签，只能 keep / drop")

s = slide("第二部分 · 模块 3 · 人在哪儿把关", "闸是回路，不是关卡", module=3,
          notes="左图是流程图里闸 2 那一块（闸 4 逻辑一样）。\n"
                "橙色回路：改了再看；右边虚线框：放行时回跑。")
px, py, pw, ph = pic(s, A("p12_gate2.png"), M, 1.6, 6.9, 5.1)
X = M + 7.25
CW = SW - M - X
for i, (head, body, col) in enumerate((
        ("回路 1 · 改了再看", "不满意 → 改决定、不加 --go 再跑，图按新决定重出。不耗 GPU，不限次", DEC),
        ("回路 2 · 放行时回跑", "只改了 drop：不回跑。改了类级决定：自动回跑受影响的步骤，已缓存的跳过", DEC),
        ("默认", "每个物体单独生成，共用要人说。15 个行人共用一个模型，就是满街同一件衬衫", GRAY),
        ("两道闸通用", "闸 2 和闸 4 的回路、能填的字段、放行时回跑，完全一样；闸 4 只多了召回的归类规则", GRAY))):
    y = 1.65 + i * 1.26
    heading(s, X, y, CW, head, color=col, size=14)
    text(s, X, y + 0.36, CW, 0.85, body, size=13.5)

s = slide("第二部分 · 模块 3 · 人在哪儿把关", "决策台：在网页上点，不用改文件", module=3,
          notes="可以现场演示 1 分钟：tools/gate_ui.py --scene <场景> --run <轮次>，默认端口 8765。\n"
                "截图是 pedestrian r2 的闸 2。")
LW = 8.4
px, py, pw, ph = pic(s, A("p13_desk.png"), M, 1.6, LW, 5.1, valign="t")
X = M + LW + 0.35
CW = SW - M - X
text(s, X, 1.7, CW, 4.8, [
    dict(t="每个类一格代表图，点开能看全部实例", bullet=True, after=12),
    dict(t="keep / drop / solo / shared / 换帧，点一下就生效", bullet=True, after=12),
    dict(t="召回的候选单独一栏：进没进、为什么没进，都列出来", bullet=True, after=12),
    dict(t="只写一个决策文件：不起流水线、不碰 GPU，坏了命令行照样能跑", bullet=True, after=12)],
    size=15)

dst = os.path.join(HERE, "recon_tool.pptx")
prs.save(dst)
print(f"wrote {dst}  ({PAGE[0]} slides)")
