#!/usr/bin/env python
"""组装时两条"只看盒子"的规则。组装(assemble_scene.py)和逐实例生成(gen_per_instance.py)都从这里拿, 用的是同一份。

  绝对守卫  盒子比 MIN_H 矮、或最长边超过 MAX_D -> 当成量错了, 不摆。
            ★ 组装在判之前会先把"立在地上"的类补到地面(盒子变高), 所以这类盒子量到的高度不算数, 见 never_placed。
  去重      同一个词的两个盒子水平重叠超过小的那个的 DUP_OVERLAP、顶面高差在 DUP_DH 以内
            -> 同一件被量了两遍, 留点数多的(只对方盒子家具 FIT_WORDS, 要开 DEDUP_BOX)。

★ 为什么搬出来(office r12, 2026-10-12): 这两条以前只写在组装里, 而组装排在生成之后。47 个物体逐个生成(每个约 1 分钟),
  其中 8 个到组装时被这两条扔掉 —— 6 个不到 15 厘米高(苹果 3 厘米、鼠标、两个笔筒...), 2 张桌子是量重了的。
  两条规则只看量到的盒子(06_all_boxes.json), 生成之前盒子就有了, 所以生成那步先问一遍, 过不了的不生成。
  组装的结论不变(它们本来就进不了场景), 只是不再白生成。
★ 门槛的默认值只在这里读一次。两个脚本各读各的环境变量, 迟早有一边改了另一边没改 —— 那样就会出现
  "生成了却被扔"或者更糟的"该摆的没生成"。
★ MIN_H=0.15 是室外留下来的: 室外的东西是车、长椅、路灯, 量出来不到 15 厘米高的盒子基本是点云噪声。室内直接复用了
  同一个组装脚本, 桌面上的小东西(苹果、鼠标、笔筒)因此永远进不了场景。这里不改这个数; 要改就改 MIN_H, 两边一起变。

只用 numpy。UP("天"的方向)由调用方给: 两个脚本本来就各自 load_up 过。
"""
import os
import numpy as np
from word_rank import base_word

MIN_H = float(os.environ.get("MIN_H", "0.15"))    # 比这薄就当量错了(真米, 所以要在尺度标定之后才有意义)
MAX_D = float(os.environ.get("MAX_D", "15"))      # 比这大就当量错了
DEDUP_BOX = os.environ.get("DEDUP_BOX", "0") == "1"
DUP_OVERLAP = float(os.environ.get("DUP_OVERLAP", "0.5"))
DUP_DH = float(os.environ.get("DUP_DH", "0.12"))
# 长宽比本来就千差万别的方盒子家具: 去重和"mesh 比盒子大就往回缩"(assemble_scene 的 FREE_TRIM)只对它们做
FIT_WORDS = {x.strip().replace("_", " ") for x in os.environ.get(
    "FIT_WORDS", "desk,table,dining table,side table,coffee table,counter,cabinet,file cabinet,"
                 "shelf,shelf unit,bookcase,credenza").split(",") if x.strip()}
# A missing leg is plausible for these classes. A VLM "floor" label alone is
# insufficient evidence to grow a measured box or deform its mesh.
LEG_EXTEND_WORDS = {"desk", "table", "dining table", "side table", "coffee table", "nightstand", "counter",
                    # 2026-11-10 椅子也会腿被挡住: hitch r1 chair#3 只量到椅背(高 0.53 m, 底离地 0.30 m), 悬着、矮一截。
                    # 补完还要过 assemble 的"不超过常识高度 2 倍"那道, 挂墙的东西不会被拉下来
                    "chair", "dining chair", "armchair", "office chair", "stool", "bar stool"}


def fit_word(w):
    return w in FIT_WORDS or base_word(w) in FIT_WORDS


def may_extend_legs(w):
    return w in LEG_EXTEND_WORDS or base_word(w) in LEG_EXTEND_WORDS


def slim_hung_rail(w, ext, rest, hung=()):
    """A measured wall rail can be thinner than the generic outdoor height guard."""
    e = np.asarray(ext, float)
    r = rest.get(w) if rest.get(w) is not None else rest.get(base_word(w))
    is_hung = r == "hung" or w in hung or base_word(w) in hung
    long, short = max(e[:2]), min(e[:2])
    return (is_hung and 0.05 <= e[2] < MIN_H and long >= 0.3
            and long >= 4 * max(short, 1e-6) and long >= 4 * e[2])


def min_height_for(w, ext, rest, hung=()):
    return 0.05 if slim_hung_rail(w, ext, rest, hung) else MIN_H


def abs_ok(ext, min_h=None, max_d=None):
    """盒子尺寸 [长, 宽, 高] 过不过绝对守卫。"""
    e = np.asarray(ext, float)
    return not (e[2] < (MIN_H if min_h is None else min_h) or e.max() > (MAX_D if max_d is None else max_d))


def box_rc(b, up):
    """盒子的水平矩形: (中心, 长轴方向, 短轴方向, 半长, 半宽)。"""
    c = np.array(b["中心"], float); a = np.array(b["长轴"], float); a = a - up * (a @ up)
    a /= max(np.linalg.norm(a), 1e-9)
    return c, a, np.cross(up, a), b["尺寸"][0] / 2, b["尺寸"][1] / 2


def box_top(b, up):
    return (float(b["底高"]) if "底高" in b else float(np.array(b["中心"]) @ up - b["尺寸"][2] / 2)) + b["尺寸"][2]


def in_share(b1, b2, up):
    """b1 的水平矩形有多大比例落在 b2 里 (24x24 网格采样)。"""
    c1, a1, s1, h1, k1 = box_rc(b1, up); c2, a2, s2, h2, k2 = box_rc(b2, up)
    g = (np.arange(24) + 0.5) / 24 * 2 - 1
    U_, V_ = np.meshgrid(g * h1, g * k1); q = c1 + U_.ravel()[:, None] * a1 + V_.ravel()[:, None] * s1 - c2
    return float(((np.abs(q @ a2) <= h2) & (np.abs(q @ s2) <= k2)).mean())


def dup_boxes(boxes, up, exclude=(), on=None, overlap=None, dh=None):
    """{(词, 实例号): (留下的那个 "词#实例号", 重叠比例)} —— 被判成"和同类另一个是同一件"的盒子。
    boxes: 06_all_boxes.json 的内容 {词: [盒子]}。on 不给就看 DEDUP_BOX。"""
    on = DEDUP_BOX if on is None else on
    overlap = DUP_OVERLAP if overlap is None else overlap
    dh = DUP_DH if dh is None else dh
    dup = {}
    if not on:
        return dup
    for w, bs in boxes.items():
        if not fit_word(w) or w in exclude or base_word(w) in exclude:
            continue
        order = sorted(bs, key=lambda b: -(b.get("点数") or 0))
        for i, bi in enumerate(order):
            if (w, bi["obj_id"]) in dup:
                continue
            for bj in order[i + 1:]:
                if (w, bj["obj_id"]) in dup or abs(box_top(bi, up) - box_top(bj, up)) > dh:
                    continue
                small, big = (bj, bi) if bj["尺寸"][0] * bj["尺寸"][1] <= bi["尺寸"][0] * bi["尺寸"][1] else (bi, bj)
                ov = in_share(small, big, up)
                if ov >= overlap:
                    dup[(w, bj["obj_id"])] = (f"{w}#{bi['obj_id']}", ov)
    return dup


def hung_labels():
    """人工改判成"挂着的"词(HUNG_LABELS): 压过 VLM 说的"立在地上"。垂到地面的灯管 VLM 会判成落地灯。"""
    return {x.strip().replace("_", " ") for x in os.environ.get("HUNG_LABELS", "").split(",") if x.strip()}


def rests_on_floor(w, rest, hung=()):
    """这个词是不是"立在地上"的类(rest: 01_vlm_rest.json 的内容)。组装只给这类盒子"补到地面":
    腿被挡住、只量到桌面的桌子, 顶不动、底拉到地面, 盒子变高(见 assemble_scene.py"立在地上的东西"那段)。"""
    r = rest.get(w) if rest.get(w) is not None else rest.get(base_word(w))
    return r == "floor" and w not in hung and base_word(w) not in hung


def never_placed(w, b, dup, may_grow=False, rest=None, hung=()):
    """这个盒子组装时一定会被扔的原因(给生成那步用: 有原因就不生成); 不会被这两条扔、或者现在还说不准 = None。
    may_grow: 组装有可能把它补到地面(rests_on_floor 且开着 FLOOR_EXTEND)。补完会变高, 所以"太矮"现在判不了 ——
      office r12 的 desk#4 量到 0.06 m(只有桌面), 组装补成 0.74 m 摆进了场景。补多高要等房间壳的地板高度(生成之后才有),
      这里不猜: 可能被补的一律照常生成。"太大"不受影响(补完只会更高)。"""
    if (w, b["obj_id"]) in dup:
        return f"和 {dup[(w, b['obj_id'])][0]} 是同一件(量了两遍, 留点数多的那个)"
    e = np.asarray(b["尺寸"], float)
    if e.max() > MAX_D:
        return f"最长边 {e.max():.1f} m, 超过 {MAX_D:g} m(当成量错了)"
    min_h = min_height_for(w, e, rest or {}, hung)
    if e[2] < min_h and not may_grow:
        return f"高 {e[2]:.2f} m, 不到 {min_h} m(当成量错了)"
    return None
