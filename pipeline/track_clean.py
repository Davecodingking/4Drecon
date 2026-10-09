#!/usr/bin/env python
"""人体跟踪的清理: "只剩零星几个像素、又接不上这个人完整的帧"的那些帧不算这个人。

★ 为什么要有(2026-10-08 office r7/r8): SAM 3 偶尔会给已经跟着的人再开一个编号, 这个编号的 mask 只是散在那个人
  轮廓边上的几十到几百个像素(office 人 6 第 74 帧: 237 个像素, 框却有 306x254)。出网格那步不看像素数, 照样拿
  这个框重建出一个完整的人 —— 画面里凭空多一个人, 只在零星几帧出现, 看着是"闪"了一下。
  以前靠"最大面积不到画面 1% 的整条丢掉"顺带筛掉了; 人体线接进流水线之后那个门槛放到 0.1%(远处的真人本来就小:
  soccer 守门员 0.69%), 它就漏了过来。面积门槛分不清"远处的真人"和"认错的小点", 所以不改回去, 换成看形状和连续性。

规则(对所有场景同一条, 不看这个人在画面里有多大):
  一帧里 mask 的像素数还不到它自己包围盒面积的 THIN_FILL = 只剩零星小点;
  这样的帧, 如果紧接着(LINK_GAP 帧以内)这个编号被认可的帧、位置也接得上(中心相距不到画面宽的 LINK_DIST) —— 是被挡住了, 留着
    (nfl1 人 3 被压在别人身下的 34 帧: 接在他前面 75 帧完整的身体后面, 一帧不摘; 后面的翻转补帧要靠这些帧知道他还在);
  接不上的 —— 凭空冒出来的小点, 这一帧摘掉。
  "被认可的帧" = 实心的帧, 加上已经接上的零星帧(一帧接一帧往外传, 被压住很久也接得上)。

三个场景(office r8 / nfl1 r6 / soccer r4)的实测: 只有 office 人 6 的 14 帧和人 5 的 1 帧被摘, 其余的人一帧不动。
只用标准库, 跟踪脚本(body4d 环境)和测试(sam3d 环境)都能 import。
"""
import math

THIN_FILL = 0.05     # 像素数 / 包围盒面积 低于这个 = 只剩零星小点。实测: 零星帧 0.000~0.045, 真人(含被挡一半的)最低 0.096
LINK_GAP = 3         # 帧: 隔这么多帧以内算"紧接着"
LINK_DIST = 0.08     # 位置接得上: 两帧 mask 中心相距不到画面宽的这么多


def fill(st):
    """这一帧的 mask 把自己的包围盒填了多少。st: dict(area, bbox=[x0, y0, x1, y1](含两端))。"""
    x0, y0, x1, y1 = st["bbox"]
    return st["area"] / float((x1 - x0 + 1) * (y1 - y0 + 1))


def stray_frames(frames, width, thin=THIN_FILL, gap=LINK_GAP, dist=LINK_DIST):
    """一个编号的逐帧统计 {帧号: dict(area, cx, cy, bbox)} -> 该摘掉的帧号(从小到大)。width: 画面宽(像素)。"""
    ok = {t for t, st in frames.items() if fill(st) >= thin}
    thin_frames = sorted(set(frames) - ok)
    grew = True
    while grew:                                   # 一帧接一帧往外传, 直到没有新的接上
        grew = False
        for t in thin_frames:
            if t in ok:
                continue
            a = frames[t]
            if any(abs(s - t) <= gap and math.hypot(frames[s]["cx"] - a["cx"], frames[s]["cy"] - a["cy"]) < dist * width
                   for s in ok):
                ok.add(t); grew = True
    return sorted(set(frames) - ok)
