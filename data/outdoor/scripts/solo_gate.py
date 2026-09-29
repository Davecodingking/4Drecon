"""逐实例生成的面积门槛: 实例在挑中的那一帧里至少要有多少**像素**, 才单独生成 mesh。

挑帧(pick / verify / 闸口 / gen_per_instance)都从这里读, 别在脚本里各写一份 ——
以前 6 个脚本各自 float(os.environ.get("SOLO_TH", "0.5")), 室内那边键名还叫 TH,
名字对不上就静默用默认值, 这种事已经出过一次。

★ 为什么是像素不是画面百分比 (2026-09-16 改):
  门槛防的是"截图太小, 生成出来是垃圾" —— 那是**像素数**的事, 跟画面多大无关。
  以前写的是百分比 SOLO_TH=0.5, 在时代广场(1280x720)上调的, 相当于 4608 像素;
  换到 1920x1080 的 pedestrian 就变成 10368 像素, 同一个本意严了 2.25 倍 ——
  car #6 最好那帧 9895 像素(99 px 见方), 放在时代广场稳过, 在这里被挡掉, 借了 car_0 的模型。
  于是每换一种分辨率就得在 SCENES 里补一个覆盖(beach/carpark 0.03, 室内 0.1), 补漏就会漂。

★ 为什么默认 256 (16 px 见方): 用户要"非常宽松"。它是所有已跑轮次里旧门槛折成像素的
  最小值(ithor312 512x512 x 0.1% = 262) —— 取它, 任何场景都只放宽、不收紧。
  实测(06_all_boxes 里的实例): pedestrian/r2 25 -> 46, timesquare/r1 85 -> 188,
  beach/r1 108 -> 113, 其余不变或全过。代价是生成数变多; 太小的烂图由 VLM 校验和人工闸兜底。
"""
import os

DEFAULT_MIN_PX = 256


def min_px():
    """当前生效的门槛(像素)。老键名还在环境里就直接报错 —— 静默忽略等于偷偷换了门槛。"""
    if "SOLO_TH" in os.environ:
        raise SystemExit(
            f"[失败] 环境里还有 SOLO_TH={os.environ['SOLO_TH']} —— 这个键(画面百分比)已废弃,"
            f" 改用 SOLO_MIN_PX(像素数, 默认 {DEFAULT_MIN_PX})。unset SOLO_TH 再跑。")
    return int(os.environ.get("SOLO_MIN_PX", DEFAULT_MIN_PX))
