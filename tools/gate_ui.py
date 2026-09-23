#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""闸口决策台 —— 05_review.json 的可视化编辑器。

这个文件**不是流水线的一环**, 它是那个决策文件的编辑器。这个定位是有意的:

  已有            05_review.json          已有
  review_gate.py  ────────────▶  review_gate.py --apply
  (出图 + 出json)   ▲        │      (校验 + 落地)
                    │        │
                    └── gate_ui.py ──┘
                        (读 / 渲染 / 写回)

所以它挂了、不用了、写坏了, 命令行照旧能跑 —— 它只碰一个文件, 而那个文件本来就是给人改的。

不做的事(边界, 别往里加):
  * 不写 05_review_applied.json —— 那是 review_gate --apply 的活, 它带着一堆校验
    (改类别+复用组冲突、改用+改类别冲突、召回待决...), 绕过去等于把校验废掉。
  * 不起流水线子进程 —— 点完保存, 人回终端敲 drive.py --go。不管子进程是它能保持
    "零风险"的前提: 没有子进程就没有僵尸、没有 GPU 争抢、没有"界面关了活还在跑"。
  * 不碰 mask / 不起 GPU / 不改任何别的文件。

------------------------------------------------------------------ 抠图逻辑是抄来的
cutout_rgb() 是 tools/review_gate.py 里 cutout() 的逐行搬运(灰底 210 / pad 12% / 居中)。
★ 改一处要改两处。没抽成公共模块是因为 review_gate.py 是**脚本不是模块** ——
  它模块级就读 os.environ["R1"] 并跑完整个出图流程, import 它会当场重新生成联络图。
  抽出来是对的, 但那要动一个已经在用的文件; prototype 阶段先抄, 证明有用再抽。

------------------------------------------------------------------ 用法
  PY=~/miniforge3/envs/sam3d/bin/python
  $PY tools/gate_ui.py --scene pedestrian            # 起服务, 默认 8765
  $PY tools/gate_ui.py --scene pedestrian --port 9000
  $PY tools/gate_ui.py --scene cyberpunk --indoor
  远程: ssh -L 8765:localhost:8765 <host>, 然后本地浏览器开 http://localhost:8765
"""
import os, sys, json, time, argparse, urllib.parse, threading
from collections import OrderedDict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np, cv2


def _is_empty(v):
    """"这个字段算没填吗"。

    ★ 不能写 `v in (None, "", [], False)` —— Python 里 `0 == False`, 于是 `0 in (..., False)`
      是 True, 值为 0 的字段会被悄悄丢掉。实测 office r1 因此崩过: 界面追加的三行
      (computer_monitor_0 / desk_chair_0 / door_0) 全是**实例 0**, 写盘时 实例 字段没了,
      下一轮 review_gate 在 `_wmap[...] = (r["词"], r["实例"])` 上 KeyError, 决策落不了地。
      判空必须按类型判, 不能靠真值。

    False 仍然算"没填": 只有 VLM放行 这类开关字段是布尔, 它的 False 本来就表示没勾。
    """
    return (v is None or v is False
            or (isinstance(v, (str, list, dict, tuple)) and len(v) == 0))
from pycocotools import mask as mask_util

HOME  = os.path.expanduser("~")
SAM3D = f"{HOME}/sam3d"
MYIMG = f"{SAM3D}/sam-3d-objects/my_images"
GEN_OUT = f"{SAM3D}/sam-3d-objects/outputs"     # assemble_scene.py 的 OUTS, 生成好的 glb 在这
# 组装的绝对尺寸守卫, 抄 assemble_scene.py:28-29(流水线不传, 用的就是这两个默认值)。
# 界面只拿它预告"这个会被组装扔掉", 不参与写入。
ASM_MIN_H = float(os.environ.get("MIN_H", "0.15"))
ASM_MAX_D = float(os.environ.get("MAX_D", "15"))
sys.path.insert(0, f"{SAM3D}/data/outdoor/scripts")
from solo_gate import DEFAULT_MIN_PX          # noqa: E402  逐实例门槛(像素)的唯一定义
import class_sample                            # noqa: E402  整类样本候选规则的唯一定义

# 决定的五种取值 + 颜色。★ 和 review_gate.py 的 BORDER 表同源(那边是 BGR, 这里是 CSS)。
DECISIONS = ["keep", "drop", "drop-word", "solo", "shared"]
DEC_COLOR = {"keep": "#5ab45a", "drop": "#f0963c", "drop-word": "#dc3c3c",
             "solo": "#3cb4dc", "shared": "#c878c8"}


# ------------------------------------------------------------------ 抠图
def _gamma(img, m):
    """暗抠图的提亮 gamma。按 **mask 内**像素的灰度中位数 med 定: med >= 110 一点不动(1.0);
    再暗就把 med 拉到 128 —— g = log(0.5) / log(med/255), 夹在 [0.4, 1.0]。
    ★ 特写和定位图必须用同一个 g, 两张图亮度才一致。"""
    b = m > 127
    if b.sum() < 30:
        return 1.0
    med = float(np.median(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[b]))
    if med >= 110:
        return 1.0
    return float(np.clip(np.log(0.5) / np.log(max(med, 1.0) / 255.0), 0.4, 1.0))


def _apply_gamma(img, g):
    """g < 1 提亮。返回新数组 —— _frame() 缓存里的那份不许改。"""
    if g >= 0.999:
        return img
    lut = (np.clip((np.arange(256) / 255.0) ** g, 0, 1) * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def cutout_rgb(img, m, size, gamma=True):
    """灰底特写: 只留 mask 里的像素, 裁到外接框再留一圈边, 缩到 size 见方居中。

    ★ 逐行抄自 review_gate.py 的 cutout(), 只把"读两个文件"改成"收两个数组" ——
      原版为了复用把数组写进 /tmp 再读回来, 一个请求两次磁盘往返, 界面上翻候选会很卡。
    """
    if img is None or m is None:
        return None
    if m.shape != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    b = m > 127
    if b.sum() < 30:
        return None
    src = _apply_gamma(img, _gamma(img, m)) if gamma else img
    out = np.full_like(img, 210)                      # 灰底
    out[b] = src[b]
    v, u = np.nonzero(b)
    pad = int(0.12 * max(u.max() - u.min(), v.max() - v.min())) + 8
    y0, y1 = max(v.min() - pad, 0), min(v.max() + pad, img.shape[0])
    x0, x1 = max(u.min() - pad, 0), min(u.max() + pad, img.shape[1])
    c = out[y0:y1, x0:x1]
    if c.size == 0:
        return None
    s = size / max(c.shape[0], c.shape[1])
    c = cv2.resize(c, (max(int(c.shape[1] * s), 1), max(int(c.shape[0] * s), 1)))
    canvas = np.full((size, size, 3), 210, np.uint8)
    oy, ox = (size - c.shape[0]) // 2, (size - c.shape[1]) // 2
    canvas[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return canvas


def locate_rgb(img, m, size=420, g=None):
    """整帧定位图 —— 物体在画面哪儿、挨着什么。mask 外压暗到 0.45, 外接框(留 3px 边)2px 红框,
    mask 轮廓 1px 红线, 长边缩到 size。红线在**缩完之后**画, 否则 1920 缩到 420 时 2px 线只剩半个像素。
    g 是提亮 gamma, 和 cutout_rgb 用同一个(None = 自己按 mask 算)。"""
    if img is None or m is None:
        return None
    if m.shape != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    b = m > 127
    if b.sum() < 30:
        return None
    src = _apply_gamma(img, _gamma(img, m) if g is None else g)
    out = (src * 0.45).astype(np.uint8)
    out[b] = src[b]
    s = size / max(out.shape[0], out.shape[1])
    sm = cv2.resize(out, (max(int(out.shape[1] * s), 1), max(int(out.shape[0] * s), 1)),
                    interpolation=cv2.INTER_AREA)
    bs = cv2.resize(b.astype(np.uint8), (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST)
    v, u = np.nonzero(bs)
    if len(v):
        cv2.rectangle(sm, (int(u.min()) - 3, int(v.min()) - 3), (int(u.max()) + 3, int(v.max()) + 3),
                      (0, 0, 230), 2)
        cs, _ = cv2.findContours(bs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(sm, cs, -1, (0, 0, 230), 1)
    return sm


# ------------------------------------------------------------------ 数据
class Store:
    """管一个场景一轮次的全部读写。所有磁盘访问都走这里, handler 只做 HTTP。"""

    def __init__(self, scene, run, indoor=False, stage=None, solo_min_px=None):
        self.scene, self.run, self.indoor = scene, run, indoor
        self.solo_min_px_arg = solo_min_px      # 换轮次时照原样传给新的 Store
        self._solo_min_px = solo_min_px   # 逐实例门槛(像素), None = 问 pipeline, 见 solo_min_px()
        if indoor:
            self.R1 = f"{SAM3D}/data/indoorOutput/{scene}/{run}"
            self.FR = self._indoor_frames()
        else:
            self.R1 = f"{SAM3D}/data/outdoor/result/{scene}/{run}"
            self.FR = f"{SAM3D}/data/outdoor/dataset/{scene}/frames"
        self.MASKS = f"{self.R1}/03_masks"
        # 阶段: pre = 闸 2(生成前), post = 闸 4(召回之后)。
        # ★ 显式 > 猜。review_gate.py 的注释记着这个坑: 靠"盘上有没有 14_recall_ready.json"
        #   猜阶段, 会在重跑「生成前」那一段时把召回后的结果覆盖掉。这里默认取**已存在的
        #   最新那份**, 并且把读的是哪个文件明明白白显示在页面顶栏上, 让人自己看得见。
        self.stage_arg = stage          # 人显式指定的(--stage), 有就一直听它的
        self.refresh_stage()
        self._mk = {}                 # word -> (npz mtime, (meta, counts, H, W))  npz 变了就重读
        self._insts = {}              # (word, 门槛, 源文件 mtime...) -> [实例元]
        self._dedup = None            # 03_dedup_words.json 的 淘汰实例, 懒加载
        self._vrej = None             # 04_picks/vlm_verify_inst.json 的 否决, 懒加载
        self._disk_samples = None     # 04_picks/picks.json, 懒加载
        # 帧 LRU: 展开一个类要抠 15 张图, 常常是 15 个不同的帧, 瓶颈是 cv2.imread 一张
        # 1080p JPEG。命中之后一张缩略图只剩 decode+resize。1080p BGR 约 6MB/张,
        # 24 张 ≈ 150MB。★ 必须加锁 —— ThreadingHTTPServer 是多线程的。
        self._frames = OrderedDict()
        self._frame_cap = 24
        self._flock = threading.Lock()
        self._lock = threading.Lock()
        self._cfg = None              # pipeline 的 --dump-cfg, 懒加载
        self.tag = None               # 决策文件的「场景」字段, load() 时填
        self.tag_ok = False           # 表里的生成名前缀对不对得上 tag

    def pipeline_cfg(self):
        """问 pipeline 要 SOLO / SOLO_KEEP —— 界面要显示"你不点的话默认是哪种"。

        ★ 别自己抄一份默认值。SOLO 住在 outdoor_pipeline.py 的 CFG 里, 还能被 SCENES
          按场景覆盖(beach/carpark 是 "*"), 抄一份就会漂移。drive.py 也是这么问的。
        """
        if self._cfg is not None:
            return self._cfg
        import subprocess
        pipe = (f"{SAM3D}/pipeline/indoor_pipeline.py" if self.indoor
                else f"{SAM3D}/pipeline/outdoor_pipeline.py")
        cmd = [sys.executable, pipe, "--scene", self.scene, "--run", self.run, "--dump-cfg"]
        if self.indoor:
            cmd += ["--chain", "obj"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            self._cfg = json.loads(r.stdout) if r.returncode == 0 else {}
        except Exception:
            self._cfg = {}
        return self._cfg

    def refresh_stage(self):
        """现在是闸 2 还是闸 4, 每次请求都按盘上的文件重判一次。

        ★ 为什么不能只在启服务时判一次: 服务常常是在闸 2 打开的, 人审完往下跑, 召回那段跑完
          盘上才出现 05_review_afterrecall.json。定死的话页面刷新多少次都还是闸 2 那份候选,
          而且它是**静默**的 —— 人以为自己在审召回后的结果, 其实在审召回前的。实测踩过。
        """
        self.stage = self.stage_arg or ("post" if os.path.exists(f"{self.R1}/05_review_afterrecall.json")
                                        else "pre")
        sfx = "_afterrecall" if self.stage == "post" else ""
        self.DEC = f"{self.R1}/05_review{sfx}.json"

    def solo_min_px(self):
        """逐实例生成的面积门槛(像素)。只用来在面板上标"低于门槛, 不会单独生成", 不参与写入。
        ★ 问 pipeline 要, 不自己抄 —— 以前这里读 SOLO_TH 环境变量、没有就用 0.5,
          而 beach/carpark 在 pipeline 里配的是 0.03, 界面不带参数启动就标错一大片。"""
        if self._solo_min_px is None:
            self._solo_min_px = int(self.pipeline_cfg().get("SOLO_MIN_PX", DEFAULT_MIN_PX))
        return self._solo_min_px

    def boxes(self):
        """组装会读的那份盒子: {词: {实例号: 尺寸}}, 以及文件名。还没量过盒子就是 (None, None)。
        选哪份照抄 outdoor_pipeline._boxes_arg: 合并版在且不比基础版旧就用合并版。"""
        m, b = f"{self.R1}/06_all_boxes_merged.json", f"{self.R1}/06_all_boxes.json"
        p = m if (os.path.exists(m) and not (os.path.exists(b)
                                             and os.path.getmtime(m) < os.path.getmtime(b))) else b
        if not os.path.exists(p):
            return None, None
        with open(p) as f:
            d = json.load(f)
        return ({w: {int(x["obj_id"]): x["尺寸"] for x in v} for w, v in d.items()},
                os.path.basename(p))

    def disk_samples(self):
        """盘上每个共用词**实际在用**的样本: picks.json(挑图 + VLM 复核 + 人工指定之后的结果)。

        ★ 前端不能拿"挑图分最高"当成"机器选的": verify_picks 会让 VLM 在前几名里重选
          (timesquare r1 的 bollard/planter/traffic light 都不是第一名), 源码表还会强行指定(table)。
          盘上是谁就是谁; 只有人改了会影响挑图的决定(排除 与 挑的时候不同)时, 才轮到前端去预告。
        """
        p = f"{self.R1}/04_picks/picks.json"
        def parse(d):
            return {w: dict(oid=int(v["obj_id"]), fr=int(v["frame"]),
                            排除=v.get(class_sample.BANNED_KEY), 人工=bool(v.get("人工指定")))
                    for w, v in d.items() if isinstance(v, dict) and "obj_id" in v and "frame" in v}
        return self._json_by_mtime("_disk_samples", p, parse) or {}

    def mesh_facts(self, words):
        """每个词"组装时会拿哪个 mesh"要用到的盘上事实。规则本身在前端(选网格), 因为它要跟着
        人没保存的点选实时变; 这里只给前端自己看不到的东西。

        ★ 为什么要有: 以前一个实例没过逐实例门槛 / 被 VLM 否掉, 组装会按名字排序借同词第一个
          mesh(assemble_scene.pick_mesh), 界面上只有一行灰字 "below solo threshold",
          不说后果 —— pedestrian r2 的 car #2/#6/#10 全借了 car_0, 场景里 4 辆一模一样的车,
          人在闸口完全看不出来。
        """
        tag = self.tag or ""
        try:
            avail = {d for d in os.listdir(GEN_OUT)
                     if d.startswith(tag + "_") and os.path.exists(f"{GEN_OUT}/{d}/{d}.glb")}
        except OSError:
            avail = set()
        bx, bname = self.boxes()
        out = {}
        for w in words:
            base = f"{tag}_{w.replace(' ', '_')}"
            mine = sorted(int(d[len(base) + 1:]) for d in avail
                          if d.startswith(base + "_") and d[len(base) + 1:].isdigit())
            ins = self.insts(w, self.solo_min_px())
            out[w] = dict(
                类网格=base in avail,
                盘上实例=mine,
                盒子=None if bx is None else {str(o): e for o, e in (bx.get(w) or {}).items()},
                实例=[dict(oid=r["oid"], 低于门槛=r["低于门槛"], VLM否决=r["VLM否决"],
                          去重淘汰=r["去重淘汰"], 挑图分=r["挑图分"], 挑图帧=r["挑图帧"])
                      for r in ins],
            )
        return out, bname

    def solo_words(self):
        """(SOLO 原样, SOLO_KEEP 集合) —— 判"这个词不点的话默认逐个造还是共用"。"""
        c = self.pipeline_cfg()
        return (c.get("SOLO", ""), {w.strip() for w in c.get("SOLO_KEEP", "").split(",") if w.strip()})

    def _indoor_frames(self):
        o = f"{SAM3D}/data/indoorOutput/{self.scene}"
        for c in (f"{o}/frames_{self.run}", f"{o}/frames_bright", f"{o}/frames"):
            if os.path.isdir(c):
                return c
        return f"{o}/frames_{self.run}"

    # ---------- 决策文件 ----------
    def load(self):
        if not os.path.exists(self.DEC):
            raise SystemExit(
                f"[失败] 没有决策文件: {self.DEC}\n"
                f"       它是 review_gate.py 的产物 —— 先把流水线跑到闸 {2 if self.stage=='pre' else 4}。")
        with open(self.DEC) as f:
            d = json.load(f)
        # TAG 从「场景」字段来 —— review_gate.py:524 写的就是 场景=TAG。
        # ★ 校验表里的生成名前缀对不对得上。对不上说明这份文件不是这个 tag 产的,
        #   那么任何按 gen_name() 拼出来的名字都会在下游静默失配。宁可关掉功能也别静默。
        self.tag = d.get("场景") or ""
        rows = d.get("候选") or []
        self.tag_ok = bool(self.tag) and all(
            str(r.get("生成名", "")).startswith(self.tag + "_") for r in rows)
        return d

    # 追加行必须带齐的字段。少一个的后果不是难看, 是**下一轮 review_gate 当场崩** ——
    # review_gate.py:226 的 _wmap[r["生成名"]] = (r["词"], r["实例"]) 是直接索引不是 .get,
    # 那会把一个只读工具变成能弄挂流水线的东西, 直接违背这个文件的定位。
    NEED = ("生成名", "词", "实例", "决定")

    def save_rows(self, rows, delete=()):
        """覆盖「候选」这一段, 并支持**追加**由界面新建的行。

        ★ 整份重写会把 review_gate 下次合并旧决策用的「历史决定」抹掉。那份东西是
          "重复跑不会覆盖已改的决定"这条性质的依据, 丢了等于每次重跑都要重填一遍。
          所以这里从头到尾**一个字都不碰它**。

        ------------------------------------------------------------------ 为什么要追加
        一个 shared/默认 的词在表里只有一行(类行), 它代表的 15 个实例一行都没有。
        于是"这一类留着, 但第 7 个实例别摆"这句话**没有地方可写**。
        而下游其实早就通了 —— assemble_scene.py:276-278 同时按词级名和实例级名挡:
            if _gen_name(w) in SKIP_NAMES or _gen_name(w, b["obj_id"]) in SKIP_NAMES
        缺的只是上游从来没喂过实例级的名字。这个方法就是补上那一截。

        追加行下一轮会不会被认领, 取决于那个词是不是 shared/solo:
          * 是 -> review_gate.py:105-111 (solo_w |= KEEP) 会逐实例枚举它, 生成名撞上,
                  review_gate.py:225 的 old[生成名] 直接认领, 行归队。
          * 否 -> 行会从表上消失, 但**决定不丢**: review_gate.py:243-249 写进「历史决定」,
                  --apply 的 _gone (313-316) 再捞回来, 不生成 里照样有它。
        """
        with self._lock:
            d = self.load()
            by = {r["生成名"]: r for r in rows}
            old = {r["生成名"] for r in d["候选"]}
            for r in d["候选"]:
                n = by.get(r["生成名"])
                if not n:
                    continue
                r["决定"] = n.get("决定", "keep")
                # 三个可选字段: 给了就写, 给空就删 —— 留一个空值在文件里,
                # review_gate --apply 会把它当"填了"来解析。
                for k in ("改用", "改类别", "复用组", "VLM放行"):
                    if _is_empty(n.get(k)):
                        r.pop(k, None)
                    else:
                        r[k] = n[k]
            # ---- 删除: 只许删界面自己追加过的行 ----
            # review_gate 产的行一律不许删 —— 那是它下一轮重建表的依据。
            dele = {g for g in delete}
            if dele:
                d["候选"] = [r for r in d["候选"]
                             if not (r["生成名"] in dele and r.get("由界面追加"))]
            # ---- 追加 ----
            added = []
            for g, r in by.items():
                if g in old or g in dele:
                    continue
                missing = [k for k in self.NEED if r.get(k) in (None, "")]
                if missing:
                    raise ValueError(f"追加行 {g} 少字段 {missing} —— "
                                     f"少了它下一轮 review_gate 会 KeyError 崩掉")
                if not str(g).startswith((self.tag or "") + "_"):
                    raise ValueError(f"追加行 {g} 的前缀和 TAG {self.tag!r} 对不上 —— "
                                     f"名字错一个字, assemble 的 SKIP_NAMES 就静默失配")
                if r.get("复用组") and r.get("决定") == "drop":
                    raise ValueError(f"{g} 同时填了 复用组 和 drop —— 落地后实例会被搬进 "
                                     f"「母词~N」, drop 查的还是老名字, 那个物体照样摆出来。"
                                     f"先只填分组跑一轮。")
                row = {k: r[k] for k in ("生成名", "词", "来源", "实例", "帧",
                                         "面积占比", "决定", "改用", "复用组", "改类别", "VLM放行")
                       if not _is_empty(r.get(k))}
                row["由界面追加"] = True
                added.append(row)
            if added:
                # 插在同词的类行之后, 让联络图和文件的顺序都读得通
                out = []
                pend = {}
                for a in added:
                    pend.setdefault(a["词"], []).append(a)
                for r in d["候选"]:
                    out.append(r)
                    if r["词"] in pend and not str(r.get("来源")) == "逐实例":
                        out += pend.pop(r["词"])
                for rest in pend.values():
                    out += rest
                d["候选"] = out
            for i, r in enumerate(d["候选"]):
                r["编号"] = i          # 编号只用于联络图标注, 重排安全
            tmp = self.DEC + ".tmp"
            with open(tmp, "w") as f:
                json.dump(d, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.DEC)     # 原子替换: 写一半崩了不会留个坏文件
            return dict(更新=len(rows), 新增=len(added), 删除=len(dele))

    # ---------- mask ----------
    def _npz(self, word):
        k = word.replace(" ", "_")
        p = f"{self.MASKS}/{k}.npz"
        mt = self._mt(p)
        if k not in self._mk or self._mk[k][0] != mt:
            if mt is None:
                self._mk[k] = (None, None)
            else:
                z = np.load(p, allow_pickle=True)
                H, W = z["hw"]
                self._mk[k] = (mt, (z["meta"], z["counts"], int(H), int(W)))
        return self._mk[k][1]

    def n_instances(self, word):
        """这个词**会进场景**的实例数。

        ★ 为什么要显示它: 一张"类"卡代表的是整个类 —— person 那张挑错了, 街上那些人
          会全长成同一件格子衬衫。而一张"逐实例"卡挑错只影响那一个。决策的分量差十几倍,
          而这个数字原来的联络图上也没有, 人只能靠对场景的印象猜。

        ★ 必须减掉**跨词去重淘汰**的。它们由同组另一个词代表, 下游根本不出盒子
          (boxes_from_masks.py:87-91 直接 continue), 摆进场景的不是它们。
          第一版数的是 mask 里的全部实例, 于是 car 显示 13 个, 而 #1/#3/#7 早被
          taxi/suv/person 代表走了, 真正会摆的只有 10 个 —— 数字比事实大 30%。
          人正是拿这个数在判"挑错了要错几份", 大了就等于在夸大后果。

        还是别当准数: boxes 那步之后还有点数门槛和三维去重会再滤一轮。当量级看。
        """
        z = self._npz(word)
        if not z:
            return 0
        dead = self._dead_insts().get(word, set())
        return len({int(r["oid"]) for r in z[0]
                    if r["area"] > 0 and int(r["oid"]) not in dead})

    def _frame(self, fi):
        """读一帧, 走 LRU。返回的是缓存里那个数组本身 —— **调用方不许改它**,
        cutout_rgb 只读不写, 满足这个约定。"""
        with self._flock:
            im = self._frames.get(fi)
            if im is not None:
                self._frames.move_to_end(fi)
                return im
        im = cv2.imread(f"{self.FR}/{fi:04d}.jpg")     # 读盘放在锁外, 别把别的线程堵住
        if im is None:
            return None
        with self._flock:
            self._frames[fi] = im
            self._frames.move_to_end(fi)
            while len(self._frames) > self._frame_cap:
                self._frames.popitem(last=False)
        return im

    # ★ 缓存必须跟着文件变。服务一开就是一整天, 中间流水线会重跑(verify 重验、track 重跟) ——
    #   按"启动时读一次"缓存, 界面就拿旧的否决名单预告, 说 trash can #2 "会生成自己的 mesh",
    #   而新一轮 verify 已经否掉它、场景里借的是 trash_can_0。所以键里带上文件的 mtime。
    @staticmethod
    def _mt(p):
        try:
            return os.path.getmtime(p)
        except OSError:
            return None

    def _have(self):
        """有深度的帧 —— pick_from_masks 只在这些帧里挑样本(HAVE)。"""
        d = f"{self.R1}/02_da3/depth"
        mt = self._mt(d)
        if getattr(self, "_havec", None) is None or self._havec[0] != mt:
            try:
                s = {int(f[:4]) for f in os.listdir(d) if f.endswith("_depth.npy")}
            except OSError:
                s = set()
            self._havec = (mt, s)
        return self._havec[1]

    def _json_by_mtime(self, attr, p, parse):
        mt = self._mt(p)
        hit = getattr(self, attr)
        if hit is None or hit[0] != mt:
            try:
                d = json.load(open(p)) if mt is not None else {}
                val = parse(d)
            except Exception:
                val = {}
            setattr(self, attr, (mt, val))
        return getattr(self, attr)[1]

    def _dead_insts(self):
        """跨词去重淘汰掉的实例 {词: {oid,...}}。这些实例下游根本不出盒子,
        摆在面板上要标出来, 否则人会对着一个永远不会进场景的东西做决定。"""
        return self._json_by_mtime(
            "_dedup", f"{self.R1}/03_dedup_words.json",
            lambda d: {w: {int(x) for x in v} for w, v in (d.get("淘汰实例") or {}).items()})

    def _vlm_rej(self):
        """verify_picks.py 逐实例那一遍否掉的实例 {词: {oid,...}}。

        ★ 和 _dead_insts 是**两个不同的通道**, 别混:
            去重淘汰 -> 同组另一个词代表它, 东西还在场景里, 只是换了个词
            VLM否决  -> gen_per_instance.py 直接 continue, 这份 mesh 根本不造
          后者摆在面板上必须标出来, 否则人会对着一个不会存在的东西按 keep。
        """
        return self._json_by_mtime(
            "_vrej", f"{self.R1}/04_picks/vlm_verify_inst.json",
            lambda d: {w: {int(x) for x in v} for w, v in (d.get("否决") or {}).items()})

    # ---------- 召回这一支 ----------
    def recall_info(self):
        """召回全量清单: 进表的和没进表的**一起**, 每条带结局。

        ★ 为什么要单开一条通道, 而不是从候选表里挑。
          闸口界面读的是候选表, 候选表里只有活下来的。而召回专挑主线漏掉的东西,
          漏掉的多半是被前景挡住的 —— 补抠 mask 的失败率天生就高。实测 pedestrian/r2
          是「找到 3 个, 活下来 0 个」, 于是这道叫"召回"的闸上, 召回找到了什么
          人一眼都看不见, 连被判死的原因都没有。
          14_recall_verdict.json 由 mask_recall.py 逐条写, 谁都不许悄悄消失。

        老轮次没有判词册(那时候三条丢弃分支只加计数器), 退回 13_recall.json 至少
        把"找到了什么"列出来, 结局标成 未知 —— **别假装知道**。
        """
        vp = f"{self.R1}/14_recall_verdict.json"
        rp = f"{self.R1}/13_recall.json"
        判词, 门槛, 来源 = [], {}, None
        try:
            if os.path.exists(vp):
                d = json.load(open(vp))
                判词 = d.get("判词") or []
                门槛 = d.get("门槛") or {}
                来源 = vp
            elif os.path.exists(rp):
                for a in (json.load(open(rp)).get("召回") or []):
                    判词.append(dict(obj_id=a.get("obj_id"), 词=a.get("词"),
                                     最佳帧=a.get("最佳帧"), 框2d=a.get("框2d"),
                                     见帧数=a.get("见帧数"), 结局="未知"))
                来源 = rp
        except Exception as e:
            return dict(有=False, 说明=f"读不了召回清单: {e}", 判词=[], 门槛={}, 文件=vp)
        n = {}
        for r in 判词:
            n[r.get("结局") or "未知"] = n.get(r.get("结局") or "未知", 0) + 1
        return dict(有=bool(判词), 判词=判词, 门槛=门槛, 统计=n, 文件=来源,
                    说明=None if 判词 else "这一轮没有召回产物(13_recall.json 都没有)")

    def recall_shot(self, oid, size=260):
        """召回实例在它最佳帧上的样子: 画出那个 2D 框, 外扩留上下文。

        ★ 不能只裁框内。召回的东西多半被前景挡着, 只给框内一块色斑, 人判不了
          "这到底是不是一辆 taxi" —— 得看见挡它的那棵树。外扩比例跟着框走,
          小框扩得多, 大框扩得少, 否则小目标裁出来还是一块色斑。
        """
        for r in self.recall_info()["判词"]:
            if int(r.get("obj_id", -1)) != int(oid):
                continue
            fi = int(r.get("最佳帧", 0))
            im = self._frame(fi)
            if im is None or not r.get("框2d"):
                return None
            H, W = im.shape[:2]
            x0, y0, x1, y1 = [int(v) for v in r["框2d"]]
            pad = int(max(40, 0.6 * max(x1 - x0, y1 - y0)))
            X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
            X1, Y1 = min(W, x1 + pad), min(H, y1 + pad)
            out = im[Y0:Y1, X0:X1].copy()
            cv2.rectangle(out, (x0 - X0, y0 - Y0), (x1 - X0, y1 - Y0), (0, 215, 255), 3)
            h = max(out.shape[0], 1)
            sc = float(size) / h
            return cv2.resize(out, (max(1, int(out.shape[1] * sc)), int(size)))
        return None

    def insts(self, word, solo_min_px=DEFAULT_MIN_PX):
        """一个词的全部实例 + 每个实例的量。**这是展开面板的数据源。**

        ------------------------------------------------------------------ 量的选取
        闸 2 排在 boxes 之前, 所以一个三维量都没有 —— 只有 03_masks 里的
        (oid, frame, area, edge) 和 RLE。但这已经够回答那个最要紧的问题了:
        **这一类该 shared 还是该拆 solo。**

        代表帧的打分 ★必须逐字抄 review_gate.py:131 / gen_per_instance.py:32:
            score = area x (1 - min(edge*4, 0.9))
        抄错的后果不是数字难看, 是**界面上显示的那张图和真正会喂给 SAM3D 的不是同一张**,
        人在看 A 判 B。

        中位面积而不是单帧最大 —— 抄 dedup_words.py:185-192 的教训:
        游走的垃圾 mask 单帧最大值会赢过真车。所以:
          * 排序、分量  用 中位面积
          * 抖动 = 最大/中位, 正是那种垃圾 mask 的指纹, 拿来当质量警报

        ar / fill 只对代表帧那一条 RLE 调 toBbox —— RLE 上是 O(1), 不解码位图
        (dedup_words.py:129-131 专门强调过 1080x1920 解一张要 2MB)。
        这两个量合起来能把**两件现在完全看不见的事**分开:
          * ar 离群 + fill 低    -> mask 坏了, 该换候选或 drop
          * ar 分成几族 fill 正常 -> 真的形状多样, 该 solo
        """
        key = (word, int(solo_min_px),
               self._mt(f"{self.MASKS}/{word.replace(' ', '_')}.npz"),
               self._mt(f"{self.R1}/03_dedup_words.json"),
               self._mt(f"{self.R1}/04_picks/vlm_verify_inst.json"),
               self._mt(f"{self.R1}/02_da3/depth"))
        if key in self._insts:
            return self._insts[key]
        z = self._npz(word)
        if not z:
            return []
        meta, counts, H, W = z
        px = float(H * W)
        dead = self._dead_insts().get(word, set())
        rej  = self._vlm_rej().get(word, set())
        by = {}
        for i, r in enumerate(meta):
            if r["area"] <= 0:
                continue
            by.setdefault(int(r["oid"]), []).append(
                (i, int(r["frame"]), int(r["area"]), float(r["edge"])))
        out = []
        have = self._have()
        cfg = self.pipeline_cfg()
        min_fr = int(cfg.get("MIN_FRAMES", 2))
        sat = float(cfg.get("FRAME_SAT", 10))
        # 挑图分: 和 pick / verify 同一份规则(class_sample.ranked) —— 只看有深度的帧, 活不满
        # MIN_FRAMES 不参选, 分 = 面积 x (1 - 贴边惩罚) x min(帧数/FRAME_SAT, 1)。
        # 这里不传 banned/dead: 挡谁要跟着人没保存的点选实时变, 前端 自动样本 自己挡。
        pk = {c[1]: (c[0], c[3]) for c in
              class_sample.ranked(meta, have, word, min_frames=min_fr, frame_sat=sat)[0]}
        for oid, rows in by.items():
            pscore, pf = pk.get(oid, (0.0, None))
            areas = sorted(a for _, _, a, _ in rows)
            med_a = float(areas[len(areas) // 2])
            max_a = float(areas[-1])
            edges = sorted(e for _, _, _, e in rows)
            med_e = float(edges[len(edges) // 2])
            # 代表帧: 和 review_gate.py:131 同一个公式
            bi, bf, ba, be = max(rows, key=lambda t: t[2] * (1 - min(t[3] * 4, 0.9)))
            x, y, bw, bh = [float(v) for v in mask_util.toBbox(
                {"counts": counts[bi], "size": [H, W]})]
            nf = len(rows)
            out.append(dict(
                oid=oid,
                生成名=self.gen_name(word, oid),
                代表帧=bf,
                中位面积pct=round(100.0 * med_a / px, 3),
                最大面积pct=round(100.0 * max_a / px, 3),
                抖动=round(max_a / max(med_a, 1.0), 2),
                贴边中位=round(med_e, 3),
                活帧数=nf,
                ar=round(bw / max(bh, 1.0), 2),
                fill=round(ba / max(bw * bh, 1.0), 3),
                # 样本分: 抄 dedup_words.py:196-198, 归一化所以跨词可比
                样本分=round(100.0 * med_a / px * (1 - min(med_e * 4, 0.9))
                             * min(nf / 10.0, 1.0), 4),
                # 比代表帧(= gen_per_instance 真正拿去生成的那帧)的像素, 不比最大面积 ——
                # 比的量不一样, 面板说"会生成"而那边其实不生成, 就又是一处对不上。
                低于门槛=ba < int(solo_min_px),
                去重淘汰=(oid in dead),
                VLM否决=(oid in rej),
                挑图分=round(float(pscore), 1),
                挑图帧=pf,
            ))
        out.sort(key=lambda r: -r["中位面积pct"])
        self._insts[key] = out
        return out

    def gen_name(self, word, oid=None):
        """★ 生成名只有这一处拼装。逐字对齐三处已有公式:
             review_gate.py:144      候选表的行
             gen_per_instance.py:53  真正生成 mesh 的目录名
             assemble_scene.py:49-52 组装时查"人工否掉"的键
           拼错一个字的后果是静默的: assemble 的 SKIP_NAMES 匹配不上, drop 白填。
        """
        tag = self.tag or ""
        base = f"{tag}_{word.replace(' ', '_')}"
        return base if oid is None else f"{base}_{oid}"

    def candidates(self, word, per_inst=6, must=None, only_oid=None):
        """一个词的全部可选候选 (实例 x 帧), 每个实例采样 per_inst 帧。

        must = [(oid, frame), ...] 是"现在正用着的那张"(和这个实例分最高的那张), **保证入选**, 见下面条 0。

        ---------------------------------------------------------------- 采样策略
        为什么要采样: taxi 的 mask 里有 117 条 (实例,帧) 记录, 全铺开既慢又没用 ——
        相邻帧长得几乎一模一样, 翻 90 张只是在看同一张图晃动。

        现在的策略(四条, 按优先级):
          0. 当前正用着的那张一定入选, 压过后面所有过滤。
             ★ 这条是实测补的: taxi 现在用的是 #0/f87, 而它 edge=0.411 会被条 3 滤掉,
               于是"我现在用的是哪张"在界面上直接消失, 人既看不到现状也退不回去。
               任何过滤都不该把现状滤没 —— 这和 review_gate.py 那条"决定自我抹除"是
               同一类病: 一个视图把自己的输入过滤掉了。
          1. 面积最大的那一帧入选 —— 抠得最全的通常就是它, 这是 pick 那步的主判据。
          2. 其余按**帧号均匀铺开**, 不按面积排。面积前 6 名往往是连续 6 帧, 等于没采样;
             帧号铺开才能看到不同角度、不同遮挡。
          3. 贴边(edge)超过 0.35 的不要 —— 物体一半在画面外, 生成必然缺一块。
             但如果滤完一个都不剩, 就把这条放掉(总比一个候选都给不出强)。
        """
        z = self._npz(word)
        if not z:
            return []
        meta, counts, H, W = z
        px = float(H * W)
        by = {}
        for i, r in enumerate(meta):
            if r["area"] <= 0:
                continue
            by.setdefault(int(r["oid"]), []).append(
                dict(oid=int(r["oid"]), frame=int(r["frame"]),
                     area=int(r["area"]), edge=round(float(r["edge"]), 3),
                     pct=round(100.0 * int(r["area"]) / px, 3)))
        # ★ 逐实例候选只许在**自己那个实例**里换帧, 不许跨实例。
        #   不是界面洁癖, 是 overrides_inst() 的键就是 (词, 实例号):
        #       out[(w, oid)] = fi        # review_decisions.py
        #   所以给 car#2 填 "改用": [5, 56], 落地会变成 out[("car",5)]=56 ——
        #   override 跑到 car#5 头上, car#2 一点没变, **而且不报错**。
        #   类候选相反, 跨实例换正是本意(overrides() 取 {词: (实例号, 帧号)} 两个都用),
        #   所以这个限制只对 only_oid 不为空时生效。
        if only_oid is not None:
            by = {int(only_oid): by.get(int(only_oid), [])}
        out = []
        for oid in sorted(by):
            rows = by[oid]
            if not rows:
                continue
            good = [r for r in rows if r["edge"] <= 0.35] or rows      # 条 3(带兜底)
            good.sort(key=lambda r: -r["area"])
            picked = [good[0]]                                          # 条 1
            rest = sorted((r for r in good[1:]), key=lambda r: r["frame"])
            if rest and per_inst > 1:                                   # 条 2
                step = max(len(rest) / float(per_inst - 1), 1.0)
                seen = {picked[0]["frame"]}
                for j in range(per_inst - 1):
                    r = rest[min(int(j * step), len(rest) - 1)]
                    if r["frame"] not in seen:
                        seen.add(r["frame"])
                        picked.append(r)
            # 条 0: 当前那张(和这个实例分最高的那张)不在采样结果里就补进来, 标记出来让前端能说明它为什么在
            for mo, mf in (must or []):
                if int(mo) == oid and not any(r["frame"] == int(mf) for r in picked):
                    cur = next((r for r in rows if r["frame"] == int(mf)), None)
                    if cur:
                        picked.append(dict(cur, 当前=True))
            picked.sort(key=lambda r: r["frame"])
            for r in picked:
                r["nframes"] = len(rows)
            out += picked
        return out

    def _mask(self, word, oid, frame):
        """这一帧这个实例的 mask(uint8 0/255), 没有就 None。cutout / locate 共用, 同一帧同一 mask。"""
        z = self._npz(word)
        if not z:
            return None
        meta, counts, H, W = z
        hit = [i for i, r in enumerate(meta)
               if int(r["oid"]) == oid and int(r["frame"]) == frame and r["area"] > 0]
        if not hit:
            return None
        return mask_util.decode({"counts": counts[hit[0]], "size": [H, W]}).astype(np.uint8) * 255

    def cutout(self, word, oid, frame, size=200):
        """现场抠一张。先看 my_images 里有没有写好盘的(那是 pick 选中的那张), 没有就从 mask 抠。

        ★ 判据是"盘上有没有这个文件", 不是来源标签 —— 抄的是 review_gate.py 那条注释,
          它踩过: 人把标签改成"整词复用"之后实例目录并不存在, 整张表变成 mask empty。
        """
        m = self._mask(word, oid, frame)
        return None if m is None else cutout_rgb(self._frame(frame), m, size)      # 走 LRU, 不直接 imread

    def locate(self, word, oid, frame, size=420):
        """整帧定位图, 和 cutout 同一帧同一 mask。"""
        m = self._mask(word, oid, frame)
        return None if m is None else locate_rgb(self._frame(frame), m, size)

    # ================================================================ 闸 3: 尺度
    # ★ 下面这一段是 world_scale.py:110-152 的**逐字搬运**。搬而不是读它的输出, 是因为
    #   world_scale.json 的「逐类」只含**通过了筛选的**词 —— 被 MIN_REAL 滤掉的
    #   (crosswalk 0.1 / manhole cover 0.3) 和实例不够的根本不在里面, 而人恰恰需要看到
    #   "这个词为什么当不了尺子", 以及"把它的参照高度改对之后能不能当"。
    #
    #   搬公式的代价是会漂。所以配一条自守的不变量(见 scale_state 的 校验):
    #   **没有任何编辑时, 这里算出的 k / 一致率 必须等于盘上 world_scale.json 里的值。**
    #   对不上就说明抄漂了, 当场暴露 —— 而不是等到"界面说达标、重跑却拒绝"。

    SCALE_DEFAULTS = dict(MIN_INST=2, MIN_REAL=0.5, OUTLIER=2.0,
                          AGREE_BAND=1.5, AGREE_MIN=0.6)

    def scale_paths(self):
        """(报告文件, 是不是估计版)。★ 两个文件名的差别就是"施加了没有" ——
        world_scale.json = 已经乘进 depth/相机/点云; _estimate.json = 只算没施加。
        顶栏必须显示读的是哪一个, 不能让人猜。"""
        d = f"{self.R1}/02_da3"
        a, b = f"{d}/world_scale.json", f"{d}/world_scale_estimate.json"
        if os.path.exists(a):
            return a, False
        return b, True

    @staticmethod
    def _wq(v, w, q):
        """按实例数加权的分位数。逐字抄 world_scale.py:129-133 ——
        不加权是不对的: 19 辆车和 2 台叉车不该一人一票。"""
        o = np.argsort(v)
        v = np.asarray(v, float)[o]
        w = np.asarray(w, float)[o]
        return float(np.interp(q, (np.cumsum(w) - 0.5 * w) / w.sum(), v))

    def _base_word(self, w):
        """派生子类 umbrella~1 在常识高度表里没有, 回退到母词。抄 world_scale.py:112 的意图。"""
        return w.split("~", 1)[0]

    def scale_rows(self, sizes=None):
        """从源头(01_vlm_sizes.json + 06_all_boxes.json)算出完整的尺子表。

        返回每个**盒子里有的词**一行, 带上"能不能当尺子、为什么不能"。
        sizes 给了就用它算(前端预览用), 不给就读盘上的。
        """
        cfg = self.pipeline_cfg()
        MIN_POINTS = int(cfg.get("MIN_POINTS", 300))      # ★ 问流水线要, 别自己抄
        C = dict(self.SCALE_DEFAULTS, MIN_POINTS=MIN_POINTS)
        sp = f"{self.R1}/01_vlm_sizes.json"
        bp = f"{self.R1}/06_all_boxes.json"
        if not (os.path.exists(sp) and os.path.exists(bp)):
            return [], C, None
        REAL = json.load(open(sp)) if sizes is None else sizes
        boxes = json.load(open(bp))
        rows = []
        for w, lst in sorted(boxes.items()):
            real = REAL.get(w)
            if real is None:
                real = REAL.get(self._base_word(w))
            # 高度够门槛的实例(点数 + 非零高度), 和 world_scale.py:117 同一个条件
            hs = [b["尺寸"][2] for b in lst
                  if b.get("点数", 0) >= C["MIN_POINTS"] and b["尺寸"][2] > 1e-4]
            r = dict(词=w, 盒子数=len(lst), 实例数=len(hs), 参照=(None if real is None else float(real)))
            if hs:
                r["量到"] = round(float(np.median(hs)), 3)
            # 当不了尺子的三种原因, 分开报 —— 人要据此决定改哪个词的高度
            if real is None:
                r["用不了"] = "no reference height"
            elif float(real) < C["MIN_REAL"]:
                r["用不了"] = f"reference {real} m < {C['MIN_REAL']} m — too flat to be a vertical ruler"
            elif len(hs) < C["MIN_INST"]:
                r["用不了"] = (f"only {len(hs)} instance(s) over {C['MIN_POINTS']} points "
                              f"(need {C['MIN_INST']})")
            else:
                r["真实"] = float(real)
                r["倍数"] = round(float(real) / r["量到"], 4)   # ★ 存的是 round 后的值
            rows.append(r)
        return rows, C, REAL

    def scale_calc(self, rows, C):
        """离群剔除 + 加权中位数 + 一致率。逐字抄 world_scale.py:136-152。"""
        cand = [r for r in rows if "倍数" in r]
        for r in rows:
            r.pop("采用", None)
        if len(cand) < 2:
            return dict(k=None, 一致率=None, 离散=None, 采用类数=0, 候选类数=len(cand),
                        闸门=C["AGREE_MIN"], 一致率达标=False,
                        不可算=f"only {len(cand)} usable ruler(s), need 2")
        f_all = np.array([r["倍数"] for r in cand])
        n_all = np.array([r["实例数"] for r in cand], float)
        med0 = self._wq(f_all, n_all, 0.5)
        keep = np.abs(np.log(f_all / med0)) < np.log(C["OUTLIER"])
        if keep.sum() < 2:                      # 全被剔光就退回不剔
            keep = np.ones(len(cand), bool)
        for r, k in zip(cand, keep):
            r["采用"] = bool(k)
        kept = [r for r in cand if r["采用"]]
        fk = np.array([r["倍数"] for r in kept])
        nk = np.array([r["实例数"] for r in kept], float)
        k_hat = self._wq(fk, nk, 0.5)
        spread = float(self._wq(fk, nk, 0.9) / max(self._wq(fk, nk, 0.1), 1e-9))
        inband = (fk > k_hat / C["AGREE_BAND"]) & (fk < k_hat * C["AGREE_BAND"])
        agree = float(nk[inband].sum() / nk.sum())
        return dict(k=round(k_hat, 4), 一致率=round(agree, 3), 离散=round(spread, 3),
                    采用类数=len(kept), 候选类数=len(cand), 闸门=C["AGREE_MIN"],
                    一致率达标=bool(agree >= C["AGREE_MIN"]))

    def scale_state(self):
        """闸 3 的全部数据 + 那条自守的不变量。"""
        rows, C, REAL = self.scale_rows()
        calc = self.scale_calc(rows, C)
        rep_p, is_est = self.scale_paths()
        rep = json.load(open(rep_p)) if os.path.exists(rep_p) else None
        # ★ 不变量: 没有编辑时, 我算的必须等于盘上的 —— **但只在"还没施加"时成立**。
        #
        #   施加过的轮次复现不了报告, 而且这不是 bug: s_scale 施加之后会**重新量盒子**,
        #   06_all_boxes.json 里的高度已经是真米了, 再比一次自然得 1.0。
        #   实测 pedestrian/r1: 盘上 k=1.4523(施加前的历史记录), 从当前数据算是 0.9993 ——
        #   两个数都对, 只是量的不是同一件事。
        #   第一版把这条不变量无条件打开, 于是它在一个完全正常的轮次上报"公式抄漂了"。
        #   **不变量本身也会写错**, 所以它报警时第一件事是问"我的假设对不对"。
        校验 = None
        if is_est and rep and rep.get("k") is not None and calc.get("k") is not None:
            dk = abs(calc["k"] - float(rep["k"]))
            da = (abs(calc["一致率"] - float(rep["一致率"]))
                  if rep.get("一致率") is not None and calc["一致率"] is not None else 0.0)
            校验 = dict(对得上=bool(dk <= 1e-3 and da <= 2e-3),
                       盘上k=rep.get("k"), 算出k=calc["k"],
                       盘上一致率=rep.get("一致率"), 算出一致率=calc["一致率"])
        if 校验 is None and rep and not is_est:
            校验 = dict(不适用=True, 盘上k=rep.get("k"), 算出k=calc.get("k"),
                       说明="scale was already applied — the report is a record from before "
                            "it was applied, and the boxes on disk were re-measured after. "
                            "Recomputing from current data gives ~1.0, which is the correct "
                            "answer for 'is it still off?'")
        return dict(场景=self.tag, R1=self.R1, 闸=3,
                    报告文件=rep_p, 已施加=(not is_est) and bool(rep and rep.get("施加")),
                    报告存在=bool(rep), 拒绝原因=(rep or {}).get("拒绝原因"),
                    无需施加=(rep or {}).get("无需施加"),
                    尺子文件=f"{self.R1}/01_vlm_sizes.json",
                    参数=C, 逐类=rows, 汇总=calc, 校验=校验, 尺子=REAL or {})

    def scale_preview(self, sizes):
        """前端改了高度之后, 不重跑就算出 k 会变成多少。"""
        rows, C, _ = self.scale_rows(sizes)
        return dict(逐类=rows, 汇总=self.scale_calc(rows, C))

    def scale_save(self, changes):
        """把改过的参照高度写回 01_vlm_sizes.json。

        ★ 只改人动过的键, 其余原样 —— 和 save_rows 同一条原则: 局部覆盖, 绝不整份重写。
          这个文件是 vlm_sizes 那步的产物, 里面可能有界面没显示的词(盒子里没有的)。
        """
        p = f"{self.R1}/01_vlm_sizes.json"
        with self._lock:
            d = json.load(open(p))
            n = 0
            for w, v in (changes or {}).items():
                if v is None:                      # 撤销 = 删字段, 不是写一个等值的进去
                    if w in d:
                        del d[w]; n += 1
                    continue
                v = float(v)
                if v <= 0:
                    raise ValueError(f"{w}: height must be > 0, got {v}")
                if d.get(w) != v:
                    d[w] = v; n += 1
            tmp = p + ".tmp"
            with open(tmp, "w") as f:
                json.dump(d, f, ensure_ascii=False, indent=1)
            os.replace(tmp, p)
            self._cfg = None                       # 让下次读重新算
            return dict(改了=n, 文件=p)

    def cutout_pick(self, gen_name, word, oid, frame, size=200, disk=True):
        """卡片上那张图。

        disk=True  用 my_images 里写好盘的那张 —— 它才是真正会喂给 SAM3D 的图,
                   所以"人没换过"的时候显示它最诚实。
        disk=False 现场从 mask 抠 (oid, frame) 指定的那一张。

        ★ 这个开关是踩出来的。第一版**无条件**优先用盘上那张, 而 pick 那步早就把目录
          写好了(my_images/ped_r2_person/ 一直在), 于是人在界面上换了帧、URL 里的
          oid/frame 也跟着变了, 服务端却照旧发回同一张 —— 换了没反应。
          实测: frame=181/225/87 三个请求返回的 md5 完全相同。
          **参数变了结果不变**是最难查的一类 bug: 前端后端看着都对, 错在两者之间的约定。
          现在由前端说了算: 有「改用」就 disk=0, 没有才 disk=1。
        """
        if disk:
            d = f"{MYIMG}/{gen_name}"
            fp, mp = f"{d}/frame.png", f"{d}/mask.png"
            if os.path.exists(fp) and os.path.exists(mp):
                return cutout_rgb(cv2.imread(fp), cv2.imread(mp, 0), size)
        return self.cutout(word, oid, frame, size)


# ------------------------------------------------------------------ 轮次目录
# ★ 为什么要有: 服务起来的时候 --scene/--run 就定死了, 换一个轮次必须重启, 而一台机器上
#   常常同时有好几轮(pedestrian r2 / r2sim / r2sim_vggt ...)。现在地址里带 s/r/i 就能换,
#   /runs 是选择页。每个轮次一个 Store, 缓存起来 —— Store 里有 mask、缩略图那些缓存,
#   每次请求新建一个等于每翻一页重读一遍 npz。
_STORES = {}
_SLOCK = threading.Lock()


def run_dirs():
    """盘上有哪些轮次。判据是"有决策文件或者驱动状态" —— 只有这种目录才有东西可看。"""
    out = []
    for indoor, root in ((False, f"{SAM3D}/data/outdoor/result"), (True, f"{SAM3D}/data/indoorOutput")):
        for scene in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            sd = f"{root}/{scene}"
            for run in sorted(os.listdir(sd)) if os.path.isdir(sd) else []:
                R1 = f"{sd}/{run}"
                if not os.path.isdir(R1):
                    continue
                decs = [f for f in ("05_review_afterrecall.json", "05_review.json") if os.path.exists(f"{R1}/{f}")]
                state = f"{R1}/logs/.drive_state.json"
                if not decs and not os.path.exists(state):
                    continue
                info = dict(scene=scene, run=run, indoor=indoor, R1=R1, 候选=None, 闸=None, 几何=geom_of(R1),
                            阶段=("post" if decs and decs[0].endswith("afterrecall.json") else "pre" if decs else None),
                            决策文件=decs[0] if decs else None, 改过=None, 段=None, 停在闸口=None)
                if decs:
                    p = f"{R1}/{decs[0]}"
                    info["改过"] = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
                    try:
                        info["候选"] = len(json.load(open(p)).get("候选") or [])
                    except Exception:
                        info["候选"] = -1
                    info["闸"] = 2 if info["阶段"] == "pre" else 4
                try:
                    st = json.load(open(state))
                    info["段"], info["停在闸口"] = st.get("段"), st.get("停在闸口")
                except Exception:
                    pass
                out.append(info)
    return out


def geom_of(R1):
    """这一轮的深度和位姿是谁出的。写在 02_da3/summary.txt 里(VGGT 那条链会打 backend: 一行),
    没有那一行就是一直在用的 DA3 —— 光看轮次名字分不出来, 而这正是两轮唯一的区别。"""
    for d in ("02_da3", "02_DA3点云"):
        p = f"{R1}/{d}/summary.txt"
        if not os.path.exists(p):
            continue
        for ln in open(p, errors="ignore"):
            if ln.startswith("backend:"):
                t = ln.split(":", 1)[1].strip()
                if "VGGT" in t:
                    sc = "Metric3D L" if "metric3d_l" in t else "Metric3D G2" if "metric3d_g2" in t else "Metric3D"
                    return f"VGGT-Ω + {sc}"
                return t.split(";")[0].strip()
        return "DA3"
    return None


def get_store(scene, run, indoor, solo_min_px=None):
    k = (scene, run, bool(indoor))
    with _SLOCK:
        if k not in _STORES:
            _STORES[k] = Store(scene, run, indoor, None, solo_min_px)
        return _STORES[k]


RUNS_PAGE = """<!doctype html><meta charset=utf-8><title>Gate runs</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#faf9f5;--panel:#fff;--ink:#141413;--dim:#6b6a65;--line:#e3e1da;--hi:#1a6acb;--red:#d0000e}
 @media (prefers-color-scheme:dark){:root{--bg:#1b1b19;--panel:#232320;--ink:#eceae4;--dim:#9a988f;--line:#36352f;--hi:#7fb2f0}}
 body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,sans-serif;padding:24px 16px 40px}
 .wrap{max-width:860px;margin:0 auto}
 h1{font-size:19px;margin:0 0 4px} p.note{color:var(--dim);margin:0 0 18px}
 h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin:22px 0 8px;font-weight:600}
 .rows{display:flex;flex-direction:column;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--panel)}
 a.row{display:grid;grid-template-columns:1fr auto;gap:2px 14px;padding:11px 14px;border-top:1px solid var(--line);
       text-decoration:none;color:inherit}
 a.row:first-child{border-top:0} a.row:hover{background:color-mix(in srgb,var(--hi) 8%,transparent)}
 .nm{font-weight:600} .nm em{font-style:normal;color:var(--dim);font-weight:400}
 .meta{color:var(--dim);font-size:12.5px;font-variant-numeric:tabular-nums}
 .right{text-align:right;font-size:12.5px;color:var(--dim);font-variant-numeric:tabular-nums}
 .cur{color:var(--hi);font-weight:600}
 .geom{font-variant-numeric:normal}
 a.row{grid-template-areas:"nm when" "meta geom"}
 a.row .nm{grid-area:nm} a.row .right:not(.geom){grid-area:when}
 a.row .meta{grid-area:meta} a.row .geom{grid-area:geom}
 .none{color:var(--dim);padding:12px 14px}
</style>
<div class=wrap>
<h1>Pick a run</h1>
<p class=note>Every run on disk that has a decision file or driver state. Opening one only changes what this page shows —
nothing on disk is touched, and each browser tab keeps its own run.</p>
<!--ROWS-->
</div>"""


def runs_html(cur):
    rows = run_dirs()
    out = []
    for indoor, head in ((False, "Outdoor"), (True, "Indoor")):
        rs = [r for r in rows if r["indoor"] == indoor]
        out.append(f"<h2>{head}</h2>")
        if not rs:
            out.append('<div class="rows"><div class="none">none</div></div>')
            continue
        out.append('<div class="rows">')
        for r in rs:
            q = urllib.parse.urlencode(dict(s=r["scene"], r=r["run"], **({"i": 1} if r["indoor"] else {})))
            is_cur = (r["scene"], r["run"], r["indoor"]) == cur
            gate = f'gate {r["闸"]}' if r["闸"] else "no decision file yet"
            seg = "" if r["段"] is None else f' · driver at segment {r["段"] + 1}{" (waiting)" if r["停在闸口"] else ""}'
            cand = "" if r["候选"] is None else f'{r["候选"]} candidates · '
            out.append(f'<a class="row" href="/?{q}">'
                       f'<span class="nm">{r["scene"]} <em>/</em> {r["run"]}'
                       f'{" <span class=cur>· open now</span>" if is_cur else ""}</span>'
                       f'<span class="right">{r["改过"] or ""}</span>'
                       f'<span class="meta">{cand}{gate}{seg}</span>'
                       f'<span class="right geom">{r["几何"] or ""}</span></a>')
        out.append("</div>")
    # 用替换不用 % 格式化: 这页的 CSS 里有 color-mix(... 8%) 这种百分号, % 格式化会当成占位符炸掉
    return RUNS_PAGE.replace("<!--ROWS-->", "\n".join(out))


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass                       # 默认每个请求打一行, 翻候选时会把终端刷爆

        # ---- 小工具 ----
        def _send(self, code, ctype, body, cache=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if cache:
                self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, "application/json; charset=utf-8",
                       json.dumps(obj, ensure_ascii=False).encode())

        def _png(self, arr):
            if arr is None:
                # 占位图: 和 review_gate 一样画一块深灰 + "mask empty",
                # 别返回 404 —— 前端一片碎图标比一块写着原因的灰块难查得多
                arr = np.full((200, 200, 3), 60, np.uint8)
                cv2.putText(arr, "mask empty", (18, 105), 0, 0.55, (255, 255, 255), 1)
            ok, buf = cv2.imencode(".png", arr)
            self._send(200, "image/png", buf.tobytes(), cache=True)

        def _store(self, q):
            """这个请求看的是哪个轮次: 地址里带 s/r/i 就按它, 否则就是启动时那个。"""
            st = (store if not q.get("s") or not q.get("r")
                  else get_store(q["s"], q["r"], q.get("i") in ("1", "true"), store.solo_min_px_arg))
            st.refresh_stage()          # 跑完一段盘上会多出召回后那份, 每次请求重判一次
            return st

        # ---- 路由 ----
        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self._send(200, "text/html; charset=utf-8", PAGE.encode())
                if u.path == "/runs":
                    cur = (store.scene, store.run, store.indoor)   # 启动时那个, 标 open now
                    return self._send(200, "text/html; charset=utf-8", runs_html(cur).encode())
                if u.path == "/api/runs":
                    return self._json(run_dirs())
                st = self._store(q)              # 下面每一条都用这个轮次
                if u.path == "/api/state":
                    d = st.load()
                    rows = d["候选"]
                    # 每条补上"这个词有几个实例" —— 类卡靠它显示分量, 前端不该自己再去数
                    for r in rows:
                        r["实例数"] = st.n_instances(r["词"])
                    facts, bname = st.mesh_facts(sorted({r["词"] for r in rows
                                                            if int(r.get("实例") or 0) < 10000}))
                    return self._json(dict(
                        场景=d.get("场景"), TAG=st.tag,
                        # ★ 「场景」那个字段其实是 TAG(ped_r2)。命令行要的是
                        #   --scene pedestrian --run r2, 拿 TAG 去拼会拼出个不存在的场景。
                        场景名=st.scene, 轮次=st.run,
                        闸=2 if st.stage == "pre" else 4,
                        阶段=st.stage, 决策文件=st.DEC, R1=st.R1,
                        solo门槛px=st.solo_min_px(), 可追加行=st.tag_ok,
                        SOLO=st.solo_words()[0],
                        SOLO_KEEP=sorted(st.solo_words()[1]),
                        历史决定=d.get("历史决定") or {},
                        网格事实=facts, 盒子文件=bname,
                        盘上样本=st.disk_samples(),
                        配置指定样本=st.pipeline_cfg().get("配置指定样本") or {},
                        组装守卫=dict(MIN_H=ASM_MIN_H, MAX_D=ASM_MAX_D),
                        候选=rows))
                if u.path == "/img/word":
                    # 只给词, 服务端自己挑"最有代表性的那个实例的代表帧" ——
                    # 尺子表只有词名, 拿不到 oid/frame; 硬拼 oid=0/frame=0 有一半的词
                    # 根本没有那个组合, 整列缩略图是空的(实测 beach 23 个词空了 6 个)。
                    ins = st.insts(q["word"], st.solo_min_px())
                    if not ins:
                        return self._png(None)
                    # 和前端 自动样本 同一路: 去重淘汰的、VLM 否掉的不当缩略图, 剩下挑图分最高
                    ok = [r for r in ins if not r["去重淘汰"] and not r["VLM否决"] and r["挑图帧"] is not None]
                    hit = max(ok, key=lambda r: r["挑图分"]) if ok else ins[0]
                    if ok:
                        return self._png(st.cutout(
                            q["word"], hit["oid"], hit["挑图帧"], int(q.get("size", "80"))))
                    return self._png(st.cutout(
                        q["word"], hit["oid"], hit["代表帧"], int(q.get("size", "80"))))
                if u.path == "/api/recall":
                    return self._json(st.recall_info())
                if u.path == "/img/recall":
                    return self._png(st.recall_shot(
                        int(q["oid"]), int(q.get("size", "260"))))
                if u.path == "/api/scale":
                    return self._json(st.scale_state())
                if u.path == "/api/insts":
                    return self._json(st.insts(q["word"], st.solo_min_px()))
                if u.path == "/img/inst":
                    # 服务端自己挑代表帧, 前端不用先等 /api/insts 回来
                    ins = st.insts(q["word"], st.solo_min_px())
                    hit = next((r for r in ins if r["oid"] == int(q["oid"])), None)
                    if not hit:
                        return self._png(None)
                    # 人换过帧就给那一帧(前端带 frame 来), 否则代表帧
                    fr = int(q["frame"]) if q.get("frame") else hit["代表帧"]
                    return self._png(st.cutout(
                        q["word"], hit["oid"], fr, int(q.get("size", "120"))))
                if u.path == "/img/frame":
                    # 整帧定位图: 和 /img/inst 同一帧(前端带 frame 就那帧, 否则代表帧)
                    ins = st.insts(q["word"], st.solo_min_px())
                    hit = next((r for r in ins if r["oid"] == int(q["oid"])), None)
                    if not hit:
                        return self._png(None)
                    fr = int(q["frame"]) if q.get("frame") else hit["代表帧"]
                    return self._png(st.locate(q["word"], hit["oid"], fr, int(q.get("size", "420"))))
                if u.path == "/api/cands":
                    must = ([(int(q["oid"]), int(q["frame"]))]
                            if "oid" in q and "frame" in q else [])
                    # 这个实例分最高的那帧也保证入选 —— 抽屉上要标出 best, 采样把它漏了就标不出来
                    if q.get("best") and q.get("only_oid"):
                        must.append((int(q["only_oid"]), int(q["best"])))
                    return self._json(st.candidates(
                        q["word"], int(q.get("n", "6")), must,
                        q.get("only_oid")))          # 逐实例卡传它, 类卡不传
                if u.path == "/img/pick":
                    return self._png(st.cutout_pick(
                        q["gen"], q["word"], int(q["oid"]), int(q["frame"]),
                        int(q.get("size", "200")), q.get("disk", "1") != "0"))
                if u.path == "/img/cand":
                    return self._png(st.cutout(
                        q["word"], int(q["oid"]), int(q["frame"]),
                        int(q.get("size", "200"))))
                return self._json(dict(error="no such route: " + u.path), 404)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return self._json(dict(error=f"{type(e).__name__}: {e}"), 500)

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            try:
                st = self._store(q)
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                if u.path == "/api/scale/preview":
                    return self._json(st.scale_preview(body.get("尺子") or {}))
                # ★ 返回值别再叫 st —— 这里的 st 是这一轮的 Store, 覆盖掉之后下面取 st.DEC
                #   就成了在字典上取属性(AttributeError, 保存直接失败)。踩过一次。
                if u.path == "/api/scale/save":
                    res = st.scale_save(body.get("改动") or {})
                    print(f"[保存尺子] {res} ", flush=True)
                    return self._json(dict(ok=True, **res))
                if u.path == "/api/save":
                    res = st.save_rows(body.get("候选", []), body.get("删除", []))
                    print(f"[保存] {res} -> {st.DEC}", flush=True)
                    return self._json(dict(ok=True, **res, 文件=st.DEC))
                return self._json(dict(error="no such route: " + u.path), 404)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return self._json(dict(error=f"{type(e).__name__}: {e}"), 500)
    return H


# ------------------------------------------------------------------ 页面
PAGE = r"""<!doctype html>
<meta charset="utf-8"><title>Gate Review</title>
<style>
 /* ================================================================ 瑞士
    国际主义平面设计的三件套 —— 网格、无衬线、客观摄影 —— 正好是这一屏的三件事:
    候选是网格里的单元, 判定是排版(选中 = 实心黑块, 正片), 抠图是客观摄影。

    ★ 后端烤的抠图灰底是 210 = #d2d2d2 (cutout_rgb 里写死的; 上一版注释写成 #b4b4b4,
      是错的)。它放在白纸上只有 1.51:1, 并排的两张灰图之间隔 1px 网格线只剩 1.26:1 ——
      一排图会糊成一条灰带。所以图版的边界**不靠色差, 靠网格**: 每个单元白底,
      图版缩进一圈白边(裱框), 图和字共用同一条左边线; 单元之间走发丝线。 */
 :root{
   --paper:#efefed;   /* 纸: 中性灰白。刻意不用暖奶油色 */
   --sheet:#ffffff;   /* 单元 */
   --ink:#000000;     /* 正文: 真黑。瑞士排版不用 #111 那种调过的近黑 */
   --rule:#bdbcb6;    /* 发丝线 = 网格本身, 只落在真实的单元边界上 */
   --mute:#6b6a65;    /* 次要文字  白底 5.42:1 */
   --edge:#8a8983;    /* 控件描边  白底 3.51:1 (WCAG 1.4.11 要 3:1) */
   --field:#f4f4f2;   /* 未选控件的底 */
   --red:#d0000e;     /* 唯一强调色, **只说"不"**。白底 5.69:1, 白字压它 5.69:1 */
   --plate:#d2d2d2;   /* 抠图底, 后端定的 */
   --yellow:#ffd400;  /* 召回的身份色(用户点名要黄)。只做底, 上面压黑字 14:1 */
   /* 旧名字留着 —— 内联样式和 RC色() 还在读它们 */
   --bg:var(--paper); --card:var(--sheet); --line:var(--rule);
   --fg:var(--ink); --dim:var(--mute); --pencil:var(--mute);
   --keep:var(--ink); --drop:var(--red); --kill:var(--red);
   --solo:var(--ink); --share:var(--ink); --flag:var(--ink);
 }
 *{box-sizing:border-box}
 body{margin:0;background:var(--paper);color:var(--ink);
      font:13px/1.5 "Helvetica Neue",Helvetica,Arial,"Liberation Sans","Nimbus Sans",
           system-ui,"PingFang SC","Microsoft YaHei",sans-serif;
      -webkit-font-smoothing:antialiased}
 /* ★ 全页没有等宽字体。数字靠同一个字族的 tabular-nums 竖着对齐 —— 这才是瑞士做法。
    等宽只留给**可复制的 shell 命令**(那是代码, 不是数据), 见 .note code / .rcsum code。 */
 .hd span,.im em,.dim,.glyph,.rt td,.rt input,.sum .big,.rcrow .num,.cand div,
 #stat,.bg,.path{font-variant-numeric:tabular-nums}

 #runs{font-size:12px;color:var(--mute);text-decoration:none;border:1px solid var(--rule);
        border-radius:5px;padding:3px 7px;white-space:nowrap}
 #runs:hover{color:var(--ink);border-color:var(--ink)}
 /* ---------------------------------------------------------------- 顶栏 */
 header{position:sticky;top:0;z-index:9;background:var(--sheet);
        border-bottom:2px solid var(--ink);
        padding:12px 20px 11px;display:flex;gap:20px;align-items:baseline;flex-wrap:wrap}
 header b{font-size:18px;font-weight:700;letter-spacing:-.025em;line-height:1.1}
 .ghd{font-size:13px;letter-spacing:-.005em}
 #gate{font-weight:700}
 .ghd.dimly{color:var(--mute);font-size:12px}
 .sp{flex:1}
 .path{color:var(--mute);font-size:11.5px;white-space:nowrap}
 .path b{color:var(--ink);font-weight:700;font-size:11.5px;letter-spacing:0}

 /* ---------------------------------------------------------------- 控件
    默认: 白底 + 3.51:1 描边。选中: 实心黑(见文末, 必须放在所有容器规则之后)。 */
 button{background:var(--sheet);color:var(--ink);border:1px solid var(--edge);border-radius:0;
        padding:5px 10px;cursor:pointer;font:inherit;font-size:12px;line-height:1.3;
        letter-spacing:0;transition:background .08s linear}
 button:hover:not(:disabled){background:var(--field)}
 button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
 /* 禁用: 真实底色, 不用 opacity(会和父级相乘) */
 button:disabled{background:var(--field);border-color:#d6d5d0;color:#85847f;cursor:not-allowed}
 #save{padding:6px 20px;font-weight:700;border-color:var(--ink)}
 #save:not(:disabled){background:var(--ink);color:#fff}
 #save:hover:not(:disabled){background:#2a2a28}
 #save:disabled{background:var(--field);border-color:#d6d5d0;color:#85847f}
 /* 闸切换: 一组分段控件, 描边用墨色 —— 它是这一屏的"总开关" */
 .gsw{display:flex;border:1px solid var(--ink)}
 .gsw button{border:0;padding:4px 14px;font-size:12px;background:var(--sheet)}
 .gsw button+button{border-left:1px solid var(--ink)}
 /* 召回钮: 黄底黑字。黄是用户点名的召回身份色, 不和"红=否"混 */
 /* ★ #rcw 也挂着 .gsw, 所以 .gsw button (0,1,1) 会压过裸 .rcb (0,1,0) —— 黄底上不去。
    写成 .gsw button.rcb (0,2,1)。另外闸 3 上 #rcw 是空的, 带着 1px 框就是顶栏一个小黑点。 */
 #rcw{border:0}
 #rcw:empty{display:none}
 .gsw button.rcb,.gsw button.rcb:hover:not(:disabled){background:var(--yellow);
      border:1px solid var(--ink);color:var(--ink);font-weight:700}

 /* ---------------------------------------------------------------- 网格
    ★ 不用 "gap:1px + 底色当线" 的做法: 最后一排没填满时, 空格子会露出一大块底色。
      用经典的**边框折叠**: 每个单元自带 1px 框, 负边距把相邻两条压成一条。
      空格子露出的是纸, 不是灰块。 */
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:0;
       padding:6px 21px 36px 20px}
 .sec{grid-column:1/-1;display:flex;align-items:baseline;gap:12px;
      padding:22px 0 8px;margin-bottom:-1px;position:relative;z-index:1;
      border-bottom:1px solid var(--ink);font-size:13px;font-weight:700;letter-spacing:-.01em}
 .sec:first-child{padding-top:10px}
 .sec span{font-weight:400;color:var(--mute);font-size:12px;letter-spacing:0}

 /* ---- 词卡 ---- */
 .wcard{background:var(--sheet);border:1px solid var(--rule);margin:0 -1px -1px 0;
        display:flex;flex-direction:column;min-width:0}
 /* 图版: 缩进 10px 白边, 图和下面的字共用同一条左边线 */
 .wcard img{display:block;width:calc(100% - 20px);margin:10px 10px 0;aspect-ratio:1;
            background:var(--plate);object-fit:cover}
 /* 整词作废: 图压淡 + 单元顶上一道红。**字不压** —— 复核要读得出来 */
 .wcard.dead{box-shadow:inset 0 3px 0 var(--red)}
 .wcard.dead img{opacity:.3}
 .wcard.dead .hd b{color:var(--mute)}
 /* ★ 标题块用 grid 定死两行: 第一行 词 | 数量, 第二行 #id f帧。
    用 flex-wrap 的话, 词长的卡(street lamp)第一行会折, 那张卡就高一截,
    一排的按钮对不齐 —— 这一屏的动作是横扫一排连着敲 drop, 偏了就点错。 */
 .hd{display:grid;grid-template-columns:minmax(0,1fr) auto;column-gap:8px;row-gap:1px;
     align-items:baseline;padding:10px 10px 8px}
 .hd b{font-size:13.5px;font-weight:700;letter-spacing:-.015em;min-width:0;
       white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .hd span{grid-column:1/-1;color:var(--mute);font-size:11px}
 /* 两个按钮一组, 共用中间那条边 —— 分段控件 */
 .wcard .row{display:grid;grid-template-columns:1fr 1fr;padding:0 10px 6px}
 .wcard .row button{padding:5px 4px;font-size:12px;min-width:0}
 .wcard .row button+button{margin-left:-1px}
 .more{margin-top:auto;border-top:1px solid var(--rule);display:flex;flex-direction:column}
 .more button{border:0;background:transparent;text-align:left;padding:8px 10px;
              font-size:12px;color:var(--ink)}
 .more button+button{border-top:1px solid var(--rule)}
 .more button:hover:not(:disabled){background:var(--field)}
 .more button:disabled{background:transparent;color:#85847f}
 .warn{margin:0 10px 8px;padding:6px 8px;border-left:2px solid var(--red);
       background:#fdf2f2;color:var(--ink);font-size:11px;line-height:1.45}

 /* ---- 徽章。位置即类别; 实线 = 人定的, 虚线 = 系统猜的 ---- */
 .bg{font-style:normal;font-size:11px;white-space:nowrap}
 /* 对象数: 不做药丸 —— 就是一个数。多到该考虑 shared 时加粗 */
 .bg.cls{color:var(--mute)}
 .bg.cls.big{color:var(--ink);font-weight:700}
 .bg.rec{background:var(--yellow);color:var(--ink);padding:0 5px;font-weight:700;font-size:10.5px}

 /* ---------------------------------------------------------------- 实例面板 */
 .panel{grid-column:1/-1;background:var(--sheet);padding:14px}
 .panel h3{margin:0 0 12px;font-size:13px;font-weight:700;letter-spacing:-.01em}
 .panel h3 span{color:var(--mute);font-weight:400;margin-left:10px;font-size:12px;letter-spacing:0}
 .insts{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:0;padding-right:1px}
 .inst{position:relative;background:var(--sheet);border:1px solid var(--rule);
       margin:0 -1px -1px 0;display:flex;flex-direction:column;min-width:0}
 .inst img{display:block;width:calc(100% - 16px);margin:8px 8px 0;aspect-ratio:1;
           background:var(--plate);object-fit:cover}
 /* 两张图并排: 左特写(是什么), 右整帧定位(在哪、挨着什么)。定位图点开是原分辨率的大图。 */
 .inst .pics{display:grid;grid-template-columns:120px 1fr;gap:8px;margin:8px 8px 0;align-items:start}
 .inst .pics img{width:100%;margin:0}
 .inst .pics img.loc{aspect-ratio:16/9;object-fit:contain;background:var(--plate)}
 .inst.flash{box-shadow:inset 0 0 0 3px var(--yellow);z-index:2}
 /* 质量可疑只挂黄徽章, 图保持彩色。★ 原来还降饱和, 但警报是自动规则会误报
    (pedestrian r2 的 car #0 镜头走近面积涨 5 倍, 好图也被变灰), 而且和"被判掉"的灰叉
    撞了同一个视觉通道 —— 人分不清"可疑"和"不进场景"。灰色只留给判掉的。 */
 /* 判掉 = 灰白 + 红框 + 一道划掉的红线(用户定的语义, 这里只换材质)。
    滤镜挂 img 不挂 .inst: 挂在 .inst 上 filter 会连 ::after 的红线一起变灰。 */
 .inst.dropped{box-shadow:inset 0 0 0 2px var(--red);z-index:1}
 .inst.dropped img{filter:grayscale(1) contrast(.85) brightness(1.08)}
 .inst.dropped:hover img{filter:grayscale(.2)}
 .inst.dropped::after{content:'';position:absolute;left:8px;top:8px;right:8px;aspect-ratio:1;
     pointer-events:none;background:linear-gradient(to top right,
       transparent 48.2%,var(--red) 48.2%,var(--red) 51.8%,transparent 51.8%)}
 /* 代表 = 实心黑框(正片), 和"判掉"的红框同一个位置、不同的颜色 */
 .inst.rep{box-shadow:inset 0 0 0 2px var(--ink);z-index:1}
 .alarm{position:absolute;top:12px;left:12px;background:var(--sheet);border:1px solid var(--ink);
        color:var(--ink);font-style:normal;font-size:10px;padding:0 4px;line-height:1.5}
 .repbadge{position:absolute;top:12px;right:12px;background:var(--ink);color:#fff;
           font-style:normal;font-size:10px;padding:0 5px;line-height:1.5}
 .repbadge.guess{background:var(--sheet);color:var(--ink);border:1px dashed var(--ink)}
 .im{padding:8px 8px 2px;font-size:11px}
 .im b{font-size:12.5px;font-weight:700;letter-spacing:-.01em}
 .im em{float:right;font-style:normal;color:var(--ink);font-size:11.5px}
 /* 面积条: 一根细黑条就是一张最小的柱状图 */
 .bar{height:2px;background:#e9e8e4;margin:6px 0 5px}
 .bar s{display:block;height:100%;background:var(--ink);text-decoration:none}
 /* AR 画成矩形字形 —— 横扫一眼就能分堆 */
 .glyph{display:flex;align-items:center;gap:6px;color:var(--mute);font-size:10.5px;margin:3px 0}
 .glyph u{display:inline-block;border:1px solid var(--ink);text-decoration:none;flex:none}
 .dim{color:var(--mute);font-size:10.5px;line-height:1.45}
 .dim.verdict{color:var(--ink)}      /* 结论, 不是第四个字段 */
 /* 组装预告里"出问题"的那几句: 不进场景 / 没有自己的 mesh。红字, 不是灰字 ——
    灰字等于没说, pedestrian r2 三辆车借 car_0 就是在灰字里漏掉的。 */
 .dim b.picked{color:var(--ink);font-weight:700}
 .red,.panel h3 span.red{color:var(--red);font-weight:600;margin-left:0;font-size:inherit}
 .meshsum{margin:0 10px 8px;font-size:11.5px;line-height:1.45;color:var(--mute)}
 /* 两列定死, 第三个按钮整行 —— 否则同一排里 keep/drop 会错开 21px */
 .iops{display:grid;grid-template-columns:1fr 1fr;padding:6px 8px 8px;margin-top:auto}
 .iops button{padding:4px 4px;font-size:11px;min-width:0}
 .iops button:nth-child(2){margin-left:-1px}
 .iops button:nth-child(3){grid-column:1/-1;margin-top:-1px}

 /* ---------------------------------------------------------------- 帧抽屉 */
 .tray{grid-column:1/-1;background:var(--sheet);padding:14px}
 .tray h3{margin:0 0 12px;font-size:13px;font-weight:700;letter-spacing:-.01em}
 .tray h3 span{color:var(--mute);font-weight:400;margin-left:10px;font-size:12px}
 .cands{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:0;padding-right:1px}
 /* .cand 是 <button>(为了键盘可达) —— 把按钮的默认外观全收掉 */
 .cand{position:relative;display:block;width:auto;margin:0 -1px -1px 0;padding:0;
       background:var(--sheet);border:1px solid var(--rule);color:inherit;font:inherit;
       text-align:left;cursor:pointer;min-width:0}
 .cand:hover:not(:disabled){background:var(--sheet);box-shadow:inset 0 0 0 1px var(--ink);z-index:1}
 .cand.cur,.cand.cur:hover:not(:disabled){box-shadow:inset 0 0 0 3px var(--ink);z-index:2}
 .cand:focus-visible{outline:2px solid var(--ink);outline-offset:2px;z-index:3}
 .cand img{display:block;width:calc(100% - 12px);margin:6px 6px 0;aspect-ratio:1;background:var(--plate)}
 .cand .ohome{position:absolute;top:10px;left:10px;background:var(--sheet);
        border:1px dashed var(--ink);color:var(--ink);font-style:normal;font-size:10px;padding:0 4px}
 .cand div{padding:5px 7px 7px;font-size:10.5px;color:var(--mute);line-height:1.45}
 .empty{padding:56px 0;color:var(--mute);line-height:1.7;max-width:60ch}

 /* ---------------------------------------------------------------- 召回信息表 */
 .rcsum{color:var(--mute);font-size:12.5px;margin-bottom:14px;line-height:1.6;max-width:78ch}
 .rcsum b{color:var(--ink);font-weight:700}
 .rcrow{display:flex;gap:18px;align-items:flex-start;padding:14px 0;border-top:1px solid var(--rule)}
 .rcrow img{background:var(--plate);flex:none}
 .rcrow .t{flex:1;min-width:0}
 .rcrow .t b{font-size:15px;font-weight:700;letter-spacing:-.02em}
 .rcrow .why{font-size:12.5px;margin-top:6px;max-width:70ch;color:var(--ink)}
 .rcrow .num{color:var(--mute);font-size:11.5px;margin-top:4px}
 .rcv{font-style:normal;font-size:10.5px;padding:0 5px;margin-left:8px;white-space:nowrap;
      background:var(--sheet)}
 .rcedge{border:1px dashed var(--ink);color:var(--ink);font-size:10.5px;padding:0 5px;
         margin-left:6px;font-style:normal}
 .rcsum .src{color:var(--mute);font-size:11px;margin-top:8px;word-break:break-all;user-select:all}

 /* ---------------------------------------------------------------- 闸 3 尺子表 */
 /* 这一屏的主角是 k。大号数字、左齐, 左边一道粗条: 黑 = 采用, 红 = 被拒 */
 .sum{display:flex;gap:40px;flex-wrap:wrap;align-items:flex-end;padding:18px 0 18px 18px;
      margin:4px 0 18px;background:transparent;border:0;border-left:6px solid var(--ink)}
 .sum .big{font-size:30px;font-weight:700;letter-spacing:-.03em;line-height:1}
 .sum .lb{color:var(--mute);font-size:11.5px;display:block;margin-bottom:6px;letter-spacing:0}
 /* 注解: 左边一道线, 不套框 —— 它不该比它注解的表还重 */
 .note{padding:2px 0 2px 14px;font-size:12px;line-height:1.6;margin:0 0 14px;max-width:96ch;
       border-left:2px solid var(--ink);background:none;color:var(--ink)}
 .note.warn{border-left-color:var(--red)}
 .note.info{border-left-color:var(--ink)}
 .note code,.rcsum code{font-family:ui-monospace,SFMono-Regular,Menlo,"Liberation Mono",monospace;
            background:var(--field);border:0;padding:1px 5px;font-size:11px;color:var(--ink);
            display:inline-block;white-space:pre;max-width:100%;overflow-x:auto;
            vertical-align:bottom;user-select:all}
 /* 可复制的命令: 按空格软换行 —— 裁掉一截的话人读不到 --run r2 就不敢跑 */
 .note code.cmd,.rcsum code.cmd{display:block;margin:8px 0;padding:8px 10px;
            white-space:pre-wrap;overflow-wrap:break-word;overflow-x:visible;max-width:none;
            line-height:1.7}
 .rtwrap{overflow-x:auto}
 .rt{width:100%;border-collapse:collapse;font-size:12.5px}
 .rt th{text-align:right;color:var(--mute);font-weight:400;font-size:11px;padding:0 10px 7px;
        border-bottom:1px solid var(--ink)}
 .rt th:first-child,.rt th:nth-child(2){text-align:left}
 .rt td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:right}
 .rt td:nth-child(2){text-align:left;font-weight:700;letter-spacing:-.01em}
 .rt tbody tr:hover td{background:#f7f7f5}
 .rt tr.off td{color:var(--mute)}
 .rt tr.off td:nth-child(2){font-weight:400}
 .rt tr.off img{filter:grayscale(1);opacity:.55}
 .rt tr.out td:nth-child(2){color:var(--red)}          /* 离群, 不采用 = "不" */
 .rt tr.kline td{border:0;border-top:1px solid var(--ink);padding:4px 10px;
                 color:var(--ink);font-size:11px;font-weight:700}
 .rt img{width:34px;height:34px;background:var(--plate);object-fit:cover;display:block}
 .rt input{width:66px;background:var(--sheet);border:1px solid var(--edge);border-radius:0;
           color:var(--ink);padding:4px 6px;font:inherit;font-size:12.5px;text-align:right}
 .rt input:focus{outline:2px solid var(--ink);outline-offset:0;border-color:var(--ink)}
 .rt tr.edited input{border-color:var(--ink);font-weight:700;box-shadow:inset 0 -2px 0 var(--ink)}
 .bar2{display:inline-block;height:6px;background:var(--ink);vertical-align:middle}
 .why{color:var(--mute);font-size:11px;font-weight:400}

 /* ---------------------------------------------------------------- 弹层 */
 .scrim{position:fixed;inset:0;z-index:50;background:rgba(20,20,18,.3);
        backdrop-filter:blur(5px);-webkit-backdrop-filter:blur(5px);
        display:flex;align-items:flex-start;justify-content:center;padding:28px 20px;
        overflow:auto;animation:fade .12s linear}
 @keyframes fade{from{opacity:0}to{opacity:1}}
 .modal{background:var(--sheet);border:1px solid var(--ink);border-radius:0;width:100%;
        max-width:1240px;box-shadow:none;animation:rise .14s ease-out}
 @keyframes rise{from{transform:translateY(6px)}to{transform:none}}
 .mhd{position:sticky;top:0;z-index:2;background:var(--sheet);border-bottom:2px solid var(--ink);
      padding:13px 16px 11px;display:flex;gap:14px;align-items:baseline}
 .mhd b{font-size:18px;font-weight:700;letter-spacing:-.025em}
 .mhd .sp{flex:1} .mhd .sub{color:var(--mute);font-size:12px}
 .mbody{padding:16px}
 .mbody .panel,.mbody .tray{background:none;border:0;padding:0}
 .mclose{font-size:14px;line-height:1;padding:5px 10px;align-self:center}

 /* ---------------------------------------------------------------- 选中态
    ★ 这一块必须放在**所有**容器按钮规则之后:
      .more button / .wcard .row button 这些 (0,1,1)~(0,2,1) 的规则一旦写了背景,
      就会和 button.on 打平甚至压过它, 于是选中的按钮看起来没选中。
      :not(:disabled) 版本 (0,2,1) 和 hover 版本 (0,3,1) 都得写, 否则光标一压上去
      就退回未选中 —— 那正是人要下手点的时刻(上一版审出来过)。
    ★ 中性选中(keep / shared / solo)**不做实心**: 这三个几乎永远处在选中态
      (keep 是默认, 模式有配置默认), 一张卡两个黑块, 八张卡就是十六个黑方块,
      比图还抢眼 —— 而这一屏的主角是图。默认状态不该喊。
      所以: 中性选中 = 2px 黑框 + 加粗(框对白底 21:1, 满足 WCAG 1.4.11 的 3:1),
      **只有"否"才实心**(红)。全页实心的东西, 就是你判掉的东西。 */
 button.on,button.on:not(:disabled),button.on:hover:not(:disabled){
        background:var(--sheet);border-color:var(--ink);color:var(--ink);font-weight:700;
        box-shadow:inset 0 0 0 1px var(--ink);position:relative;z-index:1}
 .gsw button.on,.gsw button.on:hover:not(:disabled){background:var(--ink);color:#fff}
 /* "不"是红的: drop / drop word 选中时实心红 */
 button.on[data-yn="drop-word"],button.on[data-yn="drop-word"]:not(:disabled),
 button.on[data-yn="drop-word"]:hover:not(:disabled),
 button.on[data-io="drop"],button.on[data-io="drop"]:not(:disabled),
 button.on[data-io="drop"]:hover:not(:disabled),
 button.on[data-rio="drop"],button.on[data-rio="drop"]:not(:disabled),
 button.on[data-rio="drop"]:hover:not(:disabled){
        background:var(--red);border-color:var(--red);color:#fff}

 @media (prefers-reduced-motion:reduce){
   *,*::before,*::after{animation-duration:.001ms!important;transition-duration:.001ms!important}
 }
 @media (max-width:720px){
   header{padding:10px 14px;gap:10px}
   .path{order:9;flex:1 1 100%} #save{order:10}
   #grid{padding:4px 15px 28px 14px;grid-template-columns:repeat(auto-fill,minmax(160px,1fr))}
   .sum{gap:22px} .sum .big{font-size:24px}
 }
</style>
<header>
  <a id="runs" href="/runs" title="Every run on disk: pick another one (this page stays on the run in its address)">runs ▾</a>
  <b id="scene">…</b>
  <span class="gsw" id="gsw"></span>
  <span id="gate" class="ghd"></span>
  <span class="gsw" id="rcw"></span>
  <span id="stat" class="ghd dimly"></span>
  <span class="sp"></span>
  <span id="dec" class="path"></span>
  <button id="save" disabled>Save</button>
</header>
<div id="grid"><div class="empty">Loading…</div></div>
<script>
const COL = {keep:"#bdbcb6",drop:"#d0000e","drop-word":"#d0000e",solo:"#bdbcb6",shared:"#bdbcb6"};
// ★ 卡片左边那条框线**只说一件事: 这一条被否了**(红)。其余一律和网格线同色。
//   原来它按模式上色, 但它读的是 w.模式(人显式点过的)而不是 实际模式(含配置默认),
//   于是会出现按钮亮着 solo、色条却是 keep 色 —— 两处打架。按钮就在下面写着模式,
//   色条再说一遍本来就是冗余, 去掉比修对更干净。
const MODES = ["shared","solo"];   // 只有两种结果: 共用一个 / 各造各的

// S.行 是唯一真相(决策文件的「候选」原样)。S.词 是每次改动后重建的派生索引;
// S.实例 是 /api/insts 的只读缓存, **永不写回文件**。
let S = {行:[], 词:new Map(), 实例:new Map(), 待删:new Set(), meta:{}};
let dirty = false, openTray = null;   // {词, oid|null}  oid=null 表示不限实例
// 弹层。{view:'insts'|'src', 词, oid|null, 从}  —— 沉浸式, 背景虚化
let M = null;
// 当前看的是哪道闸: 'review'(闸2/4, 编辑 05_review.json) | 'scale'(闸3, 编辑 01_vlm_sizes.json)
let GATE = 'review';
// 召回信息表的数据。只读, 永不写回 —— 这一屏是"给人看召回找到了什么", 不是编辑器。
let RC = null;
let RC弹过 = false;   // 闸 4 首次打开自动弹一次, 之后靠顶栏那个钮

const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const qs  = o => new URLSearchParams(o).toString();
// ★ 看哪个轮次写在地址里(s=场景 r=轮次 i=室内), 不在服务端记 —— 这样两个标签页可以各看各的,
//   /runs 那一页点进来就是换轮次, 服务不用重启。所有请求都得带上, 漏一个就会去读启动时那个轮次。
const RUN = (() => { const p = new URLSearchParams(location.search), o = {};
  for (const k of ['s','r','i']) if (p.get(k)) o[k] = p.get(k); return o; })();
const U = (path, o) => path + '?' + qs({...RUN, ...(o || {})});
const isInst = r => r.来源 === "逐实例";
const isRecall = r => (r.实例 ?? 0) >= 10000;

// ★ 生成名只有这一处拼装, 和后端 Store.gen_name 同一个公式。
const genName = (词, oid) =>
  `${S.meta.TAG}_${String(词).replace(/ /g,'_')}` + (oid==null ? '' : `_${oid}`);

// ★ 词级决定是**全词行的 OR**, 不是"类行说了算"。
//   review_gate.py:329-334  {r["词"] for r in saved if r["决定"]=="solo"}
//   任何一行是 solo, 整个词就进 SOLO。所以推模式必须扫全词的行。
function 推模式(行们){
  const ds = new Set(行们.map(r => r.决定));
  if (ds.has('drop-word')) return 'dropped';
  if (ds.has('solo'))      return 'solo';
  if (ds.has('shared'))    return 'shared';
  return null;                    // 没明说 —— 跟着 pipeline 的 SOLO 配置走
}

// 这个词没明说的话, 实际会是哪种。★ 这不是第三种模式, 是"跟着配置走"的结果。
// 第一版把它做成了第三个按钮 default, 那是把一个**状态**当成了一个**选项**,
// 而它的含义还随场景变: SOLO="car" 时 street lamp 默认共用, SOLO="*" 时默认各造各的。
function 配置默认(词){
  const solo = String(S.meta.SOLO || '');
  const keep = new Set(S.meta.SOLO_KEEP || []);
  if (keep.has(词)) return 'shared';
  if (solo === '*') return 'solo';
  return solo.split(',').map(x=>x.trim()).includes(词) ? 'solo' : 'shared';
}
const 实际模式 = w => w.模式 || 配置默认(w.词);

// ★ 被 drop 的行会被 review_gate 从「候选」里删掉(review_gate.py:271), 决定只留在「历史决定」。
//   只看表的话, 这些实例在面板上显示成 keep、没有红叉 —— 人回头核对决策时以为没 drop 过,
//   而组装那边照历史决定真的把它们去掉了, 两边打架。所以表里没有这一行时要回退查历史。
function 历史drop(词){
  return Object.values(S.历史 || {})
    .filter(h => h.词 === 词 && h.决定 === 'drop' && h.实例 != null)
    .map(h => +h.实例);
}

// ★ VLM 否决只是提醒, 人说了算。VLM放行 = 人看过、VLM 说不是也照样要(写在这个实例的行上)。
//   没放行的否决实例: 不生成、不摆、不借别人的 mesh(以前会按名字借, car #2 就成了 8 米的 car_0)。
function VLM否(词, oid){
  const F = (S.网格事实 || {})[词];
  const r = F && F.实例.find(x => x.oid === oid);
  return !!(r && r.VLM否决);
}
function 放行了(词, oid){
  const w = S.词.get(词);
  const i = w && w.实例行.has(oid) ? w.实例行.get(oid) : null;
  return i != null && !!S.行[i].VLM放行;
}

// ---- 整类样本(共用词): 这一类共用的那个 mesh, 照着谁的哪一帧建 ----
// 来源四种, 优先级从高到低:
//   人      行上的「改用」(有类行写类行; 没类行写被选实例自己的行, --apply 塌成裸词)
//   配置    pipeline 源码里写死的 PICK_OVERRIDE_BY_SCENE, s_override 每次都会套
//   机器    盘上 picks.json —— 挑图分排名 + VLM 选择题复核之后的结果, 不一定是分数第一
//   预测    盘上那张用不了了(它被挡掉了 / 挑它时挡掉的实例和现在不一样 / 这个词原来不是共用的),
//           保存后流水线会重挑。这里只能给分数第一的, VLM 复核可能换人 —— 要确定就点 set as rep 钉住
// ★ 以前前端把"分数第一"当成"机器选的", 而 timesquare r1 有 4 个词盘上用的根本不是分数第一。

// 能不能当整类样本 —— 和 class_sample.py 同一条: 去重判给别的词的、人 drop 的、VLM 否决没放行的不能
function 可当样本(词, oid){
  const F = (S.网格事实 || {})[词];
  const r = F && F.实例.find(x => x.oid === oid);
  if (r && r.去重淘汰) return false;
  if (人决(词, oid) === 'drop') return false;
  return !(VLM否(词, oid) && !放行了(词, oid));
}
// 这个实例分最高的那帧(set as rep 用它)。没资格参选(有深度的帧不够)就退回它的代表帧。
function 最佳帧(词, oid){
  const F = (S.网格事实 || {})[词];
  const r = F && F.实例.find(x => x.oid === oid);
  if (r && r.挑图帧 != null) return r.挑图帧;
  const m = ((S.实例.get(词) || {}).数据 || []).find(x => x.oid === oid);
  return m ? m.代表帧 : null;
}
// 分数第一的(只算现在能当样本的)
function 自动样本(词){
  const F = (S.网格事实 || {})[词];
  if (!F) return null;
  const ok = F.实例.filter(r => r.挑图分 > 0 && r.挑图帧 != null && 可当样本(词, r.oid));
  if (!ok.length) return null;
  const b = ok.reduce((a, r) => r.挑图分 > a.挑图分 ? r : a);
  return {oid: b.oid, fr: b.挑图帧};
}
function 配置钉(词){ const v = (S.配置指定样本 || {})[词]; return v ? [+v[0], +v[1]] : null; }
// 不钉的话流水线会用谁
function 机选(w){
  const 词 = w.词, 盘 = (S.盘上样本 || {})[词];
  if (盘 && !盘.人工 && 可当样本(词, 盘.oid)){
    // 挑它的时候挡掉了谁 —— 和现在一样, 重跑也还是它(drive.py 用同一个比法决定要不要重挑)
    const F = (S.网格事实 || {})[词];
    const 今 = F ? F.实例.filter(r => !r.去重淘汰 && !可当样本(词, r.oid)).map(r => r.oid) : [];
    const 同 = a => a.slice().sort((x, y) => x - y).join();
    if (盘.排除 == null || 同(盘.排除) === 同(今)) return {oid: 盘.oid, fr: 盘.fr, 预测: false};
  }
  const a = 自动样本(词);
  return a ? {oid: a.oid, fr: a.fr, 预测: true} : null;
}
// 现在这一类用的是谁。solo 下没有整类样本, 返回 null。
function 当前样本(w){
  if (实际模式(w) === 'solo') return null;
  if (w.钉) return {oid: w.钉[0], fr: w.钉[1], 来源: '人'};
  const c = 配置钉(w.词);
  if (c) return {oid: c[0], fr: c[1], 来源: '配置'};
  const m = 机选(w);
  return m ? {oid: m.oid, fr: m.fr, 来源: m.预测 ? '预测' : '机器'} : null;
}
// 词卡封面: 共用词就是整类样本; solo 词给分数第一的实例; 都没有 -> 面积最大的一行
function 封面(w){
  const 词 = w.词;
  const cur = 当前样本(w);
  if (cur) return {oid: cur.oid, fr: cur.fr, gen: genName(词),
                   disk: cur.来源 === '机器', 改: cur.来源 === '人'};
  const a = 实际模式(w) === 'solo' ? 自动样本(词) : null;
  if (a) return {oid: a.oid, fr: a.fr, gen: genName(词, a.oid), disk: false, 改: false};
  const r = w.全部.map(i => S.行[i]).filter(x => !isRecall(x))
             .sort((x, y) => (y.面积占比 || 0) - (x.面积占比 || 0))[0] || S.行[w.全部[0]];
  return {oid: r.实例, fr: r.帧, gen: r.生成名, disk: !r.改用, 改: !!r.改用};
}

// 人给这个实例换到的帧(没换 = null)。逐实例看它自己的行; 共用词看类行的「改用」是不是指它。
// ★ 实例卡片的图必须跟着它变 —— 以前卡片永远画机器挑的代表帧, 人在抽屉里点完没有任何变化,
//   分不清选上没选上。
function 人选帧(w, oid){
  const i = w.实例行.has(oid) ? w.实例行.get(oid) : null;
  const a = i != null ? S.行[i].改用 : null;
  if (a && a[0] === oid) return a[1];
  const c = w.类行 != null ? S.行[w.类行].改用 : null;
  if (c && c[0] === oid) return c[1];
  return null;
}

// 这个实例人判的是什么: 表里有行看行, 没行(drop 之后被 review_gate 删掉了)看历史。
function 人决(词, oid){
  const w = S.词.get(词);
  const i = w && w.实例行.has(oid) ? w.实例行.get(oid) : null;
  if (i != null) return S.行[i].决定;
  return 历史drop(词).includes(oid) ? 'drop' : 'keep';
}

// ---- 组装预告: 每个实例进不进场景、用谁的 mesh ----
// ★ 照抄 assemble_scene.py 的摆放循环(:273-300)和 pick_mesh(:69-80):
//     共用词且有类 mesh -> 类 mesh;  有自己的 -> 自己的;  有类 mesh -> 类 mesh;
//     否则借同词 mesh 里**生成名字符串排序**的第一个(car_10 排在 car_4 前面);
//     借到的是被 drop 的那个 -> 这个也不摆;  盒子过薄/过大 -> 不摆。
//   "自己的"= 盘上已有, 或这一轮会生成(gen_per_instance.py 的条件: 逐实例词 + 有盒子 +
//   VLM 没否 + 过像素门槛 + 没被 drop)。
//   规则放前端是因为它要跟着人还没保存的点选实时变; 盘上事实由后端 mesh_facts 给。
//   两边任何一处改了规则, 这里要跟着改 —— 不然界面说的和场景里的又是两回事。
// 返回 Map(oid -> {摆, 因, 网格:{来源, 名, 号, 待生成, 现借}, 没自有因})
function 网格去向(词){
  const F = (S.网格事实 || {})[词];
  if (!F) return null;
  const w = S.词.get(词);
  const 模式 = w ? 实际模式(w) : 配置默认(词);
  const 有盒 = oid => !F.盒子 || (String(oid) in F.盒子);
  const 盘上 = new Set(F.盘上实例);
  const 会生成 = new Set(F.实例.filter(r => 模式 === 'solo' && !r.去重淘汰
      && (!r.VLM否决 || 放行了(词, r.oid))
      && !r.低于门槛 && 有盒(r.oid) && 人决(词, r.oid) !== 'drop').map(r => r.oid));
  // 类 mesh: 盘上有, 或这一轮 pick 会写类目录(共用词 / 逐实例词但一个实例都没过门槛)
  const 类有 = F.类网格 || (模式 === 'shared')
             || (模式 === 'solo' && !F.实例.some(r => !r.去重淘汰 && !r.低于门槛));
  const 号表 = new Map([...盘上, ...会生成].map(o => [genName(词, o), o]));
  const 池 = [...号表.keys()].sort();
  // 不生成、现在就组装的话会用谁 —— 给"自己的还没生成"那句话配后半句
  const 盘池 = [...盘上].map(o => genName(词, o)).sort();
  const 现借 = F.类网格 ? 'the class mesh' : (盘池[0] || null);
  const out = new Map();
  for (const r of F.实例){
    const o = r.oid, e = F.盒子 ? F.盒子[String(o)] : null;
    let v;
    // 先判跟人的决定无关的(点 keep/drop 都改变不了), 再判人的, 最后判 mesh
    if (r.去重淘汰)              v = {摆:false, 因:'dedup'};
    else if (模式 === 'dropped') v = {摆:false, 因:'word'};
    else if (!有盒(o))           v = {摆:false, 因:'nobox'};
    else if (e && (e[2] < S.守卫.MIN_H || Math.max(...e) > S.守卫.MAX_D))
                                 v = {摆:false, 因:'size', 尺寸:e};
    else if (人决(词, o) === 'drop') v = {摆:false, 因:'drop'};
    else if (r.VLM否决 && !放行了(词, o)) v = {摆:false, 因:'vlm'};
    else {
      let g = null;
      if (模式 === 'shared' && 类有) g = {来源:'class'};
      else if (盘上.has(o) || 会生成.has(o))
        g = {来源:'own', 待生成: !盘上.has(o), 现借: 盘上.has(o) ? null : 现借};
      else if (类有) g = {来源:'class'};
      else if (池.length) g = {来源:'borrow', 名: 池[0], 号: 号表.get(池[0])};
      if (!g) v = {摆:false, 因:'nomesh'};
      // 放行了的否决实例只许用自己的 mesh(共用词用类 mesh 是人判的, 可以)
      else if (r.VLM否决 && g.来源 !== 'own' && !(模式 === 'shared' && g.来源 === 'class'))
        v = {摆:false, 因:'vlmnomesh'};
      else if (g.来源 === 'borrow' && 人决(词, g.号) === 'drop')
        v = {摆:false, 因:'borrowdropped', 网格:g};
      else v = {摆:true, 网格:g};
    }
    if (v.网格 && v.网格.来源 !== 'own')
      v.没自有因 = 模式 === 'shared' ? 'shared' : r.VLM否决 ? 'vlm' : r.低于门槛 ? 'small' : 'other';
    if (人决(词, o) === 'drop') v.人drop = true;
    v.词 = 词;
    v.放行 = r.VLM否决 && 放行了(词, o);
    out.set(o, v);
  }
  return out;
}
const 短名 = n => String(n).startsWith(S.meta.TAG + '_') ? n.slice(S.meta.TAG.length + 1) : n;

// 一个实例的去向写成人话。返回 [html, 是不是问题]。问题一律红字。
function 去向文(v){
  const R = t => '<span class="red">'+esc(t)+'</span>';
  const 因文 = {vlm: ' (VLM rejected it)', small: ' (below '+S.meta.solo门槛px+' px)', other: ''};
  if (!v.摆){
    const 也 = v.人drop && v.因 !== 'drop' ? ' — you also dropped it' : '';
    switch (v.因){
      case 'dedup': return [R('NOT PLACED — dedup: another word represents it'), true];
      case 'word':  return [R('NOT PLACED — the whole word is dropped'), true];
      case 'nobox': return [R('NOT PLACED — no 3D box (too few depth points)'+也), true];
      case 'size': {
        const e = v.尺寸, big = Math.max(...e);
        return [R('NOT PLACED — box '+(big > S.守卫.MAX_D ? big.toFixed(1)+' m long, over the '+S.守卫.MAX_D+' m limit'
                                     : 'only '+e[2].toFixed(2)+' m tall, under '+S.守卫.MIN_H+' m')+也), true];
      }
      case 'drop':  return [R('NOT PLACED — you dropped it'), true];
      case 'nomesh': return [R('NOT PLACED — no mesh at all for this word'), true];
      case 'vlm':  return [R('NOT PLACED — VLM says this is not a '+v.词+
                             '. Your call: keep it, or pick another frame'), true];
      case 'vlmnomesh': return [R('NOT PLACED — you kept it past the VLM, but it is too small '+
                                  'to get its own mesh (it will not borrow one)'), true];
      case 'borrowdropped':
        return [R('NOT PLACED — it has no own mesh and would borrow '+短名(v.网格.名)+', which you dropped'), true];
    }
  }
  const g = v.网格;
  const 放 = v.放行 ? ' (you overrode the VLM)' : '';
  if (g.来源 === 'own')
    return g.待生成
      ? ['<span>own mesh'+放+' — not generated yet</span>'+
         (g.现借 ? R('; until then it uses '+短名(g.现借)) : ''), !!g.现借]
      : ['<span>own mesh'+放+'</span>', false];
  if (g.来源 === 'class')
    return v.没自有因 === 'shared'
      ? ['<span>shared class mesh</span>', false]
      : [R('NO OWN MESH — uses the class mesh'+因文[v.没自有因]), true];
  return [R('NO OWN MESH — uses '+短名(g.名)+'\'s mesh'+因文[v.没自有因]), true];
}

// 每次 mutate 之后重建派生索引
function 重建索引(){
  const m = new Map();
  S.行.forEach((r, i) => {
    if (!m.has(r.词)) m.set(r.词, {词:r.词, 类行:null, 实例行:new Map(),
                                   召回行:[], 全部:[], 实例数:r.实例数||1});
    const w = m.get(r.词);
    w.全部.push(i);
    if (isRecall(r))            w.召回行.push(i);
    else if (isInst(r))         w.实例行.set(r.实例, i);
    else                        w.类行 = i;
  });
  for (const w of m.values()){
    // ★ 卡片上那个数字要跟着人的决定走。后端给的 实例数 已经减掉了跨词去重淘汰的
    //   (那是既定事实), 这里再减掉**人自己 drop 的**实例行 —— 否则你 drop 了三个,
    //   卡片还写着原来那个数, 界面在说一件已经不成立的事。
    w.已drop = [...w.实例行.values()].filter(i => S.行[i].决定 === 'drop').length
             + 历史drop(w.词).filter(oid => !w.实例行.has(oid)).length;
    w.实例数 = Math.max((w.实例数 || 1) - w.已drop, 0);
    w.模式 = 推模式(w.全部.map(i => S.行[i]));
    // 冲突: 同一个词既有 solo 行又有 shared 行 -> 下游同时进 SOLO 和 SOLO_KEEP
    const ds = new Set(w.全部.map(i => S.行[i].决定));
    w.冲突 = ds.has('solo') && ds.has('shared');
    // 人钉的整类样本。落点两种, 取决于这个词有没有类行:
    //   有类行(person...) -> 类行的 改用, overrides() 键就是词
    //   没类行(car/tree 这种全是逐实例行的) -> 被选实例自己行上的 改用;
    //     词在 shared 里时 review_gate 的 _ov_key 把 词#实例 塌成裸词, 于是它就是类级样本。
    //   ★ 塌成同一个键意味着共用词**只许一个**改用。不止一个且值不同 = 冲突,
    //     review_gate --apply 会硬停; 这里标出来, 点一次 set as rep 就清干净。
    //   solo 下 改用 是各实例自己的换帧, 不是整类样本, 不算。
    w.钉 = null; w.钉冲突 = false;
    if (实际模式(w) !== 'solo'){
      const 带 = [w.类行, ...w.实例行.values()].filter(i => i != null && S.行[i].改用);
      if (带.length){
        w.钉 = S.行[带[0]].改用;
        w.钉冲突 = new Set(带.map(i => String(S.行[i].改用))).size > 1;
      }
    }
  }
  S.词 = m;
  // ★ 卡片数字和汇总用组装预告算 —— 它要读 S.词(人决), 所以必须在 S.词 换成新索引之后。
  for (const w of m.values()){
    const 去向 = 网格去向(w.词);
    w.去向 = 去向;
    if (!去向) continue;
    const 摆 = [...去向.values()].filter(v => v.摆);
    w.实例数 = 摆.length;
    const 借 = new Map();
    let 类兜 = 0, 待 = 0;
    const 挡 = {};                      // 因 -> 个数, 只数和人的决定无关的那几种
    for (const v of 摆){
      if (v.网格.来源 === 'borrow') 借.set(v.网格.名, (借.get(v.网格.名) || 0) + 1);
      else if (v.网格.来源 === 'class' && v.没自有因 !== 'shared') 类兜++;
      else if (v.网格.来源 === 'own' && v.网格.待生成) 待++;
    }
    let 丢 = 0;
    for (const v of 去向.values()){
      if (v.摆) continue;
      if (v.因 === 'drop') 丢++;
      else if (['vlm', 'nobox', 'size', 'nomesh', 'borrowdropped', 'vlmnomesh'].includes(v.因))
        挡[v.因] = (挡[v.因] || 0) + 1;
    }
    // ★ 丢 只数"因为你 drop 才不摆"的。没盒子的那几个即使你也 drop 过, 记在 挡 里 ——
    //   不管 drop 不 drop 都摆不进去, 两边都数就是同一辆车数两次(car #9/#11/#12)。
    w.网格账 = {借, 类兜, 待, 挡, 丢};
  }
}

// ---- 变更器: 改 S.行 的唯一入口 ----

// 切词的模式。★ 必须扫全词的行(约束: 词级决定是 OR), 否则一条陈旧的 solo 行
//   就能把词翻回去。历史决定里的残留本轮不处理(那是批次 B 的墓碑行), 但要报警。
function 设词模式(词, m){
  const w = S.词.get(词); if(!w) return;
  const 旧 = w.模式;                     // null = 之前没明说
  w.全部.forEach(i => {
    const r = S.行[i];
    if (r.决定==='solo' || r.决定==='shared') r.决定 = 'keep';   // 先扫干净
  });
  // 再点一次已经选中的 = 取消这条决定, 退回"跟配置走"。
  // 不留一个和配置同值的决定在文件里 —— --apply 会把它当成人明确做过判断来处理。
  const 旧实际 = 实际模式(w);
  if (m !== 旧){
    const i = w.类行 != null ? w.类行 : w.全部[0];
    if (i != null) S.行[i].决定 = m;
  }
  重建索引();
  const 新实际 = 实际模式(S.词.get(词));
  if (新实际 !== 旧实际) 换模式收改用(词, 旧实际, 新实际);
  dirty = true; 重建索引(); render();
}

// ★ 「改用」在两种模式下是两回事: solo 下是"各实例自己用哪帧"(每行各一个), 共用下是
//   "整类样本"(全词只许一个, 写在哪行都塌成同一个键)。切到共用时, solo 留下的几个换帧会
//   撞成一个键, review_gate 硬停; 切到 solo 时, 整类样本会变成那个实例自己的换帧 —— 都不是人的意思。
//   所以切模式时把旧模式的 改用 收起来, 切回去再放出来(只在这次打开的页面里记着)。
function 换模式收改用(词, 旧, 新){
  S.暂存改用 = S.暂存改用 || {};
  const 存 = (S.暂存改用[词] = S.暂存改用[词] || {});
  const 收 = [];
  S.行.forEach(r => {
    if (r.词 === 词 && r.改用 && !isRecall(r)){
      收.push({g: r.生成名, oid: isInst(r) ? r.实例 : null, v: r.改用}); delete r.改用;
    }
  });
  const 放 = 存[新] || [];
  存[旧] = 收; delete 存[新];
  重建索引();
  for (const x of 放){
    let i = S.行.findIndex(r => r.生成名 === x.g);
    if (i < 0 && x.oid != null) i = 确保行(词, x.oid);   // 界面建的空行被收尾拿掉了, 重建
    if (i >= 0) S.行[i].改用 = x.v;
  }
  收尾全部(词);
}

// 界面建的行退回"什么决定都没有"的, 一次清掉(收尾 会删行, 倒着走下标才不乱)
function 收尾全部(词){
  for (let i = S.行.length - 1; i >= 0; i--) if (i < S.行.length && S.行[i].词 === 词) 收尾(i);
}

// 清掉这个词的整类样本(类行和所有实例行上的 改用)。只在共用模式下调。
function 清钉(词){
  S.行.forEach(r => { if (r.词 === 词 && !isRecall(r)) delete r.改用; });
}

function 设词要不要(词, v){        // v: 'keep' | 'drop-word'
  const w = S.词.get(词); if(!w) return;
  if (v === 'drop-word'){
    const i = w.类行 != null ? w.类行 : w.全部[0];
    S.行[i].决定 = 'drop-word';
  } else {
    w.全部.forEach(i => { if (S.行[i].决定==='drop-word') S.行[i].决定='keep'; });
  }
  dirty = true; 重建索引(); render();
}


// 拿到这个实例的行; 没有就凭空建一行。
// ★ 实例层默认零行 —— 只有人在某个实例上做了决定才建。一行 decision:"keep" 是纯噪声,
//   review_gate.py:249 下一轮会 HIST.pop 掉它。
function 确保行(词, oid){
  const w = S.词.get(词);
  if (w && w.实例行.has(oid)) return w.实例行.get(oid);
  const meta = (S.实例.get(词)||{}).数据 || [];
  const m = meta.find(r => r.oid === oid);
  const g = genName(词, oid);
  S.待删.delete(g);
  S.行.push({生成名:g, 词:词, 来源:'逐实例', 实例:oid,
             帧: m ? m.代表帧 : 0, 面积占比: m ? m.中位面积pct : 0,
             决定:'keep', 由界面追加:true});
  重建索引();
  return S.词.get(词).实例行.get(oid);
}

// 惰性清理: 界面自己建的行如果退回"什么决定都没有", 就从表里拿掉, 别留噪声。
function 收尾(i){
  const r = S.行[i];
  if (!r.由界面追加) return;
  // ★ 历史里 drop 过的, 这行 keep 就是"撤销 drop"本身, 不是噪声 —— 删了它, 显示回退查历史
  //   又是 drop, 人点了 keep 叉还在。存下去之后 review_gate 见表里有这个键就以表为准、
  //   把历史那条 drop 删掉(review_gate.py:243-249), 撤销才真正落地。
  if (历史drop(r.词).includes(+r.实例)) return;
  if (r.VLM放行) return;            // 放行本身就是一个决定, 不是噪声
  if (r.决定==='keep' && !r.改用 && !r.复用组 && !r.改类别){
    S.待删.add(r.生成名);
    S.行.splice(i, 1);
    重建索引();
  }
}

function 设实例决定(词, oid, v){
  const i = 确保行(词, oid);
  S.行[i].决定 = v;
  // VLM 否掉的: 点 keep = "我看过了, 照样要" -> 放行; 点 drop = 不要 -> 撤掉放行
  if (VLM否(词, oid)){
    if (v === 'keep') S.行[i].VLM放行 = true; else delete S.行[i].VLM放行;
  }
  // drop 掉的实例不许当整类样本(class_sample 同一条) —— 它正是人钉的那个就一起撤掉,
  // 否则保存后 --apply 照样拿它的抠图建整类 mesh
  const w = S.词.get(词);
  if (v === 'drop' && w && w.钉 && w.钉[0] === oid) 清钉(词);
  if (v === 'keep') 收尾(i); else 收尾全部(词);
  dirty = true; 重建索引(); render();
}

// 设整类样本(共用词)。所有入口(set as rep / 帧抽屉)都走这里。
//   * 先清掉这个词所有 改用 —— 共用词只许一个
//   * 选的正是盘上机器那张(而且没有配置钉) -> 就停在"不钉", 不写一个和原选等值的进去:
//     留着它, --apply 会当成人做过决定来处理
//   * 选的是预测出来的那张 -> 照样钉住: 不钉的话重挑时 VLM 复核可能换人
//   * 选了 VLM 否掉的实例 = 人看过了, 放行(drop 过的翻回 keep)
function 设样本(词, oid, 帧){
  if (!S.词.get(词) || 帧 == null) return;
  清钉(词);
  if (VLM否(词, oid) && !放行了(词, oid)){
    const i = 确保行(词, oid);
    S.行[i].VLM放行 = true;
    if (S.行[i].决定 === 'drop') S.行[i].决定 = 'keep';
  }
  重建索引();
  const w = S.词.get(词), m = 机选(w);
  const 等机 = !配置钉(词) && m && !m.预测 && m.oid === oid && m.fr === 帧;
  if (!等机){
    const i = w.类行 != null ? w.类行 : 确保行(词, oid);
    S.行[i].改用 = [oid, 帧];
  }
  收尾全部(词);
  dirty = true; 重建索引(); render();
}

// ★ 弹层的焦点。原来开了弹层 activeElement 还是 body, 接着按 Tab 会一路走到
//   **虚化背景后面**的按钮上 —— 焦点环画在 blur 底下, 人看不见自己在哪。
//   开的时候移进弹层(见 挂弹层 末尾), 关的时候还回它是从哪儿开的。
function 开弹层(v){ M = v; render(); }
// ✕ / Esc / 点背景: 从实例面板里点开的帧抽屉, 退回实例面板; 别的一律关掉。
// ★ 以前 ✕ 直接关整个弹层, 人从 person 面板 -> #10 的帧抽屉 -> ✕, 一下回到最外层, 面板要重新点开。
function 退一层(){
  if (M && M.view === 'src' && M.从 === 'insts'){
    openTray = null; 开弹层({view:'insts', 词:M.词, oid:null});
  } else if (M) 关弹层();
  else render();
}
function 关弹层(){
  // ★ 存节点没用: render() 会把整棵 DOM 重建, 原来那颗按钮已经不在文档里,
  //   document.contains() 一律 false, 焦点就掉回 body。所以存的是**怎么找回它**。
  const 从 = M ? (M.view === 'recall' ? {钮:'#rcw button'} : {词:M.词}) : null;
  M = null; openTray = null; render();
  let el = null;
  if (从 && 从.钮) el = document.querySelector(从.钮);
  else if (从 && 从.词)
    el = [...document.querySelectorAll('[data-exp]')].find(b => b.dataset.exp === 从.词);
  if (el) { try { el.focus(); } catch(e){} }
}
document.addEventListener("keydown", e => { if (e.key === "Escape" && M) 退一层(); });

async function 取实例(词){
  if (S.实例.has(词)) return S.实例.get(词);
  const st = {状态:'loading', 数据:[]};
  S.实例.set(词, st);
  try {
    const d = await (await fetch(U("/api/insts",{word:词}))).json();
    if (d.error) { st.状态='err'; st.说明=d.error; }
    else { st.状态='ok'; st.数据=d; }
  } catch(e){ st.状态='err'; st.说明=String(e); }
  return st;
}

// ================================================================ 闸 3: 尺度
// SC.改动 = {词: 新高度 | null}  —— null 表示"删掉这个词的参照高度"(撤销 = 删字段)
let SC = {state:null, 改动:{}, 预览:null, 忙:false};
const scDirty = () => Object.keys(SC.改动).length > 0;

async function scLoad(){
  try { SC.state = await (await fetch(U("/api/scale"))).json(); }
  catch(e){ SC.state = {error:String(e)}; }
  SC.改动 = {}; SC.预览 = null; render();
}

// 改一个高度 -> 问后端要新的 k。★ 不在前端重算 —— 公式住在后端一处,
// 抄两遍必然漂(而漂了的后果是"界面说达标、重跑却拒绝")。
let scT = null;
function scSet(词, v){
  const 原 = SC.state.尺子[词];
  if (v === "" || v == null) delete SC.改动[词];
  else {
    const n = parseFloat(v);
    if (!isFinite(n)) return;
    if (原 != null && Math.abs(n - 原) < 1e-9) delete SC.改动[词];  // 改回原值 = 撤销
    else SC.改动[词] = n;
  }
  clearTimeout(scT);
  scT = setTimeout(scPreview, 260);
  render();
}

async function scPreview(){
  if (!scDirty()){ SC.预览 = null; render(); return; }
  const 尺子 = Object.assign({}, SC.state.尺子);
  for (const [w,v] of Object.entries(SC.改动)){ if (v==null) delete 尺子[w]; else 尺子[w]=v; }
  SC.忙 = true; render();
  try {
    SC.预览 = await (await fetch(U("/api/scale/preview"),
      {method:"POST", headers:{"Content-Type":"application/json"},
       body: JSON.stringify({尺子})})).json();
  } catch(e){ SC.预览 = {error:String(e)}; }
  SC.忙 = false; render();
}

function scRender(){
  const g = document.getElementById("grid");
  g.innerHTML = "";
  const d = SC.state;
  document.getElementById("gate").textContent = "Gate 3 of 6 — world scale";
  if (d && d.报告文件) document.getElementById("scene").textContent = d.场景 || "";
  if (!d){ g.innerHTML = '<div class="empty">Loading…</div>'; scLoad(); return; }
  if (d.error){ g.innerHTML = '<div class="empty">'+esc(d.error)+'</div>'; return; }
  if (!d.报告存在 && !d.逐类.length){
    g.innerHTML = '<div class="empty">This run has not reached the scale step yet.<br><br>'+
      esc(d.R1)+'</div>'; return; }

  const 用 = SC.预览 && !SC.预览.error ? SC.预览 : {逐类:d.逐类, 汇总:d.汇总};
  const c = 用.汇总, 原 = d.汇总;
  const wrap = document.createElement("div");
  wrap.style.gridColumn = "1/-1";

  // ---- 汇总条 ----
  const 变 = scDirty() && c.k != null && 原.k != null && Math.abs(c.k-原.k) > 1e-9;
  const ok = c.一致率达标;
  wrap.innerHTML =
    '<div class="sum" style="border-left-color:'+(ok?'var(--ink)':'var(--red)')+'">'+
      '<div><span class="lb">scale factor k</span><span class="big">'+
        (c.k==null?'—':c.k)+'</span>'+
        (变?' <span style="color:var(--mute);font-size:11px">was '+原.k+'</span>':'')+'</div>'+
      '<div><span class="lb">agreement (gate '+c.闸门+')</span><span class="big" style="color:'+
        (ok?'var(--ink)':'var(--red)')+'">'+(c.一致率==null?'—':c.一致率)+'</span>'+
        (scDirty()&&原.一致率!=null&&c.一致率!=原.一致率?
          ' <span style="color:var(--mute);font-size:11px">was '+原.一致率+'</span>':'')+'</div>'+
      '<div><span class="lb">rulers used</span><span class="big">'+c.采用类数+'/'+c.候选类数+'</span></div>'+
      '<div><span class="lb">spread</span><span class="big">'+(c.离散==null?'—':c.离散)+'</span></div>'+
      '<span class="sp" style="flex:1"></span>'+
      (SC.忙?'<span class="lb">recomputing…</span>':'')+
    '</div>';

  // ---- 状态说明 + 界面写不了的部分 ----
  const notes = [];
  if (d.已施加)
    notes.push(['info', 'Already applied (k = '+(d.汇总&&d.校验?d.校验.盘上k:'?')+
      '). The boxes on disk were re-measured afterwards, so the table below now measures '+
      '<b>how far off it still is</b> — near 1.00 means the calibration held.']);
  else if (d.拒绝原因)
    notes.push(['warn', 'Not applied — '+esc(d.拒绝原因)+
      (ok?'' : '. Agreement '+原.一致率+' is below the gate '+原.闸门+
           ', so world_scale.py exits 2 and the segment stops.')]);
  if (d.无需施加)
    notes.push(['info', 'k is close to 1 — this scene was already in real metres, nothing to apply.']);
  // ★ 约定 8: 没有消费者的操作要说明, 不留着让人白填。
  //   施加/不施加/放宽闸门这三个决定界面**写不了** —— world_scale.py 只读 os.environ,
  //   那几个变量连 CFG 表都不在, --dump-cfg 也吐不出来。
  notes.push(['info',
    'Editing a reference height below writes <code>01_vlm_sizes.json</code> — that is the one '+
    'file channel this gate has. <b>Whether to apply, and where the gate sits, cannot be set '+
    'from here</b>: world_scale.py reads those from os.environ only. To override, run:'+
    '<code class="cmd">'+esc(scCmd(d))+'</code>']);
  wrap.innerHTML += notes.map(([k,t]) => '<div class="note '+k+'">'+t+'</div>').join('');

  // ---- 尺子表 ----
  const rows = 用.逐类.slice();
  const 可用 = rows.filter(r => r.倍数 != null);
  可用.sort((a,b) => b.倍数 - a.倍数);
  const 不可用 = rows.filter(r => r.倍数 == null)
                     .sort((a,b) => (b.实例数||0) - (a.实例数||0));
  const mx = 可用.length ? Math.max(...可用.map(r=>r.倍数)) : 1;

  const tr = (r, off) => {
    const ed = r.词 in SC.改动;
    const cls = [off?'off':'', r.倍数!=null&&!r.采用?'out':'', ed?'edited':''].filter(Boolean).join(' ');
    const v = ed ? SC.改动[r.词] : (r.真实 ?? r.参照 ?? '');
    return '<tr class="'+cls+'">'+
      '<td><img loading="lazy" src="'+U('/img/word',{word:r.词, size:80})+'"></td>'+
      '<td><b>'+esc(r.词)+'</b>'+(ed?' <span style="color:var(--ink);font-weight:700">✎</span>':'')+
        (r.用不了?'<br><span class="why">'+esc(r.用不了)+'</span>':'')+'</td>'+
      '<td>'+(r.量到!=null?r.量到.toFixed(2):'—')+'</td>'+
      '<td><input data-sc="'+esc(r.词)+'" value="'+(v===''?'':v)+'" placeholder="—"></td>'+
      '<td>'+(r.倍数!=null?r.倍数.toFixed(2):'—')+'</td>'+
      '<td style="text-align:left;width:120px">'+(r.倍数!=null?
        '<i class="bar2" style="width:'+Math.max(2,Math.round(100*r.倍数/mx))+'px"></i>':'')+'</td>'+
      '<td>'+(r.实例数||0)+'</td>'+
      '<td style="text-align:left;color:var(--dim);font-size:10px">'+
        (r.倍数==null?'':(r.采用?'':'outlier — not used'))+'</td></tr>';
  };
  // k 那条基准线插在倍数跨过 k 的位置 —— 一眼看出谁在上谁在下
  let kline = false;
  const body = 可用.map(r => {
    let pre = '';
    if (!kline && c.k != null && r.倍数 <= c.k){
      kline = true;
      pre = '<tr class="kline"><td></td><td colspan="7">k = '+c.k+
            ' — everything above measures larger than k, everything below smaller</td></tr>';
    }
    return pre + tr(r, false);
  }).join('');

  wrap.innerHTML +=
    '<div class="rtwrap"><table class="rt"><thead><tr><th></th><th>word</th><th>measured</th>'+
    '<th>real (m)</th><th>ratio</th><th></th><th>inst</th><th></th></tr></thead>'+
    '<tbody>'+body+
    (不可用.length? '<tr class="kline"><td></td><td colspan="7" style="color:var(--dim);'+
      'border-top:1px solid var(--rule)">not usable as rulers — fix a height to bring one in</td></tr>'
      + 不可用.map(r=>tr(r,true)).join('') : '')+
    '</tbody></table></div>';
  g.appendChild(wrap);

  g.querySelectorAll("[data-sc]").forEach(i => {
    i.onchange = () => scSet(i.dataset.sc, i.value);
    i.onkeydown = e => { if (e.key === "Enter") i.blur(); };
  });
  scStat();
}

// 把该敲的那行命令拼全 —— 界面写不了的决定, 至少别让人自己去凑参数
function scCmd(d){
  const m = /result\/([^/]+)\/([^/]+)$/.exec(d.R1 || "");
  const sc = m ? m[1] : "<scene>", run = m ? m[2] : "<run>";
  return "SCALE_AGREE_MIN=0.5 ~/miniforge3/envs/sam3d/bin/python "
       + "~/sam3d/pipeline/outdoor_pipeline.py --scene " + sc + " --run " + run + " --only scale";
}

function scStat(){
  const n = Object.keys(SC.改动).length;
  document.getElementById("stat").textContent =
    SC.state && SC.state.逐类
      ? SC.state.逐类.length+" words, "+(SC.state.汇总.候选类数||0)+
        " usable as rulers"+(n?" — "+n+" edited":"") : "";
  写文件名(SC.state ? SC.state.尺子文件 : "");
  const b = document.getElementById("save");
  b.disabled = !n; b.textContent = n ? "Save "+n+" ✎" : "Save";
}

async function scSave(){
  const b = document.getElementById("save");
  b.disabled = true; b.textContent = "Saving…";
  const r = await (await fetch(U("/api/scale/save"), {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({改动: SC.改动})})).json();
  if (r.error){ alert("Save failed: "+r.error); scStat(); return; }
  await scLoad();
  b.textContent = "Saved ✓";
  setTimeout(scStat, 1500);
}

async function boot(){
  let d;
  try {
    d = await (await fetch(U("/api/state"))).json();
  } catch(e) {
    document.getElementById("grid").innerHTML =
      '<div class="empty">Cannot reach the server — '+esc(e)+'<br><br>'+
      'The server is not running, or the ssh -L tunnel dropped.<br>'+
      'Check it, then reload this page.</div>';
    return;
  }
  if(d.error){ document.getElementById("grid").innerHTML =
    '<div class="empty">'+esc(d.error)+'</div>'; return; }
  S.meta = {TAG:d.TAG, 场景:d.场景, 场景名:d.场景名, 轮次:d.轮次,
            闸:d.闸, 阶段:d.阶段, 决策文件:d.决策文件,
            solo门槛px:d.solo门槛px, 可追加行:d.可追加行,
            SOLO:d.SOLO, SOLO_KEEP:d.SOLO_KEEP};   // 配置默认要靠它俩推
  S.行 = d.候选; S.待删 = new Set();
  S.历史 = d.历史决定 || {};
  S.网格事实 = d.网格事实 || {};       // 盘上哪些 glb / 盒子在, 见后端 mesh_facts
  S.盘上样本 = d.盘上样本 || {};       // picks.json: 共用词盘上实际在用的样本, 见后端 disk_samples
  S.配置指定样本 = d.配置指定样本 || {}; // pipeline 源码里写死的换候选
  S.暂存改用 = {};                     // 切模式时收起来的另一种模式的 改用(重新加载就清空)
  S.守卫 = d.组装守卫 || {MIN_H: 0.15, MAX_D: 15};
  document.getElementById("scene").textContent = d.场景;
  // 阶段名(pre/post)对人没意义, 它说的是"召回之前还是之后" —— 直接说那句话。
  document.getElementById("gate").textContent =
    "Gate " + d.闸 + " of 6 — " + (d.阶段 === "post" ? "after recall" : "before generating");
  写文件名(d.决策文件);
  重建索引();
  if (GATE === 'scale') scLoad(); else render();
}

// 顶栏那行文件路径。目录暗、文件名亮 —— 一眼看出在编辑哪一份。
function 写文件名(全路径){
  const el = document.getElementById("dec");
  if (!el) return;
  const 段 = String(全路径).split("/");
  const 尾 = 段.slice(-3);
  el.innerHTML = (段.length > 3 ? '…/' : '') +
    esc(尾.slice(0, -1).join('/')) + (尾.length > 1 ? '/' : '') +
    '<b>' + esc(尾[尾.length - 1]) + '</b>';
  el.title = "Editing this file: " + 全路径;
}

function stat(){
  const c = {};
  S.行.forEach(r => c[r.决定] = (c[r.决定]||0)+1);
  const ov = S.行.filter(r => r.改用).length;
  const 冲 = [...S.词.values()].filter(w => w.冲突).length;
  // 顶栏这行是**进度**, 不是清单: 先说规模, 再说人动过什么。
  // ★ drop 和 drop-word 不能加成一个数 —— 它们写进 JSON 的是**两个不同的字段**
  //   (单个不生成 / 整个词作废)。加起来看着简洁, 代价是人再也看不出自己废了几个词。
  //   solo / shared 同理: 那两个就是"逐实例词"和"共用词"两种决策的计数。
  const 有 = k => c[k] || 0;
  const 头 = S.词.size + (S.词.size>1?" words":" word") + ", " +
             S.行.length + (S.行.length>1?" candidates":" candidate");
  const 尾 = [];
  if (有('drop'))       尾.push(有('drop') + " dropped");
  if (有('drop-word'))  尾.push(有('drop-word') + " word" +
                               (有('drop-word')>1?"s":"") + " cut");
  if (有('solo'))       尾.push(有('solo') + " per-instance");
  if (有('shared'))     尾.push(有('shared') + " shared");
  if (ov)               尾.push(ov + " re-sourced");
  if (冲)               尾.push("⚠ " + 冲 + " mode conflict" + (冲>1?"s":""));
  document.getElementById("stat").textContent = 头 + (尾.length ? " — " + 尾.join(", ") : "");
  const b = document.getElementById("save");
  b.disabled = !dirty; b.textContent = dirty ? "Save *" : "Save";
}

// ---- 词卡(收起态) ----
function 词卡(w){
  const 作废 = w.模式 === 'dropped';
  const n = w.实例数;
  const 丢数 = w.网格账 ? w.网格账.丢 : w.已drop;
  // 封面只从 封面() 拿, 规则见那里(和面板的 class rep、pipeline 的挑图同一条)
  const 封 = 封面(w);
  const oid = 封.oid, fr = 封.fr, 有代表改 = 封.改;

  const d = document.createElement("div");
  d.className = "wcard" + (作废 ? " dead" : "");
  d.style.borderLeftColor = 作废 ? COL['drop-word'] : (COL[w.模式] || COL.keep);

  // 按钮文字说的是**效果**, 不是配置名。default 和 shared 在 SOLO 不含这个词的场景里
  // 效果一样(都是整类共用一个 mesh), 差别只在 shared 会显式写进决策文件 ——
  // 那是给 SOLO="*" 的场景把词捞回来用的(beach/carpark)。三个并排写配置名会让人以为
  // default 是第三种东西, 实际它就是"共用", 只是没写下来。
  const 实际 = 实际模式(w), 明说 = w.模式 != null;
  const 模式文 = {'shared': 'shared', 'solo': 'solo'};
  const 模式解 = {
    'shared': 'One mesh for the whole class; every instance re-uses it.',
    'solo':   'Every instance gets its own mesh. Costs GPU per instance.'};
  const 模式钮 = MODES.map(m =>
    '<button data-mode="'+m+'" data-w="'+esc(w.词)+'"'+(作废?' disabled':'')+
      ' title="'+esc(模式解[m])+'" class="'+(实际===m?'on':'')+'">'+
      模式文[m]+'</button>').join('');

  d.innerHTML =
    '<img loading="lazy" src="'+U('/img/pick',
      {gen:封.gen,word:w.词,oid:oid,frame:fr,disk:封.disk?1:0})+'">'+
    '<div class="hd"><b>'+esc(w.词)+'</b>'+
      '<i class="bg cls'+(n>=5?' big':'')+'" title="Instances of this word that will '+
        'actually be placed. Cross-word duplicates are already excluded'+
        (丢数 ? ', and '+丢数+' you dropped' : '')+'.">'+
        n+' object'+(n===1?'':'s')+(w.网格账 ? '' : (丢数?' (−'+丢数+')':''))+'</i>'+
      '<span>#'+oid+' f'+fr+(有代表改?' ✎':'')+'</span></div>'+
    网格汇总(w)+
    (w.钉冲突 ? '<div class="warn">Different class sources are set on several rows — '+
               'open the instances and click set as rep on the one you want.</div>' : '')+
    (w.冲突 ? '<div class="warn">This word has both a solo row and a shared row — '+
              'it would land in SOLO and SOLO_KEEP at once, which is undefined downstream. '+
              'Click a mode button to clear it.</div>' : '')+
    '<div class="row">'+
      '<button data-yn="keep" data-w="'+esc(w.词)+'" class="'+(!作废?'on':'')+'">keep</button>'+
      '<button data-yn="drop-word" data-w="'+esc(w.词)+'" class="'+(作废?'on dgr':'')+'">drop word</button>'+
    '</div>'+
    '<div class="row">'+模式钮+'</div>'+
    '<div class="more">'+
      '<button data-exp="'+esc(w.词)+'"'+(作废?' disabled':'')+'>'+
        'show '+n+' instance'+(n===1?'':'s')+' ▾</button>'+
      // 换整类样本的入口只在展开面板里(set as rep / 样本上的 frame ▾)。以前词卡上还有个
      // mesh source ▾, 铺开全词所有实例的帧 —— 连去重删掉的、人 drop 掉的都在里面, 和样本规则对不上。
    '</div>';
  return d;
}

// 词卡上那一行: 这个词里有几个没有自己的 mesh、借了谁的; 几个被组装挡掉。问题红字。
function 网格汇总(w){
  const a = w.网格账;
  if (!a) return '';
  const 句 = [];
  const 挡文 = {vlm: k => k+' rejected by VLM — review',
               nobox: k => k+(k === 1 ? ' has' : ' have')+' no 3D box',
               size: k => k+' with an impossible box size',
               nomesh: k => k+' with no mesh at all',
               borrowdropped: k => k+' would borrow a mesh you dropped',
               vlmnomesh: k => k+' kept past the VLM but too small for an own mesh'};
  for (const [因, k] of Object.entries(a.挡)) 句.push('<span class="red">'+挡文[因](k)+'</span>');
  if (a.丢) 句.push(a.丢+' you dropped');      // 人的决定, 不算问题, 不标红
  for (const [名, k] of a.借)
    句.push('<span class="red">'+k+(k === 1 ? ' borrows ' : ' borrow ')+esc(短名(名))+'\'s mesh</span>');
  if (a.类兜) 句.push('<span class="red">'+a.类兜+(a.类兜 === 1 ? ' falls' : ' fall')+' back to the class mesh</span>');
  if (a.待) 句.push(a.待+' own mesh'+(a.待 > 1 ? 'es' : '')+' not generated yet');
  return 句.length ? '<div class="meshsum">'+句.join(' · ')+'</div>' : '';
}

// 召回卡。比词卡简单得多: 没有模式、没有代表、没有实例层, 只有 keep / drop。
function 召回卡(r, i){
  const d = document.createElement("div");
  d.className = "wcard" + (r.决定==='drop' ? " dead" : "");
  d.style.borderLeftColor = COL[r.决定] || COL.keep;
  d.innerHTML =
    '<img loading="lazy" src="'+U('/img/pick',
      {gen:r.生成名, word:r.词, oid:r.实例, frame:r.帧})+'">'+
    '<div class="hd"><b>'+esc(r.词)+'</b>'+
      '<i class="bg rec" title="'+RECALL_WHY+'">recalled</i>'+
      '<span>#'+r.实例+' f'+r.帧+'</span></div>'+
    '<div class="row">'+
      '<button data-rio="keep" data-i="'+i+'" class="'+(r.决定!=='drop'?'on':'')+'">keep</button>'+
      '<button data-rio="drop" data-i="'+i+'" class="'+(r.决定==='drop'?'on dgr':'')+'">drop</button>'+
    '</div>'+
    '<div class="more"><button disabled title="'+RECALL_WHY+'">no other frame</button></div>';
  return d;
}

// ---- 展开面板: 跨实例的对比视图 ----
// ★ 这和"换帧抽屉"是两个不同的东西。抽屉锁死在单个实例内换帧;
//   这一屏是横着比所有实例, 功能上更接近 tools/reuse_table.py。别合并。
function 展开面板(w){
  const t = document.createElement("div");
  t.className = "panel";
  const st = S.实例.get(w.词);
  // ★ 只在**还没有状态**时才触发加载。写成 `!st || 状态==='loading'` 就会死循环:
  //   取实例() 进门第一件事就是把状态设成 loading, 于是第二次 render 必然命中这一支,
  //   又调一次 取实例 -> 它发现已有缓存立即 resolve -> .then 再 render -> 无限循环,
  //   主线程被占满, 页面上所有点击都没反应。实测踩过(Playwright 点击 30s 超时)。
  if (!st){
    t.innerHTML = '<h3>'+esc(w.词)+' instances<span>loading…</span></h3>';
    取实例(w.词).then(() => render());
    return t;
  }
  if (st.状态 === 'loading'){
    t.innerHTML = '<h3>'+esc(w.词)+' instances<span>loading…</span></h3>';
    return t;                       // 加载已在飞, 它自己会 render
  }
  if (st.状态 === 'err'){
    t.innerHTML = '<h3>'+esc(w.词)+'<span style="color:var(--red)">cannot read instances: '+
      esc(st.说明||'')+'</span></h3>';
    return t;
  }
  const ins = st.数据;
  // 形状摘要: 只在"正常"的实例里看 AR 的散布, 坏 mask 不参与 —— 否则一个横躺的
  // 破 mask 会把结论带偏成"形状很多样, 该 solo"
  const 好 = ins.filter(r => !坏(r));
  const ars = 好.map(r => r.ar).sort((a,b)=>a-b);
  const 中 = ars.length ? ars[ars.length>>1] : 0;
  // ★ 散度不能用 max/min —— 一个横躺的实例就能把 0.28~0.69 的一族拉成 0.28~1.99,
  //   结论直接从"该 shared"翻成"该 solo"。实测 person 就是这样(#15 AR 1.99)。
  //   改成"离中位数 1.8 倍以外的算离群", 主族和离群分开报, 让人自己看那几个离群的
  //   到底是真的形状不同, 还是抠错了。
  const 离 = r => r.ar > 中*1.8 || r.ar < 中/1.8;
  const 群 = 好.filter(r => !离(r)), 异 = 好.filter(离);
  const 坏数 = ins.length - 好.length;
  const 建议 =
      ars.length < 2 ? 'only one usable instance'
    : 异.length === 0 ? 'shapes agree (AR≈'+中.toFixed(2)+', '+群.length+' of them) → shared is fine'
    : 异.length*2 >= 好.length
        ? 'shapes spread out (AR '+ars[0].toFixed(2)+'–'+ars[ars.length-1].toFixed(2)+
          ') → consider solo, or split into reuse groups'
        : 群.length+' share one shape (AR≈'+中.toFixed(2)+'), '+异.length+' outlier'+
          (异.length>1?'s':'')+' ('+异.map(r=>'#'+r.oid).join(' ')+
          ') → check whether those are just bad masks';
  // 整类样本(共用词才有)。它自己带质量警报 —— 这是最该被指出来的一种情况: 整类的 mesh 要拿它来建
  const 样 = 当前样本(w);
  const 代 = 样 && ins.find(r => r.oid === 样.oid);
  const 代警 = 代 ? 警报(代) : [];

  // ★ 面板列的是**全部**实例(包括被去重淘汰的, 要让人看得见它们存在),
  //   而卡片上的数字是**会进场景的**。两个数不一样, 所以这里必须把账算给人看,
  //   否则卡片说 10 个、面板列 13 个, 人不知道该信哪个。
  // 有组装预告就按它的"因"记账(每个不摆的实例只记一个因, 加起来正好等于差额)
  const 因计 = {};
  if (w.去向) for (const v of w.去向.values()) if (!v.摆) 因计[v.因] = (因计[v.因] || 0) + 1;
  const 淘 = w.去向 ? (因计.dedup || 0) : ins.filter(r => r.去重淘汰).length;
  const 丢 = w.去向 ? (因计.drop || 0) : (w.已drop || 0);
  const 其他 = [['vlm', ' rejected by VLM (review them)'], ['vlmnomesh', ' kept past the VLM, too small'],
               ['nobox', ' without a 3D box'], ['size', ' with an impossible box size'],
               ['borrowdropped', ' borrowing a dropped mesh'], ['nomesh', ' with no mesh']]
    .filter(([k]) => 因计[k]).map(([k, t]) => '<span class="red">'+因计[k]+t+'</span>');
  const 账 = 淘||丢||其他.length
    ? ' · '+ins.length+' here, '+w.实例数+' will be placed ('+
      [淘?淘+' taken by another word':'', 丢?丢+' you dropped':'', ...其他].filter(Boolean).join(', ')+')'
    : '';
  const 自样 = 自动样本(w.词);
  // 整类样本本身不该被挡(人钉的 drop 会一起撤掉, 只有文件里带来的 / 源码表钉的会这样)
  const 样F = 样 && ((S.网格事实 || {})[w.词] || {实例: []}).实例.find(x => x.oid === 样.oid);
  const 样坏 = !样 || 可当样本(w.词, 样.oid) ? ''
    : (样F && 样F.去重淘汰) ? 'dedup gave it to another word'
    : 人决(w.词, 样.oid) === 'drop' ? 'you dropped it' : 'the VLM rejected it';
  // ★ 弹层头已经写了词名和数量(见 挂弹层 的 sub), h3 再说一遍就是 40px 内说两次。
  //   这里只留"怎么读这一屏"的说明。顺带修复 "1 instances" 的复数。
  t.innerHTML = '<h3>'+(ins.length===1?'One instance':'Largest first')+账+
    '<span>'+建议+
    (坏数?'. '+坏数+' mask'+(坏数>1?'s':'')+' look wrong — those carry a ⚠ flag':'')+
    '</span></h3>'+
    (w.钉冲突 ? '<div class="warn">Different class sources are set on several rows. '+
      'Only one can be used — click set as rep on the one you want.</div>' : '')+
    (样坏 ? '<div class="warn">The class mesh is built from #'+样.oid+', but '+样坏+
      '. Click set as rep on another instance.</div>' : '')+
    (代警.length ? '<div class="warn">The class source #'+代.oid+
      ' is itself flagged ('+esc(代警.join('; '))+') — and the mesh for this whole class '+
      'is built from it.'+(自样 && 自样.oid !== 代.oid
        ? ' The top-scoring one is #'+自样.oid+'. Worth switching.' : '')+'</div>' : '')+
    '<div class="insts"></div>';
  const box = t.querySelector(".insts");

  ins.forEach(r => {
    r.召回 = r.oid >= 10000;
    const 行i = w.实例行.has(r.oid) ? w.实例行.get(r.oid) : null;
    // ★ 闭环规则: 叉 = 不进场景, 亮 = 进场景, 按钮高亮永远和叉一致。所以 决 必须是
    //   **组装时真实会发生的事**, 不是"表里写了什么":
    //   * 去重淘汰: 下游根本不出盒子(boxes_from_masks 直接 continue), 表里写什么都不进场景。
    //     界面也没有撤销去重的通路 —— 所以固定为 drop, 按钮禁用, 不给一个点了没用的 keep。
    //   * VLM 否决: 只挡生成, 不挡组装 —— 没 drop 就借同词别人的 mesh 照样摆。所以按表/历史走,
    //     默认是 keep(亮), 不能打叉。
    //   * 早先闸口 drop 过、行已被 review_gate 删掉的: 回退查历史决定。
    const 旧drop = 行i==null && 历史drop(w.词).includes(r.oid);
    const 决 = r.去重淘汰 ? 'drop'
             : 行i!=null ? S.行[行i].决定 : (旧drop ? 'drop' : 'keep');
    // 组装预告(有盒子文件时才有)。叉 = 不摆, 以它为准; 没有预告时退回只看人的决定。
    const 去 = w.去向 ? w.去向.get(r.oid) : null;
    const 不摆 = 去 ? !去.摆 : 决 === 'drop';
    // 按钮高亮跟叉走(闭环): 不摆就亮 drop —— 哪怕人点的是 keep、是没盒子/尺寸把它挡掉的。
    // 那种情况按钮是锁着的, 原因写在卡片上和 title 里。
    const 亮 = 不摆 ? 'drop' : 'keep';
    const 去文 = 去 ? 去向文(去)[0] : '';
    // ★ solo 模式下没有"代表"这回事 —— 每个实例各造各的 mesh, 类 mesh 根本不会被读。
    //   第一版只判"有没有类行", 而点 solo 只是把类行的决定改成 solo, 行还在,
    //   于是徽章赖着不走, 人以为自己没点上。
    // 没人钉过代表时, 标出**流水线自己会挑哪个** —— 否则人看不出"我不选的话会怎样",
    // 而这恰恰是最该看见的: 不选也是一种选择, 只是选的人不是你。
    // 判据用 样本分 最高(和 pick_from_masks 的打分同源), 面板本来就按它降序排,
    // 所以就是第一张。标成虚线徽章, 和"钉死的"区分开。
    // 共用词: 标出整类样本是谁、从哪来(人钉 / 源码表 / 盘上机器挑的 / 保存后重挑的预测)。
    //   没人钉的时候也要标 —— 不选也是一种选择, 只是选的人不是你。
    const 是代表 = !!(样 && 样.oid === r.oid);
    const 警 = 警报(r);
    // 卡片画哪一帧: solo 画人换的那帧(没换 = 服务端的代表帧); 共用词的样本画它用的那帧,
    // 其余画它分最高的那帧 —— 就是点 set as rep 会用的那帧
    const 选帧 = 实际模式(w) === 'solo' ? 人选帧(w, r.oid) : null;
    const 显帧 = 实际模式(w) === 'solo' ? 选帧 : (是代表 ? 样.fr : (r.召回 ? null : 最佳帧(w.词, r.oid)));
    // 不摆的原因跟人的决定无关时(去重 / 没盒子 / 盒子尺寸离谱 / 借的被 drop 了), keep/drop
    // 都锁住: 点了不改变任何结果, 给了就是假按钮。
    // 'vlm'(VLM 否决等人判)不锁 —— 那正是要人点 keep / drop 的地方
    const 锁因 = 去 ? (!去.摆 && !['drop', 'vlm'].includes(去.因) ? 去.因 : null) : (r.去重淘汰 ? 'dedup' : null);
    const 锁 = 锁因 ? ' disabled title="'+esc({
        dedup: 'Removed by dedup: another word already represents this object, so it is never placed.',
        word: 'The whole word is dropped.',
        nobox: 'No 3D box in the current boxes file, so assembly has nothing to place.',
        size: 'Assembly rejects this box size.',
        nomesh: 'No mesh exists for this word.',
        borrowdropped: 'It would borrow a mesh you dropped. Un-drop that one, or give this one its own mesh.',
        vlmnomesh: 'Too small for its own mesh, and a VLM-rejected instance never borrows one.',
      }[锁因]+' Keep/drop cannot change that here.')+'"' : '';
    const el = document.createElement("div");
    // 被 drop 的: 灰白 + 红框。★ 和"质量可疑"的降饱和要分得开 ——
    // 那个是"这张图信不过"(黄徽章), 这个是"你判了它不进场景"(红框), 两码事。
    el.className = "inst" + (警.length ? " bad" : "") + (是代表 ? " rep" : "")
                 + (不摆 ? " dropped" : "");
    el.dataset.key = w.词 + ':' + r.oid;      // ?sel= 跳过来时靠它找到这张卡
    // AR 画成小矩形字形 —— 数字要读, 图形一扫就分堆
    const bw = Math.round(Math.min(Math.max(r.ar,0.15),4) * 14), bh = 14;
    el.innerHTML =
      (() => { const q = 显帧 != null ? {word:w.词, oid:r.oid, frame:显帧} : {word:w.词, oid:r.oid};
        return '<div class="pics">'+
          '<img loading="lazy" src="'+U('/img/inst', {...q, size:120})+'">'+
          '<a href="'+U('/img/frame', {...q, size:1280})+'" target="_blank" title="Open the full frame">'+
            '<img class="loc" loading="lazy" src="'+U('/img/frame', {...q, size:520})+'"></a>'+
        '</div>'; })()+
      (警.length ? '<i class="alarm" title="'+esc(警.join('；'))+'">⚠ '+esc(警[0])+'</i>' : '')+
      (是代表 ? 样徽章(w, 样) : '')+
      '<div class="im">'+
        '<b>#'+r.oid+'</b><em>'+r.中位面积pct+'%</em>'+
        '<div class="bar"><s style="width:'+
           Math.round(100*r.中位面积pct/Math.max(ins[0].中位面积pct,1e-6))+'%"></s></div>'+
        '<div class="glyph"><u style="width:'+bw+'px;height:'+bh+'px"></u>'+
          'AR '+r.ar.toFixed(2)+' · fill '+Math.round(r.fill*100)+'%</div>'+
        // 上一行是量(帧数/帧号), 下一行是**结论**。原来四样挂在同一根中点轨道上,
        // 人读不出 "below solo threshold" 是个判定而不是第四个字段。
        '<div class="dim">alive '+r.活帧数+' frames · '+
          (选帧 != null ? '<b class="picked">f'+选帧+' you picked</b> (auto f'+r.代表帧+')'
           : 是代表 ? '<b class="picked">f'+样.fr+' builds the class mesh</b>'
             + (样.fr !== 最佳帧(w.词, r.oid) ? ' (best f'+最佳帧(w.词, r.oid)+')' : '')
           : 'f'+(显帧 != null ? 显帧 : r.代表帧))+'</div>'+
        (去
          ? '<div class="dim verdict">'+去文+'</div>'
          : ((r.去重淘汰||r.VLM否决||r.低于门槛||旧drop)
            ? '<div class="dim verdict">'+[
                (旧drop && !r.去重淘汰)?'you dropped this at an earlier gate':'',
                r.去重淘汰?'removed by dedup — another word represents it':'',
                r.VLM否决?'VLM rejected':'',
                r.低于门槛?'below solo threshold':''].filter(Boolean).join('; ')+'</div>'
            : ''))+
      '</div>'+
      '<div class="iops">'+
        '<button data-io="keep" data-w="'+esc(w.词)+'" data-oid="'+r.oid+'"'+
          ' class="'+(亮==='keep'?'on':'')+'"'+锁+'>keep</button>'+
        '<button data-io="drop" data-w="'+esc(w.词)+'" data-oid="'+r.oid+'"'+
          ' class="'+(亮==='drop'?'on':'')+'"'+锁+
          (行i==null && !r.去重淘汰 ? ' title="Adds a row for this instance so it can be excluded '+
                       'from the scene (assemble_scene checks instance-level names too)"' : '')+
          '>drop</button>'+
        实例钮(w, r) +
      '</div>';
    box.appendChild(el);
  });
  return t;
}

// ---- 落点表: 哪些按钮在哪个模式下才有意义 ----
// 依据(硬约束, 不是审美):
//   默认模式下逐实例换帧 = 零消费者。_ov_key 生成「词#实例」键, 而 overrides() 明确
//     过滤掉带 # 的(review_decisions.py:114), gen_per_instance 又不遍历这个词。
//   shared 下换帧塌成裸词键(review_gate.py:386,393-394), 多个实例各填各的会落进
//     同一个字典键, 后者静默压掉前者, 谁赢取决于插入顺序 —— 所以做成全词单选。
//   solo 下没有类 mesh, overrides() 那条根本不会被读, 所以「设为代表」无意义。
// ★ 召回行(实例号 >= 10000)不能填「改用」—— review_gate.py:399-404 是**硬停**:
//     "召回候选不支持改用: 它的抠图是按检测框现抠的, 没有别的帧可换。
//      这一条要么 keep 要么 drop; 觉得抠得不对就 drop。"
//   人填了 -> 保存 -> --apply 当场 SystemExit 退出码 2。所以按约定 8, 禁用并说明。
const RECALL_WHY = "Recalled candidates cannot take a source override — their cutout was "
                 + "made on the fly from the detection box, there is no other frame to switch "
                 + "to. review_gate.py refuses this outright (exit 2). Keep it or drop it.";

// 整类样本的徽章: 实线 = 定了的(人钉 / 源码表 / 盘上机器挑的), 虚线 = 保存后要重挑的预测
function 样徽章(w, 样){
  const t = {
    '人':   ['class rep', 'You picked this. The one mesh every '+w.词+' shares is built from f'+样.fr+' of it.'],
    '配置': ['class rep (config)', 'Pinned in the pipeline source (PICK_OVERRIDE_BY_SCENE), applied on every run. '+
             'Click set as rep on another instance to replace it.'],
    '机器': ['class rep (auto)', 'The pipeline picked this: top scores, then a VLM check chose among them. '+
             'Nothing is pinned. Click set as rep on another instance to replace it.'],
    '预测': ['class rep (predicted)', 'Your decisions changed which instances may be used, so the pipeline '+
             're-picks after Save — and its VLM check may choose a different one. This is the top-scoring one. '+
             'Click set as rep to lock it.'],
  }[样.来源];
  return '<i class="repbadge'+(样.来源 === '预测' ? ' guess' : '')+'" title="'+esc(t[1])+'">'+t[0]+'</i>';
}

// 实例卡上除 keep/drop 以外的按钮。★ 先看模式, 两种模式下是两件事:
//   solo   -> frame ▾      这个实例自己的 mesh 用哪帧(写它自己的行, overrides_inst)
//   共用   -> set as rep   整类样本换成它, 用它分最高的那帧 —— 一下就好, 不用挑图
//             frame ▾      只在整类样本那个实例上: 嫌那帧不好再换一帧
//   以前共用词每个实例都给 use as source ▾, 换样本必须自己挑一帧; 再早还有个名字按"表里有没有行"
//   变来变去的版本。两步拆开: 选谁(一下) 和 选哪帧(要看图) 是两个问题。
// 去重删掉的不给按钮(不进场景, 也不许当样本), 原因卡片上写着。
function 实例钮(w, r){
  if (r.去重淘汰) return '';
  const solo = 实际模式(w) === 'solo';
  if (r.召回) return '<button disabled title="'+RECALL_WHY+'">'+(solo ? 'frame' : 'set as rep')+'</button>';
  const 开 = openTray && openTray.词===w.词 && openTray.oid===r.oid;
  const 帧钮 = tip => '<button data-tray="'+esc(w.词)+'" data-toid="'+r.oid+'" title="'+esc(tip)+'">'+
                     (开 ? 'close ▴' : 'frame ▾')+'</button>';
  if (solo)
    return 帧钮('Pick which frame of #'+r.oid+' builds its own mesh'+
               (r.VLM否决 ? '. Picking one also overrides the VLM rejection' : ''));
  const 样 = 当前样本(w);
  const 是 = 样 && 样.oid === r.oid;
  const 最 = 最佳帧(w.词, r.oid);
  const 设钮 = 最 == null ? ''
    : 人决(w.词, r.oid) === 'drop'
      ? '<button disabled title="You dropped #'+r.oid+'. Keep it first to use it as the class source.">set as rep</button>'
      : '<button data-rep="'+esc(w.词)+'" data-oid="'+r.oid+'" title="'+esc(
          'Build the one mesh every '+w.词+' shares from #'+r.oid+', using its best frame (f'+最+'). '+
          'Change the frame afterwards with frame ▾.'+
          (r.VLM否决 && !放行了(w.词, r.oid) ? ' This also overrides the VLM rejection.' : ''))+'">set as rep</button>';
  if (!是) return 设钮;
  // 样本本人: 换帧; 预测出来的那个再给一个 set as rep, 点了就钉住(免得重挑时 VLM 换人)
  return (样.来源 === '预测' ? 设钮 : '') +
         帧钮('#'+r.oid+' is the class source. Pick which of its frames builds the mesh every '+w.词+' shares');
}

// ---- 质量警报: 这张图信不信得过 ----
// 和"形状差异"分开成两个通道 —— 形状用几何(矩形字形/面积条), 质量用徽章。
// 徽章的意思是"这张图本身信不过, 别拿它比形状"。
function 警报(r){
  const a = [];
  if (r.抖动 >= 4)      a.push('jitter '+r.抖动+'x');   // 游走的垃圾 mask 的指纹
  if (r.fill < 0.25)    a.push('fill '+Math.round(r.fill*100)+'%');
  if (r.贴边中位 > 0.35) a.push('edge '+r.贴边中位);
  if (r.活帧数 < 5)      a.push('only '+r.活帧数+' frames');
  if (r.去重淘汰)        a.push('dropped by dedup');
  if (r.VLM否决)         a.push('VLM says this is not it');   // 最重: 根本不会生成
  return a;
}
const 坏 = r => 警报(r).length > 0;

function render(){
  document.querySelectorAll(".scrim").forEach(e => e.remove());   // 弹层每次重建
  画闸切换();
  画召回钮();
  // 闸 4 一进来自动把召回表推到脸上一次。★ 必须放在 M 判断之前挡一道:
  //   人正开着别的弹层时把它顶掉, 比看不见召回更烦。
  if (!RC弹过 && !M && GATE === 'review' && S.meta.闸 === 4 &&
      RC && RC.状态 !== 'loading' && (RC.判词||[]).length){
    RC弹过 = true; M = {view:'recall'};
  }
  if (GATE === 'scale') return scRender();     // 闸 3 走自己那套, 共用外壳
  const g = document.getElementById("grid");
  g.innerHTML = "";
  const 词们 = [...S.词.values()];
  const 类 = 词们.filter(w => w.类行 != null);
  const 纯实例 = 词们.filter(w => w.类行 == null);
  类.sort((a,b) => b.实例数 - a.实例数);      // 分量最重的排最前

  const sec = (t, n) => { const h=document.createElement("div");
    h.className="sec"; h.innerHTML=t+'<span>'+n+'</span>'; g.appendChild(h); };
  const emit = ws => ws.forEach(w => { g.appendChild(词卡(w)); });

  if (类.length){
    const tot = 类.reduce((s,w)=>s+w.实例数, 0);
    sec("Class representatives, "+类.length+" word"+(类.length>1?"s":""),
        "One word makes one mesh, shared by every object of that class. "+tot+
        " objects sit behind them — open one to see what they look like.");
    emit(类);
  }
  if (纯实例.length){
    sec("Per-instance, "+纯实例.length+" word"+(纯实例.length>1?"s":""),
        "every instance gets its own mesh");
    emit(纯实例);
  }
  // ★ 召回区。这些行原来收进了 w.召回行 却**没有任何地方渲染它** ——
  //   它们既不是类行也不是实例行, 卡片和面板都看不见, 于是闸 4 上召回捡回来的候选
  //   整批消失、人根本审不到。而召回恰恰是"主线漏掉的、质量最没保证"的那一批
  //   (review_gate 给它们在联络图上单独标黄边就是这个理由)。
  const 召 = [];
  S.词.forEach(w => w.召回行.forEach(i => 召.push(i)));
  if (召.length){
    sec("Recalled, "+召.length,
        "picked up by WildDet3D where SAM3 missed them — the least trustworthy batch, "+
        "worth a look. Keep or drop only: their cutout has no other frame to switch to");
    召.sort((a,b) => (S.行[b].面积占比||0) - (S.行[a].面积占比||0));
    召.forEach(i => g.appendChild(召回卡(S.行[i], i)));
  }

  g.querySelectorAll("[data-mode]").forEach(b => b.onclick = () => 设词模式(b.dataset.w, b.dataset.mode));
  g.querySelectorAll("[data-yn]").forEach(b => b.onclick = () => 设词要不要(b.dataset.w, b.dataset.yn));
  g.querySelectorAll("[data-exp]").forEach(b => b.onclick = () => {
    取实例(b.dataset.exp).then(()=>render());
    开弹层({view:'insts', 词:b.dataset.exp, oid:null});
  });
  绑事件(g);
  if (M) 挂弹层();
  stat();
}

// 弹层里的按钮不在 #grid 底下, 得单独绑一次 —— 两处用同一个函数, 免得漂移。
function 绑事件(root){
  root.querySelectorAll("[data-rio]").forEach(b => b.onclick = () => {
    S.行[+b.dataset.i].决定 = b.dataset.rio; dirty = true; 重建索引(); render();
  });
  root.querySelectorAll("[data-rep]").forEach(b => b.onclick = () =>
    设样本(b.dataset.rep, +b.dataset.oid, 最佳帧(b.dataset.rep, +b.dataset.oid)));
  root.querySelectorAll("[data-io]").forEach(b => b.onclick = () =>
    设实例决定(b.dataset.w, +b.dataset.oid, b.dataset.io));
  root.querySelectorAll("[data-tray]").forEach(b => b.onclick = () => {
    const 词 = b.dataset.tray;
    const oid = b.dataset.toid == null ? null : +b.dataset.toid;
    openTray = {词, oid};
    开弹层({view:'src', 词, oid, 从: (M && M.view==='insts') ? 'insts' : null});
  });
}

// ★ 弹层承载的是**已有的**展开面板/帧抽屉两个组件, 不另写一套 ——
//   那两个组件的逻辑(落点、禁用规则、撤销语义)是这个工具里最容易错的部分,
//   为了换个容器复制一份出来, 等于把最难的地方复制了两遍。
function 画闸切换(){
  const el = document.getElementById("gsw");
  if (!el) return;
  el.innerHTML = [['review','Candidates'], ['scale','Scale']]
    .map(([k,t]) => '<button data-gate="'+k+'" class="'+(GATE===k?'on':'')+'">'+t+'</button>').join('');
  el.querySelectorAll("[data-gate]").forEach(b => b.onclick = () => {
    if (GATE === b.dataset.gate) return;
    // ★ 切闸前拦一下没保存的改动 —— 两道闸写的是不同的文件, 切过去就看不见了
    const 脏 = GATE==='review' ? dirty : scDirty();
    if (脏 && !confirm("You have unsaved changes on this gate. Switch anyway?")) return;
    GATE = b.dataset.gate; M = null; openTray = null;
    if (GATE === 'scale' && !SC.state) scLoad(); else render();
  });
}

// ================================================================ 召回信息表
// 闸 4 的一半价值在这儿。候选表只装活下来的, 而召回专挑主线漏掉的东西 —— 漏掉的
// 多半被前景挡着, 补抠 mask 天生就容易失败。实测 pedestrian/r2 是「找到 3 个,
// 活下来 0 个」, 于是这道叫"召回"的闸上, 召回找到了什么人一眼都看不见。
// 这一屏只读: 列全部, 标结局, 给图。它不改任何决定 —— 没进候选表的行下游收不了,
// 给个 keep 按钮是骗人。人在这儿只该看见, 然后自己决定要不要去松阈值重跑。

function rcLoad(){
  if (RC && RC.状态 === 'loading') return;       // 已在飞, 别叠
  RC = {状态:'loading', 判词:[], 统计:{}};
  fetch(U('/api/recall')).then(r => r.json())
    .then(d => { RC = d; render(); })
    .catch(e => { RC = {有:false, 判词:[], 统计:{},
      说明:'cannot reach the server — is gate_ui still running? '+e}; render(); });
}

// 结局 -> 颜色。绿=活下来, 橙=被阈值挡的(阈值是拍的, 可以松), 红=模型没给东西(松也没用)
// ★ 直接读 CSS 里那五个变量, 别在这儿抄一份 —— 抄一份就会漂: 上一版这里的
//   #5ab45a / #f0963c / #dc3c3c 和 --keep / --drop / --kill 是三对不同的色号,
//   同一个含义在两个面板里显示成两种颜色。
const _c = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const RC色 = () => ({'进表':_c('--keep'), '抠出来太小':_c('--drop'), '框对不上':_c('--drop'),
              '框太窄':_c('--kill'), 'SAM3没给mask':_c('--kill'), '未知':_c('--pencil')});
// 结局 -> 一句人话。★ 说的是**下一步该怎么办**, 不是复述字段名。
// 结局 -> 界面上的英文短标。★ 界面文案一律英文(JSON 字段名照旧中文, 那是数据不是文案)。
const RC标 = {
  '进表':'kept', '抠出来太小':'mask too small', '框对不上':'box mismatch',
  '框太窄':'box too thin', 'SAM3没给mask':'no mask returned', '未知':'unrecorded',
};
const RC话 = {
  '进表':        'made it into the candidate table below',
  '抠出来太小':  'SAM3 did cut a mask, but it came out smaller than the size floor',
  '框对不上':    'SAM3 cut something, but it does not line up with the box recall gave',
  '框太窄':      'the 2D box is only a few pixels wide — nothing to prompt with',
  'SAM3没给mask':'SAM3 returned nothing at all for that box',
  '未知':        'this run predates per-item verdicts — reason was never recorded',
};

// 卡在门槛边上的那些, 值得人自己看一眼 —— 阈值是拍出来的, 不是真理。
// 实测 pedestrian/r2: potted plant 抠出 0.149% vs 门槛 0.150%, 差 0.001 被判死。
// r 是一条判词, 门槛 是 {REC_MIN_PCT, REC_MIN_IOU}。返回 true = 在表上打个"borderline"。
function 擦边(r, 门槛){
  // 用**比例**不用绝对差 —— 两个门槛量纲不同(抠出pct 是面积百分比, 框重合 是 0~1 的
  // IoU), 绝对差在两边不可比: 差 0.02 对 0.15 的门槛是擦边, 对 0.3 的门槛是差远了。
  // 带宽 ±30%: 再宽就全打上标, 标签本身就没信息了。
  const 近 = (v, th) => v != null && th != null && th > 0 &&
                        v >= th * 0.7 && v <= th * 1.3;
  // 每条判词自己带的 门槛 优先 —— 它记的是**当时那一跑**用的值, 顶栏那份是现在的。
  // 两者不一致时(改了阈值还没重跑)照当时的算, 否则标出来的擦边是对不上号的。
  const thP = (r.结局 === '抠出来太小' && r.门槛 != null) ? r.门槛 : 门槛.REC_MIN_PCT;
  const thI = (r.结局 === '框对不上'   && r.门槛 != null) ? r.门槛 : 门槛.REC_MIN_IOU;
  // 压线**过关**的也标: 侥幸进表和差点没进, 同样值得人自己看一眼。
  return 近(r.抠出pct, thP) || 近(r.框重合, thI);
}

function 召回表(){
  const t = document.createElement("div");
  t.className = "panel";
  if (!RC){ rcLoad(); t.innerHTML = '<div class="rcsum">loading…</div>'; return t; }
  if (RC.状态 === 'loading'){ t.innerHTML = '<div class="rcsum">loading…</div>'; return t; }
  const 判 = RC.判词 || [], 门槛 = RC.门槛 || {}, 统 = RC.统计 || {};
  if (!判.length){
    t.innerHTML = '<div class="rcsum">'+esc(RC.说明 || 'Recall produced nothing this run.')+
      '<br>Nothing was found, so nothing could be lost.</div>';
    return t;
  }
  const 活 = 统['进表'] || 0;
  let h = '<div class="rcsum"><b style="color:var(--fg)">'+判.length+
    ' object'+(判.length>1?'s':'')+' recalled</b> — found by WildDet3D, missed by SAM3 tracking. '+
    活+' made it into the candidate table, '+(判.length-活)+' did not.<br>'+
    'Everything below is listed either way, so nothing dies silently. '+
    'The rows that did not make it are <b>not editable here</b> — with no mask, '+
    'nothing downstream can consume them.';
  if (门槛.REC_MIN_PCT != null)
    h += '<br>Thresholds this run: <code>REC_MIN_PCT='+门槛.REC_MIN_PCT+
         '</code> (mask must cover this % of frame) · <code>REC_MIN_IOU='+
         门槛.REC_MIN_IOU+'</code> (cut mask vs the box recall gave)';
  // 显式 > 猜: 读的是哪一份必须看得见。文件名亮、全路径进 title, 和顶栏同一套做法 ——
  // 只留 basename 会让人分不清是这一轮的还是上一轮遗留的。
  h += '<div class="src" title="'+esc(RC.文件||'')+'">read from '+
       esc(RC.文件||'(nothing)')+'</div></div>';
  t.innerHTML = h;

  判.forEach(r => {
    const 色表 = RC色(); const 结 = r.结局 || '未知', c = 色表[结] || _c('--pencil');
    const 边 = 擦边(r, 门槛);
    const 数 = [];
    if (r.抠出pct != null) 数.push('cut '+r.抠出pct+'% (floor '+(r.门槛!=null?r.门槛:门槛.REC_MIN_PCT)+'%)');
    if (r.框重合 != null)  数.push('IoU '+r.框重合);
    if (r.框宽 != null)    数.push('box '+r.框宽+'x'+r.框高+'px');
    if (r.见帧数 != null)  数.push('seen in '+r.见帧数+' frames');
    if (r.分)              数.push('det score '+r.分);
    const d = document.createElement("div");
    d.className = "rcrow";
    d.innerHTML =
      '<img loading="lazy" width="230" src="'+U('/img/recall',{oid:r.obj_id, size:230})+'">'+
      '<div class="t">'+
        '<b>'+esc(r.词||'?')+'</b>'+
        '<span style="color:var(--dim);font-size:11px"> #'+r.obj_id+' · frame '+r.最佳帧+'</span>'+
        '<i class="rcv" style="border:1px solid '+c+';color:'+c+'">'+esc(RC标[结] || 结)+'</i>'+
        (边 ? '<i class="rcedge" title="This one sits right on the threshold — '+
              'worth looking at yourself before accepting the machine’s call.">borderline</i>' : '')+
        '<div class="why">'+esc(RC话[结] || 结)+'</div>'+
        '<div class="num">'+esc(数.join('  ·  '))+'</div>'+
      '</div>';
    t.appendChild(d);
  });

  const 死 = 判.length - 活;
  if (死 > 0){
    const f = document.createElement("div");
    f.className = "rcsum";
    f.style.cssText = 'margin:14px 0 0;padding-top:12px;border-top:1px solid var(--line)';
    const sc = esc(S.meta.场景名||''), rn = esc(S.meta.轮次||'');
    f.innerHTML = 'Want the rejected ones back? The thresholds are judgement calls, not '+
      'truth — loosen them and re-cut. This step does not touch the GPU (~1 min):'+
      '<code class="cmd">REC_MIN_PCT=0.03 REC_MIN_IOU=0.15 python pipeline/outdoor_pipeline.py'+
      ' --scene '+sc+' --run '+rn+' --only recall_mask</code>'+
      'Then re-run the gate-4 check so the survivors land in the candidate table, '+
      'and reload this page.<br>'+
      '<span style="color:var(--dim)">This console cannot run any of it — it never spawns '+
      'pipeline processes. That is the one rule that keeps it from being able to break '+
      'your run.</span>';
    t.appendChild(f);
  }
  return t;
}

function 画召回钮(){
  const el = document.getElementById("rcw");
  if (!el) return;
  // 只在闸 4 的候选视图上出现 —— 闸 2 那时候召回还没跑, 摆个空按钮是噪音
  if (GATE !== 'review' || S.meta.闸 !== 4){ el.innerHTML = ''; return; }
  if (!RC){ rcLoad(); }
  const n = (RC && RC.判词) ? RC.判词.length : null;
  const 活 = (RC && RC.统计) ? (RC.统计['进表'] || 0) : 0;
  el.innerHTML = '<button data-rcopen="1" class="rcb">Recall'+
    (n == null ? '…' : ' '+活+'/'+n+' kept')+'</button>';
  el.querySelector('[data-rcopen]').onclick = () => 开弹层({view:'recall'});
}

function 挂弹层(){
  // 召回表是第三种 view: 它**不属于任何一个词**, 所以不能走 S.词.get 那条路。
  const 是召 = M.view === 'recall';
  const w = 是召 ? null : S.词.get(M.词);
  if (!是召 && !w) { M = null; return; }
  const 是源 = M.view === 'src';
  const scrim = document.createElement("div");
  scrim.className = "scrim";
  scrim.onclick = e => { if (e.target === scrim) 退一层(); };   // 点背景退一层
  const box = document.createElement("div");
  box.className = "modal";
  const sub = 是召
    ? 'what recall found this run — kept and rejected alike, read-only'
    : 是源
    ? (M.oid!=null ? '#'+M.oid+' — the cutout you pick here is what SAM3D builds from'
                   : 'any frame of any instance — the cutout you pick is what SAM3D builds from')
    : w.实例数+' will be placed. Keep or drop each one, or pick which builds the mesh.';
  box.innerHTML =
    '<div class="mhd">'+
      (是源 && M.从 ? '<button id="mback">‹ back</button>' : '')+
      '<b>'+esc(是召 ? 'Recall' : M.词)+'</b><span class="sub">'+esc(sub)+'</span>'+
      '<span class="sp"></span><button class="mclose" id="mx">✕</button>'+
    '</div><div class="mbody"></div>';
  const body = box.querySelector(".mbody");
  body.appendChild(是召 ? 召回表() : 是源 ? tray(w, M.oid) : 展开面板(w));
  scrim.appendChild(box);
  document.body.appendChild(scrim);
  box.querySelector("#mx").onclick = 退一层;
  // 焦点落进弹层。放在挂完之后 —— 元素得先在文档里才 focus 得上。
  requestAnimationFrame(() => { const x = box.querySelector("#mx"); if (x) x.focus(); });
  const bk = box.querySelector("#mback");
  if (bk) bk.onclick = () => { openTray = null; 开弹层({view:'insts', 词:M.词, oid:null}); };
  绑事件(box);
}

// 帧抽屉: 只看一个实例的帧。★ 写哪一行由**词的模式**决定, 不由从哪儿点开决定:
//     solo           -> 写这个实例自己的行   overrides_inst(): 键 (词,实例号), 值只取帧
//     default/shared -> 交给 设样本           overrides():      键 词,      值 (实例号,帧号)
//   两条通道的键形状不同, 写错一边就静默失配 —— solo 下写类行等于给一个没有类 mesh 的词
//   设样本(没人读), default 下写实例行等于填一个带 # 的键(overrides 明确过滤掉)。
function tray(w, oid){
  const t = document.createElement("div");
  t.className = "tray";
  if (oid == null){
    t.innerHTML = '<h3>'+esc(w.词)+'<span>open an instance to pick its frames</span></h3>';
    return t;
  }
  const solo = 实际模式(w) === 'solo';
  const 词 = w.词;
  // 两种模式各有自己的"现状"和"原选":
  //   solo  现状 = 这个实例行上的 改用, 没有就是 pick 给它挑的那帧(原选); 点原选/点现状 = 撤销
  //   共用  现状 = 整类样本(只在样本本人身上打开); 原选 = 这个实例分最高的那帧(标 best)
  //         ★ 不能拿这个实例自己的行当现状: 那行的帧是它逐实例时挑的, 不是整类的图
  let cur, orig, row = null;
  if (solo){
    const i = w.实例行.has(oid) ? w.实例行.get(oid) : null;
    if (i != null) row = S.行[i];
    else {
      // 这个实例表里还没行: 先按它自己的代表帧"虚拟"一行, 真点选了再建
      const m = ((S.实例.get(词) || {}).数据 || []).find(x => x.oid === oid);
      row = {实例: oid, 帧: m ? m.代表帧 : 0, 虚: true};
    }
    cur = row.改用 || [row.实例, row.帧];
    orig = [row.实例, row.帧];
  } else {
    const 样 = 当前样本(w);
    cur = 样 && 样.oid === oid ? [样.oid, 样.fr] : null;
    const 最 = 最佳帧(词, oid);
    orig = 最 != null ? [oid, 最] : null;
  }
  t.innerHTML = '<h3>'+esc(词)+' frames<span>loading…</span></h3>';

  const q = {word: 词, only_oid: oid};
  if (cur) { q.oid = cur[0]; q.frame = cur[1]; }
  if (orig) q.best = orig[1];
  // ★ .catch 不能省: 服务没起/被杀掉时 fetch 直接 reject, 没有 catch 的话
  //   这个抽屉会**永远**停在"loading…", 人分不清是慢还是死了。实测踩过。
  fetch(U("/api/cands", q))
   .then(r => r.json())
   .then(cs => {
    if(cs.error || !cs.length){
      t.innerHTML = '<h3>'+esc(词)+'<span>no candidates'+
        (cs.error?': '+esc(cs.error):'')+'</span></h3>'; return; }
    t.innerHTML = '<h3>'+esc(词)+' frames<span>'+
      (solo ? 'only #'+oid+'\'s own frames, for its own mesh. '+cs.length+' sampled, green is in use. '+
              'Click another to switch, click the green one to go back to the pipeline pick'
            : 'frames of #'+oid+' — the one you pick builds the mesh every '+esc(词)+' shares. '+
              cs.length+' sampled, green is in use, "best" is its top-scoring frame')+
      '</span></h3><div class="cands"></div>';
    const box = t.querySelector(".cands");
    cs.forEach(c => {
      const isCur  = !!cur  && c.oid === cur[0]  && c.frame === cur[1];
      const isOrig = !!orig && c.oid === orig[0] && c.frame === orig[1];
      // ★ 这是决定"哪一帧拿去建 mesh"的控件, 用 <button>: 回车/空格、焦点环、AT 语义全都白送。
      const el = document.createElement("button");
      el.type = "button";
      el.className = "cand" + (isCur ? " cur" : "") + (isOrig ? " orig" : "");
      el.innerHTML = '<img loading="lazy" src="'+U('/img/cand',{word:词,oid:c.oid,frame:c.frame,size:160})+'">'+
        (isOrig && !isCur ? '<i class="ohome">'+(solo ? 'pipeline pick' : 'best')+'</i>' : '')+
        '<div>#'+c.oid+' f'+c.frame+(isCur?' · in use':'')+'<br>'+
        c.pct+'% · '+c.nframes+' frames'+(c.edge>0.2?' · edge '+c.edge:'')+'</div>';
      el.title = isCur ? (isOrig ? 'In use, and it is the '+(solo ? 'pipeline pick' : 'best frame')+'.'
                                 : 'In use. Click to go back to the '+(solo ? 'pipeline pick' : 'best frame')+'.')
                       : 'Click to use this one.';
      el.onclick = () => {
        if (!solo){
          // 共用: 全部交给 设样本(清掉别的改用 / 等于盘上机器那张就不钉 / VLM 放行)
          if (!isCur) 设样本(词, oid, c.frame);
          else if (!isOrig && orig) 设样本(词, oid, orig[1]);     // 点现状 = 退回 best
          退一层();
          return;
        }
        // solo: 点现状 = 撤销, 回到 pick 原本选的那张。撤销一律走"删字段",
        //   不写一个和原选等值的进去: 留着它, --apply 会当成人明确做过判断来处理。
        const 撤 = isCur || isOrig;
        let rr = row;
        if (row.虚){
          if (撤) { 退一层(); return; }                     // 虚拟行上"撤销"= 什么都没选
          rr = S.行[确保行(词, oid)];
        }
        if (撤){ delete rr.改用; delete rr.VLM放行; }       // 撤销就撤干净, 放行是跟着这次换帧来的
        else {
          rr.改用 = [c.oid, c.frame];
          // 人亲手挑了一帧 = 人看过了。VLM 否掉的那一帧不是这一帧, 以人为准。
          if (VLM否(词, rr.实例)) { rr.VLM放行 = true; if (rr.决定 === 'drop') rr.决定 = 'keep'; }
        }
        收尾全部(词);
        dirty = true; 重建索引();
        退一层();          // 选完回到上一层, 卡片上直接看到换上去的那一帧
      };
      box.appendChild(el);
    });
   })
   .catch(e => {
    t.innerHTML = '<h3>'+esc(词)+'<span style="color:var(--red)">cannot reach the server — '+
      esc(e)+'. This page may be a stale copy from before the server stopped; '+
      'make sure it is running, then reload.</span></h3>';
   });
  return t;
}

document.getElementById("save").onclick = async () => {
  if (GATE === 'scale') return scSave();
  const b = document.getElementById("save"); b.disabled = true; b.textContent = "Saving…";
  const r = await (await fetch(U("/api/save"), {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({候选: S.行, 删除: [...S.待删]})})).json();
  if(r.error){ alert("Save failed: "+r.error); b.disabled=false; b.textContent="Save *"; return; }
  dirty = false; S.待删 = new Set(); stat();
  b.textContent = "Saved ✓";
  setTimeout(stat, 1500);
};
// 起始闸由后端注入(#scale)或默认 review
if (location.hash === '#scale') GATE = 'scale';
boot();
</script>
"""


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", default="r1")
    ap.add_argument("--indoor", action="store_true")
    ap.add_argument("--stage", choices=["pre", "post"], default=None,
                    help="pre=闸2(生成前) post=闸4(召回后)。默认取盘上已存在的最新那份")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--gate", choices=["review", "scale"], default="review",
                    help="打开时停在哪道闸。review=闸2/4(候选决策台), scale=闸3(尺子表)。"
                         "两道闸在同一个服务里, 顶栏可以随时切")
    ap.add_argument("--solo-min-px", type=int, default=None,
                    help="逐实例面积门槛(像素), 只用于面板标注。默认问 pipeline 要(SOLO_MIN_PX)")
    a = ap.parse_args()

    st = Store(a.scene, a.run, a.indoor, a.stage, a.solo_min_px)
    d = st.load()                    # 起服务之前先把决策文件读通, 免得页面开出来才报错
    print(f"场景 {a.scene} 轮次 {a.run}   闸 {2 if st.stage=='pre' else 4} ({st.stage})")
    print(f"决策 {st.DEC}   {len(d['候选'])} 个候选")
    print(f"帧   {st.FR}")
    print(f"  闸 3 尺子: {st.scale_paths()[0]}")
    print(f"\n  http://localhost:{a.port}" + ("#scale" if a.gate=="scale" else "") + "\n")
    print(f"  远程: ssh -L {a.port}:localhost:{a.port} <这台机器>")
    print(f"  点完保存后, 回终端跑: drive.py --scene {a.scene}"
          f"{' --indoor' if a.indoor else ''} --go\n")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(st)).serve_forever()


if __name__ == "__main__":
    main()
