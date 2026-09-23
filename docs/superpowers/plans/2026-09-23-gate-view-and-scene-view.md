# 闸口看得清 + 场景查看页搬进闸口 —— 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 闸口网页每个实例旁边多一张"整帧定位图"(暗的自动提亮);3D 查看页由 drive.py 在闸 3/4/5 自动出数据、挂在闸口同一个端口的 `/3d`,两边按 `词:实例` 互相跳。

**Architecture:** `box_fit_view.py` 的出口从"烤进 html"改成"写 json"(算法不动);新文件 `tools/scene_view.py`(只标准库)把 json 端给浏览器,`gate_ui.py` 在 `/3d*` 才懒加载它;模板 `box_fit_view_template.html` 数据标签为空时自己去 `fetch('/3d/data')`。闸口定位图是 `gate_ui.py` 里的一个新路由 `/img/frame` + 实例卡多一张图。

**Tech Stack:** Python 3.11(`~/miniforge3/envs/sam3d/bin/python`,有 numpy/cv2/pycocotools,**没有 pytest**——测试是带 assert 的普通脚本)、`http.server`、原生 JS + three.js(CDN)。

**Spec:** `docs/superpowers/specs/2026-09-23-gate-view-and-scene-view-design.md`

## Global Constraints

- Python 一律 `PY=~/miniforge3/envs/sam3d/bin/python`;系统 python 没有 numpy。
- 界面上的文字英文,代码注释中文(memory: 界面英文/交流中文)。
- 改已有文件前先留备份:`cp X X.bak_3dtab`(仓库惯例,一个文件只备一次)。
- `gate_ui.py`:不写 `05_review_applied.json`、不起子进程、不 import 算法模块。3D 的代码**不进** `gate_ui.py`。
- 数据文件名 `10_box_fit_view.json` 在 `box_fit_view.py`、`gate_ui.py`(`box_view_state`)、`scene_view.py`、`drive.py` 各写一次,改名要一起改。
- 物体的键 = `词 + ':' + 实例号`(模板里本来就是 `key(o) = o.word + ':' + o.id`)。URL 里 `encodeURIComponent`。
- 提交信息末尾加 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`。
- 测试用的轮次:`~/sam3d/data/indoorOutput/office/r6_pillarfix`(有 `03_masks/pipe.npz`、`06_all_boxes_merged.json`、`07_scene_phy.glb`、人)。**不许改写它下面已有的文件**(测试会新增 `10_box_fit_view.json`,那是流水线本来就该出的产物,可以);临时产物写到 `/tmp/claude-1004/-home-jincheng/92d1614a-64e3-41aa-9d0b-8592d431f8f9/scratchpad/`。

---

## 文件一览

| 文件 | 做什么 |
|---|---|
| `tools/gate_ui.py`(改) | `_gamma/_apply_gamma/locate_rgb`、`Store._mask/locate`、路由 `/img/frame`、`/api/state` 多一个 `场景查看`、`/3d*` 懒加载转给 scene_view、页面:实例卡定位图 + 3D 链接 + `data-key`、顶栏 3D、`?sel=` 跳到实例 |
| `tools/scene_view.py`(新) | 只标准库。`DATA_NAME/INPUTS/data_path/stale/state/handle`。`/3d` 模板壳、`/3d/data` gzip json、`/3d/state` |
| `tools/box_fit_view.py`(改) | `--data`(默认 `<r1>/10_box_fit_view.json`,`off` 不出)、`--out off`、`data["页"]` |
| `tools/box_fit_view_template.html`(改) | 数据标签空就 `fetch('/3d/data')`;`#sel=` 初选;侧栏"回闸口"链接;`D.页` 填标题 |
| `.claude/skills/run-recon-pipeline/drive.py`(改) | `ensure_box_view(R1, dry)`,`gate_report` 在第 3/4/5 段调它 |
| `tools/tests/`(新) | `test_gate_locate.py`、`test_scene_view.py`、`test_drive_box_view.py`、`test_box_fit_view_data.sh`、`test_gate_3d.sh`、`run.sh` |
| `tools/README.md`、`devLog/devlog_9.21.md`(改) | 新工具、新行为 |

---

### Task 1: 闸口提亮 + 整帧定位图(后端)

**Files:**
- Modify: `tools/gate_ui.py`(`cutout_rgb` 在 77-104 行;`Store.cutout` 在 707-722 行;路由 `/img/inst` 在 1140-1149 行)
- Create: `tools/tests/test_gate_locate.py`

**Interfaces:**
- Produces: `_gamma(img, m) -> float`、`_apply_gamma(img, g) -> ndarray`、`cutout_rgb(img, m, size, gamma=True)`、`locate_rgb(img, m, size=420, g=None) -> ndarray|None`、`Store._mask(word, oid, frame) -> uint8 mask|None`、`Store.locate(word, oid, frame, size=420)`、路由 `GET /img/frame?word=&oid=[&frame=][&size=]`。

- [ ] **Step 1: 备份**

```bash
cd ~/sam3d/tools && cp gate_ui.py gate_ui.py.bak_3dtab
```

- [ ] **Step 2: 写失败的测试** `tools/tests/test_gate_locate.py`

```python
#!/usr/bin/env python
"""闸口的提亮和整帧定位图: 纯函数, 合成图, 不碰盘上的轮次。跑: $PY tools/tests/test_gate_locate.py"""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gate_ui as G

def dark_scene():
    img = np.full((360, 640, 3), 30, np.uint8)      # 暗背景
    img[100:200, 300:400] = 45                      # 更暗一点的物体
    m = np.zeros((360, 640), np.uint8); m[100:200, 300:400] = 255
    return img, m

def test_gamma_only_brightens_dark():
    img, m = dark_scene()
    g = G._gamma(img, m)
    assert 0.4 <= g < 1.0, g
    bright = np.full_like(img, 180)
    assert G._gamma(bright, m) == 1.0
    out = G._apply_gamma(img, g)
    med = int(np.median(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[m > 127]))
    assert 118 <= med <= 138, med                    # 中位数被拉到 128 上下

def test_cutout_and_locate_share_gamma():
    img, m = dark_scene()
    g = G._gamma(img, m)
    want = int(G._apply_gamma(np.full((1, 1, 3), 45, np.uint8), g)[0, 0, 0])
    c = G.cutout_rgb(img, m, 120)
    l = G.locate_rgb(img, m, 420)
    assert c.shape == (120, 120, 3)
    assert l.shape[1] == 420 and l.shape[0] == int(360 * 420 / 640), l.shape
    assert abs(int(c[60, 60, 0]) - want) <= 2, (c[60, 60, 0], want)          # 特写中心 = 物体
    assert abs(int(l[98, 229, 0]) - want) <= 2, (l[98, 229, 0], want)        # 定位图里物体中心, 同样亮
    bg = int(G._apply_gamma(np.full((1, 1, 3), 30, np.uint8), g)[0, 0, 0])
    assert abs(int(l[10, 10, 0]) - int(bg * 0.45)) <= 2, (l[10, 10, 0], bg)  # mask 外压暗
    red = (l[:, :, 2] >= 200) & (l[:, :, 1] < 60) & (l[:, :, 0] < 60)
    assert red.sum() > 100, red.sum()                                        # 有红框
    assert G.locate_rgb(img, np.zeros_like(m), 420) is None                  # 空 mask

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f(); print("ok", k)
```

- [ ] **Step 3: 跑,确认失败**

Run: `cd ~/sam3d && ~/miniforge3/envs/sam3d/bin/python tools/tests/test_gate_locate.py`
Expected: `AttributeError: module 'gate_ui' has no attribute '_gamma'`

- [ ] **Step 4: 实现**

在 `cutout_rgb` 前面(第 77 行之前)加:

```python
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
```

`cutout_rgb` 改签名和取像素那两行:

```python
def cutout_rgb(img, m, size, gamma=True):
    ...(docstring 不动)
    b = m > 127
    if b.sum() < 30:
        return None
    src = _apply_gamma(img, _gamma(img, m)) if gamma else img
    out = np.full_like(img, 210)                      # 灰底
    out[b] = src[b]
    ...(后面不动)
```

`cutout_rgb` 后面加:

```python
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
```

`Store.cutout`(707-722 行)拆成 `_mask` + `cutout` + `locate`:

```python
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
        """(原 docstring 原样保留)"""
        m = self._mask(word, oid, frame)
        return None if m is None else cutout_rgb(self._frame(frame), m, size)      # 走 LRU, 不直接 imread

    def locate(self, word, oid, frame, size=420):
        """整帧定位图, 和 cutout 同一帧同一 mask。"""
        m = self._mask(word, oid, frame)
        return None if m is None else locate_rgb(self._frame(frame), m, size)
```

路由:在 `if u.path == "/img/inst":` 那块(1140-1149 行)**后面**加:

```python
                if u.path == "/img/frame":
                    # 整帧定位图: 和 /img/inst 同一帧(前端带 frame 就那帧, 否则代表帧)
                    ins = st.insts(q["word"], st.solo_min_px())
                    hit = next((r for r in ins if r["oid"] == int(q["oid"])), None)
                    if not hit:
                        return self._png(None)
                    fr = int(q["frame"]) if q.get("frame") else hit["代表帧"]
                    return self._png(st.locate(q["word"], hit["oid"], fr, int(q.get("size", "420"))))
```

- [ ] **Step 5: 跑测试,通过**

Run: `cd ~/sam3d && ~/miniforge3/envs/sam3d/bin/python tools/tests/test_gate_locate.py`
Expected: 两行 `ok test_...`

- [ ] **Step 6: 真轮次上 curl 一张**

```bash
cd ~/sam3d && PY=~/miniforge3/envs/sam3d/bin/python; SP=/tmp/claude-1004/-home-jincheng/92d1614a-64e3-41aa-9d0b-8592d431f8f9/scratchpad
$PY tools/gate_ui.py --scene office --run r6_pillarfix --indoor --port 8799 > $SP/gate_8799.log 2>&1 &
sleep 4; curl -s "http://127.0.0.1:8799/img/frame?word=pipe&oid=0&size=520" -o $SP/frame_pipe0.png; curl -s "http://127.0.0.1:8799/img/inst?word=pipe&oid=0&size=200" -o $SP/inst_pipe0.png
$PY -c "import cv2;a=cv2.imread('$SP/frame_pipe0.png');print(a.shape)"; kill %1
```
Expected: shape 长边 520,不是 200×200(那是 "mask empty" 占位)。用 Read 看两张 png:定位图里红框在通风管处、背景压暗;特写比之前亮。

- [ ] **Step 7: 提交**

```bash
cd ~/sam3d && git add tools/gate_ui.py tools/tests/test_gate_locate.py && git commit -m "闸口: 暗抠图自动提亮 + 整帧定位图路由 /img/frame

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: 闸口定位图(页面)

**Files:**
- Modify: `tools/gate_ui.py`(CSS 1333-1336 行 `.insts/.inst img`;实例卡 innerHTML 2531-2533 行;`el.className` 2525 行)

**Interfaces:**
- Consumes: 路由 `/img/frame`(Task 1)。
- Produces: 每张实例卡 `el.dataset.key = 词 + ':' + oid`(Task 6 的 `?sel=` 靠它滚动)。

- [ ] **Step 1: CSS**

把 `.insts{...minmax(150px,1fr)...}` 改成 `minmax(400px,1fr)`,并在 `.inst img{...}` 那条后面加:

```css
 /* 两张图并排: 左特写(是什么), 右整帧定位(在哪、挨着什么)。定位图点开是原分辨率的大图。 */
 .inst .pics{display:grid;grid-template-columns:120px 1fr;gap:8px;margin:8px 8px 0;align-items:start}
 .inst .pics img{width:100%;margin:0}
 .inst .pics img.loc{aspect-ratio:16/9;object-fit:contain;background:var(--plate)}
 .inst.flash{box-shadow:inset 0 0 0 3px var(--yellow);z-index:2}
```

- [ ] **Step 2: 卡片**

`el.className = "inst" + ...;` 那句后面加一行:

```js
    el.dataset.key = w.词 + ':' + r.oid;      // ?sel= 跳过来时靠它找到这张卡
```

innerHTML 开头那个 `<img loading="lazy" src="'+U('/img/inst', ...)+'">'+` 换成:

```js
      (() => { const q = 显帧 != null ? {word:w.词, oid:r.oid, frame:显帧} : {word:w.词, oid:r.oid};
        return '<div class="pics">'+
          '<img loading="lazy" src="'+U('/img/inst', {...q, size:120})+'">'+
          '<a href="'+U('/img/frame', {...q, size:1280})+'" target="_blank" title="Open the full frame">'+
            '<img class="loc" loading="lazy" src="'+U('/img/frame', {...q, size:520})+'"></a>'+
        '</div>'; })()+
```

- [ ] **Step 3: 起服务看一眼**

```bash
cd ~/sam3d && ~/miniforge3/envs/sam3d/bin/python tools/gate_ui.py --scene office --run r6_pillarfix --indoor --port 8799
```
浏览器开 `http://localhost:8799`,点开 `pipe` 那个词:每张卡左特写、右整帧红框;点定位图开大图。`.inst.dropped` 的红斜线仍然盖在图上。关掉服务。

- [ ] **Step 4: 提交**

```bash
cd ~/sam3d && git add tools/gate_ui.py && git commit -m "闸口: 实例卡并排整帧定位图, 卡片带 data-key

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `box_fit_view.py` 出数据文件

**Files:**
- Modify: `tools/box_fit_view.py`(参数 44-58 行;结尾 419-432 行;文件头 docstring)
- Create: `tools/tests/test_box_fit_view_data.sh`

**Interfaces:**
- Produces: `--data <路径|off>`(默认 `<r1>/10_box_fit_view.json`)、`--out <路径|off>`;json 顶层多一个 `"页": {"title","eyebrow","notice"}`。

- [ ] **Step 1: 备份**

```bash
cd ~/sam3d/tools && cp box_fit_view.py box_fit_view.py.bak_3dtab
```

- [ ] **Step 2: 写测试** `tools/tests/test_box_fit_view_data.sh`

```bash
#!/bin/bash
# box_fit_view.py 两种出口: --data 出 json(闸口 3D tab 读), --out 出单文件 html。office r6 上跑, 约 40 s。
set -e
PY=~/miniforge3/envs/sam3d/bin/python; R1=~/sam3d/data/indoorOutput/office/r6_pillarfix
SP=/tmp/claude-1004/-home-jincheng/92d1614a-64e3-41aa-9d0b-8592d431f8f9/scratchpad; mkdir -p $SP
$PY ~/sam3d/tools/box_fit_view.py --r1 $R1 --out off --data $SP/bfv.json > $SP/bfv_data.log
$PY - <<EOF
import json; d = json.load(open("$SP/bfv.json"))
for k in ("insts", "ctx", "cams", "页"): assert k in d, k
assert d["页"]["eyebrow"] and "measured" in d["页"]["eyebrow"], d["页"]
assert len(d["insts"]) > 10, len(d["insts"])
print("json ok:", len(d["insts"]), "insts;", d["页"]["eyebrow"])
EOF
test ! -e $R1/10_box_fit_view.json.tmp
$PY ~/sam3d/tools/box_fit_view.py --r1 $R1 --data off --out $SP/bfv.html > $SP/bfv_html.log
grep -q '"insts"' $SP/bfv.html && ! grep -q '__DATA__' $SP/bfv.html && echo "html ok"
```

- [ ] **Step 3: 跑,确认失败**

Run: `bash ~/sam3d/tools/tests/test_box_fit_view_data.sh`
Expected: `error: unrecognized arguments: --data`

- [ ] **Step 4: 实现**

参数(`--out` 那行改、加 `--data`):

```python
ap.add_argument("--out", default=None, help="单文件 html; 默认 <r1>/10_box_fit_view.html, off = 不出")
ap.add_argument("--data", default=None,
                help="数据 json(闸口网页的 3D tab 读它, tools/scene_view.py); 默认 <r1>/10_box_fit_view.json, off = 不出")
```

`OUT = ...` 那行换成:

```python
OUT  = None if a.out  == "off" else (a.out  or f"{R1}/10_box_fit_view.html")
DATA = None if a.data == "off" else (a.data or f"{R1}/10_box_fit_view.json")
```

结尾:`eyebrow = ...` 那行**之后**、`t = open(...)` 之前,把写文件那 4 行换成:

```python
data["页"] = dict(title=a.title, eyebrow=eyebrow, notice=notice)   # 服务端出页时由页面自己填
blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
if DATA:
    open(DATA, "w", encoding="utf-8").write(blob)
    print(f"[数据] {DATA}  {os.path.getsize(DATA)/1e6:.1f} MB   (闸口网页 3D tab 读这个)")
if OUT:
    t = open(f"{HERE}/box_fit_view_template.html", encoding="utf-8").read()
    t = (t.replace("__TITLE__", a.title).replace("__EYEBROW__", eyebrow).replace("__NOTICE__", notice)
          .replace("__DATA__", blob))
    open(OUT, "w", encoding="utf-8").write(t)
    print(f"[输出] {OUT}  {os.path.getsize(OUT)/1e6:.1f} MB")
```

文件头 docstring 第 6 行 `出 <r1>/10_box_fit_view.html(...)` 改成:

```
出两样: <r1>/10_box_fit_view.json(闸口网页 3D tab 读的数据, --data off 不出)
        <r1>/10_box_fit_view.html(单文件, 浏览器直接开, 要发给人用它; --out off 不出)
drive.py 在闸 3/4/5 自动跑 --out off; 手动要单文件就自己跑。
```

- [ ] **Step 5: 跑测试,通过**

Run: `bash ~/sam3d/tools/tests/test_box_fit_view_data.sh`
Expected: `json ok: ... insts; office/r6_pillarfix · N measured ...` 和 `html ok`

- [ ] **Step 6: 提交**

```bash
cd ~/sam3d && git add tools/box_fit_view.py tools/tests/test_box_fit_view_data.sh && git commit -m "box_fit_view: 出口拆成 --data json 和 --out html, 页面标题存进 json

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `tools/scene_view.py`(只标准库)

**Files:**
- Create: `tools/scene_view.py`
- Create: `tools/tests/test_scene_view.py`

**Interfaces:**
- Produces: `DATA_NAME = "10_box_fit_view.json"`、`INPUTS`、`data_path(R1) -> str`、`stale(R1) -> str|None`、`state(R1) -> {"在","生成于","过时"}`、`handle(h, path, R1) -> bool`(h 要有 `_send(code, ctype, body)`、`_json(obj, code)`、`send_response/send_header/end_headers/wfile`)。

- [ ] **Step 1: 写测试** `tools/tests/test_scene_view.py`

```python
#!/usr/bin/env python
"""scene_view: 假的请求处理器 + 临时轮次目录。跑: $PY tools/tests/test_scene_view.py"""
import gzip, io, json, os, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import scene_view as V

class FakeH:
    """只记下 gate_ui 的 handler 会被调到的那几样。"""
    def __init__(self): self.code = None; self.ctype = None; self.hdr = {}; self.wfile = io.BytesIO()
    def _send(self, code, ctype, body, cache=False): self.code, self.ctype = code, ctype; self.wfile.write(body)
    def _json(self, obj, code=200): self._send(code, "application/json; charset=utf-8", json.dumps(obj, ensure_ascii=False).encode())
    def send_response(self, c): self.code = c
    def send_header(self, k, v): self.hdr[k] = v
    def end_headers(self): pass
    def body(self): return self.wfile.getvalue()

def test_state_and_stale():
    with tempfile.TemporaryDirectory() as R1:
        assert V.state(R1) == {"在": False, "生成于": None, "过时": None}
        open(V.data_path(R1), "w").write('{"insts": []}')
        s = V.state(R1); assert s["在"] and s["过时"] is None, s
        p = f"{R1}/07_scene_phy.glb"; open(p, "wb").write(b"x")
        os.utime(p, (time.time() + 10, time.time() + 10))
        assert V.stale(R1) == "07_scene_phy.glb"

def test_routes():
    with tempfile.TemporaryDirectory() as R1:
        h = FakeH(); assert V.handle(h, "/3d", R1) and h.code == 200 and b"No 3D data" in h.body()
        h = FakeH(); assert V.handle(h, "/3d/data", R1) and h.code == 404
        open(V.data_path(R1), "w").write('{"insts": [1, 2], "页": {"title": "T"}}')
        h = FakeH(); assert V.handle(h, "/3d", R1)
        assert b'<script id="data" type="application/json"></script>' in h.body()      # 数据标签留空
        assert b"__DATA__" not in h.body() and b"__TITLE__" not in h.body()
        h = FakeH(); assert V.handle(h, "/3d/data", R1)
        assert h.hdr["Content-Encoding"] == "gzip"
        assert json.loads(gzip.decompress(h.body()))["insts"] == [1, 2]
        h = FakeH(); assert V.handle(h, "/3d/state", R1) and json.loads(h.body())["在"] is True
        assert V.handle(FakeH(), "/3dx", R1) is False

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f(); print("ok", k)
```

- [ ] **Step 2: 跑,确认失败**

Run: `cd ~/sam3d && ~/miniforge3/envs/sam3d/bin/python tools/tests/test_scene_view.py`
Expected: `ModuleNotFoundError: No module named 'scene_view'`

- [ ] **Step 3: 实现** `tools/scene_view.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""闸口网页的 3D tab —— 把 box_fit_view.py 算好的数据端给浏览器。

只有标准库: 不读点云、不读 glb、不起子进程、不 import 算法模块。
gate_ui.py 在 do_GET 里碰到 /3d 开头的路径才 import 这个文件(懒加载), 所以它坏了只影响 3D tab,
决策台照常填照常存。这是有意的边界, 别往这里加会读大文件或起进程的东西。

路由(gate_ui 转过来, R1 = 闸口当前看的那一轮, 顶栏换轮次就跟着换):
  /3d          模板壳: box_fit_view_template.html, 数据标签留空, 页面自己去拿 /3d/data
  /3d/data     <R1>/10_box_fit_view.json, gzip(按 mtime 缓存压好的那份)
  /3d/state    {"在": bool, "生成于": "09-23 10:12" | None, "过时": "07_scene_phy.glb" | None}

数据由 tools/box_fit_view.py --out off 产出, drive.py 在闸 3/4/5 自动跑(ensure_box_view)。
"过时" = 有人绕过 drive.py 直接跑了流水线, 产物比数据新 —— 页面上提示, 不自动重算(那要起进程)。
文件名在 box_fit_view.py / gate_ui.py(box_view_state) / drive.py 各写一次, 改要一起改。
"""
import gzip, json, os, time

HERE = os.path.dirname(os.path.realpath(__file__))
DATA_NAME = "10_box_fit_view.json"
# 这些产物比数据新, 数据就过时了。顺序无所谓, 报第一个撞上的。
INPUTS = ["06_all_boxes.json", "06_all_boxes_merged.json", "07_scene_phy.glb",
          "07_scene_skipped.json", "10_people.json", "09_body4d_report.json"]
_GZ = {}          # (路径, mtime) -> gzip 字节。7 MB 压一次 0.3 s, 别每次刷新都压


def data_path(R1):
    return os.path.join(R1, DATA_NAME)


def stale(R1):
    """数据比哪个输入旧, 返回那个文件名; 都不比它新 -> None; 数据不在也 None(那是"没有", 不是"过时")。"""
    p = data_path(R1)
    if not os.path.exists(p):
        return None
    t = os.path.getmtime(p)
    for n in INPUTS:
        q = os.path.join(R1, n)
        if os.path.exists(q) and os.path.getmtime(q) > t + 1:
            return n
    return None


def state(R1):
    p = data_path(R1)
    if not os.path.exists(p):
        return {"在": False, "生成于": None, "过时": None}
    return {"在": True, "生成于": time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p))),
            "过时": stale(R1)}


NO_DATA = """<!doctype html><meta charset="utf-8"><title>3D view</title>
<style>body{font:15px/1.6 ui-sans-serif,system-ui;max-width:640px;margin:60px auto;padding:0 16px;color:#141413}
code{background:#eee;padding:1px 5px}</style>
<h1>No 3D data for this run yet</h1>
<p>It appears once gate 3 (boxes) has run — <code>drive.py</code> writes
<code>__R1__/10_box_fit_view.json</code> at gates 3, 4 and 5.</p>
<p>To make it by hand: <code>python tools/box_fit_view.py --r1 __R1__ --out off</code>, then reload.</p>"""


def handle(h, path, R1):
    """h 是 gate_ui 的请求处理器(用它的 _send / _json)。处理了返回 True, 不是这里的路返回 False。"""
    if path == "/3d/state":
        h._json(state(R1)); return True
    if path == "/3d/data":
        p = data_path(R1)
        if not os.path.exists(p):
            h._json({"error": f"no {DATA_NAME} in {R1} — run gate 3 (boxes) first"}, 404); return True
        k = (p, os.path.getmtime(p))
        if k not in _GZ:
            _GZ.clear(); _GZ[k] = gzip.compress(open(p, "rb").read(), 5)
        body = _GZ[k]
        h.send_response(200)
        h.send_header("Content-Type", "application/json; charset=utf-8")
        h.send_header("Content-Encoding", "gzip")
        h.send_header("Content-Length", str(len(body)))
        h.end_headers(); h.wfile.write(body); return True
    if path == "/3d":
        if not state(R1)["在"]:
            h._send(200, "text/html; charset=utf-8", NO_DATA.replace("__R1__", R1).encode()); return True
        t = open(os.path.join(HERE, "box_fit_view_template.html"), encoding="utf-8").read()
        t = (t.replace("__TITLE__", "Box Fit View").replace("__EYEBROW__", "")
              .replace("__NOTICE__", '<div id="notice"></div>').replace("__DATA__", ""))
        h._send(200, "text/html; charset=utf-8", t.encode()); return True
    return False
```

- [ ] **Step 4: 跑测试,通过**

Run: `cd ~/sam3d && ~/miniforge3/envs/sam3d/bin/python tools/tests/test_scene_view.py`
Expected: `ok test_state_and_stale`、`ok test_routes`

- [ ] **Step 5: 提交**

```bash
cd ~/sam3d && git add tools/scene_view.py tools/tests/test_scene_view.py && git commit -m "scene_view: 闸口 3D tab 的服务端, 只标准库, 端 box_fit_view 的 json

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: 模板:从服务拿数据、`#sel=` 初选、回闸口链接

**Files:**
- Modify: `tools/box_fit_view_template.html`(105 行 `__NOTICE__`;180-181 行 IIFE 开头;488 行 `select`;668-671 行结尾)

**Interfaces:**
- Consumes: `/3d/data`(Task 4)、`D.页`(Task 3)。
- Produces: 页面读 `location.hash` 里的 `sel=<词:实例>`;选中时 `#gatelink` 指向 `/?<原 query>&sel=<词:实例>`。

- [ ] **Step 1: 备份**

```bash
cd ~/sam3d/tools && cp box_fit_view_template.html box_fit_view_template.html.bak_3dtab
```

- [ ] **Step 2: 侧栏加回闸口链接**

第 105 行 `__NOTICE__` 后面加:

```html
      <p class="note" id="gatelink-wrap" hidden><a id="gatelink" href="/">‹ Back to the gate — this object</a></p>
```

- [ ] **Step 3: 数据来源**

180-181 行

```js
(function(){
const D = JSON.parse(document.getElementById('data').textContent);
```
换成
```js
(async function(){
// Single-file mode: the data is baked into the page. Served mode (gate_ui /3d): the tag is empty — fetch it.
const SERVED = !document.getElementById('data').textContent.trim();
const D = SERVED
  ? await (async () => { const r = await fetch('/3d/data' + location.search);
      if (!r.ok) throw new Error('No 3D data for this run (' + r.status + ')'); return r.json(); })()
  : JSON.parse(document.getElementById('data').textContent);
if (SERVED && D.页){
  document.title = D.页.title || 'Box Fit View';
  document.querySelector('.eyebrow').textContent = D.页.eyebrow || '';
  const nt = document.getElementById('notice'); if (nt) nt.outerHTML = D.页.notice || '';
}
```

结尾 `})();` 换成:

```js
})().catch(e => { document.body.insertAdjacentHTML('afterbegin',
  '<p style="margin:24px;font:15px ui-sans-serif,system-ui">' + String(e.message || e) + '</p>'); });
```

- [ ] **Step 4: `select` 更新回闸口链接**

`function select(ob, doFocus=true){` 后第一行 `S.sel = ob; ...` 之后加:

```js
  if (SERVED){ const q = new URLSearchParams(location.search); q.set('sel', key(ob.o));
    $('gatelink').href = '/?' + q.toString(); $('gatelink-wrap').hidden = false; }
```

- [ ] **Step 5: 初选按 `#sel=`**

结尾 `select(objs.filter(visible).sort((a,b)=>sortKey(b.o)-sortKey(a.o))[0] || objs[0]);` 换成:

```js
// Opened from the gate with #sel=word:id — start on that object; otherwise the widest-spread one.
const want = decodeURIComponent((location.hash.match(/sel=([^&]+)/) || [])[1] || '');
select(objs.find(ob => key(ob.o) === want) || objs.filter(visible).sort((a,b)=>sortKey(b.o)-sortKey(a.o))[0] || objs[0]);
```

`key` 在 380 行定义(`const`),在 `select` 和这里用之前已经定义 —— 不用挪。

- [ ] **Step 6: 单文件模式没坏**

```bash
cd ~/sam3d && SP=/tmp/claude-1004/-home-jincheng/92d1614a-64e3-41aa-9d0b-8592d431f8f9/scratchpad
~/miniforge3/envs/sam3d/bin/python tools/box_fit_view.py --r1 data/indoorOutput/office/r6_pillarfix --data off --out $SP/bfv2.html
grep -c "SERVED" $SP/bfv2.html
```
Expected: 计数 ≥ 3。浏览器开 `$SP/bfv2.html`(或发成 artifact):列表、点云、盒子照常;侧栏**没有**"Back to the gate"。

- [ ] **Step 7: 提交**

```bash
cd ~/sam3d && git add tools/box_fit_view_template.html && git commit -m "box_fit_view 模板: 数据标签空就 fetch /3d/data, #sel 初选, 回闸口链接

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `gate_ui.py` 挂 `/3d`、顶栏 3D、卡片 3D 链接、`?sel=` 跳到实例

**Files:**
- Modify: `tools/gate_ui.py`(常量 ~70 行;`/api/state` 1100-1115 行;`do_GET` 1092 行 `st = self._store(q)` 之后;header 1503-1512 行;CSS;`boot()` 2220-2262 行;实例卡 `.im` 2536 行)
- Create: `tools/tests/test_gate_3d.sh`

**Interfaces:**
- Consumes: `scene_view.handle/state`(Task 4)、卡片 `data-key`(Task 2)、模板 `#sel=`(Task 5)。
- Produces: `/api/state` 多字段 `场景查看 = {在, 生成于, 过时}`;`GET /3d*` 转 scene_view;首次 import 打印 `[3D] 挂载 scene_view.py`;页面 `?sel=词:实例` 打开那个词的弹层并滚到那张卡。

- [ ] **Step 1: 写测试** `tools/tests/test_gate_3d.sh`

```bash
#!/bin/bash
# 闸口 + 3D tab 的整链: 懒加载、/3d 三条路、数据不在时灰掉。office r6 上跑, 服务起在 8799。
set -e
PY=~/miniforge3/envs/sam3d/bin/python; R1=~/sam3d/data/indoorOutput/office/r6_pillarfix
SP=/tmp/claude-1004/-home-jincheng/92d1614a-64e3-41aa-9d0b-8592d431f8f9/scratchpad; mkdir -p $SP
U="http://127.0.0.1:8799"; Q="s=office&r=r6_pillarfix&i=1"
test -e $R1/10_box_fit_view.json || $PY ~/sam3d/tools/box_fit_view.py --r1 $R1 --out off > /dev/null
$PY ~/sam3d/tools/gate_ui.py --scene office --run r6_pillarfix --indoor --port 8799 > $SP/gate3d.log 2>&1 &
PID=$!; trap "kill $PID 2>/dev/null; mv -f $R1/10_box_fit_view.json.hidden $R1/10_box_fit_view.json 2>/dev/null; true" EXIT
for i in $(seq 1 40); do curl -sf "$U/api/runs" > /dev/null && break; sleep 0.5; done
curl -sf "$U/api/state?$Q" | $PY -c "import json,sys; d=json.load(sys.stdin); assert d['场景查看']['在'] is True, d['场景查看']; print('state ok', d['场景查看'])"
! grep -q "\[3D\] 挂载" $SP/gate3d.log && echo "lazy ok: not imported yet"
curl -sf "$U/3d?$Q" | grep -q '<script id="data" type="application/json"></script>' && echo "/3d ok"
grep -q "\[3D\] 挂载" $SP/gate3d.log && echo "lazy ok: imported on first /3d"
curl -sf --compressed "$U/3d/data?$Q" | $PY -c "import json,sys; d=json.load(sys.stdin); assert len(d['insts'])>10; print('/3d/data ok', len(d['insts']))"
curl -sf "$U/3d/state?$Q" | grep -q '"在": true' && echo "/3d/state ok"
mv $R1/10_box_fit_view.json $R1/10_box_fit_view.json.hidden
curl -sf "$U/api/state?$Q" | $PY -c "import json,sys; d=json.load(sys.stdin); assert d['场景查看']['在'] is False; print('greyed ok')"
curl -sf "$U/3d?$Q" | grep -q "No 3D data" && echo "/3d without data ok"
curl -sf "$U/api/state?$Q" | $PY -c "import json,sys; d=json.load(sys.stdin); assert d['候选']; print('gate still serves', len(d['候选']), 'rows')"
mv $R1/10_box_fit_view.json.hidden $R1/10_box_fit_view.json
echo ALL OK
```

- [ ] **Step 2: 跑,确认失败**

Run: `bash ~/sam3d/tools/tests/test_gate_3d.sh`
Expected: 第一条 `assert d['场景查看']` 处 `KeyError: '场景查看'`

- [ ] **Step 3: 后端**

常量区(`MYIMG = ...` 后)加:

```python
HERE  = os.path.dirname(os.path.realpath(__file__))   # realpath: 万一有软链接, abspath 会数错目录(见 tools/README)


def box_view_state(R1):
    """3D tab 的数据在不在。只查文件, 不 import scene_view —— 顶栏灰不灰要在 /api/state 里答,
    而那时候人还没点 3D, 不该为它把 scene_view 拉进来。文件名和 scene_view.DATA_NAME 一致。"""
    p = f"{R1}/10_box_fit_view.json"
    if not os.path.exists(p):
        return {"在": False, "生成于": None}
    return {"在": True, "生成于": time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))}
```

`/api/state` 的 `return self._json(dict(` 里, `组装守卫=...` 那行后面加:

```python
                        场景查看=box_view_state(st.R1),
```

`do_GET` 里 `st = self._store(q)` 那行**之后**、`if u.path == "/api/state":` 之前加:

```python
                if u.path.startswith("/3d"):
                    # 3D tab 住在 scene_view.py, 到这里才 import(懒加载): 不点它, 闸口不多读一个字节;
                    # 它坏了只这条路 500(下面的 except 兜住), 决策台照常。
                    if "scene_view" not in sys.modules:
                        print("[3D] 挂载 scene_view.py", flush=True)
                        if HERE not in sys.path: sys.path.insert(0, HERE)
                    import scene_view
                    if scene_view.handle(self, u.path, st.R1):
                        return
```

- [ ] **Step 4: 顶栏**

header 里 `<span class="gsw" id="rcw"></span>` 后面加:

```html
  <a id="to3d" class="ghd" target="_blank" hidden>3D ↗</a>
```

CSS(`.gsw button.on` 那条附近)加:

```css
 #to3d{text-decoration:none;border:1px solid var(--ink);padding:4px 14px;font-size:12px;color:var(--ink)}
 #to3d:hover{background:var(--ink);color:#fff}
 #to3d.off{opacity:.4;cursor:not-allowed}
 .im a.to3d{float:right;font-size:11px;text-decoration:none;border:1px solid var(--ink);padding:0 5px;color:var(--ink)}
 .im a.to3d:hover{background:var(--ink);color:#fff}
```

`boot()` 里 `S.meta = {...};` 后加:

```js
  S.meta.场景查看 = d.场景查看 || {在:false};
  const v = document.getElementById("to3d");
  v.hidden = false;
  if (S.meta.场景查看.在){ v.href = U('/3d'); v.title = "3D view: point cloud, boxes, final placement (data " + S.meta.场景查看.生成于 + ")"; }
  else { v.classList.add('off'); v.removeAttribute('href'); v.title = "No 3D data yet — it appears once gate 3 (boxes) has run"; }
```

- [ ] **Step 5: 卡片 3D 链接 + `?sel=`**

实例卡 innerHTML 里 `'<b>#'+r.oid+'</b><em>'+r.中位面积pct+'%</em>'+` 改成:

```js
        '<b>#'+r.oid+'</b><em>'+r.中位面积pct+'%</em>'+
        (S.meta.场景查看 && S.meta.场景查看.在
          ? '<a class="to3d" target="_blank" title="Open this object in the 3D view" href="'
            +U('/3d')+'#sel='+encodeURIComponent(w.词+':'+r.oid)+'">3D ↗</a>' : '')+
```

`boot()` 末尾 `if (GATE === 'scale') scLoad(); else render();` 后加:

```js
  const want = new URLSearchParams(location.search).get('sel');
  if (want && GATE === 'review') 跳到实例(want);
```

`boot` 函数后面加:

```js
// 从 3D 页跳回来: ?sel=词:实例 —— 打开那个词的弹层, 滚到那张卡, 黄框闪一下。
// 实例列表要先从服务端拿(展开面板 里 取实例 是异步的), 所以轮询等卡片出现, 最多 4 秒。
function 跳到实例(k){
  const i = k.lastIndexOf(':'); if (i < 0) return;
  const 词 = k.slice(0, i), oid = k.slice(i + 1);
  if (!S.词.has(词)) return;
  开弹层({view:'insts', 词, oid:null});
  let n = 0;
  const t = setInterval(() => {
    const el = document.querySelector('.inst[data-key="' + CSS.escape(词 + ':' + oid) + '"]');
    if (el){ clearInterval(t); el.scrollIntoView({block:'center'});
      el.classList.add('flash'); setTimeout(() => el.classList.remove('flash'), 2400); }
    else if (++n > 40) clearInterval(t);
  }, 100);
}
```

- [ ] **Step 6: 跑测试,通过**

Run: `bash ~/sam3d/tools/tests/test_gate_3d.sh`
Expected: `state ok ... lazy ok: not imported yet / /3d ok / lazy ok: imported on first /3d / /3d/data ok N / /3d/state ok / greyed ok / /3d without data ok / gate still serves N rows / ALL OK`

- [ ] **Step 7: 浏览器走一遍两边跳**

```bash
cd ~/sam3d && ~/miniforge3/envs/sam3d/bin/python tools/gate_ui.py --scene office --run r6_pillarfix --indoor --port 8799
```
`http://localhost:8799`:顶栏有 `3D ↗`;点开 `table` 的弹层,某张卡点 `3D ↗` → 新标签页 3D 视图选中的是 `Table #N`,镜头飞过去,侧栏有 "‹ Back to the gate";点它 → 回到闸口,`table` 的弹层自动开着,那张卡黄框闪。改 `/runs` 换到 r5_columns(没有 json 的话)→ 顶栏 3D 变灰。关服务。

- [ ] **Step 8: 提交**

```bash
cd ~/sam3d && git add tools/gate_ui.py tools/tests/test_gate_3d.sh && git commit -m "闸口: 挂 /3d(懒加载 scene_view), 顶栏和卡片的 3D 链接, ?sel= 跳回实例

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `drive.py` 闸 3/4/5 自动出数据

**Files:**
- Modify: `.claude/skills/run-recon-pipeline/drive.py`(`gate_report` 377-395 行;`run_steps` 后面加函数)
- Create: `tools/tests/test_drive_box_view.py`

**Interfaces:**
- Consumes: `scene_view.data_path/stale`(Task 4)、`box_fit_view.py --out off`(Task 3)。
- Produces: `ensure_box_view(R1, dry=False) -> "新鲜"|"干跑"|"重算"|"失败"`。

- [ ] **Step 1: 备份**

```bash
cd ~/sam3d/.claude/skills/run-recon-pipeline && cp drive.py drive.py.bak_3dtab
```

- [ ] **Step 2: 写测试** `tools/tests/test_drive_box_view.py`

```python
#!/usr/bin/env python
"""drive.ensure_box_view 的判"要不要重算": 临时目录 + dry, 不真跑 box_fit_view。跑: $PY tools/tests/test_drive_box_view.py"""
import importlib.util, os, tempfile, time
p = os.path.expanduser("~/sam3d/.claude/skills/run-recon-pipeline/drive.py")
spec = importlib.util.spec_from_file_location("drive", p); drive = importlib.util.module_from_spec(spec); spec.loader.exec_module(drive)

def test_missing_stale_fresh():
    with tempfile.TemporaryDirectory() as R1:
        assert drive.ensure_box_view(R1, dry=True) == "干跑"                 # 没有 -> 要算
        open(f"{R1}/10_box_fit_view.json", "w").write("{}")
        assert drive.ensure_box_view(R1, dry=True) == "新鲜"                 # 有且不旧 -> 跳过
        g = f"{R1}/07_scene_phy.glb"; open(g, "wb").write(b"x"); os.utime(g, (time.time() + 10,) * 2)
        assert drive.ensure_box_view(R1, dry=True) == "干跑"                 # 产物比它新 -> 要算

if __name__ == "__main__":
    test_missing_stale_fresh(); print("ok test_missing_stale_fresh")
```

- [ ] **Step 3: 跑,确认失败**

Run: `cd ~/sam3d && ~/miniforge3/envs/sam3d/bin/python tools/tests/test_drive_box_view.py`
Expected: `AttributeError: module 'drive' has no attribute 'ensure_box_view'`

- [ ] **Step 4: 实现**

`run_steps` 后面加:

```python
def ensure_box_view(R1, dry=False):
    """闸口网页 3D tab 要的数据 <R1>/10_box_fit_view.json: 不在、或比产物旧, 就重算(office 实测 18 s);
    否则跳过。判"旧"的规则在 tools/scene_view.py(闸口网页也用同一份报"过时"), 别在这再抄。
    失败不挡闸 —— 3D 页是看的, 不是流程的一环。"""
    if TOOLS not in sys.path: sys.path.insert(0, TOOLS)
    import scene_view
    p = scene_view.data_path(R1)
    why = "还没有" if not os.path.exists(p) else scene_view.stale(R1)
    if not why:
        return "新鲜"
    cmd = [PY, f"{TOOLS}/box_fit_view.py", "--r1", R1, "--out", "off"]
    print(f"\n[3D 页] 数据{why if why == '还没有' else '比 ' + why + ' 旧'}, 重算 (~20 s): $ {' '.join(cmd)}", flush=True)
    if dry:
        return "干跑"
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[3D 页] 失败(不挡闸): " + (r.stderr or r.stdout).strip()[-400:]); return "失败"
    print(f"[3D 页] {p}   闸口网页顶栏 3D 就是它")
    return "重算"
```

`gate_report` 里 `lines, look = CHECKS[fn](...)` 的 try/except **之后**加:

```python
    if 2 <= i <= 4:      # 闸 3/4/5: 盒子量完才有得看; 闸 6 和闸 5 之间场景没变, 不重算; 闸 1/2 没盒子
        ensure_box_view(R1)
        lines.append("3D: open the gate page (tools/gate_ui.py) and press 3D in the top bar")
```

- [ ] **Step 5: 跑测试,通过**

Run: `cd ~/sam3d && ~/miniforge3/envs/sam3d/bin/python tools/tests/test_drive_box_view.py`
Expected: `ok test_missing_stale_fresh`

- [ ] **Step 6: 真的 drive 一下(只读)**

Run: `cd ~/sam3d && ~/miniforge3/envs/sam3d/bin/python .claude/skills/run-recon-pipeline/drive.py --scene office --run r6_pillarfix --indoor --status`
Expected: 照旧打印段表(这条不调 gate_report,只是确认 import 没坏)。再跑一次没有 `--go` 的 `drive.py --scene office --run r6_pillarfix --indoor`:如果它停在闸 ≥3,会看到 `[3D 页]` 那行(数据新鲜就是不打印、直接列 `3D: open the gate page`);如果它没停在闸口,跳过这步。**不要加 `--go`**。

- [ ] **Step 7: 提交**

```bash
cd ~/sam3d && git add .claude/skills/run-recon-pipeline/drive.py tools/tests/test_drive_box_view.py && git commit -m "drive: 闸 3/4/5 自动出 3D 页数据(不在或过时才重算, 失败不挡闸)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: 测试入口、README、日志、验收

**Files:**
- Create: `tools/tests/run.sh`
- Modify: `tools/README.md`(两张表)、`devLog/devlog_9.21.md`(加一节)

- [ ] **Step 1: `tools/tests/run.sh`**

```bash
#!/bin/bash
# 跑 tools/ 的全部测试(没有 pytest, 都是带 assert 的脚本)。快的先跑; 后两个要 office r6 在盘上, 各 ~40 s。
set -e; cd "$(dirname "$0")"; PY=~/miniforge3/envs/sam3d/bin/python
$PY test_gate_locate.py; $PY test_scene_view.py; $PY test_drive_box_view.py
bash test_box_fit_view_data.sh; bash test_gate_3d.sh
echo "== tools/tests 全过 =="
```

Run: `bash ~/sam3d/tools/tests/run.sh` → 以 `== tools/tests 全过 ==` 结束。

- [ ] **Step 2: README**

"流程自动调的"表加一行:

```
| `box_fit_view.py --out off` | `drive.py` 闸 3 / 4 / 5(`ensure_box_view`, 不在或过时才重算) | `10_box_fit_view.json` —— 闸口网页 3D tab 的数据 |
```

"手动排查用的"表 `box_fit_view.py` 那行末尾补:`★ 09-23 起出口是 json + html 两份;闸口网页(`gate_ui.py`)顶栏 `3D` 直接看,不用再手敲;`--out off` 只出 json。` 并加一行:

```
| `scene_view.py` | 闸口网页 3D tab 的服务端(`/3d` `/3d/data` `/3d/state`)。只标准库, gate_ui 碰到 `/3d` 才 import。**不是**独立命令 |
```

"写新工具要注意"后加一段:

```
**闸口网页(`gate_ui.py`)里的图:** 特写(`cutout_rgb`)和整帧定位图(`locate_rgb`)共用一个提亮 gamma(`_gamma`,
mask 内中位亮度 < 110 才动)。改一个的亮度就是改两个。测试在 `tests/`(`bash tests/run.sh`, 没有 pytest)。
```

- [ ] **Step 3: 日志**

`devLog/devlog_9.21.md` 末尾加一节 `## 11. 9.23 闸口看得清 + 3D 页搬进闸口`,写:问题(灰底抠图看不出在哪/是什么、3D 页手敲 + 7 MB artifact)、做法(定位图 + 提亮;json 出口 + `/3d` 懒加载;闸 3/4/5 自动出)、进流水线的开关(`drive.py ensure_box_view`;`box_fit_view.py --data/--out off`)、验收结果(测试全过;浏览器两边跳的结果)。

- [ ] **Step 4: 对 spec 的"怎么验"逐条打勾**

1. 看得清:Task 2 Step 3 ✔ 2. 懒加载:`test_gate_3d.sh` ✔ 3. 两边定位:Task 6 Step 7 ✔ 4. 坏了不连累:`test_gate_3d.sh` ✔ 5. 单文件还能出:`test_box_fit_view_data.sh` + Task 5 Step 6 ✔ 6. 自动出:`test_drive_box_view.py` + Task 7 Step 6 ✔

- [ ] **Step 5: 提交**

```bash
cd ~/sam3d && git add tools/tests/run.sh tools/README.md devLog/devlog_9.21.md && git commit -m "文档: 闸口定位图 / 3D tab / 测试入口

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
