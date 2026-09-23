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
