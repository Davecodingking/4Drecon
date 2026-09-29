#!/usr/bin/env python
"""tools/run_page.py —— 启动页 /new、本轮页 /run 和它们的接口。假 handler + 沙盒里的 run_ctl, 不起服务。
跑: $PY tools/tests/test_run_page.py"""
import io, json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np, cv2
import run_page, run_ctl
from test_run_ctl import _sandbox, _state


class FakeH:
    def __init__(self): self.code = None; self.ctype = None; self.hdr = {}; self.wfile = io.BytesIO()
    def _send(self, code, ctype, body, cache=False): self.code, self.ctype = code, ctype; self.wfile.write(body)
    def _json(self, obj, code=200): self._send(code, "application/json; charset=utf-8", json.dumps(obj, ensure_ascii=False).encode())
    def body(self): return self.wfile.getvalue()
    def j(self): return json.loads(self.body())


def _h(method, path, q=None, body=None):
    h = FakeH()
    assert run_page.handle(h, method, path, q or {}, body), path
    return h


def test_pages_and_unknown_paths():
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        assert "新建一轮".encode() in _h("GET", "/new").body()
        assert b"/api/run/state" in _h("GET", "/run").body()
        assert run_page.handle(FakeH(), "GET", "/", {}) is False          # 不是它的路由, 交回 gate_ui
        assert run_page.handle(FakeH(), "GET", "/api/state", {}) is False


def test_new_run_flow():
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        d = _h("GET", "/api/datasets").j()
        assert [x["场景"] for x in d["outdoor"]] == ["beach", "nfl", "soccer"]
        h = _h("POST", "/api/new", body={"类型": "indoor", "场景": "office", "轮次": "r3", "有人": True, "召回": False})
        assert h.code == 200 and h.j()["url"] == "/run?s=office&r=r3&i=1", h.body()
        assert json.load(open(f"{T}/ir/office/r3/00_run_config.json"))["覆盖"] == {"PEOPLE": "1", "RECALL": "0"}
        h = _h("POST", "/api/new", body={"类型": "indoor", "场景": "office", "轮次": "r3", "有人": True, "召回": False})
        assert h.code == 400 and "已经存在" in h.j()["error"]


def test_run_state_config_people_go():
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        R1 = run_ctl.new_run("outdoor", "soccer", "r1", people="0", recall="0")
        _state(R1, 0, True)
        q = {"s": "soccer", "r": "r1"}
        assert _h("GET", "/api/run/state", q).j()["闸"] == 1
        h = _h("POST", "/api/run/config", q, {"PEOPLE": "1"})
        assert h.code == 200 and _h("GET", "/api/run/state", q).j()["有人"] == "1"
        _state(R1, 1, True)
        assert _h("POST", "/api/run/config", q, {"PEOPLE": "0"}).code == 400
        json.dump({"人": [{"id": 1, "决定": "keep"}, {"id": 2, "决定": "keep"}]}, open(f"{R1}/05_people.json", "w"))
        assert _h("POST", "/api/run/people", q, {"2": "drop"}).code == 200
        assert [r["决定"] for r in _h("GET", "/api/run/state", q).j()["人"]["人"]] == ["keep", "drop"]
        os.remove(f"{T}/go_args")
        assert _h("POST", "/api/run/go", q, {}).code == 200 and os.path.exists(f"{T}/go_args")
        assert _h("GET", "/api/run/state", {"s": "../x", "r": "r1"}).code == 400


def test_images():
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        R1 = run_ctl.new_run("outdoor", "soccer", "r1", people="1", recall="0")
        q = {"s": "soccer", "r": "r1"}
        assert _h("GET", "/run/img", dict(q, what="ground")).code == 404
        cv2.imwrite(f"{R1}/02_ground_check.jpg", np.zeros((20, 30, 3), np.uint8))
        h = _h("GET", "/run/img", dict(q, what="ground"))
        assert h.code == 200 and h.ctype == "image/jpeg"
        # 挑人缩略图: 按决策文件里的跟踪目录, 裁出那一帧里这个人的 mask 外接框
        trk = f"{T}/trk"; os.makedirs(f"{trk}/images"); os.makedirs(f"{trk}/masks")
        cv2.imwrite(f"{trk}/images/00000003.jpg", np.full((100, 100, 3), 200, np.uint8))
        m = np.zeros((100, 100), np.uint8); m[20:60, 40:50] = 2
        # body4d 的 mask 是调色板 PNG(像素值 = 人的 id): cv2.imread 会把它展开成 3 通道彩色, 必须照真的格式造
        from PIL import Image
        pm = Image.fromarray(m, mode="P"); pm.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * 759)
        pm.save(f"{trk}/masks/00000003.png")
        json.dump({"跟踪目录": trk, "人": [{"id": 2, "决定": "keep", "最大面积帧": 3}]}, open(f"{R1}/05_people.json", "w"))
        h = _h("GET", "/run/img", dict(q, what="person", id="2"))
        assert h.code == 200 and h.ctype == "image/png"
        img = cv2.imdecode(np.frombuffer(h.body(), np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[0] == 180 and img.shape[1] < img.shape[0]         # 竖长的人 -> 竖长的图
        assert _h("GET", "/run/img", dict(q, what="person", id="9")).code == 404
        assert _h("GET", "/run/img", dict(q, what="../../etc")).code == 404


def test_media_whitelist():
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        R1 = run_ctl.new_run("outdoor", "soccer", "r1", people="1", recall="0")
        q = {"s": "soccer", "r": "r1"}
        cv2.imwrite(f"{R1}/08_render_back_phy.jpg", np.zeros((20, 30, 3), np.uint8))
        open(f"{R1}/09_people_check.mp4", "wb").write(b"\x00\x00\x00 ftypisom")
        h = _h("GET", "/run/media", dict(q, what="render"))
        assert h.code == 200 and h.ctype == "image/jpeg"
        h = _h("GET", "/run/media", dict(q, what="people4d"))
        assert h.code == 200 and h.ctype == "video/mp4"
        assert _h("GET", "/run/media", dict(q, what="overlay")).code == 404          # 白名单里有, 盘上没有
        assert _h("GET", "/run/media", dict(q, what="people4d_prev")).code == 404    # 没重跑过就没有上一版
        open(f"{R1}/09_people_check_prev.mp4", "wb").write(b"\x00\x00\x00 ftypisom")
        assert _h("GET", "/run/media", dict(q, what="people4d_prev")).code == 200
        assert _h("GET", "/run/media", dict(q, what="../../etc/passwd")).code == 404


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
