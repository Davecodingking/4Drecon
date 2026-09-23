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
