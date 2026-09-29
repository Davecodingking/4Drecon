#!/usr/bin/env python
"""box_fit_view.py 的找地面(find_floor): 大片平地不许把内存吃爆。
2026-09-24 nfl1(球场, 地面内点 11 万): SVD 默认 full_matrices=True, 顺手算了个 N×N 的 U,
11 万² × 8 字节 ≈ 100 GB, 峰值 194 GB、4 分 42 秒, 驱动卡在闸口"运行中"。
box_fit_view.py 是脚本(一 import 就整个跑), 这里只把 _rot / find_floor 两个函数抠出来单测,
放进限了 8 GB 地址空间的子进程里跑 —— 再有 N×N 的东西会直接 MemoryError, 不会真的吃掉机器。
跑: $PY tools/tests/test_box_fit_floor.py"""
import ast, os, resource, subprocess, sys, textwrap, time

SAM3D = os.path.expanduser("~/sam3d")
SRC = f"{SAM3D}/tools/box_fit_view.py"
LIMIT = 8 << 30


def _funcs():
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    return "\n\n".join(ast.get_source_segment(open(SRC, encoding="utf-8").read(), n)
                       for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_rot", "find_floor"))


def test_big_flat_ground_stays_small():
    child = "import numpy as np\n" + _funcs() + textwrap.dedent("""
        r = np.random.default_rng(1)
        g = np.c_[r.uniform(-40, 40, 120000), r.normal(0, 0.01, 120000), r.uniform(-40, 40, 120000)]
        junk = np.c_[r.uniform(-40, 40, 20000), r.uniform(0.3, 3, 20000), r.uniform(-40, 40, 20000)]
        n, q0, nin = find_floor(np.r_[g, junk])
        assert nin > 100000, nin
        assert n[1] > 0.999, n
        print("ok", nin, n.round(4).tolist())
    """)
    t = time.time()
    r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True,
                       preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT)))
    took = time.time() - t
    assert r.returncode == 0, f"找地面在 8 GB 以内跑不下来:\n{r.stderr[-800:]}"
    assert took < 60, f"找地面用了 {took:.0f}s"
    print("  ", r.stdout.strip(), f"{took:.1f}s")


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
