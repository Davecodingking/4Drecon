#!/usr/bin/env python
"""两轮的 track 和出人网格结果对比(并行调度的验证用)。
  $PY tools/compare_runs.py <轮次A的结果目录> <轮次B的结果目录>
track: 每个词的 npz 里 meta(实例/帧/面积)和 RLE 是不是一模一样。
出人网格: 每人每帧的顶点差多少(毫米), 以及每人有网格的帧数。"""
import glob, os, sys
import numpy as np
import trimesh


def track(A, B):
    fa = {os.path.basename(p) for p in glob.glob(f"{A}/03_masks/*.npz")}
    fb = {os.path.basename(p) for p in glob.glob(f"{B}/03_masks/*.npz")}
    same = fa == fb
    if not same:
        print(f"  词不一样: 只在 A {sorted(fa - fb)}  只在 B {sorted(fb - fa)}")
    for f in sorted(fa & fb):
        a, b = np.load(f"{A}/03_masks/{f}", allow_pickle=True), np.load(f"{B}/03_masks/{f}", allow_pickle=True)
        ok = (a["meta"].shape == b["meta"].shape and (a["meta"] == b["meta"]).all()
              and len(a["counts"]) == len(b["counts"]) and all(x == y for x, y in zip(a["counts"], b["counts"])))
        same &= bool(ok)
        n = len(set(a["meta"]["oid"])) if len(a["meta"]) else 0
        print(f"  {f[:-4]:16s} {'一样' if ok else '不一样'}   A {len(a['meta'])} 条记录 {n} 个实例, B {len(b['meta'])} 条")
    return same


def mesh(A, B):
    worst = 0.0
    da, db = (glob.glob(f"{x}/body4d/04_mesh_occ*/mesh_4d_individual")[0] for x in (A, B))
    for pid in sorted(set(os.listdir(da)) | set(os.listdir(db)), key=int):
        fa = set(os.listdir(f"{da}/{pid}")) if os.path.isdir(f"{da}/{pid}") else set()
        fb = set(os.listdir(f"{db}/{pid}")) if os.path.isdir(f"{db}/{pid}") else set()
        d = [float(np.abs(np.asarray(trimesh.load(f"{da}/{pid}/{f}", process=False).vertices)
                          - np.asarray(trimesh.load(f"{db}/{pid}/{f}", process=False).vertices)).max())
             for f in sorted(fa & fb)]
        mx = 1000 * max(d) if d else float("nan")
        worst = max(worst, mx if d else 0.0)
        if fa != fb:
            worst = float("inf")
        print(f"  人 {pid}: A {len(fa)} 帧, B {len(fb)} 帧, 共有 {len(d)} 帧; 顶点最大差 {mx:.4f} mm, "
              f"中位 {1000 * float(np.median(d)) if d else float('nan'):.4f} mm" + ("   ★ 帧不一样" if fa != fb else ""))
    return worst


if __name__ == "__main__":
    A, B = (os.path.abspath(os.path.expanduser(x)) for x in sys.argv[1:3])
    print(f"A = {A}\nB = {B}\n== track"); t = track(A, B)
    print("== 出人网格"); m = mesh(A, B)
    print(f"\nTRACK {'一致' if t else '不一致'}  MESH 最大差 {m:.4f} mm")
