#!/usr/bin/env python
"""出人网格按人拆开跑之后, 把几份并回一个运行目录, 结构和不拆时一样(2026-09-29)。

  python mesh_merge.py --run <出网格目录> --trk <跟踪目录> --ids 1,2,4 --parts 3 [--seconds 210.5]

拆开跑时每一份(<出网格目录>/_part<k>/)是 body4d_mesh_5090.py 的一次完整输出, 只是里面只有那几个人,
而且编号是本份里的 1..n(见那个脚本"重编号"一节)。这里换成全体统一的编号 —— 规则和不拆时一样:
按挑人的顺序 1..N —— 下游(body_into_scene / body4d_report)读到的目录和不拆时没有区别。

★ 检查用的叠加图(rendered_frames)是把几份的图叠起来的: 不同份里的人互相遮挡的前后关系可能不对,
  颜色也可能重复。它只给人看, 后面的步骤不读; 验收视频 09_people_check.mp4 是另外渲染的。

sam3d 环境(numpy + PIL)。纯 CPU。
"""
import argparse, json, os, shutil
import numpy as np
from PIL import Image

# mhr_4d_individual: 模型的一手数据(每帧的 MHR 参数, 2026-10-08 起存), 老的输出没有这个目录, 没有就跳过
PER_PERSON = ("mesh_4d_individual", "focal_4d_individual", "mhr_4d_individual", "rendered_frames_individual")
WHITE = 235          # 叠加图是白底: 三个通道都不低于这个数的像素当背景。不取 250: JPEG 在人的边上有一圈
                     # 接近白的振铃, 当成前景贴过去会在别人身上留白边


def merge(run, trk, ids, parts, seconds=None):
    G = {o: i + 1 for i, o in enumerate(ids)}
    got = []
    for k in range(parts):
        p = f"{run}/_part{k}"
        rel = {int(o): int(l) for o, l in json.load(open(f"{p}/relabel.json"))["relabel"].items()}
        got.append((p, rel, json.load(open(f"{p}/mesh_report.json"))))
    have = sorted(o for _, rel, _ in got for o in rel)
    if have != sorted(ids):
        raise SystemExit(f"[失败] 几份加起来的人 {have} 和要出网格的 {sorted(ids)} 对不上")
    # ★ 空网格(文件头里写着 0 个顶点)不能收: 2026-09-29 nfl1 r3 人 2/6/7 拆开后全是空的 —— 这几个人有
    #   "模型拒绝出网格"的帧, 那一帧里没有别人时上游从此全出空网格。以前照收, 到 scale 那步才炸
    empty = {}
    for p, rel, _ in got:
        for o, l in rel.items():
            d = f"{p}/mesh_4d_individual/{l}"
            n = sum(b"element vertex 0\n" in open(f"{d}/{f}", "rb").read(400)
                    for f in (os.listdir(d) if os.path.isdir(d) else []))
            if n:
                empty[o] = n
    if empty:
        raise SystemExit(f"[失败] 拆开跑之后这些人出了空网格 {{人: 空的帧数}} = {empty}。"
                         f"这几个人不能单独拆出去, 用 FANOUT=0 重跑这一步(不拆)")
    # 先清掉上一次留下的(上次挑的人可能更多), 再把各份的搬进来
    for d in PER_PERSON + ("rendered_frames", "masks"):
        shutil.rmtree(f"{run}/{d}", ignore_errors=True)
    for p, rel, _ in got:
        for d in PER_PERSON:
            for o, l in rel.items():
                if os.path.isdir(f"{p}/{d}/{l}"):
                    os.makedirs(f"{run}/{d}", exist_ok=True)
                    shutil.move(f"{p}/{d}/{l}", f"{run}/{d}/{G[o]}")
                    if d == "rendered_frames_individual":         # 文件名里也带编号: 00000000_<编号>.jpg
                        for n in os.listdir(f"{run}/{d}/{G[o]}"):
                            stem, ext = os.path.splitext(n)
                            if stem.endswith(f"_{l}"):
                                os.rename(f"{run}/{d}/{G[o]}/{n}", f"{run}/{d}/{G[o]}/{stem[:-len(str(l)) - 1]}_{G[o]}{ext}")
    os.makedirs(f"{run}/masks"); os.makedirs(f"{run}/rendered_frames")
    for name in sorted(os.listdir(f"{got[0][0]}/masks")):
        out, pal = None, None
        for p, rel, _ in got:
            im = Image.open(f"{p}/masks/{name}")
            m = np.array(im)
            if out is None:
                out, pal = np.zeros_like(m), im.getpalette()
            for o, l in rel.items():
                out[m == l] = G[o]
        q = Image.fromarray(out).convert("P")                     # 和 body4d_mesh_5090.py 存 mask 的写法一样
        if pal:
            q.putpalette(pal)
        q.save(f"{run}/masks/{name}")
    for name in sorted(os.listdir(f"{got[0][0]}/rendered_frames")):
        out = None
        for p, _, _ in got:
            fp = f"{p}/rendered_frames/{name}"
            if not os.path.exists(fp):
                continue
            img = np.array(Image.open(fp).convert("RGB"))
            if out is None:
                out = img.copy()
            else:
                fg = (img < WHITE).any(axis=2)
                out[fg] = img[fg]
        Image.fromarray(out).save(f"{run}/rendered_frames/{name}", quality=95)
    if not os.path.lexists(f"{run}/images"):
        os.symlink(os.path.abspath(f"{trk}/images"), f"{run}/images")
    json.dump(dict(relabel={str(o): g for o, g in G.items()},
                   note="左=跟踪出的原始 id, 右=送进 SAM 3D Body 的连续 id。"
                        "mesh_4d_individual/<右>/ 对应的是 <左> 那个人"),
              open(f"{run}/relabel.json", "w"), ensure_ascii=False, indent=1)
    n = {str(g): (len(os.listdir(f"{run}/mesh_4d_individual/{g}"))
                  if os.path.isdir(f"{run}/mesh_4d_individual/{g}") else 0) for g in G.values()}
    first = got[0][2]
    json.dump(dict(ids=list(ids), ids_run=sorted(G.values()), relabel={str(o): g for o, g in G.items()},
                   frames=len(os.listdir(f"{run}/images")), meshes_per_id=n,
                   completion=first.get("completion", False), batch_size=first.get("batch_size"),
                   peak_vram_gb=max(r.get("peak_vram_gb", 0) for _, _, r in got),
                   seconds=seconds if seconds is not None else max(r.get("seconds", 0) for _, _, r in got),
                   parts=parts),
              open(f"{run}/mesh_report.json", "w"), ensure_ascii=False, indent=1)
    for p, _, _ in got:
        shutil.rmtree(p)
    print(f"[合并] {parts} 份 -> {run}   每人网格帧数: " + "  ".join(f"id{k}:{v}" for k, v in n.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--trk", required=True)
    ap.add_argument("--ids", required=True); ap.add_argument("--parts", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=None)
    a = ap.parse_args()
    merge(a.run, a.trk, [int(x) for x in a.ids.split(",")], a.parts, a.seconds)
