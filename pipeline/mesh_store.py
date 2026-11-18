#!/usr/bin/env python
"""人体网格的库: <人体目录>/04_mesh_occ{0,1}/。出网格那一步(body4d_pipeline.s_mesh)往里放, 后面的步骤从这里读。

  mesh_4d_individual/<人>/<帧>.ply   focal_4d_individual/<人>/<帧>.json   mhr_4d_individual/<人>/<帧>.npz
  masks/<帧>.png      跟踪的抠图里只留勾上的人(像素值 = 人的编号)
  images -> 跟踪目录的 images/
  store.json          登记簿: 库里有谁、各自是拿什么输入算的("印记")、这一次改完了没有
  relabel.json  mesh_report.json     给读它们的老地方留着(对照表现在两边一样)

两条规矩(2026-10-12 起):
  1. <人> = 他的**跟踪编号**(挑人卡片、03_check.jpg 上的数字), 不是"第几个勾的人"。
  2. 一个人算过、而且他的输入没变, 就不重算。点掉一个人 = 删他的目录, 别人一个字节不动。

★ 为什么(office r12, 2026-10-09): 以前目录名是"第几个勾的人"(上游拿"第几个"当"是谁"用, 我们就把勾上的人重排成 1..N)。
  闸口上点掉 7 号(一件外套), 8、9、10 号全部往前挪一个目录: 目录和人对不上了, 只能把 9 个人都重出一遍 —— 多等 3 分半,
  而且刚在闸口看过的人也被重算, 网格和看的时候差几毫米(模型在个别帧上每次算都不一样)。
  现在 8 号永远在目录 8; 重排只留在每一份自己的临时目录里(上游要连续的 1..n), 搬进库时换回跟踪编号。

"输入没变"怎么判(印记, stamps): 这个人每一帧的抠图、所有帧的图、设置(补全 / 批大小 / 焦距)、出网格的代码。
  都是按内容算的, 不看路径和文件时间 —— 把一轮整个复制到别处, 印记不变, 不重算。
  开着补全时一个人的结果要看别人挡没挡他, 所以再加上"这一批勾了谁"。
  2026-10-12 之前出的目录没有登记簿: 名字是老的排法, 也不知道是哪版代码算的 —— 第一次碰到整个清掉重算, 不猜。

改库的顺序(s_mesh 里): begin -> absorb(上次算好没收的) -> prune -> 算缺的人 -> absorb -> finish。
  begin 把库标成"没完成", finish 才标回来; 中间任何一步失败, 读它的地方(people_identity.raw_run)都会拒绝用。
  登记簿先删后加: 目录在、登记簿里没有的, 一律当垃圾清掉 —— 崩在哪一步都不会把半截的人当成算好的。

sam3d 环境(numpy + PIL)。纯 CPU。
"""
import hashlib, json, os, shutil, time
import numpy as np
# PIL 只在读写抠图的两个函数里用(现用现载): 读登记簿的地方(people_identity)不该因此多一个依赖

VERSION = 2                                # 2 = 目录名是跟踪编号 + 有登记簿
MANIFEST = "store.json"
PER_PERSON = ("mesh_4d_individual", "focal_4d_individual", "mhr_4d_individual")    # mhr: 模型的一手关节数据(10-08 起存)
# 老写法的库里还有这些: 出网格时顺手画的图。现在叠加视频由 body4d_report_5090.py 从网格现画, 库里不存图
OLD = PER_PERSON + ("rendered_frames_individual", "rendered_frames", "masks")
# 决定网格长什么样的代码(相对上游仓库 sam-body4d/)。哪个改了, 以前算的都不能再当成"同样的输入算出来的"
CODE = ("body4d_mesh_5090.py", "scripts/offline_app.py", "models/sam_3d_body/notebook/utils.py",
        "models/sam_3d_body/sam_3d_body/models/meta_arch/sam3d_body.py", "configs/body4d.yaml")


# ---------------------------------------------------------------- 印记

def focal_of(cam_json):
    """出网格用的焦距(相对图高): 场景 cameras.json 各帧 竖直焦距 / 图高 的中位数, 和 body4d_mesh_5090.py 算的是同一个数。
    没给场景相机 = None(模型自己估)。★ 不拿文件内容当印记: 尺度那步会改写 cameras.json(外参乘 k), 内参不动。"""
    if not cam_json:
        return None
    cams = json.load(open(cam_json))["cameras"]
    return float(np.median([np.array(v["intrinsic_3x3"])[1][1] / v["input_hw"][0] for v in cams.values()]))


def focal_sig(cam_json):
    """逐帧焦距的指纹(2026-11-10 起出网格每帧用自己的焦距): 中位数没变、某几帧变了也算过期。没给场景相机 = None。"""
    if not cam_json:
        return None
    cams = json.load(open(cam_json))["cameras"]
    h = hashlib.blake2b(digest_size=8)
    for k in sorted(cams):
        v = cams[k]; h.update(f"{k}:{np.array(v['intrinsic_3x3'])[1][1] / v['input_hw'][0]:.6f};".encode())
    return h.hexdigest()


def code_sig(b4d):
    h = hashlib.blake2b(digest_size=8)
    for rel in CODE:
        if os.path.exists(f"{b4d}/{rel}"):
            h.update(rel.encode()); h.update(open(f"{b4d}/{rel}", "rb").read())
    return h.hexdigest()


def frames_sig(trk):
    h = hashlib.blake2b(digest_size=8)
    for name in sorted(os.listdir(f"{trk}/images")):
        h.update(name.encode()); h.update(open(f"{trk}/images/{name}", "rb").read())
    return h.hexdigest()


def mask_sigs(trk, ids):
    """{人: 他的抠图的指纹} —— 哪些帧、每帧哪些像素。别人的抠图变了不影响他的。"""
    from PIL import Image
    hs = {int(i): hashlib.blake2b(digest_size=12) for i in ids}
    for name in sorted(os.listdir(f"{trk}/masks")):
        if not name.endswith(".png"):
            continue
        m = np.array(Image.open(f"{trk}/masks/{name}"))
        flat = m.ravel()
        idx = np.flatnonzero(flat)
        lab = flat[idx]
        for o in np.unique(lab):
            if int(o) in hs:
                h = hs[int(o)]
                h.update(name.encode()); h.update(np.int64(m.shape).tobytes()); h.update(idx[lab == o].astype(np.int64).tobytes())
    return {o: h.hexdigest() for o, h in hs.items()}


def stamps(trk, ids, settings, b4d):
    """{人: 印记}。印记 = 决定这个人网格的所有输入; 和库里登记的一样, 这个人就不用重算。
    settings: dict(补全, batch, max_occ_len, 焦距) —— 只放能原样写进 json 的值。"""
    base = dict(设置=dict(settings), 代码=code_sig(b4d), 帧=frames_sig(trk))
    ms = mask_sigs(trk, ids)
    together = sorted(int(i) for i in ids) if str(settings.get("补全")) == "1" else None
    return {int(i): dict(base, 抠图=ms[int(i)], 同批的人=together) for i in ids}


# ---------------------------------------------------------------- 登记簿

def _load(run):
    try:
        m = json.load(open(f"{run}/{MANIFEST}"))
    except (OSError, ValueError):
        return None
    return m if m.get("版本") == VERSION else None


def _save(run, man):
    os.makedirs(run, exist_ok=True)
    tmp = f"{run}/{MANIFEST}.tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    os.replace(tmp, f"{run}/{MANIFEST}")


def people(run):
    """{人: dict(印记, 帧, 算于, 显存峰值)} —— 库里登记着的人。不是新写法的库 = 空。"""
    return {int(o): e for o, e in ((_load(run) or {}).get("人") or {}).items()}


def finished(run):
    """True = 上一次改库改完了, 能往下用; False = 改到一半(出网格那步失败了); None = 不是新写法的库(老轮次)。"""
    man = _load(run)
    return None if man is None else bool(man.get("完成"))


def _forget(run, man, o):
    """先从登记簿上划掉, 再删目录: 崩在中间留下的是"没登记的目录", prune 会清。"""
    if man["人"].pop(str(o), None) is not None:
        _save(run, man)
    for d in PER_PERSON:
        shutil.rmtree(f"{run}/{d}/{o}", ignore_errors=True)


# ---------------------------------------------------------------- 改库

def begin(run):
    """开始改库。老写法的库(没有登记簿)整个清掉, 返回 True。库标成"没完成", 上一次的报告删掉(它是这一步跑完了的标志)。"""
    man, wiped = _load(run), False
    if man is None:
        leftovers = [x for x in OLD + ("relabel.json", "mesh_report.json") if os.path.lexists(f"{run}/{x}")]
        wiped = bool(leftovers)
        for x in leftovers:
            shutil.rmtree(f"{run}/{x}") if os.path.isdir(f"{run}/{x}") else os.remove(f"{run}/{x}")
        man = dict(版本=VERSION, 完成=False, 人={},
                   说明="人体网格的库(pipeline/mesh_store.py)。<人> = 跟踪编号; 印记 = 这个人是拿什么输入算的, 没变就不重算")
    man["完成"] = False
    _save(run, man)
    if os.path.exists(f"{run}/mesh_report.json"):
        os.remove(f"{run}/mesh_report.json")
    return wiped


def prune(run, want):
    """库里不该留的人删掉: 闸口上没勾的、输入变了的、没登记的(上次搬到一半)。want: {人: 印记}。返回 {人: 原因}。"""
    man, gone = _load(run), {}
    for o, e in sorted(people(run).items()):
        if o not in want:
            gone[o] = "闸口上没勾"
        elif e.get("印记") != want[o]:
            diff = [k for k in want[o] if (e.get("印记") or {}).get(k) != want[o][k]] or ["印记"]
            gone[o] = "、".join(diff) + "变了, 以前算的不能用"
    for o in gone:
        _forget(run, man, o)
    for d in PER_PERSON:
        for x in (os.listdir(f"{run}/{d}") if os.path.isdir(f"{run}/{d}") else []):
            if x not in man["人"]:
                shutil.rmtree(f"{run}/{d}/{x}", ignore_errors=True)
                gone.setdefault(int(x) if x.isdigit() else x, "目录在但没登记(上次搬到一半), 不知道完不完整")
    return gone


def seal(part, stamps_of):
    """这一份算完了(出网格脚本正常退出之后才调): 记下里面每个人是拿什么输入算的。没盖这个章的一份, absorb 不收。"""
    with open(f"{part}/part.json", "w") as f:
        json.dump(dict(人={str(o): s for o, s in stamps_of.items()}, 算于=time.strftime("%F %T")), f, ensure_ascii=False)


def absorb(run, part, want):
    """把算完的一份搬进库: 只收这次还要、而且印记对得上的人, 目录名换成跟踪编号。这一份随后删掉。返回收进来的人。
    没盖章的(没算完)一个都不收。"""
    try:
        meta = json.load(open(f"{part}/part.json"))
        rel = {int(o): int(l) for o, l in json.load(open(f"{part}/relabel.json"))["relabel"].items()}
        rep = json.load(open(f"{part}/mesh_report.json"))
        stamped = meta["人"]
    except (OSError, ValueError, KeyError, TypeError):
        shutil.rmtree(part, ignore_errors=True)
        return []
    take = [o for o in sorted(rel) if o in want and stamped.get(str(o)) == want[o]]
    # ★ 空网格(文件头里写着 0 个顶点)不能收: 2026-09-29 nfl1 r3 人 2/6/7 拆开后全是空的。以前照收, 到 scale 那步才炸
    empty = {}
    for o in take:
        d = f"{part}/mesh_4d_individual/{rel[o]}"
        n = sum(b"element vertex 0\n" in open(f"{d}/{f}", "rb").read(400) for f in (os.listdir(d) if os.path.isdir(d) else []))
        if n:
            empty[o] = n
    if empty:
        os.remove(f"{part}/part.json")           # 撕掉章: 不然下次还是收这份空的, 永远过不去
        raise SystemExit(f"[失败] 这些人出了空网格 {{人: 空的帧数}} = {empty}。这是出网格脚本的问题(看 04_mesh.log); "
                         f"重跑这一步会重算他们, 想和别人放在一份里算就加 FANOUT=0")
    man = _load(run)
    for o in take:
        _forget(run, man, o)
        for d in PER_PERSON:
            if os.path.isdir(f"{part}/{d}/{rel[o]}"):
                os.makedirs(f"{run}/{d}", exist_ok=True)
                shutil.move(f"{part}/{d}/{rel[o]}", f"{run}/{d}/{o}")
        md = f"{run}/mesh_4d_individual/{o}"
        man["人"][str(o)] = dict(印记=want[o], 帧=len(os.listdir(md)) if os.path.isdir(md) else 0,
                                 算于=meta.get("算于"), 显存峰值=rep.get("peak_vram_gb", 0))
        _save(run, man)
    shutil.rmtree(part)
    return take


def finish(run, trk, ids, new=(), seconds=0.0, parts=0, settings=None):
    """收尾: 抠图(只留勾上的人)、图的链接、对照表、报告, 最后把库标成"完成"。ids: 这次勾的人(按挑人的顺序); new: 这次新算的。"""
    from PIL import Image
    man, have = _load(run), people(run)
    miss = [o for o in ids if o not in have]
    if man is None or miss:
        raise SystemExit(f"[失败] 库里缺这些人的网格: {miss} —— 出网格没跑完, 重跑这一步")
    shutil.rmtree(f"{run}/masks", ignore_errors=True)
    os.makedirs(f"{run}/masks")
    for name in sorted(os.listdir(f"{trk}/masks")):
        im = Image.open(f"{trk}/masks/{name}")
        m = np.array(im)
        q = Image.fromarray(np.where(np.isin(m, list(ids)), m, 0).astype(m.dtype)).convert("P")   # 和 body4d_mesh_5090.py 存 mask 的写法一样
        if im.getpalette():
            q.putpalette(im.getpalette())
        q.save(f"{run}/masks/{name}")
    if os.path.lexists(f"{run}/images"):         # 链接每次重指: 这一轮可能是从别处复制来的, 老链接还指着原来那一轮
        os.remove(f"{run}/images")
    os.symlink(os.path.abspath(f"{trk}/images"), f"{run}/images")
    same = {str(o): int(o) for o in ids}
    with open(f"{run}/relabel.json", "w") as f:
        json.dump(dict(relabel=same, note="目录名就是跟踪编号(2026-10-12 起), 两边一样。留着这个文件是给读它的地方用"),
                  f, ensure_ascii=False, indent=1)
    s = settings or {}
    new = [int(o) for o in new]
    with open(f"{run}/mesh_report.json", "w") as f:
        json.dump(dict(ids=[int(o) for o in ids], ids_run=sorted(int(o) for o in ids), relabel=same,
                       frames=len(os.listdir(f"{run}/images")), meshes_per_id={str(o): have[o]["帧"] for o in ids},
                       completion=str(s.get("补全")) == "1", batch_size=int(s.get("batch") or 0),
                       peak_vram_gb=max([have[o].get("显存峰值") or 0 for o in new] or [0]),
                       seconds=seconds, parts=parts, 重算=new, 沿用=[int(o) for o in ids if o not in new]),
                  f, ensure_ascii=False, indent=1)
    for o in [o for o in have if o not in ids]:  # prune 之后不该还有; 有就是调用顺序错了, 别让它悄悄留在库里
        _forget(run, man, o)
    man["完成"] = True
    _save(run, man)
    print(f"[库] {run}   每人网格帧数: " + "  ".join(f"人{o}:{have[o]['帧']}" for o in ids)
          + f"   这次新算 {new or '无'}", flush=True)
