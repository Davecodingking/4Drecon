#!/usr/bin/env python
"""核对身份: 跟踪给的"编号"不等于"人"。把人放进场景之前, 用每个编号在房间里的位置核一遍, 出一份核对过的人体数据。

★ 为什么要有(2026-10-09 office r8/r10): 流水线原来把 SAM 3 给的编号直接当成人往下走, 而编号有三种错:
    1. 不是人的东西也有编号 —— 画面边上的一只手、别人轮廓上的几个像素(后一种在跟踪那步就摘了, 见 track_clean.py)。
       SAM 3D Body 不管给它什么都输出一个完整的人, 于是凭空多一个人。
    2. 同一个人有好几个编号 —— 人走出画面再回来, 跟踪认不出是他(office 的人 2 被拆成 2、6、10 三个编号),
       他站的位置上就多站了人。
    3. 一个编号先后指了不同的东西 —— 6 号先是一只手, 14 秒后是人 2 的头。导成骨骼动画就是一个人从这头飘到那头。
  跟踪只看画面, 不知道人在房间里的位置; 位置要到"摆人"那步才有(相机 + 场景深度)。所以核对放在摆人那步的最前面。

怎么核(对所有场景同一套, 没有按场景调的数):
  先把每个编号按时间切成段: 中断超过 GAP_S 秒就是两段(中间没看见他, 身份要重新确认)。然后段和段之间连线, 连上的是同一个人:
    - 不同编号的两段: 从不同时在场, 而且位置接得上(相距不到 SAME_M 米) -> 同一个人。
      同时在场过哪怕一帧就是两个人 —— nfl1 叠在一起的人位置一样, 但都同时在场, 一个都不并。
    - 同一个编号的前后两段: 位置接得上 -> 同一个人; 接不上但两段都够长(各 >= MIN_FRAMES 帧) -> 信跟踪的, 还是同一个人
      (人走开又回来); 接不上、又有一段只有一两帧 -> 不认(那一两帧多半是别的东西)。
    - 任何一次连线都不能把"同时在场过"的两段连进同一个人。
  连完之后, 一个人出了网格的帧不到 MIN_FRAMES -> 不算人。
已知的局限: 甲走了, 乙后来正好站到同一个位置、两人从没同框 —— 分不出来, 会并成一个人(并了谁都记在 identity.json 里)。

结果落成 <人体目录>/06_identity/, 结构和出网格目录一样(网格 / 相机 / 一手参数是链接, 抠图按新编号重画), 外加 identity.json。
摆人之后的步骤(合成、验收视频、动作导出、3D 页)都用 mesh_run_of() 找人体数据, 读到的是同一份核对过的人。
"""
import json, os, shutil
import numpy as np

GAP_S = 1.0          # 秒: 一个编号中断这么久以上, 前后算两段
SAME_M = 0.5         # 米: 两段相距不到这么多 = 站在同一个地方
MIN_FRAMES = 3       # 核对完, 出了网格的帧不到这么多的不算人
DIR = "06_identity"
PER_PERSON = {"mesh_4d_individual": ".ply", "focal_4d_individual": ".json", "mhr_4d_individual": ".npz"}


def resolve(tracks, up=None):
    """tracks: {编号: {帧号: (秒, 世界坐标 (3,))}} —— 只放出了网格的帧。up: 给了就只比水平方向的距离。
    返回 (people, log): people = {人: [(原编号, 帧号), ...] 按帧号排}; log = [dict(事, 编号, 帧, ...)] 给人看的记录。"""
    up = None if up is None else np.asarray(up, float) / np.linalg.norm(up)
    flat = (lambda p: p - (p @ up) * up) if up is not None else (lambda p: p)
    segs = []
    for rid in sorted(tracks, key=int):
        bis = sorted(tracks[rid]); cur = [bis[0]]
        for a, b in zip(bis, bis[1:]):
            if tracks[rid][b][0] - tracks[rid][a][0] > GAP_S:
                segs.append((rid, cur)); cur = []
            cur.append(b)
        segs.append((rid, cur))

    def dist(i, j):
        """两段在时间上最靠近的那一对帧, 位置差多远。"""
        (ri, bi), (rj, bj) = segs[i], segs[j]
        a, b = min(((x, y) for x in bi for y in bj), key=lambda xy: abs(tracks[ri][xy[0]][0] - tracks[rj][xy[1]][0]))
        return float(np.linalg.norm(flat(np.asarray(tracks[ri][a][1], float) - np.asarray(tracks[rj][b][1], float))))

    links = []                                      # (先连哪条: 位置接得上的按距离从近到远, 然后才是"信跟踪的", d, i, j)
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            same_id = segs[i][0] == segs[j][0]
            if same_id and j != i + 1:
                continue                            # 同一个编号只连相邻的两段
            if not same_id and set(segs[i][1]) & set(segs[j][1]):
                continue                            # 同时在场 = 两个人
            d = dist(i, j)
            if d <= SAME_M:
                links.append((0, d, i, j))
            elif same_id and min(len(segs[i][1]), len(segs[j][1])) >= MIN_FRAMES:
                links.append((1, d, i, j))
    group = list(range(len(segs)))                  # 每段属于哪一组(组 = 一个人)
    frames = [set(s[1]) for s in segs]              # 每组占了哪些帧
    for _, d, i, j in sorted(links):
        gi, gj = group[i], group[j]
        if gi == gj or frames[gi] & frames[gj]:
            continue                                # 已经是一个人了; 或者两组同时在场过, 不能连
        frames[gi] |= frames[gj]
        group = [gi if g == gj else g for g in group]

    people, log, used = {}, [], set()
    members = {g: [k for k in range(len(segs)) if group[k] == g] for g in sorted(set(group))}
    for g, ks in sorted(members.items(), key=lambda x: -len(frames[x[0]])):          # 帧多的先挑编号
        n = {}
        for k in ks:
            n[segs[k][0]] = n.get(segs[k][0], 0) + len(segs[k][1])
        if len(frames[g]) < MIN_FRAMES:
            for k in ks:
                log.append(dict(事="不算人", 编号=segs[k][0], 帧=list(segs[k][1]),
                                原因=f"核对完这个人只有 {len(frames[g])} 帧(不到 {MIN_FRAMES} 帧), 也并不到别人身上"))
            continue
        pid = max(n, key=lambda r: (n[r], -int(r)))                                  # 用出帧最多的那个编号当这个人的号
        if pid in used:                                                              # 同一个编号被拆成了两个人(极少): 另起一个号
            pid = str(max(int(r) for r in list(tracks) + list(used)) + 1)
        used.add(pid)
        people[pid] = sorted(((segs[k][0], b) for k in ks for b in segs[k][1]), key=lambda x: x[1])
        for k in ks:
            if segs[k][0] != pid:
                log.append(dict(事="并进别人", 编号=segs[k][0], 帧=list(segs[k][1]), 并进=pid,
                                原因=f"和人 {pid} 从不同时在场, 位置接得上(不到 {SAME_M} 米)"))
    return dict(sorted(people.items(), key=lambda x: int(x[0]))), sorted(log, key=lambda x: (int(x["编号"]), x["帧"][0]))


def _frame_counts(run):
    root = f"{run}/mesh_4d_individual"
    return {p: len([f for f in os.listdir(f"{root}/{p}") if f.endswith(".ply")]) for p in sorted(os.listdir(root), key=int)
            if os.path.isdir(f"{root}/{p}")}


def raw_run(body):
    """出网格那一轮的目录(跟踪原来的编号)。开了补全的优先。"""
    for d in ("04_mesh_occ1", "04_mesh_occ0"):
        if os.path.isdir(f"{body}/{d}/mesh_4d_individual"):
            return f"{body}/{d}"
    raise SystemExit(f"[失败] {body} 下没有 mesh_4d_individual, 先跑 body4d_pipeline 的 mesh 那步(people_mesh)")


def mesh_run_of(body, resolved=True):
    """人体数据在哪: 核对过身份就是 06_identity/, 没核对过(老轮次)就是出网格的目录。resolved=False 强制要原始的。
    核对之后人体网格又重出过 -> 报错, 不拿过期的身份往下走。"""
    raw = raw_run(body)
    meta = f"{body}/{DIR}/identity.json"
    if not resolved or not os.path.exists(meta):
        return raw
    m = json.load(open(meta))
    if m["来源"] != os.path.basename(raw) or m["来源帧数"] != _frame_counts(raw):
        raise SystemExit(f"[失败] {body}/{DIR} 是按旧的人体网格核对的(网格重出过) —— 重跑 people(摆人)")
    return f"{body}/{DIR}"


def write(body, run, people, log):
    """把核对结果落成 <body>/06_identity/: 每人每帧的网格 / 相机 / 一手参数链接到原文件, 抠图按新编号重画。返回目录。"""
    from PIL import Image
    dst, tmp = f"{body}/{DIR}", f"{body}/{DIR}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    for sub, ext in PER_PERSON.items():
        if not os.path.isdir(f"{run}/{sub}"):
            continue                                                   # 老的出网格目录没有一手参数
        for pid, items in people.items():
            os.makedirs(f"{tmp}/{sub}/{pid}")
            for rid, b in items:
                src = os.path.abspath(f"{run}/{sub}/{rid}/{b:08d}{ext}")
                if os.path.exists(src):
                    os.symlink(src, f"{tmp}/{sub}/{pid}/{b:08d}{ext}")
    new_id = {}                                                        # 帧号 -> {原编号: 新编号}
    for pid, items in people.items():
        for rid, b in items:
            new_id.setdefault(b, {})[int(rid)] = int(pid)
    os.makedirs(f"{tmp}/masks")
    for f in sorted(os.listdir(f"{run}/masks")):
        im = Image.open(f"{run}/masks/{f}"); m = np.array(im); out = np.zeros_like(m)
        for rid, pid in new_id.get(int(os.path.splitext(f)[0]), {}).items():
            out[m == rid] = pid
        o = Image.fromarray(out).convert("P")
        if im.getpalette():
            o.putpalette(im.getpalette())
        o.save(f"{tmp}/masks/{f}")
    by = {pid: {} for pid in people}
    for pid, items in people.items():
        for rid, b in items:
            by[pid].setdefault(rid, []).append(b)
    json.dump(dict(来源=os.path.basename(run), 来源帧数=_frame_counts(run), 人=by, 记录=log,
                   规则=dict(中断多久算两段_秒=GAP_S, 同一个位置_米=SAME_M, 至少几帧才算人=MIN_FRAMES)),
              open(f"{tmp}/identity.json", "w"), ensure_ascii=False, indent=1)
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)
    return dst
