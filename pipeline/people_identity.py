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
摆人那步(body_motion.py)用 mesh_run_of() 找人体数据, 读到的是核对过的人。

★ 人号 = 跟踪编号(2026-10-09)。编号本来有三套: 跟踪编号(挑人卡片、03_check.jpg 上的数字) -> 出网格时重排成连续的 1..N
  (<出网格目录>/relabel.json) -> 核对后的人号。三个场景里它们碰巧一样(没人被点掉过), 所以一直没出事; 但在闸口点掉
  一个人, 后面的人全部往前挪一号: FBX 里的 P5 和上一版视频里的 5 号不是同一个人, 网页上的截图也对到别人头上。
  现在核对完的人直接用他的跟踪编号命名(track_ids), 从这里往后(10_people.json、验收视频、FBX 里的名字)只有这一套号。
★ 2026-10-12 出网格目录也直接用跟踪编号命名了(mesh_store.py), 中间那套重排的号没有了; 10-12 之前出的目录还是老排法,
  所以 track_ids 照旧读 relabel.json(新目录里它两边一样)。
"""
import json, os, shutil
import numpy as np
import mesh_store

GAP_S = 1.0          # 秒: 一个编号中断这么久以上, 前后算两段
JUMP_M = 1.5         # 米: 同一个编号前后两帧水平位置差这么多以上,
JUMP_V = 8.0         #     而且比这个速度(米/秒, 冲刺)跑得还远 -> 前后不是同一个人, 也切开(只能靠位置接得上再并回去)
#   中断超过 GAP_S 的两段本来"信跟踪的"(人走开又回来); 但中断期间这个编号挂过碎片(跟踪跟丢了在乱认)就不信 ——
#   goodfellas_tray 人 4: 侍者出画 -> 编号挂在人群的碎片上 -> 挂到同样穿白衣服的厨师身上, 中间隔了 1.3 秒
# ★ 2026-11-06 goodfellas_tray 人 4: 前景侍者出画 0.6 秒后, 编号挂到了人群里别人身上, 又挂到端托盘的厨师身上
#   (一帧跳 3.4 m); 中断不到 GAP_S, 三个人被当成一个人。正常的人(office / couplelift / coupledance / stepup 骑车的)
#   前后两帧最大跳 1.13 m; 串人的 3.3~53 m。
# ★ 2026-11-11 jw4 人 1: 位置本身不可信的时候不按瞬移切 —— 坏相机的帧(bad_frames.json, 相机位姿是错的, 世界位置跟着错)
#   和隔着切点的两帧(两个机位, DA3 两个镜头之间本来就对不齐; jw4 切点两边同一个人差 2~3 m)。这时只信跟踪的编号。
#   jw4 甄子丹坐在地上的特写, 第 62~71 帧相机坏, 位置一帧跳 4.1 m, 被切成两个人(验收视频 10 秒黄变粉)。
GLITCH_N = 2         # ★ 2026-11-11 帧: 切点处挂错的一两帧(一头切点、一头同镜头里瞬移), 当成跟丢 —— 见 resolve 的 A
CUT_TRUST_S = 1.0    # ★ 2026-11-11 秒: 编号跨过切点, 切点后那一段不到这么久就不信跟踪的(jw4 11.57 秒挂到前景别人身上 0.83 秒) —— B
SAME_M = 0.5         # 米: 两段相距不到这么多 = 站在同一个地方
MIN_FRAMES = 3       # 核对完, 出了网格的帧不到这么多的不算人
MIN_PERSON_S = 0.25  # ★ 2026-11-13 秒: 还要在场(帧数 x 帧间隔)这么久才算人 —— 人体全帧(30 fps)时 3 帧只有 0.1 秒,
                     #   jw4 r4_30fps 人 8 = 一闪而过的模糊人影 + 一根棍子(两个编号的碎片拼的), 片子中间闪一下。
                     #   片头 / 片尾连着露的除外(nfl1 人 8 片尾才进画面, 24 fps 3 帧 = 0.125 秒, 是真人)。
                     #   场景那批帧(6~10 fps)里 3 帧 >= 0.3 秒, 老轮次结果不变。
DUP_M = 0.3          # ★ 2026-11-13 米: 编号跳过来的段, 和一个已经确认的人同一帧骨盆相距不到这么多 = 站在同一个位置
DUP_FRAC = 0.5       #   这样的帧占这段(位置可信的帧)这么多以上 -> 编号跳到了那个人身上(一个人挂两个编号), 跳过来的不算人。
                     #   jw4 r4_30fps 18.8~19.5 秒长发打手身上叠了两个人(人 3 + 跳过来的人 7)。只看"跳过来切出的段":
                     #   真人从头就有自己的编号(jw1 两个人抱在一起扭打, 骨盆常只差 0.25 m)不受影响。
SPLIT_MIN_S = 0.5    # 秒: 全是瞬移 / 跟丢之后切出来的段组成的人, 露的时间(帧数 x 帧间隔)不到这么久不算人 ——
#   切出来的短段多半是编号挂到了别的东西上(stepup3d_scooter 画面边上 3 帧的一小块; goodfellas_tray 厨师手里的托盘 11 帧)
DIR = "06_identity"
PER_PERSON = {"mesh_4d_individual": ".ply", "focal_4d_individual": ".json", "mhr_4d_individual": ".npz"}


def resolve(tracks, up=None, name=None, taken=(), lost=None, unreliable=(), shot_of=None, max_people=0, check_depth_jump=None):
    """tracks: {编号: {帧号: (秒, 世界坐标 (3,))}} —— 只放出了网格的帧。up: 给了就只比水平方向的距离。
    name: {tracks 里的编号: 跟踪编号}(见 track_ids), 给了人号和记录里的编号就用跟踪编号; taken: 已经被占的跟踪编号
    (包括闸口上点掉的), 另起新号时避开。lost: {编号: 跟丢的帧}(挂在碎片上的帧, 已经从 tracks 里去掉了, 见
    body_into_scene.fragment_frames): 一个编号中断的那段里挂过碎片, 说明跟踪已经跟丢、在乱认, 中断后面那段不再"信跟踪的"。
    unreliable: 位置不可信的帧号(坏相机); shot_of: 帧号 -> 镜头号。前后两帧有一帧位置不可信、或者隔着切点, 不按瞬移切;
    算两段相距多远时也先用位置可信的帧。
    max_people: 用户给的人数上限(0 = 不知道, 不管)。见 cap_people。
    返回 (people, log): people = {人: [(原编号, 帧号), ...] 按帧号排}(原编号还是 tracks 里的, 找文件用);
    log = [dict(事, 编号, 帧, ...)] 给人看的记录。"""
    name = {str(r): str((name or {}).get(str(r), r)) for r in tracks}
    # 帧间隔(人体帧的采样间隔, 秒) 和片子首尾: "在场多久"按秒算(见 MIN_PERSON_S)
    _tb = {b: float(v[0]) for r_ in tracks.values() for b, v in r_.items()}
    _d = [d for r_ in tracks.values() for d in np.diff(sorted(float(v[0]) for v in r_.values())) if d > 0]
    DTV = float(np.median(_d)) if _d else 0.15
    T0, T1 = (min(_tb.values()), max(_tb.values())) if _tb else (0.0, 0.0)

    def too_short(bs):
        """一组人体帧号: 在场太短、不算人吗。不到 MIN_FRAMES 帧照旧不算; 帧数 x 帧间隔 不到 MIN_PERSON_S 也不算,
        除非是片头 / 片尾连着露的那一下(人刚进画面 / 正要出画)。"""
        ts = sorted(_tb[b] for b in bs)
        if len(ts) < MIN_FRAMES:
            return True
        if len(ts) * DTV >= MIN_PERSON_S - 1e-9:
            return False
        joined = all(d <= 1.5 * DTV for d in np.diff(ts))
        return not (joined and (ts[0] <= T0 + 1.5 * DTV or ts[-1] >= T1 - 1.5 * DTV))
    up = None if up is None else np.asarray(up, float) / np.linalg.norm(up)
    flat = (lambda p: p - (p @ up) * up) if up is not None else (lambda p: p)
    unreliable = set(unreliable or ())
    comparable = lambda a, b: (a not in unreliable and b not in unreliable
                               and (shot_of is None or shot_of(a) == shot_of(b)))
    def brk(rid, a, b):
        """同一个编号相邻两帧之间断不断: "gap"(中断超过 GAP_S) / "jump"(同镜头、位置可信时瞬移) / "cut"(隔着切点) / None"""
        dt = tracks[rid][b][0] - tracks[rid][a][0]
        # ★ 2026-11-11 正好隔 GAP_S 的两帧(jw1 编号 1 第 75 -> 80 帧, 453/30 -> 483/30 = 1.0000000000000018 秒)
        #   以前因为浮点误差被当成"中断超过 1 秒"切开, 后面那段又被别人的碎片挡住并不回来, 另起了人 10。留 1 微秒的余量
        if dt > GAP_S + 1e-6:
            return "gap"
        jump = float(np.linalg.norm(flat(np.asarray(tracks[rid][b][1], float) - np.asarray(tracks[rid][a][1], float))))
        if jump > JUMP_M and jump > JUMP_V * dt and comparable(a, b):
            if check_depth_jump is not None and check_depth_jump(rid, a, b):
                return None
            return "jump"
        if shot_of is not None and shot_of(a) != shot_of(b):
            return "cut"
        return None

    segs, jump_cut, glitch_log = [], set(), []      # jump_cut: 第 k 段和前一段之间的跟踪编号不可信(瞬移 / 跟丢 / 切点后太短)
    for rid in sorted(tracks, key=int):
        bis = sorted(tracks[rid])
        # A. 切点处挂错的一两帧: 前后两头都断, 一头是切点、一头是同镜头里的瞬移 -> 当成跟丢, 从这个编号里拿掉
        if shot_of is not None and len(bis) > GLITCH_N:
            ks = [brk(rid, a, b) for a, b in zip(bis, bis[1:])]
            st = [0] + [i + 1 for i, k in enumerate(ks) if k]
            en = st[1:] + [len(bis)]
            drop = []
            for s0, e0 in zip(st, en):
                kin = ks[s0 - 1] if s0 > 0 else None
                kout = ks[e0 - 1] if e0 < len(bis) else None
                if e0 - s0 <= GLITCH_N and {kin, kout} == {"cut", "jump"}:
                    drop += bis[s0:e0]
            if drop:
                glitch_log.append(dict(事="不算人", 编号=name[rid], 帧=sorted(drop),
                                       原因=f"切点处挂错的 {len(drop)} 帧: 编号跨过切点后先挂到别处, 同一镜头里又瞬移回来"))
                bis = [b for b in bis if b not in set(drop)]
        if not bis:
            continue
        dts = np.diff([tracks[rid][b][0] for b in bis]); step = float(np.median(dts)) if len(dts) else 0.0
        ks = [brk(rid, a, b) for a, b in zip(bis, bis[1:])]
        cur = [bis[0]]
        for i, (a, b) in enumerate(zip(bis, bis[1:])):
            k = ks[i]
            short_after_cut = False
            if k == "cut":                           # B. 跨切点: 切点后到下一次断开不到 CUT_TRUST_S 秒 -> 不信跟踪的
                j = i + 1
                while j < len(ks) and ks[j] is None:
                    j += 1
                short_after_cut = tracks[rid][bis[j]][0] - tracks[rid][b][0] + step < CUT_TRUST_S
            if k in ("gap", "jump") or short_after_cut:
                segs.append((rid, cur)); cur = []
                if k != "gap" or any(a < x < b for x in (lost or {}).get(rid, ())):
                    jump_cut.add(len(segs))         # 瞬移切开的 / 中断期间跟丢过的 / 切点后太短的: 后面那段不信跟踪的
            cur.append(b)
        segs.append((rid, cur))

    def dist(i, j):
        """两段在时间上最靠近的那一对帧, 位置差多远。"""
        (ri, bi), (rj, bj) = segs[i], segs[j]
        gi = [x for x in bi if x not in unreliable] or bi          # 先用位置可信的帧
        gj = [y for y in bj if y not in unreliable] or bj
        a, b = min(((x, y) for x in gi for y in gj), key=lambda xy: abs(tracks[ri][xy[0]][0] - tracks[rj][xy[1]][0]))
        return float(np.linalg.norm(flat(np.asarray(tracks[ri][a][1], float) - np.asarray(tracks[rj][b][1], float))))

    links = []                                      # (先连哪条: 位置接得上的按距离从近到远, 然后才是"信跟踪的", d, i, j)
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            same_id = segs[i][0] == segs[j][0]
            # 同一个编号: 相邻两段照旧; 隔着几段的只能靠位置接得上(瞬移切出去的一小段中间夹着, 前后还是他)
            if not same_id and set(segs[i][1]) & set(segs[j][1]):
                continue                            # 同时在场 = 两个人
            d = dist(i, j)
            if d <= SAME_M:
                links.append((0, d, i, j))
            elif (same_id and j == i + 1 and j not in jump_cut and min(len(segs[i][1]), len(segs[j][1])) >= MIN_FRAMES
                  and min(len(segs[i][1]), len(segs[j][1])) * DTV >= MIN_PERSON_S - 1e-9):
                links.append((1, d, i, j))          # 信跟踪的(人走开又回来); 位置瞬移切开的不信
    group = list(range(len(segs)))                  # 每段属于哪一组(组 = 一个人)
    frames = [set(s[1]) for s in segs]              # 每组占了哪些帧
    for _, d, i, j in sorted(links):
        gi, gj = group[i], group[j]
        if gi == gj or frames[gi] & frames[gj]:
            continue                                # 已经是一个人了; 或者两组同时在场过, 不能连
        frames[gi] |= frames[gj]
        group = [gi if g == gj else g for g in group]

    people, log, used = {}, [], set()
    ppos = {}                                       # 已经确认的人: {人: {帧: 水平位置}}(查"跳过来的段是不是叠在他身上")
    members = {g: [k for k in range(len(segs)) if group[k] == g] for g in sorted(set(group))}
    for g, ks in sorted(members.items(), key=lambda x: -len(frames[x[0]])):          # 帧多的先挑编号
        n = {}
        for k in ks:
            n[segs[k][0]] = n.get(segs[k][0], 0) + len(segs[k][1])
        if too_short(frames[g]):
            for k in ks:
                log.append(dict(事="不算人", 编号=name[segs[k][0]], 帧=list(segs[k][1]),
                                原因=(f"核对完这个人只有 {len(frames[g])} 帧(不到 {MIN_FRAMES} 帧), 也并不到别人身上"
                                      if len(frames[g]) < MIN_FRAMES else
                                      f"核对完这个人只在场 {len(frames[g])} 帧 ≈ {len(frames[g]) * DTV:.2f} 秒(不到 {MIN_PERSON_S} 秒,"
                                      f" 又不是片头片尾刚进 / 出画), 也并不到别人身上: 一闪而过的多半不是人")))
            continue
        if all(k in jump_cut for k in ks):
            ts = sorted(tracks[segs[k][0]][b][0] for k in ks for b in segs[k][1])
            shown = len(ts) * (float(np.median(np.diff(ts))) if len(ts) > 1 else 0.0)
            if shown < SPLIT_MIN_S:
                for k in ks:
                    log.append(dict(事="不算人", 编号=name[segs[k][0]], 帧=list(segs[k][1]),
                                    原因=f"编号瞬移 / 跟丢之后切出来的一小段, 只露了 {shown:.2f} 秒(不到 {SPLIT_MIN_S} 秒), 多半是别的东西"))
                continue
        if any(k in jump_cut for k in ks) and ppos:
            # ★ 重复人(见 DUP_M): 有跳过来的段的这一组, 位置可信的帧里有多少和一个已经确认的人站在同一个位置
            hit, tot = {}, 0
            for k in ks:
                rid = segs[k][0]
                for b in segs[k][1]:
                    if b in unreliable:
                        continue
                    tot += 1
                    pb = flat(np.asarray(tracks[rid][b][1], float))
                    for q, pp in ppos.items():
                        if b in pp and float(np.linalg.norm(pb - pp[b])) < DUP_M:
                            hit[q] = hit.get(q, 0) + 1
                            break
            if tot and hit and sum(hit.values()) / tot >= DUP_FRAC:
                host = max(hit, key=hit.get)
                for k in ks:
                    log.append(dict(事="不算人", 编号=name[segs[k][0]], 帧=list(segs[k][1]),
                                    原因=f"编号 {name[segs[k][0]]} 跳到了人 {host} 身上: 这段 {sum(hit.values())}/{tot} 帧和他同时在同一个位置"
                                         f"(不到 {DUP_M} 米) —— 一个人挂了两个编号, 跳过来的这段不算人"))
                continue
        pid = name[max(n, key=lambda r: (n[r], -int(r)))]                            # 用出帧最多的那个编号当这个人的号
        fresh = pid in used
        if fresh:                                                                    # 同一个编号被拆成了两个人: 另起一个号
            pid = str(max(int(r) for r in list(name.values()) + list(used) + [str(x) for x in taken]) + 1)
        used.add(pid)
        people[pid] = sorted(((segs[k][0], b) for k in ks for b in segs[k][1]), key=lambda x: x[1])
        ppos[pid] = {b: flat(np.asarray(tracks[r][b][1], float)) for r, b in people[pid]}
        for k in ks:
            if name[segs[k][0]] != pid:
                why = (f"编号 {name[segs[k][0]]} 中途换了人(位置瞬移 / 跟丢后挂过碎片), 这一段另起人号 {pid}"
                       if fresh and len({segs[x][0] for x in ks}) == 1 else f"和人 {pid} 从不同时在场, 位置接得上(不到 {SAME_M} 米)")
                log.append(dict(事="并进别人", 编号=name[segs[k][0]], 帧=list(segs[k][1]), 并进=pid, 原因=why))
    log += glitch_log
    if max_people:
        people, log = cap_people(people, log, int(max_people), {r: sorted(v) for r, v in tracks.items()}, name)
    return dict(sorted(people.items(), key=lambda x: int(x[0]))), sorted(log, key=lambda x: (int(x["编号"]), x["帧"][0]))


def cap_people(people, log, k, track_frames, name, win=5, dup_frac=0.5):
    """人数上限 k(用户说这个片子一共几个人, 配置 PEOPLE_N)。2026-11-11 jw1 起头。
    1) 同一帧里多于 k 个人: 这一帧里"最不像在跟这个人"的检测不要(重复框 / 碎片 / 挂错的)。
       像不像 = 这个检测所在编号在前后 win 帧里出现了几帧(断断续续的碎片少、一直跟着的多), 一样多看编号总帧数。
    2) 还多于 k 个人: 挑重叠最少(同时在场的帧占小的那个人的比例)、时间上挨得最近的两个人并起来;
       重叠的帧上两个检测留更像的那个。重叠超过小的那个人 dup_frac 的不并 —— 那是一直同时在场的重复框, 小的整个不要。
       并成的人用帧多的那个的人号。"""
    tf = {str(r): set(v) for r, v in track_frames.items()}
    score = lambda rid, b: (sum(1 for x in range(b - win, b + win + 1) if x in tf.get(str(rid), ())), len(tf.get(str(rid), ())))
    nm = lambda rid: name.get(str(rid), str(rid))
    P = {pid: {b: rid for rid, b in items} for pid, items in people.items()}
    for b in sorted({b for v in P.values() for b in v}):
        here = [pid for pid in P if b in P[pid]]
        if len(here) > k:
            for pid in sorted(here, key=lambda q: score(P[q][b], b))[:len(here) - k]:
                log.append(dict(事="不算人", 编号=nm(P[pid][b]), 帧=[int(b)],
                                原因=f"人数上限 {k}: 这一帧有 {len(here)} 个人, 这个检测最不像在跟人(前后几帧断断续续), 多半是重复框 / 碎片"))
                del P[pid][b]
    P = {pid: v for pid, v in P.items() if v}
    while len(P) > k:
        best = None
        ids = list(P)
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                p, q = ids[x], ids[y]
                ov = len(set(P[p]) & set(P[q])) / min(len(P[p]), len(P[q]))
                gap = min(abs(b - c) for b in (min(P[p]), max(P[p])) for c in (min(P[q]), max(P[q])))
                key = (round(ov, 3), gap, -(len(P[p]) + len(P[q])))
                if best is None or key < best[0]:
                    best = (key, p, q)
        (ov, _, _), p, q = best
        keep, gone = (p, q) if len(P[p]) >= len(P[q]) else (q, p)
        if ov > dup_frac:                                   # 一直和别人同时在场: 重复框, 整个不要
            for r in sorted({str(r) for r in P[gone].values()}):
                log.append(dict(事="不算人", 编号=nm(r), 帧=sorted(int(b) for b, rr in P[gone].items() if str(rr) == r),
                                原因=f"人数上限 {k}: 和别人一直同时在场(重叠 {ov:.0%}), 是重复框 / 碎片"))
            P.pop(gone); continue
        for b in sorted(set(P[keep]) & set(P[gone])):       # 重叠的帧: 留更像的
            loser = gone if score(P[keep][b], b) >= score(P[gone][b], b) else keep
            log.append(dict(事="不算人", 编号=nm(P[loser][b]), 帧=[int(b)],
                            原因=f"人数上限 {k}: 并人时这一帧两个检测, 留下更像的那个"))
            if loser == keep:
                P[keep][b] = P[gone][b]
            del P[gone][b]
        for r in sorted({str(r) for r in P[gone].values()}):
            log.append(dict(事="并进别人", 编号=nm(r), 帧=sorted(int(b) for b, rr in P[gone].items() if str(rr) == r),
                            并进=keep, 原因=f"人数上限 {k}: 和人 {keep} 并成一个人(同时在场 {ov:.0%}, 重叠的帧留更像的检测)"))
        P[keep].update(P.pop(gone))
    return {pid: sorted(((rid, b) for b, rid in v.items()), key=lambda x: x[1]) for pid, v in P.items()}, log


def _frame_counts(run):
    root = f"{run}/mesh_4d_individual"
    return {p: len([f for f in os.listdir(f"{root}/{p}") if f.endswith(".ply")]) for p in sorted(os.listdir(root), key=int)
            if os.path.isdir(f"{root}/{p}")}


def raw_run(body):
    """出网格那一轮的目录(跟踪原来的编号)。开了补全的优先。"""
    for d in ("04_mesh_occ1", "04_mesh_occ0"):
        # ★ 出网格那步改库改到一半失败了(有的人删了、有的还没算): 不能拿它往下走。见 mesh_store.begin / finish
        if mesh_store.finished(f"{body}/{d}") is False:
            raise SystemExit(f"[失败] {body}/{d} 出网格没跑完(上一次改到一半失败了) —— 重跑 people_mesh(出人体网格)")
        if os.path.isdir(f"{body}/{d}/mesh_4d_individual"):
            return f"{body}/{d}"
    raise SystemExit(f"[失败] {body} 下没有 mesh_4d_individual, 先跑 body4d_pipeline 的 mesh 那步(people_mesh)")


def track_ids(run):
    """{出网格目录里的编号: 跟踪编号}。10-12 起目录名就是跟踪编号, 对照表两边一样; 之前出的目录是把挑中的人重排成
    连续的 1..N, 对照写在 relabel.json(左 = 跟踪编号, 右 = 重排后的)。更老的目录没有这个文件 = 没重排过, 原样。"""
    p = f"{run}/relabel.json"
    if os.path.exists(p):
        return {str(v): int(k) for k, v in json.load(open(p))["relabel"].items()}
    return {r: int(r) for r in _frame_counts(run)}


def all_track_ids(body, run):
    """跟踪给过的所有编号(连闸口上点掉的): 另起新人号时不能撞上它们。"""
    ids = set(track_ids(run).values())
    p = f"{body}/02_body4d/picked.json"
    if os.path.exists(p):
        d = json.load(open(p))
        ids |= {int(x) for x in d.get("keep_ids", [])} | {int(x) for x in d.get("dropped", {})}
    return sorted(ids)


def source_of(body):
    """这份人体网格的"指纹": 哪个目录、每个编号多少帧、编号对的是跟踪的谁。核对身份和摆人的结果都记着它,
    读的时候再算一遍对一下 —— 人体网格重出过(或者挑的人变了)就对不上。
    ★ 只比帧数不够: 点掉甲、留下乙, 两人帧数一样的话帧数表一模一样, 但 1 号已经换了人; 所以带上编号对照。"""
    return _source(raw_run(body))


def _source(run):
    """★ 算于(2026-10-12): 每个人的网格是哪一次算的。一个人重算过(跟踪变了 / 代码变了), 帧数和编号都可能不变,
      但网格已经不是那一份 —— 模型在个别帧上每次算都不一样。10-12 之前出的目录没有登记簿, 就没有这一项。"""
    src = dict(来源=os.path.basename(run), 来源帧数=_frame_counts(run), 编号对照=track_ids(run))
    who = mesh_store.people(run)
    if who:
        src["算于"] = {str(o): e.get("算于") for o, e in sorted(who.items())}
    return src


def mesh_run_of(body, resolved=True):
    """人体数据在哪: 核对过身份就是 06_identity/, 没核对过(老轮次)就是出网格的目录。resolved=False 强制要原始的。
    核对之后人体网格又重出过 -> 报错, 不拿过期的身份往下走。"""
    raw = raw_run(body)
    meta = f"{body}/{DIR}/identity.json"
    if not resolved or not os.path.exists(meta):
        return raw
    m, src = json.load(open(meta)), source_of(body)
    if any(m.get(k, v) != v for k, v in src.items()):                 # 10-09 之前写的没有编号对照那一项: 只比有的
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
    json.dump(dict(**_source(run), 人=by, 记录=log,
                   规则=dict(中断多久算两段_秒=GAP_S, 瞬移多远算两段_米=JUMP_M, 瞬移速度_米每秒=JUMP_V,
                           同一个位置_米=SAME_M, 至少几帧才算人=MIN_FRAMES, 至少在场几秒才算人=MIN_PERSON_S, 跳过来叠在别人身上_米=DUP_M, 切出来的段至少露几秒=SPLIT_MIN_S)),
              open(f"{tmp}/identity.json", "w"), ensure_ascii=False, indent=1)
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)
    return dst
