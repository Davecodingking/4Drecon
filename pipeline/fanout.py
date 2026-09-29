#!/usr/bin/env python
"""步骤内部拆分(2026-09-29): 同一个脚本同时起 K 份, 每份算一部分, 各用一张卡。

★ 为什么: 最慢的两步内部是一个一个算的 —— 出人网格 8 个人算 11 分钟, track 9 个词算 4 分钟,
  各占一张卡, 另外几张闲着。每个人、每个词的结果本来就是分开存的, 拆开跑不用动模型代码。
★ 拆几份每次现算: 要算的个数、现在的空卡数、本轮剩余名额, 三个里最小的。只有 1 就和以前一样。
★ 拆分的步骤自己不领卡, 由它起的每一份各领一张 —— 否则这一步多占一张不用。

只用标准库。
"""
import json, os, threading

from _common import GpuSlot, gpus_free_now, sh, envcmd, parallel_on


def on():
    """拆分默认关, FANOUT=1 才开。
    ★ 2026-09-29 nfl1 实测拆分会改变结果, 所以不能默认开:
      - track 按词拆: gloves 10 -> 11 个实例, referee 1 -> 2(不拆时两次重跑完全一样)。它是"一批词共用一次检测",
        批里有哪些词会影响结果。
      - 出人网格按人拆: 除了编号 1 的人, 其余的人顶点系统性地差 1~3 厘米(两次重跑只差 1~4 毫米);
        有"模型拒绝出网格"的帧的人, 单独一份时从那一帧起全是空网格(合并时会拦下, 见 mesh_merge.py)。"""
    return os.environ.get("FANOUT", "0") == "1"


def split(items, weights, k):
    """分成最多 k 组, 让各组的权重和接近: 从重到轻, 每次放进当前最轻的那组。组内保持原来的顺序。"""
    if not items:
        return []
    k = max(1, min(k, len(items)))
    groups, load = [[] for _ in range(k)], [0.0] * k
    for i in sorted(range(len(items)), key=lambda i: (-weights[i], i)):
        j = load.index(min(load))
        groups[j].append(i)
        load[j] += weights[i]
    return [[items[i] for i in sorted(g)] for g in groups if g]


def how_many(n_items, need_mb=0, **kw):
    return max(1, min(n_items, gpus_free_now(need_mb, **kw)))


def run_workers(jobs, what, need_mb=0, retries=1, **slot_kw):
    """jobs: [(名字, fn)], fn(卡号) 跑这一份, 失败就抛异常(sh() 失败抛的是 SystemExit)。
    每份一个线程, 各领一张卡。失败的那份换一张卡再试 retries 次; 其他份不受影响, 照常跑完。
    返回 {名字: None = 成功 / 错误文字}。"""
    out = {}

    def one(name, fn):
        avoid, err = [], None
        for _ in range(retries + 1):
            card = None
            try:
                with GpuSlot(f"{what}/{name}", need_mb, set_env=False, avoid=avoid, **slot_kw) as s:
                    card = s.card
                    fn(card)
                out[name] = None
                return
            except (Exception, SystemExit) as e:
                err = f"{type(e).__name__}: {e}"
                print(f"    [拆分] {what}/{name} 在 GPU {card} 上失败: {err}", flush=True)
                if card is not None:
                    avoid.append(card)
        out[name] = err

    ts = [threading.Thread(target=one, args=j) for j in jobs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def track_words(R1, extra=""):
    """这一轮要跟踪、还没有结果的词: 词表 + 附加词, 去掉已经有 npz 的。和 track_all.py 同一个规则。"""
    vocab = list(json.load(open(f"{R1}/01_vlm_vocab.json"))["vocab"])
    vocab += [w for w in (x.strip() for x in extra.split(",")) if w and w not in vocab]
    return [w for w in vocab if not os.path.exists(f"{R1}/03_masks/{w.replace(' ', '_')}.npz")]


def merge_reports(parts, out):
    """几份 json(都是 {词: ...})并成一份, 并完删掉各份。某一份没跑出来(文件不在)就跳过。"""
    d = {}
    for p in parts:
        if os.path.exists(p):
            d.update(json.load(open(p)))
            os.remove(p)
    json.dump(d, open(out, "w"), ensure_ascii=False, indent=1)
    return d


def fan_track(py, script, env, R1, log, dry, need_mb=0, max_parts=None, sh_env=None, **kw):
    """物体跟踪按词拆分。env: 给 track_all.py 的环境变量(FR / R1 / EXTRA_WORDS ...)。max_parts: 最多拆几份。
    sh_env: 每条命令额外带的环境变量(室内要带 CUDA_HOME 那一套, 见 indoor_pipeline.sh)。
    每个词的结果是单独一个 npz, 不用合并; 报告各份写各的, 这里并成一份。
    ★ 重试只补没出来的词: track_all.py 见到 npz 已经在就跳过。"""
    probe = {k: v for k, v in kw.items() if k in ("run_dir", "pool", "busy")}
    words = [] if dry else track_words(R1, env.get("EXTRA_WORDS", ""))
    k = how_many(len(words), need_mb, **probe) if (words and parallel_on() and on()) else 1
    k = min(k, max_parts) if max_parts else k
    if dry or k <= 1:
        if dry:
            return sh(envcmd(py, script, env), log, dry, env=sh_env)
        with GpuSlot("track", need_mb, **kw):
            return sh(envcmd(py, script, env), log, dry, env=sh_env)
    groups = split(words, [1] * len(words), k)
    print(f"    [拆分] {len(words)} 个词分成 {len(groups)} 份: " + " | ".join(", ".join(g) for g in groups), flush=True)
    parts = [f"{R1}/03_track_report.part{i}.json" for i in range(len(groups))]

    def job(i):
        return lambda card: sh(envcmd(py, script, dict(env, CUDA_VISIBLE_DEVICES=card, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                                                       ONLY_WORDS=",".join(groups[i]), TRACK_REPORT=parts[i])),
                               log, dry, env=sh_env)

    res = run_workers([(f"w{i}", job(i)) for i in range(len(groups))], "track", need_mb, **kw)
    merge_reports(parts, f"{R1}/03_track_report.json")
    bad = {n: e for n, e in res.items() if e}
    if bad:
        raise SystemExit(f"[失败] track 有 {len(bad)} 份重试后还是没跑成({'; '.join(bad.values())})。"
                         f"没出结果的词: {track_words(R1, env.get('EXTRA_WORDS', ''))}")
