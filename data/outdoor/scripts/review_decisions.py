#!/usr/bin/env python
"""人工闸决策的唯一一份实现 —— 室内室外共用。

这是**人的决定进入流水线的唯一入口**。

★ 2026-09-09 之前这一环整个是断的: 人工闸(review_gate.py)把结论写进
  05_review_applied.json, 而全代码库没有任何脚本读它 —— 人只能照着闸打印的
  EXCLUDE=/SOLO= 手敲回驱动源码。于是"人工干预"实际是"人工改代码",
  而路线图的目标(全自动, 只留 override 一处人工)在结构上就到不了。

  第一版只补了室外。复核发现室内的 resolve_cfg 连 R1 都不接, 也没有 override 这一步,
  于是 drive.py --indoor 的两道闸从头到尾是摆设, 而且 apply_decisions 还会打印
  "[落地决策]" 让人以为生效了 —— 比不做更糟。所以抽到这里, 两条链共用一份。

决策文件的形状(review_gate.py --apply 写的), 六个字段:
  排除词    [词...]                整个词作废        -> EXCLUDE
  逐实例词  [词...]                这个词逐实例生成  -> SOLO
  共用词    [词...]                这个词共用一个mesh-> SOLO_KEEP
  不生成    [生成名...]            单个候选不要      -> filter_dirs / 组装也挡
  指定样本  {"词"或"词#实例": [实例, 帧]}  换一张抠图 -> s_override / gen_per_instance
  改类别    {"词#实例": 目标词}    实例挪到别的词    -> relabel_words.py
  VLM放行   [生成名...]            VLM 说"不是它", 人看过还是要 -> gen_per_instance / 组装

★ 人在闸口填的"分组复用"不是第七个字段 —— 它在 review_gate --apply 里就被展开成
  "改类别 + 共用词"两条已有机制了, 运行期没有第三套代码。
"""
import os, sys, json

APPLIED = "05_review_applied.json"
SKIP    = "04_picks/skip.json"


def load(R1):
    """读决策文件。坏了硬停 —— 它是人工闸的产物, 坏了等于人的决定全丢了, 不能静默跳过。"""
    p = f"{R1}/{APPLIED}"
    if not os.path.exists(p):
        return {}
    try:
        d = json.load(open(p))
    except Exception as e:
        raise SystemExit(f"[失败] 人工决策文件读不了: {p}\n       {e}\n"
                         f"       它是人工闸的产物, 坏了就等于人的决定全丢了, 不能静默跳过。")
    if not isinstance(d, dict):
        raise SystemExit(f"[失败] {p} 不是一个对象, 内容: {str(d)[:200]}")
    return d


def merge_words(cur, add):
    """把闸判的词并进配置里已有的。**只加不减** —— 人的决定是增量,
    不该把场景配置里本来就有的排除项抹掉。

    代价是"改主意撤不回": 人在闸口把一个词从 drop-word 改回 keep, 决策文件里那条没了,
    但 EXCLUDE 里原本就有的还在。这是有意的 —— 配置里写死的排除项往往有独立理由
    (比如 manhole cover 扁得量不出高度), 不该被闸口一次改主意抹掉。
    """
    s = {w.strip() for w in str(cur).split(",") if w.strip()}
    s |= {str(w).strip() for w in add if str(w).strip()}
    return ",".join(sorted(s))


def apply_to_cfg(cfg, R1, quiet=False):
    """把决策并进配置。原地改 cfg 并返回它。

    生效顺序由调用方保证: 默认 <- 场景覆盖 <- **这里** <- 环境变量。
    闸排在环境变量之前 —— 临时敲一个环境变量试值应该压得过闸的决定;
    但闸的决定要压得过写死在场景表里的默认, 否则人在闸口的判断永远轮不上。
    """
    d = load(R1)
    if not d:
        return cfg
    # ★ 提示走 stderr: stdout 是 --dump-cfg 给程序解析的通道, 混进日志会解析失败,
    #   而调用方有 try/except 兜底 —— 失败后只是少传配置, 悄悄退回成"闸看不到候选"。
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, file=sys.stderr))
    if d.get("排除词"):
        cfg["EXCLUDE"] = merge_words(cfg.get("EXCLUDE", ""), d["排除词"])
        say(f"[人工闸] 整词作废 {len(d['排除词'])} 个: {', '.join(sorted(d['排除词']))}")
    if d.get("逐实例词"):
        # 室外的键叫 SOLO, 室内也叫 SOLO(门槛那个才叫 TH), 所以这里统一按 SOLO 处理。
        cur = str(cfg.get("SOLO", "")).strip()
        if cur == "*":
            say(f"[人工闸] 逐实例 {len(d['逐实例词'])} 个词, 但这个场景 SOLO=* (本来就全逐实例), 不用加")
        else:
            cfg["SOLO"] = merge_words(cur, d["逐实例词"])
            say(f"[人工闸] 逐实例生成 {len(d['逐实例词'])} 个: {', '.join(sorted(d['逐实例词']))}")
    # 反向: 人想让某个词**不要**逐实例(共用一个 mesh), 走 SOLO_KEEP 白名单。
    # SOLO="*" 的场景只能靠这条把词捞回来, 否则没有通道。
    if d.get("共用词"):
        cfg["SOLO_KEEP"] = merge_words(cfg.get("SOLO_KEEP", ""), d["共用词"])
        say(f"[人工闸] 共用一个 mesh {len(d['共用词'])} 个: {', '.join(sorted(d['共用词']))}")
    return cfg


def _parse_ov(R1):
    """把"指定样本"读成 {键: (实例号, 帧号)}, 顺便校验形状。

    键有两种写法, 都要认:
      "car"        —— 类级: 换这个词**类 mesh** 用的那张抠图
      "car#7"      —— 实例级: 换这个词第 7 个实例**专属 mesh** 用的那张抠图
    ★ 为什么要有实例级: beach/carpark 都配 SOLO="*", 联络图上 81/83 行都是逐实例候选,
      而老格式只有"词"一个键 —— 对这些行填"改用"一律静默丢弃, 同一个词填两次还会互相覆盖。
      而"换抠图"恰好是路线图里唯一要永久保留的那处人工, 它不能是个摆设。
    """
    out = {}
    for k, v in (load(R1).get("指定样本") or {}).items():
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise SystemExit(f"[失败] 决策文件里 {k!r} 的\"指定样本\"应该是 [实例号, 帧号], 拿到 {v!r}")
        out[str(k)] = (int(v[0]), int(v[1]))
    return out


VLM_INST = "04_picks/vlm_verify_inst.json"


def vlm_rejected(R1):
    """verify_picks.py 逐实例那一遍否掉的实例 {词: {实例号}}。文件不在当空的(老轮次没有)。"""
    p = f"{R1}/{VLM_INST}"
    if not os.path.exists(p):
        return {}
    return {w: {int(x) for x in v} for w, v in (json.load(open(p)).get("否决") or {}).items()}


def vlm_released(R1):
    """人在闸口看过、决定"VLM 说不是也照样要"的生成名集合。
    ★ VLM 的否决只是**提醒**, 不是判决: 它只看了机器自动挑的那一帧, 而且会看错。
      以前否决 = 直接不生成, 组装再按名字借同词别人的 mesh 摆上去(pedestrian r2 的 car #2
      就这样变成一辆 8 米长的 car_0), 人在闸口没有任何办法把它要回来。
      现在: 否决且没放行 = 不生成、不摆、不借; 放行 = 按人选的帧生成它自己的 mesh。"""
    return set(load(R1).get("VLM放行") or [])


def overrides(R1, fallback=None):
    """**类级**换候选: {词: (实例号, 帧号)}。给 s_override 用 —— 它写的是类目录 my_images/<tag>_<词>/。
    闸的决定压过源码里写死的那张表 —— 闸是人当场看着图做的判断, 源码表是历史遗留。"""
    ov = {w: tuple(v) for w, v in (fallback or {}).items()}
    r = {k: v for k, v in _parse_ov(R1).items() if "#" not in k}
    ov.update(r)
    if r:
        print("[人工闸] 换候选(类) " + str(len(r)) + " 个: "
              + ", ".join(f"{w}->实例{a}/第{b}帧" for w, (a, b) in sorted(r.items())),
              file=sys.stderr)
    return ov


def overrides_inst(R1):
    """**实例级**换候选: {(词, 实例号): 帧号}。给逐实例生成(gen_per_instance)用。

    类级的那条也一并算进来 —— 它的值本来就带着实例号, 人填"这个词改用实例 7 的第 40 帧"时,
    意思显然也包括"实例 7 自己的专属 mesh 就用第 40 帧", 两边分裂反而怪。
    """
    out = {}
    for k, (oid, fi) in _parse_ov(R1).items():
        w = k.split("#", 1)[0] if "#" in k else k
        out[(w, oid)] = fi
    if out:
        print("[人工闸] 换候选(实例) " + str(len(out)) + " 个: "
              + ", ".join(f"{w}#{o}->第{f}帧" for (w, o), f in sorted(out.items())),
              file=sys.stderr)
    return out


def write_skip(R1, dry=False):
    """报一声"单个不生成"有哪些。**不再落盘。**

    ★ 以前它会写 04_picks/skip.json。那个文件全代码库没有读者(filter_dirs 直接读
      05_review_applied.json, 免得依赖"谁先跑"), 而且必然过期 —— 它只在 override 那步写,
      而 drop 大多是在**后面**的召回闸判的。于是盘上躺着一份 `[]`, 决策文件里却有两条,
      人翻 run 目录会被它误导。没有读者、内容还是错的文件, 不如不写。
    """
    skip = sorted(load(R1).get("不生成") or [])
    if skip and not dry:
        print(f"[人工闸] 单个不生成 {len(skip)} 个: {', '.join(skip)}", file=sys.stderr)
    return skip


def filter_dirs(R1, dirs, tag=None):
    """从待生成目录里剔掉人工否掉的。EXCLUDE 是词级的, 挡不住"这个词留着但这一个实例不要",
    而闸口最常见的判断恰好是后者。

    ★ 直接读 05_review_applied.json, **不经过 skip.json 那个中间文件**。
      第一版是 s_override 写 skip.json、生成那步再读 —— 这制造了一个隐藏的时序契约:
      写的人必须排在读的人之前。而分段驱动打破了这个前提: 人工闸在 override **之后**,
      --go 之后直接跳到下一段, override 永远不重跑, skip.json 停留在上一轮。
      去掉中间文件, 时序问题就不存在了。
    """
    skip = set(load(R1).get("不生成") or [])
    if not skip:
        return dirs
    keep = [d for d in dirs if os.path.basename(d) not in skip]
    if len(keep) != len(dirs):
        print(f"    [人工闸] 人工否掉 {len(dirs)-len(keep)} 个候选, 不生成: "
              + ", ".join(sorted(os.path.basename(d) for d in dirs if d not in keep)), flush=True)
    return keep


def skip_names(R1):
    """人工否掉的生成名集合。给组装那步用 —— 人的语义是"这东西别出现在场景里",
    不只是"别再烧这份 GPU"。盘上已经有上一轮生成的 glb 时, 光挡生成挡不住组装。"""
    return sorted(load(R1).get("不生成") or [])


# 词内分组复用不在这里 —— 它在人工闸(review_gate --apply)里就被**展开**成
# "挪类别到派生子类 + 整词复用"两个已有机制了, 运行期没有第三套。
# 见 review_gate.py 里 "复用组" 那段注释。
