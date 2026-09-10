#!/usr/bin/env python
"""重建流水线的驱动(室内室外通用): 自动的部分自动跑, 机器判不了的地方停下来等人。

为什么不直接用 outdoor_pipeline.py —— 那个是步骤执行器, 它不知道"什么时候该停"。
一路跑到底的问题是: 地基错了(尺度、天的方向)后面全废, 但要跑完才看得出来;
生成是最贵的一步, 而该不该生成这件事机器判不了。所以要分段, 段与段之间摆一道闸。

四道闸:
  1 地基   深度 + 天 + 相机轨迹 + 词表。这三样错了后面全废, 而且这一步很便宜
  2 生成前 联络图: 每个待生成候选长什么样。机器只判得了"是不是这个词", 判不了"抠得全不全"
  3 尺度   标定出来的倍数和一致率。改错比不改更糟, 要人点头
  4 验收   渲染回图 + 和分割结果的重叠率

任何一步失败, 或者保险闸拦下来, 就地停, 不往下跑。

用法:
  drive.py --scene carpark              跑到下一道闸就停
  drive.py --scene carpark --go         人看过了, 继续跑下一段
  drive.py --scene carpark --status     现在停在哪
  drive.py --scene carpark --reset      从头开始
  加 -n 干跑, 只打印要执行的命令
"""
import argparse, json, os, subprocess, sys

SAM3D = os.path.expanduser("~/sam3d")
PIPE_OUT = f"{SAM3D}/pipeline/outdoor_pipeline.py"
PIPE_IN  = f"{SAM3D}/pipeline/indoor_pipeline.py"
PY    = os.path.expanduser("~/miniforge3/envs/sam3d/bin/python")
HERE  = os.path.dirname(os.path.abspath(__file__))
TOOLS = f"{SAM3D}/tools"      # 闸口那几张图的脚本都在 tools/ —— 常用的放一处好找
# ★ 别在这里自己抄一份 tag 表 —— pipeline 那边有 SCENE_TAG, 抄一份就会漂移
#   (ueindoor 在 pipeline 里是 "ue", 这里按 scene[:3] 会算成 "uei", 闸口一个候选都找不到)。
_TAG_CACHE = {}
def scene_tag(kind, scene, run):
    """问 pipeline 要 tag, 别自己算。"""
    k = (kind, scene, run)
    if k in _TAG_CACHE: return _TAG_CACHE[k]
    cmd = [PY, PIPE_OUT if kind == "outdoor" else PIPE_IN, "--scene", scene, "--run", run, "--dump-cfg"]
    if kind == "indoor": cmd += ["--chain", "obj"]
    r = subprocess.run(cmd + ["--tag-only"], capture_output=True, text=True)
    t = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    if not t:
        t = f"{scene[:3]}_{run}"
        print(f"[提示] 问不到 tag, 退回 {t} —— 闸口找不到候选的话就是这里错了")
    _TAG_CACHE[k] = t
    return t

# 段 = (闸名, 这一段要跑的步骤, 到闸口做什么检查)
# 每条链一套分段。步骤名必须存在于各自 pipeline 的 STEPS 里, 否则跑到那一段直接 SystemExit
# (walls 就这么炸过: 步骤摘掉了, 这张表没跟着改, 室内最后一段整段跑不了)。
#
# ★ 这张表是第五个注册点。加一个步骤要改的地方是: STEPS / ARTIFACTS / DEPS / GPU 标记
#   + 这里。漏了这里不会报错, 只会静默不跑 —— dedup 就这么漏过, 跨词去重整个没生效。
#   下面 _check_segments() 会在启动时核对, 别再靠人记得。
#
# ★ 2026-09-09 调整: 召回五步挪到 assemble 之前(原来排在 render 之后, 结果永远进不了场景),
#   顺带让"召回"那道闸也出联络图, 补抠回来的候选终于能和主线候选一起过目。
#   召回没法再往前挪到 gen 之前: detect 要吃 scale 校准过的深度 npy, 提前就成环了。
SEGMENTS = {
    "outdoor": [
        ("地基",   ["frames", "vlm", "da3", "up"],                                     "check_ground"),
        ("生成前", ["track", "dedup", "relabel", "pick", "verify", "override"],        "check_picks"),
        ("尺度",   ["gen", "ratio", "meshes", "boxes", "vlm_sizes", "scale"],          "check_scale"),
        ("召回",   ["solo", "boxes_wd", "compare", "detect", "recall", "recall_mask"], "check_recall"),
        ("验收",   ["recall_gen", "recall_merge", "assemble", "phys", "render"],       "check_scene"),
        ("收尾",   ["building_scene", "assemble_wd", "phys_wd", "render_wd"],          "check_final"),
    ],
    # 室内物体链: 多了提亮/抽帧/z朝上三步, 墙壳走 walls+shell, 没有 WD 对照那一支
    "indoor": [
        ("地基",   ["frames", "bright", "sub", "vlm", "da3", "up"],                    "check_ground"),
        ("生成前", ["track", "dedup", "relabel", "pick", "verify", "override"],        "check_picks"),
        ("尺度",   ["gen", "ratio", "meshes", "boxes", "vlm_sizes", "scale"],          "check_scale"),
        ("召回",   ["solo", "detect", "recall", "recall_mask"],                        "check_recall"),
        # walls 已从两张步骤表里摘掉(房间壳改从点云自己算足迹), 这里再列它会直接 SystemExit
        # zup 排在 scale 之后(见 indoor_pipeline 的注释), 所以它在这一段不在地基段
        ("验收",   ["recall_gen", "recall_merge", "zup", "shell",
                    "assemble", "phys", "render"],                                     "check_scene"),
    ],
}

def _pipeline_steps(kind):
    """从 pipeline 里把步骤名读出来。不 import(那会连带执行模块级的东西),
    直接跑 --list 拿它自己打印的清单 —— 谁是权威就问谁。"""
    cmd = [PY, PIPE_OUT if kind == "outdoor" else PIPE_IN, "--list"]
    if kind == "indoor": cmd += ["--chain", "obj"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[提示] {' '.join(cmd)} 退出码 {r.returncode}")
        for l in (r.stderr or "").strip().split("\n")[-5:]:
            if l.strip(): print("        " + l)
        return []
    out = []
    for l in r.stdout.split("\n"):
        l = l.strip()
        if l and "[" in l and not l.startswith("步骤") and l.split("[")[0].strip():
            out.append(l.split("[")[0].strip())
    return out

def check_segments(kind):
    """核对分段表和 pipeline 的步骤表。**这是那五个注册点里最容易漏的一个。**

    历史: walls 从步骤表摘掉了但这张表没跟着改 -> 室内最后一段整段 SystemExit;
          dedup 加进了步骤表但没加进这张表 -> 跨词去重从来没生效, 而且一声不吭。
    前者跑到那一段才炸, 后者永远不炸 —— 后者更糟。所以两种都在启动时挡住。
    """
    have = _pipeline_steps(kind)
    if not have:
        # 读不到就不挡(别让自检本身变成故障源), 但必须说一声 ——
        # pipeline 一旦有语法错或 import 失败, 自检会整个消失, 而那正是最需要它的时候。
        print(f"[提示] 分段表自检跳过: {kind} 的 --list 读不出步骤名。"
              f"pipeline 可能有语法错或 import 失败, 下面的错都可能是它引起的。")
        return
    listed, bad = [], []
    for _, steps, _ in SEGMENTS[kind]:
        for st in steps:
            listed.append(st)
            if st not in have: bad.append(st)
    if bad:
        sys.exit(f"[失败] 分段表里这些步骤在 {kind} 的 STEPS 里不存在: {sorted(set(bad))}\n"
                 f"       现有的步骤: {have}")
    dup = sorted({x for x in listed if listed.count(x) > 1})
    if dup:
        sys.exit(f"[失败] 分段表里这些步骤出现了不止一次(会跑两遍): {dup}")
    # 顺序也要核。只查"存在"挡不住把某一步放错段 —— zup 挪到 scale 之后时,
    # 这张表还把它留在地基段, 结果它会在依赖还没满足的时候先跑一遍。
    idx = {k: n for n, k in enumerate(have)}
    order = [idx[x] for x in listed]
    if order != sorted(order):
        bad = [(listed[i], listed[i+1]) for i in range(len(order)-1) if order[i] > order[i+1]]
        sys.exit(f"[失败] 分段表的顺序和 {kind} 的 STEPS 对不上, 这几处倒了: {bad}\n"
                 f"       STEPS 是权威, 照它排。")
    miss = [x for x in have if x not in listed]
    if miss:
        print(f"[提示] 这些步骤不在任何一段里, 用 drive.py 永远跑不到: {miss}")

def state_path(R1): return f"{R1}/logs/.drive_state.json"
def load_state(R1):
    p = state_path(R1)
    return json.load(open(p)) if os.path.exists(p) else dict(段=0, 停在闸口=False)
def save_state(R1, s):
    os.makedirs(f"{R1}/logs", exist_ok=True)
    json.dump(s, open(state_path(R1), "w"), ensure_ascii=False, indent=1)

def run_steps(kind, scene, run, steps, dry):
    cmd = [PY, PIPE_OUT if kind == "outdoor" else PIPE_IN,
           "--scene", scene, "--run", run, "--only", ",".join(steps)]
    if kind == "indoor": cmd += ["--chain", "obj"]
    if dry: cmd.append("-n")
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd).returncode

# ---------------- 每道闸的检查: 只报数和图, 不下结论 ----------------
def _j(p, d=None):
    return json.load(open(p)) if os.path.exists(p) else d

def da3_dir(R1):
    """室外叫 02_da3, 室内叫 02_DA3点云 —— 两条链的目录名从一开始就不一样。"""
    for n in ("02_da3", "02_DA3点云"):
        if os.path.isdir(f"{R1}/{n}"): return f"{R1}/{n}"
    return f"{R1}/02_da3"

def check_ground(R1, tag, scene, kind="outdoor", run="r1"):
    look, lines = [], []
    D3 = da3_dir(R1)
    up = _j(f"{D3}/world_up.json")
    if up:
        lines.append(f"天的方向偏离老假设 {up.get('偏离老假设度数')}°, 地面拟合内点比例 {up.get('内点比例')}")
        lines.append(f"地面高度 {up.get('ground_h'):.2f} m")
    cams = _j(f"{D3}/cameras.json")
    if cams:
        import numpy as np
        C = np.array([v["camera_center_world"] for _, v in sorted(cams["cameras"].items())])
        step = np.linalg.norm(np.diff(C, axis=0), axis=1)
        span = float(np.linalg.norm(C[0]-C[-1]))
        lines.append(f"相机 {len(C)} 帧, 轨迹总长 {step.sum():.2f} m, 首尾 {span:.2f} m")
        # 视差够不够看的是"相机挪的距离 / 物体的距离", 不是帧数。
        # 实测: timesquare 走了 19.7 m 效果最好; carpark 1.5 m、beach 0.9 m,
        # 每个物体的视角张角只有 2.2° 和 0.1°, 等于一百多张同一个视角的照片 ——
        # 朝向判不出来, 尺度也没有约束。这一条比任何算法改进都要紧, 所以要显眼地报出来。
        if span < 5.0:
            lines.append(f"  ★ 轨迹只有 {span:.1f} m —— 几乎没有视差。朝向和尺度都会不可靠,"
                         f" 细长物体容易躺倒。重拍时让相机走动, 别悬停。")
        jump = float(np.percentile(step, 99)/max(np.median(step), 1e-9))
        if jump > 5: lines.append(f"  ★ 帧间位移有跳变 (99分位是中位的 {jump:.0f} 倍), 位姿可能出错了")
    v = _j(f"{R1}/01_vlm_vocab.json")
    if v: lines.append(f"词表 {len(v['vocab'])} 个: {', '.join(v['vocab'])}")
    # 这张图现场生成: 抽帧一览 + 点云俯视(带相机轨迹) + 点云侧视(带拟合的地面线)。
    # 光看数字看不出"地面拟合到底贴不贴", 得看侧视图里那条红线是不是压在点云底部。
    sc = f"{TOOLS}/viz_ground.py"
    if os.path.exists(sc):
        env = dict(os.environ, R1=R1, FR=frames_dir(kind, scene, run), OUT=f"{R1}/02_ground_check.jpg")
        subprocess.run([PY, sc], env=env, capture_output=True, text=True)
    look += [f"{R1}/02_ground_check.jpg"]
    look += ([f"{R1}/02_da3_topview.png"] if kind == "outdoor"
             else [f"{R1}/02_DA3点云_俯视.png", f"{R1}/03_点云_分层.jpg"])
    return lines, look

def frames_dir(kind, scene, run):
    """室外: dataset/<场景>/frames。
    室内: 产物在 data/indoorOutput/<场景>/ 下, 物体链吃的是 frames_<run> ——
    那是从提亮帧里等间隔抽出来、重新编号成 0000..N-1 的一套。不能拿 frames_bright 顶替:
    DA3 的 cameras.json 键指的是"它读到的第几张", 不是原视频帧号, 配错了会用一帧的 mask
    去另一帧上取像素(轮廓对、里面的画面错位)。"""
    if kind == "outdoor": return f"{SAM3D}/data/outdoor/dataset/{scene}/frames"
    o = f"{SAM3D}/data/indoorOutput/{scene}"
    for cand in (f"{o}/frames_{run}", f"{o}/frames_bright", f"{o}/frames"):
        if os.path.isdir(cand): return cand
    return f"{o}/frames_{run}"

def gate_env(R1, tag, scene, kind, run):
    """给人工闸准备环境。**必须把 SOLO/EXCLUDE 那几个也带上。**

    ★ 闸要展示"哪些东西会被生成", 而"会不会逐实例生成"由 SOLO/SOLO_KEEP/SOLO_TH 决定,
      这几个值住在 pipeline 的 resolve_cfg 里。以前这里只传 R1/TAG/FR, 于是 review_gate
      里 SOLO_CFG 拿到空串, 整个逐实例枚举被跳过 —— 而 beach/carpark 配的是 SOLO="*"
      (全部逐实例), 等于联络图上一个真正会生成的候选都没有。
      不自己抄一份配置, 直接问 pipeline 要 —— 谁是权威就问谁。
    """
    env = dict(os.environ, R1=R1, TAG=tag, FR=frames_dir(kind, scene, run))
    cmd = [PY, PIPE_OUT if kind == "outdoor" else PIPE_IN, "--scene", scene, "--run", run, "--dump-cfg"]
    if kind == "indoor": cmd += ["--chain", "obj"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        try:
            cfg = json.loads(r.stdout)
            for k in ("SOLO", "SOLO_KEEP", "SOLO_TH", "TH", "EXCLUDE", "MIN_AREA_PCT"):
                if k in cfg: env[k] = str(cfg[k])
            # 室内那边逐实例门槛的键叫 TH 不叫 SOLO_TH
            if "SOLO_TH" not in cfg and "TH" in cfg: env["SOLO_TH"] = str(cfg["TH"])
        except Exception as e:
            print(f"[提示] 配置读不出来({e}), 闸只能看到类候选, 逐实例那批会缺")
    else:
        print(f"[提示] --dump-cfg 失败(退出码 {r.returncode}), 闸只能看到类候选")
    return env

def check_picks(R1, tag, scene, kind="outdoor", run="r1"):
    env = gate_env(R1, tag, scene, kind, run)
    # 闸 2 = 召回前。阶段显式传给闸口脚本 —— 让它自己猜的话, 只要盘上有过召回产物,
    # 这一段的重跑就会把召回前的清单写进 _afterrecall 那份, 覆盖掉真正的召回后结果。
    env = dict(env, REVIEW_STAGE="pre")
    r = subprocess.run([PY, f"{TOOLS}/review_gate.py"], env=env, capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).strip().split("\n") if l.strip()]
    # 复用表: 联络图逐格判"这一个要不要生成", 复用表横着比"这一批该不该长成同一个样子"。
    # 两个粒度不同的问题, 各一张图。这一步纯 CPU, 几秒。
    t = subprocess.run([PY, f"{TOOLS}/reuse_table.py"], env=env, capture_output=True, text=True)
    lines += [l for l in (t.stdout + t.stderr).strip().split("\n") if l.strip()]
    return lines, [f"{R1}/05_review_sheet.jpg", f"{R1}/05_reuse_table.jpg"]

def check_scale(R1, tag, scene, kind="outdoor", run="r1"):
    lines = []
    # 施加了写 world_scale.json, 没施加(SCALE_APPLY=0 或一致率不达标)写 world_scale_estimate.json。
    # 两个都要看 —— 闸 3 恰恰是人来判"这个倍数该不该信"的地方, 没施加的那次更需要摆出来。
    _sp = f"{da3_dir(R1)}/world_scale.json"
    if not os.path.exists(_sp): _sp = f"{da3_dir(R1)}/world_scale_estimate.json"
    w = _j(_sp)
    if w:
        lines.append(f"标定倍数 k = {w['k']}, 一致率 {w['一致率']}, 采用 {w['采用类数']}/{w['候选类数']} 类, 施加={w['施加']}")
        # k≈1 是**正常结果**(室内几乎每次都走这条), 不该顶着醒目星号 ——
        # 醒目警告出现在正常结果上, 看几次就没人看了。
        if w.get("无需施加"):
            lines.append(f"  倍数接近 1, 无需施加 —— 这个场景本来就是对的")
        elif not w["施加"]:
            lines.append(f"  ★ 没有施加, 原因: {w.get('拒绝原因')}")
            if w.get("一致率达标") is False:
                lines.append(f"  ★ 一致率 {w['一致率']} 没到闸门 {w['闸门']} —— 先看下面哪个词的参照高度不对")
        for r in sorted(w["逐类"], key=lambda x: -x["倍数"])[:6]:
            lines.append(f"    {r['词']:14s} {r['实例数']:3d}个  量到 {r['量到']:.2f} m  真实 {r['真实']:.2f} m  倍数 {r['倍数']:.2f}"
                         + ("" if r.get("采用") else "  <- 离群不用"))
    b = _j(f"{R1}/06_all_boxes.json", {})
    import numpy as np
    for w_ in ("car", "van", "truck", "bench", "chair", "table"):
        if w_ in b:
            lines.append(f"  {w_}: {len(b[w_])} 个, 高度中位 {np.median([x['尺寸'][2] for x in b[w_]]):.2f} m")
    return lines, [f"{R1}/05_meshes.jpg", _sp]

def check_scene(R1, tag, scene, kind="outdoor", run="r1"):
    lines = []
    log = f"{R1}/logs/07_assemble.log"
    if os.path.exists(log):
        txt = open(log).read().split("== assemble")
        # ★ [跳过] 底下那几行**是缩进的续行, 没有方括号标签** —— 按标签过滤会把它们
        #   全滤掉, 于是闸口只剩一句"共 6 个:" 后面空空如也。而 assemble 特意按原因
        #   分了组(见 assemble_scene.py:404 的注释: 分不清"阈值定错"还是"盒子量错",
        #   这两件事修法完全相反), 那份用心在这里被扔了。续行跟着它的标题一起留。
        续 = False
        for l in txt[-1].split("\n"):
            if any(k in l for k in ("[组装]", "[跳过]", "[朝向]", "[地面]")):
                lines.append(l.strip()); 续 = l.startswith("[跳过]")
            elif 续 and l.startswith("   ") and l.strip():
                lines.append("  " + l.strip())
            elif l.strip():
                续 = False
    return lines, [f"{R1}/08_render_back_phy.jpg"]

def check_recall(R1, tag, scene, kind="outdoor", run="r1"):
    lines = []
    r = _j(f"{R1}/13_recall.json")
    if r:
        from collections import Counter
        lines.append(f"召回: {r['对上']} 个和已有的对上, {len(r['召回'])} 个是漏检 (总 {r['总实例']})")
        lines.append(f"  漏检按词: {dict(Counter(a['词'] for a in r['召回']).most_common())}")
        # 光看数字不够。二维去重修好之前, 那 48 个"漏检"没有一个是真的 ——
        # 每一个底下都压着已经抠出来的东西。所以这里直接把核对图生成出来给人看。
        sc = f"{TOOLS}/viz_recall_2d.py"
        if os.path.exists(sc):
            # TAG 必须传: 脚本要拿它拼生成名, 才能和下游"真会生成"的清单对上。
            env = dict(os.environ, R1=R1, TAG=tag, FR=frames_dir(kind, scene, run),
                       OUT=f"{R1}/95_recall_check_2d.jpg", N="12")
            v = subprocess.run([PY, sc], env=env, capture_output=True, text=True)
            for l in v.stdout.strip().split("\n"):
                if "IoU" in l or "真漏检" in l or "[不画]" in l: lines.append("  " + l.strip())
    # ★ 联络图在这里再出一次。review_gate.py 本来就会读 14_recall_ready.json, 但它以前只在
    #   第 2 道闸跑, 那时候召回还没跑、文件还不存在, 于是那段代码永远扫不到东西 ——
    #   补抠回来的候选(质量最没保证的一批)是唯一没有人工把关的一支。现在补上。
    #   重复跑不会覆盖已经改过的决定, 只把新出现的候选追加进去(见 review_gate 的说明)。
    env = gate_env(R1, tag, scene, kind, run)
    env = dict(env, REVIEW_STAGE="post")      # 闸 4 = 召回并进来之后
    g = subprocess.run([PY, f"{TOOLS}/review_gate.py"], env=env, capture_output=True, text=True)
    lines += [l for l in (g.stdout + g.stderr).strip().split("\n") if l.strip()]
    # ★ 闸 4 出的是 _afterrecall 那一版, 不是闸 2 那张 —— 列错了人会去看旧图。
    #   闸 2 那张留着不动, 是"回退到 recall 之前"的依据。
    t = subprocess.run([PY, f"{TOOLS}/reuse_table.py"], env=env, capture_output=True, text=True)
    lines += [l for l in (t.stdout + t.stderr).strip().split("\n") if l.strip()]
    return lines, [f"{R1}/95_recall_check_2d.jpg",
                   f"{R1}/05_review_sheet_afterrecall.jpg",
                   f"{R1}/05_reuse_table_afterrecall.jpg",
                   f"{R1}/92_compare_pca_vs_wd.jpg"]

def check_final(R1, tag, scene, kind="outdoor", run="r1"):
    return ["全部跑完"], [f"{R1}/08_render_back_phy.jpg", f"{R1}/18_render_back_wd_phy.jpg"]

CHECKS = dict(check_ground=check_ground, check_picks=check_picks, check_scale=check_scale,
              check_scene=check_scene, check_recall=check_recall, check_final=check_final)

def gate_report(i, R1, tag, scene, kind, run, last=False):
    segs = SEGMENTS[kind]
    name, steps, fn = segs[i]
    print(f"\n{'='*72}\n闸 {i+1}/{len(segs)} — {name}\n{'='*72}")
    try:
        lines, look = CHECKS[fn](R1, tag, scene, kind, run)
    except Exception as e:
        lines, look = [f"(检查本身出错了: {e})"], []
    for l in lines: print("  " + l)
    exist = [p for p in look if os.path.exists(p)]
    if exist:
        print("\n  该看的图:")
        for p in exist: print(f"    {p}")
    missing = [p for p in look if not os.path.exists(p)]
    if missing and not exist:
        print("  (图还没生成: " + ", ".join(os.path.basename(p) for p in missing) + ")")
    flag = "" if kind == "outdoor" else " --indoor"
    if last: print(f"\n★ 这是最后一道闸, {len(segs)} 段都跑完了。")
    else:    print(f"\n★ 停在这里等人。看完之后: drive.py --scene {scene}{flag} --go")

def _applied(R1):
    """读机器版决策文件的快照。不存在或坏了一律当空 —— 这里只用来比"变没变",
    真正的坏文件硬停在 review_decisions.load 里, 不在这儿重复报。"""
    try:
        return json.load(open(f"{R1}/05_review_applied.json"))
    except Exception:
        return {}


def apply_decisions(R1, tag, scene, kind, run, seg, dry):
    """人按 --go 往下走之前, 把他在闸口做的决定落成流水线读得懂的文件。

    ★ 2026-09-09 之前这一步整个不存在: 闸只出联络图和决策文件, --apply 从来没人调,
      于是人改完 05_review.json 什么都不会发生, 得自己去改 outdoor_pipeline.py 的源码。
      现在 --go 会先跑 --apply, 产物 05_review_applied.json 由 resolve_cfg 直接读。

    只对"生成前"那道闸有意义 —— 别的闸没有决策文件, 静默跳过。
    """
    if not os.path.exists(f"{R1}/05_review.json"): return
    if SEGMENTS[kind][seg][2] not in ("check_picks", "check_recall"): return
    print(f"\n[落地决策] 把闸口的决定写成配置 ...")
    env = gate_env(R1, tag, scene, kind, run)
    # 落地时也要说清是哪道闸 —— --apply 读的决策文件按阶段分两份
    env = dict(env, REVIEW_STAGE=("post" if SEGMENTS[kind][seg][2] == "check_recall" else "pre"))
    before = _applied(R1)          # 落地前先拍一张, 待会儿比对哪几类决定真的变了
    if dry:
        print(f"    $ {PY} {TOOLS}/review_gate.py --apply"); return
    r = subprocess.run([PY, f"{TOOLS}/review_gate.py", "--apply"], env=env,
                       capture_output=True, text=True)
    print("\n".join("    " + l for l in (r.stdout + r.stderr).strip().split("\n") if l.strip()))
    if r.returncode != 0:
        # 决策落不了地就往下跑, 等于人白看一遍而且不知道自己白看了 —— 硬停。
        sys.exit(f"\n★ 决策没能落地(退出码 {r.returncode})。改好 {R1}/05_review.json 再 --go。")

    # ★ 决定落完, 必须把**依赖这些决定的步骤重跑一遍** —— 闸不是一个通过点, 是一个回路。
    #
    #   为什么:  shared/drop-word 改的是 SOLO_KEEP/EXCLUDE, 而"哪些词出类 mesh"是 pick 决定的,
    #            pick 排在闸**之前**。不重跑的话, 人标了 shared 的词既不逐实例生成
    #            (被 SOLO_KEEP 挡掉)、也没有类 mesh(pick 跑的时候还不知道要出)——
    #            **整个词从场景里消失**。beach 实测: bench/soda machine/trash can/tree
    #            27 个物体会全没, 而且不报错。
    #            relabel 更直接: 它把实例在 mask 文件之间搬家, pick 必须在搬完之后再挑。
    #
    #   顺序:    relabel(搬家) -> pick(重挑) -> verify(重验) -> override(换候选)
    #            只重跑这一段, 前面的 track/dedup 不动(它们不依赖闸的决定)。
    #   ★ 判"要不要重跑"看的是**决定本身变没变**, 不是"这一段里有没有这些步骤"。
    #     原来写的是 `x in steps_of_seg` —— 而 relabel/pick/verify/override 全在"生成前"那一段,
    #     召回闸(第 4 段)的步骤是 solo/detect/recall/... , 两个集合**交集为空**,
    #     于是闸 4 填的 shared / solo / 改类别 / 换候选**一个都不会生效**, 而且不吭声。
    #     决定消费在前面的段里, 这很正常: 闸是回路, 不是通过点。
    #   ★ 只动"不生成"的那一类不用重跑 —— 生成那步是当场读 05_review_applied.json 的
    #     (review_decisions.filter_dirs), 没有时序依赖。闸 4 绝大多数情况就是这一类,
    #     不加这个判断的话每次 --go 都要白重跑一遍挑图。
    NEED_RERUN_KEYS = ("排除词", "逐实例词", "共用词", "改类别", "指定样本")
    after = _applied(R1)
    # 空列表/空字典/缺键都当"没有决定" —— 第一次跑时 before 是 {}, before.get() 给 None,
    # 而 after 里是 [] , None != [] 会把 5 个键全判成"变了", 白重跑一遍挑图(verify 还吃 GPU)。
    _norm = lambda v: v or None
    changed = [k for k in NEED_RERUN_KEYS if _norm(before.get(k)) != _norm(after.get(k))]
    if not changed:
        print("[落地决策] 只动了'不生成'一类的决定, 生成那步当场读, 不用重跑任何步骤")
        return
    print(f"[落地决策] 这些决定变了: {', '.join(changed)} —— 消费它们的步骤要重来")
    #   ★ 重跑清单必须一直排到**真正产 mesh 和产盒子**的那几步, 否则等于没修:
    #     闸 4 填"共用词" -> pick 重挑、抠图建好了, 但 gen 不重跑 -> 没有类 mesh,
    #     这个词又被 SOLO_KEEP 挡住不逐实例 -> 组装时两头都找不到, **整个词从场景里消失**。
    #     闸 4 填"改类别" -> relabel 把实例搬进 umbrella~1.npz, 但 boxes 不重跑 ->
    #     盒子文件里还是老的 umbrella, 组装照样找不到 mesh, 14 把伞全没。
    #     两种都不报错。这就是注释开头说的那 27 个物体的形状。
    #   代价可控: 抠图现在按内容判变没变(imgio.imwrite_stable), 内容没变 mtime 就不动,
    #   gen/solo 会照常"已有且不过期, 跳过", 不会白烧 GPU。
    LOOP = ["relabel", "pick", "verify", "override", "gen", "ratio", "boxes", "solo"]
    have = _pipeline_steps(kind)
    # 跑过的才重跑(有完成戳 = 跑过)。不限本段: 闸 4 要重跑的恰恰是第 2 段的那几步。
    todo = [x for x in LOOP if x in have and os.path.exists(f"{R1}/logs/.stamp_{x}")]
    if todo:
        print(f"[落地决策] 决定变了, 重跑 {' -> '.join(todo)} ...")
        if run_steps(kind, scene, run, todo, dry) != 0:
            sys.exit("\n★ 重跑失败, 闸口的决定没能落到实处。就地停。")
    else:
        # 刚说完"这些决定变了, 消费它们的步骤要重来"却一步都没跑, 只能是问不到步骤名
        # (pipeline 有语法错 / import 失败)。静默返回等于人白填一遍还不知道。
        sys.exit(f"\n★ 说要重跑, 却一个步骤都排不出来 —— 多半是问不到 {kind} 的步骤表"
                 f"(pipeline 语法错或 import 失败)。先修好它, 决定还在文件里没丢。")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", default="r1")
    ap.add_argument("--indoor", action="store_true", help="走室内那条链 (默认室外)")
    ap.add_argument("--go", action="store_true", help="人看过了, 继续下一段")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("-n", "--dry-run", action="store_true")
    a = ap.parse_args()
    kind = "indoor" if a.indoor else "outdoor"
    check_segments(kind)      # 五个注册点里最容易漏的那个, 启动就核
    segs = SEGMENTS[kind]

    # 场景名打错的话, pipeline 会拿默认配置照跑不误, 干跑还看着像成功的 —— 先挡住
    root = f"{SAM3D}/data/outdoor/dataset" if kind == "outdoor" else f"{SAM3D}/data"
    if not os.path.isdir(f"{root}/{a.scene}"):
        have = sorted(d for d in os.listdir(root)) if os.path.isdir(root) else []
        sys.exit(f"[失败] 没有这个场景: {root}/{a.scene}\n       现有的: {', '.join(have)}")

    R1 = (f"{SAM3D}/data/outdoor/result/{a.scene}/{a.run}" if kind == "outdoor"
          else f"{SAM3D}/data/indoorOutput/{a.scene}/{a.run}")
    tag = scene_tag(kind, a.scene, a.run)
    s = load_state(R1)

    if a.reset:
        save_state(R1, dict(段=0, 停在闸口=False)); print(f"[重置] {a.scene} 回到第一段"); return
    if a.status:
        i = s["段"]
        print(f"场景 {a.scene} ({kind})  共 {len(segs)} 段")
        for k, (nm, steps, _) in enumerate(segs):
            mark = "← 现在" if k == i else ("done" if k < i else "")
            print(f"  {k+1}. {nm:8s} {' '.join(steps):64s} {mark}")
        if s["停在闸口"]: print(f"\n停在闸 {i+1} 等人。继续: --go")
        return

    if s["停在闸口"]:
        if not a.go:
            print(f"[等人] 停在闸 {s['段']+1}。看完之后加 --go 继续, 或 --status 看全貌。")
            gate_report(s["段"], R1, tag, a.scene, kind, a.run,
                        last=(s["段"] == len(segs)-1)); return
        apply_decisions(R1, tag, a.scene, kind, a.run, s["段"], a.dry_run)
        s["段"] += 1; s["停在闸口"] = False; save_state(R1, s)

    while s["段"] < len(segs):
        i = s["段"]; name, steps, _ = segs[i]
        print(f"\n{'#'*72}\n# 第 {i+1} 段: {name}   ({' -> '.join(steps)})\n{'#'*72}")
        rc = run_steps(kind, a.scene, a.run, steps, a.dry_run)
        if rc != 0:
            print(f"\n★ 第 {i+1} 段失败 (退出码 {rc}), 就地停。日志在 {R1}/logs/")
            sys.exit(rc)
        s["停在闸口"] = True; save_state(R1, s)
        gate_report(i, R1, tag, a.scene, kind, a.run, last=(i == len(segs)-1))
        return                                # 一次只跑到下一道闸

    print(f"\n[全部跑完] {len(segs)} 段都过了。")

if __name__ == "__main__":
    main()
