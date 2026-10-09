#!/usr/bin/env python
"""本轮控制 —— 闸口网页里启动页 /new 和本轮页 /run 的后端(2026-09-24)。

gate_ui.py 只接路由, 逻辑都在这里, 方便单测(tools/tests/test_run_ctl.py)。管四件事:
  本轮配置 00_run_config.json   有没有人 / 开不开召回(规则在 _common.update_run_config, 和 drive.py 同一套)
  挑人决策 05_people.json       第 2 道闸勾人(pipeline/people_decisions.py)
  驱动状态                      logs/.drive_state.json + 有没有驱动进程在跑
  开跑 / 继续                   只通过 tools/drive_go.sh

★ 为什么只通过 drive_go.sh 起驱动: 它把驱动放进后台 tmux、和网页进程脱钩 —— 网页关了活照跑,
  网页不持有任何子进程, 也就没有僵尸、没有"界面关了活还在跑一半"。gate_ui 原来"不起流水线子进程"
  那条边界的本意就是这个; 现在放开的只有"让 drive_go.sh 替人敲一次 --go", 同一轮在跑就拒绝。
"""
import glob, importlib.util, json, os, re, subprocess, sys, time

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import _common, people_decisions        # noqa: E402

# 类型 -> (数据集根, 结果根)。和 drive.py / pipeline 的路径一致; 测试会把它指到临时目录。
ROOTS = {"outdoor": (f"{SAM3D}/data/outdoor/dataset", f"{SAM3D}/data/outdoor/result"),
         "indoor":  (f"{SAM3D}/data/indoor/dataset", f"{SAM3D}/data/indoorOutput")}
PIPES = {"outdoor": f"{SAM3D}/pipeline/outdoor_pipeline.py", "indoor": f"{SAM3D}/pipeline/indoor_pipeline.py"}
PY = os.path.expanduser("~/miniforge3/envs/sam3d/bin/python")
DRIVE_GO = f"{SAM3D}/tools/drive_go.sh"
# 场景名/轮次名会进路径、进 drive_go.sh 拼的 tmux 命令行 —— 只许这几种字符, 不许 ../ 和空格引号
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _check_name(*names):
    for n in names:
        if not NAME.match(n or ""):
            raise ValueError(f"名字只能用字母数字和 _ . -(不能以符号开头): {n!r}")


def r1_of(kind, scene, run):
    return f"{ROOTS[kind][1]}/{scene}/{run}"


_DRIVE = None
def _segments(kind):
    """段数问 drive.py 的 SEGMENTS, 别自己抄一份。"""
    global _DRIVE
    if _DRIVE is None:
        p = f"{SAM3D}/.claude/skills/run-recon-pipeline/drive.py"
        spec = importlib.util.spec_from_file_location("drive", p)
        _DRIVE = importlib.util.module_from_spec(spec); spec.loader.exec_module(_DRIVE)
    return _DRIVE.SEGMENTS[kind]


def _next_run(res, sc):
    n = 1
    while os.path.exists(f"{res}/{sc}/r{n}"):
        n += 1
    return f"r{n}"


def datasets():
    """两个数据集根下有哪些场景, 以及每个场景下一个没用过的轮次名。
    场景 = 子目录里有 frames/ 或视频; ★ 根目录下散放的视频(nfl1.mp4)也算, 标"新视频" ——
    选它开跑时 new_run 会建 <场景>/video.mp4。和某个子目录同名的散放视频不重复列, 子目录说了算。"""
    out = {}
    for kind, (ds, res) in ROOTS.items():
        rows = []
        names = sorted(os.listdir(ds)) if os.path.isdir(ds) else []
        for sc in names:
            d = f"{ds}/{sc}"
            if sc.startswith(".") or not os.path.isdir(d):
                continue
            if not (os.path.isdir(f"{d}/frames") or any(f.lower().endswith((".mp4", ".mov")) for f in os.listdir(d))):
                continue
            rows.append({"场景": sc, "下一轮": _next_run(res, sc)})
        for f in names:
            stem, ext = os.path.splitext(f)
            if ext.lower() in (".mp4", ".mov") and os.path.isfile(f"{ds}/{f}") \
                    and not os.path.exists(f"{ds}/{stem}") and NAME.match(stem):
                rows.append({"场景": stem, "下一轮": _next_run(res, stem), "新视频": f})
        out[kind] = sorted(rows, key=lambda r: r["场景"])
    return out


def running(scene, run):
    """这一轮有没有驱动进程在跑。和 drive_go.sh 用同一个判据。"""
    pat = f"drive.py.*--scene {re.escape(scene)}( |$).*--run {re.escape(run)}( |$)"
    return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0


# ---------------------------------------------------------------- 在跑什么、用哪张卡(2026-09-29)
# 并行之后一段里可以有好几步同时在跑, 各自领了卡(见 pipeline/sched.py 和 _common.GpuSlot)。
# 调度和领卡的时候各写一个标记文件, 这里只管读。
# ★ 进程被杀(OOM / kill -9)的时候来不及删标记: 进程已经不在的标记不算数。
def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True                     # 进程在, 只是不归我们管
    except (OSError, ValueError):
        return False


def running_steps(R1):
    """调度正在跑的步骤(logs/.running_<步骤>, 内容是子进程号)。"""
    out = []
    for p in sorted(glob.glob(f"{R1}/logs/.running_*")):
        try:
            pid = open(p).read().strip()
        except OSError:
            continue
        if _alive(pid):
            out.append(os.path.basename(p)[len(".running_"):])
    return out


def gpu_tasks(R1):
    """谁在用哪张卡: [{"卡": "0", "步骤": "track/w0"}], 按卡号排(logs/.gpu_<卡>.json)。"""
    out = []
    for p in glob.glob(f"{R1}/logs/.gpu_*.json"):
        try:
            d = json.load(open(p))
            if _alive(d["pid"]):
                out.append({"卡": str(d["卡"]), "步骤": d["步骤"]})
        except (OSError, ValueError, KeyError):
            continue                    # 正在写的、写坏的: 下一次刷新再看
    return sorted(out, key=lambda t: (len(t["卡"]), t["卡"]))


def _drive_state(R1):
    try:
        s = json.load(open(f"{R1}/logs/.drive_state.json"))
        return int(s.get("段", 0)), bool(s.get("停在闸口"))
    except (OSError, ValueError):
        return 0, False


def _effective(kind, scene, run):
    """没有本轮配置的老轮次: 问 pipeline 要生效的配置(场景 SCENES 里可能手工配了 BODY_RUN)。"""
    cmd = [PY, PIPES[kind], "--scene", scene, "--run", run, "--dump-cfg"] + (["--chain", "obj"] if kind == "indoor" else [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout)
    except Exception:
        return {}


def state(kind, scene, run):
    """本轮页要的全部: 类型、有没有人、开不开召回、哪些还能改、驱动停在哪、在不在跑、挑人表。"""
    _check_name(scene, run)
    R1 = r1_of(kind, scene, run)
    rc = _common.load_run_config(R1)
    seg, stop = _drive_state(R1)
    total = len(_segments(kind))
    if seg >= total:                      # 段表变短了(10-08 室外六段变五段): 老轮次停在原第 6 道闸 = 全部跑完
        seg, stop = total, False
    done = seg + (1 if stop else 0)
    busy = running(scene, run)
    if rc is not None:
        over = rc.get("覆盖", {})
        people, recall, manual = over.get("PEOPLE", "0"), over.get("RECALL", "0"), False
    else:
        cfg = _effective(kind, scene, run)
        manual = bool(cfg.get("BODY_RUN", "").strip()) and cfg.get("PEOPLE") != "1"
        people = "1" if (cfg.get("PEOPLE") == "1" or manual) else "0"
        recall = cfg.get("RECALL", "0")
    until = _common.EDITABLE_UNTIL
    body = people_decisions.auto_dir(R1)
    tasks = gpu_tasks(R1) if busy else []
    return {"场景": scene, "轮次": run, "类型": (rc or {}).get("类型", kind), "配置文件": rc is not None,
            "有人": people, "召回": recall, "手工人体": manual,
            "可改": {"PEOPLE": not busy and not manual and done <= until["PEOPLE"],
                     "RECALL": not busy and done <= until["RECALL"]},
            "段": seg, "停在闸口": stop, "闸": seg + 1 if stop else None, "已跑完": done, "总段数": total,
            "在跑": busy, "GPU": ",".join(t["卡"] for t in tasks) or None, "任务": tasks,
            "段列表": segments_view(kind, scene, run, seg, stop, busy, recall),
            "下一步": None if busy else next_action(kind, seg, stop, recall),
            "能继续": not busy and seg < total and os.path.isdir(R1),
            "人": people_decisions.load(R1),
            "人表": people_decisions.table(R1),       # 摆人之后才有: 每个编号的去向和可疑原因(挑人卡片上也显示)
            "图": {"地基": os.path.exists(f"{R1}/02_ground_check.jpg"),
                   "体检": os.path.exists(f"{body}/03_check.jpg"),
                   "跟踪视频": os.path.exists(f"{body}/03_track.mp4")}}


# ---------------------------------------------------------------- 本轮页的"一步步"视图(2026-09-24)
# 每道闸摆什么图/视频。键是 /run/media?what= 的白名单, 路径只在这里拼, 不收任意路径。
def media_paths(kind, scene, run):
    R1 = r1_of(kind, scene, run); body = people_decisions.auto_dir(R1)
    mo = f"{R1}/12_motion/{os.path.basename(os.path.dirname(R1))}"     # motion 步的产物, 文件名带场景名(见 motion_export.py)
    return {"ground":    (f"{R1}/02_ground_check.jpg", "地基核对图: 抽帧 + 点云俯视(相机轨迹) + 侧视(地面线)"),
            "sheet":     (f"{R1}/05_review_sheet.jpg", "候选联络图(改决定在候选页)"),
            "check":     (f"{body}/03_check.jpg", "人体跟踪体检图"),
            "track":     (f"{body}/03_track.mp4", "人体跟踪视频"),
            "overlay":   (f"{body}/05_mesh_overlay.mp4", "人体网格叠回原片"),
            "recall2d":  (f"{R1}/95_recall_check_2d.jpg", "召回核对图"),
            "render":    (f"{R1}/08_render_back_phy.jpg", "渲染回图: 实拍 | 重建 | 叠加"),
            "people4d":  (f"{R1}/09_people_check.mp4", "带人的验收视频: 原片 | 重建+人 | 叠加"),
            "people4d_prev": (f"{R1}/09_people_check_prev.mp4", "上一版带人验收视频(重跑前的, 对比用)"),
            "blender":   (f"{mo}_blender.mp4", "Blender 里从原视频的视角渲的视频(场景 + 人, 用的就是交付的那份数据)"),
            # 交付的文件: 页面上是下载链接, 不是图
            "fbx_scene_people": (f"{mo}_scene_people.fbx", "场景 + 人 + 相机的 FBX(人带骨骼动画, 相机带每帧的位姿和焦距)"),
            "fbx_scene": (f"{mo}_scene.fbx", "只有场景的 FBX")}

GATE_MEDIA = {"check_ground": ["ground"], "check_picks": ["sheet", "check", "track"], "check_scale": ["overlay"],
              "check_recall": ["recall2d"],
              "check_scene": ["render", "people4d", "people4d_prev", "blender", "fbx_scene_people", "fbx_scene"]}


_PIPE_MOD = {}
def _pipe(kind):
    """pipeline 模块本身(只读它的表: STEPS / SCENE_TAG), 别自己抄一份。"""
    if kind not in _PIPE_MOD:
        spec = importlib.util.spec_from_file_location(f"_{kind}_pipeline_for_run_ctl", PIPES[kind])
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        _PIPE_MOD[kind] = m
    return _PIPE_MOD[kind]


def _step_desc(kind):
    """步骤说明问 pipeline 的 STEPS(和 --list 打的是同一份)。"""
    m = _pipe(kind)
    return {k: d for k, _e, d, _f in (m.STEPS if kind == "outdoor" else m.STEPS_OBJ)}


def tag_of(kind, scene, run):
    """生成的 mesh 按这个标签存目录(sam-3d-objects/my_images/<标签>_*)。公式和两个 pipeline 的 Ctx 一样。"""
    return f"{_pipe(kind).SCENE_TAG.get(scene, scene[:3])}_{run}"


def _tag_clash(kind, scene, run):
    """新轮次的标签和盘上已有的哪一轮撞了(室内外共用 my_images, 两边都查)。没撞返回 None。"""
    mine = tag_of(kind, scene, run)
    for k, (_ds, res) in ROOTS.items():
        for sc in sorted(os.listdir(res)) if os.path.isdir(res) else []:
            if (k, sc) != (kind, scene) and os.path.isdir(f"{res}/{sc}/{run}") and tag_of(k, sc, run) == mine:
                return f"{sc}/{run}"
    return None


def _latest_log(R1):
    logs = sorted(glob.glob(f"{R1}/logs/drive_*.log"))
    if not logs:
        return None, None
    try:
        first = open(logs[-1], errors="ignore").readline()
        t0 = time.mktime(time.strptime(first.split("]", 1)[1].strip(), "%Y-%m-%d %H:%M:%S"))
    except (IndexError, ValueError):
        t0 = os.path.getmtime(logs[-1])
    return logs[-1], t0


def _timing(R1):
    """logs/timing.tsv 里每步最后一次的耗时(秒)。"""
    out = {}
    try:
        for ln in open(f"{R1}/logs/timing.tsv", errors="ignore").read().splitlines()[1:]:
            c = ln.split("\t")
            if len(c) >= 7:
                out[c[0]] = float(c[4])
    except (OSError, ValueError):
        pass
    return out


def gate_report(R1, gate, total):
    """驱动打在日志里的第 gate 道闸的报告(最新那次), 去掉路径和 3D 提示 —— 那些在页面上另有地方。"""
    # 总段数不认: 室外 10-08 从六段变五段, 老日志里写的是"闸 3/6", 报告照样要找得到
    head = re.compile(rf"闸 {gate}/\d+ — ")
    for p in sorted(glob.glob(f"{R1}/logs/drive_*.log"), reverse=True):
        lines = open(p, errors="ignore").read().splitlines()
        idx = [i for i, l in enumerate(lines) if head.match(l)]
        if not idx:
            continue
        out = []
        for l in lines[idx[-1] + 2:]:
            if l.startswith("★") or l.strip() == "该看的图:":
                break
            if l.strip() and not l.startswith("[3D 页]") and not l.strip().startswith("3D: open"):
                out.append(l.strip())
        return out
    return []


def _gate_extra(check, kind, scene, run):
    """闸口要单独摆出来的结构化数字(报告原文之外)。"""
    R1 = r1_of(kind, scene, run)
    if check == "check_picks":
        # ★ 按**现在盘上**的决策文件算, 不用驱动日志里那份闸口报告 —— 那是到闸时的快照,
        #   人之后在候选页改的决定不在里面(nfl1 上就是: 报告写 25 个, 实际只剩 1 个)。
        #   语义和 review_gate 落地一样: 整词作废 = 这个词所有行都不生成; 共用 = 这个词只出一个 mesh。
        try:
            rows = json.load(open(f"{R1}/05_review.json")).get("候选") or []
        except (OSError, ValueError):
            return {}
        dead = sorted({r["词"] for r in rows if r.get("决定") == "drop-word"})
        live = [r for r in rows if r.get("决定") not in ("drop", "drop-word") and r["词"] not in dead]
        shared = {r["词"] for r in rows if r.get("决定") == "shared"}
        per = {}
        for r in live:
            per[r["词"]] = 1 if r["词"] in shared else per.get(r["词"], 0) + 1
        return {"候选决定": {"候选": len(rows), "会生成": sum(per.values()), "按词": per, "作废词": dead}}
    if check == "check_scale":
        try:
            w = json.load(open(f"{R1}/02_da3/world_scale.json"))
        except (OSError, ValueError):
            return {}
        per = (w.get("人") or {}).get("每人") or {}
        return {"尺度": {"k": w.get("k"), "来源": w.get("来源") or "家具",
                         "每人": {i: v.get("场景深度比模型") for i, v in per.items()}}}
    if check == "check_scene":
        try:
            sk = json.load(open(f"{R1}/07_scene_skipped.json")).get("skipped", [])
        except (OSError, ValueError):
            sk = []
        out = {"跳过的物体": [{"词": x.get("word"), "实例": x.get("obj_id"), "原因": x.get("why"),
                               "尺寸": x.get("ext")} for x in sk]}
        # ★ "人"那张表: 每个跟踪编号最后算谁(并进了谁 / 不算人)、谁可疑。以前只打在摆人那步的日志里,
        #   闸口上看不到(office 人 7 是搭在椅背上的一件外套, 流水线标了"深度离群", 没人看见)。
        #   按**现在盘上**的两个文件算(05_people.json + 10_people.json), 不是到闸时的快照: 勾的人改了它会标"过期"
        t = people_decisions.table(R1)
        if t:
            out["人表"] = t
        return out
    return {}


def segments_view(kind, scene, run, seg, stop, busy, recall):
    """每一段的状态、步骤、闸口报告、产出。状态: done / waiting(停在这一段的闸)
    / running / stopped(这一段开过头但没跑完, 多半失败了) / pending。"""
    R1 = r1_of(kind, scene, run)
    log, t0 = _latest_log(R1)
    # 在跑哪几步: 并行时看调度写的标记; 没有标记(PARALLEL=0 或老轮次)照旧看驱动日志里最后出现的那一步
    curs = set()
    if busy:
        curs = set(running_steps(R1))
        if not curs and log:
            hits = re.findall(r"^== (\S+)  \[", open(log, errors="ignore").read(), re.M)
            curs = set(hits[-1:])
    tm, desc, media = _timing(R1), _step_desc(kind), media_paths(kind, scene, run)
    out = []
    for i, (name, steps, check) in enumerate(_segments(kind)):
        if i < seg:
            st = "done"
        elif i == seg:
            st = "waiting" if stop else "running" if busy else "stopped" if os.path.exists(f"{R1}/logs/.stamp_{steps[0]}") and seg else "pending"
        else:
            st = "pending"
        rows = []
        for k in steps:
            stamp = f"{R1}/logs/.stamp_{k}"
            if st in ("done", "waiting"):
                s_ = "done"
            elif st == "running":
                s_ = "running" if k in curs else "done" if os.path.exists(stamp) and t0 and os.path.getmtime(stamp) >= t0 else "pending"
            else:
                s_ = "pending"
            rows.append({"步骤": k, "说明": desc.get(k, ""), "状态": s_,
                         "耗时": tm.get(k) if s_ == "done" else None})
        gate = ("跳过" if check == "check_recall" and recall != "1" and (st == "done" or i > seg or st != "waiting")
                else "等你" if st == "waiting" else "通过" if st == "done" else None)
        item = {"段": i + 1, "名": name, "状态": st, "闸": gate, "步骤": rows,
                "报告": gate_report(R1, i + 1, len(_segments(kind))) if st in ("done", "waiting") else [],
                "产出": [{"what": w, "说明": media[w][1], "视频": media[w][0].endswith(".mp4"),
                          # 不是图也不是视频的(FBX): 页面上给下载链接, 这里带上文件名和大小
                          "下载": os.path.basename(media[w][0]) if media[w][0].endswith(".fbx") else None,
                          "大小MB": round(os.path.getsize(media[w][0]) / 1e6, 1)}
                         for w in GATE_MEDIA.get(check, []) if os.path.exists(media[w][0])]}
        if st in ("done", "waiting"):
            item.update(_gate_extra(check, kind, scene, run))
        if st == "running" and t0:
            done_t = [os.path.getmtime(f"{R1}/logs/.stamp_{r['步骤']}") for r in rows if r["状态"] == "done"]
            item["已用秒"] = round(time.time() - max([t0] + done_t))
        out.append(item)
    return out


def next_action(kind, seg, stop, recall):
    """Continue 之后会跑哪一段、停在哪道闸(召回关时第 4 道闸不停, 顺延到下一道)。"""
    segs = _segments(kind)
    i = seg + 1 if stop else seg
    if i >= len(segs):
        return None
    j = i
    while j < len(segs) - 1 and segs[j][2] == "check_recall" and recall != "1":
        j += 1
    steps = segs[i][1] + [x for k in range(i + 1, j + 1) for x in segs[k][1]]
    if recall != "1":       # 召回关时这五步只打一行"跳过", 不列给人看
        steps = [x for x in steps if x not in ("detect", "recall", "recall_mask", "recall_gen", "recall_merge")]
    return {"段": i + 1, "名": segs[i][0], "步骤": steps, "停在": j + 1}


def set_config(kind, scene, run, changes):
    _check_name(scene, run)
    if running(scene, run):
        raise ValueError("驱动正在跑这一轮, 等它停在闸口再改")
    R1 = r1_of(kind, scene, run)
    seg, stop = _drive_state(R1)
    return _common.update_run_config(R1, kind, {k: str(v) for k, v in changes.items()}, seg + (1 if stop else 0))


def set_people(kind, scene, run, changes):
    _check_name(scene, run)
    if running(scene, run):
        raise ValueError("驱动正在跑这一轮(可能正在按旧决定出网格), 等它停在闸口再改")
    return people_decisions.save_decisions(r1_of(kind, scene, run), {int(k): v for k, v in changes.items()})


def go(kind, scene, run):
    """开跑 / 继续 = 替人敲一次 drive_go.sh <场景> <轮次> [--indoor]。它自己会把驱动挂进 tmux 就返回。"""
    _check_name(scene, run)
    if running(scene, run):
        raise ValueError("这一轮的驱动已经在跑了")
    if not os.path.isdir(r1_of(kind, scene, run)):
        raise ValueError(f"没有这个轮次: {r1_of(kind, scene, run)}")
    # ★ 网页进程自己的 CUDA_VISIBLE_DEVICES 不传下去: 传了所有轮次都挤在那一张卡上。
    #   不给卡号, 每个任务开始时自己领一张空卡(_common.GpuSlot)。
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    r = subprocess.run(["bash", DRIVE_GO, scene, run] + (["--indoor"] if kind == "indoor" else []),
                       capture_output=True, text=True, timeout=60, env=env)
    if r.returncode != 0:
        raise ValueError((r.stdout + r.stderr).strip() or f"drive_go.sh 退出码 {r.returncode}")
    return (r.stdout + r.stderr).strip()


def new_run(kind, scene, run, people, recall):
    """启动页: 建轮次目录, 写本轮配置, 开跑第 1 段。"""
    if kind not in ROOTS:
        raise ValueError(f"类型只能是 {sorted(ROOTS)}")
    _check_name(scene, run)
    rows = {x["场景"]: x for x in datasets()[kind]}
    if scene not in rows:
        raise ValueError(f"{ROOTS[kind][0]} 下没有场景 {scene}(类型选对了吗?)")
    R1 = r1_of(kind, scene, run)
    if os.path.isdir(R1) and os.listdir(R1):
        raise ValueError(f"轮次已经存在: {R1}")
    # ★ 标签默认取场景名前 3 个字母(nfl / nfl1 都是 nfl_r1), 撞了生成的 mesh 会互相覆盖, 而且不报错
    clash = _tag_clash(kind, scene, run)
    if clash:
        raise ValueError(f"标签 {tag_of(kind, scene, run)} 和已有的 {clash} 撞了, 生成的 mesh 会互相覆盖。"
                         f"在 pipeline 的 SCENE_TAG 里给 {scene} 配一个自己的标签, 或者换个轮次名")
    if rows[scene].get("新视频"):
        # 散放的视频: 建场景文件夹, 视频以 video.mp4 放进去(抽帧那步认这个名字)。
        # 硬链接: 不占第二份空间, 也不动原文件; 跨盘链不了就复制
        ds = ROOTS[kind][0]
        os.makedirs(f"{ds}/{scene}", exist_ok=True)
        try:
            os.link(f"{ds}/{rows[scene]['新视频']}", f"{ds}/{scene}/video.mp4")
        except OSError:
            import shutil
            shutil.copy2(f"{ds}/{rows[scene]['新视频']}", f"{ds}/{scene}/video.mp4")
    os.makedirs(f"{R1}/logs", exist_ok=True)
    _common.update_run_config(R1, kind, {"PEOPLE": str(people), "RECALL": str(recall)}, 0)
    go(kind, scene, run)
    return R1
