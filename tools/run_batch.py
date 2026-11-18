#!/usr/bin/env python
"""批量跑一个文件夹里的视频: 先自动判室内 / 室外, 再一个一个从头跑到底(闸口自动过), 最后出一张结果表。

  python tools/run_batch.py <视频文件夹> [--jobs 2] [--classify-only] [环境变量=值 ...]

  <视频文件夹>/*.mp4|mov|mkv|avi|m4v       自动判室内外(tools/scene_kind.py, Qwen3-VL 看 6 帧投票)
  <视频文件夹>/indoor/*、outdoor/*          放在这两个子文件夹里的不判, 按文件夹走(判错了就挪进去重跑)

每个视频:
  * 场景名 = 文件名(小写, 字母数字以外换成 _)。素材放进 data/<室内外>/dataset/<场景名>/video.mp4(硬链接, 不占空间)。
  * 轮次名默认 r1。★ 生成的模型按"场景名前 3 个字母_轮次"存目录, 撞了会互相覆盖而且不报错 ——
    撞了就自动换 r2、r3 ...; 这一批之内互相撞的也一样。
  * 结果目录已经有了: 跑完了(有产出)就跳过; 没跑完(中断了 / 上次失败)就接着跑。要从头重跑: 删掉那个结果目录。
  * 跑法 = tools/run_to_end.sh(按"有人"跑, 闸口全部自动过, 不作废任何词; 要作废词传 "DROP_WORDS=wall,door frame")。
  * 一段失败自动重跑这一段, 最多 2 次(生成物体偶尔显存不够, 重跑就过)。还不行就记失败, 不挡下一个。

--jobs N: 同时跑 N 个视频, 把卡平均分成 N 份(8 张卡、--jobs 2 = 0-3 和 4-7)。默认 1。
--classify-only: 只判室内外、出表, 不跑。
其余 KEY=VAL 原样带给每个视频(例: MAX_GPUS=4、CPU_THREADS=16、"DROP_WORDS=window frame,wall")。

输出在 <视频文件夹>/batch_<时间>/: 判室内外的结果 kinds.json、每个视频的日志、结果表 summary.tsv。
用 sam3d 环境的 python 跑: ~/miniforge3/envs/sam3d/bin/python tools/run_batch.py ...
"""
import argparse, filecmp, json, os, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/tools")
import run_ctl  # noqa: E402  (ROOTS / tag_of / NAME, 和网页开新一轮同一套规则)

PY = os.path.expanduser("~/miniforge3/envs/sam3d/bin/python")
EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v")


def find_videos(d):
    """[(视频路径, 指定的类型或 None)]。"""
    out = []
    for sub, kind in (("", None), ("indoor", "indoor"), ("outdoor", "outdoor")):
        p = os.path.join(d, sub)
        if os.path.isdir(p):
            out += [(os.path.join(p, f), kind) for f in sorted(os.listdir(p))
                    if f.lower().endswith(EXTS) and os.path.isfile(os.path.join(p, f))]
    return out


def scene_name(path):
    s = re.sub(r"[^a-z0-9]+", "_", os.path.splitext(os.path.basename(path))[0].lower()).strip("_")
    return s if s and run_ctl.NAME.match(s) else "video"


def place(kind, scene, video):
    """素材放进 dataset/<场景>/video.mp4。同名场景里已经是别的视频 -> 场景名加 _2、_3 ...。返回最终场景名。"""
    ds, base, i = run_ctl.ROOTS[kind][0], scene, 1
    while True:
        dst = f"{ds}/{scene}/video.mp4"
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.link(video, dst)
            except OSError:
                import shutil
                shutil.copy2(video, dst)
            return scene
        if os.path.samefile(dst, video) or filecmp.cmp(dst, video, shallow=False):
            return scene
        i += 1
        scene = f"{base}_{i}"


def pick_run(kind, scene, want, taken):
    """轮次名: 结果目录已有 -> (想要的那个, "跳过" 跑完了 / "续跑" 没跑完); 否则从 want 起找一个标签不撞的, ""。
    taken = 这一批已经占了的标签。"""
    m = re.match(r"^(.*?)(\d+)$", want)
    stem, n = (m.group(1), int(m.group(2))) if m else (want + "_", 1)
    # 以前跑过的(上一批可能因为撞名用了 r2、r3): 有跑完的就跳过, 否则接着跑最后那个没跑完的
    d = os.path.dirname(run_ctl.r1_of(kind, scene, want))
    old = sorted((int(x.group(2)), x.group(0)) for x in
                 (re.match(rf"^({re.escape(stem)})(\d+)$", r) for r in (os.listdir(d) if os.path.isdir(d) else []))
                 if x and int(x.group(2)) >= n and os.listdir(f"{d}/{x.group(0)}"))
    for _, r in old:
        if outcome(kind, scene, r):
            return r, "跳过"
    if old:
        return old[-1][1], "续跑"
    run = want
    while (os.path.exists(run_ctl.r1_of(kind, scene, run)) or run_ctl._tag_clash(kind, scene, run)
           or run_ctl.tag_of(kind, scene, run) in taken):
        n += 1
        run = f"{stem}{n}"
    return run, ""


def outcome(kind, scene, run):
    R1 = run_ctl.r1_of(kind, scene, run)
    fbx = f"{R1}/12_motion/{scene}_scene_people.fbx"
    if os.path.exists(fbx):
        return fbx
    if os.path.exists(f"{R1}/07_scene_phy.glb"):
        return f"{R1}/07_scene_phy.glb (没有检测到人, 不出 FBX)"
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--run", default="r1", help="轮次名(默认 r1, 撞了自动往后加)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--classify-only", action="store_true")
    ap.add_argument("env", nargs="*", help="KEY=VAL, 带给每个视频")
    a = ap.parse_args()
    for kv in a.env:
        if "=" not in kv:
            sys.exit(f"多余的参数 {kv!r}: 只认 KEY=VAL")
    d = os.path.abspath(a.dir)
    vids = find_videos(d)
    if not vids:
        sys.exit(f"{d} 里没有视频({'/'.join(EXTS)}), indoor/ outdoor/ 子文件夹里也没有")
    out = f"{d}/batch_{time.strftime('%m%d_%H%M%S')}"
    os.makedirs(out)
    print(f"[批量] {len(vids)} 个视频, 输出在 {out}", flush=True)

    # 1. 判室内外(放在 indoor/ outdoor/ 里的不判)
    kinds = {v: {"类型": k, "拿不准": False, "来源": "文件夹"} for v, k in vids if k}
    todo = [v for v, k in vids if not k]
    if todo:
        print(f"[判室内外] {len(todo)} 个", flush=True)
        kf = f"{out}/kinds.json"
        r = subprocess.run([PY, f"{SAM3D}/tools/scene_kind.py", *todo, "--out", kf])
        if r.returncode != 0 or not os.path.exists(kf):
            sys.exit("★ 判室内外失败, 看上面的报错。可以把视频手动放进 indoor/ outdoor/ 子文件夹绕开")
        for v, x in json.load(open(kf)).items():
            kinds[v] = dict(x, 来源="模型")
    json.dump(kinds, open(f"{out}/kinds.json", "w"), ensure_ascii=False, indent=1)

    # 2. 起名(全部先定好, 并行时不会抢同一个名字)
    jobs, taken = [], set()
    for v, _ in vids:
        k = kinds[v]["类型"]
        scene = place(k, scene_name(v), v) if not a.classify_only else scene_name(v)
        run, skip = pick_run(k, scene, a.run, taken)
        taken.add(run_ctl.tag_of(k, scene, run))
        jobs.append(dict(视频=v, 类型=k, 拿不准=kinds[v].get("拿不准"), 场景=scene, 轮次=run, 跳过=skip))
    for j in jobs:
        print(f"  {j['类型']:8s}{' (拿不准)' if j['拿不准'] else '':8s} {j['场景']}/{j['轮次']}"
              f"{'  已跑完, 跳过' if j['跳过'] == '跳过' else '  上次没跑完, 接着跑' if j['跳过'] else ''}  <- {os.path.basename(j['视频'])}", flush=True)
    if a.classify_only:
        print(f"[只判不跑] 表在 {out}/kinds.json。判错的挪进 indoor/ 或 outdoor/ 子文件夹再跑")
        return

    # 3. 跑。卡分成 jobs 份
    pool = os.environ.get("GPU_POOL") or ",".join(
        str(i) for i in range(len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())))
    gpus = [g for g in pool.split(",") if g]
    n = max(1, min(a.jobs, len(gpus)))
    pools = [",".join(gpus[i::n]) for i in range(n)]
    free, lock = list(pools), threading.Lock()
    base_env = dict(os.environ, **dict(kv.split("=", 1) for kv in a.env))
    base_env.setdefault("DROP_WORDS", "")          # run_to_end.sh 不给就默认作废五个橄榄球的词, 批量不要
    rows = []

    def one(j):
        if j["跳过"] == "跳过":
            return dict(j, 结果="已跑完, 跳过", 用时="", 产出=outcome(j["类型"], j["场景"], j["轮次"]))
        with lock:
            p = free.pop()
        env = dict(base_env)
        if j["跳过"] == "续跑":
            env["RESUME"] = "1"
        if n > 1:
            env["GPU_POOL"] = p
            env.setdefault("MAX_GPUS", str(len(p.split(","))))
        log = f"{out}/{j['场景']}_{j['轮次']}.log"
        cmd = ["bash", f"{SAM3D}/tools/run_to_end.sh", j["场景"], j["轮次"]] + (["--indoor"] if j["类型"] == "indoor" else [])
        print(f"[开始] {j['场景']}/{j['轮次']} ({j['类型']}{', 卡 ' + p if n > 1 else ''})  日志 {log}", flush=True)
        t0 = time.time()
        with open(log, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        with lock:
            free.append(p)
        mins = f"{(time.time() - t0) / 60:.0f} 分钟"
        res = outcome(j["类型"], j["场景"], j["轮次"])
        ok = "成功" if rc == 0 else f"失败(退出码 {rc}, 看 {log} 和结果目录的 logs/)"
        print(f"[{'完成' if rc == 0 else '失败'}] {j['场景']}/{j['轮次']}  {mins}  {res}", flush=True)
        return dict(j, 结果=ok, 用时=mins, 产出=res)

    with ThreadPoolExecutor(n) as ex:
        rows = list(ex.map(one, jobs))

    # 4. 结果表
    cols = ["视频", "类型", "拿不准", "场景", "轮次", "结果", "用时", "产出"]
    with open(f"{out}/summary.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    good = sum(r["结果"] == "成功" for r in rows)
    print(f"\n[批量结束] 成功 {good} / {len(rows)}。结果表 {out}/summary.tsv")
    for r in rows:
        print(f"  {r['结果'][:2]:4s} {r['类型']:8s} {r['场景']}/{r['轮次']}  {r['产出']}")


if __name__ == "__main__":
    main()
