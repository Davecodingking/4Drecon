# 并行调度 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 同一轮里能同时跑的步骤同时跑、各用一张卡, nfl1 的计算时间从约 25 分钟降到 10～12 分钟。

**Architecture:** 流水线收到多个步骤时, 按 `DEPS` 表为每个"依赖已满足"的步骤起一个子进程(`--only <步骤> --child`)。吃显存的任务开始时用 `GpuSlot` 领一张空卡, 跑完还。最慢的两步(track、出人网格)在步骤内部把同一个脚本起 K 份, 每份算一部分。闸口、决策重跑规则、模型代码都不动。

**Tech Stack:** Python 标准库(`subprocess` / `fcntl` / `threading`), numpy + PIL(只在合并脚本里), bash。测试是带 `assert` 的脚本, 没有 pytest。

**Spec:** `docs/superpowers/specs/2026-09-29-parallel-scheduling-design.md`

## Global Constraints

- 仓库 `/home/ubuntu/sam3d`, 分支 `parallel-sched`, 回退点标签 `before-parallel-sched`。
- Python: `~/miniforge3/envs/sam3d/bin/python`(下文写作 `$PY`)。测试一个个跑: `$PY tools/tests/<文件>.py`; 全量: `bash tools/tests/run.sh`。
- 锁文件名一个字不改: `/tmp/outdoor_pipeline_gpu<卡号>.lock`。`pipeline/gpulock.sh` 不动。
- 三个配置, 默认值只写在 `pipeline/_common.py`: `MAX_GPUS` 默认 `8`; `GPU_POOL` 默认全部卡; `PARALLEL` 默认 `1`, 设 `0` = 和改动前完全一样一步一步跑。
- `pipeline/_common.py`、`pipeline/sched.py`、`pipeline/fanout.py` 只用标准库。要 numpy/PIL 的放进单独的脚本, 用 `$PY` 当子进程跑。
- 闸口不变; `drive.py` 的分段表、决策落地后重跑哪些步骤, 都不改。
- 模型代码不动(`sam-body4d/` 里上游的文件、`sam3/`、`sam-3d-objects/`)。我们自己的 `body4d_mesh_5090.py` 也不用改。
- 不覆盖已有轮次和视频: 实跑验证一律新开轮次(`r2` / `r3`)。
- 不许 `pkill -f`。重启网页只用 `tmux kill-session -t gate_ui` 再起。
- 代码注释用中文, 写"为什么", 风格照周围的代码。
- 每个任务最后提交一次, 提交信息结尾带一行: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`。
- 直连不了 HuggingFace, 流水线默认 `HF_HUB_OFFLINE=1`, 不要改。

## Review Focus

规格没明说、但用的人一定会碰到的五种情况。每条都在负责那段代码的任务里有测试。

1. **进程被杀(OOM / kill -9)时手里还拿着卡。** 锁由内核自动释放, 但 `.gpu_<卡>.json` 和 `.running_<步骤>` 会留在盘上。网页不能因此一直显示"在跑"。→ Task 5 `test_stale_markers_are_ignored`。
2. **`--only` 跳着选步骤**(决策落地重跑的就是 `relabel,pick,verify,override,gen,ratio,boxes,solo`, 中间隔着没选的 `scale`)。solo 仍然要等 boxes 跑完。→ Task 2 `test_waits_through_unselected_steps`。
3. **只有一张卡可用**(用户钉了 `CUDA_VISIBLE_DEVICES=3`, 或别的卡都被人占着)。并行的步骤要排队轮流用这一张, 不能死锁, 不能报错。→ Task 3 `test_one_card_is_shared_in_turn`。
4. **没有人 / 只有 1 个人 / 要跟踪的词是 0 个。** 拆分退化成 1 份或直接跳过, 不能崩。→ Task 6 `test_split_small_inputs`。
5. **卡全被别人占着。** 任务等待并打印"排队中", 有卡空出来后继续; 不是报错退出。→ Task 3 `test_waits_until_a_card_frees_up`。

---

## File Structure

| 文件 | 负责什么 |
|---|---|
| `pipeline/sched.py`(新) | 按依赖表并行跑步骤: `check_graph` / `ancestors` / `run_graph` / `run_children` |
| `pipeline/_common.py` | 领卡: `GpuSlot` / `gpus_free_now` / `gpu_pool` / `card_busy` / `max_gpus` / `parallel_on` / `lock_path_of` |
| `pipeline/fanout.py`(新) | 步骤内部拆分: `split` / `how_many` / `run_workers` / `track_words` / `merge_reports` |
| `pipeline/mesh_merge.py`(新) | 出人网格几份并回一个目录(numpy + PIL, 当脚本跑) |
| `pipeline/outdoor_pipeline.py` | 主循环接 `sched`; `s_track` 拆分; 补 `DEPS`; `--child` / `--dump-deps` |
| `pipeline/indoor_pipeline.py` | 同上(`o_track`) |
| `pipeline/body4d_pipeline.py` | 主循环用 `GpuSlot`; `s_mesh` 拆分 |
| `data/outdoor/scripts/track_all.py` | `ONLY_WORDS` / `TRACK_REPORT` 两个环境变量 |
| `tools/drive_go.sh` | 不再开跑时挑卡 |
| `tools/run_ctl.py` | `running_steps` / `gpu_tasks`; `state` 和 `segments_view` 支持多个在跑的步骤 |
| `tools/run_page.py` | 顶上那行字列出多个步骤和卡 |
| `tools/watch_run.py` | `parse_drive` 认 `[并行] 开始` |
| `tools/compare_runs.py`(新) | 两轮的 track / 出人网格结果对比 |
| `tools/tests/test_sched.py`、`test_gpu_slot.py`、`test_fanout.py`、`test_mesh_merge.py`、`test_watch_parse.py`(新) | 测试 |
| `tools/tests/test_gpu_alloc.py`、`test_run_ctl.py`、`tools/tests/run.sh` | 更新 |

---

### Task 1: 依赖表检查 + 核对并行的步骤

**Files:**
- Create: `pipeline/sched.py`
- Create: `tools/tests/test_sched.py`
- Modify: `pipeline/outdoor_pipeline.py`(`DEPS`、`main` 加 `--dump-deps`、`GPU_STEPS` 挪到模块级)
- Modify: `pipeline/indoor_pipeline.py`(同上)
- Modify: `pipeline/body4d_pipeline.py`(`main` 加 `--dump-deps`)
- Modify: `docs/superpowers/specs/2026-09-29-parallel-scheduling-design.md`(加附录: 核对结果)
- Modify: `tools/tests/run.sh`

**Interfaces:**
- Produces: `sched.check_graph(keys: list[str], deps: dict[str, list[str]]) -> None`(不合规矩抛 `ValueError`); `sched.ancestors(deps) -> dict[str, set[str]]`。
- Produces: 三条流水线的 `--dump-deps`, 打印一行 JSON: `{"steps": [...], "deps": {...}, "gpu": [...]}`。

- [ ] **Step 1: 写失败的测试**

`tools/tests/test_sched.py`:

```python
#!/usr/bin/env python
"""按依赖表并行跑步骤(pipeline/sched.py)。不碰显卡: 步骤是假的。
跑: $PY tools/tests/test_sched.py"""
import json, os, subprocess, sys, tempfile, time

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import sched

KEYS = ["frames", "vlm", "da3", "up", "track", "boxes", "scale", "solo"]
DEPS = {"vlm": ["frames"], "da3": ["frames"], "up": ["da3"], "track": ["frames", "vlm"],
        "boxes": ["track", "up"], "scale": ["boxes"], "solo": ["scale"]}


def _raises(fn, *a):
    try:
        fn(*a)
    except ValueError as e:
        return str(e)
    raise AssertionError("该报错没报错")


def test_check_graph_accepts_a_good_table():
    sched.check_graph(KEYS, DEPS)


def test_check_graph_rejects_unknown_names():
    assert "walls" in _raises(sched.check_graph, KEYS, dict(DEPS, boxes=["walls"]))
    assert "walls" in _raises(sched.check_graph, KEYS, dict(DEPS, walls=["frames"]))


def test_check_graph_rejects_backward_edges():
    """依赖排在后面 = 顺序错或者成环。步骤表的顺序就是 PARALLEL=0 时的执行顺序。"""
    assert "solo" in _raises(sched.check_graph, KEYS, dict(DEPS, up=["solo"]))
    assert "up" in _raises(sched.check_graph, KEYS, dict(DEPS, up=["up"]))


def test_ancestors_are_transitive():
    a = sched.ancestors(DEPS)
    assert a["solo"] == {"scale", "boxes", "track", "up", "da3", "vlm", "frames"}, a["solo"]
    assert a["frames"] == set()


def test_real_tables_pass():
    """三条流水线自己的步骤表和依赖表。依赖表以前只用来判过期, 现在还决定谁先谁后, 不能有错。"""
    PY = sys.executable
    for cmd in ([f"{SAM3D}/pipeline/outdoor_pipeline.py"],
                [f"{SAM3D}/pipeline/indoor_pipeline.py", "--chain", "obj"],
                [f"{SAM3D}/pipeline/indoor_pipeline.py", "--chain", "lm"],
                [f"{SAM3D}/pipeline/body4d_pipeline.py"]):
        r = subprocess.run([PY] + cmd + ["--dump-deps"], capture_output=True, text=True)
        assert r.returncode == 0, (cmd, r.stderr[-400:])
        d = json.loads(r.stdout.strip().splitlines()[-1])
        sched.check_graph(d["steps"], d["deps"])
        assert set(d["gpu"]) <= set(d["steps"]), (cmd, set(d["gpu"]) - set(d["steps"]))


def test_track_depends_on_vlm():
    """track 读 vlm 出的词表(01_vlm_vocab.json)。以前依赖表里没写, 因为两步不在同一段里没出过事。"""
    for cmd in ([f"{SAM3D}/pipeline/outdoor_pipeline.py"], [f"{SAM3D}/pipeline/indoor_pipeline.py", "--chain", "obj"]):
        r = subprocess.run([sys.executable] + cmd + ["--dump-deps"], capture_output=True, text=True)
        d = json.loads(r.stdout.strip().splitlines()[-1])
        assert "vlm" in d["deps"]["track"], (cmd, d["deps"]["track"])


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
```

- [ ] **Step 2: 跑测试, 确认失败**

Run: `cd ~/sam3d && $PY tools/tests/test_sched.py`
Expected: `ModuleNotFoundError: No module named 'sched'`... 注意标准库里也有一个 `sched`。因为 `sys.path.insert(0, pipeline)` 在前, 文件建好后导入的是我们的。现在文件还没有, 导入的是标准库的, 报 `AttributeError: module 'sched' has no attribute 'check_graph'`。两种报错都算"按预期失败"。

- [ ] **Step 3: 写 `pipeline/sched.py` 的前半部分**

```python
#!/usr/bin/env python
"""按依赖表并行跑步骤(2026-09-29)。

★ 为什么有这个文件
  三条流水线的主循环以前都是"按步骤表从上到下, 一步跑完再跑下一步", 整轮只用一张卡。
  nfl1 一段 5 秒的视频算 25 分钟, 8 张 3090 里 7 张闲着。
  现在: 一个步骤只要它依赖的步骤都跑完了就可以开始, 几个同时满足就同时开始, 每个步骤一个子进程。
  依据就是各流水线自己的 DEPS(以前只用来判"过期"), 不另外维护一张"谁和谁能并行"的表 ——
  多一张表就多一个会漏改的地方(分段表已经是第五个注册点了)。

★ 文件名和标准库的 sched 重名。三条流水线都把 pipeline/ 插在 sys.path 最前面, 导入的是这个。

只用标准库, 不依赖任何一条流水线。
"""
import os, subprocess, sys, time


def check_graph(keys, deps):
    """依赖表要和步骤表对得上: 名字都在步骤表里; 依赖的步骤在步骤表里排在前面。
    步骤表的顺序就是 PARALLEL=0 时的执行顺序, 往后指的依赖 = 顺序错或者成环。不合规矩抛 ValueError。"""
    idx = {k: i for i, k in enumerate(keys)}
    bad = sorted({x for k, us in deps.items() for x in [k, *us] if x not in idx})
    if bad:
        raise ValueError(f"依赖表里这些名字不在步骤表里: {bad}")
    back = sorted((k, u) for k, us in deps.items() for u in us if idx[u] >= idx[k])
    if back:
        raise ValueError(f"这些依赖在步骤表里排在了后面(步骤, 依赖): {back}")


def ancestors(deps):
    """{步骤: 它直接和间接依赖的全部步骤}。"""
    memo = {}

    def anc(k):
        if k not in memo:
            memo[k] = set()                     # 先占位: 就算表里有环也不会转不出来(check_graph 会另外报)
            for u in deps.get(k, []):
                memo[k] |= {u} | anc(u)
        return memo[k]

    return {k: anc(k) for k in set(deps) | {u for us in deps.values() for u in us}}
```

- [ ] **Step 4: 三条流水线加 `--dump-deps`, `GPU_STEPS` 挪到模块级, 补 `DEPS`**

`pipeline/outdoor_pipeline.py`:

1. `DEPS` 里把 `"track": ["frames"]` 改成 `"track": ["frames", "vlm"]`, 行尾加注释 `# track 读 vlm 出的词表 01_vlm_vocab.json`。
2. 在 `DEPS = {...}` 这张表的后面(`def do_check` 之前)加模块级的:

```python
# 吃显存的步骤: 开始之前先领一张空卡(_common.GpuSlot)。
# ★ people_track / people_mesh 故意不在这里: body4d_pipeline.py 自己在 track/mesh 上领卡, 外面再领一张就多占一张不用
GPU_STEPS = {"vlm", "vlm_sizes", "da3", "track", "verify", "gen", "solo", "boxes_wd", "detect",
             "recall_mask", "recall_gen"}
```

3. `main()` 里删掉原来函数内的那两行 `GPU_STEPS = {...}` 和它上面那行 `# ★ people_track / people_mesh 故意不在这里...` 注释。
4. `main()` 的参数里, `--check` 那行后面加:

```python
    ap.add_argument("--dump-deps", action="store_true",
                    help="把步骤表/依赖表/吃显存的步骤打成一行 JSON(调度自检和测试用)")
```

5. `keys = [s[0] for s in STEPS]` 这一行后面加:

```python
    if g.dump_deps:
        print(json.dumps(dict(steps=keys, deps=DEPS, gpu=sorted(GPU_STEPS)), ensure_ascii=False))
        return
```

`pipeline/indoor_pipeline.py`: 同样五处。`DEPS_OBJ` 里 track 的依赖加 `"vlm"`(先 `grep -n '"track":' pipeline/indoor_pipeline.py` 看它现在依赖谁, 原有的保留)。`GPU_STEPS` 的内容照抄 `main()` 里现有的那个集合(两条链的名字并在一起), 连同它上面的三段注释一起挪到模块级、放在 `CHAINS = {` 之前。`--dump-deps` 的分支放在 `STEPS, DEFAULT_END, ARTIFACTS, DEPS = CHAINS[g.chain]` 和 `keys = ...` 之后, 输出里的 `gpu` 只留这条链有的步骤:

```python
    if g.dump_deps:
        print(json.dumps(dict(steps=keys, deps=DEPS, gpu=sorted(GPU_STEPS & set(keys))), ensure_ascii=False))
        return
```

`pipeline/body4d_pipeline.py`: `GPU_STEPS` 已经在模块级。只加参数和分支(放在 `keys = [s[0] for s in STEPS]` 后面):

```python
    ap.add_argument("--dump-deps", action="store_true")
```

```python
    if g.dump_deps:
        print(json.dumps(dict(steps=keys, deps=DEPS, gpu=sorted(GPU_STEPS)), ensure_ascii=False))
        return
```

- [ ] **Step 5: 跑测试, 确认通过**

Run: `cd ~/sam3d && $PY tools/tests/test_sched.py`
Expected: 6 行 `ok test_...`。如果 `test_real_tables_pass` 报"依赖在步骤表里排在了后面", 说明某条流水线的表本身有错: 看报出来的那一对, 依赖关系是真的就调步骤表的顺序, 不是真的就从依赖表里删掉。改完重跑。

- [ ] **Step 6: 核对会同时跑的步骤, 确认它们不读写同一个文件**

对室外每一段, 列出"互相不是上下游"的步骤对, 逐对查它们的脚本读什么、写什么:

```bash
cd ~/sam3d && $PY - <<'EOF'
import json, subprocess, sys, os
sys.path.insert(0, "pipeline"); import sched
sys.path.insert(0, ".claude/skills/run-recon-pipeline")
d = json.loads(subprocess.run([sys.executable, "pipeline/outdoor_pipeline.py", "--dump-deps"],
                              capture_output=True, text=True).stdout.strip().splitlines()[-1])
anc = sched.ancestors(d["deps"])
SEG = [["frames","vlm","da3","up"], ["track","people_track","dedup","relabel","pick","verify","override"],
       ["gen","ratio","meshes","boxes","vlm_sizes","people_mesh","scale"],
       ["solo","boxes_wd","compare","detect","recall","recall_mask"],
       ["recall_gen","recall_merge","assemble","people","phys","people_glb","render"],
       ["building_scene","assemble_wd","phys_wd","render_wd"]]
for i, s in enumerate(SEG):
    pairs = [(a, b) for n, a in enumerate(s) for b in s[n+1:] if a not in anc.get(b, ()) and b not in anc.get(a, ())]
    print(f"第 {i+1} 段 会同时跑的步骤对:", pairs)
EOF
```

对打印出来的每一对 `(a, b)`: 在 `pipeline/outdoor_pipeline.py` 里找到 `s_a` 和 `s_b` 调的脚本, 在脚本里 `grep -n "open(\|json.dump\|np.save\|imwrite\|\.export(\|os.replace\|shutil"` 找它写的文件, `grep -n "json.load\|np.load\|imread\|glob"` 找它读的文件。判断规则:

| 情况 | 处理 |
|---|---|
| a 写的文件 b 要读 | 这是漏掉的依赖: 往 `DEPS["b"]` 里加 `"a"` |
| a 和 b 写同一个文件 | 往 `DEPS` 里加一条边让它们分先后(后一个依赖前一个, 按步骤表顺序) |
| 只是都读同一个文件 | 没问题 |

把每一对的结论写进规格文档末尾的新一节 `## 附录: 同时跑的步骤核对结果`, 一对一行: `步骤 a | 步骤 b | a 写什么 | b 写什么 | 结论`。室内(`--chain obj`)做同样的事, 分段表见 `drive.py` 的 `SEGMENTS["indoor"]`。

已经确认过、可以直接写进附录的:
- 不在 `GPU_STEPS` 里的步骤没有一个直接用 CUDA(`pick_for_sam3d.py` / `all_instances_boxes.py` / `sam3_cutouts.py` 是已经不用的老脚本)。渲染类的步骤(`render` / `people_glb` / `meshes`)用 EGL, 占显存很少。
- `people_decisions._save` 是原子替换, 别的步骤启动时读 `05_people.json` 不会读到半截。
- `scale` 会就地改深度和相机, 但它依赖本段所有其他步骤, 不会和谁同时跑。

- [ ] **Step 7: 核对后重跑测试, 加进 run.sh, 提交**

`tools/tests/run.sh` 第 5 行末尾追加 `; $PY test_sched.py`。

Run: `cd ~/sam3d && $PY tools/tests/test_sched.py && bash tools/tests/run.sh 2>&1 | tail -2`
Expected: 最后一行 `== tools/tests 全过 ==`

```bash
cd ~/sam3d && git add pipeline/sched.py pipeline/outdoor_pipeline.py pipeline/indoor_pipeline.py pipeline/body4d_pipeline.py tools/tests/test_sched.py tools/tests/run.sh docs/superpowers/ && git commit -q -m "调度: 依赖表检查 + 核对会同时跑的步骤, 补 track 对 vlm 的依赖

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: 调度核心 `run_graph` / `run_children`

**Files:**
- Modify: `pipeline/sched.py`
- Modify: `tools/tests/test_sched.py`

**Interfaces:**
- Consumes: `sched.ancestors`。
- Produces: `sched.run_graph(sel, deps, launch, max_parallel=8, poll=0.2, on_done=None) -> dict[str, int]`。`launch(步骤)` 返回一个有 `.poll()` 的对象(返回 `None` = 还在跑, `0` = 成功, 其他 = 失败)。返回值里只有启动过的步骤。
- Produces: `sched.run_children(sel, deps, make_cmd, log_dir, max_parallel=8, env=None, out=sys.stdout) -> dict[str, int]`。`make_cmd(步骤)` 返回命令列表。在跑的步骤记在 `<log_dir>/.running_<步骤>`(内容是子进程号)。

- [ ] **Step 1: 写失败的测试**

在 `tools/tests/test_sched.py` 的 `if __name__` 之前加:

```python
class _Fake:
    """假步骤: 跑 secs 秒, 然后以 rc 退出。记下起止时刻。"""
    log = []

    def __init__(self, name, secs=0.3, rc=0):
        self.name, self.t0, self.t1, self.rc = name, time.time(), time.time() + secs, rc

    def poll(self):
        if time.time() < self.t1:
            return None
        if self not in _Fake.log:
            _Fake.log.append(self)
        return self.rc


def _run(sel, deps=DEPS, fail=(), **kw):
    _Fake.log = []
    started = []
    def launch(k):
        started.append(k)
        return _Fake(k, rc=1 if k in fail else 0)
    rc = sched.run_graph(sel, deps, launch, poll=0.02, **kw)
    return rc, started, {f.name: (f.t0, f.t1) for f in _Fake.log}


def test_independent_steps_overlap():
    """vlm 和 da3 互不依赖: 同时跑。up 等 da3, track 等 vlm。"""
    rc, started, t = _run(["vlm", "da3", "up", "track"])
    assert set(rc) == {"vlm", "da3", "up", "track"} and not any(rc.values()), rc
    assert t["vlm"][0] < t["da3"][1] and t["da3"][0] < t["vlm"][1], "vlm 和 da3 应该有重叠"
    assert t["up"][0] >= t["da3"][1] and t["track"][0] >= t["vlm"][1]


def test_max_parallel_one_is_the_old_order():
    rc, started, t = _run(["vlm", "da3", "up", "track"], max_parallel=1)
    assert started == ["vlm", "da3", "up", "track"], started
    assert t["da3"][0] >= t["vlm"][1] and t["up"][0] >= t["da3"][1] and t["track"][0] >= t["up"][1]


def test_waits_through_unselected_steps():
    """决策落地后重跑的是 ...,boxes,solo, 中间隔着没选的 scale: solo 仍然要等 boxes。"""
    rc, started, t = _run(["boxes", "solo"])
    assert t["solo"][0] >= t["boxes"][1], t


def test_failure_stops_new_steps_but_not_running_ones():
    rc, started, t = _run(["vlm", "da3", "up", "track"], fail=("vlm",))
    assert rc["vlm"] == 1 and rc["da3"] == 0, rc            # da3 已经在跑, 让它跑完
    assert "up" not in started and "track" not in started, started


def test_steps_without_deps_entry_just_run():
    rc, started, t = _run(["frames", "vlm"], deps={})
    assert set(rc) == {"frames", "vlm"}


def test_run_children_blocks_and_markers():
    """子进程的输出整块写出, 不穿插; 在跑的步骤有 .running_<步骤>, 跑完删掉。"""
    import io
    with tempfile.TemporaryDirectory() as T:
        seen = {}
        def cmd(k):
            code = (f"import time, os; print('== {k}  [env]  说明', flush=True); time.sleep(0.4); "
                    f"print('    $ 命令 {k}'); print('   0.4s')")
            return [sys.executable, "-c", code]
        out = io.StringIO()
        import threading
        def peek():
            time.sleep(0.2); seen["mid"] = sorted(f for f in os.listdir(T) if f.startswith(".running_"))
        th = threading.Thread(target=peek); th.start()
        rc = sched.run_children(["vlm", "da3"], DEPS, cmd, T, out=out)
        th.join()
        assert rc == {"vlm": 0, "da3": 0}, rc
        assert seen["mid"] == [".running_da3", ".running_vlm"], seen
        assert not [f for f in os.listdir(T) if f.startswith(".running_") or f.startswith(".out_")], os.listdir(T)
        txt = out.getvalue()
        assert "[并行] 开始 vlm" in txt and "[并行] 开始 da3" in txt
        for k in ("vlm", "da3"):                             # 每个步骤的三行连在一起
            assert f"== {k}  [env]  说明\n    $ 命令 {k}\n   0.4s\n" in txt, txt


def test_run_children_reports_failure():
    import io
    with tempfile.TemporaryDirectory() as T:
        out = io.StringIO()
        rc = sched.run_children(["vlm"], {}, lambda k: [sys.executable, "-c", "import sys; print('坏了'); sys.exit(3)"],
                                T, out=out)
        assert rc == {"vlm": 3} and "[失败] 步骤 vlm 退出码 3" in out.getvalue() and "坏了" in out.getvalue()
```

- [ ] **Step 2: 跑测试, 确认失败**

Run: `cd ~/sam3d && $PY tools/tests/test_sched.py`
Expected: 前 6 个 `ok`, 然后 `AttributeError: module 'sched' has no attribute 'run_graph'`

- [ ] **Step 3: 实现**

`pipeline/sched.py` 末尾追加:

```python
def run_graph(sel, deps, launch, max_parallel=8, poll=0.2, on_done=None):
    """sel: 这次要跑的步骤(按步骤表顺序); launch(步骤) 返回有 poll() 的对象(None = 在跑, 0 = 成功)。
    规则: 一个步骤等它在 sel 里的全部上游跑完才开始 —— 隔着没选的步骤也算(决策落地后重跑
    ...,boxes,solo, 中间的 scale 没选, solo 仍然要等 boxes); 依赖表里不在 sel 里的当作早就跑完了。
    同时够条件的按 sel 的顺序启动, 所以 max_parallel=1 就是原来的顺序。
    有步骤失败: 不再启动新的, 正在跑的等它跑完(它们的结果是有效的)。
    返回 {步骤: 退出码}, 没启动的不在里面。"""
    anc = ancestors(deps)
    need = {k: anc.get(k, set()) & set(sel) for k in sel}
    done, running, rc = set(), {}, {}
    try:
        while True:
            for k, p in list(running.items()):
                r = p.poll()
                if r is None:
                    continue
                del running[k]
                rc[k] = r
                if r == 0:
                    done.add(k)
                if on_done:
                    on_done(k, r)
            if not any(rc.values()):
                for k in sel:
                    if len(running) >= max_parallel:
                        break
                    if k not in rc and k not in running and need[k] <= done:
                        running[k] = launch(k)
            if not running:
                return rc
            time.sleep(poll)
    finally:
        for p in running.values():              # 被打断(Ctrl-C / 异常): 别留下没人管的子进程
            if hasattr(p, "terminate") and p.poll() is None:
                p.terminate()


def run_children(sel, deps, make_cmd, log_dir, max_parallel=8, env=None, out=None):
    """每个步骤起一个子进程, make_cmd(步骤) 给命令。返回 {步骤: 退出码}。
    ★ 子进程的输出先写进 log_dir/.out_<步骤>.txt, 跑完整块打印: 几个步骤同时跑, 逐行透传会互相穿插,
      而驱动日志是按 "== 步骤 [环境] 说明" 开头、"   12.3s" 结尾一块一块读的
      (watch_run.parse_drive / run_ctl.segments_view), 穿插了就配不上对。
    ★ 在跑的步骤记在 log_dir/.running_<步骤>(内容是子进程号), 跑完删掉。输出要等跑完才进驱动日志,
      网页和看板靠这个文件知道"现在在跑哪几步"。"""
    out = out or sys.stdout
    os.makedirs(log_dir, exist_ok=True)
    files = {}

    def launch(k):
        files[k] = open(f"{log_dir}/.out_{k}.txt", "w")
        p = subprocess.Popen(make_cmd(k), stdout=files[k], stderr=subprocess.STDOUT, env=env)
        with open(f"{log_dir}/.running_{k}", "w") as f:
            f.write(str(p.pid))
        print(f"[并行] 开始 {k}", file=out, flush=True)
        return p

    def on_done(k, r):
        files.pop(k).close()
        txt = open(f"{log_dir}/.out_{k}.txt", errors="replace").read()
        out.write(txt if txt.endswith("\n") or not txt else txt + "\n")
        if r != 0:
            print(f"[失败] 步骤 {k} 退出码 {r}", file=out)
        out.flush()
        for p in (f"{log_dir}/.running_{k}", f"{log_dir}/.out_{k}.txt"):
            try:
                os.remove(p)
            except OSError:
                pass

    return run_graph(sel, deps, launch, max_parallel, on_done=on_done)
```

- [ ] **Step 4: 跑测试, 确认通过**

Run: `cd ~/sam3d && $PY tools/tests/test_sched.py`
Expected: 13 行 `ok test_...`

- [ ] **Step 5: 提交**

```bash
cd ~/sam3d && git add pipeline/sched.py tools/tests/test_sched.py && git commit -q -m "调度: run_graph / run_children —— 依赖满足的步骤同时跑, 每步一个子进程

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: 领卡 `GpuSlot`

**Files:**
- Modify: `pipeline/_common.py`(`# ==== GPU 排队` 那一节)
- Create: `tools/tests/test_gpu_slot.py`
- Modify: `tools/tests/run.sh`

**Interfaces:**
- Produces(都在 `_common` 里):
  - `parallel_on() -> bool`; `max_gpus() -> int`; `lock_path_of(dev) -> str`; `gpu_pool() -> list[str]`; `card_busy(card, need_mb=0) -> bool`
  - `class GpuSlot(what="", need_mb=0, run_dir=None, pool=None, busy=None, set_env=True, avoid=(), poll=5, timeout=10800)`, 当 `with` 用。进去之后 `.card`(卡号字符串)、`.waited`(等了多少秒)。`set_env=True` 时把本进程的 `CUDA_VISIBLE_DEVICES` 设成这张卡, 出来时恢复。`run_dir` 不给就读环境变量 `RUN_DIR`。
  - `gpus_free_now(need_mb=0, run_dir=None, pool=None, busy=None) -> int`
- `GpuLock` 和 `lock_path()` 保留(Task 4 换掉调用处之后 `GpuLock` 才没人用, 到时再删)。

- [ ] **Step 1: 写失败的测试**

`tools/tests/test_gpu_slot.py`:

```python
#!/usr/bin/env python
"""领卡(_common.GpuSlot): 哪张空着用哪张, 用完还。假卡号 + 假的"卡上有没有别人", 不碰真显卡。
跑: $PY tools/tests/test_gpu_slot.py"""
import json, os, sys, tempfile, threading, time

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import _common as C

POOL = ["t91", "t92", "t93"]                    # 假卡号: 锁文件是 /tmp/outdoor_pipeline_gput91.lock, 不和真卡撞
FREE = lambda card, need_mb=0: False


def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
    return old


def _restore(old):
    for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v


def test_lock_file_name_is_unchanged():
    assert C.lock_path_of("3") == "/tmp/outdoor_pipeline_gpu3.lock"
    old = _env(CUDA_VISIBLE_DEVICES="5,6")
    try:
        assert C.lock_path() == "/tmp/outdoor_pipeline_gpu5.lock"
    finally:
        _restore(old)


def test_two_takers_get_different_cards():
    with C.GpuSlot("a", pool=POOL, busy=FREE, set_env=False) as a:
        with C.GpuSlot("b", pool=POOL, busy=FREE, set_env=False) as b:
            assert (a.card, b.card) == ("t91", "t92"), (a.card, b.card)
    with C.GpuSlot("c", pool=POOL, busy=FREE, set_env=False) as c:
        assert c.card == "t91"                                  # 还回去之后又能领到


def test_cards_used_by_others_are_skipped():
    busy = lambda card, need_mb=0: card == "t91"
    with C.GpuSlot("a", pool=POOL, busy=busy, set_env=False) as a:
        assert a.card == "t92"


def test_waits_until_a_card_frees_up():
    """卡全被占着: 等, 不报错; 有卡空出来就拿到。"""
    state = {"n": 0}
    def busy(card, need_mb=0):
        state["n"] += 1
        return state["n"] <= 4                                  # 头两轮(每轮查 2 张)都说有人
    t0 = time.time()
    with C.GpuSlot("a", pool=POOL[:2], busy=busy, set_env=False, poll=0.05) as a:
        assert a.card == "t91" and a.waited >= 0.1, (a.card, a.waited)
    assert time.time() - t0 < 3


def test_one_card_is_shared_in_turn():
    """只有一张卡可用: 第二个等第一个还了再用。不死锁。"""
    order = []
    def work(name, secs):
        with C.GpuSlot(name, pool=["t94"], busy=FREE, set_env=False, poll=0.05):
            order.append((name, "进")); time.sleep(secs); order.append((name, "出"))
    a = threading.Thread(target=work, args=("a", 0.3)); b = threading.Thread(target=work, args=("b", 0.05))
    a.start(); time.sleep(0.1); b.start(); a.join(5); b.join(5)
    assert order == [("a", "进"), ("a", "出"), ("b", "进"), ("b", "出")], order


def test_timeout_is_a_clear_failure():
    try:
        with C.GpuSlot("a", pool=["t95"], busy=lambda c, need_mb=0: True, set_env=False, poll=0.05, timeout=0.2):
            raise AssertionError("不该领到")
    except SystemExit as e:
        assert "t95" in str(e) and "排" in str(e), str(e)


def test_pinned_card_is_the_whole_pool():
    old = _env(CUDA_VISIBLE_DEVICES="t96", GPU_POOL=None)
    try:
        assert C.gpu_pool() == ["t96"]
        os.environ["GPU_POOL"] = "t97,t98"
        assert C.gpu_pool() == ["t97", "t98"]                   # GPU_POOL 说了算
    finally:
        _restore(old)


def test_set_env_and_restore():
    old = _env(CUDA_VISIBLE_DEVICES=None)
    try:
        with C.GpuSlot("a", pool=POOL, busy=FREE) as a:
            assert os.environ["CUDA_VISIBLE_DEVICES"] == a.card and os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
    finally:
        _restore(old)


def test_run_quota_caps_cards_per_run():
    """一轮最多同时用 MAX_GPUS 张: 名额用完, 卡再空也要等。"""
    with tempfile.TemporaryDirectory() as T:
        old = _env(MAX_GPUS="2")
        try:
            assert C.gpus_free_now(run_dir=T, pool=POOL, busy=FREE) == 2
            with C.GpuSlot("a", run_dir=T, pool=POOL, busy=FREE, set_env=False):
                with C.GpuSlot("b", run_dir=T, pool=POOL, busy=FREE, set_env=False):
                    assert C.gpus_free_now(run_dir=T, pool=POOL, busy=FREE) == 0
                    try:
                        with C.GpuSlot("c", run_dir=T, pool=POOL, busy=FREE, set_env=False, poll=0.05, timeout=0.2):
                            raise AssertionError("名额满了不该领到")
                    except SystemExit:
                        pass
            assert C.gpus_free_now(run_dir=T, pool=POOL, busy=FREE) == 2
        finally:
            _restore(old)


def test_marker_file_says_who_uses_which_card():
    with tempfile.TemporaryDirectory() as T:
        with C.GpuSlot("track/w0", run_dir=T, pool=POOL, busy=FREE, set_env=False) as a:
            d = json.load(open(f"{T}/logs/.gpu_{a.card}.json"))
            assert d["步骤"] == "track/w0" and d["卡"] == a.card and d["pid"] == os.getpid(), d
        assert not os.path.exists(f"{T}/logs/.gpu_{a.card}.json")


def test_avoid_prefers_another_card():
    with C.GpuSlot("a", pool=POOL, busy=FREE, set_env=False, avoid=["t91"]) as a:
        assert a.card == "t92"
    with C.GpuSlot("a", pool=["t91"], busy=FREE, set_env=False, avoid=["t91"]) as a:
        assert a.card == "t91"                                  # 只有这一张就还用它


def test_defaults():
    old = _env(MAX_GPUS=None, PARALLEL=None)
    try:
        assert C.max_gpus() == 8 and C.parallel_on() is True
        os.environ["PARALLEL"] = "0"
        assert C.parallel_on() is False
    finally:
        _restore(old)


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
```

- [ ] **Step 2: 跑测试, 确认失败**

Run: `cd ~/sam3d && $PY tools/tests/test_gpu_slot.py`
Expected: `AttributeError: module '_common' has no attribute 'lock_path_of'`

- [ ] **Step 3: 实现**

`pipeline/_common.py`: 把现有的 `def lock_path(): ...`(三行)换成下面这一段, `LOCK = lock_path()` 那行和 `class GpuLock`、`def wait_gpu` 不动。新的一节放在 `def wait_gpu` 之后、`# ==== 跑命令` 之前。

替换 `lock_path`:

```python
def lock_path_of(dev):
    return f"/tmp/outdoor_pipeline_gpu{dev}.lock"


def lock_path():
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip() or "0"
    return lock_path_of(dev)
```

新的一节:

```python
# ==================================================================== 领卡(2026-09-29)
# ★ 以前一轮开跑时 drive_go.sh 挑一张卡, 整轮所有步骤都钉在这张卡上, 另外 7 张闲着。
#   现在每个吃显存的任务开始时领一张空卡, 跑完还 —— 同一轮里几个步骤可以各用一张同时跑(见 sched.py)。
# ★ 每张卡还是那一把锁, 文件名不变(lock_path_of)。旧方式起的轮次、gpulock.sh 和这里互相认得。
MAX_GPUS = 8          # 一轮最多同时用几张卡(环境变量 MAX_GPUS 可改)。现在一个人用 8 张; 同事一起用时改成 3
FOREIGN_MB = 2000     # 卡上别人的进程占显存超过这么多 = 这张卡有人在用(和 wait_gpu 同一个数)
SLOT_POLL = 5         # 秒: 没领到卡, 隔多久再找一遍
SLOT_TIMEOUT = 10800  # 秒: 排这么久还领不到就报失败(和 wait_gpu 一样 3 小时)


def parallel_on():
    """PARALLEL=0 = 和以前完全一样一步一步跑。出问题时用它退回去。"""
    return os.environ.get("PARALLEL", "1") != "0"


def max_gpus():
    return max(1, int(os.environ.get("MAX_GPUS", MAX_GPUS)))


def _smi(query, what="gpu"):
    r = subprocess.run(["nvidia-smi", f"--query-{what}={query}", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return [[x.strip() for x in l.split(",")] for l in r.stdout.strip().splitlines() if l.strip()]


def gpu_pool():
    """可以用哪些卡。GPU_POOL 说了算; 没给但钉了 CUDA_VISIBLE_DEVICES(想钉某张卡: CUDA_VISIBLE_DEVICES=3
    drive_go.sh ...)就只用钉的; 都没给 = 机器上全部的卡。"""
    p = os.environ.get("GPU_POOL", "").strip() or os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if p:
        return [x.strip() for x in p.split(",") if x.strip()]
    return [r[0] for r in _smi("index")]


def card_busy(card, need_mb=0):
    """这张卡现在不能用: 有别人的进程占着显存(超过 FOREIGN_MB), 或者余量不够 need_mb。
    ★ 只看瞬时余量不够, 理由见 wait_gpu: 别人的活有空档期, 骗得过余量检查。"""
    row = next((r for r in _smi("index,uuid,memory.used,memory.total") if r[0] == str(card)), None)
    if row is None:
        return True
    me = os.getpid()
    others = sum(int(mb) for u, pid, mb in _smi("gpu_uuid,pid,used_memory", "compute-apps")
                 if u == row[1] and int(pid) != me and int(mb) > FOREIGN_MB)
    return bool(others) or int(row[3]) - int(row[2]) < need_mb


def _try_lock(path):
    fh = open(path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fh
    except OSError:
        fh.close()
        return None


def _unlock(fh):
    if fh not in (None, True):
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def _take_quota(run_dir):
    """本轮的名额: <本轮>/logs/.gpu_quota_<i>.lock 里拿到任意一把。几个子进程共用一轮的名额,
    所以是文件锁不是计数器。没有本轮目录(单跑一个脚本)就不限, 返回 True。满了返回 None。"""
    if not run_dir:
        return True
    os.makedirs(f"{run_dir}/logs", exist_ok=True)
    for i in range(max_gpus()):
        fh = _try_lock(f"{run_dir}/logs/.gpu_quota_{i}.lock")
        if fh:
            return fh
    return None


def _take_card(pool, busy, need_mb, avoid=()):
    """avoid: 上次在这张卡上失败了, 有别的就先用别的。"""
    for card in [c for c in pool if c not in avoid] + [c for c in pool if c in avoid]:
        fh = _try_lock(lock_path_of(card))
        if not fh:
            continue
        if busy(card, need_mb):
            _unlock(fh)
            continue
        return card, fh
    return None


def gpus_free_now(need_mb=0, run_dir=None, pool=None, busy=None):
    """现在马上能领到几张卡: 空卡数和本轮剩余名额里小的那个。只看一眼, 不占。拆分时用它定拆几份。"""
    run_dir = run_dir if run_dir is not None else os.environ.get("RUN_DIR") or None
    pool, busy = (gpu_pool() if pool is None else pool), busy or card_busy
    held, cards = [], 0
    try:
        for card in pool:
            fh = _try_lock(lock_path_of(card))
            if fh:
                held.append(fh)
                cards += 0 if busy(card, need_mb) else 1
        if not run_dir:
            return cards
        os.makedirs(f"{run_dir}/logs", exist_ok=True)
        quota = 0
        for i in range(max_gpus()):
            fh = _try_lock(f"{run_dir}/logs/.gpu_quota_{i}.lock")
            if fh:
                held.append(fh); quota += 1
        return min(cards, quota)
    finally:
        for fh in held:
            _unlock(fh)


class GpuSlot:
    """领一张空卡, 用完还。with GpuSlot("track") as s: ... s.card 是卡号。
    set_env=True: 把本进程的 CUDA_VISIBLE_DEVICES 设成这张卡(之后起的命令都落在它上面), 出来时恢复。
    一个进程里开几个线程各领一张的(拆分)用 set_env=False, 自己把卡号写进命令的环境变量。"""

    def __init__(self, what="", need_mb=0, run_dir=None, pool=None, busy=None, set_env=True, avoid=(),
                 poll=SLOT_POLL, timeout=SLOT_TIMEOUT):
        self.what, self.need_mb, self.set_env, self.avoid = what, need_mb, set_env, tuple(avoid)
        self.run_dir = run_dir if run_dir is not None else os.environ.get("RUN_DIR") or None
        self.pool, self.busy, self.poll, self.timeout = pool, busy or card_busy, poll, timeout
        self.card = self.waited = self._fh = self._quota = self._old = self._mark = None

    def __enter__(self):
        t0, said = time.time(), False
        while True:
            pool = gpu_pool() if self.pool is None else self.pool
            q = _take_quota(self.run_dir)
            got = _take_card(pool, self.busy, self.need_mb, self.avoid) if q else None
            if got:
                (self.card, self._fh), self._quota = got, q
                break
            _unlock(q)
            if time.time() - t0 > self.timeout:
                raise SystemExit(f"[失败] {self.what or '这一步'} 排了 {self.timeout / 60:.0f}min 队还是领不到卡 "
                                 f"(可用的卡 {pool}, 本轮最多 {max_gpus()} 张)")
            if not said:
                print(f"    [等待] {self.what or '这一步'}: 没有空卡, 或本轮 {max_gpus()} 张的名额用满了, 排队中...",
                      flush=True)
                said = True
            time.sleep(self.poll)
        self.waited = time.time() - t0
        if self.set_env:
            self._old = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.card)
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # nvidia-smi 的编号才和 CUDA 一致
        if self.run_dir:
            # 谁在用哪张卡: 网页和看板读这个文件。进程被杀的话文件会留下, 读的人要核对 pid 还在不在
            self._mark = f"{self.run_dir}/logs/.gpu_{self.card}.json"
            tmp = f"{self._mark}.{os.getpid()}.tmp"
            json.dump({"卡": str(self.card), "步骤": self.what, "pid": os.getpid(),
                       "开始": time.strftime("%F %T")}, open(tmp, "w"), ensure_ascii=False)
            os.replace(tmp, self._mark)
        print(f"    [卡] {self.what or '这一步'} 用 GPU {self.card}"
              + (f" (等了 {self.waited:.0f}s)" if self.waited >= 1 else ""), flush=True)
        return self

    def __exit__(self, *a):
        if self._mark:
            try:
                os.remove(self._mark)
            except OSError:
                pass
        if self._old is not None:
            for k, v in self._old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        _unlock(self._fh)
        _unlock(self._quota)
```

- [ ] **Step 4: 跑测试, 确认通过**

Run: `cd ~/sam3d && $PY tools/tests/test_gpu_slot.py && $PY tools/tests/test_gpu_alloc.py`
Expected: `test_gpu_slot.py` 12 行 `ok`; `test_gpu_alloc.py` 照旧全过(锁文件名没变)。

- [ ] **Step 5: 加进 run.sh, 提交**

`tools/tests/run.sh` 第 5 行末尾追加 `; $PY test_gpu_slot.py`。

```bash
cd ~/sam3d && git add pipeline/_common.py tools/tests/test_gpu_slot.py tools/tests/run.sh && git commit -q -m "领卡: GpuSlot —— 哪张空着用哪张, 每轮有名额上限(MAX_GPUS)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: 室外流水线和出人流水线接上调度

**Files:**
- Modify: `pipeline/outdoor_pipeline.py`(import、`main`、新函数 `_parallel`)
- Modify: `pipeline/body4d_pipeline.py`(import、`main`)
- Modify: `tools/drive_go.sh:34-47`
- Modify: `tools/tests/test_sched.py`
- Modify: `.claude/skills/run-recon-pipeline/SKILL.md`

**Interfaces:**
- Consumes: `sched.check_graph` / `sched.run_children`; `_common.GpuSlot` / `_common.parallel_on`。
- Produces: `outdoor_pipeline.py --only a,b,c`(多于一个步骤、`PARALLEL` 不是 `0`、不是干跑)并行跑; 隐藏参数 `--child`(子进程用: 不打印开头的横幅和结尾的 `[全部完成]`)。环境变量 `RUN_DIR` = 本轮结果目录, 传给所有子进程。

- [ ] **Step 1: 写失败的测试**

`tools/tests/test_sched.py` 的 `if __name__` 之前加:

```python
def test_dry_run_is_still_linear_and_prints_commands():
    """干跑不并行: 打印出来的命令要能一条条照着粘去跑, 顺序就是步骤表的顺序。"""
    r = subprocess.run([sys.executable, f"{SAM3D}/pipeline/outdoor_pipeline.py", "--scene", "nfl1", "--run", "r1",
                        "--only", "vlm,da3,up", "-n"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    heads = [l.split()[1] for l in r.stdout.splitlines() if l.startswith("== ")]
    assert heads == ["vlm", "da3", "up"], heads
    assert "[并行]" not in r.stdout


def test_child_mode_prints_no_banner():
    r = subprocess.run([sys.executable, f"{SAM3D}/pipeline/outdoor_pipeline.py", "--scene", "nfl1", "--run", "r1",
                        "--only", "up", "--child", "-n"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    assert "== up  [" in r.stdout, r.stdout[:200]
    assert "[全部完成]" not in r.stdout and "\n场景 " not in "\n" + r.stdout


def test_drive_go_does_not_pin_a_card():
    txt = open(f"{SAM3D}/tools/drive_go.sh").read()
    assert "pick_gpu" not in txt
    assert "env -u CUDA_VISIBLE_DEVICES" in txt                  # tmux 服务端的环境里可能留着旧的卡号
    for v in ("PARALLEL", "MAX_GPUS", "GPU_POOL"):
        assert v in txt, v
```

- [ ] **Step 2: 跑测试, 确认失败**

Run: `cd ~/sam3d && $PY tools/tests/test_sched.py`
Expected: 前 13 个 `ok`, `test_dry_run_is_still_linear_and_prints_commands` 通过(现在本来就是线性的), 然后 `test_child_mode_prints_no_banner` 失败: `unrecognized arguments: --child`

- [ ] **Step 3: 改 `pipeline/outdoor_pipeline.py`**

1. import 那几行改成:

```python
import _common as _cm          # 设 CUR_STEP 用 —— 必须改模块上的属性, from-import 进来的是副本
import sched                   # pipeline/sched.py(不是标准库那个): 按依赖表并行跑步骤
from _common import (LOCK, GpuSlot, wait_gpu, sh, envcmd, stamp_path,
                     timing as _timing, resolve_cfg as _resolve_cfg,
                     write_frame_map, load_frame_map)
```

2. 在 `def main():` 之前加:

```python
def _parallel(g, c, names):
    """一段里能同时跑的步骤同时跑: 每个步骤起一个子进程(--only <步骤> --child), 谁的依赖满足了谁开始。
    吃显存的步骤在子进程里自己领卡(GpuSlot)。规则和理由见 sched.py。"""
    sched.check_graph([s[0] for s in STEPS], DEPS)
    tsv = f"{c.LOG}/timing.tsv"
    if not os.path.exists(tsv):     # 表头先写好: 几个子进程同时发现"文件不存在"会各写一遍表头
        with open(tsv, "w") as f:
            f.write("步骤\t环境\t开始\t结束\t耗时秒\t等GPU秒\t结果\n")
    me = os.path.abspath(__file__)
    rc = sched.run_children(
        names, DEPS,
        lambda k: [sys.executable, "-u", me, "--scene", g.scene, "--run", g.run, "--only", k, "--child"],
        c.LOG, max_parallel=8, env=dict(os.environ, RUN_DIR=c.R1))
    bad = [k for k in names if rc.get(k, 0) != 0]
    if bad:
        left = [k for k in names if k not in rc]
        raise SystemExit(f"[失败] 这些步骤失败: {bad}" + (f"; 没轮到跑的: {left}" if left else ""))
    print("[全部完成]")
```

3. `main()` 的参数里, `--dump-deps` 后面加:

```python
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)   # sched.run_children 起的子进程
```

4. `main()` 里从 `os.makedirs(c.LOG, exist_ok=True)` 到函数结尾, 整段换成:

```python
    os.makedirs(c.LOG, exist_ok=True)
    os.environ.setdefault("RUN_DIR", c.R1)      # 领卡的名额和"谁在用哪张卡"都记在本轮目录里, 子进程也要知道
    if not g.child:
        print(f"场景 {c.scene}  轮次 {c.run}")
        print(f"帧   {c.FR}")
        print(f"结果 {c.R1}")
        print(f"步骤 {' -> '.join(s[0] for s in sel)}" + ("   [干跑]" if g.dry_run else "") + "\n")
    if len(sel) > 1 and _cm.parallel_on() and not g.dry_run:
        return _parallel(g, c, [s[0] for s in sel])
    for k, e, d, fn in sel:
        t = time.time()
        for lg in glob.glob(f"{c.LOG}/*.log"): pass
        c._sep = f"\n{'='*70}\n== {k}  {time.strftime('%F %T')}\n{'='*70}\n"
        _cm.CUR_STEP = k               # sh() 靠它往步骤日志打 "== <步骤> <时间戳>" 分隔线
        print(f"== {k}  [{e}]  {d}", flush=True)
        _o = globals()["sh"]
        globals()["sh"] = lambda *a, **kw: _o(*a, **{**kw, "sep": kw.get("sep", c._sep)})
        t_run, ok = t, "失败"
        try:
            if k in GPU_STEPS and not g.dry_run:
                with GpuSlot(k):
                    t_run = time.time()          # 等卡的时间单独记, 不算进这一步的耗时
                    fn(c, g.dry_run)
            else:
                fn(c, g.dry_run)
            ok = "ok"
        finally:
            globals()["sh"] = _o
            if not g.dry_run: _timing(c, k, e, t, t_run, ok)
        if not g.dry_run: open(stamp_path(c, k), "w").write(time.strftime("%F %T"))
        print(f"   {time.time()-t:.1f}s\n", flush=True)
    if not g.child:
        print("[全部完成]")
```

和原来相比只有四处不同: 加了 `RUN_DIR`; 四行横幅和结尾的 `[全部完成]` 只在 `not g.child` 时打印; 多个步骤时走 `_parallel`; 循环里 `GpuLock()` 换成 `GpuSlot(k)`。

- [ ] **Step 4: 改 `pipeline/body4d_pipeline.py`**

1. import 里 `GpuLock` 换成 `GpuSlot`。
2. `main()` 里 `os.makedirs(c.LOG, exist_ok=True)` 后面加一行:

```python
    os.environ.setdefault("RUN_DIR", c.R1)      # 主流程拉起时外面已经给了(整轮共用名额); 单跑时用自己的目录
```

3. 循环里 `with GpuLock():` 换成 `with GpuSlot(k):`。

出人流水线的步骤是一条直线(frames → track → check → pick → mesh → report), 步骤之间不并行。

- [ ] **Step 5: 改 `tools/drive_go.sh`**

把第 34 行(`# ★ 挑卡: ...`)到 `tmux new-session` 之前的 `for v in ...` 那一行, 整段换成:

```bash
# ★ 2026-09-29 不在这里挑卡了: 每个吃显存的任务开始时自己领一张空卡(_common.GpuSlot), 一轮里能同时用好几张。
#   想钉某张卡照旧: CUDA_VISIBLE_DEVICES=3 drive_go.sh ... —— 这一轮就只用 3 号卡。
# ★ 环境变量显式带进去: tmux 会话继承的是 tmux 服务端的环境, 不是调这个脚本的人的。
#   没钉卡时要把 CUDA_VISIBLE_DEVICES 明确去掉 —— 服务端的环境里可能留着以前的卡号, 留着就等于钉死了。
GPU="-u CUDA_VISIBLE_DEVICES"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && GPU="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
GPU="$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID"
for v in PARALLEL MAX_GPUS GPU_POOL; do [ -n "${!v:-}" ] && GPU="$GPU $v=${!v}"; done
echo "卡: ${CUDA_VISIBLE_DEVICES:-每个任务自己领}  ${PARALLEL:+PARALLEL=$PARALLEL }${MAX_GPUS:+MAX_GPUS=$MAX_GPUS }${GPU_POOL:+GPU_POOL=$GPU_POOL}"
```

`tmux new-session` 那一行里的 `env $GPU $PY ...` 不用改(`env -u X K=V 命令` 是合法写法)。

- [ ] **Step 6: 跑测试, 确认通过**

Run: `cd ~/sam3d && $PY tools/tests/test_sched.py && $PY tools/tests/test_gpu_alloc.py`
Expected: `test_sched.py` 16 行 `ok`。`test_gpu_alloc.py` 里 `test_go_lets_drive_go_pick_the_gpu` 照旧通过(它用的是假的 drive_go.sh)。

- [ ] **Step 7: 冒烟: 拿一轮已有的结果, 并行重跑两个便宜的、互不依赖的步骤**

`ratio` 和 `meshes` 都只读 gen 的产物、各写各的文件, 不吃显存, 重跑不改变结果。用 soccer r2(不是要对比的 nfl1):

```bash
cd ~/sam3d && cp -p data/outdoor/result/soccer/r2/05_mesh_ratio.json /tmp/claude-ratio-before.json && \
$PY -u pipeline/outdoor_pipeline.py --scene soccer --run r2 --only ratio,meshes 2>&1 | tail -20 && \
cmp data/outdoor/result/soccer/r2/05_mesh_ratio.json /tmp/claude-ratio-before.json && echo "ratio 结果没变" && \
ls data/outdoor/result/soccer/r2/logs/ -a | grep -c "^\.running_\|^\.out_"
```

Expected: 输出里有 `[并行] 开始 ratio` 和 `[并行] 开始 meshes`(两行挨着, 说明是同时启动的), 各自一块 `== ratio ...` / `== meshes ...`, 最后 `[全部完成]`; `ratio 结果没变`; 最后一行是 `0`(标记文件都删干净了)。

- [ ] **Step 8: 更新 SKILL.md, 提交**

`.claude/skills/run-recon-pipeline/SKILL.md`: 找到讲挑卡的那一段(`grep -n "pick_gpu" .claude/skills/run-recon-pipeline/SKILL.md`, 在 189 行附近), 整段换成:

```markdown
- **并行和卡(2026-09-29)**: 一段里互不依赖的步骤同时跑, 每个吃显存的任务开始时自己领一张空卡、跑完还
  (`_common.GpuSlot`; 依据是各流水线的 `DEPS`, 调度在 `pipeline/sched.py`)。
  - `MAX_GPUS`(默认 8): 一轮最多同时用几张卡。同事一起用时设成 3。
  - `GPU_POOL`(默认全部): 可以用哪些卡, 例如 `GPU_POOL=0,1,2,3`。
  - `PARALLEL=0`: 退回一步一步跑, 和以前完全一样。
  - 想钉某张卡: `CUDA_VISIBLE_DEVICES=3 tools/drive_go.sh <场景> <轮次>`。
  - 谁在用哪张卡: `<本轮>/logs/.gpu_<卡号>.json`; 正在跑哪几步: `<本轮>/logs/.running_<步骤>`。
```

```bash
cd ~/sam3d && git add pipeline/outdoor_pipeline.py pipeline/body4d_pipeline.py tools/drive_go.sh tools/tests/test_sched.py .claude/skills/run-recon-pipeline/SKILL.md && git commit -q -m "室外流水线接上并行调度; 每个任务自己领卡, drive_go.sh 不再开跑时挑卡

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: 网页和看板显示多个在跑的任务

**Files:**
- Modify: `tools/run_ctl.py`(挑卡那一节、`state`、`segments_view`)
- Modify: `tools/run_page.py:311-317`
- Modify: `tools/watch_run.py`(`parse_drive`、380 行附近)
- Modify: `tools/tests/test_run_ctl.py`、`tools/tests/test_gpu_alloc.py`
- Create: `tools/tests/test_watch_parse.py`
- Modify: `tools/tests/run.sh`

**Interfaces:**
- Consumes: `<R1>/logs/.running_<步骤>`(Task 2)、`<R1>/logs/.gpu_<卡>.json`(Task 3)。
- Produces: `run_ctl.running_steps(R1) -> list[str]`; `run_ctl.gpu_tasks(R1) -> list[{"卡": str, "步骤": str}]`; `run_ctl.state()` 的返回里 `"GPU"` 变成逗号连起来的卡号(没有就是 `None`), 新增 `"任务"`。`watch_run.parse_drive()` 的返回里新增 `"running_all": list[str]`。
- 删除: `run_ctl.pick_gpu` / `busy_gpus` / `run_gpu` / `_driver_procs` / `gpu_cards` / `FREE_MB`(没人用了)。

- [ ] **Step 1: 写失败的测试**

`tools/tests/test_run_ctl.py` 的 `if __name__` 之前加:

```python
def _marker(R1, step=None, card=None, what=None, pid=None):
    os.makedirs(f"{R1}/logs", exist_ok=True)
    pid = os.getpid() if pid is None else pid
    if step:
        open(f"{R1}/logs/.running_{step}", "w").write(str(pid))
    if card:
        json.dump({"卡": card, "步骤": what, "pid": pid, "开始": "x"}, open(f"{R1}/logs/.gpu_{card}.json", "w"))


def _dead_pid():
    import subprocess, sys
    p = subprocess.Popen([sys.executable, "-c", "pass"]); p.wait()
    return p.pid


def test_several_steps_running_at_once():
    """并行之后一段里可以有好几步同时在跑, 各用各的卡。"""
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        R1 = run_ctl.new_run("outdoor", "soccer", "r1", people="1", recall="0")
        run_ctl.running = lambda scene, run: True               # 先建轮次再说它在跑: 在跑的轮次 go() 会拒绝
        _state(R1, 2, False)
        open(f"{R1}/logs/drive_0929_100000.log", "w").write("[开始] 2026-09-29 10:00:00\n[并行] 开始 gen\n")
        for step, card, what in (("gen", "0", "gen"), ("vlm_sizes", "1", "vlm_sizes"), ("people_mesh", "3", "mesh/p0")):
            _marker(R1, step, card, what)
        _marker(R1, card="2", what="mesh/p1")
        s = run_ctl.state("outdoor", "soccer", "r1")
        assert s["GPU"] == "0,1,2,3", s["GPU"]
        assert s["任务"] == [{"卡": "0", "步骤": "gen"}, {"卡": "1", "步骤": "vlm_sizes"},
                             {"卡": "2", "步骤": "mesh/p1"}, {"卡": "3", "步骤": "mesh/p0"}], s["任务"]
        rows = {r["步骤"]: r["状态"] for r in s["段列表"][2]["步骤"]}
        assert [k for k, v in rows.items() if v == "running"] == ["gen", "vlm_sizes", "people_mesh"], rows
        assert rows["scale"] == "pending"


def test_stale_markers_are_ignored():
    """进程被杀的时候来不及删标记。进程已经不在的标记不算数, 不然网页一直显示"在跑"。"""
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        R1 = run_ctl.new_run("outdoor", "soccer", "r1", people="1", recall="0")
        _marker(R1, "gen", "0", "gen", pid=_dead_pid())
        assert run_ctl.running_steps(R1) == [] and run_ctl.gpu_tasks(R1) == []
        open(f"{R1}/logs/.gpu_5.json", "w").write("{坏的")                # 写到一半的文件也不能让网页崩
        assert run_ctl.gpu_tasks(R1) == []


def test_old_style_run_still_shows_its_step():
    """PARALLEL=0 或者老轮次: 没有标记文件, 照旧看驱动日志里最后出现的步骤。"""
    with tempfile.TemporaryDirectory() as T:
        _sandbox(T)
        R1 = run_ctl.new_run("outdoor", "soccer", "r1", people="1", recall="0")
        run_ctl.running = lambda scene, run: True
        _state(R1, 0, False)
        open(f"{R1}/logs/drive_0929_100000.log", "w").write("[开始] 2026-09-29 10:00:00\n== vlm  [trellis2]  x\n   3.0s\n== da3  [spatiallm]  y\n")
        s = run_ctl.state("outdoor", "soccer", "r1")
        rows = {r["步骤"]: r["状态"] for r in s["段列表"][0]["步骤"]}
        assert rows["da3"] == "running" and s["GPU"] is None and s["任务"] == [], (rows, s["GPU"])
```

`tools/tests/test_run_ctl.py` 里如果别的测试在这三个之后还要用 `run_ctl.running`, 注意 `_sandbox` 每次都会把它重设成 `lambda: False`, 不用手工恢复。

`tools/tests/test_gpu_alloc.py`: 删掉 `test_pick_gpu` 和 `test_busy_gpus_reads_driver_process_env` 两个测试(被测的函数没了, 领卡的测试在 `test_gpu_slot.py`)。文件头的说明改成 `"""每张卡一把 GPU 锁(文件名规则) + 网页开跑不把自己的卡号传下去。领卡见 test_gpu_slot.py。`。

`tools/tests/test_watch_parse.py`:

```python
#!/usr/bin/env python
"""看板读驱动日志(watch_run.parse_drive): 并行之后一段里有好几步同时在跑。
跑: $PY tools/tests/test_watch_parse.py"""
import os, sys, tempfile

sys.path.insert(0, os.path.expanduser("~/sam3d/tools"))
import watch_run as W


def _parse(txt):
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(txt)
    try:
        return W.parse_drive(f.name)
    finally:
        os.remove(f.name)


def test_parallel_log():
    d = _parse("[开始] 2026-09-29 10:00:00\n[并行] 开始 vlm\n[并行] 开始 da3\n"
               "== vlm  [trellis2]  列词表\n    $ 命令\n   34.2s\n\n[并行] 开始 up\n")
    assert d["steps"] == [("vlm", 34.2)], d["steps"]
    assert d["running_all"] == ["da3", "up"] and d["running"] == "da3", d


def test_failed_step_is_not_running():
    d = _parse("[开始] 2026-09-29 10:00:00\n[并行] 开始 vlm\n[并行] 开始 da3\n"
               "== vlm  [trellis2]  列词表\n[失败] 退出码 1: python x.py\n[失败] 步骤 vlm 退出码 1\n")
    assert d["running_all"] == ["da3"] and d["fail"], d


def test_old_log_unchanged():
    d = _parse("[开始] 2026-09-24 15:43:35\n== vlm  [trellis2]  x\n   3.0s\n\n== da3  [spatiallm]  y\n")
    assert d["steps"] == [("vlm", 3.0)] and d["running"] == "da3" and d["running_all"] == ["da3"], d


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
```

`watch_run.py` 一导入就要 `rich`。先确认 sam3d 环境里有: `$PY -c "import rich"`。没有的话看 `tools/watch_run.sh` 用的是哪个 python, `test_watch_parse.py` 就用那个跑(`run.sh` 里那一条也用它)。

- [ ] **Step 2: 跑测试, 确认失败**

Run: `cd ~/sam3d && $PY tools/tests/test_run_ctl.py; $PY tools/tests/test_watch_parse.py`
Expected: `test_run_ctl.py` 在 `test_several_steps_running_at_once` 失败(`s["GPU"]` 是 `None`); `test_watch_parse.py` 报 `KeyError: 'running_all'`。

- [ ] **Step 3: 改 `tools/run_ctl.py`**

1. 从 `# ---- 挑卡(2026-09-24)` 那行注释到 `def pick_gpu` 函数结尾, 整节(`FREE_MB`、`gpu_cards`、`_driver_procs`、`busy_gpus`、`run_gpu`、`pick_gpu`)换成:

```python
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
```

2. `state()` 里, `body = people_decisions.auto_dir(R1)` 后面加一行 `tasks = gpu_tasks(R1) if busy else []`; 返回的字典里把

```python
            "在跑": busy, "GPU": run_gpu(scene, run) if busy else None,
```

换成

```python
            "在跑": busy, "GPU": ",".join(t["卡"] for t in tasks) or None, "任务": tasks,
```

3. `segments_view()` 里, 把

```python
    cur = None
    if busy and log:
        hits = re.findall(r"^== (\S+)  \[", open(log, errors="ignore").read(), re.M)
        cur = hits[-1] if hits else None
```

换成

```python
    # 在跑哪几步: 并行时看调度写的标记; 没有标记(PARALLEL=0 或老轮次)照旧看驱动日志里最后出现的那一步
    curs = set()
    if busy:
        curs = set(running_steps(R1))
        if not curs and log:
            hits = re.findall(r"^== (\S+)  \[", open(log, errors="ignore").read(), re.M)
            curs = set(hits[-1:])
```

再把下面那行里的 `"running" if k == cur else` 换成 `"running" if k in curs else`。

4. `go()` 里的注释 `#   不给卡号, drive_go.sh 就调 pick_gpu() 挑一张空的。` 换成 `#   不给卡号, 每个任务开始时自己领一张空卡(_common.GpuSlot)。`

- [ ] **Step 4: 改 `tools/run_page.py`**

把

```javascript
    const r = cur.步骤.find(x => x.状态 === 'running');
    big = '正在跑第 ' + cur.段 + ' 段「' + cur.名 + '」' + (S.GPU ? ' · GPU ' + S.GPU : '');
    sub = r ? '当前步骤 ' + r.步骤 + ' — ' + r.说明 + (cur.已用秒 != null ? ' · 已用 ' + dur(cur.已用秒) : '') : '';
```

换成

```javascript
    // 并行之后一段里可以有好几步同时在跑, 各用各的卡
    const rs = cur.步骤.filter(x => x.状态 === 'running');
    big = '正在跑第 ' + cur.段 + ' 段「' + cur.名 + '」' + (S.GPU ? ' · GPU ' + S.GPU : '');
    sub = rs.length ? '当前步骤 ' + rs.map(x => x.步骤).join('、') + (rs.length === 1 ? ' — ' + rs[0].说明 : '') +
          ((S.任务 || []).length ? ' · ' + S.任务.map(t => '卡 ' + t.卡 + ': ' + t.步骤).join(', ') : '') +
          (cur.已用秒 != null ? ' · 已用 ' + dur(cur.已用秒) : '') : '';
```

- [ ] **Step 5: 改 `tools/watch_run.py`**

1. `parse_drive()` 里, `r = dict(start=None, ...)` 那行的字典加一项 `running_all=[]`。
2. 循环之前加 `started = []`。循环体开头(`s = re.match(r"^== ...` 之前)加:

```python
        p = re.match(r"^\[并行\] 开始 (\S+)", ln)
        if p:                                   # 并行: 步骤启动时只有这一行, 它的输出跑完才整块写进来
            started.append(p.group(1))
            continue
        f = re.match(r"^\[失败\] 步骤 (\S+) 退出码", ln)
        if f and f.group(1) in started:
            started.remove(f.group(1))
```

3. `if d and cur:` 那个分支里, `r["steps"].append(...)` 后面加:

```python
            if cur in started:
                started.remove(cur)
```

4. 函数结尾的 `r["running"] = cur` 换成:

```python
    r["running_all"] = started + ([cur] if cur and cur not in started else [])
    r["running"] = r["running_all"][0] if r["running_all"] else None
```

5. 380 行附近画步骤表的地方, `if D["running"] == s:` 换成 `if s in D["running_all"]:`。紧接着那几行里的 `drive_tail[drive_tail.rfind(f"== {s} "):]` 不用改(并行时找不到就是整段尾巴, `step_detail` 主要读的是步骤自己的日志)。

- [ ] **Step 6: 跑测试, 确认通过**

Run: `cd ~/sam3d && $PY tools/tests/test_run_ctl.py && $PY tools/tests/test_watch_parse.py && $PY tools/tests/test_gpu_alloc.py && $PY tools/tests/test_run_page.py`
Expected: 全部 `ok`。

- [ ] **Step 7: 重启网页, 看一眼, 加进 run.sh, 提交**

```bash
cd ~/sam3d && tmux kill-session -t gate_ui; tmux new-session -d -s gate_ui "$HOME/miniforge3/envs/sam3d/bin/python tools/gate_ui.py --scene soccer --run r2 --port 8765 > tools/gate_ui_8765.log 2>&1"; \
until curl -s -o /dev/null -w "%{http_code}\n" "http://127.0.0.1:8765/api/run/state?s=soccer&r=r2" | grep -q 200; do sleep 1; done; \
curl -s "http://127.0.0.1:8765/api/run/state?s=soccer&r=r2" | $PY -c "import json,sys; d=json.load(sys.stdin); print(d['在跑'], d['GPU'], d['任务'])"
```

Expected: `False None []`(这一轮没在跑)。

`tools/tests/run.sh` 第 5 行末尾追加 `; $PY test_watch_parse.py`。

```bash
cd ~/sam3d && bash tools/tests/run.sh 2>&1 | tail -1 && git add tools/run_ctl.py tools/run_page.py tools/watch_run.py tools/tests/ && git commit -q -m "网页和看板: 一段里显示多个在跑的步骤和各自的卡

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: 步骤内部拆分的公共件 `fanout.py`

**Files:**
- Create: `pipeline/fanout.py`
- Create: `tools/tests/test_fanout.py`
- Modify: `tools/tests/run.sh`

**Interfaces:**
- Consumes: `_common.GpuSlot` / `_common.gpus_free_now`。
- Produces:
  - `fanout.split(items: list, weights: list[float], k: int) -> list[list]`: 分成最多 `k` 组, 各组权重和接近, 组内保持原来的顺序, 不返回空组。
  - `fanout.how_many(n_items, need_mb=0, **kw) -> int`: 拆几份 = `n_items` 和"现在能领到几张卡"里小的那个, 至少 1。`kw` 传给 `gpus_free_now`。
  - `fanout.run_workers(jobs: list[tuple[str, callable]], what: str, need_mb=0, retries=1, **slot_kw) -> dict[str, str|None]`: 每份一个线程, 各领一张卡后调 `fn(卡号)`。返回 `{名字: None}` 表示成功, `{名字: "错误文字"}` 表示重试后还是失败。
  - `fanout.track_words(R1, extra="") -> list[str]`: 这一轮还没有结果的词。
  - `fanout.merge_reports(parts: list[str], out: str) -> dict`: 几份 json 并成一份, 并完删掉各份。

- [ ] **Step 1: 写失败的测试**

`tools/tests/test_fanout.py`:

```python
#!/usr/bin/env python
"""步骤内部拆分(pipeline/fanout.py): 同一个脚本起 K 份, 每份算一部分, 各用一张卡。假卡 + 假任务。
跑: $PY tools/tests/test_fanout.py"""
import json, os, sys, tempfile, threading, time

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import fanout as F

FREE = lambda card, need_mb=0: False
KW = dict(pool=["t81", "t82", "t83"], busy=FREE, poll=0.05)


def test_split_balances_by_weight():
    """nfl1 的 8 个人, 出现的帧数差很多(114 ~ 12): 按帧数分, 每份的工作量接近。"""
    ids, w = [1, 2, 3, 4, 5, 6, 7, 8], [114, 114, 114, 112, 114, 66, 60, 12]
    g = F.split(ids, w, 3)
    assert sorted(x for grp in g for x in grp) == ids and len(g) == 3
    load = [sum(w[ids.index(x)] for x in grp) for grp in g]
    assert max(load) - min(load) <= 66, load                     # 总共 706, 每份 235 上下
    assert all(grp == sorted(grp) for grp in g)                  # 组内保持原来的顺序


def test_split_small_inputs():
    assert F.split([], [], 3) == []                              # 没有人 / 没有词: 空的, 不崩
    assert F.split([7], [50], 3) == [[7]]                        # 只有 1 个: 1 份
    assert F.split([1, 2], [5, 5], 8) == [[1], [2]]              # 份数不超过个数
    assert F.split([1, 2, 3], [1, 1, 1], 0) == [[1, 2, 3]]       # k 给错了也至少 1 份


def test_how_many():
    assert F.how_many(8, **{k: v for k, v in KW.items() if k != "poll"}) == 3    # 3 张空卡
    assert F.how_many(2, **{k: v for k, v in KW.items() if k != "poll"}) == 2    # 只有 2 个人
    assert F.how_many(5, pool=["t81"], busy=lambda c, need_mb=0: True) == 1      # 一张空卡都没有: 1 份, 排队


def test_workers_run_at_the_same_time_on_different_cards():
    seen, lock = [], threading.Lock()
    def job(name):
        def fn(card):
            t0 = time.time(); time.sleep(0.3)
            with lock: seen.append((name, card, t0, time.time()))
        return fn
    res = F.run_workers([(f"w{i}", job(f"w{i}")) for i in range(3)], "track", **KW)
    assert res == {"w0": None, "w1": None, "w2": None}, res
    assert sorted(c for _, c, _, _ in seen) == ["t81", "t82", "t83"], seen
    assert max(t0 for _, _, t0, _ in seen) < min(t1 for _, _, _, t1 in seen), "三份应该有重叠"


def test_failed_worker_retries_on_another_card_and_others_finish():
    cards, lock = {}, threading.Lock()
    def job(name):
        def fn(card):
            with lock: cards.setdefault(name, []).append(card)
            if name == "w1" and len(cards[name]) == 1:
                raise SystemExit("[失败] 退出码 1: 假的")
        return fn
    res = F.run_workers([(f"w{i}", job(f"w{i}")) for i in range(2)], "track", **KW)
    assert res == {"w0": None, "w1": None}, res                  # 重试成功
    assert len(cards["w1"]) == 2 and cards["w1"][0] != cards["w1"][1], cards


def test_worker_that_keeps_failing_is_reported():
    def bad(card): raise RuntimeError("显存炸了")
    def good(card): pass
    res = F.run_workers([("w0", bad), ("w1", good)], "mesh", **KW)
    assert res["w1"] is None and "显存炸了" in res["w0"], res


def test_track_words_skips_finished_ones():
    with tempfile.TemporaryDirectory() as R1:
        json.dump({"vocab": ["goal post", "helmet", "sign"]}, open(f"{R1}/01_vlm_vocab.json", "w"))
        os.makedirs(f"{R1}/03_masks"); open(f"{R1}/03_masks/goal_post.npz", "w").close()
        assert F.track_words(R1, "building, helmet") == ["helmet", "sign", "building"]


def test_merge_reports():
    with tempfile.TemporaryDirectory() as T:
        json.dump({"helmet": {"实例数": 5}}, open(f"{T}/p0.json", "w"), ensure_ascii=False)
        json.dump({"sign": {"实例数": 2}}, open(f"{T}/p1.json", "w"), ensure_ascii=False)
        d = F.merge_reports([f"{T}/p0.json", f"{T}/p1.json", f"{T}/没跑出来.json"], f"{T}/all.json")
        assert d == {"helmet": {"实例数": 5}, "sign": {"实例数": 2}} and json.load(open(f"{T}/all.json")) == d
        assert sorted(os.listdir(T)) == ["all.json"]


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
```

- [ ] **Step 2: 跑测试, 确认失败**

Run: `cd ~/sam3d && $PY tools/tests/test_fanout.py`
Expected: `ModuleNotFoundError: No module named 'fanout'`

- [ ] **Step 3: 实现 `pipeline/fanout.py`**

```python
#!/usr/bin/env python
"""步骤内部拆分(2026-09-29): 同一个脚本同时起 K 份, 每份算一部分, 各用一张卡。

★ 为什么: 最慢的两步内部是一个一个算的 —— 出人网格 8 个人算 11 分钟, track 9 个词算 4 分钟,
  各占一张卡, 另外几张闲着。每个人、每个词的结果本来就是分开存的, 拆开跑不用动模型代码。
★ 拆几份每次现算: 要算的个数、现在的空卡数、本轮剩余名额, 三个里最小的。只有 1 就和以前一样。
★ 拆分的步骤自己不领卡, 由它起的每一份各领一张 —— 否则这一步多占一张不用。

只用标准库。
"""
import json, os, threading

from _common import GpuSlot, gpus_free_now


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
```

- [ ] **Step 4: 跑测试, 确认通过**

Run: `cd ~/sam3d && $PY tools/tests/test_fanout.py`
Expected: 8 行 `ok test_...`

- [ ] **Step 5: 加进 run.sh, 提交**

`tools/tests/run.sh` 第 5 行末尾追加 `; $PY test_fanout.py`。

```bash
cd ~/sam3d && git add pipeline/fanout.py tools/tests/test_fanout.py tools/tests/run.sh && git commit -q -m "拆分的公共件 fanout.py: 分组、各领一张卡同时跑、失败换卡重试

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: track 按词拆分

**Files:**
- Modify: `data/outdoor/scripts/track_all.py`(词表那一段、结尾写报告那一行)
- Modify: `pipeline/fanout.py`(加 `fan_track`)
- Modify: `pipeline/outdoor_pipeline.py`(`s_track`、`main` 的循环)
- Modify: `tools/tests/test_fanout.py`

**Interfaces:**
- Consumes: `fanout.split` / `how_many` / `run_workers` / `track_words` / `merge_reports`; `_common.sh` / `envcmd` / `GpuSlot` / `parallel_on`。
- Produces: `fanout.fan_track(py, script, env, R1, log, dry, need_mb=0, max_parts=None, **kw) -> None`。失败抛 `SystemExit`, 信息里写明没出结果的词。`max_parts`: 最多拆几份(`1` = 不拆)。`kw` 传给 `how_many` / `run_workers`(测试注入假卡用)。
- Produces: `track_all.py` 认两个新的环境变量: `ONLY_WORDS`(逗号分隔, 只跟踪这几个词)、`TRACK_REPORT`(报告写到哪, 默认 `<R1>/03_track_report.json`)。
- Produces: 流水线里 `FANOUT_STEPS = {"track"}`: 这些步骤主循环不替它领卡。

- [ ] **Step 1: 写失败的测试**

`tools/tests/test_fanout.py` 的 `if __name__` 之前加:

```python
def _fake_track(T):
    """假的 track_all.py: 只认环境变量, 给每个词写一个 npz 和一份报告。FAIL_WORD 里的词第一次跑会失败。"""
    p = f"{T}/fake_track.py"
    open(p, "w").write('''
import json, os, sys
R1 = os.environ["R1"]; only = [w for w in os.environ.get("ONLY_WORDS", "").split(",") if w]
vocab = json.load(open(f"{R1}/01_vlm_vocab.json"))["vocab"]
words = [w for w in vocab if not only or w in only]
rep = {}
os.makedirs(f"{R1}/03_masks", exist_ok=True)
for w in words:
    f = f"{R1}/03_masks/{w.replace(' ', '_')}.npz"
    if os.path.exists(f): continue
    flag = f"{R1}/.failed_once_{w}"
    if w == os.environ.get("FAIL_WORD") and not os.path.exists(flag):
        open(flag, "w").close(); sys.exit(1)
    open(f, "w").write(os.environ.get("CUDA_VISIBLE_DEVICES", "?")); rep[w] = {"实例数": 1}
json.dump(rep, open(os.environ.get("TRACK_REPORT", f"{R1}/03_track_report.json"), "w"), ensure_ascii=False)
''')
    return p


def _track_run(words, **env):
    T = tempfile.mkdtemp()
    json.dump({"vocab": words}, open(f"{T}/01_vlm_vocab.json", "w"))
    F.fan_track(sys.executable, _fake_track(T), dict(R1=T, FR="x", EXTRA_WORDS="", **env), T, f"{T}/logs/03_track.log",
                False, **KW)
    cards = {w: open(f"{T}/03_masks/{w.replace(' ', '_')}.npz").read() for w in words}
    return T, cards, json.load(open(f"{T}/03_track_report.json"))


def test_fan_track_splits_words_over_cards():
    words = ["goal post", "helmet", "sign", "jersey", "football", "gloves"]
    T, cards, rep = _track_run(words)
    assert sorted(set(cards.values())) == ["t81", "t82", "t83"], cards          # 三张卡都用上了
    assert list(rep) and set(rep) == set(words), rep                            # 报告并成了一份
    assert not [f for f in os.listdir(T) if ".part" in f], os.listdir(T)


def test_fan_track_retry_only_redoes_missing_words():
    T, cards, rep = _track_run(["helmet", "sign", "jersey"], FAIL_WORD="sign")
    assert set(cards) == {"helmet", "sign", "jersey"}, cards


def test_fan_track_names_the_words_that_never_came_out():
    T = tempfile.mkdtemp()
    json.dump({"vocab": ["helmet", "sign"]}, open(f"{T}/01_vlm_vocab.json", "w"))
    bad = f"{T}/bad.py"; open(bad, "w").write("import sys; sys.exit(2)")
    try:
        F.fan_track(sys.executable, bad, dict(R1=T, FR="x", EXTRA_WORDS=""), T, f"{T}/logs/03_track.log", False, **KW)
    except SystemExit as e:
        assert "helmet" in str(e) and "sign" in str(e), str(e)
    else:
        raise AssertionError("该失败")


def test_fan_track_nothing_to_do():
    T = tempfile.mkdtemp()
    json.dump({"vocab": ["helmet"]}, open(f"{T}/01_vlm_vocab.json", "w"))
    os.makedirs(f"{T}/03_masks"); open(f"{T}/03_masks/helmet.npz", "w").close()
    F.fan_track(sys.executable, _fake_track(T), dict(R1=T, FR="x", EXTRA_WORDS=""), T, f"{T}/logs/03_track.log",
                False, **KW)                                                    # 词都跟踪过了: 照常跑一份, 不崩


def test_fan_track_max_parts_one_means_no_split():
    T = tempfile.mkdtemp()
    json.dump({"vocab": ["helmet", "sign", "jersey"]}, open(f"{T}/01_vlm_vocab.json", "w"))
    F.fan_track(sys.executable, _fake_track(T), dict(R1=T, FR="x", EXTRA_WORDS=""), T, f"{T}/logs/03_track.log",
                False, max_parts=1, **KW)
    cards = {open(f"{T}/03_masks/{w}.npz").read() for w in ("helmet", "sign", "jersey")}
    assert len(cards) == 1, cards                                               # 一份, 一张卡


def test_track_all_reads_the_two_env_vars():
    txt = open(f"{SAM3D}/data/outdoor/scripts/track_all.py").read()
    assert 'os.environ.get("ONLY_WORDS"' in txt and 'os.environ.get("TRACK_REPORT"' in txt
```

- [ ] **Step 2: 跑测试, 确认失败**

Run: `cd ~/sam3d && $PY tools/tests/test_fanout.py`
Expected: 前 8 个 `ok`, 然后 `AttributeError: module 'fanout' has no attribute 'fan_track'`

- [ ] **Step 3: 改 `data/outdoor/scripts/track_all.py`**

1. 在 `VOCAB = VOCAB + [w for w in EXTRA if w not in VOCAB]` 这个 `if EXTRA:` 块的后面加:

```python
# ★ 2026-09-29 拆分: 同一轮里这个脚本可能同时起几份, 每份只跟踪自己那几个词(各用一张卡)。
#   报告各写各的, 由流水线并成一份 —— 几份都往 03_track_report.json 写的话后写的盖掉先写的。
ONLY = [w for w in os.environ.get("ONLY_WORDS", "").split(",") if w]
if ONLY:
    VOCAB = [w for w in VOCAB if w in ONLY]
    print(f"[拆分] 这一份只跟踪 {len(VOCAB)} 个词: {', '.join(VOCAB)}", flush=True)
REPORT = os.environ.get("TRACK_REPORT", f"{ROOT}/03_track_report.json")
```

2. 结尾的 `json.dump(report, open(f"{ROOT}/03_track_report.json","w"), ensure_ascii=False, indent=1)` 换成:

```python
json.dump(report, open(REPORT,"w"), ensure_ascii=False, indent=1)
```

- [ ] **Step 4: `pipeline/fanout.py` 加 `fan_track`**

文件头的 import 改成 `from _common import GpuSlot, gpus_free_now, sh, envcmd, parallel_on`, 末尾追加:

```python
def fan_track(py, script, env, R1, log, dry, need_mb=0, max_parts=None, **kw):
    """物体跟踪按词拆分。env: 给 track_all.py 的环境变量(FR / R1 / EXTRA_WORDS ...)。max_parts: 最多拆几份。
    每个词的结果是单独一个 npz, 不用合并; 报告各份写各的, 这里并成一份。
    ★ 重试只补没出来的词: track_all.py 见到 npz 已经在就跳过。"""
    probe = {k: v for k, v in kw.items() if k in ("run_dir", "pool", "busy")}
    words = [] if dry else track_words(R1, env.get("EXTRA_WORDS", ""))
    k = how_many(len(words), need_mb, **probe) if (words and parallel_on()) else 1
    k = min(k, max_parts) if max_parts else k
    if dry or k <= 1:
        if dry:
            return sh(envcmd(py, script, env), log, dry)
        with GpuSlot("track", need_mb, **kw):
            return sh(envcmd(py, script, env), log, dry)
    groups = split(words, [1] * len(words), k)
    print(f"    [拆分] {len(words)} 个词分成 {len(groups)} 份: " + " | ".join(", ".join(g) for g in groups), flush=True)
    parts = [f"{R1}/03_track_report.part{i}.json" for i in range(len(groups))]

    def job(i):
        return lambda card: sh(envcmd(py, script, dict(env, CUDA_VISIBLE_DEVICES=card, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                                                       ONLY_WORDS=",".join(groups[i]), TRACK_REPORT=parts[i])), log, dry)

    res = run_workers([(f"w{i}", job(i)) for i in range(len(groups))], "track", need_mb, **kw)
    merge_reports(parts, f"{R1}/03_track_report.json")
    bad = {n: e for n, e in res.items() if e}
    if bad:
        raise SystemExit(f"[失败] track 有 {len(bad)} 份重试后还是没跑成({'; '.join(bad.values())})。"
                         f"没出结果的词: {track_words(R1, env.get('EXTRA_WORDS', ''))}")
```

- [ ] **Step 5: 改 `pipeline/outdoor_pipeline.py`**

1. import 后面加 `import fanout`。
2. `s_track` 整个换成:

```python
def s_track(c, dry):
    """SAM3 视频跟踪, 每个词一个结果文件, RLE 压缩落盘。跟踪只跑这一次,
    后面挑图/量盒子/逐实例生成全部从 03_masks/*.npz 读, 不再碰 GPU。
    防 OOM 三级都在 track_all.py 里(显存卸到 CPU / 密集词提高检出阈值 / 分段)。
    ★ 2026-09-29 按词拆分: 有几张空卡就起几份 track_all.py, 每份跟踪一部分词(fanout.fan_track)。"""
    fanout.fan_track(PY["sam3"], f"{SCRIPTS}/track_all.py",
                     dict(FR=c.FR, R1=c.R1, EXTRA_WORDS=c.cfg["EXTRA_WORDS"]),
                     c.R1, f"{c.LOG}/03_track.log", dry)
```

3. 模块级的 `GPU_STEPS = {...}` 后面加:

```python
# 内部拆分的步骤: 它起的每一份各领一张卡, 主循环不替它领(领了就多占一张不用)
FANOUT_STEPS = {"track"}
```

4. `main()` 的循环里 `if k in GPU_STEPS and not g.dry_run:` 换成 `if k in GPU_STEPS - FANOUT_STEPS and not g.dry_run:`。

- [ ] **Step 6: 跑测试, 确认通过**

Run: `cd ~/sam3d && $PY tools/tests/test_fanout.py && $PY tools/tests/test_sched.py`
Expected: `test_fanout.py` 14 行 `ok`; `test_sched.py` 照旧全过(`--dump-deps` 的 `gpu` 里仍然有 `track`)。

- [ ] **Step 7: 冒烟: 干跑打印的命令没变**

Run: `cd ~/sam3d && $PY pipeline/outdoor_pipeline.py --scene nfl1 --run r1 --only track -n | grep track_all`
Expected: 一行 `$ FR=... R1=... EXTRA_WORDS=... .../envs/sam3/bin/python .../track_all.py    > 03_track.log`, 没有 `ONLY_WORDS`。

- [ ] **Step 8: 提交**

```bash
cd ~/sam3d && git add data/outdoor/scripts/track_all.py pipeline/fanout.py pipeline/outdoor_pipeline.py tools/tests/test_fanout.py && git commit -q -m "track 按词拆分: 有几张空卡起几份, 每份跟踪一部分词

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: 出人网格按人拆分

**Files:**
- Create: `pipeline/mesh_merge.py`
- Create: `tools/tests/test_mesh_merge.py`
- Modify: `pipeline/body4d_pipeline.py`(`s_mesh`、`GPU_STEPS` 后面、`main` 的循环、import)
- Modify: `tools/tests/run.sh`

**Interfaces:**
- Consumes: `fanout.split` / `how_many` / `run_workers`; `_common.GpuSlot` / `parallel_on` / `sh` / `envcmd`。
- Consumes: `sam-body4d/body4d_mesh_5090.py` 现有的参数 `--dir` `--out` `--ids` `--completion` `--batch-size` `--max-occ-len` `--cam-json`(不改这个脚本)。它在 `--out` 目录里写: `mesh_4d_individual/<本份里的编号>/*.ply`、`focal_4d_individual/<编号>/*.json`、`rendered_frames_individual/<编号>/`、`rendered_frames/*.jpg`、`masks/*.png`、`relabel.json`(`{"relabel": {"原 id": 本份里的编号}}`)、`mesh_report.json`。
- Produces: `python pipeline/mesh_merge.py --run <出网格目录> --trk <跟踪目录> --ids 1,2,3 --parts <K> --seconds <秒>`; 也可以 `import mesh_merge; mesh_merge.merge(run, trk, ids, parts, seconds)`。合并后 `<出网格目录>` 的结构和不拆时一样, `_part*` 目录删掉。
- Produces: `<出网格目录>/_part<k>/part.json`: `{"ids": [...], "args": [...], "track_mtime": 浮点数}`。

- [ ] **Step 1: 写失败的测试**

`tools/tests/test_mesh_merge.py`:

```python
#!/usr/bin/env python
"""出人网格按人拆开跑之后并回一个目录(pipeline/mesh_merge.py)。假的几份: 小图 + 空网格文件。
跑: $PY tools/tests/test_mesh_merge.py"""
import json, os, sys, tempfile
import numpy as np
from PIL import Image

SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import mesh_merge as M

H, W, NF = 20, 30, 3
WHERE = {1: (slice(0, 5), slice(0, 5)), 2: (slice(5, 10), slice(5, 10)), 4: (slice(10, 15), slice(10, 15)),
         7: (slice(15, 20), slice(20, 30))}


def _part(root, k, ids, frames_of):
    """照 body4d_mesh_5090.py 的样子造一份: 本份里的人重编号成 1..n。"""
    p = f"{root}/_part{k}"
    rel = {o: i + 1 for i, o in enumerate(ids)}
    for o, l in rel.items():
        for d, ext in (("mesh_4d_individual", "ply"), ("focal_4d_individual", "json"), ("rendered_frames_individual", "jpg")):
            os.makedirs(f"{p}/{d}/{l}")
            for f in frames_of[o]:
                open(f"{p}/{d}/{l}/{f:08d}.{ext}", "w").write(f"人{o} 帧{f}")
    os.makedirs(f"{p}/masks"); os.makedirs(f"{p}/rendered_frames")
    for f in range(NF):
        m = np.zeros((H, W), np.uint8)
        img = np.full((H, W, 3), 255, np.uint8)
        for o, l in rel.items():
            if f in frames_of[o]:
                m[WHERE[o]] = l
                img[WHERE[o]] = (10 * o, 20, 30)
        q = Image.fromarray(m).convert("P"); q.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * 759); q.save(f"{p}/masks/{f:08d}.png")
        Image.fromarray(img).save(f"{p}/rendered_frames/{f:08d}.jpg", quality=95)
    json.dump({"relabel": {str(o): l for o, l in rel.items()}}, open(f"{p}/relabel.json", "w"))
    json.dump({"ids": ids, "frames": NF, "completion": False, "batch_size": 8, "peak_vram_gb": 9.0 + k, "seconds": 100.0},
              open(f"{p}/mesh_report.json", "w"))
    return p


def _setup():
    T = tempfile.mkdtemp()
    run, trk = f"{T}/04_mesh_occ0", f"{T}/02_body4d"
    os.makedirs(f"{trk}/images")
    for f in range(NF):
        open(f"{trk}/images/{f:08d}.jpg", "w").close()
    frames_of = {1: [0, 1, 2], 2: [0, 1], 4: [1, 2], 7: [2]}
    _part(run, 0, [1, 7], frames_of); _part(run, 1, [2, 4], frames_of)
    return run, trk, frames_of


def test_merged_layout_matches_an_unsplit_run():
    run, trk, frames_of = _setup()
    M.merge(run, trk, [1, 2, 4, 7], 2, seconds=123.4)
    G = {1: 1, 2: 2, 4: 3, 7: 4}                                 # 全体统一编号: 按挑人的顺序 1..N
    for o, g in G.items():
        for d, ext in (("mesh_4d_individual", "ply"), ("focal_4d_individual", "json"), ("rendered_frames_individual", "jpg")):
            got = sorted(os.listdir(f"{run}/{d}/{g}"))
            assert got == [f"{f:08d}.{ext}" for f in frames_of[o]], (d, o, got)
            assert open(f"{run}/{d}/{g}/{got[0]}").read().startswith(f"人{o} "), "人搬错了目录"
    assert sorted(os.listdir(f"{run}/mesh_4d_individual")) == ["1", "2", "3", "4"]
    assert not [x for x in os.listdir(run) if x.startswith("_part")], os.listdir(run)
    assert os.path.islink(f"{run}/images") and len(os.listdir(f"{run}/images")) == NF


def test_masks_use_the_global_numbering():
    run, trk, frames_of = _setup()
    M.merge(run, trk, [1, 2, 4, 7], 2)
    m = np.array(Image.open(f"{run}/masks/00000002.png"))        # 第 2 帧: 人 1、4、7 在, 人 2 不在
    assert m[WHERE[1]].min() == m[WHERE[1]].max() == 1
    assert m[WHERE[4]].min() == m[WHERE[4]].max() == 3
    assert m[WHERE[7]].min() == m[WHERE[7]].max() == 4
    assert (m[WHERE[2]] == 0).all() and sorted(np.unique(m)) == [0, 1, 3, 4]
    assert Image.open(f"{run}/masks/00000002.png").mode == "P"


def test_rendered_frames_are_composited():
    run, trk, frames_of = _setup()
    M.merge(run, trk, [1, 2, 4, 7], 2)
    img = np.array(Image.open(f"{run}/rendered_frames/00000001.jpg")).astype(int)    # 第 1 帧: 人 1(第 0 份)、2 和 4(第 1 份)
    for o in (1, 2, 4):                                          # JPEG 在小块边上有振铃, 只判"这里有人(暗)"
        assert img[WHERE[o]].mean() < 128, (o, img[WHERE[o]].mean())
    assert img[WHERE[7]].mean() > 230                            # 人 7 这一帧不在: 还是白底


def test_reports_are_merged():
    run, trk, frames_of = _setup()
    M.merge(run, trk, [1, 2, 4, 7], 2, seconds=123.4)
    rel = json.load(open(f"{run}/relabel.json"))
    assert rel["relabel"] == {"1": 1, "2": 2, "4": 3, "7": 4}, rel
    rep = json.load(open(f"{run}/mesh_report.json"))
    assert rep["ids"] == [1, 2, 4, 7] and rep["ids_run"] == [1, 2, 3, 4] and rep["frames"] == NF
    assert rep["meshes_per_id"] == {"1": 3, "2": 2, "3": 2, "4": 1}, rep["meshes_per_id"]
    assert rep["peak_vram_gb"] == 10.0 and rep["seconds"] == 123.4 and rep["parts"] == 2


def test_old_outputs_are_cleared_first():
    """上一次挑了 5 个人, 这次只挑 4 个: 第 5 个人的旧目录不能留着。"""
    run, trk, frames_of = _setup()
    os.makedirs(f"{run}/mesh_4d_individual/5"); open(f"{run}/mesh_4d_individual/5/00000000.ply", "w").close()
    M.merge(run, trk, [1, 2, 4, 7], 2)
    assert sorted(os.listdir(f"{run}/mesh_4d_individual")) == ["1", "2", "3", "4"]


def test_missing_person_is_a_failure():
    run, trk, frames_of = _setup()
    try:
        M.merge(run, trk, [1, 2, 4, 7, 9], 2)
    except SystemExit as e:
        assert "9" in str(e), str(e)
    else:
        raise AssertionError("少了一个人该失败")
    assert os.path.isdir(f"{run}/_part0/mesh_4d_individual/1"), "失败时不能动各份的结果"


if __name__ == "__main__":
    for name, t in list(globals().items()):
        if name.startswith("test_") and callable(t):
            t(); print("ok", name)
```

- [ ] **Step 2: 跑测试, 确认失败**

Run: `cd ~/sam3d && $PY tools/tests/test_mesh_merge.py`
Expected: `ModuleNotFoundError: No module named 'mesh_merge'`

- [ ] **Step 3: 实现 `pipeline/mesh_merge.py`**

```python
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

PER_PERSON = ("mesh_4d_individual", "focal_4d_individual", "rendered_frames_individual")
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
    # 先清掉上一次留下的(上次挑的人可能更多), 再把各份的搬进来
    for d in PER_PERSON + ("rendered_frames", "masks"):
        shutil.rmtree(f"{run}/{d}", ignore_errors=True)
    for p, rel, _ in got:
        for d in PER_PERSON:
            for o, l in rel.items():
                if os.path.isdir(f"{p}/{d}/{l}"):
                    os.makedirs(f"{run}/{d}", exist_ok=True)
                    shutil.move(f"{p}/{d}/{l}", f"{run}/{d}/{G[o]}")
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
```

- [ ] **Step 4: 跑测试, 确认通过**

Run: `cd ~/sam3d && $PY tools/tests/test_mesh_merge.py`
Expected: 6 行 `ok test_...`

- [ ] **Step 5: 改 `pipeline/body4d_pipeline.py`**

1. import: `from _common import (...)` 里加上 `parallel_on`; 下面加两行:

```python
import fanout

PIPE = os.path.dirname(os.path.abspath(__file__))
```

2. `s_mesh` 整个换成:

```python
def _part_ok(part, meta):
    """这一份上次已经算好、而且算的是同样的东西(人、参数、跟踪结果都没变): 直接用, 不重算。"""
    try:
        return json.load(open(f"{part}/part.json")) == meta and os.path.exists(f"{part}/mesh_report.json")
    except (OSError, ValueError):
        return False


def s_mesh(c, dry):
    """SAM 3D Body 出网格。OCC=1 时先跑 Diffusion-VAS 补全 (32G 上会 OOM, 见文件头)。
    ★ 2026-09-29 按人拆分: 有几张空卡就起几份, 每份算几个人(按每人出现的帧数分, 各份工作量接近),
      跑完并回一个目录(mesh_merge.py)。只在补全关着的时候拆 —— 补全要看别人的遮挡, 不能按人拆开。"""
    g = c.cfg
    need = 26000 if g["OCC"] == "1" else 15000
    script = f"{B4D}/body4d_mesh_5090.py"
    log = f"{c.LOG}/04_mesh.log"
    args = (["--dir", c.TRK, "--completion", g["OCC"], "--batch-size", g["BATCH"], "--max-occ-len", g["MAX_OCC_LEN"]]
            + (["--cam-json", f"{g['SCENE_R1']}/02_da3/cameras.json"] if g["SCENE_R1"] else []))
    picked = json.load(open(f"{c.TRK}/picked.json")) if os.path.exists(f"{c.TRK}/picked.json") else {}
    ids = picked.get("keep_ids", [])
    k = 1
    if not dry and g["OCC"] != "1" and parallel_on() and len(ids) > 1:
        k = fanout.how_many(len(ids), need)
    if k <= 1:
        if dry:
            return sh(envcmd(PY["body4d"], script, GPUENV, args + ["--out", c.MESH]), log, dry, cwd=B4D)
        with GpuSlot("mesh", need):
            return sh(envcmd(PY["body4d"], script, GPUENV, args + ["--out", c.MESH]), log, dry, cwd=B4D)
    cover = picked.get("coverage", {})
    groups = fanout.split(ids, [cover.get(str(i), 1) for i in ids], k)
    print(f"    [拆分] {len(ids)} 个人分成 {len(groups)} 份: "
          + " | ".join(",".join(map(str, grp)) for grp in groups), flush=True)
    t0 = time.time()
    tm = os.path.getmtime(f"{c.TRK}/track_report.json") if os.path.exists(f"{c.TRK}/track_report.json") else 0

    def job(i):
        part = f"{c.MESH}/_part{i}"
        meta = dict(ids=groups[i], args=args, track_mtime=tm)

        def fn(card):
            if _part_ok(part, meta):
                print(f"    [拆分] 第 {i} 份(人 {groups[i]})上次已经算好, 直接用", flush=True)
                return
            sh(envcmd(PY["body4d"], script, dict(GPUENV, CUDA_VISIBLE_DEVICES=card, CUDA_DEVICE_ORDER="PCI_BUS_ID"),
                      args + ["--out", part, "--ids", ",".join(map(str, groups[i]))]), log, dry, cwd=B4D)
            json.dump(meta, open(f"{part}/part.json", "w"), ensure_ascii=False)
        return fn

    res = fanout.run_workers([(f"p{i}", job(i)) for i in range(len(groups))], "mesh", need)
    bad = [int(n[1:]) for n, e in res.items() if e]
    if bad:
        raise SystemExit(f"[失败] 出网格有 {len(bad)} 份重试后还是没跑成。没算出来的人: "
                         f"{sorted(x for i in bad for x in groups[i])}。算好的几份留着, 重跑这一步只补没算出来的")
    sh([PY["sam3d"], f"{PIPE}/mesh_merge.py", "--run", c.MESH, "--trk", c.TRK, "--ids", ",".join(map(str, ids)),
        "--parts", str(len(groups)), "--seconds", f"{time.time() - t0:.1f}"], log, dry)
```

注意分份的顺序: `fanout.split` 在人数和份数不变时结果是确定的, 所以重跑时同一份里还是同样的人, `part.json` 对得上。空卡数变了 → 份数变了 → 各份的人变了 → `part.json` 对不上 → 重算。这是对的。

3. `GPU_STEPS = {"track", "mesh"}` 后面加:

```python
# mesh 内部按人拆分, 它起的每一份各领一张卡(s_mesh 里), 主循环不替它领
FANOUT_STEPS = {"mesh"}
```

4. `main()` 的循环里 `if k in GPU_STEPS and not g.dry_run:` 换成 `if k in GPU_STEPS - FANOUT_STEPS and not g.dry_run:`。

- [ ] **Step 6: 跑测试和干跑**

Run: `cd ~/sam3d && $PY tools/tests/test_mesh_merge.py && $PY tools/tests/test_sched.py && BODY_OUT=$HOME/sam3d/data/outdoor/result/nfl1/r1/body4d SCENE_R1=$HOME/sam3d/data/outdoor/result/nfl1/r1 $PY pipeline/body4d_pipeline.py --scene nfl1 --run r1 --only mesh -n | grep body4d_mesh`
Expected: 测试全过; 干跑打印一行命令, 里面有 `--out .../04_mesh_occ0`, 没有 `--ids`。

- [ ] **Step 7: 加进 run.sh, 提交**

`tools/tests/run.sh` 第 5 行末尾追加 `; $PY test_mesh_merge.py`。

```bash
cd ~/sam3d && bash tools/tests/run.sh 2>&1 | tail -1 && git add pipeline/mesh_merge.py pipeline/body4d_pipeline.py tools/tests/test_mesh_merge.py tools/tests/run.sh && git commit -q -m "出人网格按人拆分: 各份按帧数分组同时跑, 跑完并回一个目录

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: nfl1 实跑验证(旧方式 r2 对 新方式 r3), 再跑 soccer

**Files:**
- Create: `tools/compare_runs.py`
- Create: `tools/run_to_end.sh`
- Create: `docs/superpowers/specs/2026-09-29-parallel-scheduling-results.md`

**Interfaces:**
- Consumes: 前面全部任务。
- Produces: `$PY tools/compare_runs.py <轮次A> <轮次B>` 打印 track 和出人网格的差别, 最后一行 `TRACK 一致|不一致  MESH 最大差 <毫米>`; `bash tools/run_to_end.sh <场景> <轮次> [环境变量...]` 把一轮从头跑到最后一道闸, 闸口按 nfl1 r1 的决定自动过。

这一步要用显卡, 两轮加起来约 40 分钟。先确认 8 张卡是空的: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`。

- [ ] **Step 1: 写对比脚本 `tools/compare_runs.py`**

```python
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
```

- [ ] **Step 2: 写跑到底的脚本 `tools/run_to_end.sh`**

闸口的选择照抄 nfl1 r1: 第 2 道闸把 `football` `gloves` `helmet` `jersey` `referee` 五个词整词作废, 人用默认勾选(帧数够的全勾); 其他闸不改。

```bash
#!/usr/bin/env bash
# 把一轮从头跑到最后一道闸, 闸口自动过(验证并行调度用, 不是日常用法 —— 日常是人在网页上看闸口)。
#   tools/run_to_end.sh <场景> <轮次> [环境变量=值 ...]      例: tools/run_to_end.sh nfl1 r2 PARALLEL=0
# 第 2 道闸按 DROP_WORDS 把词整词作废(默认是 nfl1 r1 当时作废的那五个), 其他闸不改决定。
set -eu
SC=$1; RUN=$2; shift 2
for kv in "$@"; do export "$kv"; done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=$HOME/miniforge3/envs/sam3d/bin/python
R1=$HOME/sam3d/data/outdoor/result/$SC/$RUN
DROP_WORDS=${DROP_WORDS-football,gloves,helmet,jersey,referee}      # 传空字符串 = 一个词都不作废
[ -e "$R1" ] && { echo "轮次已经存在, 不覆盖: $R1"; exit 1; }
mkdir -p "$R1/logs"
$PY -c "import sys; sys.path.insert(0, '$HOME/sam3d/pipeline'); import _common; _common.update_run_config('$R1', 'outdoor', {'PEOPLE': '1', 'RECALL': '0'}, 0)"
T0=$(date +%s)
while true; do
  LOG=$(bash "$HERE/drive_go.sh" "$SC" "$RUN" | sed -n 's/^已继续: .* 日志 //p')
  [ -n "$LOG" ] || { echo "★ drive_go.sh 没开跑"; exit 1; }
  # 等这一段跑完: 看日志里的 [结束]。不靠 pgrep —— 驱动还没起来时 pgrep 查不到, 会当成已经跑完
  until [ -e "$LOG" ] && grep -q "^\[结束\]" "$LOG"; do sleep 5; done
  grep -q "^\[结束\] .* 退出码 0$" "$LOG" || { echo "★ 失败, 看 $LOG"; tail -5 "$LOG"; exit 1; }
  read -r SEG STOP < <($PY -c "import json; s=json.load(open('$R1/logs/.drive_state.json')); print(s['段'], int(s['停在闸口']))")
  echo "[$(( $(date +%s) - T0 ))s] 停在第 $((SEG + 1)) 道闸"
  grep -q "全部跑完" "$LOG" && break
  [ "$STOP" = 1 ] || { echo "★ 没停在闸口也没跑完, 看 $LOG"; exit 1; }
  if [ "$SEG" = 1 ]; then
    $PY - "$R1" "$DROP_WORDS" <<'EOF'
import json, sys
R1, words = sys.argv[1], set(sys.argv[2].split(","))
p = f"{R1}/05_review.json"; d = json.load(open(p)); seen = set()
for c in d["候选"]:
    if c["词"] in words and c["词"] not in seen:
        c["决定"] = "drop-word"; seen.add(c["词"])
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
print(f"    闸 2: 整词作废 {sorted(seen)}; 词表里没有的 {sorted(words - seen)}")
EOF
  fi
  LAST=$($PY -c "import sys; sys.path.insert(0, '$HOME/sam3d/.claude/skills/run-recon-pipeline'); import drive; print(len(drive.SEGMENTS['outdoor']) - 1)")
  [ "$SEG" = "$LAST" ] && break
done
echo "总用时 $(( $(date +%s) - T0 )) 秒(含闸口之间的切换, 不含人看的时间)"
column -t -s$'\t' "$R1/logs/timing.tsv"
```

`chmod +x tools/run_to_end.sh`

- [ ] **Step 3: 跑旧方式 r2**

```bash
cd ~/sam3d && bash tools/run_to_end.sh nfl1 r2 PARALLEL=0 2>&1 | tee /tmp/claude-run-r2.txt
```

Expected: 每过一道闸打印一行, 最后是 `总用时 ... 秒` 和 timing 表。用 `run_in_background` 跑, 约 25～30 分钟。失败了先看它报出来的那份驱动日志。

- [ ] **Step 4: 跑新方式 r3**

等 r2 跑完(两轮同时跑会互相抢卡, 耗时就比不了了):

```bash
cd ~/sam3d && bash tools/run_to_end.sh nfl1 r3 2>&1 | tee /tmp/claude-run-r3.txt
```

跑的过程中另开一个命令看一眼并行是不是真的发生了:

```bash
ls -a ~/sam3d/data/outdoor/result/nfl1/r3/logs/ | grep "^\.running_\|^\.gpu_"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

Expected: 第 1 段能同时看到 `.running_vlm` 和 `.running_da3`, 有两张卡在用。

- [ ] **Step 5: 对比**

```bash
cd ~/sam3d && R=data/outdoor/result/nfl1 && \
echo "##### 基线: r1(09-24, 不拆) 对 r2(今天, 不拆) —— 同一种做法跑两次本身差多少" && $PY tools/compare_runs.py $R/r1 $R/r2 | tail -14 && \
echo "##### 验证: r2(不拆) 对 r3(拆)" && $PY tools/compare_runs.py $R/r2 $R/r3 | tail -14
```

判断:

| 结果 | 结论 |
|---|---|
| r2 对 r3: `TRACK 一致`, MESH 最大差不超过基线(r1 对 r2)的最大差, 每人帧数一样 | 通过 |
| r2 对 r3 的 track 不一致, 但基线的 track 也不一致 | track 本身两次跑就不一样: 记下来, 比较实例数和每个实例的帧数, 差别和基线同一个量级就算通过 |
| 基线一致, r2 对 r3 的 track 不一致 | 按词拆分改变了结果(多半是"一批词共用检测"和批的组成有关)。track 不拆: `pipeline/outdoor_pipeline.py` 的 `s_track` 调 `fan_track` 时加 `max_parts=1`, 注释写明实测的差别。不用重跑 r3, 记下结论即可 |
| MESH 的差超过基线 | 按人拆分改变了结果。出人不拆: `body4d_pipeline.py` 的 `s_mesh` 里把 `k = fanout.how_many(len(ids), need)` 换成 `k = 1`, 注释写明实测的差别 |

- [ ] **Step 6: 耗时对比, 人工看视频**

```bash
cd ~/sam3d && $PY - <<'EOF'
import os
def rows(run):
    p = os.path.expanduser(f"~/sam3d/data/outdoor/result/nfl1/{run}/logs/timing.tsv")
    return {l.split("\t")[0]: float(l.split("\t")[4]) for l in open(p).read().splitlines()[1:] if l.split("\t")[6] == "ok"}
a, b = rows("r2"), rows("r3")
print(f"{'步骤':14s} {'r2 旧方式':>10s} {'r3 新方式':>10s}")
for k in a:
    print(f"{k:14s} {a[k]:10.1f} {b.get(k, float('nan')):10.1f}")
print(f"{'各步耗时相加':14s} {sum(a.values()):10.1f} {sum(b.values()):10.1f}   (并行时相加没有意义, 看下面的总用时)")
EOF
grep "总用时" /tmp/claude-run-r2.txt /tmp/claude-run-r3.txt
```

把 r3 的验收视频抽几帧看(方法同之前: cv2 抽帧存到 scratchpad, 用 Read 看图): 人都在、位置合理、没有人站在相机上。

- [ ] **Step 7: soccer 跑一轮新方式**

闸口的选择照抄 soccer r2: 作废的词从它当时落地的决定里读。

```bash
cd ~/sam3d && export DROP_WORDS=$($PY -c "import json; print(','.join(json.load(open('data/outdoor/result/soccer/r2/05_review_applied.json'))['排除词']))") && echo "作废的词: $DROP_WORDS" && bash tools/run_to_end.sh soccer r3 2>&1 | tee /tmp/claude-run-soccer-r3.txt && $PY tools/compare_runs.py data/outdoor/result/soccer/r2 data/outdoor/result/soccer/r3 | tail -8
```

Expected: 跑完; 对比结果和 nfl1 的结论一致。

- [ ] **Step 8: 写结果文档, 提交**

`docs/superpowers/specs/2026-09-29-parallel-scheduling-results.md`, 内容是实测的数字, 照这个结构填:

```markdown
# 并行调度: 实测结果(2026-09-29)

## 耗时(nfl1, 114 帧)
| | 旧方式 r2 (PARALLEL=0) | 新方式 r3 |
|---|---|---|
| 总用时(秒) | <Step 6 的数> | <Step 6 的数> |
| track | ... | ... |
| 出人网格 | ... | ... |

## 结果一致吗
| 对比 | track | 出人网格最大差(mm) |
|---|---|---|
| 基线 r1 对 r2 | ... | ... |
| r2 对 r3 | ... | ... |

结论: <通过 / 哪一步不拆, 为什么>

## soccer
<总用时, 对比结论>

## 拆了几份
<r3 的 03_track.log 和 04_mesh.log 里 "[拆分]" 那两行原文>
```

```bash
cd ~/sam3d && git add tools/compare_runs.py tools/run_to_end.sh docs/superpowers/specs/2026-09-29-parallel-scheduling-results.md pipeline/ && git commit -q -m "并行调度: nfl1 / soccer 实跑验证和实测结果

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: 室内流水线接上

**Files:**
- Modify: `pipeline/indoor_pipeline.py`(import、`o_track`、`main`、新函数 `_parallel`)
- Modify: `tools/tests/test_sched.py`
- Modify: `tools/run_to_end.sh`(支持 `--indoor`)
- Modify: `docs/superpowers/specs/2026-09-29-parallel-scheduling-results.md`

**Interfaces:**
- Consumes: `sched.check_graph` / `run_children`; `_common.GpuSlot` / `parallel_on`; `fanout.fan_track`。
- Produces: `indoor_pipeline.py --chain obj --only a,b,c` 并行跑, 规则和室外一样。

- [ ] **Step 1: 写失败的测试**

`tools/tests/test_sched.py` 的 `if __name__` 之前加:

```python
def test_indoor_child_mode_and_dry_run():
    base = [sys.executable, f"{SAM3D}/pipeline/indoor_pipeline.py", "--chain", "obj", "--scene", "office", "--run", "r6_pillarfix"]
    r = subprocess.run(base + ["--only", "vlm,da3,up", "-n"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    assert [l.split()[1] for l in r.stdout.splitlines() if l.startswith("== ")] == ["vlm", "da3", "up"]
    r = subprocess.run(base + ["--only", "up", "--child", "-n"], capture_output=True, text=True)
    assert r.returncode == 0 and "== up  [" in r.stdout and "[全部完成]" not in r.stdout, r.stdout[:200]
```

- [ ] **Step 2: 跑测试, 确认失败**

Run: `cd ~/sam3d && $PY tools/tests/test_sched.py`
Expected: 最后一个测试失败: `unrecognized arguments: --child`

- [ ] **Step 3: 改 `pipeline/indoor_pipeline.py`**

1. import: `from _common import (...)` 里 `GpuLock` 换成 `GpuSlot`; 加 `import sched` 和 `import fanout`(`import _common as _cm` 已经有就不用加)。

2. `o_track` 里, 把

```python
    wait_gpu(23000, dry)
    sh(envcmd(PY["sam3"], f"{SCRIPTS}/track_all.py",
               dict(**_base(c), EXTRA_WORDS=c.cfg["EXTRA_WORDS"])),
       f"{c.LOG}/03_track.log", dry)
```

换成

```python
    # ★ 2026-09-29 按词拆分: 有几张空卡就起几份(fanout.fan_track)。原来的 wait_gpu(23000) 变成领卡时的余量要求
    fanout.fan_track(PY["sam3"], f"{SCRIPTS}/track_all.py",
                     dict(**_base(c), EXTRA_WORDS=c.cfg["EXTRA_WORDS"]),
                     c.R1, f"{c.LOG}/03_track.log", dry, need_mb=23000)
```

3. 模块级 `GPU_STEPS = {...}` 后面加:

```python
# 内部拆分的步骤: 它起的每一份各领一张卡, 主循环不替它领
FANOUT_STEPS = {"track"}
```

4. `def main():` 之前加(和室外的 `_parallel` 只差 `--chain`):

```python
def _parallel(g, c, names):
    """一段里能同时跑的步骤同时跑: 每个步骤起一个子进程(--only <步骤> --child)。和 outdoor_pipeline._parallel
    是同一回事, 只多带一个 --chain。规则和理由见 sched.py。"""
    sched.check_graph([s[0] for s in STEPS], DEPS)
    tsv = f"{c.LOG}/timing.tsv"
    if not os.path.exists(tsv):     # 表头先写好: 几个子进程同时发现"文件不存在"会各写一遍表头
        with open(tsv, "w") as f:
            f.write("步骤\t环境\t开始\t结束\t耗时秒\t等GPU秒\t结果\n")
    me = os.path.abspath(__file__)
    rc = sched.run_children(
        names, DEPS,
        lambda k: [sys.executable, "-u", me, "--chain", g.chain, "--scene", g.scene, "--run", g.run,
                   "--only", k, "--child"],
        c.LOG, max_parallel=8, env=dict(os.environ, RUN_DIR=c.R1))
    bad = [k for k in names if rc.get(k, 0) != 0]
    if bad:
        left = [k for k in names if k not in rc]
        raise SystemExit(f"[失败] 这些步骤失败: {bad}" + (f"; 没轮到跑的: {left}" if left else ""))
    print("[全部完成]")
```

5. `main()`: 参数加 `ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)`。从 `os.makedirs(c.LOG, exist_ok=True)` 到打印 `步骤 ...` 那一行, 这几行横幅全部包进 `if not g.child:`; 在横幅之前加 `os.environ.setdefault("RUN_DIR", c.R1)`; 横幅之后、循环之前加:

```python
    if len(sel) > 1 and _cm.parallel_on() and not g.dry_run:
        return _parallel(g, c, [s[0] for s in sel])
```

循环里 `if k in GPU_STEPS and not g.dry_run:` 换成 `if k in GPU_STEPS - FANOUT_STEPS and not g.dry_run:`, `with GpuLock():` 换成 `with GpuSlot(k):`。结尾的 `print("[全部完成]")` 换成:

```python
    if not g.child:
        print("[全部完成]")
```

6. 三条流水线都不再用 `GpuLock` 了: `grep -rn "GpuLock" pipeline/ tools/ .claude/skills/ --include=*.py --include=*.sh | grep -v "\.bak"`。除了 `_common.py` 里的定义和注释里提到的, 没有别的使用者就把 `_common.py` 里的 `class GpuLock` 整个删掉, `gpulock.sh` 第 2 行注释里的 `GpuLock` 改成 `GpuSlot`。

- [ ] **Step 4: 跑测试, 确认通过**

Run: `cd ~/sam3d && $PY tools/tests/test_sched.py && bash tools/tests/run.sh 2>&1 | tail -1`
Expected: `test_sched.py` 17 行 `ok`; `== tools/tests 全过 ==`

- [ ] **Step 5: `tools/run_to_end.sh` 支持室内**

脚本开头 `SC=$1; RUN=$2; shift 2` 后面加:

```bash
KIND=outdoor; IN=""
if [ "${1:-}" = "--indoor" ]; then KIND=indoor; IN="--indoor"; shift; fi
```

`R1=...` 那行换成:

```bash
if [ "$KIND" = indoor ]; then R1=$HOME/sam3d/data/indoorOutput/$SC/$RUN; else R1=$HOME/sam3d/data/outdoor/result/$SC/$RUN; fi
```

`update_run_config('$R1', 'outdoor', ...)` 里的 `'outdoor'` 换成 `'$KIND'`; `bash "$HERE/drive_go.sh" "$SC" "$RUN"` 后面加 ` $IN`; `drive.SEGMENTS['outdoor']` 换成 `drive.SEGMENTS['$KIND']`。

- [ ] **Step 6: office 新开一轮实跑**

闸口的选择照抄 office r6_pillarfix: 作废的词从它当时落地的决定里读。

```bash
cd ~/sam3d && export DROP_WORDS=$($PY -c "import json; print(','.join(json.load(open('data/indoorOutput/office/r6_pillarfix/05_review_applied.json'))['排除词']))") && echo "作废的词: $DROP_WORDS" && bash tools/run_to_end.sh office r7_parallel --indoor 2>&1 | tee /tmp/claude-run-office-r7.txt
```

Expected: 跑到最后一道闸; `总用时` 打印出来。和 `r6_pillarfix` 的 `logs/timing.tsv` 各步耗时对比, 把 office 的验收视频抽几帧看。

室内的帧目录是按轮次分的(`frames_<轮次>`)。如果第 1 段在抽帧那步失败, 看报错: 多半是室内的轮次要先有 `00_run_config.json` 以外的准备。这时改用网页的开跑页(`run_ctl.new_run('indoor', 'office', 'r7_parallel', '1', '0')`)建轮次, 再从 `run_to_end.sh` 的循环部分接着跑。

- [ ] **Step 7: 结果写进文档, 提交**

`docs/superpowers/specs/2026-09-29-parallel-scheduling-results.md` 末尾加一节 `## office(室内)`: 总用时、各步耗时和 r6 的对比、拆了几份。

```bash
cd ~/sam3d && git add pipeline/ tools/run_to_end.sh tools/tests/test_sched.py docs/superpowers/specs/2026-09-29-parallel-scheduling-results.md && git commit -q -m "室内流水线接上并行调度; office 实跑验证

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
