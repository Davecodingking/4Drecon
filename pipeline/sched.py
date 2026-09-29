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
