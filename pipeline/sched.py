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
