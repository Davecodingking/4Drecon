# 上游仓库的本地改动（2026-09-21 从 ES-MAL-GPU-011 导出）

代码是整目录（带 .git）带走的，新机器上**不需要**打这些补丁 —— 这里只是一份"我们在别人的仓库里改了什么"的说明。
进对应仓库 `git diff` 看到的是同样的内容。

文件名 = `<仓库目录>@<基于的 commit>.patch`；子模块用 `__` 连接路径。
只有在重新 clone 原版仓库时才需要：`git -C <仓库> checkout <commit> && git -C <仓库> apply ../patches/<文件>`

在用的：sam-3d-objects、sam-body4d、Depth-Anything-3
已弃用路线（留档）：SpatialGen（含子模块 Sparse-RaDeGS）、SpatialLM、InvSR、Pano2Room
PiSA-SR 的"改动"只是删掉的 __pycache__，没导出。

2026-10-08 起, 我们自己写在上游仓库目录里的脚本(sam-body4d 的 `body4d_*_5090.py`、`NOTES_5090.md`; sam-3d-objects 的
`bake_pool.py`、`bake_worker.py`)在对应仓库里 `git add -N` 过, 所以 `git diff` 会把它们当新文件带进快照 —— 不然它们哪里都没版本。
重出快照: `git -C <仓库> diff > patches/<文件>`。
