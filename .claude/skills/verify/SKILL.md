---
name: verify
description: 在真的网页上验证闸口网页 / 流水线的改动: 用无头浏览器当一回员工, 从新建一轮点到下载 FBX。改了 tools/gate_ui.py、run_page.py、run_ctl.py、drive.py 或想确认"整条路还通不通"时用。
---

# 怎么在真网页上走一遍

网页服务平时就开着(tmux 会话 `gate_ui`, 只听 127.0.0.1:8765)。**不要重启它**, 更不要 `pkill -f gate_ui`。

`web.sh` 是点网页的脚本(playwright + 自带的 chromium, 在 sam3d 环境里):

```bash
V=.claude/skills/verify/web.sh
$V snap "/run?s=office&r=r12&i=1" out.png      # 打开本轮页, 打印页面上的状态, 截图
$V new indoor <场景> <有人0/1> out.png          # 在 /new 选场景、点"开始"
$V go  "/run?s=<场景>&r=r1&i=1" out.png         # 点"继续"
$V dl  "/run?s=<场景>&r=r1&i=1" <目录>          # 点下载链接, 存下 FBX
```

室外的轮次地址不带 `i=1`。

## 从头走一遍(约 25 分钟, 8 张卡)

1. 备份 `tools/.last_run`(驱动每次开跑都会改写它)。
2. 把视频放进 `data/indoor/dataset/<名字>.mp4`(名字只能是字母数字和 `_ . -`; 网页没有上传入口)。
   名字前 3 个字母别和已有场景一样, 否则标签撞车, 网页拒绝开跑。
3. `web.sh new` -> 第 1 道闸 -> `web.sh go` -> 第 2 道闸(候选页 `/?s=&r=&i=1` 点 `drop word` 再点 Save; 本轮页点人的卡片再点"保存挑人")
   -> 第 3 道闸 -> `go` -> 第 5 道闸 -> `web.sh dl`。
4. 等一段跑完: 轮询 `/api/run/state?s=&r=&i=1` 的 `在跑`, 别用 `pgrep -f`。
5. FBX 用 `~/venvs/bpy/bin/python` 里的 `bpy.ops.import_scene.fbx` 打开, 数 MESH / ARMATURE / CAMERA。

## 清理

`data/indoor/dataset/<名字>*`、`data/indoorOutput/<名字>`、`sam-3d-objects/my_images/<标签>_*`、`sam-3d-objects/outputs/<标签>_*`,
还原 `tools/.last_run`。删之前先 `ls` 看一眼。

## 坑

- 截图里中文是方框: 无头浏览器没装中文字体, 不是网页的问题。页面文字用 `inner_text` 读。
- 候选页是英文的, 本轮页是中文的。
- 生成物体那步可能因显存不够失败(2026-10-12 实测过), 页面显示"第 4 段没跑完就停了", 再点一次继续只补没生成的。
- 第 1 段失败时页面显示"还没开跑", 没有失败提示; 原因只在 `<轮次>/logs/drive_*.log` 里。
