---
name: run-recon-pipeline
description: 跑场景重建流水线, 室内室外同一个驱动 —— 分段自动执行, 每段结束停在一道人工闸, 把该看的数字和图摆出来等人决策再继续。当用户说"跑一下 carpark/beach/timesquare/cyberpunk""重建这个场景""跑流水线""继续跑""现在到哪一步了""审一下候选""人工把关"时使用。室内加 --indoor。
---

# 场景重建流水线驱动 (室内室外通用)

一条命令从头跑到尾,自动的部分自动跑,**机器判不了的地方停下来等人**。

驱动:`.claude/skills/run-recon-pipeline/drive.py`(路径相对 `~/sam3d/`)。
它包着 `pipeline/outdoor_pipeline.py` —— 那个是步骤执行器,不知道什么时候该停;
这个知道。

**为什么要分段停:** 地基(深度/天/尺度)错了后面全废,但一路跑到底要跑完才看得出来;
生成是最贵的一步,而"该不该生成这个候选"机器判不了 —— VLM 只判得了"是不是这个词",
判不了"抠得全不全"。

---

## 分段和闸

**室外**(六段):


## 用法

```bash
cd ~/sam3d
D=.claude/skills/run-recon-pipeline/drive.py
PY=~/miniforge3/envs/sam3d/bin/python

$PY $D --scene carpark            # 室外: 跑到下一道闸就停
$PY $D --scene carpark --go       # 人看过了, 继续下一段
$PY $D --scene carpark --status   # 现在停在哪
$PY $D --scene carpark --reset    # 从头开始
$PY $D --scene carpark -n         # 干跑, 只打印要执行的命令

$PY $D --scene cyberpunk --indoor          # 室内: 同一套用法, 加 --indoor
$PY $D --scene cyberpunk --indoor --status
```

**Agent 在闸口必须停下,把闸口列出的图给用户看,等用户的决定,不要自己 `--go` 过去。**
唯一例外是用户明确说了"一路跑到底"。

`--status` 实测:

```
场景 carpark  共 6 段
  1. 地基       frames vlm da3 up                                            ← 现在
  2. 生成前      track pick verify override
  3. 尺度       gen ratio meshes boxes vlm_sizes scale
  ...
```

### 闸 1 实测输出

```
闸 1/6 — 地基
  天的方向偏离老假设 29.19°, 地面拟合内点比例 0.962
  地面高度 -29.71 m
  相机 132 帧, 轨迹总长 9.49 m, 首尾 7.57 m
  词表 31 个: trash can, manhole cover, traffic cone, bollard, ...
```

轨迹首尾短于 5 米会额外报一行警告。这一条比任何算法改进都要紧:
实测 timesquare 走了 19.7 米效果最好,carpark 1.5 米、beach 0.9 米,
每个物体的视角张角只有 2.2° 和 0.1° —— 等于一百多张同一个视角的照片,
朝向判不出来,尺度也没有约束。看到警告就该考虑重拍。

### 闸 2:人工挑候选

闸 2 会调 `review_gate.py`,出一张灰底联络图 `05_review_sheet.jpg`(每格一个候选,
标着 `[编号] 词` 和 `class/recall #实例 f帧 决定`)和一个决策文件 `05_review.json`;
同时调 `reuse_table.py` 出一张复用表 `05_reuse_table.jpg`。
两张图粒度不同:联络图逐格判"**这一个**要不要生成",复用表横着比"**这一批**该不该长成同一个样子"。

**图和决策文件都按阶段分两份,谁也不覆盖谁**(阶段由驱动用 `REVIEW_STAGE=pre|post` 显式给):

| | 闸 2(召回前) | 闸 4(召回后) |
|---|---|---|
| 联络图 | `05_review_sheet.jpg` | `05_review_sheet_afterrecall.jpg` |
| 复用表 | `05_reuse_table.jpg` | `05_reuse_table_afterrecall.jpg` |
| 决策文件 | `05_review.json` | `05_review_afterrecall.json` |

想整个不要召回:删掉带 `_afterrecall` 的那份决策文件,就回到召回前的状态。
两份是**叠加**关系 —— 闸 2 那份打底,闸 4 那份覆盖,合成同一个 `05_review_applied.json`。

人看完图,在决策文件里把 `决定` 改成:

| 值 | 含义 |
|---|---|
| `keep` | 照常生成(默认) |
| `drop` | 这一条不生成 |
| `drop-word` | 整个词作废,等于加进 `EXCLUDE` |
| `solo` | 这个词逐实例生成 |
| `shared` | 这个词共用一个 mesh(`SOLO="*"` 场景把某个词捞回来的唯一通道) |

| `复用组 N` | 词内分成几组,每组共用一个 mesh(展开成"挪类别到 `词~N` + 整词复用") |
| `改类别` | 这个实例其实是另一个词,mask 不动只换标签 |

换候选另加 `"改用": [实例号, 帧号]`。它的键会按这一行最终出的是**类 mesh 还是专属 mesh**
自动写成 `词` 或 `词#实例号` —— 共用词写裸词(`s_override` 消费),逐实例写带 `#` 的
(`gen_per_instance` 消费)。**召回候选不支持"改用"**,它的抠图是按检测框现抠的,没有别的帧可换,
填了会当场硬停。

**改完直接 `--go` 就行,不用手动跑 `--apply`。** 驱动会自己落地决策:
跑 `review_gate.py --apply` 写出 `05_review_applied.json`,然后**比对落地前后哪几类决定变了**:

- 只动了 `drop`(不生成)→ 什么都不用重跑,生成那步当场读这份文件。
- 动了词级决定(`排除词`/`逐实例词`/`共用词`/`改类别`/`指定样本`)→ 重跑
  `relabel → pick → verify → override → gen → ratio → boxes → solo`。

**为什么要一直重跑到 `gen`/`boxes`:** 闸是回路不是通过点。填了"共用"的词,`pick` 不重挑就
没有类 mesh、又被 `SOLO_KEEP` 挡住不逐实例,组装时两头都找不到,**整个词从场景里消失且不报错**;
填了"改类别"的实例被 `relabel` 搬进 `词~N.npz`,`boxes` 不重跑的话盒子文件里还是老词,一样全没。

重跑不贵:抠图按**内容**判变没变(`imgio.imwrite_stable`),内容没变 mtime 就不动,
`gen`/`solo` 会照常"已有且不过期,跳过"。落地失败就地停。

> ★ 2026-09-09 之前这一环是断的:`--apply` 只把 `EXCLUDE=`/`SOLO=` **打印**出来等人
> 粘回驱动源码,而那个 json 没有任何脚本读。现在 `resolve_cfg` 直接读它,
> 生效顺序是 `默认 ← 场景配置 ← 人工闸 ← 环境变量`。**别再手敲环境变量了。**

实测输出:

```
[决策] 共 67 个候选
  整个词作废 : ['bollard']
  单个不生成 : ['cp_r1_truck_10011']
  逐实例生成 : ['car']
  换候选     : {'car': [7, 60]}

下游用这些环境变量:
  EXCLUDE="$EXCLUDE,bollard"
  SOLO="car"
```

把打印出来的环境变量加在后续命令前面。

### 闸 3 实测输出

```
闸 3/6 — 尺度
  标定倍数 k = 6.4283, 一致率 0.778, 采用 11/12 类, 施加=True
      barrier          5个  量到 0.07 m  真实 1.00 m  倍数 15.39  <- 离群不用
      car             19个  量到 0.23 m  真实 1.50 m  倍数 6.65
    car: 21 个, 高度中位 1.47 m
    van: 6 个, 高度中位 2.04 m
```

`施加=False` 说明保险闸拦下来了,会一起打印拒绝原因。这时候不要硬推,
先看是哪个词的参照高度不对。

### 闸 5 实测输出

闸 5 会自己生成 `95_recall_check_2d.jpg`(召回结果逐个画在照片上),并打印:

```
闸 5/6 — 召回
  召回: 42 个和已有的对上, 4 个是漏检 (总 46)
    漏检按词: {'bus': 2, 'awning': 1, 'billboard': 1}
    IoU > 0.3:   0 个 —— SAM3 其实已经有了, 属于重复
    IoU = 0 (SAM3 那块完全空的): 1 个 —— 真漏检
```

光看"漏检多少个"是不够的:二维去重修好之前,timesquare 那 48 个"漏检"没有一个是真的,
每一个底下都压着已经抠出来的东西。所以这道闸一定要看图。

---

## 本轮配置 / 人体线 / 网页开跑(2026-09-24)

- **每轮一份 `<R1>/00_run_config.json`**:`{"类型": "outdoor", "覆盖": {"PEOPLE": "1", "RECALL": "0"}}`。
  "覆盖"就是持久化的 CFG 覆盖,写错键名直接报错。生效顺序
  `默认 ← SCENES ← 本轮配置 ← 人工闸决策 ← 环境变量`(`_common.resolve_cfg`)。
  写它的三处用同一套规则(`_common.update_run_config`):启动页 `/new`、本轮页 `/run`、
  `drive.py --people 0/1 --recall 0/1`。有没有人要在第 2 段之前定,召回要在第 4 段之前定,过了就拒绝。
- **人体线在主流程里**:`PEOPLE=1` 时第 2 段 `people_track`(SAM3 跟踪人,复用场景的帧和相机)、
  第 2 道闸挑人(网页 `/run#people`,文件 `05_people.json`)、第 3 段 `people_mesh`(尺度之前出网格)。
  `BODY_RUN` 自动指向 `<结果目录>/<场景>/<轮次>/body4d`(跟踪视频、网格叠回原片都在这);一个都没勾 = 这一轮没有人。
  跟踪只丢面积 < `PEOPLE_MIN_AREA`(0.1%)的碎片,其余全列出来、帧数够的默认勾上 ——
  soccer 的守门员就是被 body4d 默认的 1% 门槛静默丢掉的。过了闸 2 再改挑人,`--go` 会重跑出网格和摆人那几步。
  有本轮配置的新轮次不认 `SCENES` 里写死的 `BODY_RUN`;没有配置文件的老轮次(soccer r1、office)照旧。
- **网页能开跑了**:`gate_ui.py` 的 `/new`(新建一轮)和 `/run`(本轮页:配置、地基图、挑人、
  Start/Continue)。开跑只通过 `tools/drive_go.sh`(后台 tmux,会话名 `run_<场景>_<轮次>`),
  同一轮在跑就拒绝。页面在 `tools/run_page.py`,逻辑在 `tools/run_ctl.py`。
- **多卡并行**:GPU 锁是**每张卡一把**(`_common.lock_path()`,`/tmp/outdoor_pipeline_gpu<卡号>.lock`,
  shell 的 `gpulock.sh` 同一规则)。`drive_go.sh` 没拿到 `CUDA_VISIBLE_DEVICES` 就自动挑一张空卡
  (`run_ctl.pick_gpu()`:没有别的驱动在用、已用显存 < 2GB 的里面卡号最小的),网页开跑也走这条;
  想钉某张卡就 `CUDA_VISIBLE_DEVICES=3 tools/drive_go.sh <场景> <轮次>`。所以不同场景可以同时开跑、各占一张卡。
  本轮页状态条会显示这一轮在哪张卡上。

## Gotchas

- **召回默认关(2026-09-24 起)。** 两条 pipeline 的 CFG 都有 `RECALL="0"`:召回五步
  (`detect`/`recall`/`recall_mask`/`recall_gen`/`recall_merge`)照常盖完成戳但什么都不跑,
  组装和室内房间壳只认 `06_all_boxes.json`,驱动的第 4 道闸不停、它的决策也不落地。
  下面讲闸 4、召回联络图的段落只在召回打开时适用。要开:场景配置里写 `RECALL="1"`,
  或者临时用环境变量 `RECALL=1`。测试在 `tools/tests/test_recall_switch.py`。

- **场景名打错会被挡住。** `outdoor_pipeline.py` 拿默认配置照跑不误,干跑还看着像成功。
  驱动先检查 `data/outdoor/dataset/<场景>/` 存不存在,不存在就退出码 1 并列出现有场景。

- **进度存在 `logs/.drive_state.json`。** 换终端、隔天再来都接得上。想重来用 `--reset`。

- **召回的候选在闸 4 过目,不在闸 2。** 召回没法提到生成之前 —— `detect` 要吃 `scale`
  校准过的深度,提前就成环了。所以第 4 段(召回)末尾也出一张联络图,补抠回来的候选
  在那里过目,决定同样会落地。

- **分段表是第五个注册点。** 加一个步骤要改 `STEPS` / `ARTIFACTS` / `DEPS` / GPU 标记
  **加上 `drive.py` 的 `SEGMENTS`**。漏了最后这个不报错,只静默不跑(`dedup` 就这么
  漏过,跨词去重整个没生效)。现在启动时会自检:步骤名不存在、顺序和 `STEPS` 对不上,
  都会当场报错。段的实际内容以 `--status` 打印的为准,**这份文档不再抄一份**。

- **`-n` 干跑不会暴露真实错误。** 它只打印命令。要验证一个场景真能跑,得实跑。

- **重跑闸 2 不会冲掉已改的决定。** `review_gate.py` 按"生成名"合并旧决策,只追加新候选。
  候选因为自己的决定退出了表(标了 `shared` 就不在逐实例清单里),决定存在 `历史决定` 里继续生效。

- **决策填错退出码是 1。** 只认 `keep`/`drop`/`drop-word`/`solo`/`shared`。这一条要紧,
  否则流水线会当它成功继续往下跑。

- **尺度没施加时不会写 `world_scale.json`。** 一致率不达标或 `SCALE_APPLY=0` 时报告写进
  `world_scale_estimate.json`,并把旧的 `world_scale.json` 改名成 `.stale`。
  这样 `--check` 才会如实说"scale 没做完" —— 那个文件在,就代表世界真的校准过了。

- **联络图上的标签只能是 ASCII。** cv2 的字体画不了中文,写中文出来是一排问号。
  所以图上用 `class`/`recall`,中文只留在 JSON 里。

## Troubleshooting

| 现象 | 原因 / 处理 |
|---|---|
| 第 1 段 `vlm` 退出码 1 | 多半是 `frames/` 是空的。抽帧那步找的是 `dataset/<场景>/video.mp4`;素材如果叫别的名字(比如 `beach.mp4`)它会静默什么都不做。改名或建软链 |
| `[失败] 没有这个场景` | 场景名打错了,报错里会列出现有的 |
| 闸 3 `施加=False` | 保险闸拦下来了,看打印的拒绝原因。一致率低通常是某个词的参照高度被 VLM 说错了 |
| `[失败] 没找到任何待生成候选` | `pick` 没跑过,或 `TAG` 对不上 `sam-3d-objects/my_images/` 里的目录名 |
| CUDA out of memory | 显卡上有别的进程。`nvidia-smi` 看一眼,流水线内部有 flock 互斥但管不住外面的进程 |

---

## 闸 3 尺度 · 标准操作

**先看一个数决定要不要动手:**

| 一致率 | 怎么办 |
|---|---|
| ≥ 60% | 不用管,脚本自己施加,闸只是过目 |
| < 60% | 停下来,看下面那张表 |

**看表只判一列:"常识(m)"这个值配不配得上这个物体。**

判据一句话:**这东西矮而实心、高度定义明确吗?**

能当尺子:`trash can 0.80` `bollard 0.80` `soda machine 1.50` `sign 1.20` `person 1.70`
不能当尺子,三种情况:

| 例子 | 为什么不能 |
|---|---|
| `awning` / `umbrella` | 斜挂的一块布,"高度"对它没意义,量出来是投影厚度 |
| `tree` | 顶超出画面永远量小;而且树没有标准高度 |
| `billboard` | 只有 2 个样本,而且大小差十倍 |

**然后数好尺子说什么:**

- 好尺子都说 ≈1.0 → **不改**(`SCALE_APPLY=0`)
- 好尺子都说要放大 N 倍 → **照改**(放宽 `SCALE_AGREE_MIN`)
- **好尺子自己也打架** → 不是尺度问题,是**深度本身坏了**。回闸 1 看地基,或者重拍

**实例(beach 2026-09-09):** 好尺子 4 类 21 个实例全说 ≈1.0(trash can 1.00 / sign 0.96 / bollard 1.08 / soda machine 1.08),要放大的全是斜布和超框的 → 判定不改。

**反例(carpark):** trash can 6.15 / bollard 6.67 / soda machine 6.52 / sign 6.32 —— 好尺子一致说放大 6.4 倍 → 照改。那次一致率 78%,根本不会停。

---

## 闸 4 召回 · 类别继承三条规则

召回候选**排在最后**(排最前会让主线所有编号变一遍,人刚用过的号立刻失效),**边框标黄**。

| 情况 | 怎么办 |
|---|---|
| 这个词前面**有决定** | 继承(复用的继续复用) |
| 这个词前面**没决定** | 默认 solo,标黄等人判 |
| 这个词前面**被分过组**(`word~1/~2`) | **硬停**,等人指定进哪一组 |

**"没决定"包括"整个词被跨词去重清空了"。** 不要拿"当初和它重叠的那个词"的决定去顶 ——
去重判的是"**那一个**物体是同一个",不是"这两个词是同义词"。召回回来的是另一个物体。
(实例:`chair` 唯一那个实例被判成和 `bench#2` 是同一张椅子而清空;后来召回回来一把
新椅子,它该按新词走 solo,**不该**去用 bench 的 mesh。)

**为什么分过组要硬停:** 新来一个实例属于哪一组是看外观的视觉判断,机器猜不了。
猜错的代价是它长成别的组的样子,而且不报错。

---

## 分组复用 = 挪类别 + 整词复用

人在 sheet 上填"复用组 1/2/3",`--apply` 时**展开**成:
把这一组实例挪进派生子类 `umbrella~1`,再把 `umbrella~1` 整词复用。

所以**没有第三套机制** —— 生成那步、组装那步一行都不用改,sheet 上天然显示成
`umbrella~1/~2/~3` 三行粉框,和 bench 一模一样。

派生词对查表的东西是陌生的(常识高度、优先级表、排除表都是精确匹配),
所以 `word_rank.base_word()` 提供母词回退:`umbrella~1` 查不到就按 `umbrella` 查。

---

## 两条不许违反的规矩

**1. 决定只改标签,不许让候选从表里消失。**
第一版把标了 shared 的词从 sheet 拿掉,后果是双重的:人看不见,而且**决定会自我抹除**
(决定把候选移出表 → 下次重算时表里没它 → 决定蒸发 → 那批物体既不逐实例也没类 mesh,
整个从场景消失且不报错)。

**2. 图上的文字一律 ASCII。** cv2 的默认字体画不出中文,会变成一排问号,而且不报错。
`_SRC` / `_TY` 两张映射表负责中文→英文;加新标签必须同步加进去。

---

## 脚本都在 `tools/`

闸口那几张图的脚本**不在这个目录里**,在 `~/sam3d/tools/`:

| 脚本 | 闸 | 出什么 |
|---|---|---|
| `viz_ground.py` | 1 | 地基核对图 |
| `review_gate.py` | 2 / 4 | 联络图 + 决策文件 |
| `reuse_table.py` | 2 / 4 | 复用表 |
| `viz_recall_2d.py` | 4 | 补漏核对图 |

`drive.py` 用 `TOOLS = f"{SAM3D}/tools"` 指过去。想手动跑某一个:

```bash
cd ~/sam3d
R=data/outdoor/result/beach/r2
# 配置从 pipeline 拿, 别自己抄 —— 抄一份就会漂移
eval $(~/miniforge3/envs/sam3d/bin/python pipeline/outdoor_pipeline.py \
        --scene beach --run r2 --dump-cfg | \
        ~/miniforge3/envs/sam3d/bin/python -c "
import json,sys,shlex; d=json.load(sys.stdin)
for k in ('SOLO','SOLO_KEEP','SOLO_MIN_PX','EXCLUDE'): print(f'export {k}={shlex.quote(str(d[k]))}')")
R1=$PWD/$R TAG=be_r2 FR=$PWD/data/outdoor/dataset/beach/frames \
  ~/miniforge3/envs/sam3d/bin/python tools/review_gate.py
```

**为什么集中在 `tools/`:** 这些图经常看、经常改,散在三个目录里找不着。
`tools/README.md` 把 18 个脚本分"流程自动调"和"手动排查"两类列全了。

**不用软链接。** 软链接会让 `__file__` 解析到别处,那几个 import 算法模块
(`world_frame` / `word_rank` / `review_decisions`,它们仍在 `data/outdoor/scripts/`)
的脚本会找不到路 —— 实测踩过。引用方改代码指过来,一份实现。
