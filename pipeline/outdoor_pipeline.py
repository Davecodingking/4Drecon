#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""室外逐物体三维重建 —— 全流程调度脚本 (r1 版, plant pot 修复之后)

这个文件不实现算法, 只做三件散在别处、丢了就没人记得的事:
  1. 步骤的先后顺序 (依赖关系不是直觉上的顺序, 见下)
  2. 每一步该用哪个 conda 环境 (这条链跨了三个环境, 用错就 import 失败)
  3. 每一步的参数取值, 以及这些取值是怎么来的 (注释里写清楚)

算法本体仍在 ~/sam3d/data/outdoor/scripts/*.py, 每个都能单独重跑。
故意不把它们合并成一个大文件: 环境不同合不了, 而且调门槛时能只重跑一步这个性质很值钱。

------------------------------------------------------------------ 依赖顺序
词表 -> 深度 -> 跟踪 -> 挑图 -> VLM验图 -> 生成(每类一个) -> 量mesh比例
     -> 量盒子 -> 生成(逐实例) -> 组装 -> 物理修正 -> 渲染回图

反直觉的两处, 是踩过才知道的:
  * 量盒子(boxes) 要在 生成(gen) 之后。因为跨词去重时靠 mesh 的长宽高比例来判断
    "这个红花盆到底该叫 plant pot 还是 potted plant" —— 没有 mesh 就没有比例。
  * 逐实例生成(solo) 要在 量盒子 之后。要先有 06_all_boxes.json 才知道有哪些实例。

------------------------------------------------------------------ 一个潜伏很久的坑
调 SAM3D 的 run.py 必须显式设 CONDA_PREFIX (它内部 inference.py 直接读这个变量,
没有就 KeyError)。这个变量只有 conda activate 才会设, 直接调环境里的 python 是没有的。
这坑在驱动里潜伏了很久没暴露: gen / solo 两步每次都因为"产物已有, 跳过"而没真正调过
run.py, 直到第一次真的要生成新东西才炸。教训是"这步跑过了没报错"不等于"这步能跑"。

------------------------------------------------------------------ 哪些脚本是死的
scripts/ 里有 6 个文件不在这条链上, 别照着它们改:
  all_instances_boxes.py  已被 boxes_from_masks.py 取代 (首轮版, 门槛按帧数不按点数)
  pick_for_sam3d.py       已被 pick_from_masks.py 取代 (首轮版, 每次自己重跑一遍跟踪)
  sam3_video_probe.py     首轮试 SAM3 视频跟踪能不能用的探针
  sam3_cutouts.py         首轮看抠图效果的小工具
  mask_to_box.py          首轮的单张 mask 转框
上面五个都指向 result/timesquare/ 下已经删掉的目录, 现在直接跑会报文件不存在。
  diag_bld.py             还活着, 但不在链上 —— 查楼摆放的诊断工具, 手动跑

------------------------------------------------------------------ 用法
  python3 outdoor_pipeline.py --list                # 看所有步骤
  python3 outdoor_pipeline.py -n                    # 干跑, 只打印命令不执行
  python3 outdoor_pipeline.py --from boxes --to render
  python3 outdoor_pipeline.py --only phys
  python3 outdoor_pipeline.py --scene timesquare --run r1
"""
import os, sys, json, time, glob, argparse, subprocess, shlex

# ★ 驱动层的公共件搬到 _common.py 了(见那个文件的文件头): GPU 锁、跑命令、配置解析、
#   计时、帧号映射。留在本文件里的是室外真正独有的部分。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as _cm          # 设 CUR_STEP 用 —— 必须改模块上的属性, from-import 进来的是副本
import sched                   # pipeline/sched.py(不是标准库那个): 按依赖表并行跑步骤
import fanout                  # 步骤内部拆分: 同一个脚本起几份, 各用一张卡
from _common import (GpuSlot, wait_gpu, sh, envcmd, stamp_path,
                     timing as _timing, resolve_cfg as _resolve_cfg,
                     write_frame_map, load_frame_map)
import people_decisions           # 挑人闸的决策文件 + BODY_RUN 自动指向, 室内外共用

HOME   = os.path.expanduser("~")
SAM3D  = f"{HOME}/sam3d"
SCRIPTS= f"{SAM3D}/data/outdoor/scripts"
PIPE   = f"{SAM3D}/pipeline"                 # 两条链共用的步骤(body_into_scene / body_video)
TOOLS  = f"{SAM3D}/tools"      # 看结果的脚本都在这里, 常用的放一处好找
GEN    = f"{SAM3D}/sam-3d-objects"          # SAM 3D Objects 仓库
DA3REPO= f"{SAM3D}/Depth-Anything-3"

# 每个 conda 环境的 python。装了什么 (2026-09-10 实测):
#   sam3      sam3 包 + pycocotools + cv2 + torch        无 trimesh / pyrender
#   sam3d     pycocotools + trimesh + pyrender + cv2 + **transformers 4.57.1**
#   spatiallm depth_anything_3 在这里, 且只在这里
#
# ★ 2026-09-10: trellis2 这个环境被清理掉了(腾磁盘), VLM 那三步(vlm / verify / vlm_sizes)
#   随之全挂, 退出码 127。它在这条链上的唯一身份从来就是"装了 transformers 4.57.1 的那个
#   环境" —— 名字是历史遗留(当初为 TRELLIS.2 建的), 跟 TRELLIS 本身没关系。
#   所以不必重建那一整套, 把 transformers==4.57.1 装进 sam3d 就行:
#     * 不碰 torch —— Blackwell 的 2.8.0+cu128/sm_120 是调通过的, 重装等于把已解决的问题重解
#     * 不碰 huggingface_hub —— 不钉版本会把它推到 1.30, 而 gradio/moge/timm 依赖 0.36,
#       那才是真会连锁崩的地方。钉 4.57.1 之后它纹丝不动(实测 dry-run 确认)
#     * 净新增 4 个包(regex/tokenizers/transformers/accelerate), 零覆盖
#   trellis2 这个键保留但指向 sam3d, 免得步骤表和各处 envcmd 全要改。
PY = {
    "sam3":      f"{HOME}/miniforge3/envs/sam3/bin/python",
    "sam3d":     f"{HOME}/miniforge3/envs/sam3d/bin/python",
    "trellis2":  f"{HOME}/miniforge3/envs/sam3d/bin/python",   # ← 见上, 环境没了, 活挪进 sam3d
    "spatiallm": f"{HOME}/miniforge3/envs/spatiallm/bin/python",
    # WildDet3D 走 vis4d, 钉 pydantic<2 + numpy<2, 塞进上面任何一个都会打架, 必须独立
    "wilddet":   f"{HOME}/miniforge3/envs/wilddet/bin/python",
    # VGGT-Ω 几何后端(GEOM != da3 时用), 环境和权重都在 nvme 上。它内部还会调
    # /mnt/nvme/jincheng/envs/metricdepth 跑 Metric3D 定尺度, 那个路径写在脚本里
    "vggt_omega": "/mnt/nvme/jincheng/envs/vggt_omega/bin/python",
}
WD3REPO = f"{SAM3D}/WildDet3D"
VGGT_GEOM = f"{SAM3D}/vggt-omega/tools/vggt_omega_geom.py"

# ------------------------------------------------------------------ 参数
# 这些值都是调出来的, 改之前先看注释里写的"为什么是这个数"。
CFG = dict(
    NFR         = "8",      # VLM 扫几帧列词表。用户明确要求先用 8 帧
    MAX_FRAMES  = "180",    # DA3 用多少帧。深度帧越多, 每个实例可用的帧越多
    PROCESS_RES = "504",    # DA3 处理分辨率, 出来的深度图是 280x504
    GEOM        = "vggt_l", # 位姿+深度用哪个后端(da3 这一步), 产物格式完全一样, 下游不用改:
                            #   da3      Depth Anything 3 Nested(一直在用的)
                            #   vggt_l   VGGT-Ω + Metric3D v2 Large 定米制尺度(ithor312 尺度误差 0.6%)
                            #   vggt_g2  VGGT-Ω + Metric3D v2 Giant2(2.2%), 和 vggt_l 互为对照
                            # 为什么试 VGGT-Ω: ithor312 上位姿最差一帧 3.1° vs DA3 36.3°(devlog_9.16 第 9、10 节)。
                            # ★ 09-17 起默认 vggt_l。pedestrian 实测(对照 DA3 r2): 位姿朝向差 0.5°、位置差 2cm,
                            #   两边都 0 坏帧; 尺度比常识尺寸小 18%(DA3 原始小 25%), world_scale 那步照常再校;
                            #   显存峰值 21GB(DA3 24.8GB)。Large 和 Giant2 室外只差 1%, 取更快更省的 Large。
                            #   要回到老结果: GEOM=da3 python outdoor_pipeline.py ...
                            # ★ 置信度被换算成 DA3 刻度(vggt-omega/tools/build_conf_ref.py), CONF_MIN 可照用,
                            #   但没按 VGGT-Ω 重新调过; PROCESS_RES 对它无效(固定 512)。
    FX          = "",       # 已知焦距(原图宽度下的像素值); 空 = 让模型自己估。
                            # ★ 只在 GEOM != da3 时起作用。VGGT-Ω 自己不收内参, 但它后面的
                            #   Metric3D 是靠焦距把深度换成米的(深度 x fx x 系数), 焦距偏 10%
                            #   整个场景就一起大/小 10% —— 这把尺子完全架在这个数上。不填就用
                            #   VGGT-Ω 估的, 而 VGGT-Ω 本来就不做米制。
                            # ★ 现在兜得住不代表没事: 后面 world_scale 会按物体常识尺寸再校一次
                            #   (pedestrian 实测 k=1.21, 即 Metric3D 这轮给小了 21%), 但那要求
                            #   画面里有认得出的参照物。空路 + 楼这种没尺子的场景就兜不住, 错到底。
                            # ★ 室内 CFG 一直有这一项, 室外没有, 09-20 补齐。UE 那种已知镜头
                            #   可以直接填(7mm/23.76 -> 565.66)。
    MIN_FRAMES  = "2",      # 挑生成样本的下限, 只防单帧误检
    BAKE_WORKERS = "2",     # 生成 mesh 时另开几个进程烘焙贴图(run.py --bake-workers); 0 = 不拆。
                            # 32G 卡实测(8 个物体): 0 -> 257s, 2 -> 165s, 3 -> 180s 且有一个显存不够要重烘
    MIN_AREA_PCT= "0.05",   # 候选实例的最小面积占比(%)
    TOPK        = "8",      # VLM 每类最多验几个候选
    MIN_POINTS  = "300",    # 盒子门槛: 反投影合并后点数。首轮用"活够8帧"是错的 ——
                            # 近处的车 2 帧就上万点, 远处的车 20 帧也才几百点
    SAMPLE_FRAMES="8",      # 取样: 用面积最大的这么多帧。是取样不是门槛
    CONF_MIN    = "2.0",    # 深度置信度下限, 见 boxes_from_masks.py 里的注释
    VOX         = "0.15",   # 去重用的体素边长(米)
    DUP_TH      = "0.55",   # 双向包含都超过 = 同一个物体
    PART_TH     = "0.70",   # 单向包含超过 = 部件, 并进父体
    # 二维跨词去重(dedup 那步)。名字带 DW_ 前缀是为了和上面三维那套分开 —— 两边都叫
    # "包含率", 但一个量的是体素占用、一个量的是像素, 数值不可通用。
    DW_DUP_TH   = "0.55",   # 同一帧上双向包含都超过 = 这一帧算重复
    DW_PART_TH  = "0.70",   # 单向包含超过 = 可能是部件(只记录不合并, 留给三维那步)
    DW_PART_RATIO="0.55",   # 且小的面积不到大的这个比例才算部件。和 DW_DUP_TH 取同一个值:
                            # 完全包含时"较小的包含率"就等于面积比, 两个门槛是同一条线的
                            # 两侧, 取同值中间才不留既判不成重复也判不成部件的灰带
    DW_FRAME_FRAC="0.50",   # 共同帧里过半都判重复才算数。一帧说了不算: 俯视场景里两个
                            # 不相干的东西前后叠在一起, 某几帧确实会重得很厉害
    DW_WORD_FRAC= "0.60",   # 词对配对率门槛。分母取 max(两词实例数) 不取 min ——
                            # 见 dedup_words.py 里 car/taxi 那段注释
    SOLO        = "*",      # 逐实例生成的类; "*" = 除 SOLO_KEEP 外全部
                            # ★ 2026-09-10 默认从 "car" 改成 "*"。
                            #   共用一个 mesh 是特例, 不该是默认: 街上的人各不相同、车各不相同,
                            #   15 个人共用一个模型会变成满街同一件格子衬衫。
                            #   而且旧默认有个连带问题 —— 人在闸口把一个词标成 solo, 候选池
                            #   会从"1 个代表"炸成"N 个实例", pick/verify 得重跑, 而重跑出来的
                            #   那批新候选**没有再回到闸口给人看**就直接进生成了。
                            #   默认就是 "*" 的话候选池一次挑定, 不再重建, 人一次看全。
                            #   要共用的词写进 SOLO_KEEP。
    SOLO_KEEP   = "",       # 白名单: 这些类可以合并, 全类共用一个 mesh
    SOLO_MIN_PX = "256",    # 逐实例生成的面积下限(像素): 挑中那帧里实例少于这么多像素就不单独生成。
                            # 按像素不按画面百分比, 换分辨率不用再调; 为什么是 256 见 solo_gate.py。
                            # ★ 2026-09-16 以前是 SOLO_TH="0.5"(%), 在 1280x720 的时代广场上调的,
                            #   到 1920x1080 的 pedestrian 等于严了 2.25 倍, beach/carpark 只好各覆盖 0.03。
    PER_INSTANCE= "tree",   # 这些类不做类内统一尺度(树高矮差太多, 统一了全变矮)
    EXCLUDE     = "sign,billboard,pedestrian,manhole cover,bollard,awning,building",
                            # 排除: 前四个是贴在墙上/地上的平面, 当独立物体摆必然出错;
                            # bollard 实际是垃圾桶误检; awning 是雨篷, 悬空没有支撑;
                            # building 走 buildings 那条支线, 不能混进物体场景 —— 一旦
                            # 03_masks/building.npz 存在, 量盒子时楼会把墙上的广告牌和
                            # 招牌当成"部件"吸走(实测 billboard 23->19, sign 13->10),
                            # 这本身没错, 但楼不该出现在 07_scene.glb 里
    SNAP        = "0.8",    # 落地: 底面离地在这个范围内才吸附, 超过视为本就悬空
    ITERS       = "12",     # 推开穿模的迭代轮数
    MARGIN      = "0.02",   # 推开后留的缝(米)
    # 人的整个身体(整段扫过, 不含手)当空气墙, 替代只管脚下的脚印矩形。见 scene_phys 的 PEOPLE_BODY
    BODY_WALL   = "1",
    # 有人的场景填 body4d 那一轮的目录 (body4d_pipeline.py 的产物)。给了就
    #   1) 尺度那步拿人当主尺子 —— 人从头到脚拍得全, 身高是真的; 家具/护栏被遮挡只会量矮, 把世界放大
    #   2) people / people_glb 两步把人放进场景; 物理把人走过的路当路障
    # ★ 和室内同一套写法(indoor_pipeline 的 BODY_RUN) —— 两条链的人线必须一致,
    #   否则同一个 world_scale.py 在室外永远走不到"只用人"那个分支(soccer 踩过: 够格当
    #   尺子的家具只有 1 类, 直接硬停, 而画面主角恰恰是人)。
    # ★ 2026-09-24 人体线接进主流程: PEOPLE=1 时 people_track(第 2 段)跟踪人、第 2 道闸挑人,
    #   people_mesh(第 3 段, 尺度之前)出网格; BODY_RUN 空着就自动指向 <R1>/body4d。
    #   一般不在这里改, 在启动页 / 第 1 道闸 / drive.py --people 1 设(写进本轮的 00_run_config.json)。
    #   手工填了 BODY_RUN 的老轮次(soccer/office)不走这两步, 照旧读那个目录。
    PEOPLE      = "0",
    PEOPLE_MIN_AREA = "0.1",# 人体跟踪只丢最大面积不到画面这么多 % 的(碎片/误检), 其余全列到闸口让人勾。
                            # body4d 自己的默认 1.0 是 mayday 近景调的, soccer 远处守门员 0.7% 被它静默丢了
    BODY_RUN    = "",
    RB_FRAMES   = "0,40,88,120,160",   # 渲染回图用哪几个机位
    RECALL      = "0",      # 召回支路(detect/recall/recall_mask/recall_gen/recall_merge)开不开。
                            # ★ 2026-09-24 默认关(用户拍板)。关着时五步照常盖完成戳但什么都不跑,
                            #   组装只认 06_all_boxes.json(盘上留着的 merged 也不用), 驱动第 4 道闸不停。
                            #   要开: 场景里写 RECALL="1", 或环境变量 RECALL=1。
    DET_STRIDE  = "5",      # 召回: 每隔几帧全图检一次 (180 帧不必全检)
    DET_SCORE   = "0.8",    # 召回: 检测分数下限。分数上限约 1.1 不是概率;
                            # 0.8 时每帧约 60 个, 和画面里实际物体数量级相当
    MIN_VIEWS   = "3",      # 召回: 聚类后至少见于这么多帧才算数, 挡单帧幻觉
    DIST_F      = "0.5",    # 召回: 跨帧聚类半径 = 物体尺寸 x 它 (车和椅子容差差一个量级)
    MATCH_F     = "0.7",    # 召回: 和 SAM3 已有实例的匹配半径同理(只在 MATCH_2D=0 时用)
    MATCH_2D    = "1",      # 召回: 和 SAM3 比对走二维框 IoU 而不是三维距离。见 s_recall 的注释
    IOU_SELF    = "0.5",    # 召回: WD 实例之间去重, 框 vs 框
    IOU_SAM     = "0.3",    # 召回: WD 的框 vs SAM3 mask 的外接框
    SELF_3D_F   = "0.8",    # 召回: 两个 WD 实例没有共同帧时的三维退路半径
    REC_MIN_PCT = "0.15",   # 召回: 补抠出的 mask 太小就不生成, 出来必是垃圾
    REC_MIN_IOU = "0.3",    # 召回: 补抠结果和输入框重合不够就丢
    BLD_CONF    = "1.0",    # 楼的深度置信度下限。楼在 100m 外, 置信度恰好等于 DA3 的
                            # 下限 1.00, 用物体那档 2.0 会一个点都剩不下
    BUILDING_WORDS = "",    # 哪些词算"楼"。空 = 这个场景没有建筑, BUILDING_MODE 自动降成 scene
    BUILDING_MODE  = "scene",   # 楼怎么出。三选一, 见 _bmode():
                            #   building+scene     楼一个 glb + 场景一个 glb, 各自渲染回图
                            #   scenewithbuilding  一个 glb, 楼和物体在一起
                            #   scene              一个 glb, 没有楼(楼连生成都不生成)
    EXTRA_WORDS = "",       # 追加给 SAM3 跟踪的词, 词表之外的。逗号分隔。
                            # ★ 为什么必须有这个配置项: 列词那步的 prompt 明确让 VLM
                            #   "跳过建筑立面", 所以词表里永远不会有楼。楼只能从这里注入。
                            #   2026-09-08 beach 那 4 栋楼是手敲 EXTRA_WORDS= 跑出来的,
                            #   没进配置 —— 结果是重跑 track 楼就消失, 交付物不可复现。
)

# 上面那些阈值是在 timesquare 这一个"游戏引擎渲染"素材上调出来的。换实拍素材(噪声更大、
# 深度更差)多半要重调, 所以每个场景可以在这里覆盖, 不用去改 CFG 也不用每次敲环境变量。
# ★ 没列在这里的场景标签取场景名前 3 个字母 —— nfl / nfl1 都会是 nfl_rN, 生成的 mesh 按标签存目录,
#   撞了互相覆盖。前缀会撞的场景在这里配自己的标签(启动页开新轮次时也会查撞没撞)。
SCENE_TAG = {"timesquare": "ts", "warpengine": "we", "beach": "be", "carpark": "cp", "nfl1": "nf1"}

SCENES = {
    "timesquare": {},        # 基准场景 —— CFG 的默认值就是在它上面调的
    # 足球场, 1280x720 / 205 帧。★ 人是这段素材的主角: 拿人当尺子, 再把人放进场景。
    #   不配 BODY_RUN 的话 scale 那步会硬停 —— 够格当尺子的家具只有 fence post 一类
    #   (球门/球网/围栏各只有 1 个实例, 袜子球鞋 0.15 m 太矮), 不够两类互相印证。
    #   相机轨迹只有 4.2 m, 视差很小, 朝向和尺度本来就不可靠, 更要靠人这把尺子。
    #   ★ 2026-09-24 b1 -> b2: b1 按默认面积门槛把守门员丢了, b2 补回来(3 人)。尺度是 b1 那两个
    #     近处球员定的(k=0.864), 不重定 —— 守门员远、小, 当尺子不如他们准。
    "soccer":     dict(BODY_RUN=f"{SAM3D}/sam-body4d/outputs/soccer/b2"),
    "warpengine": {},        # 还没跑过, 跑的时候把调出来的值填这里
    # 下面两个是 2026-09-07 新加的。素材 1920x1080, 比时代广场(1280x720)清晰但帧少
    # (beach 109 帧 / carpark 132 帧 vs 180)。阈值先沿用时代广场那套, 跑出来再按实际调。
    # 两段都是高空俯视, 物体在画面里比平视街景小得多, 面积类门槛要放宽。
    # SOLO="*" = 除白名单外全部逐实例生成 —— 用户的判断是"基本所有类都不能合并",
    # 同类外观差异大(沙滩伞至少四种花色, 停车场轿车/皮卡/箱货/搅拌车混在一起)。
    "beach":      dict(MAX_FRAMES="109", MIN_AREA_PCT="0.02", SOLO="*", SOLO_KEEP="",   # chair 唯一那个实例被去重判成了 bench#2(4 个共同帧 IoU 0.964,
                                             # 看图确认是同一把白色板条椅), 这个白名单已经空转
                       # 建筑 glb 只用 cabin(绿顶木亭子) 和 shed(长条集装箱建筑), 两个抠得都干净。
                       # ★ building 这个词必须排除: 它抠出来的是整个蓝色餐厅露台, 连同平台上
                       # 所有遮阳伞/桌/椅/人一起烘成一坨。而那些伞桌椅本身还在被单独检测
                       # (umbrella 24 个实例 / chair 12 / table 10), 放进去同一批物体会被摆两次。
                       # 教训: 补词能救回漏检(没有 building 这个词, 22 个建筑实例一个都没有),
                       # 但补进去的词必须看抠图验一遍 —— 词太宽泛会把一大片场景圈成一个物体。
                       # 同样是补的词, cabin 对 building 错, 差别只在词的粒度。
                       # (wooden pavilion / kiosk 跑出 0 实例, archway / beach hut 被 VLM 否掉。)
                       # 2026-09-08 换了新素材(Coney Island 木栈道), 老的 cabin/shed 不存在了。
                       # 列词那步的 prompt 明确让 VLM 跳过建筑立面, 所以词表里一个建筑词都没有,
                       # 建筑词是用 track_all 的 EXTRA_WORDS 单独跟踪进来的:
                       #   building   OOM 挂掉(峰值 29.2G, 分段也不行) —— 匹配的东西太多, 粒度太粗
                       #   pavilion   0 个实例
                       #   storefront / shop / restaurant  各 5 个, 抠的是同一批楼, 质量都不错
                       #   kiosk      1 个, 抠到的是热狗车不是楼
                       # 去重那步(2026-09-08 新增)判定 shop/storefront/restaurant 是同一批楼,
                       # kiosk 抠到的是热狗车, 并进了热狗车那组。
                       # ★ 这里必须写"去重之后活下来的那个词" —— 另外两个词的实例已经被判掉,
                       #   写它们等于这个 glb 是空的, 而那 5 栋楼会留在主场景里(建筑不进主场景
                       #   靠的就是 BUILDING_WORDS 这张表), 两头都错。
                       #   去重从词级改成实例级(同日)之后胜出者从 storefront 变成 restaurant:
                       #   三个词在 RANK 表里同档、实例数都是 5, 一路平到底最后按词名定,
                       #   restaurant 字典序最小。三个词抠的是同一批楼, 换名字不换东西。
                       # ★ 2026-09-09: 这两行以前是手敲命令行的, 没进配置 ——
                       #   重跑 track 楼就消失, 昨天交付的 4 栋楼从流水线跑不出来。
                       #   现在 s_track 会把 EXTRA_WORDS 传给 track_all.py, 可复现了。
                       #   building 不写进来: 实测 OOM(峰值 29.2G), 而且粒度太粗,
                       #   会把整个餐厅露台连桌椅一起烘成一坨。
                       EXTRA_WORDS="storefront,shop,restaurant",
                       BUILDING_WORDS="restaurant",
                       BUILDING_MODE="building+scene",
                       # 沙滩纹理均一(大片沙子和海面), DA3 只有 9% 的像素置信度过 2.0
                       # (停车场 92.6%, 时代广场 77.7%)。注意这个差别是画面内容造成的,
                       # 不是相机运动: carpark 相机也只走了 1.48m 但置信度比时代广场还高。
                       # 照搬 2.0 会把整个场景滤空, 降到 1.0 保住 100% 的像素;
                       # 点数门槛跟着降, 因为可用像素虽多但深度本身更糙。
                       CONF_MIN="1.0", MIN_POINTS="150",
                       # 用户要求"直接全跑实例"。俯视素材物体极小: chair 0.14%、
                       # trash can 0.08%、bollard 0.05% 都已经是各自类里最大的那个。
                       # 以前在这里覆盖 SOLO_TH="0.03"(= 622 像素); 门槛改成像素之后默认 256 已经
                       # 比它宽, 覆盖删掉了(2026-09-16)。
                       EXCLUDE="sign,billboard,awning,manhole cover,bollard,fence,railing,person,stair,building"),
    "carpark":    dict(MAX_FRAMES="132", MIN_AREA_PCT="0.02", SOLO="*", SOLO_KEEP="van",
                       # 以前覆盖 SOLO_TH="0.03", 同 beach, 门槛改像素后删掉
                       EXCLUDE="sign,billboard,awning,manhole cover,bollard,fence,fence post,scaffolding"),
}
# 生效顺序: CFG 默认值 <- 场景覆盖 <- 环境变量 (环境变量最大, 方便临时试一个值)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("review_decisions", f"{SCRIPTS}/review_decisions.py")
review = _ilu.module_from_spec(_spec); _spec.loader.exec_module(review)

def resolve_cfg(scene, R1=None):
    """本文件的 CFG/SCENES/闸决策接到共用实现上。生效顺序和为什么是这个顺序, 见 _common.py。"""
    return _resolve_cfg(CFG, SCENES, scene, R1, review)


# 生成样本的人工指定: VLM 验图只保证"是不是这个词", 不保证"抠得全不全"。
# table 自动选中的候选只有一块光桌面没有腿, 用户看了候选图后指定第 6 个(实例8/第32帧),
# 那个连 X 形支架一起抠到了, 生成出来才是完整的折叠桌。
# 词 -> (实例号, 帧号)。★ 按场景分开 —— 这本来是个全局 dict, 但它写的是
# "timesquare 那张折叠桌该用第 8 个实例的第 32 帧", 换个场景根本没有 table 这个词,
# 读 npz 直接 FileNotFoundError 把整条链打断(carpark 实测)。
PICK_OVERRIDE_BY_SCENE = {
    # timesquare: VLM 只判"是不是桌子", 判不了"抠得全不全"。自动选中的那张只有一块光桌面
    # 没有腿, 用户看了候选图后指定第 6 个候选(实例8/第32帧), 那张连 X 形支架一起抠到了。
    "timesquare": {"table": (8, 32)},
}


# ------------------------------------------------------------------ 执行
class Ctx:
    def __init__(self, scene, run):
        self.scene = scene; self.run = run
        self.tag = f"{SCENE_TAG.get(scene, scene[:3])}_{run}"     # timesquare+r1 -> ts_r1
        self.FR  = f"{SAM3D}/data/outdoor/dataset/{scene}/frames"
        self.VID = f"{SAM3D}/data/outdoor/dataset/{scene}/video.mp4"
        self.R1  = f"{SAM3D}/data/outdoor/result/{scene}/{run}"
        self.LOG = f"{self.R1}/logs"
        self.cfg = resolve_cfg(scene, self.R1)   # R1 要先算出来: 人工闸的决策文件在它下面
        # 人体线归不归这条流水线管; 顺带把 BODY_RUN 填成自动目录(或挑人闸一个没勾时置空)
        self.body_auto = people_decisions.resolve_body_run(self.cfg, scene, run, self.R1)

# ------------------------------------------------------------------ 各步骤
def s_frames(c, dry):
    """抽帧。文件名必须补零成 0000.jpg —— 这是踩过的最大的坑:
    0.jpg/1.jpg/... 按字典序排是 0,1,10,100,101,...,2,20, DA3 拿到的就是乱序的视频,
    90 个相机位姿会塌成一个点, 而且不报错。"""
    os.makedirs(c.FR, exist_ok=True)
    if glob.glob(f"{c.FR}/0000.jpg") and not dry:
        print("    帧已存在, 跳过"); return
    # 本机没装 ffmpeg, 用 cv2 抽。原来写 ffmpeg 一直没暴露, 是因为 timesquare 的帧
    # 是当初手工抽好的, frames 这步从来没真跑过。
    code = f'''
import cv2, os
cap = cv2.VideoCapture({c.VID!r}); i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    cv2.imwrite(f"{c.FR}/{{i:04d}}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95]); i += 1
cap.release(); print(f"[done] 抽出 {{i}} 帧 (文件名补零到四位, 这是必须的)")
'''
    sh([PY["sam3d"], "-c", code], f"{c.LOG}/00_frames.log", dry)

def s_vlm(c, dry):
    """Qwen3-VL 扫 8 帧列词表。只让它列名词, 不让它画框 —— 定位交给 SAM3。
    词表一个不改: 这是"VLM 能不能自己列全"的实验, 掺了手写词就没意义了。"""
    sh(envcmd(PY["trellis2"], f"{SCRIPTS}/vlm_list_objects.py",
              dict(FR=c.FR, OD=c.R1, NFR=c.cfg["NFR"])), f"{c.LOG}/01_vlm.log", dry)

def s_da3(c, dry):
    """Depth Anything 3 出米制深度 + 相机位姿。只有 spatiallm 环境装了它。
    世界系 = 第一帧的相机系, OpenCV 约定(y 朝下), 所以后面所有脚本里 UP = [0,-1,0]。
    GEOM 不是 da3 时换成 VGGT-Ω 后端(见 CFG 里 GEOM 的注释), 输出目录和格式不变。
    室外要开天空分割: DA3 自带天空头, VGGT-Ω 没有, 不遮的话天空会变成一层假的"远处的墙"。"""
    if c.cfg["GEOM"] != "da3":
        scale = {"vggt_l": "metric3d_l", "vggt_g2": "metric3d_g2"}[c.cfg["GEOM"]]
        cmd = [PY["vggt_omega"], VGGT_GEOM, "--images", c.FR, "--out", f"{c.R1}/02_da3",
               "--max-frames", c.cfg["MAX_FRAMES"], "--scale", scale]
        # 焦距交给 Metric3D 定尺度用, 见 CFG 里 FX 的注释。VGGT-Ω 本身不收内参, 所以不传给它。
        if c.cfg["FX"]:
            cmd += ["--fx", c.cfg["FX"]]
        else:
            print("    [提示] 没给 FX, 米制尺度建立在 VGGT-Ω 估的焦距上, 只能靠后面 world_scale "
                  "按常识尺寸兜; 画面里没有认得出的参照物时这一兜会落空", flush=True)
        # PROCESS_RES 这一轮是死的: VGGT-Ω 固定 512。人调了却不生效最容易白白浪费一轮, 所以要说一声。
        if c.cfg["PROCESS_RES"] != CFG["PROCESS_RES"]:
            print(f"    [提示] PROCESS_RES={c.cfg['PROCESS_RES']} 这一轮不生效: "
                  f"VGGT-Ω 分辨率固定 512(只有 GEOM=da3 才读这个值)", flush=True)
        sh(cmd, f"{c.LOG}/02_da3.log", dry)
        _frame_map(c, dry); return
    sh(f'{shlex.quote(PY["spatiallm"])} {shlex.quote(f"{DA3REPO}/da3_to_ply.py")} '
       f'--images {shlex.quote(c.FR)} --out {shlex.quote(f"{c.R1}/02_da3")} '
       f'--max-frames {c.cfg["MAX_FRAMES"]} --process-res {c.cfg["PROCESS_RES"]} --use-ray-pose',
       f"{c.LOG}/02_da3.log", dry)
    _frame_map(c, dry)

def _frame_map(c, dry):
    """在 cameras.json 旁边写一份"帧 -> 原视频帧号 + fps"。两条链同一个路径、同一份格式,
    下游一律查它, 谁也不准自己从键名解析。为什么需要它见 _common.py 里 FRAME_MAP 那段。
    室外的帧是逐帧抽的, 键本来就是原帧号, 所以这里落下来是恒等映射 —— 但**照样要写**,
    下游才不用管自己面对的是哪条链。"""
    write_frame_map(f"{c.R1}/02_da3", c.FR, c.VID, dry)

def s_up(c, dry):
    """定出场景真正的"天"。老代码写死"首帧相机的 -y 就是天", 街头平视成立(时代广场偏
    0.4°), 高空俯拍不成立(beach 偏 26.2°, carpark 29.2°) —— 相机低着头, 它的上方是斜的,
    整个场景连同地面一起歪掉。改成从点云最大平面拟合地面法向。
    必须排在量盒子之前: 盒子高度是沿"天"量的, 方向错了盒子就错, 事后旋转救不回来。"""
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/world_up.py", dict(R1=c.R1)),
       f"{c.LOG}/02_up.log", dry)

def s_track(c, dry):
    """SAM3 视频跟踪, 每个词一个结果文件, RLE 压缩落盘。跟踪只跑这一次,
    后面挑图/量盒子/逐实例生成全部从 03_masks/*.npz 读, 不再碰 GPU。
    防 OOM 三级都在 track_all.py 里(显存卸到 CPU / 密集词提高检出阈值 / 分段)。
    ★ 2026-09-29 按词拆分: 有几张空卡就起几份 track_all.py, 每份跟踪一部分词(fanout.fan_track)。"""
    fanout.fan_track(PY["sam3"], f"{SCRIPTS}/track_all.py",
                     dict(FR=c.FR, R1=c.R1, EXTRA_WORDS=c.cfg["EXTRA_WORDS"]),
                     c.R1, f"{c.LOG}/03_track.log", dry)

def s_dedup(c, dry):
    """跨词去重(二维): 同一批物体被几个同义词各抠一遍, 赶在生成之前判掉多余的词。
    ★ 位置就是这一步的全部意义。原来的跨词去重在 boxes 那步(三维体素), 而 boxes 排在
    gen 之后 —— 它要拿 mesh 的长宽高比例来选词, 于是三个同义词已经各烧过一次 SAM3D
    (每次 1-2 分钟 GPU) 才发现是重复的, 去掉的只是盒子的重, mesh 白生成。
    挪到 track 之后, 判据只用 mask 就够: 同一帧上两个词的 mask 重得够多就是同一个东西,
    不必经过深度和尺度 —— 而深度和尺度恰恰是这条链上最不可靠的两环(beach 的深度尺度
    实测差过 6.4 倍)。同一条思路在 merge_recall.py 上验证过, 假漏检从 46 降到 4。
    排除表要和挑图那步用同一份: 排除词本来就不下游, 让它参与只会有一种坏结果 ——
    它当选了组代表, 整组物体跟着它一起被排除掉。
    boxes 那步的三维体素去重留着当兜底, 专管"两个词从来没在同一帧同时出现"的死角。"""
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/dedup_words.py",
              dict(R1=c.R1, EXCLUDE=_gen_excl(c),
                   DW_DUP_TH=c.cfg["DW_DUP_TH"], DW_PART_TH=c.cfg["DW_PART_TH"],
                   DW_PART_RATIO=c.cfg["DW_PART_RATIO"],
                   DW_FRAME_FRAC=c.cfg["DW_FRAME_FRAC"], DW_WORD_FRAC=c.cfg["DW_WORD_FRAC"])),
       f"{c.LOG}/03_dedup.log", dry)

BMODES = ("building+scene", "scenewithbuilding", "scene")

def _bwords(c, follow_dedup=True):
    """这个场景里哪些词算"楼"。

    ★ 不能只用 BUILDING_WORDS 那几个字面词。beach 的楼被 storefront/shop/restaurant
      三个词各抠一遍, 去重那步判成同一批、只留一个代表 —— 而代表是谁会变
      (三个词同档、实例数相同时按词名定, 跟踪结果稍变就换人)。
      两个后果都不报错:
        1. BUILDING_MODE=scene 只排除 restaurant, shop/storefront 顶上来照样生成、照样进场景;
        2. 代表换人后 s_assemble 用的还是原来那个词, 楼留在主场景里, 同时 building glb 少几栋。
      所以这里把"和建筑词同组的所有词"一起算进来 —— 跟组走, 不跟字面词走。
    """
    bw = {w.strip() for w in c.cfg["BUILDING_WORDS"].split(",") if w.strip()}
    # EXTRA_WORDS 注入的词就是为了抠楼的, 也一并算 —— 它们本来就是同一批东西的不同说法
    bw |= {w.strip() for w in c.cfg.get("EXTRA_WORDS", "").split(",") if w.strip()}
    if not (follow_dedup and bw):
        return bw
    dw = f"{c.R1}/03_dedup_words.json"
    if not os.path.exists(dw):
        return bw
    try:
        d = json.load(open(dw))
    except Exception:
        return bw
    grown = set(bw)
    for g in d.get("组", []):
        words = {str(m).rsplit("#", 1)[0] for m in g.get("实例", [])}
        rep = g.get("代表词") or g.get("胜出")
        if rep: words.add(rep)
        if words & bw:                      # 这一组里只要有一个是建筑词, 整组都算楼
            grown |= words
    if grown != bw:
        print(f"    [建筑词跟组] {sorted(bw)} -> {sorted(grown)}")
    return grown

def _bmode(c):
    """这个场景的建筑模式。三选一:
         building+scene     楼一个 glb + 场景一个 glb, 各自渲染回图
         scenewithbuilding  一个 glb, 楼和物体在一起
         scene              一个 glb, 没有楼(楼连生成都不生成, 不烧那份 GPU)
    没有建筑词就没有"楼"可谈, 一律降成 scene —— 否则 building+scene 会出一个空 glb,
    而空 glb 不报错, 又是一处静默失效。"""
    m = c.cfg["BUILDING_MODE"].strip()
    if m not in BMODES:
        raise SystemExit(f"[失败] BUILDING_MODE={m!r} 不认识, 只能是 {list(BMODES)}")
    if not _bwords(c):
        if m != "scene":
            print(f"    [建筑模式] BUILDING_WORDS 是空的, {m} 降成 scene")
        return "scene"
    return m

def _boxes_arg(c):
    """摆场景该读哪份盒子。有召回合并过的就用合并版 —— 这是"召回捡回来的东西进不进场景"
    的唯一开关。
    ★ 原来这里看的是环境变量 USE_RECALL, 而且默认关: 召回五步照跑、显存照烧, 结果永远
      进不了 07_scene.glb, 表面上还"全跑完了没报错"。室内一直是"文件在就用", 是对的,
      这里跟室内对齐。"""
    if c.cfg["RECALL"] != "1": return {}     # 召回关: merged 即使在盘上也是以前留下的, 不用
    m = f"{c.R1}/06_all_boxes_merged.json"
    b = f"{c.R1}/06_all_boxes.json"
    if not os.path.exists(m): return {}
    # ★ 陈旧的 merged 不许静默采用。改成"文件在就用"之后这个坑比原来的 USE_RECALL 更容易踩:
    #   重跑过 boxes/scale 之后 base 是新的、merged 还是上一轮的, 整场景会退回旧盒子摆。
    if os.path.exists(b) and os.path.getmtime(m) < os.path.getmtime(b):
        # ★ 这里原来只把话写进一个变量就 return 了 —— 既不抛也不打印, 注释说"不许静默采用",
        #   代码做的正好是静默采用。硬停: 拿上一轮的盒子摆整个场景, 出来的东西看着没毛病。
        raise SystemExit(
            f"[失败] {os.path.basename(m)} 比 {os.path.basename(b)} 旧, 拒绝使用。\n"
            f"       合并盒子 {time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(m)))}, "
            f"基础盒子 {time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(b)))}\n"
            f"       量盒子那步重跑过了, 召回的合并结果已经过期。重跑 recall_merge, "
            f"或者删掉 merged 那份用基础盒子摆。")
    return {"BOXES": m}

def _gen_excl(c):
    """挑图/逐实例生成 那两步用的排除表。
    建筑词要不要生成, 取决于模式:
      building+scene / scenewithbuilding —— 要生成, 所以从排除表里减掉建筑词
        (主场景排不排它们是 s_assemble 的事, 生成那步要是先挡掉, 楼就一个都没有)
      scene —— 不出楼, 那就连生成都别生成, 把建筑词加进排除表, 省掉那份 GPU"""
    ex = {w.strip() for w in c.cfg["EXCLUDE"].split(",") if w.strip()}
    bw = _bwords(c)
    ex = (ex | bw) if _bmode(c) == "scene" else (ex - bw)
    return ",".join(sorted(ex))

def _gen_excl_set(c):
    """_gen_excl 的集合版, 给 _gen 过滤磁盘目录用。"""
    return {w.strip() for w in _gen_excl(c).split(",") if w.strip()}

def s_pick(c, dry):
    """每类挑一个实例的一帧当生成样本。打分 = 面积 x (1-贴边惩罚) x min(帧数/10,1)。
    首轮那个"活够20帧"的硬门槛去掉了 —— 打分本来就在挑最大最完整的, 硬门槛只误杀
    (桌子 12 个实例被它全砍光, 一个 mesh 都没生成)。"""
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/pick_from_masks.py",
              dict(R1=c.R1, FR=c.FR, TAG=c.tag, EXCLUDE=_gen_excl(c),
                   MIN_FRAMES=c.cfg["MIN_FRAMES"],
                   MIN_AREA_PCT=c.cfg["MIN_AREA_PCT"],
                   # ★ 必须和 s_solo 传给 gen_per_instance.py 的三个值一模一样。
                   # 挑图这步靠它们判断"这个词会不会走逐实例", 走的话就不写类目录
                   # (不然同一张 mask 被写两遍, 生成两个一样的 mesh)。两边判据一旦不一致,
                   # 会出现两边都以为对方会写 -> 那个词整个从场景里消失。
                   SOLO=c.cfg["SOLO"], SOLO_KEEP=c.cfg["SOLO_KEEP"], SOLO_MIN_PX=c.cfg["SOLO_MIN_PX"],
                   )), f"{c.LOG}/04_pick.log", dry)

def s_verify(c, dry):
    """让 VLM 看抠图确认"这块到底是不是这个词", 不是就换下一个候选。
    起因: table 挑中的实例圈的是一辆轿车+卡车。打分只看面积/贴边/活多久, 纯几何,
    从不校验内容 —— 测量交给几何, 判断交给 VLM。
    提问要放宽("可能只是物体的一部分/可能糊"), 严格问法会把正常抠图也否掉。"""
    sh(envcmd(PY["trellis2"], f"{SCRIPTS}/verify_picks.py",
              dict(R1=c.R1, FR=c.FR, TAG=c.tag, TOPK=c.cfg["TOPK"], MIN_AREA_PCT=c.cfg["MIN_AREA_PCT"],
                   # 整类样本的候选规则和 s_pick 同一份(class_sample.py), 帧数下限也得同一个值
                   MIN_FRAMES=c.cfg["MIN_FRAMES"],
                   # ★ 不传 SOLO 的话 verify_picks 的逐实例校验整段跳过(它靠 SOLO 知道要验谁),
                   #   以前只有 shell 里碰巧 export 了 SOLO 才会跑。和 s_pick / s_solo 传同一套。
                   SOLO=c.cfg["SOLO"], SOLO_KEEP=c.cfg["SOLO_KEEP"], EXCLUDE=_gen_excl(c),
                   SOLO_MIN_PX=c.cfg["SOLO_MIN_PX"])),
       f"{c.LOG}/07_verify.log", dry)

def _overrides(c):
    return review.overrides(c.R1, PICK_OVERRIDE_BY_SCENE.get(c.scene, {}))

def s_override(c, dry):
    """人工指定生成样本。VLM 只判"是不是", 判不了"全不全"。
    ★ 这一步是路线图里唯一要永久保留的人工干预, 所以它的输入必须是机器可读的 ——
      2026-09-09 之前它只读源码里那张 dict, 闸口填的东西到不了这里。"""
    # "不生成"落一个盘上的名单, _gen 会照着挡。词级的 EXCLUDE 挡不住单个候选,
    # 而人在闸口否掉的往往就是某一个实例, 不是整个词。
    review.write_skip(c.R1, dry)
    code = f'''
import os,sys,json,numpy as np,cv2
from pycocotools import mask as mask_util
sys.path.insert(0,{SCRIPTS!r})
from imgio import imwrite_stable
R1={c.R1!r}; FR={c.FR!r}; OV={_overrides(c)!r}
if not OV: print("[人工指定] 这个场景没有人工指定, 跳过")
for w,(oid,fi) in OV.items():
    _p=f"{{R1}}/03_masks/{{w.replace(chr(32),chr(95))}}.npz"
    if not os.path.exists(_p):
        print(f"[跳过] {{w}} 这个场景里没有这个词"); continue
    z=np.load(_p,allow_pickle=True)
    meta=z["meta"]; counts=z["counts"]; H,W=z["hw"]
    hit=[i for i,(o,f,a,e) in enumerate(meta) if int(o)==oid and int(f)==fi and a>0]
    if not hit: print(f"[跳过] {{w}} 没有 实例{{oid}}/第{{fi}}帧"); continue
    m=mask_util.decode({{"counts":counts[hit[0]],"size":[int(H),int(W)]}}).astype(np.uint8)
    d=os.path.expanduser(f"~/sam3d/sam-3d-objects/my_images/{c.tag}_{{w.replace(chr(32),chr(95))}}")
    os.makedirs(d,exist_ok=True)
    imwrite_stable(f"{{d}}/frame.png",cv2.imread(f"{{FR}}/{{fi:04d}}.jpg")); imwrite_stable(f"{{d}}/mask.png",m*255)
    p=json.load(open(f"{{R1}}/04_picks/picks.json")); p.setdefault(w,{{}}).update(obj_id=oid,frame=fi,人工指定=True)
    json.dump(p,open(f"{{R1}}/04_picks/picks.json","w"),ensure_ascii=False,indent=1)
    print(f"[人工指定] {{w}} -> 实例{{oid}} 第{{fi}}帧")
'''
    sh([PY["sam3d"], "-c", code], f"{c.LOG}/07_override.log", dry)

def _word_of(tag, dirname):
    """目录名反推词: cp_r1_manhole_cover -> "manhole cover"; cp_r1_car_3 -> "car"。"""
    n = dirname[len(tag)+1:] if dirname.startswith(tag + "_") else dirname
    if is_solo(dirname): n = n.rsplit("_", 1)[0]      # 实例目录, 去掉尾巴上的编号
    return n.replace("_", " ")

def _gen(c, dirs, log, dry):
    """调 SAM 3D Objects 逐个生成。--hq = 烘焙 1024 贴图, 慢 1-2 分钟但清晰得多。
    注意它是"补全"式生成: 只看到出租车的后侧也会生成一辆完整的车, 楼也会生成四个立面。

    ★ 这里必须再过一道排除表。挑图那步已经"排除的词不写盘"了, 但那只挡得住这一次 ——
    上一轮留在 my_images/ 里的目录还躺在盘上, 而 s_gen 是 glob 磁盘目录不是读 picks.json,
    照样会捡起来生成一遍, 正好是想省掉的那段 GPU。cyberpunk 的 couch 是靠人手动删目录
    才没撞上, 靠人记得删不算修好。"""
    ex = _gen_excl_set(c)
    if ex:
        keep = [d for d in dirs if _word_of(c.tag, os.path.basename(d)) not in ex]
        if len(keep) != len(dirs):
            print(f"    [排除] 盘上还留着 {len(dirs)-len(keep)} 个排除词的目录, 不生成: "
                  + ", ".join(sorted({_word_of(c.tag, os.path.basename(d))
                                      for d in dirs if d not in keep})), flush=True)
        dirs = keep
    # 人在闸口逐个否掉的候选。EXCLUDE 是词级的, 挡不住"这个词留着但这一个实例不要",
    # 而闸口最常见的判断恰好是后者。名单由 s_override 落盘。
    dirs = review.filter_dirs(c.R1, dirs)
    def _fresh(d):
        """这个目录的 mesh 还作数吗 —— 存在, 而且不比抠图旧。

        ★ 只看"在不在"是不够的 —— 这条正是"人在闸口换候选"一直不生效的原因:
          s_override 会重写 my_images/<n>/frame.png, mesh 却还是上一轮那个,
          于是 picks.json 改了、场景里摆的没变, 而且一声不吭。
          抠图比 mesh 新就必须重跑。
          +1 秒容差: 同一次跑里两个文件的 mtime 可能只差毫秒, 不留容差会每次都重生成。
        ★ 要看 frame.png **和** mask.png 里较新的那个。只看 frame 是不够的:
          frame.png 的内容只由帧号决定, "同一帧换一个实例"时它一个字节都不变,
          真正的选择写在 mask.png 里 —— 于是"改用"在同帧换实例时静默变成空操作。
        """
        n = os.path.basename(d)
        g = f"{GEN}/outputs/{n}/{n}.glb"
        if not os.path.exists(g): return False
        newest = max((os.path.getmtime(f"{d}/{x}") for x in ("frame.png", "mask.png")
                      if os.path.exists(f"{d}/{x}")), default=0)
        return newest <= os.path.getmtime(g) + 1

    todo = []
    for i, d in enumerate(dirs, 1):
        n = os.path.basename(d)
        if _fresh(d):
            print(f"    [{i}/{len(dirs)}] {n} 已有且不过期, 跳过"); continue
        if os.path.exists(f"{GEN}/outputs/{n}/{n}.glb"):
            print(f"    [{i}/{len(dirs)}] {n} 抠图比 mesh 新, 重新生成", flush=True)
        else:
            print(f"    [{i}/{len(dirs)}] {n}", flush=True)
        todo.append(d)
    if not todo:
        print("    没有要生成的, 跳过"); return

    # ★ 一批一次交给 run.py, 不再一个物体起一个进程。
    #   实测(pedestrian r2): 装一遍模型 40 秒, 真正生成一个物体才 30 秒 ——
    #   一个一个喂的话一半 GPU 时间花在反复装同一个模型上。22 个物体白烧 14 分钟,
    #   timesquare 召回那 43 个白烧 27 分钟。run.py 本来就支持一次吃一批(模型只装一次),
    #   它的说明里写着, 只是这里从来没这么调过 —— 日志里每个物体都是 "[1/1]" 就是证据。
    #   不能改用 --image 传一串: 那些图全叫 frame.png, --image 的输出名取自文件名,
    #   一批会全写进 outputs/frame/ 互相覆盖。--dirs 按目录名命名, 就是为这个加的。
    # run.py 里 inference.py 直接读 os.environ["CONDA_PREFIX"], 没有就 KeyError。
    # 那个变量只有 conda activate 才会设, 直接调环境里的 python 是没有的。
    print(f"    [批量] {len(todo)} 个一起生成, 模型只装一次", flush=True)

    def run(dirs, card):
        env = dict(CONDA_PREFIX=os.path.dirname(os.path.dirname(PY["sam3d"])),
                   # SAM3D 每次启动都去 HuggingFace 拉 DINO 主干, HF 一抽风就 504。
                   # 权重本地已有缓存, 强制离线即可。这个坑随网络状况时好时坏。
                   HF_HUB_OFFLINE="1")
        if card is not None:
            env.update(CUDA_VISIBLE_DEVICES=card, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        try:
            sh(envcmd(PY["sam3d"], f"{GEN}/run.py", env,
                      ["--dirs", *dirs, "--hq", "--bake-workers", c.cfg["BAKE_WORKERS"]]), log, dry, cwd=GEN)
        except SystemExit as e:
            # 批量步骤不能一个失败就整批中断: 之前 HF 504 时 beach 卡在第 15 个,
            # 后面 9 个连试都没试。run.py 内部逐个 try 住了, 这里只是接住它的非零码,
            # 具体是哪几个失败的, 下面按产物点名 —— 比退出码准。
            print(f"    [提示] 这一份有失败的 ({e}), 下面按产物点名", flush=True)

    if dry:
        return run(todo, None)
    # ★ 2026-10-08 按物体分到几张空卡上同时生成(fanout.fan_gen); 这一步自己不领卡, 每份各领一张。
    fanout.fan_gen(todo, run, _fresh)
    # ★ 按"产物在不在"点名, 不按退出码。退 0 但一个文件没写出来也算失败 ——
    #   原来一个进程一个物体时靠退出码分得清谁失败, 批量之后这个信息没了, 而且
    #   查产物本来就比查退出码严。
    _gen_report([os.path.basename(d) for d in todo if not _fresh(d)])

def is_solo(name):
    """ts_r1_<词>_<实例号> 是某个实例专属的; ts_r1_<词> 是这一类共用的。
    只能靠"最后一段是不是纯数字"来分 —— 按下划线个数数会把 folding_chair /
    manhole_cover / street_lamp 这些多词类名全部误判成实例。"""
    tail = name.split("_")[-1]
    return tail.isdigit()

def _gen_report(failed):
    """★ 有失败就退非零码, 不能只 print。
    只打印的话驱动照样盖完成戳, 而 --check 看的是 logs/solo_jobs.txt(那是清单不是 mesh),
    于是"20 个逐实例 mesh 一个都没生成出来"这件事对 --check 完全不可见。
    单个失败不中断整批(上面 try 住了), 但整批跑完必须让人知道 —— 已有的下次会跳过, 只补缺的。"""
    if not failed: return
    raise SystemExit(f"    [失败] {len(failed)} 个没生成出来: {', '.join(failed)}\n"
                     f"       别的都好了。单独重跑这一步只会补这几个(已有且不过期的会跳过)。")

def s_relabel(c, dry):
    """挪类别: 人在闸口说"这几个其实是别的东西", 把实例在 mask 文件之间搬家。
    mask 一个像素不改, 只换标签。搬完下游照常读 npz, 一行不用改。
    没有决策就是空转。"""
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/relabel_words.py", dict(R1=c.R1)),
       f"{c.LOG}/04_relabel.log", dry)

def s_gen(c, dry):
    """每类生成一个 mesh(同款复制)。楼走 buildings 那条支线, 这里不管。
    ★ 只认 04_picks/picks.json 里真的挑中的词, 不再照盘上目录生成 ——
      盘是有记忆的: 上一轮留下的类目录(某个词后来改成逐实例了, 或者被去重判掉了)还躺在
      那儿, 照 glob 就会把它捡起来再生成一遍, 正是想省掉的那段 GPU。"""
    pj = f"{c.R1}/04_picks/picks.json"
    if os.path.exists(pj):
        want = {f"{c.tag}_{w.replace(' ','_')}" for w in json.load(open(pj))}
        dirs = sorted(d for d in glob.glob(f"{GEN}/my_images/{c.tag}_*")
                      if os.path.basename(d) in want)
        stale = sorted(os.path.basename(d) for d in glob.glob(f"{GEN}/my_images/{c.tag}_*")
                       if not is_solo(os.path.basename(d)) and os.path.basename(d) not in want)
        if stale:
            print(f"    [跳过] 盘上 {len(stale)} 个类目录不在这轮的挑图结果里(改成逐实例了/被去重判掉了/"
                  f"被排除了), 不生成: {', '.join(x[len(c.tag)+1:] for x in stale)}", flush=True)
    else:
        dirs = sorted(d for d in glob.glob(f"{GEN}/my_images/{c.tag}_*")
                      if not is_solo(os.path.basename(d)))
    _gen(c, dirs, f"{c.LOG}/05_gen.log", dry)

def s_ratio(c, dry):
    """量出每个类 mesh 的长宽高比例, 存成 05_mesh_ratio.json。
    下一步跨词去重要用它: 一个红花盆同时被 plant pot / planter / potted plant 三个词抠到,
    这三个词不是粗细关系而是三种形态, 优先级表本来就无从排起 —— 但比例是量得出来的,
    矮胖的盒子对上 plant pot(0.95x0.98x1.0), 细高的对上 potted plant(0.5x1.0x0.49)。
    ★ 存的必须是"从大到小排好序"的比例: boxes_from_masks.py 的 shape_cost 把盒子的
    三条边排序后直接和这里的数对齐, 存原始 x/y/z 顺序会让整个形状匹配失效。
    预先算好存盘, 是因为量盒子那步在 sam3 环境跑, 那里没有 trimesh。"""
    code = f'''
import os,glob,json,trimesh,numpy as np
r={{}}
for p in sorted(glob.glob({GEN+"/outputs/"+c.tag+"_*/*.glb"!r})):
    n=os.path.basename(os.path.dirname(p))[len({c.tag!r})+1:].replace("_"," ")
    s=trimesh.load(p); m=s.to_geometry() if hasattr(s,"to_geometry") else s
    e=np.sort(np.array(m.extents))[::-1]          # ★ 必须降序排
    r[n]=(e/max(e[0],1e-9)).tolist()
json.dump(r,open({c.R1+"/05_mesh_ratio.json"!r},"w"),indent=1)
print(f"[done] {{len(r)}} 类的 mesh 比例")
'''
    sh([PY["sam3d"], "-c", code], f"{c.LOG}/05_ratio.log", dry)

def s_meshes(c, dry):
    """把生成的 glb 渲成灰底一览图, 肉眼查有没有生成歪的。"""
    sh(envcmd(PY["sam3d"], f"{TOOLS}/render_meshes.py",
              dict(R1=c.R1, TAG=c.tag, GEN_OUT=f"{GEN}/outputs")),
       f"{c.LOG}/05_meshes.log", dry)

def s_boxes(c, dry):
    """把每个实例跨帧反投影成点云, 拟合带朝向的盒子, 再跨词去重。
    去重用三维体素占用而不是中心距 —— 中心距分不清"重复"和"部件"(树冠 vs 整个盆栽)。
    留哪个词: 先比 mesh 形状, 形状分不开再退回优先级表(car/taxi 的 mesh 都是轿车, 形状分不出来)。"""
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/boxes_from_masks.py",
              dict(R1=c.R1, MIN_POINTS=c.cfg["MIN_POINTS"], CONF_MIN=c.cfg["CONF_MIN"],
                   SAMPLE_FRAMES=c.cfg["SAMPLE_FRAMES"],
                   VOX=c.cfg["VOX"], DUP_TH=c.cfg["DUP_TH"], PART_TH=c.cfg["PART_TH"])),
       f"{c.LOG}/06_boxes.log", dry)

def s_vlm_sizes(c, dry):
    """问 Qwen 要词表里每个词的典型真实高度(米), 给下一步当尺子。
    纯文本问答, 不看图 —— "车多高"是常识问题, 混进逐帧的视觉调用里纯属浪费。
    读现成的 01_vlm_vocab.json, 不重跑列词那步(重跑会换出一批不一样的词)。"""
    sh(envcmd(PY["trellis2"], f"{SCRIPTS}/vlm_sizes.py", dict(OD=c.R1)),
       f"{c.LOG}/06_vlm_sizes.log", dry)

# ---------- 人体线: body4d_pipeline.py 的两半挂进主流程(2026-09-24) ----------
# ★ 室内 indoor_pipeline.py 有一模一样的两步(o_people_track / o_people_mesh), 改一边要改另一边。
# ★ 这两步不进 GPU_STEPS: 里面的 body4d_pipeline.py 自己会在 track/mesh 上拿同一把 GPU 锁
#   (同一个锁文件), 外面再套一层就是自己等自己, 永远死锁。
# ★ SCENE_R1=R1: 人体直接用场景的帧和相机, 每一帧人都对得上一个场景相机。自己抽帧的话对不齐 ——
#   soccer b1 的 69 帧里只有 60 帧能对上。
def _people_cmd(c, only, extra=None):
    env = dict(SCENE_R1=c.R1, BODY_OUT=people_decisions.auto_dir(c.R1),
               MIN_AREA_PCT=c.cfg["PEOPLE_MIN_AREA"], **(extra or {}))
    return envcmd(PY["sam3d"], f"{PIPE}/body4d_pipeline.py", env,
                  ["--scene", c.scene, "--run", c.run, "--only", only])

def s_people_track(c, dry):
    """人体1: SAM3 跟踪 person + 体检图, 出挑人决策 05_people.json(第 2 道闸在网页"人"那一栏勾)。
    只丢面积不到 PEOPLE_MIN_AREA 的碎片, 其余全部列出来, 帧数够的默认勾上。"""
    if not c.body_auto:
        print("    这一轮没有人(PEOPLE=0), 或 BODY_RUN 是手工指定的, 跳过"); return
    sh(_people_cmd(c, "frames,track,check"), f"{c.LOG}/05_people_track.log", dry)
    if dry: return
    d = people_decisions.build(c.R1, f"{people_decisions.auto_dir(c.R1)}/02_body4d")
    print(f"    [挑人] 跟到 {len(d['人'])} 个人, 默认勾上 {people_decisions.keep_ids(c.R1)} -> {people_decisions.path(c.R1)}")

def s_people_mesh(c, dry):
    """人体2: 按第 2 道闸勾的人出网格(SAM 3D Body) + 对账。尺度那步拿人当尺子, 所以排在 scale 前。"""
    if not c.body_auto:
        print("    这一轮没有人(PEOPLE=0), 或 BODY_RUN 是手工指定的, 跳过"); return
    keep = people_decisions.keep_ids(c.R1)
    if keep == []:
        print("    挑人闸一个都没勾, 当这一轮没有人, 跳过"); return
    if keep is None and not dry:
        raise SystemExit(f"[失败] 没有 {people_decisions.path(c.R1)} —— 先跑 people_track")
    sh(_people_cmd(c, "pick,mesh,report", dict(PICK=",".join(map(str, keep or [])))),
       f"{c.LOG}/05_people_mesh.log", dry)

def s_scale(c, dry):
    """标定并施加全局尺度, 把 DA3 那层从"它以为的米"改成真米。
    DA3 自称米制, 但俯拍素材上判得离谱: carpark 的车量出来 0.23 m, 真车 1.48 m, 差 6.55 倍。
    后果不是"看着小"这么简单 —— SNAP=0.8 变成实际 5.2 m, assemble 的 "ext[2]<0.15 就丢"
    把真实 0.85 m 的长椅误杀, WildDet3D 的真米盒子也接不进来。
    ★ 必须排在 boxes 之后(要拿盒子当尺子), 改完深度盒子就过期, 脚本自己会重量一遍。
    ★ 幂等靠 cameras.json 里的 metric_scale_applied, 乘过就拒绝再乘。原始数据备份在 02_da3/_prescale/。"""
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/world_scale.py",
              dict(R1=c.R1, MIN_POINTS=c.cfg["MIN_POINTS"], CONF_MIN=c.cfg["CONF_MIN"],
                   SAMPLE_FRAMES=c.cfg["SAMPLE_FRAMES"], VOX=c.cfg["VOX"],
                   DUP_TH=c.cfg["DUP_TH"], PART_TH=c.cfg["PART_TH"],
                   # ★ 有人就拿人当主尺子(和室内同一套)。没有人、且够格的家具不足两类时,
                   #   world_scale.py 会直接退非零码 —— 那是它在说"这段素材定不出尺度", 不是 bug。
                   SCALE_PEOPLE=c.cfg["BODY_RUN"])),
       f"{c.LOG}/06_scale.log", dry)

def s_solo(c, dry):
    """SOLO 里的类逐实例各生成一个 mesh。街上的车型号各不相同, 复用一个 mesh
    会变成满街同款。太小的实例(< SOLO_MIN_PX 像素)不单独生成 —— 那么小生成出来必是垃圾, 继续复用。
    注意: gen_per_instance.py 里的 R1/FR 是写死的, 换场景要改那个文件。"""
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/gen_per_instance.py",
              dict(R1=c.R1, FR=c.FR, TAG=c.tag, EXCLUDE=_gen_excl(c),
                   SOLO=c.cfg["SOLO"], SOLO_KEEP=c.cfg["SOLO_KEEP"], SOLO_MIN_PX=c.cfg["SOLO_MIN_PX"])),
       f"{c.LOG}/06_solo_pick.log", dry)
    jobs = f"{c.LOG}/solo_jobs.txt"
    names = open(jobs).read().split() if os.path.exists(jobs) else []
    _gen(c, [f"{GEN}/my_images/{n}" for n in names], f"{c.LOG}/06_gen_solo.log", dry)

def s_assemble(c, dry):
    """按盒子把 mesh 摆回街上, 加一个 RANSAC 拟合的地面, 导出整场景。
    尺度按高度对齐(高度几乎总能看全, 长度会被遮挡截短), 且同一类统一成一个尺度,
    基准取"当初拿去生成 mesh 的那个实例"(它是 VLM 验过的, 最可信)。
    地面必须用 RANSAC: 最小二乘会被挂在墙上的广告牌带偏 1.51m, RANSAC 残差 0.07~0.09m。
    注意地面是用 06_all_boxes.json 里"全部"盒子的底面拟合的, EXCLUDE 在这之后才生效 ——
    所以被排除的类照样影响地面高度(实测多跟了一次 building 让地面动了 2.5cm)。"""
    ex = c.cfg["EXCLUDE"]
    # 楼进不进主场景, 由模式决定 —— scenewithbuilding 就是"进", 那一支什么都不加。
    if _bmode(c) != "scenewithbuilding" and _bwords(c):
        # ★ 用跟组后的集合。写死那个词的话, 去重代表一换人楼就留在主场景里, 而且不报错。
        ex += "," + ",".join(sorted(_bwords(c)))
    print(f"    [建筑模式] {_bmode(c)}")
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/assemble_scene.py",
              dict(R1=c.R1, TAG=c.tag, GEN_OUT=f"{GEN}/outputs", SOLO_KEEP=c.cfg.get("SOLO_KEEP",""), EXCLUDE=ex,
                   PER_INSTANCE=c.cfg["PER_INSTANCE"],
                   BODY_RUN=c.cfg["BODY_RUN"],     # 有人就把人的身体当空气墙, 别往人身上推
                   **_boxes_arg(c))),
       f"{c.LOG}/07_assemble.log", dry)

def s_phys(c, dry):
    """落地 + 推开穿模。纯几何不用物理引擎 —— 室外地面就是一个平面, 比室内简单。"""
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/scene_phys.py",
              dict(R1=c.R1, SNAP=c.cfg["SNAP"], ITERS=c.cfg["ITERS"], MARGIN=c.cfg["MARGIN"],
                   # 人走过的路当路障; 整个身体当空气墙(和室内同一套)
                   PEOPLE=f"{c.R1}/10_people.json" if c.cfg["BODY_RUN"].strip() else "",
                   PEOPLE_BODY=(f"{c.R1}/10_people_body.npz"
                                if c.cfg["BODY_RUN"].strip() and c.cfg["BODY_WALL"] == "1" else ""))),
       f"{c.LOG}/07_phys.log", dry)


def _body(c, dry):
    """这个场景的人体那一轮。没配 = 没有人, 返回空; 配了但目录不在 = 直接失败, 不悄悄跳过。
    (和 indoor_pipeline._body 逐行一致 —— 改一处要改两处。)"""
    body = c.cfg["BODY_RUN"].strip()
    if body and not os.path.isdir(body) and not dry:
        raise SystemExit(f"[失败] BODY_RUN={body} 不存在 —— 先跑 body4d_pipeline.py 出人")
    return body


def s_people(c, dry):
    """摆人(物理之前): 每人每帧的远近、脚着地、走过的路线 -> 10_people.json。
    ★ 排在物理之前 —— 物理拿人走过的路当路障 (scene_phys 的 PEOPLE)。
    远近按场景深度量。尺度那步已经拿人当尺子校准过, 这里每个人的比例都该在 1 附近。"""
    body = _body(c, dry)
    if not body:
        print("    这个场景没配 BODY_RUN(没有人), 跳过"); return
    sh(envcmd(PY["sam3d"], f"{PIPE}/body_into_scene.py", None,
              ["--r1", c.R1, "--body", body, "--mode", "place", "--scene-in", "07_scene.glb"]),
       f"{c.LOG}/10_people.log", dry)


def s_people_glb(c, dry):
    """合成(物理之后): 07_scene_phy.glb 摘掉静态人形 -> 09_scene_nopeople.glb, 按 10_people.json 出几帧验收图,
    再出验收视频 09_people_check.mp4 (原视频 | 同一个相机看到的重建+人 | 两者叠加)。
    (会动的人的交付物是 motion 那步的 FBX; 顶点动画的 09_scene_4d.glb 2026-10-08 起不出了。)"""
    body = _body(c, dry)
    if not body:
        print("    这个场景没配 BODY_RUN(没有人), 跳过"); return
    sh(envcmd(PY["sam3d"], f"{PIPE}/body_into_scene.py", None,
              ["--r1", c.R1, "--body", body, "--mode", "compose", "--scene-glb", "07_scene_phy.glb",
               "--render", "--fr", c.FR]), f"{c.LOG}/10_people_glb.log", dry)
    sh(envcmd(PY["sam3d"], f"{PIPE}/body_video.py", None,
              ["--r1", c.R1, "--body", body, "--scene", "07_scene_phy.glb", "--fr", c.FR]),
       f"{c.LOG}/10_people_glb.log", dry)

def s_motion(c, dry):
    """导出动作文件(合成之后): 在模型的一手数据(关节参数)上做修复 -> 12_motion/:
    场景 + 人的 FBX(人带骨骼)、只有场景的 FBX、.blend、每人每帧的原始参数。三段脚本和自检见 motion_export.py。
    出网格那一轮没存一手数据(2026-10-08 之前跑的)会直接失败, 不悄悄跳过。"""
    body = _body(c, dry)
    if not body:
        print("    这个场景没配 BODY_RUN(没有人), 跳过"); return
    sh(envcmd(PY["sam3d"], f"{PIPE}/motion_export.py", None, ["--r1", c.R1, "--body", body]),
       f"{c.LOG}/12_motion.log", dry)

def s_render(c, dry):
    """从原视频的机位渲染重建结果, 和实拍并排比对。护栏和路灯杆应当重合。"""
    for scene, out in (("07_scene.glb", "08_render_back.jpg"),
                       ("07_scene_phy.glb", "08_render_back_phy.jpg")):
        sh(envcmd(PY["sam3d"], f"{SCRIPTS}/render_back.py",
                  dict(R1=c.R1, FR=c.FR, SCENE=scene, RBOUT=out, FRAMES=c.cfg["RB_FRAMES"])),
           f"{c.LOG}/08_render.log", dry)

def s_building_scene(c, dry):
    """把建筑类单独摆成一个 glb (beach 要"物体一个、建筑一个"), 并跑物理修正。
    做法是复用已验证的 assemble_scene + scene_phys, 只把 EXCLUDE 反过来 —— 除建筑外全排除。
    比 assemble_buildings.py 那条可靠: 那条按实心盒子摆, 已知盒心会落到马路中间。"""
    if _bmode(c) != "building+scene":
        print(f"    [建筑模式] {_bmode(c)} 不需要单独的 building glb, 跳过"); return
    # 和 assemble 读同一份盒子 —— 只靠召回捡回来的建筑, 用 base 那份会漏掉
    _bx = _boxes_arg(c).get("BOXES", f"{c.R1}/06_all_boxes.json")
    allw = set(json.load(open(_bx)))
    want = _bwords(c) & allw          # 跟组后的建筑词, 只留盒子里真有的
    # ★ BUILDING_WORDS 指向一个已经不存在的词时, 这一步会两头都错而且一声不吭:
    #   ex 变成"全部词" -> building glb 是空的; 同时 s_assemble 那句 ex += BUILDING_WORDS
    #   变成 no-op -> 那几栋楼留在主场景里。跨词去重把词换了代表(storefront -> restaurant)
    #   就会触发, 所以先去 03_dedup_words.json 里跟一下代表词, 跟不上就报错停下。
    dw = f"{c.R1}/03_dedup_words.json"
    if os.path.exists(dw):
        d = json.load(open(dw)); moved = {}
        for g in d.get("组", []):
            rep = g.get("代表词") or g.get("胜出")
            for m in g.get("实例", g.get("词", [])):
                w0 = m.rsplit("#", 1)[0] if "#" in str(m) else str(m)
                if w0 != rep: moved[w0] = rep
        for w in sorted(want - allw):
            if w in moved and moved[w] in allw:
                print(f"    [跟随去重] BUILDING_WORDS 里的 {w!r} 已并进 {moved[w]!r}, 自动改用后者")
                want.discard(w); want.add(moved[w])
    miss = sorted(want - allw)
    if miss:
        raise SystemExit(f"[失败] BUILDING_WORDS 里这些词在 06_all_boxes.json 里不存在: {miss}\n"
                         f"       硬停在这里, 因为放过去会两头错且不报错: building glb 会是空的,"
                         f" 同时这些楼会留在主场景里。\n"
                         f"       现有的词: {sorted(allw)}")
    ex = ",".join(sorted(allw - want))
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/assemble_scene.py",
              dict(R1=c.R1, TAG=c.tag, GEN_OUT=f"{GEN}/outputs", SOLO_KEEP=c.cfg.get("SOLO_KEEP",""), EXCLUDE=ex, PER_INSTANCE=c.cfg["PER_INSTANCE"],
                   SCENE_OUT="20_buildings.glb",
                   **_boxes_arg(c))),
       f"{c.LOG}/20_building.log", dry)
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/scene_phys.py",
              dict(R1=c.R1, SCENE_IN="20_buildings.glb", PHY_OUT="20_buildings_phy.glb",
                   SNAP=c.cfg["SNAP"], ITERS=c.cfg["ITERS"], MARGIN=c.cfg["MARGIN"])),
       f"{c.LOG}/20_building.log", dry)
    # building+scene 模式要"各自渲染回图" —— 楼这一份在这里出, 物体那份在 s_render 出。
    # 用同一批帧 (RB_FRAMES), 这样两张图能逐帧对着看。
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/render_back.py",
              dict(R1=c.R1, FR=c.FR, SCENE="20_buildings_phy.glb",
                   RBOUT="20_render_back_buildings.jpg", FRAMES=c.cfg["RB_FRAMES"])),
       f"{c.LOG}/20_building.log", dry)

# ---------- 召回支路: WildDet3D 和 SAM3 并行, 捡回 SAM3 漏掉的 ----------
# 为什么要有这条: 量盒子那条吃的是 SAM3 的框, SAM3 漏掉的它看都看不到,
# 而当初看中 WildDet3D 的理由之一正是召回。这条让它直接看原图, 各找各的, 最后合并。
#     帧 -> Qwen 列词 -> SAM3 跟踪       -> mask + track ID  ┐
#                     -> WildDet3D 按词检测(原图)  -> 3D 框  ┤-> 合并
#     两边都有 -> 用 SAM3 的 mask; 只有框没 mask -> 框喂回 SAM3 补抠 -> 捡回漏检
RECALL_STEPS = ("detect", "recall", "recall_mask", "recall_gen", "recall_merge")

def _recall_off(c):
    """召回开关(CFG 的 RECALL)。关着就打一行跳过, 调用方直接 return —— 完成戳照盖, 和 building_scene 的条件跳过同一套。"""
    if c.cfg["RECALL"] == "1": return False
    print("    召回关(RECALL=0), 跳过", flush=True); return True

def s_detect(c, dry):
    """召回1: WildDet3D 文本模式全图检测, 逐帧各检各的(它没有 track ID)。"""
    if _recall_off(c): return
    sh(envcmd(PY["wilddet"], f"{SCRIPTS}/detect_wilddet.py",
              dict(R1=c.R1, FR=c.FR, WD3=WD3REPO, DET_STRIDE=c.cfg["DET_STRIDE"],
                   DET_SCORE=c.cfg["DET_SCORE"])), f"{c.LOG}/12_detect.log", dry)

def s_recall(c, dry):
    """召回2: 逐帧检测按三维位置聚成实例(场景静止, 同一物体各帧世界坐标应重合),
    跨词去重, 再和 SAM3 已有实例比对, 挑出 SAM3 漏掉的。"""
    if _recall_off(c): return
    # ★ FR 必须传: 二维那条要读原帧的尺寸。
    # MATCH_2D=1 是"和 SAM3 比对时用二维框 IoU, 不用三维距离"。三维那版有个必然的毛病 ——
    # 拿 WildDet3D 的三维位置去判"是不是同一个物体", 而它的三维位置本身就不准, 判出来的
    # "新物体"多半只是同一个东西被放歪了。timesquare 48 个"漏检"逐个画到照片上一看,
    # 没有一个落在空白处; 同一辆卡车被当成三个新物体。改成二维后 48 -> 11, carpark 46 -> 4。
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/merge_recall.py",
              dict(R1=c.R1, FR=c.FR, MIN_VIEWS=c.cfg["MIN_VIEWS"], DIST_F=c.cfg["DIST_F"],
                   MATCH_F=c.cfg["MATCH_F"], MATCH_2D=c.cfg["MATCH_2D"],
                   IOU_SELF=c.cfg["IOU_SELF"], IOU_SAM=c.cfg["IOU_SAM"],
                   SELF_3D_F=c.cfg["SELF_3D_F"])), f"{c.LOG}/13_recall.log", dry)

def s_recall_mask(c, dry):
    """召回3: 漏检的只有框没 mask, 把框喂回 SAM3 补抠一张, 好让 SAM3D 能生成。
    ★ 框必须是"左上xywh 归一化", 且要在返回的一组候选里按 IoU 挑, 见脚本文件头。"""
    if _recall_off(c): return
    sh(envcmd(PY["sam3"], f"{SCRIPTS}/mask_recall.py",
              dict(R1=c.R1, FR=c.FR, TAG=c.tag, REC_MIN_PCT=c.cfg["REC_MIN_PCT"],
                   REC_MIN_IOU=c.cfg["REC_MIN_IOU"],
                   EXCLUDE=_gen_excl(c))), f"{c.LOG}/14_recall_mask.log", dry)

def s_recall_merge(c, dry):
    """召回5: 把召回实例的盒子并进主盒子文件 -> 06_all_boxes_merged.json。
    之后 assemble 传 BOXES 指向它, 召回的东西才会进场景。原文件不动, 随时能切回。"""
    if _recall_off(c): return
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/merge_boxes_recall.py",
              dict(R1=c.R1, TAG=c.tag, GEN_OUT=f"{GEN}/outputs")),
       f"{c.LOG}/15_merge_boxes.log", dry)

def s_recall_gen(c, dry):
    """召回4: 给补抠出来的漏检实例生成 mesh。清单在 logs/recall_jobs.txt。"""
    if _recall_off(c): return
    jobs = f"{c.LOG}/recall_jobs.txt"
    names = open(jobs).read().split() if os.path.exists(jobs) else []
    _gen(c, [f"{GEN}/my_images/{n}" for n in names], f"{c.LOG}/14_recall_gen.log", dry)

# ---------- 楼: 支线, 摆放是坏的 ----------
def s_buildings(c, dry):
    """★ 已知有问题, 不要当成能用的结果。
    生成没问题(SAM3D 出的楼体很好), 坏在摆放:
      1. 楼是斜着看的一面墙, 反投影出来是顺着视线方向的一片薄片, 按"实心盒子"去套
         盒心就落到马路正中间。16 号楼因此把相机整个包住, 180 帧全在楼里面。
         (盒子进深 43m/49m/40m —— 一面墙不可能有 40 米厚, 这数字本身就是证据)
      2. 楼顶超出画面, 量到的高度系统性偏小。
      3. 一大半楼在 100~240m 外, 相机总共只走了 10.4m, 那个距离根本三角化不出来。
    正确做法是拟合平面而不是盒子, 落脚点从楼底边往地面打射线; 远处的直接不要。"""
    z = f"{c.R1}/03_masks/building.npz"
    if not os.path.exists(z) and not dry:
        print("    没有 building.npz (词表里没有 building), 跳过"); return
    code = f'''
import os,json,numpy as np,cv2
from pycocotools import mask as mask_util
z=np.load({z!r},allow_pickle=True); meta=z["meta"]
live={{}}
for o,f,a,e in meta:
    if a>0: live.setdefault(int(o),[]).append((int(a),int(f)))
best={{}}
for o,rows in live.items():
    rows.sort(reverse=True); best[o]=rows[0]
open({c.LOG+"/building_jobs.txt"!r},"w").write("\\n".join(f"{c.tag}_building_{{o}}" for o in sorted(best)))
print(f"[done] {{len(best)}} 栋待生成")
'''
    sh([PY["sam3d"], "-c", code], f"{c.LOG}/08_bld_pick.log", dry)
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/assemble_buildings.py",
              dict(R1=c.R1, TAG=c.tag, CONF=c.cfg["BLD_CONF"],
                   GEN_OUT=f"{GEN}/outputs")), f"{c.LOG}/08_bld.log", dry)

def s_merge(c, dry):
    """把物体场景和楼合成一个 glb, 顺带报 AABB 穿模, 再出俯视图和渲染回图。"""
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/merge_glb.py",
              dict(R1=c.R1, SRC="07_scene_phy.glb,10_buildings.glb", DST="11_scene_full.glb")),
       f"{c.LOG}/11_merge.log", dry)
    sh(envcmd(PY["sam3d"], f"{SCRIPTS}/render_back.py",
              dict(R1=c.R1, FR=c.FR, SCENE="11_scene_full.glb", RBOUT="11_render_full.jpg",
                   TILE="520", FRAMES=c.cfg["RB_FRAMES"])), f"{c.LOG}/11_merge.log", dry)
    sh(envcmd(PY["sam3d"], f"{TOOLS}/topview.py", dict(R1=c.R1, FR=c.FR)),
       f"{c.LOG}/11_merge.log", dry)


STEPS = [
    ("frames",   "-",         "抽帧(补零成 0000.jpg)",           s_frames),
    ("vlm",      "trellis2",  "Qwen3-VL 列词表",                 s_vlm),
    ("da3",      "spatiallm", "DA3 米制深度 + 相机位姿",          s_da3),
    ("up",       "sam3d",     "定出真实的\"天\"(地面法向)",         s_up),
    ("track",    "sam3",      "SAM3 视频跟踪, 落盘",              s_track),
    ("people_track","sam3d",  "人体1: 跟踪人 + 体检图 -> 挑人决策", s_people_track),
    ("dedup",    "sam3d",     "跨词去重(二维 mask), 判掉同义词",    s_dedup),
    ("relabel",  "sam3d",     "挪类别(闸口说这几个是别的东西)",   s_relabel),
    ("pick",     "sam3d",     "每类挑生成样本",                   s_pick),
    ("verify",   "trellis2",  "VLM 验证抠图内容",                 s_verify),
    ("override", "sam3d",     "人工指定生成样本(按场景配)",       s_override),
    ("gen",      "sam3d",     "SAM3D 每类生成一个 mesh",          s_gen),
    ("ratio",    "sam3d",     "量 mesh 长宽高比例",               s_ratio),
    ("meshes",   "sam3d",     "mesh 一览图",                      s_meshes),
    ("boxes",    "sam3d",     "反投影量盒子 + 跨词去重",           s_boxes),
    ("vlm_sizes","trellis2",  "问 VLM 要每个词的真实高度(尺子)",   s_vlm_sizes),
    ("people_mesh","sam3d",   "人体2: 按闸口勾的人出网格(尺度前)", s_people_mesh),
    ("scale",    "sam3d",     "标定全局尺度 -> 真米 + 重量盒子",   s_scale),
    ("solo",     "sam3d",     "car 逐实例生成",                   s_solo),
    # ★ 2026-09-09: 召回五步整体挪到 assemble 之前。
    #   原来 assemble 排在 recall_merge 之前, 而且依赖里没有它 —— 五个召回步(三个吃满
    #   显存)照跑, 但 07_scene.glb 早就出好了, 召回捡回来的物体一个也进不去, 而且不报错。
    #   召回没法再往前挪(detect 吃深度 npy, 必须在 scale 之后), 但可以挪到组装之前。
    ("detect",   "wilddet",   "召回1: 全图按词检测",              s_detect),
    ("recall",   "sam3d",     "召回2: 聚类 + 和SAM3合并",         s_recall),
    ("recall_mask","sam3",    "召回3: 框喂回SAM3补抠mask",        s_recall_mask),
    ("recall_gen","sam3d",    "召回4: 漏检实例生成mesh",          s_recall_gen),
    ("recall_merge","sam3d",  "召回5: 盒子并进主文件",            s_recall_merge),
    ("assemble", "sam3d",     "摆回街上 + 地面 -> 07_scene.glb",  s_assemble),
    ("people",   "sam3d",     "摆人: 远近/脚着地/走过的路",        s_people),
    ("phys",     "sam3d",     "落地 + 推开穿模",                  s_phys),
    ("people_glb","sam3d",    "人放进物理后的场景 + 验收视频",      s_people_glb),
    ("motion",   "sam3d",     "导出动作文件: 场景+人的 FBX(带骨骼)", s_motion),
    ("render",   "sam3d",     "渲染回图比对",                     s_render),
    ("building_scene","sam3d", "楼单独出 glb + 物理 + 渲染回图",    s_building_scene),
    ("buildings","sam3d",     "楼(★摆放是坏的)",                  s_buildings),
    ("merge",    "sam3d",     "物体+楼合并 + 俯视图",             s_merge),
]
# 主链一路跑到 building_scene: 场景 + 渲染回图 + 召回 + (beach 那种)楼单独一份, 一条命令出全。
# ★ 2026-10-08 去掉了 WildDet3D 量盒子那一支(boxes_wd/compare/assemble_wd/phys_wd/render_wd):
#   它每轮多出一份对照场景 17_scene_wd.glb, 人从来没放进去过, 量的盒子也不看排除词, 130 秒白跑。
#   WildDet3D 本身还在用 —— 召回那一支(detect)靠它。
# 召回本来挂在支线上, 跑新场景默认根本不跑 —— 但它在 timesquare 上已经验证过(多找回
# 48 个物体), 不该再当实验。楼那两步(buildings/merge)仍是支线, 那条摆放是坏的。
DEFAULT_END = "building_scene"


# 每一步应当产出什么 (用来做 --check; 只列关键产物, 不求全)
ARTIFACTS = {
    "frames":   ["@FR/0000.jpg"],          # @FR = 帧目录, 不在 R1 底下
    "vlm":      ["01_vlm_vocab.json"],
    "da3":      ["02_da3/cameras.json", "02_da3/depth"],
    "up":       ["02_da3/world_up.json"],
    "track":    ["03_masks"],
    # 人体两步按条件跳过(没有人 / BODY_RUN 手工指定), 产物也不在 R1 底下 —— 只认完成戳
    "people_track": [],
    "people_mesh": [],
    "dedup":    ["03_dedup_words.json"],
    "pick":     ["04_picks/picks.json"],
    "verify":   ["04_picks/vlm_verify.json"],
    "gen":      [],                       # 产物在 sam-3d-objects/outputs/, 单独查
    "ratio":    ["05_mesh_ratio.json"],
    "meshes":   ["05_meshes.jpg"],
    "boxes":    ["06_all_boxes.json"],
    "vlm_sizes":["01_vlm_sizes.json"],
    "scale":    ["02_da3/world_scale.json"],
    "solo":     ["logs/solo_jobs.txt"],
    "override": [],                       # 只改 my_images/ 里的抠图, R1 底下没有产物
    "relabel": ["03_relabel_done.json"],
    "assemble": ["07_scene.glb"],
    "people":   [],                     # 没有人的场景这步是空的, 产物不能写死
    "phys":     ["07_scene_phy.glb"],
    "people_glb": [],
    "motion":   [],                     # 同上: 没有人就跳过, 产物在 12_motion/ 底下、文件名带场景名
    "render":   ["08_render_back.jpg", "08_render_back_phy.jpg"],
    "detect":   ["12_wd_detections.json"],
    "recall":   ["13_recall.json"],
    "recall_mask": ["14_recall_ready.json"],
    "recall_gen": [],
    "recall_merge": ["06_all_boxes_merged.json"],
    # ★ 这两步都会**按条件跳过**(没有建筑 mask, 或 BUILDING_MODE 不是 building+scene),
    #   而跳过之后驱动照样盖完成戳。声明了产物的话 --check 就永远报"缺", 人会以为该重跑。
    #   条件跳过的步骤只认完成戳, 不认产物。
    "building_scene": [],
    "buildings":[],
    "merge":    ["11_scene_full.glb", "11_topview.jpg"],
}

# 每一步真正依赖谁的产物。--check 靠它判断"过期", 而不是靠"比前面所有步骤旧"——
# 那种偷懒判法只要有人乱序重跑一步就会级联误报一片。
DEPS = {
    "vlm": ["frames"],          "da3": ["frames"],        "track": ["frames", "vlm"],   # track 读 vlm 出的词表 01_vlm_vocab.json
    "up": ["da3"],   "dedup": ["track"],        "relabel": ["dedup"],   "pick": ["track", "dedup", "relabel"],
    "verify": ["pick"],       "override": ["verify"],
    "gen": ["override"],        "ratio": ["gen"],         "meshes": ["gen"],
    "boxes": ["track", "ratio", "up", "dedup", "relabel"],   # relabel 会搬 03_masks/*.npz, 而量盒子读的就是它
    # 人体: track 吃 da3 的 frame_map/相机(SCENE_R1 模式); scale 拿人当尺子, 所以依赖 people_mesh
    "people_track": ["da3"],    "people_mesh": ["people_track"],
    "vlm_sizes": ["vlm"],       "scale": ["boxes", "vlm_sizes", "people_mesh"],
    "solo": ["scale"],          "assemble": ["scale", "solo", "verify", "recall_merge", "boxes"],
    "people": ["assemble", "people_mesh"],     "people_glb": ["phys", "people"],
    "motion": ["people_glb"],                  # 场景用合成那步出的 09_scene_nopeople.glb
    "phys": ["assemble", "people"],  "render": ["phys"],
    "building_scene": ["assemble"],
    # 召回不依赖盒子 —— 二维比对读的是 03_masks 的 mask。写成依赖 boxes 会把它
    # 无谓地拖到 gen/ratio 之后, 而它本可以排在生成之前。
    # detect 反投影用的是深度和相机, 尺度改了它就过期 —— 必须依赖 scale, 否则召回出来的
    # 盒子还在旧坐标系里, 并进主文件会摆错位置(carpark 实测: 用旧相机投影差 2~8 px,
    # 用新相机差 9~80 px, 差得不明显但确实是错的)。
    "detect": ["da3", "vlm", "up", "scale"],   "recall": ["detect", "track"],
    "recall_mask": ["recall"],  "recall_gen": ["recall_mask"],
    "recall_merge": ["recall_gen", "up"],
    # ★ 补 scale: assemble_buildings 读深度和相机平移, scale 就地改过它们
    "buildings": ["track", "da3", "scale"],
    "merge": ["phys", "buildings", "up"]  # topview.py 现在读 world_up.json,
}

# 吃显存的步骤: 开始之前先领一张空卡(_common.GpuSlot)。
# ★ people_track / people_mesh 故意不在这里: body4d_pipeline.py 自己在 track/mesh 上领卡, 外面再领一张就多占一张不用
GPU_STEPS = {"vlm", "vlm_sizes", "da3", "track", "verify", "gen", "solo", "detect",
             "recall_mask", "recall_gen", "motion"}
# 内部拆分的步骤: 它起的每一份各领一张卡, 主循环不替它领(领了就多占一张不用)
FANOUT_STEPS = {"track", "gen", "solo", "recall_gen"}

def do_check(c):
    """看一遍产物在不在、有没有过期。过期 = 比上一步的产物还旧, 说明上游重跑过而它没跟上。"""
    print(f"场景 {c.scene}  轮次 {c.run}\n")
    when = {}; bad = 0
    for k, e, d, _ in STEPS:
        if k == "gen":
            g = [x for x in glob.glob(f"{GEN}/outputs/{c.tag}_*/*.glb")
                 if not is_solo(os.path.basename(os.path.dirname(x)))]
            st = stamp_path(c, k)
            if g and os.path.exists(st): when[k] = os.path.getmtime(st)
            # ★ 生成类的步骤也要判过期。原来这里直接 continue, 于是"override 重跑过 ->
            #   gen 该重出"这条链在 --check 里整段隐形(assemble 又不依赖 override, 接不上)。
            _t = (os.path.getmtime(stamp_path(c, k))
                  if os.path.exists(stamp_path(c, k)) else None)
            if _t is not None: when[k] = _t
            _old = [u for u in DEPS.get(k, []) if u in when and when[u] > (_t or 0) + 1]
            if _old or not g: bad += 1
            _n = f"  ← 比依赖的 {'/'.join(_old)} 旧, 该重跑" if _old else ""
            print(f"  {k:10s} {'有' if g else '缺':2s}  {len(g)} 个类 mesh{_n}"); continue
        # 召回关着时那五步是条件跳过, 和 building_scene 一样只认完成戳, 不认产物
        arts = [] if k in RECALL_STEPS and c.cfg["RECALL"] != "1" else ARTIFACTS.get(k, [])
        paths = [p.replace("@FR", c.FR) if p.startswith("@FR") else os.path.join(c.R1, p)
                 for p in arts]
        if not paths:
            # 没有声明产物的步骤(override / recall_gen / building_scene ...)
            # ★ 也要判过期。原来这条分支只打印时间就 continue —— 于是这几步永远不会
            #   被报成"该重跑", 而 override 恰好是人工闸的落地点, 它过期是常态。
            st = stamp_path(c, k)
            if os.path.exists(st):
                when[k] = os.path.getmtime(st)
                _old = [u for u in DEPS.get(k, []) if u in when and when[u] > when[k] + 1]
                if _old: bad += 1
                print(f"  {k:10s} 跑过 {time.strftime('%m-%d %H:%M', time.localtime(when[k]))}"
                      + (f"  ← 比依赖的 {'/'.join(_old)} 旧, 该重跑" if _old else ""))
            else:
                print(f"  {k:10s} --  (没有产物, 也没跑过)")
            continue
        miss = [p for p in paths if not os.path.exists(p)]
        if miss:
            print(f"  {k:10s} 缺  {', '.join(os.path.basename(m) for m in miss)}"); bad += 1; continue
        st = stamp_path(c, k)
        if not os.path.exists(st):          # 产物在, 但不是这个驱动跑出来的
            # ★ 这也要计进 bad。原来只换个措辞就 continue, 最后一行照样打"全部就绪" ——
            #   而"产物在、戳不在"恰好是**产物已经过期**的典型形状: 比如 scale 就地改完深度、
            #   撤掉 .stamp_boxes 之后重量盒子失败, 盘上那份盒子就是缩放前的旧坐标系。
            print(f"  {k:10s} 有  (产物在, 无完成戳 —— 可能是上一轮留下的, 跑一次这步)"); bad += 1; continue
        t = os.path.getmtime(st); when[k] = t
        older = [u for u in DEPS.get(k, []) if u in when and when[u] > t + 1]
        note = f"  ← 比依赖的 {'/'.join(older)} 旧, 该重跑" if older else ""
        if older: bad += 1
        print(f"  {k:10s} 有  {time.strftime('%m-%d %H:%M', time.localtime(t))}{note}")
    print(f"\n{'全部就绪' if not bad else f'{bad} 处需要注意'}")


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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="timesquare")
    ap.add_argument("--run",   default="r1")
    ap.add_argument("--from",  dest="a", default=None)
    ap.add_argument("--to",    dest="b", default=None)
    ap.add_argument("--only",  default=None, help="逗号分隔")
    ap.add_argument("-n", "--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tag-only", action="store_true",
                    help="配合 --dump-cfg: 只打 tag。驱动别自己抄 tag 表 —— 抄一份就会漂移")
    ap.add_argument("--dump-cfg", action="store_true",
                    help="把解析后的配置打成 JSON。给人工闸用 —— 闸要知道 SOLO/EXCLUDE\n                          才能把逐实例候选也摆出来, 而那些值只有这里算得出来。")
    ap.add_argument("--check", action="store_true", help="不跑, 只看产物在不在/有没有过期")
    ap.add_argument("--dump-deps", action="store_true",
                    help="把步骤表/依赖表/吃显存的步骤打成一行 JSON(调度自检和测试用)")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)   # sched.run_children 起的子进程
    g = ap.parse_args()

    keys = [s[0] for s in STEPS]
    if g.dump_deps:
        print(json.dumps(dict(steps=keys, deps=DEPS, gpu=sorted(GPU_STEPS)), ensure_ascii=False))
        return
    if g.dump_cfg:
        _c = Ctx(g.scene, g.run)
        # 源码里写死的换候选(PICK_OVERRIDE_BY_SCENE)也吐出去: s_override 每次都会套它,
        # 闸口不知道的话, 界面上说"自动挑的是 #3", 实际用的却是这张表里的那张。
        _dump = dict(_c.cfg, 配置指定样本={w: list(v) for w, v in PICK_OVERRIDE_BY_SCENE.get(g.scene, {}).items()})
        print(_c.tag if g.tag_only else json.dumps(_dump, ensure_ascii=False))
        return
    if g.list:
        print(f"步骤 (默认跑到 {DEFAULT_END} 为止; buildings/merge 不在任何分段里, 要显式 --only):\n")
        for k, e, d, _ in STEPS:
            print(f"  {k:10s} [{e:9s}] {d}")
        return
    for x in filter(None, [g.a, g.b] + (g.only.split(",") if g.only else [])):
        if x not in keys: raise SystemExit(f"没有这个步骤: {x}\n可选: {', '.join(keys)}")

    if g.only:
        sel = [s for s in STEPS if s[0] in g.only.split(",")]
    else:
        i = keys.index(g.a) if g.a else 0
        j = keys.index(g.b) if g.b else keys.index(DEFAULT_END)
        sel = STEPS[i:j+1]

    c = Ctx(g.scene, g.run)
    if g.check: return do_check(c)
    os.makedirs(c.LOG, exist_ok=True)
    os.environ.setdefault("RUN_DIR", c.R1)      # 领卡的名额和"谁在用哪张卡"都记在本轮目录里, 子进程也要知道
    if not g.child:
        print(f"场景 {c.scene}  轮次 {c.run}")
        print(f"帧   {c.FR}")
        print(f"结果 {c.R1}")
        print(f"步骤 {' -> '.join(s[0] for s in sel)}" + ("   [干跑]" if g.dry_run else "") + "\n")
    # ★ 只在 --only 给了多个步骤时并行(驱动就是这么调的): --from/--to 会跨段, 跨段的步骤对没核对过读写冲突
    if g.only and len(sel) > 1 and _cm.parallel_on() and not g.dry_run:
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
            if k in GPU_STEPS - FANOUT_STEPS and not g.dry_run:
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

if __name__ == "__main__":
    main()
