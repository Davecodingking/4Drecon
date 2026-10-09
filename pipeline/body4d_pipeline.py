#!/usr/bin/env python
"""SAM-Body4D 链: 动态视频 -> 每个人逐帧的三维人体网格。

和 outdoor_pipeline.py / indoor_pipeline.py 同一套写法: 这里只管步骤顺序、每步用哪个环境、
参数是多少和为什么。算法在 ~/sam3d/sam-body4d/body4d_*_5090.py 里。

依赖顺序
    frames -> track -> check -> [人工闸: 挑人] -> pick -> mesh -> report

    frames  抽帧: 降 fps + 等比缩放 + 补零四位
    track   SAM 3 文字提示 "person" 跟踪, 出 images/ + masks/
    check   出 mp4 和体检图给人看 (跟丢没 / 串号没)
    pick    把人工挑的 id 落成 picked.json
    mesh    SAM 3D Body 出网格 (可选先跑 Diffusion-VAS 补全)
    report  网格叠回原片的 mp4 + 跟踪帧数 vs 网格帧数的对账

反直觉之处
  * 不用上游的 scripts/offline_app.py。它是"检人->跟踪->出网格"一条龙中间不能停,
    而且首帧检人用的是 SAM 3D Body 自带的 ViTDet、只在首帧下一次框 ——
    中途走进画面的人永远跟不到。我们用 SAM 3 的文字提示替掉了这一段。
  * 帧目录在结果树底下 (00_frames/), 不在 dataset/<scene>/frames。因为这里的帧是
    抽稀+缩放过的, 和 outdoor 那套原分辨率全帧不是一回事, 同名会互相污染。
  * 单跑时产物落在 ~/sam3d/sam-body4d/outputs/<场景>/<轮次>/(老轮次 b1/b2 都在这)。
    主流程拉起时(people_track / people_mesh)给 BODY_OUT, 落在场景轮次底下的 body4d/ ——
    一轮的东西都在 data/outdoor/result/<场景>/<轮次>/ 里, 看视频不用去第三方仓库里翻。

坑
  * ★ body4d 必须是独立 conda 环境。sam-body4d 仓库内嵌了另一份 SAM 3, 包名同样叫 sam3,
    装进我们的 sam3 环境会把 track_all.py 依赖的那份静默替换掉。
  * ★ 挑人会挑出不连续的 id (比如 1/2/4), 而上游用"第几个"冒充"是谁"。
    mesh 那步会重编号成连续 1..N 并写 relabel.json。已给上游打了补丁, 但重编号仍保留 ——
    少依赖一个补丁少一个风险。
  * ★ 开补全 (OCC=1) 在 32G 上会 OOM。Diffusion-VAS 两个 SVD 加 SAM 3D Body 一起驻留,
    实测撞在 25G 还要再申请 7.2G。要开先降 BATCH 和 MAX_OCC_LEN。
  * 视频一律 H.264/yuv420p。cv2 的 mp4v 不报错但浏览器不认。

用法
    python body4d_pipeline.py --list
    python body4d_pipeline.py --scene mayday                     # 跑到默认终点
    python body4d_pipeline.py --scene mayday --to check          # 停在人工闸
    python body4d_pipeline.py --scene mayday --from pick         # 挑完人接着跑
    python body4d_pipeline.py --scene mayday --only mesh,report
    python body4d_pipeline.py --scene mayday --check             # 只看产物在不在/过没过期
    python body4d_pipeline.py --scene mayday -n                  # 干跑, 打印能直接粘的命令
    PICK=1,2,4 python body4d_pipeline.py --scene mayday --from pick
"""
import argparse, glob, json, os, shlex, subprocess, sys, time

# ★ 驱动层的公共件在 _common.py(见那个文件的文件头)。三条流水线共用同一把 GPU 锁、
#   同一个跑命令/计时/配置解析的实现 —— 复制粘贴过的东西一定会漂移。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (GpuSlot, wait_gpu, sh, envcmd, stamp_path, parallel_on,
                     timing as _timing, resolve_cfg as _resolve_cfg)
import fanout                  # 步骤内部拆分: 同一个脚本起几份, 各用一张卡

PIPE = os.path.dirname(os.path.abspath(__file__))

HOME = os.path.expanduser("~")
SAM3D = f"{HOME}/sam3d"
B4D = f"{SAM3D}/sam-body4d"                    # 上游仓库 + 我们的 body4d_*_5090.py
OUTROOT = f"{B4D}/outputs"
DATASET = f"{SAM3D}/data/outdoor/dataset"

PY = {
    # ★ body4d 独立环境: python 3.12 + torch 2.8.0+cu128 + detectron2(自己编的)
    #   + 内嵌 sam3 + imageio-ffmpeg。装法和坑见 sam-body4d/NOTES_5090.md
    "body4d": f"{HOME}/miniforge3/envs/body4d/bin/python",
    # 抽帧只要 cv2, 借 sam3d 环境, 不占 body4d 的载入时间
    "sam3d":  f"{HOME}/miniforge3/envs/sam3d/bin/python",
}

CFG = dict(
    # ---- 抽帧 ----
    FPS_OUT    = "7.5",   # Diffusion-VAS 底子是 SVD, 上游按 ~8fps 喂。源 30fps 取 stride 4
                          # 正好 7.5, 时间轴不用重采样
    STRIDE     = "0",     # 非 0 则直接指定隔几帧取一, 忽略 FPS_OUT
    LONG_SIDE  = "1024",  # 长边缩到多少。内部补全固定跑 512x1024, 长边给到 1024 已是 1:1。
                          # 关心的是"人在画面里有多高"而不是画面多大 (SAM 3D Body 按人裁 patch)。
                          # mayday 人高 345~409px 一帧不丢; pedestrian 148~372px 仍有帧被跳。
                          # ★ 所以人像高度**不是**决定跳不跳帧的因素 —— 提分辨率救不回来, 实测过。
    MAX_FRAMES = "96",    # 帧数上限。论文自测 90 帧/5 人/A100-80G, 先不越过
    SCENE_R1   = "",      # 场景重建的结果目录。给了就不从视频抽帧, 直接用场景那套帧(frames_r1)
                          # 和它的相机焦距 —— 人要放回这个场景, 每一帧都得有场景的相机和深度。
                          # ★ 自己抽帧的话帧对不齐: pedestrian 44 帧里只有 30 帧能对上场景相机。
    START_SEC  = "0",
    JPEG_Q     = "95",

    # ---- 跟踪 ----
    PROMPT       = "person",
    CONF         = "0.5",   # SAM 3 的 new_det_thresh。人多的场景调高
    MIN_AREA_PCT = "1.0",   # 最大面积不到画面这个百分比的实例整条丢掉。
                            # ★ 0.1 太松: mayday 实测混进 3 个垃圾(0.5%/0.5%/1.6%),
                            #   真人是 16.5%/9.9%/14.2%, 中间空档很大, 1.0 卡得干净
    JUMP_TH      = "0.25",  # 质心跳变告警(占画面宽)。★ 0.08 对打斗太紧 ——
                            # 7.5fps 下挥拳扑击一帧移动 10~20% 是真实的, 实测 22 条全是误报
    IOU_TH       = "0.20",

    # ---- 人工闸 ----
    PICK = "",              # 逗号分隔的 id; 空 = 自动挑: MIN_AREA_PCT 过滤后、帧数够的全部。
                            # 看完 03_check.jpg 再填, 或者直接改 SCENES
    MIN_PICK_FRAMES = "5",  # 自动挑人时, 出现不到这么多帧的 id 不要(一闪而过的多半是误检/路过的半个人)
    PICK_BY = "",           # 这批 PICK 是谁定的, 原样记进 picked.json 的 decided_by。主流程拉起时由它给(闸口上有人改过 = human,
                            # 没人动过 = auto); 空 = 单跑: 给了 PICK 算 human, 没给算 auto

    # ---- 出网格 ----
    OCC         = "0",      # 1 开 Diffusion-VAS 补全。★ 32G 上开了会 OOM, 见文件头。
                            # ★ 2026-09-21 补全的三份权重(Diffusion-VAS x2 + Depth-Anything V2, 16G)
                            #   已删 —— 32G 上从来没开成过。要开得重新下, 见 sam-body4d/NOTES_5090.md 权重表
    BATCH       = "8",      # 上游默认 64 (论文写 32), 我们显存只有人家 40%。
                            # ★ 注意 max_occ_len 会被 min(它, BATCH) 夹, 两个参数不独立
    MAX_OCC_LEN = "25",

    # ---- 出图 ----
    RP_FPS = "8",
    ALPHA  = "0.6",         # 网格叠回原片的不透明度
)

SCENE_TAG = {"mayday": "md", "pedestrian": "pd", "office": "of"}
SCENES = {
    # 2538x1066 (2.381:1 超宽), 335 帧。酒吧三人打架, 重度互遮挡 + 运动模糊 + 低光。
    # 人占画面高 60~90%, 分辨率不是瓶颈。
    "mayday": dict(PICK="1,2,4"),      # 跟出 6 个, 1/2/4 是真正打架的三个人

    # 1920x1080, 269 帧。游戏引擎渲染, 很多行人但动作慢。
    # ★ 镜头在平移, 第 50 帧之后画面里基本没人了。SAM 3 只在首帧下提示,
    #   跑满 68 帧的话后半段全是"跟丢", 但那不是跟丢, 是人真的走了。
    #   所以截到 44 帧 (约前 6 秒), 让数字有意义。
    # ★ LONG_SIDE 用原生 1920。注意: 这**不是**为了救那几帧被跳过的 ——
    #   实测从 1024 提到 1920 (人像高 78->148), 跳过的还是同样那 8/1/3 帧,
    #   一帧没救回来。跳过的真实原因是 SAM 3D Body 自己返回全零 bbox 拒绝出网格,
    #   判据在它内部, 还没定位。用 1920 纯粹是"人本来就小, 没必要再缩"。
    "pedestrian": dict(MAX_FRAMES="44", LONG_SIDE="1920"),

    # 室内办公室, 1920x1080。人既有走动的也有坐着的。帧和焦距都直接取场景 r1 那套,
    # 人最后要放回这个场景里 (body_into_scene / indoor_pipeline 的 people 步)。
    "office": dict(SCENE_R1=f"{SAM3D}/data/indoorOutput/office/r1"),

    # 足球场远景, 1280x720 / 205 帧。★ 守门员离得远, 最大面积只有 0.7%, 默认 1.0% 会把他整条丢掉
    #   (b1 就这么丢的)。0.2% 以下那两个是碎片/误检。1/2 是近处两个球员, 3 是守门员(像素高中位 115, 69 帧全出网格)。
    "soccer": dict(MIN_AREA_PCT="0.2", PICK="1,2,3"),
}


def resolve_cfg(scene, R1=None):
    """本文件的 CFG/SCENES 接到共用实现上。这条链没有人工闸决策文件, 所以不传 review。
    生效顺序: CFG 默认 <- 场景覆盖 <- 环境变量, 详见 _common.py。"""
    return _resolve_cfg(CFG, SCENES, scene)


class Ctx:
    def __init__(self, scene, run):
        self.scene, self.run = scene, run
        self.tag = f"{SCENE_TAG.get(scene, scene[:3])}_{run}"
        d = f"{DATASET}/{scene}"
        # ★ 这两段素材叫 <scene>.mp4, 不是 outdoor 写死的 video.mp4
        cand = [f"{d}/{scene}.mp4", f"{d}/video.mp4"] + sorted(glob.glob(f"{d}/*.mp4"))
        self.VID = next((p for p in cand if os.path.exists(p)), cand[0])
        self.R1 = os.environ.get("BODY_OUT") or f"{OUTROOT}/{scene}/{run}"
        self.LOG = f"{self.R1}/logs"
        self.FR = f"{self.R1}/00_frames"
        self.TRK = f"{self.R1}/02_body4d"
        self.cfg = resolve_cfg(scene, self.R1)

    @property
    def MESH(self):
        """出网格的运行目录。开/关补全各一份, 才能并排比。"""
        return f"{self.R1}/04_mesh_occ{1 if self.cfg['OCC'] == '1' else 0}"


GPUENV = dict(CUDA_HOME="/usr/local/cuda-12.8", TORCH_CUDA_ARCH_LIST="8.6",
              PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")


# ============================ 各步骤 ============================

def s_frames(c, dry):
    """抽帧。降 fps + 等比缩放 + 补零四位。理由见 body4d_frames.py 的文件头。
    给了 SCENE_R1 就不抽: 直接软链场景那套帧, kept_indices 取场景的 frame_map ——
    这样人体的每一帧都对得上场景的一个相机, body_into_scene 按同一张表找回相机。"""
    g = c.cfg
    if g["SCENE_R1"]:
        code = f'''
import json, os, sys
sys.path.insert(0, {SAM3D + "/pipeline"!r})
from _common import load_frame_map
da3 = {g["SCENE_R1"] + "/02_da3"!r}
fm = load_frame_map(da3)
src = json.load(open(da3 + "/frame_map.json")).get("frames_dir") if os.path.exists(da3 + "/frame_map.json") else None
src = src or json.load(open(da3 + "/cameras.json"))["source"]
names = sorted(fm, key=lambda n: fm[n])
out = {c.FR!r}; os.makedirs(out, exist_ok=True)
for f in os.listdir(out):
    if f.endswith((".jpg", ".json")): os.remove(os.path.join(out, f))
# ★ 软链按位置重编号成 0000..N-1: body4d 按位置序号读帧。室外场景帧按原视频帧号命名且不连续
#   (soccer: 0000 0001 0002 0003 0005 ...), 原样链过来就对不上; 室内本来就连续, 重编号等于没变。
#   帧和相机的对应靠 kept_indices(原视频帧号), 不靠文件名 —— body_into_scene 就是按它找相机的。
for i, n in enumerate(names):
    os.symlink(os.path.join(src, n), os.path.join(out, f"{{i:04d}}.jpg"))
from PIL import Image
W, H = Image.open(os.path.join(src, names[0])).size
# 帧率给动画时间轴用(原视频帧号 / fps)。frame_map 里没记的话去数据集找原视频读
fps = json.load(open(da3 + "/frame_map.json")).get("fps") if os.path.exists(da3 + "/frame_map.json") else None
if not fps:
    import glob, cv2
    scene = os.path.basename(os.path.dirname({g["SCENE_R1"]!r}))
    vids = sorted(glob.glob({SAM3D!r} + f"/data/indoor/dataset/{{scene}}/*.mp4")
                  + glob.glob({SAM3D!r} + f"/data/outdoor/dataset/{{scene}}/*.mp4"))
    fps = cv2.VideoCapture(vids[0]).get(cv2.CAP_PROP_FPS) if vids else 0
    if not fps:
        fps = 30.0; print("[警告] 找不到原视频的帧率, 按 30fps 算 —— 动画会快慢不对")
json.dump(dict(scene_r1={g["SCENE_R1"]!r}, frames_dir=src, out_wh=[W, H], n_frames=len(names),
               src_fps=float(fps),
               kept_indices=[int(fm[n]) for n in names],
               note="帧是场景帧的软链(按位置重编号), kept_indices = 场景 frame_map 里的原视频帧号"),
          open(os.path.join(out, "frames.json"), "w"), ensure_ascii=False, indent=1)
print(f"[场景帧] {{len(names)}} 帧 {{W}}x{{H}} <- {{src}}  (软链, 原视频帧号 {{fm[names[0]]}}..{{fm[names[-1]]}})")
'''
        sh([PY["sam3d"], "-c", code], f"{c.LOG}/00_frames.log", dry)
        return
    sh(envcmd(PY["sam3d"], f"{SAM3D}/pipeline/body4d_frames.py", {},
              ["--video", c.VID, "--out", c.FR, "--fps", g["FPS_OUT"],
               "--stride", g["STRIDE"], "--long-side", g["LONG_SIDE"],
               "--max-frames", g["MAX_FRAMES"], "--start-sec", g["START_SEC"],
               "--quality", g["JPEG_Q"]]),
       f"{c.LOG}/00_frames.log", dry)


def s_track(c, dry):
    """SAM 3 文字提示跟踪。产出的 images/+masks/ 就是 SAM 3D Body 那半段要吃的格式。"""
    g = c.cfg
    wait_gpu(9000, dry)
    sh(envcmd(PY["body4d"], f"{B4D}/body4d_track_5090.py", GPUENV,
              ["--frames", c.FR, "--out", c.TRK, "--prompt", g["PROMPT"],
               "--conf", g["CONF"], "--min-area-pct", g["MIN_AREA_PCT"],
               "--jump-th", g["JUMP_TH"], "--iou-th", g["IOU_TH"]]),
       f"{c.LOG}/02_track.log", dry, cwd=B4D)


def s_check(c, dry):
    """体检: 出 03_track.mp4 + 03_check.jpg。★ 这是人工闸 —— 看完再决定挑谁。"""
    sh(envcmd(PY["body4d"], f"{B4D}/body4d_check_5090.py", {},
              ["--dir", c.TRK, "--fps", c.cfg["RP_FPS"], "--prefix", "03"]),
       f"{c.LOG}/03_check.log", dry, cwd=B4D)
    if not dry:
        print(f"    ★ 人工闸: 看 {c.R1}/03_track.mp4 和 03_check.jpg,")
        print(f"      决定要还原哪几个人, 填进 SCENES['{c.scene}']['PICK'] 或 PICK=1,2,4 环境变量")


def s_pick(c, dry):
    """把人工决定落成 picked.json。空 PICK 就是留下 track 过滤后的全部。"""
    rep = f"{c.TRK}/track_report.json"
    pick = c.cfg["PICK"]
    code = f'''
import json, os
rep = json.load(open({rep!r}))
cov = {{int(k): v for k, v in rep["coverage"].items()}}
pick = [int(x) for x in {pick!r}.split(",") if x.strip()] or sorted(i for i in cov if cov[i] >= {int(c.cfg["MIN_PICK_FRAMES"])})
miss = [i for i in pick if i not in cov]
if miss:
    raise SystemExit(f"[失败] PICK 里的 id {{miss}} 跟踪结果里没有。有的是 {{sorted(cov)}}")
json.dump(dict(keep_ids=pick, decided_by={c.cfg["PICK_BY"]!r} or ("human" if {pick!r} else "auto(帧数>={c.cfg['MIN_PICK_FRAMES']})"),
               scene={c.scene!r}, run={c.run!r},
               coverage={{str(i): cov[i] for i in pick}},
               dropped={{str(i): cov[i] for i in cov if i not in pick}}),
          open({c.TRK!r} + "/picked.json", "w"), ensure_ascii=False, indent=1)
print(f"[挑人] 留 {{pick}}, 丢 {{[i for i in cov if i not in pick]}}")
'''
    sh([PY["sam3d"], "-c", code], f"{c.LOG}/03_pick.log", dry)


def _part_ok(part, meta):
    """这一份上次已经算好、而且算的是同样的东西(人、参数、跟踪结果都没变): 直接用, 不重算。"""
    try:
        return json.load(open(f"{part}/part.json")) == meta and os.path.exists(f"{part}/mesh_report.json")
    except (OSError, ValueError):
        return False


def s_mesh(c, dry):
    """SAM 3D Body 出网格。OCC=1 时先跑 Diffusion-VAS 补全 (32G 上会 OOM, 见文件头)。
    ★ 2026-09-29 按人拆分(默认关, FANOUT=1 才拆, 原因见 fanout.on): 有几张空卡就起几份, 每份算几个人
      (按每人出现的帧数分, 各份工作量接近), 跑完并回一个目录(mesh_merge.py)。
      只在补全关着的时候拆 —— 补全要看别人的遮挡, 不能按人拆开。"""
    g = c.cfg
    need = 26000 if g["OCC"] == "1" else 15000
    script = f"{B4D}/body4d_mesh_5090.py"
    log = f"{c.LOG}/04_mesh.log"
    args = (["--dir", c.TRK, "--completion", g["OCC"], "--batch-size", g["BATCH"], "--max-occ-len", g["MAX_OCC_LEN"]]
            + (["--cam-json", f"{g['SCENE_R1']}/02_da3/cameras.json"] if g["SCENE_R1"] else []))
    picked = json.load(open(f"{c.TRK}/picked.json")) if os.path.exists(f"{c.TRK}/picked.json") else {}
    ids = picked.get("keep_ids", [])
    k = 1
    if not dry and g["OCC"] != "1" and parallel_on() and fanout.on() and len(ids) > 1:
        k = fanout.how_many(len(ids), need)
    if k <= 1:
        if dry:
            return sh(envcmd(PY["body4d"], script, GPUENV, args + ["--out", c.MESH]), log, dry, cwd=B4D)
        with GpuSlot("mesh", need):
            # ★ 不拆时模型直接往 c.MESH 里写, 它自己不清旧文件: 闸口上点掉一个人再重出(8 个人 -> 7 个), 第 8 个人的目录还在,
            #   新的 5 号目录里混着上一次 5 号(另一个人)的帧 —— 2026-10-09 nfl1 r8 对账报"3 个人身份被写串"。
            #   拆开跑的那条路合并时本来就清(mesh_merge.merge), 所以以前只在"重出时正好没有空卡"才撞上
            import mesh_merge
            mesh_merge.clear_outputs(c.MESH)
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


def s_report(c, dry):
    """叠加视频 + 对账。★ 对账那一半是硬的: 跟踪帧数和网格帧数对不上就退出码非 0。"""
    sh(envcmd(PY["body4d"], f"{B4D}/body4d_report_5090.py", {},
              ["--run", c.MESH, "--track", c.TRK, "--fps", c.cfg["RP_FPS"],
               "--alpha", c.cfg["ALPHA"], "--prefix", "05"]),
       f"{c.LOG}/05_report.log", dry, cwd=B4D)


# ============================ 步骤表 ============================

STEPS = [
    ("frames", "sam3d",  "抽帧: 降fps + 等比缩放 + 补零四位",      s_frames),
    ("track",  "body4d", "SAM 3 文字提示跟踪 -> images/ + masks/", s_track),
    ("check",  "body4d", "体检 mp4 + 总览图 (人工闸)",             s_check),
    ("pick",   "sam3d",  "把人工挑的 id 落成 picked.json",         s_pick),
    ("mesh",   "body4d", "SAM 3D Body 出人体网格",                 s_mesh),
    ("report", "body4d", "网格叠原片 mp4 + 帧数对账",              s_report),
]
DEFAULT_END = "report"
GPU_STEPS = {"track", "mesh"}
# mesh 内部按人拆分, 它起的每一份各领一张卡(s_mesh 里), 主循环不替它领
FANOUT_STEPS = {"mesh"}

# 只列关键产物, 不求全。@FR = 帧目录, @TRK = 跟踪目录, @MESH = 出网格目录
ARTIFACTS = {
    "frames": ["@FR/0000.jpg", "@FR/frames.json"],
    "track":  ["@TRK/images/00000000.jpg", "@TRK/masks", "@TRK/track_report.json"],
    "check":  ["03_track.mp4", "03_check.jpg"],
    "pick":   ["@TRK/picked.json"],
    "mesh":   ["@MESH/mesh_report.json", "@MESH/relabel.json", "@MESH/rendered_frames"],
    "report": ["05_mesh_overlay.mp4", "05_summary.txt"],
}

# ★ 加边之前先确认依赖在 STEPS 里排在前面 —— 没有拓扑排序, 列表顺序就是执行顺序,
#   往回指的依赖会静默失效。
DEPS = {
    "track":  ["frames"],
    "check":  ["track"],
    "pick":   ["track"],          # 不写 check: 闸是给人看的, 不是产物依赖
    "mesh":   ["track", "pick"],
    "report": ["mesh"],
}


def _resolve(c, p):
    for k, v in (("@FR", c.FR), ("@TRK", c.TRK), ("@MESH", c.MESH)):
        if p.startswith(k):
            return v + p[len(k):]
    return os.path.join(c.R1, p)


def do_check(c):
    when, bad = {}, 0
    for k, e, d, _ in STEPS:
        sp = stamp_path(c, k)
        t = os.path.getmtime(sp) if os.path.exists(sp) else None
        paths = [_resolve(c, p) for p in ARTIFACTS.get(k, [])]
        miss = [p for p in paths if not os.path.exists(p)]
        if not paths:
            print(f"  {k:8s} " + (f"跑过 {time.strftime('%F %T', time.localtime(t))}"
                                  if t else "--  (没有产物, 也没跑过)"))
        elif miss:
            print(f"  {k:8s} 缺 {', '.join(os.path.relpath(m, c.R1) for m in miss[:3])}")
            bad += 1
        elif t is None:
            print(f"  {k:8s} 有  (产物在, 无完成戳, 跑一次这步即可)")
        else:
            older = [u for u in DEPS.get(k, []) if u in when and when[u] > t + 1]
            note = f"  <- 比依赖的 {'/'.join(older)} 旧, 该重跑" if older else ""
            print(f"  {k:8s} OK  {time.strftime('%F %T', time.localtime(t))}{note}")
            if older:
                bad += 1
        if t:
            when[k] = t
    print("全部就绪" if not bad else f"{bad} 处需要注意")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="mayday")
    ap.add_argument("--run", default="b1")
    ap.add_argument("--from", dest="a", default=None)
    ap.add_argument("--to", dest="b", default=None)
    ap.add_argument("--only", default=None, help="逗号分隔")
    ap.add_argument("-n", "--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true", help="不跑, 只看产物在不在/有没有过期")
    ap.add_argument("--dump-cfg", action="store_true")
    ap.add_argument("--tag-only", action="store_true")
    ap.add_argument("--dump-deps", action="store_true")
    g = ap.parse_args()

    keys = [s[0] for s in STEPS]
    if g.dump_deps:
        print(json.dumps(dict(steps=keys, deps=DEPS, gpu=sorted(GPU_STEPS)), ensure_ascii=False))
        return
    if g.dump_cfg:
        c = Ctx(g.scene, g.run)
        print(c.tag if g.tag_only else json.dumps(c.cfg, ensure_ascii=False))
        return
    if g.list:
        for k, e, d, _ in STEPS:
            print(f"  {k:8s} [{e:7s}] {d}" + ("   <- 默认终点" if k == DEFAULT_END else ""))
        return
    for x in [g.a, g.b] + (g.only.split(",") if g.only else []):
        if x and x not in keys:
            raise SystemExit(f"没有这个步骤: {x}\n可选: {', '.join(keys)}")

    c = Ctx(g.scene, g.run)
    if g.check:
        do_check(c)
        return

    if g.only:
        sel = [s for s in STEPS if s[0] in g.only.split(",")]
    else:
        i = keys.index(g.a) if g.a else 0
        j = keys.index(g.b) if g.b else keys.index(DEFAULT_END)
        sel = STEPS[i:j + 1]

    os.makedirs(c.LOG, exist_ok=True)
    os.environ.setdefault("RUN_DIR", c.R1)      # 主流程拉起时外面已经给了(整轮共用名额); 单跑时用自己的目录
    print(f"场景 {c.scene}  轮次 {c.run}  标签 {c.tag}")
    print(f"视频 {c.VID}")
    print(f"结果 {c.R1}")
    print(f"补全 {'开' if c.cfg['OCC'] == '1' else '关'}  ->  {os.path.basename(c.MESH)}")
    print(f"步骤 {' -> '.join(s[0] for s in sel)}" + ("   [干跑]" if g.dry_run else ""))
    print()

    for k, e, d, fn in sel:
        t = time.time()
        sep = f"\n{'='*70}\n== {k}  {time.strftime('%F %T')}\n{'='*70}\n"
        print(f"== {k}  [{e}]  {d}", flush=True)
        _o = globals()["sh"]
        globals()["sh"] = lambda *a, **kw: _o(*a, **{**kw, "sep": kw.get("sep", sep)})
        try:
            if k in GPU_STEPS - FANOUT_STEPS and not g.dry_run:
                with GpuSlot(k):
                    fn(c, g.dry_run)
            else:
                fn(c, g.dry_run)
        finally:
            globals()["sh"] = _o
        if not g.dry_run:
            open(stamp_path(c, k), "w").write(time.strftime("%F %T"))
        print(f"   {time.time()-t:.1f}s\n", flush=True)
    print("[全部完成]")


if __name__ == "__main__":
    main()
