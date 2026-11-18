#!/usr/bin/env python
"""对不该复用的类, 每个实例各自裁一张最好的图去生成 mesh.
够大的(>= SOLO_MIN_PX 像素, 见 solo_gate.py)单独生成; 太小的(生成必是垃圾)继续复用。"""
import os, sys, json, numpy as np, cv2
from pycocotools import mask as mask_util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imgio import imwrite_stable
import review_decisions as review
TAG=os.environ.get("TAG","ts_r1")
R1=os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
FR=os.environ.get("FR","/home/jincheng/sam3d/data/outdoor/dataset/timesquare/frames")
SOLO=set(x for x in os.environ.get("SOLO","car").split(",") if x)
KEEP=set(x for x in os.environ.get("SOLO_KEEP","").split(",") if x)   # 白名单: 可合并, 共用一个 mesh
# SOLO 和 EXCLUDE 可能同时含一个词 (cyberpunk 上 sofa 就是), 排除优先 —— 否则逐实例
# 生成一堆 mesh, 组装时全被扔。
EXCLUDE=set(x.strip() for x in os.environ.get("EXCLUDE","").split(",") if x.strip())
# 门槛按像素, 定义在 solo_gate.py。老键 TH(画面百分比)还在环境里就报错, 别静默换门槛。
if "TH" in os.environ:
    raise SystemExit(f"[失败] 环境里还有 TH={os.environ['TH']} —— 已废弃, 改用 SOLO_MIN_PX(像素数)")
from solo_gate import min_px
MIN_PX=min_px()
BOXES=os.environ.get("BOXES", f"{R1}/06_all_boxes.json")
d=json.load(open(BOXES))
if "*" in SOLO: SOLO=set(d)-KEEP          # "*" = 除白名单外全部逐实例生成
SOLO=SOLO-EXCLUDE                         # ★ 必须在 "*" 展开之后减, 否则 "*" 场景漏掉
print(f"[SOLO] {len(SOLO)} 类逐实例生成, 白名单(共用mesh) {sorted(KEEP)}"
      + (f", EXCLUDE 挡掉 {sorted(EXCLUDE & (set(d)|SOLO))}" if EXCLUDE else ""))
# ★ 2026-10-12 组装时一定会被扔的盒子, 这里就不生成。规则和门槛都在 box_guards.py, 和组装(assemble_scene.py)用的是同一份:
#     太矮 / 太大(绝对守卫) —— office r12 的苹果(3 厘米)、鼠标、两个笔筒...共 6 个;
#     同一件量了两遍(去重, DEDUP_BOX 开着才判) —— office r12 的两张桌子。
#   以前照样生成(每个约 1 分钟), 到组装才扔。场景不变, 只是不白生成。
#   去重那条要看"同词的其他盒子": 开着召回时组装读的是并进了召回物体的盒子文件, 和这里读的不是同一份, 结论可能不一样,
#   所以流水线在召回开着时不把 DEDUP_BOX 传过来(见 o_solo / s_solo) —— 宁可白生成, 不能少生成。
import box_guards
_DUPB = {}
if box_guards.DEDUP_BOX:
    from world_frame import load_up
    _DUPB = box_guards.dup_boxes(d, load_up(f"{R1}/02_da3", "逐实例生成")[0], EXCLUDE)
# "太矮"那条有个例外: 组装会先把"立在地上"的类补到地面再判(腿被挡住只量到桌面的桌子, 0.06 -> 0.74 m)。
# 补不补由这三个设置决定, 流水线传的和传给组装的是同一份; 这三个条件和 assemble_scene.py 里开那一段的条件一样。
_rp = os.environ.get("REST", "")
_REST = json.load(open(_rp)) if _rp and os.path.exists(_rp) else {}
_FLOOR_EXTEND = os.environ.get("FLOOR_EXTEND", "0") == "1"
_HUNG = box_guards.hung_labels()
OV_INST=review.overrides_inst(R1)     # {(词, 实例号): 帧号} —— 人在闸口点名换的那张抠图
# ★ VLM 逐实例校验的否决名单(verify_picks.py 那一段写的)。
#   以前逐实例这条路上一个机器校验都没有: 共用的词验代表那一张, solo 的词一张都不验,
#   于是图越多越没人管。pedestrian/r2 实测 person 15 个里 3 个根本不是人。
#   文件不在就当空的 —— 老轮次没有这份, 不该因此跑不了。
import solo_plan
VLM_REJ = solo_plan.vlm_rejected(R1)
RELEASE = review.vlm_released(R1)     # 人看过、VLM 说不是也照样要的
if VLM_REJ:
    print(f"[VLM否决] {sum(len(v) for v in VLM_REJ.values())} 个实例被 VLM 否决: "
          + json.dumps({k: sorted(v) for k, v in VLM_REJ.items()}, ensure_ascii=False)
          + (f"; 其中人工放行 {sorted(RELEASE)}" if RELEASE else "; 没有人工放行的, 都不生成"))
# ★ 2026-11-10 生不生成、用哪一帧, 规则搬到 solo_plan.py —— 本轮页"会生成几个"调的是同一份。
#   以前那边自己数, hitch_cuts_indoor r1 上说 28 个, 这里实际生成 19 个。这里只管裁图落盘。
_never = []
jobs=[]
for r in solo_plan.plan(R1, d, list(SOLO), tag=TAG, min_px=MIN_PX, dup=_DUPB, rest=_REST,
                        floor_extend=_FLOOR_EXTEND, hung=_HUNG, vlm_rej=VLM_REJ, released=RELEASE, ov_inst=OV_INST):
    w, oid = r["词"], r["实例"]
    for t in r["提示"]: print(t)
    if r["类"] in (solo_plan.GUARD, solo_plan.DUP):
        _never.append((w, oid)); print(f"[不生成] {w}#{oid}: {r['原因']}")
    if r["原因"] is not None:
        continue
    i,a,fi,e = r["行"]
    z=np.load(f"{R1}/03_masks/{w.replace(' ','_')}.npz",allow_pickle=True)
    counts=z["counts"]; H,W=z["hw"]
    pct=100*a/(H*W)
    m=mask_util.decode({"counts":counts[i],"size":[int(H),int(W)]}).astype(np.uint8)
    name=r["生成名"]
    dd=os.path.expanduser(f"~/sam3d/sam-3d-objects/my_images/{name}"); os.makedirs(dd,exist_ok=True)
    imwrite_stable(f"{dd}/frame.png", cv2.imread(f"{FR}/{fi:04d}.jpg"))
    imwrite_stable(f"{dd}/mask.png", m*255)
    jobs.append(name); print(f"[{w}#{oid}] 第{fi}帧 占{pct:.2f}% ({a}px) -> {name}")
if _never:
    print(f"[不生成] 共 {len(_never)} 个: 组装时一定会被扔(太矮 / 太大当成量错了, 或者同一件量了两遍), 不白生成")
open(f"{R1}/logs/solo_jobs.txt","w").write("\n".join(jobs))
print(f"\n[done] {len(jobs)} 个待单独生成")
