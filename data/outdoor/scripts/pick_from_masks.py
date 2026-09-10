#!/usr/bin/env python
"""每类挑一个最好的实例和它最完整的一帧, 存 frame.png + mask.png 给 SAM3D.
只从 03_masks/*.npz 读, 不碰 GPU, 所以调打分可以秒级重跑.

★ 2026-09-08: 这一步只负责"整类共用一个 mesh"的那些词。逐实例的词一律交给
  gen_per_instance.py, 这里不再写它们的类目录 —— 以前两边都写, 而一个词只有一个实例时
  (beach 的 door) 两边挑中的是同一个实例的同一帧, 写出 <TAG>_door 和 <TAG>_door_1
  两个目录、同一张图, 生成那步照目录数就跑两遍 SAM3D, 出两个一模一样的 mesh。
  组装那步的 pick_mesh 有三条退路(实例 mesh -> 类 mesh -> 该词任一实例 mesh),
  所以不写类目录不会让谁消失: 太小没单独生成的实例, 会借同类别人的 mesh。

打分 = mask面积 x (1-贴边惩罚) x min(帧数/10, 1)
首轮的 20 帧硬门槛去掉了 —— 打分已经在挑最大最完整的, 硬门槛只会误杀
(桌子 12 个实例全被它砍掉, 一个 mesh 都没生成)。只留 2 帧的下限防单帧误检。
"""
import os, sys, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imgio import imwrite_stable
from word_rank import base_word
from pycocotools import mask as mask_util
TAG=os.environ.get("TAG","ts_r1")

ROOT = os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
FR   = os.environ.get("FR","/home/jincheng/sam3d/data/outdoor/dataset/timesquare/frames")
MK   = f"{ROOT}/03_masks"; OUT=f"{ROOT}/04_picks"; os.makedirs(OUT, exist_ok=True)
MIN_FRAMES  = int(os.environ.get("MIN_FRAMES","2"))       # 硬下限, 防单帧误检
FRAME_SAT   = float(os.environ.get("FRAME_SAT","10"))     # 帧数加分到此饱和
MIN_AREA_PCT= float(os.environ.get("MIN_AREA_PCT","0.05"))# 太小的生成一定是垃圾
# ★ 组装那步会排除的词, 这里就别挑图了 —— 排除以前只在组装生效, 前面照挑图、照验图、
#   照生成 mesh, 跑完再扔。cyberpunk 上那个错的 sofa 就是这么白跑出来的。
#   过滤放在最早这一步: 省掉 VLM 验图 + SAM3D 生成两段 GPU 时间。
EXCLUDE = {w.strip() for w in os.environ.get("EXCLUDE","").split(",") if w.strip()}
# ★ 跨词去重(dedup_words.py)判掉的同义词, 效果上和 EXCLUDE 一样 —— 不挑图不写盘, 后面
#   自然也不生成。但故意不并进 EXCLUDE 这个变量, 分成两个来源:
#   EXCLUDE 是人写在 pipeline CFG 里的判断("这个场景里压根没有沙发"), 一条一条有出处;
#   淘汰词是这一轮从 mask 算出来的, 换个场景/换个门槛就不一样。混成一个变量后, 一是
#   干跑打印的命令行会依赖一个还没生成的 json, 二是日志里再也分不清某个词到底是人不要它
#   还是算法判它重复 —— 而这两种情况的排查方向完全相反。
#   文件不存在就是空集: 去重那步是后加的, 老的结果目录里没有这个文件, 行为和以前一样。
DEDUP = os.environ.get("DEDUP", f"{ROOT}/03_dedup_words.json")
#   ★ 名单的粒度是"实例"不是"词": 同一个物理物体被几个词各追一遍, 该合的是实例。整词淘汰
#   会顺手扔掉细词多追到的那些物体 —— 淘汰词只在"这个词一个实例都不剩"时才有, 拿它做整词
#   短路只是为了日志里一眼看得出谁整个没了; 部分淘汰的词照常往下走, 在下面逐实例过滤。
DUPOUT = set(); DUPINST = {}
if DEDUP and os.path.exists(DEDUP):
    _d = json.load(open(DEDUP))
    DUPOUT = set(_d.get("淘汰词", []))
    DUPINST = {w: set(v) for w, v in _d.get("淘汰实例", {}).items()}
    print(f"[去重] 读到 {DEDUP}, {sum(len(v) for v in DUPINST.values())} 个实例被同组的"
          f"另一个词代表(其中 {len(DUPOUT)} 个词整个没剩), 共 {len(_d.get('组', []))} 组")
# 逐实例的词在这里跳过。判据要和 gen_per_instance.py 完全一致, 否则会出现
# "这里以为逐实例所以不写, 那边以为共用所以也不写" —— 那个词就整个消失了。
SOLO_CFG = {x.strip() for x in os.environ.get("SOLO","").split(",") if x.strip()}
KEEP     = {x.strip() for x in os.environ.get("SOLO_KEEP","").split(",") if x.strip()}
SOLO_TH  = float(os.environ.get("SOLO_TH","0.5"))
HAVE = {int(f[:4]) for f in os.listdir(f"{ROOT}/02_da3/depth") if f.endswith("_depth.npy")}

picked={}; skipped_solo=[]
for fn in sorted(os.listdir(MK)):
    z=np.load(f"{MK}/{fn}", allow_pickle=True)
    meta=z["meta"]; counts=z["counts"]; H,W=z["hw"]; word=str(z["word"])
    # 排除表是精确匹配, 派生子类(umbrella~1)不会被 "umbrella" 挡住 —— 一并按母词判。
    if word in EXCLUDE or base_word(word) in EXCLUDE:
        print(f"[{word}] 在 EXCLUDE 里, 不挑图不生成"); continue
    if word in DUPOUT:
        print(f"[{word}] 跨词去重判为重复(同组里没赢), 不挑图不生成"); continue
    if len(meta)==0 or meta["area"].max()==0: continue
    live={}
    _dead = DUPINST.get(word, ())
    for i,(oid,fi,a,e) in enumerate(meta):
        if a==0 or int(fi) not in HAVE: continue
        if int(oid) in _dead: continue      # 这个实例由同组另一个词代表, 别再送一份去生成
        live.setdefault(int(oid),[]).append((i,int(fi),int(a),float(e)))
    best=None
    for oid,rows in live.items():
        if len(rows)<MIN_FRAMES: continue
        nf=min(len(rows)/FRAME_SAT, 1.0)                     # 帧数因子, 10 帧封顶
        for i,fi,a,e in rows:
            s = a * (1-min(e*4,0.9)) * nf
            if best is None or s>best[0]: best=(s,oid,fi,a,e,i,len(rows))
    if best is None:
        # 这一支原来是静默 continue, 于是这个词既不写类目录也不上逐实例名单, 从场景里
        # 无声消失(beach 的 planter 只活在第 312 帧、traffic cone 同理, 都在深度覆盖之外)。
        # 一个词落空和被人排除掉的排查方向完全不同, 得让它在日志里留下一行。
        print(f"[{word}] {'实例都不在有深度的帧上' if not live else f'剩下的实例都活不满 {MIN_FRAMES} 帧'}"
              f", 不挑图不生成" + (f" (去重先判掉了 {len(_dead)} 个实例)" if _dead else ""))
        continue
    s,oid,fi,a,e,i,nfr = best
    # 这个词会走逐实例吗? 走的话这里不写类目录, 交给 gen_per_instance.py。
    # 但要确认它至少有一个实例够大 —— 一个都不够大的话逐实例那边什么都不会生成,
    # 那就还得由这里写一个类目录兜底, 否则这个词从场景里消失。
    _solo = (word not in KEEP) and (("*" in SOLO_CFG) or (word in SOLO_CFG))
    if _solo:
        _big = any(len(rows) >= MIN_FRAMES and
                   100*max(r[2] for r in rows)/(H*W) >= SOLO_TH
                   for rows in live.values())
        if _big:
            skipped_solo.append(word); continue
        print(f"[{word}] 逐实例词, 但没有实例过 {SOLO_TH}% 门槛, 由这里写一个类目录兜底")
    if 100*a/(H*W) < MIN_AREA_PCT: 
        print(f"[{word}] 最好的也只占 {100*a/(H*W):.3f}% 画面, 太小, 跳过"); continue
    m=mask_util.decode({"counts":counts[i],"size":[int(H),int(W)]}).astype(bool)
    name=word.replace(" ","_")
    d=os.path.expanduser(f"~/sam3d/sam-3d-objects/my_images/{TAG}_{name}"); os.makedirs(d,exist_ok=True)
    img=cv2.imread(f"{FR}/{fi:04d}.jpg")
    # 内容没变就不重写 —— mtime 一刷新, 下游 gen 会把所有 mesh 当成过期全部重烧
    imwrite_stable(f"{d}/frame.png", img); imwrite_stable(f"{d}/mask.png", m.astype(np.uint8)*255)
    ov=img.copy(); ov[m]=(0.5*ov[m]+0.5*np.array([60,60,230])).astype(np.uint8)
    cv2.imwrite(f"{OUT}/{name}.jpg", ov, [cv2.IMWRITE_JPEG_QUALITY,88])
    picked[word]=dict(obj_id=oid, frame=fi, 面积占比=round(100*a/(H*W),3), 活帧数=nfr,
                      贴边=round(e,3), 候选数=len(live), 目录=d)
    print(f"[{word}] 实例#{oid} 第{fi}帧 占{100*a/(H*W):.2f}% 活{nfr}帧 贴边{e:.3f} ({len(live)}个候选)")
json.dump(picked, open(f"{OUT}/picks.json","w"), ensure_ascii=False, indent=1)
if skipped_solo:
    print(f"\n[逐实例] {len(skipped_solo)} 个词交给 gen_per_instance.py, 这里不写类目录"
          f"(不然同一张 mask 会被写两遍): {', '.join(sorted(skipped_solo))}")
print(f"\n[done] {len(picked)} 类待生成(整类共用一个 mesh 的)")
