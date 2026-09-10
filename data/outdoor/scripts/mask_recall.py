#!/usr/bin/env python
"""召回支路第 3 步: 给"只有框没 mask"的漏检实例补一张 mask, 让它们也能进 SAM3D 生成。

WildDet3D 只给框, 但下游 SAM3D 要 mask 才能生成 mesh。做法是把它那个 2D 框当提示
喂回 SAM3 —— SAM3 本来就支持框提示, 之前只是没用过, 一直用的文字提示。
三个踩过的坑, 前两个叠在一起会让人以为是 SAM3 抠不出来:
  1. 框的格式是 [左上x, 左上y, 宽, 高] 且归一化到 0~1。源码里那个 boxes_cxcywh 是
     它内部转换后的中间量, 不是输入格式 —— 我照着变量名传中心点, 全错。
  2. 框提示返回的是一组候选 (实测 5~12 个), 取第 0 个是错的, 要按和输入框的重合度挑。
     只犯第 2 个错时, 抠出来的是旁边的绿篱和广告牌, 看起来像"模型不行"。
  3. WildDet3D 的 2D 框允许超出画面, 归一化后越界会直接断言失败, 必须先裁到画面内。
定位方法记一下: 拿 SAM3 自己找到过的实例, 用它自己 mask 的外接框反问它一次。
修好前 0/18 还原成功, 修好后 18/18 且 IoU 0.99 —— 这个对照一做, 是"我用错"还是
"模型不行"立刻就分开了。
"""
import os, json, numpy as np, cv2
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imgio import imwrite_stable
from sam3.model_builder import build_sam3_video_predictor
R1  = os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
FR  = os.environ.get("FR","/home/jincheng/sam3d/data/outdoor/dataset/timesquare/frames")
TAG = os.environ.get("TAG","ts_r1")
CKPT= os.path.expanduser("~/sam3d/sam3/checkpoints/sam3.pt")
REC = os.environ.get("RECALL_OUT", f"{R1}/13_recall.json")
GEN = os.path.expanduser("~/sam3d/sam-3d-objects/my_images")
MIN_PCT = float(os.environ.get("REC_MIN_PCT","0.15"))   # 太小的不值得生成, 出来必是垃圾
# 组装那步会排除的词, 这里就别生成了 —— 排除只在组装生效, 生成那步不看, 结果 couch 白跑
# 了一次 70 秒然后被扔掉。两边读同一个 EXCLUDE。
EXCL = {w.strip().replace(" ","_") for w in os.environ.get("EXCLUDE","").split(",") if w.strip()}
OUTDIR = f"{R1}/14_recall_masks"; os.makedirs(OUTDIR, exist_ok=True)

rec = json.load(open(REC))["召回"]
print(f"[输入] {len(rec)} 个漏检实例")
H,W = cv2.imread(f"{FR}/0000.jpg").shape[:2]
pred = build_sam3_video_predictor(checkpoint_path=CKPT)
sid  = pred.handle_request(request=dict(type="start_session", resource_path=FR,
                                        offload_video_to_cpu=True, offload_state_to_cpu=True))["session_id"]
ok = small = miss = 0
out = []
def iou(b, c):
    x0=max(b[0],c[0]); y0=max(b[1],c[1]); x1=min(b[2],c[2]); y1=min(b[3],c[3])
    if x1<=x0 or y1<=y0: return 0.0
    i=(x1-x0)*(y1-y0)
    return i/((b[2]-b[0])*(b[3]-b[1])+(c[2]-c[0])*(c[3]-c[1])-i)

MIN_IOU = float(os.environ.get("REC_MIN_IOU","0.3"))   # 挑出来的候选和输入框重合不够就不要
bad_iou = 0

# ---------------------------------------------------------------- 判词册
# 原来这三条丢弃分支只 += 一个计数器, 于是日志只说得出"丢了 3 个", 说不出**哪个因为什么**。
# 后果不是难看: 召回专挑主线漏掉的东西, 漏掉的多半是被遮挡的, 补抠失败率天生就高 ——
# 也就是说这一步**经常什么都不剩**, 而人在闸 4 只看得到活下来的候选, 于是「召回找到了
# 什么」这件事人根本审不到。闸口叫"召回"却看不见召回。
# 每个实例记一条, 结局是这五种之一, 谁都不许悄悄消失。
判词 = []
def 记(a, 结局, **kw):
    判词.append(dict(obj_id=a["obj_id"], 词=a["词"], 最佳帧=a["最佳帧"],
                     框2d=[round(float(v),1) for v in a["框2d"]],
                     见帧数=a.get("见帧数"), 分=round(float(a.get("分",0)),3),
                     结局=结局, **kw))
    if 结局 != "进表":
        extra = "".join(f" {k}={v}" for k, v in kw.items())
        w, o, f = a["词"], a["obj_id"], a["最佳帧"]
        print(f"  [丢] {w}#{o} 第{f}帧 -> {结局}{extra}", flush=True)
for a in rec:
    fi = a["最佳帧"]
    x0 = float(np.clip(a["框2d"][0], 0, W-1)); y0 = float(np.clip(a["框2d"][1], 0, H-1))
    x1 = float(np.clip(a["框2d"][2], 0, W-1)); y1 = float(np.clip(a["框2d"][3], 0, H-1))
    if x1-x0 < 4 or y1-y0 < 4:
        small += 1; 记(a, "框太窄", 框宽=round(x1-x0,1), 框高=round(y1-y0,1)); continue
    pred.handle_request(request=dict(type="reset_session", session_id=sid))
    r = pred.handle_request(request=dict(
        type="add_prompt", session_id=sid, frame_index=fi,
        bounding_boxes=[[x0/W, y0/H, (x1-x0)/W, (y1-y0)/H]],   # 左上xywh, 归一化
        bounding_box_labels=[1]))
    o = r["outputs"]
    if len(o["out_obj_ids"]) == 0:
        miss += 1; 记(a, "SAM3没给mask"); continue
    M = o["out_binary_masks"]
    M = M.cpu().numpy() if hasattr(M,"cpu") else np.asarray(M)
    best = (0.0, None)
    for k in range(len(M)):
        mk = M[k].astype(bool)
        if mk.ndim == 3: mk = mk[0]
        ys,xs = np.nonzero(mk)
        if len(xs) == 0: continue
        v = iou([x0,y0,x1,y1], [xs.min(),ys.min(),xs.max(),ys.max()])
        if v > best[0]: best = (v, mk)
    if best[1] is None or best[0] < MIN_IOU:
        bad_iou += 1; 记(a, "框对不上", 框重合=round(best[0],3), 门槛=MIN_IOU); continue
    m = best[1]; a["框重合"] = round(best[0],3)
    pct = 100*m.sum()/(H*W)
    if pct < MIN_PCT:
        small += 1; 记(a, "抠出来太小", 抠出pct=round(float(pct),3), 门槛=MIN_PCT,
                       框重合=a["框重合"]); continue
    # ★ 名字只能是这两种格式之一, 不能自创第三种:
    #     <TAG>_<词>          该类共用的 mesh
    #     <TAG>_<词>_<实例号>  这个实例专属的
    # assemble_scene.pick_mesh 是"拿名字去解析"的 —— 末段是不是纯数字, 决定它被当成实例
    # 还是当成类。曾经为了让召回实例复用类 mesh, 把名字写成 <TAG>_<词>_rec<oid>,
    # 指望 pick_mesh 找不到就自动退回类 mesh。对 chair 这种有类 mesh 的词碰巧管用, 但
    # rec10017 不是纯数字, 于是被当成类 mesh、词被解析成 "counter rec10017" —— 一个不存在
    # 的词。SAM3 完全没跟到、只有召回认出来的那些词(counter/bar/booth seat...)没有类 mesh
    # 可退, 整个从场景里消失, 而那恰恰是召回最该体现价值的部分。
    # 正确的分工是: 生不生成由人在闸口决定(复用就别生成), 名字老老实实按格式来。
    name = f"{TAG}_{a['词'].replace(' ','_')}_{a['obj_id']}"
    d = f"{GEN}/{name}"; os.makedirs(d, exist_ok=True)
    img = cv2.imread(f"{FR}/{fi:04d}.jpg")
    imwrite_stable(f"{d}/frame.png", img); imwrite_stable(f"{d}/mask.png", m.astype(np.uint8)*255)
    ov = img.copy(); ov[m] = (0.5*ov[m] + 0.5*np.array([60,230,60])).astype(np.uint8)
    cv2.imwrite(f"{OUTDIR}/{name}.jpg", ov, [cv2.IMWRITE_JPEG_QUALITY,85])
    a["面积占比"] = round(float(pct),3); a["生成名"] = name
    out.append(a); ok += 1
    记(a, "进表", 抠出pct=round(float(pct),3), 框重合=a["框重合"], 生成名=name)
    print(f"  [{a['词']}#{a['obj_id']}] 第{fi}帧 抠出 {pct:.2f}% IoU{best[0]:.2f} -> {name}", flush=True)
json.dump(out, open(f"{R1}/14_recall_ready.json","w"), ensure_ascii=False, indent=1)
# ★ 判词册和 14_recall_ready.json 是两份东西: 后者只有活下来的, 前者一个不落。
#   闸口界面读的是前者 —— 让人看得见"召回找到了什么", 而不只是"什么活下来了"。
json.dump(dict(门槛=dict(REC_MIN_PCT=MIN_PCT, REC_MIN_IOU=MIN_IOU), 判词=判词),
          open(f"{R1}/14_recall_verdict.json","w"), ensure_ascii=False, indent=1)

# 待生成清单: 只有"这个词还没有类 mesh"时才需要生成, 而且一个词只生成一次 (取抠得最大
# 的那个实例当代表), 其余实例复用。
# 为什么: 召回补回来的大多是 SAM3 已经跟到过的词 (chair/bar stool/table...), 那些词的
# 类 mesh 早就生成好了, 再生成一遍既慢又没意义。真正缺 mesh 的是 SAM3 一个实例都没跟到
# 的词 —— cyberpunk 上是 armchair/sofa/couch/side table/floor lamp/bar/counter/bar counter。
# 实测: 61 个待生成降到 8 个。
OUTS = os.path.normpath(f"{GEN}/../outputs")
_have = {d[len(TAG)+1:] for d in os.listdir(OUTS)
         if d.startswith(TAG + "_") and os.path.exists(f"{OUTS}/{d}/{d}.glb")} \
        if os.path.isdir(OUTS) else set()
jobs, seen, skip_ex = [], set(), []
for a in sorted(out, key=lambda x: -x["面积占比"]):
    w = a["词"].replace(" ", "_")
    if w in EXCL:
        if w not in seen: skip_ex.append(w)
        seen.add(w); continue                 # 组装会排除, 生成也跳过
    if w in _have or w in seen: continue      # 类 mesh 已有, 或这个词已经派过一个代表
    seen.add(w); jobs.append(a["生成名"])
if skip_ex: print(f"[跳过] EXCLUDE 里的词不生成: {sorted(set(skip_ex))}")
open(f"{R1}/logs/recall_jobs.txt","w").write("\n".join(jobs)+"\n")
print(f"\n[待生成] {len(jobs)} 个 (共 {len(out)} 个召回实例; 已有类 mesh 的词直接复用): {jobs}")
print(f"\n[done] 补抠成功 {ok}, 太小丢 {small}, 与输入框对不上丢 {bad_iou}, SAM3 没给出 mask {miss}")
print(f"       预览图 {OUTDIR}/  待生成清单 {R1}/logs/recall_jobs.txt")
