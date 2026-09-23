#!/usr/bin/env python
"""召回支路第 2 步: 把逐帧的检测聚成实例, 再和 SAM3 已有的实例合并, 挑出"漏检"。

两件事, 都是因为 WildDet3D 是单帧模型、没有 track ID:
  1. 跨帧关联 —— 同一个物体在不同帧各被检一次, 得认出它们是同一个。
     做法是反投影到世界系之后按三维位置聚类: 场景是静止的, 同一个物体在各帧的世界坐标
     应该落在一起。这比在图像上做关联可靠, 因为相机在动而物体不动。
  2. 和 SAM3 合并 —— 两边都有的, 用 SAM3 的 (它有 mask, 下游要靠 mask 去生成);
     只有 WildDet3D 有的, 就是 SAM3 漏掉的, 送去补抠 mask。

跨词去重也在这里做: chair 和 folding chair 会各检一遍同一把椅子, 位置几乎重合。
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_frame import load_up, export_T, inverse_T
from collections import Counter
R1  = os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
DET = os.environ.get("DET_OUT", f"{R1}/12_wd_detections.json")
SAM = os.environ.get("SAM_BOXES", f"{R1}/06_all_boxes.json")
OUT = os.environ.get("RECALL_OUT", f"{R1}/13_recall.json")
MIN_VIEWS = int(os.environ.get("MIN_VIEWS","3"))   # 至少在这么多帧里被检到才算数, 挡单帧幻觉
DIST_F    = float(os.environ.get("DIST_F","0.5"))  # 聚类半径 = 物体尺寸 x 这个系数
MATCH_F   = float(os.environ.get("MATCH_F","0.7")) # 和 SAM3 实例的匹配半径同理(仅 MATCH_2D=0 时用)
# 这一步的判据全在二维(MATCH_2D=1), UP 实际没参与计算 —— 留着只为一件事:
# 缺 world_up.json 时在这里就早停, 而不是等到下游量盒子才发现天没算出来。
UP, _GH, _FB = load_up(f"{R1}/02_da3", "召回合并")
# ★ 2026-09-08: 判"是不是同一个物体"改到二维做, 三维只留给第 1 步的跨帧归并。
# 为什么: WildDet3D 抬出来的三维盒子和 DA3 的深度对不上(carpark 实测盒子高度整体差 2.37 倍),
# 拿它去和 PCA 的盒子比距离, 等于用一把不准的尺子量。timesquare 48 个"漏检"用二维一验,
# 没有一个落在空白处, IoU>0.5 的有 22 个 —— 同一辆卡车被当成三个新物体。
# 二维为什么可靠: 框2d 是 WD 在原图上直接画的, 抬三维之前的原始输出; SAM3 的 mask 也按帧存着。
# 同一帧同一批像素, 不过深度、不过尺度。
MATCH_2D  = os.environ.get("MATCH_2D","1") == "1"
IOU_SELF  = float(os.environ.get("IOU_SELF","0.5"))  # WD 实例之间: 框 vs 框
IOU_SAM   = float(os.environ.get("IOU_SAM","0.3"))   # WD 的框 vs SAM3 mask 的外接框
SELF_3D_F = float(os.environ.get("SELF_3D_F","0.8")) # 两个 WD 实例没有共同帧时的三维退路半径
FR        = os.environ.get("FR", "")
MK        = f"{R1}/03_masks"

def box_iou(a, b):
    x0,y0 = max(a[0],b[0]), max(a[1],b[1]); x1,y1 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(x1-x0,0)*max(y1-y0,0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

dets = json.load(open(DET))["检测"]
print(f"[输入] {len(dets)} 个逐帧检测")

# ---------- 1. 跨帧关联: 同词的检测按三维位置贪心聚类 ----------
def cluster(items, fac):
    """贪心: 按分数从高到低, 每个检测要么并进已有簇, 要么自成一簇。
    半径取物体自身尺寸的一个比例 —— 一辆车和一把椅子该用的容差差着一个数量级。"""
    items = sorted(items, key=lambda d: -d["分"])
    cl = []
    for d in items:
        c = np.array(d["中心"]); r = fac*max(np.mean(d["尺寸"]), 0.3)
        hit = None
        for g in cl:
            if np.linalg.norm(c - g["中心"]) < max(r, fac*max(np.mean(g["尺寸"]),0.3)):
                hit = g; break
        if hit is None:
            cl.append(dict(中心=c, 尺寸=np.array(d["尺寸"]), 成员=[d]))
        else:
            hit["成员"].append(d)
            m = hit["成员"]
            hit["中心"] = np.median([x["中心"] for x in m], 0)
            hit["尺寸"] = np.median([x["尺寸"] for x in m], 0)
    return cl

inst = []
for word in sorted({d["词"] for d in dets}):
    for g in cluster([d for d in dets if d["词"]==word], DIST_F):
        frames = {x["帧"] for x in g["成员"]}
        if len(frames) < MIN_VIEWS: continue          # 单帧/两帧的当幻觉丢掉
        A = np.array([x["长轴"] for x in g["成员"]])
        sgn = np.sign(A @ A[0]); sgn[sgn==0]=1
        a1 = (A*sgn[:,None]).mean(0); a1 /= max(np.linalg.norm(a1),1e-9)
        if (sgn>0).sum()*2 < len(sgn): a1 = -a1
        best = max(g["成员"], key=lambda x: x["分"])
        inst.append(dict(词=word, 中心=g["中心"].tolist(), 尺寸=g["尺寸"].tolist(),
                         长轴=a1.tolist(), 见帧数=len(frames), 分=float(best["分"]),
                         最佳帧=best["帧"], 框2d=best["框2d"],
                         各帧框={str(x["帧"]): x["框2d"] for x in g["成员"]},
                         朝向分歧=round(float(min((sgn>0).mean(),(sgn<0).mean())),3)))
print(f"[聚类] -> {len(inst)} 个实例 (要求见于 >= {MIN_VIEWS} 帧)")

# ---------- 2. 跨词去重: 同一把椅子被 chair / folding chair 各检一遍 ----------
# 三维会漏: timesquare 的同一辆卡车因为各帧深度误差不同, 被拆成 van/truck/truck 三个实例。
# 二维做法: 两个实例只要在某个共同帧上二维框重得够多, 就是同一个东西。
inst.sort(key=lambda x: (-x["见帧数"], -x["分"]))
keep = []
for a in inst:
    if MATCH_2D:
        # 有共同帧就比二维; 没有共同帧就没法比二维, 退回三维(WD 比 WD, 偏差同向)。
        # 退路是必要的: carpark 那个旋转木马被拆成两个 bus, 帧范围一个 50~130 一个 85~95,
        # 恰好一帧都不重叠, 只靠二维永远发现不了它们是同一个。
        ca = np.array(a["中心"]); dup = False
        for b in keep:
            common = set(a["各帧框"]) & set(b["各帧框"])
            if common:
                if max(box_iou(a["各帧框"][f], b["各帧框"][f]) for f in common) > IOU_SELF:
                    dup = True; break
            else:
                r = SELF_3D_F*max(np.mean(a["尺寸"]), np.mean(b["尺寸"]), 0.3)
                if np.linalg.norm(ca - np.array(b["中心"])) < r:
                    dup = True; break
        if dup: continue
    else:
        ca = np.array(a["中心"]); ra = DIST_F*max(np.mean(a["尺寸"]),0.3)
        if any(np.linalg.norm(ca-np.array(b["中心"])) < max(ra, DIST_F*max(np.mean(b["尺寸"]),0.3))
               for b in keep): continue
    keep.append(a)
print(f"[跨实例去重] {len(inst)} -> {len(keep)}  ({'二维框 IoU' if MATCH_2D else '三维距离'})")

# ---------- 3. 和 SAM3 已有实例比对 ----------
# ★ 盒子只有 MATCH_2D=0 那条三维退路才用得上。二维那条读的是 03_masks 里的 mask,
# 跟盒子无关 —— 所以这里必须"用到才读", 否则整个召回会白白挂在 06_all_boxes.json 上,
# 而那个文件要等 gen -> ratio -> boxes 跑完才有, 召回就被迫排到生成之后去了。
sam = ([dict(b, 词=w) for w, bs in json.load(open(SAM)).items() for b in bs]
       if not MATCH_2D else [])
if not MATCH_2D: print(f"[SAM3] 已有 {len(sam)} 个盒子(三维退路)")

# 按帧索引 SAM3 的 mask —— 只解需要的帧, 别整个场景全解
_by = {}
def sam_masks(fi):
    if fi in _by: return _by[fi]
    from pycocotools import mask as mask_util
    out = []
    for fn in sorted(os.listdir(MK)):
        if not fn.endswith(".npz"): continue
        z = np.load(f"{MK}/{fn}", allow_pickle=True)
        meta, counts, (H,W) = z["meta"], z["counts"], z["hw"]
        for i, r in enumerate(meta):
            if r["area"] > 0 and int(r["frame"]) == fi:
                out.append((fn[:-4], int(r["oid"]),
                            mask_util.decode({"counts":counts[i],"size":[int(H),int(W)]}).astype(bool)))
    _by[fi] = out
    return out

# ★ 2026-09-22 新加一条判据: 和"已有实例的三维盒子投影到这一帧"也比一次。
# 起因(office 的急救箱): 它在第 39~54 帧还没有自己的编号 —— 那几帧它被圈在柱子的抠图里, 而柱子的
#   外接框是一整根柱子那么大, 框 vs 外接框只有 0.06, 于是被当成漏检补了一份; 到第 58 帧跟踪才给它
#   开编号, 一个急救箱成了两份(补出来的那份位置还偏了 1 m)。跟踪那份的盒子是拿它自己的 mask 量的,
#   是准的, 投影回第 39 帧正好落在检测框上。
# ★ 为什么不改成"被别的东西的 mask 包住就算已经有了": 那几帧急救箱确实还没有自己的编号, 它是被柱子
#   包着的 —— 那样改会把所有挂在墙上/柱子上的东西都当成已有的漏掉。
# 用 IoU 不用包含: 杯子落在桌子的投影框里这种框差太多, IoU 很小, 不会误判。
# 盒子或相机缺了就跳过这一条 —— 只少一道过滤, 不把召回硬绑死在量盒子那一步后面。
_CAMS = f"{R1}/02_da3/cameras.json"
_PBOX, _KJ, _IMWH, _DEP = [], {}, [None], {}
DEP_TOL = float(os.environ.get("RECALL_DEPTH_TOL", "0.3"))   # 检测框的距离和盒子的距离差多少还算同一个(米)
if MATCH_2D and os.path.exists(SAM) and os.path.exists(_CAMS):
    _PBOX = [dict(b, 词=w) for w, bs in json.load(open(SAM)).items() for b in bs]
    _KJ = json.load(open(_CAMS))["cameras"]
    print(f"[SAM3] 已有 {len(_PBOX)} 个盒子, 也投影到检测那一帧比一次(认出'那几帧还没被跟踪到'的同一个东西)")

def _img_wh():
    if _IMWH[0] is None:
        for fn in sorted(os.listdir(MK)):
            if fn.endswith(".npz"):
                H_, W_ = np.load(f"{MK}/{fn}", allow_pickle=True)["hw"]; _IMWH[0] = (int(W_), int(H_)); break
    return _IMWH[0]

# ★ 光比画面上的框不够: 投影不管远近, 同一条视线上远处的另一张桌子, 投影框照样能对上
#   (office 实测: 只比框会把 59 个补漏压掉 17 个, 里面有真漏掉的椅子和架子)。
#   所以还要拿那一帧的深度图量一下检测框里的东西有多远, 和盒子的距离对不上就不算同一个。
def det_depth(fi, box):
    """检测框里的中位深度(米); 没深度图或点太少返回 None。"""
    if fi in _DEP: dep = _DEP[fi]
    else:
        fp = f"{R1}/02_da3/depth/{fi:04d}_depth.npy"
        dep = _DEP[fi] = (np.load(fp) if os.path.exists(fp) else None)
    if dep is None: return None
    ih_, iw_ = _img_wh()[1], _img_wh()[0]
    h_, w_ = dep.shape
    x0_, y0_, x1_, y1_ = [int(round(v)) for v in (box[0]*w_/iw_, box[1]*h_/ih_, box[2]*w_/iw_, box[3]*h_/ih_)]
    x0_, y0_ = max(x0_, 0), max(y0_, 0); x1_, y1_ = min(x1_, w_), min(y1_, h_)
    if x1_ - x0_ < 2 or y1_ - y0_ < 2: return None
    d = dep[y0_:y1_, x0_:x1_].ravel(); d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if len(d) >= 12 else None

def proj_box(b, fi):
    """已有实例的三维盒子投影到第 fi 帧, 返回 (原图像素的外接框, 最近距离, 最远距离); 在相机后面返回 None。"""
    cam = _KJ.get(f"{fi:04d}.jpg")
    if cam is None: return None
    c = np.array(b["中心"], float); a1 = np.array(b["长轴"], float)
    a1 = a1 - UP*(a1@UP); n1 = float(np.linalg.norm(a1))
    if n1 < 1e-6: return None
    a1 /= n1; a2 = np.cross(UP, a1)
    L_, W_, H_ = b["尺寸"]; bot = float(b["底高"]); base = c - UP*(c@UP)
    P = np.array([base + sx*L_/2*a1 + sy*W_/2*a2 + UP*z
                  for sx in (-1, 1) for sy in (-1, 1) for z in (bot, bot+H_)])
    K = np.array(cam["intrinsic_3x3"], float); w2c = np.array(cam["extrinsic_w2c_4x4"], float).reshape(4, 4)
    Pc = P @ w2c[:3, :3].T + w2c[:3, 3]
    if (Pc[:, 2] < 0.05).any(): return None
    iw, ih = _img_wh()                       # 内参在深度图分辨率上, 框2d 在原图上, 按主点换算
    fx_, fy_ = iw/(2*K[0, 2]), ih/(2*K[1, 2])
    u = (Pc[:, 0]/Pc[:, 2]*K[0, 0] + K[0, 2]) * fx_
    v = (Pc[:, 1]/Pc[:, 2]*K[1, 1] + K[1, 2]) * fy_
    return [float(u.min()), float(v.min()), float(u.max()), float(v.max())], float(Pc[:, 2].min()), float(Pc[:, 2].max())

new = []; byproj = 0
for a in keep:
    if MATCH_2D:
        fi = int(a["最佳帧"]); x0,y0,x1,y1 = [float(v) for v in a["框2d"]]
        # 拿 mask 的外接框来比, 不拿 mask 本身。
        # 直接比框和 mask 会低估: 框里必然带着背景, 完全同一个物体也只有 0.2~0.3。
        # 实测 carpark: forklift 对 forklift#0, 框vs mask 只有 0.04, 框 vs 外接框是 0.33。
        best_iou, m = 0.0, None
        for w, oid, msk in sam_masks(fi):
            v, u = np.nonzero(msk)
            if len(u) == 0: continue
            s_ = box_iou([x0,y0,x1,y1], [float(u.min()), float(v.min()), float(u.max()), float(v.max())])
            if s_ > best_iou: best_iou, m = s_, dict(词=w, obj_id=oid)
        _bp = False
        if best_iou < IOU_SAM and _PBOX:             # 抠图比不出来, 再拿已有盒子的投影比(还要对得上距离)
            _dd = det_depth(fi, [x0,y0,x1,y1])
            for b in (_PBOX if _dd is not None else []):
                pb = proj_box(b, fi)
                if pb is None: continue
                bb, z0_, z1_ = pb
                if not (z0_ - DEP_TOL <= _dd <= z1_ + DEP_TOL): continue      # 一个在前一个在后, 不是同一个
                s_ = box_iou([x0,y0,x1,y1], bb)
                if s_ > best_iou: best_iou, m, _bp = s_, dict(词=b["词"], obj_id=b["obj_id"]), True
        a["二维IoU"] = round(best_iou, 3)
        if best_iou < IOU_SAM: m = None
        elif _bp: a["对上方式"] = "盒子投影"; byproj += 1
    else:
        ca = np.array(a["中心"]); r = MATCH_F*max(np.mean(a["尺寸"]),0.3)
        m = None
        for b in sam:
            cb = np.array(b["中心"])
            if np.linalg.norm(ca-cb) < max(r, MATCH_F*max(np.mean(b["尺寸"]),0.3)): m = b; break
    if m is None: new.append(a)
    else: a["对上SAM3"] = f"{m['词']}#{m['obj_id']}"
matched = len(keep)-len(new)
print(f"[合并] {matched} 个和 SAM3 对得上 (用 SAM3 的 mask), {len(new)} 个是 SAM3 漏掉的"
      f"  判定方式: {'二维 IoU > ' + str(IOU_SAM) if MATCH_2D else '三维距离'}"
      + (f"; 其中 {byproj} 个是靠已有盒子的投影认出来的(那几帧它还没被跟踪到)" if byproj else ""))
for i,a in enumerate(new): a["obj_id"] = 10000+i          # 召回实例的 ID 从 10000 起, 不和 SAM3 撞
print("  漏检按词:", dict(Counter(a["词"] for a in new).most_common(10)))
json.dump(dict(召回=new, 对上=matched, 总实例=len(keep)), open(OUT,"w"),
          ensure_ascii=False, indent=1)
print(f"\n[done] -> {OUT}")
