#!/usr/bin/env python
"""每个实例跨帧反投影合并点云, 拟合带朝向的盒子. 只从 03_masks/*.npz 读, 不碰 GPU.

和首轮的区别: 门槛从"活够 8 帧"改成"合并后点数够不够"。
帧数不代表点够: 近处的车 2 帧就有上万点, 远处的车 20 帧也只有几百点。
  MIN_POINTS    = 门槛, 合并后点数少于它就丢
  SAMPLE_FRAMES = 取样, 用面积最大的这么多帧 (不是门槛, 不存在过不过)
"""
import sys, os, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_frame import load_up, export_T, inverse_T, usable_frames
from pycocotools import mask as mask_util

ROOT = os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
MK   = f"{ROOT}/03_masks"; DA3=f"{ROOT}/02_da3"
MIN_POINTS    = int(os.environ.get("MIN_POINTS","300"))
VOX           = float(os.environ.get("VOX","0.15"))     # 体素边长(米)
DUP_TH        = float(os.environ.get("DUP_TH","0.55"))  # 双向包含都超过它 = 同一个物体
PART_TH       = float(os.environ.get("PART_TH","0.70")) # 单向包含超过它 = 部件, 归给父体
# 词优先级(同级再比点数)搬到 word_rank.py 了 —— 二维去重那步也要用同一张表, 复制两份
# 迟早分叉, 分叉就意味着同一批物体在二维叫 car、在三维叫 taxi。
# ★ 两边共用同一份表: 只 import rank 的话, 三维这步的选词还是平表, 跨类会互相比
#   (二维那边已经用 comparable 挡住了 table vs car 这种)。
from word_rank import rank, comparable
SAMPLE_FRAMES = int(os.environ.get("SAMPLE_FRAMES","8"))
# 深度置信度下限。2.0 是时代广场调的, 那里 78% 的像素过得了。但这个数是随画面内容变的,
# 不是随相机运动变的: 沙滩场景大片沙子和海面纹理均一, DA3 本来就没信心, 只有 9% 的像素
# 过得了 2.0, 照搬会把整个场景滤空; 停车场有路面标线和车身纹理, 反而有 92.6%。
CONF_MIN = float(os.environ.get("CONF_MIN","2.0"))
cams = json.load(open(f"{DA3}/cameras.json"))["cameras"]
# 位姿错的帧不许参与(像素会落在错的位置, 盒子被拉长), 见 world_frame.usable_frames
HAVE = usable_frames(DA3, "量盒子")
# "天"的方向。老的写法是写死 [0,-1,0] = 首帧相机的 -y —— 街头平视时相机水平, 这个假设
# 成立(时代广场实测只偏 0.4°); 但高空俯拍时相机低着头, 它的 -y 指向斜上后方, beach 偏
# 26.2°、carpark 偏 29.2°, 整个场景连同地面一起歪掉。
# 现在改成读 world_up.py 从点云主平面拟合出来的真实法向, 没有那个文件才退回老假设。
# ★ 必须在"量盒子"之前定好: 盒子的高度是沿"天"量出来的, 方向错了盒子本身就是错的,
#   事后旋转只是把错的盒子摆正, 数字不会变对。
UP, _GH, _FB = load_up(DA3, "量盒子")

def backproj(mask, fi, conf_min=None):
    if conf_min is None: conf_min = CONF_MIN
    fn=f"{fi:04d}.jpg"
    dep=np.load(f"{DA3}/depth/{fi:04d}_depth.npy"); cnf=np.load(f"{DA3}/depth/{fi:04d}_conf.npy")
    h,w=dep.shape
    r=int(np.clip(np.sqrt(mask.sum())/25,1,6))                 # 腐蚀半径随物体大小自适应
    m=cv2.erode(mask.astype(np.uint8),np.ones((r*2+1,)*2,np.uint8))
    m=cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
    m&=(cnf>=conf_min)&np.isfinite(dep)&(dep>0)
    if m.sum()<12: return None
    d=dep[m]; med=np.median(d); mad=np.median(np.abs(d-med))+1e-6
    k=np.abs(d-med)<4*mad                                      # 剔穿透到背景的像素
    v,u=np.nonzero(m); v,u,d=v[k],u[k],d[k]
    K=np.array(cams[fn]["intrinsic_3x3"])
    P=np.stack([(u-K[0,2])*d/K[0,0],(v-K[1,2])*d/K[1,1],d],1)
    c2w=np.linalg.inv(np.array(cams[fn]["extrinsic_w2c_4x4"]).reshape(4,4))
    return (c2w[:3,:3]@P.T).T+c2w[:3,3]

def fit_box(P):
    hgt=P@UP; Q=P-np.outer(hgt,UP); R=Q-Q.mean(0)
    ev,V=np.linalg.eigh(R.T@R/len(R)); ax=V[:,np.argsort(ev)[::-1]]
    a1=ax[:,0]-UP*(ax[:,0]@UP); a1/=np.linalg.norm(a1); a2=np.cross(UP,a1)
    pr=np.stack([P@a1,P@a2,hgt],1); lo,hi=pr.min(0),pr.max(0)
    ctr=a1*(lo[0]+hi[0])/2 + a2*(lo[1]+hi[1])/2 + UP*(lo[2]+hi[2])/2
    return dict(中心=ctr.tolist(), 长轴=a1.tolist(), 尺寸=(hi-lo).tolist(), 底高=float(lo[2]))

# 前置的二维跨词去重(dedup_words.py, 排在 track 之后)判掉的同义词, 这里也得跳过。
# 不是为了省时间, 是为了不出错: 那些词在挑图那步就没写盘, 从来没生成过 mesh。
# 让它们的盒子留在这里, 下面的三维去重有可能把某个没有 mesh 的词选成组代表 ——
# 形状打分要读 05_mesh_ratio.json, 没生成过的词读不到, 于是退回优先级表, 而优先级表里
# 大部分词同级, 最后按点数选, 完全可能选中一个没有 mesh 的词。组装那步 pick_mesh 拿不到
# mesh 就直接跳过, 整组物体一声不响地从场景里消失。
# ★ 名单的粒度是"实例"不是"词": 一个词只有部分实例被同组的另一个词代表(taxi 的 3 个实例
#   并进了 car, 剩下 17 辆 car 还得量), 整词跳过会连着扔掉那些没人代表的物体。
#   淘汰词只在"这个词一个实例都不剩"时才有, 留着做整词短路 + 日志。
DEDUP=os.environ.get("DEDUP", f"{ROOT}/03_dedup_words.json")
_dd=json.load(open(DEDUP)) if DEDUP and os.path.exists(DEDUP) else {}
DUPOUT=set(_dd.get("淘汰词",[]))
DUPINST={w:set(v) for w,v in _dd.get("淘汰实例",{}).items()}
if DUPINST: print(f"[去重] 二维那步判掉 {sum(len(v) for v in DUPINST.values())} 个实例不量盒子"
                  + (f", 其中整个没剩的词: {', '.join(sorted(DUPOUT))}" if DUPOUT else ""))

all_boxes={}; drop_pts=0; drop_none=0
for fn in sorted(os.listdir(MK)):
    z=np.load(f"{MK}/{fn}",allow_pickle=True)
    meta=z["meta"]; counts=z["counts"]; H,W=z["hw"]; word=str(z["word"])
    if word in DUPOUT: continue
    if len(meta)==0 or meta["area"].max()==0: continue
    live={}; _dead=DUPINST.get(word, ())
    for i,(oid,fi,a,e) in enumerate(meta):
        if int(oid) in _dead: continue       # 这个实例由同组另一个词代表, 它的盒子在那边量
        if a>0 and int(fi) in HAVE: live.setdefault(int(oid),[]).append((int(a),int(fi),i))
    boxes=[]
    for oid,rows in live.items():
        rows.sort(key=lambda t:-t[0])
        pts=[]
        for a,fi,i in rows[:SAMPLE_FRAMES]:
            m=mask_util.decode({"counts":counts[i],"size":[int(H),int(W)]}).astype(bool)
            P=backproj(m,fi)
            if P is not None: pts.append(P)
        if not pts: drop_none+=1; continue
        P=np.concatenate(pts)
        med=np.median(P,0); dd=np.linalg.norm(P-med,axis=1)
        P=P[dd<np.percentile(dd,95)]                          # 剔离群
        if len(P)<MIN_POINTS: drop_pts+=1; continue           # ← 门槛: 按点数, 不按帧数
        b=fit_box(P); b["obj_id"]=oid; b["点数"]=len(P); b["用帧"]=len(pts); b["总帧"]=len(rows)
        b["_vox"]=set(map(tuple, np.round(P/VOX).astype(np.int32)))   # 体素占用, 判重复用
        boxes.append(b)
    all_boxes[word]=boxes
    print(f"[{word}] {len(boxes)}/{len(live)} 个实例过点数门槛", flush=True)
# ---------- 跨词去重 ----------
# ★ 2026-09-08 起这里是兜底, 不再是主力。主力挪到了 track 之后的 dedup_words.py,
#   在同一帧的像素上比 mask —— 那一步能赶在生成之前判掉同义词, 省掉每个词 1-2 分钟的
#   SAM3D 生成; 这里排在 gen 之后, 判出来的重复 mesh 已经烧过 GPU 了。
#   但二维那步有个看不见的死角: 两个词从来没在同一帧同时出现过, 就没有可比的像素。
#   beach 上 planter 只活在第 312 帧、potted plant 只活在第 48~98 帧, 一帧都不重叠,
#   二维永远发现不了它们是同一个花盆 —— 这种只能靠下面的三维体素捞。
#   所以这段留着, 而且判据不变: 三维和二维是互补的两个视角, 不是一个替换另一个。
# 同一个物理物体被多个词各抠一遍 (planter / plant pot / potted plant 中心距 0.01m)。
# 判据用三维体素占用而不是中心距: 中心距分不清"重复"和"部件"。
#   双向包含都高  = 同一个东西  -> 合成一组, 留点数最多(抠得最全)的那个
#   单向包含高    = 部件(树冠 vs 整个盆栽) -> 部件并进父体, 不单独出现
flat=[(w,i,b) for w,bs in all_boxes.items() for i,b in enumerate(bs)]
N=len(flat)
par=list(range(N))
def find(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
def union(a,b):
    a,b=find(a),find(b)
    if a!=b: par[b]=a

merged_dup=0; merged_part=0
for i in range(N):
    Vi=flat[i][2]["_vox"]
    for j in range(i+1,N):
        Vj=flat[j][2]["_vox"]
        inter=len(Vi&Vj)
        if not inter: continue
        ci, cj = inter/len(Vi), inter/len(Vj)
        vi=np.prod(flat[i][2]["尺寸"]); vj=np.prod(flat[j][2]["尺寸"])
        big,small=max(vi,vj),min(vi,vj)
        if ci>DUP_TH and cj>DUP_TH and big < 3*small:
            union(i,j); merged_dup+=1                      # 重复: 体积也得可比
        elif max(ci,cj)>PART_TH and small < 0.7*big:
            union(i,j); merged_part+=1                     # 部件: 子必须明显比父小

groups={}
for i in range(N): groups.setdefault(find(i),[]).append(i)
# 组里选哪个词: 先比形状, 形状分不出再退回优先级表。
# 起因: 一个红花盆被 plant pot / planter / potted plant 三个词同时抠到, 优先级表把
# potted plant 排第一, 于是这个"光盆"套上了带黄杨球的细高模型, 成了前景一团大绿球。
# 三个词不是粗细关系而是三种形态, 优先级本来就无从排起 —— 但盒子的长宽高比例是量出来的:
# 花盆盒子矮胖对上 plant pot 的 mesh(0.95x0.98x1.0), 黄杨球细高对上 potted plant(0.5x1.0x0.49)。
# 局限: car/taxi 的 mesh 都是轿车、比例几乎一样, 形状分不出来, 那种还得靠优先级。
# mesh 比例由 05_mesh_ratio.json 提供 (sam3 环境没有 trimesh, 预先在 sam3d 环境算好)
_MR={k:np.array(v) for k,v in json.load(open(f"{ROOT}/05_mesh_ratio.json")).items()}
def mesh_ratio(w): return _MR.get(w)
def shape_cost(w, ext):
    r=mesh_ratio(w)
    if r is None: return None
    e=np.sort(np.array(ext))[::-1]; e=e/max(e[0],1e-9)
    return float(np.abs(np.log((e+1e-3)/(r+1e-3))).sum())

kept={}; by_shape=0; by_rank=0
for g in groups.values():
    ext=flat[max(g,key=lambda i: flat[i][2]["点数"])][2]["尺寸"]   # 点最多那个的盒子当形状基准
    scored=sorted([(c,i) for c,i in ((shape_cost(flat[i][0],ext),i) for i in g) if c is not None])
    if len(scored)>=2 and scored[1][0]-scored[0][0] > 0.15:
        best=scored[0][1]; by_shape+=1                 # 形状分得开
    else:
        # ★ 跨类不许比。RANK 是分块的(车/桌椅/植物/招牌/人), "细词赢粗词"只在同类内有意义。
        # 平表比会出事: beach 的 billboard 是 rank 1 且在 EXCLUDE 里, 它只要在三维上和一棵
        # 真树撞一组, 整组就被登记成 billboard -> 被排除 -> 那棵树从场景里消失。
        # 而 beach 全部走逐实例, 05_mesh_ratio.json 是空的, shape_cost 全 None ——
        # 100% 的组都走这条分支, 所以这不是学术问题。
        _anchor = max(g, key=lambda i: flat[i][2]["点数"])          # 点最多的当锚
        _ok = [i for i in g if comparable(flat[i][0], flat[_anchor][0])] or [_anchor]
        best=min(_ok,key=lambda i:(rank(flat[i][0]), -flat[i][2]["点数"])); by_rank+=1
    w,_,b=flat[best]
    b=dict(b); b.pop("_vox")
    b["同组候选"]=[f"{flat[i][0]}#{flat[i][2]['obj_id']}" for i in g]
    kept.setdefault(w,[]).append(b)

json.dump(kept, open(f"{ROOT}/06_all_boxes.json","w"), ensure_ascii=False, indent=1)
n0=N; n=sum(len(v) for v in kept.values())
print(f"[选词] 形状判定 {by_shape} 组, 形状分不开退回优先级 {by_rank} 组")
print(f"\n[去重] {n0} -> {n} 个实例 (重复合并 {merged_dup} 对, 部件并入父体 {merged_part} 对)")
for w in sorted(all_boxes, key=lambda w:-len(all_boxes[w])):
    a,bq=len(all_boxes[w]), len(kept.get(w,[]))
    if a: print(f"  {w:16s} {a:3d} -> {bq:3d}")
print(f"\n[done] 点数不足丢 {drop_pts}, 反投影失败丢 {drop_none}")
