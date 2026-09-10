#!/usr/bin/env python
"""挑完生成样本后, 让 VLM 确认这块 mask 到底是不是那个词。不是就换下一个候选。
起因: table 挑中的 5 号实例圈的是一辆轿车+卡车 —— 打分只看面积/贴边/活多久,
纯几何指标, 从不校验内容。测量交给几何, 判断交给 VLM。"""
import os, re, json, numpy as np, cv2, torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imgio import imwrite_stable
from PIL import Image
from pycocotools import mask as mask_util
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
TAG=os.environ.get("TAG","ts_r1")

R1  = os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
FR  = os.environ.get("FR","/home/jincheng/sam3d/data/outdoor/dataset/timesquare/frames")
TOPK= int(os.environ.get("TOPK","8"))          # 每类最多试几个候选
MIN_AREA_PCT=float(os.environ.get("MIN_AREA_PCT","0.05"))
# 场景描述进提示词。室内素材要改成 "photo of an indoor bar" 之类,
# 否则 VLM 会拿"这是街景吗"当额外判据, 把吧凳判成 no。
SCENE_DESC=os.environ.get("SCENE_DESC","street photo")
HAVE={int(f[:4]) for f in os.listdir(f"{R1}/02_da3/depth") if f.endswith("_depth.npy")}

def crop_on_gray(img, m, pad=0.12):
    ys,xs=np.nonzero(m); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    p=int(pad*max(y1-y0,x1-x0))+6
    H,W=m.shape; y0,y1,x0,x1=max(0,y0-p),min(H,y1+p),max(0,x0-p),min(W,x1+p)
    cut=np.full_like(img[y0:y1,x0:x1],200); mm=m[y0:y1,x0:x1]
    cut[mm]=img[y0:y1,x0:x1][mm]
    return cut

MODEL="Qwen/Qwen3-VL-8B-Instruct"
model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda")
proc=AutoProcessor.from_pretrained(MODEL)
def ask(pil, word):
    # 问"是不是完整的 xxx"太严: 桌子的 mask 常常只抠到桌面, 一片悬空的橙色椭圆
    # 会被判成 no —— 可实际上 12 个候选里 11 个是真桌面。SAM3D 本来就会补看不见的部分,
    # 所以只要"是这个东西或它的一部分"就够了, 真正要排掉的是抠成别的东西 / 抠了一堆东西。
    q=(f'This is a segmentation cut-out from a {SCENE_DESC}, shown on a gray background. '
       f'It may be only part of an object (e.g. just a table top without the legs), '
       f'and it may be blurry or seen from an odd angle.\n'
       f'Could this plausibly be a "{word}", or a visible part of a "{word}"?\n'
       f'Answer no ONLY if it is clearly a different kind of object, '
       f'or if it is several unrelated objects merged together.\n'
       f'Answer with ONLY one word: yes or no.')
    msgs=[{"role":"user","content":[{"type":"image","image":pil},{"type":"text","text":q}]}]
    return _gen(msgs, 6)

def _gen(msgs, ntok):
    inp=proc.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,
                                 return_dict=True,return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out=model.generate(**inp,max_new_tokens=ntok,do_sample=False)
    return proc.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip().lower()

def grid_of(cuts, cell=320):
    """把若干抠图拼成带编号的网格, 每格左上角写一个大号数字。"""
    n=len(cuts); cols=min(4,n); rows=(n+cols-1)//cols
    canvas=np.full((rows*cell, cols*cell, 3), 200, np.uint8)
    for k,c in enumerate(cuts):
        h,w=c.shape[:2]; sc=min((cell-24)/w,(cell-24)/h)
        r=cv2.resize(c,(max(int(w*sc),1),max(int(h*sc),1)))
        y=(k//cols)*cell+(cell-r.shape[0])//2; x=(k%cols)*cell+(cell-r.shape[1])//2
        canvas[y:y+r.shape[0], x:x+r.shape[1]]=r
        p=((k//cols)*cell+38,(k%cols)*cell+14)
        cv2.putText(canvas,str(k+1),(p[1],p[0]),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),6,cv2.LINE_AA)
        cv2.putText(canvas,str(k+1),(p[1],p[0]),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,90,255),2,cv2.LINE_AA)
    return canvas

def choose(pil, word, n):
    """选择题。★ 为什么不用逐个问是非 ——
    原来是拿候选 1 问 "这可能是 bar stool 吗", 答 yes 就定了。那道题故意问得很宽
    ("是它的一部分也算"), 所以一把高背餐椅问 "这可能是吧凳吗" 必然答 yes, 甚至
    问 "这可能是灯吗" 也答了 yes。它没得比, 只能顺着说。
    改成把这个词的全部候选并排给它、让它挑一个, 它就必须在吧凳和餐椅之间做区分。"""
    q=(f'These are {n} segmentation cut-outs from a {SCENE_DESC}, shown on a gray background, '
       f'numbered 1 to {n}. Some may be blurry, partial, or a different object entirely.\n'
       f'Which ONE is most clearly and completely a "{word}"?\n'
       f'Prefer a complete, unoccluded, correctly-identified example. '
       f'If none of them is a "{word}", answer 0.\n'
       f'Answer with ONLY the number.')
    msgs=[{"role":"user","content":[{"type":"image","image":pil},{"type":"text","text":q}]}]
    a=_gen(msgs, 8)
    m=re.search(r"\d+", a)
    return (int(m.group()) if m else -1), a

# ★ 只校验挑图那步真的挑中的词。
# 以前这里自己 os.listdir(03_masks) 从头枚举一遍, 谁都不认: 排除的词、跨词去重判掉的词、
# 交给逐实例生成的词, 统统会被它挑出来、写进 my_images/、然后被 gen 拿去烧一次 SAM3D。
# 更麻烦的是它最后整个覆盖 picks.json —— gen 那步是照 picks.json 过滤磁盘目录的,
# 一覆盖就等于把上游所有过滤全部作废, 而且连一句警告都不打。
# beach 实测: 该写 0 个类目录, 它会写 23 个, 白烧 20 多次 --hq(半小时 GPU)。
BASE = f"{R1}/04_picks/picks.json"
picked = json.load(open(BASE)) if os.path.exists(BASE) else {}
_todo = set(picked)
if not _todo:
    print("[注意] picks.json 是空的(这一轮没有词走整类共用), 没有要校验的东西")
report={}; _ok=set()          # ★ 必须单独记"这一轮通过的", 不能靠 word in picked ——
                              # picked 是从文件读进来的, 每个词一开始就在里面, 那个判断恒为真,
                              # 结果是被否决的词照样留在 picks.json 里, 拿着 VLM 判错的那张
                              # mask 去生成。改动前 picked 从空开始是 fail-closed, 别写回去。
for fn in sorted(os.listdir(f"{R1}/03_masks")):
    z=np.load(f"{R1}/03_masks/{fn}",allow_pickle=True)
    meta=z["meta"]; counts=z["counts"]; H,W=z["hw"]; word=str(z["word"])
    if word not in _todo: continue
    if len(meta)==0 or meta["area"].max()==0: continue
    live={}
    for i,(oid,fi,a,e) in enumerate(meta):
        if a>0 and int(fi) in HAVE: live.setdefault(int(oid),[]).append((i,int(fi),int(a),float(e)))
    cands=[]
    for oid,rows in live.items():
        if len(rows)<2: continue
        nf=min(len(rows)/10,1.0)
        i,fi,a,e=max(rows,key=lambda t:t[2]*(1-min(t[3]*4,0.9)))
        cands.append((a*(1-min(e*4,0.9))*nf, oid, i, fi, a, e, len(rows)))
    cands.sort(reverse=True)
    tried=[]
    # 先把候选都抠出来, 一次性给 VLM 做选择题
    pool=[]
    for sc_,oid,i,fi,a,e,nfr in cands[:TOPK]:
        if 100*a/(H*W) < MIN_AREA_PCT: break
        m=mask_util.decode({"counts":counts[i],"size":[int(H),int(W)]}).astype(bool)
        img=cv2.imread(f"{FR}/{fi:04d}.jpg")
        pool.append((oid,i,fi,a,e,nfr,m,img,crop_on_gray(img,m)))
    order=list(range(len(pool)))
    if len(pool) >= 2:
        g=grid_of([p[8] for p in pool])
        cv2.imwrite(f"{R1}/04_picks/{word.replace(' ','_')}_choices.jpg",g,[cv2.IMWRITE_JPEG_QUALITY,88])
        k,raw=choose(Image.fromarray(cv2.cvtColor(g,cv2.COLOR_BGR2RGB)), word, len(pool))
        print(f"[{word}] 选择题 {len(pool)} 个候选 -> VLM 选 {raw}", flush=True)
        tried.append(dict(选择题=f"{len(pool)}选1", VLM=raw))
        if 1 <= k <= len(pool):
            order=[k-1]+[j for j in order if j != k-1]   # 它选中的排第一, 其余按原顺序兜底
    # 再逐个过是非题 (它选中的第一个就过, 相当于二次确认); 全否决就放弃
    for j in order:
        oid,i,fi,a,e,nfr,m,img,cut=pool[j]
        ans=ask(Image.fromarray(cv2.cvtColor(cut,cv2.COLOR_BGR2RGB)), word)
        ok=ans.startswith("yes")
        tried.append(dict(obj_id=oid,frame=fi,占比=round(100*a/(H*W),2),VLM=ans))
        print(f"[{word}] 候选#{oid} 第{fi}帧 占{100*a/(H*W):.2f}% -> VLM: {ans}", flush=True)
        if ok:
            name=word.replace(" ","_")
            d=os.path.expanduser(f"~/sam3d/sam-3d-objects/my_images/{TAG}_{name}"); os.makedirs(d,exist_ok=True)
            imwrite_stable(f"{d}/frame.png",img); imwrite_stable(f"{d}/mask.png",m.astype(np.uint8)*255)
            ov=img.copy(); ov[m]=(0.5*ov[m]+0.5*np.array([60,60,230])).astype(np.uint8)
            cv2.imwrite(f"{R1}/04_picks/{name}.jpg",ov,[cv2.IMWRITE_JPEG_QUALITY,88])
            cv2.imwrite(f"{R1}/04_picks/{name}_cut.jpg",cut,[cv2.IMWRITE_JPEG_QUALITY,90])
            picked[word]=dict(obj_id=oid,frame=fi,面积占比=round(100*a/(H*W),2),活帧数=nfr,
                              贴边=round(e,3),候选数=len(cands),试过=len(tried))
            _ok.add(word)
            break
    report[word]=tried
    if word not in _ok: print(f"[{word}] 试了 {len(tried)} 个候选都被 VLM 否决, 放弃", flush=True)
# 只更新校验过的那些词, 不整个覆盖 —— 覆盖会把上游的排除/去重/逐实例全部作废
json.dump(picked,open(BASE,"w"),ensure_ascii=False,indent=1)
json.dump(report,open(f"{R1}/04_picks/vlm_verify.json","w"),ensure_ascii=False,indent=1)
_rej = sorted(_todo - _ok)
for w in _rej: picked.pop(w, None)          # 否决的真的踢出去, 否则 gen 照样生成
if _rej: print(f"[否决] {len(_rej)} 个词被 VLM 否决, 已从 picks.json 移除: {', '.join(_rej)}")
print(f"\n[done] 校验 {len(report)} 个词, {len(picked)} 个通过 (只校验挑图挑中的, 不再自己重新枚举)")


# ==================================================================== 逐实例校验
# ★ 为什么要有这一段: 上面那一段只验"整类共用"的词 —— 一个词一个代表, 验代表那一张。
#   而 solo 的词根本不进 picks.json(pick_from_masks.py:92 主动跳过, 交给
#   gen_per_instance.py), 于是**图变多了 N 倍, 验证却掉到零**:
#       共用  1 张图 -> 验 1 张
#       solo  N 张图 -> 验 0 张
#   越该验的时候越没人验。pedestrian/r2 实测: person 15 个实例里 3 个根本不是人
#   (一个黑背包、一堆碎片、半截身子), 全程没有任何机器环节看过它们一眼。
#
#   枚举方式和 review_gate.py:128-150 保持一致(同一个打分公式、同一个 SOLO_TH 门槛),
#   这样人在闸口看到的那批, 就是这里验过的那批 —— 否则又是"人在看 A 判 B"。
SOLO_CFG = os.environ.get("SOLO", "")
if SOLO_CFG:
    KEEP  = {w.strip() for w in os.environ.get("SOLO_KEEP", "").split(",") if w.strip()}
    EXCL  = {w.strip() for w in os.environ.get("EXCLUDE", "").split(",") if w.strip()}
    SOLO_TH = float(os.environ.get("SOLO_TH", "0.5"))
    _dd = f"{R1}/03_dedup_words.json"
    _dead = {}
    if os.path.exists(_dd):
        _dead = {w: {int(x) for x in v}
                 for w, v in (json.load(open(_dd)).get("淘汰实例") or {}).items()}
    allw = [f[:-4].replace("_", " ") for f in sorted(os.listdir(f"{R1}/03_masks"))
            if f.endswith(".npz")]
    solo_w = (set(allw) if "*" in SOLO_CFG
              else {w.strip() for w in SOLO_CFG.split(",") if w.strip()})
    solo_w = (solo_w | KEEP) - EXCL
    inst_report, inst_rej = {}, {}
    n_all = n_bad = 0
    for w in sorted(solo_w):
        fp = f"{R1}/03_masks/{w.replace(' ', '_')}.npz"
        if not os.path.exists(fp):
            continue
        z = np.load(fp, allow_pickle=True)
        meta, counts, (H, W) = z["meta"], z["counts"], z["hw"]
        per = {}
        for i, r in enumerate(meta):
            if r["area"] <= 0:
                continue
            sc = int(r["area"]) * (1 - min(float(r["edge"]) * 4, 0.9))   # 同 review_gate:131
            o = int(r["oid"])
            if sc > per.get(o, (0,))[0]:
                per[o] = (sc, int(r["area"]), int(r["frame"]), i)
        for o, (sc, a, fi, i) in sorted(per.items()):
            if o in _dead.get(w, ()):            # 跨词去重判掉的, 下游本来就不出盒子
                continue
            pct = 100 * a / (int(H) * int(W))
            if pct < SOLO_TH:                    # 太小的不逐实例生成, 不必验
                continue
            m = mask_util.decode({"counts": counts[i], "size": [int(H), int(W)]}).astype(bool)
            img = cv2.imread(f"{FR}/{fi:04d}.jpg")
            if img is None:
                continue
            ans = ask(Image.fromarray(cv2.cvtColor(crop_on_gray(img, m), cv2.COLOR_BGR2RGB)), w)
            n_all += 1
            ok = ans.startswith("yes")
            inst_report.setdefault(w, []).append(dict(obj_id=o, frame=fi,
                                                      占比=round(pct, 2), VLM=ans))
            if not ok:
                n_bad += 1
                inst_rej.setdefault(w, []).append(o)
                print(f"[否决] {w}#{o} 第{fi}帧 占{pct:.2f}% -> VLM: {ans}", flush=True)
    json.dump(dict(否决=inst_rej, 报告=inst_report),
              open(f"{R1}/04_picks/vlm_verify_inst.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n[逐实例] 验了 {n_all} 个实例, 否决 {n_bad} 个"
          + (f": {json.dumps(inst_rej, ensure_ascii=False)}" if inst_rej else "")
          + f"\n           -> {R1}/04_picks/vlm_verify_inst.json")
