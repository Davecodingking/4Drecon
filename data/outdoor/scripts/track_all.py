#!/usr/bin/env python
"""SAM3 视频跟踪, 把结果按词落盘, 后面所有步骤都从盘上读, 不再碰 GPU.

多词模式(默认开, MULTIWORD=0 关): 每 MW_GROUP 个词一批, 一批词的检测只算一遍
(画面编码 + 检测器跟词无关), 跟踪仍然一个词一个会话 —— 结果应和逐词跑一致。
为什么不把几个词塞进同一个会话、实现细节, 见 sam3_multiword.py。
  - 一批检测 OOM 就把没跑完的词对半拆, 拆到一个词就是逐词
  - 分段跑(三级)时关掉共用, 段内帧号和缓存对不上

防 OOM 三级 (按代价从低到高, 逐级升级):
  一级 offload_video_to_cpu / offload_state_to_cpu  —— 一个参数, 最省显存
  二级 密集词临时提高 new_det_thresh                —— 单帧就有上百个实例的词只能靠它
  三级 分段跑 + 按 mask IoU 接续实例 ID              —— 前两级不够才上
"""
import os, sys, json, time, gc, traceback
import numpy as np, cv2, torch
from pycocotools import mask as mask_util
from sam3.model_builder import build_sam3_video_predictor
import sam3_multiword as mw

ROOT = os.environ.get("R1", "/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
FR   = os.environ.get("FR", "/home/jincheng/sam3d/data/outdoor/dataset/timesquare/frames")
CKPT = os.path.expanduser("~/sam3d/sam3/checkpoints/sam3.pt")
OUT  = f"{ROOT}/03_masks"; os.makedirs(OUT, exist_ok=True)
SEG  = int(os.environ.get("SEG", "60"))          # 三级: 每段多少帧
MULTI    = os.environ.get("MULTIWORD", "1") == "1"
MW_GROUP = int(os.environ.get("MW_GROUP", "8"))  # 多词模式一批几个词(检测爆显存会自动拆小)
VOCAB= json.load(open(f"{ROOT}/01_vlm_vocab.json"))["vocab"]
# 附加词, 逗号分隔。给"建筑模式"用: 列词的 prompt 明确让 VLM 跳过建筑立面, 所以词表里
# 一个建筑词都没有, 但建筑要单独出一个 glb 就必须跟踪它们。
# ★ 走这个口子而不是往 01_vlm_vocab.json 里塞词 —— 词表纯 VLM 是刻意保持的实验条件,
#   混进手写的词就说不清"VLM 自己能不能列全"了。附加词单独记在 03_track_report.json 里。
EXTRA = [w.strip() for w in os.environ.get("EXTRA_WORDS","").split(",") if w.strip()]
if EXTRA:
    print(f"[附加词] {len(EXTRA)} 个(不在 VLM 词表里): {', '.join(EXTRA)}", flush=True)
    VOCAB = VOCAB + [w for w in EXTRA if w not in VOCAB]
# ★ 2026-09-29 拆分: 同一轮里这个脚本可能同时起几份, 每份只跟踪自己那几个词(各用一张卡)。
#   报告各写各的, 由流水线并成一份 —— 几份都往 03_track_report.json 写的话后写的盖掉先写的。
ONLY = [w for w in os.environ.get("ONLY_WORDS", "").split(",") if w]
if ONLY:
    VOCAB = [w for w in VOCAB if w in ONLY]
    print(f"[拆分] 这一份只跟踪 {len(VOCAB)} 个词: {', '.join(VOCAB)}", flush=True)
REPORT = os.environ.get("TRACK_REPORT", f"{ROOT}/03_track_report.json")
# 二级: 单帧就密密麻麻的词, 只要最大的那几个
DENSE = {"sign":0.90, "billboard":0.90, "window":0.90,
         "person":0.85, "pedestrian":0.85, "chair":0.85, "folding chair":0.85}

FILES = sorted(f for f in os.listdir(FR) if f.endswith(".jpg"))
NF    = len(FILES)
H, W  = cv2.imread(f"{FR}/{FILES[0]}").shape[:2]

def edge_ratio(m):
    return float(max(m[0,:].mean(), m[-1,:].mean(), m[:,0].mean(), m[:,-1].mean()))

def track(pred, frames_dir, thresh=None, word=None):
    """跑一遍跟踪, 返回 {obj_id: {frame_idx: mask_bool}}"""
    sid = pred.handle_request(request=dict(
        type="start_session", resource_path=frames_dir,
        offload_video_to_cpu=True, offload_state_to_cpu=True))["session_id"]   # ← 一级
    old = None
    if thresh is not None:                                                      # ← 二级
        old = pred.model.new_det_thresh; pred.model.new_det_thresh = thresh
    try:
        pred.handle_request(request=dict(type="add_prompt", session_id=sid, frame_index=0, text=word))
        res = {}
        for r in pred.handle_stream_request(request=dict(type="propagate_in_video", session_id=sid)):
            fi, o = r["frame_index"], r["outputs"]
            for k, oid in enumerate(o["out_obj_ids"].tolist()):
                m = o["out_binary_masks"][k].astype(bool)
                if m.any(): res.setdefault(int(oid), {})[int(fi)] = m.copy()
        return res
    finally:
        if old is not None: pred.model.new_det_thresh = old
        pred.handle_request(request=dict(type="close_session", session_id=sid))

def make_seg_dirs():
    """三级: 建软链接分段目录, 段间重叠 1 帧"""
    dirs=[]
    for s in range(0, NF, SEG-1):
        d=f"{ROOT}/_seg{s}"; os.makedirs(d, exist_ok=True)
        idx=list(range(s, min(s+SEG, NF)))
        if len(idx)<2: break
        for j,i in enumerate(idx):
            lk=f"{d}/{j:04d}.jpg"
            try:
                os.symlink(f"{FR}/{FILES[i]}", lk)
            except FileExistsError:      # 同一轮里几份同时跑到三级, 谁先建都一样(链的是同一张图)
                pass
        dirs.append((d, idx))
        if idx[-1]==NF-1: break
    return dirs

def track_segmented(pred, thresh, word):
    """分段跑, 段间共享帧上按 IoU 把实例 ID 接起来"""
    merged={}; nxt=0; prev_tail={}          # prev_tail: {全局oid: 上一段末帧的 mask}
    for d, idx in make_seg_dirs():
        part = track(pred, d, thresh, word)
        # 本段第一帧(=上一段最后一帧)上做匹配
        cur_head = {oid: fr[0] for oid, fr in
                    ((o, sorted(f.items())) for o, f in part.items())}
        remap={}
        for oid,(lf, lm) in cur_head.items():
            best, bi = 0.35, None
            if lf == 0:
                for goid, pm in prev_tail.items():
                    inter=(lm&pm).sum(); union=(lm|pm).sum()
                    if union and inter/union > best: best, bi = inter/union, goid
            if bi is None: bi = nxt; nxt += 1
            remap[oid]=bi
        for oid, fr in part.items():
            g=remap[oid]
            for lf, m in fr.items(): merged.setdefault(g, {})[idx[lf]] = m
        prev_tail={remap[o]: fr[max(fr)] for o, fr in part.items()}
    return merged

def save(word, res):
    """存成 npz: RLE 压缩的 mask + 每个实例每帧的面积/贴边"""
    recs=[]; rles=[]
    for oid, fr in res.items():
        for fi, m in sorted(fr.items()):
            r=mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
            rles.append(r["counts"]); recs.append((oid, fi, int(m.sum()), edge_ratio(m)))
    if not recs: recs=[(0,0,0,0.)]; rles=[b""]
    np.savez_compressed(f"{OUT}/{word.replace(' ','_')}.npz",
        meta=np.array(recs, dtype=[("oid","i4"),("frame","i4"),("area","i8"),("edge","f4")]),
        counts=np.array(rles, dtype=object), hw=np.array([H,W]), word=word, allow_pickle=True)

pred = build_sam3_video_predictor(checkpoint_path=CKPT)
if MULTI: mw.install_hooks(pred)
print(f"[info] {NF} 帧 {W}x{H}, {len(VOCAB)} 个词, 分段大小 {SEG}, "
      f"多词模式 {'开, 每批 %d 词' % MW_GROUP if MULTI else '关'}", flush=True)
report={}

def peak():
    g = torch.cuda.max_memory_allocated()/2**30; torch.cuda.reset_peak_memory_stats(); return g

def finish(word, res, lvl, sec):
    save(word, res)
    lives=sorted((len(v) for v in res.values()), reverse=True)
    report[word]=dict(实例数=len(res), 最长帧=lives[0] if lives else 0, 级别=lvl, 秒=round(sec,1))
    return lives[0] if lives else 0

def track_single(WORD):
    """逐词: 一级 offload -> 二级提阈值 -> 三级分段"""
    th = DENSE.get(WORD); t0=time.time(); lvl="1级"
    try:
        res = track(pred, FR, th, WORD)
        if th: lvl="1+2级"
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        th = max(th or 0.7, 0.90); lvl="1+2级(重试)"
        print(f"[{WORD}] OOM, 提高检出阈值到 {th} 重试", flush=True)
        try:
            res = track(pred, FR, th, WORD)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect(); lvl="1+2+3级(分段)"
            print(f"[{WORD}] 仍 OOM, 改分段跑", flush=True)
            try:
                with mw.paused():
                    res = track_segmented(pred, th, WORD)
            except Exception as e:
                print(f"[{WORD}] 分段也失败: {type(e).__name__}", flush=True); res={}
                lvl="失败"
    live = finish(WORD, res, lvl, time.time()-t0)
    print(f"[{WORD}] {len(res)} 个实例, 最长活 {live} 帧, {lvl}, {time.time()-t0:.0f}s "
          f"(峰值 {peak():.1f}G)", flush=True)

def track_batch(words):
    """一批词共用检测; 检测爆显存就把还没跑完的词对半拆"""
    if len(words) == 1:
        return track_single(words[0])
    done = []
    try:
        with mw.shared_detection(pred, words) as sd:
            t0 = time.time()
            for w in words:
                sd.use(w); track_single(w); done.append(w)
                if len(done) == 1:
                    print(f"[多词] 首词含整批 {len(words)} 词的检测: {time.time()-t0:.0f}s", flush=True)
    except mw.SharedDetOOM as e:
        torch.cuda.empty_cache(); gc.collect()
        rest = [w for w in words if w not in done]; h = max(len(rest)//2, 1)
        print(f"[多词] {e}, 剩下 {len(rest)} 个词拆成 {h}+{len(rest)-h}", flush=True)
        track_batch(rest[:h])
        if rest[h:]: track_batch(rest[h:])

todo = []
for w in VOCAB:
    if os.path.exists(f"{OUT}/{w.replace(' ','_')}.npz"):
        print(f"[{w}] 已有, 跳过", flush=True)
    else:
        todo.append(w)
if MULTI:
    print(f"[多词] {len(todo)} 个词, 每批 {MW_GROUP} 个共用检测", flush=True)
    for b in range(0, len(todo), MW_GROUP):
        track_batch(todo[b:b+MW_GROUP])
else:
    for w in todo:
        track_single(w)
json.dump(report, open(REPORT,"w"), ensure_ascii=False, indent=1)
print(f"\n[done] {sum(r['实例数'] for r in report.values())} 个实例 -> {OUT}")
