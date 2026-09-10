#!/usr/bin/env python
"""室内版词表: 让 Qwen3-VL 扫若干帧, 列出画面里可以单独拿出来的物体(名词短语).
输出合并去重后的词表, 供 SAM3 视频跟踪使用. 不让它画框 —— 定位交给 SAM3.

和 data/outdoor/scripts/vlm_list_objects.py 的区别只有两处, 其余逐行照抄:
  1) PROMPT 换室内版: 场景描述、示例词、SKIP 列表(原版缺 ceiling/floor)、
     以及"不要桌面小物"这一条(用户拍板: 这一轮只做大件, 蜡烛酒杯太小重建不了)
  2) 空返回重试一次并打警告。原版 `except: words=[]` 不报警不重试 ——
     ISSUES.md §3: beach 上 8 帧里 2 帧空返回, 唯一有建筑的那帧就在里面,
     整个建筑词表静默丢失。

★ 这一步是整条链能赢 SpatialLM 的全部理由: SpatialLM 的词表是写死的 59 类住宅家具
  (SpatialLM/inference.py:161), 里面没有 piano、没有 bar counter; 这里是开放词表。

用法: FR=<帧目录> OD=<输出目录> [NFR=8] python vlm_list_objects_indoor.py
"""
import os, sys, json, re, collections
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

FR   = os.environ.get("FR", "/home/jincheng/sam3d/data/indoorOutput/cyberpunk/frames_bright")
OUT  = os.environ.get("OD", "/home/jincheng/sam3d/data/indoorOutput/cyberpunk/o1")
NFR  = int(os.environ.get("NFR", "8"))          # 扫几帧
MODEL= "Qwen/Qwen3-VL-8B-Instruct"
os.makedirs(OUT, exist_ok=True)

PROMPT = """You are helping build a 3D reconstruction pipeline. Look at this frame from an indoor bar / lounge.

List every DISCRETE PHYSICAL OBJECT that could be cut out and rebuilt as its own 3D model.

Rules:
- Use short, simple, singular noun phrases (2 words max), the kind a segmentation model understands: "bar stool", "dining table", "potted palm", "grand piano".
- Only things with a clear boundary. SKIP background surfaces: floor, ceiling, wall, window glass, curtain wall, railing panel.
- SKIP anything that is a texture, a light, or a reflection rather than an object.
- Include furniture that is easy to overlook: bar counter, bar stool, booth seat, side table, planter, floor lamp, screen, shelf, sofa, armchair.
- SKIP small tabletop items (candles, glasses, cutlery, bottles, plates) - they are too small to reconstruct.
- Do not invent things you cannot actually see.

Answer with ONLY a JSON array of strings, nothing else."""

files = sorted(f for f in os.listdir(FR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
idx   = [round(i*(len(files)-1)/(NFR-1)) for i in range(NFR)]
print(f"[info] {len(files)} 帧, 抽 {NFR} 帧: {idx}", flush=True)

model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
proc  = AutoProcessor.from_pretrained(MODEL)
print(f"[info] 模型就绪, 显存 {torch.cuda.memory_allocated()/2**30:.1f}G", flush=True)

def ask(img, greedy=True):
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":PROMPT}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   return_dict=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=400, do_sample=not greedy,
                             **({} if greedy else dict(temperature=0.7, top_p=0.9)))
    txt = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.search(r"\[.*?\]", txt, re.S)
    try:    return [w.strip().lower() for w in json.loads(m.group(0))] if m else []
    except Exception: return []

votes, per_frame, empties = collections.Counter(), {}, []
for i in idx:
    img = Image.open(os.path.join(FR, files[i])).convert("RGB")
    words = ask(img)
    if not words:                       # ★ 原版到此为止, 静默丢内容
        print(f"[警告] 帧 {files[i]} 返回空, 重试一次(采样)", flush=True)
        words = ask(img, greedy=False)
        if not words: empties.append(files[i])
    per_frame[files[i]] = words
    votes.update(set(words))
    print(f"[帧 {files[i]}] {len(words)} 个: {', '.join(words)}", flush=True)

vocab = [w for w,c in votes.most_common()]
json.dump({"vocab": vocab, "votes": dict(votes), "per_frame": per_frame, "空返回帧": empties},
          open(f"{OUT}/01_vlm_vocab.json","w"), ensure_ascii=False, indent=1)
print(f"\n[合并词表] {len(vocab)} 个")
for w,c in votes.most_common(): print(f"  {c}/{NFR}帧  {w}")
if empties: print(f"\n[警告] {len(empties)} 帧重试后仍为空: {empties} —— 这些帧里的东西全丢了")
print(f"\n[done] {OUT}/01_vlm_vocab.json")
