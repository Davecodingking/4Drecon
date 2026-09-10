#!/usr/bin/env python
"""让 Qwen3-VL 扫若干帧, 只做一件事: 列出画面里有哪些可以单独拿出来的物体(名词短语).
输出合并去重后的词表, 供 SAM3 视频跟踪使用. 不让它画框 —— 定位交给 SAM3."""
import os, sys, json, re, collections
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

FR   = os.environ.get("FR", "/home/jincheng/sam3d/data/outdoor/dataset/timesquare/frames")
OUT  = os.environ.get("OD", "/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
NFR  = int(os.environ.get("NFR", "8"))          # 扫几帧
MODEL= "Qwen/Qwen3-VL-8B-Instruct"
os.makedirs(OUT, exist_ok=True)

PROMPT = """You are helping build a 3D reconstruction pipeline. Look at this street scene frame.

List every DISCRETE PHYSICAL OBJECT that could be cut out and rebuilt as its own 3D model.

Rules:
- Use short, simple, singular noun phrases (2 words max), the kind a segmentation model understands: "taxi", "trash can", "street lamp", "potted plant".
- Only things with a clear boundary. SKIP background surfaces: road, sky, sidewalk, building facade, wall, ground.
- SKIP anything that is a texture or a pattern rather than an object.
- Include things that are easy to overlook: bollards, barriers, awnings, scaffolding, benches, signs, planters, traffic cones, newspaper boxes, fire hydrants, manhole covers, billboards.
- Do not invent things you cannot actually see.

Answer with ONLY a JSON array of strings, nothing else."""

files = sorted((f for f in os.listdir(FR) if f.endswith(".jpg")), key=lambda x: int(x.split('.')[0]))
idx   = [round(i*(len(files)-1)/(NFR-1)) for i in range(NFR)]
print(f"[info] {len(files)} 帧, 抽 {NFR} 帧: {idx}", flush=True)

model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
proc  = AutoProcessor.from_pretrained(MODEL)
print(f"[info] 模型就绪, 显存 {torch.cuda.memory_allocated()/2**30:.1f}G", flush=True)

votes, per_frame = collections.Counter(), {}
for i in idx:
    img = Image.open(os.path.join(FR, files[i])).convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":PROMPT}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   return_dict=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=400, do_sample=False)
    txt = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.search(r"\[.*?\]", txt, re.S)
    try:    words = [w.strip().lower() for w in json.loads(m.group(0))] if m else []
    except Exception: words = []
    per_frame[files[i]] = words
    votes.update(set(words))
    print(f"[帧 {files[i]}] {len(words)} 个: {', '.join(words)}", flush=True)

vocab = [w for w,c in votes.most_common()]
json.dump({"vocab": vocab, "votes": dict(votes), "per_frame": per_frame},
          open(f"{OUT}/01_vlm_vocab.json","w"), ensure_ascii=False, indent=1)
print(f"\n[合并词表] {len(vocab)} 个")
for w,c in votes.most_common(): print(f"  {c}/{NFR}帧  {w}")
print(f"\n[done] {OUT}/01_vlm_vocab.json")
