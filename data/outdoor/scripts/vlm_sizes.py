#!/usr/bin/env python
"""问 Qwen 要每个词的典型真实高度(米), 存成 01_vlm_sizes.json, 给 world_scale.py 当尺子。

为什么要有这一步 —— DA3 说自己给的是米, 但在俯拍素材上判得很离谱: carpark 量出来的车高
0.23 米, 真车 1.48 米, 整个世界被压缩了 6.5 倍。要把它拉回真米, 就得有个参照物, 而参照物
只能来自"常识": 车多高、路灯多高、长椅多高。这种常识 VLM 本来就有。

为什么单独一个脚本, 而不是往列词那步里加一个字段 ——
  1. 词表纯 VLM 是刻意保持的实验条件, 重跑 vlm_list_objects.py 会换出一批不一样的词;
  2. 尺寸问答不需要看图, 是纯文本问题, 混进逐帧的视觉调用里纯属浪费。
所以这里只读现成的 01_vlm_vocab.json, 一次问完所有词。

输出扁平对象 {"car": 1.5, ...}, 不要嵌套 —— 解析用的正则和列词那步同源, 是非贪婪的,
遇到嵌套会在第一个右括号处截断。
"""
import os, json, re
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

OUT   = os.environ.get("OD", "/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
VOCAB = f"{OUT}/01_vlm_vocab.json"
DST   = f"{OUT}/01_vlm_sizes.json"

words = json.load(open(VOCAB))["vocab"]
print(f"[info] 词表 {len(words)} 个词: {', '.join(words)}", flush=True)

PROMPT = """You are calibrating the metric scale of a 3D reconstruction.

For each object name below, give the TYPICAL REAL-WORLD HEIGHT IN METRES of one such object,
standing in its normal upright orientation on the ground.

Rules:
- Height means the vertical dimension, not length or width. A car is about 1.5 (not 4.5).
- Give the typical value for an average real instance, not a record or an extreme.
- If a name is a background surface, a texture, or something with no meaningful single height
  (road, sky, wall, facade, water, sand), give null instead of a number.
- Answer with ONLY a flat JSON object mapping each name to a number or null. No nesting,
  no units, no explanation, no extra keys.

Names:
""" + json.dumps(words, ensure_ascii=False)

model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
proc  = AutoProcessor.from_pretrained(MODEL)
print(f"[info] 模型就绪, 显存 {torch.cuda.memory_allocated()/2**30:.1f}G", flush=True)

msgs = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                               return_dict=True, return_tensors="pt").to("cuda")
with torch.inference_mode():
    out = model.generate(**inp, max_new_tokens=1200, do_sample=False)
txt = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

m = re.search(r"\{.*\}", txt, re.S)          # 贪婪: 对象里没有嵌套, 取到最后一个右花括号
if not m:
    raise SystemExit(f"[失败] 没解析出 JSON 对象。模型原话:\n{txt[:800]}")
try:
    raw = json.loads(m.group(0))
except Exception as e:
    raise SystemExit(f"[失败] JSON 解析不了 ({e})。模型原话:\n{txt[:800]}")

# 只留词表里有的、是正数的。null 和离谱值(超出 0.02~200 米)在这里就丢掉,
# 免得脏数据流到下游还要再判一次。
sizes, dropped = {}, []
for w in words:
    v = raw.get(w)
    if isinstance(v, (int, float)) and 0.02 < float(v) < 200:
        sizes[w] = round(float(v), 3)
    else:
        dropped.append((w, v))

json.dump(sizes, open(DST, "w"), ensure_ascii=False, indent=1)
print(f"\n[尺子] {len(sizes)}/{len(words)} 个词拿到了高度")
for w, h in sorted(sizes.items(), key=lambda x: -x[1]):
    print(f"  {h:6.2f} m   {w}")
if dropped:
    print(f"[没给数] {len(dropped)} 个: {', '.join(w for w, _ in dropped)}")
print(f"\n[done] -> {DST}")
