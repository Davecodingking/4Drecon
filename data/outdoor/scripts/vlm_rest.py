#!/usr/bin/env python
"""问 Qwen 每类东西平时"放在哪": 立在地上 / 搁在别的东西上 / 挂着, 存成 01_vlm_rest.json。

为什么要有这一步 (2026-09-21 office 实测) ——
  量盒子只量得到拍到的那部分。桌子腿被椅子和人挡住, 盒子只剩桌面那一片薄板 (30~40cm 厚,
  悬在 1.2m 高); 物理一看悬空, 整张桌子扔到地上, 桌面从 1.2m 掉到 0.35m, 桌上的显示器
  杯子跟着一起掉。窗户、高处的箱子、灭火器也被扔到地上 —— 因为"挂着的"只认手写名单。
  要分清"腿被挡住的桌子"(该把腿补到地面)和"本来就挂在墙上的"(该原地不动), 得知道
  这一类东西平时放在哪 —— 这是常识, VLM 本来就有, 不该靠手写名单一个个补。

三类:
  floor    自己立在地上 (桌子/椅子/柜子/门)。组装时顶面不动、底补到地面 (assemble_scene)。
  surface  搁在别的东西上 (显示器/杯子/键盘)。物理只让它就近落到下面的台面上, 挪得多就不动。
  hung     挂墙/吊顶/建筑的一部分 (窗户/吊灯/风管)。物理不落它。
一类东西不总是一种放法 (盆栽有落地的有桌上的), 这里只给"平时"的答案; 逐个实例的判断
在用的地方做 —— assemble_scene 补地面前会先看它脚下是不是已经有别的东西托着。

为什么单独一个脚本, 不并进 vlm_sizes.py: 两个问题各问各的, 答案互不影响, 并在一起
一个解析失败两个都没了; 而且 vlm_sizes 的输出格式(扁平 词->数字)是下游写死读的。

输出扁平对象 {"table": "floor", ...}。人工改判直接改这个文件, 或者用 HUNG_LABELS 强制判挂着。
"""
import os, json, re
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

OUT   = os.environ.get("OD", "/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
VOCAB = f"{OUT}/01_vlm_vocab.json"
DST   = f"{OUT}/01_vlm_rest.json"
KINDS = ("floor", "surface", "hung")

words = json.load(open(VOCAB))["vocab"]
print(f"[info] 词表 {len(words)} 个词: {', '.join(words)}", flush=True)

PROMPT = """You are describing how objects in an indoor 3D reconstruction are physically supported.

For each object name below, answer how ONE TYPICAL such object rests, with exactly one of:
  "floor"   - it stands directly on the floor on its own legs, base or frame
              (for example a table, an office chair, a bookcase, a door)
  "surface" - it is normally placed on top of another piece of furniture
              (for example a computer monitor, a mug, a keyboard)
  "hung"    - it is fixed to a wall or the ceiling, or is part of the building above the floor
              (for example a window, a pendant lamp, a ceiling duct)

Rules:
- Judge the most common case in an office or a home.
- Answer with ONLY a flat JSON object mapping each name to "floor", "surface" or "hung".
  No nesting, no explanation, no extra keys.

Names:
""" + json.dumps(words, ensure_ascii=False)

model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
proc  = AutoProcessor.from_pretrained(MODEL)
print(f"[info] 模型就绪, 显存 {torch.cuda.memory_allocated()/2**30:.1f}G", flush=True)

msgs = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                               return_dict=True, return_tensors="pt").to("cuda")
with torch.inference_mode():
    out = model.generate(**inp, max_new_tokens=1500, do_sample=False)
txt = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

m = re.search(r"\{.*\}", txt, re.S)          # 贪婪: 对象里没有嵌套, 取到最后一个右花括号
if not m:
    raise SystemExit(f"[失败] 没解析出 JSON 对象。模型原话:\n{txt[:800]}")
try:
    raw = json.loads(m.group(0))
except Exception as e:
    raise SystemExit(f"[失败] JSON 解析不了 ({e})。模型原话:\n{txt[:800]}")

# 只留词表里有的、答案在三类之内的。没给或乱答的不猜, 下游按 surface 处理
# (不补地面、物理只做小修正) —— 那是最不会闯祸的一种。
rest, bad = {}, []
for w in words:
    v = raw.get(w)
    v = v.strip().lower() if isinstance(v, str) else v
    if v in KINDS: rest[w] = v
    else: bad.append((w, v))

json.dump(rest, open(DST, "w"), ensure_ascii=False, indent=1)
print(f"\n[放在哪] {len(rest)}/{len(words)} 个词拿到了答案")
for k in KINDS:
    ws = sorted(w for w, v in rest.items() if v == k)
    print(f"  {k:8s} {len(ws):3d}: {', '.join(ws)}")
if bad:
    print(f"[没答上] {len(bad)} 个, 下游按 surface 处理: {', '.join(f'{w}={v!r}' for w, v in bad)}")
print(f"\n[done] -> {DST}")
