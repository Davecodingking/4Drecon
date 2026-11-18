#!/usr/bin/env python
"""判视频是室内还是室外(批量跑之前用, 决定走哪条流水线)。

每个视频均匀抽 NFR 帧, 一帧一帧问 Qwen3-VL(流水线 vlm 步同一个模型, 本地加载不联网) indoor / outdoor / unsure, 投票:
  室内票过半 -> indoor; 否则 -> outdoor(说不准的按室外跑: 室内链会去拟合墙和地板组成的房间, 用在室外会出怪东西)。
  赢的一方不到 2/3 帧 -> 标"拿不准", 结果表里列出来让人事后看一眼。

用法(sam3d 环境, 要一张 ~20G 显存的卡; 没给 CUDA_VISIBLE_DEVICES 就挑空闲显存最多的那张):
  python tools/scene_kind.py <视频> [<视频> ...] --out kinds.json
输出: {视频路径: {"类型": "indoor|outdoor", "拿不准": bool, "票": {"indoor": n, "outdoor": n, "unsure": n}, "逐帧": [...]}}
"""
import argparse, json, os, re, subprocess, sys

NFR = int(os.environ.get("NFR", "6"))
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT = ("Was this video frame filmed indoors (inside a room or building) or outdoors (street, park, open air)? "
          "Answer with exactly one word: indoor, outdoor, or unsure.")


def pick_gpu():
    """没钉卡时挑空闲显存最多的(只在 GPU_POOL 里挑)。"""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    try:
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return
    free = {int(i): int(m) for i, m in (r.split(",") for r in rows)}
    pool = os.environ.get("GPU_POOL")
    if pool:
        free = {i: m for i, m in free.items() if str(i) in pool.split(",")} or free
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(max(free, key=free.get))


def frames(path, n):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    out = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, round((k + 0.5) * total / n))
        ok, bgr = cap.read()
        if not ok:
            continue
        h, w = bgr.shape[:2]
        s = min(1.0, 960 / max(h, w))                     # 判室内外不用大图, 小一点快
        if s < 1:
            bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        out.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return out


def decide(answers):
    """逐帧答案 -> (类型, 拿不准, 票)。纯函数, 测试直接调。"""
    votes = {k: answers.count(k) for k in ("indoor", "outdoor", "unsure")}
    n = len(answers)
    if n == 0:
        return "outdoor", True, votes
    kind = "indoor" if votes["indoor"] * 2 > n else "outdoor"
    unsure = votes[kind] * 3 < n * 2
    return kind, unsure, votes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pick_gpu()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    proc = AutoProcessor.from_pretrained(MODEL)
    res = {}
    for v in a.videos:
        answers = []
        for img in frames(v, NFR):
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=5, do_sample=False)
            txt = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).lower()
            m = re.search(r"indoor|outdoor|unsure", txt)
            answers.append(m.group(0) if m else "unsure")
        kind, unsure, votes = decide(answers)
        res[v] = {"类型": kind, "拿不准": unsure, "票": votes, "逐帧": answers}
        print(f"  {kind:8s}{' (拿不准)' if unsure else '':8s} 室内 {votes['indoor']} / 室外 {votes['outdoor']} / 说不准 {votes['unsure']}  {v}",
              flush=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
