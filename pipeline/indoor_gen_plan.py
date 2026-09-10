#!/usr/bin/env python
"""定"谁单独生成一个 mesh、谁复用同类的 mesh", 并把三个 jobs 文件写出来。

为什么要这一步: 室内场景重复件极多 (十几组一模一样的餐桌椅)。逐个 SAM3D --hq 生成
是几十次 x 1-2 分钟, 而且远处小桌子的抠图质量本来就差。所以走混合策略:
  * SOLO_CLS 里的类 (吧台/沙发/大盆栽这种形态各异的主体) -> 逐实例生成
  * 其余类 -> 该类里配图分最高的那个实例生成一个 mesh (proto), 同类其余实例复用它

复用能成立, 是因为 hybrid_place_5090.py 的 --map 本来就是逐条 "idx:名字" 加载 glb
再按各自盒子的 yaw/尺寸归位 —— 同一个名字出现多次天然就是复用, place 脚本一行都不用改。
代价是同类物体长得一模一样。

用法: python indoor_gen_plan.py --csv 06_配对/配对表.csv --tag cyb_r1 \
        --solo-cls bar,sofa,tv --out 07_生成计划.json --jobs-dir <logs目录>
产出: <out>                      人看的计划 (每个盒子的 role/名字/来源帧/理由)
      <jobs-dir>/mask_jobs.txt   抠图作业, TSV: 名字 \\t 帧文件名 \\t 类别 \\t x1,y1,x2,y2
      <jobs-dir>/gen_jobs.txt    生成作业, 一行一个名字
      <jobs-dir>/place_map.txt   归位映射, hybrid_place --map 直接吃这一行
"""
import argparse, ast, csv, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="hybrid_match_5090.py 产出的 配对表.csv")
ap.add_argument("--tag", required=True, help="产物名前缀, 如 cyb_r1")
ap.add_argument("--solo-cls", default="", help="逗号分隔, 这些类逐实例生成")
ap.add_argument("--out", required=True)
ap.add_argument("--jobs-dir", required=True)
ap.add_argument("--keep-status", default="ok,weak",
                help="哪些 status 参与生成。weak 也留着 —— r2 里 0.247 的地毯照样能用")
a = ap.parse_args()

SOLO = set(x.strip() for x in a.solo_cls.split(",") if x.strip())
KEEP = set(x.strip() for x in a.keep_status.split(",") if x.strip())

rows = list(csv.DictReader(open(a.csv)))
usable = [r for r in rows if r["status"] in KEEP and r["crop_rect"].strip().startswith("[")]
keep_ids = {id(r) for r in usable}
dropped = [r for r in rows if id(r) not in keep_ids]

# 同类里挑配图分最高的当 proto
best_of = {}
for r in usable:
    c = r["class"]
    if c in SOLO: continue
    if c not in best_of or float(r["score"]) > float(best_of[c]["score"]): best_of[c] = r

plan, mask_jobs, gen_jobs, place = [], [], [], []
for r in usable:
    idx, cls = int(r["idx"]), r["class"]
    if cls in SOLO:
        role, name = "solo", f"{a.tag}_obj{idx}_{cls}"
    elif best_of[cls] is r:
        role, name = "proto", f"{a.tag}_{cls}"
    else:
        role, name = "reuse", f"{a.tag}_{cls}"
    if role != "reuse":
        x1, y1, x2, y2 = ast.literal_eval(r["crop_rect"])
        mask_jobs.append(f"{name}\t{r['best_frame']}\t{cls}\t{x1},{y1},{x2},{y2}")
        gen_jobs.append(name)
    place.append(f"{idx}:{name}")
    plan.append(dict(idx=idx, cls=cls, role=role, name=name, status=r["status"],
                     score=float(r["score"]), frame=r["best_frame"],
                     proto_of=(best_of[cls]["idx"] if role == "reuse" else None)))

for r in dropped:
    plan.append(dict(idx=int(r["idx"]), cls=r["class"], role="skip", name=None,
                     status=r["status"], why=r["crop_rect"] or "无合格视角"))

os.makedirs(a.jobs_dir, exist_ok=True)
json.dump(plan, open(a.out, "w"), ensure_ascii=False, indent=1)
open(f"{a.jobs_dir}/mask_jobs.txt", "w").write("\n".join(mask_jobs) + "\n")
open(f"{a.jobs_dir}/gen_jobs.txt", "w").write("\n".join(gen_jobs) + "\n")
open(f"{a.jobs_dir}/place_map.txt", "w").write(",".join(place) + "\n")

n = {k: sum(1 for p in plan if p["role"] == k) for k in ("solo", "proto", "reuse", "skip")}
print(f"[done] 盒子 {len(rows)} 个: 单独 {n['solo']} + 代表 {n['proto']} + 复用 {n['reuse']} "
      f"+ 跳过 {n['skip']}  =>  要生成 {len(gen_jobs)} 个 mesh")
for p in plan:
    if p["role"] == "skip": print(f"  obj{p['idx']:2d} {p['cls']:16s} 跳过 ({p['status']})")
