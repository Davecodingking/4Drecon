#!/usr/bin/env python
"""挪类别: 把某几个实例从一个词的 mask 文件搬到另一个词。mask 一个像素都不改, 只换标签。

为什么需要这一步 ——
  SAM3 按词跟踪, 词是 VLM 列的。同一个东西被列成什么词, 决定了它后面的一切:
  和谁归成一类、拿哪个常识尺寸当尺子、类内统不统一尺度。
  实测: beach 的 street lamp 里混进了 3 个其实是 light fixture 的东西 —— 它们和真正的
  路灯高度差好几倍, 混在一类里做尺度统一, 两边都会被拉歪。
  这是人一眼看得出、机器判不了的事(VLM 只答"这是不是路灯", 答不了"它和另外 13 个
  是不是同一种路灯"), 所以放在人工闸后面。

为什么是"搬家"而不是"带一张映射表往下传" ——
  mask 是按词存的(street_lamp.npz 里装着它所有实例)。真把实例搬过去之后, 挑图、生成、
  量盒子、组装**全都照常读 npz, 一行都不用改**。带映射表往下传的话, 每一个消费者都要
  记得查表, 而"每个消费者都要记得"正是这条链上反复出事的那个形状。

幂等 + 可回退: 原始 npz 备份在 03_masks_prerelabel/ (在 03_masks 外面), 已经搬过的不重复搬。

输入: 05_review_applied.json 的 "改类别" 字段, 形如 {"street lamp#38": "light fixture"}
"""
import os, sys, json, shutil
import numpy as np

R1  = os.environ["R1"]
MK  = f"{R1}/03_masks"
# ★ 备份放在 03_masks **外面**。放里面的话 pick_from_masks 遍历目录时会把它当成一个 npz
#   文件去读, 直接 IsADirectoryError —— 实测踩过。
BAK = f"{R1}/03_masks_prerelabel"
DEC = f"{R1}/05_review_applied.json"
STATE = f"{R1}/03_relabel_done.json"


def npz_path(word):
    return f"{MK}/{word.replace(' ', '_')}.npz"


def load(word):
    p = npz_path(word)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return dict(meta=z["meta"], counts=z["counts"], hw=z["hw"],
                word=str(z["word"]) if "word" in z else word)


def save(word, d):
    np.savez_compressed(npz_path(word), meta=d["meta"], counts=d["counts"],
                        hw=d["hw"], word=word, allow_pickle=True)


def main():
    if not os.path.exists(DEC):
        print("[跳过] 没有决策文件, 不用挪类别"); return 0
    rel = json.load(open(DEC)).get("改类别") or {}
    if not rel:
        print("[跳过] 决策里没有挪类别"); return 0

    done = json.load(open(STATE)) if os.path.exists(STATE) else {}
    todo = {k: v for k, v in rel.items() if done.get(k) != v}
    if not todo:
        print(f"[跳过] {len(rel)} 条挪类别已经做过了(幂等)"); return 0

    os.makedirs(BAK, exist_ok=True)
    # 按源词分组, 一个词只读写一次
    by_src = {}
    for key, dst in todo.items():
        if "#" not in key:
            raise SystemExit(f"[失败] 改类别的键要写成 '词#实例号', 拿到 {key!r}")
        src, oid = key.rsplit("#", 1)
        by_src.setdefault(src, []).append((int(oid), dst))

    moved_total = 0
    for src, items in sorted(by_src.items()):
        S = load(src)
        if S is None:
            raise SystemExit(f"[失败] 没有 {npz_path(src)} —— 挪类别的源词不存在")
        # 备份一次, 不覆盖(第一次的才是原始)
        for w in {src} | {d for _, d in items}:
            b = f"{BAK}/{w.replace(' ', '_')}.npz"
            if os.path.exists(npz_path(w)) and not os.path.exists(b):
                shutil.copy2(npz_path(w), b)

        for oid, dst in sorted(items):
            sel = np.array([int(r["oid"]) == oid for r in S["meta"]])
            if not sel.any():
                print(f"  [跳过] {src}#{oid} 不存在(可能已经搬走了)"); continue
            D = load(dst)
            if D is None:                       # 目标词还没有 npz, 新建一个
                D = dict(meta=S["meta"][:0].copy(), counts=S["counts"][:0].copy(),
                         hw=S["hw"], word=dst)
            if tuple(D["hw"]) != tuple(S["hw"]):
                raise SystemExit(f"[失败] {src} 和 {dst} 的画幅不一样 {S['hw']} vs {D['hw']}")
            # 目标词里换一个不冲突的实例号 —— 原号很可能已经被占了
            used = {int(r["oid"]) for r in D["meta"]}
            new_oid = oid if oid not in used else (max(used) + 1 if used else 0)

            mv = S["meta"][sel].copy()
            mv["oid"] = new_oid
            D["meta"] = np.concatenate([D["meta"], mv])
            D["counts"] = np.concatenate([D["counts"], S["counts"][sel]])
            save(dst, D)

            S["meta"] = S["meta"][~sel]
            S["counts"] = S["counts"][~sel]
            n = int(sel.sum())
            moved_total += 1
            print(f"  {src}#{oid} -> {dst}#{new_oid}   ({n} 帧的 mask 跟着走)")
            done[f"{src}#{oid}"] = dst
        save(src, S)
        left = len({int(r['oid']) for r in S['meta']})
        print(f"  [{src}] 还剩 {left} 个实例")

    json.dump(done, open(STATE, "w"), ensure_ascii=False, indent=1)
    print(f"\n[done] 挪了 {moved_total} 个实例。原始 npz 备份在 {BAK}/")
    print("       ★ 挪完之后挑图/生成/量盒子/组装照常读 npz, 不用改任何代码。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
