"""流水线的 carry 步: 闸 2 标了"被人拿着"的物体, 每个求出每一帧的位姿 -> <R1>/13_carry/。
排在摆人(people)之后、导出动作(motion)之前; motion 步读这里的结果, 把物体和人一起导进 FBX / .blend。

每个物体三件事(都在 13_carry/<词>_<实例号>/ 底下):
  1. 重跟 mask(retrack.py, sam3 环境): 闸 2 时的跟踪常常只跟上一段, 拿最好的几帧当种子重新跟一遍。
     03_masks 没变就不重跟。
  2. 网格, 按顺序找:
       流水线给这个实例生成的 <生成名>              (闸 3 的网格一览图里人看过)
       流水线给这个词生成的整类网格 <标签>_<词>     (共用词 / 室内默认每类一个)
       这个词标了"共用"(闸 2 的 shared): 同一个词别的被拿着的实例已经有的网格 —— 两辆一样的滑板车用一个网格
       都没有(比如室外把会动的东西排除了没生成): make_mesh.py 从重跟的 mask 里挑一帧(大、不贴边、少挨着人)现生成
  3. 拟合每一帧的位姿(fit.py, fpose 环境)。
  4. (可选, 环境变量 CARRY_INHAND=1 才做) 挂手版(carryfit/inhand, 见那里的 README): 物体当成手的子节点重新求一遍,
     结果在 <物体目录>/inhand/。过了验收(轮廓对齐不比第二版同样那些帧低 0.10 以上、在手上的帧占一半以上)就换上:
     carry_poses.npz 换成挂手版的(第二版的留作 carry_poses_v2.npz), 汇总这一行多一栏"挂手版"; 没过就保留第二版, 栏里写原因。
     采用了的物体, motion 步导出时会把拿着它的那条手臂的修正叠加到人身上(pipeline/motion_export.py -> inhand/inhand_overlay.py)。
     开了挂手版以后还多两件事(11-17, 杀破狼逼出来的):
     · 跨镜头被拆开的同一个物体: 同一个词、都标了拿着、帧范围互不重叠的几个实例, 当成一个物体一起求(挂手版 --oid a,b,c);
       结果记在帧最多的那个实例名下, 其余几个的状态写成"并入", 不再单独进 FBX。没采用就各自保留第二版的。
     · 流水线没识别出来的物体(词表里没有): 在 <R1>/13_carry_extra.json 里写
         {"另跟": [{"词": "knife", "提示词": "dagger,blade,sword,machete", "实例": [202000, 204000, 207000]}]}
       carry 步按提示词单独跟踪(inhand/track_word.py, 结果在 13_carry/<词>_extra/track.npz 和 track.jpg); "实例"没填就停在这儿
       (这一行标失败, 原因里写着去看哪张联络图挑实例号), 填了就生成网格、求位姿, 和别的物体一样进汇总和 FBX。
全部做完出一段预览 13_carry/preview.mp4(原片 | 人 + 所有被拿着的物体 | 叠加), 汇总写 13_carry/summary.json。
★ 一个物体失败不拖垮整轮: 记进汇总, 闸 5 标红; 其余物体照常进 FBX。

sam3d 环境(要一张卡, 流水线外面领好):  python carryfit/carry_step.py --r1 <r1>
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SAM3D = os.path.dirname(HERE)
sys.path.insert(0, f"{SAM3D}/data/outdoor/scripts")
import review_decisions  # noqa: E402

HOME = os.path.expanduser("~")
PY = {k: f"{HOME}/miniforge3/envs/{k}/bin/python" for k in ("sam3", "sam3d", "fpose")}
GEN = f"{SAM3D}/sam-3d-objects/outputs"
OUT = "13_carry"
LOW = 0.5          # 四段里任何一段轮廓对齐低于它, 闸 5 标红
INHAND = os.environ.get("CARRY_INHAND", "0") not in ("", "0")     # 挂手版开关, 默认关
INHAND_DIR = f"{HERE}/inhand"
INHAND_IOU_DROP = 0.10      # 挂手版(挂回世界系的手)的轮廓对齐比第二版同样那些帧低这么多以上: 不采用
INHAND_MIN = 0.5            # 挂手版"在手上"的帧占比低于这个: 不采用


def run(cmd, log):
    print("    $ " + " ".join(os.path.relpath(x, SAM3D) if x.startswith(SAM3D) else x for x in cmd), flush=True)
    env = dict(os.environ, PYOPENGL_PLATFORM="egl", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    with open(log, "a") as f:
        f.write(f"\n== {time.strftime('%F %T')}  {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=SAM3D)
    if r.returncode:
        tail = open(log, errors="ignore").read().strip().split("\n")[-3:]
        raise RuntimeError(f"{os.path.basename(cmd[1])} 退出码 {r.returncode}: " + " | ".join(t.strip() for t in tail))


def find_mesh(h, d, r1, log, shared, have):
    """网格在哪、从哪来。见文件头的顺序。"""
    g = sorted(glob.glob(f"{GEN}/{h['生成名']}/*.glb"))
    if g:
        return g[0], "流水线生成的"
    cls = h["生成名"][: -len(f"_{h['实例']}")]            # <标签>_<词>: 整类网格
    g = sorted(glob.glob(f"{GEN}/{cls}/*.glb"))
    if g:
        return g[0], "流水线给这个词生成的整类网格"
    if h["词"] in shared and h["词"] in have:
        return have[h["词"]], f"共用: 和同词的 {os.path.basename(os.path.dirname(have[h['词']]))} 一个网格"
    name = f"{h['生成名']}_carry"
    g = sorted(glob.glob(f"{GEN}/{name}/*.glb"))
    if g:
        return g[0], "carry 步以前生成过的(从重跟的 mask 挑帧)"
    run([PY["sam3d"], f"{HERE}/make_mesh.py", "--r1", r1, "--masks", f"{d}/mask.npz", "--name", name], log)
    g = sorted(glob.glob(f"{GEN}/{name}/*.glb"))
    if not g:
        raise RuntimeError(f"make_mesh 没生成出 {GEN}/{name}/*.glb")
    return g[0], "carry 步现生成的(从重跟的 mask 挑帧)"


def inhand(h, r1, d, mesh, log, oids=None, masks=None, init=True):
    """挂手版(carryfit/inhand): 拟合 -> 手臂 IK -> 指标和四格视频, 都在 <物体目录>/inhand/。返回汇总里"挂手版"那一栏。
    失败、没过验收都不影响第二版的结果。
    oids / masks: 几个实例号当一个物体(跨镜头被拆开的), 各自的 mask 文件("" = 用 03_masks 的); init=False: 没有第二版的结果可比
    (几个实例合起来的 / 另跟的物体), 这时只按"在手上的帧占比"验收。"""
    o = f"{d}/inhand"
    shutil.rmtree(o, ignore_errors=True)
    os.makedirs(o)
    oids = oids or [h["实例"]]
    masks = masks if masks is not None else [f"{d}/mask.npz"]
    try:
        run([PY["fpose"], f"{INHAND_DIR}/fit_attach.py", "--r1", r1, "--word", h["词"], "--oid", ",".join(str(x) for x in oids), "--mesh", mesh,
             "--masks", ",".join(masks), "--out", o] + (["--init_poses", f"{d}/carry_poses.npz"] if init else []), log)
        run([PY["fpose"], f"{INHAND_DIR}/arm_ik.py", "--r1", r1, "--targets", f"{o}/wrist_targets.npz", "--out", o], log)
        run([PY["fpose"], f"{INHAND_DIR}/metrics.py", "--r1", r1, "--poses", f"{o}/carry_poses.npz",
             "--people_params", f"{o}/people_refined.npz", "--out", o, "--video"], log)
        fit, met = json.load(open(f"{o}/attach.json")), json.load(open(f"{o}/inhand_metrics.json"))
    except Exception as e:
        return dict(状态="失败", 采用=False, 原因=str(e)[:300])
    iw, i2, frac = fit.get("轮廓IoU_挂回世界系的手_中位"), fit.get("第二版同样这些帧_IoU中位"), met.get("在手上的帧占比")
    why = []
    if init and (iw is None or i2 is None or iw < i2 - INHAND_IOU_DROP):
        why.append(f"轮廓对齐 {iw} 比第二版同样这些帧的 {i2} 低了 {INHAND_IOU_DROP} 以上")
    if frac is None or frac < INHAND_MIN:
        why.append(f"在手上的帧只占 {frac}")
    return dict(状态="ok", 采用=not why, 原因="; ".join(why) or None, IoU_画面上的手=fit.get("轮廓IoU_挂着的帧_中位"), IoU_世界系的手=iw,
                第二版同样这些帧IoU=i2, 在手上的帧占比=frac, 挂在手上的帧=fit.get("挂在手上的帧"),
                没看见沿用握姿的帧=fit.get("其中没看见沿用握姿的帧"), 视频=os.path.relpath(f"{o}/inhand_view.mp4", r1))


def iou_fields(npz):
    """挂手版的位姿文件 -> 闸 5 上那几个数(轮廓对齐按"挂回世界系的手"算, 和导出的一致): 中位, 四段, 低的段"""
    import numpy as np
    iou = np.load(npz, allow_pickle=True)["iou_world"]
    medf = lambda x: round(float(np.nanmedian(x)), 3) if np.isfinite(x).any() else None
    b = np.linspace(0, len(iou), 5).astype(int)
    q = {f"关键帧{s_}-{e_ - 1}": medf(iou[s_:e_]) for s_, e_ in zip(b[:-1], b[1:])}
    return medf(iou), q, [k for k, v in q.items() if v is not None and v < LOW]


def mask_frames(npz, oid=None):
    """mask 文件里(这个实例)的帧号, 排好序; 文件不在返回 []"""
    import numpy as np
    if not os.path.exists(npz):
        return []
    m = np.load(npz, allow_pickle=True)["meta"]
    return sorted(int(f) for f, o in zip(m["frame"], m["oid"]) if oid is None or int(o) == int(oid))


def shot_trimmed_mask(dj, r1, h, more_than=0):
    """重跟的 mask 只留"这个实例原来所在的镜头"里的帧: 重跟是整段视频前后双向跟的, 会跨过硬切跟到别的东西上(杀破狼的警棍,
    三个实例合起来求时混进去 40 多帧别的镜头里的假 mask, 轮廓对齐从 0.53 掉到 0.36)。留下的帧数不超过 more_than 就返回 ""(= 用 03_masks 原来的)。
    没有切点文件 = 整段一个镜头, 原样留着。"""
    import numpy as np
    if not os.path.exists(dj):
        return ""
    fps = float(json.load(open(f"{r1}/02_da3/frame_map.json")).get("fps", 30.0))
    p = f"{r1}/03_split_cuts.json"
    cf = np.array([round(c * fps) for c in (json.load(open(p)).get("切点", []) if os.path.exists(p) else [])])
    shot = lambda f: int((cf <= f).sum()) if len(cf) else 0
    shots = {shot(f) for f in mask_frames(f"{r1}/03_masks/{h['词'].replace(' ', '_')}.npz", h["实例"])}
    z = np.load(dj, allow_pickle=True)
    m = z["meta"]
    keep = np.array([shot(int(f)) in shots for f in m["frame"]], bool)
    if keep.sum() <= more_than:
        return ""
    out = os.path.join(os.path.dirname(dj), "mask_shot.npz")
    np.savez_compressed(out, hw=z["hw"], counts=z["counts"][keep], meta=m[keep])
    return out


def held_groups(held, r1):
    """同一个词、帧范围互不重叠的几个被拿着的实例 = 跨镜头被拆开的同一个物体。返回 [[held 的下标...]], 只含两个以上的组。
    帧范围重叠的(同时出现的两辆滑板车)是不同的物体, 各管各的。"""
    import numpy as np
    keyno = {int(os.path.splitext(k)[0]) for k in json.load(open(f"{r1}/02_da3/frame_map.json"))["frames"]}
    span = {}
    for j, h in enumerate(held):
        p = f"{r1}/03_masks/{h['词'].replace(' ', '_')}.npz"
        fr = mask_frames(p, h["实例"])
        if fr:
            m = np.load(p, allow_pickle=True)["meta"]
            area = float(np.median(m["area"][m["oid"] == h["实例"]])) if "area" in m.dtype.names else 0.0
            span[j] = (fr[0], fr[-1], len([f for f in fr if f in keyno]), area)     # 帧范围, 落在关键帧上的帧数, mask 面积中位
    groups = []
    for w in sorted({h["词"] for h in held}):
        js = sorted([j for j in span if held[j]["词"] == w], key=lambda j: span[j][0])
        cur = []
        for j in js:
            if cur and span[j][0] > max(span[k][1] for k in cur):
                cur.append(j)
            else:
                if len(cur) > 1:
                    groups.append(cur)
                cur = [j]
        if len(cur) > 1:
            groups.append(cur)
    return groups, span


def extra(item, r1, root, prefix):
    """13_carry_extra.json 里"另跟"的一个物体(流水线没识别出来的): 按提示词单独跟踪 -> (人挑实例号) -> 网格 -> 挂手版。返回汇总里的一行。"""
    import numpy as np
    t0 = time.time()
    word = item["词"]
    w_ = word.replace(" ", "_")
    d = f"{root}/{w_}_extra"
    os.makedirs(d, exist_ok=True)
    log = f"{d}/log.txt"
    row = dict(生成名=f"{prefix}_{w_}_extra", 词=word, 实例=0, 目录=os.path.relpath(d, r1), 另跟=True)
    if not os.path.exists(f"{d}/track.npz"):
        run([PY["sam3"], f"{INHAND_DIR}/track_word.py", "--r1", r1, "--word", item.get("提示词") or word, "--out", f"{d}/track.npz"], log)
    oids = [int(x) for x in (item.get("实例") or [])]
    if not oids:
        return dict(row, 状态="失败", 原因=f"'{word}' 跟好了: 看 {os.path.relpath(d, r1)}/track.jpg 挑出是它的实例号, 填进 13_carry_extra.json 这一项的\"实例\", 再跑 carry")
    z = np.load(f"{d}/track.npz", allow_pickle=True)
    m = z["meta"]
    miss = [o for o in oids if not (m["oid"] == o).any()]
    if miss:
        return dict(row, 状态="失败", 原因=f"13_carry_extra.json 里 '{word}' 的实例号 {miss} 不在 track.npz 里(看 track.jpg)")
    big = max(oids, key=lambda o: int(m["area"][m["oid"] == o].max()))          # 面积最大的那个实例: 生成网格用, 也当缩略图
    k = m["oid"] == big
    np.savez_compressed(f"{d}/mask.npz", hw=z["hw"], counts=z["counts"][k], meta=m[k])
    row["实例"] = oids[0]
    mesh = item.get("网格")
    if not mesh:
        name = f"{row['生成名']}_carry"
        g = sorted(glob.glob(f"{GEN}/{name}/*.glb"))
        if not g:
            run([PY["sam3d"], f"{HERE}/make_mesh.py", "--r1", r1, "--masks", f"{d}/mask.npz", "--name", name], log)
            g = sorted(glob.glob(f"{GEN}/{name}/*.glb"))
        if not g:
            return dict(row, 状态="失败", 原因=f"make_mesh 没生成出 {GEN}/{name}/*.glb")
        mesh = g[0]
    row.update(网格=mesh, 网格来源="另跟的物体, 从面积最大的那个实例挑帧生成")
    ih = row["挂手版"] = inhand(row, r1, d, mesh, log, oids=oids, masks=[f"{d}/track.npz"] * len(oids), init=False)
    if not ih["采用"]:
        return dict(row, 状态="失败", 原因=f"挂手版 {ih['状态']}: {ih['原因']}")
    shutil.copy(f"{d}/inhand/carry_poses.npz", f"{d}/carry_poses.npz")
    iou, q, low = iou_fields(f"{d}/carry_poses.npz")
    row.update(状态="ok", 轮廓IoU=iou, 四段=q, 低的段=low, 用时秒=round(time.time() - t0))
    return row


def one(h, r1, root, shared, have, do_inhand=True):
    """一个物体: 重跟 -> 网格 -> 拟合。返回汇总里的一行。"""
    t0 = time.time()
    w_ = h["词"].replace(" ", "_")
    d = f"{root}/{w_}_{h['实例']}"
    os.makedirs(d, exist_ok=True)
    log = f"{d}/log.txt"
    row = dict(h, 目录=os.path.relpath(d, r1))
    src = f"{r1}/03_masks/{w_}.npz"
    if not (os.path.exists(f"{d}/mask.npz") and os.path.getmtime(f"{d}/mask.npz") > os.path.getmtime(src)):
        run([PY["sam3"], f"{HERE}/retrack.py", "--r1", r1, "--word", h["词"], "--oid", str(h["实例"]),
             "--out", f"{d}/mask.npz"], log)
    mesh, how = find_mesh(h, d, r1, log, shared, have)
    have.setdefault(h["词"], mesh)
    row.update(网格=mesh, 网格来源=how)
    run([PY["fpose"], f"{HERE}/fit.py", "--r1", r1, "--word", h["词"], "--oid", str(h["实例"]), "--mesh", mesh,
         "--masks", f"{d}/mask.npz", "--out", d, "--no_preview"], log)
    m = json.load(open(f"{d}/metrics.json"))
    iou0 = m.get("轮廓IoU_真人以外_中位")
    if iou0 is not None and iou0 < RETRY_IOU:
        # 拟得差: 多半是东西在被转 / 甩(伞、棍子), 放松转动约束再拟一次, 好就换上
        d2 = f"{d}/_retry_spin"; os.makedirs(d2, exist_ok=True)
        try:
            run([PY["fpose"], f"{HERE}/fit.py", "--r1", r1, "--word", h["词"], "--oid", str(h["实例"]), "--mesh", mesh,
                 "--masks", f"{d}/mask.npz", "--out", d2, "--no_preview"] + RETRY_ARGS, log)
            m2 = json.load(open(f"{d2}/metrics.json")) if os.path.exists(f"{d2}/metrics.json") else {}
        except Exception as e:                                   # 重拟失败(显存等)不拖垮这个物体: 留第一次的
            print(f"    [拿着] {h['词']} #{h['实例']}: 放松约束重拟失败({str(e)[-160:]}), 保留第一次的结果", flush=True)
            m2 = {}
        iou2 = m2.get("轮廓IoU_真人以外_中位")
        took = iou2 is not None and iou2 > iou0
        with open(log, "a") as f_:
            f_.write(f"[重拟] 重合度中位 {iou0} < {RETRY_IOU}: 放松转动约束 {RETRY_ARGS} -> {iou2}, {'换上' if took else '不换'}" + chr(10))
        print(f"    [拿着] {h['词']} #{h['实例']}: 重合度 {iou0} 偏低, 放松转动约束重拟 -> {iou2}, {'换上' if took else '保留原来的'}", flush=True)
        if took:
            for fn in os.listdir(d2):
                if os.path.isfile(f"{d2}/{fn}"): shutil.copy(f"{d2}/{fn}", f"{d}/{fn}")
            m = m2; row["重拟"] = dict(原来=iou0, 放松转动约束=iou2)
    if INHAND and do_inhand:
        ih = row["挂手版"] = inhand(h, r1, d, mesh, log, masks=[shot_trimmed_mask(f"{d}/mask.npz", r1, h) or f"{d}/mask.npz"])
        if ih["采用"]:                                     # 换上挂手版的位姿; 闸 5 上的轮廓对齐也换成它的(挂回世界系的手, 和导出的一致)
            import numpy as np
            shutil.copy(f"{d}/carry_poses.npz", f"{d}/carry_poses_v2.npz")
            shutil.copy(f"{d}/inhand/carry_poses.npz", f"{d}/carry_poses.npz")
            iou = np.load(f"{d}/carry_poses.npz", allow_pickle=True)["iou_world"]
            medf = lambda x: round(float(np.nanmedian(x)), 3) if np.isfinite(x).any() else None
            b = np.linspace(0, len(iou), 5).astype(int)
            m = dict(m, 轮廓IoU_真人以外_中位=medf(iou), 轮廓IoU_按四段={f"关键帧{s_}-{e_ - 1}": medf(iou[s_:e_]) for s_, e_ in zip(b[:-1], b[1:])})
        print(f"    [拿着] {h['词']} #{h['实例']}: 挂手版 {ih['状态']}, " + (f"采用(在手上 {ih['在手上的帧占比']}, 轮廓对齐 {ih['IoU_世界系的手']} / 第二版 {ih['第二版同样这些帧IoU']})"
              if ih["采用"] else f"不采用, 保留第二版: {ih['原因']}"), flush=True)
    q = m.get("轮廓IoU_按四段") or {}
    low = [k for k, v in q.items() if v is not None and v < LOW]
    row.update(状态="ok", 轮廓IoU=m.get("轮廓IoU_真人以外_中位"), 四段=q, 低的段=low,
               拿着它的人=m.get("拿着它的人"), 用时秒=round(time.time() - t0))
    return row


RETRY_IOU = 0.6        # 拟完轮廓重合度中位低于这个就放松转动约束重拟一次
RETRY_ARGS = ["--w_body", "0", "--held_frac", "0.1", "--w_cham", "2", "--w_grasp", "0.5"]
# ★ 11-11 加 --w_grasp 0.5(握点也放松): singin 雨伞补相机平滑后重拟, 只放松转动 0.559 / 0.561, 再放松握点 0.643(四段 0.45/0.70/0.68/0.56)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    a = ap.parse_args()
    r1 = os.path.abspath(a.r1)
    root = f"{r1}/{OUT}"
    held = review_decisions.held(r1)
    extras = []
    if INHAND and os.path.exists(f"{r1}/13_carry_extra.json"):
        extras = json.load(open(f"{r1}/13_carry_extra.json")).get("另跟") or []
    if not held and not extras:
        shutil.rmtree(root, ignore_errors=True)      # 闸口上把 held 全撤了: 不留上一次的结果给 motion 步
        print("    闸 2 没有标被人拿着的物体, 跳过")
        return
    if not os.path.exists(f"{r1}/10_people_world.npz"):
        raise SystemExit(f"[失败] 缺 {r1}/10_people_world.npz —— carry 要先有摆好的人(people 步)")
    os.makedirs(root, exist_ok=True)
    want = {f"{h['词'].replace(' ', '_')}_{h['实例']}" for h in held} | {f"{x['词'].replace(' ', '_')}_extra" for x in extras}
    for x in os.listdir(root):                        # 上一次标了、这次撤掉的
        if os.path.isdir(f"{root}/{x}") and x not in want:
            shutil.rmtree(f"{root}/{x}")
    shared = set(review_decisions.load(r1).get("共用词") or [])
    print(f"    被人拿着的 {len(held)} 个: " + ", ".join(f"{h['词']}#{h['实例']}" for h in held))
    groups, span = held_groups(held, r1) if INHAND else ([], {})
    grouped = {j for g in groups for j in g}
    for g in groups:
        print("    跨镜头的同一个物体(同词、帧不重叠), 挂手版合起来求: " + ", ".join(f"{held[j]['词']}#{held[j]['实例']}(帧 {span[j][0]}-{span[j][1]})" for j in g))
    rows, have = [], {}
    for j, h in enumerate(held):
        try:
            r = one(h, r1, root, shared, have, do_inhand=j not in grouped)
            flag = f"  ★ 有一段低于 {LOW}: {', '.join(r['低的段'])}" if r["低的段"] else ""
            print(f"    [拿着] {h['词']} #{h['实例']}: 轮廓对齐 {r['轮廓IoU']}, 四段 "
                  + " / ".join("-" if v is None else f"{v:.2f}" for v in r["四段"].values())
                  + f", 网格 {r['网格来源']}, {r['用时秒']} s{flag}", flush=True)
        except Exception as e:                          # 一个失败不拖垮整轮, 闸 5 标红
            r = dict(h, 状态="失败", 原因=str(e)[:500])
            print(f"    [失败] {h['词']} #{h['实例']}: {e}", flush=True)
        rows.append(r)
    for g in groups:                                    # 跨镜头合起来的: 记在帧最多的那个实例名下
        mem = [j for j in g if rows[j]["状态"] == "ok"]
        if len(mem) < 2:
            continue
        lead = max(mem, key=lambda j: span[j][2])
        lr, t0 = rows[lead], time.time()
        d = f"{r1}/{lr['目录']}"
        masks = []
        for j in mem:                                   # 每个实例用帧多的那份 mask: 重跟的(裁到它自己的镜头里; 细东西重跟常常反而丢帧) / 03_masks 原来的
            masks.append(shot_trimmed_mask(f"{r1}/{rows[j]['目录']}/mask.npz", r1, held[j], more_than=span[j][2]))
        # 网格: 几个实例各有各的, 用"在画面里最大、帧数又不太少(够挑出一帧好的)"的那个实例的 —— 糊的远景镜头生成出来的不像样
        mj = max([j for j in mem if span[j][2] >= 0.5 * span[lead][2]], key=lambda j: span[j][3])
        ih = lr["挂手版"] = inhand(held[lead], r1, d, rows[mj]["网格"], f"{d}/log.txt", oids=[held[j]["实例"] for j in mem], masks=masks, init=False)
        ih["合并"] = [rows[j]["生成名"] for j in mem]
        ih["网格"] = rows[mj]["网格"]
        if ih["采用"]:
            shutil.copy(f"{d}/carry_poses.npz", f"{d}/carry_poses_v2.npz")
            shutil.copy(f"{d}/inhand/carry_poses.npz", f"{d}/carry_poses.npz")
            iou, q, low = iou_fields(f"{d}/carry_poses.npz")
            lr.update(轮廓IoU=iou, 四段=q, 低的段=low, 用时秒=lr.get("用时秒", 0) + round(time.time() - t0))
            for j in mem:
                if j != lead:
                    rows[j].update(状态="并入", 原因=f"和 {lr['生成名']} 是同一个东西(跨镜头被拆成了几个实例号), 并进去一起求了")
        print(f"    [拿着] {lr['词']} " + " + ".join(f"#{held[j]['实例']}" for j in mem) + f": 挂手版 {ih['状态']}, "
              + (f"采用, 记在 #{held[lead]['实例']} 名下(在手上 {ih['在手上的帧占比']}, 挂着 {ih['挂在手上的帧']} 帧)" if ih["采用"] else f"不采用, 各自保留第二版: {ih['原因']}"), flush=True)
    if extras:
        any_name = next((h["生成名"][: -len(f"_{h['词'].replace(' ', '_')}_{h['实例']}")] for h in held), None)
        prefix = any_name or f"{os.path.basename(os.path.dirname(r1))[:3]}_{os.path.basename(r1)}"
        for item in extras:
            try:
                r = extra(item, r1, root, prefix)
            except Exception as e:
                r = dict(生成名=f"{prefix}_{item['词'].replace(' ', '_')}_extra", 词=item["词"], 实例=0, 状态="失败", 原因=str(e)[:500])
            print(f"    [另跟] {item['词']}: " + (f"挂手版采用(在手上 {r['挂手版']['在手上的帧占比']}, 挂着 {r['挂手版']['挂在手上的帧']} 帧), 轮廓对齐 {r['轮廓IoU']}"
                  if r["状态"] == "ok" else f"{r['状态']}: {r['原因']}"), flush=True)
            rows.append(r)
    ok = [r for r in rows if r["状态"] == "ok"]
    if ok:
        run([PY["fpose"], f"{HERE}/preview_multi.py", "--r1", r1, "--poses"]
            + [f"{r1}/{r['目录']}/carry_poses.npz" for r in ok] + ["--out", f"{root}/preview.mp4"], f"{root}/preview.log")
    json.dump(dict(物体=rows, 门槛=LOW, 预览="preview.mp4" if ok else None), open(f"{root}/summary.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"[done] {len(ok)}/{len(rows)} 个求出了每帧位姿 -> {root}/summary.json" + (", 预览 preview.mp4" if ok else ""))


if __name__ == "__main__":
    main()
