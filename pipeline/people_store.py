#!/usr/bin/env python
"""摆人那步的结果存在哪、怎么读。写的(body_motion.py)和读的(合成 / 验收视频 / 3D 页 / 动作导出)都只走这里。

★ 为什么有它(2026-10-09): 以前"修好的人"在盘上没有一份完整的 —— 10_people.json 只存校正表, 每个读者拿它在网格上
  重放一遍(apply_corr); 动作导出嫌网格反推不准, 又在关节数据上自己修一遍。两边是两次计算, 后加的修复(锁骨头长度、
  肢体跳变)只有导出那边有: 网页上验收的人和交付的 FBX 里的人, 身体表面平均差 1~3 cm, 个别帧几十厘米。
  现在修只做一遍(在关节数据上), 结果落成下面几个文件, 谁要都从这里读。

<R1>/ 下:
  10_people.json        地面高、钉住的椅子、每人(坐没坐 / 椅子 / 脚印 / 坐姿是怎么定的)、报告、身份、solve_id
  10_people_params.npz  每人每帧的关节参数(DA3 世界系, 见 mhr_body.py) + 体型 + 帧号        <- 动作导出读
  10_people_world.npz   每人每帧的网格顶点(DA3 世界系, 就是上面的参数喂给 MHR 的结果, 带姿势修正项) + 面  <- 网页那几样读
  10_people_body.npz    物理用的人体空气墙(从上面的网格算)
  10_people_shift.json  合成那步写: 椅子被物理挪了, 坐在上面的人跟着挪多少

★ 椅子的位移不写回前四个文件: 摆人的结果写完就不再改, 重跑合成也不会把位移加两遍。位移只在 _shift() 一处加,
  网格(顶点 + d)和参数(根关节位置 + d)读出来时各加一次 —— 两边严格相等(根关节挪 d = 每个顶点挪 d)。
★ solve_id: 摆人每跑一次换一个。位移文件上记着它是给哪一次算的; 摆人重跑了而合成没重跑, 读的时候直接报错,
  不拿上一次的位移套到这一次的人身上。

只用 numpy + 标准库(加位移到参数时调 mhr_body 的两个纯 numpy 函数), 哪个环境都能 import。
"""
import json, os, time
import numpy as np

META, PARAMS, WORLD, BODY, SHIFT = ("10_people.json", "10_people_params.npz", "10_people_world.npz",
                                    "10_people_body.npz", "10_people_shift.json")
FILES = (META, PARAMS, WORLD, BODY, SHIFT)
RERUN_PEOPLE = "重跑 people(摆人)"
RERUN_COMPOSE = "重跑 people_glb(合成)"


def new_id():
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"


def _pids(z, suffix):
    return sorted((k[1:-len(suffix)] for k in z.files if k.startswith("p") and k.endswith(suffix)), key=int)


def _save_json(path, d):
    """写到临时文件再改名: 网页随时会来读, 不能让它读到半个文件。"""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(d, f, ensure_ascii=False)
    os.replace(tmp, path)


def _save_npz(path, arrays, compress=True):
    tmp = f"{path[:-4]}.tmp{os.getpid()}.npz"                  # 要以 .npz 结尾, 不然 numpy 会自己再加一个
    (np.savez_compressed if compress else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


def save(r1, meta, res, verts, faces, body, pitch, fps):
    """摆人那步落盘。meta: 10_people.json 的内容(这里补上 solve_id); res: {人: dict(frames, vf, params, shape, conf)};
    verts: {人: (T,N,3)}(和 res 的帧一一对应); body: {人: 空气墙格子}。返回 solve_id。
    ★ 10_people.json 最后写: 它在 = 这一次的几个文件都齐了。上一次的位移文件先删, 不留给这一次。"""
    sid = new_id()
    if os.path.exists(f"{r1}/{SHIFT}"):
        os.remove(f"{r1}/{SHIFT}")
    par, wv = dict(solve_id=np.array(sid), fps=np.float32(fps)), dict(solve_id=np.array(sid), faces=np.asarray(faces, np.int32))
    for pid, r in res.items():
        par[f"p{pid}_params"] = np.asarray(r["params"], np.float64)       # 双精度: 导出要和这里逐位一样
        par[f"p{pid}_shape"] = np.asarray(r["shape"], np.float64)
        par[f"p{pid}_vf"] = np.asarray(r["vf"], np.int32); par[f"p{pid}_frames"] = np.asarray(r["frames"], np.int32)
        par[f"p{pid}_conf"] = np.asarray(r["conf"], np.float32)
        wv[f"p{pid}_V"] = np.asarray(verts[pid], np.float32); wv[f"p{pid}_vf"] = np.asarray(r["vf"], np.int32)
    _save_npz(f"{r1}/{PARAMS}", par)
    _save_npz(f"{r1}/{WORLD}", wv, compress=False)             # 顶点坐标压不动: nfl1 151 MB 压完还有 125 MB, 要多等 7 秒
    _save_npz(f"{r1}/{BODY}", dict(pitch=np.float32(pitch), **{f"p_{k}": v for k, v in body.items()}))
    _save_json(f"{r1}/{META}", dict(meta, solve_id=sid))
    return sid


def clear(r1):
    """这一轮没有人了(挑人闸一个都没勾): 上一次摆人的结果全删掉, 不留给后面的步骤和网页当成"现在的人"。返回删了哪些。"""
    gone = [f for f in FILES if os.path.exists(f"{r1}/{f}")]
    for f in gone:
        os.remove(f"{r1}/{f}")
    return gone


def clear_step(r1, step):
    """这一轮没有人了(挑人闸上全点掉了), people / people_glb / motion 三步各自跳过 —— 跳过之前把自己上一次的产物清掉:
    不清的话, 物理还拿上一次的人当空气墙, 网页还把上一次的验收视频和 FBX 摆出来, 3D 页还画着人。
    验收视频不删, 挪成 _prev(和重跑时的做法一样), 别的都能由流水线重出。返回清掉了哪些。"""
    import glob, shutil
    gone = []
    if step == "people":
        gone = clear(r1)
        if gone and os.path.exists(f"{r1}/10_box_fit_view.json"):     # 3D 页的数据里也画着人: 删了, 到闸口时会重算
            os.remove(f"{r1}/10_box_fit_view.json"); gone.append("10_box_fit_view.json")
    elif step == "people_glb":
        for f in ["09_scene_nopeople.glb", "09_body4d_report.json"] + sorted(
                os.path.basename(x) for x in glob.glob(f"{r1}/09_people_f*.glb") + glob.glob(f"{r1}/09_render_4d_f*.jpg")):
            if os.path.exists(f"{r1}/{f}"):
                os.remove(f"{r1}/{f}"); gone.append(f)
        if os.path.exists(f"{r1}/09_people_check.mp4"):
            os.replace(f"{r1}/09_people_check.mp4", f"{r1}/09_people_check_prev.mp4"); gone.append("09_people_check.mp4 -> _prev")
    elif step == "motion":
        if os.path.isdir(f"{r1}/12_motion"):
            shutil.rmtree(f"{r1}/12_motion"); gone.append("12_motion/")
    return gone


def meta(r1):
    """10_people.json; 没有返回 None。"""
    p = f"{r1}/{META}"
    return json.load(open(p)) if os.path.exists(p) else None


def _need(r1):
    m = meta(r1)
    if m is None:
        raise SystemExit(f"[失败] 没有 {r1}/{META} —— {RERUN_PEOPLE}")
    if "solve_id" not in m or not all(os.path.exists(f"{r1}/{f}") for f in (PARAMS, WORLD)):
        raise SystemExit(f"[失败] {r1}/{META} 是 2026-10-09 之前的写法(人修在网格上, 没有存修好的关节数据) —— {RERUN_PEOPLE}")
    return m


def check_fresh(r1, source):
    """摆人的结果是不是按现在这份人体网格算的。source: people_identity.source_of(人体目录)。不是就报错。"""
    m = _need(r1)
    if m.get("人体来源") != source:
        raise SystemExit(f"[失败] {r1}/{META} 是按旧的人体网格算的(人体网格重出过, 或挑的人变了) —— {RERUN_PEOPLE}")
    return m


def save_shift(r1, solve_id, shifts):
    """合成那步写: {人: 位移 (3,), DA3 世界系, 米}。没人要挪也要写(空的) —— 它在 = 合成已经核过这一次摆人的结果。"""
    _save_json(f"{r1}/{SHIFT}", dict(
        solve_id=solve_id, shift={str(p): np.round(np.asarray(d, float), 5).tolist() for p, d in shifts.items()},
        说明="坐着的人那把椅子被物理挪了, 人跟着挪这么多(DA3 世界系, 米)。读网格 / 参数时由 people_store 加上"))


def _shift(r1, solve_id):
    """{人: 位移 (3,)}。位移文件不在、或者不是给这一次摆人算的 -> 报错。"""
    p = f"{r1}/{SHIFT}"
    s = json.load(open(p)) if os.path.exists(p) else None
    if s is None or s["solve_id"] != solve_id:
        raise SystemExit(f"[失败] {p} {'不存在' if s is None else '是给上一次摆人算的'} —— {RERUN_COMPOSE}")
    return {k: np.asarray(v, float) for k, v in s["shift"].items()}


def _open(r1, name):
    m = _need(r1)
    z = np.load(f"{r1}/{name}")
    if str(z["solve_id"]) != m["solve_id"]:
        raise SystemExit(f"[失败] {r1}/{name} 和 {META} 不是同一次摆人写的(上次写到一半断了?) —— {RERUN_PEOPLE}")
    return m, z


def load_world(r1, shifted=True):
    """({人: {视频帧号: 顶点 (N,3) DA3 世界系}}, 面)。shifted=False: 不加椅子位移(只有合成那步自己算位移时用)。"""
    m, z = _open(r1, WORLD)
    sh = _shift(r1, m["solve_id"]) if shifted else {}
    world = {}
    for pid in _pids(z, "_V"):
        V = z[f"p{pid}_V"].astype(np.float64) + sh.get(pid, 0.0)
        world[pid] = dict(zip(z[f"p{pid}_vf"].tolist(), V))
    return world, z["faces"]


def load_params(r1, shifted=True):
    """({人: dict(params (T,204) DA3 世界系, shape (45,), vf, frames, conf)}, fps)。位移加在根关节位置上。"""
    import mhr_body as M
    m, z = _open(r1, PARAMS)
    sh = _shift(r1, m["solve_id"]) if shifted else {}
    out = {}
    for pid in _pids(z, "_params"):
        P = z[f"p{pid}_params"].astype(np.float64)
        if pid in sh:
            P = M.set_root(P, p=M.root_pos(P) + sh[pid])
        out[pid] = dict(params=P, shape=z[f"p{pid}_shape"].astype(np.float64), vf=z[f"p{pid}_vf"],
                        frames=z[f"p{pid}_frames"], conf=z[f"p{pid}_conf"])
    return out, float(z["fps"])
