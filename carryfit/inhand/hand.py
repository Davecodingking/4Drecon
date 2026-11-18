"""手的坐标系: 从 MHR 的一手参数直接算每帧手腕在世界里的位姿, 再定一个有含义的"握持系"。

为什么不从网格顶点反推: 10_people_params.npz 喂给 MHR 得到的就是 10_people_world.npz 的网格(差 2e-6 m, 核过),
骨架状态里手腕关节的世界位姿是现成的, 而且对参数可导 —— 后面"让物体反过来修手臂"要用。

握持系(每人每只手一个, 固定在手腕关节上, 不随手指动):
  原点  掌心: 手腕和四个掌指关节(食指~小指根)的中点, 再沿掌心法线往外 2.5 cm (握杆时杆子待的地方)
  x     手指方向(手腕 -> 中指根)
  y     掌心法线, 朝掌心外(手心朝哪 y 就朝哪)
  z     x × y: 右手朝拇指一侧, 左手朝小指一侧(左右手互为镜像, 右手系里 y 朝掌心外就只能这样)
左右手都是右手系, y 都朝掌心外 —— 所以"托着"(物体在 +y)、"握杆"(杆沿 ±z)这些说法左右手通用。
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
SAM3D = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, f"{SAM3D}/pipeline")
import mhr_body as mb  # noqa: E402

SIDE = {"hand_R": "r", "hand_L": "l"}
ARM_GROUP = {"hand_R": "r_clavicle", "hand_L": "l_clavicle"}       # mb.limb_groups() 里整条手臂(锁骨到手腕)的参数组
PALM_OUT = 0.025


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def joint_index(dev="cpu"):
    return {str(n): i for i, n in enumerate(mb.skeleton(dev)["names"])}


def grip_local(S, J, s):
    """骨架状态 S (T,127,8) -> 握持系在手腕关节局部系里的 (Hm 3x3, o 3)。掌指关节是手腕的直接孩子, 相对手腕不动, 取各帧中位。"""
    w = J[f"{s}_wrist"]
    Rw = Rotation.from_quat(S[:, w, 3:7]).as_matrix()
    tw = S[:, w, :3] / 100.0
    loc = lambda n: np.median(np.einsum("tji,tj->ti", Rw, S[:, J[n], :3] / 100.0 - tw), 0)
    mcp = {k: loc(f"{s}_{k}1") for k in ("index", "middle", "ring", "pinky")}
    f_ = unit(mcp["middle"])
    u = unit(mcp["index"] - mcp["pinky"])
    n_ = unit(np.cross(u, f_) if s == "r" else np.cross(f_, u))
    u = np.cross(f_, n_)
    Hm = np.stack([f_, n_, u], -1)
    o = 0.5 * np.mean(list(mcp.values()), 0) + PALM_OUT * n_
    thumb = loc(f"{s}_thumb1") - 0.5 * np.mean(list(mcp.values()), 0)
    # 自检用: 拇指在哪一侧; 手指弯起来时指尖往掌心法线哪边去(各帧里最弯的那些帧)
    tips = [n for n in J if n.startswith(f"{s}_") and any(k in n for k in ("index", "middle", "ring", "pinky")) and n[-1] in "34"]
    curl = np.mean([np.percentile(np.einsum("tji,tj->ti", Rw, S[:, J[n], :3] / 100.0 - tw) @ n_, 90) for n in tips]) if tips else 0.0
    return Hm, o, float(thumb @ u) * (1 if s == "r" else -1), float(curl)


def load_params(r1):
    """{"p1": dict(params (T,204), shape (45,), vf (T,))}: 摆人那步的参数。走 people_store, 椅子位移已经加上 —— 和动作导出、
    网页读到的是同一份人。位移文件不在 / 过期(老的轮次、单独调试)就退回直接读 10_people_params.npz, 不加位移。"""
    try:
        import people_store
        people, _ = people_store.load_params(r1)
        return {f"p{k}": dict(params=np.asarray(v["params"], np.float64), shape=np.asarray(v["shape"], np.float64),
                              vf=np.asarray(v["vf"]).astype(int)) for k, v in people.items()}
    except (Exception, SystemExit) as e:
        print(f"[hand] people_store.load_params 读不了({str(e)[:100]}), 直接读 10_people_params.npz(不加椅子位移)", flush=True)
        pp = np.load(f"{r1}/10_people_params.npz")
        return {f[:-len("_params")]: dict(params=pp[f].astype(np.float64), shape=pp[f[:-len("_params")] + "_shape"].astype(np.float64),
                                         vf=pp[f[:-len("_params")] + "_vf"].astype(int)) for f in pp.files if f.endswith("_params")}


def load_people(r1, dev="cpu", params_path=None):
    """{人: dict(V (T,N,3) 世界网格, vf (T,) 视频帧号, params, shape,
              grip={hand_R/hand_L: (R (T,3,3), t (T,3))} 握持系 -> 世界,
              wrist={...} 手腕关节 -> 世界, grip_local={...: (Hm, o)})}"""
    base = load_params(r1)
    if params_path and not isinstance(params_path, (list, tuple)):   # arm_ik.py 改过手臂的人: 只换这些人的参数
        ref = np.load(params_path)
        for k in ref.files:
            if k.endswith("_params") and k[:-len("_params")] in base:
                base[k[:-len("_params")]]["params"] = ref[k].astype(np.float64)
    elif params_path:                                 # 几个物体各自改过手臂(一个人左手一样右手一样 / 不同镜头拿不同东西): 改动量加在一起
        for pth in params_path:
            ref = np.load(pth)
            for k in ref.files:
                pid = k[:-len("_arm_delta")]
                if k.endswith("_arm_delta") and pid in base:
                    P = np.array(base[pid]["params"], np.float64, copy=True)
                    P[:, ref[f"{pid}_arm_cols"]] += ref[k]
                    base[pid]["params"] = P
    J = joint_index(dev)
    out = {}
    for pid, b in base.items():
        V, S = mb.forward(b["shape"], b["params"], True, dev)
        d = dict(V=V, vf=b["vf"], params=b["params"], shape=b["shape"], grip={}, wrist={}, grip_local={},
                 thumb_side={})
        for part, s in SIDE.items():
            w = J[f"{s}_wrist"]
            Rw = Rotation.from_quat(S[:, w, 3:7]).as_matrix()
            tw = S[:, w, :3] / 100.0
            Hm, o, th_n, th_u = grip_local(S, J, s)
            d["wrist"][part] = (Rw, tw)
            d["grip"][part] = (Rw @ Hm, tw + Rw @ o)
            d["grip_local"][part] = (Hm, o)
            d["thumb_side"][part] = (th_n, th_u)          # 自检: (拇指在拇指侧 > 0, 弯指时指尖朝掌心法线 > 0)
        out[pid] = d
    return out, _faces(r1)


def _faces(r1):
    return np.load(f"{r1}/10_people_world.npz")["faces"].astype(np.int32)


def rows_for(person, vf_keys):
    """关键帧的视频帧号 -> 这个人的行号(-1 = 这帧没有这个人)"""
    row = {int(v): j for j, v in enumerate(person["vf"])}
    return np.array([row.get(int(v), -1) for v in vf_keys])


def shots_of(r1, keys, fps):
    """每个关键帧在第几个镜头(切点读 03_split_cuts.json; 没有这个文件 = 整段一个镜头)。帧号 = 帧文件名里的号。"""
    import json
    fn = np.array([int(os.path.splitext(k)[0]) for k in keys])
    p = f"{r1}/03_split_cuts.json"
    cuts = json.load(open(p)).get("切点", []) if os.path.exists(p) else []
    cf = np.array([round(c * fps) for c in cuts])
    return np.array([int((cf <= f).sum()) for f in fn]) if len(cf) else np.zeros(len(fn), int)


def shape_class(ext):
    """按包围盒三边(从长到短)分: rod 细长 / plate 扁 / blob 其它。握法的手内自由度和指标的分解都按它来, 不靠物体叫什么。"""
    e = np.sort(np.asarray(ext, float))[::-1]
    if e[1] / e[0] < 0.4:
        return "rod"
    if e[2] / e[1] < 0.4:
        return "plate"
    return "blob"
