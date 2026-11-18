"""动作导出时, 把挂手版修过的手臂叠加到摆人的结果上(只有 numpy, 哪个环境都能 import)。

挂手版(fit_attach.py + arm_ik.py)为了让物体刚性地待在手里, 会小幅改拿着它的那条手臂的关节参数(手腕为主)。
物体的每帧位姿是按"改过的手"算的 —— 导出的人如果还是没改的手臂, 物体和手就对不上。
carry 步采用了挂手版结果的物体(13_carry/summary.json 这一行的 挂手版.采用 = true), 而且闸 5 上没点"不要"的, 在这里把
<物体目录>/inhand/people_refined.npz 里"手臂那几列的改动"加到 people_store.load_params 读出来的参数上。
摆人的结果文件一个字节不动; 改动只进 12_motion 的导出(FBX / blend / motion.npz)。
★ 网页上的验收视频和 3D 页读的是 10_people_world.npz(没改的手臂), 和导出的 FBX 在这条手臂上差一个手腕修正量。
"""
import json
import os

import numpy as np


def refined_files(r1, names=None):
    """[(生成名, people_refined.npz 路径)]: carry 步采用了挂手版的物体。names 给了就只要这些(闸 5 上没点不要的)。"""
    try:
        rows = json.load(open(f"{r1}/13_carry/summary.json")).get("物体") or []
    except (OSError, ValueError):
        return []
    out = []
    for r in rows:
        if r.get("状态") != "ok" or not (r.get("挂手版") or {}).get("采用"):
            continue
        if names is not None and r["生成名"] not in names:
            continue
        p = f"{r1}/{r['目录']}/inhand/people_refined.npz"
        if os.path.exists(p):
            out.append((r["生成名"], p))
    return out


def apply(r1, people, names=None):
    """people: people_store.load_params 的第一个返回值 {人号: dict(params, vf, ...)}。就地加上手臂的改动, 返回说明文字的列表。"""
    notes = []
    for name, p in refined_files(r1, names):
        z = np.load(p)
        for k in z.files:
            if not k.endswith("_arm_delta"):
                continue
            pid = k[:-len("_arm_delta")]                 # "p4"
            key = pid[1:] if pid[1:] in people else (int(pid[1:]) if int(pid[1:]) in people else None)
            if key is None:
                notes.append(f"{name}: 人 {pid} 不在这次摆人的结果里, 跳过")
                continue
            r = people[key]
            if len(z[f"{pid}_vf"]) != len(r["vf"]) or (np.asarray(z[f"{pid}_vf"]) != np.asarray(r["vf"])).any():
                raise SystemExit(f"[失败] {name}: 挂手版修手臂时的人({pid})和现在摆人的结果帧对不上 —— 摆人重跑过, 要重跑 carry")
            cols, d = z[f"{pid}_arm_cols"], z[f"{pid}_arm_delta"]
            P = np.array(r["params"], np.float64, copy=True)
            P[:, cols] += d
            r["params"] = P
            notes.append(f"{name}: 人 {pid} 的手臂参数叠加了挂手版的修正({int((np.abs(d).max(1) > 1e-6).sum())} 帧, "
                         f"最大 {np.degrees(np.abs(d).max()):.1f} 度)")
    return notes
