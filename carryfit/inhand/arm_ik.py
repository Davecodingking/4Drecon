"""把 fit_attach.py 求出来的"修正后的手腕"落到人身上: 只动拿着物体的那条手臂的关节参数(锁骨到手腕, 每条 10 个),
让 MHR 骨架里手腕关节的世界位姿对上目标。人的其它部分一个数不动。

  目标帧: 手腕位置误差 + 朝向误差;  所有帧: 修正量小、随时间变得慢(握持段两头自然过渡回原来的动作)
输出 <out>/people_refined.npz: 只含改过的人。<人>_params (T,204) 整份参数, hand.load_people(params_path=) 读它;
  <人>_arm_cols / <人>_arm_delta / <人>_vf 是"手臂那几列各改了多少", 动作导出时由 inhand_overlay.py 叠加到摆人的结果上。
报告每段握持: 手腕位置残差(cm)、朝向残差(度) —— 手臂够不着 / 关节别不过去的帧在这里现形。

  carryfit/run.sh carryfit/inhand/arm_ik.py --r1 <r1> --targets <out>/wrist_targets.npz --out <out>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hand as hd  # noqa: E402
mb = hd.mb

DEV = "cuda"


def quat_to_mat(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--w_prior", type=float, default=0.02)
    ap.add_argument("--w_smooth", type=float, default=2.0)
    a = ap.parse_args()
    t0 = time.time()
    tg = np.load(a.targets, allow_pickle=True)
    base = hd.load_params(a.r1)
    J = hd.joint_index(DEV)
    groups = mb.limb_groups(DEV)
    m = mb.model(DEV)
    by_person = {}
    for k in range(int(tg["n"])):
        pid, part = str(tg[f"{k}_name"]).split(":")
        by_person.setdefault(pid, []).append((k, part))
    out = {}
    report = []
    for pid, segs in by_person.items():
        P0 = torch.tensor(base[pid]["params"], dtype=torch.float32, device=DEV)
        shape = torch.tensor(base[pid]["shape"], dtype=torch.float32, device=DEV)
        row = {int(v): j for j, v in enumerate(base[pid]["vf"])}
        Tn = len(P0)
        cols = sorted({int(c) for _, part in segs for c in groups[hd.ARM_GROUP[part]]})
        cols_t = torch.tensor(cols, device=DEV)
        # 每只手: 哪些行有目标, 目标是什么(同一只手的几段握持盖到同一帧时, 后面的段覆盖前面的 —— 它们在联合阶段已经被拉到一致)
        tgt = {}
        for k, part in segs:
            rows = np.array([row[int(v)] for v in tg[f"{k}_vf"]])
            d = tgt.setdefault(part, dict(mask=np.zeros(Tn, bool), R=np.tile(np.eye(3), (Tn, 1, 1)), t=np.zeros((Tn, 3))))
            d["mask"][rows], d["R"][rows], d["t"][rows] = True, tg[f"{k}_Rw"], tg[f"{k}_tw"]
        tt = {part: (torch.tensor(d["mask"], device=DEV), torch.tensor(d["R"], dtype=torch.float32, device=DEV),
                     torch.tensor(d["t"], dtype=torch.float32, device=DEV)) for part, d in tgt.items()}
        # 只算握持段前后各留 1 秒的帧(过渡用), 其余帧不动
        need = np.zeros(Tn, bool)
        for d in tgt.values():
            need |= d["mask"]
        pad = 12
        grow = np.zeros(Tn, bool)                     # 不用 np.convolve(..., "same"): 这个人的帧比窗口还少时它返回的比 Tn 长, 下标越界
        for i in np.where(need)[0]:
            grow[max(0, i - pad):i + pad + 1] = True
        idx = np.where(grow)[0]
        idx_t = torch.tensor(idx, device=DEV)
        delta = torch.zeros(len(idx), len(cols), device=DEV, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=0.02)
        zeros = torch.zeros(len(idx), mb.NEXPR, device=DEV)
        contig = torch.tensor((np.diff(idx) == 1).astype(np.float32), device=DEV)
        for it in range(a.iters + 1):
            P = P0[idx_t].clone()
            P[:, cols_t] = P[:, cols_t] + delta
            _, S = m(shape.expand(len(idx), -1), P, zeros, False)
            loss = a.w_prior * (delta ** 2).mean() + a.w_smooth * (((delta[1:] - delta[:-1]) ** 2).sum(-1) * contig).mean()
            errs = {}
            for part, (mk, Rt, tt_) in tt.items():
                w = J[f"{hd.SIDE[part]}_wrist"]
                mk_i = mk[idx_t]
                e_t = ((S[:, w, :3] / 100.0 - tt_[idx_t]) ** 2).sum(-1)
                e_R = ((quat_to_mat(S[:, w, 3:7]) - Rt[idx_t]) ** 2).sum((-1, -2))
                loss = loss + ((e_t / 0.02 ** 2 + e_R / 0.1 ** 2) * mk_i).sum() / mk_i.sum().clamp(min=1)
                errs[part] = (e_t.detach(), e_R.detach(), mk_i)
            if it == a.iters:
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
        Pn = base[pid]["params"].copy()
        Pn[np.ix_(idx, cols)] += delta.detach().cpu().numpy()
        out[f"{pid}_params"] = Pn                     # 整份参数(椅子位移已加): metrics / hand.load_people 用
        full = np.zeros((Tn, len(cols)), np.float32)
        full[idx] = delta.detach().cpu().numpy()
        out[f"{pid}_arm_cols"], out[f"{pid}_arm_delta"], out[f"{pid}_vf"] = np.array(cols, np.int32), full, base[pid]["vf"]   # 只有手臂那几列的改动: 导出时叠加(inhand_overlay.py)
        for part, (e_t, e_R, mk_i) in errs.items():
            et = e_t[mk_i].sqrt().cpu().numpy() * 100
            eR = np.degrees(2 * np.arcsin(np.clip(e_R[mk_i].sqrt().cpu().numpy() / (2 * np.sqrt(2)), 0, 1)))
            dl = np.degrees(np.abs(delta.detach().cpu().numpy()))
            report.append(dict(手=f"{pid}:{part}", 目标帧数=int(mk_i.sum()), 手腕位置残差cm_中位=round(float(np.median(et)), 2),
                               手腕位置残差cm_p95=round(float(np.percentile(et, 95)), 2), 手腕朝向残差度_中位=round(float(np.median(eR)), 1),
                               手腕朝向残差度_p95=round(float(np.percentile(eR, 95)), 1), 关节参数改动度_p95=round(float(np.percentile(dl, 95)), 1)))
    np.savez(f"{a.out}/people_refined.npz", **out)
    json.dump(dict(用时秒=round(time.time() - t0, 1), 手臂=report), open(f"{a.out}/arm_ik.json", "w"), ensure_ascii=False, indent=1)
    for r in report:
        print(json.dumps(r, ensure_ascii=False))
    print("arm_ik 用时", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
