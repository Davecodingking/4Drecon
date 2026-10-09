"""第二步(body4d 环境, GPU): 用 MHR 模型本身拟合修正好的网格, 反推每帧的关节状态。
拟合时关掉姿势修正项(correctives): 导出去的 glTF/FBX 只会线性蒙皮, 这样拟合的误差就是播放时的误差。
用法: python fit_mhr.py <people.npz> <输出目录> [人id,...]"""
import os, sys, time
import numpy as np, torch

MHR = os.path.expanduser("~/sam3d/sam-body4d/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
dev = "cuda"
m = torch.jit.load(MHR, map_location=dev).eval()
ch = m.character_torch
lbs = ch.linear_blend_skinning
parts = np.load(os.path.expanduser("~/sam3d/pipeline/mhr_parts.npz"))
PEL = torch.as_tensor(parts["pelvis_idx"], device=dev).long()
NP, NS, NI = 136, 68, 45            # 姿势 / 骨骼缩放 / 体型
REG = float(os.environ.get("REG", "1e-2"))   # 体型/骨骼缩放的约束强度
CORR = os.environ.get("CORR", "0") == "1"      # 拟合时开不开姿势修正项


def fwd(idc, pose, scl):
    """idc (45,) pose (B,136) scl (68,) -> 顶点 cm (B,N,3), 骨架 (B,127,8)"""
    B = pose.shape[0]
    mp = torch.cat([pose, scl.expand(B, -1)], 1)
    return m(idc.expand(B, -1), mp, torch.zeros(B, 72, device=dev), CORR)


def kabsch(A, B):
    """R 使 (B - B.mean) ≈ (A - A.mean) @ R.T"""
    A0, B0 = A - A.mean(0), B - B.mean(0)
    U, _, Vt = torch.linalg.svd(A0.T @ B0)
    d = torch.sign(torch.det(Vt.T @ U.T))
    return Vt.T @ torch.diag(torch.tensor([1.0, 1.0, float(d)], device=dev)) @ U.T


def run(params, loss_fn, steps, lr):
    opt = torch.optim.Adam([{"params": [p], "lr": l} for p, l in zip(params, lr)])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.02 * min(lr))
    for _ in range(steps):
        opt.zero_grad(); L = loss_fn(); L.backward(); opt.step(); sch.step()
    return float(L)


def fit_person(V):
    """V: (T,N,3) 米, glTF 世界系。返回 dict。"""
    T = len(V)
    tgt = torch.as_tensor(V, device=dev) * 100.0                     # cm
    idc = torch.zeros(NI, device=dev, requires_grad=True)
    scl = torch.zeros(NS, device=dev, requires_grad=True)
    z = torch.zeros(1, NP, device=dev)
    # --- 整体朝向和位置: 平移参数的单位先探一下, 旋转从 4 个朝向起步挑最好的 ---
    with torch.no_grad():
        v0, _ = fwd(idc, z, scl)
        k = []
        for a in range(3):
            p = z.clone(); p[0, a] = 1.0
            k.append(float((fwd(idc, p, scl)[0][0, :, a] - v0[0, :, a]).mean()))
    best = None
    for yaw in (0.0, 1.5708, 3.1416, -1.5708):
        p = z.clone(); p[0, 4] = yaw
        with torch.no_grad():
            vp = fwd(idc, p, scl)[0][0]
            d = tgt[0, PEL].mean(0) - vp[PEL].mean(0)
            p[0, :3] = torch.stack([d[a] / k[a] for a in range(3)])
        p = p.requires_grad_(True)
        L = run([p], lambda: ((fwd(idc, p, scl)[0][0, PEL] - tgt[0, PEL]) ** 2).sum(-1).mean(), 300, [0.05])
        if best is None or L < best[0]:
            best = (L, p.detach().clone())
    # --- 第一帧: 全身关节 + 体型 + 骨骼缩放一起 ---
    p0 = best[1].clone().requires_grad_(True)
    run([p0, idc, scl], lambda: ((fwd(idc, p0, scl)[0][0] - tgt[0]) ** 2).sum(-1).mean()
        + REG * (idc ** 2).sum() + REG * (scl ** 2).sum(), 1500, [0.03, 0.02, 0.01])
    # --- 逐帧: 从上一帧接着拟合(体型先不动) ---
    poses = [p0.detach().clone()]
    for t in range(1, T):
        p = poses[-1].clone().requires_grad_(True)
        run([p], lambda: ((fwd(idc, p, scl)[0][0] - tgt[t]) ** 2).sum(-1).mean(), 150, [0.02])
        poses.append(p.detach().clone())
    # --- 所有帧一起精修: 体型/骨骼缩放是同一个人的, 共享 ---
    P = torch.cat(poses).requires_grad_(True)
    bs = 64
    def batch_loss():
        idx = torch.randint(0, T, (min(bs, T),), device=dev)
        return ((fwd(idc, P[idx], scl)[0] - tgt[idx]) ** 2).sum(-1).mean() \
            + REG * (idc ** 2).sum() + REG * (scl ** 2).sum()
    run([P, idc, scl], batch_loss, 600, [0.005, 0.005, 0.003])
    with torch.no_grad():
        err, skel = [], []
        for i in range(0, T, 32):
            v, s = fwd(idc, P[i:i + 32], scl)
            err.append((v - tgt[i:i + 32]).norm(dim=-1)); skel.append(s)
        err = torch.cat(err); skel = torch.cat(skel)
        rest = ch.blend_shape.base_shape + torch.einsum("k,kvd->vd", idc, ch.blend_shape.shape_vectors)
    return dict(pose=P.detach().cpu().numpy(), idc=idc.detach().cpu().numpy(), scl=scl.detach().cpu().numpy(),
                skel=skel.cpu().numpy(), rest=rest.cpu().numpy(),
                err_mean=err.mean(1).cpu().numpy(), err_max=err.max(1).values.cpu().numpy())


if __name__ == "__main__":
    src, outd = sys.argv[1], sys.argv[2]
    want = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    os.makedirs(outd, exist_ok=True)
    z = np.load(src)
    pids = sorted({k.split("_")[0][1:] for k in z.files if k.endswith("_V")}, key=int)
    common = dict(fps=z["fps"], faces=z["faces"], names=np.array(list(ch.skeleton.joint_names)),
                  parents=ch.skeleton.joint_parents.cpu().numpy(), inv_bind=lbs.inverse_bind_pose.cpu().numpy(),
                  sk_v=lbs.vert_indices_flattened.cpu().numpy(), sk_j=lbs.skin_indices_flattened.cpu().numpy(),
                  sk_w=lbs.skin_weights_flattened.cpu().numpy())
    for pid in pids:
        if want and pid not in want:
            continue
        V = z[f"p{pid}_V"]
        if os.environ.get("SLICE"): a, b = map(int, os.environ["SLICE"].split(":")); V = V[a:b]
        if len(V) < 2:
            print(f"人 {pid}: 只有 {len(V)} 帧, 跳过"); continue
        t0 = time.time()
        r = fit_person(V)
        np.savez_compressed(f"{outd}/p{pid}.npz", vf=z[f"p{pid}_vf"], **r, **common)
        print(f"人 {pid}: {len(V)} 帧  拟合误差 平均 {r['err_mean'].mean():.2f} cm (最差一帧平均 {r['err_mean'].max():.2f}), "
              f"单顶点最大 {r['err_max'].max():.2f} cm  用时 {time.time() - t0:.0f}s", flush=True)
