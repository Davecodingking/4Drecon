"""调试: 某几帧把(人渲染轮廓 红, 箱子初值轮廓 绿, SAM3 箱子 mask 蓝)叠在原图上, 检查投影对不对"""
import sys, os, numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(__file__))
import fit as C
import nvdiffrast.torch as dr
r1 = "data/indoorOutput/couplelift_indoor/r1"
D = C.load_inputs(r1, "box", 1)
n, H, W = len(D["keys"]), D["H"], D["W"]
pids = sorted(D["people"]); nv = D["people"][pids[0]].shape[1]
RW, RH = 960, 540
K = torch.tensor(D["K"] * np.array([[RW / W], [RH / H], [1]]), dtype=torch.float32, device="cuda")
w2c = torch.tensor(D["w2c"], dtype=torch.float32, device="cuda")
glctx = dr.RasterizeCudaContext()
PV = torch.cat([torch.tensor(np.nan_to_num(D["people"][p], nan=-100.0), device="cuda") for p in pids], 1)
PF = torch.tensor(np.concatenate([D["faces"] + i * nv for i in range(len(pids))]), dtype=torch.int32, device="cuda")
rast, _ = dr.rasterize(glctx, C.to_clip(PV, w2c, K, RW, RH), PF, (RH, RW))
occ = (rast[..., 3] > 0).cpu().numpy()
rows = []
FR = [int(x) for x in os.environ.get("FR", "0,40,80,110").split(",")]
for i in FR:
    im = cv2.resize(cv2.imread(f"data/indoorOutput/couplelift_indoor/frames_r1/{D['keys'][i]}"), (RW, RH))
    o = im.copy()
    o[occ[i]] = (0.5 * o[occ[i]] + [0, 0, 120]).clip(0, 255)
    m = cv2.resize(D["masks"][i].astype(np.uint8), (RW, RH)) > 0
    o[m] = (0.5 * o[m] + [120, 0, 0]).clip(0, 255)
    cv2.putText(o, f"key {i}", (10, 30), 0, 1, (255, 255, 255), 2)
    rows.append(o)
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(rows[:2]), np.hstack(rows[2:])]))

# 叠上拟合结果的箱子轮廓(绿, 不做遮挡)
if len(sys.argv) > 2:
    P = np.load(sys.argv[2], allow_pickle=True)
    M = torch.tensor(P["local_to_world"], dtype=torch.float32, device="cuda")
    BV = torch.tensor(P["box_V"], device="cuda"); BF = torch.tensor(P["box_F"], dtype=torch.int32, device="cuda")
    Xw = torch.einsum("nij,vj->nvi", M[:, :3, :3], BV) + M[:, None, :3, 3]
    rast, _ = dr.rasterize(glctx, C.to_clip(Xw, w2c, K, RW, RH), BF, (RH, RW))
    sil = (rast[..., 3] > 0).cpu().numpy()
    img = cv2.imread(sys.argv[1])
    for j, i in enumerate(FR):
        y0, x0 = (j // 2) * RH, (j % 2) * RW
        e = cv2.Canny(sil[i].astype(np.uint8) * 255, 50, 150) > 0
        e = cv2.dilate(e.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        img[y0:y0 + RH, x0:x0 + RW][e] = (0, 255, 0)
    cv2.imwrite(sys.argv[1], img)
