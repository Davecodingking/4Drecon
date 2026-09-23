#!/usr/bin/env python
"""量盒子的第二条路: 用 WildDet3D 把 2D 框抬成 3D 框, 替代 boxes_from_masks.py 的点云主轴。

为什么要有它 —— 现行 PCA 那条有两个几何上无解的毛病:
  1. 朝向只有轴没有正反。点云主轴给你一条线, 但车头朝哪它不知道, 而且素材是静止的
     游戏画面, "用运动定朝向"也用不上。正反是语义, 几何里根本没有这个信息。
  2. 尺寸系统性偏小 20~40%。点云只量得到朝相机那一面, 背面没拍到就是没有。
WildDet3D 是学出来的先验: 输出带四元数的完整朝向, 和"含被遮挡部分"的完整尺寸。
它能不能在我们这种连续遮挡的素材上兑现, 正是要测的东西。

★ 输出格式和 boxes_from_masks.py 完全一致 (06_all_boxes.json), 下游一律不用改。
★ 单独一个文件 + 单独一个 conda 环境: WildDet3D 走 vis4d, 钉 pydantic<2 和 numpy<2,
  和 sam3d 环境装一起会打架。boxes_from_masks.py 一行都不要动, 两条随时能切回去。

模型输出的 boxes3d 是 (N,10) = (cx,cy,cz, w,h,l, qw,qx,qy,qz), 相机系, 米。
文档写的字母顺序 (w,h,l) 对不上实测, 实际的配对关系是量出来的, 不是猜的:
  三边:  索引0 = 宽, 索引1 = 长, 索引2 = 高
  旋转:  第0列 = 长轴, 第1列 = 竖直轴, 第2列 = 宽轴
  即 第0列<->索引1, 第1列<->索引2, 第2列<->索引0 (一个固定的置换)
怎么定下来的: 拿 16 辆车 (taxi/car/van/truck) 让 WildDet3D 出框, 再和现行 PCA 量出的
长轴比 |cos| —— 第0列 0.992, 第1列 0.624, 第2列 0.124, 干净利落; 竖直轴 16/16 都是第1列。
交叉验证: 路灯 [0.31 0.46 3.99] 和垃圾桶 [0.48 0.52 0.82] 的索引2 正好是它们的高。
(四元数顺序 (w,x,y,z) / (x,y,z,w) 换着试结果完全一样, 所以顺序不是问题, 置换才是。)

------------------------------------------------------------------ 这版的局限 (设计上的)
本脚本走 geometric 模式: 吃的是 SAM3 mask 转出来的 2D 框, 所以它只看得见 SAM3
已经找到的东西, SAM3 漏掉的它连看都看不到。而当初看中 WildDet3D 的理由之一恰恰是召回
(论文 Anc.R 0.8811 vs Lumera-Box 0.5607), 这个接法把召回能力整个屏蔽了。

完整设计应该是 WildDet3D 和 SAM3 并行、各出各的, 再合并:
    帧 -> Qwen 列词 -> SAM3 跟踪      -> mask + track ID
                    -> WildDet3D 按词检测 (input_texts 模式, 吃原图不吃框) -> 3D 框
    合并: 两边都有的用 SAM3 的 mask; 只有框没 mask 的, 把框投回图让 SAM 补抠 (捡回漏检)

分两步做, 顺序不能反: 先用现在这版回答"它的框到底准不准"(尺寸/朝向/按遮挡分档),
确认比 PCA 好之后再加召回那条支路。框本身要是歪的, 检出更多只会让场景更糟。
另外第二步还得处理跨帧关联 —— 它单帧出框, 没有 track ID, 工作量不小。
"""
import sys, os, json, glob, numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_frame import load_up, export_T, inverse_T, usable_frames
from pycocotools import mask as mask_util

R1   = os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
FR   = os.environ.get("FR","/home/jincheng/sam3d/data/outdoor/dataset/timesquare/frames")
REPO = os.environ.get("WD3","/home/jincheng/sam3d/WildDet3D")
CKPT = os.environ.get("WD3_CKPT", f"{REPO}/ckpt/wilddet3d_alldata_all_prompt_v1.0.pt")
OUT  = os.environ.get("WD_OUT", f"{R1}/06_all_boxes_wilddet.json")
NFRAME  = int(os.environ.get("WD_FRAMES","5"))    # 每个实例取面积最大的几帧, 取中位数
MIN_PIX = int(os.environ.get("WD_MINPIX","400"))  # 2D 框太小的直接不问, 问了也是噪声
USE_DEPTH = os.environ.get("WD_DEPTH","1") == "1" # 喂 DA3 深度钉尺度
MK, DA3 = f"{R1}/03_masks", f"{R1}/02_da3"
# "天"的方向。老的写法是写死 [0,-1,0] = 首帧相机的 -y —— 街头平视时相机水平, 这个假设
# 成立(时代广场实测只偏 0.4°); 但高空俯拍时相机低着头, 它的 -y 指向斜上后方, beach 偏
# 26.2°、carpark 偏 29.2°, 整个场景连同地面一起歪掉。
# 现在改成读 world_up.py 从点云主平面拟合出来的真实法向, 没有那个文件才退回老假设。
# ★ 必须在"量盒子"之前定好: 盒子的高度是沿"天"量出来的, 方向错了盒子本身就是错的,
#   事后旋转只是把错的盒子摆正, 数字不会变对。
UP, _GH, _FB = load_up(DA3, "WD量盒子")

cams = json.load(open(f"{DA3}/cameras.json"))["cameras"]
HAVE = usable_frames(DA3, "WD量盒子")          # 跳过位姿不可信的帧

import sys; sys.path.insert(0, REPO)
from wilddet3d import build_model, preprocess

model = build_model(checkpoint=CKPT, score_threshold=0.3, skip_pretrained=True,
                    use_depth_input_test=USE_DEPTH)

def quat_to_mat(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

def lift(mask, fi):
    """一帧里把 mask 的 2D 框抬成世界系的 3D 框。返回 (中心, 长轴, 尺寸) 或 None。"""
    ys,xs = np.nonzero(mask)
    if len(xs) == 0: return None
    x0,y0,x1,y1 = xs.min(), ys.min(), xs.max(), ys.max()
    if (x1-x0)*(y1-y0) < MIN_PIX: return None
    img = cv2.cvtColor(cv2.imread(f"{FR}/{fi:04d}.jpg"), cv2.COLOR_BGR2RGB).astype(np.float32)
    H,W = img.shape[:2]
    cam = cams[f"{fi:04d}.jpg"]
    # DA3 的内参是在 504x280 上算的, 要缩放到原图分辨率
    K = np.array(cam["intrinsic_3x3"], dtype=np.float64).copy()
    sx, sy = W/(K[0,2]*2), H/(K[1,2]*2)
    K[0,:] *= sx; K[1,:] *= sy
    kw = {}
    if USE_DEPTH:
        d = np.load(f"{DA3}/depth/{fi:04d}_depth.npy").astype(np.float32)
        kw["depth"] = cv2.resize(d, (W,H), interpolation=cv2.INTER_LINEAR)
    data = preprocess(img, K, **kw)
    call = dict(images=data["images"].cuda(), intrinsics=data["intrinsics"].cuda()[None],
                input_hw=[data["input_hw"]], original_hw=[data["original_hw"]],
                padding=[data["padding"]],
                # 每张图一个平铺的 [x1,y1,x2,y2] 原图像素框, 不是框的张量
                input_boxes=[[float(x0), float(y0), float(x1), float(y1)]],
                prompt_text="geometric")          # 一对一: 给一个 2D 框, 抬成一个 3D 框
    if USE_DEPTH: call["depth_gt"] = data["depth_gt"].cuda()
    with torch.no_grad():
        _, boxes3d, *_ = model(**call)
    b = boxes3d[0]
    if b is None or len(b) == 0: return None
    b = b[0].detach().cpu().numpy().astype(np.float64)
    ctr_c, dims, quat = b[:3], b[3:6], b[6:10]
    wid, lng, hgt = dims[0], dims[1], dims[2]          # 索引0宽 索引1长 索引2高 (见文件头)
    Rc = quat_to_mat(quat)
    c2w = np.linalg.inv(np.array(cam["extrinsic_w2c_4x4"]).reshape(4,4))
    ctr = c2w[:3,:3] @ ctr_c + c2w[:3,3]
    Rw  = c2w[:3,:3] @ Rc
    a1 = Rw[:,0] - UP*(Rw[:,0]@UP)                     # 第0列 = 长轴, 投到水平面
    n = np.linalg.norm(a1)
    if n < 1e-6: return None
    a1 /= n
    # 长轴带正反 —— 这正是 PCA 那条给不了的东西, 别在这里丢掉
    return ctr, a1, np.array([lng, wid, hgt])          # 尺寸[2] 必须是高, 下游按它对齐尺度

kept = {}; n_ok = n_fail = 0
for fn in sorted(os.listdir(MK)):
    z = np.load(f"{MK}/{fn}", allow_pickle=True)
    meta, counts, (H,W), word = z["meta"], z["counts"], z["hw"], str(z["word"])
    if len(meta) == 0 or meta["area"].max() == 0: continue
    live = {}
    for i,(oid,fi,a,e) in enumerate(meta):
        if a > 0 and int(fi) in HAVE: live.setdefault(int(oid), []).append((int(a), int(fi), i, float(e)))
    out = []
    for oid, rows in live.items():
        rows.sort(key=lambda t: -t[0]*(1-min(t[3]*4, 0.9)))     # 面积大且不贴边优先
        got = []
        for a, fi, i, e in rows[:NFRAME]:
            m = mask_util.decode({"counts": counts[i], "size":[int(H),int(W)]}).astype(bool)
            try: r = lift(m, fi)
            except Exception as ex: print(f"    [{word}#{oid}] 第{fi}帧 出错 {ex}"); r = None
            if r is not None: got.append(r)
        if not got: n_fail += 1; continue
        # 跨帧融合: 中心和尺寸取中位数。
        # 长轴不能直接平均 —— 各帧要是差 180°, 平均出来是零向量。做法是按"轴"聚成一致,
        # 再用多数票定正反, 同时把分歧率记下来: 这个数就是模型自己对车头朝哪的把握,
        # 也正是 PCA 那条完全给不出的东西。
        ctr = np.median([g[0] for g in got], 0)
        ext = np.median([g[2] for g in got], 0)
        A = np.array([g[1] for g in got])
        sgn = np.sign(A @ A[0]); sgn[sgn == 0] = 1
        a1 = (A * sgn[:,None]).mean(0)                     # 先对齐成同一条轴
        a1 /= max(np.linalg.norm(a1), 1e-9)
        if (sgn > 0).sum() * 2 < len(sgn): a1 = -a1        # 少数派才是第0帧那侧, 翻回来
        flip = float(min((sgn > 0).mean(), (sgn < 0).mean()))   # 0 = 各帧一致, 0.5 = 完全对半
        out.append(dict(中心=ctr.tolist(), 长轴=a1.tolist(), 尺寸=ext.tolist(),
                        底高=float(ctr@UP - ext[2]/2), obj_id=oid,
                        点数=len(got), 用帧=len(got), 总帧=len(rows),
                        朝向分歧=round(flip,3), 来源="wilddet3d"))
        n_ok += 1
    if out:
        kept[word] = out
        print(f"[{word}] {len(out)}/{len(live)} 个实例出框", flush=True)

json.dump(kept, open(OUT,"w"), ensure_ascii=False, indent=1)
print(f"\n[done] {n_ok} 个实例出框, {n_fail} 个失败 -> {OUT}")
