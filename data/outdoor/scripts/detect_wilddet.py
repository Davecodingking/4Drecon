#!/usr/bin/env python
"""召回支路第 1 步: WildDet3D 直接在原图上按词表检测, 不吃 SAM3 的 mask。

为什么要这一步 —— geometric 那条(boxes_from_wilddet.py)吃的是 SAM3 抠出来的框,
所以 SAM3 漏掉的东西它连看都看不到。而当初看中 WildDet3D 的理由之一正是召回。
这里走 input_texts 模式: 输入原始 RGB 帧 + Qwen 那 29 个词, 它自己找。

输出的是"每帧各自的检测", 还没有跨帧身份 —— 它是单帧模型, 没有 track ID。
把检测反投影到世界系再聚类成实例, 是下一步 merge_recall.py 的事。
"""
import sys, os, json, numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_frame import load_up, export_T, inverse_T
R1   = os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
FR   = os.environ.get("FR","/home/jincheng/sam3d/data/outdoor/dataset/timesquare/frames")
REPO = os.environ.get("WD3","/home/jincheng/sam3d/WildDet3D")
CKPT = os.environ.get("WD3_CKPT", f"{REPO}/ckpt/wilddet3d_alldata_all_prompt_v1.0.pt")
OUT  = os.environ.get("DET_OUT", f"{R1}/12_wd_detections.json")
STRIDE = int(os.environ.get("DET_STRIDE","5"))     # 每隔几帧检一次, 180 帧不用全检
SCORE  = float(os.environ.get("DET_SCORE","0.35"))
LIMIT  = int(os.environ.get("DET_LIMIT","0"))      # >0 时只检这么多帧, 试跑用
DA3  = f"{R1}/02_da3"
# ★ 天方向以前写死 [0,-1,0], 而依赖表里却声明这一步依赖 up —— 依赖表在说假话。
#   后果: 俯拍场景真实的天偏 26~29°, 这里算出来的朝向/高度系统性歪掉。
UP, _GH, _FB   = load_up(DA3, "WD检测")

VOCAB = json.load(open(f"{R1}/01_vlm_vocab.json"))["vocab"]
cams  = json.load(open(f"{DA3}/cameras.json"))["cameras"]
FRAMES = sorted(int(f[:4]) for f in os.listdir(f"{DA3}/depth") if f.endswith("_depth.npy"))
FRAMES = FRAMES[::STRIDE]
if LIMIT: FRAMES = FRAMES[:LIMIT]

import sys; sys.path.insert(0, REPO)
from wilddet3d import build_model, preprocess
model = build_model(checkpoint=CKPT, score_threshold=SCORE, skip_pretrained=True,
                    use_depth_input_test=True)

def quat_to_mat(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                     [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

dets = []
for n, fi in enumerate(FRAMES, 1):
    img = cv2.cvtColor(cv2.imread(f"{FR}/{fi:04d}.jpg"), cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w = img.shape[:2]
    cam = cams[f"{fi:04d}.jpg"]
    K = np.array(cam["intrinsic_3x3"]).copy()
    K[0,:] *= w/(K[0,2]*2); K[1,:] *= h/(K[1,2]*2)
    d = cv2.resize(np.load(f"{DA3}/depth/{fi:04d}_depth.npy").astype(np.float32), (w,h))
    data = preprocess(img, K, depth=d)
    with torch.no_grad():
        b2d, b3d, sc, _, sc3, cls, _ = model(
            images=data["images"].cuda(), intrinsics=data["intrinsics"].cuda()[None],
            input_hw=[data["input_hw"]], original_hw=[data["original_hw"]],
            padding=[data["padding"]], depth_gt=data["depth_gt"].cuda(),
            input_texts=VOCAB)
    B2, B3 = b2d[0].cpu().numpy(), b3d[0].cpu().numpy()
    S, C = sc[0].cpu().numpy(), cls[0].cpu().numpy()
    c2w = np.linalg.inv(np.array(cam["extrinsic_w2c_4x4"]).reshape(4,4))
    for j in range(len(B3)):
        dims = B3[j][3:6]                                  # 索引0宽 1长 2高
        Rw   = c2w[:3,:3] @ quat_to_mat(B3[j][6:10])
        ctr  = c2w[:3,:3] @ B3[j][:3] + c2w[:3,3]
        a1   = Rw[:,0] - UP*(Rw[:,0]@UP)                   # 第0列 = 长轴
        nn   = np.linalg.norm(a1)
        if nn < 1e-6: continue
        dets.append(dict(词=VOCAB[int(C[j])], 帧=int(fi), 分=float(S[j]),
                         中心=(ctr).tolist(), 长轴=(a1/nn).tolist(),
                         尺寸=[float(dims[1]), float(dims[0]), float(dims[2])],
                         框2d=[float(v) for v in B2[j]]))
    print(f"[{n}/{len(FRAMES)}] 第{fi}帧 检出 {len(B3)}", flush=True)

json.dump(dict(帧数=len(FRAMES), 步长=STRIDE, 检测=dets), open(OUT,"w"),
          ensure_ascii=False, indent=1)
from collections import Counter
print(f"\n[done] {len(FRAMES)} 帧共 {len(dets)} 个检测 -> {OUT}")
print("按词:", dict(Counter(d['词'] for d in dets).most_common()))
