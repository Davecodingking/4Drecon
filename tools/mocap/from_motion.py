#!/usr/bin/env python
"""一手数据版的动作导出: motion.npz(摆人那步修好的参数, motion_export.py 换到 glTF 世界系写的) -> 每人一个带骨骼的 glb。**没有拟合这一步**:
参数喂给 MHR, 骨架每一帧的状态直接就有。之后照旧用 to_blender.py 转 FBX / .blend。
(对照: fit_mhr.py 是从网格反推, 每人 7 分钟, 关节角度不保证和模型一样 —— 留作对照, 不进正式流程。)

用法: python from_motion.py <motion.npz> <输出目录> <名字>
产物: <输出目录>/glb_per_person/<名字>_P<人>.glb   每人一个 glb(骨架 + 蒙皮网格 + 关节动画)
      <输出目录>/<名字>_motion_params.npz           给同事转自己骨架用的原始数据: 每人每帧 204 维参数 + 体型 + 关节名/参数名
      <输出目录>/_lbs_target.npz                    只做线性蒙皮的网格(to_blender.py 自检用: Blender 里播出来应该和它一样)
sam3d 环境。有 GPU 就用。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mhr_body as M
import export_glb

src, outd, name = sys.argv[1:4]
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(f"{outd}/glb_per_person", exist_ok=True)
z = np.load(src)
sk = M.skeleton(dev)
faces = M.model(dev).character_torch.mesh.faces.cpu().numpy()
pnames = np.array(list(M.model(dev).character_torch.parameter_transform.parameter_names)[:M.NPOSE + M.NSCALE])
CAM = {k: z[k] for k in ("cam_vf", "cam_c2w", "cam_K", "cam_hw", "video_wh")}      # 相机序列原样带下去: to_blender.py 建相机用
tgt = dict(fps=z["fps"], **CAM)
raw = dict(fps=z["fps"], joint_names=sk["names"], joint_parents=sk["parents"], param_names=pnames,
           说明=np.array("p<人>_params (T,204): MHR 参数, 世界系(glTF: 天 +y, 米)。[0:3] 根平移(1 单位 = 10 cm) [3:6] 根旋转(xyz 外旋欧拉角, 弧度, "
                        "绕根关节) [6:136] 各关节 [136:204] 骨骼缩放。p<人>_shape (45): 体型。p<人>_vf: 视频帧号。"
                        "网格/骨架 = MHR(shape, params), 见 pipeline/mhr_body.py。"
                        "相机: cam_c2w (N,4,4) 相机 -> 世界(同一个世界系; 相机轴是 OpenCV 的: x 右 y 下 z 朝前), cam_K (N,3,3) 内参"
                        "(像素是 cam_hw = (高, 宽) 那张图的), cam_vf 视频帧号, video_wh 原视频的宽高"), **CAM)
for k in sorted((k for k in z.files if k.endswith("_params")), key=lambda x: int(x[1:-7])):
    pid = k[1:-7]
    P, shape, vf = z[k].astype(np.float64), z[f"p{pid}_shape"].astype(np.float64), z[f"p{pid}_vf"]
    V, skel = M.forward(shape, P, False, dev)                    # 线性蒙皮: glTF / FBX 里播出来的就是它
    Vc, _ = M.forward(shape, P, True, dev)                       # 带姿势修正项: 验收视频里的
    d = np.linalg.norm(V - Vc, axis=2) * 100
    f = f"{outd}/_p{pid}.npz"
    np.savez(f, vf=vf, skel=skel, rest=M.rest_verts(shape, dev), faces=faces, fps=z["fps"],
             err_mean=d.mean(1), err_max=d.max(1), err_label=np.array("和验收视频里的网格差(姿势修正项, 格式放不了)"), **sk)
    os.replace(f, f"{outd}/p{pid}.npz")
    export_glb.build([f"{outd}/p{pid}.npz"], f"{outd}/glb_per_person/{name}_P{pid}.glb")
    os.remove(f"{outd}/p{pid}.npz")
    tgt[f"p{pid}_V"] = V.astype(np.float32); tgt[f"p{pid}_vf"] = vf
    raw[f"p{pid}_params"] = P.astype(np.float32); raw[f"p{pid}_shape"] = shape.astype(np.float32); raw[f"p{pid}_vf"] = vf
np.savez_compressed(f"{outd}/_lbs_target.npz", **tgt)
np.savez_compressed(f"{outd}/{name}_motion_params.npz", **raw)
print(f"-> {outd}/glb_per_person/  {outd}/{name}_motion_params.npz")
