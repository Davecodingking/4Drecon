"""对照用的小工具(sam3d 环境): 把摆人那步修好的人体网格(和验收视频里的一样)按 glTF 世界系(天在 +y, 米)存成 npz。
fit_mhr.py(从网格反推关节, 留作对照)吃它。正式流程不用: 导出直接读关节参数, 见 pipeline/motion_export.py。
用法: python dump_people.py <R1> <输出npz> [人id,...]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
sys.path.insert(0, os.path.expanduser("~/sam3d/data/outdoor/scripts"))
import body_into_scene as B
import people_store
from world_frame import export_T

R1, OUT = sys.argv[1], sys.argv[2]
want = sys.argv[3].split(",") if len(sys.argv) > 3 else None
BD = json.load(open(f"{R1}/09_body4d_report.json"))["人体"]
da3, cams, UP, GH, vid2key = B.load_scene(R1)
world, faces = people_store.load_world(R1)
fps = json.load(open(f"{BD}/00_frames/frames.json"))["src_fps"]
F = export_T(UP)
out = dict(fps=np.float32(fps), faces=faces.astype(np.int32))
for pid in sorted(world, key=int):
    if want and pid not in want:
        continue
    W = world[pid]
    vfs = sorted(W)
    V = np.stack([W[v] @ F[:3, :3].T + F[:3, 3] for v in vfs]).astype(np.float32)
    out[f"p{pid}_V"] = V
    out[f"p{pid}_vf"] = np.array(vfs, np.int32)
    print(f"人 {pid}: {len(vfs)} 帧, 视频帧 {vfs[0]}~{vfs[-1]}, 身高范围 {np.ptp(V[:, :, 1], axis=1).min():.2f}~{np.ptp(V[:, :, 1], axis=1).max():.2f} m")
np.savez_compressed(OUT, **out)
print("->", OUT)
