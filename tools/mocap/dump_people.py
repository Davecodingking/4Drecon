"""第一步(sam3d 环境): 把修正好的人体网格(和验收视频里的一样)按 glTF 世界系(天在 +y, 米)存成 npz。
用法: python dump_people.py <R1> <输出npz> [人id,...]"""
import io, contextlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_into_scene as B
from world_frame import export_T

R1, OUT = sys.argv[1], sys.argv[2]
want = sys.argv[3].split(",") if len(sys.argv) > 3 else None
BD = json.load(open(f"{R1}/09_body4d_report.json"))["人体"]
with contextlib.redirect_stdout(io.StringIO()):
    people, faces, kept = B.load_body(BD)
    da3, cams, UP, GH, vid2key = B.load_scene(R1)
MR = next(f"{BD}/{d}" for d in ("04_mesh_occ1", "04_mesh_occ0") if os.path.isdir(f"{BD}/{d}/mesh_4d_individual"))
roots = B.load_roots(MR)
fps = json.load(open(f"{BD}/00_frames/frames.json"))["src_fps"]
corr = json.load(open(f"{R1}/10_people.json"))["people"]
F = export_T(UP)
out = dict(fps=np.float32(fps), faces=faces.astype(np.int32))
for pid in sorted(corr, key=int):
    if want and pid not in want:
        continue
    W = B.apply_corr({pid: people[pid]}, {pid: corr[pid]}, cams, vid2key, kept, roots)[pid]
    vfs = sorted(W)
    V = np.stack([W[v] @ F[:3, :3].T + F[:3, 3] for v in vfs]).astype(np.float32)
    out[f"p{pid}_V"] = V
    out[f"p{pid}_vf"] = np.array(vfs, np.int32)
    print(f"人 {pid}: {len(vfs)} 帧, 视频帧 {vfs[0]}~{vfs[-1]}, 身高范围 {np.ptp(V[:, :, 1], axis=1).min():.2f}~{np.ptp(V[:, :, 1], axis=1).max():.2f} m")
np.savez_compressed(OUT, **out)
print("->", OUT)
