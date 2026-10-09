#!/usr/bin/env python
"""导出动作文件(流水线的 motion 步): 把摆人那步修好的人(关节参数)和合成那步出的场景导成 <R1>/12_motion/:
  <名字>_scene_people.fbx          场景 + 人(人带骨骼和蒙皮, 关节动画) + 相机(每帧的位姿和焦距) —— 主要交付物
  <名字>_scene.fbx                 只有场景
  <名字>_scene_people.blend        Blender 打开就能播(给人看的骨架: 骨头连到下一个关节; 动作和 FBX 一样); 小键盘 0 = 原视频的视角
  <名字>_motion_params.npz         每人每帧的 MHR 参数 + 相机序列(同事要转成自己的骨架时用)
  <名字>_blender.mp4               加 --video 才出: 在 Blender 里从原视频的视角渲的"场景 + 人", 和原视频一帧对一帧
  fbx_per_person/  glb_per_person/  motion.npz  <名字>_preview.png
<名字> = 场景名(R1 的上一级目录名)。场景用 09_scene_nopeople.glb(合成那步出的: 物理之后、摘掉了静态人形)。

★ 这一步**不修人**(2026-10-09 起): 人是摆人那步修的(body_motion.py), 这里读它存下来的参数(people_store.load_params,
  坐着的人已经带上"椅子被物理挪了"的位移), 只做坐标系转换 + 带上相机序列。网页验收视频里的人读的是同一份参数生成的网格 ——
  同事在网页上看过的人, 就是 FBX 里的人。

三段各用各的环境, 这里只管串起来:
  这里(sam3d)写 motion.npz  ->  tools/mocap/from_motion.py(sam3d, GPU)  ->  tools/mocap/to_blender.py(~/venvs/bpy, 带自检)
  [--video]  ->  tools/mocap/render_video.py(它自己起 ~/venvs/bpy 的进程; EEVEE 用的显卡不归 CUDA_VISIBLE_DEVICES 管, 见那个文件)
设计和验证: docs/superpowers/specs/2026-10-08-people-firsthand-motion-design.md

用法: python motion_export.py --r1 <场景结果目录> --body <body4d 轮次目录> [--video]
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/sam3d/data/outdoor/scripts"))
import people_identity, people_store
MOCAP = os.path.join(os.path.dirname(HERE), "tools", "mocap")
BPY = os.path.expanduser("~/venvs/bpy/bin/python")
SCENE = "09_scene_nopeople.glb"
OUT = "12_motion"


def build_motion(r1, body):
    """motion.npz 的内容: 每人的参数转到 glTF 世界系(天 +y, 和场景的 glb 同一个系) + 体型 + 视频帧号, 外加相机序列。
    参数来自摆人那步(people_store), 这里一个数都不改, 只换坐标系。"""
    import body_into_scene as B
    import mhr_body as M
    import cam_track
    from world_frame import export_T
    people_store.check_fresh(r1, people_identity.source_of(body))
    people, fps = people_store.load_params(r1)
    da3, cams, UP, GH, vid2key = B.load_scene(r1)
    fj = json.load(open(f"{body}/00_frames/frames.json"))
    FL = export_T(UP)
    mo = dict(fps=np.float32(fps))
    mo.update(cam_track.track(cams, vid2key, FL), video_wh=np.array(fj["out_wh"], np.int32))   # 相机序列: 和人同一个世界系、同一套帧号
    for pid, r in people.items():
        mo[f"p{pid}_params"] = M.transform(r["params"], FL).astype(np.float32)
        mo[f"p{pid}_shape"] = r["shape"].astype(np.float32); mo[f"p{pid}_vf"] = r["vf"].astype(np.int32)
        mo[f"p{pid}_conf"] = r["conf"].astype(np.float32)
    print(f"    人 {list(people)}: 参数读自 {r1}/{people_store.PARAMS}(摆人那步修好的), 相机 {len(mo['cam_vf'])} 帧")
    return mo


def run(cmd):
    print("    $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r1", required=True); ap.add_argument("--body", required=True)
    ap.add_argument("--video", action="store_true", help="再用 Blender 从原视频的视角渲一段视频(每帧约 0.3 秒)")
    a = ap.parse_args()
    r1 = os.path.abspath(a.r1); name = os.path.basename(os.path.dirname(r1)); out = f"{r1}/{OUT}"
    if not os.path.exists(f"{r1}/{SCENE}"):
        raise SystemExit(f"[失败] 缺 {r1}/{SCENE} —— 先跑 people_glb(合成)")
    mo = build_motion(r1, a.body)                        # 先读: 摆人的结果过期 / 缺文件在这里就报错, 不先把上一次的导出删了
    shutil.rmtree(out, ignore_errors=True)               # 整个目录都是这一步的产物: 人删了/改了, 不留上一次的文件
    os.makedirs(out)
    np.savez_compressed(f"{out}/motion.npz", **mo)
    run([sys.executable, f"{MOCAP}/from_motion.py", f"{out}/motion.npz", out, name])
    glbs = sorted(os.listdir(f"{out}/glb_per_person"), key=lambda f: int(f.rsplit("_P", 1)[1].split(".")[0]))
    run([BPY, "-u", f"{MOCAP}/to_blender.py", ",".join(f"{out}/glb_per_person/{f}" for f in glbs),
         f"{r1}/{SCENE}", f"{out}/_lbs_target.npz", out, name])
    os.remove(f"{out}/_lbs_target.npz")                  # 只给自检用的逐帧网格, 几百 MB, 不留
    if a.video:
        run([sys.executable, f"{MOCAP}/render_video.py", f"{out}/{name}_scene_people.blend", f"{out}/{name}_blender.mp4"])
    print(f"[done] {out}/{name}_scene_people.fbx  {name}_scene.fbx  (+ .blend, 原始参数, 每人一个 FBX" + (", Blender 渲的视频)" if a.video else ")"))


if __name__ == "__main__":
    main()
