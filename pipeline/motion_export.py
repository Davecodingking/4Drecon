#!/usr/bin/env python
"""导出动作文件(流水线的 motion 步): 在模型的一手数据(关节参数)上做修复, 出 <R1>/12_motion/:
  <名字>_scene_people.fbx          场景 + 人(人带骨骼和蒙皮, 关节动画) + 相机(每帧的位姿和焦距) —— 主要交付物
  <名字>_scene.fbx                 只有场景
  <名字>_scene_people.blend        Blender 打开就能播(给人看的骨架: 骨头连到下一个关节; 动作和 FBX 一样); 小键盘 0 = 原视频的视角
  <名字>_motion_params.npz         每人每帧的 MHR 参数 + 相机序列(同事要转成自己的骨架时用)
  <名字>_blender.mp4               加 --video 才出: 在 Blender 里从原视频的视角渲的"场景 + 人", 和原视频一帧对一帧
  fbx_per_person/  glb_per_person/  motion.npz  report.json  <名字>_preview.png
<名字> = 场景名(R1 的上一级目录名)。场景用 09_scene_nopeople.glb(合成那步出的: 物理之后、摘掉了静态人形)。

三段各用各的环境, 这里只管串起来:
  body_motion.py(sam3d, GPU)  ->  tools/mocap/from_motion.py(sam3d, GPU)  ->  tools/mocap/to_blender.py(~/venvs/bpy, 带自检)
  [--video]  ->  tools/mocap/render_video.py(它自己起 ~/venvs/bpy 的进程; EEVEE 用的显卡不归 CUDA_VISIBLE_DEVICES 管, 见那个文件)
设计和验证: docs/superpowers/specs/2026-10-08-people-firsthand-motion-design.md

用法: python motion_export.py --r1 <场景结果目录> --body <body4d 轮次目录> [--video]
"""
import argparse, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import people_identity
MOCAP = os.path.join(os.path.dirname(HERE), "tools", "mocap")
BPY = os.path.expanduser("~/venvs/bpy/bin/python")
SCENE = "09_scene_nopeople.glb"
OUT = "12_motion"


def mesh_run(body):
    """人体数据的目录: 和摆人、合成读的是同一份(核对过身份的, 见 people_identity.py), 另外要求存了一手数据。"""
    run = people_identity.mesh_run_of(body)
    if not os.path.isdir(f"{run}/mhr_4d_individual"):
        raise SystemExit(f"[失败] {run} 没有模型的一手数据(mhr_4d_individual; 2026-10-08 之前出的网格没有) —— 重跑 people_mesh")
    return run


def run(cmd):
    print("    $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r1", required=True); ap.add_argument("--body", required=True)
    ap.add_argument("--video", action="store_true", help="再用 Blender 从原视频的视角渲一段视频(每帧约 0.3 秒)")
    a = ap.parse_args()
    r1 = os.path.abspath(a.r1); name = os.path.basename(os.path.dirname(r1)); out = f"{r1}/{OUT}"
    mr = mesh_run(a.body)
    if not os.path.exists(f"{r1}/{SCENE}"):
        raise SystemExit(f"[失败] 缺 {r1}/{SCENE} —— 先跑 people_glb(合成)")
    shutil.rmtree(out, ignore_errors=True)               # 整个目录都是这一步的产物: 人删了/改了, 不留上一次的文件
    run([sys.executable, f"{HERE}/body_motion.py", "--r1", r1, "--body", a.body, "--mesh-run", mr, "--out", out])
    run([sys.executable, f"{MOCAP}/from_motion.py", f"{out}/motion.npz", out, name])
    glbs = sorted(os.listdir(f"{out}/glb_per_person"), key=lambda f: int(f.rsplit("_P", 1)[1].split(".")[0]))
    run([BPY, "-u", f"{MOCAP}/to_blender.py", ",".join(f"{out}/glb_per_person/{f}" for f in glbs),
         f"{r1}/{SCENE}", f"{out}/_lbs_target.npz", out, name])
    for f in ("_lbs_target.npz", "people_world.npz"):    # 只给自检 / 验收视频用的逐帧网格, 几百 MB, 不留
        os.remove(f"{out}/{f}")
    if a.video:
        run([sys.executable, f"{MOCAP}/render_video.py", f"{out}/{name}_scene_people.blend", f"{out}/{name}_blender.mp4"])
    print(f"[done] {out}/{name}_scene_people.fbx  {name}_scene.fbx  (+ .blend, 原始参数, 每人一个 FBX" + (", Blender 渲的视频)" if a.video else ")"))


if __name__ == "__main__":
    main()
