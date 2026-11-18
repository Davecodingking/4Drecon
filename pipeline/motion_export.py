#!/usr/bin/env python
"""导出动作文件(流水线的 motion 步): 把摆人那步修好的人(关节参数)和合成那步出的场景导成 <R1>/12_motion/:
  <名字>_scene_people.fbx          场景 + 人(人带骨骼和蒙皮, 关节动画) + 相机(每帧的位姿和焦距) —— 主要交付物
                                   + 被人拿着的物体(闸 2 标了 held 的, carry 步求的每帧位姿; 物体名 carried_<词>_<实例号>)
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
import carry_decisions
import gravity_from_poles
MOCAP = os.path.join(os.path.dirname(HERE), "tools", "mocap")
BPY = os.path.expanduser("~/venvs/bpy/bin/python")
SCENE = "09_scene_nopeople.glb"
OUT = "12_motion"
CARRY = "13_carry"                      # carry 步的结果: 被人拿着的物体每帧的位姿
G2B = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])   # glTF 世界(天 +y) -> Blender 世界(天 +z)


def build_motion(r1, body):
    """motion.npz 的内容: 每人的参数转到 glTF 世界系(天 +y, 和场景的 glb 同一个系) + 体型 + 视频帧号, 外加相机序列。
    参数来自摆人那步(people_store), 这里一个数都不改, 只换坐标系。"""
    import body_into_scene as B
    import mhr_body as M
    import cam_track
    from world_frame import export_T
    people_store.check_fresh(r1, people_identity.source_of(body))
    people, fps = people_store.load_params(r1)
    # 挂手版(carry 步 CARRY_INHAND=1 时)采用了、闸 5 上没点不要的物体: 拿着它的那条手臂的修正叠加到参数上。只进导出, 摆人的文件不动
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "carryfit", "inhand"))
    import inhand_overlay
    for note in inhand_overlay.apply(r1, people, set(carry_decisions.want(r1))):
        print("    " + note)
    fj = json.load(open(f"{body}/00_frames/frames.json"))
    da3, cams, UP, GH, vid2key = B.load_scene(r1, dense=bool(fj.get("dense")))   # 人体全帧时相机也逐帧(dense_cams.py)
    # 斜坡(06_gravity.json 启用时): 链上一直在地面系(地面法线 +y), 交付前整体转到真竖直朝上 —— 人、相机、场景同一个转法
    FL = gravity_from_poles.level_T(r1, UP) @ export_T(UP)
    mo = dict(fps=np.float32(fps))
    mo.update(cam_track.track(cams, vid2key, FL), video_wh=np.array(fj["out_wh"], np.int32))   # 相机序列: 和人同一个世界系、同一套帧号
    for pid, r in people.items():
        mo[f"p{pid}_params"] = M.transform(r["params"], FL).astype(np.float32)
        mo[f"p{pid}_shape"] = r["shape"].astype(np.float32); mo[f"p{pid}_vf"] = r["vf"].astype(np.int32)
        mo[f"p{pid}_conf"] = r["conf"].astype(np.float32)
    print(f"    人 {list(people)}: 参数读自 {r1}/{people_store.PARAMS}(摆人那步修好的), 相机 {len(mo['cam_vf'])} 帧")
    return mo


def carry_bundle(r1, mo, out):
    """被人拿着的物体(carry 步求出来的、闸 5 上没点"不要"的, 见 carry_decisions.py) -> <out>/_carry.npz, to_blender.py 读它
    一起导进场景+人的 FBX 和 .blend。没有就不写。返回导了哪几个(生成名)。
    物体每帧的位姿在 DA3 世界里(carry_poses.npz: 局部系 -> DA3 世界 M_k, glb 坐标 -> 局部系 A), 换到导出的世界:
      T = 导出的相机 c2w · DA3 的 w2c   (同一帧的两套相机, 每帧都该一样; 斜坡时整体转的那下也在里面)
      整体 = G2B · T · M_k · A · G2B⁻¹  (Blender 导入 glb 时自己做了 G2B, 同 carryfit/to_blend.py)
    ★ M_k = [R_k·diag(s) | t_k]: 拟合给物体的是沿它自己三根轴各一个缩放(滑板车 0.89/0.81/0.82)。连着旋转一起塞进关键帧,
      Blender 拆成 位置/四元数/缩放 时会丢掉斜切(stepup 实测差 0.5-0.7 cm)。所以拆成两半:
        B   = G2B · diag(s) · A · G2B⁻¹          不随时间变, 导入时直接烘进网格顶点
        W_k = G2B · T · [R_k | t_k] · G2B⁻¹      每帧的刚体位姿, 打关键帧(拆得开, 没有误差)"""
    names = set(carry_decisions.want(r1))                 # 求出来了、闸 5 上没点"不要"的
    rows = [r for r in carry_decisions.rows(r1) if r["生成名"] in names]
    no = sorted(carry_decisions.dropped(r1))
    if no:
        print(f"    被拿着的物体: 闸 5 点了不要, 不进 FBX: {', '.join(no)}")
    if not rows:
        return []
    cams = json.load(open(f"{r1}/02_da3/cameras.json"))["cameras"]
    row = {int(v): i for i, v in enumerate(mo["cam_vf"])}
    pack = dict(n=np.int32(len(rows)))
    for j, r in enumerate(rows):
        P = np.load(f"{r1}/{r['目录']}/carry_poses.npz", allow_pickle=True)
        keys, vf, M, A = list(P["keys"]), P["vf"], P["local_to_world"], P["mesh_to_local"]
        Ts = np.array([mo["cam_c2w"][row[int(v)]] @ np.array(cams[k]["extrinsic_w2c_4x4"], float).reshape(4, 4)
                       for k, v in zip(keys, vf)])
        dev = float(np.abs(Ts - Ts[0]).max())
        if dev > 1e-5:
            raise SystemExit(f"[失败] {r['词']}#{r['实例']}: DA3 -> 导出世界的变换每帧不一样(差 {dev:.2e}), 物体会和人错开")
        s = np.linalg.norm(M[:, :3, :3], axis=1)                 # 每帧每根轴的缩放 = 线性部分每一列的长度
        if np.abs(s - s[0]).max() > 1e-5:
            raise SystemExit(f"[失败] {r['词']}#{r['实例']}: 物体的缩放随时间变了, 拆不成 形状 + 刚体位姿")
        Rt = M.copy(); Rt[:, :3, :3] = M[:, :3, :3] / s[0]
        W = np.array([G2B @ Ts[0] @ Rt[i] @ np.linalg.inv(G2B) for i in range(len(vf))])
        orth = float(np.abs(np.einsum("kji,kjl->kil", W[:, :3, :3], W[:, :3, :3]) - np.eye(3)).max())
        if orth > 1e-5:
            raise SystemExit(f"[失败] {r['词']}#{r['实例']}: 每帧位姿不是刚体(差 {orth:.1e}), 导出的世界和 DA3 之间有缩放?")
        pack[f"o{j}_name"] = f"carried_{r['词'].replace(' ', '_')}_{r['实例']}"
        pack[f"o{j}_mesh"] = str(P["mesh"])
        pack[f"o{j}_vf"] = np.asarray(vf, np.int32)
        pack[f"o{j}_B"] = G2B @ np.diag([*s[0], 1.0]) @ A @ np.linalg.inv(G2B)
        pack[f"o{j}_W"] = W
    np.savez(f"{out}/_carry.npz", **pack)
    print(f"    被人拿着的物体 {len(rows)} 个(读自 {r1}/{CARRY}): " + ", ".join(str(pack[f'o{j}_name']) for j in range(len(rows))))
    return [r["生成名"] for r in rows]


def B_up(r1):
    from world_frame import load_up
    return load_up(f"{r1}/02_da3", "导出")[0]


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
    carried = carry_bundle(r1, mo, out)                   # 被人拿着的物体: to_blender.py 看到 _carry.npz 就一起导
    run([sys.executable, f"{MOCAP}/from_motion.py", f"{out}/motion.npz", out, name])
    glbs = sorted(os.listdir(f"{out}/glb_per_person"), key=lambda f: int(f.rsplit("_P", 1)[1].split(".")[0]))
    scene = f"{r1}/{SCENE}"
    Tl = gravity_from_poles.level_T(r1, B_up(r1))
    if not np.allclose(Tl, np.eye(4)):                 # 斜坡: 场景也转到竖直朝上(套一个父节点, 网格原样)
        scene = f"{out}/_scene_level.glb"; gravity_from_poles.tilt_glb(f"{r1}/{SCENE}", scene, Tl[:3, :3])
        print(f"    斜坡: 整体转到竖杆的竖直朝上, 转了 {np.degrees(np.arccos(np.clip((np.trace(Tl[:3,:3])-1)/2,-1,1))):.2f}°")
    run([BPY, "-u", f"{MOCAP}/to_blender.py", ",".join(f"{out}/glb_per_person/{f}" for f in glbs),
         scene, f"{out}/_lbs_target.npz", out, name])
    if scene != f"{r1}/{SCENE}": os.remove(scene)
    os.remove(f"{out}/_lbs_target.npz")                  # 只给自检用的逐帧网格, 几百 MB, 不留
    if carried: os.remove(f"{out}/_carry.npz")
    carry_decisions.write_used(r1, carried)              # 导成功了才记: 驱动拿它和闸 5 的决定比, 对不上就重新导出
    if a.video:
        run([sys.executable, f"{MOCAP}/render_video.py", f"{out}/{name}_scene_people.blend", f"{out}/{name}_blender.mp4"])
    print(f"[done] {out}/{name}_scene_people.fbx  {name}_scene.fbx  (+ .blend, 原始参数, 每人一个 FBX" + (", Blender 渲的视频)" if a.video else ")"))


if __name__ == "__main__":
    main()
