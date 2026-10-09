"""第四步(bpy 环境): 每人一个带骨骼的 glb + 场景 glb + 相机序列 -> 交付的 FBX 和 .blend。
  <名字>_scene_people.fbx          场景 + 人(人带骨骼和蒙皮, 关节动画) + 相机(每帧的位姿和焦距) —— 主要交付物
  <名字>_scene.fbx                 只有场景(不带相机: 它是给"只要一个静态场景"的人用的)
  fbx_per_person/<名字>_P<人>.fbx  每人单独一个(只要某个人的动作时用)
  <名字>_scene_people.blend        场景 + 人 + 相机, Blender 打开就能播。只给人看: 骨头从关节连到下一个关节(四棱锥),
                                   藏掉四肢上的辅助关节; 动作和 FBX 一样, 关节的本地坐标轴不一样 —— 数据用 FBX。
                                   相机设成了场景相机(小键盘 0 = 原视频的视角), 灯和渲染设置也存了(render_video.py 用它渲视频)
★ 每人单独导入: 一个 glb 里放好几个人, Blender 的 glTF 导入会把所有骨架并成一个 armature, 就拆不出每人的 FBX 了。
自检(不过就失败): glTF 导入 / FBX 导回 / 给人看的那份, 各自在 Blender 里播出来的人的网格 vs 目标网格;
FBX 导回来的场景 vs 导出前的场景(物体个数、每个物体的包围盒); FBX 导回来的相机 / .blend 里的相机 vs 流水线的相机
(位置、朝向、焦距, 以及用 Blender 自己的投影去对流水线的投影差几个像素)。最后用 Cycles 渲一张预览。
用法: python to_blender.py <人1.glb,人2.glb,...> <场景glb> <目标网格npz> <输出目录> <名字> [fps, 不给就读目标网格 npz 里的]"""
import os, sys
import numpy as np
import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "pipeline"))
from tidy_bones import tidy_bones, view_bones
import cam_track as CT

src, scene_glb, tgt_npz, outd, name = sys.argv[1:6]
z = np.load(tgt_npz)
fps = float(sys.argv[6]) if len(sys.argv) > 6 else float(z["fps"])
os.makedirs(f"{outd}/fbx_per_person", exist_ok=True)
TOL_CM = 0.1                                                          # 自检门槛: 实测都在 0.01 cm 以内, 超过就是哪里接错了
TOL_CAM_DEG, TOL_CAM_LENS = 0.01, 0.01                                # 相机朝向(度) / 焦距(%): 导出再导回只该有浮点误差
TOL_CAM_PX = 0.01                                                     # 相机投影和流水线的投影最多差画面宽的这么多(横竖焦距取平均带来的差在 0.4% 以内)
bad = []
CAM_VF = z["cam_vf"]; CAM_M = CT.blender_pose(z["cam_c2w"])           # 相机: 每个场景帧一个, 帧号和人的是同一套(原视频帧号)
CAM_LENS, CAM_SX, CAM_SY, CAM_ANISO = CT.lens(z["cam_K"], z["cam_hw"])
RES = CT.render_size(z["cam_hw"], z["video_wh"])


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps, sc.render.fps_base = round(fps), 1.0               # glTF 导入只认整数帧率(fps_base 不管), 见 export_glb
    return sc


def frame_range(sc):
    """场景的帧范围 = 人的动画和相机的关键帧合起来(相机的焦距动画挂在相机数据上, 所以两处都看)。"""
    acts = [d.animation_data.action for o in sc.objects if o.type in ("ARMATURE", "CAMERA") for d in (o, o.data)
            if d.animation_data and d.animation_data.action]
    lo = min(a.frame_range[0] for a in acts); hi = max(a.frame_range[1] for a in acts)
    sc.frame_start, sc.frame_end = int(lo), int(hi)
    return int(lo), int(hi)


srcs = src.split(",")
def import_people(bone_dir="BLENDER"):
    for f in srcs:
        bpy.ops.import_scene.gltf(filepath=f, bone_heuristic=bone_dir)

def export_fbx(path, objs, anim=True):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.ops.export_scene.fbx(filepath=path, use_selection=True, object_types={"ARMATURE", "MESH", "EMPTY", "CAMERA"},
                             add_leaf_bones=False, bake_anim=anim, bake_anim_use_all_actions=False,
                             bake_anim_use_nla_strips=False, bake_anim_simplify_factor=0.0,
                             path_mode="COPY", embed_textures=True)   # 贴图塞进 FBX: 一个文件拿走就能用
    print("->", path, f"{os.path.getsize(path) / 1e6:.1f} MB")

def add_camera(sc):
    """流水线的相机序列 -> 一个带关键帧的相机(位置、朝向、焦距), 设成场景相机。两个关键帧之间直线过渡, 和人的动画一样。"""
    cd = bpy.data.cameras.new("camera"); cam = bpy.data.objects.new("camera", cd); sc.collection.objects.link(cam)
    cd.sensor_fit, cd.sensor_width, cd.shift_x, cd.shift_y = "HORIZONTAL", CT.SENSOR_MM, CAM_SX, CAM_SY
    cd.clip_start, cd.clip_end = 0.05, 1000.0
    cam.rotation_mode = "QUATERNION"; prev = None
    for f, M, mm in zip(CAM_VF, CAM_M, CAM_LENS):
        m = Matrix(M.tolist()); q = m.to_quaternion()
        if prev is not None:
            q.make_compatible(prev)                                   # q 和 -q 是同一个朝向: 挑和上一帧同号的, 不然中间帧会绕远路
        prev = q.copy()
        cam.location, cam.rotation_quaternion, cd.lens = m.to_translation(), q, float(mm)
        cam.keyframe_insert("location", frame=int(f)); cam.keyframe_insert("rotation_quaternion", frame=int(f))
        cd.keyframe_insert("lens", frame=int(f))
    for d in (cam, cd):
        for fc in d.animation_data.action.fcurves:
            for k in fc.keyframe_points:
                k.interpolation = "LINEAR"
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.render.resolution_percentage = 100
    return cam

def render_setup(sc):
    """存进 .blend 的灯和渲染设置: 打开直接 Ctrl+F12 就是从原视频的视角渲(render_video.py 渲视频用的也是这一份)。"""
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sun.data.energy = 3; sc.collection.objects.link(sun)
    sun.rotation_euler = (0.6, 0.2, 0.8)
    sc.world = bpy.data.worlds.new("world"); sc.world.color = (0.8, 0.8, 0.85)
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    sc["src_fps"] = fps                                               # 场景的帧率只能是整数; 出视频时用这个真的

def scene_boxes(objs):
    """场景里每个物体的世界系包围盒 (N,6), 按位置排好 —— FBX 导回来物体名可能变, 不按名字对。"""
    dg = bpy.context.evaluated_depsgraph_get(); out = []
    for o in objs:
        if o.type != "MESH" or o.find_armature() or not len(o.data.vertices):
            continue
        ev = o.evaluated_get(dg)
        P = np.array([ev.matrix_world @ v.co for v in ev.data.vertices])
        out.append(np.r_[P.min(0), P.max(0)])
    return np.array(sorted(out, key=lambda b: tuple(np.round(b, 3))))


# ---------- 1. 人: 每人一个 FBX ----------
sc = reset()
import_people()
lo, hi = frame_range(sc)
arms = sorted([o for o in sc.objects if o.type == "ARMATURE"], key=lambda o: o.name)
print(f"导入: {len(arms)} 个骨架, 帧 {lo}~{hi}")
def body_of(a):
    return [o for o in sc.objects if o.type == "MESH" and o.find_armature() == a]
people = [o for a in arms for o in [a] + body_of(a)]
for a in arms:
    export_fbx(f"{outd}/fbx_per_person/{name}_{body_of(a)[0].name.split('_')[0]}.fbx", [a] + body_of(a))

# ---------- 2. 加上场景和相机: 只有场景的 FBX, 场景 + 人 + 相机的 FBX ----------
tidy_bones()          # 去掉导入器给骨头套的小圆球, 不然打开是一堆点
had = set(sc.objects)
bpy.ops.import_scene.gltf(filepath=scene_glb)
scene_objs = [o for o in sc.objects if o not in had]
ref_boxes = scene_boxes(scene_objs)
export_fbx(f"{outd}/{name}_scene.fbx", scene_objs, anim=False)
cam = add_camera(sc); lo, hi = frame_range(sc)
print(f"相机: {len(CAM_VF)} 个关键帧(视频第 {CAM_VF[0]}~{CAM_VF[-1]} 帧), 焦距 {CAM_LENS.min():.1f}~{CAM_LENS.max():.1f} mm"
      f"(片幅宽 {CT.SENSOR_MM:.0f} mm), 画面 {RES[0]}x{RES[1]}; 横竖焦距差最多 {CAM_ANISO.max() * 100:.2f}%(取了平均)")
export_fbx(f"{outd}/{name}_scene_people.fbx", scene_objs + people + [cam])

# ---------- 3. 自检: 在 Blender 里播出来的 vs 目标 ----------
def check(tag):
  sc = bpy.context.scene
  for o in sorted([o for o in sc.objects if o.type == "MESH" and o.find_armature()], key=lambda o: o.name):
    pid = o.name.split("_")[0][1:]
    vf = z[f"p{pid}_vf"]; V = z[f"p{pid}_V"]
    errs = []
    for i in np.linspace(0, len(vf) - 1, 5).astype(int):
        sc.frame_set(int(vf[i]))
        ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
        P = np.array([ev.matrix_world @ v.co for v in me.vertices])
        ev.to_mesh_clear()
        g = np.c_[P[:, 0], P[:, 2], -P[:, 1]]                         # Blender(z 上) -> glTF(y 上)
        errs.append(np.linalg.norm(g - V[i], axis=1).mean() * 100)
    print(f"自检[{tag}] 人 {pid}: 5 帧平均差 " + " ".join(f"{e:.2f}" for e in errs) + " cm")
    if max(errs) > TOL_CM:
        bad.append(f"{tag} 人 {pid} 差 {max(errs):.2f} cm")
def check_scene(tag):
    got = scene_boxes(bpy.context.scene.objects)
    d = float(np.abs(got - ref_boxes).max()) * 100 if got.shape == ref_boxes.shape and len(got) else (0.0 if got.shape == ref_boxes.shape else np.inf)
    print(f"自检[{tag}] 场景: {len(got)} 个物体(导出前 {len(ref_boxes)} 个), 包围盒最大差 {d:.3f} cm")
    if d > TOL_CM:
        bad.append(f"{tag} 场景 {len(got)}/{len(ref_boxes)} 个物体, 包围盒差 {d:.2f} cm")
def check_camera(tag):
    """相机在 Blender 里播出来的位置 / 朝向 / 焦距 vs 存进去的; 再用 Blender 自己的投影去对流水线的投影(画面四角和中间, 3 米远的点)。"""
    sc = bpy.context.scene
    cs = [o for o in sc.objects if o.type == "CAMERA"]
    if len(cs) != 1:
        bad.append(f"{tag} 里有 {len(cs)} 个相机(应该正好 1 个)"); return
    sc.camera = cs[0]; sc.render.resolution_x, sc.render.resolution_y = RES
    h, w = (float(x) for x in z["cam_hw"])
    uv = np.array([[u, v] for u in (0, w / 2, w) for v in (0, h / 2, h)])
    dp = dr = dl = px = 0.0
    for i in np.linspace(0, len(CAM_VF) - 1, 5).astype(int):
        sc.frame_set(int(CAM_VF[i])); ev = cs[0].evaluated_get(bpy.context.evaluated_depsgraph_get())
        M = np.array(ev.matrix_world); R = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)
        dp = max(dp, float(np.linalg.norm(M[:3, 3] - CAM_M[i][:3, 3])) * 100)
        # 两个朝向差的角度 = 2 asin(|R1 - R2| / 2√2)。不用 acos(迹): 角度很小时它把 float32 的 1e-7 放大成 0.01°
        dr = max(dr, float(np.degrees(2 * np.arcsin(min(1.0, np.linalg.norm(R - CAM_M[i][:3, :3]) / (2 * np.sqrt(2)))))))
        dl = max(dl, abs(ev.data.lens / ev.data.sensor_width * CT.SENSOR_MM / CAM_LENS[i] - 1) * 100)
        X = CT.unproject_cv(z["cam_c2w"][i], z["cam_K"][i], uv, 3.0) @ CT.G2B.T
        got = np.array([world_to_camera_view(sc, ev, Vector(p))[:2] for p in X])
        px = max(px, float(np.abs(np.c_[got[:, 0] * w, (1 - got[:, 1]) * h] - uv).max()) * RES[0] / w)
    print(f"自检[{tag}] 相机: 5 帧里 位置最多差 {dp:.3f} cm, 朝向 {dr:.4f}°, 焦距 {dl:.4f}%; "
          f"画面上和流水线的投影最多差 {px:.1f} 像素({RES[0]} 宽的画面)")
    if dp > TOL_CM or dr > TOL_CAM_DEG or dl > TOL_CAM_LENS or px > TOL_CAM_PX * RES[0]:
        bad.append(f"{tag} 相机 位置差 {dp:.2f} cm / 朝向差 {dr:.3f}° / 焦距差 {dl:.3f}% / 画面差 {px:.1f} 像素")
sc = reset(); import_people(); frame_range(sc); check("glTF")
sc = reset(); bpy.ops.import_scene.fbx(filepath=f"{outd}/{name}_scene_people.fbx", anim_offset=0.0); frame_range(sc)
check("场景+人 FBX"); check_scene("场景+人 FBX"); check_camera("场景+人 FBX")
sc = reset(); bpy.ops.import_scene.fbx(filepath=f"{outd}/{name}_scene.fbx"); check_scene("场景 FBX")

# ---------- 4. .blend(只出这一份, 给人看的): 骨头连到下一个关节(四棱锥), 藏掉四肢上的辅助关节; 带相机、灯、渲染设置 ----------
# FBX 保持 MHR 关节原本的朝向(Blender 沿关节的 Y 轴画骨头, 是横着的短棍, 四肢上一排像梯子 —— 数据没错, 只是不好看)。
# .blend 在导入时让 Blender 把骨头转到朝着子关节(Bone Dir = Fortune), 动作它自己换算, 下面照样自检。
# ★ 2026-10-09 以前还另存一份"关节朝向原样"的 .blend, 两个名字只差 _view, 打开的总是梯子那份 —— 不出了。
sc = reset(); import_people("FORTUNE"); frame_range(sc); view_bones(); check("给人看的")
bpy.ops.import_scene.gltf(filepath=scene_glb)
add_camera(sc); frame_range(sc); render_setup(sc); check_camera("给人看的")
sc.frame_set(sc.frame_start)
bpy.ops.wm.save_as_mainfile(filepath=f"{outd}/{name}_scene_people.blend")
print("->", f"{outd}/{name}_scene_people.blend")
if bad:
    raise SystemExit("[失败] 自检没过(门槛 %.1f cm): " % TOL_CM + "; ".join(bad))

# ---------- 5. 预览图: 场景 + 人, Cycles CPU ----------
bpy.ops.wm.open_mainfile(filepath=f"{outd}/{name}_scene_people.blend")
sc = bpy.context.scene
frame_range(sc)
f_mid = (sc.frame_start + sc.frame_end) // 2
sc.frame_set(f_mid)
dg = bpy.context.evaluated_depsgraph_get()
pts = []
for o in sc.objects:
    if o.type == "MESH" and o.find_armature():
        ev = o.evaluated_get(dg); me = ev.to_mesh()
        pts += [ev.matrix_world @ me.vertices[i].co for i in range(0, len(me.vertices), 50)]; ev.to_mesh_clear()
pts = np.array(pts); c = pts.mean(0); ext = np.ptp(pts, axis=0).max()
cam = bpy.data.objects.new("preview_cam", bpy.data.cameras.new("preview_cam")); sc.collection.objects.link(cam)   # 斜上方看全场; 不存
cam.location = (c[0] - 0.9 * ext - 3, c[1] - 0.9 * ext - 3, c[2] + 0.6 * ext + 2)
cam.rotation_euler = (Vector(c) - cam.location).to_track_quat("-Z", "Y").to_euler()
sc.camera = cam                                                        # 灯和天光 .blend 里已经有了(render_setup)
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 16
sc.render.resolution_x, sc.render.resolution_y = 960, 540
sc.render.filepath = f"{outd}/{name}_preview.png"
bpy.ops.render.render(write_still=True)
print("->", sc.render.filepath)
for f in os.listdir(outd):                 # Blender 另存时留的上一版备份
    if f.endswith(".blend1"):
        os.remove(f"{outd}/{f}")
