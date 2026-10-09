"""把 glTF 导入的骨架改成一眼能看出是骨骼的样子。
Blender 的 glTF 导入默认给每根骨头套一个小圆球当显示形状(Icosphere), 一个人 127 根骨头, 打开就是一堆点。
这里去掉圆球, 骨头显示成细棍、画在身体前面。
单独用(改已有的 .blend): python tidy_bones.py <a.blend> [b.blend ...]"""
import bpy


def tidy_bones():
    for o in list(bpy.data.objects):
        if o.type != "ARMATURE":
            continue
        for pb in o.pose.bones:
            pb.custom_shape = None
        o.data.display_type = "STICK"
        o.show_in_front = True
    for o in list(bpy.data.objects):          # 导入器留下的圆球本体: 没挂骨架、名字是 Icosphere*
        if o.type == "MESH" and o.name.startswith("Icosphere") and not o.find_armature():
            bpy.data.objects.remove(o, do_unlink=True)


def view_bones():
    """给人看的骨架。要配合导入时 Bone Dir = Fortune(骨头朝着子关节画, 导入器自己把动作换算过去):
    四肢上的辅助关节(名字带 _proc, 每段肢体 4~5 个, 拧手臂/腿时让皮肤过渡平滑用的)藏起来; 骨头拉长到下一个关节; 显示成四棱锥。
    这里只改骨头的长度和显示, 不改朝向 —— 动作不变。重复调用结果一样。"""
    for o in list(bpy.data.objects):
        if o.type != "ARMATURE":
            continue
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.mode_set(mode="EDIT")
        for eb in o.data.edit_bones:
            y = (eb.tail - eb.head).normalized()
            along = [(c.head - eb.head).dot(y) for c in eb.children                      # 正好在这根骨头延长线上的子关节
                     if "_proc" not in c.name and (c.head - eb.head).length > 1e-4 and (c.head - eb.head).angle(y) < 0.26]
            if along:
                eb.length = max(along)
        bpy.ops.object.mode_set(mode="OBJECT")
        for b in o.data.bones:
            b.hide = "_proc" in b.name or b.name.endswith("body_world")       # body_world: 钉在场景原点的节点, 不是身体的一部分
        o.data.display_type = "OCTAHEDRAL"
        o.show_in_front = True


if __name__ == "__main__":
    import os, sys
    for f in sys.argv[1:]:
        bpy.ops.wm.open_mainfile(filepath=f)
        tidy_bones()
        bpy.ops.wm.save_mainfile(filepath=f)
        if os.path.exists(f + "1"):
            os.remove(f + "1")                # Blender 自动留的上一版备份
        arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
        print(f"{f}: {len(arms)} 个骨架, 显示 {arms[0].data.display_type}, "
              f"还带圆球的骨头 {sum(pb.custom_shape is not None for a in arms for pb in a.pose.bones)}, "
              f"剩下的 Icosphere {sum(o.name.startswith('Icosphere') for o in bpy.data.objects)}")
