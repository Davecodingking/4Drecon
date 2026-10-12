#!/usr/bin/env python
"""叠加检查图用的数据: 把库里的网格(mesh_store.py)换回上游画图函数(visualize_sample_together)要的格式。
画图本身在 sam-body4d/body4d_report_5090.py 里(body4d 环境, 要上游的渲染器); 这里只有不用渲染器的那部分, 好测。

★ 为什么(2026-10-12): 以前叠加图是出网格时顺手画的。按人拆到几张卡之后, 一张图是几份叠起来的 ——
    - 闸口上点掉一个人, 图里没法只去掉他(他和别人画在同一张图里), 只能全部重出;
    - 不同份里的人谁挡谁可能画反, 单人一份的全是同一个颜色(office r12: 9 个人里 7 个同色)。
  现在每次从库里读网格现画: 库里有谁图里就有谁, 所有人画在一个场景里(前后遮挡是对的), 颜色跟跟踪编号走。
  office r12 实测: 和原来的叠加图比, 人的轮廓重合 99.2%~99.6%。

盘上的格式(上游 save_mesh_results 写的): mesh_4d_individual/<人>/<帧>.ply = (顶点 + 相机平移) 再绕 x 轴转 180°,
也就是 y、z 取反; focal_4d_individual/<人>/<帧>.json = {focal_length, camera: 相机平移}。

只用 numpy + trimesh(sam3d、body4d 两个环境都有)。
"""
import json, os
import numpy as np

UNFLIP = np.float32([1, -1, -1])           # 绕 x 轴转 180° 的逆(和它自己一样): y、z 取反


def frame_people(run, stem, dirs):
    """这一帧在场的人。dirs: {跟踪编号: 库里的目录名}(10-12 起两边一样; 老目录见 relabel.json)。
    返回 (outputs, ids, faces): outputs[i] = dict(pred_vertices, pred_cam_t, focal_length) 给上游画图函数;
    ids[i] = 跟踪编号(按编号排, 颜色跟它走); faces = 三角面。谁都不在 = ([], [], None)。"""
    import trimesh
    outs, ids, faces = [], [], None
    for tid in sorted(dirs):
        ply = f"{run}/mesh_4d_individual/{dirs[tid]}/{stem}.ply"
        fj = f"{run}/focal_4d_individual/{dirs[tid]}/{stem}.json"
        if not (os.path.exists(ply) and os.path.exists(fj)):
            continue
        m = trimesh.load(ply, process=False)
        f = json.load(open(fj))
        cam = np.asarray(f["camera"], np.float32)
        outs.append(dict(pred_vertices=np.asarray(m.vertices, np.float32) * UNFLIP - cam, pred_cam_t=cam,
                         focal_length=np.float32(f["focal_length"])))
        ids.append(int(tid))
        faces = np.asarray(m.faces)
    return outs, ids, faces


def color_slot(tid, n_colors):
    """上游按 color_list[编号 + 4] 给人上色。编号超出色表就绕回来(从 1 起)。"""
    return (int(tid) - 1) % (n_colors - 5) + 1


def blend(img, ren, alpha, white=245):
    """网格图(白底)半透明叠到原片上。白底之外的像素才算网格; white 留了点抗锯齿的余量。ren=None(这一帧没人)原片原样。"""
    if ren is None:
        return img
    ren = np.asarray(ren, np.float32)
    fg = (ren.min(axis=2) < white).astype(np.float32)[..., None]
    return np.round(img * (1 - alpha * fg) + ren * (alpha * fg)).astype(np.uint8)
