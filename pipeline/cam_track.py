"""相机序列: 流水线的相机(每个场景帧一个位姿 + 内参, OpenCV 约定) -> Blender / FBX 里一个带关键帧的相机。
只有换算, 纯 numpy: sam3d 环境(body_motion.py 存, 测试)和 bpy 环境(to_blender.py 建相机)都能 import。

★ 为什么要有(2026-10-09): 动作文件里只有场景和人, 没有相机 —— 在 Blender 里看不到"原视频是从哪拍的",
  也没法从原视频的视角渲一遍去对原视频。相机每帧的位姿流水线一直有(02_da3/cameras.json), 只是没带出来。

约定(和人、场景一致):
  motion.npz 里  cam_c2w (N,4,4)  相机 -> 世界, 世界是 glTF 的(天 +y, 米), 相机轴是 OpenCV 的(x 右, y 下, z 朝前)
                 cam_K (N,3,3)    内参, 像素是 cam_hw = (高, 宽) 那张图的(DA3 吃进去的图, 不是原视频的分辨率)
                 cam_vf (N,)      原视频帧号(和人的 p<人>_vf 同一套)
  Blender 里     世界天 +z; 相机看向自己的 -z, y 朝上
Blender / FBX 的相机只有一个焦距(像素是方的), 流水线的横、竖焦距各估各的, 差 0.3%~0.8% —— 取两者的平均。
画面最边上因此会和流水线自己的投影差几个像素(to_blender.py 的自检会打印出来)。
"""
import numpy as np

SENSOR_MM = 36.0                                                   # Blender 相机的片幅宽; 只是个约定, 焦距(mm)按它换算
G2B = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])           # glTF 世界(天 +y) -> Blender 世界(天 +z): (x, y, z) -> (x, -z, y)
CV2BL = np.diag([1.0, -1.0, -1.0])                                 # OpenCV 相机轴 -> Blender 相机轴


def track(cams, vid2key, T):
    """cameras.json 的 cameras + {原视频帧号: 相机的键} + 世界变换 T (4,4)(DA3 世界 -> 导出的世界, 即 export_T(UP))
    -> dict(cam_vf, cam_c2w, cam_K, cam_hw), 按帧号排。"""
    vfs = sorted(v for v, k in vid2key.items() if k in cams)
    if not vfs:
        raise SystemExit("[失败] 一个对得上视频帧号的相机都没有(cameras.json 和 frame_map.json 对不上)")
    hw = {tuple(cams[vid2key[v]]["input_hw"]) for v in vfs}
    if len(hw) != 1:
        raise SystemExit(f"[失败] 相机的图大小不止一种 {sorted(hw)}, 内参没法放进同一个相机")
    T = np.asarray(T, float)
    return dict(cam_vf=np.array(vfs, np.int32),
                cam_c2w=np.stack([T @ np.linalg.inv(np.array(cams[vid2key[v]]["extrinsic_w2c_4x4"], float)) for v in vfs]),
                cam_K=np.stack([np.array(cams[vid2key[v]]["intrinsic_3x3"], float) for v in vfs]),
                cam_hw=np.array(hw.pop(), np.int32))


def blender_pose(c2w):
    """(N,4,4) cam_c2w -> (N,4,4) Blender 里相机的 matrix_world。"""
    c2w = np.asarray(c2w, float)
    M = np.tile(np.eye(4), (len(c2w), 1, 1))
    M[:, :3, :3] = G2B @ c2w[:, :3, :3] @ CV2BL
    M[:, :3, 3] = c2w[:, :3, 3] @ G2B.T
    return M


def lens(K, hw):
    """-> (焦距 mm (N,), 横向偏移, 竖向偏移, 横竖焦距差的比例 (N,))。按"片幅宽 SENSOR_MM、横向对齐"换算;
    偏移是 Blender 的 shift_x / shift_y(主点不在图正中时才不是 0, 取所有帧的中位)。"""
    K = np.asarray(K, float); h, w = (float(x) for x in hw)
    f = (K[:, 0, 0] + K[:, 1, 1]) / 2
    return (f / w * SENSOR_MM, float(np.median(w / 2 - K[:, 0, 2]) / w), float(np.median(K[:, 1, 2] - h / 2) / w),
            np.abs(K[:, 1, 1] - K[:, 0, 0]) / f)


def render_size(hw, video_wh, max_w=1920):
    """渲染分辨率: 宽 = 原视频的宽(最多 max_w), 高按内参那张图的宽高比来(DA3 把图缩成 16 的倍数, 比例和原视频差一点点), 都取偶数。"""
    W = int(min(video_wh[0], max_w)); W -= W % 2
    H = int(round(W * float(hw[0]) / float(hw[1]))); H -= H % 2
    return W, H


def project_cv(c2w, K, X):
    """流水线自己的投影: glTF 世界里的点 X (M,3) -> 像素 (M,2)(cam_hw 那张图的)。"""
    Xc = (np.asarray(X, float) - c2w[:3, 3]) @ c2w[:3, :3]
    return np.c_[K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]]


def unproject_cv(c2w, K, uv, depth):
    """像素 (M,2) + 深度(米) -> glTF 世界里的点 (M,3)。"""
    uv = np.asarray(uv, float)
    Xc = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))] * depth
    return Xc @ c2w[:3, :3].T + c2w[:3, 3]


def project_blender(M, lens_mm, shift, hw, Xb):
    """Blender 相机的投影(片幅横向对齐, 方像素): Blender 世界里的点 Xb (M,3) -> 像素 (M,2)(cam_hw 那张图的)。测试用。"""
    h, w = (float(x) for x in hw)
    Xc = (np.asarray(Xb, float) - M[:3, 3]) @ M[:3, :3]
    f = lens_mm / SENSOR_MM * w
    return np.c_[w / 2 - shift[0] * w + f * Xc[:, 0] / -Xc[:, 2], h / 2 + shift[1] * w - f * Xc[:, 1] / -Xc[:, 2]]
