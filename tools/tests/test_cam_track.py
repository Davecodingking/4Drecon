#!/usr/bin/env python
"""相机序列的换算(pipeline/cam_track.py): 流水线的相机 -> Blender 的相机, 看到的画面要是同一幅。
跑: $PY tools/tests/test_cam_track.py"""
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
SAM3D = os.path.expanduser("~/sam3d")
sys.path.insert(0, f"{SAM3D}/pipeline")
import cam_track as CT

HW = (384, 688)


def _cams(n=4, fx=630.0, fy=630.0, cx=344.0, cy=192.0, seed=0):
    rng = np.random.default_rng(seed); cams, v2k = {}, {}
    for i in range(n):
        w2c = np.eye(4); w2c[:3, :3] = R.from_rotvec(rng.normal(0, 0.4, 3)).as_matrix(); w2c[:3, 3] = rng.normal(0, 1.0, 3)
        cams[f"{i:04d}.jpg"] = dict(extrinsic_w2c_4x4=w2c.tolist(), input_hw=list(HW),
                                    intrinsic_3x3=[[fx + 3 * i, 0, cx], [0, fy + 3 * i, cy], [0, 0, 1]])
        v2k[i * 4] = f"{i:04d}.jpg"
    return cams, v2k


def _world_T(seed=1):
    T = np.eye(4); T[:3, :3] = R.from_rotvec(np.random.default_rng(seed).normal(0, 0.5, 3)).as_matrix(); return T


def test_track_is_sorted_by_video_frame_and_skips_frames_without_a_camera():
    cams, v2k = _cams()
    v2k[99] = "9999.jpg"                                    # 帧号表里有、cameras.json 里没有(那一帧被 DA3 跳过了)
    t = CT.track(cams, dict(reversed(list(v2k.items()))), np.eye(4))
    assert t["cam_vf"].tolist() == [0, 4, 8, 12] and t["cam_c2w"].shape == (4, 4, 4) and t["cam_K"].shape == (4, 3, 3)
    assert t["cam_hw"].tolist() == list(HW)


def test_a_scene_point_lands_on_the_same_pixel_after_the_world_is_rotated_for_export():
    """导出时整个世界转过一次(天转到 +y)。场景里的点跟着转, 相机也跟着转 —— 点落在画面上的位置不能变。"""
    cams, v2k = _cams(); T = _world_T(); t = CT.track(cams, v2k, T)
    X = np.random.default_rng(2).normal(0, 0.5, (20, 3)) + [0, 0, 4.0]             # DA3 世界里的点
    for i, key in enumerate(sorted(cams)):
        w2c = np.array(cams[key]["extrinsic_w2c_4x4"]); K = np.array(cams[key]["intrinsic_3x3"])
        Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
        want = np.c_[K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]]
        got = CT.project_cv(t["cam_c2w"][i], t["cam_K"][i], X @ T[:3, :3].T)
        assert np.abs(got - want).max() < 1e-6, np.abs(got - want).max()


def test_blender_camera_sees_the_same_picture_as_the_pipeline_camera():
    """横竖焦距一样时, Blender 相机(看向 -z, y 朝上, 世界天 +z)投出来的像素和流水线的一模一样。"""
    cams, v2k = _cams(); t = CT.track(cams, v2k, _world_T())
    Ms = CT.blender_pose(t["cam_c2w"]); mm, sx, sy, aniso = CT.lens(t["cam_K"], t["cam_hw"])
    assert sx == 0 and sy == 0 and aniso.max() == 0
    uv = np.array([[10.0, 10], [678, 10], [10, 374], [678, 374], [344, 192], [500, 100]])
    for i in range(len(Ms)):
        X = CT.unproject_cv(t["cam_c2w"][i], t["cam_K"][i], uv, 3.0)
        got = CT.project_blender(Ms[i], mm[i], (sx, sy), HW, X @ CT.G2B.T)
        assert np.abs(got - uv).max() < 1e-6, np.abs(got - uv).max()
        assert abs(np.linalg.det(Ms[i][:3, :3]) - 1) < 1e-9                        # 还是个旋转, 没被镜像


def test_blender_up_is_plus_z_and_camera_looks_down_its_minus_z():
    """一个在 glTF 世界里平视前方(看向 -z)、头朝天(+y)的相机: 到 Blender 里是看向 +y、头朝 +z。"""
    c2w = np.eye(4); c2w[:3, :3] = np.diag([1.0, -1.0, -1.0])                       # OpenCV 轴: z 朝前 = 世界 -z, y 朝下 = 世界 -y
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    M = CT.blender_pose(c2w[None])[0]
    assert np.allclose(-M[:3, 2], [0, 1, 0]) and np.allclose(M[:3, 1], [0, 0, 1])   # 看的方向 = -z 轴; 头顶 = y 轴
    assert np.allclose(M[:3, 3], [1.0, -3.0, 2.0])


def test_principal_point_off_center_becomes_a_shift():
    """主点不在图正中(以后给了真内参时会有): 换成 Blender 的 shift, 画面还是对得上。"""
    cams, v2k = _cams(cx=360.0, cy=180.0); t = CT.track(cams, v2k, np.eye(4))
    Ms = CT.blender_pose(t["cam_c2w"]); mm, sx, sy, _ = CT.lens(t["cam_K"], t["cam_hw"])
    assert sx != 0 and sy != 0
    uv = np.array([[10.0, 10], [678, 374], [360, 180]])
    X = CT.unproject_cv(t["cam_c2w"][0], t["cam_K"][0], uv, 2.0)
    assert np.abs(CT.project_blender(Ms[0], mm[0], (sx, sy), HW, X @ CT.G2B.T) - uv).max() < 1e-6


def test_unequal_focals_use_the_mean_and_report_the_gap():
    """横竖焦距不一样(DA3 各估各的): 取平均, 并把差多少报出来; 画面边上的误差就是 差的一半 x 离中心的像素数。"""
    cams, v2k = _cams(fx=636.0, fy=630.0); t = CT.track(cams, v2k, np.eye(4))
    mm, _, _, aniso = CT.lens(t["cam_K"], t["cam_hw"])
    assert abs(mm[0] - 633.0 / 688 * CT.SENSOR_MM) < 1e-9 and abs(aniso[0] - 6 / 633.0) < 1e-12
    uv = np.array([[0.0, 192], [344, 0]])                                            # 左边缘中点, 上边缘中点
    X = CT.unproject_cv(t["cam_c2w"][0], t["cam_K"][0], uv, 2.0)
    got = CT.project_blender(CT.blender_pose(t["cam_c2w"])[0], mm[0], (0, 0), HW, X @ CT.G2B.T)
    assert abs(abs(got[0, 0] - uv[0, 0]) - 344 * 3 / 636) < 1e-6 and abs(abs(got[1, 1] - uv[1, 1]) - 192 * 3 / 630) < 1e-6


def test_render_size_follows_the_camera_picture_not_the_video():
    assert CT.render_size(HW, (1920, 1080)) == (1920, 1072)                           # 688:384, 不是 16:9
    assert CT.render_size(HW, (1280, 720)) == (1280, 714)
    assert CT.render_size(HW, (3840, 2160)) == (1920, 1072)                           # 最多 1920 宽


def test_mixed_picture_sizes_are_refused():
    cams, v2k = _cams(); cams["0001.jpg"]["input_hw"] = [400, 700]
    try:
        CT.track(cams, v2k, np.eye(4))
    except SystemExit as e:
        assert "不止一种" in str(e)
    else:
        raise AssertionError("图大小不一致应该报错")


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f(); n += 1; print("ok", k)
    print(f"{n} ok")
