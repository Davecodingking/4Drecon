"""pipeline/dense_cams.py: 人体全帧时中间帧的相机(前后两个场景相机插值, 不跨切点, 坏相机传染)。"""
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import dense_cams as D


def cam(C, yaw_deg, f=600.0):
    c2w = np.eye(4); c2w[:3, :3] = Rot.from_euler("y", yaw_deg, degrees=True).as_matrix(); c2w[:3, 3] = C
    return dict(extrinsic_w2c_4x4=np.linalg.inv(c2w).tolist(), intrinsic_3x3=[[f, 0, 320], [0, f, 180], [0, 0, 1]],
                camera_center_world=list(C), input_hw=[360, 640], mean_conf=5.0)


def scene():
    # 场景帧: 原视频 0 / 5 / 10 | 切点 0.5 秒(第 15 帧) | 20 / 25
    cams = {"0000.jpg": cam([0, 0, 0], 0), "0001.jpg": cam([1, 0, 0], 10, 700), "0002.jpg": cam([2, 0, 0], 20),
            "0003.jpg": cam([10, 0, 0], 90), "0004.jpg": cam([11, 0, 0], 90)}
    fmap = {"0000.jpg": 0, "0001.jpg": 5, "0002.jpg": 10, "0003.jpg": 20, "0004.jpg": 25}
    return cams, fmap


def centre(c):
    return np.linalg.inv(np.asarray(c["extrinsic_w2c_4x4"]))[:3, 3]


def yaw(c):
    return Rot.from_matrix(np.linalg.inv(np.asarray(c["extrinsic_w2c_4x4"]))[:3, :3]).as_euler("xyz", degrees=True)[1]


def test_scene_cameras_kept_as_is():
    cams, fmap = scene()
    c2, f2, b2 = D.build(cams, fmap, set(), [0.5], 30.0, range(0, 26))
    for k in cams:
        assert c2[k] is cams[k] and f2[k] == fmap[k]
    assert len(c2) == 26 and sorted(f2.values()) == list(range(26))


def test_in_between_frame_is_interpolated():
    cams, fmap = scene()
    c2, f2, _ = D.build(cams, fmap, set(), [0.5], 30.0, [2, 7])
    assert np.allclose(centre(c2["d0002.jpg"]), [0.4, 0, 0]) and abs(yaw(c2["d0002.jpg"]) - 4.0) < 1e-6
    assert abs(np.asarray(c2["d0002.jpg"]["intrinsic_3x3"])[0, 0] - 640.0) < 1e-6        # 焦距也插(推镜头)
    assert np.allclose(centre(c2["d0007.jpg"]), [1.4, 0, 0])


def test_no_interpolation_across_a_cut():
    cams, fmap = scene()
    c2, _, _ = D.build(cams, fmap, set(), [0.5], 30.0, [12, 14, 16, 18])
    for v in (12, 14):                                                                        # 切点前: 停在本镜头最后一个
        assert np.allclose(centre(c2[f"d{v:04d}.jpg"]), [2, 0, 0])
    for v in (16, 18):                                                                        # 切点后: 用下一个镜头第一个
        assert np.allclose(centre(c2[f"d{v:04d}.jpg"]), [10, 0, 0])


def test_bad_camera_spreads_to_frames_that_use_it():
    cams, fmap = scene()
    _, _, b2 = D.build(cams, fmap, {"0001.jpg"}, [0.5], 30.0, range(0, 11))
    assert {"0001.jpg", "d0001.jpg", "d0004.jpg", "d0006.jpg", "d0009.jpg"} <= b2
    assert "0000.jpg" not in b2 and "0002.jpg" not in b2


def test_shot_without_scene_frames_borrows_and_is_bad():
    cams, fmap = scene()
    _, _, b2 = D.build(cams, fmap, set(), [0.5, 0.6], 30.0, [17])                              # 0.5~0.6 秒这段没有场景帧
    assert "d0017.jpg" in b2



def test_identity_depth_uses_same_track_camera_anchors():
    from body_into_scene import identity_depth_ratios
    keys = {0: "a", 1: "middle", 2: "b"}
    cams = {"a": {}, "b": {}, "middle": {"插值": {"前": "a", "后": "b", "w": 0.5}}}
    roots = {i: np.array([0., 0., z]) for i, z in enumerate([8., 12., 9.])}
    def distances(measured):
        ratios = identity_depth_ratios([0, 1, 2], measured, roots, cams, keys, list(keys), 30)
        return {i: roots[i][2] * ratio for i, ratio in ratios.items()}
    assert distances({0: 1., 2: 1.}) == {0: 8., 1: 8.5, 2: 9.}
    assert distances({2: 1.})[1] == 9.
    assert distances({0: 1., 2: 2.})[1] == 18.  # Keep a true depth jump.
    cams["middle"]["插值"]["镜头外借来的"] = True
    assert 1 not in distances({0: 1., 2: 1.})

def test_depth_jump_requires_unambiguous_continuous_mask():
    from body_into_scene import isolated_mask_pair
    a = np.zeros((100, 100), np.uint8)
    a[20:60, 20:40] = 1
    b = np.roll(a, 1, axis=1)
    assert isolated_mask_pair(a, b, 1)
    assert not isolated_mask_pair(a, np.roll(b, 40, axis=1), 1)
    b[5:20, 70:80] = 2
    assert not isolated_mask_pair(a, b, 1)


if __name__ == "__main__":
    for f in [test_depth_jump_requires_unambiguous_continuous_mask, test_identity_depth_uses_same_track_camera_anchors, test_scene_cameras_kept_as_is, test_in_between_frame_is_interpolated, test_no_interpolation_across_a_cut,
              test_bad_camera_spreads_to_frames_that_use_it, test_shot_without_scene_frames_borrows_and_is_bad]:
        f(); print("ok", f.__name__)
    print("== test_dense_cams 全过 ==")
