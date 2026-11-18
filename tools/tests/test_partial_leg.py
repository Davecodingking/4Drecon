import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))


def test_cropped_ankle_does_not_hide_a_measured_thigh():
    from leg_observations import thigh_confidence
    for fps in (12, 30, 60):
        n = max(10, fps)
        uv = np.zeros((n, 2, 3, 2))
        uv[:, :, 0] = [70, 22]
        uv[:, :, 1] = [83, 64]
        uv[:, :, 2] = [90, 140]
        uv[:, 1, :, 0] += 90
        with TemporaryDirectory() as temp:
            path = Path(temp) / "mask.png"
            mask = np.zeros((100, 190), np.uint8)
            mask[:, 45:110] = 1
            Image.fromarray(mask).save(path)
            confidence = thigh_confidence(uv, np.ones(uv.shape[:-1]), np.full(n,145.),
                                          [path]*n,1,(190,100),np.zeros(n),fps)
            assert confidence[:,0].min() > .8
            assert not confidence[:,1].any()


def test_inferred_steps_do_not_translate_an_actor():
    import torch
    from leg_observations import root_displacement
    reference = torch.tensor([[.01,.02,-.04],[.42,.11,-.06]])
    delta = torch.tensor([[.26,.12,-.22],[.32,-.1,.06]])
    fixed = torch.tensor([True, False])
    out = root_displacement(delta,reference,fixed)
    torch.testing.assert_close(out[0], reference[0])
    torch.testing.assert_close(out[1], delta[1])


def test_measured_root_only_holds_the_repeated_edge_within_its_shot():
    from leg_observations import measured_root_trajectory
    centers = np.column_stack([np.arange(18)*.035,np.zeros(18),np.ones(18)])
    centers[9:,0] += 2
    shot = np.repeat([0,1],9)
    holds = np.zeros(17,bool)
    holds[4] = True
    out = measured_root_trajectory(centers,shot,30,holds)
    np.testing.assert_allclose(out[4],out[5])
    assert out[9,0]-out[8,0] > 2
    assert out[12,0] > out[10,0]


def test_thigh_measurement_uses_hip_relative_image_coordinates():
    import torch
    from leg_observations import projected_thigh_error
    projection = torch.tensor([[[200.,0.,80.,0.],[0.,200.,50.,0.],[0.,0.,1.,0.]]])
    hip = torch.tensor([[[.20,.1,2.],[-.10,.1,2.]]])
    knee = hip+torch.tensor([[[.15,.35,0.],[-.15,.35,0.]]])
    scale = torch.tensor([100.])
    image_offset = torch.tensor([[[.15,.35],[-.15,.35]]])
    error = projected_thigh_error(hip,knee,projection,scale,image_offset)
    torch.testing.assert_close(error,torch.zeros_like(error))
    wrong = projected_thigh_error(hip,knee.flip(1),projection,scale,image_offset)
    assert (wrong > .1).all()



def test_only_verified_grounded_joins_receive_a_continuous_velocity_prior():
    from leg_observations import registered_root_prior, measured_root_trajectory
    frames=np.arange(100)
    shot=np.searchsorted([30,70],frames,side="right")
    up=np.array([0.,0.,1.])
    root=np.column_stack([frames*.012,np.zeros_like(frames),np.ones_like(frames)])
    root[30:,2]+=.18
    root[70:,0]+=2.
    held=np.zeros(99,bool)
    held[31]=True
    poses=measured_root_trajectory(root,shot,30,held)
    repaired,w=registered_root_prior(root,shot,30,held,np.array([30]),root.copy(),up)
    assert abs(repaired[30,2]-repaired[29,2])<.045
    assert w[30]>.9 and not w[70]
    np.testing.assert_array_equal(repaired[31],repaired[32])
    np.testing.assert_array_equal(repaired[60:],poses[60:])
    flight=(frames>=29)&(frames<=33)
    guarded,w=registered_root_prior(root,shot,30,held,np.array([30]),root.copy(),up,flight)
    np.testing.assert_array_equal(guarded[flight],poses[flight])
    assert not w[flight].any()

if __name__ == "__main__":
    test_only_verified_grounded_joins_receive_a_continuous_velocity_prior()
    test_cropped_ankle_does_not_hide_a_measured_thigh()
    test_inferred_steps_do_not_translate_an_actor()
    test_measured_root_only_holds_the_repeated_edge_within_its_shot()
    test_thigh_measurement_uses_hip_relative_image_coordinates()
    print("PASS")
