from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2] / "pipeline"))
from landing_motion import foot_targets, turn_support_loss, smooth_turn_rotations


def test_planned_swing_and_double_support_survive_a_restarted_track():
    up = np.array([0.,0.,1.])
    # Left foot swings then plants, while the right stays planted throughout.
    sole = np.zeros((7, 2, 5, 3))
    sole[:,0,:,0] = np.array([0.,.1,.2,.3,.3,.3,.3])[:,None]
    sole[:,0,:,2] = np.array([0.,.04,.06,0.,0.,0.,0.])[:,None]
    contact = np.ones((7,2), dtype=bool)
    contact[1:3,0] = False
    # The same stance was reconstructed in another camera coordinate system.
    sole[5:,:,:,0] += .8
    plan = dict(done=np.ones(7,dtype=bool),contact=contact,sole_world=sole)
    inferred = np.zeros_like(contact)
    inferred[:,1] = True
    targets, support, _ = foot_targets(plan,inferred,np.zeros_like(sole),up)
    np.testing.assert_array_equal(contact,support)
    np.testing.assert_allclose(targets[:5],sole[:5])
    np.testing.assert_allclose(targets[5:],np.repeat(targets[4:5],2,axis=0))


def test_turn_support_tracks_weight_and_absolute_position():
    import torch
    up = torch.tensor([0.,0.,1.])
    hips = torch.tensor([[[.10,0.,1.],[-.10,0.,1.]]])
    soles = torch.tensor([[[.13,0.,0.],[-.13,0.,0.]]]).reshape(1,2,1,3).expand(-1,-1,5,-1)
    both = torch.tensor([[.5,.5]])
    err, spread = turn_support_loss(hips,soles,both,up)
    assert err.item() == 0 and spread.item() == 0
    # Same step width, but both feet are left of the pelvis.
    misplaced = soles + torch.tensor([.25,0.,0.])
    err, spread = turn_support_loss(hips,misplaced,both,up)
    assert err.item() > .03 and spread.item() > .01
    right = torch.tensor([[0.,1.]])
    err, _ = turn_support_loss(hips,soles,right,up)
    transferred, _ = turn_support_loss(hips-torch.tensor([.10,0.,0.]),soles,right,up)
    assert transferred.item() < err.item()


def test_turn_smoothing_does_not_mix_cameras_or_freeze_a_turn():
    from scipy.spatial.transform import Rotation
    yaw = np.r_[np.zeros(10), np.arange(12)*6.+95.]
    q = Rotation.from_rotvec(np.column_stack([np.zeros_like(yaw),np.zeros_like(yaw),np.deg2rad(yaw)]))
    out = smooth_turn_rotations(q,np.r_[np.zeros(10,bool),np.ones(12,bool)],[10],30)
    angles = np.unwrap(out.as_euler('xyz')[:,2])
    np.testing.assert_allclose(angles[:10],0.)
    assert np.rad2deg(angles[10]) >= 95.
    assert np.min(np.diff(angles[10:])) > 0


def test_contact_solver_uses_the_exported_skin():
    import torch
    import mhr_body as M
    import legs_proc as L
    from landing_motion import skin_vertices, vertex_binding

    shape = np.zeros(45)
    pose = np.zeros((4, M.NPOSE+M.NSCALE))
    pose[:, L.LEG_COLS['r'][3]] = [0., .3, .7, 1.]
    pose[:, L.LEG_COLS['r'][-2]] = [0., .1, -.1, .35]
    geo = L.geometry(shape,pose[0],'cpu')
    ids = np.array([geo['side'][sd]['vids'][name] for sd in L.SIDES for name in L.PTS])
    binding = vertex_binding(shape,ids,'cpu')
    vertices, joints = M.forward(shape,pose,False,'cpu')
    fitted = skin_vertices(torch.from_numpy(joints),binding).numpy()
    np.testing.assert_allclose(fitted,vertices[:,ids],atol=1.e-5)


def test_observed_swing_requires_person_evidence_and_coherent_motion():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from PIL import Image
    from landing_motion import observed_swing_weights

    with TemporaryDirectory() as directory:
        label = Path(directory) / 'mask.png'
        Image.fromarray(np.ones((200,300),np.uint8)).save(label)
        for fps in (12,30):
            n = fps
            uv = np.full((n,2,3,2),100.,dtype=float)
            uv[:,0,2,0] -= np.linspace(0,70,n)
            depth = np.ones(uv.shape[:-1])
            scale = np.full(n,150.)
            support = np.zeros((n,2),bool)
            shot = np.zeros(n)
            def check():
                return observed_swing_weights(uv,depth,scale,(300,200),shot,support,
                                              fps,[label]*n,1)[1]

            assert check()[n//2,0] > .3 and not check()[:,1].any()
            # The second leg may be entirely out of view.
            uv[:,1,2,1] = 250.
            assert check()[n//2,0] > .3
            support[:,0] = True
            assert not check().any()
            support[:,0] = False
            uv[:,0,2,0] = 100.
            uv[n//2,0,2,0] = 50.
            assert not check().any()
            # Changes of camera angle do not indicate a kick.
            uv[n//2:,0,2,0] = 70.
            shot[n//2:] = 1
            assert not check().any()
            uv[:,0,2,0] = np.linspace(30,100,n)
            assert not observed_swing_weights(uv,depth,scale,(300,200),shot,support,
                                              fps,[label]*n,2)[0].any()


def test_observed_support_and_flight_release():
    from landing_motion import support_observations
    for fps in (24,30,60):
        n = fps
        u = np.array([0.,0.,1.])
        sole = np.zeros((n,2,3));sole[:,0,2]=.30
        joints = np.zeros((n,2,3,3));joints[:,:,0,2]=.85
        joints[:,:,1,2]=[.70,.42];joints[:,:,2,2]=[.40,.08]
        seen=np.ones((n,2),bool);move=np.tile([1.,0.],(n,1))
        shot=np.zeros(n);available=np.ones(n,bool)
        def infer(displacement, upright=1., seen=seen, raw_floor=None):
            p=sole.copy();p[:,:,2]+=displacement[:,None]
            j=joints.copy();j[:,:,:,2]+=displacement[:,None,None]
            g=p@u if raw_floor is None else raw_floor
            # The upstream terrain alignment can erase a small hop entirely.
            return support_observations(j,u,p,sole@u,np.full(n,upright),seen,g,move,available,shot,fps)
        zero=np.zeros(n)
        assert infer(zero)[:,1].all() and not infer(zero)[:,0].any()
        hop=zero.copy();middle=n//2
        hop[middle:middle+3]=[.025,.05,.025]
        assert not infer(hop)[hop>0].any()
        longer = zero.copy()
        longer[middle:middle+max(3,round(fps*.12))] = .025
        assert not infer(longer)[longer>0].any()
        assert infer(longer)[:middle-2,1].any()
        # Never make a new lock from a clipped shank or an above-ground flight pose.

        assert not infer(zero,upright=.32).any()
        assert not infer(zero,seen=np.zeros_like(seen)).any()
        assert not infer(zero,raw_floor=np.ones((n,2))*.35).any()


if __name__ == "__main__":
    test_observed_support_and_flight_release()
    test_observed_swing_requires_person_evidence_and_coherent_motion()
    test_planned_swing_and_double_support_survive_a_restarted_track()
    test_turn_support_tracks_weight_and_absolute_position()
    test_turn_smoothing_does_not_mix_cameras_or_freeze_a_turn()
    test_contact_solver_uses_the_exported_skin()
    print("PASS")
