"""Regression for fixed pelvis + inferred sole collision on both delivery skins."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pipeline'))
import mhr_body as M
from sole_clearance import resolve_inferred_sole_clearance


def test_fixed_pelvis_on_flat_ground():
    import torch
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    n=22
    shape=np.zeros(M.NSHAPE)
    pose=np.zeros((n,M.NPOSE+M.NSCALE))
    pose[:,[62,53]]=np.deg2rad(36.)
    pose[:,[61,52]]=np.deg2rad(-16.)
    up=np.array([0.,1.,0.])
    rest=M.rest_verts(shape,dev)
    sole=np.flatnonzero(rest[:,1]<rest[:,1].min()+10)
    def heights(q):
        return np.stack([(M.forward(shape,q,skin,dev)[0][:,sole]@up).min(1)
                         for skin in (True,False)])
    level=heights(pose[:1]).min()
    class Flat:
        def model_h(self, xyz):
            return np.full(len(xyz),level)

    # A smooth stance dip of 5 cm + an isolated excluded turn and real flight.
    dip=np.zeros(n)
    dip[6:17]=.05*np.sin(np.linspace(0,np.pi,11))**2
    dip[2:4]=.025
    dip[-3:]=-.12
    pose=M.set_root(pose,None,M.root_pos(pose)-dip[:,None]*up)
    done=np.arange(n)>0
    done[-3:]=False
    turn=np.zeros(n,bool); turn[2:4]=True
    plan={'done':done,'turning':turn}
    # Treat the freely airborne frames as eligible to check that the flight
    # safeguard protects them, even when the planner mislabels a stance.
    flight = np.zeros(n, bool)
    flight[10:12] = True
    flight[-3:] = True
    done[-3:] = True
    before=heights(pose)-level
    assert before[:,6:17].min()<-.049
    corrected=resolve_inferred_sole_clearance(pose,shape,up,Flat(),plan,[],30.,dev,flight)
    np.testing.assert_array_equal(corrected[-3:],pose[-3:])
    after=heights(corrected)-level
    assert after[:,done&~turn].min()>-.005, after
    np.testing.assert_array_equal(corrected[~done|turn],pose[~done|turn])
    np.testing.assert_array_equal(M.root_pos(corrected),M.root_pos(pose))
    kept=np.setdiff1d(np.arange(pose.shape[1]),[59,60,61,62,64,124,50,51,52,53,55,128])
    np.testing.assert_array_equal(corrected[:,kept],pose[:,kept])
    delta=np.rad2deg(corrected[:,[62,53]]-pose[:,[62,53]])
    assert np.abs(np.diff(delta,n=2,axis=0)).max()<9, delta
    assert np.array_equal(resolve_inferred_sole_clearance(pose,shape,up,Flat(),None,[],30.,dev),pose)


def test_contact_pivot_tracks_the_same_material_across_a_rolling_foot():
    from sole_clearance import shared_contact_vertices
    # The heel is lowest on the first frame, the toe on the next. A vertex
    # at the ball of the foot remains within a millimetre of both contacts.
    gap = np.array([[.0,.0006,.010],[.008,.0010,.0],[.010,.0007,.0]])
    ids = np.array([11,23,81])
    np.testing.assert_array_equal(shared_contact_vertices(gap,ids,count=1)[:,0],[23,81])
    # A common height offset affects clearance, not the identity of the pivot.
    np.testing.assert_array_equal(shared_contact_vertices(gap+[[.2],[.1],[.05]],ids,1)[:,0],[23,81])
    assert shared_contact_vertices(gap,ids,8).shape==(2,3)

if __name__=='__main__':
    test_contact_pivot_tracks_the_same_material_across_a_rolling_foot()
    test_fixed_pelvis_on_flat_ground()
    print('PASS inferred dual-skin clearance, fixed root, turn, flight, angular continuity')
