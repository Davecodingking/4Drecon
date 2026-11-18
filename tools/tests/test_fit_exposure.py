import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline'))
import fit_exposure as fit


class TestExposureFit(unittest.TestCase):
    def test_local_phase_preserves_motion_endpoints_and_limits_neighbor_step(self):
        held = np.zeros(19, bool)
        held[9] = True
        clock = fit.exposure_clock(held, np.ones(20,bool), np.zeros(20), 30)
        self.assertEqual(clock[9], clock[10])
        np.testing.assert_array_equal(clock[[0,-1]], [0,19])
        self.assertLess(np.diff(clock).max(), 1.2)
        self.assertGreaterEqual(np.diff(clock).min(), 0)

    def test_long_hold_does_not_pull_in_moving_action(self):
        held = np.zeros(29,bool)
        held[9:20] = True
        clock = fit.exposure_clock(held, np.ones(30,bool), np.zeros(30), 60)
        np.testing.assert_array_equal(clock[9:21], np.full(12, 14.5))
        self.assertLessEqual(np.max(np.abs(clock[:9]-np.arange(9))), .5)
        self.assertTrue((np.diff(clock)>=0).all())

    def test_contact_turn_flight_and_segment_edges_are_protected(self):
        held = np.ones(7,bool)
        eligible = np.array([False,False,True,True,False,True,False,False])
        segment = np.array([0,0,0,1,1,1,1,1])
        time = fit.exposure_clock(held,eligible,segment,30)
        np.testing.assert_array_equal(time, np.arange(8))

    @staticmethod
    def model():
        def rotations(p):
            return Rotation.from_euler('z',p[:,3:4]).as_matrix()
        def setroot(p,r=None,pos=None):
            q=p.copy()
            if r is not None:
                q[:,3]=Rotation.from_matrix(r).as_euler('xyz')[:,2]
            if pos is not None:
                q[:,:3]=pos
            return q
        def forward(shape,p,corrective,dev):
            loc=p[:,:3].copy()
            loc[:,2]+=p[:,4] + (.001 if corrective else 0)
            return loc[:,None,:],None
        return SimpleNamespace(root_pos=lambda p:p[:,:3],root_rot=rotations,set_root=setroot,
                               forward=forward,rest_verts=lambda *args:np.array([[0.,0.,0.]]))

    def test_subframe_fit_shares_root_and_legs_preserves_air_and_unrelated_channels(self):
        p=np.zeros((12,8))
        t=np.arange(12)
        p[:,0]=.04*t
        p[5:7,3]=np.deg2rad([178.,-178.])
        p[:,4]=.01*t
        p[:,7]=t
        p[-3:,2]=[.3,.42,.33]
        held=np.zeros(11,bool);held[[5,9]]=True
        allowed=np.ones(12,bool);allowed[-3:]=False
        with patch.object(fit,'M',self.model()),patch.object(fit,'L',SimpleNamespace(SIDES=['left','right'],LEG_COLS={'left':[4],'right':[5]})):
            q=fit.fit_exposure_states(p,None,np.zeros(12),30,np.array([0,0,1]),
                SimpleNamespace(model_h=lambda pts:np.zeros(len(pts))),'cpu',held,allowed)
        np.testing.assert_allclose(q[5,:7],q[6,:7],atol=1e-7)
        self.assertAlmostEqual(abs(q[5,3]),np.pi)
        np.testing.assert_array_equal(q[-3:],p[-3:])
        np.testing.assert_array_equal(q[:,7],p[:,7])
        self.assertAlmostEqual(q[0,0],p[0,0])
        self.assertAlmostEqual(q[-1,0],p[-1,0])

    def test_interpolated_stair_edge_must_be_feasible_for_entire_held_group(self):
        p=np.zeros((8,8));p[:,0]=np.arange(8)*.1
        held=np.zeros(7,bool);held[3]=True
        # Source endpoints have clean soles; the interpolation hits a steep ridge.
        def ridge(pts):return np.where((pts[:,0]>.34)&(pts[:,0]<.36), .15, 0.)
        with patch.object(fit,'M',self.model()),patch.object(fit,'L',SimpleNamespace(SIDES=['left','right'],LEG_COLS={'left':[4],'right':[5]})):
            q=fit.fit_exposure_states(p,None,np.zeros(8),30,np.array([0,0,1]),
                SimpleNamespace(model_h=ridge),'cpu',held,np.ones(8,bool))
        np.testing.assert_array_equal(q[3:5],p[3:5])



if __name__ == '__main__':
    unittest.main()
