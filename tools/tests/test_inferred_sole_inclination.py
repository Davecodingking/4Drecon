
import pathlib,sys
import numpy as np
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"pipeline"))
from leg_observations import inferred_sole_weights, sole_inclination

def test_observed_and_airborne_feet_remain_free():
    fixed=np.array([True,True,True,False])
    seen=np.array([[False,False],[True,False],[False,False],[False,False]])
    flight=np.array([False,False,True,False])
    stance=np.array([[True,False],[True,True],[True,True],[True,True]])
    w=inferred_sole_weights(fixed,seen,flight,stance)
    assert w[0,0]>w[0,1]>0 and w[1,0]==0 and w[1,1]>0 and not w[2:].any()

def test_inferred_foot_follows_its_planned_roll_and_allows_different_headings():
    up=torch.tensor([0.,0.,1.])
    flat=torch.tensor([[[[-1.,0.,0.],[-.5,0.,0.],[.5,.3,0.],[.5,-.3,0.],[1.,0.,0.]]]])
    angle=torch.tensor(0.5);c,s=torch.cos(angle),torch.sin(angle)
    zero=torch.tensor(0.);one=torch.tensor(1.)
    pitch=torch.stack([torch.stack([c,zero,-s]),torch.stack([zero,one,zero]),torch.stack([s,zero,c])])
    rolled=flat@pitch
    yaw=torch.tensor([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    target = sole_inclination(rolled,up)
    assert torch.allclose(sole_inclination(rolled@yaw,up),target)
    assert (sole_inclination(flat,up)-target).square().mean()>.02

if __name__=='__main__':
    test_observed_and_airborne_feet_remain_free()
    test_inferred_foot_follows_its_planned_roll_and_allows_different_headings()
    print('PASS inferred sole inclination')
