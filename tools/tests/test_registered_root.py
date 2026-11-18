"""Synthetic ground truth for cut registration before feet are planned."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline'))
from root_registration import registered_horizontal_path

class Surface:
    e1, e2 = np.eye(3)[:2]
    ST = np.array([[0, 1., 0.]])
    main = 0
    extra = np.array([[-2., 6., -2., 2., 0., 0.]])
    def model_h(self, coords):
        return np.zeros(len(coords))

def test_bad_camera_interval():
    frames = np.arange(130)
    shot = np.searchsorted([35,75,105], frames, side='right')
    bad = (frames >= 35) & (frames < 85)
    root = np.c_[.025*frames, frames*0., 1.+.05*np.sin(frames/12)]
    tracked = root.copy()
    tracked[35:75] += [.85,.30,0]
    tracked[75:85] += [-.65,.10,0]
    foot = np.broadcast_to(root[:,None,:], (len(root),2,3)).copy()
    foot[:,:,2]=[0,.12]
    torso=np.broadcast_to([[-.10,0,0],[.10,0,0],[-.15,0,.52],[.15,0,.52]],(len(root),4,3)).copy()
    base=[tracked,frames,shot,bad,np.array([0.,0.,1.]),30.,Surface(),{'shot':4,'support_side':0},torso,foot]
    def check(args):return registered_horizontal_path(*args)[0]
    np.testing.assert_allclose(check(base),root,atol=.045)
    np.testing.assert_allclose(check(base)[~bad],tracked[~bad],atol=1e-9)
    np.testing.assert_allclose(check(base)[:,2],tracked[:,2],atol=1e-9)
    for idx,val in ((7,None),(1,frames+np.where(frames>=75,1,0)),
                    (3,np.zeros_like(bad)),(8,torso+np.where(frames[:,None,None]>=75,[0,0,.38],0))):
        args=base.copy(); args[idx]=val
        np.testing.assert_array_equal(check(args),tracked)
    args=base.copy(); elevated=foot.copy(); elevated[74:77,:,2]+=.4; args[-1]=elevated
    np.testing.assert_array_equal(check(args),tracked)

def test_verified_reverse():
    frames=np.arange(120); shot=(frames>=50).astype(int); root=np.c_[.012*frames,np.zeros(120),np.full(120,1.)]
    root[50:] -= [.9,.3,0]; foot=np.repeat(root[:,None],2,axis=1); foot[:,:,2]=[0,.1]
    t=np.zeros((120,4,3)); t[:,2:,2]=.5
    base=[root,frames,shot,np.zeros(120,bool),np.array([0.,0.,1.]),30.,Surface(),
          {'shot':1,'support_side':0,'source':'single-actor planted-foot continuity; reverse boundary checked'},t,foot]
    result,rep=registered_horizontal_path(*base)
    np.testing.assert_allclose(result[50]-result[49],[.012,0,0],atol=.005)
    np.testing.assert_array_equal(result[:50],root[:50]); np.testing.assert_array_equal(result[-12:],root[-12:])
    np.testing.assert_array_equal(result[:,2],root[:,2])
    assert np.linalg.norm(np.diff(result[50:],axis=0),axis=1).max()<.06
    # Neither the identity nor apparent model position alone establishes a match.
    for idx,val in ((7,{'shot':1}),(9,foot+np.array([0.,0.,.4])),(1,frames+(frames>=50)*2)):
        args=base.copy(); args[idx]=val
        np.testing.assert_array_equal(registered_horizontal_path(*args)[0],root)

    # Non-coincident image rays cannot both be fit exactly at an edit. Share
    # the discrepancy locally while retaining the measured incoming movement.
    n=len(root); E=np.tile(np.eye(4),(n,1,1)); E[:,2,3]=3.
    K=np.tile(np.diag([800.,800.,1.]),(n,1,1))
    target=200*root[:,:2]; sizes=np.full(n,340.)
    adjusted,_=registered_horizontal_path(*base,rays=(E,K,target,sizes))
    def loss(world):return np.mean(np.linalg.norm(200*world[:,:2]-target,axis=1)[43:80])
    assert loss(adjusted)<.65*loss(result)
    np.testing.assert_allclose(adjusted[50]-adjusted[49],[.012,0,0],atol=.004)
    np.testing.assert_allclose(adjusted[:32],root[:32],atol=1.e-9)
    np.testing.assert_array_equal(adjusted[:,2],root[:,2])

if __name__=='__main__':
    test_bad_camera_interval();test_verified_reverse();print('OK')
