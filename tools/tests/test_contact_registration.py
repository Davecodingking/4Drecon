import os,sys
import numpy as np
sys.path.insert(0,os.path.expanduser("~/sam3d/pipeline"))
from register_layer_contact import infer_offset


def test_two_boundaries_and_a_stable_support_are_required():
    vf=np.arange(30); up=np.array([0.,1.,0.])
    shot=np.repeat([0,1,2],10)
    pos=np.zeros((30,3)); pos[10:20,0]=8.
    feet=np.repeat(pos[:,None,:],2,axis=1)
    feet[:,1,2]=0.3
    assert np.allclose(infer_offset(pos,feet,vf,shot,1,up,30),[-8,0,0])
    pos[20:,0]=5
    assert infer_offset(pos,feet,vf,shot,1,up,30) is None
    pos[20:,0]=0; feet[9,:,0]=1
    assert infer_offset(pos,feet,vf,shot,1,up,30) is None


if __name__=="__main__":
    test_two_boundaries_and_a_stable_support_are_required()
    print("PASS")
