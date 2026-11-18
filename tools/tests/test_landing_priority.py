import os,sys
import numpy as np
sys.path.insert(0,os.path.expanduser('~/sam3d/data/outdoor/scripts'))
import terrain

def test_platform_owns_its_footprint_and_lower_road():
 t=terrain.Terrain.__new__(terrain.Terrain);t.model_mesh=True;t.landing_priority=True
 t.e1,t.e2,t.up=np.eye(3);t.main=0;t.ST=np.array([[0,1,0,.25,0,12,-1,1,12]])
 t.extra=np.array([[3,4,-1,1,.75,-3],[4,7,-1,1,1.,-3]])
 t.plane_h=-3.;t.snap=[]
 pts=np.array([[1,0,0],[3.5,0,0],[5.,0,0],[6.9,0,0],[8.,0,0]])
 assert np.allclose(t.model_h(pts),[-3,.75,1.,1.,2.125])
if __name__=='__main__':test_platform_owns_its_footprint_and_lower_road();print('PASS')
