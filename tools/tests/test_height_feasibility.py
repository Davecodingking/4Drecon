from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"pipeline"))
import numpy as np
from height_feasibility import height_bias,fit_height_shift,project,supported_seated


def bias(height, gap, cropped=True, squat=False):
    n=len(height); up=np.array([0.,1.,0.]); hips=np.zeros((n,2,3))
    hips[:,:,1]=height[:,None]
    drop=.12 if squat else .33
    knees=hips-np.array([0,drop,(.33**2-drop**2)**.5])
    ankles=knees-np.array([0,.33,0])
    return height_bias(hips,knees,ankles,hips.mean(1)+[0,.45,0],np.tile(gap[:,None],(1,2)),
                       up,np.zeros(n),.74,[.33,.33],.075,np.broadcast_to(cropped,(n,)),
                       np.ones((n,2)),np.zeros(n,int),30)


def test_sustained_cropped_geometry_and_visible_squat():
    high=np.full(80,.87)
    assert -.17<bias(high,np.full(80,.15))[40]<-.13
    low=np.full(80,.50)
    assert bias(low,np.full(80,-.19))[40]>.16
    assert np.all(bias(low,np.full(80,-.19),cropped=False)==0)
    assert np.all(bias(low,np.full(80,-.19),squat=True)==0)


def test_brief_and_long_cropped_flight_and_observed_contact_are_unchanged():
    for duration in (8,24):
        rise=np.zeros(100);rise[30:30+duration]=.24*np.sin(np.linspace(0,np.pi,duration))
        np.testing.assert_array_equal(bias(.74+rise,rise),0)
    np.testing.assert_array_equal(bias(np.full(80,.51),np.zeros(80)),0)


def test_interleaved_known_seat_or_registered_turn_stays_exact():
    n=100; eligible=np.ones(n,bool); eligible[40:56]=False
    result=bias(np.full(n,.50),np.full(n,-.19),cropped=eligible)
    assert result[25]>.15 and result[75]>.15
    np.testing.assert_array_equal(result[40:56],0)


def test_stationary_and_visible_seat_protected_after_a_walk():
    n=90; root=np.zeros((n,3));root[:,1]=.54
    root[:35,0]=np.arange(35)*.03;root[35:,0]=root[34,0]
    hips=np.repeat(root[:,None],2,axis=1)
    knees=hips-[0,.21,.25]; conf=np.zeros((n,2)); conf[25:30]=1
    flagged=np.ones(n,bool); flagged[70:]=False
    result=supported_seated(flagged,hips,knees,root,np.array([0,1,0]),conf,.74,np.zeros(n),np.zeros(n),30)
    assert not result[15] and result[25:30].all() and result[50:64].all() and not result[72:].any()


def test_camera_budget_and_continuous_cut_on_same_tread():
    n=50
    root=np.tile([0.,0.,3.],(n,1))
    E=np.broadcast_to(np.eye(4),(n,4,4)).copy()
    K=np.broadcast_to(np.array([[800.,0,320],[0,800.,240],[0,0,1]]),(n,3,3)).copy()
    pts=root[:,None]+np.array([[[0,.4,0],[.15,.3,0]]])
    valid=np.ones((n,2),bool)
    shot=np.arange(n)>=25; desired=np.where(shot,-.12,.16)
    rays=(E,K,np.zeros((n,2)),np.full(n,450.))
    terrain=lambda p: (p[:,2] < 2.995)*.15
    delta=fit_height_shift(root,pts,valid,rays,(640,480),desired,np.array([0.,1.,0.]),.74,
                           terrain,shot,30,joins=[25],closeup_budget=.04)
    np.testing.assert_array_equal(terrain(root+delta),terrain(root))
    np.testing.assert_allclose(delta[24:26],0)
    assert delta[10,1]>.05 and delta[35,1]<-.05
    drift=np.linalg.norm(project(pts+delta[:,None],K@E[:,:3])-project(pts,K@E[:,:3]),axis=2)
    assert drift.max()<=.04*480+.01

if __name__=="__main__":
    test_sustained_cropped_geometry_and_visible_squat()
    test_brief_and_long_cropped_flight_and_observed_contact_are_unchanged()
    test_interleaved_known_seat_or_registered_turn_stays_exact()
    test_stationary_and_visible_seat_protected_after_a_walk()
    test_camera_budget_and_continuous_cut_on_same_tread()
    print("PASS height feasibility")
