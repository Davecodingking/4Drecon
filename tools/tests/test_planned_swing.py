import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[2]/'pipeline'))
from planned_swing import contradictory_swing


def scene(n=24, fps=30):
    X=np.zeros((n,2,3,3)); X[:,0,:,0]=-.12; X[:,1,:,0]=.12
    X[:,:,:,1]=[.84,.44,.04]
    X[:,0,1:,1]=[.78,.40]; X[:,0,1:,2]=[.38,.43]
    sole=X[:,:,2].copy(); sole[:,:,1]-=.04
    return dict(raw_joints=X,raw_sole=sole,raw_gap=sole[:,:,1].copy(),upright=np.ones(n),
                joint_seen=np.ones((n,2,3),bool),thigh_confidence=np.ones((n,2)),
                world_hips=X[:,:,0].copy(),bone_lengths=np.full((2,2),.40),
                ankle_heights=np.full(2,.06),ground_height=lambda p:np.zeros(len(p)),
                planned=np.ones(n,bool),contact=np.ones((n,2),bool),shot=np.zeros(n,int),
                fps=fps,up=np.array([0.,1.,0.]))


def test_observed_lift_cropped_opposite_ankle_and_mirrored_side():
    for fps in [15,30,60]:
        x=scene(fps,fps)
        x['joint_seen'][:,1,1:]=False
        mask=contradictory_swing(**x)
        assert mask[:,0].all() and not mask[:,1].any()
        for key in ['raw_joints','raw_sole','raw_gap','joint_seen','thigh_confidence','world_hips','contact']:
            x[key]=x[key][:,::-1].copy()
        mask=contradictory_swing(**x)
        assert mask[:,1].all() and not mask[:,0].any()


def test_landing_or_unseen_support_knee_releases_immediately():
    x=scene();x['raw_gap'][14:,0]=0.;x['joint_seen'][5:8,1,0]=False
    mask=contradictory_swing(**x)
    assert mask[:5,0].all() and not mask[5:8].any() and mask[8:14,0].all() and not mask[14:].any()


def test_bridge_one_missing_shin_but_not_a_cut_or_long_gap():
    x=scene();x['joint_seen'][5,0,1]=False;x['joint_seen'][10:14,0,1]=False
    x['shot'][22:]=1
    mask=contradictory_swing(**x)
    assert mask[:10,0].all() and not mask[10:14].any() and mask[14:22,0].all() and not mask[22:].any()


def test_unequal_treads_flight_no_plan_and_already_free():
    x=scene();x['raw_gap'][:,0]=0.  # raised tread supports the high foot
    assert not contradictory_swing(**x).any()
    x=scene();x['raw_gap']+=.24      # both feet are clear of ground
    assert not contradictory_swing(**x).any()
    x=scene();x['planned'][:]=False
    assert not contradictory_swing(**x).any()
    x=scene();x['contact'][:,0]=False
    assert not contradictory_swing(**x).any()


def test_static_foot_infeasible_or_both_thighs_cropped():
    x=scene();x['world_hips'][:,:,1]+=.35
    assert not contradictory_swing(**x).any()
    x=scene();x['world_hips'][:,:,0]+=np.linspace(0,1.20,len(x['planned']))[:,None]
    assert not contradictory_swing(**x).any()
    x=scene();x['thigh_confidence'][:,1]=0
    assert not contradictory_swing(**x).any()


def test_short_grounded_handoff_and_release():
    from planned_swing import recovered_contacts
    x = scene(n=13)
    raised = np.zeros_like(x['contact']); raised[5:9,0] = True
    contact = x['contact'].copy(); contact[4:9,1] = False
    flight = np.zeros(13,bool)
    clearance = np.zeros_like(x['raw_gap'])
    def solve():
        return recovered_contacts(contact,raised,x['planned'],x['shot'],flight,clearance,
                                  x['bone_lengths'],x['fps'])
    fixed = solve()
    assert fixed[4:9,1].all() and not fixed[5:9,0].any()
    assert np.array_equal(fixed[9:], contact[9:])
    assert not contact[4:9,1].any()
    flight[4] = True
    assert not solve()[4,1]
    flight[4] = False
    clearance[4,1] = .20
    assert not solve()[4,1]
    clearance[4,1] = 0
    x['shot'][4:] = 1
    assert not solve()[4,1]
    x['shot'][:] = 0
    contact[2:4,1] = False
    assert not solve()[2:5,1].any()
    raised[:] = False
    assert np.array_equal(solve(),contact)



def test_pose_scope_and_release_velocity():
    from planned_swing import affected_legs, stance_transitions
    x = scene(n=60)
    old = np.zeros((60,2),bool); old[18:28,0] = True; old[12:19,1] = True; old[24:33,1] = True
    new = old.copy(); new[19:24,0] = False; new[19:24,1] = True
    raised = np.zeros_like(new); raised[19:24,0] = True
    scope = affected_legs(old,new,raised,x['shot'],30)
    assert np.all(scope[12:33,1] == 1) and np.all(scope[19:24] == 1)
    assert not scope[:4].any() and not scope[41:].any()
    shot = np.zeros(60,int); shot[33:] = 1
    flight = np.zeros(60,bool)
    weights = stance_transitions(new,raised,shot,flight,30)
    assert weights[11,1] and not weights[32,1] and not weights[:,0].any()
    shot[:] = 0
    weights = stance_transitions(new,raised,shot,flight,30)
    assert weights[32,1] == 1 and 0 < weights[33,1] < 1
    assert not weights[20:29,1].any()
    flight[32] = True
    assert not stance_transitions(new,raised,shot,flight,30)[32,1]



def test_unverified_edit_isolates_support_and_release():
    from planned_swing import affected_legs, stance_transitions
    shot=np.r_[np.zeros(20,int),np.ones(35,int)]
    old=np.zeros((len(shot),2),bool)
    old[8:40,1]=True; old[24:32,1]=False
    new=old.copy(); new[24:32,1]=True
    raised=np.zeros_like(old); raised[24:32,0]=True
    flight=np.zeros(len(shot),bool)
    for join,side in ((-1,1),(20,0)):
        a=affected_legs(old,new,raised,shot,30,join,side)
        w=stance_transitions(new,raised,shot,flight,30,join,side)
        assert not a[:20].any() and not w[:19].any()
    a=affected_legs(old,new,raised,shot,30,20,1)
    w=stance_transitions(new,raised,shot,flight,30,20,1)
    assert a[8:20,1].all() and w[7,1]>0


if __name__=='__main__':
    for name,fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn();print('ok',name)
