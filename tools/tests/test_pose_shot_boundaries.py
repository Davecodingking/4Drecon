"""Pose repair uses real source time and never borrows another segment's pose."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.expanduser('~/sam3d/pipeline'))
import body_into_scene as B

def rotations(angles):
    out=[]
    for angle in np.deg2rad(angles):
        c,s=np.cos(angle),np.sin(angle)
        out.append(np.array([[c,0,s],[0,1,0],[-s,0,c]]))
    return out

def test_flip_detection_is_segment_local():
    angles=np.zeros(90);angles[45::2]=170
    labels=np.r_[np.zeros(45,int),np.ones(45,int)]
    assert B.flip_segments(rotations(angles),np.arange(90)/30.,segments=labels)==[]

def test_repair_anchors_do_not_cross_segment_boundaries():
    frames=list(range(12));t={f:f/30. for f in frames}
    root={f:np.array([t[f],0.,0.]) for f in frames}
    labels=np.r_[np.zeros(6,int),np.ones(6,int)]
    plan=B.occ_plan(frames,[(3,5),(6,8)],t,root,None,segments=labels)
    assert plan['5'][:3]==[2,2,0.]
    assert plan['6'][:3]==[9,9,0.]
    for frame,(left,right,weight,_) in plan.items():
        assert labels[int(frame)]==labels[left]==labels[right]

def test_interpolation_uses_source_seconds_with_sparse_frames():
    frames=[0,1,5,6,9];t={f:f/30. for f in frames}
    root={f:np.array([t[f],0.,0.]) for f in frames}
    plan=B.occ_plan(frames,[(1,3)],t,root,None,segments=np.zeros(5,int))
    assert abs(plan['5'][2]-5/9)<.000006
    assert np.allclose(plan['5'][3],root[5],atol=.000006)

def test_whole_unanchored_segment_is_not_fabricated():
    frames=list(range(8));t={f:f/30. for f in frames}
    root={f:np.array([t[f],0.,0.]) for f in frames}
    assert B.occ_plan(frames,[(4,7)],t,root,None,segments=[0]*4+[1]*4)=={}

def test_single_segment_legacy_behavior_is_unchanged_at_each_frame_rate():
    cases=[[0]*60,list(range(60)),[0]*20+[60,60]+[0]*20,
           [0]*20+[50,50,10,60,60,60,120,5,90]]
    for angles in cases:
        n=len(angles);rots=rotations(angles);frames=list(range(n))
        for fps in [6.,24.,30.,60.]:
            t={f:f/fps for f in frames};root={f:np.array([t[f],0.,0.]) for f in frames}
            legacy=B.occ_plan(frames,B.flip_segments(rots,list(t.values())),t,root,None)
            segmented=B.occ_plan(frames,B.flip_segments(rots,list(t.values()),segments=np.zeros(n,int)),t,root,None,segments=np.zeros(n,int))
            assert legacy==segmented

def test_invalid_segment_inputs_are_rejected():
    try:
        B.flip_segments(rotations([0,0]),[0.,1.],segments=[0])
        assert False,'mismatched labels accepted'
    except ValueError:
        pass
    try:
        B.occ_plan([0,1,2],[(0,1)],{0:0.,1:1.,2:2.},{0:np.zeros(3),1:np.zeros(3),2:np.zeros(3)},None,segments=[0,1,1])
        assert False,'cross-segment repair accepted'
    except ValueError:
        pass

def test_coherent_turn_around_a_fast_step_remains_observed():
    # A confident camera segment starts part way through a genuine turn.
    angles = np.r_[np.arange(0.,40.,5.),np.repeat(61.,25)]
    assert B.flip_segments(rotations(angles),np.arange(len(angles))/30.) == []
    # A one-frame jump to an unrelated pose is still repaired.
    outlier = np.r_[np.zeros(18),145.,np.zeros(18)]
    assert B.flip_segments(rotations(outlier),np.arange(len(outlier))/30.) == [(18,18)]


if __name__=='__main__':
    for name,fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn();print('ok',name)
