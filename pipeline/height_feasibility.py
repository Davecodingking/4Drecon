"""Bound a persistent standing-height contradiction before generating unseen legs."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


def supported_seated(seated,hips,knees,root,up,confidence,standing,ground,shot,fps):
    """Confirm the existing height-based flag using posture or stationary support."""
    mask=np.asarray(seated,bool)
    result=np.zeros_like(mask)
    thigh=hips-knees
    vertical=(thigh@up)/np.maximum(1e-5,np.linalg.norm(thigh,axis=2))
    observed=(confidence>.35) & (vertical<.67)
    low=hips.mean(1)@up-ground < .86*standing
    flat=root-(root@up)[:,None]*up
    edges=np.r_[0,np.flatnonzero(np.diff(shot))+1,len(root)]
    radius=max(3,round(.30*fps))
    for start,end in zip(edges[:-1],edges[1:]):
        for i in np.flatnonzero(mask[start:end])+start:
            local=flat[max(start,i-radius):min(end,i+radius+1)]
            stationary=np.max(np.linalg.norm(local-np.median(local,axis=0),axis=1))<.075*standing
            result[i]=observed[i].all() or (stationary and low[i] and np.max(vertical[i])<.86)
    return result


def height_bias(hips, knees, ankles, chest, sole_gap, up, ground, standing, shin, ankle_h,
                editable, thigh_confidence, shot, fps):
    """Return a low-frequency vertical bias; short movements remain in the source path."""
    center = hips.mean(axis=1)
    height = center @ up - ground
    trunk = chest-center
    erect = (trunk @ up)/np.maximum(np.linalg.norm(trunk,axis=1),1e-5)
    thigh = hips-knees
    thigh_drop = thigh @ up
    thigh_vertical = thigh_drop/np.maximum(np.linalg.norm(thigh,axis=2),1e-5)
    # A high, seen knee supplies positive evidence of knee flexion. Allow
    # asymmetric gestures when the other leg can support the body.
    squatting = ((thigh_confidence > .35) & (thigh_vertical<.56)).all(axis=1)
    eligible = editable & (erect>.83) & ~squatting
    # Neither missing ankles nor an upright trunk imply a planted foot. Look
    # for a sustained contradiction with the independently modeled surface.
    support_span = (hips-ankles) @ up + ankle_h
    pose_height = np.maximum(support_span.max(axis=1),
                             (thigh_drop+.94*np.asarray(shin)+ankle_h).max(axis=1))
    lower = np.clip(pose_height-.008, .69*standing, .97*standing)
    under = np.minimum(lower-height, -sole_gap.max(axis=1))
    over = np.minimum(height-.97*standing, sole_gap.min(axis=1))
    result = np.zeros(len(height))
    edges = np.r_[0,np.flatnonzero(np.diff(shot))+1,len(height)]
    window = max(3,2*round(.16*fps)+1)
    for start,end in zip(edges[:-1], edges[1:]):
        good=eligible[start:end] & np.isfinite(under[start:end]) & np.isfinite(over[start:end])
        # When both soles clear the tread together and return, preserve an
        # interior hip arc between near-ground endpoints even with cropped feet.
        gap=sole_gap[start:end].min(axis=1)
        for lo,hi in np.flatnonzero(np.diff(np.r_[False,gap>.035,False])).reshape(-1,2):
            if lo<1 or hi==end-start or hi-lo<3 or hi-lo>1.15*fps:
                continue
            h=height[start+lo-1:start+hi+1]
            if h.max()-max(h[0],h[-1])>.055 and min(h.argmax(),len(h)-1-h.argmax())>=2:
                good[lo:hi]=False
        for a,b in np.flatnonzero(np.diff(np.r_[False,good,False])).reshape(-1,2)+start:
            if b-a<window:
                continue
            sl=slice(a,b)
            low=median_filter(under[sl],window,mode='nearest')
            high=median_filter(over[sl],window,mode='nearest')
            # Suppress isolated penetrations/brief flight in a longer unsupported
            # stretch. Ease in to the residual at the uncertainty margin.
            for signal,minimum_time in ((low,.36),(high,.65)):
                intervals=np.flatnonzero(np.diff(np.r_[False,signal>.012,False])).reshape(-1,2)
                for lo,hi in intervals:
                    if hi-lo < round(minimum_time*fps):
                        signal[lo:hi]=0
            raw_high=np.maximum(over[sl],0)
            high=np.where(high>.012,.35*high+.65*raw_high,0)
            desire=np.maximum(low-.008,0)-np.maximum(high,0)
            estimate=gaussian_filter1d(desire,max(1,.10*fps),mode='nearest')
            # Fade only against protected poses in the same shot. Across an
            # edit either camera may independently supply a correct absolute height.
            ramp=np.arange(b-a)/max(2,round(.22*fps))
            if a>start:
                estimate*=np.clip(ramp,0,1)
            if b<end:
                estimate*=np.clip(ramp[::-1],0,1)
            result[sl]=estimate
    return result


def project(points, projection):
    homogeneous=np.einsum('nij,nqj->nqi',projection[:,:,:3],points)+projection[:,None,:,3]
    return homogeneous[:,:,:2]/np.maximum(homogeneous[:,:,2:],.01)


def fit_height_shift(root, landmarks, visible, rays, image_wh, desired, up, standing, ground_at, shot, fps,
                     joins=(), closeup_budget=None, source_pixels=None):
    """Fit depth along the visible body's ray, then bound the actual image displacement."""
    E,K,_,size=rays
    P=K @ E[:,:3]
    initial=project(landmarks,P)
    valid=visible & np.isfinite(initial).all(axis=2)
    n_seen=valid.sum(axis=1)
    if not n_seen.any():
        return np.zeros_like(root)
    center=(np.where(valid[:,:,None],landmarks,0).sum(axis=1)/np.maximum(1,n_seen[:,None]))
    camera=-np.einsum('nji,nj->ni',E[:,:3,:3],E[:,:3,3])
    ray=center-camera
    ray-=((ray@up)[:,None])*up
    ray/=np.maximum(np.linalg.norm(ray,axis=1)[:,None],1e-5)
    step=.001
    jac=(project(landmarks+step*ray[:,None],P)-initial)/step
    lift=(project(landmarks+desired[:,None,None]*up,P)-initial)
    if source_pixels is not None:
        # Pursue the original observations only while fixing a measurable height
        # discrepancy. Cap their influence when the camera fit was already poor.
        residual=initial-source_pixels
        magnitude=np.linalg.norm(residual,axis=2)
        residual*=np.minimum(1,.035*size[:,None]/np.maximum(magnitude,1e-6))[:,:,None]
        lift+=residual*np.clip(abs(desired)/(.08*standing),0,1)[:,None,None]
    jac=np.where(valid[:,:,None],jac,0)
    lift=np.where(valid[:,:,None],lift,0)
    # A shallow ray requires too large a displacement in depth. The metric term
    # penalizes a size change even if all the visible joints lie close together.
    close=np.clip(size/image_wh[1]-1.,0.,1.)
    regularizer=((.11-.085*close)*size/standing)**2*n_seen
    depth=-(lift*jac).sum(axis=(1,2))/np.maximum((jac*jac).sum(axis=(1,2))+regularizer,1e-8)
    limit=(.25+.45*close)*standing
    depth=np.clip(depth,-limit,limit)
    cuts=np.r_[0,np.flatnonzero(np.diff(shot))+1,len(root)]
    for a,b in zip(cuts[:-1],cuts[1:]):
        depth[a:b]=gaussian_filter1d(depth[a:b],max(.6,.077*fps),mode='nearest')
    depth=np.where((desired!=0)&(n_seen>0),depth,0)
    # Position and velocity already agree across verified camera edits. Depth
    # is measured in different rays on either side, so release it at the seam.
    for i in joins:
        fade=np.clip((np.abs(np.arange(len(root))-i+.5)-.5)/max(2,.24*fps),0,1)
        depth*=fade**2*(3-2*fade)
        if abs((root[i]-root[i-1])@up)<.045:
            desired=desired*fade**2*(3-2*fade)
    # The longer depth displacement is justified only on the same terrain.
    # Reduce the magnitude instead of borrowing an adjacent stair's elevation.
    base=ground_at(root)
    fraction=np.ones(len(root))
    for _ in range(7):
        shifted=root+(fraction*depth)[:,None]*ray
        other=np.abs(ground_at(shifted)-base)>.007
        if not np.any(other):
            break
        fraction[other]*=.5
    depth*=fraction
    depth[np.abs(ground_at(root+depth[:,None]*ray)-base)>.007]=0.
    if source_pixels is not None:
        # The centre of a head can follow its ray while its width changes
        # substantially. Bound the cluster's measured spread independently.
        def image_spread(points):
            mean=np.where(valid[:,:,None],points,0).sum(axis=1)/np.maximum(1,n_seen[:,None])
            variance=np.where(valid,((points-mean[:,None])**2).sum(axis=2),0).sum(axis=1)
            return np.sqrt(variance/np.maximum(1,n_seen))
        actual=image_spread(source_pixels)
        known=(n_seen>=2)&(actual>4.)
        before=np.abs(image_spread(initial)/np.maximum(1,actual)-1)
        allowance=np.maximum(before+.02,np.clip(before+.09,.095,.135))
        low=np.zeros(len(root))
        high=np.ones(len(root))
        for _ in range(9):
            mid=(low+high)/2
            at=landmarks+(desired[:,None]*up+(mid*depth)[:,None]*ray)[:,None]
            scale_error=np.abs(image_spread(project(at,P))/np.maximum(1,actual)-1)
            works=~known|(scale_error<=allowance)
            low=np.where(works,mid,low)
            high=np.where(works,high,mid)
        depth*=low
    delta=desired[:,None]*up+depth[:,None]*ray
    # Preserve absolute image alignment in a tight close-up. A body cropped
    # at both hips and knees may project to several times the frame height.
    budget=.12*size
    if closeup_budget is not None:
        budget=np.minimum(budget,closeup_budget*min(image_wh))
    for _ in range(3):
        diff=project(landmarks+delta[:,None],P)-initial
        worst=np.where(valid,np.linalg.norm(diff,axis=2),0).max(axis=1)
        delta*=np.minimum(1,budget/np.maximum(worst,1e-6))[:,None]
    delta[n_seen==0]=0
    return delta


def standing_root_shift(params, shape, skin, skin_delta, source, translation, focals, cameras,
                        masks, person_id, feet_ok, seated, frames, shot, fps, up, terrain,
                        protected, root_registration, measured, dev):
    """Translation in world metres for the root and its calibrated prior."""
    from pathlib import Path
    import torch
    from PIL import Image
    import mhr_body as M
    import legs_proc as L
    from root_registration import camera_root_rays
    root=M.root_pos(params)
    zero=np.zeros_like(root)
    possible=~(np.asarray(feet_ok,bool)|np.asarray(protected,bool))
    if (terrain is None or not getattr(terrain,'model_mesh',False) or not possible.any()
            or np.mean(seated)>L.SEATED_SKIP or not masks or not Path(masks[0]).is_file()):
        return zero
    with Image.open(masks[0]) as image:
        wh=image.size
    rays=camera_root_rays(list(zip(source,translation)),focals,cameras,wh)
    if rays is None:
        return zero
    g=L.geometry(shape,params[0],dev)
    names=g['names']
    body=L.make_body(g)
    joints=[[names.index(side+'_'+joint) for joint in ('upleg','lowleg','talocrural')] for side in L.SIDES]
    # Only source landmarks on the tracked actor may determine the translation
    # along the camera ray. A head-only close-up does not observe both ankles.
    tracked=[names.index(j) for j in ('c_head','c_neck','l_clavicle','r_clavicle','c_spine2',
                                      'l_upleg','r_upleg','l_lowleg','r_lowleg','l_talocrural','r_talocrural')]
    with torch.no_grad():
        def skeleton(parameters):
            p=torch.as_tensor(parameters,dtype=torch.float32,device=dev)
            return L.skel_torch(p,dev)[:,:,:3].cpu().numpy()/100
        world=skeleton(params)
        camera=skeleton(source)[:,tracked]@M.F.T+np.asarray(translation)[:,None]
    pixels=camera[:,:,:2]/np.maximum(.05,camera[:,:,2:])*np.asarray(focals)[:,None,None]+np.asarray(wh)/2
    valid=np.zeros(pixels.shape[:2],bool)
    for i in np.flatnonzero(possible):
        try:
            with Image.open(masks[i]) as image:
                lab=np.asarray(image)
        except FileNotFoundError:
            possible[i]=False
            continue
        if lab.shape != wh[::-1] or not np.isfinite(pixels[i]).all():
            possible[i]=False
            continue
        coords=np.rint(pixels[i]).astype(int)
        inside=(coords>=0).all(1)&(coords<wh).all(1)&(camera[i,:,2]>.05)
        for j in np.flatnonzero(inside):
            x,y=coords[j]
            valid[i,j]=(lab[max(0,y-2):y+3,max(0,x-2):x+3]==person_id).any()
    possible &= ~valid[:,-2:].any(1) & valid[:,:-2].any(1)
    if not possible.any():
        return zero
    sk=world[:,joints]
    hips=sk[:,:,0]
    center=hips.mean(axis=1)
    rest=M.rest_verts(shape,dev)
    foot=rest[:,1]<rest[:,1].min()+10.
    gaps=[]
    for sign in (1,-1):
        sole=skin[:,foot & (sign*rest[:,0]>0)]+skin_delta[:,None]
        floor=terrain.model_h(sole.reshape(-1,3)).reshape(sole.shape[:2])
        gaps.append((sole@up-floor).min(axis=1))
    confidence=(np.zeros((len(root),2)) if measured is None else measured['confidence'])
    ground=terrain.model_h(center)
    possible &= ~supported_seated(seated,hips,sk[:,:,1],root,up,confidence,body.pelvis_h,ground,shot,fps)
    desired=height_bias(hips,sk[:,:,1],sk[:,:,2],world[:,names.index('c_spine3')],
                        np.column_stack(gaps),up,ground,body.pelvis_h,body.shin,
                        body.ankle_h,possible,confidence,shot,fps)
    frame_to_index={int(frame):i for i,frame in enumerate(frames)}
    joins=sorted({frame_to_index[frame] for r in root_registration for frame in r.get('joins',())
                  if frame in frame_to_index and frame_to_index[frame]>0})
    return fit_height_shift(root,world[:,tracked[:-2]],valid[:,:-2],rays,wh,desired,up,
                            body.pelvis_h,terrain.model_h,shot,fps,joins,source_pixels=pixels[:,:-2])
