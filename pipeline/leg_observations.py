"""Image observations of a thigh remain useful when the foot leaves the frame."""
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d, median_filter


def thigh_confidence(uv, depth, height, masks, person_id, image_wh, shot, fps):
    n = len(uv)
    confidence = np.zeros((n, 2), np.float32)
    if masks is None or person_id is None:
        return confidence
    width, rows = image_wh
    hip, knee = uv[:,:,0], uv[:,:,1]
    femur = (knee-hip)/height[:,None,None]
    finite = np.isfinite(femur).all(-1) & (depth[:,:,:2] > .05).all(-1)
    lengths = np.linalg.norm(knee-hip,axis=-1)
    for i, path in enumerate(masks):
        if not finite[i].any() or not np.isfinite(height[i]):
            continue
        try:
            with Image.open(path) as image:
                labels = np.asarray(image)
        except FileNotFoundError:
            continue
        if labels.shape != (rows,width):
            continue
        radius = max(1, round(.004*height[i]))
        for side in np.flatnonzero(finite[i] & (lengths[i] >= 8)):
            points = np.linspace(hip[i,side], knee[i,side], 5)
            if (points < 1).any() or (points >= [width-1, rows-1]).any():
                continue
            # Check the segment, not just two endpoints on opposite sides
            # of a foreground obstruction.
            samples = []
            for x,y in np.rint(points).astype(int):
                patch = labels[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]
                samples.append(np.any(patch == person_id))
            if samples[0] and samples[-1] and sum(samples) >= 4:
                confidence[i,side] = np.clip((lengths[i,side]/height[i]-.055)/.055,0.,1.)
    edges = np.r_[0,np.flatnonzero(np.diff(shot))+1,n]
    for lo,hi in zip(edges[:-1], edges[1:]):
        for side in range(2):
            valid = confidence[lo:hi,side] > 0
            bounds = np.flatnonzero(np.diff(np.r_[False,valid,False]))
            for a,b in bounds.reshape(-1,2)+lo:
                if b-a < max(2,round(.08*fps)):
                    confidence[a:b,side] = 0
                    continue
                direction = femur[a:b,side]
                local = median_filter(direction,size=(3,1),mode="nearest")
                confidence[a:b,side] *= np.linalg.norm(direction-local,axis=-1) < .055
    return confidence


def measured_thighs(params, translation, focal, cameras, masks, person_id, shot, fps, dev):
    import mhr_body as M
    import legs_proc as L
    if masks is None or len(masks)==0 or person_id is None or focal is None:
        return None
    try:
        with Image.open(masks[0]) as image:
            image_wh = image.size
    except FileNotFoundError:
        return None
    skeleton = M.skeleton(dev)
    names = list(skeleton["names"])
    ids = [[names.index(side+"_"+joint) for joint in ("upleg","lowleg","talocrural")]
           for side in L.SIDES]
    with torch.no_grad():
        source = L.skel_torch(torch.as_tensor(np.asarray(params),dtype=torch.float32,device=dev), dev)
        xyz = source[:,ids,:3].cpu().numpy()/100.
    camera_xyz = xyz @ M.F.T + np.asarray(translation)[:,None,None,:]
    z = camera_xyz[:,:,:,2]
    focal = np.asarray(focal,dtype=float)
    focal = np.where(np.isfinite(focal) & (focal>0),focal,np.nan)
    size = np.maximum(20.,1.7*focal/np.maximum(.1,np.mean(z[:,:,0],axis=1)))
    uv = camera_xyz[:,:,:,:2] / np.maximum(z[:,:,:,None],.05) * focal[:,None,None,None] + np.asarray(image_wh)/2
    confidence = thigh_confidence(uv,z,size,masks,person_id,image_wh,shot,fps)
    if not confidence.any():
        return None
    K = np.asarray([c["intrinsic_3x3"] for c in cameras],dtype=float).copy()
    hw = np.asarray([c["input_hw"] for c in cameras])
    K[:,:2] *= (np.asarray(image_wh)[None,:]/hw[:,::-1])[:,:,None]
    E = np.asarray([c["extrinsic_w2c_4x4"] for c in cameras])
    relative=(uv[:,:,1]-uv[:,:,0])/size[:,None,None]
    return dict(confidence=confidence, relative=np.nan_to_num(relative),
                size=np.where(np.isfinite(size),size,1.), projection=K @ E[:,:3,:])


def projected_thigh_error(hip, knee, projection, size, relative):
    # Vertices are in scene metres. Compare image-plane offsets relative to
    # the corresponding hip, so this term supplies no absolute depth target.
    a = torch.einsum('tij,tsj->tsi',projection[:,:,:3],hip)+projection[:,None,:,3]
    b = torch.einsum('tij,tsj->tsi',projection[:,:,:3],knee)+projection[:,None,:,3]
    measured = (b[:,:,:2]/b[:,:,2:].clamp(min=.1)-a[:,:,:2]/a[:,:,2:].clamp(min=.1))/size[:,None,None]
    return (measured-relative).square().sum(-1)


def root_displacement(delta, reference, fixed):
    """An inferred stride cannot update the independently estimated actor path."""
    return torch.where(fixed[:,None],reference,delta)


def measured_root_trajectory(root, shot, fps, held=None):
    """Denoise camera-relative estimation errors without a foot/stride target."""
    result = root.copy()
    edges = np.r_[0,np.flatnonzero(np.diff(shot))+1,len(root)]
    for lo,hi in zip(edges[:-1],edges[1:]):
        if hi-lo < 3:
            continue
        local = root[lo:hi]
        despiked = median_filter(local, size=(3,1), mode="nearest")
        result[lo:hi] = gaussian_filter1d(despiked,max(.5,.062*fps),axis=0,mode="nearest")
        if held is not None:
            intervals = np.flatnonzero(np.diff(np.r_[False,held[lo:hi-1],False])).reshape(-1,2)
            for a,b in intervals:
                sl = slice(lo+a,lo+b+1)
                result[sl] = result[sl].mean(0)
    return result


def stance_groups(contact, shot, registered_frame, registered_side):
    """One pivot per footfall. Only a registered plant can span a camera edit."""
    groups = np.zeros_like(contact,dtype=np.int64)
    count = 0
    cuts = np.r_[False,np.diff(shot)!=0]
    for side,stance in enumerate(contact.T):
        breaks = cuts.copy()
        if side == registered_side and 0 <= registered_frame < len(shot):
            breaks[registered_frame] = False
        new = stance & (breaks | ~np.r_[False,stance[:-1]])
        local = np.cumsum(new)
        groups[stance,side] = count+local[stance]
        count += int(new.sum())
    return groups, count+1

def registered_root_prior(root, shot, fps, held, joins, calibrated, up, flight=None):
    """Carry measured camera motion through verified continuous contacts."""
    base = measured_root_trajectory(root, shot, fps, held)
    strength = np.zeros(len(root))
    if calibrated is None or not len(joins):
        return base, strength
    # Reconcile drift in the ground plane. Keep the solver's grounded body
    # height as the vertical observation, including genuine takeoff and stairs.
    source = calibrated + ((root-calibrated) @ up)[:,None]*up
    radius = max(2, round(.36*fps))
    cuts = np.flatnonzero(np.diff(shot))+1
    separate = np.r_[0,np.setdiff1d(cuts,joins),len(root)]
    if flight is not None:
        flight = np.asarray(flight, bool)
        separate = np.union1d(separate, np.flatnonzero(np.diff(flight))+1)
    for start,end in zip(separate[:-1],separate[1:]):
        verified = joins[(joins>start)&(joins<end)]
        if not len(verified) or (flight is not None and flight[start]):
            continue
        local = source[start:end]
        path = gaussian_filter1d(median_filter(local,size=(3,1),mode='nearest'),
                                 max(.7,.095*fps),axis=0,mode='nearest')
        indices = np.arange(start,end)
        distance = np.min(np.abs(indices[:,None]-(verified-.5)),axis=1)
        phase = np.clip(1-distance/radius,0,1)
        weight = phase**2*(3-2*phase)
        strength[start:end] = weight
        base[start:end] += weight[:,None]*(path-base[start:end])
    # A source exposure remains one instant even inside the seam transition.
    if held is not None:
        intervals = np.flatnonzero(np.diff(np.r_[False,held,False])).reshape(-1,2)
        for lo,hi in intervals:
            if strength[lo:hi+1].any():
                base[lo:hi+1] = base[lo:hi+1].mean(0)
    return base, strength

def inferred_sole_weights(fixed_root, ankle_seen, airborne, contact):
    """Use the planned roll only on legs without a measured ankle or flight."""
    eligible = fixed_root[:, None] & ~ankle_seen & ~airborne[:, None]
    return eligible * np.where(contact, 1., .35)


def sole_inclination(points, up):
    """Vertical components of longitudinal and transverse plantar directions."""
    heel, inner, outer, toe = (points[..., index, :] for index in (0, 2, 3, 4))
    axes = torch.stack(((inner+outer)*.5-heel, toe-heel, outer-inner),dim=-2)
    return torch.nn.functional.normalize(axes,dim=-1) @ up

