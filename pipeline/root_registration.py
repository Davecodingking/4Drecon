"""Register horizontal actor positions where cameras share a measured landing."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


def camera_root_rays(track, focals, cameras, image_size):
    """Observed root pixels and the actual world-camera projection in one image grid."""
    import mhr_body as M
    if focals is None or any(f is None or not np.isfinite(f) or f<=0 for f in focals):
        return None
    wh = np.asarray(image_size)
    E = np.stack([c['extrinsic_w2c_4x4'] for c in cameras])
    K = np.array([c['intrinsic_3x3'] for c in cameras],float)
    sizes = np.asarray([c['input_hw'][::-1] for c in cameras])
    K[:,:2] *= (wh/sizes)[:,:,None]
    camera = M.root_pos(np.stack([row[0] for row in track])) @ M.F.T + np.stack([row[1] for row in track])
    if not np.isfinite(camera).all() or np.any(camera[:,2] <= .05):
        return None
    scale = np.asarray(focals)/camera[:,2]
    return E,K,camera[:,:2]*scale[:,None] + wh/2,np.maximum(1.7*scale,20.)


def _fit_rays(seed, prior, rays, editable, joins, flight, basis, fps):
    """Fit editable horizontal positions in both views; carry the seam velocity."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    E,K,pixels,height = rays
    n=len(seed)
    identity=sparse.eye(n,format='csr')
    D=identity[1:]-identity[:-1]
    D2=sparse.kron(D[1:]-D[:-1],sparse.eye(2),format='csr')
    D=sparse.kron(D,sparse.eye(2),format='csr')
    # The seed already locates discontinuities and retains flight displacement.
    # Strongly preserve that displacement across a verified edit or flight.
    weight=np.full(n-1,.8*fps/30)
    weight[joins-1]=100
    weight[flight[1:] & flight[:-1]]=100
    velocity=sparse.diags(weight.repeat(2))@D
    acceleration=5*(fps/30)**2*D2
    metric=.18*sparse.eye(n*2,format='csr')
    columns=np.flatnonzero(np.repeat(editable,2))
    if not len(columns):
        return seed
    fit=seed.copy()
    for _ in range(4):
        cam=np.einsum('nij,nj->ni',E[:,:3,:3],fit)+E[:,:3,3]
        homog=np.einsum('nij,nj->ni',K,cam)
        if not np.isfinite(homog).all() or np.any(homog[:,2]<=.05):
            return seed
        uv=homog[:,:2]/homog[:,2,None]
        grad=(K[:,:2]-uv[:,:,None]*K[:,2:3])@E[:,:3,:3]@basis
        grad/= (homog[:,2]*height)[:,None,None]
        jac=sparse.block_diag(grad,format='csr')
        matrix=sparse.vstack([jac,metric,velocity,acceleration],format='csr')[:,columns]
        residual=np.r_[((pixels-uv)/height[:,None]).ravel(),-metric@((fit-prior)@basis).ravel(),
                       -velocity@((fit-seed)@basis).ravel(),-acceleration@(fit@basis).ravel()]
        update=np.zeros(n*2)
        update[columns]=spsolve((matrix.T@matrix).tocsc(),matrix.T@residual)
        fit+=update.reshape(n,2)@basis.T
        if np.max(np.abs(update))<1.e-5:
            break
    return fit


def registered_horizontal_path(root, frames, shot, bad, up, fps, terrain, registration, torso, feet, rays=None,
                               feet_visible=None):
    """Return the world root before IK, preserving all measured vertical motion.

    Torso points are pelvis-relative (hips and clavicles). Feet are the two
    lowest skinned sole points. Bounds and elevations come from the *registered*
    surface, so an arbitrary flat area in another view cannot join the tracks.
    """
    out = root.copy()
    report = []
    if not registration or terrain is None or not len(getattr(terrain, 'extra', [])):
        return out, report
    landing = terrain.extra[-1]
    stair = terrain.ST[terrain.main]
    along = stair[1]*terrain.e1 + stair[2]*terrain.e2
    across = -stair[2]*terrain.e1 + stair[1]*terrain.e2

    def on_landing(x):
        s, w = x@along, x@across
        return (s>=landing[0]) & (s<=landing[1]) & (w>=landing[2]) & (w<=landing[3])

    foot_gap = feet @ up - landing[4]
    support = on_landing(feet) & (foot_gap >= -.25) & (foot_gap <= .08)
    grounded = np.any(support, axis=1)
    flight = np.all(on_landing(feet) & (foot_gap>.08),axis=1)
    if feet_visible is not None:
        flight &= np.asarray(feet_visible,bool)
    flat = root - (root@up)[:,None]*up
    look = max(2,round(.14*fps))
    cuts = np.flatnonzero(np.diff(shot))+1
    ranges = np.r_[0,cuts,len(root)]
    editable = np.zeros(len(root),bool)
    verified_joins = []

    def velocity(coords, lo, hi):
        return np.median(np.diff(coords[lo:hi],axis=0),axis=0)

    def smoothstep(t):
        return t*t*(3.-2.*t)

    # An unlocalized close-up may span multiple cuts and end partway into a
    # good view. Transfer the *within-shot* displacements between two trusted
    # world anchors. Never relocate the adjacent trusted shots.
    changes = np.flatnonzero(np.diff(np.r_[False,bad,False]))
    for lo,hi in changes.reshape(-1,2):
        if lo<2 or hi+1>=len(root) or not np.all(np.diff(frames[lo-2:hi+2])==1):
            continue
        seams = cuts[(cuts>=lo)&(cuts<=hi)]
        if not len(seams):
            continue
        joins = np.unique(np.r_[lo,seams,hi])
        endpoints = np.r_[joins,joins-1]
        if not np.all(on_landing(root[endpoints]) & grounded[endpoints]):
            continue
        # An actual time/location edit can show the same person. Require
        # matching torso orientation and posture in both world views.
        span = np.linalg.norm(torso[seams,2:].mean(1)-torso[seams,:2].mean(1),axis=1)
        change = np.linalg.norm(torso[seams]-torso[seams-1],axis=2).max(1)
        if np.any(change > .23*span):
            continue
        if np.max(np.linalg.norm(flat[joins]-flat[joins-1],axis=1)) < .12:
            continue
        obs = flat.copy()
        for start,end in zip(joins[:-1],joins[1:]):
            if end-start >= 4:
                # Bad extrinsics are not smoothed in the world-body solver.
                # Reject only the high-frequency horizontal range jitter here.
                short = median_filter(flat[start:end],size=(3,1),mode='nearest')
                obs[start:end] = gaussian_filter1d(short,max(.5,.045*fps),axis=0,mode='nearest')
                obs[start:end][flight[start:end]]=flat[start:end][flight[start:end]]
        previous = obs[lo-1]
        left = max(ranges[np.searchsorted(ranges,lo-1,side='right')-1],lo-1-look)
        if lo-left<2:
            continue
        v = velocity(obs,left,lo)
        joined = obs[lo:hi].copy()
        for start,end in zip(joins[:-1],joins[1:]):
            if end-start<2:
                break
            incoming = velocity(obs,start,min(start+look+1,end))
            joined[start-lo:end-lo] += previous + .5*(v+incoming) - obs[start]
            previous = joined[end-lo-1]
            v = velocity(obs,max(start,end-1-look),end)
        else:
            right = min(ranges[np.searchsorted(ranges,hi,side='right')],hi+look+1)
            if right-hi<2:
                continue
            finish = obs[hi] - .5*(v+velocity(obs,hi,right))
            # Release a calibration discrepancy on supported frames. A common
            # flight retains its velocity instead of absorbing part of the fit.
            travel = np.r_[0., np.cumsum(~(flight[lo+1:hi] & flight[lo:hi-1]))]
            if travel[-1] <= 0:
                continue
            joined += smoothstep(travel/travel[-1])[:,None]*(finish-joined[-1])
            offset = joined-flat[lo:hi]
            candidate = root[lo:hi]+offset
            if np.max(np.abs(terrain.model_h(candidate)-terrain.model_h(root[lo:hi]))) > .025:
                continue
            out[lo:hi] = candidate
            before = ranges[np.searchsorted(ranges,lo-1,side='right')-1]
            editable[max(before,lo-round(.45*fps)):hi] = True
            verified_joins.extend(joins)
            report.append(dict(kind='bracketed_bad_camera', first=int(frames[lo]), last=int(frames[hi-1]),
                               joins=frames[joins].tolist(), peak_m=float(np.linalg.norm(offset,axis=1).max())))

    # A reverse cut is verified by the original contact registration. Both
    # views must put the same anatomical foot on the physical platform. Carry
    # the incoming velocity, then rejoin the next shot on the same surface.
    checked = registration.get('reverse_boundary_checked') or 'reverse boundary checked' in registration.get('source','')
    layer = int(registration['shot'])-1
    for first in cuts[checked & (shot[cuts-1]==layer)&(shot[cuts]==layer+1)]:
        start = ranges[np.searchsorted(ranges,first)-1]
        end = ranges[np.searchsorted(ranges,first)+1]
        if first-start<2 or end-first<3 or not np.any(support[first]&support[first-1]):
            continue
        if not on_landing(root[first-1:first+1]).all() or frames[first]-frames[first-1]!=1:
            continue
        v = .5*(velocity(flat,max(start,first-look-1),first) + velocity(flat,first,min(end,first+look+1)))
        delta = flat[first-1]+v-flat[first]
        distance = np.linalg.norm(delta)
        if distance < max(.12,4*np.linalg.norm(v)):
            continue
        # Cubic offset decay has maximum speed 1.5*distance/duration. Rejoin
        # before the first different tread; don't move an actor up a riser.
        duration = max(.50,1.5*distance)
        count = min(end-first-1,int(np.ceil(duration*fps)))
        levels = terrain.model_h(root[first:first+count+1])
        other = np.flatnonzero(np.abs(levels-landing[4])>.025)
        if len(other):
            count = min(count,int(other[0]))
        if count < 2 or count/fps < .75*distance or not np.all(np.diff(frames[first:first+count+1])==1):
            continue
        phase = np.arange(count+1)/count
        candidate = root[first:first+count+1]+(1.-smoothstep(phase))[:,None]*delta
        if np.max(np.abs(terrain.model_h(candidate)-levels[:count+1])) > .025:
            continue
        out[first:first+count+1] = candidate
        editable[max(start,first-round(.45*fps)):first+count] = True
        verified_joins.append(first)
        report.append(dict(kind='registered_return', first=int(frames[first]), last=int(frames[first+count]),
                           joins=[int(frames[first])], peak_m=float(distance)))
    if rays is not None and verified_joins:
        fit = _fit_rays(out,root,rays,editable,np.asarray(verified_joins),flight,
                        np.stack([along,across],axis=1),fps)
        if np.max(np.abs(terrain.model_h(fit)-terrain.model_h(root))) <= .025:
            out = fit
            for record in report:
                region = (frames>=record['first']) & (frames<=record['last'])
                record['peak_m'] = float(np.linalg.norm(out[region]-root[region],axis=1).max())
    return out, report
