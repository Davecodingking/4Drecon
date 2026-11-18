"""Final leg-only collision correction for inferred skinned feet."""
import numpy as np
from scipy import sparse
from scipy.ndimage import maximum_filter1d
from scipy.sparse.linalg import spsolve
import mhr_body as M
import legs_proc as L

def measure(pose, shape, ids, up, terrain, dev):
    """Height above the actual collider for both corrective and exported skin."""
    feet = []
    for corrective in (True, False):
        v,_ = M.forward(shape,pose,corrective,dev,batch=16)
        feet.append(v[:,ids])
        del v
    pts = np.stack(feet,axis=1)
    ground = terrain.model_h(pts.reshape(-1,3)).reshape(pts.shape[:-1])
    gaps = pts@up-ground
    return pts,gaps

def resolve_inferred_sole_clearance(params, shape, up, terrain, plan, boundaries, fps, dev, airborne=None):
    boundaries = np.asarray(boundaries, dtype=int)
    if plan is None:
        return params
    eligible=np.asarray(plan['done'],bool)&~np.asarray(plan.get('turning',np.zeros(len(params),bool)),bool)
    if not eligible.any():
        return params
    out=params.copy()
    rest=M.rest_verts(shape,dev)
    soles=[np.flatnonzero((rest[:,1]<rest[:,1].min()+10)&(s*rest[:,0]>0)) for s in (1,-1)]
    rows=np.flatnonzero(eligible)
    gaps=np.full((len(params),2),np.inf)
    for group in np.array_split(rows,max(1,int(np.ceil(len(rows)/24)))):
        feet, gap=measure(params[group],shape,np.concatenate(soles),up,terrain,dev)
        gaps[group,0]=gap[:,:,:len(soles[0])].min((1,2))
        gaps[group,1]=gap[:,:,len(soles[0]):].min((1,2))
    for j,sd in enumerate(L.SIDES):
        ids=soles[j]
        cols=L.LEG_COLS[sd][:4]+[L.LEG_COLS[sd][6],L.LEG_COLS[sd][7]]
        # A flight estimate prevents any snapping of a free foot. If the mesh
        # itself intersects the collider the foot can still be lifted out,
        # independently of the pelvis and of the airborne opposite foot.
        movable = eligible if airborne is None else eligible & (~np.asarray(airborne,bool) | (gaps[:,j] < -.005))
        for a,b in zip(np.r_[0,boundaries],np.r_[boundaries,len(params)]):
            low=gaps[a:b,j]<-.005
            if not low.any():
                continue
            padding=max(2,round(.13*fps))
            active=maximum_filter1d(low,2*padding+1,mode='constant')&movable[a:b]&(gaps[a:b,j]<.04)
            for lo,hi in L.runs_of(active):
                sl=slice(a+lo,a+hi)
                out[sl]=_lift_sole(out[sl],shape,ids,cols,up,terrain,dev,fps)
    return out

def _lift_sole(base, shape, ids, cols, up, terrain, dev, fps):
    out=base.copy()
    n=len(base); c=len(cols)
    original,initial=measure(base,shape,ids,up,terrain,dev)
    target=np.maximum(0., .0005-initial.min((1,2)))
    # A low sole should translate up while retaining its world footprint
    # and tilt. Five evenly spaced vertices sample the skin for this term.
    ix=np.linspace(0,len(ids)-1,5).astype(int)
    reference=original[:,1,ix]+target[:,None,None]*up
    I=sparse.eye(n,format='csr'); eye=sparse.eye(c,format='csr')
    D=I[1:]-I[:-1]
    vel=sparse.kron(D,eye,format='csr')*(.26*fps/30)
    acc=sparse.kron(D[1:]-D[:-1],eye,format='csr')*(.72*(fps/30)**2)
    reg=.035*sparse.eye(n*c,format='csr')
    step=.005
    for _ in range(8):
        pts,gaps=measure(out,shape,ids,up,terrain,dev)
        worst=np.argmin(gaps,axis=2)
        clearance=gaps.min(2)
        floor_target=np.minimum(.0005, initial.min(2)+target[:,None])
        required=np.maximum(floor_target-clearance,0.)
        weights=np.where((clearance<floor_target+.001)&(target[:,None]>0),15.,0.)
        point_res=(reference-pts[:,1,ix]).reshape(n,-1)
        # Keep the active vertex on its current tread while differentiating.
        # Re-evaluate the true collider and extremal vertex every iteration.
        def features(v):
            extreme=v[np.arange(n)[:,None],np.arange(2)[None,:],worst]
            return np.c_[(extreme@up)*weights,v[:,1,ix].reshape(n,-1)]
        f0=features(pts)
        J=[]
        for col in cols:
            trial=out.copy();trial[:,col]+=step
            pv,_=measure(trial,shape,ids,up,terrain,dev)
            J.append((features(pv)-f0)/step)
        jac=sparse.block_diag(np.stack(J,axis=-1),format='csr')
        q=(out[:,cols]-base[:,cols]).ravel()
        mat=sparse.vstack([jac,reg,vel,acc],format='csr')
        rhs=np.r_[np.c_[required*weights,point_res].ravel(),-reg@q,-vel@q,-acc@q]
        dq=spsolve((mat.T@mat).tocsc(),mat.T@rhs).reshape(n,c)
        # An almost straight knee needs a bounded damped Newton step.
        dq*=np.minimum(1.,.20/np.maximum(1.e-8,np.max(abs(dq),axis=1)))[:,None]
        out[:,cols]+=dq
        out[:,cols[3]]=np.clip(out[:,cols[3]],np.minimum(0,base[:,cols[3]]),np.maximum(L.KNEE_MAX,base[:,cols[3]]))
        if np.max(abs(dq))<.001:
            break
    return out

def shared_contact_vertices(gaps, vertex_ids, count=8):
    """Select the same patches of skin nearest the collider on both frames."""
    relative = gaps-gaps.min(1,keepdims=True)
    shared_height = np.maximum(relative[:-1],relative[1:])
    count = min(count,len(vertex_ids))
    return vertex_ids[np.argpartition(shared_height,count-1,axis=1)[:,:count]]
