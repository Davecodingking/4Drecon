"""Refine a layered staircase reconstruction using a continuous single-actor cut.
Run on an independent copy after the first people solve and before final assembly.
Only translations are estimated. The street view must own the lower flight,
have an incompatible forward/back cut pair, and have no shared rigid-object boxes.
"""
import argparse
import json
from pathlib import Path
import shutil
import os
import tempfile
import numpy as np
from scipy.ndimage import shift as grid_shift
import body_into_scene as B
import mhr_body as M
import terrain


def infer_offset(root, feet, vf, shot, layer, up, fps):
    rows = np.flatnonzero(shot == layer)
    if len(rows) < 3 or rows[0] < 2 or rows[-1] + 1 >= len(vf):
        return None
    a, b = rows[[0, -1]]
    if not np.all(np.diff(vf[a-2:b+2]) == 1):
        return None
    flat = lambda x: x - (x @ up)[..., None] * up
    incoming = flat(root[a-1] - root[a])
    outgoing = flat(root[b+1] - root[b])
    if np.linalg.norm(incoming) < 3 or np.linalg.norm(incoming-outgoing) > min(2., .20*np.linalg.norm(incoming)):
        return None
    # Anchor the least mobile foot in the approach, using actual sole vertices.
    drifts = np.linalg.norm(flat(feet[a-1]-feet[a-2]), axis=1)
    j = int(np.argmin((feet[a-1] @ up) + drifts))
    if drifts[j] * fps > .3:
        return None
    return feet[a-1,j] - feet[a,j]


def run(path):
    path = Path(path)
    if (path/'contact_registration.json').exists():
        return
    report = json.loads((path/'07_terrain.json').read_text())
    layer_info = (report.get('地面模型') or {}).get('分层') or {}
    if '平台' not in layer_info or '最低一层的镜头' not in layer_info:
        raise ValueError('No independently reconstructed lower flight and landing.')
    layer = layer_info['最低一层的镜头'] - 1
    z = np.load(path/'10_people_params.npz')
    ids = [k[:-3] for k in z.files if k.endswith('_vf')]
    if len(ids) != 1:
        raise ValueError('Single continuous actor track required.')
    pid = ids[0]
    da, _, up, _, _ = B.load_scene(str(path))
    T = B.TERRAIN
    if layer == T.main or not T.model_mesh or not len(T.extra):
        raise ValueError('Lower layer has no independent coordinates.')
    vf = z[pid+'_vf']; root = M.root_pos(z[pid+'_params'])
    shot = np.searchsorted(T.cuts, vf/T.fps+.001, side='right')
    rest = M.rest_verts(z[pid+'_shape'])
    sole = rest[:,1] < rest[:,1].min()+10
    verts = np.load(path/'10_people_world.npz')[pid+'_V']
    feet = np.stack([verts[:,sole & (sign*rest[:,0]>0)].mean(1) for sign in (1,-1)],1)
    off = infer_offset(root, feet, vf, shot, layer, up, T.fps)
    if off is None:
        raise ValueError('No consistent contact-based translation.')
    entry = np.flatnonzero(shot == layer)[0]
    observed = feet[entry-1]-feet[entry]
    support = int(np.argmin(np.linalg.norm(observed-off,axis=1)))
    st = T.ST[T.main].copy()
    along = st[1]*T.e1+st[2]*T.e2
    across = -st[2]*T.e1+st[1]*T.e2
    ds,dw,dh = off@along, off@across, off@up
    extra = T.extra.copy()
    extra[:,:2] += ds; extra[:,2:4] += dw; extra[:,4:6] += dh
    # Join the upper flight to the re-registered landing. Require a physically
    # plausible riser and overlapping flight widths before making any changes.
    # Preserve the observed planar landing around the full planted footprint.
    # Vertices are in the provisional shot coordinates at this pass.
    indices = np.flatnonzero(shot == layer)
    samples = verts[indices] + off
    heights = samples @ up
    low = samples[heights < heights.min(axis=1)[:,None]+.12]
    longitudinal = low @ along
    landing = extra[-1]
    margin = .5*np.linalg.norm(np.ptp(verts[indices[0],sole],axis=0))
    margin = min(.3, max(.12,margin))
    low_s = min(landing[0],np.percentile(longitudinal,1)-margin)
    high_s = max(landing[1],np.percentile(longitudinal,99)+margin)
    if low_s <= extra[0,0]:
        raise ValueError("Proposed landing would erase the observed lower flight.")
    extra[:-1,:2] = extra[0,0]+(extra[:-1,:2]-extra[0,0]) * ((low_s-extra[0,0])/(landing[0]-extra[0,0]))
    landing[:2] = [low_s, high_s]
    rise = st[3]*(st[5]-st[4])/st[8]
    start = landing[1]
    if (start >= st[5] or min(st[7],landing[3]) <= max(st[6],landing[2])
        or abs(st[0]+st[3]*start-landing[4]) > 3*max(rise, extra[-2,4]-extra[-3,4])):
        raise ValueError(f'Corrected flights do not meet: {st[0]+st[3]*start} vs {landing[4]}, s={start}, offset={off}')
    plane = T.plane_h+dh
    boxes = json.loads((path/'06_all_boxes.json').read_text())
    for label, instances in boxes.items():
        for item in instances:
            evidence = item.get('切镜头', {})
            sources = {row[0]-1 for row in evidence.get('各镜头', [])}
            if layer in sources and len(sources)>1 and label not in terrain.STAIR_WORDS:
                raise ValueError('Shared static box constrains the proposed translation.')
            if evidence.get('镜头',0)-1 == layer:
                item['中心'] = (np.array(item['中心'])+off).tolist()
                item['底高'] += dh
    maps = {}
    for directory in [Path(da), Path(da)/'dense']:
        if not (directory/'cameras.json').exists():
            continue
        filename = directory/'cameras.json'
        doc = json.loads(filename.read_text())
        fm = json.loads((directory/'frame_map.json').read_text())['frames']
        for k,c in doc['cameras'].items():
            if T.shot_of(fm[k]) == layer:
                E = np.array(c['extrinsic_w2c_4x4'])
                E[:3,3] -= E[:3,:3]@off
                c['extrinsic_w2c_4x4'] = E.tolist()
                c['camera_center_world'] = (-E[:3,:3].T@E[:3,3]).tolist()
        maps[filename]=doc
    arr = dict(np.load(path/'07_terrain.npz'))
    arr['ST'][T.main]=st
    arr.update(extra=extra,plane_h=plane,joined_flights=True,landing_priority=True)
    shift=(off@T.e2/T.res,off@T.e1/T.res)
    for key in ['Hq','Hm']:
        arr[key][layer]=grid_shift(arr[key][layer],shift,order=0,mode='nearest')+dh
    # The registered landing is one physical surface shared by every view.
    uu,vv=terrain._cell_uv(T.lo,T.res,*T.H.shape)
    s=uu*st[1]+vv*st[2]; w=-uu*st[2]+vv*st[1]
    for sa,sb,w0,w1,top,_ in extra:
        arr['Hq'][layer][(s>=sa)&(s<=sb)&(w>=w0)&(w<=w1)]=top
    sa,sb,w0,w1,top,_ = landing
    footprint = (s>=sa)&(s<=sb)&(w>=w0)&(w<=w1)
    for grid in arr['Hq']:
        if np.isfinite(grid).any():
            grid[footprint] = top
    changes={'shot':int(layer+1),'support_side':support,'translation_m':off.tolist(),'source':'single-actor planted-foot continuity; reverse boundary checked','source_terrain':'07_terrain.json'}
    layer_info["路面高_m"] = round(float(plane),2)
    layer_info["最低一层_m"] = round(float(plane),2)
    flight = layer_info.get("下段楼梯")
    if isinstance(flight, dict) and "s" in flight:
        flight["s"] = [round(float(extra[0,0]),2),round(float(landing[0]),2)]
        tread = float(np.median(extra[:-1,1]-extra[:-1,0]))
        flight["踏面深_cm"] = round(100*tread,1)
        flight["坡度_度"] = round(float(np.degrees(np.arctan2(np.median(np.diff(extra[:-1,4])),tread))),1)
    layer_info["平台"].update(高_m=round(float(landing[4]),2),
                              s=[round(float(x),2) for x in landing[:2]])
    report["地面模型"].update(平面高_m=round(float(plane),2), 平台高_m=round(float(landing[4]),2))
    report["接触配准"] = changes
    with tempfile.TemporaryDirectory(dir=path) as work:
        work=Path(work)
        (work/'06_all_boxes.json').write_text(json.dumps(boxes,ensure_ascii=False))
        for review_file in ['04_picks','05_review.json','05_review_applied.json','00_run_config.json','.base_tag']:
            src=path/review_file
            if src.exists():
                (work/review_file).symlink_to(src.resolve(),target_is_directory=src.is_dir())
        arr['snap'],_=terrain._pedestals(str(work),up,st,extra,plane)
        (work/'07_terrain.json').write_text(json.dumps(report,ensure_ascii=False,indent=1))
        np.savez(work/'07_terrain.npz',**arr)
        staged={path/'07_terrain.json':work/'07_terrain.json',path/'06_all_boxes.json':work/'06_all_boxes.json',path/'07_terrain.npz':work/'07_terrain.npz'}
        for n,(file,doc) in enumerate(maps.items()):
            output=work/f'cameras_{n}.json'
            output.write_text(json.dumps(doc,ensure_ascii=False))
            staged[file]=output
        # Keep the source calibration for reproducible retries.
        originals=path/'contact_registration_source'
        if originals.exists():
            raise ValueError('Restore source calibration before retrying an interrupted refinement.')
        for file in staged:
            backup=originals/file.relative_to(path)
            backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(file,backup)
        try:
            for file,src in staged.items():
                os.replace(src,file)
            (path/'contact_registration.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
        except BaseException:
            for file in staged:
                shutil.copy2(originals/file.relative_to(path),file)
            shutil.rmtree(originals)
            raise
    print(changes,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--r1',required=True)
    run(p.parse_args().r1)
