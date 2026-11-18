#!/usr/bin/env python
"""pipeline/body_into_scene.shot_cuts 的切点判据(2026-11-11 scdet 候选 + DINOv2 复核)。
答案是人眼逐个核过的(data/scout_1110/jw_fix/cuts_*.jpg): jw1 / jw4 / irish 是老判据(scdet >= 12)漏掉的,
timesquare / moose 19.9 s 是老判据错报的(镜头晃、人快动)。视频不在本机就跳过。
跑: $PY tools/tests/test_shot_cuts.py"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.expanduser("~/sam3d/pipeline"))
import body_into_scene as B

D = os.path.expanduser("~/sam3d/data")
TRUTH = {
    "indoor/dataset/jw1_construction_indoor": [7.5, 9.53, 11.77, 12.77, 15.57, 19.63, 21.77, 23.53],   # 老判据只出 23.53
    "indoor/dataset/jw4_kitchen_indoor": [0.4, 4.8, 11.57],                                            # 老判据一个都没出
    "outdoor/dataset/irish_tapdance_outdoor": [9.63, 13.17, 16.87],                                    # 老判据只出 9.63
    "outdoor/dataset/timesquare": [],                                                                  # 老判据错报 9 个
    "outdoor/dataset/moose_cuts_outdoor": [3.7],                                                       # 老判据多报 19.9
}


def _cuts(video):
    with tempfile.TemporaryDirectory() as T:
        json.dump(dict(video=video, frames={}), open(f"{T}/frame_map.json", "w"))
        cuts = B.shot_cuts(T)
        meta = json.load(open(f"{T}/shot_cuts.json"))
    return cuts, meta


def test_cuts_match_what_a_person_checked():
    for rel, want in TRUTH.items():
        v = f"{D}/{rel}/video.mp4"
        if not os.path.exists(v):
            print("skip", rel); continue
        got, meta = _cuts(v)
        assert meta.get("method") == B.CUT_METHOD, meta.get("method")
        assert len(got) == len(want) and all(abs(a - b) < 0.05 for a, b in zip(got, want)), (rel, got, want)


def test_an_old_cache_is_still_honoured():
    """老判据算的缓存(没有 method)照旧认, 不悄悄改正在做的片子的镜头划分。"""
    v = f"{D}/indoor/dataset/jw4_kitchen_indoor/video.mp4"
    if not os.path.exists(v):
        print("skip old cache"); return
    with tempfile.TemporaryDirectory() as T:
        st = os.stat(v)
        json.dump(dict(video=v, frames={}), open(f"{T}/frame_map.json", "w"))
        json.dump(dict(video=v, size=st.st_size, mtime=st.st_mtime, thr=B.SCDET_THR, cuts=[1.23]), open(f"{T}/shot_cuts.json", "w"))
        assert B.shot_cuts(T) == [1.23]


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f(); print("ok", name)
