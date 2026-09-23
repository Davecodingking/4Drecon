#!/bin/bash
# box_fit_view.py 两种出口: --data 出 json(闸口 3D tab 读), --out 出单文件 html。office r6 上跑, 约 40 s。
set -e
PY=~/miniforge3/envs/sam3d/bin/python; R1=~/sam3d/data/indoorOutput/office/r6_pillarfix
SP=/tmp/claude-1004/-home-jincheng/92d1614a-64e3-41aa-9d0b-8592d431f8f9/scratchpad; mkdir -p $SP
$PY ~/sam3d/tools/box_fit_view.py --r1 $R1 --out off --data $SP/bfv.json > $SP/bfv_data.log
$PY - <<EOF
import json; d = json.load(open("$SP/bfv.json"))
for k in ("insts", "ctx", "cams", "页"): assert k in d, k
assert d["页"]["eyebrow"] and "measured" in d["页"]["eyebrow"], d["页"]
assert len(d["insts"]) > 10, len(d["insts"])
print("json ok:", len(d["insts"]), "insts;", d["页"]["eyebrow"])
EOF
$PY ~/sam3d/tools/box_fit_view.py --r1 $R1 --data off --out $SP/bfv.html > $SP/bfv_html.log
grep -q '"insts"' $SP/bfv.html && ! grep -q '__DATA__' $SP/bfv.html && echo "html ok"
