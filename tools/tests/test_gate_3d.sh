#!/bin/bash
# 闸口 + 3D tab 的整链: 懒加载、/3d 三条路、数据不在时灰掉。office r6 上跑, 服务起在 8799。
set -e
PY=~/miniforge3/envs/sam3d/bin/python; R1=~/sam3d/data/indoorOutput/office/r6_pillarfix
SP=/tmp/claude-1004/-home-jincheng/92d1614a-64e3-41aa-9d0b-8592d431f8f9/scratchpad; mkdir -p $SP
U="http://127.0.0.1:8799"; Q="s=office&r=r6_pillarfix&i=1"
test -e $R1/10_box_fit_view.json || $PY ~/sam3d/tools/box_fit_view.py --r1 $R1 --out off > /dev/null
$PY ~/sam3d/tools/gate_ui.py --scene office --run r6_pillarfix --indoor --port 8799 > $SP/gate3d.log 2>&1 &
PID=$!
# 收尾: 关服务、把藏起来的数据文件放回去。★ trap 里 set -e 照样生效, 文件已经放回时 mv 失败会把
#   整个脚本的退出码变成 1(明明全过) —— 所以先关掉 set -e, 再判文件在不在。
cleanup(){ set +e; kill $PID 2>/dev/null; wait $PID 2>/dev/null
           [ -e $R1/10_box_fit_view.json.hidden ] && mv -f $R1/10_box_fit_view.json.hidden $R1/10_box_fit_view.json
           return 0; }
trap cleanup EXIT
for i in $(seq 1 40); do curl -sf "$U/api/runs" > /dev/null && break; sleep 0.5; done
curl -sf "$U/api/state?$Q" | $PY -c "import json,sys; d=json.load(sys.stdin); assert d['场景查看']['在'] is True, d['场景查看']; print('state ok', d['场景查看'])"
! grep -q "\[3D\] 挂载" $SP/gate3d.log && echo "lazy ok: not imported yet"
curl -sf "$U/3d?$Q" | grep -q '<script id="data" type="application/json"></script>' && echo "/3d ok"
grep -q "\[3D\] 挂载" $SP/gate3d.log && echo "lazy ok: imported on first /3d"
curl -sf --compressed "$U/3d/data?$Q" | $PY -c "import json,sys; d=json.load(sys.stdin); assert len(d['insts'])>10; print('/3d/data ok', len(d['insts']))"
curl -sf "$U/3d/state?$Q" | grep -q '"在": true' && echo "/3d/state ok"
mv $R1/10_box_fit_view.json $R1/10_box_fit_view.json.hidden
curl -sf "$U/api/state?$Q" | $PY -c "import json,sys; d=json.load(sys.stdin); assert d['场景查看']['在'] is False; print('greyed ok')"
curl -sf "$U/3d?$Q" | grep -q "No 3D data" && echo "/3d without data ok"
curl -sf "$U/api/state?$Q" | $PY -c "import json,sys; d=json.load(sys.stdin); assert d['候选']; print('gate still serves', len(d['候选']), 'rows')"
mv $R1/10_box_fit_view.json.hidden $R1/10_box_fit_view.json
echo ALL OK
