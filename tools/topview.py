import sys, os, json, numpy as np, trimesh, cv2
# 算法模块(world_frame / word_rank / review_decisions)住在 data/outdoor/scripts/,
# 这个脚本在 tools/ —— 用仓库根定位, 别用 __file__ 所在目录(软链接会把它解析到 tools/)。
# 算法模块(world_frame / word_rank / review_decisions)住在 data/outdoor/scripts/,
# 这个脚本在 tools/ —— realpath 解析真身之后往上一层就是仓库根。
# 用 realpath 不用 abspath: 万一哪天又有人建软链接, abspath 会往上数到错的目录去。
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "data", "outdoor", "scripts"))
from world_frame import load_up, export_T, inverse_T
# R1 不给默认值: 手动跑忘了传时, 老的默认值会把俯视图静默写进 timesquare 别人的结果目录,
# 而且读的也是那个场景的 glb —— 看着像图出来了, 其实完全不是这次的结果。宁可 KeyError。
R1=os.environ["R1"]
cams=json.load(open(f"{R1}/02_da3/cameras.json"))["cameras"]
# ★ 这里曾经写死 diag(1,-1,-1): 组装那边 2026-09-08 改成按实际天方向算, 这里没跟着,
#   于是俯视图里相机轨迹和物体框不在一个坐标系(carpark 差 29.2°)。没人发现是因为
#   没人拿俯视图当依据。现在和组装同源。
_UP, _GH, _FB = load_up(f"{R1}/02_da3", "俯视图")
F=export_T(_UP)[:3,:3]
C=np.array([np.linalg.inv(np.array(v["extrinsic_w2c_4x4"]).reshape(4,4))[:3,3] for v in cams.values()])@F.T
items=[]
for f,col in (("07_scene_phy.glb",(90,90,90)),("10_buildings.glb",(40,60,220))):
    s=trimesh.load(f"{R1}/{f}")
    for n in s.graph.nodes_geometry:
        T,g=s.graph[n]; m=s.geometry[g].copy(); m.apply_transform(T)
        if n.startswith("ground"): continue
        lo,hi=m.bounds; items.append((lo[[0,2]],hi[[0,2]],col,n))
P=np.array([x for it in items for x in (it[0],it[1])]+[C[:,[0,2]].min(0),C[:,[0,2]].max(0)])
lo,hi=P.min(0)-5,P.max(0)+5; S=1400/max(hi-lo)
def px(p): return (int((p[0]-lo[0])*S), int(1400-(p[1]-lo[1])*S))
img=np.full((1400,int((hi[0]-lo[0])*S)+20,3),255,np.uint8)
for a,b,col,n in items:
    cv2.rectangle(img,px(a),px(b),col,2 if col[2]>150 else 1)
    if col[2]>150: cv2.putText(img,n.replace("building_","B"),px((a+b)/2),0,0.5,(40,60,220),1)
for p in C[:,[0,2]]: cv2.circle(img,px(p),3,(0,160,0),-1)
cv2.putText(img,"green=camera path  blue=buildings  gray=objects  (top view, 1 grid unused)",(10,25),0,0.6,(0,0,0),2)
cv2.imwrite(f"{R1}/11_topview.jpg",img,[cv2.IMWRITE_JPEG_QUALITY,88]); print("saved",img.shape)
