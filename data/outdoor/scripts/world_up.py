#!/usr/bin/env python
"""定出场景真正的"天"在哪, 存成 02_da3/world_up.json 供后面所有步骤共用。

为什么需要这一步 —— 原来整条链把"上"定义成首帧相机的 -y 轴。街头平视时相机基本水平,
这个假设成立(时代广场地面只偏 1.3°); 但高空俯拍时相机是低着头的, 它的 -y 指向斜上后方
而不是天顶, beach 和 carpark 因此整个场景歪了 26°。

怎么定: 在 DA3 的点云里找地面, 取它的法向当"天"。★ 找法是"最低的那一层里, 最多点
认同的那个平面" —— 不是"整片点云里最大的平面"。后者只在室外成立(见下面那段长注释),
室内地板被家具盖住, 会抓到家具那一层, 整个场景歪掉十几度还内部自洽。
这个判据不依赖盒子, 所以能放在量盒子之前(盒子的高度是沿"天"量的, 方向错了盒子就错了,
事后再转只是把错的盒子摆正)。
"""
import os, json, numpy as np
from world_frame import usable_frames
R1  = os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
DA3 = f"{R1}/02_da3"
OUT = f"{DA3}/world_up.json"
NFR = int(os.environ.get("UP_FRAMES","12"))      # 抽几帧的点云来拟合
BAND= float(os.environ.get("UP_BAND","0.05"))    # RANSAC 内点带宽, 相对场景尺度
# ★ 2026-09-20 从 0.30 收到 0.05。0.30 是"整片点云找最大平面"时代的值(室外能到 2m 宽),
#   换成"最低那层里找共识"之后它会把整层点全吞进内点, 共识这一关等于没做
#   (实测 timesquare 23688 个种子点里"共识" 23688 个)。
CAM_UP = np.array([0,-1,0.])                     # 老的假设: 首帧相机的 -y

cams = json.load(open(f"{DA3}/cameras.json"))["cameras"]
# 位姿错的帧不许参与拟合地面: 它的点落在错的位置, 会把"最大平面"拉歪(ithor312 内点只剩 29%)
_ok = usable_frames(DA3, "定天")
ks = sorted(k for k in cams if int(k[:4]) in _ok)
frames = [ks[i] for i in np.linspace(0, len(ks)-1, min(NFR,len(ks))).astype(int)]

P=[]
for k in frames:
    d=np.load(f"{DA3}/depth/{k[:4]}_depth.npy"); c=np.load(f"{DA3}/depth/{k[:4]}_conf.npy")
    h,w=d.shape; K=np.array(cams[k]["intrinsic_3x3"])
    v,u=np.mgrid[0:h:3,0:w:3]; v=v.ravel(); u=u.ravel()
    dd=d[v,u]; cc=c[v,u]
    m=np.isfinite(dd)&(dd>0)&(cc>=np.median(c))
    v,u,dd=v[m],u[m],dd[m]
    Q=np.stack([(u-K[0,2])*dd/K[0,0],(v-K[1,2])*dd/K[1,1],dd],1)
    c2w=np.linalg.inv(np.array(cams[k]["extrinsic_w2c_4x4"]).reshape(4,4))
    P.append((c2w[:3,:3]@Q.T).T+c2w[:3,3])
P=np.concatenate(P)
scale=float(np.linalg.norm(P.max(0)-P.min(0)))
tol=max(0.03, BAND*scale/20.0)
print(f"[点云] {len(P)} 点, 场景尺度 {scale:.1f}m, 内点带宽 {tol:.2f}m")

# ---------------- 找地面 (2026-09-20 换判据) ----------------
# ★ 原来是"整片点云里内点最多的那个平面", 带宽还随场景尺度放大到 0.28m 那么宽。
#   室外对 —— 马路、沙滩、停车场本来就是最大的一片平地。
#   室内错得离谱 —— 地板被家具盖住, 露出来的碎片比桌面椅面加起来还少, 抓到的是家具
#   那一层: 既不是地板, 也不水平。实测 office 它认定的"地面点"平均比真地板高 1.4m,
#   方向偏 9.5°; ithor312 偏 12.9°。整个世界连同盒子沿这个错方向量一遍, 内部自洽,
#   看渲染图发现不了 —— 只有把房间壳(另一条链自己算的天)并进来才露馅。
#
# 现在两个条件叠加: **先把搜索限制在最低的那一层点, 再在这一层里找共识。**
#   只靠"最低"太脆 —— 一个坏深度点、一个楼梯口、一块黑洞就能把地板拽下去;
#   只靠"最多"就是上面那个毛病。两个一起才稳。
#   "最低"要沿"天"量, 而天正是要求的东西, 所以从相机自己的上起步迭代几轮, 很快收敛。
#
# 室外不受影响: 地面本来既是最低的也是最大的, 两种问法给同一个答案。
# 和独立判据(法线共识, 完全不碰地面)对照, 改前 -> 改后的偏差:
#   timesquare 1.78->1.76  pedestrian 0.95->0.75  realstreet 3.43->0.32
#   beach 0.75->0.37  carpark 2.76->2.12  office 9.51->0.95  ithor312 12.89->0.76
# 七个场景无一变差, 最担心的斜坡场景 beach 反而更好。
LOW_Q = float(os.environ.get("UP_LOW_Q", "25"))      # 取最低百分之几当搜索范围

def _fit(Q, tol):
    """在 Q 里 RANSAC 一个平面, 返回 (法向, 面上一点, 内点数)。"""
    rng=np.random.default_rng(0); b=(0,None)
    for _ in range(2000):
        i=rng.choice(len(Q),3,replace=False); p,q,r=Q[i]
        m=np.cross(q-p,r-p); ln=np.linalg.norm(m)
        if ln<1e-9: continue
        m=m/ln
        k=int((np.abs((Q-p)@m)<tol).sum())
        if k>b[0]: b=(k,(m,p))
    m,p=b[1]
    for _ in range(3):                                # 用内点精修
        sel=np.abs((Q-p)@m)<tol; W=Q[sel]-Q[sel].mean(0)
        m=np.linalg.svd(W,full_matrices=False)[2][-1]; p=Q[sel].mean(0)
    return m,p,b[0]

# 起步的"天": 各帧相机自己的上取平均。俯拍时它能偏 25°, 但这里只用来挑"哪一层最低",
# 这点误差够用了, 迭代一两轮就跟上。同一个 cam_up 下面定正反还要再用一次。
_cu = [np.linalg.inv(np.array(cams[_k]["extrinsic_w2c_4x4"]).reshape(4,4))[:3,:3] @ CAM_UP
       for _k in frames]
cam_up = np.mean(_cu, 0); cam_up /= np.linalg.norm(cam_up)

up = cam_up.copy()
for _it in range(3):
    _h = P @ up
    S  = P[_h < np.percentile(_h, LOW_Q)]
    n, a, _k = _fit(S, tol)
    if n @ up < 0: n = -n                             # 只为迭代稳定, 最终正反在下面定
    print(f"[地面] 第{_it+1}轮: 最低{LOW_Q:.0f}% 共 {len(S)} 点, 共识 {_k} 点, "
          f"方向挪了 {np.degrees(np.arccos(np.clip(float(n@up),-1,1))):.2f}°")
    up = n

# 内点比例仍按整片点云报, 这样和改之前的日志可比
best = (int((np.abs((P-a)@n) < tol).sum()), (n, a))

# 临时定个正反: 让多数点落在法向正侧。下面"定正反"那段会用相机再判一次并覆盖它,
# 这里留着是因为相机判据失效时要退回这一条。
d = P @ n - (a @ n)
if np.median(d) < 0: n = -n
above = float((P @ n > (a @ n)).mean())
print(f"[正反] 地面上方的点占 {100*above:.0f}% (应显著过半, 否则法向反了)")
ang=float(np.degrees(np.arccos(np.clip(abs(n@(-CAM_UP)),-1,1))))
print(f"[地面] 内点 {best[0]}/{len(P)} ({100*best[0]/len(P):.0f}%)  法向 {np.round(n,4)}")
print(f"[结果] 新的\"天\"偏离老假设(首帧相机-y) {ang:.1f}°" + ("   -> 老假设基本没错, 影响很小" if ang<3 else "   -> 老假设错得明显, 必须转正"))
# 地面高度也一并存: 之前只存法向, 高度靠盒底反推 —— 而盒底本身在俯视场景里是错的
# (beach 盒底跨了 14.9m), 拿错的盒底推地面, 推出来的线穿过物体中间, 上下都贴不上。
# 地面在点云里本来就有, 直接取内点沿"天"方向的高度中位数。
# ---------------- 定正反 (2026-09-20 换判据) ----------------
# 平面只给一个法向, UP 和 -UP 数学上等价, 必须另外挑一个。挑错整个场景上下颠倒。
#
# ★ 原来靠上面那句"多数点在平面哪一侧"。室内不成立, 室外也只是侥幸 —— 点云中位高度
#   离地面的余量实测: timesquare +1.22m  ithor312 +0.73m  realstreet **+0.11m**
#                     office **-0.44m** (翻了)。
#   相机在盒子里的时候看到的一大半是脚下的地板和桌面, 天花板又远又白点又少,
#   整团点云的重心压在地面附近, 余量随时会掉到负的。realstreet 只富余 11 厘米。
#
# 现在用相机自己的"上": 视频是正着拍的, 相机坐标系的 -y 就是画面上方, 用位姿转到
#   世界系就是一个大致朝天的向量。它和正确 UP 的夹角实测:
#     timesquare 0.5°  beach 3.4°  realstreet 6.1°  ithor312 17.6°  carpark 25.4°
#     office 171°  <- 一眼看出翻了
#
# ★ 文件头那句"不能拿相机 -y 当判据"说的是**定方向**: 俯拍时它偏 25~29°, 当方向不够准,
#   所以方向仍然用平面法向。但**定正反**只要一个符号, 要偏过 90° 才判错, 25° 还有 65°
#   的余量。原代码为了第一件事把相机 -y 扔了, 连第二件事也一起扔了 —— 而且它其实算过
#   这个角(下面的 ang), 但取了 abs 把符号丢了: office 真值 171° 被打印成 9.29°。
#
# 失效情形: 镜头笔直朝下的正射俯视 —— 画面上方在世界里是水平的, 点乘接近 0, 二选一
#   没有意义。低于 SIGN_MIN 就不信它, 退回老判据并在日志里喊一声。
SIGN_MIN = float(os.environ.get("UP_SIGN_MIN", "0.20"))
# cam_up 已在上面找地面那段算过, 这里直接用 —— 两处必须是同一个值。

UP = -n                                        # 下游一律用 P@UP 取高度
dot = float(UP @ cam_up)
if abs(dot) >= SIGN_MIN:
    if dot < 0:
        UP = -UP; n = -n
        print(f"[正反] 相机自己的上 · UP = {dot:+.3f} ({np.degrees(np.arccos(-dot)):.0f}°) "
              f"-> **反了, 已翻正**")
    else:
        print(f"[正反] 相机自己的上 · UP = {dot:+.3f} "
              f"(偏 {np.degrees(np.arccos(min(dot,1.0))):.1f}°) -> 一致")
else:
    print(f"[正反] ★ 镜头几乎笔直朝下(点乘 {dot:+.3f} < {SIGN_MIN}), 相机判据失效, "
          f"沿用老判据'多数点在上方'。老判据在室内会判错, 出图务必核对上下。")

# ★ 符号必须和下游一致: 整条链取高度用的是 P@UP。
# 第一版写成了 P@n, 高度正负颠倒, 结果地面跑到物体下方几米, 所有东西被压进地里。
_h = float(np.median(P[np.abs((P-a)@UP)<tol] @ UP))
_camh = np.array([np.linalg.inv(np.array(cams[k]["extrinsic_w2c_4x4"]).reshape(4,4))[:3,3]
                  for k in frames]) @ UP
_ch = float(np.median(_camh) - _h)
print(f"[核对] 相机在地面上方 {_ch:+.2f} m (必须为正; 手持约 1.4~1.6m, "
      f"明显偏小说明米制尺度也偏小)")
json.dump(dict(up=UP.tolist(), ground_h=_h, 偏离老假设度数=round(ang,2),
               相机朝上点乘=round(dot,3), 相机离地=round(_ch,3),
               内点比例=round(best[0]/len(P),3), 用帧=frames, 说明="up 是 DA3 世界系里**指向天**的单位向量。取高度一律用 P@UP, 值越大越高。(实测核过: 相机沿 UP 的高度中位 -0.29, ground_h -29.71, 相机在地面之上, 符号对)"),
          open(OUT,"w"), ensure_ascii=False, indent=1)
print(f"[done] -> {OUT}")
