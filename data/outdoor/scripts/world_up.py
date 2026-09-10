#!/usr/bin/env python
"""定出场景真正的"天"在哪, 存成 02_da3/world_up.json 供后面所有步骤共用。

为什么需要这一步 —— 原来整条链把"上"定义成首帧相机的 -y 轴。街头平视时相机基本水平,
这个假设成立(时代广场地面只偏 1.3°); 但高空俯拍时相机是低着头的, 它的 -y 指向斜上后方
而不是天顶, beach 和 carpark 因此整个场景歪了 26°。

怎么定: 拿 DA3 的点云找最大的那个平面 —— 沙滩、停车场、街道都是大片平地, 点云里
占比最大的平面就是地面, 它的法向就是"天"。这个判据不依赖盒子, 所以能放在量盒子之前
(盒子的高度是沿"天"量的, 方向错了盒子就错了, 事后再转只是把错的盒子摆正)。
"""
import os, json, numpy as np
R1  = os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
DA3 = f"{R1}/02_da3"
OUT = f"{DA3}/world_up.json"
NFR = int(os.environ.get("UP_FRAMES","12"))      # 抽几帧的点云来拟合
BAND= float(os.environ.get("UP_BAND","0.30"))    # RANSAC 内点带宽, 相对场景尺度
CAM_UP = np.array([0,-1,0.])                     # 老的假设: 首帧相机的 -y

cams = json.load(open(f"{DA3}/cameras.json"))["cameras"]
ks = sorted(cams)
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
tol=BAND*scale/20.0
print(f"[点云] {len(P)} 点, 场景尺度 {scale:.1f}m, 内点带宽 {tol:.2f}m")

rng=np.random.default_rng(0); best=(0,None)
for _ in range(2000):
    i=rng.choice(len(P),3,replace=False)
    a,b,c=P[i]
    n=np.cross(b-a,c-a); ln=np.linalg.norm(n)
    if ln<1e-9: continue
    n=n/ln
    inl=int((np.abs((P-a)@n)<tol).sum())
    if inl>best[0]: best=(inl,(n,a))
n,a=best[1]
# 用内点重新精修
for _ in range(3):
    m=np.abs((P-a)@n)<tol
    Q=P[m]-P[m].mean(0)
    n=np.linalg.svd(Q,full_matrices=False)[2][-1]; a=P[m].mean(0)
# 定正反。★ 不能拿"和首帧相机 -y 同侧"当判据: 相机低着头时那个基准本身就歪了 26~29°,
# 临界情况会判反, 结果整个场景上下颠倒(实测 beach 的 07_scene_phy.glb 就是这么翻的)。
# 可靠的判据来自场景本身: 地面是最低的那一层, 绝大多数点应该在地面之上。
d = P @ n - (a @ n)                       # 各点到平面的有符号距离
if np.median(d) < 0: n = -n               # 让"多数点在正侧"= 法向朝天
above = float((P @ n > (a @ n)).mean())
print(f"[正反] 地面上方的点占 {100*above:.0f}% (应显著过半, 否则法向反了)")
ang=float(np.degrees(np.arccos(np.clip(abs(n@(-CAM_UP)),-1,1))))
print(f"[地面] 内点 {best[0]}/{len(P)} ({100*best[0]/len(P):.0f}%)  法向 {np.round(n,4)}")
print(f"[结果] 新的\"天\"偏离老假设(首帧相机-y) {ang:.1f}°" + ("   -> 老假设基本没错, 影响很小" if ang<3 else "   -> 老假设错得明显, 必须转正"))
# 地面高度也一并存: 之前只存法向, 高度靠盒底反推 —— 而盒底本身在俯视场景里是错的
# (beach 盒底跨了 14.9m), 拿错的盒底推地面, 推出来的线穿过物体中间, 上下都贴不上。
# 地面在点云里本来就有, 直接取内点沿"天"方向的高度中位数。
# ★ 符号必须和下游一致: 整条链取高度用的是 P@UP, 而 UP = -n。
# 第一版写成了 P@n, 高度正负颠倒, 结果地面跑到物体下方几米, 所有东西被压进地里。
_h = float(np.median(P[np.abs((P-a)@n)<tol] @ (-n)))
json.dump(dict(up=(-n).tolist(), ground_h=_h, 偏离老假设度数=round(ang,2),
               内点比例=round(best[0]/len(P),3), 用帧=frames, 说明="up 是 DA3 世界系里**指向天**的单位向量。取高度一律用 P@UP, 值越大越高。(实测核过: 相机沿 UP 的高度中位 -0.29, ground_h -29.71, 相机在地面之上, 符号对)"),
          open(OUT,"w"), ensure_ascii=False, indent=1)
print(f"[done] -> {OUT}")
