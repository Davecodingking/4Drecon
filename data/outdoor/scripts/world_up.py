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
import os, sys, json, numpy as np
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

# ---------------- 室内: 上面挑的地面先过检验, 不合格才重找 (2026-11-05) ----------------
# ★ carl 上"最低一层里点最多"挑错了: 床尾(栏杆+垂下的被子)那一条点数是露出来的地板的两倍,
#   被当成地面, 天歪 13.5°, 相机离地只剩 0.13 m, 日志只提醒没停。
#   室内地板有两条硬约束, 先拿来检验上面的结果:
#     1. 地板下面没有东西: 平面下方 5 cm 以外的点不超过 0.5%(床尾下面就是地板, 一票否决)
#     2. 相机在地板上方 0.4~2.5 m(这时还没做米制缩放, 实测缩放倍数 0.9~1.5, 所以比手持的 0.5~2.2 放宽)
#   合格就不动(office/couplelift/ithor312 结果逐位不变)。不合格才在满足两条的平面里重找,
#   ★ 重找时按"窄带(带宽的 1/3)里的点数"打分: 真地板是很薄的一层, carl 上带宽 ±3cm -> ±1cm 只少 4%;
#     斜穿被子、床边的面是一片散开的点, 同样收窄少 60%。按宽带打分 carl 会挑中一个侧倾 12° 的面。
#   ★ 曾经还加过"和后墙垂直 ±3°": carl 用不上(窄带打分就对了), office r1 墙只占 6% 找不准, 反把地面带歪 15°, 去掉了。
#   只在室内开(UP_INDOOR=1, 室内链传): 室外观景台、山路下面就是山谷城市, 第 1 条在室外不成立。
#   重找也找不到(squat 几乎看不到地板) -> 保留上面的结果, 交给下面单墙那一支 / 相机高度检查。
INDOOR = os.environ.get("UP_INDOOR") == "1"
CAM_H_MIN, CAM_H_MAX = 0.4, 2.5
_C = np.array([np.linalg.inv(np.array(cams[k]["extrinsic_w2c_4x4"]).reshape(4,4))[:3,3] for k in frames])

def _floor_ok(m, p, Q):
    """m 朝相机上。返回 (合格?, 下方比例, 相机离地)。"""
    m = m * (np.sign(m @ cam_up) or 1)
    below = float((((Q - p) @ m) < -0.05).mean()); ch = float(np.median((_C - p) @ m))
    return below <= 0.005 and CAM_H_MIN <= ch <= CAM_H_MAX, below, ch

def _indoor_floor(P, tol, iters=6000):
    """在"下方几乎无点 + 相机高度合理 + 离相机上 25° 内"的平面里, 挑窄带点数最多的。返回 (法向, 面上一点) 或 None。"""
    rng = np.random.default_rng(0); t = tol / 3
    Ps = P[rng.choice(len(P), min(len(P), 60000), replace=False)]       # 打分用的子集, 精修和复检用全量
    h = Ps @ cam_up; seeds = Ps[h < np.percentile(h, LOW_Q)]
    best = None
    for _ in range(iters):
        p, q, r = seeds[rng.choice(len(seeds), 3, replace=False)]
        m = np.cross(q-p, r-p); ln = np.linalg.norm(m)
        if ln < 1e-9: continue
        m /= ln; m *= np.sign(m @ cam_up) or 1
        if m @ cam_up < np.cos(np.radians(25)): continue
        k = int((np.abs((Ps - p) @ m) < t).sum())
        if best is not None and k <= best[0]: continue
        if not _floor_ok(m, p, Ps)[0]: continue
        best = (k, m, p)
    if best is None: return None
    _, m, p = best
    for _ in range(3):                                    # 窄带内点精修
        sel = np.abs((P-p) @ m) < t; W = P[sel] - P[sel].mean(0)
        m2 = np.linalg.svd(W, full_matrices=False)[2][-1]; m = m2 * (np.sign(m2 @ cam_up) or 1); p = P[sel].mean(0)
    return (m, p) if _floor_ok(m, p, P)[0] else None      # 精修会挪平面, 挪完再验一遍

_refit = False
if INDOOR:
    _ok, _bl, _chk = _floor_ok(n, a, P)
    if _ok:
        print(f"[地面·室内] 检验通过: 下方 {100*_bl:.2f}%, 相机离地 {_chk:.2f} m")
    else:
        print(f"[地面·室内] ★ 检验不过(下方 {100*_bl:.2f}% 应 ≤0.5%, 相机离地 {_chk:.2f} m 应在 {CAM_H_MIN}~{CAM_H_MAX}), 重找")
        _r = _indoor_floor(P, tol)
        if _r is None:
            print("[地面·室内] ★ 重找也没有合格的平面(地板几乎看不到?), 保留原结果")
        else:
            _ang0 = np.degrees(np.arccos(np.clip(abs(_r[0] @ n), 0, 1)))
            n, a = _r; _refit = True
            _ok, _bl, _chk = _floor_ok(n, a, P)
            print(f"[地面·室内] 重找到: 和原结果差 {_ang0:.1f}°, 下方 {100*_bl:.2f}%, 相机离地 {_chk:.2f} m")

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
# 墙占满画面时，最低层 RANSAC 仍会给出很高的内点比例；方向与相机的上几乎垂直，
# 这不是可信的地面。无可见地板时必须显式给出高度，不能把墙面高度写给下游。
_fit_up = UP.copy()
_fit_ratio = round(best[0]/len(P), 3)
_angle_cam = float(np.degrees(np.arccos(np.clip(UP @ cam_up, -1, 1))))
_override = f"{DA3}/up_override.json"
_source = "地面平面拟合" + ("(室内重找: 下方无点+相机高度+窄带)" if _refit else "")
def _upward_strip_floor(frames, cams, DA3, cam_up, tol):
    """墙占满画面(_angle_cam > 60°)时, 窄条马路/地板在点数上拼不过大墙, 但墙面法向水平、路面法向朝上。
    用深度图逐像素局部法向把竖直墙筛掉(只留和 cam_up 夹角 < 35° 的朝上面元), 在最低层里拟合真地面。
    真的一点地面都没拍到(纯墙/天花板) -> 朝上点不够, 返回 None, 照旧走单墙兜底。"""
    pts_u, nrm_u, n_valid = [], [], 0
    cos_max = float(np.cos(np.radians(35.0)))
    for k in frames:
        d = np.load(f"{DA3}/depth/{k[:4]}_depth.npy").astype(np.float64)
        c = np.load(f"{DA3}/depth/{k[:4]}_conf.npy")
        H, W = d.shape
        if H < 8 or W < 8:
            continue
        K = np.asarray(cams[k]["intrinsic_3x3"], float)
        if "input_hw" in cams[k]:
            h0, w0 = cams[k]["input_hw"]; sx, sy = W / w0, H / h0
        else:
            sx = sy = 1.0
        fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy
        ys, xs = np.mgrid[0:H, 0:W]
        Pc = np.stack([(xs - cx) / fx * d, (ys - cy) / fy * d, d], -1)
        dx = Pc[1:-1, 2:] - Pc[1:-1, :-2]
        dy = Pc[2:, 1:-1] - Pc[:-2, 1:-1]
        nv = np.cross(dx, dy)
        dc = d[1:-1, 1:-1]
        jump = np.maximum(np.abs(d[1:-1, 2:] - d[1:-1, :-2]), np.abs(d[2:, 1:-1] - d[:-2, 1:-1])) > 0.06 * dc
        ok = np.isfinite(dc) & (dc > 0) & (dc < 12.0) & ~jump & (c[1:-1, 1:-1] >= np.percentile(c, 35))
        n_valid += int(ok.sum())
        if not ok.any():
            continue
        P_sub, n_sub = Pc[1:-1, 1:-1][ok], nv[ok]
        nn = np.linalg.norm(n_sub, axis=1)
        good = nn > 1e-12
        P_sub, n_sub = P_sub[good], n_sub[good] / nn[good, None]
        n_sub[(n_sub * P_sub).sum(1) > 0] *= -1
        c2w = np.linalg.inv(np.asarray(cams[k]["extrinsic_w2c_4x4"], float).reshape(4, 4))
        Pw = (P_sub @ c2w[:3, :3].T) + c2w[:3, 3]
        Nw = n_sub @ c2w[:3, :3].T
        up_m = (Nw @ cam_up) >= cos_max
        if up_m.any():
            pts_u.append(Pw[up_m]); nrm_u.append(Nw[up_m])
    if not pts_u:
        return None
    Pu = np.concatenate(pts_u); Nu = np.concatenate(nrm_u)
    if len(Pu) < 500 or len(Pu) < 0.015 * max(n_valid, 1):
        return None
    u0 = np.median(Nu, axis=0); u0 /= np.linalg.norm(u0)
    inl_n = (Nu @ u0) >= float(np.cos(np.radians(10.0)))
    if inl_n.sum() < 400:
        return None
    u_up = np.median(Nu[inl_n], axis=0); u_up /= np.linalg.norm(u_up)
    if u_up @ cam_up < 0:
        u_up = -u_up
    hu = Pu[inl_n] @ u_up
    low_band = Pu[inl_n][hu <= np.percentile(hu, 60)]
    if len(low_band) < 200:
        return None
    n_fit, p_fit, n_in = _fit(low_band, max(0.03, tol))
    u_final = n_fit * (1.0 if (n_fit @ cam_up) >= 0 else -1.0)
    u_final /= np.linalg.norm(u_final)
    if (u_final @ cam_up) < cos_max or n_in < 0.35 * len(low_band):
        u_final = u_up
        gh = float(np.median(low_band @ u_final))
    else:
        gh = float(np.median(low_band[np.abs((low_band - p_fit) @ u_final) < max(0.03, tol)] @ u_final))
    ch = float(np.median(np.array([np.linalg.inv(np.asarray(cams[k]["extrinsic_w2c_4x4"], float).reshape(4, 4))[:3, 3]
                                   for k in frames]) @ u_final) - gh)
    if ch < 0.20 or ch > 5.0:
        return None
    return u_final, gh, ch, round(len(Pu) / max(n_valid, 1), 3)


if os.path.exists(_override):
    manual = json.load(open(_override))
    raw = manual.get("up")
    UP = cam_up.copy() if raw == "camera" else np.asarray(raw, float)
    if UP.shape != (3,) or not np.all(np.isfinite(UP)) or np.linalg.norm(UP) < 1e-9:
        sys.exit(f"[失败] {_override} 的 up 必须是 'camera' 或 3 维非零向量")
    UP = UP / np.linalg.norm(UP)
    _h = float(manual["ground_h"])
    if not np.isfinite(_h): sys.exit(f"[失败] {_override} 的 ground_h 必须是有限数")
    _ch = float(np.median(np.array([np.linalg.inv(np.array(cams[k]["extrinsic_w2c_4x4"]).reshape(4,4))[:3,3]
                                 for k in frames]) @ UP) - _h)
    ang = float(np.degrees(np.arccos(np.clip(UP @ CAM_UP, -1, 1))))
    _source = f"人工覆盖: {_override}"
    print(f"[人工覆盖] 自动拟合与相机上差 {_angle_cam:.1f}°，改用 up={np.round(UP,4)}，"
          f"ground_h={_h:.3f} m，相机离地 {_ch:.2f} m")
elif _angle_cam > 60:
    _up_strip = _upward_strip_floor(frames, cams, DA3, cam_up, tol)
    if _up_strip is not None:
        UP, _h, _ch, _fit_ratio = _up_strip
        ang = float(np.degrees(np.arccos(np.clip(UP @ CAM_UP, -1, 1))))
        _source = "朝上局部平面拟合(筛除大墙)"
        print(f"[朝上地面] 大墙占优(原偏 {_angle_cam:.1f}°), 筛除竖墙后抓到朝上地面: "
              f"up={np.round(UP, 4)}, ground_h={_h:.3f} m, 相机离地 {_ch:.2f} m, 偏老假设 {ang:.1f}°")
    else:

        if os.environ.get("UP_WALL_BODY") != "1":
            sys.exit(f"[失败] 拟合平面与相机上差 {_angle_cam:.1f}°，不是可靠地面；"
                     f"请核对视频并在 {_override} 指定 up 和 ground_h")
        # 这片平面更像墙。墙面法线只能约束地面法线与它垂直，剩余方向取相机的上；
        # 暂以相机离地 1.5m 留一个可用的地基，people_mesh 后须用 Body 脚底重定高度。
        wall_n = _fit_up / np.linalg.norm(_fit_up)
        wall_mismatch = float(np.degrees(np.arcsin(np.clip(abs(cam_up @ wall_n), 0, 1))))
        if wall_mismatch <= 5:
            UP = cam_up - float(cam_up @ wall_n) * wall_n
            UP /= np.linalg.norm(UP)
            _source = "单墙约束+相机上（待Body脚底定高度）"
        else:
            # DA3 的单面墙可能被深度扭斜；这时强制垂直会把地板一起转歪。
            UP = cam_up.copy()
            _source = "单墙检验+相机上（待Body脚底定高度）"
        _h = float(np.median(np.array([np.linalg.inv(np.array(cams[k]["extrinsic_w2c_4x4"]).reshape(4,4))[:3,3]
                                       for k in frames]) @ UP) - 1.5)
        _ch = 1.5
        ang = float(np.degrees(np.arccos(np.clip(UP @ CAM_UP, -1, 1))))
        print(f"[单墙] 拟合平面与相机上差 {_angle_cam:.1f}°；暂用{('墙约束后的' if wall_mismatch <= 5 else '')}相机上，"
              f"墙面垂直偏差 {wall_mismatch:.1f}°，地面 {_h:.3f} m 待 Body 脚底校准")
dot = float(UP @ cam_up)  # 记录最终采用的方向，不能沿用翻转前或自动拟合的点乘
# ★ 室内自动拟合的地面, 相机高度不合理就停在闸 1(carl r1 当初是 0.13 m, 只打了一行提醒就往下跑了)。
#   人工覆盖和单墙那一支(高度等 Body 脚底定)不在这里拦。
if INDOOR and _source.startswith("地面平面拟合") and not CAM_H_MIN <= _ch <= CAM_H_MAX:
    sys.exit(f"[失败] 室内地面拟合后相机离地 {_ch:.2f} m, 不在 {CAM_H_MIN}~{CAM_H_MAX} m —— 多半把家具顶面当成了地板。"
             f"核对 02_ground_check.jpg, 在 {_override} 指定 up 和 ground_h 后重跑这一步")
print(f"[核对] 相机在地面上方 {_ch:+.2f} m (必须为正; 手持约 1.4~1.6m, "
      f"明显偏小说明米制尺度也偏小)")
json.dump(dict(up=UP.tolist(), ground_h=_h, 偏离老假设度数=round(ang,2),
               相机朝上点乘=round(dot,3), 相机离地=round(_ch,3),
               内点比例=_fit_ratio if _source.startswith("地面平面拟合") else None,
               自动拟合up=_fit_up.tolist(), 自动拟合内点比例=_fit_ratio, 来源=_source,
               用帧=frames, 说明="up 是 DA3 世界系里指向天的单位向量；高度为 P@up。"),
          open(OUT,"w"), ensure_ascii=False, indent=1)
print(f"[done] -> {OUT}")
