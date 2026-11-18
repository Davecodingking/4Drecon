#!/usr/bin/env python
"""多镜头的片子: 把相机贴在主场景地面上的镜头整体竖直挪到正常离地高度。2026-11-11 加, joker 起头。

为什么: 一个片子的所有镜头在 DA3 里是一起重建的, 落在同一个世界系(统一的场景), 但各镜头的深度对不齐。
  joker 3 号镜头(296~353 帧, 在楼梯上往下拍街道)的相机被放在主镜头(4 号)楼梯底部、离台阶只有 0.22 m ——
  交付的场景里从这个相机看出去整个画面都是台阶的侧面(检查视频 10 秒左右全灰), 相机穿进台阶。
  这个镜头自己的深度里相机离脚下地面 1.5 m 左右, 是主场景那块地和它对不上。
做法(只在 GROUND_MODE=terrain 时由 boxes 那步调用; 跑在量盒子之前, 后面盒子 / 地形 / 摆人都用挪过的相机):
  1. 用 terrain.py 建一遍地形(各镜头各一张, 主镜头 = 量到格子最多的)。
  2. 每个非主镜头: 只看相机落在主镜头"真量到的地面或楼梯"上的帧(主镜头没看到的地方不算), 量
       主离地 = 相机高 - 主镜头地面高,  自离地 = 相机高 - 这个镜头自己的地面高。
  3. 主离地中位 < MIN_CLEAR 就整个镜头(这个镜头的所有相机)沿天挪 中位(自离地 - 主离地): 相机回到它自己深度里的离地高度。
     只挪位置不转, 这个镜头的深度点 / 人 / 物体跟着相机一起走, 镜头内部一点不变。
  另: 镜头转反了的(joker 3 号镜头: 4 个镜头的相机在世界系里都朝上坡, 但 3 号自己的深度里地面沿视线每米降 0.50 m,
  另外三个每米升 0.19~0.22 m —— 真实画面里它是在楼梯上往下拍街道, DA3 把它的朝向放反了 180°):
  自己的地面沿视线的坡 和 主镜头地面沿同一方向的坡 符号相反、都超过 FLIP_SLOPE -> 整个镜头绕自己相机中心的竖直轴转 180°,
  再做上面的竖直对齐。
  原件存 02_da3/cameras_before_shotreg.json(第一次); 结果写进 cameras.json 的"镜头竖直对齐"。
  cam_smooth --redo 会从平滑前的原件重来, 所以每次都要在平滑之后跑; 跑过一次再跑, 离地已经够了, 不会再挪。
"""
import argparse, json, os, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data", "outdoor", "scripts"))

MIN_CLEAR = 1.0     # 相机离主场景地面中位低于这个(m)才挪(手持 / 肩扛相机一般 1.2~1.8 m)
MIN_FRAMES = 3      # 至少这么多帧的相机落在主镜头量到的地面上才算得出来
FLIP_SLOPE = 0.07   # 坡(每米升降)小于这个(约 4°)不判朝向: 平地上分不出朝上坡还是下坡
PLACE_GAIN = 0.8    # 转过向的镜头: 搜到的位置误差要低于原地误差的这个比例才挪过去
PLACE_COVER = 0.08  # 搜位置时, 这个镜头自己量到的地面至少这么大比例落在统一场景量到的地方才算数
MIN_FEET = 5        # 至少这么多个人脚点落在主镜头量到的地面上, 才按人脚定这个镜头的高度
CAM_MIN_CLEAR = 0.5 # 米: 相机离脚下的地 / 台阶至少这么高
SEARCH_M, SEARCH_STEP = 12.0, 0.5   # 以人为锚搜平移的范围 / 步长(米)
FEET_TOL = 0.3      # 米: 主场景地面和人脚差的中位超过这个才挪(1、2 号镜头的地和主场景对得上, 差在 ±0.1 m 内)


def place_search(T, s, P0, own_clear, ks_known=None, flip=True):
    """转过 180° 的镜头: 在统一场景里搜相机(镜头中心)的水平位置, 让"这个镜头自己量到的地面(相对相机, 转过的)"和统一地面最吻合。
    joker 3 号镜头: 原地(楼梯底)误差中位 3.4 m, 搜到楼梯中段 2.3 m —— 自己的深度里前方十几米地面一路往下, 相机该在楼梯上部往下拍。
    返回 (最好的位置 p(水平, 世界系), 它的误差, 原地误差, 覆盖) 或 None。"""
    import terrain
    m = T.main; Ho = T.Hm[s]; ok = np.isfinite(Ho)
    if ok.sum() < 200:
        return None
    uu, vv = terrain._cell_uv(T.lo, T.res, *Ho.shape)
    Pw = uu[ok][:, None] * T.e1 + vv[ok][:, None] * T.e2 + Ho[ok][:, None] * T.up
    Pw = Pw[np.random.default_rng(0).choice(len(Pw), min(3000, len(Pw)), replace=False)]
    Rz = (2 * np.outer(T.up, T.up) - np.eye(3)) if flip else np.eye(3); rel = (Pw - P0) @ Rz.T
    dz = rel @ T.up; rh = rel - np.outer(dz, T.up)
    Hm = T.Hm[m]; known = np.isfinite(Hm); st = T.ST[m]
    if np.isfinite(st).all():
        sv = uu * st[1] + vv * st[2]; wv = -uu * st[2] + vv * st[1]
        known |= (sv >= st[4]) & (sv <= st[5]) & (wv >= st[6]) & (wv <= st[7])

    def score(p):
        zc = float(T.height_at(p)[0]) + own_clear; Q = p + rh
        ij = np.floor((np.c_[Q @ T.e1, Q @ T.e2] - T.lo) / T.res).astype(int)
        inb = (ij[:, 0] >= 0) & (ij[:, 0] < Hm.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < Hm.shape[0])
        kn = np.zeros(len(Q), bool); kn[inb] = known[ij[inb, 1], ij[inb, 0]]
        if kn.mean() < PLACE_COVER:
            return None
        return float(np.median(np.abs(T.height_at(Q[kn]) - (zc + dz[kn])))), float(kn.mean())
    p_here = P0 - (P0 @ T.up) * T.up
    here = score(p_here)
    best = None
    for j, i in np.argwhere(known)[::10]:
        p = uu[j, i] * T.e1 + vv[j, i] * T.e2; r = score(p)
        if r and (best is None or r[0] < best[1]):
            best = (p, r[0], r[1])
    if best is None:
        return None
    return best[0], best[1], (here[0] if here else None), best[2]


def main(r1):
    import terrain
    D = f"{r1}/02_da3"
    terrain.build(r1)
    T = terrain.Terrain(f"{r1}/07_terrain.npz")
    if len(T.Hq) < 2:
        print("[镜头对齐] 只有一个镜头, 不用对"); return
    main_s = T.main
    Hm = T.Hm[main_s] if T.Hm is not None else T.H
    st = T.ST[main_s]

    def main_known(C):
        """相机脚下主镜头真量到了(网格格子) 或者在主镜头楼梯脚印里。"""
        uv = np.array([C @ T.e1, C @ T.e2]); ij = np.floor((uv - T.lo) / T.res).astype(int)
        if 0 <= ij[0] < Hm.shape[1] and 0 <= ij[1] < Hm.shape[0] and np.isfinite(Hm[ij[1], ij[0]]):
            return True
        if np.isfinite(st).all():
            s = uv @ st[1:3]; w = -uv[0] * st[2] + uv[1] * st[1]
            return st[4] <= s <= st[5] and st[6] <= w <= st[7]
        return False

    def mesh_h(S):
        """主场景铺了网格的地方(地面网格 H 有值, 或者在台阶脚印里)的高度, 其余 NaN(那里没有东西挡视线)。"""
        uv = np.c_[S @ T.e1, S @ T.e2]; ij = np.floor((uv - T.lo) / T.res).astype(int)
        i = np.clip(ij[:, 0], 0, T.H.shape[1] - 1); j = np.clip(ij[:, 1], 0, T.H.shape[0] - 1)
        out = np.where(np.isfinite(T.H[j, i]), T.Hq[main_s][j, i], np.nan)
        if np.isfinite(st).all():
            s = uv @ st[1:3]; w = -uv[:, 0] * st[2] + uv[:, 1] * st[1]
            ins = (s >= st[4]) & (s <= st[5]) & (w >= st[6]) & (w <= st[7])
            out = np.where(ins, T.Hq[main_s][j, i], out)
        return out

    def people_place(pf, cc, mh):
        """有地面模型时: 整个镜头水平平移 o(在 ±SEARCH_M 内按 SEARCH_STEP 搜) + 竖直按人脚落地, 让
             1) 人脚落在模型上(脚的高低和模型的起伏对得上: 残差中位), 2) 每帧相机到人的视线不被模型挡,
             3) 相机离脚下的地 / 台阶 >= CAM_MIN_CLEAR; 同样好就挪得少的(每米 0.02)。
           只用人和相机, 不用这个镜头自己量到的地面(各镜头的深度互相对不上, 拿它对模型会被大平面带偏: joker 1 号镜头被挪 43 m)。
           返回 (o: 要减去的平移, 报告)。o 已含竖直; 被挡 / 相机太低就再往上抬(人会悬空, 报告里写)。"""
        F = np.concatenate([P_ for P_ in pf.values()]); Fv = np.concatenate([[vf_] * len(P_) for vf_, P_ in pf.items()])
        Cs = np.array([cc[v] for v in pf])
        fh = F @ T.up; best = None
        g = np.arange(-SEARCH_M, SEARCH_M + 1e-6, SEARCH_STEP)
        for dx in g:
            for dy in g:
                oh = dx * T.e1 + dy * T.e2
                Fm = mh(F - oh)
                if not np.isfinite(Fm).all():
                    continue
                gap = Fm - fh; d = float(np.median(gap)); fres = float(np.median(np.abs(gap - d)))
                # 视线: 每个脚点和它那一帧的相机
                need = []
                for (vf_, P_), C0 in zip(pf.items(), Cs):
                    C_ = C0 - oh + d * T.up
                    for q in P_:
                        Q_ = q - oh + (d + 0.5) * T.up; S_ = C_ + np.linspace(0.1, 0.9, 17)[:, None] * (Q_ - C_)
                        need.append(float(np.nanmax(mh(S_) - S_ @ T.up)))
                los = float(np.percentile(need, 90))
                cam = float(np.max(CAM_MIN_CLEAR - ((Cs - oh) @ T.up + d - mh(Cs - oh))))
                lift = max(0.0, los + 0.1 if los > 0 else 0.0, cam if cam > 0 else 0.0)
                score = fres + lift + 0.02 * float(np.hypot(dx, dy))
                if best is None or score < best[0]:
                    best = (score, oh - (d + lift) * T.up, dict(水平挪_m=round(float(np.hypot(dx, dy)), 2), 按人脚挪_m=round(d, 2),
                            脚和模型残差_m=round(fres, 2), 视线还被挡_m_p90=round(los, 2), 相机最低还差_m=round(cam, 2), 再抬_人悬空_m=round(lift, 2)))
        if best is None:
            return None, dict(结论="搜不到人脚都落在模型上的位置")
        return best[1], best[2]

    p = f"{D}/cameras.json"; doc = json.load(open(p)); cams = doc["cameras"]
    vfm = json.load(open(f"{D}/frame_map.json"))["frames"]; BAD = terrain._bad(D)
    by = {}
    for k, c in cams.items():
        s = T.shot_of(vfm[k])
        if s != main_s:
            by.setdefault(s, []).append(k)
    uu, vv = terrain._cell_uv(T.lo, T.res, *T.H.shape)

    def slope(H):
        ok = np.isfinite(H)
        if ok.sum() < 100:
            return None
        x = np.linalg.lstsq(np.c_[np.ones(ok.sum()), uu[ok], vv[ok]], H[ok], rcond=None)[0]
        return x[1] * T.e1 + x[2] * T.e2
    g_main = slope(Hm)
    if T.model and np.isfinite(st).all():   # 地面模型: 主场景的"坡"取台阶的坡(模型大部分是平面, 整体拟坡几乎是 0, 判不出转向)
        g_main = st[3] * (st[1] * T.e1 + st[2] * T.e2)
    _camh = float(json.load(open(f"{D}/world_up.json")).get("相机离地") or 1.5)

    def own_h(P, s):
        """这个镜头自己量到的地面高(有地面模型时 height_at 查到的是模型, 自己的在 Hm[s]); 查不到 = None。"""
        if not T.model:
            return float(T.height_at(P, vf=vfm[by[s][0]])[0])
        G = T.Hm[s]; uv = np.array([P @ T.e1, P @ T.e2]); ij = np.floor((uv - T.lo) / T.res).astype(int)
        if 0 <= ij[0] < G.shape[1] and 0 <= ij[1] < G.shape[0] and np.isfinite(G[ij[1], ij[0]]):
            return float(G[ij[1], ij[0]])
        return None
    rep, moved = [], False
    for s, ks in sorted(by.items()):
        r = dict(镜头=s + 1, 帧=len(ks)); C_own = {}; shift = np.zeros(3)     # 转过的镜头: 自己的地面要在转之前的位置上查(地面跟着一起转了)
        # 朝向反了没有: 自己的地面沿视线的坡 vs 主镜头地面沿同一方向的坡
        g_own = slope(T.Hm[s]) if T.Hm is not None else None
        if g_own is not None and g_main is not None:
            F = np.mean([np.array(cams[k]["extrinsic_w2c_4x4"], float)[2, :3] for k in ks], 0)
            fh = F - (F @ T.up) * T.up; fh /= np.linalg.norm(fh)
            own_a, main_a = float(g_own @ fh), float(g_main @ fh)
            r.update(自己的坡_沿视线=round(own_a, 2), 主场景的坡_沿视线=round(main_a, 2))
            if own_a * main_a < 0 and min(abs(own_a), abs(main_a)) > FLIP_SLOPE:
                P0 = np.mean([-np.array(cams[k]["extrinsic_w2c_4x4"], float)[:3, :3].T @ np.array(cams[k]["extrinsic_w2c_4x4"], float)[:3, 3] for k in ks], 0)
                Rz = 2 * np.outer(T.up, T.up) - np.eye(3)          # 绕竖直轴转 180°
                # 转了向, 原来的位置也不可信: 搜统一场景里和这个镜头自己的地面最吻合的位置(误差要明显更低才挪)
                _oh = own_h(P0, s); own_clear = float(P0 @ T.up - _oh) if _oh is not None else _camh
                ps = place_search(T, s, P0, own_clear)
                if ps and ps[2] is not None and ps[1] < PLACE_GAIN * ps[2]:
                    p_new = ps[0] + (float(T.height_at(ps[0])[0]) + own_clear) * T.up
                    shift = p_new - P0
                    r.update(搜位置=dict(原地误差_m=round(ps[2], 2), 新位置误差_m=round(ps[1], 2), 覆盖=round(ps[3], 2),
                                       水平挪_m=round(float(np.linalg.norm(shift - (shift @ T.up) * T.up)), 2)))
                elif ps:
                    r.update(搜位置=dict(原地误差_m=None if ps[2] is None else round(ps[2], 2), 最好_m=round(ps[1], 2), 结论="不挪"))
                for k in ks:
                    E = np.array(cams[k]["extrinsic_w2c_4x4"], float); R = E[:3, :3]; C = -R.T @ E[:3, 3]
                    C2 = Rz @ (C - P0) + P0 + shift; R2 = R @ Rz.T; C_own[k] = C
                    E[:3, :3] = R2; E[:3, 3] = -R2 @ C2; cams[k]["extrinsic_w2c_4x4"] = E.tolist()
                    if cams[k].get("camera_center_world") is not None:
                        cams[k]["camera_center_world"] = C2.tolist()
                r["转了_度"] = 180; moved = True
        # ★ 2026-11-12 按人定这个镜头放在哪(有人的镜头; 没人 / 人都在坏相机帧里的退回下面"按相机离地"):
        #   1) 人脚踩在主场景地面上: 人脚(terrain.people_feet)落在主场景量到的地面 / 台阶上的, 量"主场景地面 - 脚"的中位,
        #      超过 FEET_TOL 就整个镜头(相机 + 跟着相机的人和深度)竖直挪这么多。
        #   2) 相机看得见人: 每帧相机到人(脚上 0.5 m)连线, 线上不能低于主场景铺了网格的地面 / 台阶; 按 1) 挪完还被挡的
        #      (90% 分位超过 0.3 m), 再往上挪到不挡(多 0.1 m), 这时人会悬空, 记进报告。
        #   转过向又搜过位置的镜头, "搜到的位置"和"原地只转向"两个都按 1) 2) 算一遍, 取挡得少的(一样就取搜到的)。
        #   joker 往下看的镜头(9~12 秒): 按相机离地放, Joker 在主场景楼梯下 4.23 m, 检查视频那段全灰(第 18 条);
        #   重跑时搜位置挪了 7.5 m, 人脚踩在主场景地面上(差 0.0 m), 但视线被中间的台阶挡 4.9 m, 又全灰 ——
        #   原地只转向再按人脚抬 4.23 m, 人踩地、视线也通。主场景的楼梯量成 11°, 这个镜头自己约 27°, 两套几何对不上。
        nf_shift = None
        _kg = [k for k in ks if k not in BAD and os.path.exists(f"{D}/depth/{k[:-4]}_depth.npy")]
        if not _kg:   # 整个镜头都是坏相机帧(joker 推特写): 人和相机是一体的(cam_smooth 已拟合成匀速), 用这些帧的人脚
            _kg = [k for k in ks if os.path.exists(f"{D}/depth/{k[:-4]}_depth.npy")]; r["用坏相机帧的人脚"] = True
        _pf = terrain.people_feet(r1, D, _kg, cams, vfm, T.up) if _kg else {}
        _cc = {}
        for k_ in _kg:
            E_ = np.array(cams[k_]["extrinsic_w2c_4x4"], float); _cc[int(vfm[k_])] = -E_[:3, :3].T @ E_[:3, 3]

        def _eval(o):
            gaps, occ = [], []
            for vf_, P_ in _pf.items():
                C_ = _cc[vf_] - o
                for q in P_ - o:
                    if main_known(q):
                        gaps.append(float(T.height_at(q)[0]) - float(q @ T.up))
                    Q_ = q + 0.5 * T.up; S_ = C_ + np.linspace(0.05, 0.95, 40)[:, None] * (Q_ - C_)
                    hz = S_ - np.outer(S_ @ T.up, T.up) - (C_ - (C_ @ T.up) * T.up)
                    S_ = S_[np.linalg.norm(hz, axis=1) > 0.5]
                    if len(S_):
                        mh = mesh_h(S_)
                        if np.isfinite(mh).any():
                            occ.append(float(np.nanmax(mh - S_ @ T.up)))
            d = float(np.median(gaps)) if len(gaps) >= MIN_FEET else 0.0
            d = d if abs(d) > FEET_TOL else 0.0
            resid = float(np.percentile(np.array(occ) - d, 90)) if len(occ) >= MIN_FEET else None
            # 相机不能在地面下: 每帧相机离脚下(主场景铺了网格的地 / 台阶)要 >= CAM_MIN_CLEAR, 不够就算进"还要抬多少"
            cn = []
            for vf_ in _pf:
                C_ = (_cc[vf_] - o)[None]; mh = mesh_h(C_)
                if np.isfinite(mh[0]):
                    cn.append(CAM_MIN_CLEAR - (float(C_[0] @ T.up) + d - float(mh[0])))
            if cn:
                cneed = float(np.max(cn))
                resid = cneed if resid is None else max(resid, cneed)
            return dict(人脚_落在主场景地面上的=len(gaps), 主场景地面减人脚_中位_m=round(float(np.median(gaps)), 2) if gaps else None,
                        按人脚挪_m=round(d, 2), 挪完视线还被挡_m_p90=None if resid is None else round(resid, 2), 视线条数=len(occ)), d, resid, len(gaps), len(occ)
        if T.model and _pf:
            o_best, rp = people_place(_pf, _cc, mesh_h)
            r["以人为锚"] = rp
            if o_best is not None:      # o_best 已含竖直(人脚落地 + 必要的再抬), 直接用, 不再走下面的按人脚 / 视线
                for k in ks:
                    E = np.array(cams[k]["extrinsic_w2c_4x4"], float); R = E[:3, :3]; C = -R.T @ E[:3, 3] - o_best
                    E[:3, 3] = -R @ C; cams[k]["extrinsic_w2c_4x4"] = E.tolist()
                    if cams[k].get("camera_center_world") is not None:
                        cams[k]["camera_center_world"] = C.tolist()
                moved = True; r["选"] = "以人为锚"; rep.append(r)
                continue
            cands = [("当前位置", np.zeros(3))]
        elif nf_shift is not None:
            cands = [("当前位置", np.zeros(3)), ("搜到的位置", -nf_shift)]
        else:
            cands = [("搜到的位置" if "搜位置" in r and "水平挪_m" in r.get("搜位置", {}) else "当前位置", np.zeros(3))]
            if cands[0][0] == "搜到的位置":
                cands.append(("原地只转向", shift))
        best = None
        for name, o in cands:
            rp, d, resid, ng, no = _eval(o)
            if ng < MIN_FEET and no < MIN_FEET:
                continue
            score = max(resid or 0.0, 0.0) + (0.0 if ng >= MIN_FEET else 0.5)
            r.setdefault("按人定位置", {})[name] = rp
            if best is None or score < best[0] - 1e-6:
                best = (score, name, o, d, resid)
        if best is not None:
            _, name, o, d, resid = best
            lift = (resid + 0.1) if (resid is not None and resid > 0.3) else 0.0
            dz = d + lift
            if np.linalg.norm(o) > 0 or abs(dz) > 0:
                for k in ks:
                    E = np.array(cams[k]["extrinsic_w2c_4x4"], float); R = E[:3, :3]; C = -R.T @ E[:3, 3] - o + dz * T.up
                    E[:3, 3] = -R @ C; cams[k]["extrinsic_w2c_4x4"] = E.tolist()
                    if cams[k].get("camera_center_world") is not None:
                        cams[k]["camera_center_world"] = C.tolist()
                moved = True
            r.update(选=name, 竖直挪_m=round(dz, 2), 人悬空_m=round(lift, 2))
            rep.append(r)
            continue
        mc, oc = [], []
        for k in ks:
            E = np.array(cams[k]["extrinsic_w2c_4x4"], float); C = -E[:3, :3].T @ E[:3, 3]
            if not main_known(C):
                continue
            h = C @ T.up
            _oh = own_h(C_own.get(k, C), s)
            mc.append(h - float(T.height_at(C)[0])); oc.append(h - _oh if _oh is not None else _camh)
        r["落在主场景地面上的帧"] = len(mc)
        if len(mc) >= MIN_FRAMES:
            r.update(主离地_中位=round(float(np.median(mc)), 2), 自离地_中位=round(float(np.median(oc)), 2))
            if np.median(mc) < MIN_CLEAR:
                off = float(np.median(np.array(oc) - np.array(mc)))
                for k in ks:
                    E = np.array(cams[k]["extrinsic_w2c_4x4"], float); R = E[:3, :3]; C = -R.T @ E[:3, 3] + off * T.up
                    E[:3, 3] = -R @ C; cams[k]["extrinsic_w2c_4x4"] = E.tolist()
                    if cams[k].get("camera_center_world") is not None:
                        cams[k]["camera_center_world"] = C.tolist()
                r["竖直挪_m"] = round(off, 2); moved = True
        rep.append(r)
    out = dict(主镜头=main_s + 1, 门槛_m=MIN_CLEAR, 各镜头=rep)
    if moved:
        bak = f"{D}/cameras_before_shotreg.json"
        if not os.path.exists(bak):
            shutil.copy(p, bak)
        doc["镜头竖直对齐"] = out
        json.dump(doc, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"[镜头对齐] {json.dumps(out, ensure_ascii=False)}" + ("" if moved else " (没有要挪的)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    main(ap.parse_args().r1)
