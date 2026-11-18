#!/usr/bin/env python
"""看不到地板的场景(world_up.py 走了"单墙 + 相机上"那一支): 用人的脚底定地面。

排在 people_mesh 之后、world_scale 之前(world_scale 最后会按这里定的"天"把盒子重量一遍)。
world_up.py 那时只能拿"相机的上"当天, 高度暂按相机离地 1.5 m。这里拿每个人抽 24 帧的脚底点:
  - 先按现在的"天"量一个水平面(脚底高度中位), 这是原来的做法;
  - ★ 2026-11-09 再试一个可以斜的平面(fit_feet_plane)。相机边走边拍、又没端平时, "相机的上"本身就歪了,
    水平面贴不住: goodfellas 相机走了 2.8 m, 脚底离散 0.195 m(门槛 0.15), 起点附近的人被贴地抬高 0.3 m,
    画面上高出一个头; 斜平面离散 0.065 m, 斜 6.1°, 相机离地从"1.03 -> 1.27 m 越走越高"变成 1.39 -> 1.44 m。
    斜平面明显更贴(离散降到 TILT_GAIN 以下)、斜得不离谱(<= TILT_MAX_DEG)时用它的法线当新的天; 否则照旧水平面。
    squat 那种相机不动、脚点挤在一处的, 人走过的范围不够, 不会斜。
"""
import argparse
import json
import os

import numpy as np

TILT_MAX_DEG = 10.0   # 脚底平面和现在的"天"最多差这么多度, 再多就不信(多半是脚底本身乱)
TILT_GAIN = 0.6       # 斜平面的离散要降到水平面的这么多倍以下才换
SPREAD_MIN = 0.4      # 米: 脚底点沿某个水平方向的标准差不到这么多(走过的范围不到约 1.5 m), 这个方向不许斜
BAND = 0.08           # 米: 拟合时离平面这么近的脚底点算内点(其余是没站在地上 / 量错的帧)

import body_into_scene as B
import body_motion as BM
import mhr_body as M
import people_identity


def fit_feet_plane(P, up):
    """脚底点 P (N,3), 现在的天 up -> 斜平面 dict(up=新的天, h=新的 ground_h, mad, tilt_deg, 斜的方向) 或 None(点太少)。
    在水平面上建坐标(脚底点的两个主方向), 拟合 高度 = c + g1*u1 + g2*u2; 某个方向走过的范围不够(标准差 < SPREAD_MIN)
    就把那个方向的斜率钉成 0 —— 人只沿一条走廊走时, 横着的那个方向没有约束, 放开拟合会随便歪。
    离群点: 先全体拟合, 再只用离平面 BAND 以内的点重拟, 迭代几轮。"""
    up = np.asarray(up, float) / np.linalg.norm(up)
    P = np.asarray(P, float)
    if len(P) < 10:
        return None
    h = P @ up
    Ph = P - np.outer(h, up)
    o = Ph.mean(0)
    _, sv, vt = np.linalg.svd(Ph - o, full_matrices=False)
    E = vt[:2]                                            # 水平面里的两个主方向(和 up 垂直)
    U = (Ph - o) @ E.T
    use = [k for k in range(2) if U[:, k].std() >= SPREAD_MIN]
    inl = np.ones(len(P), bool)
    for _ in range(5):
        A = np.c_[np.ones(inl.sum()), U[inl][:, use]] if use else np.ones((inl.sum(), 1))
        coef, *_ = np.linalg.lstsq(A, h[inl], rcond=None)
        g = np.zeros(2)
        for j, k in enumerate(use):
            g[k] = coef[1 + j]
        res = h - (coef[0] + U @ g)
        new = np.abs(res - np.median(res[inl])) < BAND
        if new.sum() < 10 or (new == inl).all():
            break
        inl = new
    n = up - g @ E                                        # 平面 高度 = c + g·u 的法线(没归一化)
    n /= np.linalg.norm(n)
    P0 = o + coef[0] * up                                 # 平面上一点(水平原点正上方)
    r = (P - P0) @ n
    return dict(up=n, h=float(P0 @ n), mad=float(np.median(np.abs(r - np.median(r)))),
                tilt_deg=float(np.degrees(np.arccos(np.clip(n @ up, -1, 1)))),
                放开斜的方向=len(use), 内点比例=round(float(inl.mean()), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--force", action="store_true", help="also replace an existing manual height")
    a = ap.parse_args()
    up_path = f"{a.r1}/02_da3/world_up.json"
    world = json.load(open(up_path))
    if not a.force and "待Body脚底" not in world.get("来源", ""):
        print("[地面] 已有地面高度，跳过 Body 回填")
        return

    raw = people_identity.raw_run(a.body)
    params, shapes = BM.load_params(raw)
    da3, cams, up, _, vid2key = B.load_scene(a.r1)
    kept = json.load(open(f"{a.body}/00_frames/frames.json"))["kept_indices"]
    roots = B.load_roots(raw)
    all_feet = []
    used = []
    for pid, frames in params.items():
        ids = sorted(b for b in frames if b < len(kept) and kept[b] in vid2key and b in roots.get(pid, {}))
        if len(ids) < 5:
            continue
        take = [ids[i] for i in np.unique(np.linspace(0, len(ids)-1, min(a.samples, len(ids))).astype(int))]
        ratios = B.depth_ratio(pid, take, roots[pid], raw, da3, vid2key, kept)
        if len(ratios) < 3:
            continue
        ratio = float(np.median(list(ratios.values())))
        ct = np.stack([frames[b][1] * ratio for b in take])
        par = np.stack([frames[b][0] for b in take])
        c2w = np.stack([np.linalg.inv(np.asarray(cams[vid2key[kept[b]]]["extrinsic_w2c_4x4"], float))
                        for b in take])
        posed = M.to_world(par, ct, c2w)
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        verts, _ = M.forward(shapes[pid], posed, True, device)
        verts = np.asarray(verts)
        lo = np.argmin(verts @ up, axis=1)
        all_feet.extend(verts[i, lo[i]] for i in range(len(take)))     # 脚底点的 3D 位置(斜平面要用)
        used.append(pid)
    if len(all_feet) < 5:
        raise SystemExit("[失败] 单墙场景没有足够的 Body 脚底帧，需人工指定地面高度")
    P = np.asarray(all_feet)
    feet = P @ up
    h = float(np.median(feet))
    mad = float(np.median(np.abs(feet - h)))
    mad_level, tilt = mad, None
    t = fit_feet_plane(P, up)
    if t and t["tilt_deg"] <= TILT_MAX_DEG and t["mad"] <= TILT_GAIN * mad and t["放开斜的方向"]:
        tilt = t
        up, h, mad = t["up"] / np.linalg.norm(t["up"]), t["h"], t["mad"]
        print(f"[地面] 脚底更贴一个斜平面: 离散 {mad_level:.3f} -> {mad:.3f} m, 天转 {t['tilt_deg']:.1f}°"
              f"(放开 {t['放开斜的方向']} 个方向, 内点 {t['内点比例']:.0%}) —— 相机的上不准, 改用脚底平面的法线当天")
    elif t:
        print(f"[地面] 斜平面没用上: 离散 {mad_level:.3f} -> {t['mad']:.3f} m, 斜 {t['tilt_deg']:.1f}°, "
              f"放开 {t['放开斜的方向']} 个方向(要求离散降到 {TILT_GAIN:.0%} 以下、斜 <= {TILT_MAX_DEG:.0f}°)")
    cam_h = float(np.median([np.linalg.inv(np.asarray(c["extrinsic_w2c_4x4"], float))[:3, 3] @ up
                             for c in cams.values()]) - h)
    if mad > .15 or not .5 <= cam_h <= 2.2:
        raise SystemExit(f"[失败] Body 脚底不稳定或相机高度不合理：MAD={mad:.3f}m，"
                         f"相机离地={cam_h:.2f}m；需人工核对")
    source = ("单墙检验+相机上+Body脚底高度" if "单墙检验" in world.get("来源", "")
              else "单墙约束+Body脚底高度")
    if tilt:
        source += "+脚底平面定天"
        world.update(up=[float(x) for x in up], 相机上的天=world.get("up"), 脚底平面转了度=round(tilt["tilt_deg"], 2),
                     水平面时的脚底MAD=round(mad_level, 3))
    world.update(ground_h=h, 相机离地=round(cam_h, 3), 来源=source,
                 Body脚底帧数=len(feet), Body脚底MAD=round(mad, 3), Body人物=used)
    out = a.out or up_path
    temp = out + ".tmp"
    with open(temp, "w") as f:
        json.dump(world, f, ensure_ascii=False, indent=1)
    os.replace(temp, out)
    print(f"[地面] {len(feet)} 帧 Body 脚底中位 {h:.3f}m，MAD={mad:.3f}m；"
          f"相机离地 {cam_h:.2f}m -> {out}")


if __name__ == "__main__":
    main()
