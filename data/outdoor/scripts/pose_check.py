"""DA3 坏位姿帧筛选 —— 不用真值, 只用 DA3 自己的深度和位姿互相对账。

★ 为什么要有 (devLog/devlog_9.16.md 第 2、3 节):
  ithor312 上 DA3 逐帧深度很准(误差中位 0.9%), 但有 11 帧位姿错了 10–36°(画面几乎全是白墙,
  轨迹每帧转约 16°)。深度准、位置放错 -> 重影墙、斜墙, 地面拟合内点只剩 29%, 定天也跟着不稳。
  流水线原来没有任何逐帧位姿检查。

原理: 深度准的话, 用估出的位姿把第 i 帧的点投到相邻帧, 落点深度应该和那一帧自己的深度对得上。
  分数 = 落点深度误差 < 5% 的像素占比, 取和前后各 2 个(已保留)帧比的中位数。
  分数 < 0.6 算可疑; 每轮只剔最可疑的三分之一, 剩下的重新打分(一次性剔的话, 好帧会被旁边的
  坏帧连累); 最多剔 25%。
  ithor312 对真值核对: 剔 15 帧, 命中真坏帧 12/13, 误剔 3 帧, 漏 1 帧。

已知代价: 会误伤少量好帧; 只被坏帧拍到的区域会变空; 大面积运动物体可能导致误判。

谁用: da3_to_ply.py 推理完当场筛, 坏帧不进 points.ply, 名单写 bad_frames.json;
  "把像素放进 3D"的步骤(定天 / 量盒子 / 召回检测)通过 world_frame.usable_frames() 跳过它们。
  这个文件只用 numpy —— spatiallm 和 sam3d 两个环境都要 import 它。

单独跑(给已有轮次补一份名单, 不重跑 DA3):
  python pose_check.py <02_da3 目录> [--write]
  ★ 这样补的名单只影响之后的步骤; 已经拼好的 points.ply 里坏帧的点还在, 要干净得重跑 da3。
"""
import json
import os
import sys

import numpy as np

BAD_FRAMES_FILE = "bad_frames.json"
PARAMS = dict(stride=4, tol=0.05, th=0.6, cap_frac=0.25, win=2, max_rounds=10, min_px=200)


def find_bad_frames(names, depths, w2c, K, stride=4, tol=0.05, th=0.6, cap_frac=0.25,
                    win=2, max_rounds=10, min_px=200):
    """names: 帧名列表(按时间顺序); depths: 每帧 HxW 深度; w2c: 每帧 4x4; K: 每帧 3x3。
    返回 dict(坏帧=[名字], 分数={名字: 最后一次打分}, 每轮=[[名字...]], 参数=...)。
    逻辑逐条照抄 analysis_9.16/filter_iter.py(那边对真值核过), 只多了 (i,j) 结果缓存 —— 同一对帧
    每轮算出来一样, 缓存不改结果。"""
    n = len(names)
    w2c = [np.asarray(e, np.float64) for e in w2c]
    K = [np.asarray(k, np.float64) for k in K]
    H, W = np.asarray(depths[0]).shape
    wp = {}

    def world_pts(i):
        if i not in wp:
            d0 = np.asarray(depths[i])
            v, u = np.mgrid[0:H:stride, 0:W:stride]
            d = d0[v, u]
            P = np.stack([(u - K[i][0, 2]) * d / K[i][0, 0], (v - K[i][1, 2]) * d / K[i][1, 1],
                          d, np.ones_like(d)], -1).reshape(-1, 4)
            wp[i] = (np.linalg.inv(w2c[i]) @ P.T).T
        return wp[i]

    memo = {}

    def cons(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        C = (w2c[j] @ world_pts(i).T).T
        z = C[:, 2]
        ok = z > 0.05
        u = K[j][0, 0] * C[ok, 0] / z[ok] + K[j][0, 2]
        v = K[j][1, 1] * C[ok, 1] / z[ok] + K[j][1, 2]
        inb = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        if inb.sum() < min_px:
            r = None
        else:
            dj = np.asarray(depths[j])[v[inb].astype(int), u[inb].astype(int)]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = float(np.mean(np.abs(z[ok][inb] - dj) / dj < tol))
        memo[(i, j)] = r
        return r

    keep = np.ones(n, bool)
    cap = int(cap_frac * n)
    rounds, score, drop_score = [], {}, {}
    for _ in range(max_rounds):
        idx = np.where(keep)[0]
        score = {}
        for p, i in enumerate(idx):
            nb = [idx[q] for q in range(max(0, p - win), min(len(idx), p + win + 1)) if q != p]
            s = [x for x in (cons(i, j) for j in nb) if x is not None]
            score[i] = float(np.median(s)) if s else 1.0
        bad = [i for i in idx if score[i] < th]
        if not bad:
            break
        worst = sorted(bad, key=lambda i: score[i])[:max(1, len(bad) // 3)]   # 每轮只剔最差的三分之一
        if (~keep).sum() + len(worst) > cap:
            break                                                             # 再剔就超 25%, 停手
        keep[worst] = False
        rounds.append([names[i] for i in worst])
        drop_score.update({i: score[i] for i in worst})      # 被剔的帧记它被剔那一轮的分数
    return dict(
        坏帧=[names[i] for i in np.where(~keep)[0]],
        # 留下的帧: 最后一轮的分数; 剔掉的帧: 被剔那一轮的分数
        分数={names[i]: round(float(drop_score.get(i, score.get(i, float("nan")))), 3) for i in range(n)},
        每轮=rounds,
        参数=dict(stride=stride, tol=tol, th=th, cap_frac=cap_frac, win=win,
                 max_rounds=max_rounds, min_px=min_px),
        帧数=n,
    )


def check_dir(da3_dir, **kw):
    """从盘上的 02_da3(cameras.json + depth/*.npy) 算坏帧。帧顺序 = cameras.json 键名排序(帧名补零过)。"""
    cams = json.load(open(f"{da3_dir}/cameras.json"))["cameras"]
    names = sorted(cams)
    depths = [np.load(f"{da3_dir}/depth/{os.path.splitext(k)[0]}_depth.npy") for k in names]
    return find_bad_frames(names,
                           depths,
                           [cams[k]["extrinsic_w2c_4x4"] for k in names],
                           [cams[k]["intrinsic_3x3"] for k in names], **{**PARAMS, **kw})


def write(da3_dir, res, how):
    res = dict(res, 来源=how)
    tmp = f"{da3_dir}/{BAD_FRAMES_FILE}.tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(tmp, f"{da3_dir}/{BAD_FRAMES_FILE}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = sys.argv[1].rstrip("/")
    r = check_dir(d)
    print(f"[坏位姿] {len(r['坏帧'])}/{r['帧数']} 帧: {[os.path.splitext(x)[0] for x in r['坏帧']]}")
    print(f"         每轮: {[[os.path.splitext(x)[0] for x in rr] for rr in r['每轮']]}")
    if "--write" in sys.argv:
        write(d, r, "pose_check.py 事后补算(points.ply 未重拼)")
        print(f"         -> {d}/{BAD_FRAMES_FILE}")
