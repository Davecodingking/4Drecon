#!/usr/bin/env python
"""标定并施加全局尺度, 把 DA3 那一层从"它以为的米"改成真米。

为什么需要这一步 —— DA3 自称输出米制(它在 model/da3.py 里已经把 scale_factor 乘进深度和
相机了), 但它的米制头在俯拍素材上判得很离谱。拿场景里已知真实尺寸的物体当尺子量:
  timesquare  车高量到 1.09 m (真实 1.48) -> 差 1.36 倍
  carpark     车高量到 0.23 m (真实 1.48) -> 差 6.55 倍
carpark 八类里六类挤在 6.1~7.2, 说明这是一个干净的全局常数, 一个数就能修。

不修会怎样(都不是理论问题, 都实测过):
  1. 物理参数是绝对米。SNAP=0.8 在压缩 6.55 倍的世界里等于 5.2 米, 完全不是设计意图。
  2. assemble_scene 的 "ext[2]<0.15 就丢" 在误杀: carpark 长椅真实 0.85 m 量出来 0.13 m, 被跳过。
  3. 接不进任何米制的东西。WildDet3D 输出真米, 和压缩世界差 2.37 倍。
  4. 阈值不能跨场景搬。

★ 改在 DA3 那一层, 不在下游。相反的做法(存个 json 让每个消费者自己乘)刚踩过同款坑:
  render_back.py 不知道 assemble_scene.py 换了导出旋转, 四张渲染回图全废。读 cameras.json
  或深度的脚本有 14 个, 每一个都是一个"忘了乘"的机会。就地改数据, 下游一行都不用动。

乘 k:  depth/*_depth.npy、extrinsic_w2c_4x4 的平移列、camera_center_world、
       points.ply 的顶点、world_up.json 的 ground_h
不动:  intrinsic_3x3、input_hw、mean_conf、*_conf.npy、world_up.json 的 up(单位向量)

环境变量:
  R1                结果目录
  SCALE_MIN_POINTS  实例点数下限, 默认跟 MIN_POINTS 走(观测太稀的高度不可信)
  SCALE_MIN_INST    每类至少几个实例才当尺子, 默认 2
  SCALE_MIN_REAL    参照物真实高度下限(米), 默认 0.5 —— 井盖、地砖这种扁的当不了竖直尺子
  SCALE_OUTLIER     离群倍数, 和加权中位数差这么多倍的类丢掉, 默认 2.0
  SCALE_AGREE_BAND  "算作一致"的带宽, 默认 1.5 倍
  SCALE_AGREE_MIN   保险闸: 落在带内的实例占比低于它就拒绝施加, 默认 0.6
  SCALE_APPLY=0     只估计不施加(想先看看数字时用)
  SCALE_FORCE=1     绕过幂等保护 (危险: 会二次缩放)
"""
import os, json, shutil, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from word_rank import base_word

R1   = os.environ.get("R1", "/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
DA3  = f"{R1}/02_da3"
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
CAMJ = f"{DA3}/cameras.json"
OUT  = f"{DA3}/world_scale.json"
# ★ 没施加时报告写到**另一个文件**。原来不管施加没施加都写 world_scale.json, 而它正是
#   流水线声明的 scale 那一步的产物 —— 于是 SCALE_APPLY=0 或一致率不达标之后, --check
#   从此永远显示 scale "全部就绪", 重跑也不会被判过期。那不是绕过闸门, 是把失败洗成绿灯。
EST  = f"{DA3}/world_scale_estimate.json"

def write_est(report):
    """没施加时: 报告写进 EST, 并且**把旧的 world_scale.json 清掉**。
    ★ 光写 EST 不够 —— 上一轮留下的 world_scale.json 还躺在盘上, --check 只看文件在不在,
      于是闸门照样是绿的, 而且闸 3 会一直展示那份旧数字。"没施加"这件事必须体现在
      "那个代表已施加的文件不存在"上, 否则等于没改。
    """
    json.dump(report, open(EST, "w"), ensure_ascii=False, indent=1)
    if os.path.exists(OUT):
        os.replace(OUT, f"{OUT}.stale")
        print(f"[清理] 旧的 {os.path.basename(OUT)} 是上一轮的, 这次没施加 -> 改名成 .stale")
BAK  = f"{DA3}/_prescale"

MIN_POINTS = int(os.environ.get("SCALE_MIN_POINTS", os.environ.get("MIN_POINTS", "300")))
MIN_INST   = int(os.environ.get("SCALE_MIN_INST", "2"))
MIN_REAL   = float(os.environ.get("SCALE_MIN_REAL", "0.5"))
OUTLIER    = float(os.environ.get("SCALE_OUTLIER", "2.0"))
AGREE_BAND = float(os.environ.get("SCALE_AGREE_BAND", "1.5"))
AGREE_MIN  = float(os.environ.get("SCALE_AGREE_MIN", "0.6"))
APPLY      = os.environ.get("SCALE_APPLY", "1") == "1"
FORCE      = os.environ.get("SCALE_FORCE", "0") == "1"

# ---------- 0. 幂等: 乘过就不能再乘 ----------
# ★ "上次跑到一半断了"的标记要在**最前面**查。原来它埋在第 5 节施加代码里, 而 k≈1
#   那条路在第 3 节就 exit 0 了 —— 于是半缩放的数据会被盖章成"本来就是对的",
#   之后每次重跑都在下面这个幂等分支退出, 那个标记永远不会再被看到。
_prog = f"{DA3}/.scaling_in_progress"
if os.path.exists(_prog) and os.environ.get("SCALE_FORCE", "0") != "1":
    raise SystemExit(f"[失败] 上一次缩放跑到一半就断了({_prog} 还在)。\n"
                     f"       盘上的深度可能已经乘过一部分, 再往下走会二次缩放。\n"
                     f"       只有一条路: 从 {DA3}/_prescale/ 还原原始数据, 再删掉那个标记。\n"
                     f"       ★ 别用 SCALE_FORCE=1 绕 —— 已经乘过的那批会再乘一遍变成 k1*k2,\n"
                     f"         而账本上只会记 k2, 之后再也查不出来。'半缩放'这件事没法确认没问题。")
cams_doc = json.load(open(CAMJ))
already = cams_doc.get("metric_scale_applied")
if already is not None and not FORCE:
    print(f"[跳过] 这个场景已经标定过了, 施加过的倍数是 {already:.3f}。")
    print("       再乘一次会把世界放大两遍。真要重来: 先从 02_da3/_prescale/ 还原, 或者 SCALE_FORCE=1。")
    # ★ 报告丢了就补一份再退。否则会卡死在一个自相矛盾的状态: cameras.json 说已标定,
    #   而 world_scale.json 不在 -> --check 永远报"缺", 重跑每次都在这里 exit 0 什么也不做。
    if not os.path.exists(OUT):
        # ★ 字段必须凑齐 —— 闸 3 是直接下标取 k/一致率/采用类数/候选类数/施加/逐类 的,
        #   少一个就 KeyError, 而 gate_report 有 try/except 兜着, 于是闸 3 整段退化成
        #   一行"检查本身出错了", 连该看的图都不列。让报告的形状始终自洽, 别指望消费者各自防守。
        json.dump(dict(k=round(float(already), 4), 一致率=None, 离散=None,
                       采用类数=0, 候选类数=0, 闸门=AGREE_MIN, 逐类=[], 一致率达标=None,
                       施加=True, 说明="补写: 世界已经标定过, 原报告丢了, 只有倍数是真的",
                       历史=cams_doc.get("metric_scale_history")),
                  open(OUT, "w"), ensure_ascii=False, indent=1)
        print(f"       (顺手补回了 {os.path.basename(OUT)} —— 它之前不在, --check 会一直说没做完)")
    sys.exit(0)

# ---------- 1. 尺子 ----------
sizes_p = f"{R1}/01_vlm_sizes.json"
if not os.path.exists(sizes_p):
    raise SystemExit(f"[失败] 没有尺子 {sizes_p}, 先跑 vlm_sizes.py")
REAL = json.load(open(sizes_p))
boxes = json.load(open(f"{R1}/06_all_boxes.json"))
print(f"[尺子] {len(REAL)} 个词有参考高度; 盒子 {len(boxes)} 类 "
      f"{sum(len(v) for v in boxes.values())} 个实例")

# ---------- 2. 逐类算倍数 ----------
rows = []
for w, lst in boxes.items():
    # 派生子类(umbrella~1)在常识高度表里没有 —— 回退到母词, 否则那一类不参与标定。
    real = REAL.get(w) if REAL.get(w) is not None else REAL.get(base_word(w))
    # 扁的东西当不了竖直尺子: 井盖真实厚度几厘米, 量出来是噪声, 比出来的倍数能到 11 倍。
    # 用"参照高度本身够不够高"来筛, 不用手写黑名单。
    if real is None or float(real) < MIN_REAL: continue
    hs = [b["尺寸"][2] for b in lst if b.get("点数", 0) >= MIN_POINTS and b["尺寸"][2] > 1e-4]
    if len(hs) < MIN_INST: continue
    med = float(np.median(hs))
    rows.append(dict(词=w, 实例数=len(hs), 量到=round(med, 3), 真实=float(real),
                     倍数=round(float(real)/med, 4)))
if len(rows) < 2:
    raise SystemExit(f"[失败] 能当尺子的类只有 {len(rows)} 个, 不够。"
                     f"(要求: 词在尺子表里、点数 >= {MIN_POINTS}、每类 >= {MIN_INST} 个实例)")

def wq(v, w, q):
    """按实例数加权的分位数。不加权是不对的: 19 辆车和 2 台叉车不该一人一票 ——
    实例多的类中位数更稳, 也更可能是场景的主体。"""
    o = np.argsort(v); v = np.asarray(v, float)[o]; w = np.asarray(w, float)[o]
    return float(np.interp(q, (np.cumsum(w) - 0.5*w)/w.sum(), v))

f_all = np.array([r["倍数"] for r in rows])
n_all = np.array([r["实例数"] for r in rows], float)
med0  = wq(f_all, n_all, 0.5)
keep  = np.abs(np.log(f_all/med0)) < np.log(OUTLIER)
if keep.sum() < 2:                       # 全被剔光就退回不剔
    print("[注意] 剔离群后不足 2 类, 这一步不剔了")
    keep = np.ones(len(rows), bool)
for r, k in zip(rows, keep): r["采用"] = bool(k)

kept = [r for r in rows if r["采用"]]
fk = np.array([r["倍数"] for r in kept]); nk = np.array([r["实例数"] for r in kept], float)
k_hat = wq(fk, nk, 0.5)
spread = float(wq(fk, nk, 0.9)/max(wq(fk, nk, 0.1), 1e-9))
# ★ 闸门看的是"一致率"不是"离散"。
# 离散是个太苛刻的判据: 总有一两类的参照高度被 VLM 说错(路灯说成 3.5 米), 或者高度被遮挡
# 截短(俯拍只看得到卡车顶), 这几类会把 90/10 分位撑开, 但它们并不影响加权中位数。
# 真正该问的是"大部分证据同不同意", 所以统计落在 k 的 ±AGREE_BAND 倍带内的实例占比。
inband = (fk > k_hat/AGREE_BAND) & (fk < k_hat*AGREE_BAND)
agree = float(nk[inband].sum()/nk.sum())

print(f"\n{'词':14s} {'实例':>4s} {'量到高':>7s} {'真实':>6s} {'倍数':>6s}")
for r in sorted(rows, key=lambda x: -x["倍数"]):
    print(f"{r['词']:14s} {r['实例数']:4d} {r['量到']:7.2f} {r['真实']:6.2f} {r['倍数']:6.2f}"
          + ("" if r["采用"] else "   <- 离群, 不用"))
print(f"\n[结果] 采用 {len(kept)}/{len(rows)} 类, 加权中位倍数 k = {k_hat:.3f}")
print(f"       一致率 {100*agree:.0f}% 的实例落在 k 的 {1/AGREE_BAND:.2f}~{AGREE_BAND:.2f} 倍带内 "
      f"(闸门 {100*AGREE_MIN:.0f}%), 加权离散(90/10) {spread:.2f}")
for r, ib in zip(kept, inband):
    if not ib: print(f"       带外: {r['词']} ({r['实例数']}个, {r['倍数']:.2f}) —— 参照高度或量到的高度有问题")

report = dict(k=round(k_hat, 4), 一致率=round(agree, 3), 离散=round(spread, 3),
              采用类数=len(kept), 候选类数=len(rows), 闸门=AGREE_MIN, 逐类=rows,
              一致率达标=bool(agree >= AGREE_MIN),
              说明="k = 真实米 / DA3 米")

# ---------- 3. 保险闸 ----------
# ★ 一致率的判定要**先记进报告再分支**。原来 SCALE_APPLY=0 排在闸门前面直接 exit,
#   于是"一致率其实不达标"这件事根本不进 JSON —— 人后来翻报告只看到"拒绝原因: SCALE_APPLY=0",
#   会以为是自己主动跳过的, 而真相是这个场景的尺子本身就对不上。
if not APPLY:
    report["施加"] = False
    report["拒绝原因"] = ("SCALE_APPLY=0, 只估计" if agree >= AGREE_MIN else
                          f"SCALE_APPLY=0, 只估计(而且一致率 {agree:.2f} 本来也低于闸门 {AGREE_MIN})")
    write_est(report)
    print(f"[只估计] SCALE_APPLY=0, 没动数据。报告 -> {EST}")
    print(f"       ★ 没有写 {os.path.basename(OUT)} —— 这一步在 --check 里仍然算没做完, 这是故意的。")
    sys.exit(0)
if agree < AGREE_MIN:
    report["施加"] = False
    report["拒绝原因"] = f"一致率 {agree:.2f} 低于闸门 {AGREE_MIN}"
    write_est(report)
    print(f"[拒绝] 只有 {100*agree:.0f}% 的实例同意这个倍数, 低于闸门 {100*AGREE_MIN:.0f}%, 不敢施加。")
    print("       ★ 退非零码: 未校准的世界继续往下跑, 所有绝对米阈值都会静默失准\n       (SNAP/MARGIN/合法性守卫)。")
    print( "       三条能正确往下走的路, 按推荐顺序:\n"
           "         1. 改 01_vlm_sizes.json 里那个词的参照高度 —— 报告里 带外 那几行就是嫌疑人,\n"
           "            VLM 把路灯说成 1 米这种错, 改一个数就能把一致率拉回来;\n"
          f"         2. 放宽 SCALE_AGREE_MIN(现在 {AGREE_MIN})或 SCALE_AGREE_BAND, 前提是你看过带外那几行;\n"
           "         3. SCALE_APPLY=0 —— 只估计不施加, 明确接受 这个场景不校准。")
    print(f"       尺度改错比不改更糟。报告写在 {EST}, 自己看一眼是哪个词的参照高度不对。")
    sys.exit(2)
if abs(np.log(k_hat)) < np.log(1.05):
    print(f"[无需施加] k = {k_hat:.3f}, 和 1 差不到 5%, 当作本来就是对的。")
    # ★ 还是要写进 cameras.json。否则下游(assemble_scene 的守卫)看不到这个字段,
    #   会当成"这个场景从没标定过"天天报警 —— 而这其实是**正常结果**, 室内几乎每次都走这条。
    #   醒目警告出现在正常结果上, 练几次就没人看了。
    _cd = json.load(open(CAMJ))
    _cd.setdefault("metric_scale_applied", 1.0)
    _cd.setdefault("metric_scale_history", []).append(round(k_hat, 6))
    _cd["metric_scale_checked"] = True
    json.dump(_cd, open(CAMJ, "w"), ensure_ascii=False, indent=1)
    # ★ 这条写 OUT 不写 EST, 是有意的: 它算"这一步做完了, 结论是无需施加" ——
    #   cameras.json 拿到了 metric_scale_applied=1.0, 下游守卫满足, 重跑幂等。
    #   和上面两条"拒绝"不是一回事, 别按"没施加就写 EST"一刀切。
    report["施加"] = False; report["拒绝原因"] = "k 已经接近 1"; report["无需施加"] = True
    for _junk in (f"{OUT}.stale", EST):      # 一个场景只该留一个说法
        if os.path.exists(_junk): os.remove(_junk)
    json.dump(report, open(OUT, "w"), ensure_ascii=False, indent=1)
    sys.exit(0)


# ---------- 4. 备份 ----------
if not os.path.exists(BAK):
    os.makedirs(BAK)
    shutil.copy2(CAMJ, f"{BAK}/cameras.json")
    shutil.copytree(f"{DA3}/depth", f"{BAK}/depth")
    for f in ("points.ply", "world_up.json"):
        if os.path.exists(f"{DA3}/{f}"): shutil.copy2(f"{DA3}/{f}", f"{BAK}/{f}")
    print(f"[备份] 原始数据 -> {BAK}")
else:
    print(f"[备份] {BAK} 已存在, 不覆盖(里面是真正的原始数据, 别弄丢)")

# ---------- 5. 施加 ----------
# ★ 先落一个"正在缩放"的标记再动数据。深度是逐张就地乘的, 而幂等标记原来要等到
#   全部乘完、写 cameras.json 时才落盘 —— 中途 Ctrl-C 或崩溃, 下次重跑会把已经乘过的
#   那批再乘一遍, 而且事后查不出来。宁可多一个要人手动清的标记, 也不要二次缩放。
open(_prog, "w").write(json.dumps(dict(k=k_hat), ensure_ascii=False))   # 检查在文件开头
n = 0
for f in sorted(os.listdir(f"{DA3}/depth")):
    if not f.endswith("_depth.npy"): continue      # 置信度不是长度, 不动
    p = f"{DA3}/depth/{f}"
    np.save(p, (np.load(p) * k_hat).astype(np.float32)); n += 1
print(f"[深度] {n} 张 x{k_hat:.3f}")

for name, cam in cams_doc["cameras"].items():
    flat = np.array(cam["extrinsic_w2c_4x4"]).ndim == 1     # 存的是嵌套 4x4 还是拍平的 16, 原样写回
    E = np.array(cam["extrinsic_w2c_4x4"], dtype=float).reshape(4, 4)
    E[:3, 3] *= k_hat                              # 只动平移, 旋转不动
    cam["extrinsic_w2c_4x4"] = E.reshape(-1).tolist() if flat else E.tolist()
    if "camera_center_world" in cam:
        cam["camera_center_world"] = (np.array(cam["camera_center_world"], float) * k_hat).tolist()
# ★ 累乘不覆盖。SCALE_FORCE=1 再跑一次时, 世界已经被乘过一遍 k1, 这次又乘 k2,
#   真实总倍数是 k1*k2。原来直接写 k2, 幂等记录本身就成了假的, 而且再也查不出来。
_prev = float(cams_doc.get("metric_scale_applied") or 1.0)
cams_doc["metric_scale_applied"] = _prev * k_hat
_hist = cams_doc.setdefault("metric_scale_history", [])
if not _hist and _prev != 1.0:
    _hist.append(round(_prev, 6))    # 存量场景: 之前那次没记账, 这里补上, 否则总倍数仍然查不出来
_hist.append(round(k_hat, 6))
if _prev != 1.0:
    print(f"[累乘] 之前已施加 {_prev:.3f}, 这次 {k_hat:.3f}, 总倍数 {_prev*k_hat:.3f}")
cams_doc["convention"] = (cams_doc.get("convention", "") +
                          f"  | 已用 world_scale.py 乘 {k_hat:.4f} 校准到真米")
json.dump(cams_doc, open(CAMJ, "w"), ensure_ascii=False, indent=1)
print(f"[相机] {len(cams_doc['cameras'])} 个位姿的平移量 x{k_hat:.3f} (旋转和内参没动)")

up_p = f"{DA3}/world_up.json"
if os.path.exists(up_p):
    u = json.load(open(up_p))
    if u.get("ground_h") is not None:
        u["ground_h"] = float(u["ground_h"]) * k_hat          # up 是单位向量, 不动
        json.dump(u, open(up_p, "w"), ensure_ascii=False, indent=1)
        print(f"[地面高度] x{k_hat:.3f} -> {u['ground_h']:.2f} m")

ply = f"{DA3}/points.ply"
if os.path.exists(ply):
    try:
        import trimesh
        pc = trimesh.load(ply)
        pc.vertices = np.asarray(pc.vertices) * k_hat
        pc.export(ply)
        print(f"[点云] points.ply 顶点 x{k_hat:.3f}")
    except Exception as e:
        print(f"[点云] points.ply 缩放失败({e}) —— 它只给可视化用, 不影响主链")

report["施加"] = True
report["说明"] = "k = 真实米 / DA3 米。已乘进 depth、相机平移、camera_center_world、points.ply、world_up.ground_h"
json.dump(report, open(OUT, "w"), ensure_ascii=False, indent=1)
for _junk in (f"{OUT}.stale", EST):
    if os.path.exists(_junk): os.remove(_junk)
os.remove(_prog)      # 数据动完了, 撤掉"正在缩放"的标记
print(f"[报告] -> {OUT}")

# ---------- 6. 把上游的完成戳往前推 ----------
# ★ 必须排在重跑 boxes **之前**: 重跑失败会直接 SystemExit, 推戳就永远不执行,
#   而那时数据已经就地乘过了 —— 洞原样复现。
# ★ 这是"就地改数据"这个设计漏掉的一环。驱动判断谁过期, 比的是 logs/.stamp_<步骤> 的
#   时间戳 —— 也就是"步骤什么时候跑的"。而这一步改的是**步骤脚下的数据**(深度 npy、
#   相机外参、points.ply), 不动 .stamp_da3。于是任何只声明依赖 da3 的步骤, 永远不会
#   被判成过期, --check 一直显示"新的"。
#   把 da3 和 boxes 的戳推到现在, 依赖它们的步骤才会被正确判成该重跑。
_now = time.time()
for _k in ("da3", "up"):
    _sp = f"{R1}/logs/.stamp_{_k}"
    if os.path.exists(_sp):
        os.utime(_sp, (_now, _now))
        print(f"[戳] 推新 .stamp_{_k} —— 它脚下的数据被就地改过了, 下游该重跑")
# ★ boxes 的戳反过来: 现在盘上那份盒子是**缩放前**量的, 已经作废了。
#   原来把它和 da3/up 一起推到 now, 等于宣布"盒子是新的" —— 而万一下面重跑失败并 SystemExit,
#   --check 会理直气壮显示 boxes 全部就绪, 拿旧坐标系的盒子一路跑到组装。
#   正确做法是先把戳拿掉(此刻它确实没做完), 重跑成功之后再补回去。
_bs = f"{R1}/logs/.stamp_boxes"
if os.path.exists(_bs):
    os.remove(_bs)
    print("[戳] 撤掉 .stamp_boxes —— 盘上的盒子是缩放前量的, 重跑成功之前它就是没做完")

# ---------- 6. 盒子过期了, 重量一遍 ----------
# 纯 CPU, 只读 03_masks/*.npz, 十几秒。环境变量原样继承(MIN_POINTS/CONF_MIN/VOX 都在里面)。
print("\n[重跑] 深度变了, 盒子过期, 重新量一遍 ...", flush=True)
r = subprocess.run([sys.executable, f"{SCRIPTS}/boxes_from_masks.py"], env=os.environ)
if r.returncode != 0:
    raise SystemExit(f"[失败] 重跑 boxes_from_masks.py 退出码 {r.returncode} —— "
                     f"数据已经缩放过了, 别再跑一次 scale, 直接单独重跑量盒子那步")
os.makedirs(os.path.dirname(_bs), exist_ok=True)   # 手动单跑时 logs/ 可能还不存在
open(_bs, "w").close()      # 重跑成功了, 现在盒子才真的是新的
print(f"\n[done] 全场景已校准到真米, k = {k_hat:.3f}")
