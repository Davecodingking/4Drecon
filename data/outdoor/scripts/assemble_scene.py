#!/usr/bin/env python
"""同款复制: 一类一个 mesh, 按各实例的盒子摆回街上, 加地面, 导出整场景 glb.
尺度用高度对齐(高度几乎总能看全, 长度会被遮挡截短); 朝向用盒子长轴, 180° 未解。"""
import os, sys, json, numpy as np, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_frame import load_up, export_T, inverse_T
from word_rank import base_word
TAG=os.environ.get("TAG","ts_r1")

ROOT=os.environ.get("R1","/home/jincheng/sam3d/data/outdoor/result/timesquare/r1")
# 写死绝对路径换台机器就 glob 到空, 而空的结果是"什么都没摆进去"而不是报错 —— 静默失败。
OUTS=os.environ.get("GEN_OUT", os.path.expanduser("~/sam3d/sam-3d-objects/outputs"))
# "天"的方向。老的写法是写死 [0,-1,0] = 首帧相机的 -y —— 街头平视时相机水平, 这个假设
# 成立(时代广场实测只偏 0.4°); 但高空俯拍时相机低着头, 它的 -y 指向斜上后方, beach 偏
# 26.2°、carpark 偏 29.2°, 整个场景连同地面一起歪掉。
# 现在改成读 world_up.py 从点云主平面拟合出来的真实法向, 没有那个文件才退回老假设。
# ★ 必须在"量盒子"之前定好: 盒子的高度是沿"天"量出来的, 方向错了盒子本身就是错的,
#   事后旋转只是把错的盒子摆正, 数字不会变对。
# 天的方向和导出旋转都在 world_frame.py 里 —— 这个约定以前每个文件各抄一份, 已经漂移过。
UP, _GH, _FB = load_up(f"{ROOT}/02_da3", "组装")


# ---- 合法性守卫: 这两个数是**绝对真米**, 所以它们的含义取决于尺度校准成没成功 ----
# ★ 校准前 carpark 整个世界被压缩 6.43 倍: 0.85 m 的长椅量出来 0.13 m, 被 MIN_H 当成
#   "太薄"扔掉 —— 上一轮那句"[跳过] 12 个尺寸不合理"大半是这么来的。
#   反过来校准后世界放大了, MAX_D 才第一次有能力误杀。
#   守卫本身没错, 错在它不知道自己站在校准过还是没校准的世界里。所以这里先去问一句。
MIN_H = float(os.environ.get("MIN_H", "0.15"))    # 比这薄就当量错了
MAX_D = float(os.environ.get("MAX_D", "15"))      # 比这大就当量错了
_cj = f"{ROOT}/02_da3/cameras.json"
_k = None
if os.path.exists(_cj):
    try: _k = json.load(open(_cj)).get("metric_scale_applied")
    except Exception: pass
if _k is None:
    print(f"★★ 这个场景没有做过尺度标定(cameras.json 里没有 metric_scale_applied)。\n"
          f"   下面 MIN_H={MIN_H} / MAX_D={MAX_D} 这两个守卫写的是真米, 在未校准的世界里\n"
          f"   含义可能差好几倍, 会误杀。先跑 scale 那一步, 或显式覆盖 MIN_H/MAX_D。", flush=True)
else:
    print(f"[尺度] 这个世界已按 {float(_k):.3f} 倍校准过, 守卫 MIN_H={MIN_H} MAX_D={MAX_D} 是真米", flush=True)
EXCLUDE=set(os.environ.get("EXCLUDE","sign,billboard,pedestrian,manhole cover").split(","))
# ---- 人在闸口逐个否掉的候选 ----
# ★ EXCLUDE 是**词级**的, 挡不住"这个词留着但这一个实例不要", 而闸口最常见的判断恰好是后者。
#   以前这份名单只挡了生成(不烧 GPU), 组装这步从来不看 —— 于是盘上但凡有上一轮留下的 glb,
#   被人否掉的东西照样摆进场景, 而且一声不吭。人的语义是"这东西别出现在场景里",
#   不只是"别再烧这份 GPU", 所以这里必须再挡一道。
import review_decisions as _review
SKIP_NAMES=set(_review.skip_names(ROOT))
def _gen_name(w, oid=None):
    """生成名的拼法: 类共用 = <TAG>_<词>, 实例专属 = <TAG>_<词>_<实例号>。词里的空格换成下划线。"""
    s = f"{TAG}_{w}" if oid is None else f"{TAG}_{w}_{oid}"
    return s.replace(" ", "_")
if SKIP_NAMES:
    print(f"[人工闸] 人工否掉 {len(SKIP_NAMES)} 个, 不摆进场景: {', '.join(sorted(SKIP_NAMES))}", flush=True)
VLM_REJ = _review.vlm_rejected(ROOT)
VLM_REL = _review.vlm_released(ROOT)
if VLM_REJ:
    print(f"[VLM否决] {sum(len(v) for v in VLM_REJ.values())} 个实例被否决, 人工放行 {len(VLM_REL)} 个", flush=True)
# 自动扫描 outputs/。ts_r1_<词> = 该类共用的 mesh; ts_r1_<词>_<实例号> = 这个实例专属的
AVAIL={d for d in os.listdir(OUTS) if d.startswith(TAG+"_") and os.path.exists(f"{OUTS}/{d}/{d}.glb")}
MESH={d[len(TAG)+1:].replace("_"," "):d for d in AVAIL}
# 一个词的全部实例 mesh, 给"没有类 mesh"的词兜底用
INST={}
for _d in AVAIL:
    if _d.rsplit("_",1)[-1].isdigit():
        INST.setdefault(_d[len(TAG)+1:].rsplit("_",1)[0].replace("_"," "), []).append(_d)
# 人在闸口判了"共用一个 mesh"的词。组装必须知道这件事 ——
# ★ 否则会这样: 一个词先 solo 生成过(outputs/ 里躺着 be_r2_light_fixture_7/8/13/14/15),
#   后来在闸 4 被标成 shared, 类 mesh 也出来了, 但 pick_mesh **优先**用实例专属 mesh,
#   于是那 5 个孤儿照样各用各的 —— "shared"对组装是个空操作, 而且不报错。
#   盘上的旧 mesh 不会自己消失, 所以只能在这里按人的决定压过它们。
KEEP_W=set(x.strip() for x in os.environ.get("SOLO_KEEP","").split(",") if x.strip())
def pick_mesh(w, oid):
    """优先用这个实例专属的 mesh, 没有才退回该类共用的。
    第三条退路: 有些词 SAM3 一个实例都没跟到, 全靠召回捡回来 —— 它们从来没生成过类 mesh,
    只有召回代表那一个实例 mesh。没有这条退路, 这些词会整个从场景里消失, 而它们正是召回
    最该体现价值的部分。借这个词自己的任一实例 mesh 用, 比整个丢掉合理。"""
    if (w in KEEP_W or base_word(w) in KEEP_W) and w in MESH:
        return MESH[w]           # 人判了共用, 压过盘上残留的实例专属 mesh(见上面 KEEP_W)
    solo=f"{TAG}_{w.replace(' ','_')}_{oid}"
    if solo in AVAIL: return solo
    if w in MESH: return MESH[w]
    c=INST.get(w)
    return sorted(c)[0] if c else None
SNAP_GROUND=os.environ.get("SNAP_GROUND","1")=="1"   # 强制所有物体落在地面上
# 房间壳 (room_shell.py 出的 floor + wall_*)。给了就用它当地面, 不再铺那张覆盖全场的大方片。
# 室内必须给 —— 大方片是按物体范围 +8m 铺的, cyberpunk 上铺成 30x30m 一块灰板, 一眼就穿帮。
# 室外不给, 走原来的大方片 (街景本来就没有闭合房间)。
# ★ 壳是 room_shell.py 用 --up y 导出的, 已经在最终的 glTF 系里, 所以必须在文件末尾
#   那个 FLIP 之后才并进来, 否则会被转两次。
# ★ 这个变量原来叫 SHELL, 撞上了系统自带的同名环境变量(每个 shell 里它都等于 /bin/bash)。
# 脚本无条件继承到它, 于是每次都走"用房间壳"那条路, 然后拿 /bin/bash 当 mesh 加载崩掉。
# 室内那边显式传了 SHELL= 空值覆盖掉才没事, 室外没传, 整个组装步骤直接挂死(退出码 1)。
# 教训: 环境变量名要挑不会和系统撞的。
# 旧名字兼容一轮: 只认看起来像文件的值。
ROOM_SHELL=os.environ.get("ROOM_SHELL","") or (
    os.environ.get("SHELL","") if os.environ.get("SHELL","").endswith((".glb",".ply",".obj")) else "")
FLOAT_WORDS=set(x for x in os.environ.get("FLOAT_WORDS","awning,traffic light,billboard,sign").split(",") if x)
BOXES=os.environ.get("BOXES", f"{ROOT}/06_all_boxes.json")   # 换成 06_all_boxes_merged.json 即含召回
boxes=json.load(open(BOXES))

cache={}
def get_mesh(key):
    w=key
    if w not in cache:
        s=trimesh.load(f"{OUTS}/{w}/{w}.glb")
        m=s.dump(concatenate=True) if hasattr(s,"dump") else s
        m.apply_translation(-m.bounds.mean(0))          # 中心到原点
        cache[w]=m
    return cache[w].copy()

# --- 地面: 用所有实例的盒底拟合一个平面 (物体都站在地上) ---
pts=[]
for w,bs in boxes.items():
    for b in bs:
        c=np.array(b["中心"]); h=b["尺寸"][2]
        pts.append(c - UP*h/2)                          # 盒底中心
pts=np.array(pts)
# 地面必须是水平的 —— 真实的"天"已经由 up 那步从点云主平面拟合出来了(world_up.json),
# 在这个坐标系里地面按定义就是水平面, 只剩"多高"一个未知数。
#
# 老写法拟合的是带坡度的 y = a*x + b*z + c, 结果被盒底噪声带歪了约 10°:
# carpark 的地面片在 96x67m 的范围上竖直方向跨了 17 米(y 2.08~19.34),
# 直接斜插穿过物体群(物体 y 0.38~14.05)。以前地面是单面背朝上、看不见, 没人发现;
# 一把法向修正过来, 它就糊住了整个画面。
#
# 所以只沿 UP 求一个高度: 1D RANSAC 取"盒底最扎堆"的那一层。
# 用众数而不是最低点/中位数 —— 最低点会被一个飘到地下的误检拉走,
# 中位数会被半空中的广告牌、树冠盒底抬高。
# ★ 地面高度优先取点云直接拟合的那个 (world_up.py 存的 ground_h)。
# 从盒底反推是错的源头: 俯视场景里盒子的高度被压扁, 盒底本身就散了十几米
# (beach 实测跨 14.9m), 拿它推出来的地面必然穿过物体群中间, 上下都贴不上。
# 地面在点云里本来就是一整片实测数据, 不该用一堆错的盒底去猜它。
# ★ 直接用文件开头 load_up 已经拿到的 _GH, 不再自己 json.load 一遍。
#   原来那份是 try/except Exception: pass —— world_up.json 一旦损坏就悄悄退回盒底反推,
#   日志只会说"用 N 个盒底拟合", 看不出这是降级。而降级的后果是整个地面高度错掉。
_gh = _GH

hh = pts @ UP                                    # 每个盒底沿"天"方向的高度
cand = np.sort(hh)
best_n, best_h = -1, float(np.median(hh))
for h0 in cand:                                  # 盒底本身就是最好的候选高度
    n_in = int((np.abs(hh - h0) < 0.5).sum())
    if n_in > best_n: best_n, best_h = n_in, h0
inl = np.abs(hh - best_h) < 0.5
best_h = float(hh[inl].mean())                   # 用内点精修
# ★★ 有房间壳的时候, 地面以**壳的地板**为准, 不用 ground_h。
#   为什么翻过来(2026-09-20 office 实测):
#     ground_h = -0.41, world_up.json 自己记着 "相机离地 0.222 m"、"内点比例 0.378"。
#     手持拍摄不可能离地 22 厘米; 38% 内点说明那个平面本来就没拟稳。
#     用壳的地板 -1.95 算, 相机离地 1.75 m —— 手持正好。
#   物理解释: 办公室里**面积最大的水平面是桌面, 不是地板** —— 地板被桌椅和走动的人挡住了,
#   露出来的点还不如桌面多。world_up.py 那次平面拟合多半拟到了桌面上, 整整高了 1.53 m。
#   (同一份数据的第三个证据: 盒底反推给 -0.88, 在壳地板上方 1.07 m, 正是升降桌的高度 ——
#    因为检测到的绝大多数是桌上的东西, 它们的底本来就该在桌面上。)
#   壳的地板来自 zup 在 03_点云_z朝上.ply 上的拟合, 那条路吃的是完整点云, 更可信。
#   ★ 这只在有壳的时候成立(室内)。室外没有壳, 照旧用 ground_h ——
#     室外马路是画面里最大的一块, 那次拟合本来就稳。
_shell_h = None
if ROOM_SHELL and os.path.exists(ROOM_SHELL):
    try:
        _sh0 = trimesh.load(ROOM_SHELL, force="scene")
        for _nd in _sh0.graph.nodes_geometry:
            if _nd == "floor":
                _T0, _g0 = _sh0.graph[_nd]
                _m0 = _sh0.geometry[_g0].copy(); _m0.apply_transform(_T0)
                # 壳已经是最终 glTF 系(UP 对准 +y), 所以 y 就是沿 UP 的高度, 和 best_h 同一把尺子
                _shell_h = float(np.median(_m0.vertices[:, 1])); break
    except Exception as _e:
        print(f"[地面] 读不出房间壳的地板高度({_e}), 退回 ground_h")
if _shell_h is not None:
    print(f"[地面] 用房间壳的地板 {_shell_h:.2f} m "
          f"(ground_h 给 {_gh:.2f} m, 但它记着相机离地只有 0.22 m、内点 38% —— 多半拟到桌面上了; "
          f"盒底反推给 {best_h:.2f} m)")
    best_h = _shell_h
elif _gh is not None:
    print(f"[地面] 用点云拟合的高度 {_gh:.2f} m (盒底反推会给出 {best_h:.2f} m, 不用它)")
    best_h = float(_gh)
else:
    print(f"[地面] {len(pts)} 个盒底中 {int(inl.sum())} 个在地面上, "
          f"残差中位 {np.median(np.abs(hh[inl]-best_h)):.2f} m (水平面, 高度 {best_h:.2f} m)")

# ---- 立在地上的东西: 盒子补到地面 (2026-09-21) ----
# ★ 盒子只量得到拍到的部分。办公室的桌子腿被椅子和人挡住, 盒子只剩桌面一片薄板
#   (office: table_0 量到 0.83~1.21 m, 厚 0.38 m, 悬在半空), 按这个高度缩 mesh 就是一张 38cm 高的
#   迷你桌; 物理一看悬空, 整张扔到地上, 桌面从 1.2 m 掉到 0.35 m, 桌上的显示器杯子跟着掉。
#   量到的桌面是准的(物理之前渲染回原镜头能对上), 该补的是被挡住的腿: 顶不动, 底放到地面。
#   只补 VLM 说"立在地上"的类 (01_vlm_rest.json), 而且逐个实例再看两道:
#     1. 脚下已经有别的盒子托着(那个盒子的顶就在它底下 EXT_SUP_TOL 内, 它的中心落在那个盒子
#        的水平范围里) -> 它是搁在上面的(桌上的盆栽), 不补。
#     2. 补完的高度超过常识高度(01_vlm_sizes.json)的 EXT_MAX_RATIO 倍 -> 多半不是腿被挡住,
#        是它本来就不在地上(挂墙的、柜顶的), 或者 VLM 判错了类。不补, 交给物理原地不动。
#   室外不给 FLOOR_EXTEND, 整段不走, 逐位不变 (室外靠 SNAP_GROUND 整体落地)。
FLOOR_EXTEND = os.environ.get("FLOOR_EXTEND", "0") == "1"
REST_P = os.environ.get("REST", "")
# 补过地面的, 如果 mesh 自己也是照着"只看得到桌面"那张图生成的(矮桌, 高只有长的 0.24),
# 按补完的高度等比缩会把桌面撑到 5 m 长 —— 这种改成"按量到的那截缩 + 腿往下拉到地面", 见摆放循环。
FOOT_RATIO = float(os.environ.get("FOOT_RATIO", "1.5"))   # 等比缩完水平边超过量到的这么多倍 = mesh 是矮的
LEG_KEEP = float(os.environ.get("LEG_KEEP", "0.2"))       # 拉腿时 mesh 顶上这一截(桌面)原样不变形
_legged = []
if FLOOR_EXTEND and REST_P and os.path.exists(REST_P):
    EXT_MAX_RATIO = float(os.environ.get("EXT_MAX_RATIO", "2.0"))
    EXT_MIN = float(os.environ.get("EXT_MIN", "0.05"))           # 底离地不到这么多, 不用补
    EXT_SUP_TOL = float(os.environ.get("EXT_SUP_TOL", "0.15"))   # 托着它的顶面离它的底要这么近
    _REST = json.load(open(REST_P))
    _szp = f"{ROOT}/01_vlm_sizes.json"; _REAL = json.load(open(_szp)) if os.path.exists(_szp) else {}
    _look = lambda d, w: d.get(w) if d.get(w) is not None else d.get(base_word(w))
    def _bot(b):
        return float(b["底高"]) if "底高" in b else float(np.array(b["中心"]) @ UP - b["尺寸"][2]/2)
    def _rect(b):
        """盒子的水平矩形: 中心, 长轴, 短轴, 两个半长。"""
        c = np.array(b["中心"], float); a = np.array(b["长轴"], float); a = a - UP*(a @ UP)
        a /= max(np.linalg.norm(a), 1e-9)
        return c, a, np.cross(UP, a), b["尺寸"][0]/2, b["尺寸"][1]/2
    _all = [(w, b) for w, bs in boxes.items() if not (w in EXCLUDE or base_word(w) in EXCLUDE) for b in bs]
    # 人工改判成"挂着的"(HUNG_LABELS)压过 VLM: 垂到地面的灯管 VLM 会判成落地灯
    _HUNG = {x.strip().replace("_", " ") for x in os.environ.get("HUNG_LABELS", "").split(",") if x.strip()}
    _ext, _no = [], []
    for w, b in _all:
        if _look(_REST, w) != "floor" or w in _HUNG or base_word(w) in _HUNG: continue
        bot = _bot(b); top = bot + b["尺寸"][2]; gap = bot - best_h
        if gap <= EXT_MIN: continue
        c0 = np.array(b["中心"], float); on = None
        for w2, b2 in _all:
            if b2 is b: continue
            t2 = _bot(b2) + b2["尺寸"][2]                     # 顶不会被补地面改掉, 顺序无关
            if not (bot - EXT_SUP_TOL <= t2 <= bot + 0.05): continue
            c2, a1, a2, h1, h2 = _rect(b2); q = c0 - c2
            if abs(q @ a1) <= h1 and abs(q @ a2) <= h2:
                on = f"{w2}#{b2['obj_id']}"; break
        if on:
            _no.append(f"{w}#{b['obj_id']} 搁在 {on} 上"); continue
        real = _look(_REAL, w); newh = top - best_h
        if real is None or newh > EXT_MAX_RATIO*float(real):
            _no.append(f"{w}#{b['obj_id']} 补完会有 {newh:.2f} m" +
                       (f", 超过常识 {float(real):.2f} m 的 {EXT_MAX_RATIO:g} 倍" if real else ", 没有常识高度可比"))
            continue
        b["_量到尺寸"] = list(b["尺寸"]); b["_量到底高"] = bot     # 摆放时判断要不要改成拉腿
        b["中心"] = (c0 - UP*gap/2).tolist(); b["尺寸"] = [b["尺寸"][0], b["尺寸"][1], newh]; b["底高"] = best_h
        _ext.append((w, b["obj_id"], gap, top - bot, newh))
    print(f"[补到地面] 立在地上的类里 {len(_ext)} 个底没量到(腿被挡住), 顶不动、底补到地面:")
    for w, i, g_, h0, h1 in sorted(_ext, key=lambda x: -x[2]):
        print(f"    {w}#{i}: 底离地 {g_:.2f} m, 高 {h0:.2f} -> {h1:.2f} m")
    if _no:
        print(f"[没补] {len(_no)} 个: " + "; ".join(_no))

# ---- 桌子/柜子这类: 去重 + 按"相机看到的空地"缩 (2026-09-22) ----
# ★ 这一步所有别的物体都等比缩, 原因见 o_assemble 的注释: 点云量出来的长宽不可信, 逐轴拉伸
#   吃过亏(吧凳量成 0.94x0.31 的板子被压扁)。可等比有反向的坑: 生成的 mesh 比例不对时照样
#   原样放大 —— office table_8 量到的是 1.25x0.58 的窄长站立桌, 生成的是方桌, 按高度等比
#   放成 1.46x1.46, 桌沿往人站的地方多伸出 0.44 m, 把站在桌前的人整个吞进桌面里。
#   所以只对"长宽比本来就千差万别、拉一拉看不出来"的方盒子家具(FIT_WORDS), 而且只做两件
#   有证据的事:
#     去重(DEDUP_BOX): 同一个词两个盒子, 水平重叠超过小的那个的 DUP_OVERLAP、顶面高差在
#       DUP_DH 以内 = 同一件被量了两遍(office table_7/8: 重叠 72%, 顶面 1.11/1.15 m), 留点数多的。
#     缩(FREE_TRIM): mesh 超出量到的盒子边的那一片, 投到每一帧相机里比深度 —— 相机明确
#       "看穿过去"(那里更远处有东西)才说明那片是空地, 多数点被看穿才往回缩到盒子边;
#       被挡住、出了画面、没看到的一律不缩。量到的盒子被截短(遮挡/出画)的时候, 截掉的那段
#       本来就没被看到, 所以不会被误缩 —— 这正是逐轴拉伸当年栽的地方。
#   只缩不拉: 盒子比 mesh 大的方向不动(可能是 mask 抠多了)。
#   人也算证据(BODY_RUN, 给了才用): 深度图里那一点拍到的是人(body4d 的人物抠图), 说明人站在那,
#   家具就不可能在那 —— 算"看穿"。不这么做的话, 人站在桌前的那一片恰好全是"看到表面", 人越贴着
#   桌子站, 那张桌子越缩不回去(office table_7: 超出的那片 604 个点只有 84 个被看穿)。
FIT_WORDS = {x.strip().replace("_", " ") for x in os.environ.get(
    "FIT_WORDS", "desk,table,dining table,side table,coffee table,counter,cabinet,file cabinet,"
                 "shelf,shelf unit,bookcase,credenza").split(",") if x.strip()}
_fit_word = lambda w: w in FIT_WORDS or base_word(w) in FIT_WORDS
DEDUP_BOX = os.environ.get("DEDUP_BOX", "0") == "1"
FREE_TRIM = os.environ.get("FREE_TRIM", "0") == "1"
DUP_OVERLAP = float(os.environ.get("DUP_OVERLAP", "0.5"))
DUP_DH = float(os.environ.get("DUP_DH", "0.12"))
TRIM_OVER = 0.05      # mesh 超出盒子边这么多以上才看
TRIM_EMPTY = 0.6      # 那一片里被看穿的点占有观测的点的比例
TRIM_MIN_OBS = 30     # 有观测的点至少这么多才下结论
TRIM_KEEP = 0.02      # 缩到盒子边外 2cm
TRIM_MIN_W = 0.35     # 缩完不能窄过这个
def _box_rc(b):
    c = np.array(b["中心"], float); a = np.array(b["长轴"], float); a = a - UP*(a @ UP)
    a /= max(np.linalg.norm(a), 1e-9)
    return c, a, np.cross(UP, a), b["尺寸"][0]/2, b["尺寸"][1]/2
def _box_top(b):
    return (float(b["底高"]) if "底高" in b else float(np.array(b["中心"]) @ UP - b["尺寸"][2]/2)) + b["尺寸"][2]
def _in_share(b1, b2):
    """b1 的水平矩形有多大比例落在 b2 里 (24x24 网格采样)。"""
    c1, a1, s1, h1, k1 = _box_rc(b1); c2, a2, s2, h2, k2 = _box_rc(b2)
    g = (np.arange(24) + 0.5)/24*2 - 1
    U_, V_ = np.meshgrid(g*h1, g*k1); q = c1 + U_.ravel()[:, None]*a1 + V_.ravel()[:, None]*s1 - c2
    return float(((np.abs(q @ a2) <= h2) & (np.abs(q @ s2) <= k2)).mean())
_DUP = {}
if DEDUP_BOX:
    for w, bs in boxes.items():
        if not _fit_word(w) or w in EXCLUDE or base_word(w) in EXCLUDE: continue
        order = sorted(bs, key=lambda b: -(b.get("点数") or 0))
        for i, bi in enumerate(order):
            if (w, bi["obj_id"]) in _DUP: continue
            for bj in order[i+1:]:
                if (w, bj["obj_id"]) in _DUP or abs(_box_top(bi) - _box_top(bj)) > DUP_DH: continue
                small, big = (bj, bi) if bj["尺寸"][0]*bj["尺寸"][1] <= bi["尺寸"][0]*bi["尺寸"][1] else (bi, bj)
                ov = _in_share(small, big)
                if ov >= DUP_OVERLAP:
                    _DUP[(w, bj["obj_id"])] = (f"{w}#{bi['obj_id']}", ov)
    if _DUP:
        print(f"[去重] {len(_DUP)} 个盒子和同类另一个是同一件(水平重叠 >= {DUP_OVERLAP:.0%}, 顶面高差 <= {DUP_DH:.2f} m), 留点数多的: "
              + "; ".join(f"{w}#{i} 并进 {k} (重叠 {ov:.0%})" for (w, i), (k, ov) in _DUP.items()))
_FS = None
def _people_masks():
    """{相机帧名: 人物抠图的路径}。body4d 和场景用的是同一套帧: 相机帧名 -> 原视频帧号(02_da3/frame_map.json)
    -> body4d 的帧序号(00_frames/frames.json 的 kept_indices) -> <mesh 轮>/masks/<序号>.png (值 = 人的 id)。"""
    br = os.environ.get("BODY_RUN", "").strip()
    if not br: return {}
    run = next((f"{br}/{d}" for d in ("04_mesh_occ1", "04_mesh_occ0") if os.path.isdir(f"{br}/{d}/masks")), None)
    da3 = f"{ROOT}/02_da3" if os.path.isdir(f"{ROOT}/02_da3") else f"{ROOT}/02_DA3点云"
    if not run or not os.path.exists(f"{da3}/frame_map.json"): return {}
    fmap = json.load(open(f"{da3}/frame_map.json"))["frames"]
    kept = {v: i for i, v in enumerate(json.load(open(f"{br}/00_frames/frames.json"))["kept_indices"])}
    return {k: f"{run}/masks/{kept[v]:08d}.png" for k, v in fmap.items()
            if v in kept and os.path.exists(f"{run}/masks/{kept[v]:08d}.png")}
def _fs_load():
    """每一帧: (w2c, 内参, 深度图, 人在哪些像素上 或 None)。内参和深度图是同一个分辨率(DA3 处理分辨率)。"""
    global _FS
    if _FS is None:
        da3 = f"{ROOT}/02_da3" if os.path.isdir(f"{ROOT}/02_da3") else f"{ROOT}/02_DA3点云"
        cams = json.load(open(f"{da3}/cameras.json"))["cameras"]; _FS = []; pm = _people_masks()
        for k, cm in sorted(cams.items()):
            dp = f"{da3}/depth/{k.rsplit('.', 1)[0]}_depth.npy"
            if os.path.exists(dp):
                D = np.load(dp); M = None
                if k in pm:
                    import cv2
                    from PIL import Image
                    M = cv2.resize((np.array(Image.open(pm[k])) > 0).astype(np.uint8), (D.shape[1], D.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                _FS.append((np.array(cm["extrinsic_w2c_4x4"], float), np.array(cm["intrinsic_3x3"], float), D, M))
        if pm: print(f"[按看到的空地缩] {sum(f[3] is not None for f in _FS)}/{len(_FS)} 帧有人物抠图: 拍到人的地方算空地")
    return _FS
def _seen(P):
    """每个点: 被相机看穿(视线上那一点是空的, 更远处才有东西)的帧数, 和看到表面就在这里的帧数。"""
    em = np.zeros(len(P), int); su = np.zeros(len(P), int)
    for w2c, K, D, M in _fs_load():
        X = P @ w2c[:3, :3].T + w2c[:3, 3]; z = X[:, 2]; ok = z > 0.3; zz = np.maximum(z, 1e-6)
        u = np.round(K[0, 0]*X[:, 0]/zz + K[0, 2]).astype(int); v = np.round(K[1, 1]*X[:, 1]/zz + K[1, 2]).astype(int)
        ok &= (u >= 0) & (u < D.shape[1]) & (v >= 0) & (v < D.shape[0])
        d = np.zeros(len(P)); d[ok] = D[v[ok], u[ok]]; ok &= np.isfinite(d) & (d > 0)
        tau = 0.08 + 0.03*z
        here = ok & (np.abs(z - d) <= tau)
        onp = np.zeros(len(P), bool)
        if M is not None: onp[ok] = M[v[ok], u[ok]]
        em += (ok & (z < d - tau)) | (here & onp)                  # 看到的表面是人 = 人站在这, 家具不在这
        su += here & ~onp
    return em, su
_trimmed = []
def free_trim(m, b, w):
    """mesh 超出量到的盒子边、而相机明确看到那里是空的 -> 那一侧缩回盒子边。只缩不拉。"""
    c, a, sd, ha, hs = _box_rc(b)
    hgt = m.vertices @ UP; lo, hi = float(hgt.min()), float(hgt.max())
    P, _ = trimesh.sample.sample_surface(m, 6000, seed=0)
    P = P[(P @ UP) > lo + 0.5*(hi - lo)]                # 上半截(桌面、柜顶): 最常被看到的部分
    V = m.vertices.copy(); notes = []
    for nm, e, h in (("长", a, ha), ("宽", sd, hs)):
        t = (V - c) @ e; t0, t1 = float(t.min()), float(t.max()); n0, n1 = t0, t1; tp = (P - c) @ e
        for sg in (1, -1):
            if (t1 if sg > 0 else -t0) <= h + TRIM_OVER: continue
            sel = sg*tp > h + TRIM_OVER
            if sel.sum() < TRIM_MIN_OBS: continue
            em, su = _seen(P[sel]); emp = (em >= 2) & (em >= 2*su); srf = (su >= 2) & (su > em)
            nobs = int(emp.sum() + srf.sum())
            tag = f"{nm}{'+' if sg > 0 else '-'}侧 超出 {(t1 if sg > 0 else -t0) - h:.2f} m, 看穿 {int(emp.sum())}/{nobs}"
            if nobs < TRIM_MIN_OBS or emp.sum() < TRIM_EMPTY*nobs:
                notes.append(tag + " -> 证据不够, 不缩"); continue
            if sg > 0: n1 = h + TRIM_KEEP
            else: n0 = -(h + TRIM_KEEP)
            notes.append(tag + " -> 缩")
        if (n0, n1) != (t0, t1):
            if n1 - n0 < TRIM_MIN_W:
                notes.append(f"{nm}: 缩完只剩 {n1 - n0:.2f} m < {TRIM_MIN_W} m, 不缩"); continue
            V = V + np.outer(n0 + (t - t0)*(n1 - n0)/(t1 - t0) - t, e)
            P = P + np.outer(n0 + (tp - t0)*(n1 - n0)/(t1 - t0) - tp, e)
    if notes:
        ext3 = lambda X: [round(float(np.ptp(X @ ax)), 2) for ax in (a, UP, sd)]   # 沿它自己的 长/高/宽
        e0 = ext3(m.vertices); m.vertices = V
        _trimmed.append((w, b["obj_id"], e0, ext3(m.vertices), notes))
    return m

scene=trimesh.Scene(); n=0; skipped=[]
_snapped = []                    # 朝向被吸到房间网格上的: (词, 实例, 转了几度)
# ---- 每类一个统一缩放 ----
# 基准 = 生成这个类 mesh 时用的那个实例。它是 VLM 校验过、抠得最完整的那一个,
# 拿它定尺寸和 mesh 本身自洽。查不到(被去重合并了)就退回"剔掉离群后的中位数"。
#
# 试过但不对的两条:
#   点数最多的  —— table 点数最多的 5 号其实是误检(抠了一辆红车+卡车, 盒子 2.7x2.4m),
#                  当基准把整类桌子撑到 1.18 米(真实 0.7)。
#   取分布上半段 —— 理由是"遮挡只会让盒子变小", 但漏了反向的错: mask 抠多了盒子会变大
#                  (table 有 4.8m 和 8.2m 的盒子), 两头都有噪声, 取大端等于专挑抠多的。
PICKS=json.load(open(f"{ROOT}/04_picks/picks.json"))
# 这些类真实世界里本来就有大有小, 不统一 —— 树 0.66~4.84m 全压成 2.78m 会把大树压矮一半
PER_INSTANCE=set(os.environ.get("PER_INSTANCE","tree").split(","))
def one_scale(ext, me):
    ext=np.array(ext)
    return (np.sort(ext)[::-1][0]/max(np.sort(me)[::-1][0],1e-6)) if ext[2]<0.2 \
           else ext[2]/max(me[1],1e-6)

def ruler(ext):
    """one_scale 实际拿哪个量去定尺寸。类内守卫必须比**这个**量, 不能比 ext.max()。
    ★ 为什么: 高度和最长边是两个量, 一个物体可以最长边只差 1.5 倍而高度差 5 倍
      (mask 横向漂移就是这个形状)。守卫要判的是"摆出来会不会大小离谱",
      那就必须比 one_scale 真正拿去算的那个量, 否则判的和发生的不是一回事。
    ★ 注意: 守卫和中位数两处都要求 ext[2]>=0.2, 所以在守卫这条路上只会走 ext[2] 分支;
      max 那半是给别的调用者留的(和 one_scale 保持同构)。"""
    ext=np.asarray(ext, float)
    return float(np.max(ext)) if ext[2] < 0.2 else float(ext[2])

# --- SAM3D 生成的 mesh 没有统一朝向 ---
# 下面 R 的写法默认"mesh 的长轴在自己的 +Z 上", 但 SAM3D 并不保证这件事:
# carpark 里 19 辆 car 全在 Z(对), 5 个 van、5 个 barrier、14 个 container 里 10 个却在 X(错 90°)。
# 结果就是盒子明明摆对了, 物体自己却横过来 —— 集装箱围墙整片转 90°。实测 51 个物体里 17 个中招。
# 修法: 摆之前先量 mesh 自己在水平面(XZ)上的长轴, 绕自己的 Y 转到 +Z, 再交给 R。
# 只对确实细长的做(长宽比 > YAW_MIN_AR); 接近方的物体长轴是噪声, 转了反而乱。
# 这解决的是 90°, 解决不了 180° 正反 —— 那个还是没解。
YAW_MIN_AR = float(os.environ.get("YAW_MIN_AR","1.15"))

# --- 朝向吸附到房间网格 ---
# ★ 先例: LM 链早就有这一步 —— pipeline/layout_boxfix.py 的 boxfix, 算法是 SpatialGen
#   prepare_spatiallm_scene_5090.py 里的 fix_boxes(snap_deg=30): "水平转角离 0/90 的倍数
#   <= snap_deg 就吸附"。物体链一直没有, 所以物体链摆出来才是歪的。
#   两处差别: LM 链敢直接吸 0/90, 因为它在 zup(墙对齐坐标轴) 之后的系里跑;
#   这里跑在**未旋转的 DA3 系**(load_up 读 02_da3), 坐标轴和墙没关系, 只能从墙线里数。
#   改其中一处的时候想想另一处。
# 盒子的长轴来自点云拟合。桌面这种"只拍到一个薄片"的东西, 长轴基本是噪声 ——
# 实测 office 2026-09-20: 39 个物体离房间网格中位偏 15°, 5 个偏 30° 以上,
# 摆出来整片桌子朝向乱七八糟, 而物理那步只平移不旋转, 治不了。
# 房间壳的墙线给了建筑自己的方向, 贴着建筑摆的东西(桌/柜/窗/显示器)可以吸附过去。
# ★ 默认关。椅子被人转出来是**真实的**, 无差别吸附会把真信息抹掉;
#   室外更不能开(车爱朝哪朝哪)。所以按词开, 而且只吸附偏得不多的 ——
#   偏 40° 那种要么是另一套朝向, 要么盒子彻底错了, 硬掰过去是假装修好了。
YAW_SNAP     = {w.strip() for w in os.environ.get("YAW_SNAP","").split(",") if w.strip()}
YAW_SNAP_MAX = float(os.environ.get("YAW_SNAP_MAX", "25"))   # 偏超过这个角度就不动它
_GRID = None                      # [主方向, 主方向+90°] 的水平单位向量; None = 没网格


def _room_grid(shell_path, up):
    """从房间壳的墙线里数出建筑的主方向。按墙长加权, ±5° 聚合取最高的那一档。
    取不到就返回 None —— 没有网格就不吸附, 别瞎猜一个方向出来。"""
    jp = os.path.splitext(shell_path)[0] + "_墙线.json"
    if not (shell_path and os.path.exists(jp)):
        return None
    try:
        segs = json.load(open(jp)).get("墙线") or []
    except Exception:
        return None
    # 墙线是平面坐标, up 轴被抽掉了。把它铺回三维: 找 up 最小的那一维当"被抽掉的轴"。
    ax = int(np.argmax(np.abs(up)))                  # up 在哪个轴上
    o1, o2 = [i for i in range(3) if i != ax]        # 平面的两个轴, 顺序和墙线一致
    hist = np.zeros(180)
    for sg in segs:
        try:
            a_, b_ = np.array(sg["a"], float), np.array(sg["b"], float)
        except Exception:
            continue
        v = b_ - a_; L = float(np.linalg.norm(v))
        if L < 1e-6: continue
        hist[int(round(np.degrees(np.arctan2(v[1], v[0])) % 180)) % 180] += L
    if hist.sum() <= 0: return None
    sm = np.array([hist[(np.arange(i-5, i+6)) % 180].sum() for i in range(180)])
    _c = int(np.argmax(sm))
    # ★ 用这一档里的**按长度加权圆均值**, 不要用档的中心角。
    #   直方图是 1° 一格、±5° 聚合, 取 argmax 等于拿格子编号当答案 —— 墙长在档内不对称时
    #   就会系统性偏掉。实测 office 2026-09-20: 档中心给 89°, 而墙的真实主方向是 91.3°,
    #   物体整体比墙歪 2.3°, 看上去"互相平行了但和房间不正交"。
    #   圆均值要按 2θ 算(方向是 mod 180° 的, 0° 和 179° 是邻居, 直接平均会跑到 90°)。
    _idx = (np.arange(_c - 5, _c + 6)) % 180
    _w = hist[_idx]
    _a2 = np.radians(2.0 * _idx)
    th = float(np.arctan2((_w * np.sin(_a2)).sum(), (_w * np.cos(_a2)).sum())) / 2.0 % np.pi
    g = []
    for t in (th, th + np.pi/2):
        v = np.zeros(3); v[o1] = np.cos(t); v[o2] = np.sin(t)
        v = v - up*(v@up); g.append(v/np.linalg.norm(v))
    print(f"[网格] 墙的主方向 {np.degrees(th):.0f}° (±5° 内总长 {sm.max():.1f} m), "
          f"吸附目标 2 个方向, 阈值 {YAW_SNAP_MAX}°", flush=True)
    return g


def snap_yaw(fwd, word):
    """把水平朝向吸到最近的网格方向。返回 (新朝向, 转了多少度)。不吸就原样返回。"""
    if _GRID is None or not YAW_SNAP or not (word in YAW_SNAP or "*" in YAW_SNAP):
        return fwd, 0.0
    best, bd = fwd, 1e9
    for g in _GRID:
        for cand in (g, -g):                       # 180° 这里不解, 取近的那头
            d = np.degrees(np.arccos(np.clip(fwd @ cand, -1, 1)))
            if d < bd: bd, best = d, cand
    return (best, bd) if bd <= YAW_SNAP_MAX else (fwd, 0.0)


# ★ 网格必须在**摆放循环之前**算出来 —— 第一版放在下面"用房间壳代替大方片"那句旁边,
#   那已经是循环跑完之后了, _GRID 永远还是 None, 吸附一个都不生效而且不报错。
if YAW_SNAP:
    _GRID = _room_grid(ROOM_SHELL, UP)
YAW_FIX    = os.environ.get("YAW_FIX","1")=="1"
_yaw_n=[0,0]
def yaw_to_z(m):
    if not YAW_FIX: return np.eye(4)
    V=m.vertices[:,[0,2]]                      # 只看水平面
    e=V.max(0)-V.min(0)
    if max(e)/max(min(e),1e-6) < YAW_MIN_AR:
        _yaw_n[1]+=1; return np.eye(4)         # 太方, 不动
    Q=V-V.mean(0); ev,W=np.linalg.eigh(Q.T@Q)
    a=W[:,np.argmax(ev)]                       # (x,z) 上的长轴
    th=np.arctan2(a[0],a[1])                   # 要把它转到 +Z
    ct,st=np.cos(-th),np.sin(-th)
    T=np.eye(4); T[0,0],T[0,2],T[2,0],T[2,2]=ct,st,-st,ct
    _yaw_n[0]+= int(abs(np.degrees(th))>45)
    return T

# ---- 类内相对守卫的基准: 每个词自己的"最长边中位数" ----
# ★ 为什么还要再来一道: MIN_H/MAX_D 是**绝对**阈值, 它只知道"这个盒子本身离谱吗",
#   不知道"这个盒子和同类的另外 12 个自相矛盾吗"。beach 实测 street lamp 13 个实例最长边
#   从 2.89 m 到 42.39 m 差 14 倍, MAX_D=15 只拦下 4 个, 12.63/11.43/11.19/10.70 米的路灯
#   和 0.87 米的路灯并肩摆进同一个场景, 一个都没拦。
#   而绝对阈值往两边都不对: 校准前压缩的世界里它拦不住, 校准后放大的世界里它开始误杀
#   (restaurant 真实 17.5 m 会被 MAX_D 冤枉)。相对判据不受全局尺度影响, 正好补上这一段。
# ★ 必须在下面那个 CLASS_SCALE 循环**之外**单独扫一遍: 那个循环会因为"排除词"和"没有类 mesh"
#   提前 continue, 而这些词的实例仍然能靠 pick_mesh 的第三条退路被摆出来, 拿不到中位数。
REL_N     = float(os.environ.get("REL_N", "2.5"))   # 和类内中位数差这么多倍就当量错了
REL_MIN_N = int(os.environ.get("REL_MIN_N", "4"))   # 类内实例数不到这个数, 中位数不可信, 不判
# ★ 中位数只用**过了绝对守卫**的那些盒子算。
#   拿全量算是错的, 而且错得很隐蔽: street lamp 13 个实例里有 42.39/31.32/17.08/15.67 四个
#   垃圾盒子, 它们把中位数抬到 11.19 m —— 于是"和中位差 2.5 倍"判下来, 被拦掉的反而是
#   2.89/3.35/3.98/4.01 这四根**唯一符合常识**的路灯, 12.63 那根倒留下了。方向整个是反的。
#   剔掉垃圾之后中位变成 5.05 m, 拦的对象立刻翻转 —— 一行之差, 结论相反。
def _abs_ok(b):
    e=np.array(b["尺寸"]); return not (e[2]<MIN_H or e.max()>MAX_D)
# ★ 只收 ext[2]>=0.2 的盒子进中位数。ruler 会按这个界在"最长边"和"高度"之间切换,
#   而绝对守卫的下限是 MIN_H=0.15 —— 高度落在 [0.15,0.20) 的盒子会过绝对守卫却贡献最长边,
#   和贡献高度的盒子混进同一个统计量, 两个量纲搅在一起。扁的那批也不判(它们本来就该被
#   当成薄片处理)。
CLASS_EXT_MED={}
_by_base={}
for _w,_bs in boxes.items():
    _good=[ruler(b["尺寸"]) for b in _bs if _abs_ok(b) and b["尺寸"][2] >= 0.2]
    if len(_good) >= REL_MIN_N:
        CLASS_EXT_MED[_w]=float(np.median(_good))
    _by_base.setdefault(base_word(_w), []).extend(_good)
# ★ 派生子类退回母词。人在闸口把 umbrella 分成三组各 4 个之后, 每组自己算中位数 ——
#   大伞自成一组就不再"离群", 同样那两把在分组前会被拦、分组后一个都不拦。
#   再分细一点(某组只有 3 个)那一整组就彻底没有相对守卫了, 而且不吭声。
#   守卫的强度不该由人怎么分组来决定, 所以查不到原词时用母词的中位数兜底。
def _med_of(w):
    m = CLASS_EXT_MED.get(w)
    if m is not None: return m
    g = _by_base.get(base_word(w)) or []
    return float(np.median(g)) if len(g) >= REL_MIN_N else None

CLASS_SCALE={}; dropped=0; DEGENERATE=set()
for w,bs in boxes.items():
    if w in EXCLUDE or base_word(w) in EXCLUDE or not bs: continue
    mk=MESH.get(w)
    if mk is None: continue
    me=get_mesh(mk).extents
    S=np.array([one_scale(b["尺寸"], me) for b in bs])
    keep=np.abs(np.log(S/np.median(S)))<np.log(2.0)        # 和中位差 2 倍以上 = 离群, 不参与
    dropped += int((~keep).sum())
    oid=PICKS.get(w,{}).get("obj_id")
    hit=[i for i,b in enumerate(bs) if b["obj_id"]==oid]
    # ★ 生成实例是首选基准, 但**它自己也可能是那个离群的**。一旦是, 拿它定基准等于
    #   把整类一起带歪 —— 一个词只有一个 CLASS_SCALE, 错一次错全类, 这是这里最贵的一种错。
    #   所以先问一句它在不在 keep 里, 不在就退回中位数。
    if hit and keep[hit[0]]:
        CLASS_SCALE[w]=float(S[hit[0]]); src="生成实例"
    elif hit:
        CLASS_SCALE[w]=float(np.median(S[keep])); src="生成实例离群→改用中位"
    else:
        CLASS_SCALE[w]=float(np.median(S[keep])); src="离群后中位"
    # 整类的兜底: 单个盒子已经有 ext.max()>15 的守卫, 但那守的是"盒子", 守不住"mesh 烂".
    # carpark 的 barrier: SAM3D 把护栏生成成了一张 1.00x0.09x0.01 的薄片, 高度只有 9cm。
    # 对高度时要除以这 0.09, 尺度被放大 17.5 倍 —— 9 个护栏全变成 17.6 米的巨大平板,
    # 横在整个停车场上把别的都挡住了。盒子本身是正常的, 所以老守卫一个都没拦下。
    if CLASS_SCALE[w]*me.max() > 15:
        print(f"  [{w:16s}] 整类丢弃: 摆出来会是 {CLASS_SCALE[w]*me.max():.1f} m "
              f"(mesh {np.round(me,3).tolist()} 太扁, 对高度时除出了大尺度)")
        CLASS_SCALE.pop(w); DEGENERATE.add(w); continue
    print(f"  [{w:16s}] {len(bs):3d} 个 -> 统一 {CLASS_SCALE[w]*me.max():.2f} m  ({src})")
print(f"  定基准时忽略了 {dropped} 个离群实例")

solo_used=0
for w,bs in boxes.items():
    if w in EXCLUDE or base_word(w) in EXCLUDE or w in DEGENERATE: continue
    for b in bs:
        # 人在闸口否掉的: 类名被否 = 整个词不要, 实例名被否 = 只这一个不要
        if _gen_name(w) in SKIP_NAMES or _gen_name(w, b["obj_id"]) in SKIP_NAMES:
            skipped.append((w,b["obj_id"],[],"人工否掉")); continue
        if (w, b["obj_id"]) in _DUP:
            skipped.append((w,b["obj_id"],[],f"重复: 和 {_DUP[(w, b['obj_id'])][0]} 是同一件")); continue
        # ★ VLM 否决只是提醒。没被人放行 -> 不摆; 放行了 -> 只许用它自己的 mesh
        #   (或人判了共用的类 mesh), 不许按名字借同词别人的 —— 那就是 8 米长的 car_0。
        _vetoed = b["obj_id"] in VLM_REJ.get(w, ())
        if _vetoed and _gen_name(w, b["obj_id"]) not in VLM_REL:
            skipped.append((w,b["obj_id"],[],"VLM否决(人没放行)")); continue
        mk=pick_mesh(w, b["obj_id"])
        if mk is None:
            # ★ 以前这里一声不吭地 continue, 跳过名单里没有它, 查"为什么没进场景"时查不到。
            skipped.append((w,b["obj_id"],[],"没有 mesh")); continue
        if _vetoed and mk != _gen_name(w, b["obj_id"]) and not (
                (w in KEEP_W or base_word(w) in KEEP_W) and mk == MESH.get(w)):
            skipped.append((w,b["obj_id"],[],"VLM否决已放行, 但还没有自己的 mesh(不借别人的)")); continue
        # ★ 还要拿**实际用上的那个 mesh** 再比一次。pick_mesh 的第三条退路会借同词
        #   另一个实例的 mesh —— 人否掉了 be_r2_w_0, 实例 1 没有自己的 mesh, 于是借了
        #   be_r2_w_0 摆出来: 名义上的名字不在名单里, 被否掉的东西照样出现在场景里。
        if mk in SKIP_NAMES:
            skipped.append((w,b["obj_id"],[],"人工否掉(借的是被否的mesh)")); continue
        ext=np.array(b["尺寸"])
        if ext[2]<MIN_H or ext.max()>MAX_D:
            skipped.append((w,b["obj_id"],ext.round(2).tolist(),"绝对")); continue
        # ---- 类内相对守卫 ----
        # ★ 只守**盒子决定尺寸**的那条路: 专属 mesh 和逐实例词走下面的 one_scale(ext, me),
        #   盒子量错多少, 摆出来就大多少倍 —— 12 米的垃圾桶就是这么来的。
        #   走类统一缩放(CLASS_SCALE)的那条路不守: 那些实例摆出来全是同一个尺寸,
        #   盒子离谱只影响它自己的位置, 拦掉等于白删一个本来好好的物体。
        #   第一版守反了(守在类统一那条), 净效果是只误杀不多拦, 实测 57 个变 51 个。
        _self_scaled = (mk != MESH.get(w)) or (base_word(w) in PER_INSTANCE)
        # ★ 类内守卫对**所有**实例生效, 不只是自缩放的那些。
        #   共用词的实例虽然摆出来都是同一个尺寸(尺寸来自 CLASS_SCALE), 但它的**位置和朝向**
        #   仍然是从这个盒子来的 —— 一个在定尺寸的那个量上偏了 2.5 倍的盒子, 中心多半也是错的。
        #   实测(beach, 中位数剔过垃圾之后): 全类生效只拦下 street lamp 的 10.29m 和 10.44m
        #   两根, 别的类一个都没误杀 —— 正是渲染回图里那根巨大的路灯。
        #   只限自缩放分支的话, 这两根照样摆进场景。
        _med=_med_of(w)
        if _med and ext[2] >= 0.2:
            _dev = np.log(max(ruler(ext),1e-6)/_med)
            # ★ 方向是不对称的, 这一点很关键:
            #   **偏大**  = mask 跨帧漂移 / 深度拖尾, 盒子被横着拉开 —— 这种盒子的中心也是错的,
            #             不管尺寸从哪来, 位置都不能要。beach 那两根 10.4 米的路灯就是:
            #             水平占地 11 米、厚度 0.65 米、点数只有正常杆子的 1/10。
            #   **偏小**  = 遮挡截短。这是这条链上最常见的噪声, 而且**只影响尺寸不影响中心**。
            #             对走类统一缩放的实例, 尺寸本来就来自 CLASS_SCALE 不来自这个盒子,
            #             按偏小拦掉纯属白删 —— 实测会误杀 timesquare 的一根路灯、一辆卡车,
            #             以及 WD 支线上 36 个盆栽里的 5 个(它们高度是连续分布, 不是离群)。
            #   所以: 自缩放的两头都判(尺寸真由盒子定); 类统一缩放的只判偏大。
            if _dev > np.log(REL_N) or (_self_scaled and -_dev > np.log(REL_N)):
                _why = "类内偏大(盒子横向拉开, 中心也不可信)" if _dev > 0 else "类内偏小"
                skipped.append((w,b["obj_id"],
                                f"{ruler(ext):.2f}m vs 类内中位 {_med:.2f}m", _why)); continue
        m=get_mesh(mk); me=m.extents
        solo_used += (mk != MESH.get(w))
        me=m.extents
        if _self_scaled:                                 # 专属 mesh(8 辆车) 和树: 各按各的
            s = one_scale(ext, me)
        else:
            s = CLASS_SCALE.get(w)
            if s is None: continue
        # ★ 补过地面的: 生成 mesh 的时候腿也被挡着, office 一半的桌子被生成成矮桌(高只有长的
        #   0.24, 腿很短)。按补完的高度等比缩, 桌面会被撑到 3~4 m 长, 而量到的桌面才 1 m。
        #   这种就按量到的桌面大小缩(水平最长边对上盒子的长边 —— 从上往下拍, 桌面是量得最准的),
        #   摆好以后把下面的腿往下拉到地面, 顶上 LEG_KEEP 不动。
        #   (第一版按"量到的那截高度"缩: 矮桌碰巧也对, 但完整的高柜只露出顶上一截时会被缩成
        #    0.37 m 宽, 比它自己的盒子 0.7 m 还窄。按桌面大小缩两种都对。)
        #   mesh 本来就完整的(椅子)等比缩完水平边和量到的对得上, 照旧等比。
        _legs = None
        if "_量到尺寸" in b:
            _foot = float(max(me[0], me[2]) * s)
            if _foot > FOOT_RATIO * float(max(ext[0], ext[1])):
                _s0 = float(max(ext[0], ext[1])) / max(float(max(me[0], me[2])), 1e-6)
                _legs = (_foot, float(max(me[0], me[2]) * _s0)); s = _s0
        # ★ 守门要再查一次**摆出来的尺寸**, 不能只查盒子。
        #   one_scale 是拿盒子**高度**去缩 mesh 的, 所以最终多大 = mesh 自己的比例 x 高度 ——
        #   mesh 比例和盒子比例对不上的时候, 一个过了 MAX_D 的盒子照样能摆出个超标的东西。
        #   实测 office 2026-09-20 的 ventilation duct: 盒子 8.97x1.65x1.52(最长边 8.97 < 15,
        #   放行), 而 mesh 是根细管(归一化 0.105 x 0.087 x 1.0), 按高度 1.52 缩 = 17.5 倍,
        #   长轴被拉成 17.5 m, 飞到离场景中心 27 m 的地方, 一声不吭。
        _placed = float(np.max(np.asarray(me, float) * s))
        if _placed > MAX_D:
            skipped.append((w, b["obj_id"],
                            f"摆出来 {_placed:.1f}m > MAX_D={MAX_D} (盒子最长边才 {float(np.max(ext)):.1f}m, "
                            f"mesh 比例和盒子对不上)", "摆出来超上限")); continue
        m.apply_scale(s)
        m.apply_transform(yaw_to_z(m))                   # ★ 先把 mesh 自己的长轴摆到 +Z
        fwd=np.array(b["长轴"]); fwd=fwd-UP*(fwd@UP); fwd/=np.linalg.norm(fwd)
        fwd, _sn = snap_yaw(fwd, w)
        if _sn > 0: _snapped.append((w, b["obj_id"], round(_sn)))
        rgt=np.cross(UP,fwd)
        R=np.eye(4); R[:3,0],R[:3,1],R[:3,2]=rgt,UP,fwd  # mesh X->右 Y->上 Z->前
        m.apply_transform(R)
        c=np.array(b["中心"])
        bottom=c-UP*ext[2]/2
        # ★ 强制落地: 盒子高度在俯视场景里量不准, 用它算出来的"底"有的悬空有的埋地。
        # 物体本来就该站在地上, 所以直接把底放到地面高度, 不信盒子自己算的。
        # 本该悬空的(雨篷/树冠/灯头)靠 FLOAT_WORDS 放过, 不参与强制落地。
        if SNAP_GROUND and w not in FLOAT_WORDS:
            bottom = bottom - UP*((bottom@UP) - best_h)
        # 精确落地: 把 mesh 最低点对到盒底
        h_now=(m.vertices@UP); m.apply_translation(UP*((bottom@UP)-h_now.min()))
        if _legs:                                        # 拉腿: 底已在地面, 把顶拉到量到的桌面高度
            h=m.vertices@UP; lo=float(h.min()); H0=float(h.max()-lo); T=lo+float(ext[2])
            if T > lo+H0+1e-3:
                cut=lo+(1-LEG_KEEP)*H0                   # cut 以上(桌面)整体平移, 以下(腿)线性拉长, 两段在 cut 处接上
                nh=np.where(h>=cut, h+(T-lo-H0), lo+(h-lo)*(T-LEG_KEEP*H0-lo)/max(cut-lo,1e-9))
                m.vertices=m.vertices+np.outer(nh-h, UP)
                _legged.append((w, b["obj_id"], _legs[0], _legs[1], H0, T-lo))
        # 水平对中
        d=c-m.bounds.mean(0); d=d-UP*(d@UP); m.apply_translation(d)
        if FREE_TRIM and _fit_word(w):
            m = free_trim(m, b, w)
        scene.add_geometry(m, node_name=f"{w.replace(' ','_')}_{b['obj_id']}")
        n+=1

if _trimmed:
    print(f"[按看到的空地缩] 查了 {len(_trimmed)} 个超出量到的盒子的方盒子家具 (沿它自己的 长/高/宽, 米):")
    for w, i, e0, e1, notes in _trimmed:
        print(f"    {w}#{i}: {e0} -> {e1}; " + "; ".join(notes))
if _legged:
    print(f"[拉腿] {len(_legged)} 个补过地面的 mesh 是矮的, 按量到的桌面大小缩、腿拉到地面 (顶上 {LEG_KEEP:.0%} 不变形):")
    for w, i, f_, f0, h0, h1 in _legged:
        print(f"    {w}#{i}: 等比缩会有 {f_:.2f} m 长 -> 按量到的桌面 {f0:.2f} m; 高 {h0:.2f} -> {h1:.2f} m")

# --- 地面片: 覆盖所有物体的范围 (没有房间壳时才铺) ---
allp=np.array([b["中心"] for bs in boxes.values() for b in bs])
mn,mx=allp.min(0)-8, allp.max(0)+8
gx=np.linspace(mn[0],mx[0],2); gz=np.linspace(mn[2],mx[2],2)
# 平面上任取一点 = best_h*UP, 再沿两个与 UP 正交的方向铺开
_e1 = np.array([1,0,0.]); _e1 = _e1 - UP*(_e1@UP)
if np.linalg.norm(_e1) < 1e-6: _e1 = np.array([0,0,1.]) - UP*(np.array([0,0,1.])@UP)
_e1 /= np.linalg.norm(_e1); _e2 = np.cross(UP,_e1)
_ctr = allp.mean(0); _ctr = _ctr - UP*(_ctr@UP) + UP*best_h
_r = float(np.abs((allp-_ctr) @ np.c_[_e1,_e2]).max()) + 8
V=[_ctr + a*_r*_e1 + b*_r*_e2 for a in (-1,1) for b in (-1,1)]
V=np.array(V); F=np.array([[0,1,3],[0,3,2]])
ground=trimesh.Trimesh(V,F)
# 单面渲染时法向朝哪边决定从哪边看得见。不校正的话从上往下看地面是隐形的,
# 只有从地底下才看得见, 一眼像"没生成地面"。
# 注意这里还在 DA3 世界系, 上 = UP = [0,-1,0], 也就是 -y;
# 文件末尾的 FLIP 才把它翻成 glTF 约定的 +y。所以这里要让法向朝 -y。
if float(ground.face_normals[0] @ UP) < 0: ground.faces = ground.faces[:, ::-1]
ground.visual.face_colors=[130,130,135,255]
if not ROOM_SHELL:
    scene.add_geometry(ground,node_name="ground")
else:
    print(f"[地面] 用房间壳代替大方片: {os.path.basename(ROOM_SHELL)}")

# DA3 世界系 = 首帧相机系 (OpenCV, y 朝下), glTF 约定 y 朝上, 差 180°。
# 内部一直用 DA3 系(渲染回图要对上原相机), 只在导出时翻过来。
# 导出前把整个世界转正: 让真实的"天"对准 glTF 的 +y。
# 老代码只做了 FLIP(翻 y/z), 那等于默认"天"就是 -y —— 相机低着头时这个默认是错的,
# 于是地面和物体一起歪着导出去(beach 地面在竖直方向跨了 33 米, 本该接近 0)。
# 注意这一步只解决"歪", 解决不了"矮": 高度是在量盒子那步定的, 那边也已经改用真实的天了。
# ★ UP 本身就是"天": 整条链取高度用 P@UP, 值越大越高。老代码 FLIP=diag(1,-1,-1)
# 把 UP=[0,-1,0] 映射到 +y, 也印证了这一点。第一版这里写成 -UP, 等于把天当成地,
# 导出后地面跑到物体上方、路灯整根挂在地面下面(顶贴地、底 -4.7m)。
# 导出旋转从 world_frame 拿。渲染回图那边用 inverse_T, 两边同源, 不许各算各的 ——
# 2026-09-08 就是这里改了、渲染回图没跟着改, carpark 差 29°、beach 差 154°,
# 四张验收图全废还被当依据看了好几天。
from world_frame import deviation_deg
FLIP = export_T(UP)
print(f"[转正] 把真实的天转到 +y, 相对老写法差 {deviation_deg(UP):.1f}°")
scene.apply_transform(FLIP)

# --- 房间壳: FLIP 之后并进来 (它自己已经是最终 glTF 系) ---
nw=0; npl=0
if ROOM_SHELL:
    if not os.path.exists(ROOM_SHELL): sys.exit(f"[失败] 找不到房间壳 {ROOM_SHELL}")
    shl=trimesh.load(ROOM_SHELL, force="scene")
    # ★ 把壳整体挪到"物体站的那个地面高度"上。
    #   地面其实被独立算了两遍, 而且从来没人核对过:
    #     物体 -> 底部被强制放到 best_h (= world_up.json 的 ground_h, 见上面 SNAP_GROUND 那段)
    #     壳   -> shell 那步从 03_点云_z朝上.ply 自己建的, zup 在那份点云上**另外**拟了
    #             一次地板并压到 z=0, 带着那个高度进来
    #   两个数差多少, 物体就悬空/陷进地板多少。实测 office 2026-09-20 就是这个毛病:
    #   54 个物体彼此齐平(都绑在 best_h 上), 但整体和壳的地板不在一个平面。
    #   ★ 以物体为准, 不是以壳为准: 物体的位置、尺寸、落地、支撑面判定全都已经绑死在
    #     best_h 上, 动它等于动 54 个物体; 壳只是个外框, 平移它代价最小、影响面最窄。
    #     (这不解决"地面高度准不准" —— ground_h 本身就偏高, 实测相机离地只有 0.34m。
    #      那是 world_up.py 的地面拟合问题, 另算。这里只保证两边用**同一个**数。)
    #   这段在 FLIP 之后, UP 已经对准 +y, 所以"高度"就是 y。
    _dy, _fl = 0.0, None
    for node in shl.graph.nodes_geometry:
        if node == "floor":
            _T,_gn = shl.graph[node]; _g = shl.geometry[_gn].copy(); _g.apply_transform(_T)
            _fl = float(np.median(_g.vertices[:, 1])); break
    if _fl is None:
        print("[房间壳] ★ 壳里找不到 floor 节点, 没法对齐地面 —— 按原高度并入, "
              "物体可能悬空或陷进地板。去看 room_shell.py 的节点命名。")
    else:
        _dy = float(best_h) - _fl
        if abs(_dy) > 3.0:
            print(f"[房间壳] ★ 壳的地板 {_fl:+.2f} m 和物体的地面 {best_h:+.2f} m 差了 {_dy:+.2f} m。"
                  f"差这么多说明两次地面拟合里至少有一次彻底错了 —— 照样对齐, 但值得人看一眼。")
        elif abs(_dy) > 1e-6:
            print(f"[房间壳] 壳整体平移 {_dy:+.2f} m: 地板 {_fl:+.2f} -> {best_h:+.2f} m (物体站的高度)")
    for node in shl.graph.nodes_geometry:
        T,gn=shl.graph[node]; g=shl.geometry[gn].copy(); g.apply_transform(T)
        if _dy: g.apply_translation([0.0, _dy, 0.0])
        # 节点名要保住 floor / wall_N —— 下游 scene_phys 靠名字区分"地面/静态墙/可动物体"
        scene.add_geometry(g, node_name=node)
        nw += node.startswith("wall"); npl += node.startswith("pillar")
    print(f"[房间壳] 并入 floor + {nw} 面墙" + (f" + {npl} 根柱子(pillar_N, 物理当钉死的)" if npl else ""))

dst=f"{ROOT}/"+os.environ.get("SCENE_OUT","07_scene.glb"); scene.export(dst)
print(f"[组装] {n} 个物体 + {'房间壳(floor+%d墙)'%nw if ROOM_SHELL else '地面'} -> {dst}"
      f"  (其中 {solo_used} 个用了实例专属 mesh, 排除了 {sorted(EXCLUDE)})")
print(f"[朝向] 把 mesh 自身长轴对齐到盒子长轴: 其中 {_yaw_n[0]} 个原本歪了 45°以上, {_yaw_n[1]} 个太方没动")
if YAW_SNAP:
    if _snapped:
        _d = sorted(x[2] for x in _snapped)
        print(f"[朝向] 吸到房间网格 {len(_snapped)} 个, 转了 {_d[0]}~{_d[-1]}° (中位 {_d[len(_d)//2]}°): "
              + ", ".join(f"{w}#{i}+{d}°" for w, i, d in sorted(_snapped, key=lambda x: -x[2])[:8]))
    else:
        print(f"[朝向] 开了吸附(YAW_SNAP={sorted(YAW_SNAP)}) 但一个都没吸 —— "
              f"要么没有墙线算不出网格, 要么都偏超过 {YAW_SNAP_MAX}°")
if skipped:
    # 按原因分开报 —— 混在一起的话, 排查时分不清是"阈值定错了"还是"盒子本身量错了",
    # 而这两件事的修法完全相反。
    _by={}
    for w,oid,ext,why in skipped: _by.setdefault(why,[]).append((w,oid,ext))
    print(f"[跳过] 共 {len(skipped)} 个:")
    for why,items in sorted(_by.items()):
        print(f"   {why:14s} {len(items):3d} 个: {items[:5]}")
# ★ 完整的跳过名单落盘。日志里每条原因只打前 5 个, 查"这个实例为什么没进场景"时拿不全。
#   只写一个旁路文件, 不影响任何下游; 查看页(tools/box_fit_view.py)会读它。
json.dump(dict(scene=os.path.basename(dst), exclude=sorted(x for x in EXCLUDE if x),
               skipped=[dict(word=w, obj_id=int(oid), why=why, ext=ext) for w, oid, ext, why in skipped]),
          open(dst[:-4] + "_skipped.json", "w"), ensure_ascii=False, indent=1)
