#!/usr/bin/env python
"""室内外两条流水线共用的驱动层。

★ 为什么有这个文件(2026-09-20)
  算法脚本早就共用了 —— indoor_pipeline.py 的 SCRIPTS 直接指向 data/outdoor/scripts,
  一边改另一边立刻跟上。但**驱动层**一直是复制粘贴的: _timing / resolve_cfg /
  stamp_path 三个函数两边逐字相同, GpuLock 只差一句提示文字, envcmd 只差个默认值。
  复制粘贴的东西一定会漂移, 而且漂移的时候不报错 —— 显卡那两套"等 GPU"的机制
  就是这么分叉的(一边 flock 一边查 nvidia-smi, 结果谁也拦不住谁, 双双 OOM)。
  所以: 不该分歧的地方一律搬到这里, 两边 import。

  留在各自文件里的是**真该不同**的那些: Ctx(路径不同)、s_frames/s_da3/s_gen/s_render
  (室内外做的事就是不一样)、do_check/main(室内多一个 --chain)。

★ 这个文件不依赖任何一条流水线, 只用标准库。别往里加场景相关的东西。
"""
import fcntl, json, os, shlex, subprocess, time


# ==================================================================== GPU 排队
# ★ 两条流水线(以及 body4d 那条)在**同一张卡上**共用同一把锁。锁文件名的规则一个字都不能改 ——
#   改了就等于同一张卡又分成两个队, 而分成两个队的后果见文件头。shell 那边(gpulock.sh)是同一个规则。
# ★ 2026-09-24 从"整台机器一把锁"改成"每张卡一把": 8x3090 上各轮次钉在不同的卡上
#   (drive_go.sh 开跑时挑空闲的卡), 锁只该挡同一张卡上的人。卡号取 CUDA_VISIBLE_DEVICES 的第一个
#   (torch 的 cuda:0), 不设就是 0 —— 和 wait_gpu() 看的是同一张卡。
def lock_path_of(dev):
    return f"/tmp/outdoor_pipeline_gpu{dev}.lock"


def lock_path():
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip() or "0"
    return lock_path_of(dev)


LOCK = lock_path()      # 老的 import 名; 进程的卡在启动时就定了, 算一次就够


class GpuLock:
    """吃显存的步骤串行执行。不是调参数能解决的问题: 单份 SAM3D 驻留 20GB 出头,
    32G 卡装不下两份。之前 beach 和 carpark 并行跑 gen, 两边同时被 CUDA OOM 杀掉,
    而且日志一停就再没有新内容, 靠"看日志有没有更新"的监控反而永远静默。

    和 wait_gpu() 是两层, 两层都要:
      锁       管"我们自己的几条流水线之间" —— 对方即使还没吃上显存也得排队,
               因为它可能正在加载模型, 这时候查显存是看不出来的
      wait_gpu 管"流水线之外的第三方进程" —— 那些进程不会来拿这把锁, 只能靠查显存看见
    """
    def __enter__(self):
        self.f = open(lock_path(), "a+")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("    [等待] 另一条流水线正在占用 GPU, 排队中...", flush=True)
            fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()


def wait_gpu(need_mb, dry=False, timeout=10800):
    """等到卡上没有别人的进程、而且余量够了才返回。

    ★ 只看瞬时余量不够: 别人一个实例只跑一分多钟, 空档期能骗过检查, 然后在我们跑到
      一半时又把显存吃回去(实测被这么坑过, DA3 前向到 dinov2 attention 才炸)。
      所以要等到**没有别的进程占着卡**。
    ★ 只等不杀 —— 别人的活不归我们管。
    ★ 2026-09-24 迁到 8x3090 共享机: 只看 CUDA_VISIBLE_DEVICES 的第一张卡(torch 的 cuda:0)。
      原来数的是**所有卡**上的进程、查的是 0 号卡的余量 —— 别的卡上有人(UnrealEditor)就永远
      轮不上, 排满 3 小时报失败。配 CUDA_DEVICE_ORDER=PCI_BUS_ID, nvidia-smi 的编号才和 CUDA 一致。
    """
    if dry:
        return
    me, t0 = os.getpid(), time.time()
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    while True:
        g = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        rows = [[x.strip() for x in l.split(",")] for l in g.stdout.strip().splitlines() if l.strip()]
        row = next((r for r in rows if dev in (r[0], r[1]) or (dev.startswith("GPU-") and r[1].startswith(dev))), rows[0])
        uuid, used, total = row[1], int(row[2]), int(row[3])
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        others = 0
        for line in q.stdout.strip().splitlines():
            if not line.strip():
                continue
            u, pid, mb = [x.strip() for x in line.split(",")]
            if u == uuid and int(pid) != me and int(mb) > 2000:
                others += int(mb)
        if not others and total - used >= need_mb:
            return
        if time.time() - t0 > timeout:
            raise SystemExit(f"[失败] 排了 {timeout/60:.0f}min 队还是轮不上 "
                             f"(需要 {need_mb}MB, 别人占着 {others}MB)")
        print(f"    别的活占着 {others}MB, 空闲 {total-used}MB / 需要 {need_mb}MB, "
              f"等 30s (已等 {(time.time()-t0)/60:.0f}min)", flush=True)
        time.sleep(30)


# ==================================================================== 领卡(2026-09-29)
# ★ 以前一轮开跑时 drive_go.sh 挑一张卡, 整轮所有步骤都钉在这张卡上, 另外 7 张闲着。
#   现在每个吃显存的任务开始时领一张空卡, 跑完还 —— 同一轮里几个步骤可以各用一张同时跑(见 sched.py)。
# ★ 每张卡还是那一把锁, 文件名不变(lock_path_of)。旧方式起的轮次、gpulock.sh 和这里互相认得。
MAX_GPUS = 8          # 一轮最多同时用几张卡(环境变量 MAX_GPUS 可改)。现在一个人用 8 张; 同事一起用时改成 3
FOREIGN_MB = 2000     # 卡上别人的进程占显存超过这么多 = 这张卡有人在用(和 wait_gpu 同一个数)
SLOT_POLL = 5         # 秒: 没领到卡, 隔多久再找一遍
SLOT_TIMEOUT = 10800  # 秒: 排这么久还领不到就报失败(和 wait_gpu 一样 3 小时)
CPU_THREADS = 8       # 每条命令的 CPU 线程数上限(环境变量 CPU_THREADS 可改), 理由见 cpu_threads()


def parallel_on():
    """PARALLEL=0 = 和以前完全一样一步一步跑。出问题时用它退回去。"""
    return os.environ.get("PARALLEL", "1") != "0"


def max_gpus():
    return max(1, int(os.environ.get("MAX_GPUS", MAX_GPUS)))


def cpu_threads():
    """每条命令默认的 CPU 线程数上限(OMP / MKL / OPENBLAS / NUMEXPR)。
    ★ 这台机器 255 个核, torch / numpy 默认按核数开线程, 大多数在空转还互相抢:
      2026-09-29 实测出人网格(nfl1 人 1): 默认 载模型 80s + 出网格 143s, CPU 用户时间 17354s;
      限成 8 -> 48s + 109s, CPU 用户时间 211s, 结果的差别在两次重跑的差别以内(1 mm)。
      不限的话几个步骤一起跑会互相拖慢: 6 个出网格进程一起跑, 机器负载 500+, 载模型从 59s 变成 265s。"""
    return max(1, int(os.environ.get("CPU_THREADS", CPU_THREADS)))


def _smi(query, what="gpu"):
    r = subprocess.run(["nvidia-smi", f"--query-{what}={query}", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return [[x.strip() for x in l.split(",")] for l in r.stdout.strip().splitlines() if l.strip()]


def gpu_pool():
    """可以用哪些卡。GPU_POOL 说了算; 没给但钉了 CUDA_VISIBLE_DEVICES(想钉某张卡: CUDA_VISIBLE_DEVICES=3
    drive_go.sh ...)就只用钉的; 都没给 = 机器上全部的卡。"""
    p = os.environ.get("GPU_POOL", "").strip() or os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if p:
        return [x.strip() for x in p.split(",") if x.strip()]
    return [r[0] for r in _smi("index")]


def card_busy(card, need_mb=0):
    """这张卡现在不能用: 有别人的进程占着显存(超过 FOREIGN_MB), 或者余量不够 need_mb。
    ★ 只看瞬时余量不够, 理由见 wait_gpu: 别人的活有空档期, 骗得过余量检查。"""
    row = next((r for r in _smi("index,uuid,memory.used,memory.total") if r[0] == str(card)), None)
    if row is None:
        return True
    me = os.getpid()
    others = sum(int(mb) for u, pid, mb in _smi("gpu_uuid,pid,used_memory", "compute-apps")
                 if u == row[1] and int(pid) != me and int(mb) > FOREIGN_MB)
    return bool(others) or int(row[3]) - int(row[2]) < need_mb


def _try_lock(path):
    fh = open(path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fh
    except OSError:
        fh.close()
        return None


def _unlock(fh):
    if fh not in (None, True):
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def _take_quota(run_dir):
    """本轮的名额: <本轮>/logs/.gpu_quota_<i>.lock 里拿到任意一把。几个子进程共用一轮的名额,
    所以是文件锁不是计数器。没有本轮目录(单跑一个脚本)就不限, 返回 True。满了返回 None。"""
    if not run_dir:
        return True
    os.makedirs(f"{run_dir}/logs", exist_ok=True)
    for i in range(max_gpus()):
        fh = _try_lock(f"{run_dir}/logs/.gpu_quota_{i}.lock")
        if fh:
            return fh
    return None


def _take_card(pool, busy, need_mb, avoid=()):
    """avoid: 上次在这张卡上失败了, 有别的就先用别的。"""
    for card in [c for c in pool if c not in avoid] + [c for c in pool if c in avoid]:
        fh = _try_lock(lock_path_of(card))
        if not fh:
            continue
        if busy(card, need_mb):
            _unlock(fh)
            continue
        return card, fh
    return None


def gpus_free_now(need_mb=0, run_dir=None, pool=None, busy=None):
    """现在马上能领到几张卡: 空卡数和本轮剩余名额里小的那个。只看一眼, 不占。拆分时用它定拆几份。"""
    run_dir = run_dir if run_dir is not None else os.environ.get("RUN_DIR") or None
    pool, busy = (gpu_pool() if pool is None else pool), busy or card_busy
    held, cards = [], 0
    try:
        for card in pool:
            fh = _try_lock(lock_path_of(card))
            if fh:
                held.append(fh)
                cards += 0 if busy(card, need_mb) else 1
        if not run_dir:
            return cards
        os.makedirs(f"{run_dir}/logs", exist_ok=True)
        quota = 0
        for i in range(max_gpus()):
            fh = _try_lock(f"{run_dir}/logs/.gpu_quota_{i}.lock")
            if fh:
                held.append(fh); quota += 1
        return min(cards, quota)
    finally:
        for fh in held:
            _unlock(fh)


class GpuSlot:
    """领一张空卡, 用完还。with GpuSlot("track") as s: ... s.card 是卡号。
    set_env=True: 把本进程的 CUDA_VISIBLE_DEVICES 设成这张卡(之后起的命令都落在它上面), 出来时恢复。
    一个进程里开几个线程各领一张的(拆分)用 set_env=False, 自己把卡号写进命令的环境变量。"""

    def __init__(self, what="", need_mb=0, run_dir=None, pool=None, busy=None, set_env=True, avoid=(),
                 poll=SLOT_POLL, timeout=SLOT_TIMEOUT):
        self.what, self.need_mb, self.set_env, self.avoid = what, need_mb, set_env, tuple(avoid)
        self.run_dir = run_dir if run_dir is not None else os.environ.get("RUN_DIR") or None
        self.pool, self.busy, self.poll, self.timeout = pool, busy or card_busy, poll, timeout
        self.card = self.waited = self._fh = self._quota = self._old = self._mark = None

    def __enter__(self):
        t0, said = time.time(), False
        while True:
            pool = gpu_pool() if self.pool is None else self.pool
            q = _take_quota(self.run_dir)
            got = _take_card(pool, self.busy, self.need_mb, self.avoid) if q else None
            if got:
                (self.card, self._fh), self._quota = got, q
                break
            _unlock(q)
            if time.time() - t0 > self.timeout:
                raise SystemExit(f"[失败] {self.what or '这一步'} 排了 {self.timeout / 60:.0f}min 队还是领不到卡 "
                                 f"(可用的卡 {pool}, 本轮最多 {max_gpus()} 张)")
            if not said:
                print(f"    [等待] {self.what or '这一步'}: 没有空卡, 或本轮 {max_gpus()} 张的名额用满了, 排队中...",
                      flush=True)
                said = True
            time.sleep(self.poll)
        self.waited = time.time() - t0
        if self.set_env:
            self._old = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.card)
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # nvidia-smi 的编号才和 CUDA 一致
        if self.run_dir:
            # 谁在用哪张卡: 网页和看板读这个文件。进程被杀的话文件会留下, 读的人要核对 pid 还在不在
            self._mark = f"{self.run_dir}/logs/.gpu_{self.card}.json"
            tmp = f"{self._mark}.{os.getpid()}.tmp"
            json.dump({"卡": str(self.card), "步骤": self.what, "pid": os.getpid(),
                       "开始": time.strftime("%F %T")}, open(tmp, "w"), ensure_ascii=False)
            os.replace(tmp, self._mark)
        print(f"    [卡] {self.what or '这一步'} 用 GPU {self.card}"
              + (f" (等了 {self.waited:.0f}s)" if self.waited >= 1 else ""), flush=True)
        return self

    def __exit__(self, *a):
        if self._mark:
            try:
                os.remove(self._mark)
            except OSError:
                pass
        if self._old is not None:
            for k, v in self._old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        _unlock(self._fh)
        _unlock(self._quota)


# ==================================================================== 跑命令
# 正在跑的步骤名。两个 pipeline 的主循环每进一步就设一次(`_common.CUR_STEP = k`),
# sh() 拿它往步骤日志头上打分隔线。不用参数传是因为 sh() 的调用点有上百处。
CUR_STEP = None
_SEP_DONE = set()          # (日志文件, 步骤) 已经打过分隔线了 —— 一步里 sh() 调好几次只打一条


def sh(cmd, log=None, dry=False, cwd=None, sep=None, env=None):
    """跑一条命令, 打印出来的那行要能直接粘去复现。

    sep: 写进日志的分隔条(室外每步一条, 便于在一个大日志里定位)。
    env: 额外的环境变量(室内要给 CUDA_HOME 那一套; 室外把环境变量写在命令行里, 不用这个)。
         给了 env 的时候也会打印在命令前面, 保证"打印的 == 实际跑的"。
    """
    line = cmd if isinstance(cmd, str) else " ".join(shlex.quote(str(x)) for x in cmd)
    pre = " ".join(f"{k}={v}" for k, v in (env or {}).items())
    print(f"    $ {pre + ' ' if pre else ''}{line}" +
          (f"   > {os.path.basename(log)}" if log else ""), flush=True)
    if dry:
        return 0
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        # ★ 每一步第一次写它的日志之前, 打一条 "== <步骤>  <时间戳>"。两个用处:
        #   1) 看板要知道"正在跑的这一步什么时候开始的"。耗时是从 logs/timing.tsv 读的,
        #      而那个文件**跑完才追行** —— 在跑的时候 started/took 两列只能空着,
        #      偏偏 solo 这种二十分钟的步骤最需要看已经跑了多久。
        #      格式得和 tools/watch_run.py 的 SEP 正则对上, 改这里要一起改那边。
        #   2) 步骤日志是**追加**的, 同一个文件里叠着好几轮的输出, 没有分隔线人分不出
        #      哪一段是哪一轮(实测 office 的 06_solo.log 叠了三轮, head 读到的是最老那轮)。
        #   一步里 sh() 可能调好几次(da3 就调四次写同一个日志), 只在第一次打 ——
        #   否则看板取最后一条, 显示的成了最后一条子命令的开始时刻, 不是这一步的。
        _k = (os.path.abspath(log), CUR_STEP)
        if sep:
            with open(log, "a") as f:
                f.write(sep)
        elif _k not in _SEP_DONE:
            _SEP_DONE.add(_k)
            with open(log, "a") as f:
                f.write(f"\n== {CUR_STEP or '?'}  {time.strftime('%F %T')}\n")
    # ★ 2026-09-24: 每一步默认 HF_HUB_OFFLINE=1。权重全在 ~/.cache/huggingface, 但 transformers
    #   装模型时每个文件都要上网 HEAD 一次核对版本; 本机直连不了 huggingface.co, 只能走 ~/.bashrc
    #   的代理(借用户电脑的网), 隧道一卡就 10 s 超时 × 重试 5 次 —— nfl1 的 verify 验 1 个实例
    #   用了 18 分钟。以前只有 SAM3D 生成那步单独加了, Qwen-VL 那几步(vlm/verify/vlm_sizes)漏了;
    #   放在这里所有步骤一起管。要下新模型: 环境变量或 env= 里显式给 HF_HUB_OFFLINE=0。
    run_env = dict(os.environ); run_env.setdefault("HF_HUB_OFFLINE", "1")
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        run_env.setdefault(_k, str(cpu_threads()))          # 人明确设了的不动; 步骤自己给的(env=)在下面覆盖
    if env:
        run_env.update(env)
        if "CUDA_HOME" in env:
            run_env["PATH"] = env["CUDA_HOME"] + "/bin:" + run_env.get("PATH", "")
    with (open(log, "a") if log else open(os.devnull, "w")) as f:
        out = f if log else None
        r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=run_env,
                           stdout=out or None, stderr=subprocess.STDOUT if out else None)
    if r.returncode:
        raise SystemExit(f"[失败] 退出码 {r.returncode}: {line}")
    return r.returncode


def envcmd(py, script, env=None, args=None):
    """把环境变量写在命令行最前面, 这样 --dry-run 打印出来的就是能直接粘去跑的命令。
    data/outdoor/scripts 底下的脚本大多没有 argparse, 只吃环境变量, 所以这个用得很多。"""
    pre = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items())
    # ★ env 为空时不能留下前导空格 —— 打印出来的那行是给人直接粘去跑的, "$  python ..."
    #   看着像笔误。body4d 原来的写法是对的, 合并时差点把它弄丢。
    return (pre + " " if pre else "") + f"{shlex.quote(py)} {shlex.quote(script)} " + " ".join(
        shlex.quote(str(x)) for x in (args or []))


# ==================================================================== 每轮配置
# ★ 2026-09-24: <R1>/00_run_config.json —— 这一轮的配置, 启动页 / 驱动命令行写, 第 1 道闸可改。
#   {"类型": "outdoor"|"indoor", "覆盖": {CFG 键: 值}}
#   "覆盖"就是持久化的 CFG 覆盖(效果同环境变量, 但重跑不丢), 以后调一轮的参数不用再改 SCENES。
RUN_CONFIG = "00_run_config.json"


def load_run_config(R1):
    p = f"{R1}/{RUN_CONFIG}"
    return json.load(open(p)) if R1 and os.path.exists(p) else None


def save_run_config(R1, d):
    os.makedirs(R1, exist_ok=True)
    tmp = f"{R1}/{RUN_CONFIG}.tmp"
    json.dump(d, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, f"{R1}/{RUN_CONFIG}")


# 启动页 / 闸 1 网页 / drive.py 命令行能改的项, 以及"跑完第几段之后就不许改"。
# 有没有人: people_track 在第 2 段(下标 1), 跑完 >1 段再改不会生效; 召回在第 4 段(下标 3)。
EDITABLE_UNTIL = {"PEOPLE": 1, "RECALL": 3}


def update_run_config(R1, kind, changes, done):
    """改本轮配置, 驱动命令行和网页共用这一套规则。done = 已经跑完的段数。
    类型开跑后锁定; 过了那一段还改的直接拒绝(改了也不会生效, 让人新开一轮)。不合规矩抛 ValueError。"""
    rc = load_run_config(R1) or {"类型": kind, "覆盖": {}}
    if rc.get("类型") != kind:
        raise ValueError(f"这一轮是 {rc.get('类型')}, 这里给的是 {kind}。类型开跑后不能改, 新开一轮")
    over = rc.setdefault("覆盖", {})
    for key, val in changes.items():
        if key not in EDITABLE_UNTIL:
            raise ValueError(f"{key} 不在可改的项里: {sorted(EDITABLE_UNTIL)}")
        if val not in ("0", "1"):
            raise ValueError(f"{key} 只能是 0 或 1, 给的是 {val!r}")
        if over.get(key) == val:
            continue
        if done > EDITABLE_UNTIL[key]:
            raise ValueError(f"{key} 的那一段已经跑过了(已跑完 {done} 段), 现在改不会生效 —— 新开一轮")
        over[key] = val
    save_run_config(R1, rc)
    return rc


# ==================================================================== 配置 / 戳 / 计时
def resolve_cfg(CFG, SCENES, scene, R1=None, review=None):
    """生效顺序: CFG 默认值 <- 场景覆盖 <- 本轮配置 <- 人工闸决策 <- 环境变量。

    人工闸排在环境变量之前 —— 临时敲一个环境变量试值, 应该压得过闸的决定;
    但闸的决定要压得过写死在 SCENES 里的默认, 否则人在闸口做的判断永远轮不上。
    本轮配置(00_run_config.json)在场景之后、闸之前: 它是开跑时人对这一轮定的调子。

    ★ 环境变量那一层只认已经在 CFG 里的键, 所以 CFG 同时也是"可调参数白名单"。
      写错名字的环境变量会被静默忽略 —— 这是故意的, 否则整个 os.environ 都会灌进来。
    ★ 本轮配置里写错键**直接报错**: 它是人写进盘里要长期生效的, 静默忽略等于这个设置从来没生效过。
    """
    cfg = dict(CFG)
    cfg.update(SCENES.get(scene, {}))
    rc = load_run_config(R1)
    if rc:
        over = rc.get("覆盖", {})
        bad = sorted(k for k in over if k not in CFG)
        if bad:
            raise SystemExit(f"[失败] {R1}/{RUN_CONFIG} 的覆盖里有不认识的键: {bad}")
        cfg.update({k: str(v) for k, v in over.items()})
    if R1 and review is not None:
        review.apply_to_cfg(cfg, R1)
    cfg.update({k: v for k, v in os.environ.items() if k in CFG})
    return cfg


def stamp_path(c, key):
    return f"{c.LOG}/.stamp_{key}"


def timing(c, k, env, t0, t_run, status):
    """每一步的时间开销追加进 logs/timing.tsv(成功失败都记)。
    列: 步骤 环境 开始 结束 耗时秒 等GPU秒 结果。终端上那行 "12.3s" 滚走就没了, 这份留得住,
    跑完用 `column -t -s$'\\t' logs/timing.tsv` 看。
    ★ 为什么失败也记: "跑了 40 分钟才挂"和"一上来就崩"在日志里长得很像, 在这张表里一眼分得开。
    ★ 为什么等锁的时间单独一列: 不分开的话, 改动到底有没有让这一步变快会被排队时间淹掉。"""
    p = f"{c.LOG}/timing.tsv"
    new = not os.path.exists(p)
    t1 = time.time()
    with open(p, "a") as f:
        if new:
            f.write("步骤\t环境\t开始\t结束\t耗时秒\t等GPU秒\t结果\n")
        f.write(f"{k}\t{env}\t{time.strftime('%F %T', time.localtime(t0))}\t"
                f"{time.strftime('%F %T', time.localtime(t1))}\t{t1 - t_run:.1f}\t{t_run - t0:.1f}\t{status}\n")


# ==================================================================== 帧号 <-> 原视频
# ★ 为什么需要这个(2026-09-20)
#   两条链的 cameras.json 里, 键的含义**不一样**:
#     室外  键就是原视频帧号, 不等间隔(realstreet: 0000,0002,0003,0005,...,0284)
#     室内  键是重编号 0000..N-1, 原号另存在 frames_<run>/映射.json (office 步长 3.6, 不是整数)
#   任何要回到原视频时间轴的模块(人体动画、配音、字幕、和别的视频算法交叉验证)
#   如果自己从键名解析或者自己算公式, 在另一条链上必错, 而且错得很隐蔽 ——
#   拿一帧的内容去配另一帧的相机, 表现是"轮廓对、里面的画面错位"。
#
#   所以: da3 那一步跑完就在 cameras.json 旁边写一份 frame_map.json, 路径两条链完全一样,
#   下游一律查它, 谁也不准自己解析键名、不准算公式。
#
#   为什么不放在 frames 目录里: 室内的帧目录按轮次分(frames_<run>/), 室外的不分
#   (dataset/<场景>/frames), 放那儿的话下游还得写 if 室内/else, 正是要消灭的东西。
#   放在 cameras.json 旁边, 谁拿到 R1 谁就能查, 而且它是由写 cameras.json 的同一步
#   从 cameras.json 自己的键生成的, 不会漂移。
FRAME_MAP = "frame_map.json"


def _video_fps(video):
    """读视频的 fps。读不到就返回 None —— 不让它变成硬失败, 下游需要时间的自己判断。"""
    if not video or not os.path.exists(video):
        return None
    try:
        import cv2
        cap = cv2.VideoCapture(video)
        fps = float(cap.get(cv2.CAP_PROP_FPS)); cap.release()
        return round(fps, 4) if fps and fps > 0 else None
    except Exception:
        return None


def _orig_index(name, sub):
    """一个帧文件名 -> 原视频帧号。sub 是 映射.json 的内容(没有就是恒等)。"""
    i = int(os.path.splitext(os.path.basename(name))[0])
    return int(sub[str(i)]) if sub and str(i) in sub else i


def write_frame_map(da3_dir, frames_dir, video=None, dry=False):
    """da3 那步跑完调它, 在 cameras.json 旁边写 frame_map.json。

    frames_dir 是这一轮 da3 **实际吃的**那个帧目录 —— 室内两条链吃的不是同一个
    (SpatialLM 链吃 frames_bright, 是恒等; 物体链吃 frames_<run>, 有映射),
    光看目录里有没有 映射.json 去猜是靠巧合, 所以由知道答案的调用方传进来。
    """
    if dry:
        return
    cj = f"{da3_dir}/cameras.json"
    if not os.path.exists(cj):
        return
    names = json.load(open(cj))["images_used"]
    mp = f"{frames_dir}/映射.json"
    sub = json.load(open(mp)) if os.path.exists(mp) else None
    json.dump(dict(video=video, fps=_video_fps(video), frames_dir=frames_dir,
                   identity=sub is None,
                   frames={n: _orig_index(n, sub) for n in names}),
              open(f"{da3_dir}/{FRAME_MAP}", "w"), ensure_ascii=False, indent=1)


def load_frame_map(da3_dir):
    """读 frame_map.json, 返回 {帧文件名: 原视频帧号}。

    ★ 兼容旧轮次: 文件不存在时就地推一份 —— cameras.json 里有 source(当初吃的帧目录),
      顺着它找 映射.json, 找不到就是恒等。所以 09-20 之前跑的结果也能用, 不用重跑。
    """
    p = f"{da3_dir}/{FRAME_MAP}"
    if os.path.exists(p):
        return json.load(open(p))["frames"]
    c = json.load(open(f"{da3_dir}/cameras.json"))
    src = c.get("source") or ""
    mp = f"{src}/映射.json"
    sub = json.load(open(mp)) if src and os.path.exists(mp) else None
    return {n: _orig_index(n, sub) for n in c["images_used"]}
