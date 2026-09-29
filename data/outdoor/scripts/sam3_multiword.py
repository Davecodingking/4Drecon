"""SAM3 跟踪: 一批词共用一次检测, 跟踪仍然一个词一个会话。给 track_all.py 用。

为什么这样拆 (实测踩过, 别改回"一个会话挂多个词"):
  每帧的耗时 = 画面编码 + 检测器(拿词找东西) + 跟踪器(新建物体/去重/抢像素)。
  - 前两步跟别的词无关: 检测器本来就吃一批词(pred_logits 第一维就是词), 每个词
    各算各的、NMS 也按词做。一批词放一起算, 结果和逐词算一样, 但画面只编码一次。
  - 跟踪器不行: 它判断"新检测是不是已经在跟的物体"、开头几帧删重复、重叠处抢像素,
    都是拿会话里所有物体一起比。几个词挤进一个会话, 不同词的物体会互相压 ——
    pedestrian r2 的公交站就是被同批的雨篷(awning)压没的, 盘上根本没有它。
  所以: 第一个词的会话跑时, 用整批词把每帧检测算出来缓存(按词切好); 后面每个词
  照旧开自己的会话, 检测那一步直接拿缓存, 跟踪那一步和逐词跑一模一样。

实现: 只挂 run_backbone_and_detection 一个钩子, 不改 SAM3 源码。
  - 在 shared_detection(pred, words) 里跑 track() 就生效, 出了 with 就恢复原样
  - 缓存按帧存在内存(每帧的跟踪器特征要搬到 CPU, 整段视频十几 G, 机器够)
  - 整批检测 OOM 抛 SharedDetOOM (不是 torch.OutOfMemoryError), 免得被逐词那套
    "提阈值重试"接住 —— 那套管的是跟踪器显存, 这里是检测批太大, 该做的是拆小批
"""
import copy, inspect
import torch

class SharedDetOOM(RuntimeError):
    pass

_CTX = None          # 当前批: dict(words, index, cache, ...); None = 不共用, 走原逻辑
_STASH = {}          # 检测器刚输出的 pred_logits


def _to(obj, dev):
    if torch.is_tensor(obj): return obj.to(dev, non_blocking=False)
    if isinstance(obj, dict): return {k: _to(v, dev) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return type(obj)(_to(v, dev) for v in obj)
    return obj


def install_hooks(pred):
    model = pred.model
    if getattr(model, "_shared_det_hooked", False):
        return
    det = model.detector
    o_fwd = det.forward_video_grounding_multigpu
    def fwd(*a, **k):
        out = o_fwd(*a, **k)
        img_out = out[0] if isinstance(out, tuple) else out
        if isinstance(img_out, dict) and "pred_logits" in img_out:
            _STASH["logits"] = img_out["pred_logits"]
        return out
    det.forward_video_grounding_multigpu = fwd

    o_run = model.run_backbone_and_detection
    sig = inspect.signature(o_run)
    def run(*a, **k):
        if _CTX is None:
            return o_run(*a, **k)
        b = sig.bind(*a, **k); b.apply_defaults()
        return _shared(model, o_run, dict(b.arguments))
    model.run_backbone_and_detection = run
    model._shared_det_hooked = True


def _batch_inputs(C, ib):
    """整批词的查询: 每帧 N 个查询(同一张图 x N 个词), 几何占位符批大小 N"""
    from sam3.model.data_misc import FindStage, convert_my_tensors
    from sam3.model.utils.misc import copy_data_to_device
    N = len(C["words"]); dev = ib.find_inputs[0].text_ids.device
    mib = copy.copy(ib)
    mib.find_text_batch = list(C["words"]) + ["visual"]
    mib.find_inputs = copy_data_to_device([convert_my_tensors(FindStage(
        img_ids=[t] * N, text_ids=list(range(N)),
        input_boxes=[torch.zeros(258)] * N,
        input_boxes_mask=[torch.empty(0, dtype=torch.bool)] * N,
        input_boxes_label=[torch.empty(0, dtype=torch.long)] * N,
        input_points=[torch.empty(0, 257)] * N,
        input_points_mask=[torch.empty(0)] * N, object_ids=[]))
        for t in range(len(ib.find_inputs))], dev)
    z = lambda *s, dt=torch.float32: torch.zeros(*s, device=dev, dtype=dt)
    P = C["prompt_cls"]
    geo = P(box_embeddings=z(0, N, 4), point_embeddings=z(0, N, 2),
            box_mask=z(N, 0, dt=torch.bool), point_mask=z(N, 0, dt=torch.bool),
            box_labels=z(0, N, dt=torch.long), point_labels=z(0, N, dt=torch.long),
            mask_labels=z(0, N, dt=torch.long), mask_mask=z(N, 0, dt=torch.bool))
    return mib, geo


def _shared(model, o_run, A):
    C = _CTX; fi = A["frame_idx"]; rev = A["reverse"]; fc = A["feature_cache"]
    ib = A["input_batch"]; dev = ib.find_inputs[0].text_ids.device
    key = (fi, rev)
    if key not in C["cache"]:
        if C["prompt_cls"] is None:
            C["prompt_cls"] = type(A["geometric_prompt"])
        if C["inputs"] is None:
            C["inputs"] = _batch_inputs(C, ib)
        mib, geo = C["inputs"]
        mfc = C["fc"]
        if "tracking_bounds" in fc:
            mfc["tracking_bounds"] = fc["tracking_bounds"]
        try:
            d = o_run(frame_idx=fi, num_frames=A["num_frames"], reverse=rev, input_batch=mib,
                      geometric_prompt=geo, feature_cache=mfc, allow_new_detections=True)
        except torch.OutOfMemoryError as e:
            raise SharedDetOOM(f"{len(C['words'])} 个词一起检测爆显存") from e
        # 扁平检测按词切开: 逐字复现 sam3_video_base.run_backbone_and_detection 的筛选,
        # torch.where 的第一维就是词序号, 行优先顺序 = 逐词跑时的查询顺序
        rows = torch.where(_STASH["logits"].squeeze(-1).sigmoid() > model.score_threshold_detection)[0]
        if rows.numel() != d["scores"].shape[0]:
            raise RuntimeError(f"帧 {fi}: 检测条数对不上 ({rows.numel()} vs {d['scores'].shape[0]}), 不猜")
        per_word = [{k: v[rows == i].to("cpu") for k, v in d.items()} for i in range(len(C["words"]))]
        C["cache"][key] = (per_word, _to(mfc[fi][1], "cpu"))
    per_word, backbone = C["cache"][key]
    # 和原函数一样: 本帧跟踪器特征放进这个会话自己的 feature_cache, 丢掉上一帧
    fc[fi] = (ib.img_batch[fi], _to(backbone, dev))
    fc.pop(fi - 1 if not rev else fi + 1, None)
    out = {k: v.to(dev) for k, v in per_word[C["index"]].items()}
    if not A["allow_new_detections"]:
        out = {k: v[:0] for k, v in out.items()}
    return out


class shared_detection:
    """with shared_detection(pred, words) as sd:
           for w in words: sd.use(w); track(...)"""
    def __init__(self, pred, words):
        self.words = list(words)
    def __enter__(self):
        global _CTX
        _CTX = dict(words=self.words, index=0, cache={}, fc={}, inputs=None, prompt_cls=None)
        return self
    def use(self, word):
        _CTX["index"] = self.words.index(word)
    def __exit__(self, *exc):
        global _CTX
        _CTX = None; _STASH.clear()
        return False


class paused:
    """分段跑(三级)时帧号是段内的, 和缓存对不上 —— 临时关掉共用"""
    def __enter__(self):
        global _CTX
        self.saved, _CTX = _CTX, None
    def __exit__(self, *exc):
        global _CTX
        _CTX = self.saved
        return False
