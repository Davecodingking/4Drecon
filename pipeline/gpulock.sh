# 用法: . ~/sam3d/pipeline/gpulock.sh ; gpu <命令...>
# 和 _common.GpuSlot 用同一把锁(文件名规则 lock_path_of) —— 同一张卡上室内室外两条链必须串起来。
# 轮询 nvidia-smi 是错的做法(有竞态: 两边同时看到空档同时冲进去), flock 才是对的。
# ★ 每张卡一把锁, 规则和 _common.lock_path() 一样: CUDA_VISIBLE_DEVICES 的第一个, 不设就是 0
_gpu_dev=${CUDA_VISIBLE_DEVICES:-0}; _gpu_dev=${_gpu_dev%%,*}
GPU_LOCK=/tmp/outdoor_pipeline_gpu${_gpu_dev:-0}.lock
gpu () {
  echo "  [排队] 等 GPU 锁 $GPU_LOCK ..." >&2
  flock 9 || exit 1
  echo "  [拿到锁] $(date '+%H:%M:%S')" >&2
  "$@"
  local rc=$?
  return $rc
} 9>>"$GPU_LOCK"
