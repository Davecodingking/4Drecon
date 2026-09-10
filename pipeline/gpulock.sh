# 用法: . ~/sam3d/pipeline/gpulock.sh ; gpu <命令...>
# 和 outdoor_pipeline.py 的 GpuLock 用同一把锁 —— 室内室外两条链共卡, 必须串起来。
# 轮询 nvidia-smi 是错的做法(有竞态: 两边同时看到空档同时冲进去), flock 才是对的。
GPU_LOCK=/tmp/outdoor_pipeline_gpu.lock
gpu () {
  echo "  [排队] 等 GPU 锁 $GPU_LOCK ..." >&2
  flock 9 || exit 1
  echo "  [拿到锁] $(date '+%H:%M:%S')" >&2
  "$@"
  local rc=$?
  return $rc
} 9>>"$GPU_LOCK"
