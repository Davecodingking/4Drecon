#!/usr/bin/env bash
# 无头浏览器(playwright 自带的 chromium)缺系统库 libasound.so.2, 借 conda 包缓存里那份, 只对这一个进程生效
export LD_LIBRARY_PATH=/home/ubuntu/miniforge3/pkgs/alsa-lib-1.2.16.1-h7cc23a3_1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec /home/ubuntu/miniforge3/envs/sam3d/bin/python "$(dirname "$0")/web.py" "$@"
