#!/usr/bin/env python
"""写图片: 内容没变就不落盘。

为什么需要这个 ——
  生成那步现在按 mtime 判过期("抠图比 mesh 新就重跑"), 这是对的: 人在闸口换了候选,
  抠图变了, mesh 必须跟着重出, 否则换候选永远是空操作。
  但挑图(pick)、验图(verify)、逐实例出图(gen_per_instance) 这三处都是**无条件** cv2.imwrite,
  内容一模一样也照写一遍, mtime 必然刷新。两件事凑在一起, 后果是:
  任何一次重跑挑图, 都会让盘上**全部** mesh 变成"过期", 下一次 gen/solo 全部重烧。
  beach r2 实测: 29 个已有 mesh 里 19 个的抠图比 mesh 新, 而图的内容其实一个像素都没变 ——
  白烧 20~40 分钟 GPU, 而且不吭声。

  所以判据不能靠"写没写过", 要靠"内容变没变"。这里做的就是这件事:
  编码成 png 字节和盘上那份逐字节比, 一样就原样不动, mtime 保持不变。
"""
import os
import cv2


def imwrite_stable(path, img, ext=".png"):
    """内容变了才写。返回 True = 真写了。

    比字节不比像素: 编码参数、通道顺序、alpha 有没有被丢掉, 这些差异都该算"变了" ——
    它们确实会让 SAM3D 吃到不一样的输入。
    """
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"[失败] 编码不了 {path}")
    new = buf.tobytes()
    if os.path.exists(path):
        try:
            if open(path, "rb").read() == new:
                return False
        except OSError:
            pass                      # 读不了就当它变了, 重写一遍
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(new)
    return True
