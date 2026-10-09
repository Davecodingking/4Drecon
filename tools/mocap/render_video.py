"""从原视频的视角, 在 Blender 里把"场景 + 人"渲成一段视频。
吃 to_blender.py 出的 <名字>_scene_people.blend: 相机序列、灯、分辨率、渲染引擎都已经存在里面, 这里只管渲和压。
  相机 = 流水线的相机(每个场景帧一个位姿 + 焦距), 帧号 = 原视频帧号 -> 这段视频和原视频一帧对一帧。
  引擎 = EEVEE(不开窗口也能用显卡渲, 1080p 一帧约 1 秒); 分 --jobs 个进程各渲一部分帧, 最后用 ffmpeg 压成 mp4。
  实测(2026-10-09 office r10): 432 帧 1920x1072, 4 个进程 118 秒; 4 个进程已经把一张卡用满, 再加进程不会更快。
★ EEVEE 用哪张卡不归 CUDA_VISIBLE_DEVICES 管(它走的是 OpenGL, 不是 CUDA): 这台机器上总是落在 3 号卡,
  4 个进程一共占约 5.5 GB 显存。别的任务正好也在 3 号卡上时两边都会慢, 但不会出错。

用法(sam3d 环境; 它自己去起 ~/venvs/bpy 的进程渲):  python render_video.py <.blend> <输出.mp4> [--jobs 4] [--samples 16]
内部用(bpy 环境):                                   python render_video.py <.blend> <帧目录> --part <第几份> <一共几份>
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time

BPY = os.path.expanduser("~/venvs/bpy/bin/python")


def render_part(blend, frames_dir, i, n, samples):
    """bpy 环境: 从第 (起始帧 + i) 帧起每隔 n 帧渲一帧, 文件名带帧号。第 0 份顺手把帧范围 / 帧率写下来给压视频的用。"""
    import bpy
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(blend))
    sc = bpy.context.scene
    if sc.camera is None:
        raise SystemExit(f"[失败] {blend} 里没有相机(2026-10-09 之前出的 .blend 没有) —— 重跑 motion")
    if i == 0:
        json.dump(dict(lo=sc.frame_start, hi=sc.frame_end, fps=float(sc.get("src_fps", sc.render.fps)),
                       wh=[sc.render.resolution_x, sc.render.resolution_y]), open(f"{frames_dir}/info.json", "w"))
    sc.frame_start, sc.frame_step = sc.frame_start + i, n
    sc.eevee.taa_render_samples = samples
    sc.render.image_settings.file_format = "JPEG"; sc.render.image_settings.quality = 95
    sc.render.filepath = os.path.join(frames_dir, "f_")
    if sc.frame_start <= sc.frame_end:
        bpy.ops.render.render(animation=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("blend"); ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=4, help="几个进程一起渲")
    ap.add_argument("--samples", type=int, default=16, help="每帧采样数: 越多边缘越平滑、越慢")
    ap.add_argument("--part", type=int, nargs=2, metavar=("I", "N"), help="内部用: 只渲第 I 份(一共 N 份), <输出> 是帧目录")
    a = ap.parse_args()
    if a.part:
        return render_part(a.blend, a.out, a.part[0], a.part[1], a.samples)
    if not os.path.exists(a.blend):
        raise SystemExit(f"[失败] 没有 {a.blend} —— 先跑 motion")
    t0 = time.time(); tmp = tempfile.mkdtemp(prefix="blender_video_")
    try:
        procs = [subprocess.Popen([BPY, os.path.abspath(__file__), a.blend, tmp, "--part", str(i), str(a.jobs), "--samples", str(a.samples)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) for i in range(a.jobs)]
        errs = [p.communicate()[1] for p in procs]
        info = json.load(open(f"{tmp}/info.json")) if os.path.exists(f"{tmp}/info.json") else None
        want = info["hi"] - info["lo"] + 1 if info else -1
        got = len([f for f in os.listdir(tmp) if f.endswith(".jpg")])
        if any(p.returncode for p in procs) or got != want:
            raise SystemExit(f"[失败] 渲出来 {got} 帧(应该 {want} 帧), 进程返回值 {[p.returncode for p in procs]}\n"
                             + "\n".join(e[-800:] for e in errs if e and e.strip()))
        import imageio_ffmpeg
        part = a.out + ".part.mp4"                     # 压完再改名: 压到一半断了不留半个文件
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", f"{info['fps']:.6f}",
                        "-start_number", str(info["lo"]), "-i", f"{tmp}/f_%04d.jpg",
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", part], check=True)
        os.replace(part, a.out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"-> {a.out}  {os.path.getsize(a.out) / 1e6:.1f} MB, 第 {info['lo']}~{info['hi']} 帧({want} 帧), "
          f"{info['wh'][0]}x{info['wh'][1]}, {info['fps']:.3f} fps, {a.jobs} 个进程渲了 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
