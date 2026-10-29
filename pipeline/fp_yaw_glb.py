"""FP 定朝向(fpose 那一步)里不用模型的部分。fp_yaw.py 在 fpose 环境里调它, 测试在 sam3d 环境里调它(只要 numpy)。

  rot_about / yaw_of   绕"天"转的矩阵; FP 给的位姿相对原摆放绕天转了几度(它顺带挪的位置、歪的一点都扔掉)
  apply_yaws           改 glb 的节点矩阵, 让点名的物体绕自己盒子中心转 —— 只改 JSON 里的 matrix, 网格和贴图原样不动
  source / mark_done   重跑时从组装出的原摆放(07_scene_摆放.glb)转, 不在已经转过的 07_scene.glb 上再转一遍

★ 07_scene.glb 每个节点都是同一个导出旋转、顶点烘在 DA3 世界系(assemble_scene 导出时就是这样)。
  所以"DA3 世界系里绕天、过中心 c 转"写成 glTF 里的矩阵 = inverse_T⁻¹ · 绕(UP, c) · inverse_T, 左乘到节点矩阵上。"""
import hashlib, json, os, shutil, struct
import numpy as np
from world_frame import inverse_T

BACKUP = "07_scene_摆放.glb"      # 组装出的原摆放(这一步之前的 07_scene.glb)
MARK = "07_fp_yaw.json"           # 这一步的结果 + 它写出的 07_scene.glb 的指纹


def rot_about(axis, deg, center=None):
    a = np.deg2rad(deg); k = np.asarray(axis, float) / np.linalg.norm(axis)
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    T = np.eye(4); T[:3, :3] = np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx
    if center is not None:
        c = np.asarray(center, float); T[:3, 3] = c - T[:3, :3] @ c
    return T


def yaw_of(Pc, Q, w2c, UP):
    """Q 相对 Pc(都是物体->相机)在世界系里绕天转了多少度, 0~360; 以及天被它歪了几度。"""
    Rw = np.linalg.inv(w2c[:3, :3]) @ Q[:3, :3] @ Pc[:3, :3].T @ w2c[:3, :3]
    e1 = np.cross(UP, [1, 0, 0])
    if np.linalg.norm(e1) < 1e-6: e1 = np.cross(UP, [0, 0, 1])
    e1 = e1 / np.linalg.norm(e1); e2 = np.cross(UP, e1); x = Rw @ e1
    return (float(np.degrees(np.arctan2(x @ e2, x @ e1))) % 360,
            float(np.degrees(np.arccos(np.clip((Rw @ UP) @ UP, -1, 1)))))


def _read(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[12:16])[0]
    return json.loads(b[20:20 + n]), b[20 + n:]        # JSON, 后面的二进制块原样带走


def _write(path, j, rest):
    s = json.dumps(j, separators=(",", ":")).encode(); s += b" " * (-len(s) % 4)
    body = struct.pack("<I", len(s)) + b"JSON" + s + rest
    tmp = path + ".tmp"
    with open(tmp, "wb") as f: f.write(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    os.replace(tmp, path)


def apply_yaws(src, dst, yaws, UP):
    """yaws = {节点名: (度, DA3 世界系里的中心)}。没点名的节点一个字节都不改。"""
    j, rest = _read(src)
    INV = inverse_T(np.asarray(UP, float) / np.linalg.norm(UP))
    left = set(yaws)
    for n in j["nodes"]:
        if n.get("name") not in yaws: continue
        assert not n.get("children"), f"{n['name']} 有子节点, 这里只认平的场景"
        deg, c = yaws[n["name"]]
        M = np.asarray(n.get("matrix", np.eye(4).ravel()), float).reshape(4, 4).T     # glTF 是列主序
        for k in ("translation", "rotation", "scale"):
            assert k not in n, f"{n['name']} 用的是 {k}, 不是 matrix"
        M = np.linalg.inv(INV) @ rot_about(UP, deg, c) @ INV @ M
        n["matrix"] = M.T.ravel().tolist(); left.discard(n["name"])
    assert not left, f"glb 里没有这些物体: {sorted(left)}"
    _write(dst, j, rest)


def _sha(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


def source(R1):
    """这一步该从哪个 glb 转: 07_scene.glb 还是上次这一步写的那份 -> 原摆放是备份; 否则(组装新出的)先备份它。"""
    cur, bak, mk = f"{R1}/07_scene.glb", f"{R1}/{BACKUP}", f"{R1}/{MARK}"
    done = json.load(open(mk)).get("出_sha") if os.path.exists(mk) else None
    if not (done and os.path.exists(bak) and _sha(cur) == done):
        shutil.copyfile(cur, bak)
    return bak


def mark_done(R1, info=None):
    d = dict(info or {}); d["出_sha"] = _sha(f"{R1}/07_scene.glb")
    json.dump(d, open(f"{R1}/{MARK}", "w"), ensure_ascii=False, indent=1)



def visible_frac(mask, sil):
    """露出 = 抠图面积 / 按原摆放渲染出的整个轮廓面积(只渲这一个物体, 不管谁挡它)。
    这一帧物体被挡得多, FP 能比的东西就少: office table_7 只露 29%, FP 把它转错了 87°、伸进人 2 走的路里。
    轮廓为空(物体不在画面里)当 0。超过 1 是摆放和照片有偏差, 不算问题。"""
    n = int(np.count_nonzero(sil))
    return float(np.count_nonzero(mask)) / n if n else 0.0
