"""第三步(sam3d 环境): 拟合结果 -> 带骨骼的 glb(每人一套骨架 + 蒙皮网格 + 关节动画)。
关节的全局变换直接用 MHR 的骨架状态(平移 + 四元数 + 均匀缩放), 逆绑定矩阵用 MHR 自带的 ——
和 MHR 自己的线性蒙皮逐项一致(验证过: 差 0.0002 cm), 所以播放出来就是拟合出来的网格。
用法: python export_glb.py <拟合目录> <输出glb> [人id,...]"""
import glob, os, sys
import numpy as np, pygltflib as gl, trimesh
from scipy.spatial.transform import Rotation


def tqs_to_mat(x, unit=0.01):
    """(...,8) [t(cm) q(xyzw) s] -> (...,4,4), 米"""
    M = np.zeros(x.shape[:-1] + (4, 4))
    M[..., :3, :3] = Rotation.from_quat(x[..., 3:7].reshape(-1, 4)).as_matrix().reshape(x.shape[:-1] + (3, 3)) * x[..., 7:8, None]
    M[..., :3, 3] = x[..., :3] * unit
    M[..., 3, 3] = 1.0
    return M


def local_trs(G, parents):
    """全局 (...,J,4,4) -> 局部的 平移/四元数(xyzw)/缩放。缩放是均匀的, 所以 TRS 分解是精确的。"""
    L = G.copy()
    for j, p in enumerate(parents):
        if p >= 0:
            L[..., j, :, :] = np.linalg.inv(G[..., p, :, :]) @ G[..., j, :, :]
    s = np.linalg.norm(L[..., :3, 0], axis=-1)
    R = L[..., :3, :3] / s[..., None, None]
    q = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat().reshape(R.shape[:-2] + (4,))
    return L[..., :3, 3], q, s


class Buf:
    def __init__(self):
        self.data = bytearray(); self.views = []; self.acc = []

    def add(self, arr, typ, comp, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        while len(self.data) % 4: self.data.append(0)
        off = len(self.data); self.data += arr.tobytes()
        self.views.append(gl.BufferView(buffer=0, byteOffset=off, byteLength=arr.nbytes, target=target))
        n = arr.shape[0]
        a = gl.Accessor(bufferView=len(self.views) - 1, componentType=comp, count=n, type=typ)
        if minmax:
            a.min = arr.reshape(n, -1).min(0).tolist(); a.max = arr.reshape(n, -1).max(0).tolist()
        self.acc.append(a)
        return len(self.acc) - 1


def build(files, out):
    g = gl.GLTF2(scenes=[gl.Scene(nodes=[])], scene=0, nodes=[], meshes=[], skins=[], animations=[], materials=[])
    b = Buf()
    anim = gl.Animation(name="mocap", samplers=[], channels=[])
    cols = [(0.9, 0.47, 0.16), (0.16, 0.63, 0.9), (0.24, 0.75, 0.35), (0.78, 0.27, 0.63), (0.86, 0.78, 0.16), (0.47, 0.35, 0.86), (0.16, 0.78, 0.78)]
    for k, f in enumerate(files):
        r = np.load(f); pid = os.path.basename(f)[1:-4]
        names, parents = [str(x) for x in r["names"]], r["parents"].astype(int)
        J = len(names)
        # 按整数帧率打时间戳: Blender 的 glTF 导入只认整数帧率, 这样视频第 n 帧正好落在 Blender 第 n 帧
        # (soccer 21.07 fps -> 21, 播放慢 0.3%)
        fps = float(round(float(r["fps"]))); times = (r["vf"] / fps).astype(np.float32)
        G = tqs_to_mat(r["skel"])                                   # (T,J,4,4)
        IB = tqs_to_mat(r["inv_bind"])                              # (J,4,4)
        Bind = np.linalg.inv(IB)
        t0, q0, s0 = local_trs(Bind, parents)
        t, q, s = local_trs(G, parents)
        for i in range(1, len(q)):                                  # q 和 -q 同一个朝向, 接成连续的再插值
            flip = (q[i] * q[i - 1]).sum(-1) < 0
            q[i][flip] *= -1
        base = len(g.nodes)
        for j in range(J):
            ch = [base + c for c in range(J) if parents[c] == j]
            g.nodes.append(gl.Node(name=f"P{pid}_{names[j]}", translation=t0[j].tolist(), rotation=q0[j].tolist(),
                                   scale=[float(s0[j])] * 3, children=ch or None))
        root = base + int(np.nonzero(parents < 0)[0][0])
        # 网格: 这个人的静止体型(模板 + 体型参数), 米
        rest = (r["rest"] / 100.0).astype(np.float32); faces = r["faces"].astype(np.uint32)
        nrm = trimesh.Trimesh(rest, faces, process=False).vertex_normals.astype(np.float32)
        jw = np.zeros((len(rest), 4), np.uint16); ww = np.zeros((len(rest), 4), np.float32); cnt = np.zeros(len(rest), int)
        for v, j, w in zip(r["sk_v"], r["sk_j"], r["sk_w"]):
            jw[v, cnt[v]] = j; ww[v, cnt[v]] = w; cnt[v] += 1
        ww /= ww.sum(1, keepdims=True)
        pos = b.add(rest, "VEC3", gl.FLOAT, gl.ARRAY_BUFFER, True)
        nor = b.add(nrm, "VEC3", gl.FLOAT, gl.ARRAY_BUFFER)
        jo = b.add(jw, "VEC4", gl.UNSIGNED_SHORT, gl.ARRAY_BUFFER)
        we = b.add(ww, "VEC4", gl.FLOAT, gl.ARRAY_BUFFER)
        ind = b.add(faces.reshape(-1), "SCALAR", gl.UNSIGNED_INT, gl.ELEMENT_ARRAY_BUFFER)
        g.materials.append(gl.Material(name=f"P{pid}", pbrMetallicRoughness=gl.PbrMetallicRoughness(
            baseColorFactor=list(cols[k % len(cols)]) + [1.0], metallicFactor=0.0, roughnessFactor=0.7)))
        g.meshes.append(gl.Mesh(name=f"P{pid}_body", primitives=[gl.Primitive(
            attributes=gl.Attributes(POSITION=pos, NORMAL=nor, JOINTS_0=jo, WEIGHTS_0=we), indices=ind,
            material=len(g.materials) - 1)]))
        ibm = b.add(np.stack([m.T for m in IB]).astype(np.float32).reshape(J, 16), "MAT4", gl.FLOAT)
        g.skins.append(gl.Skin(name=f"P{pid}", joints=list(range(base, base + J)), inverseBindMatrices=ibm, skeleton=root))
        g.nodes.append(gl.Node(name=f"P{pid}_body", mesh=len(g.meshes) - 1, skin=len(g.skins) - 1))
        g.scenes[0].nodes += [root, len(g.nodes) - 1]
        tin = b.add(times, "SCALAR", gl.FLOAT, minmax=True)
        for j in range(J):
            for path, val, typ in (("translation", t[:, j], "VEC3"), ("rotation", q[:, j], "VEC4"),
                                   ("scale", np.repeat(s[:, j, None], 3, 1), "VEC3")):
                o = b.add(val.astype(np.float32), typ, gl.FLOAT)
                anim.samplers.append(gl.AnimationSampler(input=tin, output=o, interpolation="LINEAR"))
                anim.channels.append(gl.AnimationChannel(sampler=len(anim.samplers) - 1,
                                                         target=gl.AnimationChannelTarget(node=base + j, path=path)))
        lab = str(r["err_label"]) if "err_label" in r.files else "拟合误差"       # 一手数据版没有拟合, 这个数是"姿势修正项"带来的差
        print(f"人 {pid}: {J} 个关节, {len(times)} 帧 ({times[0]:.2f}~{times[-1]:.2f} s), {lab}平均 {r['err_mean'].mean():.2f} cm")
    g.animations.append(anim)
    g.bufferViews = b.views; g.accessors = b.acc
    g.buffers = [gl.Buffer(byteLength=len(b.data))]
    g.set_binary_blob(bytes(b.data))
    g.save_binary(out)
    print("->", out, f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    d, out = sys.argv[1], sys.argv[2]
    want = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    fs = sorted(glob.glob(f"{d}/p*.npz"), key=lambda x: int(os.path.basename(x)[1:-4]))
    build([f for f in fs if not want or os.path.basename(f)[1:-4] in want], out)
