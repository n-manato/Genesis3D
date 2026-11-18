#!/usr/bin/env python3
"""RBF 形状転写 — テンプレート素体を、識別形状に合わせて変形する。

`game_avatar`（docs/08）の中核。写真ごとにメッシュを起こすのをやめ、**1体のテンプレート
素体の頂点を動かすだけ**にする。トポロジ・UV・ボーン・ウェイト・モーフ・マテリアルは
一切変わらないので、テンプレートで一度作り込んだ品質がそのまま出力の品質になる。

手口は MeInGame (AAAI 2021) §3.3 と同じ:

  1. ICT-FaceKit の識別形状とテンプレートのあいだに **68点のランドマーク対応**を張る
  2. テンプレート側のランドマークを放射基底関数の中心 c_i に置く
  3. 対応点で「そこがどれだけ動くべきか」を満たす重み w_i を最小二乗で解く
  4. 全頂点 x について f(x) を評価し、x + f(x) へ動かす

**Blender を経由しない。** `blender/vrm_finalize.py` の冒頭が書いているとおり、VRM の
再インポート・再エクスポートはテクスチャを再エンコードして個体を別物にしうる。RBF 変形は
POSITION accessor の書き換えでしかないので、glTF の JSON/BIN を直接いじる。
結果として spring bone・MToon・node constraint・テクスチャは「保たれる」のではなく
**一度も触られない**。

基底関数は phi(r) = r（3次元の biharmonic）に一次多項式を足したもの。3次元ではこれが
薄板スプラインに相当し、**中心から離れたところで発散せず線形に落ち着く**。ガウシアンだと
台の外で変位がゼロに戻り、首や後頭部に段差が出る。

モーフ差分は d' = d + f(x + d) - f(x) で送る。基準形状が動けば表情の差分も動く。
f を2回評価するだけなので、ヤコビアンによる一次近似より厳密で、しかも同じくらい安い。

法線はヤコビアンの逆転置で送る。f は位置の関数なので、プリミティブ境界で重複している
頂点には**同じ変位と同じヤコビアンが入る**。継ぎ目は構造的に開かない。

使い方:
    python3 -m preprocessing.shape_transfer \\
        --template <template.vrm> --pairs <pairs.json> --out <deformed.vrm> \\
        --report <report.json>
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blender.vrm_finalize import read_glb, write_glb  # noqa: E402

NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
FLOAT = 5126

# 変形させるメッシュの既定。Body と Hair も動かさないと、頭だけ変わって首から下が
# 取り残される。RBF は多項式項のおかげで遠方でも滑らかに落ちるので、全部入れてよい。
DEFAULT_MESHES = ("Face", "Body", "Hair")


# --------------------------------------------------------------------- accessor

def read_accessor(doc: dict, blob: bytes, idx: int) -> np.ndarray | None:
    """accessor を (count, ncomp) の float32 配列で読む。byteStride を尊重する。

    sparse は返さない（POSITION/NORMAL が sparse なことは無い）。
    """
    acc = doc["accessors"][idx]
    if acc.get("bufferView") is None or acc["componentType"] != FLOAT:
        return None
    nc = NCOMP[acc["type"]]
    bv = doc["bufferViews"][acc["bufferView"]]
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or nc * 4
    count = acc["count"]
    if stride == nc * 4:
        flat = np.frombuffer(blob, dtype="<f4", count=count * nc, offset=base)
        return flat.reshape(count, nc).astype(np.float64)
    # インターリーブされている場合は1頂点ずつ拾う
    out = np.empty((count, nc), dtype=np.float64)
    for i in range(count):
        out[i] = struct.unpack_from(f"<{nc}f", blob, base + i * stride)
    return out


def read_sparse(doc: dict, blob: bytes, idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    """sparse accessor を (indices, values) で読む。密なら None。

    モーフ差分は VRoid の書き出しでほぼ sparse になっている（`Face` の 2286頂点の
    プリミティブで非ゼロが 223個など）。密に展開すると 57本 × 3メッシュで数MB増えるので、
    **非ゼロの位置はそのままに、値だけ差し替える**。
    """
    acc = doc["accessors"][idx]
    sp = acc.get("sparse")
    if not sp:
        return None
    nc = NCOMP[acc["type"]]
    n = sp["count"]
    ibv = doc["bufferViews"][sp["indices"]["bufferView"]]
    ioff = ibv.get("byteOffset", 0) + sp["indices"].get("byteOffset", 0)
    itype = {5121: "<u1", 5123: "<u2", 5125: "<u4"}[sp["indices"]["componentType"]]
    ind = np.frombuffer(blob, dtype=itype, count=n, offset=ioff).astype(np.int64)
    vbv = doc["bufferViews"][sp["values"]["bufferView"]]
    voff = vbv.get("byteOffset", 0) + sp["values"].get("byteOffset", 0)
    val = np.frombuffer(blob, dtype="<f4", count=n * nc, offset=voff)
    return ind, val.reshape(n, nc).astype(np.float64)


def write_sparse_values(doc: dict, out: bytearray, idx: int, values: np.ndarray) -> None:
    """sparse の値だけを同じ長さで上書きする。indices も bufferView も動かさない。"""
    sp = doc["accessors"][idx]["sparse"]
    vbv = doc["bufferViews"][sp["values"]["bufferView"]]
    off = vbv.get("byteOffset", 0) + sp["values"].get("byteOffset", 0)
    payload = np.ascontiguousarray(values, dtype="<f4").tobytes()
    out[off:off + len(payload)] = payload


def base_positions(doc: dict, blob: bytes, prim: dict) -> np.ndarray | None:
    return read_accessor(doc, blob, prim["attributes"]["POSITION"])


def append_accessor(doc: dict, out: bytearray, idx: int, data: np.ndarray) -> None:
    """変形後の値を BIN 末尾に密で追記し、accessor をそこへ向け直す。

    元の bufferView は書き換えない。インターリーブされている場合に他の属性を壊さない
    ためと、`repack` が未参照になった領域を後で落とすため。
    """
    acc = doc["accessors"][idx]
    nc = NCOMP[acc["type"]]
    payload = np.ascontiguousarray(data, dtype="<f4").tobytes()
    offset = len(out)
    out += payload
    doc["bufferViews"].append(
        {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)})
    acc["bufferView"] = len(doc["bufferViews"]) - 1
    acc.pop("byteOffset", None)
    if acc["type"] in ("VEC3", "VEC2", "SCALAR"):
        acc["min"] = [float(v) for v in data.min(axis=0)]
        acc["max"] = [float(v) for v in data.max(axis=0)]
    else:
        # MAT4（inverseBindMatrices）に min/max は意味が無い。仕様上も任意。
        acc.pop("min", None)
        acc.pop("max", None)


# -------------------------------------------------------------------------- RBF

class RBF:
    """phi(r) = r の放射基底＋一次多項式による3次元の変位場。

    [[K  P]  [w]   [d]
     [P' 0]] [a] = [0]

    K_ij = ||c_i - c_j||、P = [1 x y z]。d は各中心で実現したい変位。
    多項式項があるので、対応点が純粋な相似変換で説明できる場合は w がゼロに近づき、
    外挿もその相似変換に従う。
    """

    def __init__(self, centers: np.ndarray, deltas: np.ndarray, smooth: float = 0.0):
        n = len(centers)
        d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        # 平滑化項。対応点にノイズがあるとき、厳密補間をやめて近似に落とす。
        # MeInGame は最小二乗と書いているが、実装上はこの正則化が同じ役割をする。
        if smooth > 0:
            d = d + np.eye(n) * smooth * d.max()
        p = np.hstack([np.ones((n, 1)), centers])
        top = np.hstack([d, p])
        bot = np.hstack([p.T, np.zeros((4, 4))])
        a = np.vstack([top, bot])
        b = np.vstack([deltas, np.zeros((4, 3))])
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        self.centers = centers
        self.w = sol[:n]          # (n, 3)
        self.poly = sol[n:]       # (4, 3) — 定数項と線形項
        self.residual = float(np.abs(self.apply(centers) - deltas).max())

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """変位 f(x) を返す（x そのものではない）。"""
        r = np.linalg.norm(pts[:, None, :] - self.centers[None, :, :], axis=2)
        return r @ self.w + self.poly[0] + pts @ self.poly[1:]

    def jacobian(self, pts: np.ndarray) -> np.ndarray:
        """d f / d x を (N, 3, 3) で返す。法線を送るのに使う。

        phi(r) = r なので d r / d x = (x - c) / r。r=0 の中心そのものでは
        微分が定義されないので、その点だけ多項式項だけを使う。
        """
        diff = pts[:, None, :] - self.centers[None, :, :]      # (N, C, 3)
        r = np.linalg.norm(diff, axis=2)                        # (N, C)
        safe = np.where(r > 1e-12, r, 1.0)
        unit = diff / safe[:, :, None]
        unit[r <= 1e-12] = 0.0
        # J[n] = sum_c outer(w[c], unit[n, c]) + poly_linear^T
        j = np.einsum("cd,nce->nde", self.w, unit)
        return j + self.poly[1:].T[None, :, :]


# ------------------------------------------------------- 局所アフィン（変形勾配）

def read_indices(doc: dict, blob: bytes, prim: dict) -> np.ndarray | None:
    if "indices" not in prim:
        return None
    acc = doc["accessors"][prim["indices"]]
    bv = doc["bufferViews"][acc["bufferView"]]
    off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    dt = {5121: "<u1", 5123: "<u2", 5125: "<u4"}[acc["componentType"]]
    return np.frombuffer(blob, dtype=dt, count=acc["count"],
                         offset=off).reshape(-1, 3).astype(np.int64)


def welded_topology(doc: dict, blob: bytes, mesh_names: set[str]):
    """変形対象の全プリミティブを位置で溶接して一輪近傍を作る。

    **プリミティブ単位で一輪を作ってはいけない。** UV の切れ目で頂点が重複しており、
    片側だけの近傍から局所アフィンを解くと継ぎ目の両側で違う値になり、モーフが割れる。
    """
    vs, fs, slices, off = [], [], [], 0
    for mesh in doc.get("meshes", []):
        name = mesh.get("name", "")
        if mesh_names and name not in mesh_names:
            continue
        for pi, prim in enumerate(mesh.get("primitives", [])):
            pos = base_positions(doc, blob, prim)
            if pos is None:
                continue
            tri = read_indices(doc, blob, prim)
            slices.append((name, pi, off, len(pos)))
            vs.append(pos)
            if tri is not None:
                fs.append(tri + off)
            off += len(pos)
    if not vs:
        return None
    v = np.vstack(vs)
    f = np.vstack(fs) if fs else np.zeros((0, 3), dtype=np.int64)
    _, first, inv = np.unique(np.round(v, 6), axis=0,
                              return_index=True, return_inverse=True)
    return v, f, v[first], inv.ravel(), slices


def local_affines(wv: np.ndarray, wf: np.ndarray, wvp: np.ndarray,
                  eps: float = 1e-3, clamp: float = 1.2) -> np.ndarray:
    """一輪近傍から頂点ごとの変形勾配 A を解く（Sumner & Popovic の変形転写）。

    A_i = argmin_A  sum_j | A (x_j - x_i) - (x'_j - x'_i) |^2
        = (S'  + lam I)(S + lam I)^{-1},   S = sum_j e e^T,  S' = sum_j e' e^T

    **なぜ大域 RBF のヤコビアンを使わないか。** phi(r)=r の厳密補間は中心が近いと
    場が振動し、(I+J) の特異値が最大 91.6 / 最小 0.0 になった（docs/09 §6.7）。
    位置は補間の目標なので正確に出るが、ヤコビアンを通るモーフ差分と法線が壊れ、
    モーフの振幅が最大 2.94倍に膨らんだ。

    局所アフィンはメッシュの近傍だけを見るので、中心の配置に依存しない。
    lam は近傍が平面的・退化しているときに A を恒等へ落とすための正則化。
    """
    n = len(wv)
    e = np.vstack([wv[wf[:, 1]] - wv[wf[:, 0]],
                   wv[wf[:, 2]] - wv[wf[:, 1]],
                   wv[wf[:, 0]] - wv[wf[:, 2]]])
    ep = np.vstack([wvp[wf[:, 1]] - wvp[wf[:, 0]],
                    wvp[wf[:, 2]] - wvp[wf[:, 1]],
                    wvp[wf[:, 0]] - wvp[wf[:, 2]]])
    idx = np.concatenate([wf[:, 0], wf[:, 1], wf[:, 2]])
    jdx = np.concatenate([wf[:, 1], wf[:, 2], wf[:, 0]])

    s_ee = np.zeros((n, 3, 3))
    s_pe = np.zeros((n, 3, 3))
    oe = np.einsum("ki,kj->kij", e, e)
    op = np.einsum("ki,kj->kij", ep, e)
    for src in (idx, jdx):                 # 辺は両端に効く（e と -e で外積は同じ）
        np.add.at(s_ee, src, oe)
        np.add.at(s_pe, src, op)

    tr = np.trace(s_ee, axis1=1, axis2=2) / 3.0
    lam = np.maximum(eps * tr, 1e-12)[:, None, None] * np.eye(3)
    a = np.matmul(s_pe + lam, np.linalg.inv(s_ee + lam))
    # 近傍が無い孤立頂点は恒等に。
    isolated = np.trace(s_ee, axis1=1, axis2=2) <= 0
    a[isolated] = np.eye(3)

    # **特異値をクランプする。** 目的はモーフの振幅を抑えることなので、
    # 局所の伸縮に上限を置くのが直接的。既定 1.5 は実測で決めた
    # （全175モーフ中、差分が有意な98個の振幅比。docs/09 §6.7）:
    #
    #   clamp   比 中央   90%     最大   1.2倍超
    #    2.0     1.366   1.803   1.990     61個
    #    1.5     1.141   1.393   1.494     39個
    #    1.3     1.073   1.259   1.299     18個
    #    1.2     1.055   1.173   1.199      0個  ← 既定
    #    1.1     1.017   1.085   1.100      0個
    #
    # **どのクランプでも 0.8倍を下回るモーフは無い**（痩せる方向には壊れない）。
    # 1.2 で全モーフが ±20% に収まる。1.1 まで締めると本当の局所変形も潰す。
    #
    # ラップは顔を平均 2.5mm しか動かさないので、本当の局所伸縮が 1.5倍を
    # 超えることはほぼ無い。超える分は振動か退化した一輪と見なして潰す。
    if clamp and clamp > 1.0:
        u, sv, vt = np.linalg.svd(a)
        sv = np.clip(sv, 1.0 / clamp, clamp)
        a = np.matmul(u * sv[:, None, :], vt)
    return a


# ------------------------------------------------------------------- 骨も動かす

def node_matrices(doc: dict):
    """各ノードのローカル 4x4 と親、そして幅優先の順序を返す。"""
    nodes = doc.get("nodes", [])
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []) or []:
            parent[c] = i

    def local(n):
        if "matrix" in n:
            return np.asarray(n["matrix"], dtype=np.float64).reshape(4, 4).T
        m = np.eye(4)
        if "rotation" in n:
            x, y, z, w = n["rotation"]
            m[:3, :3] = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        if "scale" in n:
            m[:3, :3] = m[:3, :3] @ np.diag(n["scale"])
        if "translation" in n:
            m[:3, 3] = n["translation"]
        return m

    order, seen = [], set()

    def visit(i):
        if i in seen:
            return
        p = parent.get(i)
        if p is not None:
            visit(p)
        seen.add(i)
        order.append(i)

    for i in range(len(nodes)):
        visit(i)
    return [local(n) for n in nodes], parent, order


def move_joints(doc: dict, out: bytearray, rbf, exclude=()) -> dict:
    """**骨も変位場で動かす。既定では呼ばれない（`--joints` で有効）。**

    **未解決の問題がある。** glTF のレベルでは正しく動く（下記の実測）が、
    Blender の glTF インポータを通すと**評価後のメッシュが 1.5958m → 1.7214m に
    伸びる**（生の POSITION は 1.5958m のまま）。倍率 1.0787 は体型変形の
    0.91777 の逆数に近い。下流の `project_face.py` はこの評価後メッシュで
    カメラの画角を決めて焼くので、顔が引き伸ばされて held-out ArcFace が
    0.6814 → 0.3922 に落ちた。IBM の並び（列優先）と符号は確認済みで、
    原因はまだ掴めていない。

    これが無いと、メッシュだけが変形してスケルトンが取り残される。実測
    （`jobs/gw_007`、身長 1.743m → 1.60m の体型変形）:

        ノードの translation の総和   924.8770 → 924.8737   ← ほぼ不変
        メッシュの実測身長            1.7433m  → 1.5997m

    骨の位置が約 8% ずれる。バインド姿勢は整合しているので静止時は正常に見え、
    だから長いあいだ気づかなかったが、回転の支点がずれるのでポーズを付けると
    変形がわずかに誤る。

    **inverseBindMatrices も同時に更新する。** glTF の skinning は
    `jointMatrix = globalTransform * inverseBindMatrix` なので、ノードだけ
    動かして IBM を据え置くと静止姿勢でメッシュが崩れる。新しい世界行列の
    逆行列を入れ直せば、静止姿勢は変形後の頂点そのままになる。
    """
    nodes = doc.get("nodes", [])
    if not nodes:
        return {"moved": 0}
    locals_, parent, order = node_matrices(doc)
    old_world = {}
    for i in order:
        p = parent.get(i)
        old_world[i] = (old_world[p] @ locals_[i]) if p is not None else locals_[i]

    # 世界位置を変位させ、その位置になるようローカルの translation を解き直す
    new_world, moved = {}, 0
    pos = np.array([old_world[i][:3, 3] for i in order])
    disp = rbf.apply(pos)
    dmap = {i: d for i, d in zip(order, disp)}
    # **肩の骨は動かさない。** 肘と手首は骨を体表面の中心へ動かすと改善したが
    # （剛体の歪み 手首 42.5→10.9%）、肩は**悪化した** —— KEHAI の待機姿勢で
    # 肩が内側に潰れた。Rocketbox のウェイトは腕の支点が肩の頂部にある前提で
    # 塗られていて、支点を 11cm 下げると肩の頂部ごと腕と一緒に回ってしまう。
    # 除外した骨は世界位置を据え置く（親が動くのでローカルは解き直す）。
    excluded = {i for i, n in enumerate(nodes) if n.get("name") in set(exclude)}
    for i in order:
        p = parent.get(i)
        want = old_world[i][:3, 3] + (0.0 if i in excluded else dmap[i])
        m = locals_[i].copy()
        if p is None:
            m[:3, 3] = want
        else:
            inv = np.linalg.inv(new_world[p])
            m[:3, 3] = (inv @ np.append(want, 1.0))[:3]
        new_world[i] = (new_world[p] @ m) if p is not None else m
        if i not in excluded and np.linalg.norm(dmap[i]) > 1e-9:
            moved += 1
        n = nodes[i]
        if "matrix" in n:
            n["matrix"] = [float(v) for v in m.T.reshape(-1)]
        else:
            n["translation"] = [float(v) for v in m[:3, 3]]

    # **同乗しているアニメーションを落とす。**
    #
    # Rocketbox のテンプレートは FBX 由来の `Bip01|Take 001|BaseLayer`（243チャンネル）
    # を持っている。骨を動かした時点でこのキーフレームは無効になる ——
    # というより、**体型変形をした時点で既に無効**（メッシュだけ縮んでいる）。
    #
    # これが「Blender だけが誤読する」の正体だった。**Blender はインポート時に
    # アニメーションを1フレーム目で適用**し、ノードの TRS を古い値で上書きする。
    # 結果 `oldWorld · IBM_new` になり、評価後のメッシュが 1.5958m → 1.7210m に
    # 伸びていた。three.js（KEHAI）は自動再生しないので正しく出ていた。
    dropped = len(doc.get("animations") or [])
    if dropped:
        doc.pop("animations", None)

    # skin の inverseBindMatrices を張り直す
    skins = 0
    for skin in doc.get("skins", []) or []:
        acc_i = skin.get("inverseBindMatrices")
        if acc_i is None:
            continue
        joints = skin.get("joints", [])
        ibm = np.stack([np.linalg.inv(new_world[j]) for j in joints])
        append_accessor(doc, out, acc_i,
                        ibm.transpose(0, 2, 1).reshape(len(joints), 16))
        skins += 1
    return {"moved": moved, "nodes": len(nodes), "skins_rebound": skins,
            "excluded": sorted(nodes[i].get("name") for i in excluded),
            "animations_dropped": dropped,
            "max_joint_displacement": float(np.linalg.norm(disp, axis=1).max())}


# ------------------------------------------------------------------------ 変形

def deform(doc: dict, blob: bytes, rbf: RBF, mesh_names: set[str],
           move_normals: bool = True, local: bool = True,
           clamp: float = 1.2, joints: bool = False,
           joints_exclude=()) -> tuple[bytes, dict]:
    """変形を適用する。

    `local=True` のとき、モーフ差分と法線は**一輪近傍から求めた局所アフィン**で送る。
    位置は変わらない（どちらのモードでも x + f(x)）。docs/09 §6.7 の理由により既定で有効。
    """
    out = bytearray(blob)
    aff, aff_stats = None, None
    if local:
        topo = welded_topology(doc, blob, mesh_names)
        if topo is not None:
            _, wfa, wv, winv, _ = topo
            wf = winv[wfa]
            wf = wf[(wf[:, 0] != wf[:, 1]) & (wf[:, 1] != wf[:, 2])
                    & (wf[:, 0] != wf[:, 2])]
            a_w = local_affines(wv, wf, wv + rbf.apply(wv), clamp=clamp)
            sv = np.linalg.svd(a_w, compute_uv=False)
            aff = (a_w, winv)
            aff_stats = {"welded_vertices": int(len(wv)), "clamp": clamp,
                         "singular_max": float(sv.max()),
                         "singular_min": float(sv.min()),
                         "singular_median": float(np.median(sv))}
    touched: list[dict] = []
    morphs_fixed = 0
    voff = 0
    for mesh in doc.get("meshes", []):
        name = mesh.get("name", "")
        if mesh_names and name not in mesh_names:
            continue
        for pi, prim in enumerate(mesh.get("primitives", [])):
            attrs = prim.get("attributes", {})
            pos_i = attrs.get("POSITION")
            if pos_i is None:
                continue
            pos = read_accessor(doc, out, pos_i)
            if pos is None:
                continue
            disp = rbf.apply(pos)
            a_p = None
            if aff is not None:
                a_w, winv = aff
                a_p = a_w[winv[voff:voff + len(pos)]]
            voff += len(pos)

            # モーフ差分を先に直す。**必ず変形前の pos を使う。**
            #
            # glTF が持つのは基準形状からの差分 d で、表情時の位置は x + d。
            # 基準が x + f(x) へ動いたあと、表情時は (x + d) + f(x + d) へ行くべきなので、
            #
            #     d' = (x + d + f(x + d)) - (x + f(x)) = d + f(x + d) - f(x)
            #
            # **d を足し忘れないこと。** 落とすと純粋な相似変換ですら
            # 差分が A·d ではなく (A-I)·d になり、表情の振幅が壊れる。
            # ヤコビアンによる一次近似ではなく f を2回評価する。同じくらい安い。
            for tgt in prim.get("targets") or []:
                acc_i = tgt.get("POSITION")
                if acc_i is None:
                    continue
                sp = read_sparse(doc, out, acc_i)
                if sp is not None:
                    ind, d = sp
                    if a_p is not None:
                        nd = np.einsum("kij,kj->ki", a_p[ind], d)
                    else:
                        base = pos[ind]
                        nd = d + rbf.apply(base + d) - rbf.apply(base)
                    write_sparse_values(doc, out, acc_i, nd)
                    morphs_fixed += 1
                else:
                    d = read_accessor(doc, out, acc_i)
                    if d is None:
                        continue
                    nd = (np.einsum("kij,kj->ki", a_p, d) if a_p is not None
                          else d + rbf.apply(pos + d) - disp)
                    append_accessor(doc, out, acc_i, nd)
                    morphs_fixed += 1

            append_accessor(doc, out, pos_i, pos + disp)

            nrm_i = attrs.get("NORMAL")
            if move_normals and nrm_i is not None:
                nrm = read_accessor(doc, out, nrm_i)
                if nrm is not None:
                    mats = a_p if a_p is not None else np.eye(3) + rbf.jacobian(pos)
                    # 法線は変形勾配の逆転置で送る。特異になった点は元のまま残す。
                    moved = np.empty_like(nrm)
                    for k in range(len(nrm)):
                        try:
                            moved[k] = np.linalg.inv(mats[k]).T @ nrm[k]
                        except np.linalg.LinAlgError:
                            moved[k] = nrm[k]
                    ln = np.linalg.norm(moved, axis=1, keepdims=True)
                    moved = np.where(ln > 1e-9, moved / np.maximum(ln, 1e-9), nrm)
                    append_accessor(doc, out, nrm_i, moved)

            touched.append({
                "mesh": name,
                "primitive": pi,
                "vertices": int(len(pos)),
                "max_displacement": float(np.linalg.norm(disp, axis=1).max()),
                "mean_displacement": float(np.linalg.norm(disp, axis=1).mean()),
            })
    joint_stats = move_joints(doc, out, rbf, joints_exclude) if joints else None
    stats = {
        "joints": joint_stats,
        "morph_mode": "local_affine" if aff is not None else "rbf_field",
        "local_affine": aff_stats,
        "primitives": len(touched),
        "morph_targets_corrected": morphs_fixed,
        "vertices": sum(t["vertices"] for t in touched),
        "max_displacement": max((t["max_displacement"] for t in touched), default=0.0),
        "detail": touched,
    }
    return bytes(out), stats


# -------------------------------------------------------------------------- CLI

def load_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """対応点を読む。

    形式:
      {"centers": [[x,y,z], ...],   # テンプレート側の現在位置（RBF の中心）
       "targets": [[x,y,z], ...]}   # そこが動くべき先
    """
    data = json.loads(path.read_text())
    centers = np.asarray(data["centers"], dtype=np.float64)
    targets = np.asarray(data["targets"], dtype=np.float64)
    if centers.shape != targets.shape or centers.ndim != 2 or centers.shape[1] != 3:
        raise SystemExit(
            f"{path}: centers と targets は同じ形の (N, 3) でなければならない "
            f"（{centers.shape} と {targets.shape}）")
    if len(centers) < 4:
        raise SystemExit(f"{path}: 一次多項式を解くのに最低4点必要（{len(centers)}点）")
    return centers, targets


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path, required=True, help="テンプレート VRM/GLB")
    ap.add_argument("--pairs", type=Path, required=True, help="対応点 JSON")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--meshes", nargs="*", default=list(DEFAULT_MESHES),
                    help="変形するメッシュ名。空にすると全メッシュ")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="平滑化。対応点にノイズがあるとき 0.01〜0.1 程度")
    ap.add_argument("--no-normals", action="store_true",
                    help="法線を送らない（デバッグ用）")
    ap.add_argument("--joints-only", action="store_true",
                    help="**頂点を触らず骨だけ**を変位場で動かす。Blender の工程を"
                         "全部終えた最後に掛けるための入口（docs/09 §21）")
    ap.add_argument("--joints-exclude", default="",
                    help="骨の移動から除外する骨名（カンマ区切り）。肩の鎖を除外する")
    ap.add_argument("--joints", action="store_true",
                    help="骨も変位場で動かす。**既定で切っている** —— glTF の"
                         "レベルでは正しく動くが、Blender の往復でメッシュが"
                         "1.60m→1.72m に伸びて下流の投影が壊れる（docs/09 §19）")
    ap.add_argument("--affine-clamp", type=float, default=1.2,
                    help="局所アフィンの特異値の上限（下限はその逆数）。"
                         "モーフの振幅の暴走をここで止める")
    ap.add_argument("--morph-field", dest="local", action="store_false",
                    help="モーフと法線を大域 RBF のヤコビアンで送る（旧挙動）。"
                         "中心が密なとき振幅が最大 2.94倍に膨らむ（docs/09 §6.7）")
    args = ap.parse_args()

    doc, blob = read_glb(args.template)
    centers, targets = load_pairs(args.pairs)
    rbf = RBF(centers, targets - centers, smooth=args.smooth)

    if args.joints_only:
        out = bytearray(blob)
        js = move_joints(doc, out, rbf,
                         [b for b in args.joints_exclude.split(",") if b])
        write_glb(args.out, doc, bytes(out))
        rep = {"template": str(args.template), "pairs": str(args.pairs),
               "out": str(args.out), "mode": "joints_only", "joints": js}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
        print(f"骨だけ移動: {js['moved']}/{js['nodes']}本  最大 "
              f"{js['max_joint_displacement']*1000:.1f}mm  "
              f"skin {js['skins_rebound']}個の IBM を張り直した"
              f"  アニメ {js['animations_dropped']}個を除去 → {args.out}")
        return

    blob, stats = deform(doc, blob, rbf, set(args.meshes),
                         move_normals=not args.no_normals, local=args.local,
                         clamp=args.affine_clamp, joints=args.joints,
                         joints_exclude=[b for b in args.joints_exclude.split(",") if b])
    write_glb(args.out, doc, blob)

    report = {
        "template": str(args.template),
        "pairs": str(args.pairs),
        "out": str(args.out),
        "centers": int(len(centers)),
        "smooth": args.smooth,
        # 対応点そのものでどれだけ誤差が残ったか。smooth=0 なら実質ゼロになる。
        "fit_residual": rbf.residual,
        "meshes": args.meshes,
        **stats,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    if stats.get("joints"):
        j = stats["joints"]
        print(f"骨 {j['moved']}/{j['nodes']}本を移動  最大 "
              f"{j['max_joint_displacement']*1000:.1f}mm  "
              f"skin {j['skins_rebound']}個の IBM を張り直した")
    print(f"対応点 {len(centers)}点   残差 {rbf.residual:.2e}")
    for t in stats["detail"]:
        print(f"  {t['mesh']:6s}[{t['primitive']}] {t['vertices']:6,d}頂点   "
              f"変位 平均 {t['mean_displacement']:.4f} / 最大 {t['max_displacement']:.4f}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
