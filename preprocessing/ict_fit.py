"""ICT-FaceKit の素体を、対象の頭部メッシュに合わせ込む。

いまの頭は TRELLIS が作る一枚の殻で、目・口・顎という部品を持たない。写真の表情が
焼き付き、まばたきも視線も口パクも原理的に不可能（docs/05 §36〜38）。ICT-FaceKit は
眼球・歯・舌・53個の ARKit 命名ブレンドシェイプを最初から持つので、ここを差し替える。

**写真からではなく、既にある頭に合わせる。** その頭は実写由来で本人の形を持っており、
正面 ArcFace 0.708 を出している。ICT の識別空間（100個の線形 PCA）を、その形に
最も近くなるよう解く。

手順:
  1. 対象を正射投影で正面から撮る（blender/ortho_front.py）
  2. 68点ランドマークを検出し、画素 →(X, Z) の一次式と光線で 3D に戻す
  3. 相似変換と識別係数を交互に解く（識別基底は線形なので最小二乗の閉形式）
  4. `idx_to_fitting_verts` の最近傍点で数回反復して精緻化

座標系: **ICT-FaceKit も glTF も「Y が上・+Z が前」で同じ規約**なので、変換は要らない。
Blender だけが Z 上なので、`ortho_front.py` の側でメタを glTF 座標に直して書き出している。
（一度ここを混ぜて、光線が別の軸へ飛んで顔が 0.2 ずれた。）

  python -m preprocessing.ict_fit --target <head.glb> --render <front.png> \
      --render-meta <front.json> --out <fit.json>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")

ICT_DIR = Path("vendor/ICT-FaceKit/FaceXModel")
PACK = "antelopev2"
N_IDENTITY = 100
# 識別係数の正則化。ICT の識別基底は 4000 スキャンの PCA なので、係数が大きいほど
# 「ありえない頭」になる。対象メッシュのノイズに引きずられないよう軽く押さえる。
#
# **スケール不変にすること。** ICT は cm 単位（頭の高さ 34）だが対象は正規化されて
# いて高さ 1.0 しかない。相似変換のスケールが 0.018 なので、基底を変換すると
# 行列の要素が桁で小さくなり、固定値の ridge を足すと係数がまったく動かなくなる
# （実測: alpha_norm 0.023 で反復しても誤差が変わらなかった）。
# trace(AᵀA)/n で正規化して、データの大きさに追随させる。
#
# 値は実測で決めた（docs/05 §41）。拘束が顔面にしか無いので、弱いと顔を合わせる
# ために頭蓋が後ろへ膨らむ。0.05 では奥行が対象メッシュ全体より深くなった。
# 1.0 なら表面距離は最良（0.00284）のまま係数が 55→21 に収まる。
RIDGE = 1.0


def parse_obj_verts(path: Path):
    import numpy as np
    return np.array([[float(x) for x in l.split()[1:4]]
                     for l in open(path, errors="ignore") if l.startswith("v ")])


def load_identity_basis(cache: Path | None = None):
    """素体と 100個の識別差分を読む。obj は1本 2.5MB × 100 なので結果を保存する。"""
    import numpy as np
    if cache and cache.exists():
        d = np.load(cache)
        return d["neutral"], d["basis"]
    neutral = parse_obj_verts(ICT_DIR / "generic_neutral_mesh.obj")
    basis = np.empty((N_IDENTITY, len(neutral), 3), dtype=np.float32)
    for k in range(N_IDENTITY):
        basis[k] = parse_obj_verts(ICT_DIR / f"identity{k:03d}.obj") - neutral
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, neutral=neutral, basis=basis)
    return neutral, basis


def detect_68(image):
    """レンダ画像から 68点ランドマークを取る。"""
    import numpy as np
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=PACK, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(1024, 1024))
    rgb = np.asarray(image.convert("RGB"))[:, :, ::-1]
    faces = app.get(rgb)
    if not faces:
        raise SystemExit("レンダ画像で顔を検出できません")
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    lm = getattr(f, "landmark_3d_68", None)
    if lm is None:
        raise SystemExit("68点ランドマークが取れません（antelopev2 の 1k3d68 が必要）")
    return np.asarray(lm)[:, :2]


def backproject(uv, meta, mesh):
    """画素 →(X, Y) を一次式で戻し、-Z 方向の光線でメッシュ表面の Z を求める。

    すべて glTF 座標（Y上・+Z前）。顔は +Z を向いているので、光線は前方から
    後方へ飛ばして**最初に当たった面＝顔の表面**を取る。
    """
    import numpy as np
    res = meta["res"]; span = meta["ortho_span"]
    cx, cy = meta["center_xy"]
    lo = np.array(meta["bounds_gltf"]["min"], dtype=float)
    hi = np.array(meta["bounds_gltf"]["max"], dtype=float)
    X = cx + (uv[:, 0] / res - 0.5) * span
    Y = cy - (uv[:, 1] / res - 0.5) * span
    origins = np.stack([X, Y, np.full_like(X, hi[2] + (hi[2] - lo[2]))], axis=1)
    dirs = np.tile(np.array([[0.0, 0.0, -1.0]]), (len(X), 1))
    loc, idx_ray, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=False)
    out = np.full((len(X), 3), np.nan)
    out[idx_ray] = loc
    return out


def procrustes(src, dst):
    """相似変換（等方スケール・回転・平行移動）を最小二乗で解く。"""
    import numpy as np
    mu_s, mu_d = src.mean(0), dst.mean(0)
    A, B = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    scale = (S * np.array([1.0, 1.0, d])).sum() / (A ** 2).sum()
    return scale, R, mu_d - scale * (R @ mu_s)


def apply_sim(v, s, R, t):
    return (s * (R @ v.T)).T + t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="合わせ込む先の頭部メッシュ (glb)")
    ap.add_argument("--render", required=True, help="ortho_front.py が撮った正面 png")
    ap.add_argument("--render-meta", required=True)
    ap.add_argument("--out", required=True, help="識別係数と変換を書き出す json")
    ap.add_argument("--out-mesh", help="合わせ込んだ素体を obj で書き出す（確認用）")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--ridge", type=float, default=RIDGE,
                    help=("識別係数の正則化。弱すぎると顔を合わせるために頭蓋が"
                          "膨らむ（拘束は顔面にしか無いため）"))
    ap.add_argument("--cache", default="vendor/ICT-FaceKit/identity_basis.npz")
    args = ap.parse_args()

    import numpy as np
    import trimesh
    from PIL import Image

    meta = json.loads(Path(args.render_meta).read_text())
    tgt = trimesh.load(args.target, process=False)
    if isinstance(tgt, trimesh.Scene):
        tgt = trimesh.util.concatenate(list(tgt.geometry.values()))

    uv = detect_68(Image.open(args.render))
    L_t = backproject(uv, meta, tgt)
    ok = ~np.isnan(L_t[:, 0])
    if ok.sum() < 40:
        raise SystemExit(f"逆投影できたランドマークが {int(ok.sum())} 点しかありません")

    neutral, basis = load_identity_basis(Path(args.cache))
    idx = json.loads((ICT_DIR / "vertex_indices.json").read_text())
    lm_idx = np.array(idx["idx_to_landmark_verts"])
    fit_idx = np.array(idx["idx_to_fitting_verts"])

    V0 = neutral                                    # ICT は既に glTF と同じ規約
    B = basis                                       # (100, N, 3)
    alpha = np.zeros(N_IDENTITY)

    def current():
        return V0 + np.tensordot(alpha, B, axes=(0, 0))

    # --- 1) ランドマークで相似変換と識別係数を交互に解く
    hist = []
    for it in range(args.iters):
        V = current()
        s, R, t = procrustes(V[lm_idx][ok], L_t[ok])
        # 変換後の残差を、変換した基底で説明する
        Bl = np.einsum("ij,knj->kni", s * R, B[:, lm_idx][:, ok])   # (100, m, 3)
        r = L_t[ok] - apply_sim(V[lm_idx][ok], s, R, t)
        A = Bl.reshape(N_IDENTITY, -1).T                            # (3m, 100)
        y = r.reshape(-1)
        AtA = A.T @ A
        lam = args.ridge * np.trace(AtA) / N_IDENTITY
        alpha = alpha + np.linalg.solve(AtA + lam * np.eye(N_IDENTITY), A.T @ y)
        err = float(np.linalg.norm(
            apply_sim(current()[lm_idx][ok], s, R, t) - L_t[ok], axis=1).mean())
        hist.append({"stage": "landmark", "iter": it, "mean_error": round(err, 5)})

    # --- 2) 顔全体の最近傍点で精緻化
    from trimesh.proximity import ProximityQuery
    pq = ProximityQuery(tgt)
    for it in range(args.iters):
        V = current()
        s, R, t = procrustes(V[lm_idx][ok], L_t[ok])
        Pw = apply_sim(V[fit_idx], s, R, t)
        close, dist, _ = pq.on_surface(Pw)
        # 極端に遠い対応は外す（髪や襟に吸われるのを防ぐ）
        keep = dist < np.percentile(dist, 85)
        Bf = np.einsum("ij,knj->kni", s * R, B[:, fit_idx][:, keep])
        r = close[keep] - Pw[keep]
        A = Bf.reshape(N_IDENTITY, -1).T
        AtA = A.T @ A
        lam = args.ridge * np.trace(AtA) / N_IDENTITY
        alpha = alpha + np.linalg.solve(AtA + lam * np.eye(N_IDENTITY), A.T @ r.reshape(-1))
        hist.append({"stage": "surface", "iter": it,
                     "mean_dist": round(float(dist[keep].mean()), 5),
                     "used": int(keep.sum())})

    V = current()
    s, R, t = procrustes(V[lm_idx][ok], L_t[ok])
    out = {
        "target": args.target,
        "landmarks_used": int(ok.sum()),
        "alpha": [round(float(a), 6) for a in alpha],
        "alpha_norm": round(float(np.linalg.norm(alpha)), 4),
        "similarity": {"scale": round(float(s), 6),
                       "rotation": [[round(float(x), 6) for x in row] for row in R],
                       "translation": [round(float(x), 6) for x in t]},
        "history": hist,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    if args.out_mesh:
        # 面は素体からそのまま持ってくる（位相は変わらない）。マテリアルの
        # 区切り（usemtl）も保つ — 後段で顔だけに投影するために要る。
        Vw = apply_sim(V, s, R, t)
        Path(args.out_mesh).parent.mkdir(parents=True, exist_ok=True)
        src = (ICT_DIR / "generic_neutral_mesh.obj").read_text(errors="ignore").splitlines()
        with open(args.out_mesh, "w") as f:
            f.write("mtllib ICTFaceModelMaterial.mtl\n")
            for p in Vw:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for line in src:
                if line.startswith(("vt ", "vn ", "f ", "usemtl ", "g ", "o ", "s ")):
                    f.write(line + "\n")
        out["out_mesh"] = args.out_mesh

    print("ICTFIT_OK " + json.dumps(
        {k: v for k, v in out.items() if k != "alpha"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
