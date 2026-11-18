#!/usr/bin/env python3
"""複数視点の写真から 68点の**3D座標**を三角測量で出す。

## なぜこれが要るか

前回の失敗の核心は「写真を他人の形をした頭に貼った」ことで、そうなった原因は
**単一写真では奥行きが観測できない**という原理的な壁だった。穴を埋める手段は二つしかない:

  1. 生成モデルに幻視させる（TRELLIS で held-out ArcFace 0.78。ただし今回は方針として使わない）
  2. PCA の事前分布に頼る（実測上限 0.39〜0.46。前回はこれで 0.455 だった）

**三角測量は三つめの道で、奥行きを事前分布ではなくデータから取る。** 生成ではないので
今回の方針に反しない。手持ちの `subject_a/subject_face_yaw+32_600.jpg` と `yaw-25` で
68点が検出できることは確認済み（顔 70px / 64px）。

## 解き方

未知数は 3D の 68点 X と、視点ごとの弱透視カメラ (s, R, t2)。交互に解く。

  1. X を固定して各視点のカメラを解く（`ict_photo_fit.weak_perspective`）
  2. カメラを固定して各点を独立に解く。視点 v について
     `uv_v ≈ s_v R_v[:2] X + t_v` なので、3視点なら 6式・3未知数の最小二乗

**ゲージの固定が要る。** 相似変換の分だけ解が不定なので、初期値と最終出力を
ICT-FaceKit の平均顔の 68点に procrustes で合わせる。これで cm 単位の座標系に乗る。

**視点ごとに重みを付ける。** 横向きの写真は顔が 64〜70px しかなく、ランドマークの
精度が正面（395px）と桁で違う。顔の幅で重み付けしないと、低解像度の視点が
奥行きを支配して形が崩れる。

使い方:
    python -m preprocessing.multiview_landmarks \\
        --view 0:subject_a/subject_face_downtown_restored.png \\
        --view +32:subject_a/subject_face_yaw+32_600.jpg \\
        --view -25:subject_a/subject_face_yaw-25_600.jpg \\
        --out jobs/x/scores/lm3d.json [--review jobs/x/renders/lm3d.png]

`--view` の数値は方位角のヒント（度）。カメラは推定するので厳密でなくてよいが、
初期化の向きを決めるのに使う。正面を最初に置くこと。

**負の方位角は `--view=-25:path` と `=` で渡すこと。** 空白で区切ると argparse が
先頭の `-` をオプションと解釈して落ちる。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.ict_fit import ICT_DIR, load_identity_basis          # noqa: E402
from preprocessing.ict_photo_fit import project, weak_perspective       # noqa: E402

# 点の位置に対する正則化。平均顔からの逸脱をこの重みで押さえる。
# 横向きが低解像度なので、これが無いと奥行きが暴れる。
SHAPE_PRIOR = 0.05


def detect(path: Path):
    """68点（2D）と顔の幅を返す。幅は視点の重みに使う。"""
    from PIL import Image
    from insightface.app import FaceAnalysis

    if not hasattr(detect, "_app"):
        app = FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(1024, 1024))
        detect._app = app
    arr = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1]
    faces = detect._app.get(arr)
    if not faces:
        return None, 0.0
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    lm = getattr(f, "landmark_3d_68", None)
    if lm is None:
        return None, 0.0
    return np.asarray(lm, dtype=np.float64)[:, :2], float(f.bbox[2] - f.bbox[0])


def init_rotation(azim_deg: float) -> np.ndarray:
    """方位角のヒントから初期回転。Y軸まわり。"""
    a = np.radians(azim_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def solve_points(uvs, cams, weights, mean_shape, prior=SHAPE_PRIOR):
    """カメラを固定して 68点を独立に解く。

    点 i について、視点 v ごとに 2式:  s_v R_v[:2] X_i + t_v = uv_{v,i}
    3視点で 6式・3未知数。平均顔への正則化を 3式ぶん足して安定させる。
    """
    n = len(mean_shape)
    out = np.empty((n, 3))
    for i in range(n):
        rows, rhs, w = [], [], []
        for (s, R, t2), uv, wt in zip(cams, uvs, weights):
            rows.append(s * R[:2])
            rhs.append(uv[i] - t2)
            w += [wt, wt]
        # 平均顔への引き戻し。スケールを合わせるため視点の重みの総和に比例させる。
        lam = prior * float(np.sum(weights))
        rows.append(np.eye(3) * lam)
        rhs.append(mean_shape[i] * lam)
        w += [1.0, 1.0, 1.0]
        A = np.vstack(rows)
        b = np.concatenate(rhs)
        W = np.sqrt(np.asarray(w))[:, None]
        sol, *_ = np.linalg.lstsq(A * W, b * W[:, 0], rcond=None)
        out[i] = sol
    return out


def procrustes_to(src: np.ndarray, dst: np.ndarray):
    """src を dst に合わせる相似変換（ゲージ固定用）。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    A, B = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    scale = (S * np.array([1.0, 1.0, d])).sum() / (A ** 2).sum()
    return scale, R, mu_d - scale * (R @ mu_s)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", action="append", required=True,
                    metavar="AZIM:PATH",
                    help="方位角(度):画像。正面(0)を最初に。最低2視点、3視点推奨")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--prior", type=float, default=SHAPE_PRIOR,
                    help="平均顔への正則化。横向きが低解像度なとき上げる")
    ap.add_argument("--cache", default="vendor/ICT-FaceKit/identity_basis.npz")
    ap.add_argument("--review", type=Path, help="再投影を重ねた確認画像")
    args = ap.parse_args()

    specs = []
    for v in args.view:
        if ":" not in v:
            raise SystemExit(f"--view は AZIM:PATH の形（{v}）")
        a, p = v.split(":", 1)
        specs.append((float(a), Path(p)))
    if len(specs) < 2:
        raise SystemExit("2視点以上必要。1視点では奥行きが解けない")

    # ICT の平均顔の 68点。ゲージと事前分布の両方に使う（cm 単位）。
    neutral, _ = load_identity_basis(Path(args.cache))
    lm_idx = np.asarray(json.loads(
        (ICT_DIR / "vertex_indices.json").read_text())["idx_to_landmark_verts"])
    mean_shape = neutral[lm_idx].astype(np.float64)

    uvs, weights, views = [], [], []
    for azim, path in specs:
        uv, w = detect(path)
        if uv is None:
            print(f"警告: {path} で68点が取れない。この視点を外す", file=sys.stderr)
            continue
        uvs.append(uv)
        weights.append(w)
        views.append({"azimuth": azim, "path": str(path), "face_px": round(w, 1)})
    if len(uvs) < 2:
        raise SystemExit("68点が取れた視点が2つ未満")

    # 顔の幅で重み付け。正面 395px と横 64px では精度が桁で違う。
    wmax = max(weights)
    weights = [w / wmax for w in weights]
    for v, w in zip(views, weights):
        v["weight"] = round(w, 3)

    X = mean_shape.copy()
    hist = []
    cams = [(1.0, init_rotation(v["azimuth"]), np.zeros(2)) for v in views]
    for it in range(args.iters):
        cams = [weak_perspective(X, uv) for uv in uvs]
        X = solve_points(uvs, cams, weights, mean_shape, args.prior)
        # ゲージを毎回戻す。放っておくとスケールと向きが漂う。
        s, R, t = procrustes_to(X, mean_shape)
        X = (s * (R @ X.T)).T + t
        cams = [weak_perspective(X, uv) for uv in uvs]
        rms = [float(np.sqrt(((project(X, *c) - uv) ** 2).sum(1).mean()))
               for c, uv in zip(cams, uvs)]
        hist.append({"iter": it, "rms_px": [round(r, 3) for r in rms]})

    for v, c, uv, r in zip(views, cams, uvs, hist[-1]["rms_px"]):
        v["rms_px"] = r
        v["rms_over_face"] = round(r / max(v["face_px"], 1e-6), 4)

    # 平均顔からどれだけ離れたか。ここが 0 に近いなら三角測量が効いていない。
    dev = float(np.linalg.norm(X - mean_shape, axis=1).mean())
    # 奥行き方向だけの逸脱。三角測量が本当に効いているかの指標。
    dev_z = float(np.abs((X - mean_shape)[:, 2]).mean())

    out = {
        "views": views,
        "landmarks_3d": [[round(float(c), 6) for c in p] for p in X],
        "unit": "cm（ICT-FaceKit の平均顔に procrustes で合わせてある）",
        "prior": args.prior,
        "mean_deviation_cm": round(dev, 4),
        "mean_depth_deviation_cm": round(dev_z, 4),
        "cameras": [{"scale": round(float(s), 6),
                     "rotation": [[round(float(x), 6) for x in r] for r in R],
                     "translation_2d": [round(float(x), 3) for x in t]}
                    for s, R, t in cams],
        "history": hist,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    if args.review:
        from PIL import Image, ImageDraw
        tiles = []
        for (azim, path), c, uv in zip(specs, cams, uvs):
            im = Image.open(path).convert("RGB")
            d = ImageDraw.Draw(im)
            for x, y in uv:
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(90, 220, 90))
            for x, y in project(X, *c):
                d.ellipse([x - 2, y - 2, x + 2, y + 2], outline=(255, 80, 80))
            tiles.append(im.resize((420, int(420 * im.size[1] / im.size[0]))))
        h = max(t.size[1] for t in tiles)
        sheet = Image.new("RGB", (420 * len(tiles), h), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, (i * 420, 0))
        args.review.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.review)

    print(f"視点 {len(views)}  " + " / ".join(
        f"{v['azimuth']:+.0f}° 顔{v['face_px']:.0f}px 重み{v['weight']} "
        f"RMS {v['rms_px']}px" for v in views))
    print(f"平均顔からの逸脱 {dev:.3f}cm（奥行きだけ {dev_z:.3f}cm）")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
