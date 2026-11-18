#!/usr/bin/env python3
"""写真1枚から ICT-FaceKit の識別係数を解く。3Dメッシュも FLAME も要らない。

`preprocessing/ict_fit.py` は **既にある3Dの頭**（TRELLIS の出力）に ICT を合わせ込む。
`game_avatar`（docs/08）ではその3Dの頭を作らないので、写真から直接解く必要がある。
あちらのモードは実績があるので触らず、共通部品（識別基底・68点検出）だけ借りている。

MICA は使わない。`vendor/MICA` にチェックポイントはあるが FLAME2020 の
`generic_model.pkl` が無く、取得に MPI の登録と個別のライセンス同意が要る
（`envs/setup_face.sh:118-140` が明示的に拒否している）。ICT-FaceKit は MIT で
基底も手元に揃っている。

## 表情を同時に解いて捨てる

写真の人物が笑っていれば、その笑いが**同一性として吸収される**。実測で確かめた ——
口を開けて笑っている写真から素直に α を解き、デフォルメ係数 2.5 で誇張したところ、
テンプレートの口が開き、顔が非対称に歪んだ。

これは PokerFace-GAN (2020) が名指しで解いた問題そのもの（docs/08 §2）。
ICT-FaceKit は識別100本とは別に **表情53本**を持っているので、

    V = V_mean + Σ α_k · Id_k + Σ β_j · Expr_j

として α と β を同時に解き、**β を捨てて α だけを返す**。線形なので追加の
反復も学習も要らない。β は報告のために出力するが（何の表情として説明されたか）、
形状転写に渡すのは常に α だけ。`--no-expression` は分離しない診断用。

**リッジはブロックごとに正規化する。** 共通の `trace(AtA)/(n_id+n_ex)` を使うと、
表情基底の変位が識別より大きいぶん trace が表情に支配され、識別側が過剰に抑えられる。
実測では alpha_norm が 15.2（表情なし）から 2.25 まで潰れた。各ブロック自身の
対角平均で割ること。

`EXPR_RIDGE` は実測で 1.0 にした。弱くすると表情が識別を食う（0.1 で alpha_norm 10.5）、
強くすると笑いが識別へ漏れる。掃引の結果は docs/08 §10.7。

## 解き方

未知数は識別係数 α（100次元）、表情係数 β（53次元）、弱透視カメラ (s, R, t2)。
交互に解く。

  1. α を固定して (s, R, t2) を解く — 2D と 3D の対応から 2×3 の行列を最小二乗で出し、
     SVD で相似変換に射影する（弱透視 POS）
  2. カメラを固定して α を線形最小二乗で解く。正則化は **trace(AᵀA)/n で正規化した
     リッジ**（`ict_fit.py:181` と同じ）。画素とcmでは桁が4つ違うので、
     固定値のリッジだと係数がまったく動かないか暴れるかのどちらかになる

**奥行きはデータから決まらない。** 写真1枚では原理的に観測できないので、PCA の
事前分布だけが支えている。これは誤魔化さずにそう書いておく。係数は ±`CLAMP_SIGMA` で
一律に頭打ちにする — 外れると「ありえない頭」になり、テンプレートに転写したとき破綻する。

出力は `ict_fit.py` と同じ形（alpha / alpha_norm / similarity / history）に
`mode: "photo"` と再投影誤差を足したもの。

使い方:
    python -m preprocessing.ict_photo_fit --photo <顔写真.png> \\
        --out jobs/<id>/scores/fit.json [--review overlay.png]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.ict_fit import (ICT_DIR, N_IDENTITY, detect_68,  # noqa: E402
                                   load_identity_basis, parse_obj_verts)


def load_expression_basis(neutral, cache: Path | None = None):
    """53本の表情差分を読む。識別基底と同じくキャッシュする。"""
    names = json.loads((ICT_DIR / "vertex_indices.json").read_text())["expressions"]
    if cache and cache.exists():
        d = np.load(cache)
        if len(d["basis"]) == len(names):
            return d["basis"], names
    basis = np.empty((len(names), len(neutral), 3), dtype=np.float32)
    for k, n in enumerate(names):
        basis[k] = parse_obj_verts(ICT_DIR / f"{n}.obj") - neutral
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, basis=basis, names=np.array(names))
    return basis, names

RIDGE = 1.0
CLAMP_SIGMA = 3.0
# 表情側の正則化。**識別より強く**する。弱いと表情が識別を食い、強いと笑いが識別へ漏れる。
# 掃引の実測は docs/08 §10.7。
EXPR_RIDGE = 1.0
N_EXPR = 53


def weak_perspective(V: np.ndarray, uv: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """3D点 V を 2D点 uv に写す弱透視 (s, R, t2) を解く。

    まず `uv ≈ M V + t`（M は 2×3、拘束なし）を最小二乗で解き、M を SVD で
    「スケール × 回転の上2行」に射影する。直接回転を推定するより素直で、
    退化した配置でも壊れにくい。
    """
    mu_v, mu_uv = V.mean(0), uv.mean(0)
    A, B = V - mu_v, uv - mu_uv
    M, *_ = np.linalg.lstsq(A, B, rcond=None)     # (3, 2)
    M = M.T                                        # (2, 3)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R2 = U @ Vt                                    # 回転の上2行（正規直交）
    s = float(S.mean())
    r3 = np.cross(R2[0], R2[1])
    R = np.vstack([R2, r3])
    # 反射になっていたら3行目を反転して右手系に戻す
    if np.linalg.det(R) < 0:
        R[2] *= -1.0
    t2 = mu_uv - s * (R2 @ mu_v)
    return s, R, t2


def project(V: np.ndarray, s: float, R: np.ndarray, t2: np.ndarray) -> np.ndarray:
    return s * (V @ R[:2].T) + t2


def fit(uv, neutral, id_basis, ex_basis, lm_idx, iters=10,
        ridge=RIDGE, expr_ridge=EXPR_RIDGE, clamp_sigma=CLAMP_SIGMA):
    """識別 α と表情 β を同時に解く。返すのは α（β は捨てる用に別で返す）。"""
    n_id, n_ex = len(id_basis), len(ex_basis)
    alpha = np.zeros(n_id)
    beta = np.zeros(n_ex)
    # ICT の identity/expression は **1本が係数1.0で1モード分**の変位そのものなので、
    # 成分ごとの重み付けはせず一律で切る。
    #
    # 最初は基底ノルムで正規化した σ を使ったが、モード0が突出しているため σ が
    # 大半で 0.1 未満になり、**100個中89個が頭打ち**になって係数が動かなくなった。
    # `ict_fit.py` のメッシュ当て込みは正則化だけで alpha_norm 21 まで動く
    # （成分あたり RMS 約2.1）ので、一律 ±3 は十分な余裕がある。
    hist = []
    s, R, t2 = 1.0, np.eye(3), np.zeros(2)

    def build(a, b):
        return (neutral + np.tensordot(a, id_basis, axes=(0, 0))
                + np.tensordot(b, ex_basis, axes=(0, 0)))

    for it in range(iters):
        V = build(alpha, beta)
        s, R, t2 = weak_perspective(V[lm_idx], uv)

        # 識別と表情をまとめて1本の線形系にする。カメラを固定すれば両方とも線形。
        Ai = np.einsum("ij,knj->kni", s * R[:2], id_basis[:, lm_idx])
        blocks = [Ai.reshape(n_id, -1).T]
        if n_ex:                      # --no-expression では表情ブロックが空になる
            Ae = np.einsum("ij,knj->kni", s * R[:2], ex_basis[:, lm_idx])
            blocks.append(Ae.reshape(n_ex, -1).T)
        A = np.hstack(blocks)
        r = (uv - project(V[lm_idx], s, R, t2)).reshape(-1)

        AtA = A.T @ A
        # **スケール不変にすること。** 画素とcmでは桁が4つ違う（ict_fit.py:181 と同じ理由）。
        #
        # **ブロックごとに正規化する。** 共通の trace(AtA)/(n_id+n_ex) を使うと、
        # 表情基底の変位が識別より大きいぶん trace が表情に支配され、識別側が
        # 過剰に抑えられる。実測: alpha_norm が 15.2（表情なし）から 2.25 まで潰れ、
        # 再投影誤差も 2.90px → 5.44px と悪化した。各ブロック自身の trace で割る。
        d = np.diag(AtA)
        scale_id = float(d[:n_id].mean()) if n_id else 1.0
        scale_ex = float(d[n_id:].mean()) if n_ex else 1.0
        lam = np.concatenate([np.full(n_id, ridge * scale_id),
                              np.full(n_ex, expr_ridge * scale_ex)])
        step = np.linalg.solve(AtA + np.diag(lam), A.T @ r)
        alpha = alpha + step[:n_id]
        beta = beta + step[n_id:]

        clamped = int((np.abs(alpha) > clamp_sigma).sum())
        alpha = np.clip(alpha, -clamp_sigma, clamp_sigma)
        # 表情は 0..1 が本来の定義域。負や1超は「ありえない表情」になる。
        beta = np.clip(beta, 0.0, 1.0)

        V = build(alpha, beta)
        rms = float(np.sqrt(((project(V[lm_idx], s, R, t2) - uv) ** 2).sum(1).mean()))
        hist.append({"iter": it, "rms_px": round(rms, 4),
                     "alpha_norm": round(float(np.linalg.norm(alpha)), 4),
                     "beta_norm": round(float(np.linalg.norm(beta)), 4),
                     "clamped": clamped})
    return alpha, beta, (s, R, t2), hist


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--review", type=Path, help="検出点と再投影を重ねた確認画像")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--ridge", type=float, default=RIDGE)
    ap.add_argument("--expr-ridge", type=float, default=EXPR_RIDGE,
                    help="表情側の正則化。強いと表情が説明しきれず識別へ漏れる")
    ap.add_argument("--no-expression", action="store_true",
                    help="表情を解かない。**写真の笑いが同一性に焼き付く**ので診断用")
    ap.add_argument("--expr-cache", default="vendor/ICT-FaceKit/expression_basis.npz")
    ap.add_argument("--clamp-sigma", type=float, default=CLAMP_SIGMA)
    ap.add_argument("--cache", default="vendor/ICT-FaceKit/identity_basis.npz")
    args = ap.parse_args()

    from PIL import Image

    img = Image.open(args.photo)
    uv = detect_68(img)

    neutral, basis = load_identity_basis(Path(args.cache))
    idx = json.loads((ICT_DIR / "vertex_indices.json").read_text())
    lm_idx = np.array(idx["idx_to_landmark_verts"])

    id_basis = basis.astype(np.float64)
    if args.no_expression:
        ex_basis = np.zeros((0, len(neutral), 3))
        ex_names = []
    else:
        eb, ex_names = load_expression_basis(neutral, Path(args.expr_cache))
        ex_basis = eb.astype(np.float64)

    alpha, beta, (s, R, t2), hist = fit(
        uv, neutral, id_basis, ex_basis, lm_idx, iters=args.iters,
        ridge=args.ridge, expr_ridge=args.expr_ridge, clamp_sigma=args.clamp_sigma)

    # **識別だけを返す。** 表情は解いたうえで捨てる（PokerFace-GAN と同じ狙い）。
    V = neutral + np.tensordot(alpha, id_basis, axes=(0, 0))
    proj = project(V[lm_idx], s, R, t2)
    # 目の間の距離で正規化した誤差。画素数に依存しないので個体間で比べられる。
    iod = float(np.linalg.norm(uv[36] - uv[45]))

    out = {
        "mode": "photo",
        "photo": str(args.photo),
        "alpha": [round(float(a), 6) for a in alpha],
        "alpha_norm": round(float(np.linalg.norm(alpha)), 4),
        "similarity": {"scale": round(float(s), 8),
                       "rotation": [[round(float(x), 6) for x in row] for row in R],
                       "translation_2d": [round(float(x), 4) for x in t2]},
        "landmarks_used": int(len(uv)),
        "interocular_px": round(iod, 2),
        "rms_px": round(float(np.sqrt(((proj - uv) ** 2).sum(1).mean())), 4),
        "rms_over_interocular": round(
            float(np.sqrt(((proj - uv) ** 2).sum(1).mean())) / max(iod, 1e-6), 5),
        "note": "奥行きは写真1枚では観測できない。PCA の事前分布だけが支えている",
        "expression": {
            "solved": not args.no_expression,
            "beta_norm": round(float(np.linalg.norm(beta)), 4),
            # 何の表情として説明されたか。写真が笑っていれば口角と頬が上位に来る。
            "top": [{"name": ex_names[k], "weight": round(float(beta[k]), 4)}
                    for k in np.argsort(-np.abs(beta))[:8]] if len(beta) else [],
            "note": "解いたうえで捨てている。返す alpha には表情が入っていない",
        },
        "history": hist,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    if args.review:
        from PIL import ImageDraw
        im = img.convert("RGB")
        d = ImageDraw.Draw(im)
        for (x, y) in uv:
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(90, 220, 90))     # 検出
        for (x, y) in proj:
            d.ellipse([x - 2, y - 2, x + 2, y + 2], outline=(255, 80, 80))  # 再投影
        args.review.parent.mkdir(parents=True, exist_ok=True)
        im.save(args.review)

    print(f"検出 {len(uv)}点  目間 {iod:.1f}px")
    print(f"再投影 RMS {out['rms_px']}px  （目間で正規化 {out['rms_over_interocular']}）")
    print(f"alpha_norm {out['alpha_norm']}  頭打ちになった係数 {hist[-1]['clamped']}個")
    if not args.no_expression:
        top = ", ".join(f"{t['name']} {t['weight']}" for t in out["expression"]["top"][:5])
        print(f"表情として分離 beta_norm {out['expression']['beta_norm']}  → {top}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
