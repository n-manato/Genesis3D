#!/usr/bin/env python3
"""ICT の識別係数を、テンプレートを動かす対応点に変換する。

`ict_photo_fit.py` が出した α と、`template_landmarks.py` が打った68点、
`template_anchors.py` が打ったアンカーを突き合わせて、
`shape_transfer.py` にそのまま渡せる `{centers, targets}` を作る。

## 絶対座標ではなく「平均からのズレ」を運ぶ

    target_i = template_landmark_i + k · S · ( V(α) − V_mean )[ ict_lm_i ]

**ICT の顔をテンプレートに被せるのではない。** ICT の平均顔からのズレだけを取り出して、
テンプレートの造形の上に足す。こうしないとテンプレートが ICT の顔に置き換わってしまい、
素体の作り込み（トポロジは残るが造形の意図）が消える。§42 で ICT の幾何をそのまま
使ったら ArcFace が 0.708 → 0.394 に落ちたのは、まさにそれをやったからだった。

`S` は ICT（cm）からテンプレート（m）への相似変換で、**ICT の平均顔の68点と
テンプレートの68点を procrustes で合わせて一度だけ求める**。ズレの変換には
回転とスケールだけを使う（平行移動はズレには効かない）。

`k` はデフォルメ係数。1.0 で「そのまま」、大きくすると誇張される。
ゲームの捏脸で言えばスライダーの効き幅にあたる。

## アンカーは必ず入る

顔の68点だけで解くと、多項式項が全身に効いて足まで動く。実測（docs/08 §10.3）で
Rocketbox（全身1.8m）では 40cm 以上離れた頂点が **13.113mm → 0.134mm**。
`--no-anchors` は診断用で、通常は外さない。

使い方:
    python -m preprocessing.deform_params \\
        --fit <fit.json> --template rocketbox_male_01 \\
        --out <pairs.json> [--exaggeration 1.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.ict_fit import ICT_DIR, N_IDENTITY, load_identity_basis, procrustes  # noqa: E402

TEMPLATES = Path("configs/templates.yaml")

# 68点の左右対応（dlib / insightface の並び）。中央の点は自分自身と対になる。
MIRROR = {}
for _a, _b in [(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
               (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
               (31, 35), (32, 34),
               (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
               (48, 54), (49, 53), (50, 52), (55, 59), (56, 58),
               (60, 64), (61, 63), (65, 67)]:
    MIRROR[_a] = _b
    MIRROR[_b] = _a
for _c in (8, 27, 28, 29, 30, 33, 51, 57, 62, 66):
    MIRROR[_c] = _c


def symmetrize(dev: np.ndarray, order: list[int]) -> np.ndarray:
    """識別のズレを左右対称にする。

    **顔の同一性はほぼ左右対称で、非対称は写真の側の事情**（頭のわずかな傾き、
    片側だけの笑い、照明）であることが多い。実測でも、正面に見える写真から
    素直に解いてデフォルメ係数を上げると、眉と口が片側だけ吊り上がった。

    テンプレートは x=0 を矢状面として作られているので、対になる点の変位を
    x 反転して平均する。中央の点は x 成分だけ落とす。
    """
    pos = {i: k for k, i in enumerate(order)}
    out = dev.copy()
    flip = np.array([-1.0, 1.0, 1.0])
    for k, i in enumerate(order):
        j = MIRROR.get(i)
        if j is None or j not in pos:
            continue
        if j == i:
            out[k] = dev[k] * np.array([0.0, 1.0, 1.0])   # 中央は左右に動かさない
        else:
            out[k] = 0.5 * (dev[k] + dev[pos[j]] * flip)
    return out


def load_template(name: str) -> dict:
    cfg = yaml.safe_load(TEMPLATES.read_text())
    if name not in cfg["templates"]:
        raise SystemExit(f"テンプレート '{name}' が無い。候補: {list(cfg['templates'])}")
    return cfg["templates"][name]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", type=Path, required=True, help="ict_photo_fit.py の出力")
    ap.add_argument("--template", required=True, help="configs/templates.yaml のキー")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exaggeration", type=float, default=None,
                    help="デフォルメ係数。既定はテンプレート設定の値")
    ap.add_argument("--no-symmetry", action="store_true",
                    help="左右対称化をしない。**写真の傾きや片笑いが形に残る**ので診断用")
    ap.add_argument("--no-anchors", action="store_true",
                    help="アンカーを外す。**診断用**。外すと顔の変形が全身に漏れる")
    ap.add_argument("--cache", default="vendor/ICT-FaceKit/identity_basis.npz")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    tmpl = load_template(args.template)
    if not tmpl.get("landmarks68"):
        raise SystemExit(f"{args.template} にランドマークが無い。"
                         f"preprocessing/template_landmarks.py を先に走らせること")

    fit = json.loads(args.fit.read_text())
    alpha = np.asarray(fit["alpha"], dtype=np.float64)
    if len(alpha) != N_IDENTITY:
        raise SystemExit(f"識別係数が {len(alpha)}個（{N_IDENTITY}個のはず）")

    lm_file = json.loads(Path(tmpl["landmarks68"]).read_text())
    # 68点の並びは dlib/insightface の規約で ICT と共通。番号で突き合わせる。
    by_i = {p["i"]: np.asarray(p["xyz"], dtype=np.float64) for p in lm_file["landmarks"]}
    have = sorted(by_i)
    T = np.stack([by_i[i] for i in have])                    # テンプレートの68点

    neutral, basis = load_identity_basis(Path(args.cache))
    idx = json.loads((ICT_DIR / "vertex_indices.json").read_text())
    ict_lm = np.asarray(idx["idx_to_landmark_verts"])[have]

    V_mean = neutral[ict_lm]                                  # ICT 平均顔の68点（cm）
    V_id = (neutral + np.tensordot(alpha, basis.astype(np.float64), axes=(0, 0)))[ict_lm]
    dev_ict = V_id - V_mean                                   # ズレ（cm）

    # ICT → テンプレートの相似変換。**平均顔**とテンプレートを合わせる（本人ではなく）。
    s, R, _ = procrustes(V_mean, T)
    k = args.exaggeration
    if k is None:
        k = float((tmpl.get("deform") or {}).get("exaggeration", 1.0))
    dev_tmpl = s * (dev_ict @ R.T)                            # 回転とスケールのみ
    asym = float(np.linalg.norm(dev_tmpl - symmetrize(dev_tmpl, have), axis=1).mean())
    if not args.no_symmetry:
        dev_tmpl = symmetrize(dev_tmpl, have)
    dev_tmpl = k * dev_tmpl

    centers = [T]
    targets = [T + dev_tmpl]
    n_anchor = 0
    if not args.no_anchors:
        af = tmpl.get("anchors")
        if not af:
            print("警告: アンカーが登録されていない。顔の変形が全身に漏れる "
                  "（docs/08 §10.3）", file=sys.stderr)
        else:
            A = np.asarray(json.loads(Path(af).read_text())["points"], dtype=np.float64)
            centers.append(A)
            targets.append(A)                                 # 変位ゼロ
            n_anchor = len(A)

    C = np.vstack(centers)
    G = np.vstack(targets)
    mag = np.linalg.norm(dev_tmpl, axis=1)

    out = {
        "centers": C.tolist(), "targets": G.tolist(),
        "_meta": {
            "template": args.template,
            "fit": str(args.fit),
            "photo": fit.get("photo"),
            "exaggeration": k,
            "ict_to_template": {"scale": round(float(s), 8),
                                "rotation": [[round(float(x), 6) for x in r] for r in R]},
            "landmarks": len(T), "anchors": n_anchor,
            "symmetrized": not args.no_symmetry,
            # 対称化で落ちた量。大きいほど写真が傾いていたか片側だけ笑っていた。
            "asymmetry_mm": round(asym * 1000, 3),
            "displacement_mm": {"mean": round(float(mag.mean()) * 1000, 3),
                                "max": round(float(mag.max()) * 1000, 3)},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(out["_meta"], ensure_ascii=False, indent=2))

    m = out["_meta"]
    print(f"テンプレート {args.template}  デフォルメ係数 {k}")
    print(f"ICT→テンプレ スケール {s:.6f}（ICTはcm、テンプレはm）")
    print(f"顔の変位  平均 {m['displacement_mm']['mean']}mm / "
          f"最大 {m['displacement_mm']['max']}mm")
    print(f"左右対称化 {'あり' if m['symmetrized'] else 'なし'}"
          f"（落とした非対称成分 {m['asymmetry_mm']}mm）")
    print(f"対応点 {len(T)}（顔） + {n_anchor}（アンカー） = {len(C)}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
