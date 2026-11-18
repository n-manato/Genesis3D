#!/usr/bin/env python3
"""生成された全身メッシュを、変形済みテンプレートの座標系へ合わせる相似変換を求める。

## 何のためか

`blender/add_hair.py` は髪を生成メッシュから切り出して載せた。同じことを**服**で
やりたい。テンプレート（Rocketbox）の服はメッシュではなくテクスチャに描いた絵なので、
色は寄せられても**構造**（ブレザーの前開き・襟・インナー）は寄せられなかった
（docs/09 §13.4）。生成メッシュにはそれが入っている。

髪のときは位置合わせをラップが与えてくれた（`wrap.json` の `align`）。全身には
それが無いので、ここで求める。

## 解き方

MediaPipe Pose の33点を**生成メッシュの正面レンダ**で検出して逆投影し、3Dの
関節位置を得る。テンプレート側は glTF のノード（骨）の世界位置をそのまま使う。
両者を対応させてロバスト Umeyama で解く。

**左右の取り違えに注意。** MediaPipe の left は被写体の左（画像では右側）で、
Mixamo の `Left*` も被写体の左。同じ規約だが、取り違えると鏡像の解が出るので
残差で検証する（対応を入れ替えた解も計算して、残差の小さい方を採る）。

使い方:
    $HOME/miniconda3/envs/genesis-face/bin/python -m preprocessing.align_generated_body \\
        --generated body_clean.glb --render front.png --render-meta front.json \\
        --model jobs/x/body_deformed.glb --out align_body.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.body_params import node_globals              # noqa: E402
from preprocessing.wrap_head import robust_umeyama              # noqa: E402
from blender.vrm_finalize import read_glb                       # noqa: E402

# MediaPipe Pose の点番号 → テンプレートのボーン名
PAIRS = [
    (11, "LeftArm"), (12, "RightArm"),
    (13, "LeftForeArm"), (14, "RightForeArm"),
    (15, "LeftHand"), (16, "RightHand"),
    (23, "LeftUpLeg"), (24, "RightUpLeg"),
    (25, "LeftLeg"), (26, "RightLeg"),
    (27, "LeftFoot"), (28, "RightFoot"),
]
MIRROR = {"Left": "Right", "Right": "Left"}


def swap_side(name: str) -> str:
    for a, b in MIRROR.items():
        if name.startswith(a):
            return b + name[len(a):]
    return name


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generated", type=Path, required=True)
    ap.add_argument("--render", type=Path, required=True,
                    help="ortho_front.py が撮った生成メッシュの正面（大きさ参照用）")
    ap.add_argument("--landmarks", type=Path, required=True,
                    help="preprocessing.body_landmarks を **その正面レンダに対して** "
                         "掛けた出力。**MediaPipe は genesis-eval、trimesh は "
                         "genesis-face にあり env をまたげない**ので分けてある")
    ap.add_argument("--render-meta", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True,
                    help="変形済みテンプレート（メッシュ表面へ逆投影する）")
    ap.add_argument("--model-render", type=Path, required=True,
                    help="**アバター側**の正面レンダ")
    ap.add_argument("--model-render-meta", type=Path, required=True)
    ap.add_argument("--model-landmarks", type=Path, required=True,
                    help="そのレンダに body_landmarks を掛けた出力")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--points", default="11,12,23,24,25,26",
                    help="使う MediaPipe の点番号。**腕を入れてはいけない** —— "
                         "アバターは腕が下がったA字、生成された全身はTポーズで"
                         "姿勢が違うため、相似変換では合わない（実測 RMS: "
                         "胴のみ 22.3 / 胴＋膝 21.7 / 全身12点 99.2mm）")
    args = ap.parse_args()

    import trimesh
    from PIL import Image
    from preprocessing.ict_fit import backproject

    # --- 両側とも「レンダに逆投影した**表面**点」で対応させる ------------
    #
    # **骨の位置と比べてはいけない。** 逆投影した点は服の表面に当たり、骨の内部
    # 位置とは肩と腰で数cm系統的にずれる。実測で残差 116.6mm になり使えなかった
    # （docs/09 §18.1）。両側を表面点にすればその系統差は相殺する。
    def surface_points(mesh_path, render, meta_path, lm_path):
        m = trimesh.load(mesh_path, process=False)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(list(m.geometry.values()))
        lmj = json.loads(Path(lm_path).read_text())
        px = np.asarray(lmj["landmarks_px"], dtype=float)
        im = Image.open(render).convert("RGB")
        if tuple(lmj.get("image_size", im.size)) != tuple(im.size):
            raise SystemExit(f"{lm_path} の画像 {lmj.get('image_size')} と "
                             f"{render} の {im.size} が違う")
        uv = np.array([px[i] for i, _ in PAIRS])
        p3 = backproject(uv, json.loads(Path(meta_path).read_text()), m)
        return p3

    gen3 = surface_points(args.generated, args.render,
                          args.render_meta, args.landmarks)
    mod3 = surface_points(args.model, args.model_render,
                          args.model_render_meta, args.model_landmarks)

    want = {int(x) for x in args.points.split(",")}
    ok = (~np.isnan(gen3[:, 0])) & (~np.isnan(mod3[:, 0]))
    src, dst, used = [], [], []
    for k, (idx, bone) in enumerate(PAIRS):
        if ok[k] and idx in want:
            src.append(gen3[k])
            dst.append(mod3[k])
            used.append(bone)
    if len(src) < 4:
        raise SystemExit(f"対応が {len(src)}組しか取れない（可視 {int(ok.sum())}）")
    src, dst = np.array(src), np.array(dst)

    # **左右の取り違えを残差で検証する。** 入れ替えた対応でも解いて良い方を採る。
    swap_idx = {b: i for i, b in enumerate(used)}
    best = None
    for tag, d in (("as-is", dst),
                   ("swapped", np.array([dst[swap_idx.get(swap_side(b),
                                                          swap_idx[b])]
                                         for b in used]))):
        s, r, t, keep = robust_umeyama(src, d)
        res_mm = np.linalg.norm((s * (r @ src.T).T) + t - d, axis=1) * 1000
        rms = float(np.sqrt((res_mm[keep] ** 2).mean()))
        print(f"  {tag:8s} scale {s:.4f}  採用 {int(keep.sum())}/{len(src)}  "
              f"RMS {rms:.1f}mm")
        if best is None or rms < best[0]:
            best = (rms, tag, s, r, t, keep)
    rms, tag, s, r, t, keep = best
    if rms > 40:
        print(f"警告: 残差 {rms:.1f}mm は大きい。服の切り出しがずれる恐れ",
              file=sys.stderr)

    # **対応点そのものを出す。** 相似変換だけでは体型（比率）の違いを吸収できない。
    # RBF の変形場なら吸収でき、しかも投影と違って衣服の緩みが潰れない（docs/09 §27）。
    pairs = {"centers": [[round(float(c), 6) for c in p] for p in src],
             "targets": [[round(float(c), 6) for c in p] for p in dst]}
    out = {"generated": str(args.generated), "model": str(args.model),
           "pairs": pairs,
           "correspondence": tag, "used_bones": used, "points": sorted(want),
           "inliers": int(keep.sum()), "rms_mm": round(rms, 2),
           "align": {"scale": float(s),
                     "rotation": [[float(x) for x in row] for row in r],
                     "translation": [float(x) for x in t]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"採用 {tag}  スケール {s:.4f}  RMS {rms:.1f}mm  → {args.out}")


if __name__ == "__main__":
    main()
