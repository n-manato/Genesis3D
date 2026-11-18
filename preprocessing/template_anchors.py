#!/usr/bin/env python3
"""テンプレート素体にゼロ変位のアンカー点を打つ。**一度きりの作業。**

RBF の基底 phi(r)=r に一次多項式を足したものは**大域的な台**を持つ。顔の68点だけで
変位場を解くと、多項式項が全身に効いて足まで動く。実測（docs/08 §10.3）:

    鼻先を 12mm 前に出したとき、40cm 以上離れた頂点の変位

      アンカー無し   1.200 mm
      アンカー200点  0.045 mm      ← 27分の1。鼻先の 12.00mm は変わらない

**これが無いと鼻を動かすと足が動く。** 任意項目ではない。

首から下の頂点を最遠点サンプリングで拾い、変位ゼロの対応点として置く。首のすぐ下から
始めると顎の変形が首で急に止まって段差になるので、**顔の下端から少し離した位置**から
アンカーを張り、あいだは RBF に滑らかに繋がせる。

使い方:
    python -m preprocessing.template_anchors \\
        --mesh assets/templates/rocketbox/male_adult_01.glb \\
        --face-material head --count 200 \\
        --out configs/landmarks/rocketbox_male_01_anchors.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blender.vrm_finalize import read_glb                          # noqa: E402
from preprocessing.shape_transfer import read_accessor             # noqa: E402
from preprocessing.template_landmarks import face_primitive        # noqa: E402


def farthest_point(pts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """最遠点サンプリング。偏るとその周辺だけが強く拘束されて歪む。"""
    n = min(n, len(pts))
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for _ in range(n - 1):
        i = int(d.argmax())
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return np.asarray(idx)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--face-material", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--gap", type=float, default=0.08,
                    help="顔の下端からこれだけ下を拘束の開始点にする（m）。"
                         "0 にすると顎の変形が首で急に止まって段差が出る")
    ap.add_argument("--template-id", default=None)
    args = ap.parse_args()

    doc, blob = read_glb(args.mesh)
    _, _, face_v, _, matname = face_primitive(doc, blob, args.face_material)
    face_min_y = float(face_v[:, 1].min())
    cut = face_min_y - args.gap

    # 全メッシュの頂点から、首より下だけを集める。**顔プリミティブは除かない** ——
    # 顔マテリアルは首まで伸びていることがあり、cut より下ならそこも拘束してよい。
    pts = []
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            v = read_accessor(doc, blob, prim["attributes"]["POSITION"])
            if v is not None:
                pts.append(v)
    allv = np.vstack(pts)
    low = allv[allv[:, 1] < cut]
    if len(low) < args.count:
        raise SystemExit(
            f"y < {cut:.3f} の頂点が {len(low)}個しかない（{args.count}点を要求）。"
            f"--gap を小さくすること")

    pick = low[farthest_point(low, args.count)]
    out = {
        "template": args.template_id or args.mesh.stem,
        "mesh": str(args.mesh),
        "face_material": matname,
        "face_min_y": round(face_min_y, 6),
        "gap": args.gap,
        "cut_y": round(cut, 6),
        "candidates": int(len(low)),
        "count": int(len(pick)),
        "purpose": "RBF のゼロ変位アンカー。無いと顔の変形が全身に漏れる（docs/08 §10.3）",
        "points": [[round(float(c), 6) for c in p] for p in pick],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    print(f"顔の下端 y={face_min_y:.3f}  拘束の開始 y={cut:.3f}（gap {args.gap}m）")
    print(f"候補 {len(low):,}頂点 → 最遠点サンプリングで {len(pick)}点")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
