#!/usr/bin/env python3
"""あるマテリアルの三角形が使う UV の範囲をマスク画像にする。

生成メッシュから切り出した髪（`Material_0`）は TRELLIS の頭全体のテクスチャを
共有していて、髪メッシュが参照するのはその 6% ほど。髪の色替え（`hair_color`）を
テクスチャ全体の統計で掛けると顔の肌まで基準に入るので、**髪メッシュが実際に
参照する画素**だけを取り出す。

使い方:
    python3 -m texture_processing.uv_mask --model jobs/x/t2.glb \\
        --material Material_0 --out jobs/x/scores/mask_hairgeom.png
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blender.vrm_finalize import read_glb                       # noqa: E402
from preprocessing.template_landmarks import face_primitive     # noqa: E402
from texture_processing.face_texture_blend import rasterize, uv_of  # noqa: E402
from vrm.inject_texture import base_color_image                 # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--material", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=0,
                    help="マスクの一辺。0 ならそのマテリアルのテクスチャと同じ")
    ap.add_argument("--dilate", type=int, default=2,
                    help="縁のにじみ分だけ広げる（px）")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    from PIL import Image, ImageFilter

    doc, blob = read_glb(args.model)
    mi, pi, verts, tris, matname = face_primitive(doc, blob, args.material)
    prim = doc["meshes"][mi]["primitives"][pi]
    uv = uv_of(doc, blob, prim)
    if uv is None:
        raise SystemExit(f"'{matname}' に TEXCOORD_0 が無い")
    size = args.size
    if not size:
        img = doc["images"][base_color_image(doc, matname)]
        bv = doc["bufferViews"][img["bufferView"]]
        off = bv.get("byteOffset", 0)
        size = Image.open(io.BytesIO(blob[off:off + bv["byteLength"]])).size[0]
    m = rasterize(uv, tris, np.ones(len(tris), bool), size)
    if args.dilate > 0:
        m = m.filter(ImageFilter.MaxFilter(args.dilate * 2 + 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    m.save(args.out)
    cov = float((np.asarray(m) > 128).mean())
    rep = {"model": str(args.model), "material": matname, "triangles": int(len(tris)),
           "size": size, "coverage": round(cov, 4), "out": str(args.out)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"UV_MASK_OK {json.dumps(rep, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
