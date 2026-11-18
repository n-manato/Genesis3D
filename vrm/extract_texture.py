#!/usr/bin/env python3
"""GLB/VRM からマテリアルのベースカラー画像を取り出す。

`texture_processing/skin_blend.py` に渡す「テンプレートの肌」と「投影後の肌」を
用意するため。Blender を経由しない（docs/08 §6）。

使い方:
    python -m vrm.extract_texture --input <in.glb> --material head --out <out.png>
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blender.vrm_finalize import read_glb                      # noqa: E402
from vrm.inject_texture import base_color_image                # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--material", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from PIL import Image

    doc, blob = read_glb(args.input)
    idx = base_color_image(doc, args.material)
    img_def = doc["images"][idx]
    bv = doc["bufferViews"][img_def["bufferView"]]
    off = bv.get("byteOffset", 0)
    data = blob[off:off + bv["byteLength"]]
    im = Image.open(io.BytesIO(data))
    # **アルファを落とさない。** Rocketbox の `opacity`（髪カードとまつ毛）は
    # RGBA で 78.3% が透明。RGB に落として書き戻すとカードが不透明な板になり、
    # 顔に被さって held-out ArcFace が 0.66 → 0.56 に落ちた。
    im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") and
                    "A" in im.convert("RGBA").getbands() else "RGB")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    im.save(args.out)
    print(f"'{args.material}' の画像{idx}（{img_def.get('name')}, "
          f"{img_def.get('mimeType')}）{im.size} → {args.out}")


if __name__ == "__main__":
    main()
