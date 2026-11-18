#!/usr/bin/env python3
"""GLB/VRM の画像を差し替える。Blender を経由しない。

`texture_processing/skin_blend.py` が作った肌テクスチャを、投影済みのモデルへ戻す。

**再インポート・再エクスポートしない。** `blender/vrm_finalize.py` の冒頭にあるとおり、
Blender の往復はテクスチャを再エンコードして個体を別物にしうる。ここでやるのは
「画像バイト列の差し替えと bufferView の付け替え」だけなので、頂点もリグも
モーフもマテリアルも一度も触らない（docs/08 §6）。

差し替え先は**マテリアル名**で指定する。画像の並び順は書き出し側の都合で変わるし、
画像名は `Map #99` のように意味を持たないことがあるため。

使い方:
    python -m vrm.inject_texture --input <in.glb> --out <out.glb> \\
        --material head --png <blended.png> [--report r.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blender.vrm_finalize import read_glb, write_glb  # noqa: E402

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp"}


def base_color_image(doc: dict, material: str) -> int:
    """マテリアル名から baseColorTexture の image 番号を引く。

    WebP のテクスチャは `source` を持たず `EXT_texture_webp.source` に入る。
    両方見ないと取りこぼす（Rocketbox のテンプレートは WebP）。
    """
    hits = [m for m in doc.get("materials", []) if material in (m.get("name") or "")]
    if not hits:
        names = [m.get("name") for m in doc.get("materials", [])]
        raise SystemExit(f"マテリアル '{material}' が無い。候補: {names}")
    if len(hits) > 1:
        print(f"注意: '{material}' に {len(hits)}個一致。最初のものを使う",
              file=sys.stderr)
    pbr = hits[0].get("pbrMetallicRoughness") or {}
    tex = pbr.get("baseColorTexture")
    if tex is None:
        raise SystemExit(f"マテリアル '{hits[0].get('name')}' に baseColorTexture が無い")
    t = doc["textures"][tex["index"]]
    src = t.get("source")
    if src is None:
        src = ((t.get("extensions") or {}).get("EXT_texture_webp") or {}).get("source")
    if src is None:
        raise SystemExit("テクスチャに image の参照が無い")
    return int(src)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--material", required=True, help="差し替え先のマテリアル名（部分一致）")
    ap.add_argument("--png", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    doc, blob = read_glb(args.input)
    idx = base_color_image(doc, args.material)
    data = args.png.read_bytes()

    out = bytearray(blob)
    offset = len(out)
    out += data
    out += b"\x00" * ((-len(out)) % 4)          # 4バイト境界（GLB 仕様）
    doc["bufferViews"].append(
        {"buffer": 0, "byteOffset": offset, "byteLength": len(data)})

    img = doc["images"][idx]
    before = {"name": img.get("name"), "mimeType": img.get("mimeType")}
    img["bufferView"] = len(doc["bufferViews"]) - 1
    img["mimeType"] = MIME.get(args.png.suffix.lower(), "image/png")
    img.pop("uri", None)

    # WebP から PNG に変わると EXT_texture_webp の前提が崩れる。参照を source へ戻す。
    moved = 0
    for t in doc.get("textures", []):
        ext = (t.get("extensions") or {}).get("EXT_texture_webp")
        if ext and ext.get("source") == idx and img["mimeType"] != "image/webp":
            t["source"] = idx
            t["extensions"].pop("EXT_texture_webp")
            if not t["extensions"]:
                t.pop("extensions")
            moved += 1
    if moved:
        for k in ("extensionsUsed", "extensionsRequired"):
            if "EXT_texture_webp" in (doc.get(k) or []) and not any(
                    "EXT_texture_webp" in (t.get("extensions") or {})
                    for t in doc.get("textures", [])):
                doc[k].remove("EXT_texture_webp")

    doc["buffers"][0]["byteLength"] = len(out)
    write_glb(args.out, doc, bytes(out))

    rep = {"input": str(args.input), "out": str(args.out), "material": args.material,
           "image_index": idx, "replaced": before,
           "mime": img["mimeType"], "png_bytes": len(data),
           "webp_refs_moved_to_source": moved,
           "size_mb": round(args.out.stat().st_size / 1048576, 1)}
    if args.report:
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"'{args.material}' の画像{idx}（{before['name']}）を差し替え  "
          f"{len(data)/1048576:.1f}MB  → {args.out}（{rep['size_mb']}MB）")


if __name__ == "__main__":
    main()
