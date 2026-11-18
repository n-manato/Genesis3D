#!/usr/bin/env python3
"""GLB/VRM の画像を WebP に入れ替えて軽くする。**Blender を経由しない。**

## なぜ要るか

`blender/project_face.py` が焼くテクスチャは 2048² の PNG で、出力が 28.6MB に
なる。テンプレート自身は同じ 2048² を WebP で持っていて 7.2MB なので、
差はほぼ符号化だけ。KEHAI（three.js）も Blender も WebP を読む
（glTF の `EXT_texture_webp`）。

**再インポート・再エクスポートしない。** 画像バイト列の差し替えと bufferView の
付け替えだけなので、頂点・リグ・モーフ・マテリアルは一度も触らない
（`vrm/inject_texture.py` と同じ方針）。

**アルファは保つ。** Rocketbox の `opacity`（髪カードとまつ毛）は RGBA で
78.3% が透明で、落とすとカードが不透明な板になって顔に被さる（docs/09 §14.2）。

使い方:
    python3 -m vrm.compress_textures --input a.vrm --out a.vrm --quality 90
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blender.vrm_finalize import read_glb, write_glb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--max-size", type=int, default=0,
                    help="この辺長より大きい画像を縮める（0 で縮めない）")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    from PIL import Image

    doc, blob = read_glb(args.input)
    out = bytearray()
    detail, saved = [], 0

    # bufferView を全部作り直す。画像だけ差し替えると穴が残って逆に太る。
    old_views = doc.get("bufferViews", [])
    keep_idx: dict[int, int] = {}
    new_views: list[dict] = []

    def copy_view(i: int) -> int:
        if i in keep_idx:
            return keep_idx[i]
        bv = old_views[i]
        o = bv.get("byteOffset", 0)
        data = bytes(blob[o:o + bv["byteLength"]])
        while len(out) % 4:
            out.append(0)
        off = len(out)
        out.extend(data)
        nv = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        for k in ("byteStride", "target"):
            if k in bv:
                nv[k] = bv[k]
        new_views.append(nv)
        keep_idx[i] = len(new_views) - 1
        return keep_idx[i]

    # 画像以外の bufferView を先に運ぶ（accessor が参照する順は問わない）
    img_views = {im["bufferView"] for im in doc.get("images", [])
                 if "bufferView" in im}
    for i in range(len(old_views)):
        if i not in img_views:
            copy_view(i)

    for idx, im in enumerate(doc.get("images", [])):
        if "bufferView" not in im:
            continue
        bv = old_views[im["bufferView"]]
        o = bv.get("byteOffset", 0)
        raw = bytes(blob[o:o + bv["byteLength"]])
        pil = Image.open(io.BytesIO(raw))
        has_a = pil.mode in ("RGBA", "LA", "P") and "A" in pil.convert("RGBA").getbands()
        pil = pil.convert("RGBA" if has_a else "RGB")
        if args.max_size and max(pil.size) > args.max_size:
            pil = pil.resize((args.max_size, args.max_size), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, "WEBP", quality=args.quality, method=6)
        enc = buf.getvalue()
        use = enc if len(enc) < len(raw) else raw
        while len(out) % 4:
            out.append(0)
        off = len(out)
        out.extend(use)
        new_views.append({"buffer": 0, "byteOffset": off, "byteLength": len(use)})
        im["bufferView"] = len(new_views) - 1
        if use is enc:
            im["mimeType"] = "image/webp"
        saved += len(raw) - len(use)
        detail.append({"image": im.get("name"), "before_kb": round(len(raw) / 1024),
                       "after_kb": round(len(use) / 1024),
                       "alpha": has_a, "webp": use is enc})

    # accessor と texture の参照を張り替える
    for acc in doc.get("accessors", []):
        if acc.get("bufferView") is not None:
            acc["bufferView"] = keep_idx[acc["bufferView"]]
        sp = acc.get("sparse")
        if sp:
            for k in ("indices", "values"):
                if k in sp:
                    sp[k]["bufferView"] = keep_idx[sp[k]["bufferView"]]
    doc["bufferViews"] = new_views
    doc["buffers"][0]["byteLength"] = len(out)

    # WebP になった画像は EXT_texture_webp で参照する必要がある
    webp = {i for i, im in enumerate(doc.get("images", []))
            if im.get("mimeType") == "image/webp"}
    used = doc.setdefault("extensionsUsed", [])
    for t in doc.get("textures", []):
        src = t.get("source")
        ext = (t.get("extensions") or {}).get("EXT_texture_webp")
        if src is not None and src in webp:
            t.pop("source", None)
            t.setdefault("extensions", {})["EXT_texture_webp"] = {"source": src}
        elif ext and ext.get("source") not in webp:
            t["source"] = ext["source"]
            t["extensions"].pop("EXT_texture_webp")
            if not t["extensions"]:
                t.pop("extensions")
    if any((t.get("extensions") or {}).get("EXT_texture_webp")
           for t in doc.get("textures", [])):
        if "EXT_texture_webp" not in used:
            used.append("EXT_texture_webp")
    elif "EXT_texture_webp" in used:
        used.remove("EXT_texture_webp")

    # 薄いシェル（髪）が裏面カリングで透けないよう両面にする（make_vrm.py と同じ）
    for m in doc.get("materials", []):
        m["doubleSided"] = True
    write_glb(args.out, doc, bytes(out))
    before = args.input.stat().st_size
    after = args.out.stat().st_size
    rep = {"input": str(args.input), "out": str(args.out),
           "quality": args.quality, "images": len(detail),
           "before_mb": round(before / 1048576, 1),
           "after_mb": round(after / 1048576, 1),
           "detail": detail}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"画像 {len(detail)}枚を WebP へ  "
          f"{rep['before_mb']}MB → {rep['after_mb']}MB "
          f"（{100 - after * 100 // max(before,1)}% 減）→ {args.out}")


if __name__ == "__main__":
    main()
