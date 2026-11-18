#!/usr/bin/env python3
"""Rocketbox の FBX を GLB にする。テクスチャの貼り直しを含む。

`scripts/prepare_rocketbox.py` から呼ばれる。単体でも動く。

**テクスチャの参照が壊れている。** Rocketbox の FBX は 3ds Max から書き出された
ときの絶対パスをそのまま持っている:

    D:\\temp\\Humans\\with_opacity_version\\m002\\textures\\m002_head_color.tga

当然どこにも無いので、Blender は画像データブロックだけ作ってファイルを読めず、
glTF 書き出しは**エラーを出さずに** `images: 0` の GLB を吐く。顔テクスチャ 2048² が
この素体を選んだ理由なので、これは静かな失敗になる。Microsoft 自身も Unity 用に
`Assets/Editor/FixRocketboxMaxImport.cs` を同梱していて、既知の問題である。

ここではベース名だけを見て、FBX の隣の `../Textures/` から貼り直す。

使い方:
    blender --background --factory-startup --python blender/rocketbox_to_glb.py -- \\
        --fbx <...>_facial.fbx --out out.glb [--image-format WEBP] [--report r.json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import bpy


def relink_images(fbx: Path) -> dict:
    """壊れた絶対パスを、FBX の隣の Textures/ にあるファイルへ貼り直す。"""
    # Export/<name>.fbx の隣、つまり <avatar>/Textures/ を探す。
    # 見つからない場合に備えて FBX と同階層も候補に入れる。
    roots = [fbx.parent.parent / "Textures", fbx.parent / "Textures", fbx.parent]
    found: dict[str, Path] = {}
    for r in roots:
        if r.is_dir():
            for f in r.iterdir():
                found.setdefault(f.name.lower(), f)

    rep = {"searched": [str(r) for r in roots], "relinked": [], "missing": []}
    for img in bpy.data.images:
        if img.name == "Render Result":
            continue
        # filepath は Windows 区切りなので basename が取れない。両方で割る。
        base = img.filepath.replace("\\", "/").rsplit("/", 1)[-1]
        hit = found.get(base.lower())
        if hit is None:
            rep["missing"].append(base)
            continue
        img.filepath = str(hit)
        img.source = "FILE"
        try:
            img.reload()
        except RuntimeError:
            rep["missing"].append(base)
            continue
        # reload しても壊れていれば size が 0 のまま残る
        if tuple(img.size) == (0, 0):
            rep["missing"].append(base)
        else:
            rep["relinked"].append({"name": base, "size": list(img.size)})
    return rep


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-format", default="WEBP")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(args.fbx))
    rep = relink_images(args.fbx)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(args.out), export_format="GLB",
        export_morph=True, export_skins=True,
        export_image_format=args.image_format,
        export_morph_normal=False, export_tangents=False)

    rep["exported"] = args.out.exists()
    rep["size_mb"] = round(args.out.stat().st_size / 1048576, 1) if rep["exported"] else 0
    if args.report:
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print("ROCKETBOX_GLB " + json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
