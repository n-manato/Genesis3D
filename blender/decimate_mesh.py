#!/usr/bin/env python3
"""メッシュの頂点数を落とす。UV は保つ。

TRELLIS.2 が出す頭は **593,748頂点** ある（`decimation_target` を 100万で呼んだ場合）。
そのまま体に接合すると全体で 597k 頂点になり、旧パイプラインの成果物
（`subject_a_v2.vrm` = 151,974頂点 / 15.7MB）の 4倍で、ブラウザでは重すぎる。

**削減はテクスチャ投影の前に行うこと。** 後にすると、焼き込んだ UV が
削減で歪んで顔がずれる。投影は削減後のトポロジに対して焼く。

Blender の Decimate（COLLAPSE）を使う。`use_collapse_triangulate` は付けない
（三角形化されると後段の判定が変わる）。

    blender --background --factory-startup --python blender/decimate_mesh.py -- \
        --input head.glb --out head_small.glb --target 100000
"""
import argparse
import json
import sys
from pathlib import Path

import bpy


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=100_000, help="目標の頂点数")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.input)
    # マテリアルを持たないメッシュ（インポータが毎回作る Icosphere）は対象外
    meshes = [o for o in bpy.data.objects
              if o.type == "MESH" and [m for m in o.data.materials if m]]
    if not meshes:
        raise SystemExit("メッシュが無い")

    before = sum(len(o.data.vertices) for o in meshes)
    ratio = min(1.0, args.target / max(before, 1))
    if ratio >= 1.0:
        print(f"削減不要（{before:,} <= {args.target:,}）")
    for o in meshes:
        if ratio < 1.0:
            m = o.modifiers.new("Decimate", "DECIMATE")
            m.decimate_type = "COLLAPSE"
            m.ratio = ratio
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.modifier_apply(modifier=m.name)
    after = sum(len(o.data.vertices) for o in meshes)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    for o in bpy.data.objects:
        if o.type == "ARMATURE":
            o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=args.out, export_format="GLB",
                              use_selection=True, export_image_format="AUTO",
                              export_morph=True, export_morph_normal=False,
                              export_morph_tangent=False)
    rep = {"input": args.input, "out": args.out, "ratio": round(ratio, 5),
           "vertices_before": before, "vertices_after": after,
           "size_mb": round(Path(args.out).stat().st_size / 1048576, 1)}
    if args.report:
        Path(args.report).write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print("DECIMATE_OK " + json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
