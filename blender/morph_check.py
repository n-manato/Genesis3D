#!/usr/bin/env python3
"""モーフが**実際に動くか**を確かめる。名前の存在だけでは分からない。

`scripts/check_kehai_model.py` はモーフの**名前**を見る。名前があれば通るので、
中身が空でも合格してしまう。実測でこれに引っかかった —— 接合後のアバターは
`eyeBlinkLeft` を持っていたが変位が **0.0mm** で、まばたきしなかった
（docs/08 §10.14）。原因は二つあり、どちらも名前検査では見えない:

  1. `transfer_expr.py` が対象のローカル座標で最近傍を探していた。対象が
     アーマチュアに親子付けされているとローカルとワールドがずれ、
     全頂点が falloff の外に落ちて転写が 0 個になる
  2. 接合後のメッシュには素体由来の**空のシェイプキー**が残っており、
     名前が衝突して転写分が `eyeBlinkLeft.001` になる

**ワールド座標で測ること。** 対象のローカル空間はスケールが入っていることがあり、
実測では 91倍に膨らんでいて、ローカルで測ると jawOpen が 5.7m と出た。

    blender --background --factory-startup --python blender/morph_check.py -- \
        --input avatar.glb --morphs eyeBlinkLeft,jawOpen --report r.json
"""
import argparse
import json
import sys
from pathlib import Path

import bpy

# KEHAI が実際に駆動するもの。ここが動かないと会話アバターとして成立しない。
DEFAULT = ["eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "mouthSmileLeft",
           "browDownLeft", "mouthFunnel"]
# あれば良いが必須ではないもの。ICT は ARKit 命名しか出さないので、
# Oculus のビセーム（aa/ih/ou…）は付かない。KEHAI のリップシンクは
# `MOUTH_CANDIDATES` に `jawOpen` を含むので、それで成立する。
OPTIONAL = ["aa", "ih", "ou"]
# 変位がこれ未満なら「動いていない」とみなす（頭 200mm に対して 1mm）。
MIN_MM = 1.0


def world_verts(meshes):
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for o in meshes:
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        mw = ev.matrix_world
        out.append([tuple(mw @ v.co) for v in me.vertices])
        ev.to_mesh_clear()
    return out


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--morphs", default=",".join(DEFAULT))
    ap.add_argument("--optional", default=",".join(OPTIONAL),
                    help="無くても合否に影響しないモーフ")
    ap.add_argument("--min-mm", type=float, default=MIN_MM)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.input)
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.data.shape_keys]
    if not meshes:
        raise SystemExit("シェイプキーを持つメッシュが無い")

    base = world_verts(meshes)
    res = {}
    optional = {n.strip() for n in args.optional.split(",") if n.strip()}
    for name in list(args.morphs.split(",")) + sorted(optional):
        name = name.strip()
        if not name:
            continue
        hit = 0
        for o in meshes:
            kb = o.data.shape_keys.key_blocks
            if name in kb:
                kb[name].value = 1.0
                hit += 1
        if not hit:
            res[name] = {"found": False, "moves": False,
                         "optional": name in optional}
            continue
        cur = world_verts(meshes)
        d = 0.0
        for bb, cc in zip(base, cur):
            for p, q in zip(bb, cc):
                d = max(d, sum((a - b) ** 2 for a, b in zip(p, q)) ** 0.5)
        for o in meshes:
            kb = o.data.shape_keys.key_blocks
            if name in kb:
                kb[name].value = 0.0
        mm = round(d * 1000, 2)
        res[name] = {"found": True, "meshes": hit, "max_delta_mm": mm,
                     "moves": mm >= args.min_mm, "optional": name in optional}

    ok = all(v["moves"] for v in res.values() if not v.get("optional"))
    out = {"input": args.input, "min_mm": args.min_mm, "ok": ok, "morphs": res}
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    for k, v in res.items():
        mark = "✓" if v.get("moves") else ("−" if v.get("optional") else "✗")
        detail = (f"{v['max_delta_mm']}mm" if v["found"] else "名前が無い")
        print(f"  {mark} {k:18s} {detail}")
    print("MORPH_CHECK " + json.dumps(out, ensure_ascii=False))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
