"""頭部メッシュを正射投影で正面から撮り、画素と世界座標の対応を書き出す。

ICT-FaceKit を対象の頭に当て込むには、対象側の 68 点ランドマークが 3D で要る。
2D の検出器を使うので、**画素から世界座標へ戻せる撮り方**でなければならない。
透視投影だと depth が分からないと戻せないが、正射投影なら画素 →(X, Y) が
一次式で決まり、残る Z は光線を飛ばせば求まる。

**座標系に注意。** glTF インポータはメッシュデータ自体を Y上 → Z上 に変換して
読み込む。したがって Blender 内では「Z が上、顔は -Y を向く」。glTF に戻すときは
(x, y, z)_blender → (x, z, -y)_gltf。ここは Blender 座標のまま撮り、メタに
両方の対応を書き出す。

  blender --background --factory-startup --python ortho_front.py -- \
      --input <head.glb> --out <front.png> --meta <bounds.json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

RES = 1024
MARGIN = 1.06     # 顔が画枠に触れると検出が落ちるので少しだけ余白を取る
# 顔マテリアルで枠取りするときの既定。**寄りすぎても検出は落ちる。**
# 実測: Rocketbox 男性を頭のbboxぴったり(span 0.39)で撮ると SCRFD が顔を見つけられず、
# 1.35倍(span 0.53)にしたら通った。SCRFD は画枠に対して顔が大きすぎると外す。
MATERIAL_MARGIN = 1.35


def import_mesh(path: Path):
    before = set(bpy.data.objects)
    ext = path.suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise SystemExit(f"未対応の拡張子: {ext}")
    return [o for o in bpy.data.objects if o not in before and o.type == "MESH"]


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--res", type=int, default=RES)
    ap.add_argument("--frame", choices=("full", "head"), default="full",
                    help="head は肩から上だけを枠に入れる。**全身メッシュには必須**。"
                         "全身を 1024 に収めると顔が 150px 程度になり、68点検出が落ちる")
    ap.add_argument("--frame-material", default=None,
                    help="このマテリアルを持つ面だけで枠を決める（部分一致）。"
                         "**シルエットからの肩検出より確実**なので、顔マテリアルが"
                         "分かっているテンプレートではこちらを使う")
    ap.add_argument("--head-fraction", type=float, default=0.25,
                    help="肩の自動検出が失敗したときに上から取る割合")
    ap.add_argument("--margin", type=float, default=None,
                    help=f"枠の余白。既定は全体なら {MARGIN}、"
                         f"--frame-material 指定時は {MATERIAL_MARGIN}")
    args = ap.parse_args(argv)

    bpy.ops.wm.read_homefile(use_empty=True)
    meshes = import_mesh(Path(args.input))
    if not meshes:
        raise SystemExit("メッシュがありません")

    # インポータが入れた変換をメッシュデータへ焼く（以降 matrix_world は単位行列）
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.context.view_layer.update()

    lo = Vector((float("inf"),) * 3)
    hi = Vector((float("-inf"),) * 3)
    for o in meshes:
        for v in o.data.vertices:
            p = o.matrix_world @ v.co
            for i in range(3):
                lo[i] = min(lo[i], p[i]); hi[i] = max(hi[i], p[i])

    # 頭だけを枠に入れる場合は、肩から上の頂点だけで bbox を取り直す。
    # 帯の検出は blender/mesh_landmarks.py と共用する（project_face.py と同じ）。
    band = None
    if args.frame_material:
        # マテリアルで枠を決める。肩のシルエット検出は体型に左右される
        # （実測: Rocketbox の女性は絞ったシャツのウエストを肩と誤検出して
        # 全身が枠に入り、顔が小さすぎて68点の精度が落ちた）。
        # テンプレートは固定でマテリアル名も分かっているので、推定に頼る理由がない。
        hlo = Vector((float("inf"),) * 3); hhi = Vector((float("-inf"),) * 3)
        n = 0
        for o in meshes:
            slots = [i for i, m in enumerate(o.data.materials)
                     if m and args.frame_material in m.name]
            if not slots:
                continue
            keep = set()
            for poly in o.data.polygons:
                if poly.material_index in slots:
                    keep.update(poly.vertices)
            for vi in keep:
                w = o.matrix_world @ o.data.vertices[vi].co
                for i in range(3):
                    hlo[i] = min(hlo[i], w[i]); hhi[i] = max(hhi[i], w[i])
                n += 1
        if n == 0:
            raise SystemExit(
                f"--frame-material '{args.frame_material}' に一致する面がない: "
                + str([m.name for o in meshes for m in o.data.materials if m]))
        band = {"by_material": args.frame_material, "vertices": n,
                "z_min": hlo.z, "z_max": hhi.z}
        lo, hi = hlo, hhi
    elif args.frame == "head":
        from mesh_landmarks import shoulder_split
        pts = [(( o.matrix_world @ v.co).x, (o.matrix_world @ v.co).z)
               for o in meshes for v in o.data.vertices]
        shoulder_z, _, conf = shoulder_split(pts, hi.z, lo.z, args.head_fraction)
        # 顎と首が切れないよう肩より少し下まで含める（project_face と同じ 0.023）
        z0 = max(lo.z, shoulder_z - (hi.z - lo.z) * 0.023)
        band = {"shoulder_z": shoulder_z, "z_min": z0, "confidence": conf}
        hlo = Vector((float("inf"),) * 3); hhi = Vector((float("-inf"),) * 3)
        for o in meshes:
            for v in o.data.vertices:
                w = o.matrix_world @ v.co
                if w.z >= z0:
                    for i in range(3):
                        hlo[i] = min(hlo[i], w[i]); hhi[i] = max(hhi[i], w[i])
        if hlo.x <= hhi.x:
            lo, hi = hlo, hhi

    # 正面から見えるのは X（左右）と Z（上下）。奥行きは Y。
    cx = (lo.x + hi.x) / 2.0
    cz = (lo.z + hi.z) / 2.0
    margin = args.margin
    if margin is None:
        margin = MATERIAL_MARGIN if args.frame_material else MARGIN
    span = max(hi.x - lo.x, hi.z - lo.z) * margin

    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.render.resolution_x = scn.render.resolution_y = args.res
    scn.eevee.taa_render_samples = 32
    scn.render.film_transparent = True

    w = bpy.data.worlds.new("W"); scn.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 2.6

    cam_d = bpy.data.cameras.new("C")
    cam_d.type = "ORTHO"
    cam_d.ortho_scale = span
    cam = bpy.data.objects.new("C", cam_d)
    scn.collection.objects.link(cam); scn.camera = cam
    # 顔は -Y を向いているので、カメラは -Y 側に置いて +Y を見る。
    cam.location = (cx, lo.y - max(1.0, hi.y - lo.y) * 2.0, cz)
    cam.rotation_euler = (math.pi / 2.0, 0.0, 0.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scn.render.filepath = args.out
    bpy.ops.render.render(write_still=True)

    # **メタは glTF 座標で書き出す。** 下流（trimesh / ICT-FaceKit）はどちらも
    # 「Y が上・+Z が前」なので、そこに合わせておけば以降で変換が要らない。
    # Blender (x,y,z) → glTF (x, z, -y)。
    #   X_gltf = cx + (u/res - 0.5) * span
    #   Y_gltf = cz - (v/res - 0.5) * span      （v は下向きなので反転）
    # 残る Z_gltf（奥行き）は、この (X, Y) を通る -Z 方向の光線を当てて求める。
    def to_gltf(v):
        return [v[0], v[2], -v[1]]
    g_lo = [min(a, b) for a, b in zip(to_gltf(lo), to_gltf(hi))]
    g_hi = [max(a, b) for a, b in zip(to_gltf(lo), to_gltf(hi))]
    meta = {
        "input": args.input, "output": args.out, "res": args.res,
        "frame": args.frame, "head_band": band,
        "ortho_span": span, "center_xy": [cx, cz], "margin": margin,
        "bounds_gltf": {"min": g_lo, "max": g_hi},
        "mapping": "X = cx + (u/res - 0.5)*span ; Y = cy - (v/res - 0.5)*span",
        "depth_axis": "Z", "depth_dir": -1,
        "note": "glTF 座標（Y上・+Z前）。ICT-FaceKit と同じ規約",
    }
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("ORTHO_OK " + json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
