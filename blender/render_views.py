"""Blender headless で候補メッシュを固定視点からレンダリングする（工程4）。

評価工程(工程5)が候補どうしを公平に比較できるよう、すべての候補で
カメラ・ライティング・露出・被写体スケールを厳密に統一する。
候補ごとに条件が変わると、スコア差がモデルの差ではなく描画条件の差になる。

使い方:
  blender --background --factory-startup --python render_views.py -- \
      --input <mesh.glb> --outdir <renders/cand_id> [--res 768] [--views front,left,...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

# 方位角(度)。0 = 正面。VRM は +Z 前方だが glTF 由来のメッシュは向きが不定なので、
# 「最も情報量の多い面」ではなく固定方位で撮る。比較の一貫性を優先する。
VIEW_ANGLES = {
    "front": (0.0, 0.0),
    "front_left": (-35.0, 0.0),
    "front_right": (35.0, 0.0),
    "left": (-90.0, 0.0),
    "right": (90.0, 0.0),
    "back": (180.0, 0.0),
    "three_quarter": (35.0, 15.0),
    "top": (0.0, 70.0),
}


def ensure_vrm_addon() -> None:
    """--factory-startup ではアドオンが読まれないので明示的に有効化する。"""
    import addon_utils
    for m in addon_utils.modules():
        if "vrm" in m.__name__.lower():
            try:
                addon_utils.enable(m.__name__, default_set=True, persistent=True)
            except Exception:
                pass


def clear_scene() -> None:
    bpy.ops.wm.read_homefile(use_empty=True)


def import_mesh(path: Path) -> None:
    ext = path.suffix.lower()
    if ext == ".glb" or ext == ".gltf":
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".obj":
        # Blender 4.x は新しい OBJ インポータ
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    elif ext == ".vrm":
        ensure_vrm_addon()
        bpy.ops.import_scene.vrm(filepath=str(path))
    else:
        raise SystemExit(f"未対応の拡張子: {ext}")


def scene_bounds() -> tuple[Vector, float]:
    """メッシュ全体の中心と外接半径。カメラ距離をここから決める。"""
    objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if not objs:
        raise SystemExit("メッシュが1つも読み込まれませんでした")
    lo = Vector((float("inf"),) * 3)
    hi = Vector((float("-inf"),) * 3)
    for o in objs:
        for corner in o.bound_box:
            w = o.matrix_world @ Vector(corner)
            for i in range(3):
                lo[i] = min(lo[i], w[i])
                hi[i] = max(hi[i], w[i])
    center = (lo + hi) / 2.0
    radius = max((hi - lo).length / 2.0, 1e-4)
    return center, radius


def setup_world() -> None:
    """三点照明 + 均一な環境光。影で形状が潰れないようにする。"""
    world = bpy.data.worlds.new("W")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 0.55

    def lamp(name, loc, energy):
        d = bpy.data.lights.new(name, type="AREA")
        d.energy = energy
        d.size = 4.0
        o = bpy.data.objects.new(name, d)
        o.location = loc
        bpy.context.collection.objects.link(o)
        return o

    lamp("key", (3.0, -4.0, 4.0), 900)
    lamp("fill", (-4.0, -2.0, 2.0), 350)
    lamp("rim", (0.0, 4.5, 3.0), 500)


def setup_camera(center: Vector, radius: float) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 60.0  # やや望遠。広角だと顔が歪んで identity 評価が不当に下がる
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    target = bpy.data.objects.new("Target", None)
    target.location = center
    bpy.context.collection.objects.link(target)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    return cam


def place_camera(cam, center: Vector, radius: float, azim_deg: float, elev_deg: float) -> None:
    dist = radius * 3.0
    a = math.radians(azim_deg)
    e = math.radians(elev_deg)
    cam.location = Vector((
        center.x + dist * math.sin(a) * math.cos(e),
        center.y - dist * math.cos(a) * math.cos(e),
        center.z + dist * math.sin(e),
    ))


def setup_render(res: int, samples: int, transparent: bool) -> None:
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT" if hasattr(
        bpy.types, "SceneEEVEE") and "BLENDER_EEVEE_NEXT" in [
            e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
        ] else "BLENDER_EEVEE"
    sc.render.resolution_x = res
    sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = transparent
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    try:
        sc.eevee.taa_render_samples = samples
    except Exception:
        pass
    sc.view_settings.view_transform = "Standard"  # 露出を候補間で揃える


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--views", default="front,left,right,back,three_quarter")
    ap.add_argument("--transparent", action="store_true", default=True)
    args = ap.parse_args(argv)

    mesh_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    import_mesh(mesh_path)
    center, radius = scene_bounds()
    setup_world()
    cam = setup_camera(center, radius)
    setup_render(args.res, args.samples, args.transparent)

    meta = {
        "input": str(mesh_path),
        "center": [center.x, center.y, center.z],
        "radius": radius,
        "resolution": args.res,
        "camera_lens_mm": 60.0,
        "camera_distance": radius * 3.0,
        "views": {},
        # 候補間で同一であることを評価工程が検証できるようにする
        "render_signature": f"lens60_dist3r_res{args.res}_std",
    }

    written = []
    for name in [v.strip() for v in args.views.split(",") if v.strip()]:
        if name not in VIEW_ANGLES:
            print(f"未知のビュー: {name} (スキップ)")
            continue
        azim, elev = VIEW_ANGLES[name]
        place_camera(cam, center, radius, azim, elev)
        out = outdir / f"{name}.png"
        bpy.context.scene.render.filepath = str(out)
        bpy.ops.render.render(write_still=True)
        meta["views"][name] = {"azimuth_deg": azim, "elevation_deg": elev,
                               "file": out.name}
        written.append(out.name)
        print(f"rendered {name} -> {out}")

    # 統計もここで出しておく（工程5のトポロジー評価が再計算せずに済む）
    tris = 0
    verts = 0
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        m = o.data
        m.calc_loop_triangles()
        tris += len(m.loop_triangles)
        verts += len(m.vertices)
    meta["polygon_count"] = tris
    meta["vertex_count"] = verts
    meta["material_count"] = len(bpy.data.materials)
    meta["image_count"] = len(bpy.data.images)
    meta["max_texture_size"] = max(
        [max(i.size[0], i.size[1]) for i in bpy.data.images if i.size[0]] or [0])

    (outdir / "render.json").write_text(json.dumps(meta, indent=2))
    print("RENDER_OK " + json.dumps({"views": written, "tris": tris,
                                     "verts": verts}))


if __name__ == "__main__":
    main()
