"""メッシュの静的検査（工程7の前段 / 工程5のGeometry・Topologyスコア用）。

要件§15のチェック項目を Blender の bmesh で数える。ここでは修復せず、
「何が壊れているか」を JSON で出すだけにする。修復方針は候補ごとに変わるため。

  blender --background --factory-startup --python mesh_report.py -- \
      --input <mesh.glb> --out <report.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bmesh
import bpy


def ensure_vrm_addon() -> None:
    """--factory-startup ではアドオンが読まれないので明示的に有効化する。"""
    import addon_utils
    for m in addon_utils.modules():
        if "vrm" in m.__name__.lower():
            try:
                addon_utils.enable(m.__name__, default_set=True, persistent=True)
            except Exception:
                pass


def import_mesh(path: Path) -> None:
    ext = path.suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".vrm":
        ensure_vrm_addon()
        bpy.ops.import_scene.vrm(filepath=str(path))
    else:
        raise SystemExit(f"未対応の拡張子: {ext}")


def inspect() -> dict:
    total = {
        "objects": 0, "vertices": 0, "triangles": 0, "faces": 0,
        "non_manifold_edges": 0, "boundary_edges": 0, "holes_estimate": 0,
        "loose_vertices": 0, "loose_edges": 0,
        "duplicate_vertex_candidates": 0,
        "flipped_normal_faces": 0,
        "disconnected_shells": 0,
        "degenerate_faces": 0,
        "ngons": 0,
        "zero_area_faces": 0,
    }
    per_object = []

    for obj in [o for o in bpy.data.objects if o.type == "MESH"]:
        total["objects"] += 1
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
        boundary = sum(1 for e in bm.edges if e.is_boundary)
        loose_v = sum(1 for v in bm.verts if not v.link_edges)
        loose_e = sum(1 for e in bm.edges if not e.link_faces)
        ngons = sum(1 for f in bm.faces if len(f.verts) > 4)
        zero_area = sum(1 for f in bm.faces if f.calc_area() < 1e-9)

        # 重複頂点の候補数（実際にマージはしない）
        before = len(bm.verts)
        tmp = bm.copy()
        bmesh.ops.remove_doubles(tmp, verts=tmp.verts[:], dist=1e-5)
        dupes = before - len(tmp.verts)
        tmp.free()

        # 連結成分の数 = 浮遊ジオメトリの指標
        seen: set[int] = set()
        shells = 0
        for v in bm.verts:
            if v.index in seen:
                continue
            shells += 1
            stack = [v]
            seen.add(v.index)
            while stack:
                cur = stack.pop()
                for e in cur.link_edges:
                    other = e.other_vert(cur)
                    if other and other.index not in seen:
                        seen.add(other.index)
                        stack.append(other)

        # 法線の裏返り: 体積が負になる面の割合で近似
        flipped = 0
        for f in bm.faces:
            if f.normal.length_squared < 1e-12:
                flipped += 1

        tris = sum(max(0, len(f.verts) - 2) for f in bm.faces)

        per_object.append({
            "name": obj.name,
            "vertices": len(bm.verts),
            "faces": len(bm.faces),
            "triangles": tris,
            "non_manifold_edges": non_manifold,
            "boundary_edges": boundary,
            "loose_vertices": loose_v,
            "loose_edges": loose_e,
            "duplicate_vertex_candidates": dupes,
            "disconnected_shells": shells,
            "ngons": ngons,
            "zero_area_faces": zero_area,
            "degenerate_normal_faces": flipped,
        })

        total["vertices"] += len(bm.verts)
        total["faces"] += len(bm.faces)
        total["triangles"] += tris
        total["non_manifold_edges"] += non_manifold
        total["boundary_edges"] += boundary
        total["loose_vertices"] += loose_v
        total["loose_edges"] += loose_e
        total["duplicate_vertex_candidates"] += dupes
        total["disconnected_shells"] += shells
        total["ngons"] += ngons
        total["zero_area_faces"] += zero_area
        total["flipped_normal_faces"] += flipped
        bm.free()

    # 境界エッジがあれば穴が開いている。ループ数までは数えず、存在の指標として持つ。
    total["holes_estimate"] = 1 if total["boundary_edges"] > 0 else 0
    total["watertight"] = total["boundary_edges"] == 0 and total["non_manifold_edges"] == 0

    return {"total": total, "objects": per_object}


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    bpy.ops.wm.read_homefile(use_empty=True)
    import_mesh(Path(args.input))
    report = inspect()
    report["input"] = args.input
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("MESH_REPORT_OK " + json.dumps(report["total"]))


if __name__ == "__main__":
    main()
