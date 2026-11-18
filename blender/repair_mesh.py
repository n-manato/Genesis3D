"""工程7: メッシュ修復。

要件§15のうち「安全に自動化できるもの」だけを行う。顔のディテールを失う
強い Decimation は禁止（要件§15）なので、面数削減はここでは一切しない。

行うこと:
  - 浮遊ジオメトリの除去（本体に対して十分小さい連結成分のみ）
  - 重複頂点のマージ
  - 法線の再計算（外向きに統一）
  - 面積ゼロの面・孤立頂点/辺の除去

行わないこと:
  - Decimation / Remesh（顔が潰れる）
  - 穴埋め（image-to-3D の穴は下着や口内など「閉じてはいけない」場所にも空く）

  blender --background --factory-startup --python repair_mesh.py -- \
      --input <mesh.glb> --out <clean.glb> --report <repair.json> \
      [--speck-ratio 0.005]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bmesh
import bpy


def ensure_vrm_addon() -> None:
    import addon_utils
    for m in addon_utils.modules():
        if "vrm" in m.__name__.lower():
            try:
                addon_utils.enable(m.__name__, default_set=True, persistent=True)
            except Exception:
                pass


def import_any(path: Path) -> None:
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


def shells(bm, weld: float = 1e-5) -> list[list]:
    """連結成分ごとの面リスト。**位置で溶接した上で**数える。

    辺の共有だけで数えると、UV シームのために分割された頂点で成分が切れる。
    実測: 成果物の VRM は素直に数えると 2,159 成分・最大でも全体の 3.3% だが、
    距離 1e-5 で溶接すると **37 成分・最大 90.8%** になる。前者は「バラバラの
    破片の山」に見えるが、実体は 1つの胴体である（docs/05-measurements.md §5 が
    警告していた「実用アバターは watertight ではない」の正体がこれ）。

    UV 分割のまま面積比で浮遊片を判定すると、基準になる「本体」が全体の 3% しか
    ないので閾値が極端に小さくなり、**正常な表面の小さな UV 島を削って穴を開け、
    本物のゴミは残す**。実測で面の 8.28% を失っていた。

    ここでは頂点座標を格子に量子化して溶接IDを作り、それを共有する面同士を
    繋いで成分を出す。メッシュ自体は一切変更しない（UV は無傷）。
    """
    inv = 1.0 / max(weld, 1e-12)
    weld_id: dict[tuple[int, int, int], int] = {}
    vert_weld: dict[int, int] = {}
    for v in bm.verts:
        key = (int(round(v.co.x * inv)), int(round(v.co.y * inv)),
               int(round(v.co.z * inv)))
        wid = weld_id.setdefault(key, len(weld_id))
        vert_weld[v.index] = wid

    # 溶接ID -> その位置に触れている面
    by_weld: dict[int, list] = {}
    for f in bm.faces:
        for v in f.verts:
            by_weld.setdefault(vert_weld[v.index], []).append(f)

    seen: set[int] = set()
    groups: list[list] = []
    for f in bm.faces:
        if f.index in seen:
            continue
        group = []
        stack = [f]
        seen.add(f.index)
        while stack:
            cur = stack.pop()
            group.append(cur)
            for v in cur.verts:
                for nf in by_weld.get(vert_weld[v.index], ()):
                    if nf.index not in seen:
                        seen.add(nf.index)
                        stack.append(nf)
        groups.append(group)
    return groups


def repair(obj, speck_ratio: float, merge_dist: float,
           recalc_normals: bool = False) -> dict:
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    before = {"verts": len(bm.verts), "faces": len(bm.faces)}

    # 1) 浮遊ジオメトリ: 本体面積の speck_ratio 未満の連結成分を落とす。
    #    髪の房やアクセサリを消さないよう、閾値は保守的に取る。
    groups = shells(bm)
    areas = [sum(f.calc_area() for f in g) for g in groups]
    removed_shells = 0
    if areas:
        main = max(areas)
        doomed = []
        for g, a in zip(groups, areas):
            if a < main * speck_ratio:
                doomed.extend(g)
                removed_shells += 1
        if doomed:
            bmesh.ops.delete(bm, geom=doomed, context="FACES")

    # 2) 重複頂点のマージは既定で行わない。
    #    テクスチャ付きメッシュでは UV シームのために頂点が意図的に分割されており、
    #    座標だけを見てマージすると UV が壊れる。実測で Tシャツ全面に黒斑が出た。
    if merge_dist > 0:
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=merge_dist)

    # 3) 面積ゼロの面
    bm.faces.ensure_lookup_table()
    zero = [f for f in bm.faces if f.calc_area() < 1e-9]
    if zero:
        bmesh.ops.delete(bm, geom=zero, context="FACES")

    # 4) 孤立した頂点・辺
    loose_v = [v for v in bm.verts if not v.link_edges]
    if loose_v:
        bmesh.ops.delete(bm, geom=loose_v, context="VERTS")
    bm.edges.ensure_lookup_table()
    loose_e = [e for e in bm.edges if not e.link_faces]
    if loose_e:
        bmesh.ops.delete(bm, geom=loose_e, context="EDGES")

    # 5) 法線の再計算も既定では行わない。image-to-3D の出力は法線が正しいことが
    #    多く、断片化したメッシュに recalc をかけると逆に一部が裏返る。

    if recalc_normals:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    bm.to_mesh(me)
    bm.free()
    me.update()

    return {
        "object": obj.name,
        "verts_before": before["verts"], "verts_after": len(me.vertices),
        "faces_before": before["faces"], "faces_after": len(me.polygons),
        "shells_removed": removed_shells,
    }


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    # 実測: 0.005 では表面の一部である小片まで削り、Tシャツに穴が開いた。
    # TRELLIS.2 の出力は表面自体が細かく分断されているため、閾値は桁で小さく取る。
    ap.add_argument("--speck-ratio", type=float, default=0.0002,
                    help="本体面積に対するこの比率未満の連結成分を浮遊物として除去")
    ap.add_argument("--merge-dist", type=float, default=0.0,
                    help="0 なら重複頂点マージを行わない（UV破壊を避けるため既定は0）")
    ap.add_argument("--recalc-normals", action="store_true",
                    help="法線を再計算する（断片化メッシュでは裏返ることがある）")
    args = ap.parse_args(argv)

    bpy.ops.wm.read_homefile(use_empty=True)
    import_any(Path(args.input))

    results = []
    for obj in [o for o in bpy.data.objects if o.type == "MESH"]:
        results.append(repair(obj, args.speck_ratio, args.merge_dist,
                              args.recalc_normals))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB")

    report = {
        "input": args.input, "output": str(out),
        "speck_ratio": args.speck_ratio, "merge_dist": args.merge_dist,
        "objects": results,
        "totals": {
            "verts_before": sum(r["verts_before"] for r in results),
            "verts_after": sum(r["verts_after"] for r in results),
            "faces_before": sum(r["faces_before"] for r in results),
            "faces_after": sum(r["faces_after"] for r in results),
            "shells_removed": sum(r["shells_removed"] for r in results),
        },
        "note": "Decimation / Remesh / 穴埋めは意図的に行っていない",
    }
    t = report["totals"]
    lost = 1.0 - (t["faces_after"] / max(1, t["faces_before"]))
    report["face_loss_ratio"] = round(lost, 4)
    # 削りすぎは穴として現れる。閾値を超えたら「修復した」と主張しない。
    # 閾値は実測で決めた: 5.7%除去は目視で劣化なし、18.8%はTシャツに穴が開いた。
    report["status"] = "ok" if lost <= 0.10 else "excessive_removal"
    if lost > 0.10:
        report["warning"] = (
            f"面の {lost*100:.1f}% を除去した。表面の一部を削っている可能性が高い。"
            "--speck-ratio を下げるか修復をスキップすること")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("REPAIR_OK " + json.dumps({**t, "face_loss_ratio": report["face_loss_ratio"],
                                     "status": report["status"]}))


if __name__ == "__main__":
    main()
