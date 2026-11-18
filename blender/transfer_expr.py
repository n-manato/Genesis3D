"""ICT-FaceKit の表情を、既存の頭部メッシュへ移植する。

ICT の幾何をそのまま使うと本人に見えなくなる（正面 ArcFace 0.708 → 0.394、
docs/05 §42）。原因は、現行の頭が**彼女の写真から起こした形**なのに対し、
ICT は 4000スキャンの100次元線形 PCA で、表面を 1.3mm まで寄せても顔の比率までは
再現しきれないこと。

そこで **幾何は現行のまま残し、表情の差分だけを借りる**。
`ict_fit.py` が当て込んだ ICT 素体は対象の表面に 1.3mm まで寄っているので、
対象の各頂点について「ICT 表面上の最も近い点」を求め、その三角形の重心座標で
53個の差分を補間すれば、対象の頂点がどう動くべきかが決まる。

対応が遠い頂点（髪・肩・襟など、ICT に対応物がない部分）は動かさない。
距離で滑らかに減衰させないと、髪の生え際に段差が出る。

  blender --background --factory-startup --python transfer_expr.py -- \
      --target <head.glb> --ict-neutral <fitted.obj> --fit <fit.json> \
      --out <head_expr.glb> --report <report.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ICT_DIR = Path("vendor/ICT-FaceKit/FaceXModel")
# この距離まで対応を信用する。これを超えたら 0 へ滑らかに落とす。
# 単位は当て込み後の世界座標で、頭+肩の高さが 1.0。0.012 は実寸およそ 5mm。
FALLOFF_NEAR = 0.012
FALLOFF_FAR = 0.030
MIN_DELTA = 1.0e-3
MERGE_LR = {"browInnerUp", "cheekPuff"}
SKIP_EXPR = ("PupilDilate",)     # 眼球が無いメッシュには意味がない


def verts_of(path: Path) -> np.ndarray:
    return np.array([[float(x) for x in l.split()[1:4]]
                     for l in path.read_text(errors="ignore").splitlines()
                     if l.startswith("v ")], dtype=np.float64)


def arkit_name(n: str) -> str:
    if n.endswith("_L"):
        return n[:-2] + "Left"
    if n.endswith("_R"):
        return n[:-2] + "Right"
    return n


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
    ap.add_argument("--target", required=True, help="表情を付けたい頭部 (glb)")
    ap.add_argument("--ict-neutral", required=True,
                    help="ict_fit.py の --out-mesh（当て込んだ ICT 素体 obj）")
    ap.add_argument("--fit", required=True, help="ict_fit.py の json（相似変換を使う）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--replace-keys", action="store_true",
                    help="対象に既にあるシェイプキーを**捨ててから**転写する。"
                         "接合後のメッシュにはテンプレート由来の空のキーが残っており、"
                         "名前が衝突して転写分が `eyeBlinkLeft.001` になる"
                         "（KEHAI は生の名前を探すので動かない）")
    ap.add_argument("--head-bone", default="Head",
                    help="ICT から足す歯・舌を追従させるボーン")
    ap.add_argument("--report")
    ap.add_argument("--ict-parts", nargs="*",
                    default=["M_Teeth", "M_GumsTongue"],
                    help=("ICT からそのまま持ち込むパーツ。既定は歯と歯茎・舌 — "
                          "対象メッシュは口が閉じた殻なので、jawOpen で開けると"
                          "中が空洞になり暗い穴が見える"))
    ap.add_argument("--near", type=float, default=FALLOFF_NEAR)
    ap.add_argument("--far", type=float, default=FALLOFF_FAR)
    args = ap.parse_args(argv)

    bpy.ops.wm.read_homefile(use_empty=True)

    tgt = import_mesh(Path(args.target))
    if not tgt:
        raise SystemExit("対象メッシュがありません")
    # 変換をデータへ焼いて、以降ワールド座標＝ローカル座標にする
    bpy.ops.object.select_all(action="DESELECT")
    for o in tgt:
        o.select_set(True)
    bpy.context.view_layer.objects.active = tgt[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    ict_objs = import_mesh(Path(args.ict_neutral))
    if not ict_objs:
        raise SystemExit("ICT 素体を読めません")
    bpy.ops.object.select_all(action="DESELECT")
    for o in ict_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = ict_objs[0]
    if len(ict_objs) > 1:
        bpy.ops.object.join()
    ict = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # ICT の頂点は obj の並びのまま（自前で書き出しているので割れていない）
    n_ict = len(ict.data.vertices)

    # 表情の差分を作る。相似変換の回転とスケールだけ掛け、Blender 座標へ直す。
    fit = json.loads(Path(args.fit).read_text())
    s = float(fit["similarity"]["scale"])
    R = np.array(fit["similarity"]["rotation"], dtype=np.float64)
    neutral_raw = verts_of(ICT_DIR / "generic_neutral_mesh.obj")

    def to_blender(a):
        return np.stack([a[..., 0], -a[..., 2], a[..., 1]], axis=-1)

    named: dict[str, np.ndarray] = {}
    for p in sorted(ICT_DIR.glob("*.obj")):
        if p.stem.startswith("identity") or p.stem == "generic_neutral_mesh":
            continue
        if any(k in p.stem for k in SKIP_EXPR):
            continue
        d = to_blender((s * (R @ (verts_of(p) - neutral_raw).T)).T)
        base = p.stem[:-2] if p.stem.endswith(("_L", "_R")) else p.stem
        if base in MERGE_LR:
            named[base] = named.get(base, 0.0) + d
        else:
            named[arkit_name(p.stem)] = d

    # --- 対応: 対象の各頂点 → ICT 表面の最近傍点（三角形と重心座標）
    tri = [list(p.vertices) for p in ict.data.polygons]
    ict_co = np.array([v.co[:] for v in ict.data.vertices])

    # **ワールド座標で突き合わせること。** `closest_point_on_mesh` は対象の
    # **ローカル**座標を期待する。対象がアーマチュアに親子付けされていると
    # ローカルとワールドがずれ、最近傍距離が桁で狂う
    # （実測: リグ付きの体に接合した後で 48.5m と出て、全頂点が falloff の外に
    #  落ち、主メッシュへの転写が 0 個になった。旧パイプラインは接合前の
    #  リグ無しメッシュを渡していたので local == world で偶然成立していた）。
    to_ict = ict.matrix_world.inverted()

    stats = {"vertices": 0, "corresponded": 0, "far": 0}
    for obj in tgt:
        # 対象ローカル → ワールド → ICT ローカル
        to_ict_from_obj = to_ict @ obj.matrix_world
        # 変位は ICT 空間で出るので、対象ローカルへ戻すのに回転とスケールだけ使う
        disp_to_obj = (obj.matrix_world.inverted() @ ict.matrix_world).to_3x3()
        n = len(obj.data.vertices)
        stats["vertices"] += n
        w = np.zeros(n)
        bary = np.zeros((n, 3))
        idx = np.zeros((n, 3), dtype=np.int64)
        for i, v in enumerate(obj.data.vertices):
            q = to_ict_from_obj @ v.co
            ok, loc, _, fi = ict.closest_point_on_mesh(q)
            if not ok or fi < 0:
                continue
            f = tri[fi]
            if len(f) < 3:
                continue
            a, b, c = (Vector(ict_co[f[0]]), Vector(ict_co[f[1]]), Vector(ict_co[f[2]]))
            # 重心座標（面積比）
            n0 = (b - a).cross(c - a)
            A = n0.length
            if A < 1e-12:
                continue
            u = (b - loc).cross(c - loc).length / A
            vv = (c - loc).cross(a - loc).length / A
            ww = max(0.0, 1.0 - u - vv)
            d = (Vector(loc) - q).length
            if d >= args.far:
                stats["far"] += 1
                continue
            t = 1.0 if d <= args.near else 1.0 - (d - args.near) / (args.far - args.near)
            w[i] = t
            bary[i] = (u, vv, ww)
            idx[i] = (f[0], f[1], f[2])
            stats["corresponded"] += 1

        if w.max() <= 0.0:
            continue
        if args.replace_keys and obj.data.shape_keys:
            # 名前の衝突を避ける。接合後は素体の空キーが残っていて、
            # そのままだと転写分に .001 が付き、KEHAI が見つけられない。
            obj.shape_key_clear()
        base_key = obj.shape_key_add(name="Basis", from_mix=False)
        made = 0
        for key_name, D in sorted(named.items()):
            disp = (D[idx[:, 0]] * bary[:, 0:1] + D[idx[:, 1]] * bary[:, 1:2]
                    + D[idx[:, 2]] * bary[:, 2:3]) * w[:, None]
            if float(np.abs(disp).max()) < MIN_DELTA:
                continue
            sk = obj.shape_key_add(name=key_name, from_mix=False)
            for i in range(n):
                if w[i] > 0.0:
                    sk.data[i].co = base_key.data[i].co + disp_to_obj @ Vector(disp[i])
            made += 1
        obj["shape_keys"] = made

    # --- ICT からそのまま持ち込むパーツ（歯・歯茎・舌）
    added = []
    if args.ict_parts:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import ict_head as IH
        raw_v, raw_uv, groups = IH.parse_obj(ICT_DIR / "generic_neutral_mesh.obj")
        basis_cache = Path("vendor/ICT-FaceKit/identity_basis.npz")
        basis = (np.load(basis_cache)["basis"] if basis_cache.exists()
                 else np.stack([verts_of(ICT_DIR / f"identity{k:03d}.obj") - raw_v
                                for k in range(100)]))
        alpha = np.array(fit["alpha"], dtype=np.float64)
        t = np.array(fit["similarity"]["translation"], dtype=np.float64)
        shaped = raw_v + np.tensordot(alpha, basis, axes=(0, 0))
        Vp = to_blender((s * (R @ shaped.T)).T + t)
        for mat_name in args.ict_parts:
            if mat_name not in groups:
                continue
            faces = groups[mat_name]
            keep = np.unique(np.concatenate([[vi for vi, _ in f] for f in faces]))
            obj, remap = IH.build_object(mat_name, mat_name, faces, Vp, raw_uv, keep)
            bk = obj.shape_key_add(name="Basis", from_mix=False)
            n_keys = 0
            for key_name, D in sorted(named.items()):
                dk = D[keep]
                if float(np.abs(dk).max()) < MIN_DELTA:
                    continue
                sk = obj.shape_key_add(name=key_name, from_mix=False)
                for old, new in remap.items():
                    sk.data[new].co = bk.data[new].co + Vector(D[old])
                n_keys += 1
            obj["shape_keys"] = n_keys
            added.append(obj)
            tgt.append(obj)

    bpy.data.objects.remove(ict, do_unlink=True)

    # **ICT から持ち込んだ歯・歯茎・舌にもスキンを付ける。** 付けないと
    # 首を回したときに口の中だけ空間に取り残される。頭は剛体なので
    # `Head` ボーンに丸ごと預けてよい。
    arm_obj = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    skinned_parts = []
    if arm_obj is not None and added:
        bone = args.head_bone
        if bone not in {b.name for b in arm_obj.data.bones}:
            print(f"警告: アーマチュアに '{bone}' が無いので ICT の部品を"
                  f"スキンできない", file=sys.stderr)
        else:
            for o in added:
                g = o.vertex_groups.get(bone) or o.vertex_groups.new(name=bone)
                g.add(range(len(o.data.vertices)), 1.0, "REPLACE")
                if o.parent is None:
                    o.parent = arm_obj
                if not any(m.type == "ARMATURE" for m in o.modifiers):
                    m = o.modifiers.new("Armature", "ARMATURE")
                    m.object = arm_obj
                skinned_parts.append(o.name)

    bpy.ops.object.select_all(action="DESELECT")
    for o in tgt:
        o.select_set(True)
    # **アーマチュアも選択に入れる。** 入れないとスキンが落ちる。
    # このスクリプトは元々「リギングの前」に走る前提だったが、
    # `game_avatar` では手塗りウェイトを持つ体に接合した後で走らせるので、
    # ここで落とすと指も肩も動かなくなる（実測: skins=1 → 0）。
    n_arm = 0
    for o in bpy.data.objects:
        if o.type == "ARMATURE":
            o.select_set(True)
            n_arm += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=args.out, export_format="GLB",
                              use_selection=True, export_morph=True,
                              export_morph_normal=False, export_morph_tangent=False)

    rep = {"target": args.target, "output": args.out, "armatures_kept": n_arm,
           "expressions_available": len(named),
           "ict_parts_added": [o.name for o in added],
           "ict_parts_skinned": skinned_parts,
           "falloff": [args.near, args.far], **stats,
           "objects": [{"name": o.name, "verts": len(o.data.vertices),
                        "shape_keys": o.get("shape_keys", 0)} for o in tgt]}
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print("EXPRTRANSFER_OK " + json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
