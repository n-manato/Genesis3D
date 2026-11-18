"""顔メッシュと体メッシュを接合する。

なぜ分けて作るか（docs/05-measurements.md §15〜§17）:

- TRELLIS.2 の顔ジオメトリは元画像を高解像度にしても ArcFace 0.38 前後で
  頭打ちになる。本人性は**実写を顔テクスチャに焼く**しか担保できない。
- ところが生成した全身Tポーズ画像は顔が 83px しかなく、そこから作った
  メッシュの顔は平坦で、実写を重ねるための顔検出自体が通らない。
- 実写のポートレートを単体で3D化すると顔はよく出る（ArcFace 0.68）が、
  脚も足も無いのでリギングが破綻する。

→ 顔は実写ポートレートから、体は生成Tポーズ画像から作り、ここで繋ぐ。

接合位置は形状から測る。肩で横幅が急に広がる点を頭部と胴の境目とみなし
（blender/mesh_landmarks.py）、頭側メッシュを「頭頂〜肩」の高さが揃うよう
相似変形してから、首のあたりで両方を切って結合する。

頂点のマージはしない。UV シームのために意図的に分割された頂点を座標だけで
繋ぐとテクスチャが壊れる（docs/05 §9 の実測）。首の内側で少し重ねて隙間を
隠す方式にしてある。実用アバターは watertight でなくてよい（同 §5）。

  blender --background --factory-startup --python blender/merge_head.py -- \
      --body <body.glb> --head <head.glb> --out <merged.glb>
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_landmarks import shoulder_split  # noqa: E402

MARKER = "MERGE_HEAD_OK"


def import_mesh(path: Path) -> list:
    before = set(bpy.data.objects)
    ext = path.suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        raise SystemExit(f"未対応の拡張子: {ext}")
    objs = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    # **マテリアルを持たないメッシュを外す。** Blender 4.2 の glTF インポータは
    # 読み込みのたびに 42頂点の `Icosphere` を作る（subject_a.glb でも Rocketbox でも
    # 再現）。これが混ざると外接が狂い、肩の検出も接合スケールも壊れる
    # （実測: 1.6m の体が 2.596m と測られ、スケールが 2.30 と 1.61 で 43% 食い違った）。
    objs = [o for o in objs if [m for m in o.data.materials if m]]
    if not objs:
        raise SystemExit(f"メッシュが読み込めませんでした: {path}")
    return objs


def apply_transforms(objs) -> None:
    """ローカル座標＝ワールド座標にする。以降の切断面の指定が単純になる。"""
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def world_points(objs):
    for o in objs:
        mw = o.matrix_world
        for v in o.data.vertices:
            w = mw @ v.co
            yield w


def bounds(objs) -> tuple[Vector, Vector]:
    lo = Vector((float("inf"),) * 3)
    hi = Vector((float("-inf"),) * 3)
    for w in world_points(objs):
        lo = Vector((min(lo[i], w[i]) for i in range(3)))
        hi = Vector((max(hi[i], w[i]) for i in range(3)))
    return lo, hi


def measure(objs) -> dict:
    """頭頂・肩・頭部幅・頭部帯の水平中心を測る。"""
    lo, hi = bounds(objs)
    pts = [(w.x, w.z) for w in world_points(objs)]
    shoulder_z, head_w, conf = shoulder_split(pts, hi.z, lo.z)

    # 頭部帯（肩より上）の水平中心。全体の重心だと腕や胴に引っ張られる。
    xs, ys = [], []
    for w in world_points(objs):
        if w.z > shoulder_z:
            xs.append(w.x)
            ys.append(w.y)
    cx = (min(xs) + max(xs)) / 2 if xs else (lo.x + hi.x) / 2
    cy = (min(ys) + max(ys)) / 2 if ys else (lo.y + hi.y) / 2

    # 頭部帯の中での最大横幅。頭頂付近の細いスライスと違い、
    # これは「その人の頭のいちばん広いところ」なので構図が違うメッシュ同士でも
    # 直接比較できる。スケール合わせの基準はこちらを使う。
    head_max_w = (max(xs) - min(xs)) if xs else 0.0
    return {"lo": lo, "hi": hi, "shoulder_z": shoulder_z, "head_width": head_w,
            "head_max_width": head_max_w,
            "confidence": conf, "head_center": Vector((cx, cy, 0.0)),
            "head_span": hi.z - shoulder_z, "height": hi.z - lo.z}


def cut_z(objs, z: float, keep: str) -> None:
    """平面 z で切って片側を捨てる。keep="below" なら下側を残す。

    シェイプキーを持つオブジェクトは**編集モードの bmesh で頂点を消す**。
    `bmesh.new()` + `bm.to_mesh()` の経路はシェイプキーのレイヤを持ち越さず、
    表情を持つ頭を切ると 53個のモーフが全部消える（docs/05 §44）。
    かといって切らずに飛ばすと、頭の切り出しに含まれる襟と肩が体に食い込んで
    半透明の当て布のような影を作る（§48）。編集モードの bmesh は
    シェイプキーを保ったまま削除できるので、そちらを使う。
    """
    for o in objs:
        if o.data.shape_keys is not None:
            _cut_z_keyed(o, z, keep)
            continue
        bm = bmesh.new()
        bm.from_mesh(o.data)
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(
            bm, geom=geom, dist=1e-6,
            plane_co=(0.0, 0.0, z), plane_no=(0.0, 0.0, 1.0),
            clear_outer=(keep == "below"),   # +Z 側を捨てる
            clear_inner=(keep == "above"),   # -Z 側を捨てる
        )
        bm.to_mesh(o.data)
        bm.free()
        o.data.update()


def _cut_z_keyed(o, z: float, keep: str) -> None:
    """シェイプキーを保ったまま平面 z の片側を消す。

    切り口は補間せず頂点単位で落とす。頭と体は首の内側で重ねる設計なので、
    数ミリの粗さは体のジャケットに隠れる。
    """
    import numpy as np
    n = len(o.data.vertices)
    co = np.empty(n * 3)
    o.data.vertices.foreach_get("co", co)
    world = co.reshape(-1, 3) @ np.array(o.matrix_world)[:3, :3].T \
        + np.array(o.matrix_world)[:3, 3]
    zs = world[:, 2]
    doomed = np.nonzero(zs > z if keep == "below" else zs < z)[0]
    if len(doomed) == 0 or len(doomed) == n:
        print(f"cut_z: {o.name} は z={z:.4f} で切っても変化なし", flush=True)
        return
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(o.data)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[int(i)] for i in doomed], context="VERTS")
    bmesh.update_edit_mesh(o.data)
    bpy.ops.object.mode_set(mode="OBJECT")
    keys = len(o.data.shape_keys.key_blocks) - 1 if o.data.shape_keys else 0
    print(f"cut_z: {o.name} を z={z:.4f} で切った "
          f"({n} → {len(o.data.vertices)}頂点, シェイプキー {keys}個を保持)", flush=True)


def skin_head(merged, arm, mb, head_bone: str, neck_bone: str, blend: float) -> dict:
    """接合した顔メッシュを、体のアーマチュアに追従させる。

    TRELLIS が作る頭はウェイトを持たないので、そのままだと**腕を動かしても
    頭が置き去りになる**。頭は実際にはほぼ剛体なので、`Head` ボーンに
    まるごと預けてよい。首のあたりだけ `Neck` と重みを分け合わせて、
    首を回したときに段差が出ないようにする。

    体側のウェイトは触らない（手塗りのものが入っている）。
    """
    names = {b.name for b in arm.data.bones}
    for n in (head_bone, neck_bone):
        if n not in names:
            raise SystemExit(f"アーマチュアに '{n}' が無い: {sorted(names)[:12]}…")

    mark = merged.vertex_groups.get("__head__")
    if mark is None:
        raise SystemExit("__head__ グループが無い（印付けに失敗した）")
    gh = merged.vertex_groups.get(head_bone) or merged.vertex_groups.new(name=head_bone)
    gn = merged.vertex_groups.get(neck_bone) or merged.vertex_groups.new(name=neck_bone)

    band = mb["height"] * blend
    z0 = mb["shoulder_z"]
    n_head = 0
    for v in merged.data.vertices:
        if not any(g.group == mark.index for g in v.groups):
            continue
        n_head += 1
        z = (merged.matrix_world @ v.co).z
        # 肩の高さから band ぶん上までを混合帯にする。下ほど Neck に預ける。
        t = min(max((z - z0) / band, 0.0), 1.0)
        # 体側のウェイトが乗っている場合に備えて REPLACE で上書きする
        gh.add([v.index], float(t), "REPLACE")
        gn.add([v.index], float(1.0 - t), "REPLACE")
    merged.vertex_groups.remove(mark)

    if merged.parent is None:
        merged.parent = arm
    if not any(m.type == "ARMATURE" for m in merged.modifiers):
        m = merged.modifiers.new("Armature", "ARMATURE")
        m.object = arm
    else:
        for m in merged.modifiers:
            if m.type == "ARMATURE":
                m.object = arm
    return {"head_vertices_skinned": n_head, "head_bone": head_bone,
            "neck_bone": neck_bone, "blend_band_m": round(band, 4)}


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, help="全身メッシュ（脚と足があるもの）")
    ap.add_argument("--head", required=True, help="顔メッシュ（実写から作ったもの）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neck-offset", type=float, default=0.012,
                    help=("**体側**をどこで切るか。肩から上へ全高の何割の位置。"
                          "既定 0.012 は首の中ほど。0.030 だと切断面が顎の線に来て"
                          "帯として見え、0.065 以上では顔を切ってしまう（実測）"))
    ap.add_argument("--head-offset", type=float, default=-0.055,
                    help=("**顔側**をどこで切るか。肩から上へ全高の何割の位置。"
                          "負の値＝肩より下まで残す。ボブの毛先は肩の高さより下に"
                          "垂れるので、体側と同じ位置で切ると髪が削られる"
                          "（実測: 接合前は豊かだった髪が接合後に薄くなっていた）。"
                          "顔メッシュの胴は元々細いので、多少残っても体に隠れる"))
    ap.add_argument("--overlap", type=float, default=0.012,
                    help="首の内側で重ねる帯の幅（全高比）。隙間を隠すため")
    ap.add_argument("--keep-rig", action="store_true",
                    help="**体のアーマチュアとウェイトを保つ。** 既定では捨てる"
                         "（旧パイプラインは接合の後に autorig でリギングしていた）。"
                         "手塗りウェイトを持つテンプレート素体に載せる場合は必須")
    ap.add_argument("--head-bone", default="Head",
                    help="--keep-rig のとき、顔メッシュを追従させるボーン")
    ap.add_argument("--neck-bone", default="Neck",
                    help="首の帯で重みを分け合うボーン")
    ap.add_argument("--neck-blend", type=float, default=0.05,
                    help="首の帯の幅（全高比）。ここで Head と Neck の重みを混ぜる")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    body = import_mesh(Path(args.body))
    # 体と一緒に入ってきたアーマチュア。--keep-rig のときこれを残す。
    arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    head = import_mesh(Path(args.head))
    apply_transforms(body + head)

    mb = measure(body)
    mh = measure(head)
    if mb["confidence"] < 1.0:
        print("警告: 体メッシュの肩を検出できませんでした（既定割合で切ります）",
              flush=True)
    if mh["confidence"] < 1.0:
        print("警告: 顔メッシュの肩を検出できませんでした（既定割合で切ります）",
              flush=True)

    # 「頭頂〜肩」の高さを揃える相似変形。頭部の横幅ではなくこの区間を使うのは、
    # 体側の頭は作り直す対象で形が当てにならないのに対し、肩の位置は
    # 胴の形状から決まるので安定しているため。
    if mh["head_max_width"] <= 1e-9 or mh["head_span"] <= 1e-9:
        raise SystemExit("顔メッシュの頭部が測れませんでした")

    # **頭部の最大幅で合わせる。** 「頭頂〜肩」で合わせると、その区間は
    # 頭＋首の合計なので、首の長さが違う2つのメッシュでは頭の大きさが揃わない。
    # 実測では頭が小さくなりすぎて顎が肩の近くまで下がり、首だけが長く
    # 飛び出した（切る高さを上げると今度は顔を切ってしまう）。
    # 頭の横幅は同じ人物なら一致するはずの量なので、こちらを基準にする。
    scale = mb["head_max_width"] / mh["head_max_width"]
    by_span = mb["head_span"] / mh["head_span"]

    # 肩の検出がずれるとスケールが丸ごと狂い、顎が切り落とされた頭になる。
    # 頭頂付近の横幅から出したスケールと突き合わせて、大きく食い違ったら警告する
    # （実測: 顔メッシュに胴が入りすぎると頭部帯が 0.563 → 0.344 に縮み、
    #  スケールが 1.64 倍ずれた）。値は上書きしない — どちらが正しいかは
    #  入力次第で、黙って別の値に差し替える方が危ない。
    # 閾値 0.5 では実測の食い違い（0.470 対 0.716 = 0.34）を見逃した。
    # 髪が広がった写真だと肩検出が髪の膨らみを肩と誤認し、頭部帯が
    # 0.563 → 0.218 に縮んでスケールが 1.5 倍狂う。0.25 まで下げる。
    if abs(scale / by_span - 1.0) > 0.25:
        print(f"警告: スケールの2つの推定が食い違っています "
              f"（頭幅基準 {scale:.4f} / 頭部帯基準 {by_span:.4f}）。"
              f"肩の検出がずれている可能性があります", flush=True)

    for o in head:
        o.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    apply_transforms(head)

    mh2 = measure(head)
    delta = Vector((
        mb["head_center"].x - mh2["head_center"].x,
        mb["head_center"].y - mh2["head_center"].y,
        mb["shoulder_z"] - mh2["shoulder_z"],
    ))
    for o in head:
        o.location = delta
    bpy.context.view_layer.update()
    apply_transforms(head)

    # 首の途中で切る。体は上を捨て、顔は下を捨てる。頂点マージはしない
    # （UV が壊れるため）。
    #
    # **体側と顔側で切る高さを分ける。** 同じ高さで切ると、肩より下に垂れる
    # 髪（ボブの毛先）まで削れてしまう。顔側を低く切って髪を残し、体側は
    # 首の途中で切る。重なった部分は顔メッシュの細い胴が体に隠れるだけで、
    # 髪は体の外側に出るので自然に見える。
    h = mb["height"]
    body_seam = mb["shoulder_z"] + h * args.neck_offset
    head_seam = mb["shoulder_z"] + h * args.head_offset
    cut_z(body, body_seam + h * args.overlap / 2.0, keep="below")
    cut_z(head, head_seam, keep="above")

    # **結合はシェイプキーを持つ側をアクティブにする。** join はアクティブ
    # オブジェクトのシェイプキーを残し、他方は Basis に取り込む。逆にすると
    # 頭の表情が消える。
    # **接合の前に顔側の頂点へ印を付ける。** join 後は頂点の並びが
    # アクティブオブジェクト次第で変わるので、番号では追えない。
    # 頂点グループは join をまたいで残るので、これが確実。
    if args.keep_rig:
        for o in head:
            g = o.vertex_groups.new(name="__head__")
            g.add(range(len(o.data.vertices)), 1.0, "REPLACE")

    keyed = [o for o in head + body if o.data.shape_keys is not None]
    bpy.ops.object.select_all(action="DESELECT")
    for o in body + head:
        o.select_set(True)
    bpy.context.view_layer.objects.active = keyed[0] if keyed else body[0]
    bpy.ops.object.join()
    merged = bpy.context.view_layer.objects.active

    if args.keep_rig:
        if arm is None:
            raise SystemExit("--keep-rig を指定したが体にアーマチュアが無い")
        skin_head(merged, arm, mb, args.head_bone, args.neck_bone, args.neck_blend)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    merged.select_set(True)
    if args.keep_rig and arm is not None:
        arm.select_set(True)          # 選択に入れないとスキンが落ちる
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB",
                              use_selection=True, export_image_format="AUTO",
                              export_morph=True,
                              # モーフの法線・接線は書き出さない。53本ぶんで
                              # 50MB になるうえ、KEHAI は位置しか駆動しない。
                              export_morph_normal=False,
                              export_morph_tangent=False)

    lo, hi = bounds([merged])
    report = {
        "body": args.body, "head": args.head, "output": str(out),
        "body_landmarks": {"height": round(mb["height"], 5),
                           "shoulder_z": round(mb["shoulder_z"], 5),
                           "head_span": round(mb["head_span"], 5),
                           "head_width": round(mb["head_width"], 5),
                           "head_max_width": round(mb["head_max_width"], 5),
                           "detected": mb["confidence"] == 1.0},
        "head_landmarks": {"height": round(mh["height"], 5),
                           "head_span": round(mh["head_span"], 5),
                           "head_width": round(mh["head_width"], 5),
                           "head_max_width": round(mh["head_max_width"], 5),
                           "detected": mh["confidence"] == 1.0},
        "applied_scale": round(scale, 5),
        "scale_by_span": round(by_span, 5),
        "body_seam_z": round(body_seam, 5),
        "head_seam_z": round(head_seam, 5),
        "overlap": args.overlap,
        "merged": {"vertices": len(merged.data.vertices),
                   "polygons": len(merged.data.polygons),
                   "materials": [m.name for m in merged.data.materials if m],
                   "height": round(hi.z - lo.z, 5)},
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(MARKER + " " + json.dumps(
        {"output": str(out), "scale": report["applied_scale"],
         "verts": report["merged"]["vertices"],
         "materials": len(report["merged"]["materials"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
