"""リグ済みメッシュ → VRM 1.0（工程9）+ 検証（工程10）。

VRM Humanoid は「必須ボーンが揃っていること」が全て。UniRig が生成した
スケルトンのボーン名は VRM の規約名と一致しないので、ここで対応付ける。

  blender --background --factory-startup --python export_vrm.py -- \
      --input <rigged.glb> --out <character.vrm> --report <validation.json> \
      [--name "Character"] [--author "Genesis3D"]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import bpy

# VRM 1.0 Humanoid の必須ボーン。これが1つでも欠けると VRM として成立しない。
REQUIRED_BONES = [
    "hips", "spine", "chest", "neck", "head",
    "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot",
    "rightUpperLeg", "rightLowerLeg", "rightFoot",
]
# あると品質が上がるが必須ではないもの
OPTIONAL_BONES = ["upperChest", "leftShoulder", "rightShoulder",
                  "leftToes", "rightToes", "leftEye", "rightEye", "jaw"]

# ボーン名から (左右, 中核名) を構造的に取り出して照合する。
# VRoid は J_Bip_C_Hips / J_Bip_L_UpperArm、Mixamo は mixamorig:LeftUpperArm、
# Blender 慣習は UpperArm.L のように、流儀ごとに左右の位置が違う。
# 文字列の末尾から l/r を削るような場当たり的処理は Shoulder→rightShoulde の
# ような誤変換を生むので、接頭辞・接尾辞を明示的に剥がす。

# 中核名の別名（左右に依存しない部分）
CORE_ALIASES: dict[str, list[str]] = {
    "hips":       ["hips", "hip", "pelvis", "waist", "センター", "下半身"],
    "spine":      ["spine", "spine1", "spine01", "abdomen", "上半身"],
    "chest":      ["chest", "spine2", "spine02", "torso", "上半身2"],
    "upperchest": ["upperchest", "spine3", "spine03"],
    "neck":       ["neck", "首"],
    "head":       ["head", "頭"],
    "shoulder":   ["shoulder", "clavicle", "肩"],
    "upperarm":   ["upperarm", "arm", "腕"],
    "lowerarm":   ["lowerarm", "forearm", "elbow", "ひじ"],
    "hand":       ["hand", "wrist", "手首"],
    # **Mixamo は "UpLeg"=腿 / "Leg"=脛。** 別名表の "leg" を upperleg に入れて
    # いたので、Rocketbox（Mixamo 命名）で leftUpperLeg←LeftLeg と誤対応し、
    # lowerLeg が空になって VRM 書き出しが必須ボーン不足で落ちた。
    # KEHAI 側の対応表（glb-humanoid.ts）は最初から正しく分けている。
    # **"upleg" を先に置くこと。** 部分一致なので順序が効く。
    "upperleg":   ["upleg", "upperleg", "thigh", "腿", "足"],
    "lowerleg":   ["lowerleg", "shin", "calf", "knee", "ひざ", "leg"],
    "foot":       ["foot", "ankle", "足首"],
    "toes":       ["toes", "toe", "toebase", "つま先"],
    "eye":        ["eye", "faceeye", "目"],
    "jaw":        ["jaw", "あご"],
}
# 別名 -> 正規の中核名
_ALIAS_TO_CORE = {a: core for core, al in CORE_ALIASES.items() for a in al}

# 接頭辞の優先度。数値が小さいほど「本体の骨格」に近い。
# VRoid の J_Sec_* は揺れもの、J_Aim_/J_Roll_ は補助骨なので humanoid には使わない。
# ただし目は VRoid では J_Adj_L_FaceEye が本体なので Adj を許容する。
PREFIX_RANK = [
    (re.compile(r"^j_bip_([clr])_", re.I), 0),
    (re.compile(r"^j_adj_([clr])_", re.I), 1),
    (re.compile(r"^j_opt_([clr])_", re.I), 2),
    (re.compile(r"^j_(?:aim|roll)_([clr])_", re.I), 3),
    (re.compile(r"^j_sec_([clr])_", re.I), 4),
    (re.compile(r"^j_sec_", re.I), 4),
]
_GENERIC_PREFIX = re.compile(
    r"^(?:mixamorig[:_]?|bip0?1[-_.: ]*|def[-_.]|org[-_.]|mch[-_.]|ctrl[-_.])", re.I)

_SIDE_PREFIX = re.compile(r"^(left|right|l|r)[-_.:]?(?=[a-z0-9])", re.I)
_SIDE_SUFFIX = re.compile(r"[-_.:]?(left|right|l|r)$", re.I)


def _side_word(tok: str) -> str:
    t = tok.lower()
    if t in ("l", "left"):
        return "left"
    if t in ("r", "right"):
        return "right"
    return ""


def parse_bone(name: str) -> tuple[str, str, int]:
    """ボーン名 -> (左右, 中核名, 優先度)。中核名が未知なら空文字。"""
    raw = name.strip()
    if raw.lower().endswith("_end"):
        return "", "", 99  # 末端ダミーは humanoid に使わない

    rank = 0
    side = ""
    rest = raw

    for pat, r in PREFIX_RANK:
        m = pat.match(raw)
        if m:
            rank = r
            if m.groups():
                side = _side_word(m.group(1))
            rest = raw[m.end():]
            break
    else:
        rest = _GENERIC_PREFIX.sub("", raw)

    # 残りから左右トークンを剥がす（接頭・接尾どちらでも）
    if not side:
        m = _SIDE_PREFIX.match(rest)
        if m:
            side = _side_word(m.group(1))
            rest = rest[m.end():]
    if not side:
        m = _SIDE_SUFFIX.search(rest)
        if m:
            side = _side_word(m.group(1))
            rest = rest[: m.start()]

    core = re.sub(r"[\s\-_.:]+", "", rest).lower()
    core = re.sub(r"\.?\d{3}$", "", core)
    return side, _ALIAS_TO_CORE.get(core, ""), rank


def _to_snake(camel: str) -> str:
    """leftUpperArm -> left_upper_arm（アドオンのプロパティ名に合わせる）。"""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel).lower()


def vrm_target(vrm_name: str) -> tuple[str, str]:
    """VRM規約名 -> (左右, 中核名)。"""
    if vrm_name.startswith("left"):
        return "left", vrm_name[4:].lower()
    if vrm_name.startswith("right"):
        return "right", vrm_name[5:].lower()
    return "", vrm_name.lower()


def build_mapping(armature) -> tuple[dict[str, str], list[str]]:
    """VRM規約名 -> 実ボーン名 の対応表と、不足している必須ボーン一覧。

    同じ (左右, 中核名) に複数のボーンが該当した場合は優先度が高い方を採る
    （例: J_Bip_L_UpperLeg が J_Aim_L_UpperLeg より優先される）。
    """
    best: dict[tuple[str, str], tuple[int, str]] = {}
    for b in armature.data.bones:
        side, core, rank = parse_bone(b.name)
        if not core:
            continue
        key = (side, core)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, b.name)

    mapping: dict[str, str] = {}
    for vrm_name in REQUIRED_BONES + OPTIONAL_BONES:
        key = vrm_target(vrm_name)
        hit = best.get(key)
        if hit:
            mapping[vrm_name] = hit[1]

    missing = [b for b in REQUIRED_BONES if b not in mapping]
    return mapping, missing


def import_any(path: Path) -> None:
    ext = path.suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".vrm":
        bpy.ops.import_scene.vrm(filepath=str(path))
    else:
        raise SystemExit(f"未対応の拡張子: {ext}")


def ensure_vrm_addon() -> None:
    import addon_utils
    for m in addon_utils.modules():
        if "vrm" in m.__name__.lower():
            try:
                addon_utils.enable(m.__name__, default_set=True, persistent=True)
            except Exception:
                pass
    if not hasattr(bpy.ops.export_scene, "vrm"):
        raise SystemExit("VRM Add-on が有効になっていません")


def set_double_sided(path: Path) -> dict:
    """書き出した GLB の全マテリアルに doubleSided=true を立てる。

    GLB は [12Bヘッダ][JSONチャンク][BINチャンク] の並び。JSON を書き換えると
    長さが変わるので、**4バイト境界にスペースで詰め直し**、チャンク長と
    ファイル全長を書き直す必要がある。ここを間違えると読み込めない GLB になる。
    """
    import struct

    raw = path.read_bytes()
    magic, ver, _ = struct.unpack("<III", raw[:12])
    if magic != 0x46546C67:
        return {"status": "skipped", "reason": "GLB ではありません"}

    jlen, jtype = struct.unpack("<II", raw[12:20])
    doc = json.loads(raw[20:20 + jlen])
    rest = raw[20 + jlen:]                      # BIN チャンク以降はそのまま

    mats = doc.get("materials") or []
    for m in mats:
        m["doubleSided"] = True

    new_json = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    pad = (-len(new_json)) % 4
    new_json += b" " * pad                      # JSON チャンクはスペースで詰める
    out = bytearray()
    out += struct.pack("<III", magic, ver, 12 + 8 + len(new_json) + len(rest))
    out += struct.pack("<II", len(new_json), jtype)
    out += new_json
    out += rest
    path.write_bytes(bytes(out))
    return {"status": "ok", "materials": len(mats)}


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--name", default="Genesis3D Character")
    ap.add_argument("--author", default="Genesis3D")
    ap.add_argument("--fix-double-sided", action="store_true",
                    help=("--input の GLB/VRM を書き換えて全マテリアルを両面表示に"
                          "する。薄いシェル（髪）が裏面カリングで透けるのを防ぐ。"
                          "パイプラインの VRM 書出は autorig の出力をコピーする"
                          "だけなので、ここで後から当てる"))
    ap.add_argument("--dry-run", action="store_true",
                    help="ボーン対応の検査だけ行い、書き出さない")
    args = ap.parse_args(argv)

    bpy.ops.wm.read_homefile(use_empty=True)
    ensure_vrm_addon()
    import_any(Path(args.input))

    report: dict = {"input": args.input, "output": args.out}

    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    report["armature_count"] = len(armatures)
    report["mesh_count"] = len(meshes)

    if not armatures:
        report["status"] = "fail"
        report["error"] = ("アーマチュアがありません。VRM Humanoid は骨格が必須です"
                           "（工程8のリギングが未実行か失敗しています）")
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print("VRM_EXPORT_FAIL " + report["error"])
        sys.exit(1)

    arm = armatures[0]
    mapping, missing = build_mapping(arm)
    report["bone_count"] = len(arm.data.bones)
    report["mapped_bones"] = mapping
    report["missing_required_bones"] = missing
    report["optional_present"] = [b for b in OPTIONAL_BONES if b in mapping]

    # スケール・原点の検査。VRM は1.0 = 1メートル、原点は足元が基本。
    zs = []
    for m in meshes:
        for c in m.bound_box:
            zs.append((m.matrix_world @ __import__("mathutils").Vector(c)).z)
    if zs:
        report["height_m"] = round(max(zs) - min(zs), 4)
        report["foot_z"] = round(min(zs), 4)
        report["height_plausible"] = 0.5 <= (max(zs) - min(zs)) <= 3.0
        report["origin_at_feet"] = abs(min(zs)) < 0.05

    report["materials"] = [m.name for m in bpy.data.materials]
    report["images"] = [{"name": i.name, "size": list(i.size)} for i in bpy.data.images]

    # スキニングの検査。**必須ボーンの割当と同格の検査として扱う。**
    # Blender の heat map は解が出なくても例外を投げず全ウェイト0で完了するため、
    # 「頂点グループもモディファイアも骨の割当も揃っているのに1頂点もスキン
    # されていない VRM」が作れてしまう。glTF 側は skin を書かず、読み込んでも
    # メッシュが骨に追従しない。見た目は完璧なので気づけない
    # （docs/05-measurements.md §28）。
    skinned = unskinned = 0
    for m in meshes:
        has_mod = any(mod.type == "ARMATURE" for mod in m.modifiers)
        nz = sum(1 for v in m.data.vertices
                 if any(g.weight > 1e-6 for g in v.groups))
        if has_mod and nz > 0:
            skinned += 1
        else:
            unskinned += 1
        report.setdefault("skinning", []).append(
            {"mesh": m.name, "armature_modifier": has_mod,
             "weighted_vertices": nz, "vertices": len(m.data.vertices)})
    report["skinned_meshes"] = skinned
    report["unskinned_meshes"] = unskinned

    if skinned == 0 and meshes:
        report["status"] = "fail"
        report["error"] = ("どのメッシュにもスキンウェイトが入っていません。"
                           "アーマチュアはあるがメッシュが追従しないVRMになります")
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print("VRM_EXPORT_FAIL " + report["error"])
        sys.exit(2)

    if missing:
        report["status"] = "fail"
        report["error"] = f"VRM必須ボーンが {len(missing)} 本不足: {missing}"
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print("VRM_EXPORT_FAIL " + report["error"])
        sys.exit(2)

    if args.fix_double_sided:
        report["double_sided"] = set_double_sided(Path(args.input))

    if args.dry_run:
        report["status"] = "ok_dry_run"
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print("VRM_DRYRUN_OK " + json.dumps({"mapped": len(mapping)}))
        return

    # --- VRM Humanoid の割当 ---
    # **アーマチュアのスケールと回転を焼く。**
    #
    # Rocketbox 由来のリグはルートノード `Bip01` が scale 0.01 と回転を持ち、
    # 子の translation はセンチ単位で入っている（メッシュの POSITION は
    # メートルの世界座標なので、見た目は正しい）。VRM 1.0 は単位スケール・Y上を
    # 要求するので、この状態で書き出すとアドオンの構造探索が破裂して
    # **何も書かずに CANCELLED を返す**（実測: 探索 151,539,405回 / 上限 100万、
    # しかもアドオンが独自の誤った対応を生成する —— hips←Spine、
    # leftUpperArm←LeftShoulder と1つずれる）。
    arm_objs = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    before = [round(v, 6) for v in (arm_objs[0].scale if arm_objs else (1, 1, 1))]
    if arm_objs:
        bpy.ops.object.select_all(action="DESELECT")
        for o in arm_objs + mesh_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = arm_objs[0]
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    report["armature_scale_before"] = before
    report["armature_scale_after"] = [
        round(v, 6) for v in (arm_objs[0].scale if arm_objs else (1, 1, 1))]
    print(f"アーマチュアの変換を焼いた: scale {before} → "
          f"{report['armature_scale_after']}")

    ext = getattr(arm.data, "vrm_addon_extension", None)
    if ext is None:
        report["status"] = "fail"
        report["error"] = "armature に vrm_addon_extension がありません"
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print("VRM_EXPORT_FAIL " + report["error"])
        sys.exit(3)

    ext.spec_version = "1.0"
    human_bones = ext.vrm1.humanoid.human_bones
    # VRM 仕様は camelCase (leftUpperArm) だが、アドオンのプロパティは
    # snake_case (left_upper_arm)。ここを取り違えると単語1つのボーンしか
    # 割り当たらず、見た目は成功して中身が壊れた VRM ができる。
    assigned = 0
    unassigned: list[str] = []
    for vrm_name, bone_name in mapping.items():
        attr = _to_snake(vrm_name)
        hb = getattr(human_bones, attr, None)
        if hb is not None and hasattr(hb, "node"):
            hb.node.bone_name = bone_name
            assigned += 1
        else:
            unassigned.append(vrm_name)
    report["assigned_human_bones"] = assigned
    report["unassigned_human_bones"] = unassigned

    # 必須ボーンが1本でも割り当たっていなければ、書き出す前に止める。
    required_unassigned = [b for b in REQUIRED_BONES if b in unassigned]
    if required_unassigned:
        report["status"] = "fail"
        report["error"] = (f"必須ボーンをVRM拡張に割り当てられませんでした: "
                           f"{required_unassigned}")
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print("VRM_EXPORT_FAIL " + report["error"])
        sys.exit(4)

    meta = ext.vrm1.meta
    meta.vrm_name = args.name
    if hasattr(meta, "authors"):
        try:
            if len(meta.authors) == 0:
                meta.authors.add()
            meta.authors[0].value = args.author
        except Exception:
            pass

    # **マテリアルを持たないメッシュを消す。** Blender 4.2 の glTF インポータは
    # 読み込みのたびに 42頂点の `Icosphere` を作る。アーマチュアに紐づいていない
    # メッシュがシーンに残っていると、VRM アドオンの構造探索が破裂して
    # **何も書き出さずに黙って終わる**（実測: 探索 151,539,405回 / 上限 1,000,000、
    # 出力ファイルが存在せず、後段の set_double_sided が FileNotFoundError）。
    junk = [o for o in bpy.data.objects
            if o.type == "MESH" and not [m for m in o.data.materials if m]]
    junk_names = [o.name for o in junk]   # 削除後は o.name を読めない
    for o in junk:
        bpy.data.objects.remove(o, do_unlink=True)
    if junk_names:
        print(f"マテリアル無しのメッシュを除去: {junk_names}")
    report["removed_meshes"] = junk_names

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res = bpy.ops.export_scene.vrm(filepath=str(args.out))
    # **戻り値を見る。** アドオンは失敗しても例外を投げず CANCELLED を返す。
    if "FINISHED" not in res or not Path(args.out).exists():
        report["status"] = "fail"
        report["error"] = f"export_scene.vrm が {res} を返し、ファイルが無い"
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print("VRM_EXPORT_FAIL " + report["error"], file=sys.stderr)
        sys.exit(5)

    # 薄いシェル（髪・衣装の裾）は裏面カリングされると内側が見える角度で消える。
    # glTF の doubleSided は省略時 false なので明示的に立てる。image-to-3D の
    # 髪は板状の薄いシェルになりやすく、これが無いと側面や 3/4 で髪が透ける
    # （docs/05-measurements.md §30）。
    #
    # Blender 側で use_backface_culling=False にしても VRM アドオンの書き出しは
    # これを反映しない（実測: doubleSided が出力されなかった）。書き出し後の
    # GLB の JSON チャンクを直接書き換える。
    report["double_sided"] = set_double_sided(Path(args.out))

    out_p = Path(args.out)
    report["status"] = "ok" if out_p.exists() and out_p.stat().st_size > 1024 else "fail"
    report["output_bytes"] = out_p.stat().st_size if out_p.exists() else 0
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("VRM_EXPORT_OK " + json.dumps({"bytes": report["output_bytes"],
                                         "bones": assigned}))


if __name__ == "__main__":
    main()
