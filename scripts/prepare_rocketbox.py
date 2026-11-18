#!/usr/bin/env python3
"""Microsoft Rocketbox のアバターを `game_avatar` のテンプレートに仕立てる。

Rocketbox（MIT・115体・実写寄り）は素材として理想的だが、**名前だけ**が合わない。
実測で確かめたところ、KEHAI の検査に落ちる原因はジオメトリではなく命名だった:

  - 骨が 3ds Max Biped 命名（`Bip01 Pelvis` / `Bip01 L Finger0`）。
    KEHAI の `glb-humanoid.ts` は Mixamo 命名を見るので、そのままでは
    リターゲットが効かない
  - モーフが `AK_09_EyeBlinkLeft` / `AA_VI_10_aa` という接頭辞つき。
    KEHAI は生のモーフ名を見る（`vrm-expression.ts:343`）ので、
    ARKit の `eyeBlinkLeft` を探して見つけられない

どちらも**文字列の書き換えだけ**で済み、頂点も UV もウェイトも触らない。
このスクリプトはそれを一度だけ行う。テンプレートは使い回すので、実行も一度だけ。

FBX → GLB の変換にだけ Blender を使う。以降の改名は glTF の JSON を直接いじる
（`preprocessing/shape_transfer.py` と同じ方針。docs/08 §6）。

使い方:
    python3 scripts/prepare_rocketbox.py \\
        --fbx <Rocketbox>/Assets/Avatars/Adults/Male_Adult_01/Export/Male_Adult_01_facial.fbx \\
        --out assets/templates/rocketbox/male_adult_01.glb --report r.json

`_facial.fbx` のほうを渡すこと。接尾辞の無い素体には表情が入っていない。

**テクスチャは FBX の隣に置かれた `../Textures/` から解決される。** FBX だけを
取り出して渡すと Blender が TGA を見つけられず、`images: 0` の GLB ができる
（顔テクスチャ 2048² がこの素体を選んだ理由なので、これは静かな失敗になる）。
リポジトリの `Assets/Avatars/<Name>/` をフォルダごと持ってくること。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blender.vrm_finalize import read_glb, write_glb  # noqa: E402

# --------------------------------------------------------------------- 骨の改名
#
# Biped → Mixamo。Rocketbox の実物（80 joints）から起こした対応で、推測は入れていない。
# Mixamo 命名にするのは KEHAI の `.glb` 経路がそれを見るため。VRM 化する場合も
# humanBones は「役割 → ノード番号」の対応なので、名前が Mixamo でも支障はない
# （`blender/export_vrm.py` の照合器は Mixamo 命名を認識する）。

_BODY = {
    "Bip01 Pelvis": "Hips",
    "Bip01 Spine": "Spine",
    "Bip01 Spine1": "Spine1",
    "Bip01 Spine2": "Spine2",
    "Bip01 Neck": "Neck",
    "Bip01 Head": "Head",
    "Bip01 LEye": "LeftEye",
    "Bip01 REye": "RightEye",
}
for _s, _m in (("L", "Left"), ("R", "Right")):
    _BODY.update({
        f"Bip01 {_s} Clavicle": f"{_m}Shoulder",
        f"Bip01 {_s} UpperArm": f"{_m}Arm",
        f"Bip01 {_s} Forearm": f"{_m}ForeArm",
        f"Bip01 {_s} Hand": f"{_m}Hand",
        f"Bip01 {_s} Thigh": f"{_m}UpLeg",
        f"Bip01 {_s} Calf": f"{_m}Leg",
        f"Bip01 {_s} Foot": f"{_m}Foot",
        f"Bip01 {_s} Toe0": f"{_m}ToeBase",
    })

# Biped の指: Finger0=親指 / 1=人差し指 / 2=中指 / 3=薬指 / 4=小指。
# 接尾辞なし=第1関節、1=第2、2=第3。
_FINGER = {"0": "Thumb", "1": "Index", "2": "Middle", "3": "Ring", "4": "Pinky"}
for _s, _m in (("L", "Left"), ("R", "Right")):
    for _d, _name in _FINGER.items():
        _BODY[f"Bip01 {_s} Finger{_d}"] = f"{_m}Hand{_name}1"
        _BODY[f"Bip01 {_s} Finger{_d}1"] = f"{_m}Hand{_name}2"
        _BODY[f"Bip01 {_s} Finger{_d}2"] = f"{_m}Hand{_name}3"

BONE_MAP = _BODY

# 顔ボーン26本は Mixamo に対応物が無いので、接頭辞だけ落として残す。
# ブレンドシェイプと併存していて、消すと表情の一部が動かなくなる。
FACE_BONE_PREFIX = "Bip01 "

# Biped のルート。humanoid の骨ではないので改名しない（未対応として報告もしない）。
ROOT_BONE = "Bip01"


# ------------------------------------------------------------------ モーフの改名
#
# `AK_NN_XxxYyy` は ARKit の52種そのもの。先頭を小文字にすれば ARKit 命名になる。
# `AA_VI_NN_X` は Oculus のビセーム15種。KEHAI の `vrm-lipsync.ts` と
# Avaturn 由来の `subject_a.glb` が使っているのと同じ語彙に合わせる。
VISEME = {
    "00_Sil": "sil", "01_PP": "PP", "02_FF": "FF", "03_TH": "TH", "04_DD": "DD",
    "05_KK": "kk", "06_CH": "CH", "07_SS": "SS", "08_nn": "nn", "09_RR": "RR",
    "10_aa": "aa", "11_E": "E", "12_I": "ih", "13_O": "oh", "14_U": "ou",
}


def rename_morph(name: str) -> str | None:
    """Rocketbox のモーフ名を ARKit / ビセーム命名へ。対象外は None。"""
    m = re.fullmatch(r"AK_\d{2}_([A-Za-z]+)", name)
    if m:
        s = m.group(1)
        return s[0].lower() + s[1:]
    m = re.fullmatch(r"AA_VI_(\d{2}_\w+)", name)
    if m:
        return VISEME.get(m.group(1))
    return None


# ----------------------------------------------------------------- マテリアル名
#
# `m002_head` のようにアバターごとの番号が入っている。テンプレートを差し替えても
# 設定を書き換えずに済むよう、役割だけの名前に正規化する。
MATERIAL_RE = re.compile(r"^[mfc]\d{3}_(body|head|opacity)$")


def convert_fbx(fbx: Path, out: Path, blender: str, image_format: str) -> dict:
    """Blender で FBX → GLB。ここでしか Blender を使わない。

    貼り直しを含む実処理は `blender/rocketbox_to_glb.py` にある。Rocketbox の FBX は
    3ds Max の絶対パス（`D:\\temp\\Humans\\...`）を持っていて、そのままだと
    テクスチャが1枚も入らない GLB が**エラー無しで**できるため。

    画像形式の既定は WEBP。元の TGA は 2048² の無圧縮で1枚12MBあり、PNG でも
    7枚で数十MBになる。KEHAI は `subject_a_v2.vrm` で EXT_texture_webp を
    すでに読めている（three.js が対応している）ので、圧縮して差し支えない。
    """
    script = Path(__file__).resolve().parent.parent / "blender" / "rocketbox_to_glb.py"
    r = subprocess.run(
        [blender, "--background", "--factory-startup", "--python", str(script),
         "--", "--fbx", str(fbx), "--out", str(out), "--image-format", image_format],
        capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("ROCKETBOX_GLB ")), None)
    if not out.exists() or line is None:
        raise SystemExit(f"Blender の変換に失敗した:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.loads(line[len("ROCKETBOX_GLB "):])


def retarget(doc: dict) -> dict:
    """ノード名・モーフ名・マテリアル名を書き換える。頂点データには触らない。"""
    rep: dict = {"bones": {"renamed": 0, "face_kept": 0, "unmatched": []},
                 "morphs": {"arkit": 0, "viseme": 0, "kept": 0},
                 "materials": {}}

    for node in doc.get("nodes", []):
        name = node.get("name") or ""
        if name in BONE_MAP:
            node["name"] = BONE_MAP[name]
            rep["bones"]["renamed"] += 1
        elif name.startswith(FACE_BONE_PREFIX):
            # 顔ボーン。Mixamo に対応物が無いので接頭辞だけ落とす。
            node["name"] = name[len(FACE_BONE_PREFIX):]
            rep["bones"]["face_kept"] += 1
        elif name == ROOT_BONE:
            pass                      # アーマチュアのルート。そのままでよい
        elif name.startswith("Bip01"):
            rep["bones"]["unmatched"].append(name)

    for mesh in doc.get("meshes", []):
        extras = mesh.setdefault("extras", {})
        names = extras.get("targetNames") or []
        out = []
        for n in names:
            new = rename_morph(n)
            if new is None:
                out.append(n)
                rep["morphs"]["kept"] += 1
            else:
                out.append(new)
                rep["morphs"]["viseme" if n.startswith("AA_VI") else "arkit"] += 1
        if names:
            extras["targetNames"] = out

    for mat in doc.get("materials", []):
        m = MATERIAL_RE.match(mat.get("name") or "")
        if m:
            rep["materials"][mat["name"]] = m.group(1)
            mat["name"] = m.group(1)

        # **metallicFactor を 0 にする。** Rocketbox の FBX は PBR の情報を持たないので、
        # glTF 化すると `metallicFactor` が省略され、仕様の既定値 **1.0**（＝完全な金属）
        # として解釈される。肌と布のアバターとしてそもそも誤りで、実害も出た:
        #
        #   - Cycles の DIFFUSE bake が**全面黒**になる（金属には拡散成分が無い）。
        #     写真のテクスチャ投影がこれで丸ごと失敗した
        #   - KEHAI のような PBR レンダラで正しい環境光を当てると金属として映る
        #
        # EEVEE の簡易プレビューではそれらしく見えてしまうので気づきにくい。
        pbr = mat.setdefault("pbrMetallicRoughness", {})
        if pbr.get("metallicFactor", 1.0) != 0.0:
            rep.setdefault("metallic_zeroed", []).append(mat.get("name"))
            pbr["metallicFactor"] = 0.0

    return rep


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fbx", type=Path, required=True, help="Rocketbox の *_facial.fbx")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--blender", default=os.environ.get("BLENDER_BIN")
                    or "/home/ai-ccore/opt/blender/blender")
    ap.add_argument("--image-format", default="WEBP", choices=("WEBP", "JPEG", "AUTO"),
                    help="AUTO は PNG。既定の WEBP は KEHAI で実績あり")
    args = ap.parse_args()

    if "_facial" not in args.fbx.name:
        print(f"注意: {args.fbx.name} は表情つきではないかもしれない "
              f"（`_facial.fbx` を渡すこと）", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".raw.glb")
    conv = convert_fbx(args.fbx, tmp, args.blender, args.image_format)

    doc, blob = read_glb(tmp)
    rep = retarget(doc)
    write_glb(args.out, doc, blob)
    tmp.unlink()

    verts = sum(doc["accessors"][p["attributes"]["POSITION"]]["count"]
                for m in doc.get("meshes", []) for p in m["primitives"])
    images = len(doc.get("images", []))
    rep.update({"source": str(args.fbx), "out": str(args.out), "images": images,
                "image_format": args.image_format,
                "vertices": verts, "size_mb": round(args.out.stat().st_size / 1048576, 1)})
    rep["textures"] = conv
    if conv["missing"]:
        print(f"**貼り直せなかったテクスチャ**: {conv['missing']}\n"
              f"  探した場所: {conv['searched']}", file=sys.stderr)
    if images == 0:
        print("**テクスチャが1枚も入っていない。** FBX の隣に ../Textures/ が要る",
              file=sys.stderr)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))

    b, mo = rep["bones"], rep["morphs"]
    print(f"骨       Mixamo へ改名 {b['renamed']}本 / 顔ボーン {b['face_kept']}本を保持")
    if b["unmatched"]:
        print(f"         **対応表に無い骨**: {b['unmatched']}")
    print(f"モーフ   ARKit {mo['arkit']}個 / ビセーム {mo['viseme']}個 / "
          f"そのまま {mo['kept']}個（FACS AU など）")
    print(f"マテリアル {rep['materials']}")
    print(f"頂点 {verts:,}   テクスチャ {images}枚（{args.image_format}）   {rep['size_mb']}MB")
    print(f"→ {args.out}")
    print("\n次に: python3 scripts/check_kehai_model.py " + str(args.out))


if __name__ == "__main__":
    main()
