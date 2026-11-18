#!/usr/bin/env python3
"""KEHAI に入れる前に .glb / .vrm を検分する。

KEHAI は VRM 専用ではない。`App.tsx` は `.vrm` と `.glb` を同じビューアに送り、
`.glb` は `utils/glb-humanoid.ts` が **Mixamo 系のボーン名**を見て VRMHumanoid を
被せる（Ready Player Me / Avaturn / Character Creator はどれもこの骨格を出す）。
表情は VRM の expression ではなく **生のモーフ名**を直接見る（utils/vrm-expression.ts）。

つまり外部で作ったアバターを入れるとき、確認すべきは三つだけ:

  1. ボーンが揃っているか
  2. まばたきと口のモーフ名があるか
  3. 破綻がないか（頂点数・身長・マテリアル数）

**1 は .vrm と .glb で問うべきことが違う。** `App.tsx` は `.vrm` を three-vrm に
渡すので、そちらは `VRMC_vrm.humanoid.humanBones` を読む。Mixamo 名を見るのは
`glb-humanoid.ts` だけで、それは `.glb` 専用の経路。VRM に Mixamo 名を要求すると、
54本の humanBones を持つ健全な VRM が「必須ボーン16本すべて欠落」と出る
（実測: VRoid のサンプル素体がこれで落ちていた）。

**2 は「どちらの語彙か」が先に決まる。** `vrm-expression.ts:343` は
`profile = dict["まばたき"] != null ? "pmx" : "arkit"` と、**生のモーフ名だけ**を見て
語彙を選ぶ。VRM の expression プリセットは読まない — `vrm-expression.ts:558` と
`vrm-viewer.tsx:5117` が「vrm.update() の後に走って expressionManager を上書きする」と
明記している。つまり VRoid の `Fcl_*` 命名は PMX でも ARKit でもないので arkit と
判定され、探しに行った `eyeBlinkLeft` が無くて何も動かない。

VRM への変換は要らない。glb のまま `Live2D Characters/vrm/` に置けば動く。

使い方:
    python3 scripts/check_kehai_model.py <model.glb|model.vrm>
"""
import json
import struct
import sys

# VRM の場合。three-vrm が VRMC_vrm.humanoid.humanBones から読む名前。
VRM_REQUIRED = [
    "hips", "spine", "neck", "head",
    "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot",
    "rightUpperLeg", "rightLowerLeg", "rightFoot",
]

# plain .glb の場合。glb-humanoid.ts の BONE_MAP と同じ。
REQUIRED_BONES = [
    "Hips", "Spine", "Neck", "Head",
    "LeftArm", "LeftForeArm", "LeftHand",
    "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "RightUpLeg", "RightLeg", "RightFoot",
]
NICE_BONES = ["Spine1", "Spine2", "LeftShoulder", "RightShoulder", "LeftEye", "RightEye"]
# vrm-expression.ts:198 ARKIT_BLINK / :236 EYE_MORPHS(pmx)
BLINK_ARKIT = ["eyeBlinkLeft", "eyeBlinkRight"]
BLINK_PMX = ["まばたき"]
# vrm-lipsync.ts:29 MOUTH_CANDIDATES をそのまま写す（順序も同じ）
MOUTH = ["あ", "ア", "A", "a", "jawOpen", "JawOpen", "jaw_open",
         "aa", "viseme_aa", "mouthOpen", "Mouth_Open"]


def read_gltf(path: str):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != b"glTF":
        raise SystemExit(f"{path} は GLB ではない")
    jlen = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + jlen]), len(data), data[20 + jlen + 8:]


def effective_metallic(g: dict, blob: bytes, mat: dict) -> float:
    """three.js が使う金属度: metallicFactor × テクスチャ B の平均（テクスチャ無しは 1）。"""
    pbr = mat.get("pbrMetallicRoughness") or {}
    factor = float(pbr.get("metallicFactor", 1.0))
    t = pbr.get("metallicRoughnessTexture")
    if not t or factor == 0.0:
        return factor
    try:
        import io
        import numpy as np
        from PIL import Image
        tex = g["textures"][t["index"]]
        src = (tex.get("extensions", {}).get("EXT_texture_webp", {}).get("source",
               tex.get("source")))
        img = g["images"][src]
        bvw = g["bufferViews"][img["bufferView"]]
        off = bvw.get("byteOffset", 0)
        arr = np.asarray(Image.open(io.BytesIO(
            blob[off:off + bvw["byteLength"]])).convert("RGB"))
        return factor * float(arr[..., 2].mean()) / 255.0
    except Exception as e:  # 読めなければ factor だけで判定
        print(f"  注意: metallicRoughness テクスチャを読めない（{e}）")
        return factor


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    g, size, blob = read_gltf(path)

    ext = g.get("extensions") or {}
    vrm = ext.get("VRMC_vrm") or {}
    is_vrm = bool(vrm) or "VRM" in ext

    nodes = [n.get("name") or "" for n in g.get("nodes", [])]
    have = set(nodes)

    if is_vrm:
        # three-vrm が読むのは humanBones であってノード名ではない。
        human = set((vrm.get("humanoid") or {}).get("humanBones") or {})
        if not human and "VRM" in ext:  # VRM 0.x
            human = {b.get("bone") for b in
                     ((ext["VRM"].get("humanoid") or {}).get("humanBones") or [])}
        missing = [b for b in VRM_REQUIRED if b not in human]
        bone_kind, bone_total = "VRM humanBones", len(human)
    else:
        missing = [b for b in REQUIRED_BONES if b not in have]
        # mixamorig: 接頭辞つきでも許す（glb-humanoid.ts が剥がす）
        if missing:
            stripped = {n.split(":", 1)[-1] for n in nodes}
            missing = [b for b in missing if b not in stripped]
        bone_kind, bone_total = "Mixamo 系ノード名", len(nodes)

    morphs: set[str] = set()
    verts = 0
    for m in g.get("meshes", []):
        for name in (m.get("extras") or {}).get("targetNames", []) or []:
            morphs.add(name)
        for prim in m.get("primitives", []):
            acc = prim.get("attributes", {}).get("POSITION")
            if acc is not None:
                verts += g["accessors"][acc]["count"]

    # vrm-expression.ts:343 と同じ判定。生のモーフ名だけで語彙が決まる。
    profile = "pmx" if BLINK_PMX[0] in morphs else "arkit"
    wanted_blink = BLINK_PMX if profile == "pmx" else BLINK_ARKIT
    blink = [b for b in wanted_blink if b in morphs]
    mouth = [m for m in MOUTH if m in morphs]

    springs = len(((ext.get("VRMC_springBone") or {}).get("springs")) or [])
    mtoon = sum(1 for m in g.get("materials", [])
                if "VRMC_materials_mtoon" in (m.get("extensions") or {}))
    presets = len([k for k, v in
                   ((vrm.get("expressions") or {}).get("preset") or {}).items()
                   if (v or {}).get("morphTargetBinds")])

    print(f"ファイル       {path}")
    print(f"サイズ         {size / 1048576:.1f}MB")
    print(f"形式           {'VRM' if is_vrm else 'plain glTF'}  "
          f"（KEHAI はどちらでも読む）")
    print(f"ノード         {len(nodes)}   メッシュ {len(g.get('meshes', []))}   "
          f"マテリアル {len(g.get('materials', []))}   頂点 {verts:,}")
    print(f"モーフ         {len(morphs)}   語彙 {profile}   "
          f"spring {springs}   MToon {mtoon}")
    print()
    ok = True
    if missing:
        print(f"✗ 必須ボーンが足りない（{bone_kind}）: {missing}")
        print("  → 動きがリターゲットできない")
        ok = False
    else:
        print(f"✓ 必須ボーン16本すべてある（{bone_kind} {bone_total}）")
        if not is_vrm:
            extra = [b for b in NICE_BONES if b in have]
            if extra:
                print(f"  加えて: {extra}")
    if blink:
        print(f"✓ まばたき: {'+'.join(blink)}")
    else:
        print(f"✗ まばたきのモーフがない（語彙 {profile} なので "
              f"{' / '.join(wanted_blink)} を探した）")
        ok = False
    if mouth:
        print(f"✓ 口パク: {mouth[0]}")
    else:
        print("✗ 口の開閉モーフがない（あ / jawOpen など）")
        ok = False
    if is_vrm and presets and profile == "arkit" and not blink:
        # ここを取り違えると「表情はあるのに動かない」で延々悩むことになる。
        print(f"  注意: VRM expression プリセットが {presets}個 結線されているが、"
              f"KEHAI はそれを読まない（vrm-expression.ts:558 が上書きする）。"
              f"生のモーフ名が要る")
    # **金属のマテリアルは KEHAI で真っ黒になる。** three.js の PBR は環境マップが
    # 無いと金属の反射を描けない。glTF は `metallicFactor` 省略＝1.0（完全な金属）
    # なので、素材情報を持たない FBX/生成物はそのままだと黒い切り絵になる
    # （髪でユーザーが「壊れてる」と指摘）。Blender の EEVEE はライトの反射で
    # それらしく見せるので、検査で止めないと気づけない。
    # 実効値 = metallicFactor × metallicRoughness テクスチャの B チャンネル平均。
    # TRELLIS の glb は factor を省略（1.0）しつつテクスチャの B≈0 で非金属にしている
    # ので、factor だけ見ると誤検出する。
    metallic = [m.get("name") for m in g.get("materials", [])
                if effective_metallic(g, blob, m) >= 0.5
                and "KHR_materials_unlit" not in (m.get("extensions") or {})]
    if metallic:
        print(f"✗ 金属のマテリアル（metallicFactor ≥ 0.5 か省略）: {metallic}")
        print("  → KEHAI には環境マップが無いので真っ黒に描かれる")
        ok = False
    else:
        print("✓ 金属のマテリアルなし")
    print()
    print("KEHAI_CHECK " + json.dumps({
        "ok": ok, "is_vrm": is_vrm, "bones_missing": missing,
        "metallic_materials": metallic,
        "bone_kind": bone_kind, "profile": profile,
        "blink": blink, "mouth": mouth[:1], "verts": verts,
        "morphs": len(morphs), "springs": springs, "mtoon_materials": mtoon,
        "expression_presets_bound": presets,
        "expression_driven_by_kehai": False,
        "mb": round(size / 1048576, 1),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
