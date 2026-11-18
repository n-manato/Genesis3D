#!/usr/bin/env python3
"""テンプレート素体の要件検査 — `game_avatar` に使える VRM かを判定する。

`game_avatar`（docs/08）は写真ごとにメッシュを起こすのをやめ、**1体のテンプレート素体を
変形させる**。そのためテンプレートの品質がそのまま出力の品質上限になる。逆に言えば、
テンプレートさえ要件を満たしていれば、指も肩も揺れものも入力写真によらず常に成立する。

検査するのは八つ:

  1. VRM 1.0 であること（VRM 0.x は humanoid の定義が別物）
  2. Humanoid の必須17本 + **指30本**が揃っていること
     — 指が無いテンプレートを選ぶと §38 の「指が動かない」を持ち越す
  3. 全メッシュに JOINTS_0 / WEIGHTS_0 があること（人が塗ったウェイト）
     — これが無いと自動ウェイトに戻り、§37 の破断を持ち越す
  4. 表情モーフがあり、VRM expression プリセットに結線されていること
  5. VRMC_springBone があること — 髪とスカートの揺れは作り置きでしか入らない
  6. 全マテリアルが MToon であること（KEHAI のトゥーン表示が前提にしている）
  7. **ライセンスが改変と再配布を許していること**
     — テンプレートの幾何は出力に残る。skeleton only の言い訳が効かない
  8. **頂点数が実用の範囲にあること**
     — VRoid の初期エクスポートは削減が掛かっておらず、髪だけで88万頂点ある個体が
       実在する（assets/templates/README.md）。ライセンスも骨も表情も揃っているのに
       ゲームアバターとして成立しない、という落ち方をするので独立した項目にした

加えて、68点ランドマーク対応を張る作業量を見積もるためにメッシュ構成を表示する。

使い方:
    python3 scripts/check_template_vrm.py <candidate.vrm>
    python3 scripts/check_template_vrm.py <candidate.vrm> --json   # 機械可読のみ
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

# blender/export_vrm.py:21-30 と同じ定義。あちらは bpy を要求するので import せず写す。
REQUIRED_BONES = [
    "hips", "spine", "chest", "neck", "head",
    "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot",
    "rightUpperLeg", "rightLowerLeg", "rightFoot",
]

# VRM 1.0 の指ボーン。親指だけ Metacarpal/Proximal/Distal、他4本は
# Proximal/Intermediate/Distal で、片手15本・両手30本。
_FINGERS = [
    ("Thumb", ["Metacarpal", "Proximal", "Distal"]),
    ("Index", ["Proximal", "Intermediate", "Distal"]),
    ("Middle", ["Proximal", "Intermediate", "Distal"]),
    ("Ring", ["Proximal", "Intermediate", "Distal"]),
    ("Little", ["Proximal", "Intermediate", "Distal"]),
]
FINGER_BONES = [
    f"{side}{finger}{joint}"
    for side in ("left", "right")
    for finger, joints in _FINGERS
    for joint in joints
]

# 表情として最低限これだけは動いてほしい、という VRM 1.0 プリセット。
# まばたきと母音が無いと KEHAI 側でリップシンクもアイドルも成立しない。
#
# `neutral` は入れない。VRM 1.0 の neutral は「素の状態」を指すので
# morphTargetBinds が空なのが正常で、結線を要求すると健全なモデルまで落ちる。
WANTED_EXPRESSIONS = ["blink", "aa", "ih", "ou", "ee", "oh", "happy"]

# 頂点数の上限。現行の女性テンプレート（VRoid の削減済みエクスポート）が 24,951 頂点なので、
# その4倍を上限にする。男女で極端に差があるとテクスチャ解像度も生成時間も揃わない。
MAX_VERTICES = 100_000


def read_gltf(path: Path) -> tuple[dict, int]:
    """GLB の JSON チャンクだけを読む。BIN は触らない。"""
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise SystemExit(f"{path} は GLB ではない")
    off = 12
    while off < len(data):
        length, kind = struct.unpack_from("<II", data, off)
        if kind == 0x4E4F534A:  # 'JSON'
            return json.loads(data[off + 8:off + 8 + length]), len(data)
        off += 8 + length
    raise SystemExit(f"{path} に JSON チャンクがない")


def check(doc: dict, size: int) -> dict:
    """要件ごとに (ok, 詳細) を集める。表示はせず結果だけ返す。"""
    ext = doc.get("extensions") or {}
    vrm = ext.get("VRMC_vrm") or {}
    res: dict = {"checks": {}}

    # 1. VRM 1.0
    spec = vrm.get("specVersion")
    res["checks"]["vrm10"] = {
        "ok": bool(vrm) and str(spec).startswith("1."),
        "spec_version": spec,
        "is_vrm0": "VRM" in ext and not vrm,
    }

    # 2. Humanoid — 必須17本と指30本
    human = ((vrm.get("humanoid") or {}).get("humanBones") or {})
    missing_req = [b for b in REQUIRED_BONES if b not in human]
    missing_fin = [b for b in FINGER_BONES if b not in human]
    res["checks"]["humanoid"] = {
        "ok": not missing_req and not missing_fin,
        "total": len(human),
        "missing_required": missing_req,
        "fingers_present": len(FINGER_BONES) - len(missing_fin),
        "fingers_expected": len(FINGER_BONES),
    }

    # 3. スキンウェイト — 全プリミティブに JOINTS_0 / WEIGHTS_0
    unskinned = []
    meshes = []
    for mesh in doc.get("meshes", []):
        prims = mesh.get("primitives", [])
        verts = 0
        targets = 0
        for i, prim in enumerate(prims):
            attrs = prim.get("attributes", {})
            if "JOINTS_0" not in attrs or "WEIGHTS_0" not in attrs:
                unskinned.append(f"{mesh.get('name', '?')}[{i}]")
            acc = attrs.get("POSITION")
            if acc is not None:
                verts += doc["accessors"][acc]["count"]
            targets = max(targets, len(prim.get("targets") or []))
        meshes.append({
            "name": mesh.get("name", "?"),
            "primitives": len(prims),
            "vertices": verts,
            "morphs": targets,
        })
    res["meshes"] = meshes
    res["checks"]["skinned"] = {"ok": not unskinned, "unskinned": unskinned}

    # 4. 表情 — モーフが存在し、expression プリセットに結線されている
    morphs: set[str] = set()
    for mesh in doc.get("meshes", []):
        for name in (mesh.get("extras") or {}).get("targetNames", []) or []:
            morphs.add(name)
    presets = (vrm.get("expressions") or {}).get("preset") or {}
    bound = [k for k, v in presets.items() if (v or {}).get("morphTargetBinds")]
    res["checks"]["expressions"] = {
        "ok": len(morphs) > 0 and all(w in bound for w in WANTED_EXPRESSIONS),
        "morphs": len(morphs),
        "presets_bound": len(bound),
        "missing_presets": [w for w in WANTED_EXPRESSIONS if w not in bound],
    }

    # 5. spring bone
    spring = ext.get("VRMC_springBone") or {}
    res["checks"]["spring_bone"] = {
        "ok": bool(spring.get("springs")),
        "springs": len(spring.get("springs") or []),
        "colliders": len(spring.get("colliders") or []),
    }

    # 6. MToon — 全マテリアル
    mats = doc.get("materials", [])
    non_mtoon = [
        m.get("name", f"#{i}") for i, m in enumerate(mats)
        if "VRMC_materials_mtoon" not in (m.get("extensions") or {})
    ]
    res["checks"]["mtoon"] = {
        "ok": bool(mats) and not non_mtoon,
        "total": len(mats),
        "non_mtoon": non_mtoon,
    }

    # 7. ライセンス — 改変と再配布が許されているか
    meta = vrm.get("meta") or {}
    modification = meta.get("modification")
    redistribution = meta.get("allowRedistribution")
    res["checks"]["license"] = {
        # allowModificationRedistribution だけが「改変して再配布してよい」を意味する。
        # allowModification は改変だけで再配布は不可、prohibited は改変そのものが不可。
        "ok": modification == "allowModificationRedistribution" and redistribution is True,
        "name": meta.get("name"),
        "authors": meta.get("authors"),
        "modification": modification,
        "allowRedistribution": redistribution,
        "commercialUsage": meta.get("commercialUsage"),
        "avatarPermission": meta.get("avatarPermission"),
        "licenseUrl": meta.get("licenseUrl"),
    }

    total_v = sum(m["vertices"] for m in meshes)
    res["checks"]["vertex_budget"] = {
        "ok": 0 < total_v <= MAX_VERTICES,
        "vertices": total_v,
        "limit": MAX_VERTICES,
        "primitives": sum(m["primitives"] for m in meshes),
    }

    res["ok"] = all(c["ok"] for c in res["checks"].values())
    res["mb"] = round(size / 1048576, 1)
    return res


_LABELS = {
    "vrm10": "VRM 1.0",
    "humanoid": "Humanoid（必須17 + 指30）",
    "skinned": "スキンウェイト",
    "expressions": "表情",
    "spring_bone": "spring bone",
    "mtoon": "MToon",
    "license": "ライセンス",
    "vertex_budget": "頂点数",
}

# 落ちたときに何をすればよいか。検査結果だけ出して放り出さない。
_REMEDY = {
    "vrm10": "VRM 0.x は humanoid の定義が別物。VRM 1.0 で書き出し直すか、1.0 版を探す",
    "humanoid": "指が無いテンプレートは §38 の「指が動かない」をそのまま持ち越す。別候補を探す",
    "skinned": "人が塗ったウェイトが無いと自動ウェイトに戻り、§37 の破断を持ち越す",
    "expressions": "まばたきと母音が無いと KEHAI 側でリップシンクもアイドルも成立しない",
    "spring_bone": "髪とスカートの揺れは作り置きでしか入らない。無いテンプレートは揺れない",
    "mtoon": "KEHAI のトゥーン表示が MToon を前提にしている",
    "license": "テンプレートの幾何は出力に残る。改変・再配布可でないと配布できない",
    "vertex_budget": "削減が掛かっていない。VRoid Studio 側で削減して書き出し直すこと",
}


def report(path: Path, res: dict) -> None:
    lic = res["checks"]["license"]
    print(f"ファイル       {path}")
    print(f"サイズ         {res['mb']}MB")
    print(f"名前           {lic['name']}   作者 {lic['authors']}")
    print()
    print("メッシュ構成（68点ランドマーク対応を張る対象）")
    for m in res["meshes"]:
        morph = f"モーフ {m['morphs']}" if m["morphs"] else "モーフ なし"
        print(f"  {m['name']:12s} プリミティブ {m['primitives']:2d}   "
              f"頂点 {m['vertices']:6,d}   {morph}")
    print()

    for key, c in res["checks"].items():
        mark = "✓" if c["ok"] else "✗"
        print(f"{mark} {_LABELS[key]}")
        if key == "humanoid":
            print(f"    humanBones {c['total']}本   "
                  f"指 {c['fingers_present']}/{c['fingers_expected']}本")
            if c["missing_required"]:
                print(f"    必須の欠落: {c['missing_required']}")
        elif key == "skinned" and c["unskinned"]:
            print(f"    ウェイトが無い: {c['unskinned']}")
        elif key == "expressions":
            print(f"    モーフ {c['morphs']}個   結線済みプリセット {c['presets_bound']}個")
            if c["missing_presets"]:
                print(f"    未結線: {c['missing_presets']}")
        elif key == "spring_bone":
            print(f"    springs {c['springs']}   colliders {c['colliders']}")
        elif key == "mtoon":
            print(f"    マテリアル {c['total']}個")
            if c["non_mtoon"]:
                print(f"    MToon でない: {c['non_mtoon']}")
        elif key == "vertex_budget":
            print(f"    合計 {c['vertices']:,} 頂点 / {c['primitives']} プリミティブ"
                  f"（上限 {c['limit']:,}）")
        elif key == "license":
            print(f"    modification={c['modification']}   "
                  f"allowRedistribution={c['allowRedistribution']}")
            print(f"    commercialUsage={c['commercialUsage']}   "
                  f"avatarPermission={c['avatarPermission']}")
        if not c["ok"]:
            print(f"    → {_REMEDY[key]}")
    print()
    print("合格" if res["ok"] else "不合格 — 上の ✗ を満たす候補を探すこと")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--json", action="store_true", help="機械可読な行だけを出す")
    args = ap.parse_args()

    doc, size = read_gltf(args.path)
    res = check(doc, size)
    if not args.json:
        report(args.path, res)
        print()
    print("TEMPLATE_CHECK " + json.dumps(res, ensure_ascii=False))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
