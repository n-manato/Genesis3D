#!/usr/bin/env python3
"""リグ付き glb → VRM 1.0。**VRM アドオンを使わず JSON を直接書く。**

## なぜアドオンを使わないか

`blender/export_vrm.py`（VRM Add-on 4.5.0 経由）は Rocketbox 由来のリグでは
**何も書き出さずに CANCELLED を返す**。アドオンの humanoid 構造探索が破裂する:

    Structure search completed: Total searches: 80,686,221 (soft limit: 1,000,000)

原因を切り分けた（どれも違った）:

  - 脚の別名  … `leftUpperLeg←LeftLeg` の誤対応は直したが失敗は続く
  - Icosphere … 除去しても続く
  - ルートの scale 0.01 … 焼いても続く（探索回数は同一）
  - 顔ボーン26本の名前 … `Face_` を付けても**探索回数が 80,686,221 で完全に同一**
  - 私が足した髪のボーン … 無くても失敗する（**素のテンプレートでも失敗**）

残るのは階層の形そのもの。アドオンの自動推定に頼る必要は無い ——
**どのボーンがどの humanoid ボーンかはこちらが知っている**。

glTF のスキンはノードを**番号**で参照するので、拡張の追記だけで済む。頂点・
リグ・モーフ・テクスチャは一度も触られない（`vrm_finalize.py` と同じ方針で、
あちらが拡張を消すのに対してこちらは書く）。

## 対応表

KEHAI の `glb-humanoid.ts` が持っているものと同じ。Mixamo 系の命名は
Ready Player Me / Avaturn / Character Creator も共有する事実上の標準。
**`UpLeg` が腿・`Leg` が脛**であることに注意（ここを取り違えると VRM が壊れる）。

使い方:
    python3 -m vrm.make_vrm --input avatar.glb --out avatar.vrm \\
        --name "Subject A" --author Genesis3D --report r.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blender.vrm_finalize import read_glb, write_glb  # noqa: E402

# VRM humanoid ボーン名 → ノード名の候補（良い順）。KEHAI の glb-humanoid.ts と同一。
BONE_MAP: dict[str, list[str]] = {
    "hips": ["Hips"], "spine": ["Spine"], "chest": ["Spine1"],
    "upperChest": ["Spine2"], "neck": ["Neck"], "head": ["Head"],
    "leftEye": ["LeftEye"], "rightEye": ["RightEye"], "jaw": ["MJaw", "Jaw"],
    "leftShoulder": ["LeftShoulder"], "leftUpperArm": ["LeftArm"],
    "leftLowerArm": ["LeftForeArm"], "leftHand": ["LeftHand"],
    "rightShoulder": ["RightShoulder"], "rightUpperArm": ["RightArm"],
    "rightLowerArm": ["RightForeArm"], "rightHand": ["RightHand"],
    # **UpLeg が腿、Leg が脛。** 逆にすると VRM の脚が壊れる。
    "leftUpperLeg": ["LeftUpLeg"], "leftLowerLeg": ["LeftLeg"],
    "leftFoot": ["LeftFoot"], "leftToes": ["LeftToeBase"],
    "rightUpperLeg": ["RightUpLeg"], "rightLowerLeg": ["RightLeg"],
    "rightFoot": ["RightFoot"], "rightToes": ["RightToeBase"],
}
FINGERS = {
    f"{side}{name}{seg}": [f"{'Left' if side == 'left' else 'Right'}Hand"
                           f"{cap}{i}"]
    for side in ("left", "right")
    for name, cap in (("Thumb", "Thumb"), ("Index", "Index"),
                      ("Middle", "Middle"), ("Ring", "Ring"),
                      ("Little", "Pinky"))
    for seg, i in (("Metacarpal", 1), ("Proximal", 2), ("Distal", 3))
    if False  # 指の3段は VRM の命名規則が段ごとに違うので下で明示的に作る
}
FINGER_MAP: dict[str, list[str]] = {}
for _side, _pre in (("left", "Left"), ("right", "Right")):
    for _vrm, _mix in (("Thumb", "Thumb"), ("Index", "Index"),
                       ("Middle", "Middle"), ("Ring", "Ring"),
                       ("Little", "Pinky")):
        for _seg, _i in (("Metacarpal", 1), ("Proximal", 2), ("Distal", 3)):
            FINGER_MAP[f"{_side}{_vrm}{_seg}"] = [f"{_pre}Hand{_mix}{_i}"]
BONE_MAP.update(FINGER_MAP)

REQUIRED = ["hips", "spine", "neck", "head",
            "leftUpperArm", "leftLowerArm", "leftHand",
            "rightUpperArm", "rightLowerArm", "rightHand",
            "leftUpperLeg", "leftLowerLeg", "leftFoot",
            "rightUpperLeg", "rightLowerLeg", "rightFoot"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="Genesis3D Character")
    ap.add_argument("--author", default="Genesis3D")
    ap.add_argument("--license-url", default="https://vrm.dev/licenses/1.0/")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    doc, blob = read_glb(args.input)
    nodes = doc.get("nodes", [])
    by_name: dict[str, int] = {}
    for i, n in enumerate(nodes):
        nm = n.get("name")
        if nm and nm not in by_name:
            by_name[nm] = i

    human, missing = {}, []
    for vrm_name, cands in BONE_MAP.items():
        hit = next((by_name[c] for c in cands if c in by_name), None)
        if hit is None:
            missing.append(vrm_name)
        else:
            human[vrm_name] = {"node": hit}
    req_missing = [b for b in REQUIRED if b in missing]
    if req_missing:
        raise SystemExit(
            f"必須ボーンが見つからない: {req_missing}\n"
            f"ノード名の例: {sorted(by_name)[:20]}")

    ext = doc.setdefault("extensions", {})
    ext["VRMC_vrm"] = {
        "specVersion": "1.0",
        "meta": {
            "name": args.name,
            "version": "1",
            "authors": [args.author],
            "licenseUrl": args.license_url,
            # テンプレートは Microsoft Rocketbox（MIT）。制限を掛ける理由が無い。
            "avatarPermission": "everyone",
            "allowExcessivelyViolentUsage": False,
            "allowExcessivelySexualUsage": False,
            "commercialUsage": "corporation",
            "allowPoliticalOrReligiousUsage": False,
            "allowAntisocialOrHateUsage": False,
            "creditNotation": "unnecessary",
            "allowRedistribution": True,
            "modification": "allowModificationRedistribution",
        },
        "humanoid": {"humanBones": human},
        # 目線と一人称は省略可。KEHAI は自前の look-at を持っている
        # （utils/space/look-at-viewer.ts）ので、ここで縛らない。
    }
    used = doc.setdefault("extensionsUsed", [])
    for k in ("VRMC_vrm",):
        if k not in used:
            used.append(k)

    # **薄いシェルは両面にする。** 髪（生成メッシュから切り出した 2〜3万頂点の
    # 穴だらけのシェル）は裏面カリングされると穴から向こう側の裏面が消えて
    # **透けて見える**（斑点状）。docs/05 §30 が衣装の裾で同じ現象を記録しており、
    # `blender/export_vrm.py` の `set_double_sided` がそのための処置だったが、
    # アドオンを迂回した本スクリプトはそれを落としていた。
    ds = 0
    for m in doc.get("materials", []):
        if not m.get("doubleSided"):
            m["doubleSided"] = True
            ds += 1
    write_glb(args.out, doc, blob)
    rep = {"input": str(args.input), "out": str(args.out),
           "double_sided_set": ds,
           "human_bones": len(human), "required_ok": True,
           "optional_missing": missing,
           "mapped": {k: nodes[v["node"]].get("name") for k, v in human.items()},
           "size_mb": round(args.out.stat().st_size / 1048576, 1)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"VRM 1.0 を書いた: humanoid {len(human)}本"
          f"（任意で未対応 {len(missing)}）  {rep['size_mb']}MB → {args.out}")
    if missing:
        print(f"  未対応（任意）: {missing}")


if __name__ == "__main__":
    main()
