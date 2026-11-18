#!/usr/bin/env python3
"""VRM に `VRMC_springBone` を足して**髪を揺らす**。

## なぜ VRM でないと揺れないのか

KEHAI の `frontend-src/src/renderer/src/utils/glb-humanoid.ts` が明記している ——
プレーンなリグ付き glb は

    What such a model does NOT get, because the file cannot describe them:
    spring bones (hair/cloth physics), VRM expression presets, ...

を得られない。KEHAI は `@pixiv/three-vrm` の `springBoneManager` を毎フレーム
更新しているが、それが作られるのは `VRMC_vrm` を持つ本物の VRM のときだけ。
だから髪を揺らすには **VRM で書き出して `VRMC_springBone` を持たせる**必要がある。

## 節の数

VRM の spring は `joints` の**隣り合う2節**で1区間を作る（three-vrm は
`joints[i+1]` を `joints[i]` の子として使う）。だから N本の骨を揺らすには
**N+1個の節**が要る。`blender/add_hair.py` は末端に `*_tip` という
ウェイトを持たない骨を1本余分に作ってある。

Blender を経由しない。JSON の追記だけなので頂点・リグ・モーフ・テクスチャは
一度も触られない（`vrm_finalize.py` と同じ方針）。

使い方:
    python3 -m vrm.add_springs --input character.vrm --out character.vrm \\
        --prefix HairSpring --report r.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blender.vrm_finalize import read_glb, write_glb  # noqa: E402


def chains_from_nodes(doc: dict, prefix: str) -> dict[str, list[int]]:
    """`<prefix><group>_<k>` という名前からチェーンを組み立てる。

    親子の実際の繋がりで並べる。名前の数字順に頼ると、書き出し側が名前を
    変えた場合に静かに壊れる。
    """
    nodes = doc.get("nodes", [])
    idx = {i: n.get("name", "") for i, n in enumerate(nodes)}
    hit = {i: nm for i, nm in idx.items() if nm.startswith(prefix)}
    if not hit:
        return {}
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    groups: dict[str, list[int]] = {}
    for i, nm in hit.items():
        m = re.match(rf"{re.escape(prefix)}(\d+)_", nm)
        groups.setdefault(m.group(1) if m else nm, []).append(i)
    out = {}
    for g, members in groups.items():
        s = set(members)
        roots = [i for i in members if parent.get(i) not in s]
        if len(roots) != 1:
            print(f"警告: グループ {g} の根が {len(roots)}個。飛ばす", file=sys.stderr)
            continue
        chain, cur = [], roots[0]
        while cur is not None:
            chain.append(cur)
            nxt = [c for c in nodes[cur].get("children", []) if c in s]
            cur = nxt[0] if nxt else None
        out[g] = chain
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix", default="HairSpring")
    ap.add_argument("--stiffness", type=float, default=1.0,
                    help="元の姿勢へ戻る強さ。大きいと硬い")
    ap.add_argument("--drag", type=float, default=0.45, help="減衰。大きいと早く止まる")
    ap.add_argument("--gravity", type=float, default=0.15)
    ap.add_argument("--hit-radius", type=float, default=0.02,
                    help="コライダーとの当たり半径（m）")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    doc, blob = read_glb(args.input)
    ext = doc.setdefault("extensions", {})
    if "VRMC_vrm" not in ext:
        raise SystemExit(
            f"{args.input} は VRM ではない（VRMC_vrm が無い）。"
            f"**プレーンな glb では KEHAI は髪を揺らせない** —— "
            f"blender/export_vrm.py を通してから実行すること")

    chains = chains_from_nodes(doc, args.prefix)
    if not chains:
        names = [n.get("name") for n in doc.get("nodes", [])
                 if (n.get("name") or "").startswith("Hair")]
        raise SystemExit(f"'{args.prefix}' で始まるノードが無い。Hair系: {names[:8]}")

    springs = []
    for g, chain in sorted(chains.items()):
        if len(chain) < 2:
            print(f"警告: グループ {g} は節が {len(chain)}個で区間が作れない",
                  file=sys.stderr)
            continue
        springs.append({
            "name": f"hair_{g}",
            "joints": [{
                "node": n,
                "hitRadius": args.hit_radius,
                "stiffness": args.stiffness,
                "gravityPower": args.gravity,
                "gravityDir": {"x": 0.0, "y": -1.0, "z": 0.0},
                "dragForce": args.drag,
            } for n in chain],
        })
    if not springs:
        raise SystemExit("区間を作れるチェーンが1本も無い")

    ext["VRMC_springBone"] = {"specVersion": "1.0",
                              "colliders": [], "colliderGroups": [],
                              "springs": springs}
    used = doc.setdefault("extensionsUsed", [])
    if "VRMC_springBone" not in used:
        used.append("VRMC_springBone")

    write_glb(args.out, doc, blob)
    rep = {"input": str(args.input), "out": str(args.out),
           "springs": len(springs),
           "joints": sum(len(s["joints"]) for s in springs),
           "chains": {g: [doc["nodes"][i].get("name") for i in c]
                      for g, c in sorted(chains.items())},
           "settings": {"stiffness": args.stiffness, "dragForce": args.drag,
                        "gravityPower": args.gravity,
                        "hitRadius": args.hit_radius}}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"spring {len(springs)}本 / 節 {rep['joints']}個 → {args.out}")
    for g, c in rep["chains"].items():
        print(f"  hair_{g}: {' → '.join(c)}")


if __name__ == "__main__":
    main()
