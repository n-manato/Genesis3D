#!/usr/bin/env python3
"""スケルトンのルートノードの変換を子へ焼いて、ルートを単位行列にする。

## なぜ要るか

Rocketbox 由来のリグはルートノード `Bip01` が **scale 0.01 と回転**を持ち、
子の translation はセンチ単位で入っている（メッシュの POSITION はメートルの
世界座標なので、見た目は正しい）。これが2つの実害を生んでいた:

  1. **VRM アドオンの構造探索が破裂する**（docs/09 §17.2）
  2. **Blender の glTF インポータがルートの translation の変更を無視する**。
     骨を動かす補正（§19）を掛けると、Blender が読む rest が
     「私の書いた値 ＋ ルートの移動量」になる。実測でその差は 0.07592 で、
     Bip01 の移動量 0.92329 → 0.84737 の差と**完全に一致**した

ルートの変換を子へ焼けば、どの読み手から見ても素直な階層になる。**世界変換は
変わらない**ので、頂点・ウェイト・IBM は触らなくてよい（ルート自身が joint の
場合だけ、その IBM を単位行列にする）。

使い方:
    python3 -m vrm.normalize_rig --input in.glb --out out.glb --report r.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blender.vrm_finalize import read_glb, write_glb          # noqa: E402
from preprocessing.shape_transfer import (append_accessor, node_matrices)  # noqa: E402


def to_trs(m: np.ndarray):
    t = m[:3, 3]
    lin = m[:3, :3]
    s = np.linalg.norm(lin, axis=0)
    s = np.where(s < 1e-12, 1.0, s)
    r = lin / s
    # 回転行列 → クォータニオン（x, y, z, w）
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        k = np.sqrt(tr + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / k, (r[0, 2] - r[2, 0]) / k,
             (r[1, 0] - r[0, 1]) / k, 0.25 * k]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        k = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [0.25 * k, (r[0, 1] + r[1, 0]) / k, (r[0, 2] + r[2, 0]) / k,
             (r[2, 1] - r[1, 2]) / k]
    elif r[1, 1] > r[2, 2]:
        k = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 1] + r[1, 0]) / k, 0.25 * k, (r[1, 2] + r[2, 1]) / k,
             (r[0, 2] - r[2, 0]) / k]
    else:
        k = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[0, 2] + r[2, 0]) / k, (r[1, 2] + r[2, 1]) / k, 0.25 * k,
             (r[1, 0] - r[0, 1]) / k]
    return ([float(v) for v in t], [float(v) for v in q], [float(v) for v in s])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    doc, blob = read_glb(args.input)
    nodes = doc.get("nodes", [])
    locs, parent, order = node_matrices(doc)
    roots = [i for i in range(len(nodes)) if i not in parent]

    changed = []
    for r in roots:
        m = locs[r]
        if np.allclose(m, np.eye(4), atol=1e-9):
            continue
        for c in nodes[r].get("children", []) or []:
            newc = m @ locs[c]
            t, q, s = to_trs(newc)
            nodes[c].pop("matrix", None)
            nodes[c]["translation"] = t
            nodes[c]["rotation"] = q
            nodes[c]["scale"] = s
            locs[c] = newc
        for k in ("matrix", "translation", "rotation", "scale"):
            nodes[r].pop(k, None)
        locs[r] = np.eye(4)
        changed.append({"node": r, "name": nodes[r].get("name"),
                        "children": len(nodes[r].get("children", []) or [])})

    # ルート自身が joint なら、その IBM を単位行列に直す（世界変換が I になったため）
    out = bytearray(blob)
    rebound = 0
    if changed:
        locs2, parent2, order2 = node_matrices(doc)
        world = {}
        for i in order2:
            p = parent2.get(i)
            world[i] = (world[p] @ locs2[i]) if p is not None else locs2[i]
        for skin in doc.get("skins", []) or []:
            acc_i = skin.get("inverseBindMatrices")
            if acc_i is None:
                continue
            joints = skin.get("joints", [])
            ibm = np.stack([np.linalg.inv(world[j]) for j in joints])
            append_accessor(doc, out, acc_i,
                            ibm.transpose(0, 2, 1).reshape(len(joints), 16))
            rebound += 1

    write_glb(args.out, doc, bytes(out))
    rep = {"input": str(args.input), "out": str(args.out),
           "roots": len(roots), "normalized": changed, "skins_rebound": rebound}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    if changed:
        print("ルートの変換を子へ焼いた: "
              + ", ".join(f"{c['name']}（子 {c['children']}）" for c in changed)
              + f"  skin {rebound}個の IBM を張り直した → {args.out}")
    else:
        print(f"ルートは既に単位行列。何もしなかった → {args.out}")


if __name__ == "__main__":
    main()
