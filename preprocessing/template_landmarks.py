#!/usr/bin/env python3
"""テンプレート素体に68点ランドマークを打つ。**一度きりの作業。**

`game_avatar`（docs/08）の RBF 形状転写は、ICT-FaceKit の識別形状とテンプレートの
あいだに張った68点の対応で駆動される。ICT 側は `vertex_indices.json` の
`idx_to_landmark_verts` として最初から用意されているので、作るのはテンプレート側だけ。

手順:

  1. 頭を正射投影で正面から撮る（`blender/ortho_front.py --frame head`）
  2. その画像に insightface の 68点検出を当てる
  3. 画素 → 3D をレイキャストで戻す（`ict_fit.backproject` と同じ式）
  4. 一番近い**顔マテリアルの頂点**へ吸着させ、頂点番号として保存する

3 で得た点はメッシュ表面上にあるが、頂点そのものではない。RBF の中心は位置さえ
あれば動くものの、**頂点番号で持つほうが後から目視で直せる**ので吸着させている。

素体は固定なので、この作業はテンプレートごとに一度だけ。出力は
`configs/landmarks/<template>.json` に置いてリポジトリに入れる。

`--review` を付けると、検出点を描き込んだ画像も出す。**必ず目で見ること。**
自動検出は正面向きの実写顔では概ね当たるが、当たったかどうかは数字では分からない。

使い方:
    python -m preprocessing.template_landmarks \\
        --mesh assets/templates/rocketbox/male_adult_01.glb \\
        --render <ortho.png> --render-meta <ortho.json> \\
        --face-material head \\
        --out configs/landmarks/rocketbox_male_01.json \\
        --review jobs/_template/landmarks_male.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blender.vrm_finalize import read_glb                      # noqa: E402
from preprocessing.shape_transfer import read_accessor         # noqa: E402


def face_primitive(doc: dict, blob: bytes, material: str):
    """顔マテリアルのプリミティブを (mesh_i, prim_i, 頂点, 三角形) で返す。

    **スロット0の決め打ちはしない。** glTF のインポータは複数プリミティブを
    1オブジェクトに畳むので、スロット0が顔とは限らない（docs/08 §10.4c）。
    """
    hits = []
    for mi, mesh in enumerate(doc.get("meshes", [])):
        for pi, prim in enumerate(mesh.get("primitives", [])):
            mat = prim.get("material")
            name = doc["materials"][mat].get("name", "") if mat is not None else ""
            if material in name:
                v = read_accessor(doc, blob, prim["attributes"]["POSITION"])
                idx = doc["accessors"][prim["indices"]]
                bv = doc["bufferViews"][idx["bufferView"]]
                off = bv.get("byteOffset", 0) + idx.get("byteOffset", 0)
                dt = {5121: "<u1", 5123: "<u2", 5125: "<u4"}[idx["componentType"]]
                tri = np.frombuffer(blob, dtype=dt, count=idx["count"],
                                    offset=off).reshape(-1, 3).astype(np.int64)
                hits.append((mi, pi, v, tri, name))
    if not hits:
        names = sorted({doc["materials"][p["material"]].get("name", "")
                        for m in doc["meshes"] for p in m["primitives"]
                        if p.get("material") is not None})
        raise SystemExit(f"マテリアル '{material}' が見つからない。候補: {names}")
    if len(hits) > 1:
        print(f"注意: '{material}' に {len(hits)}個のプリミティブが一致した "
              f"({[h[4] for h in hits]})。最初のものを使う", file=sys.stderr)
    return hits[0]


def snap(points: np.ndarray, verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """各点を最近傍の頂点へ吸着。(頂点番号, 距離) を返す。"""
    idx = np.empty(len(points), dtype=np.int64)
    dist = np.full(len(points), np.nan)
    for i, p in enumerate(points):
        if not np.isfinite(p).all():
            idx[i] = -1
            continue
        d = np.linalg.norm(verts - p, axis=1)
        j = int(d.argmin())
        idx[i] = j
        dist[i] = float(d[j])
    return idx, dist


def draw_review(render: Path, uv: np.ndarray, ok: np.ndarray, out: Path) -> None:
    """検出点を描き込む。**目視用。数字では当たり外れが分からない。**"""
    from PIL import Image, ImageDraw
    im = Image.open(render).convert("RGB")
    d = ImageDraw.Draw(im)
    for i, (x, y) in enumerate(uv):
        # dlib/insightface の68点の並び: 0-16 輪郭 / 17-26 眉 / 27-35 鼻 /
        # 36-47 目 / 48-67 口
        col = ((255, 80, 80) if i < 17 else (255, 200, 0) if i < 27 else
               (80, 200, 255) if i < 36 else (120, 255, 120) if i < 48 else
               (255, 120, 255))
        if not ok[i]:
            col = (255, 0, 0)
        r = 3
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)
        if i % 6 == 0:
            d.text((x + 5, y - 5), str(i), fill=col)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", type=Path, required=True, help="テンプレートの GLB/VRM")
    ap.add_argument("--render", type=Path, required=True, help="ortho_front の出力")
    ap.add_argument("--render-meta", type=Path, required=True)
    ap.add_argument("--face-material", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--review", type=Path)
    ap.add_argument("--template-id", default=None)
    args = ap.parse_args()

    from PIL import Image
    import trimesh
    from preprocessing.ict_fit import backproject, detect_68

    doc, blob = read_glb(args.mesh)
    mi, pi, verts, tri, matname = face_primitive(doc, blob, args.face_material)
    meta = json.loads(args.render_meta.read_text())

    uv = detect_68(Image.open(args.render))

    # レイキャストは**顔のプリミティブだけ**に当てる。全身を混ぜると、髪や襟に
    # 先に当たった点が顔の表面として採用されてしまう。
    mesh = trimesh.Trimesh(vertices=verts, faces=tri, process=False)
    pts = backproject(uv, meta, mesh)
    ok = np.isfinite(pts).all(axis=1)

    idx, dist = snap(pts, verts)
    good = ok & (idx >= 0)

    lm = [{"i": int(i), "vertex": int(idx[i]),
           "xyz": [round(float(c), 6) for c in verts[idx[i]]],
           "uv_px": [round(float(uv[i, 0]), 2), round(float(uv[i, 1]), 2)],
           "snap_mm": None if not good[i] else round(float(dist[i]) * 1000, 3)}
          for i in range(len(uv)) if good[i]]

    out = {
        "template": args.template_id or args.mesh.stem,
        "mesh": str(args.mesh),
        "face_material": matname,
        "gltf": {"mesh": mi, "primitive": pi, "vertices": int(len(verts))},
        "render": str(args.render),
        "source": "auto-bootstrap (insightface antelopev2 landmark_3d_68)",
        "detected": int(len(uv)),
        "landed": int(good.sum()),
        "missed": [int(i) for i in range(len(uv)) if not good[i]],
        "snap_mm": {"mean": round(float(np.nanmean(dist[good])) * 1000, 3),
                    "max": round(float(np.nanmax(dist[good])) * 1000, 3)},
        "landmarks": lm,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    if args.review:
        draw_review(args.render, uv, good, args.review)

    print(f"マテリアル '{matname}'  mesh{mi}/prim{pi}  {len(verts):,}頂点")
    print(f"検出 {len(uv)}点 / 面に着地 {good.sum()}点"
          + (f"  **外れ: {out['missed']}**" if out["missed"] else ""))
    print(f"頂点への吸着距離  平均 {out['snap_mm']['mean']}mm / "
          f"最大 {out['snap_mm']['max']}mm")
    print(f"→ {args.out}")
    if args.review:
        print(f"→ {args.review}  **目で見て確かめること**")


if __name__ == "__main__":
    main()
