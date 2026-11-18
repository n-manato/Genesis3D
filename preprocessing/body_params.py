#!/usr/bin/env python3
"""体の寸法から、テンプレート素体を変形する対応点を作る。

`body_landmarks.py` が測った比率と `--height-cm` を、`shape_transfer.py` に渡せる
`{centers, targets}` に変換する。

## 何を動かして、何を動かさないか

**動かす**: 身長（全体の相似拡大）、肩幅、腰幅。
**動かさない**: 顔（生成メッシュに差し替わるので無意味）、手足の先端（アンカー）。

対応点は**関節位置**に置く。頂点に置くと衣服のシワまで動かそうとして破綻するし、
関節なら手塗りウェイトの構造と一致する。

## 身長は指定、比率は写真

写真1枚にスケールの手がかりは無い。身長は `--height-cm` で与え、
肩幅・腰幅の比率だけを写真から取る（`body_landmarks.py` の説明を参照）。

比率が信用できないとき（全身が写っていない写真は可視度が 0.02〜0.08 まで落ちる）は
`--ratios-only-if-visible` の既定 0.5 で自動的に身長だけの相似拡大に落ちる。

使い方:
    python -m preprocessing.body_params --body <body.json> \
        --template rocketbox_female_01 --out <pairs_body.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blender.vrm_finalize import read_glb                       # noqa: E402
from preprocessing.shape_transfer import read_accessor          # noqa: E402

TEMPLATES = Path("configs/templates.yaml")

# 動かす関節（Mixamo 命名）。scripts/prepare_rocketbox.py が改名済み。
SHOULDER = ("LeftArm", "RightArm")
HIP = ("LeftUpLeg", "RightUpLeg")
ANKLE = ("LeftFoot", "RightFoot")
SPINE = ("Hips", "Spine", "Spine1", "Spine2", "Neck")
# 横幅の調整を受ける関節。腕は肩に、脚は腰に追随させる。
SHOULDER_CHAIN = ("Shoulder", "Arm", "ForeArm", "Hand")
HIP_CHAIN = ("UpLeg", "Leg", "Foot", "ToeBase")


def node_globals(doc: dict) -> dict:
    """ノード名 → ワールド座標。glTF の TRS を親から辿る。"""
    nodes = doc.get("nodes", [])
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []) or []:
            parent[c] = i

    def local(n):
        m = np.eye(4)
        if "matrix" in n:
            return np.array(n["matrix"], dtype=float).reshape(4, 4).T
        if "scale" in n:
            m[:3, :3] = np.diag(n["scale"])
        if "rotation" in n:
            x, y, z, w = n["rotation"]
            r = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            m[:3, :3] = r @ m[:3, :3]
        if "translation" in n:
            m[:3, 3] = n["translation"]
        return m

    cache = {}

    def world(i):
        if i in cache:
            return cache[i]
        m = local(nodes[i])
        p = parent.get(i)
        cache[i] = (world(p) @ m) if p is not None else m
        return cache[i]

    out = {}
    for i, n in enumerate(nodes):
        name = n.get("name")
        if name:
            out[name] = world(i)[:3, 3]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", type=Path, required=True, help="body_landmarks.py の出力")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-visibility", type=float, default=0.5,
                    help="これ未満なら比率を信用せず身長だけ合わせる")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(TEMPLATES.read_text())["templates"]
    if args.template not in cfg:
        raise SystemExit(f"テンプレート '{args.template}' が無い: {list(cfg)}")
    tmpl = cfg[args.template]
    doc, blob = read_glb(Path(tmpl["mesh"]))
    J = node_globals(doc)

    missing = [k for k in SHOULDER + HIP + ANKLE + SPINE if k not in J]
    if missing:
        raise SystemExit(f"テンプレートに関節が無い: {missing}")

    V = np.vstack([read_accessor(doc, blob, p["attributes"]["POSITION"])
                   for m in doc["meshes"] for p in m["primitives"]])
    t_height = float(V[:, 1].max() - V[:, 1].min())

    # テンプレート自身の比率を、写真と同じ定義で測る
    t_sh = float(np.linalg.norm(J[SHOULDER[0]] - J[SHOULDER[1]]))
    t_hip = float(np.linalg.norm(J[HIP[0]] - J[HIP[1]]))
    sh_c = (J[SHOULDER[0]] + J[SHOULDER[1]]) / 2
    ank_c = (J[ANKLE[0]] + J[ANKLE[1]]) / 2
    t_span = float(np.linalg.norm(sh_c - ank_c))
    t_ratio = {"shoulder_over_span": t_sh / t_span, "hip_over_shoulder": t_hip / t_sh}

    body = json.loads(args.body.read_text())
    vis = body.get("min_visibility", 0.0)
    use_ratios = vis >= args.min_visibility
    if not use_ratios:
        print(f"比率を使わない（可視度 {vis} < {args.min_visibility}）。"
              f"身長だけ合わせる", file=sys.stderr)

    # 身長。指定が無ければテンプレートのまま。
    height_cm = body.get("height_cm")
    s_height = (height_cm / 100.0) / t_height if height_cm else 1.0
    # 横幅。比率が使えないときは 1.0（相似拡大のみ）。
    s_sh = (body["shoulder_over_span"] / t_ratio["shoulder_over_span"]) if use_ratios else 1.0
    s_hip = (body["hip_over_shoulder"] / t_ratio["hip_over_shoulder"]) if use_ratios else 1.0

    # 対応点。関節を「身長で相似拡大 → 肩と腰だけ横に伸縮」させる。
    # 足元（y=0）を基準に拡大するので、原点が足元という VRM の約束を壊さない。
    centers, targets, moved = [], [], {}
    for name, p in J.items():
        if name.startswith("Bip01") and name != "Bip01":
            continue                                    # 顔ボーンは触らない
        q = p.copy()
        q[1] *= s_height                                 # 身長（足元基準）
        q[0] *= s_height
        q[2] *= s_height
        if any(k in name for k in SHOULDER_CHAIN):
            q[0] *= s_sh
        elif any(k in name for k in HIP_CHAIN):
            q[0] *= s_hip
        if np.linalg.norm(q - p) > 1e-9:
            centers.append(p)
            targets.append(q)
            moved[name] = round(float(np.linalg.norm(q - p)) * 1000, 2)

    if not centers:
        raise SystemExit("動く関節が無い。身長も比率も既定値のままでは変形しない")

    C, G = np.asarray(centers), np.asarray(targets)
    d = np.linalg.norm(G - C, axis=1)
    out = {
        "centers": C.tolist(), "targets": G.tolist(),
        "_meta": {
            "template": args.template, "body": str(args.body),
            "template_height_m": round(t_height, 4),
            "target_height_cm": height_cm,
            "scale_height": round(s_height, 5),
            "scale_shoulder": round(s_sh, 5), "scale_hip": round(s_hip, 5),
            "ratios_used": use_ratios, "min_visibility": vis,
            "template_ratios": {k: round(v, 5) for k, v in t_ratio.items()},
            "photo_ratios": {k: body.get(k) for k in t_ratio},
            "joints_moved": len(centers),
            "displacement_mm": {"mean": round(float(d.mean()) * 1000, 2),
                                "max": round(float(d.max()) * 1000, 2)},
            "per_joint_mm": dict(sorted(moved.items(), key=lambda x: -x[1])[:10]),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False))
    if args.report:
        args.report.write_text(json.dumps(out["_meta"], ensure_ascii=False, indent=2))

    m = out["_meta"]
    print(f"テンプレ身長 {t_height:.3f}m → 目標 {height_cm}cm（倍率 {s_height:.4f}）")
    print(f"肩 ×{s_sh:.4f}  腰 ×{s_hip:.4f}  比率を使った: {use_ratios}")
    print(f"関節 {len(centers)}個  変位 平均 {m['displacement_mm']['mean']}mm / "
          f"最大 {m['displacement_mm']['max']}mm")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
