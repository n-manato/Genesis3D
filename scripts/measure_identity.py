#!/usr/bin/env python3
"""出来上がったアバターの**同一性**を測る。held-out ArcFace。

## なぜ専用スクリプトにするか

この指標を測らずに「完了」と言ったのが、このプロジェクトで最大の失敗だった
（docs/08 §10.11）。テンプレート方式は見た目が整っていたので合格に見えたが、
実際は 0.4554 で旧パイプラインの 0.7790 から 0.32 の後退だった。

**基準にする写真は、テクスチャの由来と別のセッションのものでなければならない。**
由来写真と比べると「同じ画像を貼ったか」を測ることになり、同一性の指標にならない。
実測（antelopev2）:

    実写どうし（tweed vs studio）           0.8203   ← 事実上の上限
    旧パイプライン subject_a_v2             0.7790 / 0.7680
    game_avatar v2（生成頭＋テンプレ体）    0.6917（顔493px）
    テンプレート変形のみ（v1）              0.4554 / 0.4469
    Rocketbox 素体そのまま（本人性ゼロ）   -0.0751 / -0.0782

**顔の画素数を必ず併記する。** 全身の枠では顔が 55px しかなく、これ抜きの
ArcFace は解釈できない（docs/05 §18）。

使い方:
    python3 scripts/measure_identity.py --model jobs/x/output/avatar.glb \\
        --refs subject_a/subject_face_tweed_head.png subject_a/subject_face_studio_600.jpg \\
        --source subject_a/subject-a_head_1024.png \\
        --out jobs/x/scores/identity.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLENDER = os.environ.get("BLENDER_BIN", "/home/ai-ccore/opt/blender/blender")
FACE_PY = Path.home() / "miniconda3/envs/genesis-face/bin/python"

MEASURE = r'''
import json, sys, numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(1024, 1024))

def emb(path):
    im = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1]
    fs = app.get(im)
    if not fs:
        return None, 0.0
    f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return f.normed_embedding, float(f.bbox[2] - f.bbox[0])

spec = json.loads(sys.argv[1])
out = {"renders": {}, "refs": {}}
for name, p in spec["refs"].items():
    e, px = emb(p)
    out["refs"][name] = {"path": p, "face_px": round(px, 1), "detected": e is not None}
    if e is not None:
        out["refs"][name]["_e"] = e.tolist()
for name, p in spec["renders"].items():
    e, px = emb(p)
    r = {"path": p, "face_px": round(px, 1), "detected": e is not None, "vs": {}}
    if e is not None:
        for rn, rd in out["refs"].items():
            if "_e" in rd:
                r["vs"][rn] = round(float(np.dot(e, np.array(rd["_e"]))), 4)
    out["renders"][name] = r
# 参照どうし（上限の目安）
names = [n for n, d in out["refs"].items() if "_e" in d]
out["ref_pairs"] = {f"{a}|{b}": round(float(np.dot(
        np.array(out["refs"][a]["_e"]), np.array(out["refs"][b]["_e"]))), 4)
    for i, a in enumerate(names) for b in names[i + 1:]}
for d in out["refs"].values():
    d.pop("_e", None)
print("MEASURE_JSON " + json.dumps(out, ensure_ascii=False))
'''


def render(model: Path, out_png: Path, meta: Path, frame: str,
           frame_material: str | None, res: int) -> bool:
    cmd = [BLENDER, "--background", "--factory-startup", "--python",
           str(ROOT / "blender/ortho_front.py"), "--",
           "--input", str(model), "--out", str(out_png), "--meta", str(meta),
           "--res", str(res), "--frame", frame]
    if frame_material:
        cmd += ["--frame-material", frame_material]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return r.returncode == 0 and out_png.exists()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--refs", type=Path, nargs="+", required=True,
                    help="**テクスチャの由来と別セッション**の実写")
    ap.add_argument("--source", type=Path, default=None,
                    help="テクスチャの由来写真。参考値として一緒に測る（合否には使わない）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--renders", type=Path, default=None, help="レンダの保存先")
    ap.add_argument("--face-material", default="head",
                    help="バスト枠を決めるマテリアル。生成頭の混ざったモデルでは Material_0")
    ap.add_argument("--res", type=int, default=1024)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="ident_"))
    rd = args.renders or tmp
    rd.mkdir(parents=True, exist_ok=True)

    renders = {}
    # バスト枠。顔の画素を稼ぐ。マテリアル指定が効かない個体は head_fraction に落とす。
    if render(args.model, rd / "bust.png", tmp / "bust.json", "head",
              args.face_material, args.res):
        renders["bust"] = str(rd / "bust.png")
    elif render(args.model, rd / "bust.png", tmp / "bust.json", "head", None, args.res):
        renders["bust"] = str(rd / "bust.png")
    # 全身枠。**こちらが主用途**（モーション）。顔は数十pxしかない。
    if render(args.model, rd / "full.png", tmp / "full.json", "full", None, args.res):
        renders["full"] = str(rd / "full.png")
    if not renders:
        raise SystemExit("レンダに失敗した")

    refs = {p.stem: str(p) for p in args.refs}
    if args.source:
        refs["SOURCE:" + args.source.stem] = str(args.source)

    r = subprocess.run([str(FACE_PY), "-c", MEASURE,
                        json.dumps({"renders": renders, "refs": refs})],
                       capture_output=True, text=True, cwd=ROOT)
    line = next((l for l in r.stdout.splitlines() if l.startswith("MEASURE_JSON ")), None)
    if line is None:
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        raise SystemExit("測定に失敗した")
    res = json.loads(line[len("MEASURE_JSON "):])
    res["model"] = str(args.model)

    # held-out の平均。**SOURCE: を除く。** 由来写真は合否に混ぜない。
    for name, d in res["renders"].items():
        held = [v for k, v in d["vs"].items() if not k.startswith("SOURCE:")]
        d["held_out_mean"] = round(sum(held) / len(held), 4) if held else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2))

    # **同一セッションの写真を held-out と誤認しないこと。** 実際に踏んだ:
    # subject_face_studio_600 と由来写真の類似度が 0.9773 で、別セッションではなかった。
    # それを基準に混ぜていたので、docs/08 までの数値は実際より高く出ている。
    warn = []
    if args.source:
        sk = "SOURCE:" + args.source.stem
        for k, v in res["ref_pairs"].items():
            a, b = k.split("|")
            other = b if a == sk else (a if b == sk else None)
            if other is not None and v > 0.93:
                warn.append((other, v))
    res["same_session_warning"] = [{"ref": r, "vs_source": v} for r, v in warn]

    print(f"モデル {args.model}")
    for n, d in res["ref_pairs"].items():
        print(f"  参照どうし {n}: {d}")
    for r, v in warn:
        print(f"  ! 警告: '{r}' は由来写真との類似度が {v:.4f} で、"
              f"**同一セッションの可能性が高い**。held-out の基準にならない")
    for name, d in res["renders"].items():
        if not d["detected"]:
            print(f"  {name:5s}: 顔を検出できず")
            continue
        vs = "  ".join(f"{k}={v:+.4f}" for k, v in d["vs"].items())
        print(f"  {name:5s}: 顔 {d['face_px']:.0f}px   held-out 平均 "
              f"{d['held_out_mean']:+.4f}   [{vs}]")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
