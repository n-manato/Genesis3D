#!/usr/bin/env python3
"""【非推奨】写真 → ゲームアバター。生成した頭 ＋ テンプレートの体を首で接合する。

**これは接合方式で、ラップ方式（`scripts/gen_wrap_avatar.py`・docs/09）に
全指標で劣る。** 比較の基準として残しているだけで、新規に使う理由は無い。

    | | 接合（これ） | ラップ |
    |---|---|---|
    | held-out ArcFace バスト | 0.6189 | 0.6798 |
    | held-out ArcFace 全身 | 0.5329 | 0.6471 |
    | 頂点数 | 129,458 | 33,433 |
    | 身長 | 指定と 7.7cm ずれる | 誤差 0.3mm |
    | モーフ | 53（転写） | 175（テンプレ本来） |
    | 髪 | 生成メッシュを削減（黒い斑点） | 切り出して別メッシュ・揺れる |


## なぜこの構成か（docs/08 §10.11〜§10.13）

同一性とリグ品質は別々の場所から来る。片方だけを取ると片方が壊れる:

| | 頭 | 体 | held-out ArcFace | リグ |
|---|---|---|---|---|
| 旧パイプライン | 生成 | **生成** | 0.774 | **指が死に肩が破断** |
| テンプレート方式 | **テンプレ** | テンプレ | **0.451** | 完璧 |
| **これ** | **生成** | **テンプレ** | **0.746**（バスト枠） | 完璧 |

- **同一性は「写真に一致した頭の幾何」に写真を貼ることで出る。** 他人の形の頭に
  貼ると顔が歪む（テンプレート方式が 0.451 に落ちた原因）
- **リグ品質はテンプレートから来る。** 生成した体は指の骨がメッシュの外に浮く（§38）

## 順序が決定的

**投影は接合の前。** 逆にすると顔が全身の一部になり、整列レンダの顔が
248.8px → 65.5px に落ちて ArcFace が 0.6634 → 0.3445 になる（docs/05 §19）。

**削減は投影の前。** 逆にすると焼いた UV が削減で歪んで顔がずれる。

使い方:
    python3 scripts/gen_game_avatar.py \
        --face subject_a/subject-a_head_1024.png \
        --body subject_a/subject_tpose_front.png --height-cm 160 \
        --template rocketbox_female_01 --out-dir jobs/ga_v2_002

**顔写真は加工していないものを選ぶこと。** SDXL で復元した画像は解像度が高くても
held-out で 0.642、素の写真が 0.773 だった（docs/08 §10.12）。
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = {k: Path.home() / "miniconda3/envs" / v / "bin/python"
       for k, v in {"eval": "genesis-eval", "face": "genesis-face",
                    "trellis": "trellis2"}.items()}
BLENDER = os.environ.get("BLENDER_BIN", "/home/ai-ccore/opt/blender/blender")


def run(cmd: list[str], what: str, log: Path) -> str:
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    dt = time.time() - t
    if r.returncode != 0:
        print(f"  ✗ {what}（{dt:.1f}s） — {log}", file=sys.stderr)
        print(r.stderr[-1500:], file=sys.stderr)
        raise SystemExit(1)
    print(f"  ✓ {what}（{dt:.1f}s）")
    return r.stdout


def blender(script: str, args: list[str], what: str, log: Path) -> str:
    return run([BLENDER, "--background", "--factory-startup", "--python",
                str(ROOT / "blender" / script), "--"] + args, what, log)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--face", type=Path, required=True, help="顔の写真（加工していないもの）")
    ap.add_argument("--body", type=Path, default=None, help="全身写真。体型を寄せる")
    ap.add_argument("--height-cm", type=float, default=None)
    ap.add_argument("--template", default="rocketbox_female_01")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resolution", default="1024", choices=("512", "1024"))
    ap.add_argument("--head-target", type=int, default=100_000,
                    help="頭の削減目標。既定で最終 12〜16万頂点になる")
    ap.add_argument("--gpu", default="1", help="TRELLIS を回す GPU")
    args = ap.parse_args()

    d = args.out_dir
    (d / "logs").mkdir(parents=True, exist_ok=True)
    (d / "scores").mkdir(exist_ok=True)
    (d / "output").mkdir(exist_ok=True)
    L = d / "logs"
    t0 = time.time()
    print(f"顔 {args.face}  体 {args.body}  テンプレ {args.template}")

    # --- 1) 顔: 頭部切り出し → 背景除去 -----------------------------------
    run([str(ENV["eval"]), "-m", "preprocessing.head_crop", "--input", str(args.face),
         "--out", str(d / "head_crop.png"), "--meta", str(d / "scores/crop.json")],
        "頭部の切り出し", L / "01_crop.log")
    run([str(ENV["eval"]), "-m", "preprocessing.rmbg", "--input", str(d / "head_crop.png"),
         "--output", str(d / "head_rgba.png"), "--meta", str(d / "scores/rmbg.json")],
        "背景除去", L / "02_rmbg.log")

    # --- 2) 顔: TRELLIS をワンショットで（常駐させない）-------------------
    run([str(ENV["trellis"]), str(ROOT / "adapters/trellis/oneshot.py"),
         "--image", str(d / "head_rgba.png"), "--out", str(d / "head_raw.glb"),
         "--resolution", args.resolution, "--gpu", args.gpu,
         "--report", str(d / "scores/trellis.json")],
        f"頭の生成（TRELLIS {args.resolution}・GPU{args.gpu}）", L / "03_trellis.log")

    blender("repair_mesh.py", ["--input", str(d / "head_raw.glb"),
                               "--out", str(d / "head_clean.glb"),
                               "--report", str(d / "scores/repair.json")],
            "メッシュ修復", L / "04_repair.log")
    # **削減は投影の前。** 逆にすると焼いた UV が歪む。
    blender("decimate_mesh.py", ["--input", str(d / "head_clean.glb"),
                                 "--out", str(d / "head_small.glb"),
                                 "--target", str(args.head_target),
                                 "--report", str(d / "scores/decimate.json")],
            "頭の削減", L / "05_decimate.log")
    # **削減の後にもう一度修復する。** COLLAPSE は髪のような細い構造で
    # 退化面と法線反転を作り、レンダに黒い斑点として出る。それが顔検出を壊す
    # （実測: 修復無しでは全ての余白で検出できなかった）。
    blender("repair_mesh.py", ["--input", str(d / "head_small.glb"),
                               "--out", str(d / "head_ready.glb"),
                               "--report", str(d / "scores/repair2.json")],
            "削減後の修復", L / "05b_repair2.log")

    # --- 3) 顔: 実写を投影（**接合の前**）---------------------------------
    #
    # **余白は当たり外れがある。** 顔が画枠を占めすぎると SCRFD が検出できず、
    # 小さすぎると解像度が落ちる。頭の形は生成のたびに違うので、
    # 広い順に試して最初に通ったものを使う（実測で 1.6 が通る個体と
    # 2.0 でないと通らない個体があった）。
    # **顔が最大になる余白を選ぶ。** 「最初に通ったもの」だと、たまたま
    # 広い余白が選ばれて顔の画素が減り、テクスチャの質が落ちる
    # （実測: 余白 2.0 で held-out 0.745、2.6 が選ばれた回は 0.684）。
    # レンダは1枚2秒なので全部試して最良を採る。
    best = None
    for m in ("1.35", "1.6", "2.0", "2.6"):
        blender("project_face.py", ["--input", str(d / "head_ready.glb"),
                                    "--emit-align-render", str(d / f"align_{m}.png"),
                                    "--frame", "head", "--frame-margin", m,
                                    "--align-res", "2048"],
                f"整列レンダ（余白 {m}）", L / f"06_align_{m}.log")
        r = subprocess.run(
            [str(ENV["eval"]), "-m", "preprocessing.face_projection",
             "--photo", str(args.face), "--render", str(d / f"align_{m}.png"),
             "--out", str(d / f"warped_{m}.png"),
             "--meta", str(d / f"scores/warped_{m}.json")],
            capture_output=True, text=True, cwd=ROOT)
        (L / f"07_warp_{m}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        if r.returncode != 0 or not (d / f"warped_{m}.png").exists():
            print(f"  … 余白 {m} では顔を検出できず", file=sys.stderr)
            continue
        w = json.loads((d / f"scores/warped_{m}.json").read_text())
        px = float(w.get("render_face_width_px", 0))
        print(f"  ・余白 {m}: レンダの顔 {px:.0f}px")
        if best is None or px > best[1]:
            best = (m, px)
    margin = best[0] if best else None
    if margin is None:
        raise SystemExit("どの余白でも整列レンダから顔を検出できなかった。"
                         f"{d / 'align.png'} を見ること")
    print(f"  ✓ 写真の整列（余白 {margin} を採用・顔 {best[1]:.0f}px）")
    blender("project_face.py", ["--input", str(d / "head_ready.glb"),
                                "--photo", str(d / f"warped_{margin}.png"),
                                "--out", str(d / "head_tex.glb"),
                                "--frame", "head", "--frame-margin", margin,
                                "--align-res", "2048", "--bake-size", "2048"],
            "顔テクスチャの投影", L / "08_project.log")

    # --- 4) 体: 寸法を測ってテンプレートを変形 -----------------------------
    import yaml
    tmpl = yaml.safe_load((ROOT / "configs/templates.yaml").read_text())["templates"][args.template]
    body_glb = ROOT / tmpl["mesh"]
    if args.body or args.height_cm:
        cmd = [str(ENV["eval"]), "-m", "preprocessing.body_landmarks",
               "--input", str(args.body or args.face),
               "--out", str(d / "scores/body.json")]
        if args.height_cm:
            cmd += ["--height-cm", str(args.height_cm)]
        run(cmd, "体のランドマーク", L / "09_body_lm.log")
        run([sys.executable, "-m", "preprocessing.body_params",
             "--body", str(d / "scores/body.json"), "--template", args.template,
             "--out", str(d / "scores/pairs_body.json"),
             "--report", str(d / "scores/body_params.json")],
            "体の変形パラメータ", L / "10_body_params.log")
        run([sys.executable, "-m", "preprocessing.shape_transfer",
             "--template", str(body_glb), "--pairs", str(d / "scores/pairs_body.json"),
             "--out", str(d / "body_deformed.glb"), "--meshes"],
            "体の変形", L / "11_body_deform.log")
        body_glb = d / "body_deformed.glb"

    # --- 5) 接合（リグを保つ）---------------------------------------------
    blender("merge_head.py", ["--body", str(body_glb), "--head", str(d / "head_tex.glb"),
                              "--out", str(d / "merged.glb"), "--keep-rig",
                              "--report", str(d / "scores/merge.json")],
            "頭と体の接合", L / "12_merge.log")

    # --- 6) 表情（ICT の53個を ARKit 命名で）-------------------------------
    blender("ortho_front.py", ["--input", str(d / "merged.glb"),
                               "--out", str(d / "merged_front.png"),
                               "--meta", str(d / "scores/merged_front.json"),
                               "--res", "1024", "--frame-material", "Material_0"],
            "正面レンダ", L / "13_front.log")
    run([str(ENV["face"]), "-m", "preprocessing.ict_fit",
         "--target", str(d / "merged.glb"), "--render", str(d / "merged_front.png"),
         "--render-meta", str(d / "scores/merged_front.json"),
         "--out", str(d / "scores/ict_fit.json"),
         "--out-mesh", str(d / "ict_fitted.obj")],
        "ICT の当て込み", L / "14_ictfit.log")
    blender("transfer_expr.py", ["--target", str(d / "merged.glb"),
                                 "--ict-neutral", str(d / "ict_fitted.obj"),
                                 "--fit", str(d / "scores/ict_fit.json"),
                                 "--out", str(d / "output/avatar.glb"),
                                 "--replace-keys",
                                 "--report", str(d / "scores/expr.json")],
            "表情の転写", L / "15_expr.log")

    # --- 7) 検査 -----------------------------------------------------------
    blender("pose_check.py", ["--input", str(d / "output/avatar.glb"),
                              "--outdir", str(d / "renders/pose"),
                              "--poses", "arms,lowarm,legs,neck,spine,fingers",
                              "--res", "500", "--report", str(d / "scores/pose.json")],
            "ポーズ検査", L / "16_pose.log")
    run([sys.executable, str(ROOT / "scripts/check_kehai_model.py"),
         str(d / "output/avatar.glb")], "KEHAI 検査（名前）", L / "17_kehai.log")
    # **名前だけでは不足。** 実際に動くかを見る（§10.14 でまばたきが 0.0mm だった）
    blender("morph_check.py", ["--input", str(d / "output/avatar.glb"),
                               "--report", str(d / "scores/morph.json")],
            "モーフ検査（変位）", L / "18_morph.log")

    pose = json.loads((d / "scores/pose.json").read_text())
    fingers = next((p for p in pose["poses"] if p["pose"] == "fingers"), None)
    summary = {
        "face_photo": str(args.face), "body_photo": str(args.body),
        "template": args.template, "height_cm": args.height_cm,
        "seconds": round(time.time() - t0, 1),
        "output": str(d / "output/avatar.glb"),
        "pose_all_deformed": pose["all_deformed"],
        "finger_displacement_mm": round(fingers["max_delta"] * 1000, 1) if fingers else None,
    }
    (d / "output/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n完了 {summary['seconds']}s  → {summary['output']}")
    print(f"ポーズ検査 {'合格' if summary['pose_all_deformed'] else '不合格'}  "
          f"指の変位 {summary['finger_displacement_mm']}mm")
    print("同一性は別途測ること: 由来写真ではなく**別セッションの実写**を基準に")


if __name__ == "__main__":
    main()
