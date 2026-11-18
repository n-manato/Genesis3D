#!/usr/bin/env python3
"""写真 → ゲームアバター。**生成メッシュは測るために回し、出荷はテンプレートのトポロジ。**

## 接合方式（scripts/gen_game_avatar.py）との違い

接合方式は「生成した頭」を出荷物にして、テンプレートの体に繋いでいた。その実測
（`jobs/ga_v2_005/scores/merge.json`）:

    applied_scale : 0.31479     ← head-width 基準。根拠なく採用されていた
    scale_by_span : 0.45527     ← 44.6% 違う
    merged height : 1.52286     ← 指定 1.60m。7.7cm 足りない

**頭が正しい寸法の 69% に縮んで胴に付いていた。** 首が長く見えた原因はこれ。
2つのランドマーク比のどちらを信じるかを決める根拠が無く、頭と体が別人由来に
なった時点でどちらも正当化できない。

ラップは逆をやる。テンプレートの頭を生成された頭の**表面に吸着**させ、出荷するのは
テンプレートのトポロジにする。位置合わせは2つの比ではなく数千点の対応から解く。

| | 接合（v2） | **ラップ（これ）** |
|---|---|---|
| 頂点数 | 129,458 | **5,569** |
| 身長 | 1.5229m（指定と 7.7cm ずれ） | **1.7433m（テンプレどおり）** |
| モーフ | 53（転写・壊れやすい） | **175（テンプレ本来のもの）** |
| held-out ArcFace バスト | 0.6189 | 0.6158 |
| held-out ArcFace **全身** | 0.5329 | **0.6098** |
| 削減 | 682,766→179,527（斑点の原因） | **無し** |
| 接合 | 有り | **無し** |

全身が主用途（モーション）なので、全身で +0.077 なのが効く。

## 工程

    写真 → 頭部切り出し → 背景除去 → TRELLIS（測定用・出荷しない）→ 修復
         → 正面レンダ → **ラップ（非剛体ICP）** → テンプレートを変形
         → 体型を寄せる → 写真を投影 → 検査

**削減・接合・ICT当て込み・表情転写の4工程が消える。**

使い方:
    python3 scripts/gen_wrap_avatar.py \\
        --face subject_a/subject-a_head_1024.png \\
        --body subject_a/subject_tpose_front.png --height-cm 160 \\
        --template rocketbox_female_01 --out-dir jobs/gw_001

**顔写真は加工していないものを選ぶこと。** SDXL で復元した画像は解像度が高くても
held-out 0.642、素の写真が 0.773 だった（docs/08 §10.12）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = {k: Path.home() / "miniconda3/envs" / v / "bin/python"
       for k, v in {"eval": "genesis-eval", "face": "genesis-face",
                    "trellis": "trellis2"}.items()}
BLENDER = os.environ.get("BLENDER_BIN", "/home/ai-ccore/opt/blender/blender")
MARGINS = ("1.35", "1.6", "2.0", "2.6")


def run(cmd, what, log, check=True):
    t = time.time()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=ROOT)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    dt = time.time() - t
    if r.returncode != 0:
        if not check:
            return None
        print(f"  x {what}（{dt:.1f}s） — {log}", file=sys.stderr)
        print(r.stderr[-1500:], file=sys.stderr)
        raise SystemExit(1)
    print(f"  o {what}（{dt:.1f}s）")
    return r.stdout


def blender(script, args, what, log, check=True):
    return run([BLENDER, "--background", "--factory-startup", "--python",
                ROOT / "blender" / script, "--"] + list(args), what, log, check)


def best_margin(d, L, model, frame_material, stage, photo=None):
    """余白を振って**顔が最大に写る**ものを選ぶ。

    顔が画枠を占めすぎると SCRFD が検出できず、小さすぎると解像度が落ちる。
    頭の形は個体ごとに違うので、全部試して最良を採る（1枚2秒）。
    「最初に通ったもの」だと広い余白が選ばれて顔の画素が減る（docs/08 §10.12）。
    """
    best = None
    for m in MARGINS:
        png = d / f"{stage}_{m}.png"
        meta = d / f"scores/{stage}_{m}.json"
        if photo is None:      # ortho_front：ラップの入力にする
            ok = blender("ortho_front.py",
                         ["--input", model, "--out", png, "--meta", meta,
                          "--res", "1024", "--margin", m],
                         f"正面レンダ 余白{m}", L / f"{stage}_{m}.log", check=False)
            if ok is None:
                continue
            r = run([ENV["face"], "-c",
                     "import sys,json;from PIL import Image;"
                     "sys.path.insert(0,'.');"
                     "from preprocessing.ict_fit import detect_68;"
                     "uv=detect_68(Image.open(sys.argv[1]));"
                     "print('PX '+str(float(uv[:,0].max()-uv[:,0].min())))",
                     png], f"68点検出 余白{m}", L / f"{stage}_det_{m}.log", check=False)
            px = float(r.split("PX ")[1].split()[0]) if r and "PX " in r else 0.0
        else:                  # project_face：写真を整列させる
            args = ["--input", model, "--emit-align-render", png,
                    "--frame", "head", "--frame-margin", m, "--align-res", "2048"]
            if frame_material:
                args += ["--frame-material", frame_material]
            blender("project_face.py", args, f"整列レンダ 余白{m}",
                    L / f"{stage}_{m}.log", check=False)
            if not png.exists():
                continue
            run([ENV["eval"], "-m", "preprocessing.face_projection",
                 "--photo", photo, "--render", png,
                 "--out", d / f"warped_{m}.png", "--meta", meta],
                f"写真の整列 余白{m}", L / f"{stage}_warp_{m}.log", check=False)
            if not meta.exists():
                continue
            px = float(json.loads(meta.read_text()).get("render_face_width_px", 0))
        print(f"    余白 {m}: 顔 {px:.0f}px")
        if px > 0 and (best is None or px > best[1]):
            best = (m, px)
    if best is None:
        raise SystemExit(f"{stage}: どの余白でも顔を検出できなかった")
    print(f"  o 余白 {best[0]} を採用（顔 {best[1]:.0f}px）")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--face", type=Path, required=True)
    ap.add_argument("--body", type=Path, default=None)
    ap.add_argument("--height-cm", type=float, default=None)
    ap.add_argument("--template", default="rocketbox_female_01")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resolution", default="1024", choices=("512", "1024"))
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--refs", type=Path, nargs="*",
                    default=[Path("subject_a/subject_face_tweed_head.png")],
                    help="同一性の基準。**テクスチャの由来と別セッション**の実写")
    ap.add_argument("--no-body-projection", dest="body_projection",
                    action="store_false",
                    help="服の構造（襟・前開き・インナー）を胴に投影しない。"
                         "既定は有効（docs/09 §18.3）")
    # **肩の鎖は骨の移動から除外する。** 骨をメッシュ断面の中心へ動かすと肘と手首は
    # 改善する（手首の剛体歪み 42.5→10.9%）が、肩は KEHAI の待機姿勢で内側に潰れた。
    # Rocketbox のウェイトは腕の支点が肩の頂部にある前提で塗られており、支点を
    # 11cm 下げると肩の頂部ごと腕と一緒に回る。4方式を並べて確認（docs/09 §29）:
    #   v3 全部据え置き … 肩は正常・肘手首は破綻
    #   v7 全部移動     … 肘手首は改善・肩が潰れる
    #   v8a 肩だけ据え置き … **肩は正常・肘手首は改善**（剛体歪み v7 と同一）← 既定
    #   v8b 全部移動     … v7 と同じく潰れる（再現）
    ap.add_argument("--joints-exclude",
                    default="LeftShoulder,RightShoulder,LeftArm,RightArm",
                    help="骨の移動から除外する骨（カンマ区切り）。既定は肩の鎖")
    ap.add_argument("--no-vrm", action="store_true",
                    help="VRM を書き出さない。**glb だけでは髪が揺れない**")
    ap.add_argument("--vrm-name", default="Genesis3D Character")
    ap.add_argument("--no-hair", action="store_true",
                    help="髪のジオメトリを載せない（比較用）")
    ap.add_argument("--hair-outside", type=float, default=0.008)
    ap.add_argument("--hair-outside-front", type=float, default=0.030)
    ap.add_argument("--hair-target", type=int, default=20000)
    ap.add_argument("--facing-min", type=float, default=0.01,
                    help="投影の facing の下限。既定 0.15 では頭頂に届かない")
    ap.add_argument("--no-textures", action="store_true",
                    help="髪と服の色替えを行わない（比較用）")
    ap.add_argument("--hair-geom-material", default="Material_0",
                    help="生成メッシュから切り出した髪のマテリアル名（TRELLIS 由来）")
    ap.add_argument("--hair-material", default="opacity",
                    help="髪カードとまつ毛のマテリアル")
    ap.add_argument("--skip-trellis", type=Path, default=None,
                    help="生成済みの頭 glb を使い回す（実験用）")
    args = ap.parse_args()

    import yaml
    tmpl = yaml.safe_load((ROOT / "configs/templates.yaml").read_text())["templates"][args.template]
    d = args.out_dir
    for sub in ("logs", "scores", "output", "renders"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    L = d / "logs"
    t0 = time.time()
    face_mat = tmpl.get("face_material", "head")
    print(f"顔 {args.face}  体 {args.body}  テンプレ {args.template}（{face_mat}）")

    # --- 1) 頭の生成。**出荷しない。ラップの測定対象にするだけ** -----------
    # 髪色の標本は**背景を抜いた頭の切り抜き**から採る。生の写真だと背景が単色で
    # ない限り（街路の木や空が）「眉より上・顔の左右」の帯に入り、髪色が背景色に
    # なる（別人・女性で L 58 の金髪になった。本人は暗い茶髪）。rmbg の RGBA なら
    # alpha で背景が落ちる。生成済みの頭を使い回すときは、その頭を作った
    # ジョブの切り抜きを探す。
    hair_photo = args.face
    if args.skip_trellis:
        head = args.skip_trellis
        cand = Path(args.skip_trellis).parent / "head_rgba.png"
        if cand.exists():
            hair_photo = cand
        print(f"  - 生成を飛ばして {head} を使う")
    else:
        run([ENV["eval"], "-m", "preprocessing.head_crop", "--input", args.face,
             "--out", d / "head_crop.png", "--meta", d / "scores/crop.json"],
            "頭部の切り出し", L / "01_crop.log")
        run([ENV["eval"], "-m", "preprocessing.rmbg", "--input", d / "head_crop.png",
             "--output", d / "head_rgba.png", "--meta", d / "scores/rmbg.json"],
            "背景除去", L / "02_rmbg.log")
        hair_photo = d / "head_rgba.png"
        run([ENV["trellis"], ROOT / "adapters/trellis/oneshot.py",
             "--image", d / "head_rgba.png", "--out", d / "head_raw.glb",
             "--resolution", args.resolution, "--gpu", args.gpu,
             "--report", d / "scores/trellis.json"],
            f"頭の生成（TRELLIS {args.resolution}・GPU{args.gpu}）", L / "03_trellis.log")
        # **削減しない。** 出荷しないので間引く理由が無い。削減は顔の黒い斑点の
        # 原因だった（COLLAPSE が髪のような細い構造で退化面と法線反転を作る）。
        blender("repair_mesh.py", ["--input", d / "head_raw.glb",
                                   "--out", d / "head_clean.glb",
                                   "--report", d / "scores/repair.json"],
                "メッシュ修復", L / "04_repair.log")
        head = d / "head_clean.glb"

    # --- 2) ラップ: テンプレートの頭を対象の表面へ吸着 ---------------------
    m, _ = best_margin(d, L, head, None, "hf")
    run([ENV["face"], "-m", "preprocessing.wrap_head",
         "--template", ROOT / tmpl["mesh"],
         "--landmarks", ROOT / tmpl["landmarks68"],
         "--anchors", ROOT / tmpl["anchors"],
         "--target", head,
         "--render", d / f"hf_{m}.png", "--render-meta", d / f"scores/hf_{m}.json",
         "--face-material", face_mat,
         "--out", d / "scores/pairs_head.json", "--report", d / "scores/wrap.json"],
        "ラップ（非剛体ICP）", L / "05_wrap.log")
    run([ENV["face"], "-m", "preprocessing.shape_transfer",
         "--template", ROOT / tmpl["mesh"], "--pairs", d / "scores/pairs_head.json",
         "--out", d / "wrapped.glb", "--report", d / "scores/wrapped.json", "--meshes"],
        "テンプレートを変形", L / "06_transfer.log")

    # --- 2b) 髪を生成メッシュから切り出して載せる --------------------------
    #
    # ラップは生成メッシュを「測るためだけ」に回して捨てていたが、**本人の髪は
    # そこにジオメトリとして入っている**。テンプレート（Rocketbox）は髪の
    # ボリュームを持たず色しか寄せられなかった（docs/09 §13.2）。
    #
    # **髪は固定トポロジを必要としない。** ラップが要るのは表情モーフを持つ顔と
    # 手塗りウェイトを持つ体だけで、髪に要るのは骨1本。だから髪だけ生成のまま
    # 載せてよい。結合はしない（シェイプキー175個を壊しうる）。
    model = d / "wrapped.glb"
    if not args.no_hair:
        blender("add_hair.py",
                ["--model", model, "--generated", head,
                 "--wrap-report", d / "scores/wrap.json",
                 "--out", d / "haired.glb", "--face-material", face_mat,
                 "--outside", str(args.hair_outside),
                 "--outside-front", str(args.hair_outside_front),
                 "--target", str(args.hair_target),
                 "--landmarks", ROOT / tmpl["landmarks68"],
                 "--report", d / "scores/hair_geom.json"],
                "髪のジオメトリを切り出して載せる", L / "06b_hair.log")
        model = d / "haired.glb"

    # --- 3) 体型を寄せる ---------------------------------------------------
    if args.body or args.height_cm:
        cmd = [ENV["eval"], "-m", "preprocessing.body_landmarks",
               "--input", args.body or args.face, "--out", d / "scores/body.json"]
        if args.height_cm:
            cmd += ["--height-cm", str(args.height_cm)]
        run(cmd, "体のランドマーク", L / "07_body_lm.log")
        run([sys.executable, "-m", "preprocessing.body_params",
             "--body", d / "scores/body.json", "--template", args.template,
             "--out", d / "scores/pairs_body.json",
             "--report", d / "scores/body_params.json"],
            "体の変形パラメータ", L / "08_body_params.log")
        # **`--joints` で骨も動かす。** メッシュだけ変形すると骨が取り残され、
        # 肘と手首が壊れる（docs/09 §21）。同乗している FBX 由来のアニメーション
        # も同時に落ちる —— それが残っていると Blender がインポート時に1フレーム目を
        # 適用してノードを古い値に上書きし、評価後のメッシュが 1.60m → 1.72m に
        # 伸びる（docs/09 §26）。
        run([ENV["face"], "-m", "preprocessing.shape_transfer",
             "--template", model, "--pairs", d / "scores/pairs_body.json",
             "--out", d / "body_deformed.glb", "--joints",
             "--joints-exclude", args.joints_exclude,
             "--report", d / "scores/body_deformed.json", "--meshes"],
            "体型を寄せる（骨も）", L / "09_body_deform.log")
        model = d / "body_deformed.glb"

    # --- 4) 写真を投影。**最終形状に対して行う**（UVはテンプレのまま）------
    m, px = best_margin(d, L, model, face_mat, "al", photo=args.face)
    blender("project_face.py",
            ["--input", model, "--photo", d / f"warped_{m}.png",
             "--out", d / "output/avatar.glb", "--frame", "head",
             "--frame-material", face_mat, "--face-material", face_mat,
             "--frame-margin", m, "--align-res", "2048", "--bake-size", "2048",
             # **facing の下限を下げる。** 既定 0.15 では法線が上向きの頭頂に
             # 写真が乗らない。0.01 まで下げると held-out ArcFace が
             # バスト +0.008 / 全身 +0.007（docs/09 §13.2）。斜めから当たる分の
             # 引き伸ばしは、顔ではなく髪と頭皮なので目立たない。
             "--facing-min", str(args.facing_min),
             "--report", d / "scores/project.json"],
            "顔テクスチャの投影", L / "10_project.log")
    out = d / "output/avatar.glb"

    # --- 5) テクスチャ: 髪と服を写真に寄せる -------------------------------
    #
    # **どちらもメッシュには触らない。** 画像1枚の差し替えだけなので、リグ・
    # モーフ・UV は一度も動かない。Rocketbox は髪も服もテクスチャに描いてある
    # （髪のボリュームを持つジオメトリが無い。診断でマテリアルを色分けすると
    # 頭は禿げている）ので、色を寄せるのが2Dの問題として解ける（docs/09 §13）。
    if not args.no_textures:
        proj = out
        blend_tmp = d / "scores/blend_head.png"
        # 写真の髪の領域はセグメンテーションで取る（genesis-eval / MediaPipe）。
        # 幾何のヒューリスティックは背景が単色で額の狭い写真でしか当たらなかった
        # （docs/09 §28.8）。失敗したら hair_color がヒューリスティックに落ちる。
        photo_mask = d / "scores/photo_hair_mask.png"
        try:
            run([ENV["eval"], "-m", "preprocessing.hair_mask", "--input", hair_photo,
                 "--out", photo_mask, "--report", d / "scores/photo_hair_mask.json"],
                "写真の髪のセグメンテーション", L / "10a_hair_mask.log")
            photo_mask_args = ["--photo-mask", photo_mask]
        except SystemExit:
            print("  注意: 髪のセグメンテーションに失敗。ヒューリスティックで続ける")
            photo_mask_args = []
        run([ENV["face"], "-m", "texture_processing.face_texture_blend",
             "--model", model, "--projected", proj, "--material", face_mat,
             "--out", blend_tmp,
             "--scalp-mask-out", d / "scores/mask_scalp.png",
             "--hair-card-mask-out", d / "scores/mask_haircard.png",
             "--landmarks", ROOT / tmpl["landmarks68"],
             "--hair-material", args.hair_material,
             "--report", d / "scores/masks.json"],
            "髪のマスク（頭皮とカードを分ける）", L / "10b_masks.log")

        # 頭皮（head テクスチャの、投影が届かなかった領域）
        run([ENV["face"], "-m", "vrm.extract_texture", "--input", proj,
             "--material", face_mat, "--out", d / "tex_head.png"],
            "head テクスチャの抽出", L / "10c_ex_head.log")
        run([ENV["face"], "-m", "texture_processing.hair_color",
             "--photo", hair_photo, *photo_mask_args, "--texture", d / "tex_head.png",
             "--mask", d / "scores/mask_scalp.png",
             "--out", d / "tex_head_hair.png",
             "--report", d / "scores/hair_scalp.json"],
            "頭皮に髪色を移す", L / "10d_hair_scalp.log")
        run([sys.executable, "-m", "vrm.inject_texture", "--input", proj,
             "--out", d / "t1.glb", "--material", face_mat,
             "--png", d / "tex_head_hair.png"],
            "頭皮テクスチャの注入", L / "10e_inj_head.log")

        # 髪カード（opacity）。**アルファを保つこと。**
        run([ENV["face"], "-m", "vrm.extract_texture", "--input", d / "t1.glb",
             "--material", args.hair_material, "--out", d / "tex_hair.png"],
            "髪カードの抽出", L / "10f_ex_hair.log")
        # **テンプレの髪カード（ポニーテール）は消す。** 生成メッシュから切り出した
        # 髪を載せるので冗長で、残すと生成髪の隙間から「元のモデルの髪」が見える。
        # 色替えではなくアルファをゼロにする（まつ毛は残す）。
        run([ENV["face"], "-m", "texture_processing.hair_color",
             "--photo", hair_photo, *photo_mask_args, "--texture", d / "tex_hair.png",
             "--mask", d / "scores/mask_haircard.png", "--strength", "0",
             "--clear-mask", d / "scores/mask_haircard.png",
             "--out", d / "tex_hair_new.png",
             "--report", d / "scores/hair_cards.json"],
            "テンプレの髪カードを消す", L / "10g_hair_cards.log")
        run([sys.executable, "-m", "vrm.inject_texture", "--input", d / "t1.glb",
             "--out", d / "t2.glb", "--material", args.hair_material,
             "--png", d / "tex_hair_new.png"],
            "髪カードの注入", L / "10h_inj_hair.log")
        cur = d / "t2.glb"

        # 生成した髪のジオメトリ（Material_0）。**TRELLIS のテクスチャは髪を
        # 真っ黒に焼く。** 髪メッシュが参照する画素は Lab の L が平均 5.5（写真の
        # 髪は 24.3）で、そのまま出すと陰影の無い黒い切り絵になる（docs/09 §28.7）。
        # 髪メッシュが使う UV の範囲だけを写真の髪の統計に合わせる。
        # `--min-luma 0`: 既定の 6 だと真っ黒の画素（中央値 4）が対象から外れる。
        if not args.no_hair:
            run([ENV["face"], "-m", "vrm.extract_texture", "--input", cur,
                 "--material", args.hair_geom_material, "--out", d / "tex_hairgeom.png"],
                "髪ジオメトリのテクスチャ抽出", L / "10h2_ex_hairgeom.log")
            run([ENV["face"], "-m", "texture_processing.uv_mask", "--model", cur,
                 "--material", args.hair_geom_material,
                 "--out", d / "scores/mask_hairgeom.png",
                 "--report", d / "scores/mask_hairgeom.json"],
                "髪メッシュが使う UV の範囲", L / "10h3_uv_mask.log")
            run([ENV["face"], "-m", "texture_processing.hair_color",
                 "--photo", hair_photo, *photo_mask_args, "--texture", d / "tex_hairgeom.png",
                 "--mask", d / "scores/mask_hairgeom.png", "--min-luma", "0",
                 "--outlier-sigma", "2.5",
                 "--out", d / "tex_hairgeom_new.png",
                 "--report", d / "scores/hair_geom_color.json"],
                "髪ジオメトリの色を写真に寄せる", L / "10h4_hairgeom_color.log")
            run([sys.executable, "-m", "vrm.inject_texture", "--input", cur,
                 "--out", d / "t2b.glb", "--material", args.hair_geom_material,
                 "--png", d / "tex_hairgeom_new.png"],
                "髪ジオメトリのテクスチャ注入", L / "10h5_inj_hairgeom.log")
            cur = d / "t2b.glb"

        # 服（body テクスチャ）
        if args.body:
            run([ENV["face"], "-m", "vrm.extract_texture", "--input", cur,
                 "--material", tmpl.get("body_material", "body"),
                 "--out", d / "tex_body.png"],
                "body テクスチャの抽出", L / "10i_ex_body.log")
            run([ENV["face"], "-m", "texture_processing.garment_color",
                 "--model", model, "--photo", args.body,
                 "--body-landmarks", d / "scores/body.json",
                 "--texture", d / "tex_body.png",
                 "--body-material", tmpl.get("body_material", "body"),
                 "--out", d / "tex_body_new.png",
                 "--mask-out", d / "scores/mask_garment.png",
                 "--report", d / "scores/garment.json"],
                "服の色を写真に寄せる", L / "10j_garment.log")
            run([sys.executable, "-m", "vrm.inject_texture", "--input", cur,
                 "--out", d / "t3.glb",
                 "--material", tmpl.get("body_material", "body"),
                 "--png", d / "tex_body_new.png"],
                "body テクスチャの注入", L / "10k_inj_body.log")
            cur = d / "t3.glb"

        # **服の構造は胴だけ投影する。**
        #
        # 全身を投影すると部位の混合が胴の内容を脚へ滲ませ、赤い襷が脚まで
        # 伸びた（docs/09 §18.2）。服の構造（襟・前開き・インナー）は胴にしか
        # 無く、袖と裾は単色なので色転写で足りる。胴に限ると
        # インナーの白・両側の赤・ボタンまで乗り、同一性はほぼ中立
        # （バスト 0.6814→0.6813 / 全身 0.6544→0.6474）。
        if args.body_projection and args.body:
            # **服の構造を写真から投影する。** 色転写では色しか移せない ——
            # ブレザーの襟・前開き・中のインナーはテンプレートに無い形なので、
            # 顔と同じように写真をテクスチャに焼く（docs/09 §18）。
            #
            # 生成メッシュから服を切り出す方も試したが、位置合わせで破綻した
            # （関節対応の残差 116.6mm、外接箱の軸ごと合わせは奥行きを 1.72倍に
            # 伸ばして形を壊した）。シルエットは諦めて絵で入れる。
            run([ENV["eval"], "-m", "preprocessing.rmbg", "--input", args.body,
                 "--output", d / "body_rgba.png",
                 "--meta", d / "scores/body_rmbg.json"],
                "全身写真の背景除去", L / "10l_body_rmbg.log")
            blender("project_face.py",
                    ["--input", cur, "--emit-align-render", d / "align_body.png",
                     "--frame", "full", "--align-res", "2048"],
                    "全身の整列レンダ", L / "10m_align_body.log")
            for tag, img in (("render", d / "align_body.png"),
                             ("photo", d / "body_rgba.png")):
                run([ENV["eval"], "-m", "preprocessing.body_landmarks",
                     "--input", img, "--out", d / f"scores/lm_{tag}.json"],
                    f"33点（{tag}）", L / f"10n_lm_{tag}.log")
            run([ENV["eval"], "-m", "preprocessing.body_projection",
                 "--photo", d / "body_rgba.png", "--render", d / "align_body.png",
                 "--photo-landmarks", d / "scores/lm_photo.json",
                 "--render-landmarks", d / "scores/lm_render.json",
                 "--out", d / "warped_body.png", "--torso-only",
                 "--meta", d / "scores/warp_body.json"],
                "全身写真をレンダに合わせる", L / "10o_warp_body.log")
            blender("project_face.py",
                    ["--input", cur, "--photo", d / "warped_body.png",
                     "--out", d / "t4.glb", "--frame", "full",
                     "--face-material", tmpl.get("body_material", "body"),
                     "--align-res", "2048", "--bake-size", "2048",
                     "--facing-min", str(args.facing_min),
                     "--report", d / "scores/body_project.json"],
                    "服の構造を投影", L / "10p_body_project.log")
            cur = d / "t4.glb"
        cur.replace(out)

    # --- 5b) VRM で書き出して spring を付ける -------------------------------
    #
    # **プレーンな glb では KEHAI は髪を揺らせない。** KEHAI の
    # `utils/glb-humanoid.ts` が明記している ——「What such a model does NOT get,
    # because the file cannot describe them: spring bones (hair/cloth physics)」。
    # `springBoneManager` が作られるのは `VRMC_vrm` を持つ本物の VRM のときだけ。
    #
    # **VRM アドオンは使わない。** Rocketbox 由来のリグでは humanoid の構造探索が
    # 破裂して何も書き出さない（探索 80,686,221回 / 上限 100万）。素のテンプレートでも
    # 同じで、名前・スケール・Icosphere・髪のボーンのどれも原因ではなかった
    # （docs/09 §17）。どのボーンがどの humanoid ボーンかはこちらが知っているので、
    # JSON を直接書く。
    vrm_out = None
    if not args.no_vrm:
        run([sys.executable, "-m", "vrm.make_vrm", "--input", out,
             "--out", d / "output/avatar.vrm", "--name", args.vrm_name,
             "--author", "Genesis3D", "--report", d / "scores/vrm.json"],
            "VRM 1.0 で書き出す", L / "10l_vrm.log")
        vrm_out = d / "output/avatar.vrm"
        if not args.no_hair:
            run([sys.executable, "-m", "vrm.add_springs", "--input", vrm_out,
                 "--out", vrm_out, "--report", d / "scores/springs.json"],
                "髪に spring bone を付ける", L / "10m_springs.log")
        # **テクスチャを WebP にする。** 焼いたテクスチャは 2048² の PNG で、
        # テンプレート自身は同じ寸法を WebP で持っている。差はほぼ符号化だけ。
        for tgt in (out, vrm_out):
            run([ENV["face"], "-m", "vrm.compress_textures",
                 "--input", tgt, "--out", tgt,
                 "--report", d / f"scores/compress_{tgt.suffix.lstrip('.')}.json"],
                f"テクスチャ圧縮（{tgt.name}）", L / f"10o_compress_{tgt.suffix.lstrip('.')}.log")
        run([sys.executable, ROOT / "scripts/check_kehai_model.py", vrm_out],
            "KEHAI 検査（VRM）", L / "10n_kehai_vrm.log")

    # --- 6) 検査 -----------------------------------------------------------
    blender("pose_check.py", ["--input", out, "--outdir", d / "renders/pose",
                              "--poses", "arms,lowarm,legs,neck,spine,fingers",
                              "--res", "500", "--report", d / "scores/pose.json"],
            "ポーズ検査", L / "11_pose.log")
    run([sys.executable, ROOT / "scripts/check_kehai_model.py", out],
        "KEHAI 検査（名前）", L / "12_kehai.log")
    blender("morph_check.py", ["--input", out, "--report", d / "scores/morph.json"],
            "モーフ検査（変位）", L / "13_morph.log")
    # **これを測らずに完了と言わない。** それが v1 で 0.32 の後退を見逃した原因。
    run([sys.executable, ROOT / "scripts/measure_identity.py", "--model", out,
         "--refs", *args.refs, "--source", args.face,
         "--face-material", face_mat, "--renders", d / "renders",
         "--out", d / "scores/identity.json"],
        "同一性（held-out ArcFace）", L / "14_identity.log")

    # 骨のピボットは工程3（体型を寄せる）で同時に直している。以前は Blender が
    # 誤読するため最後に掛けていたが、原因（同乗アニメーション）が分かったので
    # 途中に戻した。おかげで `pose_check` が**補正後の**ファイルを検査する。
    pose = json.loads((d / "scores/pose.json").read_text())
    kehai = json.loads([l for l in (L / "12_kehai.log").read_text().splitlines()
                        if l.startswith("KEHAI_CHECK ")][0][len("KEHAI_CHECK "):])
    ident = json.loads((d / "scores/identity.json").read_text())
    morph = json.loads((d / "scores/morph.json").read_text())
    fingers = next((p for p in pose["poses"] if p["pose"] == "fingers"), None)
    wrap = json.loads((d / "scores/wrap.json").read_text())
    summary = {
        "method": "wrap（非剛体ICP・接合なし・削減なし）",
        "face_photo": str(args.face), "body_photo": str(args.body),
        "template": args.template, "height_cm": args.height_cm,
        "seconds": round(time.time() - t0, 1), "output": str(out),
        "vertices": kehai["verts"], "morphs": kehai["morphs"],
        "kehai_ok": kehai["ok"], "morphs_ok": morph["ok"],
        "pose_all_deformed": pose["all_deformed"],
        "finger_displacement_mm": round(fingers["max_delta"] * 1000, 1) if fingers else None,
        "textures": not args.no_textures, "hair_geometry": not args.no_hair,
        "joint_fix": bool(args.body),
        "vrm": str(vrm_out) if vrm_out else None,
        "wrap_residual_mm": wrap["iterations"][-1]["mean_dist_mm"],
        "wrap_landmark_rms_mm": wrap["iterations"][-1]["lm_rms_mm"],
        "identity": {k: {"face_px": v["face_px"], "held_out": v["held_out_mean"]}
                     for k, v in ident["renders"].items()},
    }
    (d / "output/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if vrm_out and (d / "scores/springs.json").exists():
        sp = json.loads((d / "scores/springs.json").read_text())
        summary["springs"] = {"count": sp["springs"], "joints": sp["joints"]}
    print(f"\n完了 {summary['seconds']}s  → {out}")
    if vrm_out:
        print(f"  VRM → {vrm_out}"
              + (f"  spring {summary['springs']['count']}本"
                 if "springs" in summary else ""))
    print(f"  頂点 {summary['vertices']:,}  モーフ {summary['morphs']}  "
          f"指 {summary['finger_displacement_mm']}mm  "
          f"ラップ残差 {summary['wrap_residual_mm']}mm")
    for k, v in summary["identity"].items():
        print(f"  同一性 {k:5s}: 顔 {v['face_px']:.0f}px  held-out {v['held_out']:+.4f}")


if __name__ == "__main__":
    main()
