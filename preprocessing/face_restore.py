"""実写ポートレートの顔だけを拡大し、生成モデルで質感を足す。

**これは情報を増やす処理ではない。作る処理である。**

素材の天井を実測してある: `subject_a/` の全16枚で最大の顔は
`subject_face_downtown_1200.jpg` の 205×281px（目間99px）。所属機関の公式サイトの公開解像度は
1200×800 が上限で、1600/1920/2400/originals はいずれも 404（docs/05 §39）。
テクスチャの情報量はここで頭打ちで、ベイク解像度を上げても増えない（§35）。

そこで拡大＋質感付与を行うが、**実在の人物の顔でこれをやると別人になりうる**。
NTIRE 2026 の実世界顔復元チャレンジは identity を合否判定にしており、
10チーム中9チームしか通らなかった。既知の失敗として扱う。

したがってこのモジュールは**必ず同一性を測って返す**。呼び出し側は
`meta["arcface_to_source"]` を見て、落ちていたら採用しないこと。

顔専用の復元モデル（GFPGAN / CodeFormer）ではなく SDXL の img2img を使うのは:
  - 追加ダウンロードが要らない（RealVisXL V5.0 は既に 6.5GB 導入済み）
  - denoise 強度がそのまま忠実度のつまみになる
  - 顔復元 GAN は「目の形が不自然になる」「identity が突然変わる」と報告されている

  python -m preprocessing.face_restore --input <portrait.jpg> --out <restored.png> \
      --strength 0.25 --meta <meta.json>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")

MODEL_ID = "SG161222/RealVisXL_V5.0"
PACK = "antelopev2"
# SDXL の素の解像度。顔をこの大きさまで拡大してから通す。
NATIVE = 1024

# 被写体を言葉で固定する。曖昧なプロンプトほどモデルが「一般的な美しい顔」へ
# 寄せてしまい、同一性が落ちる。年齢と民族を明示するのはそのため。
PROMPT = ("a candid photograph of an older East Asian woman, natural aged skin "
          "with fine wrinkles and pores, realistic skin texture, sharp focus, "
          "natural daylight, unretouched")
NEGATIVE = ("smooth airbrushed skin, plastic skin, beautified, younger, "
            "heavy makeup, cgi, 3d render, illustration, distorted eyes, "
            "asymmetric face, extra teeth")


def detect(image, app):
    """最大の顔の bbox と5点ランドマークを返す。"""
    import numpy as np
    faces = app.get(np.asarray(image.convert("RGB"))[:, :, ::-1])
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def face_box(image, face, margin: float = 0.45):
    """顔の周りに余白をつけた正方形の切り出し範囲。

    余白を取るのは、生成モデルに輪郭と髪の生え際まで見せるため。顔だけを
    切ると、境界で肌の階調が合わずに貼り戻したときに継ぎ目が出る。
    """
    x0, y0, x1, y1 = face.bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (0.5 + margin)
    w, h = image.size
    bx0 = int(max(0, cx - half)); by0 = int(max(0, cy - half))
    bx1 = int(min(w, cx + half)); by1 = int(min(h, cy + half))
    return bx0, by0, bx1, by1


def arcface(app, a, b) -> float | None:
    """2枚の画像の顔がどれだけ同じ人か。1.0 が同一。"""
    import numpy as np

    def emb(im):
        f = detect(im, app)
        return None if f is None else f.normed_embedding
    ea, eb = emb(a), emb(b)
    if ea is None or eb is None:
        return None
    return float(np.dot(ea, eb))


def restore(image, face, strength: float, steps: int, guidance: float,
            seed: int, offload: bool):
    """顔まわりを切り出して SDXL の img2img に通し、元画像へ貼り戻す。"""
    import torch
    from PIL import Image
    from diffusers import StableDiffusionXLImg2ImgPipeline

    box = face_box(image, face)
    crop = image.convert("RGB").crop(box)
    src_px = crop.size[0]
    crop_big = crop.resize((NATIVE, NATIVE), Image.LANCZOS)

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    # VRAM は GPU0 を他サービスが常時 9GB 使っている（docs/05 §19）。
    # offload 前提でリースを申告しているので、既定で有効にする。
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    out = pipe(prompt=PROMPT, negative_prompt=NEGATIVE, image=crop_big,
               strength=strength, num_inference_steps=steps,
               guidance_scale=guidance,
               generator=torch.Generator("cuda").manual_seed(seed)).images[0]

    # 貼り戻しは元の大きさではなく **拡大したまま** にする。元画像全体を
    # 同じ倍率で引き伸ばし、顔の位置に高精細な結果を置く。
    scale = NATIVE / max(1, src_px)
    W, H = image.size
    base = image.convert("RGB").resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    base.paste(out, (int(box[0] * scale), int(box[1] * scale)))
    return base, {"crop_box": list(box), "source_face_px": src_px,
                  "upscale": round(scale, 3), "output_size": list(base.size)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta")
    # 既定値は実測で決めた（docs/05-measurements.md §40）。強度と guidance を
    # 振って同一性を測ったところ、guidance 5.0 では最弱の 0.15 でも 0.612 まで落ち、
    # 目視でも別人だった。guidance 2.0 まで下げると 0.05 で 0.887 を保つ。
    ap.add_argument("--strength", type=float, default=0.05,
                    help="img2img の denoise 強度。忠実度のつまみ。上げるほど作る")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=2.0,
                    help="上げるとプロンプトの「理想の顔」に引っ張られて別人になる")
    # 同一人物を別セッションで撮った実写どうしが 0.806〜0.857。復元結果がそれを
    # 下回るなら、本人から遠ざかっているということなので採用しない。
    ap.add_argument("--min-identity", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-offload", action="store_true")
    args = ap.parse_args()

    from PIL import Image
    from insightface.app import FaceAnalysis

    src = Image.open(args.input).convert("RGB")
    app = FaceAnalysis(name=PACK, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(1024, 1024))

    face = detect(src, app)
    if face is None:
        raise SystemExit(f"顔を検出できません: {args.input}")

    out, meta = restore(src, face, args.strength, args.steps, args.guidance,
                        args.seed, not args.no_offload)

    # **同一性の測定と合否判定。** 落ちていたら採用しない。
    ident = arcface(app, src, out)
    meta["arcface_to_source"] = ident
    meta["min_identity"] = args.min_identity
    meta["accepted"] = bool(ident is not None and ident >= args.min_identity)
    if not meta["accepted"]:
        meta["reason"] = ("同一性が閾値を下回った。復元後の顔は本人から遠い。"
                          "元画像をそのまま使うこと")
    meta.update({"input": args.input, "output": args.out,
                 "strength": args.strength, "steps": args.steps,
                 "guidance": args.guidance, "seed": args.seed})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    if args.meta:
        Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
        Path(args.meta).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("FACERESTORE_OK " + json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
