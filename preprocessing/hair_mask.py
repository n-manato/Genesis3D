#!/usr/bin/env python3
"""写真の**髪の領域**を MediaPipe の多クラス人物セグメンテーションで取る。

`hair_color` は写真から髪色の標本を採るが、それまでの幾何のヒューリスティック
（眉より上・顔の左右の帯・肌色との距離）は写真によって破綻した:

  - 背景が単色でない街路の写真 … 木と空が帯に入り、暗い茶髪が L 58 の金髪になった
  - 短髪で額の広い男性 … 強く照らされた額と青い上着が入り、髪が L 64 になった

クラス分けをする器に任せる。selfie_multiclass_256x256 は
0 背景 / 1 髪 / 2 体の肌 / 3 顔の肌 / 4 服 / 5 その他 を返す。

環境は genesis-eval（mediapipe）。genesis-face の hair_color とは別プロセス。

使い方:
    python3 -m preprocessing.hair_mask --input jobs/x/head_rgba.png \\
        --out jobs/x/scores/photo_hair_mask.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODEL = Path("vendor/mediapipe/selfie_multiclass_256x256.tflite")
HAIR = 1


def segment(path: Path, model: Path) -> np.ndarray:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if not model.exists():
        raise SystemExit(
            f"{model} が無い。以下で取得できる:\n"
            f"  curl -sL -o {model} https://storage.googleapis.com/mediapipe-models/"
            f"image_segmenter/selfie_multiclass_256x256/float32/latest/"
            f"selfie_multiclass_256x256.tflite")
    opts = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True, output_confidence_masks=False)
    from PIL import Image
    rgb = np.asarray(Image.open(path).convert("RGB"))
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = seg.segment(img)
    cat = res.category_mask.numpy_view().copy()
    return cat[..., 0] if cat.ndim == 3 else cat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="髪=255 の PNG")
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    from PIL import Image
    cat = segment(args.input, args.model)
    im = Image.open(args.input)
    hair = cat == HAIR
    if im.mode in ("RGBA", "LA"):
        hair &= np.asarray(im.convert("RGBA"))[:, :, 3] > 128
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((hair * 255).astype(np.uint8)).save(args.out)
    counts = {int(k): int(v) for k, v in zip(*np.unique(cat, return_counts=True))}
    rep = {"input": str(args.input), "out": str(args.out), "hair_pixels": int(hair.sum()),
           "hair_fraction": round(float(hair.mean()), 4), "class_pixels": counts}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"HAIR_MASK_OK {json.dumps(rep, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
