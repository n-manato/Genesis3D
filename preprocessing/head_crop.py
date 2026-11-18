"""実写ポートレートから頭部＋肩を切り出す。

顔と体を別メッシュで作る二トラック構成（docs/05-measurements.md §18）で、
顔トラックの入力を作るために使う。全身画像をそのまま TRELLIS.2 に入れると
顔が 83px しかなく造形が作れないので、顔まわりだけを切って拡大する。

切り出しは ArcFace の顔検出を基準にする。固定の割合で切ると、写真ごとの
構図の差で顎や頭頂が欠ける。

  python -m preprocessing.head_crop --input <portrait.jpg> --out <head.png>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")

PACK = "antelopev2"


def crop(image, app, target_height: int = 1024,
         pad_x: float = 1.15, pad_top: float = 1.05, pad_bottom: float = 1.65):
    """顔 bbox を基準に頭部＋肩を切り、指定の高さへ拡大する。

    余白は顔幅を単位に取る。上は髪、下は肩まで入れる — 首から下がまったく
    無いと、接合時に肩の位置を測れない（blender/mesh_landmarks.py が
    「横幅が跳ねる点」を肩と見なすため）。
    """
    import numpy as np

    arr = np.asarray(image.convert("RGB"))[:, :, ::-1]
    faces = app.get(arr)
    if not faces:
        raise SystemExit("顔を検出できませんでした")
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    x0, y0, x1, y1 = f.bbox
    w = x1 - x0
    box = (max(0, x0 - w * pad_x), max(0, y0 - w * pad_top),
           min(image.width, x1 + w * pad_x), min(image.height, y1 + w * pad_bottom))
    c = image.convert("RGB").crop([int(v) for v in box])

    from PIL import Image

    s = target_height / c.height
    out = c.resize((max(1, int(c.width * s)), target_height), Image.LANCZOS)
    return out, {"face_bbox": [int(v) for v in f.bbox],
                 "face_width_px": int(w),
                 "crop_box": [int(v) for v in box],
                 "output_size": list(out.size),
                 "upscale": round(s, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--height", type=int, default=1024)
    args = ap.parse_args()

    from PIL import Image
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=PACK, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(1024, 1024))

    out, meta = crop(Image.open(args.input), app, args.height)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.save(p)
    meta["input"] = args.input
    meta["output"] = str(p)
    if args.meta:
        Path(args.meta).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("HEADCROP_OK " + json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
