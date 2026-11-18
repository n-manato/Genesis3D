"""工程1: 前処理 — 背景除去 + 正規化。

Genesis3D で最も費用対効果の高い工程。背景を残したまま image-to-3D に渡すと、
背景の模様やエフェクトがそのまま浮遊ジオメトリになる（実測で連結成分が6個に増え、
キャラクター周囲がノイズだらけになった）。

背景除去には briaai/RMBG-2.0 (BiRefNet, 220M) を使う。既存 Nexus サイドカーの
rembg(u2net) も :8000/remove-bg で使えるが、実測で**魔女帽子ごと削り落とした**。
帽子・武器・獣耳のような大きな装飾はキャラクターの同一性そのものなので、
ここは精度を優先して RMBG-2.0 を採る。

独立サブプロセスとして実行される（要件§30/§31）。
  conda run -n genesis-eval python -m preprocessing.rmbg \
      --input in.png --output rgba.png --meta meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_ID = "briaai/RMBG-2.0"
# BiRefNet の学習解像度。ここを変えると精度が落ちる。
INFER_SIZE = (1024, 1024)


def load_model(device: str):
    import torch
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        MODEL_ID, trust_remote_code=True)
    model.eval().to(device)
    # 3090 は TF32 が効く。精度への影響は無視できる範囲でマスクが速くなる。
    torch.set_float32_matmul_precision("high")
    return model


def remove_background(model, image, device: str):
    import torch
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(INFER_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(x)[-1].sigmoid().cpu()
    mask = transforms.ToPILImage()(preds[0].squeeze()).resize(image.size)
    out = image.convert("RGBA")
    out.putalpha(mask)
    return out, mask


def square_pad(image, margin: float = 0.06):
    """被写体を中央に置いた正方形へ整える。

    image-to-3D モデルは正方形・中央配置・一定の余白を前提に学習されている。
    候補間で構図がばらつくと、後段の評価が「モデルの差」ではなく
    「構図の差」を測ってしまうので、ここで必ず揃える。
    """
    from PIL import Image

    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        raise SystemExit("前景が検出できませんでした（マスクが空です）")
    cropped = image.crop(bbox)
    w, h = cropped.size
    side = int(max(w, h) * (1.0 + margin * 2))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(cropped, ((side - w) // 2, (side - h) // 2), cropped)
    return canvas, bbox


def analyze(image, mask) -> dict:
    """後段が判断に使う指標。ここでは加工せず観測だけする。"""
    import numpy as np

    a = np.array(image.getchannel("A"), dtype=np.float32) / 255.0
    coverage = float((a > 0.5).mean())

    # 前景の連結成分数。多いほど「浮遊物が写り込んでいる」入力。
    try:
        import cv2
        binary = (a > 0.5).astype("uint8")
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        areas = sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True)
        main = areas[0] if areas else 0
        # 主要成分の1%未満の島は「ノイズ片」とみなす
        specks = sum(1 for x in areas[1:] if x > main * 0.01)
    except Exception:
        n_labels, specks = -1, -1

    # アルファ境界の柔らかさ。半透明が多いとテクスチャに滲みが出やすい。
    soft = float(((a > 0.05) & (a < 0.95)).mean())

    return {
        "foreground_coverage": round(coverage, 4),
        "foreground_components": int(n_labels - 1) if n_labels > 0 else -1,
        "significant_specks": int(specks),
        "soft_alpha_ratio": round(soft, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--margin", type=float, default=0.06)
    args = ap.parse_args()

    import torch
    from PIL import Image

    device = args.device if torch.cuda.is_available() else "cpu"
    src = Path(args.input)
    image = Image.open(src)
    original_size = image.size

    model = load_model(device)
    rgba, mask = remove_background(model, image, device)
    if args.mask:
        Path(args.mask).parent.mkdir(parents=True, exist_ok=True)
        mask.save(args.mask)

    stats = analyze(rgba, mask)
    squared, bbox = square_pad(rgba, args.margin)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    squared.save(out)

    meta = {
        "input": str(src),
        "output": str(out),
        "model": MODEL_ID,
        "device": device,
        "original_size": list(original_size),
        "subject_bbox": list(bbox),
        "output_size": list(squared.size),
        "margin": args.margin,
        **stats,
    }
    # 明らかに問題のある入力は後段へ警告として渡す（処理は止めない）
    warnings_: list[str] = []
    if stats["foreground_coverage"] < 0.05:
        warnings_.append("前景が画像の5%未満。被写体が小さすぎる可能性")
    if stats["foreground_coverage"] > 0.95:
        warnings_.append("前景が95%超。背景除去が効いていない可能性")
    if stats["significant_specks"] > 2:
        warnings_.append(
            f"主要でない前景の島が {stats['significant_specks']} 個。"
            "エフェクトや小物が浮遊ジオメトリになる恐れ")
    # 高さの割合ではなく「境界に接しているか」で見切れを判定する。
    # バストアップ像は被写体が画像下端に接するが、高さ比では 0.92 程度にしかならず
    # 閾値では拾えなかった（実測）。VRM は全身が要るので、この検出は重要。
    w0, h0 = original_size
    touch = []
    if bbox[0] <= 1:
        touch.append("左")
    if bbox[1] <= 1:
        touch.append("上")
    if bbox[2] >= w0 - 1:
        touch.append("右")
    if bbox[3] >= h0 - 1:
        touch.append("下")
    meta_touch = touch
    if "下" in touch:
        warnings_.append(
            "被写体が画像下端で見切れている。全身像ではないため VRM 用の"
            "脚部・足元が推定されず、リギングが破綻する可能性が高い")
    elif touch:
        warnings_.append(f"被写体が画像の{'・'.join(touch)}端に接している。見切れの可能性")
    meta["edges_touched"] = meta_touch
    meta["warnings"] = warnings_

    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("PREPROCESS_OK " + json.dumps(
        {k: meta[k] for k in ("output_size", "foreground_coverage",
                              "significant_specks", "warnings")},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
