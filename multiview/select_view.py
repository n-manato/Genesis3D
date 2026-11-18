"""生成した参照ビューを選別・順位付けする（工程2の後段）。

TRELLIS.2 は画像を1枚しか受け取らないので、複数シードを振る意味は
「融合するため」ではなく**最良の1枚を選ぶため**。目視ではなく測って選ぶ。

見るのは3つ:

1. **足が見切れていないか** — 下端に接していたら不採用。全身像でないと
   VRM 用の脚部・足元が推定されない（docs/05-measurements.md §4）。
   判定は高さ比ではなく端への接触で行う（`preprocessing/rmbg.py` と同じ根拠）。
2. **Tポーズになっているか** — 指極 ≒ 身長 なので、正しいTポーズなら被写体の
   外接矩形はほぼ正方形になる。腕が下りていると縦長になる。
3. **見た目が本人に寄っているか** — DINOv3 で参照写真と比較。顔ではなく
   髪型・体型・衣装を見るための指標（顔は工程6で実写に置き換わる）。

  conda run -n genesis-eval python -m multiview.select_view \
      --views refs/ --reference subject_a/subject_body_red_1200.jpg --out refs/selection.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def silhouette_stats(rgba, original_size):
    """アルファから被写体の外接矩形・端接触・占有率を出す。"""
    import numpy as np

    a = np.asarray(rgba.getchannel("A"))
    ys, xs = np.where(a > 25)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    w0, h0 = original_size
    touch = []
    if x0 <= 1:
        touch.append("左")
    if y0 <= 1:
        touch.append("上")
    if x1 >= w0 - 2:
        touch.append("右")
    if y1 >= h0 - 2:
        touch.append("下")
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
    m = a > 25

    # 連結成分。人物が1つに繋がっていない生成は腕や脚が欠けている。
    import cv2
    n_lbl, lbl, stats_cc, _ = cv2.connectedComponentsWithStats(
        m.astype("uint8"), connectivity=8)
    areas = sorted(stats_cc[1:, cv2.CC_STAT_AREA], reverse=True) if n_lbl > 1 else []
    main = areas[0] if areas else 0
    stray = sum(1 for x in areas[1:] if x > main * 0.01)   # 主要部の1%超の島

    # 左右対称性。Tポーズ正面は概ね左右対称になるはずで、崩れていれば
    # 腕が下りているか片腕が消えている。被写体の外接矩形内だけで測る。
    sub = m[y0:y1 + 1, x0:x1 + 1]
    sym = float((sub == sub[:, ::-1]).mean()) if sub.size else 0.0

    # 肩の高さの帯が外接矩形の幅をどれだけ占めるか。**スコアには使わない** —
    # 外接矩形は被写体に追随するので、腕が片方消えると矩形も縮んで常に 1.0 に
    # なる（実測で確認）。診断用にだけ残す。腕が伸びていないことは
    # bbox_aspect が、片腕だけ消えたことは symmetry が拾う。
    band = sub[: max(1, int(sub.shape[0] * 0.25))]
    reach = float(band.any(axis=0).mean()) if band.size else 0.0

    return {
        "bbox": [x0, y0, x1, y1],
        "edges_touched": touch,
        "coverage": round(float(m.mean()), 4),
        "bbox_aspect": round(bw / bh, 4),     # Tポーズなら 1.0 付近
        "stray_islands": int(stray),
        "symmetry": round(sym, 4),
        "arm_reach": round(reach, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True, help="生成画像のあるディレクトリ")
    ap.add_argument("--reference", action="append", required=True,
                    help="比較する実写（複数指定可）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default="view_*.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from PIL import Image

    from evaluation.score import ImageEncoder
    from preprocessing.rmbg import load_model, remove_background

    device = args.device if torch.cuda.is_available() else "cpu"
    enc = ImageEncoder(device)
    seg = load_model(device)

    def torso(img, rgba=None):
        """被写体の外接矩形を基準に胴（衣装）だけを切り出す。

        画像全体を比べると、参照写真は普通に立っているのに候補はTポーズなので
        **姿勢の差が類似度を支配する**（実測: 上位3枚が 0.66〜0.67 で横並びに
        なり選別に効かなかった）。外接矩形からの相対位置で胴を切れば、
        姿勢が違っても同じ場所を比べられる。
        """
        import numpy as np

        if rgba is None:
            rgba, _ = remove_background(seg, img, device)
        a = np.asarray(rgba.getchannel("A"))
        ys, xs = np.where(a > 25)
        if len(xs) == 0:
            return img
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        # 頭を外し、腕を外し、胴の中央だけ。Tポーズでも通常立位でも同じ部位になる。
        cx0 = x0 + int(bw * 0.38)
        cx1 = x0 + int(bw * 0.62)
        cy0 = y0 + int(bh * 0.22)
        cy1 = y0 + int(bh * 0.52)
        return img.convert("RGB").crop((cx0, cy0, max(cx0 + 8, cx1), max(cy0 + 8, cy1)))

    refs = [enc.embed(torso(Image.open(r))) for r in args.reference]

    def appearance(img, rgba):
        e = enc.embed(torso(img, rgba))
        sims = [float((r * e).sum()) for r in refs]
        return round(sum(sims) / len(sims), 4)

    rows = []
    for p in sorted(Path(args.views).glob(args.pattern)):
        img = Image.open(p).convert("RGB")
        rgba, _ = remove_background(seg, img, device)
        st = silhouette_stats(rgba, img.size)
        if st is None:
            rows.append({"view": p.name, "rejected": "前景を検出できません"})
            continue

        # 足が見切れていたら不採用。ここは加点ではなくゲート。
        rejected = None
        if "下" in st["edges_touched"]:
            rejected = "被写体が下端で見切れている（足元が推定できない）"
        elif st["coverage"] < 0.05:
            rejected = "被写体が小さすぎる"

        # Tポーズらしさ: 外接矩形の縦横比が 1.0 からどれだけ離れているか。
        # 0.75〜1.30 を実用範囲として 0..1 に写す。
        ar = st["bbox_aspect"]
        tpose = max(0.0, 1.0 - abs(ar - 1.0) / 0.30)

        # シルエットの健全性。生成の失敗を 3D 化する前に落とすための指標。
        # 片腕が消えた画像で実測: 対称性 0.946 → 0.527 / overall 0.789 → 0.376。
        sil = (0.75 * min(1.0, st["symmetry"] / 0.92)
               + 0.25 * (1.0 if st["stray_islands"] == 0 else
                         0.5 if st["stray_islands"] == 1 else 0.0))

        app = appearance(img, rgba)
        rows.append({
            "view": p.name, "path": str(p), **st,
            "tpose_score": round(tpose, 4),
            "silhouette_score": round(sil, 4),
            "appearance": app,
            # 選ぶべきは「3D化に耐える形」。顔の本人性は後段の実写投影が
            # 担保するので、見た目の一致はここでは補助に留める。
            "overall": round(0.40 * tpose + 0.35 * sil + 0.25 * app, 4),
            "rejected": rejected,
        })

    ok = [r for r in rows if not r.get("rejected")]
    ok.sort(key=lambda r: r["overall"], reverse=True)
    out = {
        "views_dir": args.views,
        "references": args.reference,
        "encoder": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "segmenter": "briaai/RMBG-2.0",
        "ranking": ok,
        "rejected": [r for r in rows if r.get("rejected")],
        "best": ok[0]["path"] if ok else None,
        "note": ("tpose_score は外接矩形の縦横比（指極≒身長）、"
                 "silhouette_score は左右対称性・腕の到達・浮遊島の数、"
                 "appearance は胴クロップの DINOv3。"
                 "画像全体で比べると姿勢差に支配されるため胴だけを見る。"
                 "顔の本人性は後段の実写投影が担保する。"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("SELECTVIEW_OK " + json.dumps(
        {"best": Path(out["best"]).name if out["best"] else None,
         "ranking": [(r["view"], r["overall"]) for r in ok],
         "rejected": len(out["rejected"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
