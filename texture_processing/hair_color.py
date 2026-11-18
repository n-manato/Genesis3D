#!/usr/bin/env python3
"""写真の髪色をテンプレートの髪テクスチャへ移す。

## なぜテクスチャだけで足りるか

Rocketbox の髪は `opacity` プリミティブ（680頂点のアルファカード）＋ 2048² のテクスチャ
1枚でできている。髪の**形**はカードが持ち、**色**はテクスチャが持つ。だから色を
合わせるのは3Dの問題ではなく2Dの問題で、メッシュ・UV・ウェイト・モーフには触らない。

これは出荷ゲームと同じ考え方。**ゲームは髪を再構成しない。** 分類してライブラリから
選び、色だけ写真から移す。ラップ方式でテンプレートのトポロジを出荷するようになった
ので、この道が使えるようになった（docs/09 §10）。

## 手口

Reinhard の色転写（Lab 空間で平均と標準偏差を合わせる）。輝度の構造（毛束の陰影）は
そのまま残し、色味と明るさの分布だけを写真に合わせる。単純な色相回転だと金髪→黒髪の
ように明度が大きく変わる組み合わせで破綻する。

**髪の画素の採り方が肝。** 顔の内側は肌なので混ぜてはいけない。68点の眉より上、
かつ背景を除いた領域から採る。鏡面反射と影が平均を引っ張るので上下 10% を捨てる。

使い方:
    $HOME/miniconda3/envs/genesis-face/bin/python -m texture_processing.hair_color \\
        --photo jobs/x/head_rgba.png --texture tmpl_hair.png \\
        --out recolored.png --report jobs/x/scores/hair.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB(0-255) → CIELAB。cv2 を使わず numpy だけで。"""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    e, k = 216 / 24389, 24389 / 27
    f = np.where(xyz > e, np.cbrt(xyz), (k * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def to_rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    e, k = 216 / 24389, 24389 / 27
    def inv(t):
        return np.where(t ** 3 > e, t ** 3, (116 * t - 16) / k)
    xyz = np.stack([inv(fx), np.where(lab[..., 0] > k * e, fy ** 3,
                                      lab[..., 0] / k), inv(fz)], axis=-1)
    xyz = xyz * np.array([0.95047, 1.0, 1.08883])
    m = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    c = xyz @ m.T
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(c * 255.0, 0, 255)


def hair_pixels(photo: Path, log, debug_out: Path | None = None,
                photo_mask: Path | None = None):
    """写真から**髪の画素**を集める。顔の内側（肌）を混ぜないこと。

    `photo_mask`（`preprocessing/hair_mask.py` のセグメンテーション）があれば
    それを使う。無いときだけ下の幾何のヒューリスティックに落ちる —— これは
    背景が単色で額が狭い写真でしか当たらない（docs/09 §28.8）。
    """
    from PIL import Image

    im = Image.open(photo)
    has_alpha = im.mode in ("RGBA", "LA")
    alpha = np.asarray(im.convert("RGBA"))[:, :, 3] if has_alpha else None
    rgb = np.asarray(im.convert("RGB"))
    if photo_mask is not None:
        mk = Image.open(photo_mask).convert("L")
        if mk.size != im.size:
            mk = mk.resize(im.size, Image.NEAREST)
        m = np.asarray(mk) > 128
        if alpha is not None:
            m &= alpha > 128
        return _finish(rgb, m, None, log, debug_out, source="segmentation")

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(1024, 1024))
    faces = app.get(rgb[:, :, ::-1])
    if not faces:
        raise SystemExit(f"{photo}: 顔を検出できない。髪の位置が決まらない")
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    lm = np.asarray(f.landmark_3d_68)[:, :2]
    x0, y0, x1, y1 = f.bbox
    brow_y = float(lm[17:27, 1].min())
    fw = x1 - x0

    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    # 眉より上（額と髪）、または顔の左右に張り出した帯（サイドの髪）
    above = yy < brow_y
    # 側方の帯は**顎より上**に限る。顎より下は肩と上着で、本人の上着は髪と同じ
    # 暗い色なので髪色の標本に混ざっていた。
    # さらに**顔のすぐ隣（顔幅の 0.4 倍以内）・口より上**に限る。短髪の男性では
    # 帯が肩と上着（青いジャケット）を拾い、髪色が上着色になった（L 64）。
    chin_y = float(lm[8, 1])
    mouth_y = y0 + 0.8 * (y1 - y0)
    lateral = (((xx < x0 - 0.05 * fw) & (xx > x0 - 0.45 * fw)) |
               ((xx > x1 + 0.05 * fw) & (xx < x1 + 0.45 * fw))) & (yy < min(mouth_y, chin_y))
    m = above | lateral
    if alpha is not None:
        m &= alpha > 128
    else:
        # 背景が単色なら四隅の色に近い画素を落とす
        corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]]).mean(0)
        m &= np.linalg.norm(rgb.astype(float) - corners, axis=2) > 40
    # 額の肌が混ざるので、肌に近い明るく彩度の低い画素を落とす。
    # **髪の判定を色で決め打ちしない** —— 金髪も黒髪もあるので、
    # 顔の内側から採った肌色との距離で相対的に切る。
    inner = rgb[int(y0 + 0.55 * (y1 - y0)):int(y0 + 0.75 * (y1 - y0)),
                int(x0 + 0.30 * fw):int(x0 + 0.70 * fw)].reshape(-1, 3)
    skin = np.median(inner, axis=0)
    m &= np.linalg.norm(rgb.astype(float) - skin, axis=2) > 45
    # RGB の距離だけでは**強く照らされた額**が残る（頬の中央値より 45 以上明るい）。
    # 肌は色相が同じで明るいだけなので、Lab の a/b が肌に近く L が肌以上の画素も落とす。
    lab_all = to_lab(rgb.reshape(-1, 3).astype(np.float64)).reshape(h, w, 3)
    skin_lab = to_lab(skin[None, :].astype(np.float64))[0]
    chroma_d = np.hypot(lab_all[..., 1] - skin_lab[1], lab_all[..., 2] - skin_lab[2])
    bright_skin = (chroma_d < 12) & (lab_all[..., 0] > skin_lab[0] - 8)
    m &= ~bright_skin

    return _finish(rgb, m, skin, log, debug_out, source="heuristic",
                   extra={"brow_y": round(brow_y, 1),
                          "bbox": [round(float(v), 1) for v in f.bbox]})


def _finish(rgb, m, skin, log, debug_out, source, extra=None):
    from PIL import Image
    if debug_out is not None:
        # 標本にした画素を緑で塗って保存する。髪色が狂ったとき、どこを採ったかを見る
        dbg = rgb.astype(float).copy()
        dbg[m] = dbg[m] * 0.4 + np.array([0, 255, 0]) * 0.6
        debug_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(dbg.astype(np.uint8)).save(debug_out)
    px = rgb[m]
    if len(px) < 500:
        raise SystemExit(f"髪と判定できた画素が {len(px)}個しかない（{source}）")
    lab = to_lab(px)
    # 鏡面反射と影を捨てる
    lo, hi = np.percentile(lab[:, 0], [10, 90])
    keep = (lab[:, 0] >= lo) & (lab[:, 0] <= hi)
    lab = lab[keep]
    log(f"写真の髪 {int(m.sum()):,}画素（外れ値を除いて {len(lab):,}）  {source}"
        + (f"  肌基準 RGB {skin.astype(int).tolist()}" if skin is not None else ""))
    info = {"pixels": int(m.sum()), "used": int(len(lab)), "source": source,
            "skin_rgb": skin.astype(int).tolist() if skin is not None else None}
    info.update(extra or {})
    return lab, info


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", type=Path, required=True,
                    help="髪が写った顔写真。背景除去済み（RGBA）が望ましい")
    ap.add_argument("--texture", type=Path, required=True, help="テンプレートの髪テクスチャ")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--photo-mask", type=Path, default=None,
                    help="写真の髪マスク（preprocessing/hair_mask.py）。"
                         "あれば幾何のヒューリスティックを使わない")
    ap.add_argument("--debug-mask", type=Path, default=None,
                    help="写真のどこを髪の標本にしたかを塗った画像を保存する")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="0=元のまま 1=完全に写真の分布へ")
    ap.add_argument("--outlier-sigma", type=float, default=0.0,
                    help="輝度が髪の分布からこれ以上（σ）外れる画素は髪の平均色で塗る。"
                         "0 で無効。生成髪のテクスチャは 4 が目安")
    ap.add_argument("--min-luma", type=float, default=6.0,
                    help="この輝度以下は透明部分とみなして触らない")
    ap.add_argument("--clear-mask", type=Path, default=None,
                    help="このマスクの領域の**アルファをゼロにする**。"
                         "まつ毛カードを消して写真のまつ毛に任せるのに使う")
    ap.add_argument("--mask", type=Path, default=None,
                    help="変える範囲のマスク（白=変える）。"
                         "head テクスチャの頭皮だけを直すときに使う。"
                         "**顔を含むテクスチャに無指定で当てないこと**")
    args = ap.parse_args()

    from PIL import Image

    def log(s):
        print(s, flush=True)

    tgt_lab, info = hair_pixels(args.photo, log, args.debug_mask, args.photo_mask)

    tex = Image.open(args.texture)
    arr = np.asarray(tex.convert("RGB")).astype(np.float64)
    a_in = np.asarray(tex.convert("RGBA"))[:, :, 3] if tex.mode in ("RGBA", "LA") else None
    lab = to_lab(arr)
    # 透明部（黒く潰れている）は触らない。触ると縁に色が乗る。
    m = lab[:, :, 0] > args.min_luma
    if a_in is not None:
        m &= a_in > 8
    mask_w = None
    if args.mask:
        mk = Image.open(args.mask).convert("L")
        if mk.size != tex.size:
            mk = mk.resize(tex.size, Image.LANCZOS)
        mask_w = np.asarray(mk).astype(np.float64) / 255.0
        m &= mask_w > 0.02
    if m.sum() < 100:
        raise SystemExit("テクスチャに髪の画素が見つからない（--min-luma を下げる）")

    src = lab[m]
    s_mu, s_sd = src.mean(0), src.std(0) + 1e-6
    t_mu, t_sd = tgt_lab.mean(0), tgt_lab.std(0) + 1e-6
    # Reinhard。輝度の構造（毛束の陰影）は残し、分布だけ移す。
    z_raw = (src - s_mu) / s_sd
    z = np.clip(z_raw, -2.5, 2.5)   # 外挿を止める（色域外へ飛ぶ）
    new = z * t_sd + t_mu
    # **髪の分布から大きく外れる画素は髪ではない。** 生成メッシュから切り出した髪の
    # テクスチャは生え際・耳のまわりに肌の色がにじんでいる（L≈60、髪は 5.6±4）。
    # 分布合わせを掛けると 13σ の外れ値が +2.5σ に切られて「明るい茶色の斑」に
    # なる。外れ値は髪の平均色で塗る。
    outliers = 0
    if args.outlier_sigma > 0:
        far = np.abs(z_raw[:, 0]) > args.outlier_sigma
        new[far] = t_mu
        outliers = int(far.sum())
    new = src + args.strength * (new - src)
    new[:, 0] = np.clip(new[:, 0], 0, 100)

    out_lab = lab.copy()
    out_lab[m] = new
    out = arr.copy()
    recolored = to_rgb(out_lab[m])
    if mask_w is not None:
        # マスクの縁で急に色が変わると継ぎ目が出る。羽根の分だけ混ぜる。
        w = mask_w[m][:, None]
        out[m] = arr[m] * (1.0 - w) + recolored * w
    else:
        out[m] = recolored
    img = Image.fromarray(out.astype(np.uint8))
    cleared = 0
    if a_in is not None:
        a_out = a_in.astype(np.float64)
        if args.clear_mask:
            ck = Image.open(args.clear_mask).convert("L")
            if ck.size != tex.size:
                ck = ck.resize(tex.size, Image.LANCZOS)
            cw = np.asarray(ck).astype(np.float64) / 255.0
            a_out = a_out * (1.0 - cw)
            cleared = int((cw > 0.5).sum())
        img.putalpha(Image.fromarray(a_out.astype(np.uint8)))
    elif args.clear_mask:
        print("警告: 入力にアルファが無いので --clear-mask は効かない",
              file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)

    rep = {"photo": str(args.photo), "texture": str(args.texture),
           "out": str(args.out), "strength": args.strength,
           "photo_hair": info,
           "texture_pixels": int(m.sum()),
           "mask": str(args.mask) if args.mask else None,
           "clear_mask": str(args.clear_mask) if args.clear_mask else None,
           "alpha_cleared_px": cleared,
           "lab_before": [round(float(v), 2) for v in s_mu],
           "lab_after": [round(float(v), 2) for v in to_lab(out)[m].mean(0)],
           "lab_target": [round(float(v), 2) for v in t_mu],
           "outliers_painted": outliers}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log(f"テンプレの髪 Lab {rep['lab_before']} → {rep['lab_after']}"
        f"（狙い {rep['lab_target']}）")
    log(f"→ {args.out}")


if __name__ == "__main__":
    main()
