#!/usr/bin/env python3
"""投影した実写を、テンプレートの肌テクスチャへ馴染ませる（MeInGame §3.1 の後半）。

`blender/project_face.py` は実写をテンプレートの UV へ焼くところまでやる。
そのままだと**写真の照明がそのまま乗る**:

  - 逆光の写真なら顔だけ白飛びし、首から下は素体の肌のまま段差になる
  - 顔の外周に色の縁が出る

単純に色を合わせにいくのは筋が悪い。`face_projection.color_match` の実測では
strength 0.85 で ArcFace が 0.5824 → 0.5520 に落ちた。**寄せる先は別人の顔**なので、
色を揃えるほど本人性が減る。

Poisson 合成（勾配領域の合成）はそこを分ける。**勾配＝皺・目・唇・毛穴は実写のまま、
絶対値＝照明と肌の色みだけが周囲に従う。** 継ぎ目は定義上ゼロになる。

被覆マスクは「投影後」と「テンプレート」の差分から起こす。投影は焼き込みの
一発勝負で、被覆だけを別に焼き出す経路が無いため。写真とテンプレートの色が
たまたま一致した画素は取りこぼすが、そこは合成しても見た目が変わらない。

使い方:
    python -m texture_processing.skin_blend \\
        --base <テンプレの head.png> --patch <投影後の head.png> \\
        --out <合成後.png> [--symmetry-fill] [--report r.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def coverage_mask(base: np.ndarray, patch: np.ndarray, thresh: int = 12,
                  open_k: int = 15, close_k: int = 9,
                  keep_ratio: float = 0.05) -> np.ndarray:
    """実写が乗った領域を差分から起こす。0/255 の uint8。

    **開処理の核を十分大きく取ること。** 焼き込みの margin 処理はテンプレート側と
    投影側で微妙に違うので、**UV 島の輪郭に沿って細い差分の線が出る**。3x3 の
    開処理では消えず、アトラス全体の縁取りがマスクに入って Poisson 合成が
    そこでも走ってしまう（実測: 顔以外に19個の成分が拾われた）。
    線の幅より大きい核で開けば消える。

    残った成分のうち、**最大の 5% 未満のものは落とす**。実写が乗るのは顔という
    一つの塊なので、それより桁が小さいものは縁の残りかノイズ。
    """
    d = np.abs(base.astype(np.int16) - patch.astype(np.int16)).max(axis=2)
    m = (d > thresh).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas.max() * keep_ratio
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= keep:
            out[lab == i] = 255
    return out


def poisson(base: np.ndarray, patch: np.ndarray, mask: np.ndarray,
            mode: str = "normal") -> np.ndarray:
    """連結成分ごとに seamlessClone する。

    OpenCV の seamlessClone は**マスク全体の外接矩形の中心**を貼り付け位置に使うので、
    離れた島をまとめて渡すと位置がずれる。成分ごとに呼ぶ。
    """
    flag = cv2.MIXED_CLONE if mode == "mixed" else cv2.NORMAL_CLONE
    out = base.copy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    used = 0
    for i in range(1, n):
        sub = np.where(lab == i, 255, 0).astype(np.uint8)
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        # 画像の縁に接している成分は seamlessClone が落ちる。1画素だけ内側へ寄せる。
        if x <= 0 or y <= 0 or x + w >= base.shape[1] or y + h >= base.shape[0]:
            sub[0, :] = sub[-1, :] = sub[:, 0] = sub[:, -1] = 0
            if sub.max() == 0:
                continue
        center = (int(x + w / 2), int(y + h / 2))
        try:
            out = cv2.seamlessClone(patch, out, sub, center, flag)
            used += 1
        except cv2.error:
            # 稀に「マスクが小さすぎる」で落ちる。その成分は単純合成に落とす。
            a = (sub > 0)[:, :, None]
            out = np.where(a, patch, out)
    return out, used


def symmetry_fill(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """被覆されなかった側を、左右反転した被覆側から埋める。

    横向きぎみの写真では片側が隠れる。UV が u=0.5 で左右対称なテンプレートなら、
    反対側の texel をそのまま持ってこられる。
    **対称でない UV では何もしないほうがよい**ので、埋めた画素数を返して判断させる。
    """
    flipped = img[:, ::-1]
    fmask = mask[:, ::-1]
    need = (mask == 0) & (fmask > 0)
    if not need.any():
        return img, 0
    out = img.copy()
    out[need] = flipped[need]
    return out, int(need.sum())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="テンプレートの肌テクスチャ")
    ap.add_argument("--patch", type=Path, required=True, help="実写を焼いたテクスチャ")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("normal", "mixed"), default="normal",
                    help="mixed は下地の勾配も混ぜる。素体の肌に模様がある場合のみ")
    ap.add_argument("--threshold", type=int, default=12)
    ap.add_argument("--open-kernel", type=int, default=15,
                    help="UV島の縁に出る細い差分を消す。小さいとアトラス全体の"
                         "縁取りがマスクに入る")
    ap.add_argument("--symmetry-fill", action="store_true")
    ap.add_argument("--mask-out", type=Path, help="被覆マスクを書き出す（確認用）")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    base = cv2.imread(str(args.base), cv2.IMREAD_COLOR)
    patch = cv2.imread(str(args.patch), cv2.IMREAD_COLOR)
    if base is None or patch is None:
        raise SystemExit("画像を読めない")
    if base.shape != patch.shape:
        patch = cv2.resize(patch, (base.shape[1], base.shape[0]),
                           interpolation=cv2.INTER_LANCZOS4)

    mask = coverage_mask(base, patch, thresh=args.threshold, open_k=args.open_kernel)
    cov = int((mask > 0).sum())
    if cov == 0:
        raise SystemExit("実写が乗った領域が見つからない。投影が失敗しているのでは")

    out, used = poisson(base, patch, mask, args.mode)
    filled = 0
    if args.symmetry_fill:
        out, filled = symmetry_fill(out, mask)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), out)
    if args.mask_out:
        cv2.imwrite(str(args.mask_out), mask)

    rep = {"base": str(args.base), "patch": str(args.patch), "out": str(args.out),
           "size": [base.shape[1], base.shape[0]], "mode": args.mode,
           "coverage_px": cov, "coverage_ratio": round(cov / mask.size, 4),
           "components_blended": used, "symmetry_filled_px": filled}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))

    print(f"被覆 {cov:,}画素（{rep['coverage_ratio']*100:.1f}%）  合成した成分 {used}個"
          + (f"  対称補完 {filled:,}画素" if args.symmetry_fill else ""))
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
