"""ControlNet-OpenPose に渡す Tポーズ骨格画像を生成する。

手描きの骨格素材を持ち込まずに済ませるため、COCO-18 のキーポイントを
プログラムで配置して OpenPose の描画規約どおりに描く。**色と描き方は規約
どおりでなければならない** — ControlNet はこの配色で学習されているので、
色が違うと姿勢として認識されない。

Tポーズは腕を真横に伸ばすため、全身の外接矩形はほぼ正方形になる
（成人の指極 ≒ 身長）。縦長キャンバスにすると手首が画面外に出るので、
既定は 1024x1024 の正方形。

  python -m multiview.tpose_skeleton --out refs/ --views front,back
"""
from __future__ import annotations

import argparse
from pathlib import Path

# COCO-18。索引は OpenPose の定義そのもの。
NOSE, NECK = 0, 1
R_SHO, R_ELB, R_WRI = 2, 3, 4
L_SHO, L_ELB, L_WRI = 5, 6, 7
R_HIP, R_KNE, R_ANK = 8, 9, 10
L_HIP, L_KNE, L_ANK = 11, 12, 13
R_EYE, L_EYE, R_EAR, L_EAR = 14, 15, 16, 17

# 描画される 17 本の肢。OpenPose の limbSeq を 0-index にしたもの。
LIMBS = [
    (NECK, R_SHO), (NECK, L_SHO), (R_SHO, R_ELB), (R_ELB, R_WRI),
    (L_SHO, L_ELB), (L_ELB, L_WRI), (NECK, R_HIP), (R_HIP, R_KNE),
    (R_KNE, R_ANK), (NECK, L_HIP), (L_HIP, L_KNE), (L_KNE, L_ANK),
    (NECK, NOSE), (NOSE, R_EYE), (R_EYE, R_EAR), (NOSE, L_EYE), (L_EYE, L_EAR),
]

# OpenPose の標準配色。変更してはいけない。
COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

# 正規化キーポイント (x, y) を 0..1 で。正面向き＝人物の右手が画面左。
#
# 比率は成人女性の 7.5 頭身 canon で置く。適当に置くと ControlNet が
# 従った結果そのまま歪んだ体型の人物が生成され、後段のリギングまで狂う。
#   全高 H = 0.905 - 0.055 = 0.85 / 頭高 = H/7.5 = 0.1133
#   頭頂 0.055 / 目 1.5頭身 / 肩 1.5頭身 / 腰 3.75頭身 / 膝 5.5頭身 / 足首 7.4頭身
# 指極 ≒ 身長 なので手首は中心から ±H/2 = ±0.425 が理屈だが、そこに置くと
# 指先が枠に接して見切れる（実測: 生成4枚すべてが左端に接触）。±0.402 に
# 引いて左右に約10%の余白を作る。指極/身長 = 0.96 で人体の範囲内。
# 足首を 0.893 に置いて下端に約 7% の余白を残す — 足が見切れると VRM 用の
# 足元が推定できない（docs/05-measurements.md §4）。
_FRONT = {
    NOSE:  (0.500, 0.130),
    NECK:  (0.500, 0.225),          # OpenPose の neck は両肩の中点
    R_SHO: (0.410, 0.225), L_SHO: (0.590, 0.225),
    R_ELB: (0.255, 0.225), L_ELB: (0.745, 0.225),
    R_WRI: (0.098, 0.225), L_WRI: (0.902, 0.225),
    R_HIP: (0.425, 0.480), L_HIP: (0.575, 0.480),
    R_KNE: (0.435, 0.678), L_KNE: (0.565, 0.678),
    R_ANK: (0.445, 0.893), L_ANK: (0.555, 0.893),
    R_EYE: (0.480, 0.112), L_EYE: (0.520, 0.112),
    R_EAR: (0.458, 0.118), L_EAR: (0.542, 0.118),
}


def keypoints(view: str) -> dict[int, tuple[float, float]]:
    """指定ビューの正規化キーポイント。

    背面は左右反転する。OpenPose のキーポイントは**人物自身の左右**で
    ラベル付けされるので、後ろを向けば人物の右手は画面右に来る。
    """
    if view == "front":
        return dict(_FRONT)
    if view == "back":
        return {k: (1.0 - x, y) for k, (x, y) in _FRONT.items()}
    raise SystemExit(f"未対応のビュー: {view}（front / back のみ）")


def draw(kps: dict[int, tuple[float, float]], width: int, height: int):
    """OpenPose の描画規約で骨格を描く。肢は楕円ポリゴン、関節は円。"""
    import cv2
    import numpy as np

    canvas = np.zeros((height, width, 3), dtype="uint8")
    pts = {i: (x * width, y * height) for i, (x, y) in kps.items()}
    stick = max(1, int(min(width, height) / 180))   # 実装の慣例に合わせた太さ

    for i, (a, b) in enumerate(LIMBS):
        if a not in pts or b not in pts:
            continue
        (x0, y0), (x1, y1) = pts[a], pts[b]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        length = float(np.hypot(x0 - x1, y0 - y1))
        angle = float(np.degrees(np.arctan2(y0 - y1, x0 - x1)))
        poly = cv2.ellipse2Poly((int(mx), int(my)),
                                (int(length / 2), stick), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, COLORS[i])

    for i, (x, y) in pts.items():
        cv2.circle(canvas, (int(x), int(y)), stick + 1, COLORS[i], thickness=-1)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="出力ディレクトリ")
    ap.add_argument("--views", default="front,back")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    args = ap.parse_args()

    import cv2

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for v in [s.strip() for s in args.views.split(",") if s.strip()]:
        img = draw(keypoints(v), args.width, args.height)
        p = outdir / f"pose_{v}.png"
        cv2.imwrite(str(p), img)
        written.append(str(p))
    print("TPOSE_OK " + str({"written": written,
                             "size": [args.width, args.height]}))


if __name__ == "__main__":
    main()
