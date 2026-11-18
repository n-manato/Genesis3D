#!/usr/bin/env python3
"""全身写真をアバターの正面レンダに合わせて歪める（体テクスチャの投影用）。

## なぜこれをやるか

服の**構造**（ブレザーの襟・前開き・中のインナー）はテンプレートに無い。
色転写では色しか移せず（docs/09 §13.4）、生成メッシュから服を切り出す試みは
位置合わせで破綻した（軸ごとの外接箱合わせが奥行きを 1.72倍に伸ばして形が壊れた）。

**顔と同じことを体でやる。** 写真をレンダに合わせて歪め、`blender/project_face.py`
でテクスチャに焼く。シルエットは変わらないが、襟も前開きもインナーも絵として乗る。

## 合わせ方

MediaPipe Pose の33点。顔の 68点と違って点が疎なので、薄板スプライン（TPS）ではなく
**部位ごとの区分アフィン**にする。胴・左右の腕・左右の脚をそれぞれ3〜4点で
アフィン変換し、重み付きで混ぜる。点が疎なまま TPS を掛けると、点の無い場所
（脇・裾）で大きく波打つ。

使い方（**MediaPipe は genesis-eval にある**）:
    $HOME/miniconda3/envs/genesis-eval/bin/python -m preprocessing.body_projection \\
        --photo subject_a/subject_tpose_front.png --render front.png \\
        --photo-landmarks p.json --render-landmarks r.json \\
        --out warped.png --meta warped.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 部位ごとの点（MediaPipe Pose の番号）。区分アフィンの単位。
PARTS = {
    "torso": [11, 12, 23, 24],
    "arm_l": [11, 13, 15],
    "arm_r": [12, 14, 16],
    "leg_l": [23, 25, 27],
    "leg_r": [24, 26, 28],
}


def affine_from(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """最小二乗のアフィン（2x3）。点が3点未満なら相似に落とす。"""
    n = len(src)
    if n >= 3:
        a = np.hstack([src, np.ones((n, 1))])
        m, *_ = np.linalg.lstsq(a, dst, rcond=None)
        return m.T                                    # (2,3)
    # 2点しか無いときは回転＋スケール＋平行移動
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    scale = np.linalg.norm(d) / max(np.linalg.norm(s), 1e-9)
    m = np.eye(2) * scale
    return np.hstack([m, (mu_d - m @ mu_s)[:, None]])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", type=Path, required=True)
    ap.add_argument("--render", type=Path, required=True)
    ap.add_argument("--photo-landmarks", type=Path, required=True)
    ap.add_argument("--render-landmarks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--sigma", type=float, default=0.25,
                    help="部位の影響半径。画像の短辺に対する比")
    ap.add_argument("--torso-only", action="store_true",
                    help="**胴だけを投影する。** 服の構造（襟・前開き・インナー）は"
                         "胴にしかなく、袖と裾は単色なので色転写で足りる。"
                         "全身を投影すると部位の混合が胴の内容を脚へ滲ませ、"
                         "赤い襷が脚まで伸びた（docs/09 §18.2）")
    ap.add_argument("--torso-pad", type=float, default=0.12,
                    help="胴の多角形を外へ広げる比（肩幅に対して）")
    args = ap.parse_args()

    import cv2
    from PIL import Image

    P = np.asarray(json.loads(args.photo_landmarks.read_text())["landmarks_px"],
                   dtype=np.float32)
    R = np.asarray(json.loads(args.render_landmarks.read_text())["landmarks_px"],
                   dtype=np.float32)
    if len(P) < 29 or len(R) < 29:
        raise SystemExit(f"点が足りない: 写真 {len(P)} / レンダ {len(R)}")

    src_im = Image.open(args.photo)
    photo = np.asarray(src_im.convert("RGB"))
    # **背景除去済み（RGBA）を渡すこと。** 背景付きのまま投影すると、体の
    # シルエットが写真と違うところに**写真の背景（灰色）が焼き付く** ——
    # 実測で腕と脚が灰色と白の斑になった。
    photo_a = (np.asarray(src_im.convert("RGBA"))[:, :, 3]
               if src_im.mode in ("RGBA", "LA", "P") else None)
    rh, rw = np.asarray(Image.open(args.render).convert("RGB")).shape[:2]

    # 出力画素 → どの写真画素から引くか（逆写像）。部位ごとのアフィンを重み混合。
    parts = {"torso": PARTS["torso"]} if args.torso_only else PARTS
    ys, xs = np.mgrid[0:rh, 0:rw].astype(np.float32)
    acc_x = np.zeros((rh, rw), np.float32)
    acc_y = np.zeros((rh, rw), np.float32)
    acc_w = np.zeros((rh, rw), np.float32)
    sigma = args.sigma * min(rh, rw)
    # **胴の横方向を「写真のジャケット幅 / 肩幅」で補正する。**
    #
    # 関節どうしで合わせると、写真の**ゆったりしたジャケットの外縁**がモデルの
    # 外へ落ち、内側のインナー（白）だけが胸に来る。実測でそうなった ——
    # 胸が白い面に赤い斜線という、実物と似ていない絵になった。
    #
    # **レンダ側の輪郭は使えない。** アバターの腕は下に垂れているので、胸や腰の
    # 高さで輪郭を測ると腕を拾う（実測: レンダ幅 423/577/695 対 写真幅
    # 233/249/294）。写真は腕が肩の高さに水平なので、腰の高さの輪郭は
    # ジャケットだけ。そこで**写真側だけで比を測り**、逆写像の x を中心から
    # その比で広げる（＝写真の内容を圧縮して、ジャケットの縁が肩に来る）。
    kx, kx_info = 1.0, None
    if photo_a is not None:
        y_waist = int(P[[11, 12], 1].mean()
                      + 0.6 * (P[[23, 24], 1].mean() - P[[11, 12], 1].mean()))
        row = np.where(photo_a[np.clip(y_waist, 0, photo_a.shape[0] - 1)] > 128)[0]
        sh = float(np.linalg.norm(P[11] - P[12]))
        if len(row) > 4 and sh > 1:
            jacket_w = float(row.max() - row.min())
            kx = max(1.0, min(jacket_w / sh, 1.8))
            kx_info = {"jacket_px": round(jacket_w, 1), "shoulder_px": round(sh, 1),
                       "kx": round(kx, 4), "y_waist": int(y_waist)}
            print(f"  胴の横補正 kx={kx:.3f}"
                  f"（ジャケット {jacket_w:.0f}px / 肩 {sh:.0f}px）")
    cx_photo = float(P[[11, 12, 23, 24], 0].mean())

    used = {}
    for name, idx in parts.items():
        idx = [i for i in idx if i < len(P) and i < len(R)]
        if len(idx) < 2:
            continue
        m = affine_from(R[idx], P[idx])              # レンダ → 写像（逆写像）
        if name == "torso" and kx != 1.0:
            # px' = cx + (px - cx) * kx を後段に掛ける
            m = m.copy()
            m[0, :] *= kx
            m[0, 2] += cx_photo * (1.0 - kx)
        used[name] = [[round(float(v), 4) for v in row] for row in m]
        px = m[0, 0] * xs + m[0, 1] * ys + m[0, 2]
        py = m[1, 0] * xs + m[1, 1] * ys + m[1, 2]
        # その部位の点からの距離で重みを付ける
        d2 = np.full((rh, rw), np.inf, np.float32)
        for i in idx:
            d2 = np.minimum(d2, (xs - R[i, 0]) ** 2 + (ys - R[i, 1]) ** 2)
        w = np.exp(-d2 / (2.0 * sigma * sigma))
        acc_x += px * w
        acc_y += py * w
        acc_w += w
    acc_w = np.maximum(acc_w, 1e-6)
    map_x, map_y = acc_x / acc_w, acc_y / acc_w

    warped = cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # 写真の外を指した画素は透明にする（焼くときに合成しない）
    inside = ((map_x >= 0) & (map_x < photo.shape[1]) &
              (map_y >= 0) & (map_y < photo.shape[0])).astype(np.uint8) * 255
    if photo_a is not None:
        wa = cv2.remap(photo_a, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inside = np.minimum(inside, wa)
    if args.torso_only:
        # 胴の多角形の外は投影しない。**縁は羽根でぼかす** ——
        # 切りっぱなしだと色転写した領域との境目に段差が出る。
        sh_l, sh_r = R[11], R[12]
        hp_l, hp_r = R[23], R[24]
        pad = args.torso_pad * float(np.linalg.norm(sh_l - sh_r))
        cen = (sh_l + sh_r + hp_l + hp_r) / 4.0
        poly = []
        for q in (sh_l, sh_r, hp_r, hp_l):
            d = q - cen
            poly.append(q + d / max(np.linalg.norm(d), 1e-6) * pad)
        m = np.zeros((rh, rw), np.uint8)
        cv2.fillPoly(m, [np.asarray(poly, np.int32)], 255)
        m = cv2.GaussianBlur(m, (0, 0), max(pad * 0.5, 2.0))
        inside = np.minimum(inside, m)
        print(f"  胴に限定（肩幅 {np.linalg.norm(sh_l-sh_r):.0f}px、"
              f"外へ {pad:.0f}px 広げて羽根ぼかし）")
    rgba = np.dstack([warped, inside])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(args.out)

    # 残差（レンダの点を写像して写真の点に一致するか）
    res = []
    for name, idx in parts.items():
        idx = [i for i in idx if i < len(P)]
        if name not in used or len(idx) < 2:
            continue
        m = np.asarray(used[name], dtype=float)  # noqa: PLW2901
        got = (m[:, :2] @ R[idx].T).T + m[:, 2]
        res.append(float(np.linalg.norm(got - P[idx], axis=1).mean()))
    rep = {"photo": str(args.photo), "render": str(args.render),
           "out": str(args.out), "render_size": [rw, rh],
           "photo_size": list(photo.shape[1::-1]),
           "coverage": round(float((inside > 128).mean()), 4),
           "photo_had_alpha": photo_a is not None, "torso_kx": kx_info,
           "parts": list(used), "torso_only": args.torso_only, "affines": used,
           "mean_residual_px": round(float(np.mean(res)), 2) if res else None}
    if args.meta:
        args.meta.parent.mkdir(parents=True, exist_ok=True)
        args.meta.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"部位 {len(used)}  被覆 {rep['coverage']*100:.1f}%  "
          f"残差 {rep['mean_residual_px']}px  → {args.out}")


if __name__ == "__main__":
    main()
