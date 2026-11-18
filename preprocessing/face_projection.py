"""実写ポートレートを、生成メッシュの正面レンダに重なるよう整列させる。

なぜ必要か: TRELLIS.2 の顔ジオメトリは元画像を2.5倍にしても ArcFace 0.38 前後で
頭打ちになる（docs/05-measurements.md §15）。形で本人性を稼げない以上、実写を
顔テクスチャに焼き込むしかない。その前段として、写真をメッシュの正面カメラの
画角へ合わせる。

整列は ArcFace の 5 点ランドマーク（両目・鼻・口角）から相似変換を解いて行う。
アフィン全体ではなく相似（回転・等方スケール・平行移動）に制限するのは、
せん断が入ると顔の比率が崩れて本人性がむしろ落ちるため。

  python -m preprocessing.face_projection \
      --photo <portrait.jpg> --render <front.png> --out <warped_rgba.png>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")

PACK = "antelopev2"


def _analyzer():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=PACK, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(1024, 1024))
    return app


def landmarks(app, image, want68: bool = False):
    """最大の顔のランドマークを返す。無ければ None。

    want68=False: 5点を (5,2) で返す（従来どおり）
    want68=True:  (5点, 68点 or None) を返す
    """
    import numpy as np

    arr = np.asarray(image.convert("RGB"))[:, :, ::-1]  # PIL(RGB) -> insightface(BGR)
    faces = app.get(arr)
    if not faces:
        return (None, None) if want68 else None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    kps = np.asarray(f.kps, dtype="float32")
    if not want68:
        return kps
    lm = getattr(f, "landmark_3d_68", None)
    return kps, (None if lm is None else np.asarray(lm, dtype="float32")[:, :2])


def face_mask_68(shape, lm68, feather: float = 0.55):
    """68点の輪郭から顔領域のマスクを作る。

    **5点から作る楕円より締まる。** 楕円は目と口の広がりから半径を決めるので、
    `ry = eye_w * 2.15` が顎を越えて**首と襟**に、`rx = 1.55` が**髪**に掛かる。
    実測でその漏れが出た —— 赤いジャケットの色が顎の下に乗り、髪と背景が
    額から生え際に焼き込まれた。

    68点は顎の輪郭（0〜16）を持っているので、そこを下辺にする。額の点は
    68点に無いため、眉（17〜26）を「眉から顎までの距離」の比率で上へ伸ばして補う。
    """
    import cv2
    import numpy as np

    h, w = shape
    jaw = lm68[0:17]
    brow = lm68[17:27]
    chin = lm68[8]
    # 額の高さ。眉から顎までを 1 として、その 0.55 倍だけ上に取る。
    # 大きくすると生え際を越えて髪を拾い、小さいと額が欠けて額だけ素体の肌になる。
    up = (brow.mean(axis=0) - chin) * 0.55
    forehead = brow + up

    poly = np.vstack([jaw, forehead[::-1]]).astype(np.int32)
    m = np.zeros((h, w), dtype="float32")
    cv2.fillConvexPoly(m, cv2.convexHull(poly), 1.0)
    eye_w = float(np.linalg.norm(lm68[45] - lm68[36]))
    k = max(3, int(eye_w * feather) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def face_mask(shape, kps, feather: float = 0.55):
    """ランドマークから顔領域の楕円マスクを作る。境界はぼかす。

    投影を顔だけに限るためのもの。髪や耳まで実写を貼ると、メッシュ側の形状と
    合わずに二重像になる。楕円は目と口の広がりから決め、顎方向に長くとる。

    実測メモ: 顎のあたりに明るい帯が見えることがあるが、これはマスクの縁では
    なく**頭部メッシュ自身のテクスチャが淡い**ため。マスクを首まで伸ばして
    境界を逃がす案を試したが、帯は消えず ArcFace は 0.6634 → 0.6455 に下がった
    （貼る面積が増えて位置ずれの影響も増えるため）。境界の位置ではなく
    メッシュ側のテクスチャ品質の問題なので、ここでは対処しない。
    """
    import cv2
    import numpy as np

    h, w = shape
    eye_c = kps[:2].mean(axis=0)
    mouth_c = kps[3:].mean(axis=0)
    eye_w = float(np.linalg.norm(kps[1] - kps[0]))

    center = (eye_c + mouth_c) / 2.0
    rx = eye_w * 1.55
    ry = eye_w * 2.15
    center = (center[0], center[1] + eye_w * 0.10)

    m = np.zeros((h, w), dtype="float32")
    cv2.ellipse(m, (int(center[0]), int(center[1])), (int(rx), int(ry)),
                0, 0, 360, 1.0, -1)
    k = max(3, int(eye_w * feather) | 1)          # 奇数化
    m = cv2.GaussianBlur(m, (k, k), 0)
    return m


def delight(rgb, mask, strength: float = 0.7, sigma_ratio: float = 0.22):
    """写真から低周波の照明を取り除き、アルベドに近づける。

    元写真には撮影時の照明が焼き付いている。それをそのまま3Dに貼ると、
    **シーンの光を左右反転しても動かない明部**が残り、「ステッカーを貼った」
    ような違和感になる（実測: キーライトを反転しても頬と髪の明部が動かなかった）。

    大きくぼかした自身で割ることで、緩やかな明暗＝照明を平坦化する。
    細かいコントラスト（皺・眉・唇）はぼかしに含まれないので残る。

    strength=1.0 で完全に割り、0 で無効。

    **既定は 0（無効）。実測で悪化した。** 0 / 0.5 / 0.85 を比較すると
    ArcFace は 0.6601 → 0.6403 → 0.6327 と単調に下がり、見た目も首まわりに
    緑がかった色被りが出た。影部を割ると彩度ノイズが増幅されるためで、
    低周波の除算という単純な手法では扱えない。まともに消すには学習ベースの
    de-lighting が要る（docs/05-measurements.md §33）。
    """
    import cv2
    import numpy as np

    if strength <= 0:
        return rgb

    img = rgb.astype("float32") + 1e-3
    h, w = img.shape[:2]
    sigma = max(3.0, min(h, w) * sigma_ratio)
    k = int(sigma * 4) | 1
    lo = cv2.GaussianBlur(img, (k, k), sigma)

    # マスク内の平均輝度を保って割る（全体が暗く/明るくならないように）
    m = (mask > 0.02)
    if m.sum() < 64:
        return rgb
    ref = float(lo[m].mean())
    flat = img * (ref / np.clip(lo, 1e-3, None))
    out = img * (1.0 - strength) + flat * strength
    return np.clip(out, 0, 255).astype("uint8")


def color_match(warped, render, mask, strength: float = 0.85, radius: int = 0):
    """実写の低周波の色だけをメッシュ側に合わせる。

    投影の継ぎ目に色の帯が出るのは、実写の肌の色とメッシュ側テクスチャの色が
    ずれているため。単純に平均・分散を合わせると実写のディテールまで
    メッシュ側の眠い階調に引き寄せられて、本人性を稼ぐという目的に反する。

    そこで **低周波だけ** 差し替える:
        補正後 = 実写 + (ぼかしたメッシュ − ぼかした実写)
    照明と肌の色みはメッシュ側に揃い、皺・目・唇のディテールは実写のまま残る。

    **既定では無効。** 継ぎ目は確かに滑らかになるが、寄せる先のメッシュの顔は
    本人ではない（生成Tポーズ由来）ので、本人性を削る。実測 strength=0.85 で
    ArcFace 0.5824 → 0.5520。継ぎ目の見た目を優先したいときだけ上げる。
    マスク端のぼかし幅(feather)を広げる方は本人性を下げないので、
    そちらは既定で 0.55 に上げてある。
    """
    import cv2
    import numpy as np

    if radius <= 0:
        radius = max(9, int(min(mask.shape) * 0.06)) | 1

    m = (mask > 0.02).astype("uint8")
    if m.sum() < 64:
        return warped

    w = warped.astype("float32")
    r = np.asarray(render.convert("RGB"), dtype="float32")

    # マスク外の画素で平均が引っ張られないよう、マスク内だけで加重平均を取る
    wf = m.astype("float32")
    denom = cv2.GaussianBlur(wf, (radius, radius), 0) + 1e-6
    lo_w = cv2.GaussianBlur(w * wf[..., None], (radius, radius), 0) / denom[..., None]
    lo_r = cv2.GaussianBlur(r * wf[..., None], (radius, radius), 0) / denom[..., None]

    out = w + (lo_r - lo_w) * strength
    return np.clip(out, 0, 255).astype("uint8")


def _retry_detect(app, render):
    """検出に落ちたレンダから顔を拾い直す。縮小 → ガンマの順に試す。

    **縮小が効くのは直感に反するが実測でそうなる。** 生成した全身Tポーズ画像は
    顔が 83px しかなく、TRELLIS.2 はそこから顔の造形を作れない（docs/05 §2 の
    「res512 でのっぺらぼう」と同じ現象が入力側で起きる）。その平坦な顔を頭部に
    寄せて 311px に拡大すると、細部が無いので検出器が顔と認識しない。同じ顔を
    全身の画角で 61px にすると検出できる。縮小してランドマークを取り、
    元の座標系へ戻せば、拡大したレンダのまま整列できる。

    ここで拾えれば投影は成立する。貼るのは実写で、レンダは位置合わせに
    しか使っていない。
    """
    import numpy as np
    from PIL import Image

    rgb = render.convert("RGB")
    for scale in (0.35, 0.2, 0.5):
        w, h = max(64, int(rgb.width * scale)), max(64, int(rgb.height * scale))
        kp = landmarks(app, rgb.resize((w, h), Image.LANCZOS))
        if kp is not None:
            print(f"[faceproj] レンダの顔検出に縮小 {scale} で復帰しました", flush=True)
            return kp / scale          # 元の解像度の座標系へ戻す

    base = np.asarray(rgb, dtype="float32") / 255.0
    for gamma in (1.8, 2.4, 0.7):
        adj = np.clip(np.power(base, gamma), 0.0, 1.0)
        fg = adj[adj.sum(axis=2) > 0.05]          # 背景の黒は無視する
        if fg.size:
            lo, hi = float(fg.min()), float(fg.max())
            if hi - lo > 1e-3:
                adj = np.clip((adj - lo) / (hi - lo), 0.0, 1.0)
        kp = landmarks(app, Image.fromarray((adj * 255).astype("uint8"), "RGB"))
        if kp is not None:
            print(f"[faceproj] レンダの顔検出に gamma={gamma} で復帰しました",
                  flush=True)
            return kp
    return None


def align(photo, render, app, feather: float = 0.55,
          color_match_strength: float = 0.0,
          delight_strength: float = 0.0):
    """photo を render の画角へ相似変換で重ねる。(warped_rgba, meta) を返す。"""
    import cv2
    import numpy as np
    from PIL import Image

    kp_p = landmarks(app, photo)
    if kp_p is None:
        raise SystemExit("写真から顔を検出できませんでした")

    kp_r, lm_r = landmarks(app, render, want68=True)
    if kp_r is None:
        # TRELLIS の出力は当たり外れがあり、顔が白く潰れた候補が混じる。
        # そのままだと検出が落ちるので、ガンマとコントラストを寄せて再試行する。
        # ここで復帰できれば投影は成立する（貼るのは実写なので、レンダは
        # 位置合わせにしか使っていない）。
        kp_r = _retry_detect(app, render)
    if kp_r is None:
        raise SystemExit("レンダから顔を検出できませんでした。"
                         "メッシュの顔が壊れているか、正面ビューではありません")

    # 相似変換のみ（回転+等方スケール+平行移動）。fullAffine=False 相当。
    M, inl = cv2.estimateAffinePartial2D(kp_p, kp_r, method=cv2.LMEDS)
    if M is None:
        raise SystemExit("ランドマークから変換を解けませんでした")

    W, H = render.size
    src = np.asarray(photo.convert("RGB"))
    warped = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REPLICATE)

    # 68点が取れれば輪郭マスク、駄目なら従来の楕円に落ちる。
    if lm_r is not None:
        mask = face_mask_68((H, W), lm_r, feather=feather)
        mask_kind = "landmark68"
    else:
        mask = face_mask((H, W), kp_r, feather=feather)
        mask_kind = "ellipse5"
    # レンダのアルファ（シルエット）で外側を落とす。メッシュの無い場所へは貼らない。
    if render.mode == "RGBA":
        sil = np.asarray(render.getchannel("A"), dtype="float32") / 255.0
        mask = mask * sil

    if delight_strength > 0:
        warped = delight(warped, mask, delight_strength)
    if color_match_strength > 0:
        warped = color_match(warped, render, mask, color_match_strength)

    rgba = np.dstack([warped, (mask * 255).astype("uint8")])
    scale = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
    meta = {
        "photo_size": list(photo.size),
        "render_size": [W, H],
        "similarity_scale": round(scale, 4),
        "rotation_deg": round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 3),
        "inliers": int(inl.sum()) if inl is not None else None,
        "photo_face_width_px": round(float(np.linalg.norm(kp_p[1] - kp_p[0])), 1),
        "render_face_width_px": round(float(np.linalg.norm(kp_r[1] - kp_r[0])), 1),
        "feather": feather,
        "mask": mask_kind,
        "color_match_strength": color_match_strength,
        "delight_strength": delight_strength,
    }
    return Image.fromarray(rgba, "RGBA"), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--render", required=True, help="生成メッシュの正面レンダ(RGBA)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--feather", type=float, default=0.55,
                    help="顔マスク端のぼかし幅（目幅比）。大きいほど継ぎ目が目立たない")
    ap.add_argument("--delight", type=float, default=0.0,
                    help=("写真に焼き付いた照明を取り除く強さ 0..1。"
                          "シーンの光と矛盾する明部を消せるが、上げすぎると"
                          "形状由来の正しい陰影まで平坦になる"))
    ap.add_argument("--color-match", type=float, default=0.0,
                    help=("実写の低周波の色をメッシュ側へ寄せる強さ 0..1。"
                          "既定 0（無効）— 継ぎ目は滑らかになるが本人性が下がる。"
                          "実測 0.85 で ArcFace 0.5824 → 0.5520"))
    args = ap.parse_args()

    from PIL import Image

    photo = Image.open(args.photo)
    render = Image.open(args.render)
    app = _analyzer()
    warped, meta = align(photo, render, app,
                         feather=args.feather,
                         color_match_strength=args.color_match,
                         delight_strength=args.delight)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    warped.save(out)
    meta["output"] = str(out)
    if args.meta:
        Path(args.meta).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("FACEPROJ_OK " + json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
