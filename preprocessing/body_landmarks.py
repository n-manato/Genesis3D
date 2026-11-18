#!/usr/bin/env python3
"""全身写真から体の寸法を測る。MediaPipe Pose の 33点を使う。

`game_avatar` の体トラック（docs/08 §10.13）。顔は生成メッシュで作るが、
**体はテンプレート素体を変形して本人の身長・体型に寄せる**。

## 写真から取れるもの・取れないもの

**比率は取れるが絶対寸法は取れない。** 写真1枚にはスケールの手がかりが無いので、
身長は `--height-cm` で与える。これが最も確実で、モデルの追加もいらない。

比率のほうは 33点から素直に出る:

  - 肩幅 / 肩〜足首          体格の主要因
  - 腰幅 / 肩幅              体型（逆三角か寸胴か）
  - 脚長 / 全高              頭身

**頭頂はランドマークに無い。** 鼻から上は推定になるので、身長の基準には
「肩〜足首」を使い、`--height-cm` との比で頭までを外挿する。

MediaPipe 1.x は Tasks API のみ（`mp.solutions` は消えた）。モデルは
`vendor/mediapipe/pose_landmarker_heavy.task`（30MB）。

使い方:
    python -m preprocessing.body_landmarks --input subject_a/subject_body_red_1200.jpg \
        --height-cm 160 --out jobs/x/scores/body.json [--review r.png]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MODEL = Path("vendor/mediapipe/pose_landmarker_heavy.task")

# MediaPipe Pose の 33点のうち使うもの
L = {"nose": 0, "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
     "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24,
     "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
     "l_heel": 29, "r_heel": 30}


def detect(path: Path, model: Path):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if not model.exists():
        raise SystemExit(
            f"{model} が無い。以下で取得できる:\n"
            f"  curl -sL -o {model} https://storage.googleapis.com/mediapipe-models/"
            f"pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE, num_poses=1)
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        img = mp.Image.create_from_file(str(path))
        res = lm.detect(img)
    if not res.pose_landmarks:
        raise SystemExit(f"{path} で人体を検出できない")
    h, w = img.height, img.width
    pts = np.array([[p.x * w, p.y * h] for p in res.pose_landmarks[0]])
    vis = np.array([getattr(p, "visibility", 1.0) for p in res.pose_landmarks[0]])
    return pts, vis, (w, h)


def measure(pts: np.ndarray, vis: np.ndarray) -> dict:
    """比率を出す。すべて「肩〜足首」で正規化した無次元量。"""
    def mid(a, b):
        return (pts[L[a]] + pts[L[b]]) / 2.0

    shoulder = np.linalg.norm(pts[L["l_shoulder"]] - pts[L["r_shoulder"]])
    hip = np.linalg.norm(pts[L["l_hip"]] - pts[L["r_hip"]])
    sh_c, hip_c = mid("l_shoulder", "r_shoulder"), mid("l_hip", "r_hip")
    ank_c = mid("l_ankle", "r_ankle")
    # 縦の基準。**肩〜足首**を使う。頭頂はランドマークに無いため。
    torso_to_ankle = float(np.linalg.norm(sh_c - ank_c))
    if torso_to_ankle < 1e-6:
        raise SystemExit("肩と足首が同じ位置に出ている。全身が写っていないのでは")
    leg = float(np.linalg.norm(hip_c - ank_c))
    upper_arm = float(np.linalg.norm(pts[L["l_shoulder"]] - pts[L["l_elbow"]]))
    fore_arm = float(np.linalg.norm(pts[L["l_elbow"]] - pts[L["l_wrist"]]))

    used = ["l_shoulder", "r_shoulder", "l_hip", "r_hip", "l_ankle", "r_ankle"]
    return {
        "shoulder_over_span": round(shoulder / torso_to_ankle, 5),
        "hip_over_shoulder": round(hip / max(shoulder, 1e-6), 5),
        "leg_over_span": round(leg / torso_to_ankle, 5),
        "arm_over_span": round((upper_arm + fore_arm) / torso_to_ankle, 5),
        "_px": {"shoulder": round(shoulder, 1), "hip": round(hip, 1),
                "shoulder_to_ankle": round(torso_to_ankle, 1), "leg": round(leg, 1)},
        "min_visibility": round(float(vis[[L[k] for k in used]].min()), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="全身写真")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--height-cm", type=float, default=None,
                    help="本人の身長。**写真からは取れないので与えること。**"
                         "無ければテンプレートの身長のままになる")
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--review", type=Path)
    args = ap.parse_args()

    pts, vis, (w, h) = detect(args.input, args.model)
    m = measure(pts, vis)
    m.update({"input": str(args.input), "image_size": [w, h],
              "height_cm": args.height_cm,
              "landmarks_px": [[round(float(x), 1), round(float(y), 1)] for x, y in pts]})

    if m["min_visibility"] < 0.5:
        print(f"警告: 主要な関節の可視度が低い（{m['min_visibility']}）。"
              f"全身がはっきり写った写真のほうがよい", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(m, ensure_ascii=False, indent=1))

    if args.review:
        from PIL import Image, ImageDraw
        im = Image.open(args.input).convert("RGB")
        d = ImageDraw.Draw(im)
        for k, i in L.items():
            x, y = pts[i]
            c = (90, 220, 90) if vis[i] > 0.5 else (255, 80, 80)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=c)
        args.review.parent.mkdir(parents=True, exist_ok=True)
        im.save(args.review)

    print(f"肩幅/肩足首 {m['shoulder_over_span']}  腰/肩 {m['hip_over_shoulder']}  "
          f"脚/肩足首 {m['leg_over_span']}  可視度 {m['min_visibility']}")
    print(f"身長 {args.height_cm if args.height_cm else '未指定（テンプレのまま）'}"
          + (" cm" if args.height_cm else ""))
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
