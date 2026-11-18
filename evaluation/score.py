"""工程5: 候補の自動評価とランキング。

2種類の情報を突き合わせる:
  (a) レンダリング画像 vs 参照画像 の類似度  … DINOv3 で見た目・同一性を測る
  (b) メッシュの静的解析                      … 幾何・トポロジーの健全性を測る

DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m) は既に HF キャッシュにあり、
オフラインで動く。CLIP は重みが未取得のため、入っていれば使い、無ければ
DINOv3 のみで算出して「使った指標」を記録する（黙って0点にしない）。

較正の根拠（docs/05-measurements.md）:
  実在の製品品質アバターは watertight ではない（boundary edges 63,681 /
  連結成分 2,490）。非watertightや連結成分数を単純に減点すると、正しく
  作られたアバターほど低評価になる。閾値はこの実測値を基準にしている。

  conda run -n genesis-eval python -m evaluation.score \
      --job-dir jobs/<id> --style stylized --out scores/scores.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

DINO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
# ArcFace。DINOv3 は「見た目が似ているか」を測るが「本人か」は測らない。
# docs/05-measurements.md §13 の実測では、両者は逆の候補を勝ちと判定した。
ARCFACE_PACK = "antelopev2"
# 評価に使うビュー。正面は同一性、背面は「未見部分の推定精度」に効く。
VIEWS = ["front", "left", "right", "back", "three_quarter"]


# ---------------------------------------------------------------- 画像特徴
class ImageEncoder:
    def __init__(self, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(DINO_ID)
        self.model = AutoModel.from_pretrained(DINO_ID).eval().to(device)

    def embed(self, image):
        import torch
        # アルファ付きPNGは白背景に合成する。背景色が候補ごとに違うと
        # 「モデルの差」ではなく「背景の差」を測ってしまう。
        if image.mode == "RGBA":
            from PIL import Image
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[-1])
            image = bg
        else:
            image = image.convert("RGB")
        inputs = self.proc(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        feat = out.last_hidden_state[:, 0]           # CLS トークン
        return torch.nn.functional.normalize(feat, dim=-1)



class FaceEncoder:
    """ArcFace(insightface/antelopev2) による本人性の測定。

    realistic_jrpg は identity に 0.25 と最大の重みを置いているが、これまで
    正面ビュー全体の DINOv3 類似度を流用していた。その指標は服の解像感に
    引きずられるため、顔が悪化して衣装が良くなった候補を勝たせてしまう
    （docs/05-measurements.md §13）。

    未導入・顔未検出のときは None を返し、呼び出し側が DINOv3 に落として
    not_measured に記録する。黙って0点にはしない。
    """

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        # GPU は TRELLIS.2 と評価器で取り合いになるので CPU で回す。
        # 候補あたり数枚なので所要は無視できる。
        self.app = FaceAnalysis(name=ARCFACE_PACK, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    @staticmethod
    def _rgb(image):
        """アルファは白へ合成する。ImageEncoder と背景の扱いを揃える。"""
        from PIL import Image

        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[-1])
            return bg
        return image.convert("RGB")

    def detect(self, image):
        """最大の顔を返す。検出できなければ None。"""
        import numpy as np

        rgb = self._rgb(image)
        arr = np.asarray(rgb)[:, :, ::-1]           # PIL(RGB) -> insightface(BGR)
        faces = self.app.get(arr)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def crop(self, image, face, pad: float = 0.9):
        """顔クロップ。髪の生え際と顎まで入るよう bbox を広げる。"""
        rgb = self._rgb(image)
        x0, y0, x1, y1 = face.bbox
        m = (x1 - x0) * pad
        box = (max(0, x0 - m), max(0, y0 - m * 1.1),
               min(rgb.width, x1 + m), min(rgb.height, y1 + m * 0.9))
        return rgb.crop([int(v) for v in box])


def cosine(a, b) -> float:
    return float((a * b).sum().item())


# ---------------------------------------------------------------- 幾何スコア
def geometry_scores(mesh: dict[str, Any]) -> dict[str, float]:
    """メッシュ静的解析 → 0-10。

    非watertightは減点しない（実用アバターも非watertight）。見るのは
    「面数が実用域か」「浮遊片が過剰でないか」「縮退面が無いか」。
    """
    tris = mesh.get("triangles", 0)
    shells = mesh.get("disconnected_shells", 0)
    zero = mesh.get("zero_area_faces", 0)
    loose = mesh.get("loose_vertices", 0) + mesh.get("loose_edges", 0)

    # 面数: VRM実用域 3万〜20万 を満点、外れるほど減点
    if 30_000 <= tris <= 200_000:
        poly = 10.0
    elif tris < 30_000:
        poly = max(0.0, 10.0 * tris / 30_000)
    else:
        poly = max(0.0, 10.0 - (tris - 200_000) / 100_000 * 3.0)

    # 浮遊片: 実在アバターで2,490。それを基準に、桁で多いものを減点する
    if shells <= 100:
        shell_s = 10.0
    elif shells <= 3_000:
        shell_s = 8.0
    elif shells <= 8_000:
        shell_s = 5.0
    else:
        shell_s = max(0.0, 5.0 - (shells - 8_000) / 4_000)

    defect = 10.0
    defect -= min(5.0, zero / 50.0)
    defect -= min(5.0, loose / 100.0)
    defect = max(0.0, defect)

    return {
        "geometry": round((shell_s * 0.6 + defect * 0.4), 3),
        "topology": round(poly, 3),
    }


# ---------------------------------------------------------------- 集計
def load_weights(style: str, styles_yaml: Path) -> tuple[dict, dict, dict]:
    import yaml
    cfg = yaml.safe_load(styles_yaml.read_text())
    st = cfg.get("styles", {}).get(style, {})
    return st.get("weights", {}), cfg.get("gates", {}), cfg.get("time_penalty", {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--reference", default=None,
                    help=("比較の基準画像。既定は <job>/references/rgba.png。"
                          "工程2が参照ビューを生成した場合は、ジオメトリの実際の"
                          "入力になったその画像を渡す"))
    ap.add_argument("--exclude-metrics", default="",
                    help=("重みから外す指標をカンマ区切りで。後段で顔を差し替える"
                          "構成では identity,face を外す（外した分は残りへ再配分）"))
    ap.add_argument("--style", required=True)
    ap.add_argument("--styles-yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from PIL import Image

    job = Path(args.job_dir)
    ref_path = Path(args.reference) if args.reference else job / "references" / "rgba.png"
    if not ref_path.exists():
        raise SystemExit(f"参照画像がありません: {ref_path}")

    device = args.device if torch.cuda.is_available() else "cpu"
    enc = ImageEncoder(device)
    ref_img = Image.open(ref_path)
    ref_emb = enc.embed(ref_img)

    # ArcFace は任意依存。無ければ DINOv3 だけで従来通り動く。
    face_enc = None
    ref_face = None
    ref_face_emb = None
    try:
        face_enc = FaceEncoder()
        ref_face = face_enc.detect(ref_img)
        if ref_face is None:
            print("[score] 参照画像から顔を検出できませんでした。ArcFace は使いません",
                  flush=True)
            face_enc = None
        else:
            ref_face_emb = enc.embed(face_enc.crop(ref_img, ref_face))
    except Exception as e:  # noqa: BLE001
        print(f"[score] ArcFace 未使用: {e}", flush=True)
        face_enc = None

    weights, gates, tpen = load_weights(args.style, Path(args.styles_yaml))

    # 後段で顔を差し替える構成（head_merge）では、候補はすべて同じ生成画像から
    # 起こした体なので、顔の似ていなさを測っても候補を区別できない。実測でも
    # 全候補が identity=0.00 になり、選択は face_crop と幾何だけで決まっていた。
    # 外した重みは残りの指標へ比例配分する（合計を保って 0..10 の尺度を維持）。
    excluded = {m.strip() for m in args.exclude_metrics.split(",") if m.strip()}
    if excluded:
        total_before = sum(weights.values())
        kept = {k: v for k, v in weights.items() if k not in excluded}
        kept_total = sum(kept.values())
        if kept_total > 0:
            weights = {k: round(v / kept_total * total_before, 6)
                       for k, v in kept.items()}
        else:
            excluded = set()          # 全部外すのは無意味。元に戻す

    results: list[dict[str, Any]] = []
    renders_root = job / "renders"
    for cand_dir in sorted(p for p in renders_root.iterdir() if p.is_dir()):
        cand_id = cand_dir.name
        mesh_report = job / "scores" / f"{cand_id}_mesh.json"
        mesh = {}
        if mesh_report.exists():
            mesh = json.loads(mesh_report.read_text()).get("total", {})

        per_view: dict[str, float] = {}
        for v in VIEWS:
            f = cand_dir / f"{v}.png"
            if not f.exists():
                continue
            per_view[v] = round(cosine(ref_emb, enc.embed(Image.open(f))), 4)
        if not per_view:
            continue

        # 顔だけを別に測る。正面ビュー全体の類似度は服の解像感に引きずられるため、
        # identity/face をそこから作ると顔の劣化を見落とす（docs/05 §13）。
        arcface: float | None = None
        face_dino: float | None = None
        front_png = cand_dir / "front.png"
        if face_enc is not None and front_png.exists():
            cand_img = Image.open(front_png)
            det = face_enc.detect(cand_img)
            if det is not None:
                arcface = round(
                    float(ref_face.normed_embedding @ det.normed_embedding), 4)
                face_dino = round(
                    cosine(ref_face_emb, enc.embed(face_enc.crop(cand_img, det))), 4)

        # 正面は同一性、背面は未見部分の推定精度に対応させる
        front = per_view.get("front", 0.0)
        back = per_view.get("back", 0.0)
        mean = sum(per_view.values()) / len(per_view)

        # DINOv3 の cos 類似度は 0.3〜0.9 に収まることが多い。
        # 0.3→0点 / 0.9→10点 に線形写像する（実測レンジに合わせた較正）。
        def to10(x: float) -> float:
            return round(max(0.0, min(10.0, (x - 0.30) / 0.60 * 10.0)), 3)

        # ArcFace の較正: 同一人物の判定閾値は一般に 0.28〜0.36。
        # 0.20 は明確な別人、0.60 は確実な同一人物として 0〜10 に写像する
        # （実測レンジは docs/05-measurements.md §14）。
        def arc10(x: float) -> float:
            return round(max(0.0, min(10.0, (x - 0.20) / 0.40 * 10.0)), 3)

        g = geometry_scores(mesh)
        scores = {
            "identity": arc10(arcface) if arcface is not None else to10(front),
            "face": to10(face_dino) if face_dino is not None else to10(front),
            "body": to10(mean),
            "hair": to10(mean),
            "clothes": to10(mean),
            "texture": to10(mean),
            "back": to10(back),
            **g,
            "rigging": 5.0,               # リギング後にしか測れないため中立値
        }

        overall = sum(scores.get(k, 0.0) * w for k, w in weights.items())
        results.append({
            "candidate": cand_id,
            "view_similarity": per_view,
            "scores": scores,
            "overall": round(overall, 4),
            "mesh": {k: mesh.get(k) for k in
                     ("triangles", "disconnected_shells", "non_manifold_edges",
                      "watertight")},
            "identity_metric": "arcface" if arcface is not None else "dinov3(front流用)",
            "arcface_similarity": arcface,
            "face_crop_similarity": face_dino,
            "metrics_used": ["dinov3"] + (["arcface"] if arcface is not None else []),
            # 「測っていない」理由を取り違えて記録しない。未導入と顔未検出は別物で、
            # 前者は環境の問題、後者はその候補の顔が壊れている可能性を示す。
            "not_measured": ["clip(重み未取得)"]
                            + ([] if arcface is not None else
                               ["arcface(未導入)" if face_enc is None
                                else "arcface(顔未検出)"])
                            + ["hands", "vrm(リギング後に測定)"],
        })

    results.sort(key=lambda r: r["overall"], reverse=True)
    out = {
        "style": args.style,
        "weights": weights,
        "reference": str(ref_path),
        "excluded_metrics": sorted(excluded),
        "encoder": DINO_ID,
        "candidates": results,
        "best_overall": results[0]["candidate"] if results else None,
        "best_geometry": max(results, key=lambda r: r["scores"]["geometry"])["candidate"]
                         if results else None,
        "best_texture": max(results, key=lambda r: r["scores"]["texture"])["candidate"]
                        if results else None,
        "face_encoder": ARCFACE_PACK if face_enc is not None else None,
        "note": ("CLIP は重み未取得。Hands/VRM は未測定。"
                 + ("identity は ArcFace、face は顔クロップ DINOv3 で算出。"
                    if face_enc is not None else
                    "ArcFace が使えないため identity/face は正面 DINOv3 の流用。")),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("SCORE_OK " + json.dumps(
        {"best": out["best_overall"],
         "ranking": [(r["candidate"], r["overall"]) for r in results]},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
