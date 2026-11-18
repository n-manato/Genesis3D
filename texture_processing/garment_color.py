#!/usr/bin/env python3
"""全身写真の服の色を、テンプレートの体テクスチャへ移す。

## なぜ2Dの問題として解けるか

Rocketbox の服は**メッシュではなく `f001_body_color.tga`（2048²）に描いてある**。
シャツの襟もジーンズの縫い目も全部テクスチャ。だから衣装合わせは3Dの
ガーメントフィッティングではなく、**画像の色替え**になる。メッシュ・UV・ボーン・
ウェイト・モーフには触らない（docs/09 §12）。

UV レイアウトが都合よくモジュラーで、幾何の帯がそのまま島に対応する（実測）:

    領域          三角形   UV被覆   テンプレの平均色
    胴(シャツ)      895    21.0%   [188,104,116]  ピンク
    腕              522     6.6%   [180,105,109]  肌
    脚(ジーンズ)     397    15.3%   [ 58, 81, 93]  デニム
    靴              515     3.5%   [ 38, 44, 46]
    首              164     1.4%   [170, 93,104]

**色で服と肌を分けようとしてはいけない。** テンプレートのシャツはピンクで、
肌 [205,151,137] とほとんど区別が付かない。幾何（高さと中心からの距離）で切る。

## 手口

写真からは MediaPipe Pose の33点で胴と脚の領域を決めて中位色を採る。胴は
中央を避けて左右の胸元から採る ——ブレザーの前が開いていて中央にインナーが
写るため。転写は Reinhard（Lab で平均と標準偏差を合わせる）で、布のしわと
縫い目の陰影は輝度構造として残る。

使い方:
    $HOME/miniconda3/envs/genesis-face/bin/python -m texture_processing.garment_color \\
        --model jobs/x/body_deformed.glb --photo subject_a/subject_tpose_front.png \\
        --body-landmarks jobs/x/scores/body.json \\
        --texture body.png --out body_new.png --report jobs/x/scores/garment.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blender.vrm_finalize import read_glb                          # noqa: E402
from preprocessing.shape_transfer import read_accessor             # noqa: E402
from preprocessing.template_landmarks import face_primitive        # noqa: E402
from texture_processing.face_texture_blend import rasterize        # noqa: E402
from texture_processing.hair_color import to_lab, to_rgb           # noqa: E402

# MediaPipe Pose の点番号
L_SHO, R_SHO, L_HIP, R_HIP, L_KNEE, R_KNEE = 11, 12, 23, 24, 25, 26


def sample(img: np.ndarray, pts: np.ndarray, log, tag: str):
    """多角形の内側の画素を Lab で返す。上下 15% を捨てて影と反射を除く。"""
    from PIL import Image, ImageDraw
    h, w = img.shape[:2]
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).polygon([(float(x), float(y)) for x, y in pts], fill=255)
    sel = np.asarray(m) > 128
    px = img[sel]
    if len(px) < 200:
        raise SystemExit(f"{tag}: 標本が {len(px)}画素しかない")
    lab = to_lab(px)
    lo, hi = np.percentile(lab[:, 0], [15, 85])
    lab = lab[(lab[:, 0] >= lo) & (lab[:, 0] <= hi)]
    log(f"  {tag}: {len(px):,}画素（外れ値を除いて {len(lab):,}）  "
        f"Lab {np.round(lab.mean(0), 1).tolist()}  "
        f"RGB {to_rgb(lab.mean(0)[None]).astype(int)[0].tolist()}")
    return lab


def transfer(arr: np.ndarray, mask: np.ndarray, tgt_lab: np.ndarray,
             strength: float, min_luma: float):
    """マスクの内側を Reinhard で目標の分布へ寄せる。"""
    lab = to_lab(arr)
    m = (lab[:, :, 0] > min_luma) & (mask > 0.02)
    if m.sum() < 100:
        return arr, 0, None
    src = lab[m]
    s_mu, s_sd = src.mean(0), src.std(0) + 1e-6
    t_mu, t_sd = tgt_lab.mean(0), tgt_lab.std(0) + 1e-6
    # **正規化偏差をクリップする。** 領域の中に元の分布から外れた画素
    # （襟元に覗くインナーの肌色など）があると、標準偏差合わせが外挿して
    # 色域外へ飛ぶ。実際に襟の中央が緑の斑になった。
    z = np.clip((src - s_mu) / s_sd, -2.5, 2.5)
    new = z * t_sd + t_mu
    new = src + strength * (new - src)
    new[:, 0] = np.clip(new[:, 0], 0, 100)
    new[:, 1:] = np.clip(new[:, 1:], -100.0, 100.0)
    out = arr.copy()
    w = mask[m][:, None]
    out[m] = arr[m] * (1.0 - w) + to_rgb(new) * w
    return out, int(m.sum()), [round(float(v), 2) for v in s_mu]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="幾何と UV を取る glb")
    ap.add_argument("--photo", type=Path, required=True, help="全身写真（Tポーズ推奨）")
    ap.add_argument("--body-landmarks", type=Path, required=True,
                    help="preprocessing.body_landmarks の出力（33点の画素座標）")
    ap.add_argument("--texture", type=Path, required=True, help="体のテクスチャ")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--body-material", default="body")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--mask-out", type=Path, help="マスクの確認画像")
    # 相対指定なので体型変形でスケールが変わっても効く
    ap.add_argument("--shirt-y", default="0.63,0.97", help="体の高さの比（下,上）")
    ap.add_argument("--shirt-x", type=float, default=0.34,
                    help="体の半幅のこの比より内側。腕を除くための境界")
    ap.add_argument("--pants-y", default="0.07,0.60")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--min-luma", type=float, default=4.0)
    ap.add_argument("--feather", type=int, default=4)
    args = ap.parse_args()

    from PIL import Image, ImageFilter

    def log(s):
        print(s, flush=True)

    doc, blob = read_glb(args.model)
    mi, pi, v, tri, matname = face_primitive(doc, blob, args.body_material)
    uv = read_accessor(doc, blob, doc["meshes"][mi]["primitives"][pi]["attributes"]["TEXCOORD_0"])
    c = v[tri].mean(axis=1)
    y0, y1 = float(v[:, 1].min()), float(v[:, 1].max())
    xh = float(np.abs(v[:, 0]).max())
    sy = [float(t) for t in args.shirt_y.split(",")]
    py = [float(t) for t in args.pants_y.split(",")]
    sel = {
        "shirt": (np.abs(c[:, 0]) < args.shirt_x * xh)
                 & (c[:, 1] > y0 + sy[0] * (y1 - y0))
                 & (c[:, 1] < y0 + sy[1] * (y1 - y0)),
        "pants": (c[:, 1] > y0 + py[0] * (y1 - y0))
                 & (c[:, 1] < y0 + py[1] * (y1 - y0)),
    }

    tex = Image.open(args.texture)
    arr = np.asarray(tex.convert("RGB")).astype(np.float64)
    size = tex.size[0]

    lmj = json.loads(args.body_landmarks.read_text())
    lm = np.asarray(lmj["landmarks_px"], dtype=float)
    photo = np.asarray(Image.open(args.photo).convert("RGB"))

    # 胴は**中央を避けて**左右の胸元から採る。ブレザーの前が開いていると
    # 中央はインナーの色になり、上着の色を取り違える。
    sh_l, sh_r, hp_l, hp_r = lm[L_SHO], lm[R_SHO], lm[L_HIP], lm[R_HIP]
    def lerp(a, b, t):
        return a + (b - a) * t
    quads = {}
    for side, (s, h) in (("L", (sh_l, hp_l)), ("R", (sh_r, hp_r))):
        mid = lerp(s, h, 0.35)
        inner = lerp(s, sh_r if side == "L" else sh_l, 0.35)
        quads[side] = np.array([
            lerp(s, mid, 0.35), lerp(inner, lerp(inner, mid, 1.0), 0.35),
            lerp(mid, h, 0.35), lerp(s, h, 0.55)])
    torso = np.vstack([
        lerp(sh_l, sh_r, 0.12), lerp(sh_l, sh_r, 0.30),
        lerp(hp_l, hp_r, 0.30), lerp(hp_l, hp_r, 0.12)])
    torso2 = np.vstack([
        lerp(sh_r, sh_l, 0.12), lerp(sh_r, sh_l, 0.30),
        lerp(hp_r, hp_l, 0.30), lerp(hp_r, hp_l, 0.12)])
    log(f"写真から採る（{args.photo.name}）:")
    shirt_lab = np.vstack([sample(photo, torso, log, "上着 左"),
                           sample(photo, torso2, log, "上着 右")])
    leg = np.vstack([lerp(hp_l, lm[L_KNEE], 0.25), lerp(hp_r, lm[R_KNEE], 0.25),
                     lerp(hp_r, lm[R_KNEE], 0.85), lerp(hp_l, lm[L_KNEE], 0.85)])
    pants_lab = sample(photo, leg, log, "ズボン")

    out = arr
    rep = {"model": str(args.model), "photo": str(args.photo),
           "texture": str(args.texture), "material": matname, "regions": {}}
    # **幾何の帯をそのままマスクにしてはいけない。** 帯は UV の島の一部しか
    # 覆わない。実測では脚の帯（高さ 0.07〜0.60）がジーンズの島の上半分だけを
    # 拾い、腰まわりのデニムが別の島として青く残った。
    #
    # 帯を**種**にして、UV の連結成分へ広げる。服はテクスチャ上で1枚の島に
    # なっているので、島単位で選べば帯の上下限を詰める必要が無くなる。
    from scipy import ndimage
    covered = np.asarray(rasterize(uv, tri, np.ones(len(tri), bool), size)) > 0
    lab_img, ncomp = ndimage.label(covered)
    log(f"UV の連結成分 {ncomp}個（幾何のある領域 {covered.mean()*100:.1f}%）")

    seeds = {n: np.asarray(rasterize(uv, tri, k, size)) > 0 for n, k in sel.items()}
    # 成分ごとに、どちらの種が多く乗っているかで割り当てる
    counts = {n: np.bincount(lab_img[s], minlength=ncomp + 1)
              for n, s in seeds.items()}
    owner = {}
    for ci in range(1, ncomp + 1):
        best = max(sel, key=lambda n: counts[n][ci])
        if counts[best][ci] > 0:
            owner.setdefault(best, []).append(ci)
    masks = {}
    for name in sel:
        comps = owner.get(name, [])
        m = np.isin(lab_img, comps)
        log(f"  {name}: 種 {int(seeds[name].sum()):,}画素 → 成分 {comps} "
            f"に広げて {m.mean()*100:.1f}%")
        im = Image.fromarray((m * 255).astype(np.uint8))
        if args.feather > 0:
            im = im.filter(ImageFilter.GaussianBlur(args.feather))
        masks[name] = np.asarray(im).astype(np.float64) / 255.0

    for name, tgt in (("shirt", shirt_lab), ("pants", pants_lab)):
        out, n, before = transfer(out, masks[name], tgt, args.strength, args.min_luma)
        rep["regions"][name] = {
            "triangles": int(sel[name].sum()), "pixels": n,
            "lab_before": before,
            "lab_target": [round(float(x), 2) for x in tgt.mean(0)]}
        log(f"  {name}: {n:,}画素  {before} → 狙い "
            f"{[round(float(x),2) for x in tgt.mean(0)]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out.astype(np.uint8)).save(args.out)
    if args.mask_out:
        vis = np.zeros(arr.shape, dtype=np.uint8)
        vis[:, :, 0] = (masks["shirt"] * 255).astype(np.uint8)
        vis[:, :, 2] = (masks["pants"] * 255).astype(np.uint8)
        Image.fromarray(vis).save(args.mask_out)
    rep["out"] = str(args.out)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log(f"→ {args.out}")


if __name__ == "__main__":
    main()
