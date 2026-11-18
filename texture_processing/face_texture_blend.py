#!/usr/bin/env python3
"""投影テクスチャを**見えている面だけ**に限り、残りはテンプレートのまま保つ。

## 直している不具合

`blender/project_face.py` は写真をカメラから投影して焼くが、**遮蔽を見ていない**
（facing-dot の重みだけ）。眼球・歯・口内は前を向いているので判定を通り、
瞼と唇の裏にあるのに写真が焼き込まれる。実物の UV を見ると:

    テンプレの head テクスチャ左下 ＝ 眼球・歯・口内の UV 島
    投影後                       ＝ そこが写真の画素で塗り潰されている

これが「目が黒い塊に見える」「歯が変に見える」の原因。

## 手口

カメラから各三角形の重心へ光線を飛ばし、**最初に当たるのがその三角形か**を見る。
瞼に隠れた眼球、唇に隠れた歯はここで落ちる。通った三角形の UV を 2048² に
ラスタライズしてマスクにし、

    出力 = マスク × 焼いたテクスチャ ＋ (1 - マスク) × 元のテクスチャ

で合成する。**メッシュ・UV・リグ・モーフには触らない**（画像1枚の差し替えだけ）。

境界は羽根ぼかしを掛ける。切りっぱなしだと瞼の縁に段差が出る。

使い方:
    $HOME/miniconda3/envs/genesis-face/bin/python -m texture_processing.face_texture_blend \\
        --model jobs/x/body_deformed.glb --projected jobs/x/output/avatar.glb \\
        --material head --out blended.png --report jobs/x/scores/blend.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blender.vrm_finalize import read_glb                       # noqa: E402
from preprocessing.shape_transfer import read_accessor          # noqa: E402
from preprocessing.template_landmarks import face_primitive     # noqa: E402
from vrm.inject_texture import base_color_image                 # noqa: E402


def all_geometry(doc, blob):
    """全プリミティブを1つのメッシュに束ねる。**遮蔽の判定に全部要る**。

    三角形ごとのマテリアル名も返す。「何に隠れているか」で頭皮を切り分けるため ——
    髪のカードに隠れた肌が頭皮、瞼に隠れた面が眼球。
    """
    vs, fs, mats, off = [], [], [], 0
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            v = read_accessor(doc, blob, prim["attributes"]["POSITION"])
            if v is None or "indices" not in prim:
                continue
            acc = doc["accessors"][prim["indices"]]
            bv = doc["bufferViews"][acc["bufferView"]]
            o = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            dt = {5121: "<u1", 5123: "<u2", 5125: "<u4"}[acc["componentType"]]
            tri = np.frombuffer(blob, dtype=dt, count=acc["count"],
                                offset=o).reshape(-1, 3).astype(np.int64)
            m = prim.get("material")
            name = doc["materials"][m].get("name", "") if m is not None else ""
            vs.append(v)
            fs.append(tri + off)
            mats += [name] * len(tri)
            off += len(v)
    return np.vstack(vs), np.vstack(fs), np.array(mats)


def _named_primitive(doc, blob, material):
    """マテリアル名でプリミティブを引く（face_primitive と同型）。"""
    return face_primitive(doc, blob, material)


def uv_of(doc, blob, prim):
    a = prim["attributes"].get("TEXCOORD_0")
    return None if a is None else read_accessor(doc, blob, a)


def visible_triangles(verts, tris, mesh, mats, tol: float = 2e-3):
    """カメラ（+Z の無限遠）から見て**最初に当たる**三角形を選ぶ。

    重心へ光線を飛ばし、当たった点が重心より手前（+Z 側）に tol より大きく
    離れていたら、その三角形は何かの裏にある。何に隠れているかも返す。
    """
    c = verts[tris].mean(axis=1)
    zmax = float(verts[:, 2].max())
    origins = np.column_stack([c[:, 0], c[:, 1], np.full(len(c), zmax + 1.0)])
    dirs = np.tile([0.0, 0.0, -1.0], (len(c), 1))
    loc, idx_ray, idx_tri = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=False)
    hit_z = np.full(len(c), -np.inf)
    occluder = np.full(len(c), "", dtype=object)
    hit_z[idx_ray] = loc[:, 2]
    occluder[idx_ray] = mats[idx_tri]
    return hit_z <= c[:, 2] + tol, occluder


def main_shell(verts, tris):
    """head プリミティブの最大の連結成分に属する三角形。歯・舌・眼球は別シェル。"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(verts)
    i = np.r_[tris[:, 0], tris[:, 1], tris[:, 2]]
    j = np.r_[tris[:, 1], tris[:, 2], tris[:, 0]]
    _, lab = connected_components(
        coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)
    return lab[tris[:, 0]] == np.bincount(lab).argmax()


def rasterize(uv, tris, keep, size: int):
    """UV 三角形をマスク画像へ描く。

    **V を反転してはいけない。** glTF のテクスチャ座標は原点が画像の左上で、
    PIL の座標系と同じ。`1 - v` にすると上下が入れ替わる。実際に踏んだ ——
    頭皮マスクが金髪の UV 島から外れ、髪色を移しても最終レンダに何も反映
    されなかった。既知の頂点で裏を取った（テンプレの head テクスチャを直接引く）:

        頂点     UV            反転なし              反転あり
        上唇  (0.500,0.404)  [136, 44, 39] 唇の赤    [190,140,109] 肌
        頭頂  (0.305,0.033)  [188,151,100] 金髪      [ 80, 24,  6] 暗部
        鼻先  (0.500,0.338)  [208,152,123] 肌        [182,132,101] 肌
    """
    from PIL import Image, ImageDraw
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    for t in tris[keep]:
        p = [(float(uv[i, 0]) * size, float(uv[i, 1]) * size) for i in t]
        d.polygon(p, fill=255)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True,
                    help="**投影前**の glb（元テクスチャと幾何を取る）")
    ap.add_argument("--projected", type=Path, required=True, help="投影後の glb")
    ap.add_argument("--material", default="head")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mask-out", type=Path, help="マスクの確認画像")
    ap.add_argument("--scalp-mask-out", type=Path,
                    help="**髪のカードに隠れた肌＝頭皮**のマスク。髪色の移し先")
    ap.add_argument("--scalp-above", type=float, default=0.72,
                    help="頭の高さのこの比より上を頭皮の候補にする。"
                         "最終判定は「投影が届かなかった」との論理積")
    ap.add_argument("--hair-material", default="opacity",
                    help="髪カードとまつ毛が入っているマテリアル")
    ap.add_argument("--lash-mask-out", type=Path,
                    help="まつ毛だけのマスク（髪カードの補集合）")
    ap.add_argument("--hair-card-mask-out", type=Path,
                    help="**まつ毛を除いた**髪カードのマスク（opacity マテリアル用）")
    ap.add_argument("--landmarks", type=Path,
                    help="configs/landmarks/*.json。目の位置を知るのに使う")
    ap.add_argument("--eye-radius", type=float, default=0.030,
                    help="目のランドマークからこの距離内の opacity 三角形は"
                         "まつ毛とみなして髪カードから外す（m）")
    ap.add_argument("--scalp-always-above", type=float, default=0.82,
                    help="この比より上は投影の有無を問わず頭皮。生え際より上")
    ap.add_argument("--hair-detail-sd", type=float, default=3.0,
                    help="テンプレのテクスチャの局所標準偏差がこれ以上を髪とみなす。"
                         "描かれた生え際の曲線をここから取る。0 で無効")
    ap.add_argument("--nape-margin", type=float, default=0.04,
                    help="耳の後ろの髪（襟足）も頭皮に入れる。顎の輪郭の端"
                         "（68点の 0 と 16）よりこれだけ後ろの三角形が対象（m）。"
                         "0 で無効")
    ap.add_argument("--nape-min-blob", type=int, default=6000,
                    help="襟足マスクで残す連結成分の最小画素数（首の肌の粒を捨てる）")
    ap.add_argument("--unprojected-tol", type=int, default=3,
                    help="焼いた結果と元の差がこれ以下なら投影が届いていない")
    ap.add_argument("--feather", type=int, default=6, help="境界の羽根ぼかし（px）")
    ap.add_argument("--erode", type=int, default=2,
                    help="マスクを内側に縮める（px）。瞼の縁に写真が乗るのを防ぐ")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    import trimesh
    from PIL import Image, ImageFilter

    doc, blob = read_glb(args.model)
    mi, pi, verts, tris, matname = face_primitive(doc, blob, args.material)
    prim = doc["meshes"][mi]["primitives"][pi]
    uv = uv_of(doc, blob, prim)
    if uv is None:
        raise SystemExit(f"'{matname}' に TEXCOORD_0 が無い")

    av, af, amat = all_geometry(doc, blob)
    mesh = trimesh.Trimesh(vertices=av, faces=af, process=False)
    keep, occ = visible_triangles(verts, tris, mesh, amat)
    # **頭皮は「髪のカードの裏」ではなかった。** 遮蔽元の実測内訳:
    #
    #     head 1,306（後頭部が前頭部に隠れているだけ）
    #     opacity  13（髪カードの裏の肌はこれだけ）
    #     body     16
    #
    # 頭頂の頭皮は正面から見えているが、投影した写真の枠（顔に合わせた 1.35倍）の
    # 外なので元のテクスチャ（テンプレの金髪）が残る。だから幾何で切る:
    # 眉より上を頭皮とみなす。**相対高さで指定**するので、体型変形で
    # スケールが変わっても効く（テンプレ実測: (1.635-1.365)/0.374 = 0.722）。
    c = verts[tris].mean(axis=1)
    y0, y1 = float(verts[:, 1].min()), float(verts[:, 1].max())
    # 2段の閾値。
    #   scalp_always_above より上 … 生え際より確実に上。投影の有無を問わず頭皮
    #   scalp_above より上        … 眉より上。投影が届かなかった分だけ頭皮
    # 1段だけだと破綻する: 低い閾値では額の肌が暗くなり、「投影が届かなかった」
    # 条件だけを足すと**投影が額の肌を頭皮に塗った部分**（帯の 34%）が残る。
    hi = c[:, 1] > y0 + args.scalp_always_above * (y1 - y0)
    mid = c[:, 1] > y0 + args.scalp_above * (y1 - y0)
    scalp = mid

    # 元テクスチャと焼いたテクスチャ
    orig = Image.open(_extract(doc, blob, args.material)).convert("RGB")
    pdoc, pblob = read_glb(args.projected)
    bake = Image.open(_extract(pdoc, pblob, args.material)).convert("RGB")
    if bake.size != orig.size:
        orig = orig.resize(bake.size, Image.LANCZOS)
    size = bake.size[0]

    m = rasterize(uv, tris, keep, size)
    if args.erode > 0:
        m = m.filter(ImageFilter.MinFilter(args.erode * 2 + 1))
    if args.feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(args.feather))
    if args.mask_out:
        args.mask_out.parent.mkdir(parents=True, exist_ok=True)
        m.save(args.mask_out)
    if args.scalp_mask_out:
        # **幾何の高さだけで頭皮を切ってはいけない。** 眉の高さ（頭の 0.72）で
        # 切ると額の肌まで頭皮に入り、髪色を移すと額に暗い帯が出る。生え際の
        # 高さはテンプレートの幾何からは分からない。
        #
        # 正しい境界は**投影が上書きしなかった領域**。投影は顔（額を含む）を
        # 覆い、頭頂は法線が上向きで facing の重みが閾値（0.15）に届かないので
        # テンプレートの絵が残る。焼いた結果と元を引き算すれば、そこが取れる。
        arr_o = np.asarray(orig).astype(np.int16)
        arr_b = np.asarray(bake).astype(np.int16)
        untouched = (np.abs(arr_b - arr_o).max(axis=2) <= args.unprojected_tol)
        m_mid = (np.asarray(rasterize(uv, tris, mid, size)) > 0) & untouched
        m_hi = np.asarray(rasterize(uv, tris, hi, size)) > 0
        # **高さの閾値だけでは生え際にならない。** 平面で切ると水泳帽のような
        # 水平な境界が出る。テンプレートのテクスチャは**描かれた生え際**を
        # 持っているので、毛束の高周波成分でそれを拾う（実測の局所標準偏差:
        # 頬 0.83 / 額 1.76 / 鼻先 3.46 / 側頭 4.30 / 頭頂 13.45）。
        if args.hair_detail_sd > 0:
            from scipy import ndimage
            g = np.asarray(Image.fromarray(
                np.asarray(orig).astype(np.uint8)).convert("L")).astype(float)
            mu = ndimage.uniform_filter(g, 5)
            sd = np.sqrt(np.maximum(ndimage.uniform_filter(g * g, 5) - mu ** 2, 0))
            detail = sd > args.hair_detail_sd
            # 小さな穴と孤立点を潰す（毛束の隙間で穴だらけになる）
            detail = ndimage.binary_closing(detail, np.ones((9, 9)))
            detail = ndimage.binary_opening(detail, np.ones((5, 5)))
            m_hi = m_hi & detail
            m_mid = m_mid | ((np.asarray(rasterize(uv, tris, mid, size)) > 0) & detail)
        # **眉より上だけでは足りない。** テンプレートの髪は側頭から襟足まで
        # 描かれていて、UV では顔の島の左右の縁（後頭部）に広がる。眉の帯で
        # 切ると襟足が金髪のまま残り、**後ろから見ると生成した髪の下から
        # テンプレートの髪色が覗く**（KEHAI でユーザーが指摘）。
        #
        # 色では分けられない（テンプレの金髪と肌は Lab で中央値 11.6 しか離れて
        # おらず、肌自身の散らばり 21 より小さい）。幾何で分ける: 顎の輪郭の端
        # （68点の 0/16、耳の前）より `nape_margin` 後ろの三角形に限り、そこで
        # 「投影が届いていない ∧ 毛束の高周波がある」画素を襟足とみなす。
        # 耳は輪郭点の 2〜3cm 後ろまで伸びるので 4cm の余白で外す。歯・舌・
        # 眼球は head の別シェルなので最大の連結成分に限って除く。
        rep_nape = None
        if args.nape_margin > 0 and args.landmarks and args.hair_detail_sd > 0:
            lmj_ = json.loads(args.landmarks.read_text())["landmarks"]
            z_ear = float(verts[[lmj_[0]["vertex"], lmj_[16]["vertex"]], 2].min())
            behind = main_shell(verts, tris) & (c[:, 2] < z_ear - args.nape_margin)
            ext = (np.asarray(rasterize(uv, tris, behind, size)) > 0) & untouched & detail
            ext &= ~(m_mid | m_hi)
            # 毛束の分類はギザギザなので、襟足の縁は太めに閉じて丸める
            ext = ndimage.binary_closing(ext, np.ones((15, 15)))
            ext = ndimage.binary_opening(ext, np.ones((9, 9)))
            labels, nl = ndimage.label(ext)
            if nl:
                sizes = ndimage.sum(ext, labels, range(1, nl + 1))
                ext &= np.isin(labels, np.where(sizes >= args.nape_min_blob)[0] + 1)
            m_mid = m_mid | ext
            rep_nape = {"triangles": int(behind.sum()), "ear_line_z": round(z_ear, 4),
                        "coverage": round(float(ext.mean()), 4), "blobs": int(nl)}
            print(f"襟足: 耳の線 z={z_ear:.3f} より {args.nape_margin*1000:.0f}mm 後ろの"
                  f" {int(behind.sum())} 三角形から {100*ext.mean():.2f}% を追加")
        sm = Image.fromarray(((m_mid | m_hi) * 255).astype(np.uint8))
        if args.feather > 0:
            sm = sm.filter(ImageFilter.MinFilter(3))
            sm = sm.filter(ImageFilter.GaussianBlur(max(args.feather // 2, 1)))
        args.scalp_mask_out.parent.mkdir(parents=True, exist_ok=True)
        sm.save(args.scalp_mask_out)
        rep_scalp = float((np.asarray(sm) > 128).mean())

    a = np.asarray(m).astype(np.float32)[:, :, None] / 255.0
    out = np.asarray(bake).astype(np.float32) * a + \
        np.asarray(orig).astype(np.float32) * (1.0 - a)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out.astype(np.uint8)).save(args.out)

    # 髪カードのマスク。**まつ毛を外す。**
    #
    # `opacity` マテリアルは髪カードとまつ毛の両方を持つ。テクスチャを丸ごと
    # 髪色（本人はほぼ無彩色の黒髪）にすると、**まつ毛カードが目の上に太い
    # 黒い眼線として乗る**。マテリアルを色分けした診断では、まつ毛カードは
    # 細すぎて見落とした（後から head だけ / head+opacity の2枚を並べて特定）。
    rep_hair = None
    if args.hair_card_mask_out:
        if not args.landmarks:
            raise SystemExit("--hair-card-mask-out には --landmarks が必要")
        # **ランドマークの `xyz` を使ってはいけない。** あれはテンプレートの
        # 座標で、ここに来るモデルはラップと体型変形を経ている（身長 1.743 → 1.60m、
        # 倍率 0.918）。目の位置が 130mm ずれ、半径 30mm では 1枚も拾えず
        # 「まつ毛 0件」になった。**`vertex` は頂点番号なのでトポロジ不変**。
        # 変形後のモデルから引き直す。
        lmj = json.loads(args.landmarks.read_text())["landmarks"]
        eyes = verts[np.asarray([e["vertex"] for e in lmj[36:48]])]
        hv, hi, hverts, htris, _ = _named_primitive(doc, blob, args.hair_material)
        hc = hverts[htris].mean(axis=1)
        d = np.linalg.norm(hc[:, None, :] - eyes[None, :, :], axis=2).min(axis=1)
        keep_hair = d > args.eye_radius
        huv = uv_of(doc, blob, doc["meshes"][hv]["primitives"][hi])
        hm = rasterize(huv, htris, keep_hair, size)
        if args.feather > 0:
            hm = hm.filter(ImageFilter.GaussianBlur(max(args.feather // 2, 1)))
        args.hair_card_mask_out.parent.mkdir(parents=True, exist_ok=True)
        hm.save(args.hair_card_mask_out)
        if args.lash_mask_out:
            # **まつ毛だけのマスク。** 髪カードの補集合。ラップ後はテンプレの
            # 目の位置に合わせたカードが写真の目とずれる。写真にはまつ毛が
            # 写っているのでカードは冗長 —— アルファをゼロにして写真に任せる。
            lm_img = rasterize(huv, htris, ~keep_hair, size)
            if args.feather > 0:
                lm_img = lm_img.filter(ImageFilter.GaussianBlur(
                    max(args.feather // 2, 1)))
            args.lash_mask_out.parent.mkdir(parents=True, exist_ok=True)
            lm_img.save(args.lash_mask_out)
        rep_hair = {"triangles": int(len(htris)), "hair": int(keep_hair.sum()),
                    "lashes": int((~keep_hair).sum()),
                    "coverage": round(float((np.asarray(hm) > 128).mean()), 4)}
        print(f"髪カード {int(keep_hair.sum()):,} / まつ毛 {int((~keep_hair).sum()):,}"
              f"（目から {args.eye_radius*1000:.0f}mm 以内をまつ毛とみなす）")

    rep_scalp = locals().get("rep_scalp")
    rep = {"model": str(args.model), "scalp_coverage": rep_scalp,
           "nape": locals().get("rep_nape"),
           "hair_cards": rep_hair, "projected": str(args.projected),
           "material": matname, "triangles": int(len(tris)),
           "visible": int(keep.sum()), "occluded": int((~keep).sum()),
           "mask_coverage": round(float(a.mean()), 4),
           "scalp_triangles": int(scalp.sum()),
           "scalp_above": args.scalp_above,
           "occluders": {k: int(v) for k, v in zip(
               *np.unique(np.array([str(o) for o in occ])[~keep],
                          return_counts=True))},
           "size": size, "erode": args.erode, "feather": args.feather,
           "out": str(args.out)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"三角形 {len(tris):,} のうち 見える {int(keep.sum()):,} / "
          f"隠れている {int((~keep).sum()):,}（眼球・歯・口内）")
    print(f"マスクの被覆 {a.mean()*100:.1f}%  → {args.out}")


def _extract(doc, blob, material):
    """glb の中の画像を一時ファイルに出す（vrm.extract_texture と同じ引き方）。"""
    import tempfile
    idx = base_color_image(doc, material)
    img = doc["images"][idx]
    bv = doc["bufferViews"][img["bufferView"]]
    o = bv.get("byteOffset", 0)
    data = blob[o:o + bv["byteLength"]]
    ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(
        img.get("mimeType"), ".png")
    f = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    f.write(data)
    f.close()
    return f.name


if __name__ == "__main__":
    main()
