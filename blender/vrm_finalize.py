#!/usr/bin/env python3
"""VRM 仕上げ — リグテンプレート由来の残骸を落とし、中身と一致した VRM にする。

パイプラインの VRM は VRoid のサンプル（`.env` の RIG_TEMPLATE_VRM）を被せて作る。
autorig.py はテンプレートのアーマチュアを取り込んでそのまま書き出すので、**テンプレート
側の VRM 拡張が丸ごと同乗してくる**。autorig の出力を見ただけでは気づかない:

  1. meta が pixiv 名義のまま。名前・作者・著作権表示だけでなく、**ライセンス条項も
     テンプレートの値**（誰でもアバターに使ってよい／商用可／再配布可／政治宗教利用可／
     過度な性的利用可）。実在の人物を素材にしたモデルに付けてよい表明ではない。
  2. サムネイルがテンプレートのアニメキャラの顔。VRM を一覧表示するツールでは、
     これが本人の顔として出る。
  3. humanoid の leftEye / rightEye が生きている。VRoid では眼球が独立メッシュなので
     正しいが、こちらは顔が単一メッシュで、目はテクスチャに描かれている。自動ウェイトで
     この2本が顔の下半分（実測 leftEye 12,267頂点・幅16cm。head 本体の 8,299頂点より多い）
     を掴んでいるため、ビューアが視線追従で回すと**顔が変形する**。
  4. VRMC_springBone が 22本（79ノード、うち69本が実ウェイトを持つ）。VRoid のズボン用
     二次ボーンで、こちらでは衣服が体のシルエットに焼き込まれているため、揺らすべき別
     メッシュが存在しない。物理を回すと**焼き込まれたズボンが布のように揺れる**。
  5. モーフターゲットが0個。表情は作れないが、これが原因で **KEHAI がアイドル動作の設定を
     丸ごとスキップする**（vrm-viewer.tsx の `if (faceMeshes.length)` の中に、口パクだけで
     なくアイドルクリップ探索・モーション一覧の取得・ピッカーの表示が入っている）。

Blender を経由せず glTF の JSON/BIN を直接書き換える。再インポート・再エクスポートすると
テクスチャが再エンコードされるなどして、実測で選んだ個体（正面 ArcFace 0.7082）が別物に
なりうるため。BIN は使われている bufferView だけを詰め直すので、テンプレート由来の未参照
画像もここで落ちる。

使い方:
    python3 blender/vrm_finalize.py --input in.vrm --out out.vrm --report r.json \
        --name "Subject A" --author "AI-CCORE" [--thumbnail head.png]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

# 顔メッシュに1本だけ足すダミーのモーフ。**顔を動かすためのものではない。**
# KEHAI は「モーフ属性を持つメッシュがあるか」でアイドル動作の設定に入るかどうかを
# 決めており（vrm-viewer.tsx の faceMeshes ゲート）、モーフが皆無だと
# morphTargetDictionary が undefined になってアイドルの配線ごと飛ぶ。
#
# 全ゼロだとビューア側の未使用モーフ削除に消されるので、頂点1個だけ 0.1mm 動かす。
# influence は誰も動かさないので常に 0。将来ここを本物の jawOpen に差し替えれば、
# KEHAI 側は無改修で口パクが動く（生のモーフ名を直接見る実装のため）。
STUB_MORPH_NAME = "_kehai_stub"
STUB_DELTA = 1.0e-4


def read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    magic, ver, total = struct.unpack("<III", raw[:12])
    if magic != GLB_MAGIC:
        raise SystemExit(f"GLB ではありません: {path}")
    doc, blob, off = None, b"", 12
    while off < total:
        ln, ty = struct.unpack("<II", raw[off:off + 8])
        data = raw[off + 8:off + 8 + ln]
        if ty == CHUNK_JSON:
            doc = json.loads(data.decode("utf-8"))
        elif ty == CHUNK_BIN:
            blob = data
        off += 8 + ln
    if doc is None:
        raise SystemExit("JSON チャンクがありません")
    return doc, blob


def write_glb(path: Path, doc: dict, blob: bytes) -> None:
    """JSON はスペース、BIN はゼロで4バイト境界に詰める（GLB 仕様）。"""
    js = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    bn = blob + b"\x00" * ((-len(blob)) % 4)
    out = bytearray()
    out += struct.pack("<III", GLB_MAGIC, 2, 12 + 8 + len(js) + 8 + len(bn))
    out += struct.pack("<II", len(js), CHUNK_JSON) + js
    out += struct.pack("<II", len(bn), CHUNK_BIN) + bn
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


# --------------------------------------------------------------------------- meta

def fix_meta(doc: dict, name: str, author: str, note: str) -> dict:
    """作者表示とライセンス条項を、実在人物を素材にしたモデルとして妥当な値にする。

    元の値はテンプレート（pixiv のサンプル）のもので、`avatarPermission: everyone` /
    `allowRedistribution: true` / `commercialUsage: corporation` など**全方位に許可**
    していた。KEHAI の characters/subject_a.yaml が書いている方針（stand-in であって
    なりすましではない・公開物は本人の承認が要る）と正面から矛盾する。
    """
    meta = doc["extensions"]["VRMC_vrm"]["meta"]
    before = {k: meta.get(k) for k in
              ("name", "authors", "copyrightInformation", "avatarPermission",
               "commercialUsage", "allowRedistribution", "modification")}
    meta["name"] = name
    meta["version"] = "1"
    meta["authors"] = [author]
    meta["copyrightInformation"] = f"(c) {author}"
    meta["licenseUrl"] = "https://vrm.dev/licenses/1.0/"
    meta["otherLicenseUrl"] = ""
    # 実在の人物の肖像を素材にしている。使ってよいのは作者だけ、商用不可、再配布不可、
    # 改変不可、クレジット必須、政治宗教・性的・暴力的・反社会的用途は不可。
    meta["avatarPermission"] = "onlyAuthor"
    meta["commercialUsage"] = "personalNonProfit"
    meta["creditNotation"] = "required"
    meta["allowRedistribution"] = False
    meta["modification"] = "prohibited"
    meta["allowExcessivelySexualUsage"] = False
    meta["allowExcessivelyViolentUsage"] = False
    meta["allowPoliticalOrReligiousUsage"] = False
    meta["allowAntisocialOrHateUsage"] = False
    meta["thirdPartyLicenses"] = note
    return {"before": before, "after": {k: meta.get(k) for k in before}}


def set_thumbnail(doc: dict, blob: bytes, png: Path | None) -> tuple[bytes, dict]:
    """サムネイルを差し替える。--thumbnail を渡さない場合は参照を外すだけ。

    テンプレートのサムネイル（アニメキャラの顔・1.2MB）が残っていると、VRM を一覧する
    ツールで別人の顔が本人として表示される。
    """
    meta = doc["extensions"]["VRMC_vrm"]["meta"]
    old = meta.get("thumbnailImage")
    if png is None:
        meta.pop("thumbnailImage", None)
        return blob, {"action": "参照を外した", "old_image": old}
    data = png.read_bytes()
    bv = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
    blob = blob + data + b"\x00" * ((-len(data)) % 4)
    doc["bufferViews"].append(bv)
    doc.setdefault("images", []).append(
        {"name": "Thumbnail", "mimeType": "image/png",
         "bufferView": len(doc["bufferViews"]) - 1})
    meta["thumbnailImage"] = len(doc["images"]) - 1
    return blob, {"action": "差し替えた", "old_image": old,
                  "new_image": meta["thumbnailImage"], "bytes": len(data)}


# ------------------------------------------------------------------ humanoid / spring

def drop_eye_bones(doc: dict) -> dict:
    """humanoid から leftEye / rightEye を外す。

    ボーンとウェイトはメッシュに残すが、humanoid に載っていなければビューアは回さない
    （three-vrm は目ボーンの有無を `if (leftEye)` で守っている）。首と胴による視線追従は
    そのまま効く。目がテクスチャに描かれている以上、眼球ボーンは原理的に使えない。
    """
    hb = doc["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"]
    removed = [b for b in ("leftEye", "rightEye") if hb.pop(b, None) is not None]
    return {"removed": removed, "remaining": len(hb)}


def drop_spring_bones(doc: dict) -> dict:
    """VRMC_springBone を拡張本体と extensionsUsed の両方から消す。"""
    ext = doc.get("extensions", {})
    sb = ext.pop("VRMC_springBone", None)
    used = doc.get("extensionsUsed", [])
    if "VRMC_springBone" in used:
        used.remove("VRMC_springBone")
    # ノード側にぶら下がる VRMC_springBone_extended_collider などの残骸も落とす
    for n in doc.get("nodes", []):
        (n.get("extensions") or {}).pop("VRMC_springBone", None)
    if sb is None:
        return {"removed": False}
    return {"removed": True, "springs": len(sb.get("springs") or []),
            "colliders": len(sb.get("colliders") or [])}


def add_stub_morph(doc: dict, blob: bytes) -> tuple[bytes, dict]:
    """全プリミティブに、ほぼ変位ゼロのモーフを1本足す（理由は STUB_MORPH_NAME 参照）。

    実体は sparse accessor。bufferView を持たない accessor は glTF 仕様でゼロ初期化
    されるので、非ゼロにする1頂点ぶん（インデックス4B＋値12B）しか BIN が増えない。
    """
    mesh = doc["meshes"][0]
    if any(p.get("targets") for p in mesh["primitives"]):
        return blob, {"added": False, "reason": "既にモーフがある"}

    base = len(blob)
    payload = struct.pack("<I", 0) + struct.pack("<3f", STUB_DELTA, 0.0, 0.0)
    blob = blob + payload + b"\x00" * ((-len(payload)) % 4)
    bvs = doc["bufferViews"]
    bvs.append({"buffer": 0, "byteOffset": base, "byteLength": 4})       # indices
    bvs.append({"buffer": 0, "byteOffset": base + 4, "byteLength": 12})  # values
    bv_idx, bv_val = len(bvs) - 2, len(bvs) - 1

    for prim in mesh["primitives"]:
        count = doc["accessors"][prim["attributes"]["POSITION"]]["count"]
        doc["accessors"].append({
            "type": "VEC3", "componentType": 5126, "count": count,
            "min": [0.0, 0.0, 0.0], "max": [STUB_DELTA, 0.0, 0.0],
            "sparse": {
                "count": 1,
                "indices": {"bufferView": bv_idx, "byteOffset": 0,
                            "componentType": 5125},
                "values": {"bufferView": bv_val, "byteOffset": 0},
            },
        })
        prim["targets"] = [{"POSITION": len(doc["accessors"]) - 1}]

    mesh["weights"] = [0.0]
    mesh.setdefault("extras", {})["targetNames"] = [STUB_MORPH_NAME]
    return blob, {"added": True, "name": STUB_MORPH_NAME,
                  "primitives": len(mesh["primitives"])}


# --------------------------------------------------------------------------- repack

def drop_morph_normals(doc: dict) -> dict:
    """モーフターゲットから法線・接線を外す。

    VRM アドオンの書き出しは位置・法線を両方持たせるので、53個の表情で
    容量が倍以上になる（実測 243MB）。KEHAI は生のモーフ名で**位置だけ**を
    駆動するので、法線は使われないまま容量を食う。
    """
    n = 0
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            for tgt in prim.get("targets", []) or []:
                for attr in ("NORMAL", "TANGENT"):
                    if tgt.pop(attr, None) is not None:
                        n += 1
    return {"removed": n}


def gc_accessors(doc: dict) -> dict:
    """どこからも参照されない accessor を捨てる。

    モーフの法線を外しただけでは容量が減らない。accessor 自体は配列に残り、
    後段の詰め直しが「まだ参照されている bufferView」と見なしてデータを持ち越す
    （実測 154MB → 197MB と逆に増えた）。accessor を消して番号を振り直す。
    """
    used: set[int] = set()

    def mark(v):
        if isinstance(v, int):
            used.add(v)

    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            for a in prim.get("attributes", {}).values():
                mark(a)
            mark(prim.get("indices"))
            for tgt in prim.get("targets", []) or []:
                for a in tgt.values():
                    mark(a)
    for sk in doc.get("skins", []):
        mark(sk.get("inverseBindMatrices"))
    for anim in doc.get("animations", []):
        for smp in anim.get("samplers", []):
            mark(smp.get("input")); mark(smp.get("output"))

    keep = sorted(used)
    remap = {old: i for i, old in enumerate(keep)}
    before = len(doc.get("accessors", []))
    doc["accessors"] = [doc["accessors"][i] for i in keep]

    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            prim["attributes"] = {k: remap[v] for k, v in prim["attributes"].items()}
            if prim.get("indices") is not None:
                prim["indices"] = remap[prim["indices"]]
            for tgt in prim.get("targets", []) or []:
                for k in list(tgt):
                    tgt[k] = remap[tgt[k]]
    for sk in doc.get("skins", []):
        if sk.get("inverseBindMatrices") is not None:
            sk["inverseBindMatrices"] = remap[sk["inverseBindMatrices"]]
    for anim in doc.get("animations", []):
        for smp in anim.get("samplers", []):
            smp["input"] = remap[smp["input"]]; smp["output"] = remap[smp["output"]]
    return {"before": before, "after": len(doc["accessors"])}


def sparsify_morphs(doc: dict, blob: bytes, eps: float = 1e-4) -> tuple[bytes, dict]:
    """モーフの差分を sparse accessor に変換する。

    表情は顔の頂点しか動かさないのに、頭と体を結合すると **20万頂点ぶんの差分**を
    53本ぶん持つことになる（実測 151.5MB、うち 127MB がほぼゼロの数値）。

    glTF の sparse accessor は「ゼロでない要素だけ」を持てる。`bufferView` を
    外した accessor は仕様上ゼロ初期化されるので、動く頂点だけを列挙すれば足りる。

    密のままの方が小さい場合（顔だけのメッシュなど）は変換しない。
    """
    import struct as _s

    def read(acc_i):
        a = doc["accessors"][acc_i]
        if a.get("bufferView") is None:
            return None
        bv = doc["bufferViews"][a["bufferView"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        if a["componentType"] != 5126:
            return None
        n = a["count"] * nc
        return _s.unpack_from(f"<{n}f", blob, off), nc, a

    out = bytearray(blob)
    converted = saved = 0
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            for tgt in prim.get("targets", []) or []:
                for attr, acc_i in list(tgt.items()):
                    got = read(acc_i)
                    if got is None:
                        continue
                    vals, nc, a = got
                    cnt = a["count"]
                    nz = [i for i in range(cnt)
                          if any(abs(vals[i * nc + k]) > eps for k in range(nc))]
                    dense = cnt * nc * 4
                    sparse = len(nz) * (4 + nc * 4)
                    # **まったく動かないモーフこそ削る価値がある。** メッシュを
                    # 結合すると体のプリミティブにも 53本のターゲットが付くが、
                    # 顔の表情で体は動かないので中身は全部ゼロ。これを密のまま
                    # 残すと 1本 2.4MB × 数十本で 62MB を食う（実測）。
                    # glTF は「同じメッシュの全プリミティブが同数のターゲットを
                    # 持つこと」を要求するので、消さずに最小の sparse にする。
                    if not nz:
                        nz = [0]
                        sparse = 4 + nc * 4
                    # 閾値の根拠: 表情の実際の変位は 1e-3〜1e-2 のオーダー。
                    # 1e-4 未満は書き出しの往復で入る数値誤差。
                    if sparse >= dense * 0.95:
                        continue
                    ib = _s.pack(f"<{len(nz)}I", *nz)
                    vb = _s.pack(f"<{len(nz) * nc}f",
                                 *[0.0 if len(nz) == 1 and not any(
                                       abs(vals[nz[0] * nc + q]) > eps for q in range(nc))
                                   else vals[i * nc + k]
                                   for i in nz for k in range(nc)])
                    base = len(out)
                    out += ib
                    voff = len(out)
                    out += vb
                    doc["bufferViews"].append(
                        {"buffer": 0, "byteOffset": base, "byteLength": len(ib)})
                    doc["bufferViews"].append(
                        {"buffer": 0, "byteOffset": voff, "byteLength": len(vb)})
                    a.pop("bufferView", None)
                    a.pop("byteOffset", None)
                    a["sparse"] = {
                        "count": len(nz),
                        "indices": {"bufferView": len(doc["bufferViews"]) - 2,
                                    "byteOffset": 0, "componentType": 5125},
                        "values": {"bufferView": len(doc["bufferViews"]) - 1,
                                   "byteOffset": 0},
                    }
                    converted += 1
                    saved += dense - sparse
    return bytes(out), {"converted": converted,
                        "saved_bytes": saved,
                        "saved_mb": round(saved / 1e6, 1)}


def repack(doc: dict, blob: bytes) -> tuple[bytes, dict]:
    """参照されている bufferView だけを詰め直し、未参照の画像とデータを落とす。

    テンプレート由来の未参照テクスチャや、差し替え前のサムネイルがここで消える。
    bufferView / image の番号が変わるので、それらを指している全箇所を張り替える。
    """
    # --- 未参照の画像を落とす（textures と meta.thumbnailImage から辿る）
    used_img: set[int] = set()
    for t in doc.get("textures", []):
        for src in (t.get("source"),
                    (t.get("extensions") or {}).get("EXT_texture_webp", {}).get("source")):
            if src is not None:
                used_img.add(src)
    vext = (doc.get("extensions") or {}).get("VRMC_vrm") or {}
    thumb = (vext.get("meta") or {}).get("thumbnailImage")
    if thumb is not None:
        used_img.add(thumb)
    images = doc.get("images", [])
    keep_img = [i for i in range(len(images)) if i in used_img]
    img_map = {old: new for new, old in enumerate(keep_img)}
    dropped_img = [images[i].get("name") for i in range(len(images)) if i not in used_img]
    doc["images"] = [images[i] for i in keep_img]
    for t in doc.get("textures", []):
        if t.get("source") is not None:
            t["source"] = img_map[t["source"]]
        webp = (t.get("extensions") or {}).get("EXT_texture_webp")
        if webp and webp.get("source") is not None:
            webp["source"] = img_map[webp["source"]]
    if thumb is not None:
        vext["meta"]["thumbnailImage"] = img_map[thumb]

    # --- 生き残っている bufferView を集める
    def bv_refs():
        for a in doc.get("accessors", []):
            if a.get("bufferView") is not None:
                yield a, "bufferView"
            sp = a.get("sparse")
            if sp:
                yield sp["indices"], "bufferView"
                yield sp["values"], "bufferView"
        for im in doc.get("images", []):
            if im.get("bufferView") is not None:
                yield im, "bufferView"

    used_bv = sorted({holder[key] for holder, key in bv_refs()})
    old_bvs = doc["bufferViews"]
    new_blob = bytearray()
    bv_map: dict[int, int] = {}
    new_bvs = []
    for old in used_bv:
        bv = dict(old_bvs[old])
        start = bv.get("byteOffset", 0)
        data = blob[start:start + bv["byteLength"]]
        pad = (-len(new_blob)) % 4
        new_blob += b"\x00" * pad
        bv["byteOffset"] = len(new_blob)
        new_blob += data
        bv_map[old] = len(new_bvs)
        new_bvs.append(bv)
    for holder, key in bv_refs():
        holder[key] = bv_map[holder[key]]
    doc["bufferViews"] = new_bvs
    doc["buffers"] = [{"byteLength": len(new_blob)}]
    return bytes(new_blob), {
        "bufferViews": f"{len(old_bvs)} → {len(new_bvs)}",
        "bytes": f"{len(blob)} → {len(new_blob)}",
        "dropped_images": dropped_img,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    ap.add_argument("--name", default="Genesis3D Character")
    ap.add_argument("--author", default="Genesis3D")
    ap.add_argument("--thumbnail", help="差し替えるサムネイル PNG（省略時は参照を外す）")
    ap.add_argument("--note", default=(
        "Generated by Genesis3D from a photograph. Rigged with a VRoid humanoid "
        "template (VRM1_Constraint_Twist_Sample, (c) 2022 pixiv Inc., "
        "https://vrm.dev/licenses/1.0/): skeleton only, no template mesh, "
        "textures or blendshapes remain."))
    ap.add_argument("--keep-eye-bones", action="store_true")
    ap.add_argument("--keep-spring-bones", action="store_true")
    ap.add_argument("--no-stub-morph", action="store_true")
    ap.add_argument("--keep-morph-normals", action="store_true",
                    help="モーフの法線を残す（既定は外す。KEHAI は位置しか使わない）")
    ap.add_argument("--no-sparse-morphs", action="store_true",
                    help="モーフ差分を sparse にしない（既定は変換して容量を削る）")
    args = ap.parse_args()

    src = Path(args.input)
    doc, blob = read_glb(src)
    report: dict = {"input": str(src), "output": args.out,
                    "bytes_before": src.stat().st_size}

    # VRM 拡張が無い素の GLB でも呼べるようにする。リギング前のメッシュに対して
    # sparse 変換だけ掛けたい場面があるため（結合直後は 151MB になる）。
    is_vrm = "VRMC_vrm" in (doc.get("extensions") or {})
    report["is_vrm"] = is_vrm
    if is_vrm:
        report["meta"] = fix_meta(doc, args.name, args.author, args.note)
        blob, report["thumbnail"] = set_thumbnail(
            doc, blob, Path(args.thumbnail) if args.thumbnail else None)
        if not args.keep_eye_bones:
            report["eye_bones"] = drop_eye_bones(doc)
        if not args.keep_spring_bones:
            report["spring_bones"] = drop_spring_bones(doc)
    if not args.no_stub_morph:
        blob, report["stub_morph"] = add_stub_morph(doc, blob)
    if not args.keep_morph_normals:
        report["morph_normals"] = drop_morph_normals(doc)
    if not args.keep_morph_normals:
        report["accessor_gc"] = gc_accessors(doc)
    if not args.no_sparse_morphs:
        blob, report["sparse_morphs"] = sparsify_morphs(doc, blob)
    blob, report["repack"] = repack(doc, blob)

    write_glb(Path(args.out), doc, blob)
    report["bytes_after"] = Path(args.out).stat().st_size
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("VRM_FINALIZE " + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
