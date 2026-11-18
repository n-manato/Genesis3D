"""DiffLocks が出した毛束を、VRM に載るヘアカードへ変換する。

**VRM は毛束を持てない。** glTF にあるのはメッシュだけなので、ゲーム制作と同じ
「ヘアカード」— アルファテクスチャを貼った薄いリボン — に落とす必要がある。
HairLRM の著者も "strands can be converted into hair cards" と書いている通り、
これはどの手法を使っても避けられない別工程。既製の変換ツール（StrandKit /
Hair Tool）は有料の Blender アドオンなので自作する。

やること:
  1. 95,400本の毛束を、根元の位置と流れの向きでクラスタリングする
  2. クラスタごとに代表の毛筋を取り、それに沿ってリボンを張る
  3. リボンに UV を割り当て、毛束の房を描いたアルファテクスチャを貼る
  4. 対象の頭に合わせる（毛束は DiffLocks の正規化された頭部空間にある）

  blender --background --factory-startup --python hair_cards.py -- \
      --strands <difflocks_output_strands.npz> --target <head.glb> \
      --out <hair.glb> --report <report.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

# カードの本数。多いほど密に見えるが、VRM の容量と描画負荷に効く。
# 実写のボブなら 1500 本前後で隙間が目立たなくなる。
N_CARDS = 1600
# 1本のカードに使う節の数。少ないと毛先の曲がりが角張る。
N_SEG = 12
# カードの幅（対象の頭の大きさに対する比）。房の太さ。
CARD_WIDTH = 0.05        # 頭の横幅に対する比
# アルファテクスチャの大きさ
TEX = 512


def load_strands(path: Path) -> np.ndarray:
    d = np.load(path)
    P = d["positions"] if "positions" in d else d[list(d.keys())[0]]
    return np.asarray(P, dtype=np.float64)


def import_mesh(path: Path):
    before = set(bpy.data.objects)
    ext = path.suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise SystemExit(f"未対応の拡張子: {ext}")
    return [o for o in bpy.data.objects if o not in before and o.type == "MESH"]


def head_scalp(objs, frac: float = 0.42):
    """対象メッシュの上側 frac を「頭皮のある範囲」とみなして返す。"""
    pts = []
    for o in objs:
        for v in o.data.vertices:
            pts.append((o.matrix_world @ v.co)[:])
    P = np.array(pts)
    z = P[:, 2]
    hi, lo = z.max(), z.min()
    keep = P[z > hi - (hi - lo) * frac]
    return P, keep


def fit_similarity(roots: np.ndarray, scalp: np.ndarray) -> tuple[float, np.ndarray]:
    """毛束を対象の頭に合わせる。等方スケールと平行移動だけ。

    回転は入れない。毛束も対象も「Z が上・顔が前」で揃っているので、
    向きまで推定すると誤差で頭が傾くだけになる。

    **大きさは左右の幅で合わせる。** 重心からの平均距離で合わせると、毛束の
    根元は頭皮の帽子状に分布し、対象側は顔まで含む塊なので分布の形が違い、
    倍率が 2.2 倍にまで膨らんだ（実測で髪が頭を飲み込んだ）。左右の幅は
    どちらも「頭の横幅」を表すので直接比べられる。
    """
    ws = np.percentile(roots[:, 0], 98) - np.percentile(roots[:, 0], 2)
    wd = np.percentile(scalp[:, 0], 98) - np.percentile(scalp[:, 0], 2)
    s = wd / max(ws, 1e-9)
    # 上下は頭頂を合わせる。根元の最上点と頭皮の最上点が重なるようにする。
    cs = np.array([roots[:, 0].mean(), roots[:, 1].mean(), roots[:, 2].max()])
    cd = np.array([scalp[:, 0].mean(), scalp[:, 1].mean(), scalp[:, 2].max()])
    return s, cd - s * cs


def build_cards(P: np.ndarray, n_cards: int, n_seg: int, width: float,
                head_center: np.ndarray):
    """毛束をまとめてリボンの頂点・面・UV を作る。"""
    roots = P[:, 0, :]
    tips = P[:, -1, :]
    # 根元の位置と「根元→毛先」の向きでまとめる。向きを入れるのは、
    # 同じ場所から前と後ろに分かれる毛を別の房として扱うため。
    d = tips - roots
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    feat = np.concatenate([roots, d * 0.35], axis=1)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(feat), size=min(n_cards, len(feat)), replace=False)
    centers = feat[idx]
    # 1回だけ割り当て直す（k-means の1反復）。完全収束は要らない。
    for _ in range(3):
        dist = ((feat[:, None, :3] - centers[None, :, :3]) ** 2).sum(-1) \
             + ((feat[:, None, 3:] - centers[None, :, 3:]) ** 2).sum(-1)
        lab = dist.argmin(1)
        for c in range(len(centers)):
            m = lab == c
            if m.any():
                centers[c] = feat[m].mean(0)

    verts, faces, uvs = [], [], []
    t = np.linspace(0, 1, n_seg)
    made = 0
    for c in range(len(centers)):
        m = np.where(lab == c)[0]
        if len(m) < 3:
            continue
        # 代表の毛筋 = クラスタの平均。房の中心を通る線になる。
        curve = P[m].mean(0)
        # 節を等間隔に取り直す
        seg = np.stack([np.interp(t, np.linspace(0, 1, curve.shape[0]), curve[:, k])
                        for k in range(3)], axis=1)
        # 房の太さは根元で太く毛先で細く
        w = width * (1.0 - 0.65 * t)
        tang = np.gradient(seg, axis=0)
        tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
        # リボンの幅方向は「頭の中心から外へ向かう向き」と接線の外積で決める。
        # 毛筋自身の重心からのずれを使うと、毛が真下に垂れている房で向きが
        # 定まらず、カードが平たく開いてしまう（実測で髪が円盤になった）。
        radial = seg - head_center[None, :]
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
        side = np.cross(tang, radial)
        n_side = np.linalg.norm(side, axis=1, keepdims=True)

        # **頭頂では外積が退化する。** つむじの周りは毛が外向きにそのまま伸びるので
        # 接線と外向きが平行になり、幅方向が定まらずカードが潰れて見えなくなる
        # （実測で頭頂に地肌が透ける穴があいた）。別の軸で取り直す。
        bad = (n_side < 1e-3).ravel()
        if bad.any():
            alt = np.cross(tang[bad], np.array([0.0, 0.0, 1.0]))
            worse = np.linalg.norm(alt, axis=1) < 1e-3
            if worse.any():
                alt[worse] = np.cross(tang[bad][worse], np.array([1.0, 0.0, 0.0]))
            side[bad] = alt
            n_side = np.linalg.norm(side, axis=1, keepdims=True)
        side /= np.maximum(n_side, 1e-9)

        base = len(verts)
        for i in range(n_seg):
            verts.append(seg[i] - side[i] * w[i])
            verts.append(seg[i] + side[i] * w[i])
            uvs.append((0.0, t[i]))
            uvs.append((1.0, t[i]))
        for i in range(n_seg - 1):
            a = base + i * 2
            faces.append((a, a + 1, a + 3, a + 2))
        made += 1
    return np.array(verts), faces, uvs, made


def hair_texture(name: str, size: int, melanin: float, redness: float):
    """房を描いたアルファテクスチャを作る。

    DiffLocks が返す髪の材質（melanin / redness）から色を決める。
    毛の一本一本は縦線として描き、端に向けてアルファを落とす。
    """
    img = bpy.data.images.new(name, width=size, height=size, alpha=True)
    rng = np.random.default_rng(1)
    px = np.zeros((size, size, 4), dtype=np.float32)

    base = np.array([0.34, 0.22, 0.13]) * (1.0 - 0.55 * melanin) \
        + np.array([0.25, 0.09, 0.05]) * redness * 0.5
    n_hair = size // 2
    for _ in range(n_hair):
        x = rng.integers(0, size)
        w = max(1, int(rng.normal(2.2, 0.8)))
        a = float(np.clip(rng.normal(0.85, 0.15), 0.2, 1.0))
        shade = float(np.clip(rng.normal(1.0, 0.22), 0.45, 1.5))
        y0 = int(rng.uniform(0, size * 0.25))
        y1 = int(rng.uniform(size * 0.75, size))
        xs = slice(max(0, x - w), min(size, x + w))
        px[y0:y1, xs, :3] = np.clip(base * shade, 0, 1)
        px[y0:y1, xs, 3] = np.maximum(px[y0:y1, xs, 3], a)
    # 左右の端を落として、カードの継ぎ目を目立たなくする
    ramp = np.minimum(np.linspace(0, 1, size), np.linspace(1, 0, size))
    px[:, :, 3] *= np.clip(ramp * 4.0, 0, 1)[None, :]
    img.pixels = px.reshape(-1).tolist()
    return img


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--strands", required=True)
    ap.add_argument("--target", required=True, help="髪を載せる頭（位置合わせに使う）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    ap.add_argument("--material", help="DiffLocks の hair.json")
    ap.add_argument("--cards", type=int, default=N_CARDS)
    ap.add_argument("--segments", type=int, default=N_SEG)
    ap.add_argument("--width", type=float, default=CARD_WIDTH)
    ap.add_argument("--scalp-frac", type=float, default=0.42,
                    help=("対象の上から何割を『頭皮のある範囲』とみなすか。"
                          "頭だけのメッシュなら 0.42、全身なら 0.09 程度"))
    args = ap.parse_args(argv)

    bpy.ops.wm.read_homefile(use_empty=True)
    tgt = import_mesh(Path(args.target))
    if not tgt:
        raise SystemExit("対象メッシュがありません")
    bpy.ops.object.select_all(action="DESELECT")
    for o in tgt:
        o.select_set(True)
    bpy.context.view_layer.objects.active = tgt[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    P = load_strands(Path(args.strands))
    # 毛束は「Y が上」の空間。Blender は Z 上なので直す。
    P = np.stack([P[..., 0], -P[..., 2], P[..., 1]], axis=-1)

    _, scalp = head_scalp(tgt, args.scalp_frac)
    s, tvec = fit_similarity(P[:, 0, :], scalp)
    P = P * s + tvec

    # カードの幅は頭の横幅に対する比で決める。倍率をそのまま掛けると
    # 頭の大きさと無関係な値になる。
    head_w = float(np.percentile(scalp[:, 0], 98) - np.percentile(scalp[:, 0], 2))
    head_center = scalp.mean(0)
    verts, faces, uvs, made = build_cards(P, args.cards, args.segments,
                                          args.width * head_w, head_center)

    me = bpy.data.meshes.new("Hair")
    me.from_pydata([Vector(v) for v in verts], [], faces)
    me.validate()
    uv = me.uv_layers.new(name="UVMap")
    for i, loop in enumerate(me.loops):
        uv.data[i].uv = uvs[loop.vertex_index]

    mat_info = json.loads(Path(args.material).read_text()) if args.material else {}
    img = hair_texture("HairAlpha", TEX,
                       float(mat_info.get("melanin", 0.2)),
                       float(mat_info.get("redness", 0.5)))
    mat = bpy.data.materials.new("Hair")
    mat.use_nodes = True
    mat.blend_method = "HASHED"           # アルファのある薄板
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.links.new(bsdf.inputs["Base Color"], tex.outputs["Color"])
    nt.links.new(bsdf.inputs["Alpha"], tex.outputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = 0.35
    me.materials.append(mat)

    obj = bpy.data.objects.new("Hair", me)
    bpy.context.collection.objects.link(obj)
    obj.data.materials[0].use_backface_culling = False

    for o in tgt:
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=args.out, export_format="GLB",
                              use_selection=True, export_image_format="AUTO")

    rep = {"strands": int(P.shape[0]), "cards": made,
           "verts": len(verts), "faces": len(faces),
           "align_scale": round(float(s), 5),
           "align_translation": [round(float(x), 5) for x in tvec],
           "material": mat_info, "output": args.out}
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print("HAIRCARDS_OK " + json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
