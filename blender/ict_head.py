"""当て込んだ ICT-FaceKit から、VRM に載る頭部を組み立てる。

いまの頭は目・口・顎という部品を持たない一枚の殻で、まばたきも視線も口パクも
原理的に不可能だった（docs/05 §36〜38）。ICT-FaceKit は眼球・歯・舌を独立した
ジオメトリとして持ち、53個のブレンドシェイプが **ARKit の命名規約**で入っている。

KEHAI は `vrm.expressionManager` を使わず**生のモーフ名を直接**見る実装なので
（`utils/vrm-lipsync.ts` / `utils/vrm-expression.ts`）、名前さえ合わせれば
口パク・まばたき・表情がコード改修ゼロで動く。ICT は左右を `_L` / `_R` と書くが
ARKit は `Left` / `Right` なので、そこだけ直す。

`browInnerUp` と `cheekPuff` は ARKit では左右に分かれていないので、ICT の
`_L` と `_R` を足して1本にまとめる。

メッシュは**マテリアルごとに別オブジェクト**にする。理由は二つ:
  - `project_face.py` は `materials[0]` しか見ないので、顔だけに写真を投影するには
    顔が独立したオブジェクトである必要がある
  - 眼球・歯・睫毛には投影しないのが正しい（それぞれ固有の質感を保つ）

頂点の対応を確実にするため、OBJ は自前で読む。Blender の OBJ インポータは UV の
継ぎ目で頂点を割るので、ICT の 26,719 という並びが崩れて差分を当てられなくなる。

  blender --background --factory-startup --python ict_head.py -- \
      --fit <fit.json> --out <head.glb> --report <report.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

ICT_DIR = Path("vendor/ICT-FaceKit/FaceXModel")
N_IDENTITY = 100

# パーツごとの素の色。**投影しないパーツにはこれが最終的な色になる**ので、
# 眼球・歯・睫毛はここで決めておく。加えて、整列用レンダで ArcFace が顔を
# 見つけられるようにする役目もある — 既定の明るい灰色のままだと真っ白に飛んで
# 「顔を検出できません」で整列が落ちる（`project_face.py` の照明は TRELLIS の
# テクスチャ前提で較正されているため）。
BASE_COLOR = {
    "M_Face":          (0.66, 0.48, 0.40, 1.0),
    "M_BackHead":      (0.62, 0.45, 0.37, 1.0),
    "M_GumsTongue":    (0.45, 0.16, 0.17, 1.0),
    "M_Teeth":         (0.86, 0.83, 0.78, 1.0),
    "M_ScleraLeft":    (0.88, 0.87, 0.85, 1.0),
    "M_ScleraRight":   (0.88, 0.87, 0.85, 1.0),
    "M_IrisLeft":      (0.14, 0.09, 0.06, 1.0),
    "M_IrisRight":     (0.14, 0.09, 0.06, 1.0),
    "M_EyeLashes":     (0.02, 0.02, 0.02, 1.0),
    "M_EyeOcclusion":  (0.05, 0.04, 0.04, 1.0),
    "M_LacrimalFluid": (0.55, 0.42, 0.38, 1.0),
    "M_EyeBlend":      (0.60, 0.44, 0.38, 1.0),
}

# 既定で外すパーツ。**いずれもアルファテクスチャがあって初めて成立する**もので、
# 不透明のまま入れると害になる。実測: 睫毛を不透明の黒で入れたら目の上に黒い帯が
# 乗り、眼球が完全に隠れて ArcFace が 0.168 まで落ちた。
# ICT 付属の .mtl にはアルファの指定があるが、テクスチャ画像は同梱されていない。
SKIP_PARTS = ("M_EyeLashes", "M_EyeOcclusion", "M_EyeBlend", "M_LacrimalFluid")

# ARKit では左右に分かれていない表情。ICT の _L と _R を足して1本にする。
MERGE_LR = {"browInnerUp", "cheekPuff"}
# 差分がこれ以下の部位にはシェイプキーを作らない（歯に「まばたき」は要らない）。
# 単位は当て込み後の世界座標で、頭の高さがおよそ 0.35。1e-3 は実寸およそ 0.7mm で、
# それ未満の動きは見えないうえにモーフ1本ぶんの容量を食う。
MIN_DELTA = 1.0e-3


def parse_obj(path: Path):
    """v / vt / f と usemtl の区切りを、ICT の並びのまま読む。"""
    verts, uvs, groups, cur = [], [], {}, None
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            uvs.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("usemtl "):
            cur = line.split(maxsplit=1)[1].strip()
            groups.setdefault(cur, [])
        elif line.startswith("f "):
            face = []
            for tok in line.split()[1:]:
                p = tok.split("/")
                vi = int(p[0]) - 1
                ti = int(p[1]) - 1 if len(p) > 1 and p[1] else -1
                face.append((vi, ti))
            if cur is not None:
                groups[cur].append(face)
    return (np.array(verts, dtype=np.float64),
            np.array(uvs, dtype=np.float64) if uvs else None, groups)


def verts_of(path: Path) -> np.ndarray:
    return np.array([[float(x) for x in l.split()[1:4]]
                     for l in path.read_text(errors="ignore").splitlines()
                     if l.startswith("v ")], dtype=np.float64)


def arkit_name(ict: str) -> str:
    """ICT の表情名を ARKit の綴りへ。"""
    if ict.endswith("_L"):
        return ict[:-2] + "Left"
    if ict.endswith("_R"):
        return ict[:-2] + "Right"
    return ict


def build_object(name: str, mat_name: str, faces, V: np.ndarray, uvs, keep):
    """1マテリアル分のオブジェクトを作る。keep は使う頂点の元インデックス。"""
    remap = {old: i for i, old in enumerate(keep)}
    me = bpy.data.meshes.new(name)
    tri = [[remap[vi] for vi, _ in f] for f in faces]
    me.from_pydata([Vector(V[i]) for i in keep], [], tri)
    me.validate()
    if uvs is not None:
        uv = me.uv_layers.new(name="UVMap")
        li = 0
        for f in faces:
            for _, ti in f:
                if 0 <= ti < len(uvs):
                    uv.data[li].uv = (uvs[ti][0], uvs[ti][1])
                li += 1
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = BASE_COLOR.get(
            mat_name, (0.66, 0.48, 0.40, 1.0))
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.62
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = 0.0
    me.materials.append(mat)
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    return obj, remap


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="ict_fit.py が出した json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    ap.add_argument("--basis-cache", default="vendor/ICT-FaceKit/identity_basis.npz")
    ap.add_argument("--no-shape-keys", action="store_true")
    ap.add_argument("--keep-all-parts", action="store_true",
                    help="睫毛など、アルファテクスチャ前提のパーツも残す")
    args = ap.parse_args(argv)

    bpy.ops.wm.read_homefile(use_empty=True)

    fit = json.loads(Path(args.fit).read_text())
    alpha = np.array(fit["alpha"], dtype=np.float64)
    sim = fit["similarity"]
    s = float(sim["scale"])
    R = np.array(sim["rotation"], dtype=np.float64)
    t = np.array(sim["translation"], dtype=np.float64)

    neutral, uvs, groups = parse_obj(ICT_DIR / "generic_neutral_mesh.obj")

    # 識別を効かせた素体を作り、相似変換で対象の位置へ運ぶ
    cache = Path(args.basis_cache)
    if cache.exists():
        basis = np.load(cache)["basis"]
    else:
        basis = np.stack([verts_of(ICT_DIR / f"identity{k:03d}.obj") - neutral
                          for k in range(N_IDENTITY)])
    shaped = neutral + np.tensordot(alpha, basis, axes=(0, 0))
    V = (s * (R @ shaped.T)).T + t

    # **ここで Blender 座標へ直す。** ICT も当て込みの結果も glTF 規約（Y上・+Z前）
    # だが、Blender は Z 上。そのまま入れて既定の export_yup で書き出すと
    # Y上変換が二重に掛かり、**頭が寝た GLB** になる（実測で確認）。
    #   glTF (x, y, z) → Blender (x, -z, y)
    def to_blender(a):
        return np.stack([a[..., 0], -a[..., 2], a[..., 1]], axis=-1)
    V = to_blender(V)

    # 表情は識別と独立な線形差分。回転とスケールだけ掛ければよい（平行移動は不要）
    expr_files = sorted(p for p in ICT_DIR.glob("*.obj")
                        if not p.stem.startswith("identity")
                        and p.stem != "generic_neutral_mesh")
    deltas: dict[str, np.ndarray] = {}
    for p in expr_files:
        d = verts_of(p) - neutral
        deltas[p.stem] = to_blender((s * (R @ d.T)).T)

    # ARKit の綴りへ。左右が分かれていないものは足し合わせる。
    named: dict[str, np.ndarray] = {}
    for ict_name, d in deltas.items():
        base = ict_name[:-2] if ict_name.endswith(("_L", "_R")) else ict_name
        if base in MERGE_LR:
            named[base] = named.get(base, 0.0) + d
        else:
            named[arkit_name(ict_name)] = d

    skip = set() if args.keep_all_parts else set(SKIP_PARTS)
    made, dropped = [], []
    for mat_name, faces in groups.items():
        if mat_name in skip:
            dropped.append(mat_name)
            continue
        keep = np.unique(np.concatenate([[vi for vi, _ in f] for f in faces]))
        obj, remap = build_object(mat_name, mat_name, faces, V, uvs, keep)
        made.append(obj)
        if args.no_shape_keys:
            continue
        base_key = obj.shape_key_add(name="Basis", from_mix=False)
        n_keys = 0
        for key_name, d in sorted(named.items()):
            dk = d[keep]
            if float(np.abs(dk).max()) < MIN_DELTA:
                continue        # この部位は動かない（歯にまばたきは要らない）
            sk = obj.shape_key_add(name=key_name, from_mix=False)
            for old, new in remap.items():
                sk.data[new].co = base_key.data[new].co + Vector(d[old])
            n_keys += 1
        obj["shape_keys"] = n_keys

    bpy.ops.object.select_all(action="SELECT")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # **モーフ法線は書き出さない。** 既定では位置・法線・接線の3本が出るので容量が
    # 3倍になる（実測 77.7MB）。KEHAI は生のモーフ名で位置だけを駆動するので要らない。
    bpy.ops.export_scene.gltf(filepath=args.out, export_format="GLB",
                              use_selection=True, export_morph=True,
                              export_morph_normal=False,
                              export_morph_tangent=False,
                              export_apply=False)

    rep = {
        "fit": args.fit, "output": args.out,
        "alpha_norm": round(float(np.linalg.norm(alpha)), 4),
        "expressions_total": len(named), "dropped_parts": dropped,
        "objects": [{"name": o.name, "verts": len(o.data.vertices),
                     "shape_keys": o.get("shape_keys", 0)} for o in made],
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print("ICTHEAD_OK " + json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
