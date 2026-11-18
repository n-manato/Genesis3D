"""生成された全身メッシュから**服だけ**を切り出して、アバターに別メッシュとして載せる。

## なぜ色で切るのか

髪は「肌表面からの符号付き距離」で切れた（`add_hair.py`）。位置合わせをラップが
高精度に与えてくれたから成立した。全身にはそれが無い。関節位置で合わせようとすると
**残差 116.6mm** で使えなかった —— MediaPipe の点を逆投影すると**服の表面**に
当たり、骨の内部位置とは数cm系統的にずれる（肩と腰で特に大きい）。

一方で**生成メッシュの服は色で明確に分かれる**。テンプレートのときは
ピンクのシャツ [188,104,116] と肌 [205,151,137] が近すぎて色で切れなかったが
（docs/09 §13.4）、本人の服は

    赤いブレザー [172, 21, 40]      肌 [205,151,137]
    黒いスラックス [ 32, 31, 35]

と離れている。距離ではなく色で切れば、位置合わせは「服が体を包む」程度の精度で足りる
（服は体より緩いので、差分を取るのではなく外側の層として載せるだけ）。

## 位置合わせ

外接箱。身長でスケールし、足元と左右中心を合わせる。Tポーズどうしで、しかも
テンプレートは既に本人の身長・肩幅・腰幅の比に変形済みなので、これで足りる。

  blender --background --factory-startup --python blender/add_garment.py -- \\
      --model jobs/x/body_deformed.glb --generated body_clean.glb \\
      --colors '[[172,21,40],[32,31,35]]' --out jobs/x/dressed.glb
"""
import argparse
import json
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_head import apply_transforms, import_mesh  # noqa: E402
from add_hair import drop_small_parts  # noqa: E402

MARKER = "ADD_GARMENT_OK"


def bbox(objs):
    """**`bound_box` を使わない。** あれはキャッシュで、頂点を直接書き換えた直後は
    古い値を返す。実測で RBF の変形場を掛けた後に外接箱が変形前のまま
    （z ±0.485）になり、頭の切り捨て閾値が誤って計算されて体の大半を捨てた。"""
    pts = [o.matrix_world @ v.co for o in objs for v in o.data.vertices]
    return (Vector(map(min, zip(*pts))), Vector(map(max, zip(*pts))))


def _bbox_align(g, body):
    from mathutils import Matrix
    bl, bh = bbox(body)
    gl, gh = bbox([g])
    sx = (bh.x - bl.x) / max(gh.x - gl.x, 1e-9)
    sy = (bh.y - bl.y) / max(gh.y - gl.y, 1e-9)
    sz = (bh.z - bl.z) / max(gh.z - gl.z, 1e-9)
    g.matrix_world = Matrix.Diagonal((sx, sy, sz, 1.0)) @ g.matrix_world
    apply_transforms([g])
    gl2, gh2 = bbox([g])
    g.matrix_world = Matrix.Translation(
        Vector(((bl.x + bh.x) / 2 - (gl2.x + gh2.x) / 2,
                (bl.y + bh.y) / 2 - (gl2.y + gh2.y) / 2,
                bl.z - gl2.z))) @ g.matrix_world
    apply_transforms([g])
    print(f"（外接箱で合わせた）スケール {sx:.3f}/{sy:.3f}/{sz:.3f}")



def sample_colors(obj):
    """各頂点の位置にあるテクスチャの色を取る。

    UV から画像を引く。頂点色ではなくテクスチャを見るのは、TRELLIS の出力が
    頂点色を持たないため。ループ（面の角）ごとに UV があるので、頂点には
    最初に見つかったループの色を入れる。
    """
    me = obj.data
    if not me.uv_layers:
        raise SystemExit(f"{obj.name}: UV が無い")
    # **最初に見つかった TEX_IMAGE を使ってはいけない。** 法線マップや ORM が
    # 混ざる。Principled BSDF の Base Color に繋がっている画像をたどる
    # （project_face.py が `materials[0]` 決め打ちで焼き先を間違えたのと同じ話）。
    def base_color_image(mat):
        if not mat or not mat.use_nodes:
            return None
        bsdf = next((n for n in mat.node_tree.nodes
                     if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            return None
        inp = bsdf.inputs.get("Base Color")
        if inp is None or not inp.is_linked:
            return None
        node = inp.links[0].from_node
        seen = set()
        while node is not None and node.name not in seen:
            seen.add(node.name)
            if node.type == "TEX_IMAGE":
                return node.image
            nxt = None
            for i in node.inputs:
                if i.is_linked:
                    nxt = i.links[0].from_node
                    break
            node = nxt
        return None

    img = next((im for m in me.materials
                if (im := base_color_image(m)) is not None), None)
    if img is None:
        raise SystemExit(f"{obj.name}: Base Color のテクスチャが見つからない "
                         f"（マテリアル {[m.name for m in me.materials if m]}）")
    w, h = img.size
    px = list(img.pixels)          # RGBA float、下から上
    nch = 4
    uvl = me.uv_layers[0].data
    cols = [None] * len(me.vertices)
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            if cols[vi] is not None:
                continue
            u, v = uvl[li].uv
            x = min(max(int(u * w), 0), w - 1)
            y = min(max(int(v * h), 0), h - 1)   # Blender の pixels は下が原点
            o = (y * w + x) * nch
            cols[vi] = (px[o] * 255.0, px[o + 1] * 255.0, px[o + 2] * 255.0)
    return [c or (0.0, 0.0, 0.0) for c in cols], (w, h), img.name


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="変形済みテンプレート（アーマチュア付き）")
    ap.add_argument("--generated", required=True, help="生成した全身")
    ap.add_argument("--out", required=True)
    ap.add_argument("--colors", required=True,
                    help="服の代表色の JSON 配列。`[r,g,b]` または `[r,g,b,許容]`。"
                         "例 [[172,21,40],[32,31,35],[202,191,185,30]]")
    ap.add_argument("--tol", type=float, default=70.0,
                    help="代表色からの RGB 距離の許容")
    ap.add_argument("--head-frac", type=float, default=0.86,
                    help="身長のこの比より上（頭）は服ではない")
    ap.add_argument("--target", type=int, default=40000)
    ap.add_argument("--close-iters", type=int, default=3)
    ap.add_argument("--open-iters", type=int, default=1)
    ap.add_argument("--min-part", type=int, default=800)
    ap.add_argument("--align", help="preprocessing.align_generated_body の出力。"
                                    "**外接箱より正確。** 無ければ外接箱に落ちる")
    ap.add_argument("--warp", help="align_generated_body の出力。**相似変換ではなく "
                                   "RBF の変形場**で運ぶ。体型の違いを吸収する")
    ap.add_argument("--warp-smooth", type=float, default=0.0,
                    help="RBF の平滑化。対応点にノイズがあるとき 0.01〜0.1")
    ap.add_argument("--shrink", type=float, default=0.010,
                    help="体表面の外側このぶんに沿わせる（m）。0 で沿わせない")
    ap.add_argument("--shrink-smooth", type=int, default=6,
                    help="沿わせた後の平滑の回数")
    ap.add_argument("--max-x", type=float, default=0.24,
                    help="この横位置より外は袖とみなして捨てる（m）。"
                         "**袖は載せられない** —— アバターは腕が下がったA字、"
                         "生成された全身はTポーズで姿勢が違う")
    ap.add_argument("--body-material", default="body")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    # 色ごとに許容を変えられる: [r,g,b] か [r,g,b,tol]。
    # **インナーの生成りは肌と近い**（[202,191,185] 対 [205,164,138] で距離 54.3）
    # ので、赤や黒と同じ許容にすると肌まで拾う。
    colors = []
    for c in json.loads(args.colors):
        if len(c) >= 4:
            colors.append(((float(c[0]), float(c[1]), float(c[2])), float(c[3])))
        else:
            colors.append(((float(c[0]), float(c[1]), float(c[2])), args.tol))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    body = import_mesh(Path(args.model))
    apply_transforms(body)
    arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if arm is None:
        raise SystemExit(f"{args.model} にアーマチュアが無い")

    gen = import_mesh(Path(args.generated))
    apply_transforms(gen)
    if len(gen) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in gen:
            o.select_set(True)
        bpy.context.view_layer.objects.active = gen[0]
        bpy.ops.object.join()
        gen = [gen[0]]
    g = gen[0]
    g.name = "Garment"

    # --- 位置合わせ ------------------------------------------------------
    from mathutils import Matrix as _M
    if args.warp:
        # **相似変換ではなく RBF の変形場で運ぶ。**
        #
        # 相似変換は体型（比率）の違いを吸収できない。実測で対応点8組の残差は
        # 34.2mm 残り、ジャケットの縁が体の縁に合わなかった。RBF（phi(r)=r ＋
        # 一次多項式）は対応点で残差ゼロ、間は滑らかに補間する。**投影と違って
        # 衣服と体の間隙（緩み）が潰れない**のが要点 —— シュリンクラップは
        # フィットしたが襟と前開きを潰した（docs/09 §23.1）。
        import numpy as np
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from preprocessing.shape_transfer import RBF
        pr = json.loads(Path(args.warp).read_text())["pairs"]
        c = np.asarray(pr["centers"], dtype=float)
        t = np.asarray(pr["targets"], dtype=float)
        rbf = RBF(c, t - c, smooth=args.warp_smooth)
        mw = g.matrix_world
        # Blender(Z上) ⇄ glTF(Y上)。対応点は glTF 座標で入っている。
        pts = np.array([[(mw @ v.co).x, (mw @ v.co).z, -(mw @ v.co).y]
                        for v in g.data.vertices])
        moved = pts + rbf.apply(pts)
        for v, q in zip(g.data.vertices, moved):
            v.co = Vector((float(q[0]), -float(q[2]), float(q[1])))
        g.matrix_world = _M.Identity(4)
        g.data.update()
        bl, bh = bbox(body)
        gl3, gh3 = bbox([g])
        s = (float(np.linalg.norm(t.max(0) - t.min(0))
                   / max(np.linalg.norm(c.max(0) - c.min(0)), 1e-9)),) * 3
        print(f"RBF の変形場で運んだ（対応点 {len(c)}組・残差 {rbf.residual:.2e}）")
        print(f"アバター  x[{bl.x:+.3f},{bh.x:+.3f}] z[{bl.z:+.3f},{bh.z:+.3f}]")
        print(f"服        x[{gl3.x:+.3f},{gh3.x:+.3f}] z[{gl3.z:+.3f},{gh3.z:+.3f}]")
    elif args.align:
        a = json.loads(Path(args.align).read_text())["align"]
        sc = float(a["scale"])
        R = _M([[float(x) for x in row] for row in a["rotation"]])
        T = Vector([float(x) for x in a["translation"]])
        # glTF(Y上) → Blender(Z上) の共役（add_hair.similarity と同じ話）
        C = _M(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
        lin = C @ (R * sc) @ C.transposed()
        g.matrix_world = (_M.Translation(C @ T) @ lin.to_4x4()) @ g.matrix_world
        apply_transforms([g])
        bl, bh = bbox(body)
        gl3, gh3 = bbox([g])
        s = (sc, sc, sc)
        print(f"相似変換で合わせた（RMS {json.loads(Path(args.align).read_text())['rms_mm']}mm）"
              f"  スケール {sc:.4f}")
        print(f"アバター  x[{bl.x:+.3f},{bh.x:+.3f}] z[{bl.z:+.3f},{bh.z:+.3f}]")
        print(f"服        x[{gl3.x:+.3f},{gh3.x:+.3f}] z[{gl3.z:+.3f},{gh3.z:+.3f}]")
    else:
        _bbox_align(g, body)
        bl, bh = bbox(body)
        gl3, gh3 = bbox([g])
        s = (0, 0, 0)

    #
    # 身長だけで等方にスケールすると腕幅が合わない（実測: 生成 0.814m 対
    # アバター 0.579m で 23cm はみ出し、袖が腕から浮く）。Tポーズどうしなので
    # 軸ごとの外接箱合わせで足りる。服は外側の層として載せるだけなので、
    # 差分を取る用途と違って等方性に拘る理由が無い。
    # --- 色で切る --------------------------------------------------------
    cols, size, imgname = sample_colors(g)
    zmax = gh3.z
    zmin = gl3.z
    head_z = zmin + args.head_frac * (zmax - zmin)
    mw = g.matrix_world
    me = g.data
    keep = []
    for i, v in enumerate(me.vertices):
        p = mw @ v.co
        if p.z > head_z or abs(p.x) > args.max_x:
            keep.append(False)
            continue
        c = cols[i]
        keep.append(any(sum((a - b) ** 2 for a, b in zip(c, k)) ** 0.5 < tol
                        for k, tol in colors))
    n0 = len(me.vertices)
    print(f"テクスチャ {imgname} {size}  服と判定 {sum(keep):,} / {n0:,}"
          f"（許容 {args.tol}、頭は z>{head_z:.3f} を除外）")

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()

    def dilate(sel):
        nxt = list(sel)
        for i, v in enumerate(bm.verts):
            if not sel[i] and any(sel[e.other_vert(v).index] for e in v.link_edges):
                nxt[i] = True
        return nxt

    def erode(sel):
        nxt = list(sel)
        for i, v in enumerate(bm.verts):
            if sel[i] and any(not sel[e.other_vert(v).index] for e in v.link_edges):
                nxt[i] = False
        return nxt

    for _ in range(args.close_iters):
        keep = dilate(keep)
    for _ in range(args.close_iters):
        keep = erode(keep)
    for _ in range(args.open_iters):
        keep = erode(keep)
    for _ in range(args.open_iters):
        keep = dilate(keep)
    bmesh.ops.delete(bm, geom=[v for i, v in enumerate(bm.verts) if not keep[i]],
                     context="VERTS")
    bm.to_mesh(me)
    bm.free()
    n1 = len(me.vertices)
    ngroups, nkept = drop_small_parts(g, min_verts=args.min_part)
    n2 = len(me.vertices)
    print(f"形態学処理の後 {n1:,} → 小片を除いて {n2:,}"
          f"（連結成分 {ngroups} のうち {nkept}個）")
    if n2 < 500:
        raise SystemExit(f"服として残ったのが {n2}頂点。--tol を上げること")

    if args.target and n2 > args.target:
        bpy.context.view_layer.objects.active = g
        m = g.modifiers.new("dec", "DECIMATE")
        m.decimate_type = "COLLAPSE"
        m.ratio = args.target / n2
        bpy.ops.object.modifier_apply(modifier=m.name)
        print(f"削減 {n2:,} → {len(me.vertices):,}")
    n3 = len(me.vertices)

    src = next((o for o in body
                if any(mm and args.body_material in mm.name
                       for mm in o.data.materials)), body[0])

    # --- 体に沿わせる（シュリンクラップ）---------------------------------
    #
    # **相似変換だけでは緩い衣服は置けない。** 3手試して全部失敗した
    # （外接箱・関節対応・表面対応。docs/09 §18.1 / §23）。体型と姿勢が違うので、
    # どんな相似変換でもジャケットの縁が体の縁に合わない。
    #
    # 発想を変えて、合わせるのをやめて**沿わせる**。体表面へ投影しつつ外側に
    # オフセットを置けば、体型が違っても必ずフィットする。代償は「緩み」で、
    # 襟・前開き・ラペルは**起伏として残る**。
    if args.shrink > 0:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = g
        sw = g.modifiers.new("shrink", "SHRINKWRAP")
        sw.target = src
        sw.wrap_method = "NEAREST_SURFACEPOINT"
        sw.wrap_mode = "OUTSIDE"
        sw.offset = args.shrink
        bpy.ops.object.modifier_apply(modifier=sw.name)
        if args.shrink_smooth > 0:
            sm = g.modifiers.new("sm", "SMOOTH")
            sm.factor = 0.5
            sm.iterations = args.shrink_smooth
            bpy.ops.object.modifier_apply(modifier=sm.name)
        gl4, gh4 = bbox([g])
        print(f"体に沿わせた（外側 {args.shrink*1000:.0f}mm・平滑 {args.shrink_smooth}回）"
              f"  x[{gl4.x:+.3f},{gh4.x:+.3f}] z[{gl4.z:+.3f},{gh4.z:+.3f}]")

    # --- テンプレートの体からウェイトを転送 -------------------------------
    #
    # 服は胴・腕・脚をまたぐので骨1本では足りない（髪と違うところ）。
    # テンプレートの手塗りウェイトを面上の最近傍で借りる。`autorig._transfer_weights`
    # と同じ方式（`layers_select_dst="NAME"` なので、先に同名の頂点グループを作る）。
    for vg in src.vertex_groups:
        if vg.name not in g.vertex_groups:
            g.vertex_groups.new(name=vg.name)
    bpy.ops.object.select_all(action="DESELECT")
    g.select_set(True)
    src.select_set(True)
    bpy.context.view_layer.objects.active = src
    bpy.ops.object.data_transfer(
        use_reverse_transfer=False, data_type="VGROUP_WEIGHTS",
        vert_mapping="POLYINTERP_NEAREST", layers_select_src="ALL",
        layers_select_dst="NAME", mix_mode="REPLACE")
    weighted = sum(1 for v in g.data.vertices if v.groups)
    print(f"ウェイト転送: {weighted:,}/{n3:,} 頂点（{src.name} から）")

    g.parent = arm
    g.matrix_parent_inverse = arm.matrix_world.inverted()
    am = g.modifiers.new("Armature", "ARMATURE")
    am.object = arm

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(args.out), export_format="GLB",
                              use_selection=True, export_morph=True, export_morph_normal=False,
                              export_skins=True, export_apply=False)
    out = {"model": args.model, "generated": args.generated, "out": args.out,
           "align_scale": [round(v, 6) for v in s],
           "colors": [[list(k), t] for k, t in colors], "tol": args.tol,
           "generated_vertices": n0, "after_morphology": n1,
           "after_small_parts": n2, "garment_vertices": n3,
           "weighted": weighted, "weight_source": src.name,
           "head_cut_z": round(head_z, 4), "max_x": args.max_x,
           "align_file": args.align, "warp_file": args.warp,
           "shrink": args.shrink}
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(MARKER + " " + json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
