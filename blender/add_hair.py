"""生成メッシュから**髪だけ**を切り出して、ラップ済みアバターに別メッシュとして載せる。

## なぜこれで解けるのか

ラップ方式（docs/09）は生成メッシュを「測るためだけ」に回して捨てていた。捨てた
ものの中に**本人の髪がジオメトリとして入っている**。テンプレート（Rocketbox）は
髪のボリュームを持たず、髪は頭皮テクスチャに描いた絵なので、色は寄せられても
髪型は寄せられなかった（docs/09 §13.2）。

**髪は固定トポロジを必要としない。** ラップが必要だったのは、

  - 顔 … 表情モーフ（175個）が頂点番号で定義されている
  - 体 … 手塗りウェイトが頂点番号で定義されている

からで、髪に要るのは骨1本（またはスプリングチェーン）だけ。だから髪だけは
生成メッシュのまま載せてよい。**固定トポロジの制約は顔と体にしか要らない。**

## 何がジオメトリで、何がテクスチャか（実測で分けた）

生成メッシュの頂点をアバター表面からの符号付き距離で見ると、**頭頂では髪の表面が
頭皮とほぼ一致していた**（上位15%の高さで中央値 −1.3mm、外側は 44.5% だけ）。
ラップ後の頭皮はテンプレート由来で、そこに髪が描いてあるぶん元から髪の体積を
持っているため。

    領域            頂点     距離の分位(mm) 5/25/50/75/95
    前(顔)       205,683    -47.7  -11.8   +0.5  +12.4  +41.9
    上(頭頂)     163,673    -47.3  -11.7   -0.6  +12.9  +44.5
    後           221,103    -40.9   -1.7  +13.4  +28.1  +56.1
    側           175,085    -46.5   -3.6   +6.2  +18.9  +46.0

→ **頭頂に足すべきジオメトリは無い。色だけ**（docs/09 §13.2 の色転写が担う）。
→ **ジオメトリが足すのは側頭と後頭部のシルエット**。テンプレートは髪を後ろで
   束ねた形なので、ボブのように張り出す髪型はここでしか表現できない。

## 切り出し方

ラップが求めた相似変換（`wrap.json` の `align`）で生成メッシュをテンプレート空間へ
運ぶと、肌の表面はラップ済みの頭とほぼ一致する（顔面の残差 1.961mm）。髪はその
**外側**にある。だから符号付き距離で切れる:

    髪 = 肌表面から外側へ `--outside` 以上離れた生成メッシュの頂点

色や材質で判定しない。金髪と肌が色で分けられなかったのと同じ理由で、生成メッシュの
髪と肌も色では分けられない。

## 結合しない

Blender で結合するとシェイプキーの扱いが絡んでモーフ175個を壊しうる。アーマチュアに
別メッシュとして紐づけるだけにする。glTF は複数のスキンメッシュを問題なく持てる。

  blender --background --factory-startup --python blender/add_hair.py -- \\
      --model jobs/x/wrapped.glb --generated jobs/x/head_clean.glb \\
      --wrap-report jobs/x/scores/wrap.json --out jobs/x/haired.glb
"""
import argparse
import json
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_head import apply_transforms, import_mesh  # noqa: E402

MARKER = "ADD_HAIR_OK"
# glTF(Y上) → Blender(Z上) の変換。インポータが掛けるのと同じもの。
GLTF_TO_BLENDER = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))


def scene_reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def find_armature():
    return next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)


def similarity(rep: dict) -> Matrix:
    """`wrap.json` の相似変換を Blender 空間の 4x4 にする。

    保存されているのは glTF 座標での  x_t = s (R x) + t。Blender は C = GLTF_TO_BLENDER
    で回した空間なので、**共役を取る**必要がある:

        x_t^b = C x_t = C(sR)x + Ct = (C sR C^T)(C x) + C t

    ここを間違えると髪が頭から離れた場所に出る。ログで外接を突き合わせて確認する。
    """
    a = rep["align"]
    s = float(a["scale"])
    r = Matrix([[float(x) for x in row] for row in a["rotation"]])
    t = Vector([float(x) for x in a["translation"]])
    c = GLTF_TO_BLENDER
    lin = c @ (r * s) @ c.transposed()
    return Matrix.Translation(c @ t) @ lin.to_4x4()


def signed_distances(objs, pts, exclude=None):
    """pts（世界座標）から objs の表面までの符号付き距離と、最近点の法線。

    外側が正。法線は「その点がアバターのどこの上にあるか」を表すので、
    顔の上か・頭頂や後頭部の上かを分けるのに使う。

    `exclude`: (材質名の部分一致, 判定関数(世界座標の重心)→bool)。該当する面は
    表面から外す。**テンプレの耳**を外すのに使う —— 耳は本人の耳より外に突き出て
    いるので、写真で耳を覆っている髪の点が「テンプレの耳の中」になって捨てられ、
    髪に耳の穴が空く（§28.9）。
    """
    verts, faces, off = [], [], 0
    skipped = 0
    for o in objs:
        mw = o.matrix_world
        me = o.data
        names = [m.name if m else "" for m in me.materials]
        verts += [mw @ v.co for v in me.vertices]
        for p in me.polygons:
            if exclude is not None and exclude[0] in names[p.material_index]:
                c = sum((mw @ me.vertices[i].co for i in p.vertices), Vector()) / len(p.vertices)
                if exclude[1](c):
                    skipped += 1
                    continue
            idx = [i + off for i in p.vertices]
            for k in range(1, len(idx) - 1):
                faces.append((idx[0], idx[k], idx[k + 1]))
        off += len(me.vertices)
    if exclude is not None:
        print(f"  距離の表面から除外した面 {skipped:,}（耳）")
    tree = BVHTree.FromPolygons(verts, faces, all_triangles=True)
    out, nrm = [], []
    for p in pts:
        loc, nor, _, _ = tree.find_nearest(p)
        if loc is None:
            out.append(1e9)
            nrm.append(Vector((0, 0, 1)))
        else:
            d = (p - loc)
            out.append(d.length * (1.0 if d.dot(nor) >= 0 else -1.0))
            nrm.append(nor)
    return out, nrm


def strip_template_hair(objs, face_material: str, margin: float = 0.01) -> dict:
    """テンプレートの髪シェル（不透明な head メッシュ側）を外す。

    Rocketbox の女性素体は髪を **2系統** で持つ: 不透明な head マテリアルの中に
    ポニーテールの束（独立したシェル、頭蓋の後ろに 8cm はみ出す）、その上に
    半透明の毛束カード（opacity マテリアル）。カードは alpha を 0 にして消せるが、
    束は head と同じ不透明マテリアルなのでテクスチャでは消せず、生成した髪の
    後ろから金髪の束が覗く（KEHAI でユーザーが指摘）。

    判定は幾何だけで行う: head マテリアルの面を連結成分に分け、最大の成分（頭蓋）
    より **後ろ（Blender +y）にはみ出す** 副シェルを落とす。目・歯・舌も head の
    副シェルだが顔の前側にあるので該当しない。男性素体には該当シェルが無い。
    モーフは bmesh の shape layer ごと残るので、頂点を消しても 175 個のまま。
    """
    removed = 0
    dropped = []
    for o in objs:
        names = [m.name if m else "" for m in o.data.materials]
        keep = {i for i, n in enumerate(names) if face_material in n}
        if not keep:
            continue
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        head_faces = [f for f in bm.faces if f.material_index in keep]
        if not head_faces:
            bm.free()
            continue
        # head の面だけで連結成分を取る
        seen = set()
        comps = []
        for f0 in head_faces:
            if f0.index in seen:
                continue
            stack, comp = [f0], set()
            seen.add(f0.index)
            while stack:
                f = stack.pop()
                comp.update(v.index for v in f.verts)
                for v in f.verts:
                    for g in v.link_faces:
                        if g.material_index in keep and g.index not in seen:
                            seen.add(g.index)
                            stack.append(g)
            comps.append(comp)
        comps.sort(key=len, reverse=True)
        mw = o.matrix_world
        back = lambda idx: max((mw @ bm.verts[i].co).y for i in idx)
        main_back = back(comps[0])
        doomed = set()
        for comp in comps[1:]:
            b = back(comp)
            if b > main_back + margin:
                doomed |= comp
                dropped.append({"verts": len(comp),
                                "behind_skull_mm": round((b - main_back) * 1000, 1)})
        if doomed:
            bmesh.ops.delete(bm, geom=[bm.verts[i] for i in doomed], context="VERTS")
            bm.to_mesh(o.data)
            o.data.update()
            removed += len(doomed)
        bm.free()
    print(f"テンプレートの髪シェル: {len(dropped)}個 {removed}頂点を外した "
          f"{dropped}")
    return {"template_hair_shells": dropped, "template_hair_verts_removed": removed}


def keep_hair(obj, dists, nrm, outside: float, outside_front: float,
              front_cos: float, min_z: float, open_iters: int = 1,
              close_iters: int = 3, down_cos: float = 0.4) -> tuple:
    """髪と判定した頂点だけを残す。**閾値は場所によって変える。**

    単一の閾値では分けられないことを実測した（`--outside` を 8/15/25mm で掃引）:

        8mm  … 頭頂の髪は残るが**顔の表面まで全部残る**
        15mm … 顔はまだ残る
        25mm … 顔は消えるが**頭頂も禿げる**

    頭頂の髪は頭皮から 8〜15mm しか浮いていないのに、顔の上の生成メッシュは
    ラップの残差（約2mm）＋αで数mm浮くため、量的に重なってしまう。

    そこで**最近点の法線の向き**で分ける。glTF は +Z が前で、Blender へは
    (x,y,z)→(x,-z,y) と入るので前は -Y。顔の上（法線が前向き）は厳しい閾値、
    側頭・頭頂・後頭部は緩い閾値にする。
    """
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    mw = obj.matrix_world
    keep, nfront = [], 0
    for v, d, n in zip(bm.verts, dists, nrm):
        # **前向きと下向きは厳しい閾値。** 顔の上は言うまでもないが、顎の裏・
        # 首の上面（法線が下を向く）にも髪は生えない。閾値を 8→5mm に緩めたら
        # Subject A の顎の下に肌の破片が残った。3人で掃引した結果（docs/09 §28）:
        #   8mm/400 … 短髪の男性は髪が 2,523頂点しか残らず（分布の 90% 点が 7.6mm）
        #   5mm/150 … 3人とも髪が出る。ただし下向き面を緩くすると顎の下に肌が残る
        strict = (n.y < -front_cos) or (n.z < -down_cos)
        nfront += int(strict)
        thr = outside_front if strict else outside
        keep.append(bool(d > thr and (mw @ v.co).z > min_z))

    # **穴を埋めてから縁を整える（closing → opening）。**
    #
    # 頬と側頭に見えていた「黒い粒」の実体は、拡大して初めて分かった ——
    # **髪メッシュに開いた穴から肌が透けていた**。距離の切り口が髪の内部を通り、
    # 髪の表面が一時的に肌へ近づくところが虫食いになる。だから
    #
    #   - 連結成分で落とす      … 粒は髪本体の一部なので落ちない
    #   - 空間クラスタで落とす  … 半径 1.5mm でも上位2塊に含まれたまま
    #   - 収縮→膨張（opening） … 穴は埋まらない。**逆をやる必要がある**
    #
    # 膨張 k 回 → 収縮 k 回（closing）で 2k 頂点ぶんまでの穴が埋まり、
    # 外周は元に戻る。そのあと opening で縁のヒゲを落とす。
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

    for _ in range(close_iters):
        keep = dilate(keep)
    for _ in range(close_iters):
        keep = erode(keep)
    for _ in range(open_iters):
        keep = erode(keep)
    for _ in range(open_iters):
        keep = dilate(keep)
    doomed = [v for i, v in enumerate(bm.verts) if not keep[i]]
    bmesh.ops.delete(bm, geom=doomed, context="VERTS")
    bm.to_mesh(me)
    bm.free()
    return len(me.vertices), nfront


def spatial_clusters(obj, radius: float, min_verts: int, max_clusters: int):
    """**位相ではなく距離**で塊にまとめ、小さい塊を落とす。

    生成メッシュの髪はメッシュとしては 7,753個の連結成分に割れているが、
    空間的には数個の塊（左・右・後ろ）でしかない。位相の連結成分で切ろうとすると

      - 「大きい順に N 個」…上位8個でも細い筋しか残らない
      - 「頂点数の下限」  …頬や肩の粒は 800〜2000頂点あり、下限を上げると
                          髪の体積まで削れて左右非対称になる

    のどちらも破綻した。半径 `radius` 以内を同じ塊とみなす union-find なら、
    割れていても髪は1つにまとまり、離れた粒は別の塊として落ちる。
    """
    me = obj.data
    n = len(me.vertices)
    if n == 0:
        return 0, 0
    mw = obj.matrix_world
    pts = [mw @ v.co for v in me.vertices]
    kd = KDTree(n)
    for i, p in enumerate(pts):
        kd.insert(p, i)
    kd.balance()

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, p in enumerate(pts):
        for _, j, _ in kd.find_range(p, radius):
            if j != i:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    order = sorted(groups.values(), key=len, reverse=True)
    print(f"  クラスタの大きさ 上位8: {[len(g) for g in order[:8]]}")
    keep = [g for g in order if len(g) >= min_verts]
    if max_clusters:
        keep = keep[:max_clusters]
    kept = {i for g in keep for i in g}

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    doomed = [bm.verts[i] for i in range(n) if i not in kept]
    bmesh.ops.delete(bm, geom=doomed, context="VERTS")
    bm.to_mesh(me)
    bm.free()
    return len(order), len(keep)


def drop_small_parts(obj, min_frac: float = 0.02, min_verts: int = 200,
                     max_parts: int = 0) -> tuple:
    """浮遊した小片を落とす。生成メッシュは肌の外側にノイズを持つ。

    **「小さい」で切るだけでは足りない。** 頬や肩に乗る粒は 800〜2000頂点あり、
    そこまで下限を上げると髪の体積まで削れて左右非対称になった（実測: min_verts
    2000 で肩の粒は消えるが片側の髪がほぼ消えた）。

    髪は数個の大きな塊（左・右・後ろ）になるので、**大きい順に max_parts 個だけ
    残す**のが効く。粒は数が多くて小さいので自然に落ちる。
    """
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    seen, groups = set(), []
    for v in bm.verts:
        if v in seen:
            continue
        stack, comp = [v], []
        seen.add(v)
        while stack:
            x = stack.pop()
            comp.append(x)
            for e in x.link_edges:
                o = e.other_vert(x)
                if o not in seen:
                    seen.add(o)
                    stack.append(o)
        groups.append(comp)
    if not groups:
        bm.free()
        return 0, 0
    groups.sort(key=len, reverse=True)
    big = len(groups[0])
    keep = [g for g in groups if len(g) >= max(big * min_frac, min_verts)]
    if max_parts:
        keep = keep[:max_parts]
    kept = {v for g in keep for v in g}
    doomed = [v for v in bm.verts if v not in kept]
    bmesh.ops.delete(bm, geom=doomed, context="VERTS")
    bm.to_mesh(me)
    bm.free()
    return len(groups), len(keep)


def push_out_of_head(h, center, head_faces, gap: float) -> int:
    """**頭（耳を含む）が髪を突き抜けないようにする。** 殻の作り直し・平滑化・削減の
    **後**に掛けること。平滑化は閉じた殻を縮め、削減は面を粗くするので、先に押し
    出しても耳がまた突き抜けた（実測）。

    テンプレの耳は本人の耳より外に突き出ているので、最近点の法線や中心からの
    放射で押すだけでは耳の内側の面へ逃げて耳先が残った（試して両方だめだった）。
    頭の中心から各頂点へ伸ばした半直線を**外側から内側へ**頭のマテリアルの面に
    当て、最初に当たる面（＝その方向で一番外にある皮膚、耳なら耳先）より
    `gap` 外に頂点を置く。首から下（肩・上着）は head マテリアルでないので
    髪の毛先が肩に押し出されることはない。
    """
    if center is None or not head_faces or gap <= 0:
        return 0
    tree = BVHTree.FromPolygons(head_faces[0], head_faces[1], all_triangles=True)
    mw = h.matrix_world
    inv = mw.inverted()
    pushed = 0
    for v in h.data.vertices:
        p = mw @ v.co
        direction = (p - center)
        dist = direction.length
        if dist < 1e-6:
            continue
        direction.normalize()
        origin = center + direction * 1.0
        loc, _, _, _ = tree.ray_cast(origin, -direction, 1.0)
        if loc is None:
            continue
        outer = (loc - center).dot(direction)
        if dist < outer + gap:
            v.co = inv @ (center + direction * (outer + gap))
            pushed += 1
    h.data.update()
    print(f"  頭に沈んだ {pushed:,}頂点を {gap*1000:.0f}mm 外へ（耳の外側まで）")
    return pushed


def rebuild_shell(h, radius: float, voxel: float, body=None, gap: float = 0.002,
                  center=None, head_faces=None, density: float = 5.0,
                  threshold: float = 0.01) -> dict:
    """毛束の三角形スープを**穴の無い殻**に作り直す。

    TRELLIS の髪は三角形スープで、切り出した後は辺の 38.6% が境界（＝穴だらけ）。
    両面描画にしても網目（ユーザー曰く「あみあみ」）に見える。頂点マスクの
    closing では埋まらない —— 穴はマスクではなくスープそのものの性質。

    やること: 頂点を点群にして半径 `radius` の球で体積化（Points to Volume）→
    等値面をメッシュ化（Volume to Mesh）。点群の外皮が滑らかな閉曲面になる。
    球の分だけ膨らむので法線方向に `radius` 縮める。UV は元のスープの最寄り面から
    移す（テクスチャの毛流れはそのまま載る）。

    単位に注意: 髪は Bip01（0.01 倍）の子なのでローカル座標は cm。引数はメートル。
    """
    unit = h.matrix_world.to_scale()[0]
    r_loc, v_loc = radius / unit, voxel / unit
    n_before = len(h.data.vertices)
    orig = h.copy()
    orig.data = h.data.copy()
    orig.name = h.name + "_soup"
    bpy.context.scene.collection.objects.link(orig)
    mat = h.data.materials[0] if h.data.materials else None

    ng = bpy.data.node_groups.new("HairShell", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi = ng.nodes.new("NodeGroupInput")
    go = ng.nodes.new("NodeGroupOutput")
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    m2p.mode = "VERTICES"
    p2v = ng.nodes.new("GeometryNodePointsToVolume")
    p2v.resolution_mode = "VOXEL_SIZE"
    p2v.inputs["Voxel Size"].default_value = v_loc
    p2v.inputs["Radius"].default_value = r_loc
    p2v.inputs["Density"].default_value = density
    v2m = ng.nodes.new("GeometryNodeVolumeToMesh")
    v2m.resolution_mode = "VOXEL_SIZE"
    v2m.inputs["Voxel Size"].default_value = v_loc
    v2m.inputs["Threshold"].default_value = threshold
    sm = ng.nodes.new("GeometryNodeSetMaterial")
    if mat is not None:
        sm.inputs["Material"].default_value = mat
    ng.links.new(gi.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(m2p.outputs["Points"], p2v.inputs["Points"])
    ng.links.new(p2v.outputs["Volume"], v2m.inputs["Volume"])
    ng.links.new(v2m.outputs["Mesh"], sm.inputs["Geometry"])
    ng.links.new(sm.outputs["Geometry"], go.inputs[0])

    bpy.ops.object.select_all(action="DESELECT")
    h.select_set(True)
    bpy.context.view_layer.objects.active = h
    mod = h.modifiers.new("shell", "NODES")
    mod.node_group = ng
    bpy.ops.object.modifier_apply(modifier=mod.name)
    n_shell = len(h.data.vertices)
    if n_shell < 100:
        raise SystemExit(f"髪の殻が {n_shell}頂点しかできなかった（半径 {radius*1000:.1f}mm）")

    # 球の分だけ膨らんでいるので法線方向に戻す
    disp = h.modifiers.new("shrink", "DISPLACE")
    disp.direction = "NORMAL"
    disp.mid_level = 0.0
    disp.strength = -r_loc
    bpy.ops.object.modifier_apply(modifier=disp.name)

    # UV を元のスープの最寄り面から移す
    if not h.data.uv_layers:
        h.data.uv_layers.new(name="UVMap")
    dt = h.modifiers.new("uv", "DATA_TRANSFER")
    dt.object = orig
    dt.use_loop_data = True
    dt.data_types_loops = {"UV"}
    dt.loop_mapping = "POLYINTERP_NEAREST"
    bpy.ops.object.modifier_apply(modifier=dt.name)
    # **頭の中に沈んだ殻を外へ出す。** 球の分だけ縮めると、髪が薄い側頭では殻が
    # テンプレの頭蓋（髪の厚みを含む）の内側に入り、肌が髪を突き抜けて斑に見える。
    # 体表面からの符号付き距離が `gap` 未満の頂点を、表面の法線に沿って `gap` まで押し出す。
    pushed = 0

    h.data.materials.clear()
    if mat is not None:
        h.data.materials.append(mat)
    # 等値面はフラットで出てくる。スムーズにしないと書き出しで面ごとに頂点が
    # 割れて 3 倍（20k → 120k）になる
    h.data.polygons.foreach_set("use_smooth", [True] * len(h.data.polygons))
    h.data.update()
    bpy.data.objects.remove(orig)
    bpy.data.node_groups.remove(ng)
    print(f"殻に作り直し: スープ {n_before:,}頂点 → 殻 {n_shell:,}頂点"
          f"（半径 {radius*1000:.1f}mm・voxel {voxel*1000:.1f}mm）")
    return {"shell_vertices": n_shell, "shell_radius_mm": radius * 1000,
            "shell_voxel_mm": voxel * 1000}


def build_hair_chains(hair, arm, head_bone: str, njoints: int, ngroups: int):
    """髪をいくつかの束に分け、束ごとにボーンチェーンを作る。

    **KEHAI はプレーンな glb では髪を揺らせない。** `glb-humanoid.ts` が明記して
    いる ——「What such a model does NOT get, because the file cannot describe
    them: spring bones (hair/cloth physics)」。揺らすには VRM で書き出して
    `VRMC_springBone` を持たせる必要があり、そのためにまず**骨が要る**。

    束の分け方は頭の中心から見た方位角。ボブなら左・右・後ろの3束で足りる。
    各束の z を等分し、その高さの帯にある頂点の重心をチェーンの節にする ——
    束が横に張り出していてもチェーンが芯を通る。
    """
    import math
    mw = hair.matrix_world
    pts = [mw @ v.co for v in hair.data.vertices]
    cx = sum(p.x for p in pts) / len(pts)
    cy = sum(p.y for p in pts) / len(pts)
    groups = [[] for _ in range(ngroups)]
    for i, p in enumerate(pts):
        a = math.atan2(p.y - cy, p.x - cx) + math.pi
        groups[min(int(a / (2 * math.pi) * ngroups), ngroups - 1)].append(i)

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    parent = eb.get(head_bone)
    inv = arm.matrix_world.inverted()
    chains = []
    for gi, idx in enumerate(groups):
        if len(idx) < 200:
            chains.append([])
            continue
        zs = sorted(pts[i].z for i in idx)
        z_hi = zs[int(len(zs) * 0.90)]
        z_lo = zs[int(len(zs) * 0.05)]
        if z_hi - z_lo < 0.02:
            chains.append([])
            continue
        nodes = []
        for k in range(njoints + 1):
            z = z_hi - (z_hi - z_lo) * k / njoints
            band = [pts[i] for i in idx if abs(pts[i].z - z) < (z_hi - z_lo) / njoints]
            if not band:
                band = [pts[i] for i in idx]
            nodes.append(Vector((sum(p.x for p in band) / len(band),
                                 sum(p.y for p in band) / len(band), z)))
        bones, prev = [], parent
        for k in range(njoints):
            b = eb.new(f"HairSpring{gi}_{k + 1}")
            b.head = inv @ nodes[k]
            b.tail = inv @ nodes[k + 1]
            b.parent = prev
            b.use_connect = k > 0
            bones.append(b.name)
            prev = b
        # **末端ノードを1つ余分に作る。** VRM の spring は joints の**隣り合う
        # 2節**で1区間を作る（three-vrm は joints[i+1] を joints[i] の子として
        # 使う）ので、N本を揺らすには N+1個の節が要る。この骨にウェイトは載せない。
        tip = eb.new(f"HairSpring{gi}_tip")
        d = (nodes[njoints] - nodes[njoints - 1])
        d = d.normalized() * 0.02 if d.length > 1e-6 else Vector((0, 0, -0.02))
        tip.head = inv @ nodes[njoints]
        tip.tail = inv @ (nodes[njoints] + d)
        tip.parent = prev
        tip.use_connect = True
        bones.append(tip.name)
        chains.append(bones)
    bpy.ops.object.mode_set(mode="OBJECT")
    return chains


def weight_to_chains(hair, arm, chains, head_bone: str) -> None:
    """各頂点を最も近いチェーンの節へ。**節の間は線形に混ぜる。**

    段差があると揺れたときに面が割れる。チェーンの根本より上は頭に付ける
    （頭皮に接している部分は揺れてはいけない）。
    """
    # 末端の tip にはウェイトを載せない（揺れの計算にだけ使う節）
    names = [head_bone] + [n for c in chains for n in c if not n.endswith("_tip")]
    vgs = {n: (hair.vertex_groups.get(n) or hair.vertex_groups.new(name=n))
           for n in names}
    mw = hair.matrix_world
    heads = {}
    for c in chains:
        for n in [x for x in c if not x.endswith("_tip")]:
            b = arm.data.bones[n]
            heads[n] = (arm.matrix_world @ b.head_local,
                        arm.matrix_world @ b.tail_local)
    for i, v in enumerate(hair.data.vertices):
        p = mw @ v.co
        best, bd, bt = None, 1e9, 0.0
        for c in chains:
            for n in [x for x in c if not x.endswith("_tip")]:
                a, b = heads[n]
                ab = b - a
                t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-12)))
                d = (p - (a + ab * t)).length
                if d < bd:
                    best, bd, bt = n, d, t
        if best is None:
            vgs[head_bone].add([i], 1.0, "REPLACE")
            continue
        c = [x for x in next(c for c in chains if best in c)
             if not x.endswith("_tip")]
        k = c.index(best)
        # 根本の節の上半分は頭に寄せる。頭皮側が揺れないようにする。
        if k == 0 and bt < 0.5:
            w = bt / 0.5
            vgs[head_bone].add([i], 1.0 - w, "REPLACE")
            vgs[best].add([i], w, "REPLACE")
        else:
            vgs[best].add([i], 1.0, "REPLACE")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ラップ済みアバター（アーマチュア付き）")
    ap.add_argument("--generated", required=True, help="TRELLIS が生成した頭")
    ap.add_argument("--wrap-report", required=True, help="相似変換を読む")
    ap.add_argument("--out", required=True)
    ap.add_argument("--face-material", default="head")
    ap.add_argument("--outside", type=float, default=0.005,
                    help="側頭・頭頂・後頭部で髪とみなす外側距離（m）。"
                         "8mm だと短髪で髪が消える（3人で掃引・docs/09 §28）")
    ap.add_argument("--outside-front", type=float, default=0.030,
                    help="**顔の上**で髪とみなす外側距離（m）。ここを緩めると"
                         "生成メッシュの顔が丸ごと残る")
    ap.add_argument("--down-cos", type=float, default=0.40,
                    help="最近点の法線がこれ以上下を向いていたら顎の裏とみなし、"
                         "顔と同じ厳しい閾値にする")
    ap.add_argument("--front-cos", type=float, default=0.40,
                    help="最近点の法線がこれ以上前を向いていたら「顔の上」")
    ap.add_argument("--cut-frac", type=float, default=0.40,
                    help="顔マテリアルの高さのこの比より下は捨てる。顎が約 0.38。"
                         "**0.30 だと顎の 3cm 下まで残り、写真の上着の襟が髪として"
                         "出荷された**（本人の上着が髪と同じ暗い色で目立った）")
    ap.add_argument("--neck-gap", type=float, default=0.0,
                    help="cut_frac の位置からさらにこれだけ下まで残す（m）")
    # 半径は毛束の隙間を埋める大きさ。3mm では殻が 8 個の塊に分かれて塊の間に穴が
    # 残り、KEHAI では依然「あみあみ」に見えた（背景が穴から透ける）。6mm と 8mm を
    # 並べて 8mm を採った（§28.9）。球の分は法線方向に戻すので輪郭は膨らまない。
    ap.add_argument("--shell-radius", type=float, default=0.008,
                    help="毛束のスープを穴の無い殻に作り直す球の半径（m）。0 で作り直さない")
    ap.add_argument("--shell-voxel", type=float, default=0.002,
                    help="殻の voxel（m）")
    ap.add_argument("--shell-gap", type=float, default=0.003,
                    help="殻を体表面からこれだけ外に保つ（m）。頭が髪を突き抜けるのを防ぐ")
    ap.add_argument("--smooth", type=int, default=3,
                    help="ラプラシアン平滑の回数。生成メッシュの髪は波打っている")
    ap.add_argument("--close-iters", type=int, default=3,
                    help="膨張→収縮の回数。**髪に開いた穴を埋める**。"
                         "拡大して見ると粒の正体は穴から透けた肌だった")
    ap.add_argument("--open-iters", type=int, default=1,
                    help="収縮→膨張の回数。縁のヒゲを落とす")
    ap.add_argument("--cluster-radius", type=float, default=0.0,
                    help="この距離以内を同じ塊とみなす（m）。0 で無効")
    ap.add_argument("--cluster-min", type=int, default=1500,
                    help="この頂点数未満の空間クラスタは捨てる")
    ap.add_argument("--max-clusters", type=int, default=6,
                    help="大きい順にこの個数の空間クラスタだけ残す")
    ap.add_argument("--max-parts", type=int, default=0,
                    help="大きい順にこの個数だけ残す。0 で無制限。"
                         "髪は左・右・後ろの数塊になるので少数で足りる")
    ap.add_argument("--min-part", type=int, default=150,
                    help="これ未満の連結成分は捨てる。生成メッシュのノイズ対策")
    ap.add_argument("--target", type=int, default=20000, help="髪の頂点予算")
    ap.add_argument("--landmarks", default=None,
                    help="configs/landmarks/*.json。テンプレの耳を距離の表面から外すのに使う")
    ap.add_argument("--head-bone", default="Head")
    ap.add_argument("--keep-template-hair", action="store_true",
                    help="テンプレートの髪シェル（ポニーテール等）を外さない")
    ap.add_argument("--spring-joints", type=int, default=3,
                    help="チェーンあたりの骨の数。0 で骨を作らず Head に固定")
    ap.add_argument("--spring-groups", type=int, default=3,
                    help="髪を方位角で何束に分けるか")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    scene_reset()
    body = import_mesh(Path(args.model))
    apply_transforms(body)
    strip_rep = ({} if args.keep_template_hair
                 else strip_template_hair(body, args.face_material))
    arm = find_armature()
    if arm is None:
        raise SystemExit(f"{args.model} にアーマチュアが無い。--keep-rig を通したか")
    if args.head_bone not in arm.data.bones:
        raise SystemExit(f"ボーン '{args.head_bone}' が無い: "
                         f"{sorted(b.name for b in arm.data.bones)[:12]}")

    # 顔マテリアルの下端。ここより下は髪ではない（首・肩）
    face_z = []
    for o in body:
        names = [m.name if m else "" for m in o.data.materials]
        keep = {i for i, n in enumerate(names) if args.face_material in n}
        if not keep:
            continue
        mw = o.matrix_world
        for p in o.data.polygons:
            if p.material_index in keep:
                face_z += [(mw @ o.data.vertices[i].co).z for i in p.vertices]
    if not face_z:
        raise SystemExit(f"マテリアル '{args.face_material}' が見つからない")
    # **切る高さは顔マテリアルの下端ではなく、頭の高さの比で指定する。**
    # 顔マテリアルは首まで伸びているので下端で切ると首の残骸が残り、体型変形
    # （身長 1.743→1.60m、倍率 0.918）で縮んで**肩の高さに黒い粒として乗る**。
    # 顎は頭の高さの約 0.38 なので、その少し下で切ると髪（ボブは顎の高さで終わる）
    # を残したまま首の残骸を落とせる。
    fz0, fz1 = min(face_z), max(face_z)
    min_z = fz0 + args.cut_frac * (fz1 - fz0) - args.neck_gap

    before = set(bpy.data.objects)
    hair = import_mesh(Path(args.generated))
    apply_transforms(hair)
    if len(hair) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in hair:
            o.select_set(True)
        bpy.context.view_layer.objects.active = hair[0]
        bpy.ops.object.join()
        hair = [hair[0]]
    h = hair[0]
    h.name = "Hair"
    # **金属にしない。** TRELLIS の glb は `metallicFactor` を持たず、glTF の既定値 1.0
    # （＝完全な金属）として読まれる。Blender の EEVEE はライトの反射で黒髪らしく
    # 見せてしまうが、KEHAI（three.js、環境マップ無し）では金属は**真っ黒の
    # 切り絵**になる（ユーザーが「壊れてる」と指摘したのがこれ）。素体は
    # `prepare_rocketbox.py` で 0 にしてあるが、髪は生成物なのでここで直す。
    # 値を書くだけでは足りない: TRELLIS の glb は metallicRoughness **テクスチャ**を
    # 持ち、Blender はそれを Separate Color 経由で Metallic/Roughness に結線する。
    # 結線が残ると書き出しでテクスチャが復活し、`metallicFactor` は省略（=1.0）に戻る。
    # リンクを切ってから定数を入れる。
    for slot in h.material_slots:
        mat = slot.material
        if mat and mat.use_nodes:
            nt = mat.node_tree
            for node in nt.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    for name, val in (("Metallic", 0.0), ("Roughness", 0.7)):
                        sock = node.inputs[name]
                        for link in list(sock.links):
                            nt.links.remove(link)
                        sock.default_value = val

    rep = json.loads(Path(args.wrap_report).read_text())
    if "align" not in rep:
        raise SystemExit(f"{args.wrap_report} に align が無い。wrap_head.py を"
                         f"更新してから回し直すこと")
    m = similarity(rep)
    h.matrix_world = m @ h.matrix_world
    apply_transforms([h])

    def bb(objs):
        pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
        return (Vector(map(min, zip(*pts))), Vector(map(max, zip(*pts))))
    bl, bh = bb(body)
    hl, hh = bb([h])
    print(f"アバター 外接 z[{bl.z:.3f},{bh.z:.3f}]  "
          f"変換後の生成メッシュ z[{hl.z:.3f},{hh.z:.3f}]")
    print(f"顔マテリアル z[{fz0:.3f},{fz1:.3f}]  切り捨て z<{min_z:.3f}"
          f"（比 {args.cut_frac}）")
    overlap = min(bh.z, hh.z) - max(bl.z, hl.z)
    if overlap <= 0:
        raise SystemExit("変換後の生成メッシュがアバターと重なっていない。"
                         "相似変換の共役が誤っている疑い")

    n0 = len(h.data.vertices)
    mw = h.matrix_world
    pts = [mw @ v.co for v in h.data.vertices]
    ear = None
    if args.landmarks:
        # 68点の頂点番号は head プリミティブ内の番号。Blender は全プリミティブを
        # 1メッシュに畳むので、head マテリアルが使う最小の頂点番号を足す。
        lmj = json.loads(Path(args.landmarks).read_text())["landmarks"]
        for o in body:
            names = [m.name if m else "" for m in o.data.materials]
            keep = {i for i, n in enumerate(names) if args.face_material in n}
            if not keep:
                continue
            lo = min(i for p in o.data.polygons if p.material_index in keep for i in p.vertices)
            omw = o.matrix_world
            P = {k: omw @ o.data.vertices[lo + lmj[k]["vertex"]].co for k in (0, 16, 36, 45, 8)}
            x_jaw = max(abs(P[0].x), abs(P[16].x))
            z_lo = min(P[0].z, P[16].z) - 0.02
            z_hi = max(P[36].z, P[45].z) + 0.03
            y_min = min(P[0].y, P[16].y) - 0.01      # Blender では前が -y。耳は顎の輪郭点より後ろ
            ear = (args.face_material,
                   lambda c, xj=x_jaw, zl=z_lo, zh=z_hi, ym=y_min:
                       abs(c.x) > xj + 0.004 and zl < c.z < zh and c.y > ym)
            print(f"  耳の箱: |x|>{x_jaw+0.004:.3f} z[{z_lo:.3f},{z_hi:.3f}] y>{y_min:.3f}")
            break
    dists, nrm = signed_distances(body, pts, exclude=ear)
    # 側頭・頭頂・後頭部の距離分布を記録する。閾値の当否を後から検証できるように。
    import numpy as _np
    _d = _np.array(dists); _ny = _np.array([q.y for q in nrm]); _nz = _np.array([q.z for q in nrm])
    _z = _np.array([(mw @ v.co).z for v in h.data.vertices])
    _m = (_ny >= -args.front_cos) & (_nz >= -args.down_cos) & (_z > min_z)
    pct = {p: round(float(v) * 1000, 1) for p, v in
           zip((50, 70, 90), _np.percentile(_d[_m], [50, 70, 90]))} if _m.any() else {}
    print(f"  頭皮領域 {int(_m.sum()):,}頂点  距離の分位(mm) "
          + " ".join(f"{k}%={v}" for k, v in pct.items()))
    n1, nfront = keep_hair(h, dists, nrm, args.outside, args.outside_front,
                           args.front_cos, min_z, args.open_iters,
                           args.close_iters, args.down_cos)
    print(f"  顔の上と判定 {nfront:,} / {n0:,}頂点"
          f"（閾値 前 {args.outside_front*1000:.0f}mm / その他 {args.outside*1000:.0f}mm）")
    ngroups, nkept = drop_small_parts(h, min_verts=args.min_part,
                                      max_parts=args.max_parts)
    if args.cluster_radius > 0:
        nc, nck = spatial_clusters(h, args.cluster_radius,
                                   args.cluster_min, args.max_clusters)
        print(f"  空間クラスタ {nc}個 → 大きい {nck}個を残して "
              f"{len(h.data.vertices):,}頂点（半径 {args.cluster_radius*1000:.0f}mm）")
    n2 = len(h.data.vertices)
    print(f"生成 {n0:,}頂点 → 外側かつ首より上 {n1:,} → 小片を除いて {n2:,}"
          f"（連結成分 {ngroups} のうち大きい {nkept}個を残した）")
    if n2 < 200:
        raise SystemExit(f"髪として残ったのが {n2}頂点。--outside を下げること")

    shell_rep = {}
    if args.shell_radius > 0:
        # 中心は head マテリアルの頂点の重心。ボーンの head_local から出すと
        # Rocketbox の Bip01 の軸のせいで 1.5m ずれた（実測）。
        hv, hf, off = [], [], 0
        for o in body:
            names = [m.name if m else "" for m in o.data.materials]
            keep = {i for i, n in enumerate(names) if args.face_material in n}
            mw = o.matrix_world
            hv += [mw @ v.co for v in o.data.vertices]
            for poly in o.data.polygons:
                if poly.material_index in keep:
                    idx = [i + off for i in poly.vertices]
                    for k in range(1, len(idx) - 1):
                        hf.append((idx[0], idx[k], idx[k + 1]))
            off += len(o.data.vertices)
        used = sorted({i for f in hf for i in f})
        head_center = sum((hv[i] for i in used), Vector()) / max(1, len(used))
        print(f"  頭の中心（head 頂点の重心） {tuple(round(c, 3) for c in head_center)}")
        shell_ctx = {"center": head_center, "head_faces": (hv, hf)}
        shell_rep = rebuild_shell(h, args.shell_radius, args.shell_voxel)
        n2 = len(h.data.vertices)

    # **平滑化する。** 生成メッシュの髪は毛束の隙間で細かく波打っており、
    # そのまま載せると縁がギザギザに見える。ラプラシアン平滑を数回かける。
    if args.smooth > 0:
        me = h.data
        bm = bmesh.new()
        bm.from_mesh(me)
        for _ in range(args.smooth):
            bmesh.ops.smooth_vert(bm, verts=bm.verts, factor=0.5,
                                  use_axis_x=True, use_axis_y=True, use_axis_z=True)
        bm.to_mesh(me)
        bm.free()

    if args.target and n2 > args.target:
        bpy.context.view_layer.objects.active = h
        mod = h.modifiers.new("dec", "DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = args.target / n2
        bpy.ops.object.modifier_apply(modifier=mod.name)
        print(f"削減 {n2:,} → {len(h.data.vertices):,}頂点（比 {mod.ratio:.4f}）")
    if shell_rep:
        shell_rep["shell_pushed_out"] = push_out_of_head(
            h, shell_ctx["center"], shell_ctx["head_faces"], args.shell_gap)
    n3 = len(h.data.vertices)

    # **結合しない。** アーマチュアに別メッシュとして紐づけるだけ。
    # 結合するとシェイプキーの扱いでモーフ175個を壊しうる。
    chains = []
    if args.spring_joints > 0:
        chains = build_hair_chains(h, arm, args.head_bone,
                                   args.spring_joints, args.spring_groups)
        weight_to_chains(h, arm, chains, args.head_bone)
        print(f"  髪のボーン {sum(len(c) for c in chains)}本"
              f"（{len([c for c in chains if c])}チェーン × "
              f"{args.spring_joints}+末端）"
              f" ← KEHAI で揺らすには VRM で書き出すこと")
    else:
        vg = h.vertex_groups.new(name=args.head_bone)
        vg.add(list(range(n3)), 1.0, "REPLACE")
    h.parent = arm
    h.matrix_parent_inverse = arm.matrix_world.inverted()
    am = h.modifiers.new("Armature", "ARMATURE")
    am.object = arm

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(args.out), export_format="GLB",
        use_selection=True, export_morph=True, export_morph_normal=False, export_skins=True,
        export_apply=False)

    out = {"model": args.model, "generated": args.generated, "out": args.out,
           "align_scale": rep["align"]["scale"],
           **strip_rep, **shell_rep,
           "generated_vertices": n0, "outside_and_above": n1,
           "after_small_parts": n2, "components": ngroups, "kept_parts": nkept,
           "hair_vertices": n3, "target": args.target,
           "spring_chains": [c for c in chains if c],
           "outside_mm": args.outside * 1000,
           "outside_front_mm": args.outside_front * 1000,
           "front_vertices": nfront, "scalp_distance_pct_mm": pct,
           "open_iters": args.open_iters,
           "close_iters": args.close_iters,
           "cut_z": round(min_z, 4),
           "cut_frac": args.cut_frac,
           "head_bone": args.head_bone,
           "bbox": {"avatar_z": [round(bl.z, 4), round(bh.z, 4)],
                    "hair_z": [round(hl.z, 4), round(hh.z, 4)]}}
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(MARKER + " " + json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
