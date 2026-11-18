"""実写の顔をメッシュのテクスチャへ焼き込む。

TRELLIS.2 の顔ジオメトリは元画像を2.5倍にしても ArcFace 0.38 前後で頭打ちに
なる（docs/05-measurements.md §15）。形で本人性を稼げない以上、実写を顔
テクスチャに焼くのが唯一の担保になる。

入力の写真は preprocessing/face_projection.py で「正面レンダの画角に整列済み・
顔領域がアルファに入った RGBA」であること。ここではその画像を正面カメラから
UV 投影し、既存のベースカラーと合成して焼き直す。

正面カメラは render_views.py と同一条件（60mm / 距離=radius*3 / 方位0）にする。
整列に使ったレンダと画角が1度でもずれると顔が二重像になる。

  blender --background --factory-startup --python blender/project_face.py -- \
      --input <mesh.glb> --photo <warped_rgba.png> --out <out.glb> [--bake-size 2048]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

# --factory-startup では実行スクリプトのディレクトリが sys.path に入らない
sys.path.insert(0, str(Path(__file__).resolve().parent))

MARKER = "PROJECT_FACE_OK"
LENS_MM = 60.0          # render_views.py と揃える
DIST_FACTOR = 3.0       # 同上


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(path: Path) -> list:
    before = set(bpy.data.objects)
    ext = path.suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".vrm":
        # 成果物そのものを点検したいことがあるので VRM も読めるようにする
        import addon_utils
        for m in addon_utils.modules():
            if "vrm" in m.__name__.lower():
                try:
                    addon_utils.enable(m.__name__, default_set=True, persistent=True)
                except Exception:
                    pass
        bpy.ops.import_scene.vrm(filepath=str(path))
    else:
        raise SystemExit(f"未対応の拡張子: {ext}")
    new = [o for o in bpy.data.objects if o not in before]
    return [o for o in new if o.type == "MESH"]


def scene_bounds(objs):
    lo = Vector((float("inf"),) * 3)
    hi = Vector((float("-inf"),) * 3)
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            lo = Vector((min(lo[i], w[i]) for i in range(3)))
            hi = Vector((max(hi[i], w[i]) for i in range(3)))
    center = (lo + hi) / 2.0
    radius = max((hi - lo).length / 2.0, 1e-6)
    return center, radius, lo, hi


def material_band(meshes, material: str) -> tuple[float, float] | None:
    """指定マテリアルの面だけから上下の帯を取る。

    **シルエットからの肩検出より確実。** `shoulder_split` は体型に左右され、実測では
    Rocketbox の女性素体で**絞ったシャツのウエストを肩と誤検出**して全身が枠に入り、
    整列レンダの顔が 91.7px しかなくなった（テクスチャの解像度がそこで決まる）。
    テンプレートは固定で顔マテリアル名も分かっているので、推定に頼る理由がない。
    """
    z0, z1, n = float("inf"), float("-inf"), 0
    for o in meshes:
        slots = [i for i, m in enumerate(o.data.materials) if m and material in m.name]
        if not slots:
            continue
        keep = set()
        for poly in o.data.polygons:
            if poly.material_index in slots:
                keep.update(poly.vertices)
        for vi in keep:
            z = (o.matrix_world @ o.data.vertices[vi].co).z
            z0 = min(z0, z); z1 = max(z1, z); n += 1
    return None if n == 0 else (z0, z1)


def head_band(meshes, lo: Vector, hi: Vector, fallback: float,
              material: str | None = None) -> tuple[float, float]:
    """頭頂から肩までの帯。検出は blender/mesh_landmarks.py と共用する。"""
    if material:
        band = material_band(meshes, material)
        if band is not None:
            return band
        print(f"警告: マテリアル '{material}' の面が無いので肩の検出に落とす",
              file=sys.stderr)

    from mesh_landmarks import shoulder_split

    pts = []
    for o in meshes:
        mw = o.matrix_world
        for v in o.data.vertices:
            w = mw @ v.co
            pts.append((w.x, w.z))
    if not pts:
        return hi.z - (hi.z - lo.z) * fallback, hi.z

    shoulder_z, _, _ = shoulder_split(pts, hi.z, lo.z, fallback)
    # 肩よりわずかに下まで含める（顎と首がフレームから切れないように）
    height = hi.z - lo.z
    return max(lo.z, shoulder_z - height * 0.023), hi.z


def setup_front_camera(center: Vector, radius: float, lo: Vector, hi: Vector,
                       meshes, frame: str, head_fraction: float,
                       azimuth_deg: float = 0.0, name: str = "ProjCam",
                       frame_material: str | None = None,
                       frame_margin: float | None = None):
    """正面カメラ。frame="full" は render_views.py の front と同一条件。

    frame="head" は頭部に寄せる。全身メッシュを全身のまま投影すると、
    2048 のレンダでも顔は 60px 程度にしかならず、実写を貼っても解像度が
    足りない（実測: 全身Tポーズで render_face_width 61px）。頭部を画面いっぱいに
    収めれば同じ 2048 でも顔が 10 倍以上の画素を持つ。
    """
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = LENS_MM
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    if frame == "full":
        look_at = center
        dist = radius * DIST_FACTOR
    else:
        # 頭部の帯をメッシュ形状から実測して画角に収める。
        z0, z1 = head_band(meshes, lo, hi, head_fraction, frame_material)
        band = max(z1 - z0, 1e-6)
        look_at = Vector((center.x, center.y, (z0 + z1) / 2.0))
        # 垂直画角の半分の tan。センサー既定 36mm・レンズ 60mm。
        half_tan = (cam_data.sensor_width / 2.0) / cam_data.lens
        # 余白。**上下どちらにも失敗の側がある。** 狭いと顔が枠を占めすぎて
        # SCRFD が検出できず（実測: Rocketbox の男性は髪が枠上端で切れる framing で
        # 顔検出が落ちた）、広いとテクセル密度が下がる。
        # マテリアルで枠を決めるときは髪が帯に入らないぶん顔が大きく写るので、
        # 既定を広めに取る（ortho_front.py の MATERIAL_MARGIN=1.35 と同じ理由）。
        margin = frame_margin
        if margin is None:
            margin = 1.45 if frame_material else 1.25
        dist = (band / 2.0) / half_tan * margin

    target = bpy.data.objects.new(name + "Target", None)
    target.location = look_at
    bpy.context.collection.objects.link(target)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"

    # 方位角 0 = 正面（-Y 側から見る）。render_views.py と同じ規約。
    a = math.radians(azimuth_deg)
    cam.location = Vector((look_at.x + dist * math.sin(a),
                           look_at.y - dist * math.cos(a),
                           look_at.z))
    bpy.context.view_layer.update()
    return cam


def camera_forward(azimuth_deg: float) -> tuple[float, float, float]:
    """カメラを向いた面の法線が向くべき方向。内積が正なら「こちらを向いている」。

    カメラは look_at から見て (sin a, -cos a, 0) の側にある（方位0＝-Y側）。
    したがってカメラを向いた面の法線もその向きになる。

    **符号を間違えるとマスクが反転し、正面ではなく背面に貼る。**
    実際 (−sin a, cos a, 0) と書いていたせいで、背面の後頭部に顔が
    まるごと投影されていた（多視点対応リファクタで入れた退行）。
    """
    a = math.radians(azimuth_deg)
    return (math.sin(a), -math.cos(a), 0.0)


def setup_world_for_align() -> None:
    """整列レンダ用の照明。

    **`render_views.py` の setup_world と同一の値**にしてある。あちらは
    候補間で露出を揃えるために較正された設定で、勝手に明るくすると
    顔が白飛びして ArcFace のランドマーク検出が落ちる（実測: 背景強度を
    1.6 にしたら目が飛んで「顔を検出できません」で整列が失敗した）。
    """
    world = bpy.data.worlds.new("W")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 0.55

    def lamp(name, loc, energy):
        d = bpy.data.lights.new(name, type="AREA")
        d.energy = energy
        d.size = 4.0
        o = bpy.data.objects.new(name, d)
        o.location = loc
        bpy.context.collection.objects.link(o)

    lamp("key", (3.0, -4.0, 4.0), 900)
    lamp("fill", (-4.0, -2.0, 2.0), 350)
    lamp("rim", (0.0, 4.5, 3.0), 500)


def flatten_materials(meshes) -> list[str]:
    """整列レンダの前に、金属反射を切ってベースカラーだけで描かせる。

    TRELLIS.2 の出力は候補によって PBR の作りが違う。metallicFactor=0 で
    metallicRoughnessTexture を持たない候補もあれば、MR テクスチャ付きで
    factor が既定(=1.0)の候補もある。後者は Blender 上で顔が金属面として
    描画され、白く飛んで ArcFace のランドマーク検出が落ちる
    （実測: 同一入力でも候補ごとに整列が成功したり exit=1 で失敗したりした）。

    整列レンダは顔の位置合わせにしか使わないので、鏡面反射は不要どころか害。
    ここで一律に潰して、どの候補でも同じ条件で撮る。
    """
    touched = []
    for obj in meshes:
        for slot in obj.material_slots:
            mat = slot.material
            if not mat or not mat.use_nodes:
                continue
            bsdf = next((n for n in mat.node_tree.nodes
                         if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                continue
            for name, value in (("Metallic", 0.0), ("Roughness", 0.65),
                                ("Specular IOR Level", 0.25)):
                inp = bsdf.inputs.get(name)
                if inp is None:
                    continue
                for link in list(inp.links):
                    mat.node_tree.links.remove(link)
                inp.default_value = value
            touched.append(mat.name)
    return touched


def render_align_view(out: Path, res: int, samples: int = 32) -> None:
    """投影に使うのと同一のカメラで整列用の正面画像を出す。

    別スクリプトで撮ると画角がずれて顔が二重像になるので、必ずここで撮る
    （docs/05-measurements.md §16 の落とし穴）。
    """
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [
        e.identifier for e in
        bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    ] else "BLENDER_EEVEE"
    sc.render.resolution_x = res
    sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    try:
        sc.eevee.taa_render_samples = samples
    except Exception:
        pass
    sc.view_settings.view_transform = "Standard"
    out.parent.mkdir(parents=True, exist_ok=True)
    sc.render.filepath = str(out)
    bpy.ops.render.render(write_still=True)


def add_projection_uv(obj, cam, name: str = "faceproj") -> None:
    """カメラ投影の UV を第2レイヤとして作る。

    uv.project_from_view は 3D ビューの文脈を要求しヘッドレスで不安定なので、
    world_to_camera_view で自前に計算する。
    """
    from bpy_extras.object_utils import world_to_camera_view

    me = obj.data
    uv = me.uv_layers.get(name) or me.uv_layers.new(name=name)
    scene = bpy.context.scene
    mw = obj.matrix_world
    for loop in me.loops:
        co = mw @ me.vertices[loop.vertex_index].co
        p = world_to_camera_view(scene, cam, co)
        uv.data[loop.index].uv = (p.x, p.y)


def base_color_source(nodes, bsdf):
    """Base Color に今つながっているものを返す。無ければ定数色ノードを作る。"""
    inp = bsdf.inputs["Base Color"]
    if inp.is_linked:
        link = inp.links[0]
        return link.from_node, link.from_socket
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = tuple(inp.default_value)
    rgb.location = (-600, 300)
    return rgb, rgb.outputs[0]


def pick_material(obj, name: str | None):
    """写真を貼るマテリアルを選ぶ。

    **スロット0の決め打ちは危険。** glTF のインポータは複数プリミティブのメッシュを
    1オブジェクト＋複数スロットに畳むので、スロット0が顔とは限らない。実測では
    VRoid テンプレートの `Face` はスロット0が `FaceMouth_00_FACE`、Rocketbox は
    スロット0が `body` で、どちらも顔ではない。例外も出ずに間違った場所へ焼ける。
    """
    mats = [m for m in obj.data.materials if m]
    if not mats:
        raise SystemExit(f"{obj.name}: マテリアルがありません")
    if name:
        hit = [m for m in mats if name in m.name]
        if not hit:
            raise SystemExit(f"{obj.name}: マテリアル '{name}' がありません: "
                             f"{[m.name for m in mats]}")
        return hit[0]
    if len(mats) > 1:
        print(f"警告: {obj.name} はマテリアルが {len(mats)}個ある "
              f"({[m.name for m in mats]})。--face-material で指定しないと "
              f"'{mats[0].name}' に焼く", file=sys.stderr)
    return mats[0]


def build_projection_material(obj, views, facing_min: float, facing_max: float,
                              material=None):
    """既存のベースカラーに、投影した実写をマスク合成するノードを挿す。

    views は (画像, UV名, カメラ方位角) のリスト。**正面を最後に置く**こと。
    頬のあたりで正面と側面のマスクが重なるが、そこは解像度の高い正面を
    採りたいため、後から混ぜたものが勝つ順序にしてある。
    """
    mat = material if material is not None else pick_material(obj, None)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        raise SystemExit(f"{obj.name}: Principled BSDF が見つかりません")

    src_node, current = base_color_source(nodes, bsdf)

    geo = nodes.new("ShaderNodeNewGeometry")
    geo.location = (-1400, -800)

    for i, (photo_img, uv_name, azim) in enumerate(views):
        y = -200 - i * 420
        uvmap = nodes.new("ShaderNodeUVMap")
        uvmap.uv_map = uv_name
        uvmap.location = (-1200, y)

        tex = nodes.new("ShaderNodeTexImage")
        tex.image = photo_img
        tex.interpolation = "Cubic"
        tex.extension = "CLIP"          # 顔の外へタイリングさせない
        tex.location = (-1000, y)
        links.new(uvmap.outputs["UV"], tex.inputs["Vector"])

        # そのカメラを向いた面だけに貼る。斜めの面に貼ると引き伸ばされる。
        dot = nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        dot.inputs[1].default_value = camera_forward(azim)
        dot.location = (-800, y - 200)
        links.new(geo.outputs["Normal"], dot.inputs[0])

        facing = nodes.new("ShaderNodeMapRange")
        facing.inputs["From Min"].default_value = facing_min
        facing.inputs["From Max"].default_value = facing_max
        facing.clamp = True
        facing.location = (-600, y - 200)
        links.new(dot.outputs["Value"], facing.inputs["Value"])

        fac = nodes.new("ShaderNodeMath")
        fac.operation = "MULTIPLY"
        fac.location = (-400, y - 100)
        links.new(tex.outputs["Alpha"], fac.inputs[0])
        links.new(facing.outputs["Result"], fac.inputs[1])

        mix = nodes.new("ShaderNodeMixRGB")   # 4.2 でもソケット名が安定
        mix.blend_type = "MIX"
        mix.location = (-200, y + 100)
        links.new(fac.outputs[0], mix.inputs["Fac"])
        links.new(current, mix.inputs["Color1"])
        links.new(tex.outputs["Color"], mix.inputs["Color2"])
        current = mix.outputs["Color"]

    links.new(current, bsdf.inputs["Base Color"])
    return mat, nt


def _stub_bake_targets(obj, mat, size: int):
    """本命以外のマテリアルにも焼き先を用意する。

    Blender の bake は**対象オブジェクトの全マテリアル**にアクティブな image node が
    あることを要求し、1つでも欠けると "No active image found" で落ちる。
    マテリアルごとにテクスチャが別（Rocketbox の body / head / opacity、
    VRoid の Face 7スロット）なので、共有の画像に焼くと UV 空間が重なって壊れる。
    捨てる前提の小さい画像を1枚ずつ割り当て、焼き終わったらノードごと外す。
    """
    stubs = []
    for m in obj.data.materials:
        if m is None or m is mat:
            continue
        m.use_nodes = True
        img = bpy.data.images.new(f"{m.name}_stub", 8, 8, alpha=False)
        node = m.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = img
        for n in m.node_tree.nodes:
            n.select = False
        node.select = True
        m.node_tree.nodes.active = node
        stubs.append((m, node, img))
    return stubs


def _drop_stubs(stubs) -> None:
    for m, node, img in stubs:
        m.node_tree.nodes.remove(node)
        bpy.data.images.remove(img)


def _as_emission(obj):
    """全マテリアルの出力を、ベースカラーをそのまま出す Emission に一時的に差し替える。

    焼き終わったら `_restore_surfaces` で元に戻す。
    """
    saved = []
    for m in obj.data.materials:
        if m is None or not m.use_nodes:
            continue
        nt = m.node_tree
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if out is None or bsdf is None or not out.inputs["Surface"].is_linked:
            continue
        prev = out.inputs["Surface"].links[0].from_socket
        em = nt.nodes.new("ShaderNodeEmission")
        em.location = (out.location.x - 200, out.location.y - 200)
        src = bsdf.inputs["Base Color"]
        if src.is_linked:
            nt.links.new(src.links[0].from_socket, em.inputs["Color"])
        else:
            em.inputs["Color"].default_value = tuple(src.default_value)
        nt.links.new(em.outputs[0], out.inputs["Surface"])
        saved.append((nt, out, prev, em))
    return saved


def _restore_surfaces(saved) -> None:
    for nt, out, prev, em in saved:
        nt.links.new(prev, out.inputs["Surface"])
        nt.nodes.remove(em)


def bake_to_image(obj, mat, nt, size: int):
    """ベースカラーを新しい画像へ焼く。焼き先は uv_layers[0]（元のUV）。"""
    me = obj.data
    if not me.uv_layers:
        raise SystemExit(f"{obj.name}: UV がありません")
    me.uv_layers[0].active_render = True
    me.uv_layers.active = me.uv_layers[0]

    img = bpy.data.images.new(f"{obj.name}_face", size, size, alpha=False)
    target = nt.nodes.new("ShaderNodeTexImage")
    target.image = img
    target.location = (200, -400)
    for n in nt.nodes:
        n.select = False
    target.select = True
    nt.nodes.active = target

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 1              # テクスチャ参照だけなので1で足りる
    sc.cycles.use_denoising = False
    sc.render.bake.use_pass_direct = False
    sc.render.bake.use_pass_indirect = False
    sc.render.bake.use_pass_color = True
    sc.render.bake.margin = max(8, size // 128)
    stubs = _stub_bake_targets(obj, mat, size)
    emits = _as_emission(obj)
    try:
        # **EMIT で焼く。DIFFUSE では BSDF の設定に結果が左右される。**
        # 実測: Rocketbox は Metallic=1.0 で入ってくる（FBX に PBR 情報が無く、
        # glTF の既定値がそのまま効く）ため、金属には拡散成分が無く
        # DIFFUSE bake が**全面黒**になった。EMIT はシェーダを経由せず
        # 発光色をそのまま書くので、BSDF の状態に依存しない。
        bpy.ops.object.bake(type="EMIT")
    finally:
        _restore_surfaces(emits)
        _drop_stubs(stubs)
    return img, target


def finalize_material(nt, bsdf_img, target_node):
    """焼いた画像を素のベースカラーに戻す。合成ノードは不要になる。"""
    nodes, links = nt.nodes, nt.links
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    for l in list(bsdf.inputs["Base Color"].links):
        links.remove(l)
    baked = nodes.new("ShaderNodeTexImage")
    baked.image = bsdf_img
    baked.location = (-400, 300)
    links.new(baked.outputs["Color"], bsdf.inputs["Base Color"])

    # 合成に使ったノードを全部消す。残すと glTF 側が
    # 「複数の tex image がある」と警告し、どれが採用されるか不定になる。
    keep = {baked, bsdf}
    keep |= {n for n in nodes if n.type == "OUTPUT_MATERIAL"}
    for n in list(nodes):
        if n not in keep:
            nodes.remove(n)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None, help="投影後の GLB（投影モードで必須）")
    ap.add_argument("--photo", default=None,
                    help="整列済み RGBA（顔領域がアルファ）。単一ビュー投影用")
    ap.add_argument("--view", action="append", default=[], metavar="AZIM:PATH",
                    help=("多視点投影。方位角:整列済みRGBA を繰り返し指定する。"
                          "例 --view=-30:l.png --view 0:f.png --view 30:r.png。"
                          "**負の方位角は --view=-30:... と = で繋ぐこと** — "
                          "空白区切りだと argparse がオプション名と誤認する。"
                          "正面(0)は最後に指定すること — 頬で重なる部分は"
                          "解像度の高い正面を採りたいため"))
    ap.add_argument("--azimuth", type=float, default=0.0,
                    help="--emit-align-render の方位角（0=正面）")
    ap.add_argument("--emit-align-render", default=None,
                    help="このパスに整列用の正面レンダを出して終了する")
    ap.add_argument("--frame", choices=("full", "head"), default="head",
                    help="head は頭部に寄せる。全身メッシュでは必須")
    ap.add_argument("--head-fraction", type=float, default=0.25,
                    help="肩の自動検出に失敗したときに使う割合（上端から）")
    ap.add_argument("--align-res", type=int, default=2048,
                    help="整列レンダの一辺。投影時も同じ値でなければならない")
    ap.add_argument("--bake-size", type=int, default=2048)
    ap.add_argument("--facing-min", type=float, default=0.15,
                    help="この内積未満の面には貼らない")
    ap.add_argument("--facing-max", type=float, default=0.55,
                    help="この内積以上で全面的に貼る")
    ap.add_argument("--only-materials", nargs="*", default=None,
                    help=("この文字列を名前に含むマテリアル／オブジェクトにだけ"
                          "投影する。省略すると全部に焼く"))
    ap.add_argument("--frame-material", default=None,
                    help="このマテリアルの面だけで画角を決める（--frame head のとき）。"
                         "肩のシルエット検出は体型に左右されるので、"
                         "**テンプレートではこちらを使う**")
    ap.add_argument("--frame-margin", type=float, default=None,
                    help="画角の余白。既定は --frame-material 指定時 1.45、"
                         "それ以外 1.25。狭いと顔検出が落ち、広いと解像度が下がる")
    ap.add_argument("--face-material", default=None,
                    help="写真を焼くマテリアル名（部分一致）。**複数スロットのメッシュでは必須**。"
                         "glTF のインポータは複数プリミティブを1オブジェクトに畳むので、"
                         "スロット0は顔とは限らない（VRoid の Face は FaceMouth、"
                         "Rocketbox は body がスロット0）")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    clear_scene()
    meshes = import_mesh(Path(args.input))
    if not meshes:
        raise SystemExit("メッシュが読み込めませんでした")

    center, radius, lo, hi = scene_bounds(meshes)
    cam = setup_front_camera(center, radius, lo, hi, meshes, args.frame,
                             args.head_fraction, args.azimuth,
                             frame_material=args.frame_material,
                             frame_margin=args.frame_margin)

    # world_to_camera_view はレンダ解像度のアスペクト比を使う。整列レンダと
    # 投影で同じ正方形にしないと顔が横に伸びて二重像になる。
    sc = bpy.context.scene
    sc.render.resolution_x = args.align_res
    sc.render.resolution_y = args.align_res
    sc.render.resolution_percentage = 100
    bpy.context.view_layer.update()

    if args.emit_align_render:
        flat = flatten_materials(meshes)
        setup_world_for_align()
        render_align_view(Path(args.emit_align_render), args.align_res)
        print(MARKER + " " + json.dumps(
            {"align_render": args.emit_align_render, "frame": args.frame,
             "azimuth": args.azimuth, "align_res": args.align_res,
             "flattened_materials": flat}, ensure_ascii=False))
        return

    if not args.out or not (args.photo or args.view):
        raise SystemExit("投影モードでは --out と、--photo か --view が要ります")

    # (方位角, パス) の並び。--photo は --view 0:PATH の省略形。
    specs = []
    for v in args.view:
        azim, _, path = v.partition(":")
        if not path:
            raise SystemExit(f"--view の書式は AZIM:PATH です: {v}")
        specs.append((float(azim), path))
    if args.photo:
        specs.append((args.azimuth, args.photo))
    if not specs:
        raise SystemExit("投影するビューがありません")

    loaded = []
    for i, (azim, path) in enumerate(specs):
        img = bpy.data.images.load(str(Path(path).resolve()))
        img.colorspace_settings.name = "sRGB"
        uv_name = f"faceproj{i}"
        vcam = setup_front_camera(center, radius, lo, hi, meshes, args.frame,
                                  args.head_fraction, azim, name=f"ProjCam{i}",
                                  frame_material=args.frame_material,
                                  frame_margin=args.frame_margin)
        loaded.append((img, uv_name, azim, vcam))

    # 投影する対象を絞れるようにする。ICT-FaceKit の頭はマテリアルごとに
    # 12 オブジェクトに分かれていて、**眼球・歯・睫毛には写真を貼らないのが正しい**
    # （それぞれ固有の質感を保つ）。指定が無ければ従来どおり全オブジェクトに焼く。
    def wanted(o) -> bool:
        # マテリアルを持たないメッシュは投影対象になり得ない。
        # **Blender 4.2 の glTF インポータは読み込みのたびに 42頂点の `Icosphere` を
        # 作る**（subject_a.glb でも Rocketbox でも再現する）。素通ししていると
        # マテリアル選択で落ちるので、ここで弾く。
        if not [m for m in o.data.materials if m]:
            return False
        # **顔マテリアルを持たないメッシュは対象外にする（落とさない）。**
        # 髪を生成メッシュから切り出して別メッシュとして載せると
        # （blender/add_hair.py）、そのメッシュは `Material_0` しか持たない。
        # 以前はここで pick_material が例外を投げてパイプラインが止まった。
        if args.face_material and not any(
                args.face_material in m.name for m in o.data.materials if m):
            return False
        if not args.only_materials:
            return True
        names = [m.name for m in o.data.materials if m] + [o.name]
        return any(pat in n for pat in args.only_materials for n in names)

    targets = [o for o in meshes if wanted(o)]
    if not targets:
        raise SystemExit(f"--only-materials {args.only_materials} に一致する"
                         f"オブジェクトがありません: {[o.name for o in meshes]}")

    done = []
    for obj in targets:
        for img, uv_name, azim, vcam in loaded:
            add_projection_uv(obj, vcam, uv_name)
        face_mat = pick_material(obj, args.face_material)
        mat, nt = build_projection_material(
            obj, [(img, uv, az) for img, uv, az, _ in loaded],
            args.facing_min, args.facing_max, material=face_mat)
        img, target = bake_to_image(obj, mat, nt, args.bake_size)
        finalize_material(nt, img, target)
        # 投影用 UV は成果物に残さない（glTF のサイズと混乱の元）
        for _, uv_name, _, _ in loaded:
            uv = obj.data.uv_layers.get(uv_name)
            if uv:
                obj.data.uv_layers.remove(uv)
        done.append({"object": obj.name, "material": face_mat.name,
                     "baked": img.name, "size": [img.size[0], img.size[1]]})
    skipped = [o.name for o in meshes if o not in targets]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    exported = [o for o in meshes if [m for m in o.data.materials if m]]
    for o in exported:
        o.select_set(True)
    # アーマチュアも一緒に出す（選択に含めないとスキンが落ちる）
    for o in bpy.data.objects:
        if o.type == "ARMATURE":
            o.select_set(True)
    # **モーフ法線を書き出さない。** Blender の既定は True で、テンプレートが持つ
    # 175個のモーフ全部に NORMAL 差分が付いて容量が倍になる（実測 10.0 → 20.9MB）。
    # three.js は無くても動く。他の blender/*.py も同じ設定にしてある。
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB",
                              use_selection=True, export_image_format="AUTO",
                              export_morph=True, export_morph_normal=False,
                              export_morph_tangent=False)

    report = {"input": args.input,
              "views": [{"azimuth": a, "photo": pth} for a, pth in specs],
              "output": str(out),
              "frame": args.frame, "head_fraction": args.head_fraction,
              "align_res": args.align_res, "bake_size": args.bake_size,
              "facing_range": [args.facing_min, args.facing_max],
              "objects": done, "skipped": skipped}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(MARKER + " " + json.dumps({"output": str(out), "objects": len(done)},
                                    ensure_ascii=False))


if __name__ == "__main__":
    main()
