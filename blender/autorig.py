"""工程8: テンプレート転写による自動リギング（UniRig 導入までの経路）。

UniRig が本命だが 13GB 必要で、現在のディスク残量では入れられない。一方で
image-to-3D の出力は T ポーズになるため、既知の VRM Humanoid アーマチュアを
スケールして被せ、Blender の自動ウェイト（heat map）で結合する方式が成立する。

テンプレート由来のアーマチュアは humanoid 割当を最初から持っているので、
そのまま VRM として書き出せる。UniRig を入れたらこの工程を差し替える。

限界（正直に記録する）:
  - 骨の位置は「身長と腕幅を合わせた相似変形」で決めるため、体型が
    テンプレートと大きく異なるモデルでは関節位置がずれる
  - 指の骨はテンプレートのまま。生成メッシュの指形状には追従しない
  - 髪・スカートの揺れもの(spring bone)は設定しない

  blender --background --factory-startup --python autorig.py -- \
      --input <mesh.glb> --template <template.vrm> --out <rigged.blend|glb> \
      --report <rig.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def ensure_vrm_addon() -> None:
    import addon_utils
    for m in addon_utils.modules():
        if "vrm" in m.__name__.lower():
            try:
                addon_utils.enable(m.__name__, default_set=True, persistent=True)
            except Exception:
                pass


def import_any(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    ext = path.suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".vrm":
        ensure_vrm_addon()
        bpy.ops.import_scene.vrm(filepath=str(path))
    else:
        raise SystemExit(f"未対応の拡張子: {ext}")
    return [o for o in bpy.data.objects if o not in before]


def world_bounds(objs) -> tuple[Vector, Vector]:
    lo = Vector((float("inf"),) * 3)
    hi = Vector((float("-inf"),) * 3)
    for o in objs:
        if o.type != "MESH":
            continue
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for i in range(3):
                lo[i] = min(lo[i], w[i])
                hi[i] = max(hi[i], w[i])
    return lo, hi


def normalize_target(meshes: list[bpy.types.Object], target_height: float) -> dict:
    """生成メッシュを「原点が足元・身長 target_height」に正規化する。

    image-to-3D の出力はスケールも原点も不定。TRELLIS.2 は単位AABB
    （高さ1.0・原点が中心）で返すため、VRM の「1.0 = 1メートル・原点は足元」
    に合わせないと、リギングもVRM検証も破綻する。

    注意: 親エンプティに変換を掛けて子メッシュへ transform_apply しても
    親の変換は焼き付かない（子自身の matrix_basis しか適用されない）。
    各メッシュの matrix_world を直接操作して適用する。
    """
    from mathutils import Matrix

    lo, hi = world_bounds(meshes)
    height = hi.z - lo.z
    if height <= 1e-6:
        raise SystemExit("メッシュの高さが0です")
    scale = target_height / height

    # 1) スケールを各メッシュへ直接適用
    S = Matrix.Scale(scale, 4)
    for m in meshes:
        m.matrix_world = S @ m.matrix_world
    bpy.context.view_layer.update()

    # 2) スケール後の実寸から、足元を z=0・左右前後の中心を原点へ寄せる
    lo2, hi2 = world_bounds(meshes)
    offset = Vector((-(lo2.x + hi2.x) / 2.0, -(lo2.y + hi2.y) / 2.0, -lo2.z))
    T = Matrix.Translation(offset)
    for m in meshes:
        m.matrix_world = T @ m.matrix_world
    bpy.context.view_layer.update()

    # 3) 変換をメッシュデータへ焼き付ける（アーマチュア結合の前提）
    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.select_all(action="DESELECT")

    lo3, hi3 = world_bounds(meshes)
    result = {
        "source_height": round(height, 5),
        "scale_applied": round(scale, 5),
        "normalized_height": round(hi3.z - lo3.z, 5),
        "normalized_foot_z": round(lo3.z, 5),
    }
    # 正規化が効いていないまま進むと VRM のスケール検証で必ず落ちるので、ここで止める
    if abs(result["normalized_height"] - target_height) > target_height * 0.02:
        raise SystemExit(
            f"正規化に失敗しました: 目標 {target_height}m に対し "
            f"{result['normalized_height']}m")
    if abs(result["normalized_foot_z"]) > 0.01:
        raise SystemExit(
            f"足元が原点に来ていません: foot_z={result['normalized_foot_z']}")
    return result


def fit_armature(arm: bpy.types.Object, meshes: list[bpy.types.Object]) -> dict:
    """テンプレートのアーマチュアを対象メッシュの寸法に合わせる。

    身長で相似スケールし、足元を原点に合わせる。腕幅は T ポーズなら
    身長比でおおむね決まるので、ここでは追加補正しない
    （過剰に合わせると関節が体外へ出る方が害が大きい）。
    """
    m_lo, m_hi = world_bounds(meshes)
    mesh_h = m_hi.z - m_lo.z

    # アーマチュアの高さはボーンのワールド座標から取る
    bpy.context.view_layer.update()
    zs = []
    for b in arm.data.bones:
        zs.append((arm.matrix_world @ b.head_local).z)
        zs.append((arm.matrix_world @ b.tail_local).z)
    arm_h = max(zs) - min(zs)
    if arm_h <= 1e-6:
        raise SystemExit("テンプレートアーマチュアの高さが0です")

    s = mesh_h / arm_h
    arm.scale = (s, s, s)
    bpy.context.view_layer.update()

    zs2 = [(arm.matrix_world @ b.head_local).z for b in arm.data.bones]
    zs2 += [(arm.matrix_world @ b.tail_local).z for b in arm.data.bones]
    arm.location = (arm.location.x, arm.location.y,
                    arm.location.z + (m_lo.z - min(zs2)))
    bpy.context.view_layer.update()

    return {"template_height": round(arm_h, 5), "mesh_height": round(mesh_h, 5),
            "armature_scale": round(s, 5)}


def fit_arms(arm: bpy.types.Object, meshes: list[bpy.types.Object],
             also: list[bpy.types.Object] | None = None) -> dict:
    """腕の骨を、対象メッシュの実際の腕の長さに合わせて X 方向に伸ばす。

    fit_armature の相似スケールは身長だけを見るので、**腕幅／身長の比が
    テンプレートと違うと腕の骨が丸ごと内側に寄る**。実測（docs/05 §37）:

        VRoid テンプレート  幅 1.389m / 身長 1.615m = 0.860
        image-to-3D の T ポーズ  幅 1.577m / 身長 1.600m = 0.986

    ヒトの腕を広げた幅はおおむね身長に等しい（比 ≒ 1.0）ので、image-to-3D の
    出力の方が解剖学的に正しく、テンプレートが 15% 短い。身長で合わせると
    指の骨の塊が **前腕の上に乗り**、自動ウェイトが前腕・手首・手のひらを
    指先の骨に割り当てる。ポーズを付けると袖が裂け、手が潰れる。

    腕は T ポーズで X 軸に伸びているので、肩関節を支点に X 座標だけを
    引き伸ばせば足りる。骨の向き（roll）は変えない。
    """
    m_lo, m_hi = world_bounds(meshes)
    tip_mesh = max(abs(m_lo.x), abs(m_hi.x))

    bpy.context.view_layer.update()
    pivots, tip_arm = {}, 0.0
    for b in arm.data.bones:
        for p in (arm.matrix_world @ b.head_local, arm.matrix_world @ b.tail_local):
            tip_arm = max(tip_arm, abs(p.x))
        n = b.name.lower()
        if "upperarm" in n and "roll" not in n and "sec" not in n:
            side = "L" if (arm.matrix_world @ b.head_local).x >= 0 else "R"
            pivots[side] = (arm.matrix_world @ b.head_local).x
    if not pivots or tip_arm <= 1e-6:
        return {"status": "skipped", "reason": "上腕ボーンが見つからない"}

    pivot = sum(abs(v) for v in pivots.values()) / len(pivots)
    if tip_arm - pivot <= 1e-6:
        return {"status": "skipped", "reason": "腕の長さが0"}
    k = (tip_mesh - pivot) / (tip_arm - pivot)
    # 極端な値はテンプレート取り違えなどの異常。合わせない方が安全。
    if not (0.6 <= k <= 1.8):
        return {"status": "skipped", "reason": f"補正係数が範囲外 k={k:.3f}",
                "k": round(k, 4)}

    prev = arm.mode
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    moved = 0
    inv = arm.matrix_world.inverted()
    for eb in arm.data.edit_bones:
        for attr in ("head", "tail"):
            p = arm.matrix_world @ getattr(eb, attr)
            if abs(p.x) <= pivot:
                continue                      # 肩より内側（胴・脚・頭）は動かさない
            s = 1.0 if p.x >= 0 else -1.0
            p.x = s * (pivot + (abs(p.x) - pivot) * k)
            setattr(eb, attr, inv @ p)
            moved += 1
    bpy.ops.object.mode_set(mode=prev if prev in {"OBJECT", "EDIT"} else "OBJECT")

    # テンプレートのメッシュを使う経路では、メッシュも骨と同じだけ伸ばさないと
    # 近接転送で「こちらの前腕」に「向こうの手」が当たる。
    for o in also or []:
        mw = o.matrix_world
        imw = mw.inverted()
        for v in o.data.vertices:
            p = mw @ v.co
            if abs(p.x) <= pivot:
                continue
            sgn = 1.0 if p.x >= 0 else -1.0
            p.x = sgn * (pivot + (abs(p.x) - pivot) * k)
            v.co = imw @ p
        o.data.update()
    bpy.context.view_layer.update()

    return {"status": "ok", "k": round(k, 4),
            "pivot_x": round(pivot, 4),
            "arm_tip_before": round(tip_arm, 4),
            "mesh_tip": round(tip_mesh, 4),
            "endpoints_moved": moved}


def disable_garment_bones(arm: bpy.types.Object,
                          patterns=("Tops", "Bottoms", "Skirt", "Coat")) -> dict:
    """テンプレートの**衣装用**二次ボーンだけを変形から外す。

    VRoid のリグは二次ボーン `J_Sec_*` を 83本持つが、中身は用途が違う:

        Hair   59本   髪（頭に付いていて、頭と一緒に動く）
        Tops   20本   トップスのパネル（**独立した衣装メッシュ**が前提）
        Bust    4本   胸

    image-to-3D の出力は衣服が体のシルエットに焼き込まれた単一メッシュなので、
    Tops に対応する衣装メッシュが存在しない。それでも自動ウェイトはこれを候補に
    含め、実測（docs/05 §37）では**肩の頂点の 78.7% が 0.5 超**をこの骨に持って
    いた。腕を動かすと肩と上腕が別キャラのトップス用ボーンに従い、赤い板が飛び出す。

    **Hair は外さない。** 一度 `J_Sec_` を全部外したところ、顔の頂点が拠り所を
    失って上腕ボーンに拾われ（顔上部で R_UpperArm 14%）、腕を回すと顔が崩れた。
    髪の骨は頭の子なので頭と一緒に動き、顔を保持する役として機能している。

    `use_deform=False` にすると自動ウェイトの候補から外れるだけで、骨自体も
    階層も残るので VRM 出力は壊れない。
    """
    off = []
    for b in arm.data.bones:
        if not b.name.startswith("J_Sec_"):
            continue
        if any(pat in b.name for pat in patterns):
            b.use_deform = False
            off.append(b.name)
    return {"disabled": len(off), "total_bones": len(arm.data.bones),
            "patterns": list(patterns), "sample": off[:5]}


def weighted_ratio(meshes: list[bpy.types.Object]) -> tuple[int, int]:
    """ウェイトが1つでも入っている頂点の数を数える。"""
    nz = tot = 0
    for m in meshes:
        for v in m.data.vertices:
            tot += 1
            if any(g.weight > 1e-6 for g in v.groups):
                nz += 1
    return nz, tot


def _parent(arm, meshes, mode: str) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type=mode)


# エンベロープ由来のウェイトをならす回数。8 では肩と胸が裂けた。実測で
# 伸び 3倍超の辺は 8→1574 / 24→1276 / 48→1140 / 96→1038 と減り、48 で
# ほぼ頭打ち。どの値でもポーズの変形量は 0.4945 のままで、なまり過ぎない
# （docs/05-measurements.md §37）。--weight-smooth で上書きできる。
SMOOTH_REPEAT = 48
FOLD_FINGERS = True    # --keep-fingers で無効化


def smooth_weights(meshes: list[bpy.types.Object], factor: float = 0.5,
                   repeat: int | None = None) -> int:
    """エンベロープのウェイトをならす。

    エンベロープは骨からの距離だけで決まるので、境界が階段状になり、
    曲げると衣装が裂ける。近傍頂点間で平滑化すると変形が滑らかになる。
    heat map と違って解が必ず出る代わりに、この後処理が要る。
    """
    repeat = SMOOTH_REPEAT if repeat is None else repeat
    done = 0
    for m in meshes:
        if not m.vertex_groups:
            continue
        bpy.ops.object.select_all(action="DESELECT")
        m.select_set(True)
        bpy.context.view_layer.objects.active = m
        try:
            bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
            bpy.ops.object.vertex_group_smooth(
                group_select_mode="ALL", factor=factor, repeat=repeat)
            bpy.ops.object.mode_set(mode="OBJECT")
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"ウェイト平滑化に失敗 ({m.name}): {e}", flush=True)
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
    return done


def calibrate_envelopes(arm: bpy.types.Object, k: float = 1.35) -> dict:
    """短い骨のエンベロープ影響範囲だけを絞る。

    テンプレート VRM は全ての骨に `envelope_distance = 0.25` という**固定値**を
    持つ。腕・脚・前腕にとってこれは妥当で、実測でも個別に動かすと無傷だった。
    問題は短い骨で、首（長さ 0.0737）には自分の 3.4倍、指先には 26倍にあたる。
    実測: 首を 25° 振ると首に巻き込まれた**ジャケットの胴が裂けた**。

    そこで**上限だけ**を骨の長さに比例させる（`min(元の値, 長さ×k)`）。
    長い骨は 0.25 のまま変わらず、首や指だけが絞られる。
    全部を長さ比例で置き換える案も試したが、腕の影響範囲まで狭まって
    袖が細い筋状に伸びた（過補正）。**効いていたものは触らない。**

    k は 1.35 / 0.8 / 0.5 を実測して決めた。上腕は長さ約 0.25 なので
    k=1.35 では上限 0.34 > 0.25 となり元の値のまま、k=0.8 以下だと
    0.2 に絞られて袖が裂ける。首（長さ 0.0737）は k=1.35 で 0.25 → 0.0995 に
    絞られる。腕+脚のポーズはこれで無傷になった。
    """
    changed = []
    for b in arm.data.bones:
        L = (b.tail_local - b.head_local).length
        if L <= 1e-6:
            continue
        cap = L * k
        if b.envelope_distance > cap:
            changed.append((b.name, round(b.envelope_distance, 4), round(cap, 4)))
            b.envelope_distance = cap
    return {"tightened": len(changed), "k": k,
            "examples": changed[:5]}


def _build_proxy(meshes: list[bpy.types.Object], voxel: float):
    """ウェイトを解くためだけの、連結した使い捨てメッシュを作る。

    image-to-3D の出力は連結成分が数千に分かれている。距離ベースの
    エンベロープだと隣り合う破片が別々の骨に付き、曲げたときに衣装が裂ける。

    そこでボクセルリメッシュで**1つに繋がった代理メッシュ**を作り、そこで
    ウェイトを解いてから実メッシュへ空間的に転送する。UV を壊す操作は
    代理側だけで完結するので、本体のテクスチャは無傷のまま
    （頂点マージが UV を壊す件は docs/05-measurements.md §9）。
    """
    bpy.ops.object.select_all(action="DESELECT")
    copies = []
    for m in meshes:
        c = m.copy()
        c.data = m.data.copy()
        bpy.context.collection.objects.link(c)
        copies.append(c)
        c.select_set(True)
    bpy.context.view_layer.objects.active = copies[0]
    if len(copies) > 1:
        bpy.ops.object.join()
    proxy = bpy.context.view_layer.objects.active
    proxy.name = "WeightProxy"

    # 頂点グループは代理側で作り直すので、複製元のものは消しておく
    proxy.vertex_groups.clear()
    mod = proxy.modifiers.new("Remesh", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel
    mod.adaptivity = 0.0
    bpy.context.view_layer.objects.active = proxy
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return proxy


def _transfer_weights(proxy, meshes: list[bpy.types.Object]) -> None:
    """代理メッシュのウェイトを実メッシュへ最近傍で転送する。"""
    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
    proxy.select_set(True)
    bpy.context.view_layer.objects.active = proxy      # active が転送元
    bpy.ops.object.data_transfer(
        use_reverse_transfer=False,
        data_type="VGROUP_WEIGHTS",
        vert_mapping="POLYINTERP_NEAREST",   # 面上の最近傍点で補間する
        layers_select_src="ALL",
        layers_select_dst="NAME",
        mix_mode="REPLACE",
    )


def _bind_via_template(arm, meshes: list[bpy.types.Object],
                       tmpl_meshes: list[bpy.types.Object]) -> dict:
    """テンプレートの体メッシュが持つウェイトを近接で転送する。

    エンベロープは骨からの距離だけでウェイトを決めるので、テンプレートに
    あってこちらに対応物がない骨（髪・衣装・指）が近くの面を勝手に拾う。
    実測（§37）では**髪の骨が胸と肩のウェイトを持ち**、体を動かすとそこだけ
    頭に追従して裂けた。骨を除外しても被害が頭へ移るだけで総量は減らなかった
    （エンベロープのまま除外パターンを4通り試して 5209〜6239本で横ばい）。

    VRoid のテンプレートは**人が作ったウェイト**を持っている。両者とも T ポーズ
    の人体で、身長と腕幅を合わせてあるので、面の最近傍で引いてくれば
    「腕は腕から、胴は胴から」正しい配分が入る。骨からの距離ではなく
    **人体のどこか**で決まるのが本質的な違い。
    """
    if not tmpl_meshes:
        return {"mode": "TEMPLATE_TRANSFER", "ok": False, "reason": "テンプレートメッシュがない"}
    try:
        # 転送元は1つにまとめる（VRoid は顔・体・髪が別オブジェクト）
        bpy.ops.object.select_all(action="DESELECT")
        for o in tmpl_meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = tmpl_meshes[0]
        if len(tmpl_meshes) > 1:
            bpy.ops.object.join()
        src = bpy.context.view_layer.objects.active
        # アーマチュア変形が乗ったままだと転送元の姿勢が狂う
        src.parent = None
        for mod in list(src.modifiers):
            if mod.type == "ARMATURE":
                src.modifiers.remove(mod)

        bpy.ops.object.select_all(action="DESELECT")
        for m in meshes:
            m.select_set(True)
        src.select_set(True)
        bpy.context.view_layer.objects.active = src
        bpy.ops.object.data_transfer(
            use_reverse_transfer=False,
            data_type="VGROUP_WEIGHTS",
            vert_mapping="POLYINTERP_NEAREST",
            layers_select_src="ALL",
            layers_select_dst="NAME",
            mix_mode="REPLACE",
        )
        _parent(arm, meshes, "ARMATURE_NAME")
        nz, tot = weighted_ratio(meshes)
        src_verts = len(src.data.vertices)
        # 転送元はここで捨てる。残すと成果物に同梱されてしまう（実測 15MB→55MB）。
        bpy.data.objects.remove(src, do_unlink=True)
        if not tot or nz / tot < 0.5:
            return {"mode": "TEMPLATE_TRANSFER", "ok": False,
                    "reason": f"転送後のウェイトが不足 {nz}/{tot}"}
        folded = fold_to_core(arm, meshes, fold_fingers=FOLD_FINGERS)
        smoothed = smooth_weights(meshes)
        return {"mode": "TEMPLATE_TRANSFER", "ok": True, "fingers": folded,
                "source_vertices": src_verts,
                "weighted_vertices": nz, "vertices": tot,
                "smoothed_meshes": smoothed}
    except Exception as e:  # noqa: BLE001
        return {"mode": "TEMPLATE_TRANSFER", "ok": False, "reason": f"例外: {e}"}


FINGER_KEYS = ("Index", "Middle", "Ring", "Little", "Thumb")
KEEP_PREFIXES = ("J_Bip_", "J_Roll_", "J_Adj_")
GARMENT_KEYS = ("Tops", "Bottoms", "Skirt", "Coat")
# たたむ範囲。**広げるほど悪くなる**ことを実測で確認している（§37）:
#   fingers 251本 / garment 371本 / 全部（ひねり骨も）214本だが見た目は最悪
# ひねり骨や補助骨は、こちらのメッシュにも意味のある変形を与えている。
FOLD_POLICY = "fingers"


def fold_to_core(arm: bpy.types.Object, meshes: list[bpy.types.Object],
                 fold_fingers: bool = True) -> dict:
    """本体の骨以外に付いたウェイトを、親の本体骨へたたむ。

    テンプレート転送はウェイトを**名前で**写すので、テンプレートが衣装・髪・
    補助のために持っている骨のウェイトもそのまま入る。それらはこちらのメッシュに
    対応物がないので、動かすと破綻する。気配の実姿勢を再現して、破断した頂点を
    支配していた骨を数えたら（§37）:

        J_Aim_R_TopsUpperArm  69   ← 衣装用の補助骨
        J_Sec_Hair2_10        40   ← 髪
        J_Roll_L_Hand         32   ← 手のひねり
        J_Aim_L_Shoulder      30

    いずれも「独立した衣装メッシュ／髪メッシュがある」前提の骨。本体の骨
    （VRoid の `J_Bip_*`）だけを残し、他は親を辿って最初に見つかった本体骨へ
    加算する。骨は消さないので humanoid の割当も VRM 出力も変わらない。

    指も同じ扱いにする（`fold_fingers`）。image-to-3D の手は指の割れていない
    塊で、テンプレートの指の骨は40本すべてがメッシュの外に浮いている。それでも
    KEHAI の GestureLSM は指を 120〜132° 曲げてくるので、たたまないと棘状に
    飛び散る。指は動かなくなるが、**動かない塊のほうが飛び散る塊より正しい**。
    """
    def is_core(name: str) -> bool:
        """たたまずに残す骨か。

        残すのは本体（`J_Bip_`）とひねり配分（`J_Roll_` / `J_Adj_`）。
        ひねり骨まで巻き込むと腕のひねりが1関節に集中して上腕が裂ける
        （実測で上腕の破断が 12 → 171 に悪化）。KEHAI 側も
        「ひねり補助が無いと1関節でせん断する」と警告している。

        たたむのは、対応する独立メッシュがあることを前提にした骨:
        二次ボーン（`J_Sec_`）、衣装用の補助（名前に Tops/Bottoms/Skirt/Coat）、
        末端（`_end`）、そして指。
        """
        if fold_fingers and any(k in name for k in FINGER_KEYS):
            return False
        if FOLD_POLICY == "fingers":
            return True
        if any(k in name for k in GARMENT_KEYS):
            return False
        if name.endswith("_end"):
            return False
        return name.startswith(KEEP_PREFIXES)

    # 各ボーンについて、たたみ先（自分か、親を辿って最初の本体骨）を決める
    target: dict[str, str] = {}
    for b in arm.data.bones:
        cur = b
        while cur is not None and not is_core(cur.name):
            cur = cur.parent
        if cur is not None and cur.name != b.name:
            target[b.name] = cur.name

    folded, moved = 0, 0
    for m in meshes:
        vgs = m.vertex_groups
        plan = {g.name: target[g.name] for g in vgs
                if g.name in target and target[g.name] in vgs}
        if not plan:
            continue
        acc: dict[int, dict[str, float]] = {}
        for v in m.data.vertices:
            add: dict[str, float] = {}
            for ge in v.groups:
                gname = vgs[ge.group].name
                if gname in plan:
                    add[plan[gname]] = add.get(plan[gname], 0.0) + ge.weight
            if add:
                base = {vgs[ge.group].name: ge.weight for ge in v.groups}
                acc[v.index] = {k: min(1.0, base.get(k, 0.0) + w) for k, w in add.items()}
        for vi, ws in acc.items():
            for gname, w in ws.items():
                vgs[gname].add([vi], w, "REPLACE")
            moved += 1
        for gname in plan:
            vgs.remove(vgs[gname])
            folded += 1
    return {"folded_groups": folded, "vertices_touched": moved,
            "fingers_folded": fold_fingers}


def _bind_via_proxy(arm, meshes: list[bpy.types.Object],
                    voxel: float = 0.012) -> dict:
    """連結した代理メッシュでウェイトを解き、実メッシュへ転送して結合する。"""
    proxy = None
    try:
        proxy = _build_proxy(meshes, voxel)
        pv = len(proxy.data.vertices)

        # 代理は1つに繋がっているので heat map が解ける見込みがある。
        # 駄目ならエンベロープ。どちらも代理側の話なので本体は無傷。
        used = None
        for pmode in ("ARMATURE_AUTO", "ARMATURE_ENVELOPE"):
            if pmode == "ARMATURE_ENVELOPE":
                calibrate_envelopes(arm)
            _parent(arm, [proxy], pmode)
            nz, tot = weighted_ratio([proxy])
            if tot and nz / tot >= 0.5:
                used = pmode
                break
        if used is None:
            return {"mode": "PROXY_AUTO", "ok": False,
                    "reason": f"代理メッシュにもウェイトが入りません (verts={pv})"}

        # 代理は骨に結ばれたままだと転送先の判定が狂うので、親子だけ解く
        proxy.parent = None
        for mod in list(proxy.modifiers):
            if mod.type == "ARMATURE":
                proxy.modifiers.remove(mod)

        _transfer_weights(proxy, meshes)
        # 実メッシュ側にアーマチュアを付ける（ウェイトは転送済みなので NAME でよい）
        _parent(arm, meshes, "ARMATURE_NAME")
        nz, tot = weighted_ratio(meshes)
        if not tot or nz / tot < 0.5:
            return {"mode": "PROXY_AUTO", "ok": False,
                    "reason": f"転送後のウェイトが不足 {nz}/{tot}"}
        smoothed = smooth_weights(meshes, factor=0.5, repeat=max(4, SMOOTH_REPEAT // 2))
        return {"mode": "PROXY_AUTO", "ok": True, "proxy_mode": used,
                "voxel_size": voxel, "proxy_vertices": pv,
                "weighted_vertices": nz, "vertices": tot,
                "smoothed_meshes": smoothed}
    except Exception as e:  # noqa: BLE001
        return {"mode": "PROXY_AUTO", "ok": False, "reason": f"例外: {e}"}
    finally:
        if proxy is not None and proxy.name in bpy.data.objects:
            bpy.data.objects.remove(proxy, do_unlink=True)


def bind(arm: bpy.types.Object, meshes: list[bpy.types.Object],
         tmpl_meshes: list[bpy.types.Object] | None = None) -> dict:
    """自動ウェイトでメッシュをアーマチュアへ結合する。

    **ウェイトが実際に入ったかを必ず数える。** `ARMATURE_AUTO`（heat map）は
    解が出なくても**例外を投げずに全ウェイト0のまま完了する**。モディファイアの
    有無だけで成功判定していたため、頂点グループ167個・Armatureモディファイア付き
    ・deform骨167本という「見た目は完璧」な状態で、実際には1頂点もスキンされて
    いない VRM を出し続けていた（実測 0/121,228）。glTF 側は「スキンする対象が
    無い」と判断して skin を書かないので、`skins=0` の動かないVRMになる。

    image-to-3D の出力は連結成分が数千に分かれ非多様体エッジも数万あるため、
    heat map は原理的に解けない。エンベロープなら距離で決まるので通る
    （同じメッシュで 111,140/121,228 = 91.7%）。
    """
    attempts = []
    for mode in ("TEMPLATE_TRANSFER", "ARMATURE_AUTO", "PROXY_AUTO", "ARMATURE_ENVELOPE"):
        if mode == "ARMATURE_ENVELOPE":
            calibrate_envelopes(arm)
        if mode == "TEMPLATE_TRANSFER":
            r = _bind_via_template(arm, meshes, tmpl_meshes or [])
            attempts.append(r)
            if r.get("ok"):
                return {"bind_method": "TEMPLATE_TRANSFER",
                        "meshes_bound": len(meshes), "meshes_total": len(meshes),
                        "weighted_vertices": r["weighted_vertices"],
                        "vertices": r["vertices"],
                        "weighted_ratio": round(r["weighted_vertices"] / r["vertices"], 4),
                        "smoothed_meshes": r.get("smoothed_meshes", 0),
                        "attempts": attempts}
            print(f"TEMPLATE_TRANSFER: {r.get('reason')} — 次の方式を試します", flush=True)
            continue
        if mode == "PROXY_AUTO":
            r = _bind_via_proxy(arm, meshes)
            attempts.append(r)
            if r.get("ok"):
                bound = sum(1 for m in meshes
                            if any(mod.type == "ARMATURE" for mod in m.modifiers))
                return {"bind_method": mode, "meshes_bound": bound,
                        "meshes_total": len(meshes),
                        "weighted_vertices": r["weighted_vertices"],
                        "vertices": r["vertices"],
                        "weighted_ratio": round(r["weighted_vertices"] / r["vertices"], 4),
                        "proxy": {k: r[k] for k in ("voxel_size", "proxy_vertices",
                                                    "proxy_mode")},
                        "smoothed_meshes": r.get("smoothed_meshes", 0),
                        "attempts": attempts}
            print(f"PROXY_AUTO: {r.get('reason')} — 次の方式を試します", flush=True)
            continue
        try:
            _parent(arm, meshes, mode)
        except Exception as e:  # noqa: BLE001
            attempts.append({"mode": mode, "error": str(e)})
            continue
        nz, tot = weighted_ratio(meshes)
        attempts.append({"mode": mode, "weighted_vertices": nz, "vertices": tot})
        if tot and nz / tot >= 0.5:
            smoothed = smooth_weights(meshes) if mode == "ARMATURE_ENVELOPE" else 0
            bound = sum(1 for m in meshes
                        if any(mod.type == "ARMATURE" for mod in m.modifiers))
            return {"bind_method": mode, "meshes_bound": bound,
                    "meshes_total": len(meshes),
                    "weighted_vertices": nz, "vertices": tot,
                    "weighted_ratio": round(nz / tot, 4),
                    "smoothed_meshes": smoothed,
                    "attempts": attempts}
        print(f"{mode}: ウェイトが入った頂点 {nz}/{tot} — 次の方式を試します",
              flush=True)

    # どちらも駄目ならここで止める。動かないVRMを「成功」として出さない。
    raise SystemExit(
        "スキニングに失敗しました（どの方式でもウェイトが入りません）: "
        + json.dumps(attempts, ensure_ascii=False))


def main() -> None:
    global SMOOTH_REPEAT, FOLD_FINGERS, FOLD_POLICY
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--fold-policy", default="fingers",
                    choices=("garment", "fingers"),
                    help="たたむ範囲。garment=衣装・二次・末端も、fingers=指だけ")
    ap.add_argument("--keep-fingers", action="store_true",
                    help=("指のウェイトを手にたたまない。生成メッシュの手は指が"
                          "割れていない塊なので、既定ではたたんで剛体にする"))
    ap.add_argument("--no-deform", default="",
                    help=("変形に使わないボーンの部分一致パターン（カンマ区切り）。"
                          "テンプレートにあってメッシュに対応物がない骨を外す"))
    ap.add_argument("--weight-smooth", type=int, default=SMOOTH_REPEAT,
                    help=("エンベロープ由来のウェイトをならす回数。エンベロープは"
                          "骨からの距離だけで決まるため境界が階段状になり、曲げると"
                          "衣装が裂ける（docs/05 §37）"))
    ap.add_argument("--height", type=float, default=1.6,
                    help="正規化後の身長(m)。VRM は 1.0 = 1m")
    args = ap.parse_args(argv)
    SMOOTH_REPEAT = max(0, args.weight_smooth)
    FOLD_FINGERS = not args.keep_fingers
    FOLD_POLICY = args.fold_policy

    bpy.ops.wm.read_homefile(use_empty=True)
    ensure_vrm_addon()

    report: dict = {"input": args.input, "template": args.template}

    # 1) 対象メッシュ
    target_objs = import_any(Path(args.input))
    meshes = [o for o in target_objs if o.type == "MESH"]
    if not meshes:
        raise SystemExit("対象にメッシュがありません")
    report["target_meshes"] = len(meshes)
    report["normalize"] = normalize_target(meshes, args.height)

    # 2) テンプレート（アーマチュアだけ使う）
    tmpl_objs = import_any(Path(args.template))
    tmpl_arms = [o for o in tmpl_objs if o.type == "ARMATURE"]
    if not tmpl_arms:
        raise SystemExit("テンプレートにアーマチュアがありません")
    arm = tmpl_arms[0]
    # テンプレートの体メッシュは**捨てない**。VRoid の手作りウェイトを近接転送
    # する経路（_bind_via_template）で使う。エンベロープより桁違いに素性が良い。
    # 髪メッシュは転送元にしない。テンプレートでは髪が肩や胸にかかっているので、
    # 近接で引くとこちらの胸が「髪の骨」を持ってしまい、体を動かすと胸だけ頭に
    # 追従して裂ける（§37）。髪は Face/Body から引けば頭のボーンに入る。
    tmpl_meshes = [o for o in tmpl_objs
                   if o.type == "MESH" and o.vertex_groups
                   and len(o.data.vertices) > 500
                   and "hair" not in o.name.lower()]
    for o in tmpl_objs:
        if o.type != "ARMATURE" and o not in tmpl_meshes:
            bpy.data.objects.remove(o, do_unlink=True)
    arm.name = "Armature"
    report["template_meshes"] = [
        {"name": o.name, "verts": len(o.data.vertices)} for o in tmpl_meshes]
    report["template_bones"] = len(arm.data.bones)

    # 3) 寸法合わせ → 結合
    report["fit"] = fit_armature(arm, meshes)
    # 身長だけの相似では腕幅／身長の比の差を吸収できない。指の骨が前腕に
    # 乗ったまま自動ウェイトを取ると、袖が裂けて手が潰れる（docs/05 §37）。
    report["fit_arms"] = fit_arms(arm, meshes, also=tmpl_meshes)
    # 衣装用の二次ボーンを変形候補から外す。入れたままだと体の 64% が
    # 別キャラのスカート用ボーンに従って動く（docs/05 §37）。
    extra = tuple(x for x in args.no_deform.split(",") if x)
    report["garment_bones"] = disable_garment_bones(arm)
    if extra:
        n = 0
        for b in arm.data.bones:
            if any(x in b.name for x in extra):
                b.use_deform = False
                n += 1
        report["no_deform"] = {"patterns": list(extra), "disabled": n}
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    report["bind"] = bind(arm, meshes, tmpl_meshes)

    # 4) 書き出し
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".blend":
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
    elif out.suffix.lower() == ".vrm":
        bpy.ops.export_scene.vrm(filepath=str(out))
    else:
        bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB")

    report["output"] = str(out)
    report["output_bytes"] = out.stat().st_size if out.exists() else 0
    report["status"] = "ok" if report["output_bytes"] > 1024 else "fail"
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("AUTORIG_OK " + json.dumps(
        {"bones": report["template_bones"], "bind": report["bind"]["bind_method"],
         "bytes": report["output_bytes"]}))


if __name__ == "__main__":
    main()
