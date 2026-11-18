"""VRM のリグを実際に動かして、変形が破綻しないかを見る。

なぜ要るか: 検証（`export_vrm.py`）は「必須ボーンが割り当たっているか」と
「スキンが存在するか」までしか見ない。**骨を回したときにメッシュがどう動くか**は
別の話で、そこは目で見るしかない。実際、この工程を作る前は:

  - 頂点グループ167個・Armatureモディファイア・deform骨167本が揃っているのに
    1頂点もスキンされていない VRM を出し続けていた（docs/05-measurements.md §28）
  - スキンを直した後も、首を振るとジャケットの胴が裂けていた（同 §29）

どちらも検証は通っていた。ポーズを当てて初めて分かる。

骨は1本ずつ分けて動かせる。**症状が出る最小の操作を特定する**ためで、
§29 では「連結成分が多いから」という仮説に2手費やした末に、
骨ごとに切り分けて初めて原因（首の骨の影響範囲）に辿り着いた。

  blender --background --factory-startup --python blender/pose_check.py -- \
      --input <character.vrm> --outdir <renders/pose> [--poses arms,legs,neck,all]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import addon_utils
from mathutils import Euler, Vector

MARKER = "POSE_CHECK_OK"

# 命名の流儀ごとの言い換え。VRoid 系（J_Bip_L_UpperArm）と Mixamo 系（LeftArm）で
# 骨の名前がまったく違うので、両方を試す。Rocketbox 由来のテンプレートは Mixamo 命名
# （scripts/prepare_rocketbox.py が改名する）。
# 一致しないと `bones: []` で「変形しない」と報告され、**リグが壊れているように見える**。
ALIASES: dict[str, tuple[str, ...]] = {
    "L_UpperArm": ("L_UpperArm", "LeftArm"),
    "R_UpperArm": ("R_UpperArm", "RightArm"),
    "L_LowerArm": ("L_LowerArm", "LeftForeArm"),
    "R_LowerArm": ("R_LowerArm", "RightForeArm"),
    "L_UpperLeg": ("L_UpperLeg", "LeftUpLeg"),
    "L_LowerLeg": ("L_LowerLeg", "LeftLeg"),
    "R_UpperLeg": ("R_UpperLeg", "RightUpLeg"),
    "R_LowerLeg": ("R_LowerLeg", "RightLeg"),
    "C_Spine":    ("C_Spine", "Spine"),
    "Neck":       ("Neck",),
    # 指。**M1 の合否そのもの**（生成メッシュでは指が塊で動かせなかった。docs/05 §38）
    "L_Index1":   ("L_Index_Proximal", "LeftHandIndex1"),
    "L_Thumb1":   ("L_Thumb_Proximal", "LeftHandThumb1"),
    "R_Index1":   ("R_Index_Proximal", "RightHandIndex1"),
}

# ポーズの定義。(ボーン名の部分一致, XYZ回転[度])
POSES: dict[str, list[tuple[str, tuple[float, float, float]]]] = {
    "arms":   [("L_UpperArm", (0, 0, -55)), ("R_UpperArm", (0, 0, 55))],
    "lowarm": [("L_LowerArm", (0, -25, 0)), ("R_LowerArm", (0, 25, 0))],
    # **両脚を動かすこと。** 判定は外接の変化量なので、片脚だけ曲げても
    # もう一方の脚と腕が外接を決めてしまい、変形しているのに 0.0mm と報告される
    # （実測でこれに引っかかった）。
    "legs":   [("L_UpperLeg", (-25, 0, 0)), ("L_LowerLeg", (40, 0, 0)),
               ("R_UpperLeg", (-25, 0, 0)), ("R_LowerLeg", (40, 0, 0))],
    "neck":   [("Neck", (10, 0, 25))],
    "spine":  [("C_Spine", (15, 0, 0))],
    # 指を握らせる。生成メッシュでは指の骨がメッシュの外に浮いていて、
    # 動かすと棘状に飛び散った（docs/05 §38）。固定テンプレートなら成立するはず。
    "fingers": [("L_Index1", (0, 0, -70)), ("L_Thumb1", (0, 0, -45)),
                ("R_Index1", (0, 0, 70))],
    "all":    [("L_UpperArm", (0, 0, -55)), ("R_UpperArm", (0, 0, 55)),
               ("L_UpperLeg", (-20, 0, 0)), ("L_LowerLeg", (35, 0, 0))],
}


def enable_vrm() -> None:
    for m in addon_utils.modules():
        if "vrm" in m.__name__.lower():
            try:
                addon_utils.enable(m.__name__, default_set=True, persistent=True)
            except Exception:
                pass


def mesh_bounds() -> tuple[Vector, Vector]:
    """アーマチュア変形を反映した実際の頂点位置から外接を取る。

    `bound_box` は変形前の値なので、ポーズが効いたかの判定には使えない。
    """
    dg = bpy.context.evaluated_depsgraph_get()
    lo = Vector((float("inf"),) * 3)
    hi = Vector((float("-inf"),) * 3)
    for o in [x for x in bpy.data.objects if x.type == "MESH"]:
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        for v in me.vertices:
            w = ev.matrix_world @ v.co
            lo = Vector((min(lo[i], w[i]) for i in range(3)))
            hi = Vector((max(hi[i], w[i]) for i in range(3)))
        ev.to_mesh_clear()
    return lo, hi


def mesh_points() -> list:
    """アーマチュア変形を反映した全頂点のワールド座標を、順序を保って返す。

    ポーズ前後で同じ順序になるので、そのまま引き算して**頂点ごとの変位**が取れる。
    """
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in sorted([x for x in bpy.data.objects if x.type == "MESH"],
                    key=lambda x: x.name):
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        mw = ev.matrix_world
        pts.extend([mw @ v.co for v in me.vertices])
        ev.to_mesh_clear()
    return pts


def apply_pose(arm, name: str) -> list[str]:
    """指定のポーズを当てる。J_Bip_ を優先して1本だけ回す。

    VRoid 系のリグは J_Sec_/J_Adj_/J_Roll_ の補助骨が同じ語を含むので、
    素直な部分一致だと変形しない骨を掴む（docs/05-measurements.md §6）。
    """
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    used = []
    for sub, xyz in POSES[name]:
        prim = []
        for cand in ALIASES.get(sub, (sub,)):
            hits = [b for b in arm.pose.bones if cand.lower() in b.name.lower()]
            prim = [b for b in hits if "j_bip_" in b.name.lower()] or hits
            if prim:
                break
        if not prim:
            continue
        b = prim[0]
        b.rotation_mode = "XYZ"
        b.rotation_euler = Euler([math.radians(v) for v in xyz], "XYZ")
        used.append(b.name)
    bpy.ops.object.mode_set(mode="OBJECT")
    return used


def setup_scene(res: int, samples: int) -> None:
    """render_views.py と同じ照明・露出。候補間で見た目を揃えるため。"""
    world = bpy.data.worlds.new("W")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 0.55
    for n, loc, e in (("key", (3, -4, 4), 900),
                      ("fill", (-4, -2, 2), 350),
                      ("rim", (0, 4.5, 3), 500)):
        d = bpy.data.lights.new(n, type="AREA")
        d.energy = e
        d.size = 4.0
        o = bpy.data.objects.new(n, d)
        o.location = loc
        bpy.context.collection.objects.link(o)

    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [
        x.identifier for x in
        bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    ] else "BLENDER_EEVEE"
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"
    try:
        sc.eevee.taa_render_samples = samples
    except Exception:
        pass


def place_camera(lo: Vector, hi: Vector, azim_deg: float):
    center = (lo + hi) / 2.0
    radius = max((hi - lo).length / 2.0, 1e-4)
    cd = bpy.data.cameras.new("C")
    cd.lens = 60.0
    cam = bpy.data.objects.new("C", cd)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    t = bpy.data.objects.new("T", None)
    t.location = center
    bpy.context.collection.objects.link(t)
    c = cam.constraints.new("TRACK_TO")
    c.target = t
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    a = math.radians(azim_deg)
    d = radius * 3.0
    cam.location = Vector((center.x + d * math.sin(a),
                           center.y - d * math.cos(a), center.z))
    return cam


def rigidity(arm, bone: str, axis: int, ang: float, thresh: float = 0.9) -> dict:
    """**剛体であるべき部位が剛体のままか。**

    その骨に `thresh` 以上のウェイトを持つ頂点は、その骨を回しても互いの距離が
    変わらないはず。変わる＝メッシュが裂けている。

    **関節の周りの頂点間距離で測ってはいけない。** 正しい肘の曲げでも皮膚は
    伸びるので、支点が正しいほど大きくなる。実際それで判断を誤りかけた
    （骨のピボットを直したら「悪化した」と読んだ。docs/09 §21.3）。
    """
    import numpy as np
    pb = arm.pose.bones.get(bone)
    if pb is None:
        return {"bone": bone, "error": "骨が無い"}
    dg = bpy.context.evaluated_depsgraph_get()
    targets = []
    for o in bpy.data.objects:
        if o.type != "MESH" or not [m for m in o.data.materials if m]:
            continue
        vg = o.vertex_groups.get(bone)
        if vg is None:
            continue
        idx = [v.index for v in o.data.vertices
               if any(g.group == vg.index and g.weight > thresh for g in v.groups)]
        if len(idx) >= 12:
            targets.append((o, idx))
    if not targets:
        return {"bone": bone, "skipped": f"ウェイト>{thresh} の頂点が足りない"}

    def snap():
        bpy.context.view_layer.update()
        d = bpy.context.evaluated_depsgraph_get()
        out = []
        for o, idx in targets:
            ev = o.evaluated_get(d)
            me = ev.to_mesh()
            mw = o.matrix_world
            out.append(np.array([[*(mw @ me.vertices[i].co)] for i in idx]))
            ev.to_mesh_clear()
        return np.vstack(out)

    rest = snap()
    old = pb.rotation_mode, tuple(pb.rotation_euler)
    pb.rotation_mode = "XYZ"
    e = list(pb.rotation_euler)
    e[axis] += ang
    pb.rotation_euler = e
    bent = snap()
    pb.rotation_euler = old[1]
    pb.rotation_mode = old[0]
    bpy.context.view_layer.update()

    n = min(len(rest), 250)
    worst = mean = 0.0
    cnt = 0
    for a in range(n):
        for b in range(a + 1, n, 3):
            d0 = float(np.linalg.norm(rest[a] - rest[b]))
            if d0 < 0.003:
                continue
            r = abs(float(np.linalg.norm(bent[a] - bent[b])) / d0 - 1.0)
            worst = max(worst, r)
            mean += r
            cnt += 1
    return {"bone": bone, "verts": int(len(rest)), "pairs": cnt,
            "mean_pct": round(mean / max(cnt, 1) * 100, 3),
            "worst_pct": round(worst * 100, 3)}


# 剛体検査に回す骨と回し方。KEHAI 側（scripts/kehai_pose_check.py）と揃えてある。
RIGID = [("LeftArm", 2, -1.0), ("LeftForeArm", 0, -1.2), ("RightForeArm", 0, -1.2),
         ("LeftHand", 2, -0.8), ("LeftUpLeg", 0, -0.9), ("LeftLeg", 0, 0.9)]


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--poses", default="all",
                    help=f"カンマ区切り。使えるもの: {','.join(POSES)}")
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--report", default=None)
    ap.add_argument("--no-rigidity", action="store_true",
                    help="剛体検査を行わない")
    # 現行の実測（骨のピボット補正あり）: 肘 0.458/10.775%、手首 0.697/10.878%
    ap.add_argument("--max-mean", type=float, default=1.5)
    ap.add_argument("--max-worst", type=float, default=20.0)
    ap.add_argument("--min-verts", type=int, default=25,
                    help="これ未満の頂点しか持たない骨は合否に使わない")
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.poses.split(",") if n.strip()]
    unknown = [n for n in names if n not in POSES]
    if unknown:
        raise SystemExit(f"未知のポーズ: {unknown}（使えるもの: {list(POSES)}）")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = []

    for name in names:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        enable_vrm()
        bpy.ops.import_scene.vrm(filepath=str(Path(args.input).resolve()))
        arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
        if arm is None:
            raise SystemExit("アーマチュアがありません")

        rest_lo, rest_hi = mesh_bounds()
        rest_pts = mesh_points()
        used = apply_pose(arm, name)
        posed_lo, posed_hi = mesh_bounds()
        posed_pts = mesh_points()

        # **判定は頂点ごとの変位で行う。外接の変化では局所の動きを取りこぼす。**
        # 実測: 指を握らせたとき、女性素体は親指が奥行きの最大を押し広げて
        # 0.005 と出たが、男性素体は既存の外接の内側で曲がったため 0.000 になり、
        # 骨も見つかって回っているのに「変形せず」と誤報告された。
        # 脚も同じで、片脚だけ曲げるともう一方の脚と腕が外接を決めてしまう。
        if len(rest_pts) == len(posed_pts) and rest_pts:
            moved = max((a - b).length for a, b in zip(posed_pts, rest_pts))
        else:
            moved = max(abs((posed_hi - posed_lo)[i] - (rest_hi - rest_lo)[i])
                        for i in range(3))
        bbox_delta = max(abs((posed_hi - posed_lo)[i] - (rest_hi - rest_lo)[i])
                         for i in range(3))
        setup_scene(args.res, args.samples)
        place_camera(posed_lo, posed_hi, 0.0)
        bpy.context.scene.render.filepath = str(outdir / f"{name}.png")
        bpy.ops.render.render(write_still=True)

        results.append({
            "pose": name, "bones": used,
            "rest_size": [round(v, 4) for v in (rest_hi - rest_lo)],
            "posed_size": [round(v, 4) for v in (posed_hi - posed_lo)],
            "max_delta": round(moved, 4),          # 頂点ごとの最大変位
            "bbox_delta": round(bbox_delta, 4),    # 参考。外接の変化
            "vertices": len(rest_pts),
            # 変形が起きていなければスキンが効いていない。§28 の再発検知。
            "deformed": moved > 1e-3,
            "render": str(outdir / f"{name}.png"),
        })

    rig = []
    if not args.no_rigidity:
        for bone, axis, ang in RIGID:
            rig.append(rigidity(arm, bone, axis, ang))
    # **頂点が少ない骨は判定に使わない。** 股は 0.9 超のウェイトを持つ頂点が
    # 19個しかなく、Blender と KEHAI で 2.120% 対 0.630% と食い違う（骨の
    # ローカル軸の取り方の違い）。報告はするが合否には入れない。
    bad = [r for r in rig if "mean_pct" in r and r.get("verts", 0) >= args.min_verts
           and (r["mean_pct"] > args.max_mean or r["worst_pct"] > args.max_worst)]
    report = {"input": args.input, "poses": results,
              "all_deformed": all(r["deformed"] for r in results),
              "rigidity": rig, "rigidity_ok": not bad,
              "rigidity_failed": [r["bone"] for r in bad]}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(MARKER + " " + json.dumps(
        {"poses": [(r["pose"], r["deformed"], r["max_delta"]) for r in results],
         "all_deformed": report["all_deformed"],
         "rigidity_ok": report["rigidity_ok"],
         "rigidity": [(r.get("bone"), r.get("mean_pct"), r.get("worst_pct"))
                      for r in rig]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
