#!/usr/bin/env python3
"""KEHAI の実機（three.js）でポーズ品質を測る。**Blender の pose_check の代わり。**

## なぜ Blender では測れないのか

体型変形で骨が取り残される不具合を直すと（`shape_transfer --joints-only`）、
**Blender の glTF インポータが評価後のメッシュを 1.5958m → 1.7210m に伸ばす**。
倍率は体型変形の逆数にあたる。glTF の式は

    v_skinned = Σ w_i · (globalTransform_i · inverseBindMatrix_i) · v

で、静止姿勢を保つには IBM = inverse(globalTransform) が正しい。ファイルはそう
書けており、three.js はそのとおり読む（KEHAI では正しい寸法で表示され、
ポーズ品質も改善する）。Blender の読み方が非標準。

だから**補正後のファイルの検証は実機で行う**。`blender/pose_check.py` は補正前の
ファイルに対して走り、ウェイトとモーフの健全性を見る役に留める。

## 何を測るか

**剛体であるべき部位が剛体のままか。** その骨に 0.9 以上のウェイトを持つ頂点は、
その骨を回しても互いの距離が変わらないはず。変わる＝メッシュが裂けている。

「関節の周りの頂点間距離の変化」で測ってはいけない ——**正しい肘の曲げでも
皮膚は伸びる**ので、支点が正しいほど大きくなる。実際それで判断を誤りかけた
（補正あり 11.9% 対 補正なし 5.12% で「悪化した」と読んだ）。

使い方:
    python3 scripts/kehai_pose_check.py --model subject_a_v4.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

JS = r"""
(async () => {
  const v = window.__vrm;
  if (!v) return JSON.stringify({error: 'window.__vrm が無い'});
  let mesh = null, verts = 0, meshes = 0;
  const all = [];
  v.scene.traverse(o => { if (o.isSkinnedMesh) { all.push(o); meshes++;
    verts += o.geometry.attributes.position.count;
    if (o.material && /body/.test(o.material.name)) mesh = o; } });
  if (!mesh) mesh = all.reduce((a,b)=>
    a.geometry.attributes.position.count > b.geometry.attributes.position.count ? a : b);
  const bt = (mesh.applyBoneTransform || mesh.boneTransform).bind(mesh);
  const pos = mesh.geometry.attributes.position;
  const si = mesh.geometry.attributes.skinIndex;
  const sw = mesh.geometry.attributes.skinWeight;
  const bones = mesh.skeleton.bones;
  const Vec = mesh.position.constructor;

  // 全体の寸法（スキニング後）
  v.scene.updateMatrixWorld(true);
  let lo = [1e9,1e9,1e9], hi = [-1e9,-1e9,-1e9];
  for (const m of all) {
    const p = m.geometry.attributes.position;
    const b = (m.applyBoneTransform || m.boneTransform).bind(m);
    for (let i = 0; i < p.count; i += 7) { const t = new Vec();
      t.fromBufferAttribute(p, i); b(i, t); m.localToWorld(t);
      lo = [Math.min(lo[0],t.x),Math.min(lo[1],t.y),Math.min(lo[2],t.z)];
      hi = [Math.max(hi[0],t.x),Math.max(hi[1],t.y),Math.max(hi[2],t.z)]; }
  }

  const test = (name, axis, ang) => {
    const b = v.humanoid.getRawBoneNode(name);
    if (!b) return {bone: name, error: 'humanoid に無い'};
    const bi = bones.indexOf(b);
    if (bi < 0) return {bone: name, error: 'skeleton に無い'};
    const idx = [];
    for (let i = 0; i < pos.count; i++) { let w = 0;
      for (let k = 0; k < 4; k++) if (si.getComponent(i,k) === bi) w += sw.getComponent(i,k);
      if (w > 0.9) idx.push(i); }
    if (idx.length < 12) return {bone: name, verts: idx.length, skipped: '頂点が少ない'};
    const snap = () => { v.scene.updateMatrixWorld(true);
      return idx.map(i => { const p = new Vec(); p.fromBufferAttribute(pos, i);
        bt(i, p); mesh.localToWorld(p); return p; }); };
    const rest = snap();
    const a0 = b.rotation[axis]; b.rotation[axis] = a0 + ang;
    const bent = snap(); b.rotation[axis] = a0; v.scene.updateMatrixWorld(true);
    let worst = 0, sum = 0, n = 0;
    const N = Math.min(idx.length, 250);
    for (let a = 0; a < N; a++) for (let c = a + 1; c < N; c += 3) {
      const d0 = rest[a].distanceTo(rest[c]); if (d0 < 0.003) continue;
      const r = Math.abs(bent[a].distanceTo(bent[c]) / d0 - 1);
      worst = Math.max(worst, r); sum += r; n++; }
    return {bone: name, verts: idx.length, pairs: n,
      mean_pct: +(sum/Math.max(n,1)*100).toFixed(3),
      worst_pct: +(worst*100).toFixed(3)};
  };

  const sbm = v.springBoneManager;
  return JSON.stringify({
    model: localStorage.lastConfigFile,
    meshes, verts,
    height_m: +(hi[1] - lo[1]).toFixed(4),
    bbox_y: [+lo[1].toFixed(4), +hi[1].toFixed(4)],
    spring_joints: sbm ? (sbm.joints ? (sbm.joints.size ?? sbm.joints.length) : 0) : null,
    rigid: [test('leftUpperArm','z',-1.0), test('leftLowerArm','x',-1.2),
            test('rightLowerArm','x',-1.2), test('leftHand','z',-0.8),
            test('leftUpperLeg','x',-0.9), test('leftLowerLeg','x',0.9)],
  });
})()
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="切り替える conf（例 subject_a_v4.yaml）")
    ap.add_argument("--wait", type=float, default=22.0, help="読み込みを待つ秒数")
    ap.add_argument("--out", type=Path)
    # 剛体であるべき部位が壊れているとみなす閾値。**現行の実測**:
    #   補正あり 肘 0.458/10.775%、手首 0.697/10.878%
    #   補正なし 肘 0.703/17.371%、手首 2.820/42.506%
    ap.add_argument("--max-mean", type=float, default=1.5)
    ap.add_argument("--max-worst", type=float, default=20.0)
    args = ap.parse_args()

    probe = [sys.executable, str(ROOT / "kehai_probe.py")]
    if args.model:
        subprocess.run(probe + ["--model", args.model], check=True,
                       capture_output=True, text=True)
        time.sleep(args.wait)
    js = ROOT.parent / "jobs" / ".kehai_pose_check.js"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(JS)
    r = subprocess.run(probe + ["--file", str(js), "--await"],
                       capture_output=True, text=True)
    js.unlink(missing_ok=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr, file=sys.stderr)
        raise SystemExit("実機の評価に失敗")
    rep = json.loads(r.stdout.strip())
    if "error" in rep:
        raise SystemExit(rep["error"])

    bad = [x for x in rep["rigid"]
           if "mean_pct" in x and (x["mean_pct"] > args.max_mean
                                   or x["worst_pct"] > args.max_worst)]
    rep["ok"] = not bad
    rep["failed"] = [x["bone"] for x in bad]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2))

    print(f"モデル {rep['model']}  メッシュ {rep['meshes']}  頂点 {rep['verts']:,}  "
          f"身長 {rep['height_m']}m  spring {rep['spring_joints']}")
    for x in rep["rigid"]:
        if "mean_pct" not in x:
            print(f"  {x['bone']:16s} — {x.get('error') or x.get('skipped')}")
            continue
        mark = "x" if x in bad else "o"
        print(f"  {mark} {x['bone']:16s} 剛体の歪み 平均 {x['mean_pct']:6.3f}% / "
              f"最大 {x['worst_pct']:7.3f}%（{x['verts']}頂点）")
    print("合格" if rep["ok"] else f"不合格: {rep['failed']}")
    if not rep["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
