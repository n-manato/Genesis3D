#!/usr/bin/env python3
"""テンプレートの頭を、生成された頭の表面に吸着させる（非剛体 ICP）。

## なぜ接合をやめてラップにするか（docs/09）

これまでは「生成した頭」を出荷して、テンプレートの体に**接合**していた。接合の実測
（`jobs/ga_v2_005/scores/merge.json`）:

    applied_scale : 0.31479     ← head-width 基準。これが黙って採用された
    scale_by_span : 0.45527     ← 44.6% 違う
    merged height : 1.52286     ← 指定 1.60m。7.7cm 足りない

**頭が正しい寸法の69%に縮んで胴に付いていた。** 首が長く見えたのはこれ。原因は
「2つのランドマーク比のどちらを信じるか」を決める根拠が無いことで、頭と体が別人由来に
なった時点でどちらの比も正当化できない。

ラップは逆をやる。**生成メッシュは測るためだけに回し、出荷するのはテンプレートの
トポロジ**にする。すると:

  - 接合が存在しない → 首の問題が構造的に消える
  - 削減が存在しない → 顔の黒い斑点が消える（682,766頂点を間引く工程ごと無い）
  - モーフを転写しない → テンプレートが最初から持つ 175個がそのまま出る
  - 頂点数 121,907 → 約 11k

**位置合わせは2つの比ではなく数千点の対応から解く。** これが接合との本質的な違い。

## 解き方（Amberg et al. 2007, Optimal Step Nonrigid ICP）

頂点ごとに 3x4 のアフィン X_i を置き、次を最小化する:

    E = Σ w_i |X_i [v_i;1] - u_i|²          データ項（u_i は対象表面の最近点）
      + α Σ_(i,j)∈E |(X_i - X_j) G|_F²      剛性項（G = diag(1,1,1,γ)）
      + β Σ |X_i [v_i;1] - l_i|²            ランドマーク項

α を大きい値から小さい値へ焼きなます。α が大きいうちは全体が相似変換に近く動き、
小さくすると局所が表面に貼り付く。**これがランドマークだけの RBF との差** ——
v1 の RBF は 68点しか拘束が無く、頂点は平均 4.7mm しか動かずテンプレートの頭のままだった。

## 髪に吸われないようにする

対象は TRELLIS が生成した頭で、**髪が付いている**。テンプレートの頭皮を最近点で
引っ張ると髪の表面に吸着して頭蓋が膨らむ。そこで **顔マスク**を掛ける:
68点ランドマークからの距離で重みを 1 →0 に落とし、頭皮と後頭部はデータ項をゼロにする。
剛性項が残るので、そこは顔の変形に滑らかに追従するだけになる。

ArcFace が見るのは顔面だけなので、同一性の観点でも失うものは無い。

## 出力

`shape_transfer.py` が読む `{"centers": [...], "targets": [...]}` を書く。変形の適用・
モーフ差分・法線・glTF 手術は既存のものをそのまま使う（一度も Blender を通さない）。

**アンカーを必ず混ぜる。** 無いと RBF の多項式項が全身に効いて足が動く（docs/08 §10.3）。

使い方:
    $HOME/miniconda3/envs/genesis-face/bin/python -m preprocessing.wrap_head \\
        --template assets/templates/rocketbox/female_adult_01.glb \\
        --landmarks configs/landmarks/rocketbox_female_01.json \\
        --anchors configs/landmarks/rocketbox_female_01_anchors.json \\
        --target jobs/x/head_ready.glb \\
        --render jobs/x/head_front.png --render-meta jobs/x/scores/head_front.json \\
        --out jobs/x/scores/pairs_head.json --report jobs/x/scores/wrap.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blender.vrm_finalize import read_glb                       # noqa: E402
from preprocessing.ict_fit import backproject, detect_68        # noqa: E402
from preprocessing.template_anchors import farthest_point       # noqa: E402  # noqa: F401
from preprocessing.template_landmarks import face_primitive     # noqa: E402
from texture_processing.face_texture_blend import all_geometry  # noqa: E402

# 剛性の焼きなまし。大きいところから始めて相似変換に近い動きで大域を合わせ、
# 下げながら局所を表面へ貼り付ける。値はメッシュを原点中心・m単位にした前提。
#
# **下限は 2.0。実測で非単調**（held-out ArcFace / tweed 基準・docs/09 §9）:
#
#   下限 5.0   バスト 0.5738   全身 0.5644      硬すぎて対象に当たらない
#   下限 2.0   バスト 0.6208   全身 0.5994   ← 最適
#   下限 0.5   バスト 0.5804   全身 0.5933      TRELLIS の高周波を追う
#
# 幾何の残差だけ見ると 0.5 のほうが小さい（1.691mm 対 2.4mm 程度）。
# **残差を最小化すると同一性が落ちる。** 指標を取り違えないこと。
STIFFNESS = (200.0, 80.0, 30.0, 12.0, 5.0, 2.0)


def thin_centers(pts: np.ndarray, spacing: float) -> np.ndarray:
    """間隔が spacing 未満の中心を間引く。**変形ヤコビアンの条件数のために必須。**

    phi(r)=r の厳密補間は、中心が近いと行列がほぼ特異になり、**中心の間で
    場が振動する**。頂点位置は補間の目標なので正確に出るが、法線とモーフ差分は
    ヤコビアンを通るので壊れる。実測（テンプレの頭 1,752頂点をそのまま中心にした場合）:

        中心の最近傍間隔      最小 0.125mm（中央 5.31mm）
        (I+J) の特異値        最大 91.59 / 最小 0.00   ← 局所的に91倍伸びと退化
        |f(x+d)-f(x)|/|d|     最大 21.8（中央 0.151）
        モーフの振幅          ou が 2.94倍、mouthSmileLeft が 2.61倍に膨らんだ

    間引くと素直に効く（アンカー込み・顔面の位置誤差は補間しきれない分）:

        最小間隔  中心   顔面の位置誤差   σmax
          なし    1752      0.001mm      91.59
         2.0mm    1540      0.443mm       5.75
         3.0mm    1384      0.617mm       5.04
         6.0mm    1022      0.998mm       3.53

    **平滑化（`shape_transfer --smooth`）では直らない。** ほぼ特異な行列に
    対角を足すので余計に悪化し、位置誤差が平均 8.8mm・最大 734mm になった。

    外側（顔の縁）から残す。中心が中央に偏ると縁の再現が落ちる。
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    keep = np.ones(len(pts), dtype=bool)
    for i in np.argsort(-np.linalg.norm(pts - pts.mean(0), axis=1)):
        if not keep[i]:
            continue
        for j in tree.query_ball_point(pts[i], spacing):
            if j != i:
                keep[j] = False
    return np.where(keep)[0]


def weld(pos: np.ndarray, tol: int = 6):
    """位置が同じ頂点をまとめる。

    UV の切れ目でテンプレートは頂点を重複して持つ。重複を別々に解くと継ぎ目が
    割れるし、RBF の中心が同一点になって行列が特異になる。**溶接は必須。**
    """
    _, first, inv = np.unique(np.round(pos, tol), axis=0,
                              return_index=True, return_inverse=True)
    return pos[first], inv.ravel(), first


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(ln > 1e-12, ln, 1.0)


def umeyama(src: np.ndarray, dst: np.ndarray):
    """src を dst に重ねる相似変換 (s, R, t)。スケールを含む。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    scale = (s * np.array([1.0, 1.0, d])).sum() / (a ** 2).sum()
    return scale, r, mu_d - scale * (r @ mu_s)


def robust_umeyama(src, dst, iters: int = 6, k: float = 2.5):
    """外れ値を切りながら相似変換を解く。

    **最小二乗のままでは使えない。** 逆投影した68点には粗大な外れ値が混じる ——
    レンダの穴を通り抜けた光線や、前髪に当たった点。実測では点#4 が 219mm、
    点#47 が 192mm ずれており、これだけで全体のスケール推定が 0.12〜0.59 の
    範囲で暴れた（部分集合ごとに解くと桁で違う値が出た）。

    残差の中央絶対偏差で切って解き直すのを繰り返す。返り値に採用した点の
    マスクを含める ——**同じ外れ値を ICP のランドマーク項でも外す**ため。
    """
    keep = np.ones(len(src), dtype=bool)
    s = r = t = None
    for _ in range(iters):
        s, r, t = umeyama(src[keep], dst[keep])
        res = np.linalg.norm((s * (r @ src.T).T) + t - dst, axis=1)
        med = np.median(res[keep])
        mad = np.median(np.abs(res[keep] - med)) + 1e-9
        new = res < med + k * 1.4826 * mad
        if new.sum() < 8 or (new == keep).all():
            break
        keep = new
    return s, r, t, keep


def edges_of(faces: np.ndarray) -> np.ndarray:
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def visible_mask(pts, all_v, all_f, tol: float = 1.5e-3):
    """正面から見えている頂点だけ 1。**眼球・歯・口内・後頭部を外すのに必須。**

    これが無いと眼球のジオメトリまで顔表面に吸着する。眼球は前を向いていて
    68点の XY 凸包の内側にあるので、法線ゲートと凸包だけでは落ちない。実際に
    起きていたのは、瞼の裏にある眼球が瞼の面まで引き出されて相互貫入し、
    レンダで**目の周りに金属質の裂けた筋**として見えていた（当初これを
    「まつ毛カード」だと誤診した。マテリアルを色分けして描くと、目の位置に
    `opacity` プリミティブは一切無いことが分かる）。

    判定はカメラ（+Z の無限遠）から各頂点へ光線を飛ばし、最初に当たる点が
    その頂点自身かどうか。tol は表面の厚み分の余裕。
    """
    import trimesh
    mesh = trimesh.Trimesh(vertices=all_v, faces=all_f, process=False)
    zmax = float(all_v[:, 2].max())
    origins = np.column_stack([pts[:, 0], pts[:, 1], np.full(len(pts), zmax + 1.0)])
    dirs = np.tile([0.0, 0.0, -1.0], (len(pts), 1))
    loc, idx_ray, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=False)
    hit_z = np.full(len(pts), -np.inf)
    hit_z[idx_ray] = loc[:, 2]
    return hit_z <= pts[:, 2] + tol


def face_mask(pts, normals, lm, falloff: float):
    """顔面だけを 1 にする重み。頭皮・後頭部・耳を髪から守る。

    **距離で切ってはいけない。** 68点からの半径で切ると、頭が高さ 374mm しか
    無いので半径 95mm でも頭の 79% が候補に入り、頭皮が対象（TRELLIS が生成した
    **髪付き**の頭）の髪表面に吸着して頭蓋が膨らむ。

    正しい境界は 68点が張る輪郭そのもの。正面から見た XY 凸包の内側を 1 とし、
    外側は最寄りの点までの距離で落とす。さらに**前を向いた面だけ**にする ——
    XY 凸包は後頭部も同じ足元に射影されるので、法線で表裏を分ける必要がある。
    """
    from scipy.spatial import Delaunay
    hull = Delaunay(lm[:, :2])
    inside = hull.find_simplex(pts[:, :2]) >= 0
    d2 = np.linalg.norm(pts[:, None, :2] - lm[None, :, :2], axis=2).min(axis=1)
    w = np.where(inside, 1.0, np.clip((falloff - d2) / max(falloff, 1e-9), 0.0, 1.0))
    # 前向きゲート。法線の Z が 0 → 0.3 のあいだで滑らかに立ち上げる。
    w = w * np.clip(normals[:, 2] / 0.3, 0.0, 1.0)
    return w, d2


def nicp(v0, faces, target, lm_idx, lm_pos, lm_w, w_face, args, log):
    """非剛体 ICP 本体。頂点ごとの 3x4 アフィンを疎な最小二乗で解く。

    **アフィンは常に元の頂点 v0 に対して解く（Amberg の定式化）。** 反復ごとの
    増分として解いてはいけない。増分にすると剛性項が「増分どうしの差」を罰する
    ことになり、累積した変形は何も正則化されない。実測では 68点の残差が
    反復のたびに増えた（20.99 → 39.16mm）。
    """
    from trimesh.proximity import closest_point

    n = len(v0)
    e = edges_of(faces)
    v1 = np.hstack([v0, np.ones((n, 1))])          # 同次座標。**固定**

    # 剛性項: 各辺について 4行（G = diag(1,1,1,gamma)）
    rows, cols, vals = [], [], []
    for k, (i, j) in enumerate(e):
        for c in range(4):
            g = args.gamma if c == 3 else 1.0
            rows += [4 * k + c, 4 * k + c]
            cols += [4 * i + c, 4 * j + c]
            vals += [g, -g]
    m_stiff = sp.csr_matrix((vals, (rows, cols)), shape=(4 * len(e), 4 * n))

    def affine_rows(idx, w):
        """行 r が頂点 idx[r] の同次座標を持つ疎行列。X に掛けると変換後の点。"""
        return sp.csr_matrix(
            ((w[:, None] * v1[idx]).ravel(),
             (np.repeat(np.arange(len(idx)), 4),
              (4 * idx[:, None] + np.arange(4)[None, :]).ravel())),
            shape=(len(idx), 4 * n))

    lm_rows = affine_rows(lm_idx, lm_w)
    lm_rhs = lm_w[:, None] * lm_pos
    all_idx = np.arange(n)

    # ダンピング項: |X_i - I|^2 を、顔マスクの外ほど強く罰する。
    #
    # **これが無いと後頭部が暴れる。** 剛性項が罰するのは「隣どうしのアフィンの差」
    # だけで、同じアフィンでも適用する点が遠ければ変位は大きくなる。顔で解いた
    # 回転・スケールが後頭部に外挿されて、実測で最大 260mm 動いた（頭の高さは 374mm）。
    # データの無い領域はテンプレートの形のままにする、という事前分布を入れる。
    w_damp = args.damping * (1.0 - w_face)
    damp_rows = sp.csr_matrix(
        (np.repeat(w_damp, 4),
         (np.arange(4 * n), np.arange(4 * n))), shape=(4 * n, 4 * n))
    ident = np.tile(np.vstack([np.eye(3), np.zeros((1, 3))]), (n, 1))
    damp_rhs = w_damp.repeat(4)[:, None] * ident

    # 初回の最近点は変形前の位置で取る。stiffness が大きいので最初の解は
    # 全体が1つのアフィンに近く、そこから局所へ落としていく。
    x = v0.copy()
    hist = []
    n_alpha = len(args.stiffness)
    for it, alpha in enumerate(args.stiffness):
        # 対応の許容距離も焼きなます。初期は位置合わせが粗いので広く取らないと
        # 対応が全部棄却されて動かない。
        md = args.max_dist * (1.0 + 2.0 * (n_alpha - 1 - it) / max(n_alpha - 1, 1))
        close, dist, tri = closest_point(target, x)
        tn = target.face_normals[tri]
        vn = vertex_normals(x, faces)
        dot = (tn * vn).sum(axis=1)
        # **符号は見ない。** 対象は TRELLIS の出力で、三角形スープ（境界辺 130,409）
        # なので面の巻き方向が揃っておらず、外向きの面は 51.7%＝偶然と同じだった。
        # fix_normals も fix_winding も効かない。法線テストの目的は「かすめた対応
        # （面に対して斜めから当たった点）を捨てる」ことなので、絶対値で足りる。
        if not args.signed_normal:
            dot = np.abs(dot)
        cand = w_face > 1e-6
        ok = (dist < md) & (dot > args.normal_cos) & cand
        w = np.where(ok, w_face, 0.0)
        rej_d = int((cand & (dist >= md)).sum())
        rej_n = int((cand & (dist < md) & (dot <= args.normal_cos)).sum())

        a = sp.vstack([affine_rows(all_idx, w), alpha * m_stiff,
                       lm_rows, damp_rows]).tocsr()
        b = np.vstack([w[:, None] * close, np.zeros((4 * len(e), 3)),
                       lm_rhs, damp_rhs])
        ata = (a.T @ a).tocsc() + sp.identity(4 * n, format="csc") * args.ridge
        x_aff = spla.splu(ata).solve(a.T @ b).reshape(n, 4, 3)
        x = np.einsum("nij,ni->nj", x_aff, v1)

        _, dist2, _ = closest_point(target, x)
        rec = {"iter": it, "alpha": alpha, "max_dist_mm": round(md * 1000, 1),
               "corr": int(ok.sum()), "cand": int(cand.sum()),
               "rejected": {"distance": rej_d, "normal": rej_n},
               "mean_dist_mm": round(float(dist2[w_face > 0.5].mean()) * 1000, 3),
               "lm_rms_mm": round(float(np.sqrt(
                   ((x[lm_idx] - lm_pos) ** 2).sum(1).mean())) * 1000, 3)}
        hist.append(rec)
        log(f"  a={alpha:7.1f} 許容{rec['max_dist_mm']:5.0f}mm  "
            f"対応 {rec['corr']:4d}/{rec['cand']:4d}"
            f"(遠{rej_d:4d} 法線{rej_n:4d})  残差 {rec['mean_dist_mm']:7.3f}mm  "
            f"68点 RMS {rec['lm_rms_mm']:7.3f}mm")
    return x, hist


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--landmarks", type=Path, required=True,
                    help="configs/landmarks/*.json（テンプレート側の68点）")
    ap.add_argument("--anchors", type=Path, required=True,
                    help="ゼロ変位アンカー。**任意項目ではない**（docs/08 §10.3）")
    ap.add_argument("--target", type=Path, required=True, help="生成された頭 glb")
    ap.add_argument("--render", type=Path, required=True,
                    help="ortho_front.py が撮った対象の正面 png")
    ap.add_argument("--render-meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="対応点 json")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--face-material", default="head")
    ap.add_argument("--no-visibility", dest="visibility", action="store_false",
                    help="隠れている頂点もデータ項に入れる。**眼球が顔面に"
                         "引き出されて目が裂ける**ので通常は使わない")
    ap.add_argument("--no-brow-cap", dest="brow_cap", action="store_false",
                    help="眉より上もデータ項に入れる。前髪の無い対象のときだけ")
    ap.add_argument("--brow-band", type=float, default=0.015,
                    help="眉の上端から重みを0に落とすまでの距離（m）")
    ap.add_argument("--face-falloff", type=float, default=0.025,
                    help="68点の凸包の外側、この距離で重みが0になる（m）")
    ap.add_argument("--max-dist", type=float, default=0.045,
                    help="これより遠い最近点は対応にしない（m）")
    ap.add_argument("--normal-cos", type=float, default=0.5, help="法線の一致（60度）")
    ap.add_argument("--signed-normal", action="store_true",
                    help="法線の符号も見る。対象が閉じたメッシュのときだけ意味がある")
    ap.add_argument("--gamma", type=float, default=1.0, help="剛性項の平行移動の重み")
    ap.add_argument("--beta", type=float, default=2.0, help="ランドマーク項の重み")
    ap.add_argument("--contour-weight", type=float, default=0.25,
                    help="輪郭 0-16 の重み倍率。逆投影が斜めに当たるので下げる")
    ap.add_argument("--damping", type=float, default=0.3,
                    help="顔マスクの外を恒等アフィンに留める強さ。"
                         "0 にすると後頭部が外挿で暴れる")
    ap.add_argument("--ridge", type=float, default=1e-8)
    ap.add_argument("--stiffness", type=float, nargs="*", default=list(STIFFNESS))
    ap.add_argument("--min-center-spacing", type=float, default=0.002,
                    help="RBF 中心の最小間隔（m）。**0 にしてはいけない** ——"
                         "近い中心が変形ヤコビアンを壊し、モーフの振幅が最大 2.9倍に"
                         "膨らむ（thin_centers の docstring に実測値）")
    ap.add_argument("--centers", type=int, default=0,
                    help="間引いた後さらに最遠点で絞る。0 = 絞らない")
    args = ap.parse_args()

    import trimesh
    from PIL import Image

    def log(s):
        print(s, flush=True)

    # --- テンプレート側 ---------------------------------------------------
    doc, blob = read_glb(args.template)
    mi, pi, verts, tris, matname = face_primitive(doc, blob, args.face_material)
    wv, inv, _ = weld(verts)
    wf = inv[tris]
    wf = wf[(wf[:, 0] != wf[:, 1]) & (wf[:, 1] != wf[:, 2]) & (wf[:, 0] != wf[:, 2])]
    log(f"テンプレ {matname}: {len(verts)}頂点 → 溶接 {len(wv)}頂点 / {len(wf)}三角形")

    lmj = json.loads(args.landmarks.read_text())
    lm_v = np.array([e["vertex"] for e in lmj["landmarks"]], dtype=np.int64)
    lm_idx = inv[lm_v]

    # --- 対象側 -----------------------------------------------------------
    tgt = trimesh.load(args.target, process=False)
    if isinstance(tgt, trimesh.Scene):
        tgt = trimesh.util.concatenate(list(tgt.geometry.values()))
    meta = json.loads(args.render_meta.read_text())
    uv = detect_68(Image.open(args.render))
    L_t = backproject(uv, meta, tgt)
    ok = ~np.isnan(L_t[:, 0])
    if ok.sum() < 40:
        raise SystemExit(f"逆投影できたランドマークが {int(ok.sum())}点しかない")
    log(f"対象 {len(tgt.vertices):,}頂点  逆投影できた68点: {int(ok.sum())}")

    # --- 位置合わせ: 対象をテンプレ空間へ --------------------------------
    #
    # **テンプレートを動かさず、対象を寄せる。** テンプレートは体と同じ座標系に
    # 居なければならない。ここでスケールを吸収するので、頭の絶対寸法は
    # テンプレートのものになる。接合方式が 69% 間違えていた量である。
    s, r, t, keep = robust_umeyama(L_t[ok], wv[lm_idx][ok])
    tgt.vertices = (s * (r @ np.asarray(tgt.vertices).T).T) + t
    lm_pos = ((s * (r @ L_t[ok].T).T) + t)[keep]
    lm_use = lm_idx[ok][keep]
    lm_no = np.arange(68)[ok][keep]
    log(f"外れ値を除いた68点: {int(keep.sum())}/{int(ok.sum())}  "
        f"除外 {sorted(set(np.arange(68)[ok]) - set(lm_no))}")
    res0 = np.linalg.norm(wv[lm_use] - lm_pos, axis=1)
    grp = np.where(lm_no < 17, 0, 1)
    log(f"相似変換で対象を寄せた: スケール {s:.4f}  68点の初期残差 "
        f"{np.sqrt((res0**2).mean())*1000:.2f}mm  "
        f"(輪郭0-16 {res0[grp==0].mean()*1000 if (grp==0).any() else 0:.1f}mm / "
        f"内側17-67 {res0[grp==1].mean()*1000:.1f}mm)")
    # 輪郭点は視線に対して斜めに当たるので逆投影が滑る。重みを分ける。
    lm_w = np.where(grp == 0, args.beta * args.contour_weight, args.beta)

    # --- 原点中心へ。世界座標のまま解くと y≈1.5 でアフィンの条件数が壊れる ---
    c0 = wv.mean(axis=0)
    wv_c = wv - c0
    tgt.vertices = np.asarray(tgt.vertices) - c0
    lm_c = lm_pos - c0

    vn0 = vertex_normals(wv_c, wf)
    w_face, d_lm = face_mask(wv_c, vn0, lm_c, args.face_falloff)
    if args.visibility:
        av, af, _ = all_geometry(doc, blob)
        vis = visible_mask(wv_c, av - c0, af)
        log(f"  可視性: 見える {int(vis.sum())} / {len(wv)}頂点"
            f"（隠れている分をデータ項から外す。眼球・歯・口内・後頭部）")
        w_face = w_face * vis
    # **眉より上はデータ項を切る。** 対象は生成された頭で前髪が額を覆う。
    # 実測では額の頂点が 152.5mm 動いて髪の表面に貼り付いた（頭の高さは 374mm）。
    # 観測できない領域は剛性項に任せる ——「証拠が無いなら動かさない」。
    if args.brow_cap:
        brow = lm_c[(lm_no >= 17) & (lm_no <= 26)]
        if len(brow):
            cap = float(brow[:, 1].max())
            band = args.brow_band
            above = np.clip((wv_c[:, 1] - cap) / max(band, 1e-9), 0.0, 1.0)
            w_face = w_face * (1.0 - above)
            log(f"  眉の上端 y={cap + c0[1]:.3f} から {band*1000:.0f}mm で重みを0へ")
    log(f"顔マスク: 重み>0.5 が {int((w_face>0.5).sum())}頂点 / {len(wv)}"
        f"（68点の XY 凸包の内側＋前向き）")

    x, hist = nicp(wv_c, wf, tgt, lm_use, lm_c, lm_w, w_face, args, log)
    moved = np.linalg.norm(x - wv_c, axis=1)
    face_m, out_m = moved[w_face > 0.5], moved[w_face <= 0.5]
    imax = int(moved.argmax())
    log(f"変位 顔面 平均{face_m.mean()*1000:6.2f} 最大{face_m.max()*1000:7.2f}mm / "
        f"マスク外 平均{out_m.mean()*1000:6.2f} 最大{out_m.max()*1000:7.2f}mm")
    log(f"  最大は y={wv[imax][1]:.3f}（頭は {wv[:,1].min():.3f}〜{wv[:,1].max():.3f}）"
        f"  顔マスク重み {w_face[imax]:.2f}")

    # --- 対応点を書く -----------------------------------------------------
    centers, targets = wv, x + c0
    if args.min_center_spacing > 0:
        pick = thin_centers(centers, args.min_center_spacing)
        log(f"中心を間引き: {len(centers)} → {len(pick)}点"
            f"（最小間隔 {args.min_center_spacing*1000:.1f}mm）")
        centers, targets = centers[pick], targets[pick]
    if args.centers and args.centers < len(centers):
        pick = farthest_point(centers, args.centers)
        centers, targets = centers[pick], targets[pick]
    anchors = np.asarray(json.loads(args.anchors.read_text())["points"], dtype=float)
    out_c = np.vstack([centers, anchors])
    out_t = np.vstack([targets, anchors])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "centers": [[round(float(q), 6) for q in p] for p in out_c],
        "targets": [[round(float(q), 6) for q in p] for p in out_t],
        "source": "preprocessing.wrap_head",
        "template": str(args.template), "target": str(args.target),
        "anchors": int(len(anchors)),
    }, ensure_ascii=False), encoding="utf-8")

    rep = {
        "template": str(args.template), "target": str(args.target),
        "face_material": matname,
        "vertices": {"raw": int(len(verts)), "welded": int(len(wv)),
                     "face_masked": int((w_face > 0.5).sum()),
                     "visible": int(vis.sum()) if args.visibility else None},
        "landmarks": {"backprojected": int(ok.sum()),
                      "initial_rms_mm": round(float(np.sqrt(
                          ((wv[lm_use] - lm_pos) ** 2).sum(1).mean())) * 1000, 3)},
        "align_scale": round(float(s), 6),
        # 対象（生成された頭）をテンプレート空間へ運ぶ相似変換。
        #   x_template = scale * (R @ x_target) + t
        # **髪を生成メッシュから切り出すのにこれが要る**（blender/add_hair.py）。
        # 保存していなかったので後から足した。
        "align": {"scale": float(s),
                  "rotation": [[float(x) for x in row] for row in r],
                  "translation": [float(x) for x in t]},
        "displacement_mm": {
            "face_mean": round(float(face_m.mean()) * 1000, 3),
            "face_max": round(float(face_m.max()) * 1000, 3),
            "outside_mean": round(float(out_m.mean()) * 1000, 3),
            "outside_max": round(float(out_m.max()) * 1000, 3),
            "argmax_y": round(float(wv[imax][1]), 4),
        },
        "iterations": hist,
        "pairs": {"centers": int(len(out_c)), "anchors": int(len(anchors)),
                  "head_centers": int(len(centers)),
                  "min_center_spacing_mm": round(args.min_center_spacing * 1000, 2)},
        "out": str(args.out),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log(f"対応点 {len(out_c)}点（うちアンカー {len(anchors)}）→ {args.out}")


if __name__ == "__main__":
    main()
