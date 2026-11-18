# 01 — 品質最優先パイプライン設計

対象: `stylized`（3Dアニメ映画的デフォルメ） / `realistic_jrpg`（高精細リアル寄りJRPG）

前提となる実機環境（すべて実測確認済み）:

| 項目 | 値 |
|---|---|
| GPU0 | RTX 3090 24GB — 既存サイドカーが約9.0GB使用、**空き約15.2GB** |
| GPU1 | RTX 3090 24GB — TRELLIS.2が17.0GB常駐、**空き約7.1GB** |
| CUDA | driver 12.8 / nvcc 12.4 |
| 既存TRELLIS.2 env | Python 3.10.20 / torch 2.6.0+cu124 |

**GPU割当の基本方針**

- **GPU0 = Genesis3D 専有**。CharacterGen / UniRig / 評価器 / Blender をここで時分割実行。
- **GPU1 = 既存TRELLIS.2 を `127.0.0.1:8811` 経由で利用**。Genesis3Dは GPU1 に新規モデルを載せない（空きが7.1GBしかないため）。HTTP経由なので**追加VRAM消費ゼロ**。

これにより要件§9「GPU0でModel A / GPU1でModel B」を、既存サービスを一切止めずに実現する。

---

## Style A — `stylized`

```
input.png
  │
  ├─[1] 前処理            GPU0   RMBG-2.0
  │                              → references/rgba.png, meta.json
  │
  ├─[2] マルチビュー生成   GPU0   CharacterGen 2D Stage
  │                              → references/view_{front,left,right,back}.png（A-pose正規化）
  │
  ├─[3] ジオメトリ候補生成（並列）
  │     ├─[3a] GPU0  CharacterGen 3D Stage × Nseed → candidates/charactergen/cand_{i}.obj
  │     └─[3b] GPU1  TRELLIS.2 (:8811) × Nseed × 解像度 → candidates/trellis/cand_{i}.glb
  │
  ├─[4] レンダリング       GPU0   Blender headless (EEVEE)
  │                              → renders/<cand>/{front,left,right,back,three_quarter}.png
  │
  ├─[5] 自動評価・順位付け GPU0   CLIP + DINOv3 + ArcFace + mesh静的解析
  │                              → scores/scores.json, scores/ranking.csv
  │
  ├─[6] 再テクスチャリング GPU1   Trellis2TexturingPipeline (:8811)
  │                              → selected/best_textured.glb（PBR: base/normal/roughness/metallic）
  │
  ├─[7] mesh修復           CPU    Blender bmesh
  │                              → selected/clean.glb
  │
  ├─[8] リギング           GPU0   UniRig (skeleton → skin → merge)
  │                              → rigged/rigged.glb
  │
  ├─[9] VRM化              CPU    Blender + VRM Add-on 4.5.0
  │                              → output/character.vrm (VRM 1.0)
  │
  └─[10] 検証              CPU    VRM Humanoid validator
                                 → output/validation.json
```

### 工程詳細

| # | 工程 | 使用モデル/ツール | GPU | 入力 | 出力 | VRAM(推定) | 次工程への受け渡し |
|---|---|---|---|---|---|---|---|
| 1 | 前処理 | RMBG-2.0（HFキャッシュ既存） | GPU0 | `input/input.png` | RGBA・正方形パディング・アルファ境界クリーンアップ | ~2GB | PNG(RGBA) + `meta.json`(元解像度/bbox/被写体スケール) |
| 2 | マルチビュー | CharacterGen 2D Stage | GPU0 | RGBA | 4面ビュー(front/left/right/back)、**A-poseへ姿勢正規化** | ~8GB | PNG×4 + カメラ方位 |
| 3a | 幾何候補 | CharacterGen 3D Stage | GPU0 | 4-view | `.obj` + テクスチャ、A-pose | ~10GB | OBJ+MTL+PNG |
| 3b | 幾何候補 | TRELLIS.2 Image-to-3D | GPU1(HTTP) | RGBA | `.glb` PBR付き | 0（外部プロセス） | GLB |
| 4 | レンダ | Blender 4.2 headless | GPU0 | 全候補 | 5方向PNG（統一ライティング・統一カメラ） | ~2GB | PNG群 + カメラ行列JSON |
| 5 | 評価 | CLIP ViT-L/14, DINOv3 ViT-L/16, ArcFace, trimesh | GPU0 | レンダ + 参照画像 | 項目別スコア | ~4GB | `scores.json` |
| 6 | 再テクスチャ | **Trellis2TexturingPipeline** | GPU1(HTTP) | 最良mesh + 参照画像 | 高品質PBRテクスチャ付きGLB | 0（外部プロセス） | GLB |
| 7 | mesh修復 | Blender bmesh | CPU | GLB | 穴埋め・法線修正・重複頂点除去・浮遊ジオメトリ除去 | 0 | GLB |
| 8 | リギング | UniRig | GPU0 | GLB | skeleton + skinning weights | ~8GB | GLB(rigged) |
| 9 | VRM化 | Blender + VRM Add-on 4.5.0 | CPU | rigged GLB | VRM Humanoidボーン割当・MToon変換・表情スロット | 0 | `.vrm` |
| 10 | 検証 | VRM Add-on validator + 自作チェック | CPU | `.vrm` | 必須ボーン/スケール/原点/マテリアル/テクスチャパス検査 | 0 | `validation.json` |

**工程6が品質の要**。§13で要求された「best geometry + best texture」の融合を、Boolean結合ではなく**別モデルによる再テクスチャリング**として実装する。工程3aで幾何が最良（CharacterGen）でも、テクスチャはTRELLIS.2で貼り直せる。追加ディスク・追加VRAMともにゼロ。

---

> **実装の更新（2026-08-24）**: 工程2は CharacterGen ではなく
> **SDXL(RealVisXL V5.0) + ControlNet-OpenPose** で全身Tポーズ参照を生成する形で
> 実装した（`multiview/`、`envs/setup_imagegen.sh`）。工程1が見切れを検出した
> ときだけ走る。あわせて、生成画像の顔は本人ではないため（実測 ArcFace 0.20）、
> 顔だけ実写から作って接合する工程 `head_merge` を追加した。
> 実装順は `mesh_repair → head_merge → retexture`（工程6が工程7の後）。
> 経緯と実測値は docs/05-measurements.md §12〜§20。

## Style B — `realistic_jrpg`

Tier 0 構成。工程2でCharacterGenのアニメ調正規化を使うと破綻するため、参照ビュー生成の方式を変える。

```
input.png
  │
  ├─[1] 前処理           GPU0   RMBG-2.0                       （共通）
  ├─[2] 参照ビュー        GPU0   TRELLIS.2の内部多視点 + 幾何プリアを利用
  ├─[3] 幾何候補          GPU1   TRELLIS.2 × Nseed × {512,1024} が主軸
  │                             （CharacterGenは弱い二次候補として任意）
  ├─[4]〜[10] stylized と共通
```

| # | 工程 | 使用モデル | GPU | VRAM | stylizedとの差分 |
|---|---|---|---|---|---|
| 2 | 参照ビュー | TRELLIS.2 内部多視点 | GPU1(HTTP) | 0 | A-pose強制正規化を**しない**（リアル系は元ポーズの人体比率を保持する方が identity が保たれる） |
| 3 | 幾何候補 | TRELLIS.2 多シード×多解像度 | GPU1(HTTP) | 0 | CharacterGen 3D Stage を主軸から外す |
| 5 | 評価 | 同左 + 人体比率チェック強化 | GPU0 | ~4GB | `styles.yaml` の weights を realistic 用に切替（Identity/Face の比重を上げる） |
| 8 | リギング | UniRig | GPU0 | ~8GB | A-poseでないため、リグ前にBlenderで**A-pose正規化を1ステップ挟む** |

### realistic_jrpg の2モデル化

当初はディスク47GB枠に PSHuman(~25GB) が入らず、realistic_jrpg は TRELLIS.2 単一モデルになる見込みだった。

その後の調査で **`/usr/share/ollama` に53GBの孤立した旧ollama置き場**が見つかり（稼働中の ollama は `~/.ollama/models` を使用、systemd override 済み）、これを回収すれば枠が**約90GB**になる。したがって **LLM重みを1つも削らずに PSHuman を導入でき、realistic_jrpg も工程3で2モデル体制**にできる。

```
工程3（realistic_jrpg）
 ├─[3a] GPU0  PSHuman × Nseed        → candidates/pshuman/cand_*.obj
 └─[3b] GPU1  TRELLIS.2 × Nseed×解像度 → candidates/trellis/cand_*.glb
```

PSHuman を選定した理由（SiTH/SIFU/ECON より優先する根拠）は `02-install-order.md` を参照。

---

## quality モードによる工程の増減

| quality | 候補数 | 工程2 マルチビュー | 工程5 評価 | 工程6 再テクスチャ | 工程7 mesh修復 |
|---|---|---|---|---|---|
| `fast` | 1（TRELLIS.2のみ, seed固定） | スキップ | 簡易（CLIPのみ） | スキップ | 基本cleanupのみ |
| `high` | 2〜4 | 実行 | 全項目 | 実行 | 実行 |
| `maximum` | 4〜8（全モデル×全seed） | 実行 | 全項目 + 複数レンダ角度 | 実行（BEST_GEOMETRY と BEST_TEXTURE 双方について） | 実行 + retopology検討 |

`quality=maximum` では §25 の全工程を実行し、`BEST_GEOMETRY` / `BEST_TEXTURE` / `BEST_OVERALL` の3つを保存する。

---

## 中間生成物のフォーマット規約

工程間の受け渡しは**ファイル + JSON manifest** で行う。プロセス境界をまたぐため、Pythonオブジェクトを直接渡さない。

```
jobs/<job_id>/
├── input/input.png                   # 原本（ハッシュ記録）
├── references/
│   ├── rgba.png                      # 工程1出力
│   ├── meta.json                     # bbox, scale, 元解像度
│   └── view_*.png                    # 工程2出力
├── candidates/
│   ├── <model>/cand_<i>.glb|obj
│   └── <model>/cand_<i>.json         # seed, params, 生成時間, VRAM peak, model version
├── renders/<cand_id>/*.png
├── scores/
│   ├── scores.json                   # 候補×評価項目の生スコア
│   └── ranking.csv                   # 重み付け後の順位
├── selected/
│   ├── best_geometry.glb
│   ├── best_textured.glb             # 工程6出力
│   └── clean.glb                     # 工程7出力
├── rigged/rigged.glb
└── output/
    ├── character.vrm
    ├── validation.json
    └── job.json                      # §22の完全な再現情報
```

各 `cand_<i>.json` に model version / git commit / seed / パラメータ / GPU / 生成時間 / VRAM peak を記録し、§22の再現性要件を満たす。
