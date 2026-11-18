# Genesis3D

画像1枚から、複数の3Dモデルで候補を生成し、自動評価で最良を選び、リギングして
**VRM 1.0** まで持っていく完全ローカルのパイプライン。RTX 3090 ×2 上で、
生成速度ではなく**最終3Dモデル品質の最大化**を目的とする。

入口は同一PC上の **AICCORE-Nexus Gateway**。Genesis3D は Gateway を書き換えず、
既存のサイドカーパターン（motion-service / gesture-service / trellis-service と同型）で接続する。

```
Client → AICCORE-Nexus Gateway (:3000)  →  Genesis3D (:8820)  →  10工程  →  character.vrm
```

## ドキュメント

| | 内容 |
|---|---|
| [docs/01-pipeline.md](docs/01-pipeline.md) | stylized / realistic_jrpg 各工程の使用モデル・GPU・入出力・VRAM |
| [docs/02-install-order.md](docs/02-install-order.md) | モデル比較表と Tier 0/1/2 の導入順、依存衝突の回避策 |
| [docs/03-benchmark.md](docs/03-benchmark.md) | 評価項目・Overall Score・style別重み・実験管理スキーマ |
| [docs/04-runbook.md](docs/04-runbook.md) | input.png → character.vrm の具体的手順・API・検証 |
| [docs/05-measurements.md](docs/05-measurements.md) | **実測メモ** — 設計時の想定と食い違った点（既定値・解像度・背景除去・VRM API） |
| [docs/06-gateway-integration.md](docs/06-gateway-integration.md) | Gateway 統合（追加した API・非同期設計・変更ファイル・検証状況） |
| [docs/07-mcp.md](docs/07-mcp.md) | **MCP サーバー**（17ツール・登録方法・gen.sh） |
| [docs/08-game-avatar.md](docs/08-game-avatar.md) | **game_avatar** — 固定トポロジ方式。出荷ゲームの調査と、テンプレート素体を変形する設計 |
| [docs/09-wrap-avatar.md](docs/09-wrap-avatar.md) | **ラップ方式（現行）** — 生成メッシュは測るために回し、出荷はテンプレートのトポロジ。接合と削減を廃止 |

## 現状

**ポートレート1枚から、脚と足のある全身リグ付き VRM まで到達する**（実測 323秒）。
実写の顔を投影して ArcFace 0.68〜0.71（同一人物の実写どうしが 0.79）。
書き出したVRMは工程11で**骨を回して変形を確認**してから出る。
全身画像を直接入れる場合は約110秒。

**方向転換（2026-09）**: 写真ごとにメッシュを生成するのをやめ、**1体のテンプレート素体を
変形する**第3のスタイル `game_avatar` を足した。残っていた4項目（指が動かない・肩が破断・
spring bone が無い・髪が焼き込み）は「毎回メッシュが違う」という同じ原因から出ていたため、
固定トポロジにすると構造的に消える。→ [docs/08-game-avatar.md](docs/08-game-avatar.md)

**さらに方向転換（2026-09-09）**: 接合をやめて**ラップ**（非剛体 ICP）にした。
生成メッシュは**測るためだけ**に回し、出荷するのはテンプレートのトポロジ。
接合方式で KEHAI に出ていた2つの欠陥は、どちらも生成メッシュを出荷物にしたがゆえの
後処理（削減と接合）が原因だった —— 頭が正しい寸法の 69% に縮んで付いており
（スケール推定2系統が 44.6% 食い違い、根拠なく片方が採用されていた）、
顔の黒い斑点は 682,766→179,527頂点の削減が作っていた。
→ [docs/09-wrap-avatar.md](docs/09-wrap-avatar.md)

| | 接合（v2） | **ラップ（現行）** |
|---|---|---|
| 頂点数 | 129,458 | **5,565** |
| 身長 | 1.5229m（指定と 7.7cm ずれ） | **指定どおり** |
| モーフ | 53（転写・2回壊した） | **175（テンプレ本来）** |
| held-out ArcFace バスト | 0.6189 | **0.6764** |
| held-out ArcFace **全身** | 0.5329 | **0.6714** |
| 出力サイズ | 32.9MB | **8.9MB** |
| 髪 | テンプレの金髪・ボリューム無し | **生成メッシュから切り出して穴の無い殻にしたジオメトリ＋写真の髪色（セグメンテーション）**（テンプレの束とカードは除去、襟足まで髪色） |
| 髪の揺れ | 無し | **VRM 1.0 + spring bone 3本** |
| 肘・手首の破綻 | 骨が 46〜61mm ずれる | **補正済み（手首の歪み 42.5%→10.9%）。肩の鎖は据え置き（動かすと肩が潰れる）** |
| 服の色 | テンプレのピンク＋デニム | **写真から転写** |
| 服の構造 | テンプレのシャツのまま | **胴に写真を投影（襟・前開き・インナー）** |
| 所要時間 | 395秒 | **161秒** |
| 削減・接合 | 有り | **無し** |

| Phase | 内容 | 状態 |
|---|---|---|
| 0 | ディスク整理 | 完了。`/usr/share/ollama` 53GB を回収済み |
| 1 | 設計4項目 | 完了（docs/01〜04） |
| 2 | Blender 4.2.23 LTS + VRM Add-on 4.5.0 | 完了。実物VRMの読み書きを検証済み |
| 3 | orchestrator + TRELLIS アダプタ | 完了。`POST /jobs` → GLB の E2E が **52秒**（2候補） |
| 4 | 前処理・レンダ・評価・修復・リギング・VRM書出 | 完了。10工程中8工程が実行される |
| 5 | 参照ビュー生成（SDXL+ControlNet）・顔/体の二トラック | **完了**。ポートレート1枚→全身リグ付きVRM（322秒・ArcFace 0.71） |
| 5b | CharacterGen / UniRig | 未着手 |
| 5c | PSHuman | **導入不可**。公式要件が 1枚あたり VRAM 40GB超・単一GPU実行で、24GB×2 では動かない（docs/02 末尾） |
| 7 | Gateway 統合 | **完了**。認証済み E2E で `character.vrm` のダウンロードまで確認（323秒 / 13MB） |
| 8 | MCP サーバー | **完了**。17ツール登録済み（`claude mcp list` で Connected）。**利用にはセッション再起動が必要** |
| 9 | **game_avatar（接合方式）** | 完了だが KEHAI で首の伸びと顔の斑点が出た。原因を測って 10 に移行。[docs/08](docs/08-game-avatar.md) |
| 10 | **ラップ方式（現行）** | **完了**。写真1枚 → **335秒**（生成済みの頭を使う場合）。頂点 32,788・モーフ 175・held-out ArcFace バスト 0.6764 / 全身 0.6714（接合方式は 0.6189 / 0.5329）。髪は生成メッシュから切り出し（テンプレの髪は束もカードも除去）、VRM 1.0 で spring bone 3本。KEHAI 登録名 `subject_a_v10`（docs/09 §28）。[docs/09](docs/09-wrap-avatar.md) |

```bash
# ラップ方式（現行・docs/09）。写真1枚 → 171秒。glb と VRM の両方が出る
./scripts/gen.sh <全身写真> --face <顔写真> --style wrap --height-cm 160

# Gateway / MCP / API 経由でも回せる（style=wrap）
curl -X POST http://127.0.0.1:8820/jobs -F "image=@<全身写真>" \
     -F "face_image=@<顔写真>" -F "style=wrap" -F "height_cm=160"

# 実機（KEHAI）でポーズ品質を測る（剛体であるべき部位が剛体か）
python3 scripts/kehai_pose_check.py --model subject_a_v7.yaml
python3 scripts/gen_wrap_avatar.py --face <顔写真> \
    --body <全身写真> --height-cm 160 --out-dir jobs/gw_001

# KEHAI の実機で確かめる（依存なしの最小 CDP クライアント）
python3 scripts/kehai_probe.py --model subject_a_v3.yaml
python3 scripts/kehai_probe.py --shot /tmp/kehai.png

# 同一性は必ず別途測る（**由来写真ではなく別セッションの実写**を基準に）
python3 scripts/measure_identity.py --model jobs/gw_001/output/avatar.glb \
    --refs subject_a/subject_face_tweed_head.png --out /tmp/ident.json

# game_avatar — 接合方式（比較用に残してある。docs/08）
python3 scripts/gen_game_avatar.py --face <顔写真> --out-dir jobs/ga_001

# ポートレート1枚だけ（工程2が全身Tポーズを生成し、顔は実写から作る）
./scripts/gen.sh subject_a/subject_face_downtown_1200.jpg --style realistic_jrpg --quality high

# 全身画像＋顔用の実写を明示する場合
./scripts/gen.sh <全身画像> --face <実写ポートレート> --style realistic_jrpg --quality high
```

実際に走る工程（ポートレート1枚・実測 323秒）:

| # | 工程 | 実装 | 結果 |
|---|---|---|---|
| 1 | 前処理 | RMBG-2.0 (BiRefNet) | 背景除去・正方形正規化・見切れ検出 |
| 2 | 参照ビュー生成 | RealVisXL V5.0 + ControlNet-OpenPose | **見切れを検出したときだけ**走り、全身Tポーズを生成して選別 |
| 3 | 幾何候補 | TRELLIS.2 × res1024/512 | 2候補 |
| 4 | レンダ | Blender EEVEE | 候補ごとに5視点＋メッシュ静的解析 |
| 5 | 評価 | DINOv3 + **ArcFace** | 基準はジオメトリの実入力。顔を後で差し替える構成では identity/face を重みから外す |
| 7 | mesh修復 | Blender bmesh | 浮遊片除去のみ。**工程6より先に走る** |
| — | 顔メッシュの接合 | head_crop → TRELLIS.2 → 実写投影 → merge_head | 顔は実写から、体は生成Tポーズから作って首で繋ぐ |
| 6 | 再テクスチャ | 実写の顔テクスチャ投影 | 接合側で投影済みならスキップ |
| 8 | リギング | テンプレート転写 | 167ボーン。**テンプレートの手作りウェイトを近接転送**（エンベロープだと髪の骨が胸を掴んでポーズで裂ける） |
| — | 表情 | ICT-FaceKit の変形転写 | **ARKit 命名の53個**。幾何は変えずに差分だけ移すので同一性が落ちない（docs/05 §43） |
| — | 髪 | **不採用** | DiffLocks のヘアカード化は全身の距離で「濡れた藁」になり撤回。TRELLIS の髪のまま（docs/05 §46, §48） |
| 9 | VRM書出 | Blender + VRM Add-on | VRM 1.0。書出後に **vrm_finalize** でテンプレート由来の同乗物（pixiv名義のmeta・ライセンス条項・サムネイル・眼球ボーン・spring bone）を落とす |
| 10 | 検証 | Humanoid validator + **スキン検査** | 必須17本＋任意7本・身長1.6m・原点足元・`skins=1` |
| 11 | **自動QA** | pose_check + mesh_report + 整列検査 | 骨を回して**実際に変形するか**を毎回確認する（門番ではなく点検） |

工程の順序と各工程の判断根拠は `output/job.json` に残る（選定ビュー・評価の基準と
除外指標・接合スケールの2系統推定・投影の画角と整列パラメータ）。

### リギングについて

本命の UniRig（13GB）は未導入のため、**既知の VRM Humanoid アーマチュアを
スケールして被せる転写方式**で代替している。image-to-3D の出力が T ポーズに
なるため成立する。限界は正直に記録してある:

- 骨の位置は身長で相似変形したうえで、**腕だけメッシュの実寸に合わせて伸ばす**
  （テンプレートの腕幅／身長は 0.860、image-to-3D の T ポーズは 0.986 で 15% 違う）。
  それ以外の体型差は残る
- **指は動かない**。生成メッシュの手は指の割れていない塊で、テンプレートの指の骨は
  40本すべてがメッシュの外に浮く。動かすと棘状に飛び散るので、指のウェイトは手に
  たたんである（docs/05 §38）
- テンプレートの**衣装用**二次ボーン（`J_Sec_*Tops*`）は変形から外す。入れたままだと
  体のウェイトの64%を握り、腕を動かすと肩から板が飛び出す（docs/05 §37）
- ポーズ時の破断は**ゼロではない**。全身をランダムに回す試験で伸び3倍超の辺が
  8329 → 753（テンプレートのウェイト転送による）。残りは肩・胸に集中。
  heat map は骨167本中130本がメッシュ外に浮くため解けない（docs/05 §37）
- spring bone（髪・スカートの揺れもの）は未設定。**テンプレートから同乗してくる分は
  `blender/vrm_finalize.py` が削除する** — 以前はこれが 22本そのまま残っていて、
  焼き込まれた衣服が揺れていた（docs/05 §36）

UniRig を導入したら `orchestrator/pipeline.py` の `_rigging` を差し替える。

実機の制約（2026-08-20 実測）:

| | |
|---|---|
| ディスク | 916GB中 空き**29GB**（キャッシュ整理済）。sudo分を実行すると **+61GB → 約90GB枠** |
| GPU0 | 24GB中 空き15.2GB → **Genesis3D 専有** |
| GPU1 | 24GB中 空き7.1GB（TRELLIS.2が17GB常駐）→ **:8811経由でのみ利用、新規モデルは載せない** |

> **2026-09-02 再実測でこの前提は変わっている。** TRELLIS.2 は常駐しておらず（:8811 も停止）、
> **GPU1 は 24.0GB 丸ごと空き**。GPU0 の 9.0GB は AICCORE-Nexus のサイドカー2つ
> （`ActionPlan-Code` :8893 と `model-service` :8000）で、Genesis3D 用の空きは 15.1GB。
> 「GPU1 に新規モデルを載せない」制約は撤回できる。なお `game_avatar` は
> GPU を使う工程が処理時間の上位に来ない（重いのは 68点検出と RBF 評価で、どちらも CPU）。

**約90GB枠**なら Tier 0（TRELLIS.2再利用 + CharacterGen + UniRig + Blender + 評価器 ≒45GB）に
加えて **PSHuman（25GB）も導入でき、realistic_jrpg が2モデル体制になります**。
LLM重みは1つも削らずに済みます。Hunyuan3D 2.1（40GB）はまだ入らず Tier 2。詳細は
[docs/02-install-order.md](docs/02-install-order.md)。

## 成果物を見る

```bash
cd Subject A && python3 -m http.server 8000   # → http://127.0.0.1:8000/viewer/
```

マウスで回せる three.js ビューア（`subject_a/viewer/`）。`file://` では fetch が
CORS で止まるので必ずサーバー経由で開く。詳細は `subject_a/viewer/README.md`。

## セットアップ

```bash
./scripts/cleanup_disk.sh          # dry run: 削除対象と容量を表示（ハードリンク考慮済）
./scripts/cleanup_disk.sh --apply  # ユーザー領域のみ（sudo不要・実行済み、+12GB）
./scripts/cleanup_disk.sh --sudo   # システム領域も（+61GB。sudoパスワードを要求）
```

`--sudo` の最大の対象は **`/usr/share/ollama`（53GB）** — 稼働中の ollama は
`OLLAMA_MODELS=~/.ollama/models` を使っており（systemd override 済み）、この旧デフォルト位置は
2025年6〜7月の `deepseek-r1:70b` と `devstral:latest` を残したまま孤立しています。
現行 `ollama list` に存在せず、稼働プロセスからの参照も0件です。

以降の手順は [docs/04-runbook.md](docs/04-runbook.md) を参照。

## 設計上の約束

- **127.0.0.1 のみ listen**。外部公開しない。重み取得後は `HF_HUB_OFFLINE=1` で完全オフライン。
- **one model = one conda env**。env をまたぐ import をしない（Docker は sudo 必須で使用不可）。
- **各モデルは独立サブプロセス**。CUDA OOM やクラッシュがシステム全体を落とさず、
  プロセス終了で VRAM が完全解放される。1モデル失敗してもジョブは残り候補で続行する。
- **既存 Gateway を書き換えない**。Nexus 側は新規ファイル追加と、既存3ファイルへの
  import 追記・分岐追加のみ。`prisma/schema.prisma` は変更しない。
