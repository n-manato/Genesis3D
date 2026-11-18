# 07 — MCP サーバー

Genesis3D を MCP ツールとして呼べるようにする。既存の
`AICCORE-Nexus/mcp-tools/server.py` と同じ形（FastMCP 系 + `add_tool` + stdio）。

## ⚠ 反映にはセッション再起動が必要

MCP サーバーは **Claude Code のセッション開始時に接続される**。登録しても、
実行中のセッションのツール一覧には現れない。**セッションを再起動して初めて使える。**

それまでは `scripts/gen.sh` を使う（後述）。

## 構成

```
genesis_mcp/server.py    17ツール。FastMCP(v1) / MCPServer(v2) どちらでも動く
scripts/mcp.sh           ランチャー（conda activate genesis → python -m genesis_mcp.server）
.mcp.json                プロジェクトスコープ登録
```

パッケージ名を `mcp/` に**してはいけない**。SDK の `mcp` を shadow して
`import mcp` が壊れる。そのため `genesis_mcp/` にしてある。

新しい venv は作らず、既存の `genesis` conda env に `mcp` を追加した（約50MB）。

## ツール

### ジョブ制御

| ツール | 内容 |
|---|---|
| `genesis_health` | orchestrator / TRELLIS.2 / Blender の到達性 |
| `genesis_gpus` | NVML 実測（空きVRAM・役割・予約・新規ロード可否） |
| `genesis_models` | 導入済みモデルと状態 |
| `genesis_create_job` | 画像 + style + quality → job_id（即返る） |
| `genesis_job_status` | 工程 / 進捗 / 割当GPU |
| `genesis_wait_job` | **完了まで待ち、工程サマリとVRM検証結果まで返す**。反復作業の主力 |
| `genesis_job_results` | 候補一覧・評価スコア・ランキング |
| `genesis_list_jobs` | 過去ジョブ一覧 |
| `genesis_cancel_job` | キャンセル |
| `genesis_artifact_path` | 成果物の絶対パス（PNG は Read で直接開ける） |

### 工程単体

任意のメッシュ・画像に対して個別に実行できる。候補の見比べや部分的なやり直し用。

| ツール | 工程 | 備考 |
|---|---|---|
| `genesis_preprocess_image` | 1 | RMBG-2.0。見切れ検出つき |
| `genesis_render_mesh` | 4 | 統一カメラ・照明。glb/obj/fbx/ply/vrm 対応 |
| `genesis_mesh_report` | 5 | 穴・非多様体・浮遊片・重複頂点 |
| `genesis_repair_mesh` | 7 | 浮遊片除去のみ。削りすぎ検知つき |
| `genesis_autorig` | 8 | テンプレート転写リギング |
| `genesis_export_vrm` | 9 | VRM 1.0。必須ボーン不足なら書き出さず失敗 |
| `genesis_validate_vrm` | 10 | Humanoid 必須ボーン・スケール・原点 |

### 設計上の約束

- すべて `127.0.0.1:8820` への HTTP か、Blender / conda のサブプロセス実行に閉じる
- ツールは**パスを返す**。巨大なバイナリを MCP 応答に載せない
- 長時間かかるものは `timeout_sec` を必ず持つ
- `configs/*.yaml` を書き換えるツールは**作らない**
- stdio でプロトコルを喋るため、**標準出力を汚さない**。サブプロセス出力は全て捕捉する

## 登録

```bash
# ユーザースコープ（どのプロジェクトのセッションからでも使える）
claude mcp add -s user genesis3d -- /home/ai-ccore/Documents/DEV/Genesis3D/scripts/mcp.sh

# プロジェクトスコープは .mcp.json に同梱済み
```

確認:

```bash
claude mcp list
# genesis3d: .../scripts/mcp.sh - ✔ Connected
```

## セッション再起動前に使う: scripts/gen.sh

```bash
./scripts/gen.sh <画像> [--style stylized|realistic_jrpg]
                        [--quality fast|high|maximum]
                        [--render] [--gateway]
```

ジョブ投入 → 完了待ち → 工程サマリ → VRM検証結果 → レンダPNGのパス、までを1コマンドで出す。
`--gateway` を付けると Gateway(:3000) 経由になる（`GENESIS_API_KEY` が必要）。

実測（`--quality fast --render`）: **47秒**で `character.vrm` まで到達。

## 途中で踏んだ問題

- **`.env` の値にスペースがあると bash の `source` が壊れる**。
  `RIG_TEMPLATE_VRM` のパスに空白が含まれており、`. ./.env` が
  `Characters/vrm/sample.vrm` をコマンドとして実行しようとしていた。
  結果、**その行以降の環境変数が bash 側で読まれていなかった**
  （Python 側 loader は正しく読めていたため気づきにくい）。値を引用符で囲んで解消。
- **`GENESIS_HOST` の意味が2つある**。`.env` では uvicorn の**バインドアドレス**
  (`127.0.0.1`)、Nexus の `genesis.ts` では**完全なURL**。MCP サーバーは
  スキームの有無を見てどちらでも解釈できるようにした。

## 更新（2026-08-24）: `genesis_create_job` に `face_image_path` を追加

顔と体を別メッシュで作る二トラック構成に対応した。

```
genesis_create_job(image_path, style, quality, output_format,
                   multi_candidate, face_image_path="")
```

`face_image_path` は顔に貼る実写ポートレート。生成したTポーズ画像から体を作る
場合、その顔は本人ではない（実測 ArcFace 0.20）ので、ここに実写を渡すと
工程 `head_merge` が顔だけ実写から作って接合する（同 0.66）。

**省略しても構わない。** 工程1が見切れを検出すると工程2が全身Tポーズを生成し、
そのとき入力画像が自動的に顔用として立つ（ポートレート1枚だけで完結、同 0.71）。

CLI は `./scripts/gen.sh <画像> [--face <実写>]`。
Gateway は `POST /v1/3d/characters` の multipart に `face_image` を追加した。
