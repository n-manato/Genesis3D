# 04 — フォルダ構成と実行手順（input.png → character.vrm）

## フォルダ構成

```
/home/ai-ccore/Documents/DEV/Genesis3D/
├── orchestrator/
│   ├── app.py                  # FastAPI :8820。全APIエンドポイント
│   ├── scheduler.py            # multiprocessing + SQLite ジョブキュー
│   ├── gpu_manager.py          # NVMLで空きVRAM確認・GPUリース
│   ├── pipeline.py             # 工程1〜10のオーケストレーション
│   ├── events.py               # job.* イベント通知（WebSocket / callback）
│   └── store.py                # jobs.db / experiments.db
├── adapters/
│   ├── base.py                 # ModelAdapter: load/generate/get_status/cleanup
│   ├── trellis/adapter.py      # :8811 HTTPクライアント（既存サービス再利用）
│   ├── charactergen/{adapter.py,runner.py}
│   └── unirig/{adapter.py,runner.py}
├── preprocessing/rmbg.py
├── multiview/{tpose_skeleton.py,generate_views.py,select_view.py}
├── evaluation/score.py                  # DINOv3 + ArcFace。render/metrics は score.py に統合
├── mesh_processing/{inspect.py,repair.py}
├── texture_processing/retexture.py
├── blender/
│   ├── render_views.py         # Blender --background で実行するスクリプト
│   ├── repair_mesh.py
│   └── export_vrm.py
├── vrm/{humanoid_map.py,expressions.py,validate.py}
├── configs/{models.yaml,gpu.yaml,styles.yaml,gateway.yaml}
├── envs/{setup_genesis.sh,setup_eval.sh,setup_imagegen.sh,setup_blender.sh}
├── jobs/<job_id>/…            # 01-pipeline.md の中間生成物レイアウト
├── benchmarks/inputs/<style>/
├── docs/                      # 本ドキュメント群
├── scripts/{cleanup_disk.sh,run.sh,gen.sh,mcp.sh,healthcheck.sh}
├── logs/
├── .env / .env.example
└── requirements.txt
```

## セットアップ手順（初回のみ）

```bash
cd /home/ai-ccore/Documents/DEV/Genesis3D

# 0. ディスク確保（削除対象を確認してから実行）
./scripts/cleanup_disk.sh              # dry run: 削除対象と容量を表示
./scripts/cleanup_disk.sh --apply      # 実行（約31GB回収 → 空き約48GB）

# 1. Blender 4.2 LTS + VRM Add-on 4.5.0（sudo不要、~/opt/blender へ）
./envs/setup_blender.sh

# 2. Genesis3D 本体の conda env
./envs/setup_genesis.sh                # conda create -n genesis python=3.11

# 3. 前処理と評価器（RMBG-2.0 / DINOv3 / ArcFace）
./envs/setup_eval.sh                   # conda create -n genesis-eval python=3.11

# 4. 参照ビュー生成（工程2。SDXL + ControlNet-OpenPose、約15GB）
./envs/setup_imagegen.sh               # conda create -n genesis-imagegen python=3.11

# 5. 設定
cp .env.example .env                   # GATEWAY_* / GENESIS_PORT / TRELLIS_HOST を確認
```

**UniRig / CharacterGen の setup スクリプトは存在しない**（未導入のため）。
PSHuman は要件が 1枚あたり VRAM 40GB超で**この機材では動かない**ことが判明した
（docs/02-install-order.md 末尾）。

各 setup スクリプトは末尾で `conda clean -a` を実行し、pkgsキャッシュの再肥大化を防ぐ。

## 起動

```bash
./scripts/run.sh                       # 127.0.0.1:8820 で起動（localhostのみバインド）
./scripts/healthcheck.sh               # /health /gpus /models を確認
```

前提: 既存の TRELLIS.2 サービスが `127.0.0.1:8811` で稼働していること。停止している場合は Gateway の Model Runtime 画面から起動するか、`AICCORE-Nexus/trellis-service/run.sh` を直接実行する。

## 実行 — Gateway を経由しない単体テスト

```bash
# ジョブ投入
curl -s -X POST http://127.0.0.1:8820/jobs \
  -F image=@/path/to/input.png \
  -F style=stylized \
  -F quality=maximum \
  -F output_format=vrm \
  -F models=auto \
  -F multi_candidate=true
# → {"job_id":"a81f...","status":"queued"}

# 進捗
curl -s http://127.0.0.1:8820/jobs/a81f...
# → {"job_id":"a81f...","status":"running","stage":"trellis_geometry","progress":63,"gpu":1}

# イベントをストリームで見る
websocat ws://127.0.0.1:8820/events/a81f...

# 結果
curl -s http://127.0.0.1:8820/jobs/a81f.../results
curl -sO http://127.0.0.1:8820/jobs/a81f.../results/character.vrm
```

## 実行 — Gateway 経由（最終形）

```bash
curl -s -X POST http://127.0.0.1:3000/v1/3d/characters \
  -H "Authorization: Bearer <APIキー>" \
  -F image=@/path/to/input.png \
  -F style=realistic_jrpg \
  -F quality=maximum
# → {"job_id":"..."}

curl -s http://127.0.0.1:3000/v1/3d/characters/<job_id> \
  -H "Authorization: Bearer <APIキー>"
```

Gateway側は既存の `gateRequest` でAPIキー認証・アクセス権・使用量記録を行い、`ModelRegistry` の `genesis3d` エントリを通じて VRAMリースを取得してから Genesis3D に転送する。

## 工程ごとのGPU割当と中間生成物

| 工程 | 実行形態 | GPU | 生成物 |
|---|---|---|---|
| 1 前処理 | `genesis` env 内 | GPU0 | `references/rgba.png`, `meta.json` |
| 2 マルチビュー | `conda run -n charactergen` | GPU0 | `references/view_*.png` |
| 3a 幾何(CharacterGen) | `conda run -n charactergen` | GPU0 | `candidates/charactergen/cand_*.obj` |
| 3b 幾何(TRELLIS.2) | HTTP → :8811 | GPU1 | `candidates/trellis/cand_*.glb` |
| 4 レンダ | `blender --background` | GPU0 | `renders/<cand>/*.png` |
| 5 評価 | `genesis` env 内 | GPU0 | `scores/scores.json`, `ranking.csv` |
| 6 再テクスチャ | HTTP → :8811 | GPU1 | `selected/best_textured.glb` |
| 7 mesh修復 | `blender --background` | CPU | `selected/clean.glb` |
| 8 リギング | `conda run -n unirig` | GPU0 | `rigged/rigged.glb` |
| 9 VRM化 | `blender --background` | CPU | `output/character.vrm` |
| 10 検証 | `blender --background` | CPU | `output/validation.json` |

工程3aと3bは**並列実行**する（GPU0とGPU1が別ジョブとして走る）。工程6はGPU1が空くのを待ってから実行。

## API 一覧

| メソッド | パス | 用途 |
|---|---|---|
| `POST` | `/jobs` | ジョブ作成 |
| `GET` | `/jobs/{job_id}` | ステータス（stage / progress / gpu を含む） |
| `GET` | `/jobs/{job_id}/results` | 結果一覧 + 候補スコア |
| `GET` | `/jobs/{job_id}/results/{filename}` | 成果物ダウンロード |
| `POST` | `/jobs/{job_id}/cancel` | キャンセル |
| `GET` | `/models` | 導入済みモデルと状態 |
| `GET` | `/health` | 死活 |
| `GET` | `/gpus` | NVML実測のGPU状態 |
| `WS` | `/events/{job_id}` | `job.created` `job.preprocessing` `job.generating` `candidate.completed` `job.evaluating` `job.rigging` `job.exporting` `job.completed` `job.failed` |

## 検証

```bash
# VRMの妥当性（Humanoid必須ボーン・スケール・原点・マテリアル）
~/opt/blender/blender --background --python vrm/validate.py -- \
  jobs/<job_id>/output/character.vrm

# 既知の正解VRMと比較（KEHAIの実物を参照に使う）
~/opt/blender/blender --background --python vrm/validate.py -- \
  "/home/ai-ccore/Documents/DEV/KEHAI/Live2D Characters/vrm/sample.vrm"
```

必須ボーン: `hips` `spine` `chest` `upperChest` `neck` `head` / `left|rightUpperArm` `lowerArm` `hand` / `left|rightUpperLeg` `lowerLeg` `foot`

## 既存Gatewayへの影響がないことの確認

```bash
cd /home/ai-ccore/Documents/DEV/AICCORE-Nexus
npm run build                                        # 型チェックが通る
curl -s 127.0.0.1:8811/healthz                       # TRELLIS.2 が従来通り
curl -s -X POST 127.0.0.1:3000/v1/3d/generations …   # 既存エンドポイントが従来通り
```

## 成果物の確認（VRM が本当に使えるか）

`export_vrm.py` の検証は「必須ボーンが割り当たっているか」「スキンが存在するか」
までしか見ない。**骨を回したときの変形**は別の話で、そこは動かして見るしかない。

```bash
$BLENDER_BIN --background --factory-startup --python blender/pose_check.py -- \
    --input jobs/<id>/output/character.vrm \
    --outdir /tmp/pose --poses arms,legs,neck,all --report /tmp/pose.json
```

`deformed: false` が出たらスキンが効いていない（docs/05-measurements.md §28）。
骨を1本ずつ動かせるのは、**症状が出る最小の操作を特定する**ため。
§29 では「連結成分が多いから」という仮説に2手費やした末に、骨ごとに切り分けて
初めて原因（首の骨の影響範囲）に辿り着いた。
