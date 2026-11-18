# 06 — AICCORE-Nexus Gateway 統合

Genesis3D を Gateway の唯一の入口から使えるようにする（要件§2, §3, §33）。
方針は**既存ファイルを書き換えず追加する**こと。実際の変更は
**5ファイルに49行追加・削除0行**と、新規ファイル7つ。

## 追加した API

| メソッド | パス | 用途 |
|---|---|---|
| `POST` | `/v1/3d/characters` | ジョブ作成。**202 で job_id を即返す** |
| `GET` | `/v1/3d/characters/{id}` | stage / progress / 割当GPU |
| `GET` | `/v1/3d/characters/{id}/results` | 候補一覧・スコア・成果物名 |
| `GET` | `/v1/3d/characters/{id}/results/{filename}` | `character.vrm` 等のダウンロード |
| `POST` | `/v1/3d/characters/{id}/cancel` | キャンセル |

既存の `POST /v1/3d/generations`（TRELLIS・同期・GLBバイト列を返す）とは
**別の API 名**（`3D Character` / カテゴリ `3D`）にした。管理画面で個別に
有効化・使用量追跡ができ、既存APIの挙動は一切変わらない。

## 非同期であることの設計上の帰結

Genesis3D のジョブは80秒〜数分かかる。Gateway が全期間 VRAM リースを握ると、
モデルスケジューラを長時間ブロックし、`beginLease` の15分セーフティタイマにも
抵触する。そこで:

- **`POST`** はリースを取るが、ジョブを投入して即座に返すので**リースは一瞬**。
  この一瞬のリースが「Genesis3D が起動していること」を保証する（On Demand 起動）。
- **`GET` / `cancel`** は `skipModelLease: true`。メタデータ読み取りが
  モデルの同時実行数上限に並ばないようにする（vLLM と同じ既存の作法）。
- ジョブ本体の GPU 管理は **Genesis3D 側が NVML で行う**。GPU0 で事前チェックし、
  GPU1（TRELLIS.2 が常駐）には新規モデルを一切ロードしない。

### 退避（eviction）の扱い

VRAM スケジューラの退避候補フィルタは `OLLAMA_MAP` / `SIDECAR_MODELS` /
`MOTION_MODELS` / `TRELLIS_MODELS` に含まれるモデルだけを対象にしている。
**`GENESIS_MODELS` は意図的にこのフィルタへ追加していない** — 実行中のジョブを
途中で殺すと数分の計算が失われるため。`unloadModel` には登録してあるので、
管理者が明示的に無効化したときだけ停止する。

## 変更したファイル

### 新規（7）

```
src/lib/genesis.ts                                       HTTPクライアント + GENESIS_MODELS
src/lib/genesis-process.ts                               起動/停止（pidfile方式）
src/app/v1/3d/characters/route.ts                        POST
src/app/v1/3d/characters/[id]/route.ts                   GET status
src/app/v1/3d/characters/[id]/results/route.ts           GET results
src/app/v1/3d/characters/[id]/results/[filename]/route.ts GET artifact
src/app/v1/3d/characters/[id]/cancel/route.ts            POST cancel
```

### 既存への追記のみ（5ファイル / 49行追加・0行削除）

| ファイル | 追記内容 |
|---|---|
| `src/lib/gateway.ts` | `ENDPOINT_API` に4パス |
| `src/lib/vram.ts` | `gpuAffinityForModel` に `case "genesis3d": return "GPU0"` |
| `src/lib/vram-scheduler.ts` | import 2行 + 常駐判定・load・unload の分岐 |
| `src/lib/worker.ts` | `reconcileModels()` に genesis3d の起動/停止分岐 |
| `prisma/seed.ts` | `ModelRegistry` と `ApiStatus` に1行ずつ |

`prisma/schema.prisma` は**変更していない**（マイグレーション不要）。

稼働中の DB へは `scripts/register-genesis3d.ts`（冪等な upsert）で登録する。
既存データには触れない。

## 別件で直したもの（Genesis3D とは無関係）

`npm run build` が **2026-08-06 以降ずっと失敗していた**。原因は tsconfig の
`include: ["**/*.ts"]` に対し `exclude` が `node_modules` のみで、`stack/` 配下の
Python 仮想環境に含まれる gradio の TypeScript ソースを型チェックが拾っていたため。
`.gitignore` は `**/.venv/` のみで、ドット無しの `venv` を拾えていなかった。

`tsconfig.json` の `exclude` に `**/venv/**` / `**/.venv/**` / `**/site-packages/**`
を追加して解消。元ファイルは `tsconfig.json.bak-genesis3d` に退避してある。

## 検証状況

| 項目 | 結果 |
|---|---|
| `npx tsc --noEmit`（src/） | エラー 0 |
| `npm run build` | 成功。新規5ルートが登録されている |
| 4ルートの疎通（認証なし） | すべて `401 invalid_api_key` = ルーティングと `gateRequest` が正しく連結 |
| クライアント契約の突合 | `GenesisHealth` / `GenesisJob` / results / 成果物DL / パストラバーサル拒否 すべて一致 |
| **認証済み E2E** | **完了**（2026-08-24。末尾の更新を参照） |

### 認証済み E2E に必要なもの

1. **Gateway の再起動** — 本番は `next start`（pid 202919）で稼働しており、
   起動時のビルドを保持している。新ルートは再起動するまで有効にならない。
   研究室の共用サービスなので、こちらでは再起動していない。
2. **API キー** — DB にある3件はいずれも実在の利用者のもの（ハッシュ保存）。
   検証のために他人の鍵を復号・使用することはしていない。

再起動と鍵の用意ができれば、次で完走するはず:

```bash
curl -X POST http://127.0.0.1:3000/v1/3d/characters \
  -H "Authorization: Bearer <APIキー>" \
  -F image=@input.png -F style=stylized -F quality=high
# → 202 {"job_id":"...","poll":"/v1/3d/characters/..."}

curl -H "Authorization: Bearer <APIキー>" \
  http://127.0.0.1:3000/v1/3d/characters/<job_id>
curl -H "Authorization: Bearer <APIキー>" -O \
  http://127.0.0.1:3000/v1/3d/characters/<job_id>/results/character.vrm
```

## 更新（2026-08-24）: `face_image` の追加と VRAM 登録値

### 追加した multipart フィールド

`POST /v1/3d/characters` に **`face_image`（任意・ファイル）** を追加した。
顔に貼る実写ポートレート。生成したTポーズ画像から体を作る場合その顔は本人では
ないので（実測 ArcFace 0.20）、これを渡すと Genesis3D が顔だけ実写から作って
接合する（同 0.66〜0.70）。省略時は Genesis3D 側が入力画像で代用する。

変更は既存2ファイルへの**追記のみ**（`prisma/schema.prisma` は未変更）:

| ファイル | 追記内容 |
|---|---|
| `src/lib/genesis.ts` | `genesisCreateJob(..., faceImage?: Blob \| null)` |
| `src/app/v1/3d/characters/route.ts` | `face_image` の受け取りと検証 |

### VRAM 登録値を実測に合わせた

`estimatedVramMaxGb` を **12 → 14** に更新（`scripts/register-genesis3d.ts` /
`prisma/seed.ts`）。工程2の参照ビュー生成（SDXL + ControlNet）の実測ピークが
**13.9GB** で、旧値を超えていたため。

あわせて `register-genesis3d.ts` の **upsert の `update` 分岐が `installed` しか
更新していなかった**のを直した。これまでは値を変えても再実行が無反応だった。

**GPU0 は Genesis3D 専有ではない。** `actionplan_service(:8893)` と `:8000` の
サイドカーが常時 9GB 使っており、参照ビュー生成時の GPU0 合計は 23.2GB/24GB
（余裕 1.3GB）まで上がる。このためパイプラインからは `--offload` 前提で
8.0GB のリースで回している。PSHuman(~16GB) を入れるならこの2プロセスを
退かせる算段が先。

### 検証状況

| 項目 | 結果 |
|---|---|
| `npx tsc --noEmit` | エラー 0 |
| `npm run build` | 成功。`/v1/3d/characters` 系5ルートが登録されている |
| `face_image` 付き POST の疎通（未認証） | `401 invalid_api_key` = ルーティングと `gateRequest` が正しく連結 |
| Genesis3D 直叩きでの `face_image` | **完走**（`./scripts/gen.sh <画像> --face <実写>`） |
| **認証済み E2E** | **完了**（2026-08-24）。下記 |

### 認証済み E2E（実測）

`./scripts/gen.sh subject_a/subject_face_downtown_1200.jpg --style realistic_jrpg --quality high --gateway`

| 段階 | 結果 |
|---|---|
| `POST /v1/3d/characters` | 202 で job_id を即返す |
| `GET /v1/3d/characters/{id}` | `completed` / stage `done` / progress 100 |
| `GET /v1/3d/characters/{id}/results` | 候補一覧・スコア・成果物名 |
| `GET .../results/character.vrm` | **HTTP 200 / 13,130,136 bytes / `model/gltf-binary`**（GLB マジック確認済み） |
| 総所要 | 323.6秒（工程2の参照ビュー生成を含む） |

### 検証用のサービスアカウント

DB の `ApiKey` は既存5件すべて**実在の個人アカウント**に紐づいている。他人の鍵を
復号することも他人名義で発行することもせず、**検証専用の口を切り離して用意した**:

```
scripts/create-genesis3d-e2e-key.ts
  user     : genesis3d-e2e@localhost（パスワードはランダムハッシュでログイン不可）
  category : 3D のみ許可
  key      : Genesis3D/.env の GENESIS_API_KEY へ書き込む（標準出力には出さない）
```

不要になったら管理画面で失効させるか
`UPDATE ApiKey SET status='revoked' WHERE keyPrefix='lag_VrMF';`。
再実行すると既存の鍵を失効させてから新しく発行する（重複を残さない）。
