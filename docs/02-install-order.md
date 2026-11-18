# 02 — モデル導入優先順位と導入順

## 前提: ディスクが最大の制約

| | 容量 | 状態 |
|---|---|---|
| 当初の空き | 17GB | — |
| ユーザー領域のキャッシュ整理 | +12GB | ✅ 実行済み |
| **現在の空き** | **29GB** | — |
| システム領域の整理（`--sudo`） | +61GB | ⏳ sudoパスワードが必要 |
| **達成可能な枠** | **約90GB** | — |

内訳（システム領域）: **`/usr/share/ollama` 53GB**（孤立した旧ollama置き場）/ CUDA・cuDNNインストーラ残骸 4.4GB / snap無効revision 10個 / journald / aptキャッシュ。

### ハードリンクによる過大評価に注意

`du` は uv / conda のキャッシュを大幅に過大評価する。両者は稼働中の venv・env へ**ハードリンク**しており、キャッシュ側のリンクを消してもデータは解放されない。

| 対象 | `du` の表示 | 実際に解放される量 |
|---|---|---|
| `~/.cache/uv` | 18GB | **4.73 GiB**（13.11 GiB は共有） |
| `~/miniconda3/pkgs` | 6.9GB | **0.22 GiB**（6.57 GiB は共有） |

`cleanup_disk.sh` は `nlink==1` のファイルのみを集計するため、表示される数値がそのまま実際の解放量になる。

### 削除しないもの

LLM重み（`~/.ollama/models` 181GB）、HFキャッシュ（27GB）、稼働中モデル（HY-Motion 26GB / world-perception runners 17GB）、`/usr/local/cuda-11.8`（**gesturelsm が torch 2.1.2+cu118 で現用中**）、そして**他ユーザーのホーム 240GB**（rodiseri 203G 他 — 共用マシンのため触らない）。

---

## モデル比較表

VRAM欄は「単体推論時のピーク（3090 24GB上の実測または公称）」。`—` は未文書。

| Pri | Model | Purpose | Quality | VRAM | 3090互換 | Multi-GPU | Python | PyTorch | CUDA | Input | Output | License | 導入難易度 | 採用理由 / 備考 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0** | **TRELLIS.2-4B** | 幾何+PBRテクスチャ / 再テクスチャ | ★★★★★ | 〜17GB (res512) | ✅ **導入済・稼働中** | ✗ | 3.10.20 | 2.6.0+cu124 | 12.4 | 単一画像 / mesh+画像 | GLB (PBR) | MIT | **済** | 2026年時点のオープン系image-to-3D SOTA。**既に:8811で稼働**。`Trellis2TexturingPipeline` により再テクスチャ工程も追加コストゼロで得られる |
| **1** | **Blender 4.2 LTS + VRM Add-on 4.5.0** | レンダ / mesh修復 / VRM書き出し | — | 〜2GB | ✅ | ✗ | 同梱 | — | — | GLB/FBX/OBJ | **VRM 1.0** | GPL / MIT | 低 | **VRM出力に必須**。sudo不要のportable tarballで `~/opt/blender` へ。Add-onはBlender 2.93〜5.2対応、VRM 0.x/1.0両対応、**Python APIで自動化可能** |
| **2** | **UniRig** | 自動リギング | ★★★★☆ | 〜8GB | ✅ | ✗ | 3.11 | ≥2.3.1 | 12.x | obj/fbx/glb/**vrm** | fbx/glb | 要確認 | 中 | SIGGRAPH'25。**VRMを入出力でサポート**し、VRoidデータセットで学習＝VRM Humanoid骨格と相性が良い。VRM-Addon-for-Blenderの改造版を同梱。VRAM 8GBでGPU0に収まる |
| **3** | **CharacterGen** | stylized 幾何 + マルチビュー | ★★★★☆ | 〜10GB (推定) | ✅ | ✗ | 3.9 | — | — | 単一画像 | OBJ (**A-pose**) | 要確認 | **高** | SIGGRAPH'24。VRoid Hub由来データで学習＝アニメ調に最適。**A-pose正規化出力**がVRM Humanoid割当の成功率を大きく上げる。依存が古くtorch/CUDA未文書のため conda env 完全分離が必須 |
| 4 | PSHuman | realistic 幾何 | ★★★★★ | 〜16GB (推定) | ✅ | ✗ | 3.10 | — | 12.x | 単一画像 | mesh | 要確認 | 高 | **Tier 1 の最優先追加**。論文ベンチで PIFu/PIFuHD/ICON/**ECON**/GTA/**SIFU**/HiLo/**SiTH** をすべて上回る。local ID diffusion により顔ディテールが強い |
| 5 | Hunyuan3D 2.1 | 幾何 + PBR（第3の比較軸） | ★★★★☆ | 幾何≥3GB / テクスチャ≥6GB / 公式フル約29GB | △ 逐次ロード必須 | ✗ | 3.10 | — | 12.x | 単一画像 | GLB (PBR) | Apache-2.0 | 中 | Tier 2。フルパイプラインが約29GBでGPU1の空き7.1GBに載らない。段階ロードで回避可能だが優先度は低い |
| 6 | InstantMesh | 高速な追加候補 | ★★★☆☆ | 〜10GB | ✅ | ✗ | 3.10 | — | 11.8/12.x | 単一画像 | OBJ/GLB | Apache-2.0 | 低 | Tier 2。品質はTRELLIS.2に劣るが候補数を増やす用途 |
| 7 | SiTH | realistic 幾何 | ★★★☆☆ | 〜12GB | ✅ | ✗ | 3.10 | — | 11.8 | 単一画像 | mesh | 要確認 | 中 | Tier 2。**未見部分のテクスチャに非現実的アーティファクト**が出ると論文で明示されており、PSHumanに劣後 |
| 8 | SIFU | realistic 幾何 | ★★★☆☆ | 〜12GB | ✅ | ✗ | 3.8 | — | 11.x | 単一画像 | mesh | 要確認 | 中 | Tier 2。幾何は良好だが**未見部テクスチャの生成に失敗する**と論文で明示。幾何のみ採用し工程6で貼り直す使い方なら価値あり |
| 9 | ECON / TeCH | realistic 幾何 | ★★★☆☆ | 〜12GB | ✅ | ✗ | 3.8 | — | 11.x | 単一画像 | mesh | 要確認 | 高 | Tier 2。PSHumanに包含されるため優先度最低 |
| — | Stable Fast 3D / TripoSR | 高速候補 | ★★☆☆☆ | 〜6GB | ✅ | ✗ | 3.10 | — | 12.x | 単一画像 | GLB | 一部非商用 | 低 | 品質がTRELLIS.2に大きく劣り、品質最優先の本要件では採用しない |

### ご指定順からの変更点

要件§6では realistic の優先順位が **SiTH → SIFU → Hunyuan3D → TRELLIS → ECON/TeCH** でした。調査結果に基づき次のとおり変更しています。

1. **PSHuman を realistic 主力に昇格**。PSHuman論文の比較実験で SiTH・SIFU・ECON を含む既存手法すべてを上回っており、特に顔と衣服のシワで優位。
2. **SiTH / SIFU を降格**。両者とも「未見部分のテクスチャ生成に失敗/アーティファクト」と明示されている。本パイプラインは工程6で再テクスチャリングするため幾何のみ使う手はあるが、その用途ならPSHumanで足りる。
3. **TRELLIS.2 を realistic でも主力に**。既に導入済みかつオープン系SOTAであり、追加ディスクゼロで最高品質が得られる。

---

## 実際の導入順

依存衝突の回避が最重要。**one model = one conda env** を厳守する。venvではなくcondaを選ぶ理由は、TRELLIS.2が既にcondaで構築されており、かつ CUDA拡張（flash-attn/nvdiffrast/spconv/torch_scatter）のビルドに env-local な CUDA toolkit が必要なため。Dockerは**sudo権限がなく使用不可**。

### Tier 0 — 約45GB（今回実施）

| 順 | 対象 | 容量 | env | 前提 | 検証 |
|---|---|---|---|---|---|
| **0** | `cleanup_disk.sh --apply` → `--sudo` | −73GB | — | sudoパスワード | `df -h` で空き85GB以上 |
| **1** | Blender 4.2 LTS + VRM Add-on 4.5.0 | ~1.5GB | 同梱Python | sudo不要 | headlessで `sample.vrm` をインポートできる |
| **2** | Genesis3D orchestrator 骨格 + TRELLISアダプタ | ~0.5GB | `genesis` (py3.11) | :8811稼働 | 画像→GLB のE2E |
| **3** | UniRig | ~13GB | `unirig` (py3.11) | torch≥2.3.1, spconv, torch_scatter | GLB→rigged GLB |
| **4** | 評価器 | ~2GB | `genesis` に同居 | CLIP/DINOv3はHFキャッシュ流用 | scores.json 生成 |
| **5** | CharacterGen | ~18GB | `charactergen` (py3.9) | 依存が最も脆いので**最後** | 画像→A-pose OBJ |
| | ジョブ作業領域 | ~10GB | — | retention付き | — |
| | **合計** | **~45GB** | | | |

**CharacterGenを最後に置く理由**: Python 3.9固定でtorch/CUDAバージョンが未文書のため、導入失敗リスクが最も高い。先にTRELLIS.2単独でVRMまでのE2Eを通しておけば、CharacterGenが入らなくてもシステムは動作する。

### Tier 1 — PSHuman（約90GB枠なら同時に導入可能）

| 順 | 対象 | 容量 | 効果 |
|---|---|---|---|
| 6 | PSHuman | ~25GB | **realistic_jrpg が2モデル体制になる**（当初の最大の穴が埋まる） |

`/usr/share/ollama` の53GB回収により、**LLM重みを1つも削らずに PSHuman が入る**。当初は「最小のGGUF blob 1つ（23.3GiB）を消す必要がある」としていたが、その必要はなくなった。

残枠の見積り: 90GB − Tier 0（45GB）− PSHuman（25GB）= **約20GB**。Hunyuan3D 2.1（40GB）はまだ入らない。

### Tier 2 — ドライブ増設後

Hunyuan3D 2.1 → InstantMesh → SIFU（幾何のみ）→ SiTH → ECON の順。この段階で§26③の6モデル×2スタイルのベンチマークが完全に実施できる。

---

## 依存衝突の回避策

| リスク | 対策 |
|---|---|
| torch/CUDAバージョン差 | conda env 完全分離。envをまたぐ import を一切しない |
| プロセス内でのVRAM残留 | 各モデルを `conda run -n <env> python -m adapters.<name>.runner` の**独立サブプロセス**で実行し、プロセス終了でVRAMを完全解放（§31） |
| CUDA OOM / segfault のシステム全体波及 | サブプロセス境界で遮断。1モデル失敗でもジョブは残り候補で続行（§30） |
| conda pkgs キャッシュの再肥大化 | 各インストール後に `conda clean -a` を実行（`envs/*.sh` に組み込み済みにする） |
| HFからの再ダウンロード | 導入後は `HF_HUB_OFFLINE=1`（既存 `trellis-service/run.sh` と同方針） |
| 既存サービスとのVRAM衝突 | Genesis3DはGPU1に新規モデルを載せない。GPU0でもNVMLで空きを確認してから起動 |

---

## 訂正（2026-08-24）: PSHuman はこの機材では動かない

本書は PSHuman を「VRAM ~16GB (est) / Tier 1 最優先追加」としていたが、
導入前に一次情報を当たったところ **公式 README は 768解像度モデルの推論に
「over 40GB of VRAM」を要求**していた（https://github.com/pengHTYX/PSHuman）。
学習ではなく**推論**の数字で、著者は RTX 4090(24GB) 向けに 512版を
**学習し直す**ことを検討中と書いている。見積りが **2.5倍以上外れていた**。

| | 実測の空き |
|---|---|
| GPU0 | 15.1GB（他サービスが約9GB常駐） |
| GPU1 | 23.3GB（TRELLIS を On Demand 化した後） |
| カード上限 | 24GB |

### GPU 2台では解決しない

合計 48GB あるが、**要件は1枚あたり 40GB**。推論スクリプトは
`CUDA_VISIBLE_DEVICES=$GPU`（単数）の単一GPU実行で、device_map・モデル並列・
CPU offload のいずれも公式には未対応。分割されないモデルに 24GB×2 は使えない。
自前でシャーディングを入れる案は、多視点アテンションを GPU 跨ぎで動かすことに
なり、他者の研究コードへの改造として成否が読めない。

### 試す余地があるとすれば

`num_views=7` / `crop_size=740` は推論時の引数で、メモリは概ね
（視点数 × 解像度²）に比例する。視点を減らせば 23GB に収まる可能性はある。
ただしチェックポイントは 768/7視点で学習されており、著者自身が
512版は学習し直しが要ると書いていることから、単に下げるだけでは品質が落ちる。
**25GB の取得と env 構築を先に払う賭け**になる。

→ 現時点では PSHuman を選択肢から外す。realistic_jrpg の2モデル化は、
512版の公開を待つか、別手法（SiTH/SIFU/ECON は本書で PSHuman に劣ると
結論したが、VRAM 要件は同様に一次情報で再確認が要る）を当たり直す。

なお PSHuman 無しでも、現構成（TRELLIS.2 + 参照ビュー生成 + 顔/体の二トラック）で
ポートレート1枚から ArcFace 0.69〜0.71 に到達している
（docs/05-measurements.md §20〜§22）。

### 表の訂正

| # | モデル | 旧記載 | 実際 |
|---|---|---|---|
| 4 | PSHuman | VRAM ~16GB (est) / Tier 1 最優先 | **VRAM 40GB超（1枚あたり）/ 導入不可** |
