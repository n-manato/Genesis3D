# 03 — 実験管理・ベンチマーク設計

目的: モデルを客観比較し、`styles.yaml` の重みを根拠を持って調整できるようにする。

## 評価項目（各 0〜10点）

| 項目 | 何を測るか | 測定方法 | 自動/人手 |
|---|---|---|---|
| **Identity** | 入力人物との同一性 | ArcFace 顔埋め込みのcos類似度 + DINOv3 全身特徴のcos類似度 | 自動 |
| **Face** | 顔の造形品質 | 顔領域クロップのCLIP-IQA + ランドマーク左右対称性 + 目/鼻/口の破綻検出 | 自動＋人手 |
| **Body** | 人体比率の正しさ | 頭身比・肩幅/腰幅比・四肢長を統計レンジと比較 | 自動 |
| **Hands** | 手指の破綻 | 手領域の連結成分数・指の本数推定・貫通検出 | 自動＋人手 |
| **Hair** | 髪の品質 | **後頭部・前髪・サイド・浮遊房・厚み**を個別採点（§19） | 自動＋人手 |
| **Clothes** | 衣服の品質 | シワの解像感・体との貫通・境界の明瞭さ | 自動＋人手 |
| **Back** | 背面の推定精度 | 背面レンダのCLIP-IQA + 正面との整合性 + のっぺり度（法線分散） | 自動 |
| **Texture** | テクスチャ品質 | 解像感・シーム・UV歪み・色にじみ | 自動 |
| **Geometry** | 幾何アーティファクト | 穴・非多様体・反転法線・浮遊ジオメトリの個数 | 自動 |
| **Topology** | トポロジーの素性 | 三角形品質分布・頂点密度の偏り・エッジフロー | 自動 |
| **Rigging** | リグ適性 | UniRigが必須ボーンを全て生成できたか・スキンウェイトの妥当性 | 自動 |
| **VRM** | VRM実用性 | Humanoid必須ボーン充足・スケール・原点・MToon変換後の見え方 | 自動 |
| Generation Time | 生成時間（秒） | 実測 | 自動 |
| VRAM | ピークVRAM（GB） | NVML実測 | 自動 |

自動指標は0〜10へ正規化して保存し、人手採点は同じ0〜10で上書き可能にする（`scores.json` に `auto` と `manual` を併記）。

## Overall Score

要件§12の重みを基準にし、**Generation Time の影響を極小（2%）** にする。

```
Overall = 0.20*Identity + 0.20*Face + 0.15*Body + 0.10*Hair + 0.10*Clothes
        + 0.10*Texture + 0.05*Topology + 0.05*Back + 0.05*Rigging
        - 0.02*TimePenalty
```

`TimePenalty = min(1.0, generation_time_sec / 1800)` を10点満点に換算。30分かけても減点は最大0.2点にしかならず、**品質差1点は時間差30分を常に上回る**。

`Hands` / `Geometry` / `VRM` は加点ではなく**ゲート**として扱う。いずれかが閾値未満の候補は Overall によらず `BEST_OVERALL` の対象から除外する（VRMとして使えないものを高得点にしないため）。

```
Gate: Hands >= 4.0 かつ Geometry >= 5.0 かつ VRM >= 7.0
```

## style別の重み

`configs/styles.yaml` で切替可能にする。

| 項目 | stylized | realistic_jrpg | 変更理由 |
|---|---|---|---|
| Identity | 0.15 | **0.25** | リアル系は本人性が生命線。デフォルメ系は元画像から意図的に離れる |
| Face | **0.25** | 0.20 | デフォルメ系は目の大きさ・輪郭の整い方が印象を支配する |
| Body | 0.10 | **0.20** | リアル系は人体比率の破綻が即座に不自然に見える |
| Hair | 0.10 | 0.10 | 両者とも最頻出の破綻要因 |
| Clothes | 0.10 | 0.10 | — |
| Texture | 0.15 | 0.10 | デフォルメ系は滑らかな肌の質が効く |
| Topology | 0.05 | 0.05 | — |
| Back | 0.05 | 0.05 | — |
| Rigging | 0.05 | 0.05 | — |

## ベンチマーク設計

### 入力データセット

styleごとに **10枚**の入力画像を固定する。すべて `benchmarks/inputs/<style>/` に置きハッシュを記録。

| # | 想定ケース | 意図 |
|---|---|---|
| 1-2 | 正面・全身・単純背景 | ベースライン |
| 3-4 | 斜め・上半身 | 隠れ部位の推定能力 |
| 5 | 長い髪 / 複雑な髪型 | §19の重点項目 |
| 6 | 複雑な衣装（レイヤー・装飾） | 衣服の破綻耐性 |
| 7 | 手が体から離れたポーズ | Handsゲートの検証 |
| 8 | 暗い/低コントラスト | 前処理の頑健性 |
| 9 | 背景が複雑 | RMBG-2.0の分離精度 |
| 10 | 極端なデフォルメ / 極端なリアル | style両端の耐性 |

### 比較対象モデル

要件§26③は「最低6モデルずつ」ですが、ディスク枠の都合で段階的に埋めます。

| Tier | stylized | realistic_jrpg | 前提 |
|---|---|---|---|
| **Tier 0+1（今回）** | TRELLIS.2, CharacterGen | **TRELLIS.2, PSHuman** | 約90GB枠（`/usr/share/ollama` 53GB回収後） |
| Tier 2 | + Hunyuan3D 2.1, InstantMesh, SFast3D, TripoSR | + Hunyuan3D 2.1, SiTH, SIFU, ECON | ドライブ増設 |

**両styleとも2モデル**でスタートできます。「6モデル比較」は Tier 2 到達時。表とDBスキーマは6モデル分を想定して作り、そのまま埋められる形にします。

なお TRELLIS.2 の多シード×多解像度は、モデル比較とは別軸の**条件比較**として同じスキーマに記録します（`model` は同一、`seed` と `parameters` が異なる行）。

### 記録スキーマ（SQLite `experiments.db`）

要件§23の全項目 + §22の再現情報。

| カラム | 型 | 備考 |
|---|---|---|
| `experiment_id` | TEXT PK | |
| `job_id` | TEXT | jobs/<job_id> と対応 |
| `timestamp` | TEXT | ISO8601 |
| `model` / `model_version` / `git_commit` | TEXT | 再現性 |
| `style` | TEXT | stylized / realistic_jrpg |
| `input_path` / `input_hash` | TEXT | SHA-256 |
| `seed` | INTEGER | |
| `parameters` | TEXT(JSON) | 解像度・texture_size等 |
| `resolution` | INTEGER | |
| `gpu` | INTEGER | 0 / 1 |
| `generation_time_sec` | REAL | |
| `vram_peak_gb` | REAL | NVML実測 |
| `polygon_count` / `texture_resolution` | INTEGER | |
| `identity` `face` `body` `hands` `hair` `clothes` `back` `texture` `geometry` `topology` `rigging` `vrm` | REAL | 各0-10。auto値 |
| `*_manual` | REAL NULL | 人手採点の上書き |
| `overall_score` | REAL | 上式で算出 |
| `gate_passed` | INTEGER | 0/1 |
| `output_path` | TEXT | |

CSV / JSON へのエクスポートを用意し、将来のWeb UIからそのまま読める形にする（§23）。

### 実行方法

```bash
# ※ scripts/run_benchmark.sh は未実装。現状は gen.sh を回して
#    jobs/<id>/scores/scores.json を突き合わせる。
./scripts/gen.sh <画像> --style stylized --quality maximum
```

各 (入力 × モデル × seed) を1 experiment として記録。10枚 × 2モデル × 3seed = 60 experiments が Tier 0 の stylized 一回分。
