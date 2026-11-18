# テンプレート素体の候補

`game_avatar`（`docs/08-game-avatar.md`）は、写真ごとにメッシュを起こすのをやめて
**1体のテンプレート素体を変形させる**。テンプレートの品質がそのまま出力の品質上限になるので、
候補は `scripts/check_template_vrm.py` の8項目で検査してから採用する。

```bash
python3 scripts/check_template_vrm.py <candidate.vrm>
```

## 採用

| | |
|---|---|
| 女性 | `/home/ai-ccore/Documents/DEV/KEHAI/Live2D Characters/vrm/sample.vrm` — pixiv `VRM1_Constraint_Twist_Sample`。**8項目すべて合格**、24,951頂点 |
| 男性 | **未定**。下記のとおり、無料で取得できる範囲では要件を満たすものが見つかっていない |

## 却下した候補（2026-09-02 調査）

同じ調査を繰り返さないための記録。

### madjin/vrm-samples の VRoid サンプル群 — 頂点数で失格

`https://github.com/madjin/vrm-samples` は VRoid 公式サンプルをミラーしている。
胸ボーンの本数が素体の性別をよく表しており（女性素体は `J_Sec_L/R_Bust1/2` の4本）、
**`AvatarSample_C` と `HairSample_Male` は0本＝男性素体**。`HairSample_Male` は
ライセンスも明示的に **CC0** で、採用中の女性テンプレートより緩い。

しかし**すべて VRM 0.x** で、かつ**削減が掛かっていない**:

| 候補 | 合計頂点 | 髪 | ライセンス |
|---|---|---|---|
| `HairSample_Male.vrm` | **965,948** | 886,350頂点 / **114プリミティブ** | CC0 |
| `AvatarSample_C.vrm` | **920,810** | 835,794頂点 / 118プリミティブ | Other（everyone / commercial allow） |
| 参考: 採用中の女性テンプレ | 24,951 | 12,868頂点 / 1プリミティブ | 改変・再配布可 |

**約38倍。** VRoid Studio の初期エクスポートで、ゲームアバターとして成立しない。
Blender で間引く案は、114プリミティブに分かれた髪の MToon マテリアル割り当てと
手塗りウェイトを壊すので採らない。

### Seed-san（VirtualCast） — 汎用素体ではない

`madjin/vrm-samples/Seed-san/vrm/Seed-san.vrm`。**VRM 1.0**、34,059頂点、
humanBones 51（指30）、結線済みプリセット17個と、形式面は良い。却下の理由:

- ロボット腕（`robo_arm`）を持つ**特定のキャラクター**であって、男性の汎用素体ではない
- `creditNotation: required`
- モーフ命名が `eye_close` / `blink_L` 系で、VRoid の `Fcl_*` とは別系統。
  ARKit 別名の対応表を素体ごとに別途持つことになる

## 次に当たるべきところ

**VRoid Hub の `AvatarSample_*_VRM10`。** 公式の AvatarSample 系列は VRM 0.0 版と
**VRM 1.0 版の両方**がアップロードされており（モデル名の `_VRM00` / `_VRM10` が版を表す）、
現行の VRoid Studio 書き出しなので削減が掛かっている。利用条件は営利・非営利を問わず
利用可・改変可・**再配布可**・クレジット不要。

ダウンロードに VRoid Hub のログインが要るため自動取得できない（`hub.vroid.com` は
HTTP 403 を返す）。取得したら上の検査器に通す。胸ボーン0本のものが男性素体。

代替として、VRoid Studio で男性素体を書き出す方法もある（削減設定を掛けること）。
ただし VRoid Studio は Windows / Mac のみで、この Linux 機では動かない。

---

## 入手先 — VRoid Hub 公式 AvatarSample（VRM 1.0）

公式の AvatarSample 系列は1つのキャラクターページに **Q〜Z が VRM00 / VRM10 の両版**で
置かれている。`_VRM10` が VRM 1.0 版で、現行の VRoid Studio 書き出しなので削減が掛かっている。

一覧: https://hub.vroid.com/en/characters/2175138617022572467

サムネイルを並べたもの: `assets/templates/avatarsample_qz.png`（2026-09-02 取得）

| | 見た目 | 直リンク |
|---|---|---|
| **Y** ★本命 | 男性・学生服（濃紺の髪、ブレザーとネクタイ） | https://hub.vroid.com/en/characters/2175138617022572467/models/5301857324365695083 |
| **Z** ★対抗 | 男性・学生服（Y とほぼ同系） | https://hub.vroid.com/en/characters/2175138617022572467/models/6485298233030563882 |
| W | 男性・和装の羽織 | https://hub.vroid.com/en/characters/2175138617022572467/models/7786195163720861113 |
| Q | 男性・赤髪の軍装風 | https://hub.vroid.com/en/characters/2175138617022572467/models/8463623704164693993 |
| S | 男性・銀髪褐色肌のタクティカル装 | https://hub.vroid.com/en/characters/2175138617022572467/models/3700033875907456842 |
| U | 男性寄り・眼鏡とマント | https://hub.vroid.com/en/characters/2175138617022572467/models/2969590353316340402 |
| R / T / V / X | 女性 | （省略） |

**Y を薦める理由**: 採用中の女性テンプレート（`sample.vrm`）が Tops / Bottoms / Shoes という
単純な構成なので、装飾の少ない学生服が一番近い。軍装（Q）やタクティカル装（S）は
`J_Sec_*` の衣装用二次ボーンが増え、和装（W）は袖が体から大きく離れる。

ダウンロードは VRoid Hub のログインが要る（`hub.vroid.com` は自動取得に HTTP 403 を返す）。
落としたら必ず検査器に通すこと:

```bash
python3 scripts/check_template_vrm.py <落としたVRM>
```

**胸ボーン0本が男性素体の目印**（女性素体は `J_Sec_L/R_Bust1/2` の4本）。


---

## 実写寄りトラック — Microsoft Rocketbox（2026-09-02 調査）

方針を「実写寄り（NBA 2K / 逆水寒 系）」に決めたため、権利的に使える中立な実写素体を探した。
**Microsoft Rocketbox が最有力。**

https://github.com/microsoft/Microsoft-Rocketbox

| | 実測値 |
|---|---|
| ライセンス | **MIT**（改変・再配布・商用・サブライセンス、すべて無条件） |
| 体数 | 成人 男21 / 女20、子供、職業別72 — **男女ペアが最初から揃う** |
| 頂点 | `Male_Adult_01` 4,025 ／ `Female_Adult_01` 4,734 |
| モーフ | **175** — **ARKit 52 完備**（`AK_09_EyeBlinkLeft` 〜 `AK_52_TongueOut`）＋ Oculus ビセーム15（`AA_VI_*`）＋ FACS Action Unit |
| テクスチャ | head / body それぞれ **2048²** の color + normal + specular（12MB の無圧縮TGA） |
| 骨 | 80本（指30本） |
| マテリアル | 3（body / head / opacity）— **顔だけを対象にできる** |
| FBX → GLB 変換 | Blender で通る。4.7MB、モーフ175を保持 |

各アバターに `<name>.fbx`（素体）と `<name>_facial.fbx`（表情つき）がある。使うのは後者。

### 準備に要る作業（どちらも機械的）

KEHAI 検査に落ちるのは**名前だけ**で、ジオメトリの手当ては要らない。

1. **骨名**: 3ds Max Biped（`Bip01 Pelvis` / `Bip01 L Finger0`）→ Mixamo か VRM humanoid へ改名。
   `blender/export_vrm.py:32-40` の照合器が流儀ごとの差を吸収する作りなので、Biped を足す
2. **モーフ名**: `AK_09_EyeBlinkLeft` → `eyeBlinkLeft`、`AK_25_JawOpen` → `jawOpen`。
   `extras.targetNames` の文字列を書き換えるだけで **0バイト**

### 正直な懸念

- **4,025頂点は低い。** VRoid のテンプレートは顔だけで4,106頂点ある。RBF 形状転写が
  動かせる自由度は小さい。ただし `docs/05` §16 で「**同一性はテクスチャが担う（ArcFace +0.286）、
  幾何ではない**」と実測済みで、2048² のテクスチャがそれを補う見込み。
  これは NBA 2K 世代のゲームキャラクターの標準的な予算配分そのもの
- **髪が独立メッシュでない**（opacity マテリアルとして同一メッシュ内）。
  spring bone を入れるにはメッシュ分割が要る
- VRM ではない。KEHAI は `.glb` も読む（`glb-humanoid.ts`）ので必須ではないが、
  VRM 化するなら `blender/export_vrm.py` を通すことになる

### 4,025頂点では足りないと判明した場合の次点

**Reallusion CC Base** — 無料のリグ付き素体5体、顔モーフ150以上。Headshot 2 が
想定している素体そのもので、ポリゴン数も上。ただしライセンスの確認が要る。
