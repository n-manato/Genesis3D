#!/usr/bin/env bash
# Genesis3D — 顔トラック用の env（MICA + FLAME）。
#
# 目的: **1枚の写真から、無表情の頭部メッシュを FLAME 位相で得る**。
#
# いまの頭は TRELLIS が作る「彫像」で、目・口・顎という部品の概念がない。
# 写真の表情がそのまま固まっているので、まばたきも視線も口パクも原理的に不可能
# （docs/05-measurements.md §36〜38）。FLAME は眼球・顎・口内・表情基底を
# 最初から持つパラメトリック頭部モデルなので、ここを差し替えると:
#
#   - 無表情の頭が手に入る（焼き付いた笑顔と細めた目が消える）
#   - 眼球が独立ジオメトリになり、視線追従が効く
#   - 顎の関節と表情基底から名前付きモーフ（jawOpen / aa / blink …）を作れる
#     → KEHAI は生のモーフ名を直接見るので、名前を合わせるだけで口パクが動く
#
# MICA を使うのは、これが「1枚 → **無表情の**同一人物 FLAME 形状」を目的に
# 作られた手法だから。Pixel3DMM の方が幾何は細かいが、pytorch3d と nvdiffrast の
# ソースビルドが要り、動画トラッキング前提で今回は過剰。
#
# ── 手作業が要る箇所（ライセンス同意のため代行できない）──
# FLAME2020 は Max Planck の配布サイトでアカウントを作り、ライセンスに同意した
# 人だけがダウンロードできる。標準ライセンスは**非商用の研究・教育用**
# （大学は該当）、FLAME 2023 Open 版は CC-BY-4.0。
#
#   1. https://flame.is.tue.mpg.de/ で登録してライセンスに同意する
#   2. FLAME2020.zip をダウンロードする
#   3. このスクリプトに --flame-zip /path/to/FLAME2020.zip を渡す
#
set -euo pipefail

ENV_NAME="${ENV_FACE:-genesis-face}"
PY_VER=3.10
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor"
FLAME_ZIP=""

while [ $# -gt 0 ]; do
  case "$1" in
    --flame-zip) FLAME_ZIP="$2"; shift 2 ;;
    *) echo "不明な引数: $1"; exit 2 ;;
  esac
done

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' は作成済み"
else
  echo "── conda env '$ENV_NAME' (python $PY_VER) を作成 ──"
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"

echo "── torch (cu124: 他の env と同じ CUDA 系列) ──"
python -m pip install --no-cache-dir -q \
  torch torchvision --index-url https://download.pytorch.org/whl/cu124

# chumpy は FLAME の .pkl を unpickle するために必要だが、numpy 2.x でも
# 1.24+ でも壊れる（削除された np.bool / np.object を参照する）。
# ここだけ numpy を 1.23 系に固定し、pip の依存解決を通すためにパッチを当てる。
echo "── 依存（numpy は chumpy のため 1.23 系に固定）──"
python -m pip install --no-cache-dir -q \
  "numpy==1.23.5" "scipy" "scikit-image" "opencv-python-headless" \
  "trimesh" "loguru" "yacs" "easydict" "insightface==0.7.3" \
  "onnxruntime-gpu" "pyyaml" "tqdm" "Pillow"
python -m pip install --no-cache-dir -q --no-build-isolation "chumpy==0.70" || {
  echo "  chumpy の通常インストールに失敗 — パッチ版を入れる"
  python -m pip install --no-cache-dir -q --no-deps "chumpy==0.70"
}
# chumpy は import 時に np.bool 等を触るので、無ければ補う
python - <<'PY'
import pathlib, chumpy, re
p = pathlib.Path(chumpy.__file__).parent / "__init__.py"
s = p.read_text()
if "np.bool," in s:
    s = s.replace("from numpy import bool, int, float, complex, object, unicode, str, nan, inf",
                  "from numpy import nan, inf\nbool = __import__('builtins').bool\n"
                  "int = __import__('builtins').int\nfloat = __import__('builtins').float\n"
                  "complex = __import__('builtins').complex\nobject = __import__('builtins').object\n"
                  "unicode = str\nstr = __import__('builtins').str")
    p.write_text(s)
    print("  chumpy の numpy 互換パッチを適用")
PY

echo "── MICA を取得 ──"
mkdir -p "$VENDOR"
if [ -d "$VENDOR/MICA/.git" ]; then
  echo "  vendor/MICA は取得済み"
else
  git clone --depth 1 -q https://github.com/Zielon/MICA.git "$VENDOR/MICA"
fi

echo "── MICA の学習済み重み ──"
mkdir -p "$VENDOR/MICA/data/pretrained"
if [ -s "$VENDOR/MICA/data/pretrained/mica.tar" ]; then
  echo "  取得済み"
else
  python -m pip install --no-cache-dir -q gdown
  # gdown 5.x で --id は廃止された。URL 形式で渡す。
  gdown "https://drive.google.com/uc?id=1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt" \
        -O "$VENDOR/MICA/data/pretrained/mica.tar" || {
    echo "  !! MICA の重みを取得できませんでした（Google Drive の制限の可能性）"
    echo "     手動: https://drive.google.com/uc?id=1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt"
    echo "     → $VENDOR/MICA/data/pretrained/mica.tar"
  }
fi

# insightface の顔検出器は評価器 env と共用（~/.insightface に既にある）
if [ -d "$HOME/.insightface/models/antelopev2" ]; then
  echo "── insightface antelopev2: 既存を共用 ──"
else
  echo "  !! ~/.insightface/models/antelopev2 がありません。envs/setup_eval.sh を先に実行してください"
fi

echo "── FLAME2020 ──"
FLAME_DIR="$VENDOR/MICA/data/FLAME2020"
if [ -f "$FLAME_DIR/generic_model.pkl" ]; then
  echo "  配置済み: $FLAME_DIR/generic_model.pkl"
elif [ -n "$FLAME_ZIP" ] && [ -f "$FLAME_ZIP" ]; then
  mkdir -p "$FLAME_DIR"
  unzip -o -q "$FLAME_ZIP" -d "$FLAME_DIR"
  echo "  展開しました: $FLAME_DIR"
else
  cat <<'MSG'
  !! FLAME2020 がまだありません。ここだけは手作業が要ります。

     ライセンスへの同意が必要で、これは組織として行う判断なので代行しません。

       1. https://flame.is.tue.mpg.de/ で登録し、ライセンスに同意する
          （標準ライセンス = 非商用の研究・教育・芸術用途。大学は該当する）
       2. FLAME2020.zip をダウンロードする
       3. 再実行:  ./envs/setup_face.sh --flame-zip /path/to/FLAME2020.zip

     FLAME_masks.pkl も同梱されていることを確認してください（マスク処理に使う）。
MSG
fi

echo "── 確認 ──"
python - <<'PY'
import torch, numpy, trimesh
print("torch      ", torch.__version__, "| cuda", torch.cuda.is_available())
print("numpy      ", numpy.__version__)
print("trimesh    ", trimesh.__version__)
try:
    import chumpy; print("chumpy      OK")
except Exception as e:
    print("chumpy      NG:", e)
try:
    import insightface; print("insightface", insightface.__version__)
except Exception as e:
    print("insightface NG:", e)
PY

conda clean -a -y >/dev/null 2>&1 || true
echo "完了: conda activate $ENV_NAME"
df -h / | tail -1
