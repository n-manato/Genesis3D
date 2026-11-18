#!/usr/bin/env bash
# Genesis3D — 髪の生成（DiffLocks, CVPR 2025）の env。
#
# 目的: **1枚の写真から本人の髪を毛束として起こす。**
#
# いまの髪は TRELLIS が体と一緒に作る塊で、蝋が溶けたような見た目になる。
# 3/4 と横から最も目立つ（docs/05 §36）。DiffLocks は拡散モデルで頭皮テクスチャに
# 毛束の潜在コードを描き、そこから 3D の毛束を直接デコードする。汎用の 3D 生成器が
# 「塊」しか作れないのとは原理的に別物。
#
# torch を 2.5.0 に固定するのは DiffLocks の要求。他の env（2.6.0）と噛み合わないので
# 独立させる。
#
# ライセンス: **非商用の研究用のみ**（Meshcapade GmbH）。VRM のメタも
# `commercialUsage: personalNonProfit` にしてあるので整合する。
#
# 重みは既に取得・展開済み: vendor/difflocks_ckpt/checkpoints/
#   difflocks_diffusion/scalp_v9_40k_06730000.pth   9.7GB
#   strand_vae/strand_codec.pt                      7.1MB
#   rgb2material/rgb2material.pt                    531KB
set -euo pipefail

ENV_NAME="${ENV_HAIR:-genesis-hair}"
PY_VER=3.10
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' は作成済み"
else
  echo "── conda env '$ENV_NAME' (python $PY_VER) を作成 ──"
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"

echo "── torch 2.5.0 (DiffLocks の固定要求) ──"
python -m pip install --no-cache-dir -q \
  "torch==2.5.0" "torchvision==0.20.0" --index-url https://download.pytorch.org/whl/cu124

echo "── DiffLocks の依存 ──"
python -m pip install --no-cache-dir -q \
  "accelerate==0.31.0" "opencv-python==4.9.0.80" "mediapipe==0.10.10" \
  "trimesh==4.0.4" "jsonmerge==1.9.2" "dctorch==0.1.2" "einops==0.8.1" \
  "torchdiffeq==0.2.5" "torchsde==0.2.6" "libigl==2.5.1" "Pillow" "numpy<2"

echo "── 確認 ──"
python - <<'PY'
import torch
print("torch      ", torch.__version__, "| cuda", torch.cuda.is_available())
for m in ("cv2", "mediapipe", "trimesh", "igl", "dctorch", "torchdiffeq"):
    try:
        __import__(m); print(f"{m:12s} OK")
    except Exception as e:
        print(f"{m:12s} NG: {e}")
PY

conda clean -a -y >/dev/null 2>&1 || true
echo "完了: conda activate $ENV_NAME"
df -h / | tail -1
