#!/usr/bin/env bash
# Genesis3D — 自動リギング（UniRig, SIGGRAPH 2025）の env。
#
# いまのリギングは VRoid のテンプレートを被せてウェイトを転送している。
# テンプレートのウェイトは「衣装が独立メッシュで専用の骨で動く」前提なので、
# ジャケットが体に溶接されたこのモデルでは腕を回すと袖が裂ける。骨の選び方を
# 6通り試したが被害の場所が移るだけだった（docs/05 §38）。
#
# UniRig は **メッシュ自体から骨格とスキニングウェイトを推定する**ので、
# 借り物に起因する不整合が原理的に消える。MIT・登録不要・VRAM 8GB。
#
# flash_attn は skin モデルが `flash_attn.modules.mha.MHA` を直接使うため必須。
# ソースビルドは RAM を大量に食う（このマシンは 62GB 中 45GB を他サービスが使用中）
# ので、**公式のビルド済み wheel を使う**。
set -euo pipefail

ENV_NAME="${ENV_UNIRIG:-genesis-unirig}"
PY_VER=3.10
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' は作成済み"
else
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"

echo "── torch (cu124) ──"
python -m pip install --no-cache-dir -q torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "── flash_attn（ビルド済み wheel）──"
python -m pip install --no-cache-dir -q "$FA_URL" || {
  echo "  !! wheel が入りません。torch の版と cxx11abi を確かめてください"; exit 1; }

echo "── UniRig の依存 ──"
python -m pip install --no-cache-dir -q \
  "transformers==4.51.3" python-box einops omegaconf pytorch_lightning lightning \
  addict timm fast-simplification trimesh open3d huggingface_hub \
  "numpy<2" pyyaml tqdm

echo "── 確認 ──"
python - <<'PY'
import torch
print("torch      ", torch.__version__, "| cuda", torch.cuda.is_available())
for m in ("flash_attn", "transformers", "lightning", "trimesh", "open3d"):
    try:
        __import__(m); print(f"{m:14s} OK")
    except Exception as e:
        print(f"{m:14s} NG: {str(e)[:60]}")
PY
conda clean -a -y >/dev/null 2>&1 || true
echo "完了: conda activate $ENV_NAME"
df -h / | tail -1
