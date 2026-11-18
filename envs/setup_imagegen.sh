#!/usr/bin/env bash
# Genesis3D Phase 1 — 参照ビュー生成（工程2）の env。
#
# 目的は「実在の人物の全身Tポーズ画像を作る」こと。元の写真がバストアップだと
# 脚も足も存在せず、リギングが破綻する（docs/05-measurements.md §4）。
#
# 顔の本人性はここでは追わない。実写の顔テクスチャ投影（工程6）が ArcFace を
# 0.38 → 0.68 まで持ち上げるのに対し、生成画像側の顔は最終的に上書きされる
# （同 §16）。ここで欲しいのは **正しい人体比率・Tポーズ・脚と足の存在** と衣装。
#
# orchestrator 本体(genesis env)とは分ける。torch を API プロセスに同居させると
# CUDA クラッシュが API ごと落とすため（要件§30/§31）。
set -euo pipefail

ENV_NAME="${ENV_IMAGEGEN:-genesis-imagegen}"
PY_VER=3.11
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

echo "── torch (cu124: 既存 env と同じ CUDA 系列) ──"
python -m pip install --no-cache-dir -q \
  torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "── diffusers 一式 ──"
python -m pip install --no-cache-dir -q \
  "diffusers>=0.31" "transformers>=4.44" "accelerate>=0.34" "safetensors" \
  "peft" "Pillow>=10.4" "numpy<2.2" "opencv-python-headless>=4.10" \
  "huggingface_hub>=0.25"

echo "── 重みの取得 ──"
HF_HUB_OFFLINE=0 python - <<'PYEOF'
from huggingface_hub import snapshot_download

# フォトリアル SDXL。素の SDXL より実在人物の肌・髪・布に強い。
# fp16 の diffusers レイアウトだけ取る（fp32 と単一ファイル版は落とさない）。
print("RealVisXL V5.0 ...")
snapshot_download(
    "SG161222/RealVisXL_V5.0",
    allow_patterns=["*.json", "*.txt", "**/*.fp16.safetensors"],
)

# Tポーズ骨格の強制。xinsir 版は thibaud 版より姿勢追従が強い。
print("ControlNet OpenPose SDXL ...")
snapshot_download(
    "xinsir/controlnet-openpose-sdxl-1.0",
    allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"],
)
print("done")
PYEOF

echo "── 確認 ──"
python - <<'PYEOF'
import torch, diffusers, transformers, cv2
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("cuda 利用可 ", torch.cuda.is_available(), "| GPU数", torch.cuda.device_count())
print("diffusers   ", diffusers.__version__)
print("transformers", transformers.__version__)
print("opencv      ", cv2.__version__)
PYEOF

conda clean -a -y >/dev/null 2>&1 || true
echo "完了: conda activate $ENV_NAME"
df -h / | tail -1
