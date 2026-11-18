#!/usr/bin/env bash
# Genesis3D Phase 4 — 前処理(RMBG-2.0) と 評価器(CLIP/DINOv3/ArcFace) の env。
#
# orchestrator 本体(genesis env)とは分ける。torch を API プロセスに同居させると
# CUDA クラッシュが API ごと落とすため。この env のコードは常に独立サブプロセスで
# 実行され、終了時に VRAM が完全解放される（要件§30/§31）。
#
# 重みはすべて既に ~/.cache/huggingface にある:
#   briaai/RMBG-2.0                              背景除去（rembg より帽子・小物に強い）
#   openai/clip-vit-large-patch14                CLIP スコア
#   facebook/dinov3-vitl16-pretrain-lvd1689m     全身特徴の類似度
set -euo pipefail

ENV_NAME="${ENV_EVAL:-genesis-eval}"
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

echo "── torch (cu124: 既存 trellis2 env と同じ CUDA 系列) ──"
python -m pip install --no-cache-dir -q \
  torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "── 前処理・評価の依存 ──"
python -m pip install --no-cache-dir -q \
  "transformers>=4.44" "timm>=1.0" "Pillow>=10.4" "numpy<2.2" \
  "scikit-image>=0.24" "opencv-python-headless>=4.10" "einops" \
  kornia   # RMBG-2.0 の remote code が要求する

# ArcFace（本人性の測定）。GPU 版 onnxruntime は CUDA ライブラリを丸ごと引き込んで
# 数GB になるうえ、この env は DINOv3 用に torch+cu124 を既に持っている。顔は
# 候補あたり数枚しか測らないので CPU 実行で十分（docs/05-measurements.md §13）。
python -m pip install --no-cache-dir -q insightface onnxruntime

# antelopev2 の重み(408MB)を取得する。zip が antelopev2/antelopev2/ と二重に
# 展開されるため FaceAnalysis が 'detection' を見つけられない。平坦化して直す。
echo "── ArcFace(antelopev2) の重み ──"
python - <<'PYEOF'
import os, shutil
from pathlib import Path
root = Path.home() / ".insightface" / "models" / "antelopev2"
if not list(root.glob("*.onnx")):
    from insightface.app import FaceAnalysis
    try:
        FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
    except AssertionError:
        pass                      # 二重展開のため初回は必ずここに来る
    nested = root / "antelopev2"
    if nested.is_dir():
        for f in nested.glob("*.onnx"):
            shutil.move(str(f), root / f.name)
        nested.rmdir()
    zip_ = root.parent / "antelopev2.zip"
    if zip_.exists():
        zip_.unlink()
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("ArcFace OK")
PYEOF

echo "── 確認 ──"
python - <<'PY'
import torch, transformers, timm, insightface
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("cuda 利用可 ", torch.cuda.is_available(), "| GPU数", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("timm        ", timm.__version__)
print("insightface ", insightface.__version__)
PY

conda clean -a -y >/dev/null 2>&1 || true
echo "完了: conda activate $ENV_NAME"
df -h / | tail -1
