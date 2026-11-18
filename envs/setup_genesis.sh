#!/usr/bin/env bash
# Genesis3D Phase 3 — orchestrator 本体の conda env。
#
# この env には torch を入れない。オーケストレータ自身は FastAPI + HTTP + NVML だけで
# 動き、重い推論はすべて各モデル専用 env の独立サブプロセスか :8811 の TRELLIS.2 が担う。
# 評価器(torch/CLIP/DINOv3)は Phase 4 で setup_eval.sh により追加する。
set -euo pipefail

ENV_NAME="${ENV_GENESIS:-genesis}"
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
echo "── 依存パッケージ ──"
python -m pip install --no-cache-dir -q -r "$(dirname "$0")/../requirements.txt"

echo "── 確認 ──"
python - <<'PY'
import fastapi, uvicorn, httpx, yaml, pydantic, PIL, numpy, trimesh
print("python     ", __import__("sys").version.split()[0])
print("fastapi    ", fastapi.__version__)
print("pydantic   ", pydantic.VERSION)
print("trimesh    ", trimesh.__version__)
try:
    import pynvml; pynvml.nvmlInit()
    n = pynvml.nvmlDeviceGetCount()
    print("NVML       ", f"{n} GPU 検出")
    for i in range(n):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        name = pynvml.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else name
        print(f"  GPU{i}      {name}  空き {m.free/1073741824:.1f} GiB / {m.total/1073741824:.0f} GiB")
    pynvml.nvmlShutdown()
except Exception as e:
    print("NVML        利用不可:", e)
PY

# ディスクが逼迫しているのでパッケージキャッシュを残さない
conda clean -a -y >/dev/null 2>&1 || true
echo
echo "完了: conda activate $ENV_NAME"
df -h / | tail -1
