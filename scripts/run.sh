#!/usr/bin/env bash
# Genesis3D orchestrator を 127.0.0.1 で起動する（外部公開しない・要件§32）。
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${ENV_GENESIS:-genesis}"

# .env を読み込む。値にスペースが含まれる場合に備え、失敗しても set -a を残さない。
if [ -f .env ]; then set -a; . ./.env || echo "警告: .env の読み込みに失敗" >&2; set +a; fi

HOST="${GENESIS_HOST:-127.0.0.1}"
PORT="${GENESIS_PORT:-8820}"

# 127.0.0.1 以外へのバインドを拒否する。設定ミスで研究室LANへ露出させない。
if [ "$HOST" != "127.0.0.1" ] && [ "$HOST" != "localhost" ]; then
  echo "拒否: GENESIS_HOST=$HOST。Genesis3D は localhost のみで待ち受けます。" >&2
  exit 1
fi

mkdir -p logs
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "Genesis3D orchestrator → http://$HOST:$PORT"
exec python -m uvicorn orchestrator.app:app --host "$HOST" --port "$PORT" "$@"
