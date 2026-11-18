#!/usr/bin/env bash
# Genesis3D MCP サーバーのランチャー（stdio）。
#
# ~/nexus-mcp.sh と同じ「パスを内側に閉じ込めるラッパ」方式。MCP クライアントには
# このスクリプトだけを渡せばよく、conda 環境やパスの詳細を設定に書かずに済む。
#
# stdio で JSON-RPC を喋るので、**このスクリプトは標準出力に何も書いてはいけない**。
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${ENV_GENESIS:-genesis}"

# .env を読み込む（BLENDER_BIN / TRELLIS_HOST / RIG_TEMPLATE_VRM など）。
# 診断出力は stderr へ逃がす。
if [ -f .env ]; then set -a; . ./.env; set +a; fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# .env の GENESIS_HOST はバインドアドレス。サーバーが使う完全なURLは別名で渡す。
export GENESIS_URL="http://${GENESIS_HOST:-127.0.0.1}:${GENESIS_PORT:-8820}"

exec python -m genesis_mcp.server
