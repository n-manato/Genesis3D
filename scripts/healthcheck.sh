#!/usr/bin/env bash
# /health /gpus /models を叩いて状態を表示する。
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
BASE="http://${GENESIS_HOST:-127.0.0.1}:${GENESIS_PORT:-8820}"

j() { python3 -c "import sys,json;print(json.dumps(json.load(sys.stdin),indent=2,ensure_ascii=False))" 2>/dev/null || cat; }

for ep in health gpus models; do
  echo "═══ GET $BASE/$ep ═══"
  curl -sS -m 10 "$BASE/$ep" | j
  echo
done
