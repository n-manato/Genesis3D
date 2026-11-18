#!/usr/bin/env bash
# 画像1枚 → character.vrm を1コマンドで回す。
#
# MCP サーバー(genesis3d)はセッション再起動後に使えるようになる。それまで、
# および CLI から回したいときはこれを使う。
#
#   ./scripts/gen.sh <画像> [--face <実写ポートレート>]
#                            [--style stylized|realistic_jrpg|wrap]
#                           [--quality fast|high|maximum]
#                           [--render] [--gateway]
#
#   --style wrap  ラップ方式（docs/09）。**ジョブサーバを経由せずローカルで回す。**
#                 位置引数は全身写真、--face が顔写真。glb と VRM の両方が出る。
#                   ./scripts/gen.sh <全身写真> --face <顔写真> --style wrap \
#                       --height-cm 160 [--template rocketbox_female_01]
#   --render   完成した VRM を5視点でレンダリングし、PNG のパスを出す
#   --gateway  Gateway(:3000) 経由で投げる（APIキーが必要: GENESIS_API_KEY）
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env || echo "警告: .env 読み込み失敗" >&2; set +a; fi

IMAGE=""; FACE=""; STYLE="stylized"; QUALITY="high"; RENDER=0; VIA_GATEWAY=0
HEIGHT_CM=""; TEMPLATE="rocketbox_female_01"; OUT_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --face)    FACE="$2"; shift 2 ;;
    --style)   STYLE="$2"; shift 2 ;;
    --quality) QUALITY="$2"; shift 2 ;;
    --render)  RENDER=1; shift ;;
    --gateway) VIA_GATEWAY=1; shift ;;
    --height-cm) HEIGHT_CM="$2"; shift 2 ;;
    --template)  TEMPLATE="$2"; shift 2 ;;
    --out-dir)   OUT_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *)         IMAGE="$1"; shift ;;
  esac
done
[ -n "$IMAGE" ] || { echo "使い方: $0 <画像> [--face <実写>] [--style ...] [--quality ...] [--render]" >&2; exit 2; }
[ -f "$IMAGE" ] || { echo "画像が見つかりません: $IMAGE" >&2; exit 2; }
[ -z "$FACE" ] || [ -f "$FACE" ] || { echo "顔画像が見つかりません: $FACE" >&2; exit 2; }

# ---- ラップ方式はローカル実行。ジョブサーバを経由しない（docs/09）----------
if [ "$STYLE" = "wrap" ] || [ "$STYLE" = "game_avatar" ]; then
  [ -n "$FACE" ] || { echo "--style wrap には --face <顔写真> が要ります" >&2; exit 2; }
  OUT_DIR="${OUT_DIR:-jobs/gw_$(date +%Y%m%d_%H%M%S)}"
  ARGS=(--face "$FACE" --body "$IMAGE" --template "$TEMPLATE" --out-dir "$OUT_DIR")
  [ -n "$HEIGHT_CM" ] && ARGS+=(--height-cm "$HEIGHT_CM")
  echo "ラップ方式: 顔=$FACE 体=$IMAGE テンプレ=$TEMPLATE → $OUT_DIR"
  exec python3 scripts/gen_wrap_avatar.py "${ARGS[@]}"
fi

BASE="http://${GENESIS_HOST:-127.0.0.1}:${GENESIS_PORT:-8820}"
AUTH=()
if [ "$VIA_GATEWAY" -eq 1 ]; then
  [ -n "${GENESIS_API_KEY:-}" ] || { echo "GENESIS_API_KEY が未設定です" >&2; exit 2; }
  BASE="http://${GATEWAY_HOST:-127.0.0.1}:${GATEWAY_PORT:-3000}/v1/3d/characters"
  AUTH=(-H "Authorization: Bearer ${GENESIS_API_KEY}")
  CREATE="$BASE"; STATUS="$BASE"; RESULTS="$BASE"
else
  CREATE="$BASE/jobs"; STATUS="$BASE/jobs"; RESULTS="$BASE/jobs"
fi

echo "入力: $IMAGE  style=$STYLE quality=$QUALITY  経由=$([ $VIA_GATEWAY -eq 1 ] && echo Gateway || echo 直接)"
[ -z "$FACE" ] || echo "顔テクスチャ: $FACE"
# --face は工程6の顔投影に使う実写。生成した全身Tポーズから体を作るときは
# その顔が本人ではないので、ここで実写を分けて渡す。
FACE_ARG=()
[ -z "$FACE" ] || FACE_ARG=(-F "face_image=@$FACE")
RESP=$(curl -sS "${AUTH[@]}" -X POST "$CREATE" \
  -F "image=@$IMAGE" "${FACE_ARG[@]}" \
  -F "style=$STYLE" -F "quality=$QUALITY" -F "output_format=vrm")
JOB=$(printf '%s' "$RESP" | python3 -c "import sys,json;print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null)
[ -n "$JOB" ] || { echo "ジョブ作成失敗: $RESP" >&2; exit 1; }
echo "job_id=$JOB"

START=$(date +%s); LAST=""
while :; do
  S=$(curl -sS "${AUTH[@]}" "$STATUS/$JOB" 2>/dev/null)
  LINE=$(printf '%s' "$S" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f\"{d['status']}|{d['stage']}|{d['progress']}%|gpu={d['gpu']}\")" 2>/dev/null)
  [ "$LINE" != "$LAST" ] && { printf '  [%3ds] %s\n' "$(( $(date +%s)-START ))" "$LINE"; LAST="$LINE"; }
  case "$LINE" in completed*) break ;; failed*|canceled*) echo "$S"; exit 1 ;; esac
  python3 -c "import time;time.sleep(5)"
done

echo
echo "── 工程サマリ ──"
python3 - "$JOB" <<'PY'
import json,sys,pathlib
j=sys.argv[1]; f=pathlib.Path("jobs")/j/"output"/"job.json"
if not f.exists(): print("  job.json なし"); raise SystemExit
d=json.loads(f.read_text())
print(f"  総所要 {d.get('duration_sec')}秒")
for k,v in d["stages"].items():
    st=v.get("status","?")
    extra=""
    if k=="evaluate" and v.get("ranking"): extra="  " + " / ".join(f"{c}={s:.2f}" for c,s in v["ranking"])
    elif k=="rigging" and st=="ok": extra=f"  {v.get('method')} bones={v.get('bones')}"
    elif k=="validate" and st=="ok": extra=f"  必須不足={v.get('missing_required_bones')} 身長={v.get('height_m')}m 足元原点={v.get('origin_at_feet')}"
    elif st in ("skipped","degraded"): extra="  " + str(v.get("reason",""))[:60]
    print(f"  {k:14} {st:9}{extra}")
print(f"  出力: {d.get('output')}")
PY

OUT="jobs/$JOB/output/character.vrm"
if [ -f "$OUT" ]; then
  echo; echo "VRM: $(pwd)/$OUT  ($(stat -c%s "$OUT" | numfmt --to=iec))"
  if [ "$RENDER" -eq 1 ]; then
    echo "── レンダリング ──"
    "${BLENDER_BIN:-$HOME/opt/blender/blender}" --background --factory-startup \
      --python blender/render_views.py -- --input "$OUT" \
      --outdir "jobs/$JOB/renders/final" --res 640 --samples 32 \
      --views front,three_quarter,back 2>&1 | grep -oE 'RENDER_OK.*' | head -1
    for v in front three_quarter back; do
      p="jobs/$JOB/renders/final/$v.png"; [ -f "$p" ] && echo "  $(pwd)/$p"
    done
  fi
else
  echo; echo "警告: character.vrm が生成されていません（リギングに到達しなかった可能性）"
  ls -1 "jobs/$JOB/output" 2>/dev/null | sed 's/^/  /'
fi
