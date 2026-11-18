#!/usr/bin/env bash
# Genesis3D Phase 0 — reclaim disk space.
#
#   ./cleanup_disk.sh              # dry run (default): report only, delete nothing
#   ./cleanup_disk.sh --apply      # delete the user-owned targets (no sudo needed)
#   ./cleanup_disk.sh --sudo       # ALSO delete the system targets (prompts for password)
#
# Sizes are HARDLINK-AWARE. uv and conda hardlink their caches into live venvs/envs,
# so `du` massively overstates what deleting them actually frees. Only files with a
# link count of 1 are counted as freeable — everything else stays on disk anyway.
set -uo pipefail

APPLY=0; DOSUDO=0; DOOTHER=0
for a in "$@"; do
  case "$a" in
    --apply)       APPLY=1 ;;
    --sudo)        APPLY=1; DOSUDO=1 ;;
    --other-users) APPLY=1; DOSUDO=1; DOOTHER=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

# 休眠アカウントの conda インストール。研究データ・実験結果・独自収集データは対象外。
# 判定根拠（実測）: 2026-06-01以降に更新されたファイル 0 件、参照する実行中プロセス 0 件。
DORMANT_CONDA=(
  "/home/rodiseri/anaconda3"
  "/home/dilanga/anaconda3"
)

VS="$HOME/.vscode-server"
gib() { awk -v b="$1" 'BEGIN{printf "%.2f GiB", b/1073741824}'; }

# Bytes actually freed by removing a path: files with nlink==1 only.
freeable() {
  [ -e "$1" ] || { echo 0; return; }
  find "$1" -type f -links 1 -printf '%s\n' 2>/dev/null | awk '{s+=$1}END{printf "%.0f", s+0}'
}
# Bytes that will NOT be freed because another link exists.
shared() {
  [ -e "$1" ] || { echo 0; return; }
  find "$1" -type f -links +1 -printf '%s\n' 2>/dev/null | awk '{s+=$1}END{printf "%.0f", s+0}'
}

# Commits with a LIVE PROCESS behind them. Deliberately NOT derived from
# ~/.vscode-server/*.token — those linger after every update and would pin ~4GB.
live_commits() {
  for p in $(pgrep -f vscode-server 2>/dev/null); do
    readlink -f "/proc/$p/exe" 2>/dev/null
    readlink -f "/proc/$p/cwd" 2>/dev/null
  done | grep -oE '[0-9a-f]{40}' | sort -u
}
KEEP_COMMITS="$(live_commits)"
stale_vscode() {
  for d in "$VS"/bin/*/ "$VS"/cli/servers/*/; do
    [ -d "$d" ] || continue
    c="$(basename "$d" | grep -oE '[0-9a-f]{40}')" || continue
    [ -n "$c" ] || continue
    grep -qx "$c" <<< "$KEEP_COMMITS" && continue
    echo "${d%/}"
  done
}

USER_PATHS=()
add_user() { USER_PATHS+=("$1|$2|$3"); }
add_user "$HOME/.cache/uv"        "uv cache"           "uvが次回インストール時に再取得"
add_user "$HOME/miniconda3/pkgs"  "conda package cache" "condaが次回インストール時に再取得"
add_user "$HOME/.cache/vllm"      "vLLM compile cache"  "vLLMが次回起動時に再コンパイル"
add_user "$HOME/.npm/_cacache"    "npm cache"           "npmが次回インストール時に再取得"
mapfile -t STALE < <(stale_vscode)
for d in "${STALE[@]}"; do add_user "$d" "停止済VS Codeビルド" "そのVS Code版を再度開いた時のみ再取得"; done

echo "════════════════════════════════════════════════════════════════"
echo " 1. ユーザー所有・sudo不要・再生成可能"
echo "════════════════════════════════════════════════════════════════"
utotal=0
for e in "${USER_PATHS[@]}"; do
  IFS='|' read -r p name regen <<< "$e"
  [ -e "$p" ] || continue
  f=$(freeable "$p"); s=$(shared "$p"); utotal=$((utotal+f))
  printf "  %-24s 解放 %-12s %s\n" "$name" "$(gib "$f")" "$p"
  [ "$s" -gt 0 ] && printf "  %-24s %-17s ハードリンク共有のため解放されない\n" "" "(+$(gib "$s"))"
  printf "  %-24s %-17s %s\n" "" "" "$regen"
done
echo "  ──────────────────────────────────────────────────────────────"
printf "  小計: %s\n" "$(gib "$utotal")"

echo
echo "════════════════════════════════════════════════════════════════"
echo " 2. システム領域・sudo必要"
echo "════════════════════════════════════════════════════════════════"
printf "  %-34s %s\n" "孤立した旧ollama置き場" "$(du -sh /usr/share/ollama 2>/dev/null | cut -f1)"
echo "    /usr/share/ollama — 稼働中のollamaは OLLAMA_MODELS=~/.ollama/models を"
echo "    使用（systemd override 済、User=ai-ccore）。この置き場の中身は2025年6-7月の"
echo "    deepseek-r1:70b と devstral:latest のみで、現行 ollama list に存在しない。"
echo "    稼働プロセスからの参照数 0。"
printf "  %-34s %s\n" "CUDA/cuDNN インストーラ残骸" "$(du -shc /var/cuda-repo-ubuntu2004-11-8-local /var/cudnn-local-repo-ubuntu2004-9.5.1 2>/dev/null | tail -1 | cut -f1)"
echo "    導入後は不要な .deb リポジトリ。CUDA本体は削除しない。"
printf "  %-34s %s\n" "snap 無効revision" "$(snap list --all 2>/dev/null | awk '$NF ~ /disabled/' | wc -l) 個"
printf "  %-34s %s\n" "journald ログ" "$(journalctl --disk-usage 2>/dev/null | grep -oE '[0-9.]+[MG]' | head -1)"
printf "  %-34s %s\n" "apt キャッシュ" "$(du -sh /var/cache/apt 2>/dev/null | cut -f1)"

echo
echo "════════════════════════════════════════════════════════════════"
echo " 3. 休眠アカウントの conda インストール（--other-users で削除）"
echo "════════════════════════════════════════════════════════════════"
for d in "${DORMANT_CONDA[@]}"; do
  [ -e "$d" ] || continue
  u=$(echo "$d" | cut -d/ -f3)
  recent=$(find "/home/$u" -type f -newermt "2026-06-01" 2>/dev/null | wc -l)
  procs=0
  for pp in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    t=$(readlink -f /proc/$pp/exe 2>/dev/null)
    case "$t" in "/home/$u/"*) procs=$((procs+1));; esac
  done
  printf "  %-28s %s\n" "$(basename "$d") ($u)" "$(du -sh "$d" 2>/dev/null | cut -f1)"
  printf "  %-28s 2026-06-01以降の更新 %s 件 / 参照プロセス %s 件\n" "" "$recent" "$procs"
done
echo "    conda env とパッケージキャッシュのみ。研究データ・実験結果には触れない。"

echo
echo "════════════════════════════════════════════════════════════════"
echo " 4. 削除しないもの（明示）"
echo "════════════════════════════════════════════════════════════════"
cat <<'KEEP'
  rodiseri/ViLP/data       106GB  UCF101（公開DSの派生フレーム62G+pose38G）— 研究データ
  rodiseri/ViLP/output      4.9GB  実験結果 — 再現に数日の計算が必要
  rodiseri/ViLP/data/unmcvideos 1.9GB  独自収集の可能性がある動画
  samira/ 全体               21GB  eye / Dr_Sani_project / Singh — 研究プロジェクト
  vishvaprasanth... 全体    7.5GB  本日(2026-08-20)も稼働中のアカウント
  /usr/local/cuda-11.8      5.4GB  gesturelsm が torch 2.1.2+cu118 で現用中
  /usr/local/MATLAB          19GB  他ユーザーの資産
  ~/.ollama/models          181GB  Gateway が配信中のLLM重み（6 blob）
  ~/.cache/huggingface       27GB  TRELLIS.2 / DINOv3 / RMBG-2.0 / CLIP
  motion-service/assets      26GB  HY-Motion-1.0（/v1/motion で稼働中）
  world-perception/runners   17GB  moge / grounding_dino / sam2（稼働中）
KEEP

echo
echo "現在の空き: $(df -h --output=avail / | tail -1 | tr -d ' ')"

if [ "$APPLY" -eq 0 ]; then
  echo
  echo "DRY RUN — 何も削除していません。"
  echo "  ./cleanup_disk.sh --apply        ユーザー領域のみ（約$(gib "$utotal")・sudo不要）"
  echo "  ./cleanup_disk.sh --sudo         + システム領域（約+61GB）"
  echo "  ./cleanup_disk.sh --other-users  + 休眠アカウントのconda（約+81GB）→ 合計 約171GB"
  exit 0
fi

echo
echo "── ユーザー領域を削除 ──"
for e in "${USER_PATHS[@]}"; do
  IFS='|' read -r p _ _ <<< "$e"
  [ -e "$p" ] || continue
  echo "  rm -rf $p"
  rm -rf "${p:?}"
done

if [ "$DOSUDO" -eq 1 ]; then
  echo
  echo "── システム領域を削除（sudoパスワードを求められます）──"
  sudo rm -rf /usr/share/ollama
  sudo rm -rf /var/cuda-repo-ubuntu2004-11-8-local /var/cudnn-local-repo-ubuntu2004-9.5.1
  sudo journalctl --vacuum-size=200M
  sudo apt-get clean
  # 無効化された古いsnap revisionを削除
  snap list --all 2>/dev/null | awk '$NF ~ /disabled/ {print $1, $3}' |
    while read -r name rev; do sudo snap remove "$name" --revision="$rev"; done
fi

if [ "$DOOTHER" -eq 1 ]; then
  echo
  echo "── 休眠アカウントの conda インストールを削除 ──"
  for d in "${DORMANT_CONDA[@]}"; do
    [ -e "$d" ] || continue
    echo "  sudo rm -rf $d"
    sudo rm -rf "${d:?}"
  done
fi

echo
echo "削除後の空き: $(df -h --output=avail / | tail -1 | tr -d ' ')"
