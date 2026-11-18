#!/usr/bin/env bash
# Genesis3D Phase 2 — Blender 4.2 LTS + VRM Add-on, installed to ~/opt (no sudo).
#
# Blender 4.2 is the LTS the VRM Add-on has the deepest track record with, and it
# is the first release with the Extensions system — so we install the "Extension"
# zip, not the legacy add-on zip.
#
#   ./setup_blender.sh            # install
#   ./setup_blender.sh --verify   # only re-run the verification step
set -euo pipefail

BLENDER_VER="${BLENDER_VER:-4.2.23}"
BLENDER_SERIES="4.2"
VRM_VER="${VRM_VER:-4.5.0}"
VRM_TAG="v${VRM_VER}"
VRM_ZIP_NAME="VRM_Addon_for_Blender-Extension-${VRM_VER//./_}.zip"

OPT="$HOME/opt"
BL_DIR="$OPT/blender-${BLENDER_VER}-linux-x64"
BL_LINK="$OPT/blender"
BL_BIN="$BL_LINK/blender"
CACHE="$OPT/.downloads"

verify() {
  echo "── 検証 ──"
  local script; script="$(mktemp /tmp/genesis_vrm_check_XXXX.py)"
  cat > "$script" <<'PY'
import sys, bpy, addon_utils

print("blender:", bpy.app.version_string)

# The extension lands under bl_ext.*; find whatever module provides VRM.
mods = [m.__name__ for m in addon_utils.modules() if "vrm" in m.__name__.lower()]
print("vrm modules:", mods)
enabled = [m for m in mods if addon_utils.check(m)[1]]
print("enabled:", enabled)
if not enabled:
    for m in mods:
        try:
            addon_utils.enable(m, default_set=True, persistent=True)
            enabled.append(m)
        except Exception as e:
            print("enable failed", m, e)
if not enabled:
    print("RESULT: FAIL - VRM add-on not available")
    sys.exit(1)

has_import = hasattr(bpy.ops.import_scene, "vrm")
has_export = hasattr(bpy.ops.export_scene, "vrm")
print("import_scene.vrm:", has_import, "| export_scene.vrm:", has_export)

# Round-trip a real VRM if one is reachable.
sample = "/home/ai-ccore/Documents/DEV/KEHAI/Live2D Characters/vrm/sample.vrm"
import os
if has_import and os.path.exists(sample):
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.import_scene.vrm(filepath=sample)
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    print(f"imported sample.vrm: armatures={len(arms)} meshes={len(meshes)}")
    if arms:
        hum = getattr(arms[0].data, "vrm_addon_extension", None)
        if hum:
            print("vrm_addon_extension present on armature: True")
    if not arms:
        print("RESULT: FAIL - no armature in imported VRM")
        sys.exit(1)
else:
    print("sample.vrm not found - skipped round-trip")

print("RESULT: OK")
PY
  "$BL_BIN" --background --factory-startup --python "$script" 2>&1 |
    grep -E "^(blender:|vrm modules:|enabled:|import_scene|imported|vrm_addon|RESULT:)" || true
  rm -f "$script"
}

if [ "${1:-}" = "--verify" ]; then
  [ -x "$BL_BIN" ] || { echo "Blender が見つかりません: $BL_BIN" >&2; exit 1; }
  verify; exit $?
fi

mkdir -p "$OPT" "$CACHE"

# ---- 1. Blender ----
if [ -x "$BL_DIR/blender" ]; then
  echo "Blender ${BLENDER_VER} は導入済み: $BL_DIR"
else
  TARBALL="blender-${BLENDER_VER}-linux-x64.tar.xz"
  URL="https://download.blender.org/release/Blender${BLENDER_SERIES}/${TARBALL}"
  echo "── Blender ${BLENDER_VER} を取得 ──"
  echo "  $URL"
  curl -fL --retry 3 --progress-bar -o "$CACHE/$TARBALL" "$URL"
  echo "── 展開 ──"
  tar -xJf "$CACHE/$TARBALL" -C "$OPT"
  rm -f "$CACHE/$TARBALL"          # ディスクが逼迫しているので即削除
fi
ln -sfn "$BL_DIR" "$BL_LINK"
echo "  $("$BL_BIN" --version 2>/dev/null | head -1)"

# ---- 2. VRM Add-on (Extension 形式) ----
echo "── VRM Add-on ${VRM_VER} を取得 ──"
VRM_URL="https://github.com/saturday06/VRM-Addon-for-Blender/releases/download/${VRM_TAG}/${VRM_ZIP_NAME}"
curl -fL --retry 3 --progress-bar -o "$CACHE/$VRM_ZIP_NAME" "$VRM_URL"

echo "── インストール ──"
# Blender 4.2+ の Extensions CLI。失敗したら Python API にフォールバック。
if "$BL_BIN" --command extension install-file \
      --repo user_default --enable "$CACHE/$VRM_ZIP_NAME" 2>/dev/null; then
  echo "  extension install-file: OK"
else
  echo "  extension CLI が使えないため Python API で導入"
  "$BL_BIN" --background --python-expr "
import bpy
bpy.ops.extensions.package_install_files(
    filepath='$CACHE/$VRM_ZIP_NAME', repo='user_default', enable_on_install=True)
bpy.ops.wm.save_userpref()
" >/dev/null 2>&1 || {
    echo "  Extension 形式が失敗。レガシー add-on 形式で再試行"
    LEG="VRM_Addon_for_Blender-${VRM_VER//./_}.zip"
    curl -fL --retry 3 -sS -o "$CACHE/$LEG" \
      "https://github.com/saturday06/VRM-Addon-for-Blender/releases/download/${VRM_TAG}/${LEG}"
    "$BL_BIN" --background --python-expr "
import bpy
bpy.ops.preferences.addon_install(filepath='$CACHE/$LEG', overwrite=True)
bpy.ops.preferences.addon_enable(module='io_scene_vrm')
bpy.ops.wm.save_userpref()
" >/dev/null 2>&1
    rm -f "$CACHE/$LEG"
  }
fi
rm -f "$CACHE/$VRM_ZIP_NAME"

verify

echo
echo "Blender: $BL_BIN"
echo ".env に BLENDER_BIN=$BL_BIN を設定してください"
df -h / | tail -1
