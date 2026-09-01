#!/usr/bin/env bash
# One-shot environment check + install for using ZED cameras (pyzed) from robots_realtime.
#
#   ./scripts/setup_zed.sh            # check SDK, init submodules, create .venv with sensors+realsense extras
#   ./scripts/setup_zed.sh --no-sync  # checks only
#
# pyzed is a thin Cython wrapper: it dlopen()s /usr/local/zed/lib/libsl_zed.so at import time, so the
# SDK directory must exist, be readable by the current user, and match the wheel's major.minor.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
ZED_DIR=/usr/local/zed
DO_SYNC=1
[[ "${1:-}" == "--no-sync" ]] && DO_SYNC=0

echo "== ZED SDK =="
if [[ ! -d "$ZED_DIR" ]]; then
  echo "ERROR: ZED SDK not found at $ZED_DIR."
  echo "       Install ZED SDK 5.x for CUDA 12 / Ubuntu $(lsb_release -rs 2>/dev/null || echo '?') from"
  echo "       https://www.stereolabs.com/developers/release  then re-run this script."
  exit 1
fi
if [[ ! -r "$ZED_DIR/lib" || ! -x "$ZED_DIR" ]]; then
  echo "ERROR: $ZED_DIR is not readable by $(id -un) ($(stat -c '%A %U:%G' "$ZED_DIR"))."
  echo "       pyzed will fail with 'libsl_zed.so: cannot open shared object file'. Fix (needs sudo):"
  echo "         sudo chmod -R o+rX $ZED_DIR"
  exit 1
fi

HDR="$ZED_DIR/include/sl/Camera.hpp"
SDK_VER=""
if [[ -r "$HDR" ]]; then
  MAJ=$(grep -E '#define ZED_SDK_MAJOR_VERSION' "$HDR" | awk '{print $3}')
  MIN=$(grep -E '#define ZED_SDK_MINOR_VERSION' "$HDR" | awk '{print $3}')
  PAT=$(grep -E '#define ZED_SDK_PATCH_VERSION' "$HDR" | awk '{print $3}')
  SDK_VER="$MAJ.$MIN.$PAT"
fi
WHEEL=$(ls dependencies/pyzed-*.whl 2>/dev/null | tail -1 || true)
WHEEL_VER=$(basename "${WHEEL:-pyzed-unknown}" | sed -E 's/pyzed-([0-9]+\.[0-9]+).*/\1/')
echo "SDK: ${SDK_VER:-unknown}   bundled wheel: ${WHEEL:-none} (${WHEEL_VER})"
if [[ -n "$SDK_VER" && "$SDK_VER" != "$WHEEL_VER"* ]]; then
  echo "WARNING: wheel $WHEEL_VER != SDK $SDK_VER. Regenerate the wheel for this SDK:"
  echo "           python3 $ZED_DIR/get_python_api.py     # downloads pyzed-<ver>-cp311-...whl into cwd"
  echo "         then copy it into dependencies/ and point [tool.uv.sources].pyzed at it in pyproject.toml."
fi

echo
echo "== USB bus =="
if lsusb 2>/dev/null | grep -qi "2b03"; then
  lsusb | grep -i 2b03
else
  echo "No Stereolabs USB device (vendor 2b03) enumerated right now."
  echo "  - USB ZED / ZED 2i / ZED Mini: plug into a USB 3 (blue) port, then re-run."
  echo "  - ZED X / ZED X One are GMSL2 and only attach to a Jetson: run the SDK streaming sender there"
  echo "    (ZED_Streaming_Sender or xdof's zed bridge) and use stream_ip in the CameraNode config."
fi

echo
echo "== git submodules (i2rt, yam active leader) =="
git submodule update --init --recursive

if [[ "$DO_SYNC" == "1" ]]; then
  echo
  echo "== python env (uv sync --extra sensors --extra realsense) =="
  AVAIL_GB=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
  if [[ "${AVAIL_GB:-0}" -lt 6 ]]; then
    echo "WARNING: only ${AVAIL_GB} GB free on the filesystem holding $REPO; uv may need several GB for"
    echo "         torch/jax wheels that are not yet in ~/.cache/uv. Free space first if this fails."
  fi
  uv sync --extra sensors --extra realsense
  echo
  echo "== verify =="
  uv run python -c "import pyzed.sl as sl; print('pyzed OK, SDK runtime', sl.Camera.get_sdk_version())"
  uv run python -c "import openpi_client, pyrealsense2; print('openpi_client + pyrealsense2 OK')"
fi

echo
echo "Next:  uv run scripts/probe_zed_cameras.py --yaml"
