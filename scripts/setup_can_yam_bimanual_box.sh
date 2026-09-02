#!/usr/bin/env bash
# Bring up the two YAM follower CAN buses at the us05 station with *stable*
# names (can_left / can_right), independent of kernel enumeration order.
#
# The two CANable 2.5 adapters are identified by their USB serial numbers, so
# replugging them into different ports (or a reboot that swaps can0/can1) still
# yields the same left/right assignment:
#
#   207B34A158455017  ->  can_right   (right YAM follower)
#   208237984546500A  ->  can_left    (left  YAM follower)
#
# These names are what robot_configs/yam/xdof_hq/{left,right}.yaml expect.
#
# This script renames for the *current boot only*. To make the names stick
# across reboots and replugs, add these two lines to the station's existing
# /etc/udev/rules.d/90-can.rules (whose other NAME= entries reference CANable
# adapters that are no longer attached here) and replug the adapters:
#
#   SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="208237984546500A", NAME="can_left"
#   SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="207B34A158455017", NAME="can_right"
#
# That same rules file also auto-applies bitrate 1000000 and brings can* up on
# every net event, so with those lines in place this script becomes a no-op.
#
# Usage:
#   ./scripts/setup_can_yam_bimanual_us05.sh            # rename + bring up (needs sudo)
#   ./scripts/setup_can_yam_bimanual_us05.sh --check    # print mapping only, no changes
#
# Verifying / fixing the left-right assignment (if the arms are swapped):
#   1. Run with --check to see which serial currently backs which interface.
#   2. Swap the two serials in the LEFT_SERIAL / RIGHT_SERIAL lines below.
# The mapping in this file was determined on 2026-08-20 by reading both buses
# passively and matching each arm's forward kinematics against the overhead
# camera view.

# Re-exec under bash: this script uses bash features, and `sh setup_...sh`
# (dash) dies on `set -o pipefail`.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

BITRATE=1000000
LEFT_SERIAL="208237984546500A"
RIGHT_SERIAL="207B34A158455017"

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

# Echo the current netdev name for a CANable USB serial ("" if not present).
iface_for_serial() {
    local want="$1" iface serial
    for iface in $(ls /sys/class/net); do
        [[ -d "/sys/class/net/$iface/device" ]] || continue
        serial=$(udevadm info -a -p "/sys/class/net/$iface" 2>/dev/null \
                 | grep -m1 'ATTRS{serial}=="[0-9A-Fa-f]\{16\}"' \
                 | sed 's/.*=="\(.*\)"/\1/') || true
        if [[ "$serial" == "$want" ]]; then
            echo "$iface"
            return 0
        fi
    done
    echo ""
}

LEFT_IF=$(iface_for_serial "$LEFT_SERIAL")
RIGHT_IF=$(iface_for_serial "$RIGHT_SERIAL")

echo "left  arm: serial $LEFT_SERIAL  -> ${LEFT_IF:-<not found>}"
echo "right arm: serial $RIGHT_SERIAL -> ${RIGHT_IF:-<not found>}"

if [[ -z "$LEFT_IF" || -z "$RIGHT_IF" ]]; then
    echo ""
    echo "ERROR: a CANable adapter is missing. Plugged-in adapters:"
    lsusb | grep -i "1d50:606f" || echo "  (none — check power / USB cable)"
    exit 1
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo ""
    ip -brief link show | grep can || true
    exit 0
fi

current_bitrate() {
    ip -details link show "$1" 2>/dev/null | grep -oP 'bitrate \K[0-9]+' | head -1
}

# Bring an interface to $BITRATE and UP, tolerating the station's udev rule
# (/etc/udev/rules.d/90-can.rules) having already done it. That rule fires on
# every net event for can* — including a rename — and races us: setting the
# bitrate on an interface it has already re-upped fails with EBUSY.
ensure_bitrate_and_up() {
    local ifn="$1"
    if [[ "$(current_bitrate "$ifn")" != "$BITRATE" ]]; then
        sudo ip link set "$ifn" down
        sudo ip link set "$ifn" type can bitrate "$BITRATE"
    fi
    if ! ip link show "$ifn" | grep -q "state UP"; then
        sudo ip link set "$ifn" up
    fi
    local got
    got=$(current_bitrate "$ifn")
    if [[ "$got" != "$BITRATE" ]] || ! ip link show "$ifn" | grep -q "state UP"; then
        echo "  ✗ $ifn is not up @ ${BITRATE} bit/s (bitrate=${got:-unset})"
        return 1
    fi
    echo "  ✓ $ifn up @ ${BITRATE} bit/s"
}

rename_and_up() {
    local cur="$1" target="$2"
    if [[ "$cur" != "$target" ]]; then
        # A stale interface already holding the target name would block the
        # rename; push it out of the way first.
        if ip link show "$target" &>/dev/null; then
            sudo ip link set "$target" down
            sudo ip link set "$target" name "${target}_old"
        fi
        sudo ip link set "$cur" down
        sudo ip link set "$cur" name "$target"
        # Let the udev rule finish its bitrate+up pass on the new name before
        # we inspect the interface, otherwise we race it.
        sleep 0.5
    fi
    ensure_bitrate_and_up "$target"
}

rename_and_up "$LEFT_IF" can_left
rename_and_up "$RIGHT_IF" can_right

echo ""
ip -brief link show | grep can
echo ""
echo "Ready. Launch with:"
echo "  uv run python -m robots_realtime.main configs/yam/yam_bimanual_viser_teleop_xdof_hq_us05.yaml"
