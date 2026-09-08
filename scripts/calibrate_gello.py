"""First-time GELLO (YAM active leader) calibration.

Workaround for lerobot-calibrate, which calls connect(calibrate=False) and then
trips the plugin's trigger calibration before any arm calibration exists.
connect(calibrate=True) runs the interactive arm calibration first, then the
trigger calibration, in the right order.

Usage:
    uv run python calibrate_gello.py /dev/ttyACM0 left
    uv run python calibrate_gello.py /dev/ttyACM1 right
"""
import sys

import lerobot.robots  # noqa: F401 — resolves lerobot's circular import
from lerobot_teleoperator_yamactiveleader import (
    YamActiveLeaderTeleoperator,
    YamActiveLeaderTeleoperatorConfig,
)

port, arm_id = sys.argv[1], sys.argv[2]
teleop = YamActiveLeaderTeleoperator(
    YamActiveLeaderTeleoperatorConfig(port=port, id=arm_id, use_degrees=True)
)
try:
    teleop.connect(calibrate=True)
    print(f"\nDone. Calibration file: {teleop.calibration_fpath}")
finally:
    if teleop.is_connected:
        teleop.disconnect()
