import time

import lerobot.robots  # noqa: F401 — resolve circular import in lerobot

from lerobot_teleoperator_yamactiveleader import (
    YamActiveLeaderTeleoperatorConfig,
    YamActiveLeaderTeleoperator,
)

teleop = YamActiveLeaderTeleoperator(
    YamActiveLeaderTeleoperatorConfig(port="/dev/tty.usbmodem5AE60805531")
)
teleop.connect()

try:
    print("Reading joint angles at ~50Hz. Press Ctrl+C to stop.\n")
    while True:
        action = teleop.get_action()
        parts = [f"{name}: {val:7.2f}" for name, val in action.items()]
        print("  |  ".join(parts), end="\r")
        time.sleep(0.02)

except KeyboardInterrupt:
    print()

finally:
    teleop.disconnect()
    print("Disconnected.")