"""CAN protocol helpers for passive GELLO encoders and buttons."""

from __future__ import annotations

import struct
import time
from typing import Optional

ENCODER_REPORT_ID = 0x50F
ENCODER_STRUCT = "!B h h B"
ENCODER_STRUCT_SIZE = struct.calcsize(ENCODER_STRUCT)
DEFAULT_BUTTON_DEVICE_ID = 6


def decode_encoder_report(data: bytes) -> Optional[tuple[int, int, int, int]]:
    """Decode a 0x50F encoder report, returning None for a bad payload size."""
    if len(data) != ENCODER_STRUCT_SIZE:
        return None
    device, position, velocity, digital_inputs = struct.unpack(ENCODER_STRUCT, data)
    return int(device), int(position), int(velocity), int(digital_inputs)


class ButtonTracker:
    """Debounce a button byte using the timestamps attached to CAN frames."""

    def __init__(self, debounce_s: float = 0.02) -> None:
        if debounce_s < 0:
            raise ValueError(f"debounce_s must be non-negative, got {debounce_s}")
        self._debounce_s = float(debounce_s)
        self._value = 0
        self._candidate = 0
        self._candidate_since = time.monotonic()

    @property
    def value(self) -> int:
        return self._value

    def update(self, value: int, now: Optional[float] = None) -> bool:
        """Consume a button byte and return whether the stable value changed."""
        timestamp = time.monotonic() if now is None else float(now)
        value = int(value)
        if value == self._value:
            self._candidate = value
            self._candidate_since = timestamp
            return False
        if value != self._candidate:
            self._candidate = value
            self._candidate_since = timestamp
            if self._debounce_s > 0:
                return False
        elif timestamp - self._candidate_since < self._debounce_s:
            return False

        self._value = value
        return True
