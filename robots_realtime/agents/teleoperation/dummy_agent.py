"""DummyAgent — synthetic single-arm agent with periodic random targets.

Returns ``{"pos": array}`` from act() with no hardware dependency.
Used for testing and simulation without physical leader arms.
"""

from __future__ import annotations

import time

import numpy as np

_DEFAULT_INIT_Q: dict[str, np.ndarray] = {
    "left": np.array(
        [-0.20656902, 0.47283894, 0.99431604, -0.7043946, -0.30842298, -0.32864118, 0.9987507],
        dtype=np.float32,
    ),
    "right": np.array(
        [0.20160982, 0.39005876, 1.1182956, -0.8726253, 0.13332571, 0.42629892, 0.9895317],
        dtype=np.float32,
    ),
}


class DummyAgent:
    """Synthetic single-arm agent that returns periodic random joint targets.

    Args:
        arm:           Which arm — "left" or "right".  Selects the default base pose.
        init_q:        Override the 7D base pose.
        target_std:    Per-joint std dev of random offsets (radians). Gripper unchanged.
        pose_interval: Wall-clock seconds between new random target draws.
    """

    def __init__(
        self,
        arm: str = "left",
        init_q: list | None = None,
        target_std: float = 0.3,
        pose_interval: float = 1.0,
    ) -> None:
        if init_q is not None:
            self._base_q = np.asarray(init_q, dtype=np.float32).copy()
        elif arm in _DEFAULT_INIT_Q:
            self._base_q = _DEFAULT_INIT_Q[arm].copy()
        else:
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        self._target_std = target_std
        self._pose_interval = pose_interval
        self._current_q = self._base_q.copy()
        self._next_draw_t: float = 0.0

    def reset(self) -> None:
        self._current_q = self._base_q.copy()
        self._next_draw_t = time.time() + self._pose_interval

    def act(self, obs: dict) -> dict:
        now = time.time()
        if now >= self._next_draw_t:
            offset = np.random.randn(len(self._base_q)).astype(np.float32) * self._target_std
            offset[-1] = 0.0  # leave gripper unchanged
            self._current_q = self._base_q + offset
            self._current_q[-1] = float(np.clip(self._base_q[-1], 0.0, 1.0))
            self._next_draw_t = now + self._pose_interval
        return {"pos": self._current_q.copy()}
