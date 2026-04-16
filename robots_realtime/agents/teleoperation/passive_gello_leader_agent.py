"""Agent that reads joint positions from a *passive* GELLO leader arm.

A passive GELLO has no actuators — it is a kinematically-matched skeleton
whose joints carry magnetic encoders. Each encoder streams its angle on a
shared SocketCAN bus (1 Mbit/s) as a fixed-format frame at CAN ID ``0x50F``.

Frame layout (8 bytes? actually 6 bytes — see struct below):

    struct !B h h B
      device_id          : uint8      (0..6 for 7-joint passive gello)
      position_ticks     : int16 BE   (scaled by 2*pi/4096 to radians)
      velocity_ticks     : int16 BE   (same scaling, rad/s)
      digital_inputs     : uint8      (bitfield of switches)

Unlike the active GELLO (which uses ``YamActiveLeaderTeleoperator`` over
serial), we never transmit to the leader — it is genuinely passive, so the
agent is a read-only CAN listener.

Why we do NOT call ``validate_encoders()`` / write encoder EEPROM here:
some encoder firmware revisions stream reports fine but refuse to answer
0x50E command queries, which causes validation to hang or raise. Since we
do not need to reconfigure the report rate, we skip that path entirely
and simply accept whatever rate the EEPROM is already set to (typically
~40 Hz per device, more than enough for teleop).
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from typing import Any, Dict, List, Optional

import can
import numpy as np
from dm_env.specs import Array

from robots_realtime.agents.agent import Agent

logger = logging.getLogger(__name__)

NUM_ARM_JOINTS = 6
GRIPPER_DEVICE_ID = 6
NUM_DEVICES = NUM_ARM_JOINTS + 1

_ENCODER_REPORT_ID = 0x50F
_ENCODER_STRUCT = "!B h h B"
_ENCODER_STRUCT_SIZE = struct.calcsize(_ENCODER_STRUCT)
_TICKS_PER_REV = 4096
_TICKS_TO_RAD = (2.0 * np.pi) / _TICKS_PER_REV

# Distance (rad) from the leader gripper encoder's rest position (~0) to the
# fully-squeezed position, in *either* direction. Lab42's market42 package uses
# the same 0.67 rad constant for all passive gello leaders: the encoder is
# mechanically zeroed at rest, and the trigger throw is ~0.67 rad regardless
# of which way the encoder rotates when squeezed. Overriding per-instance is
# supported via the agent's ``leader_gripper_range_rad`` kwarg.
DEFAULT_LEADER_GRIPPER_RANGE_RAD = 0.67


class _PassiveGelloReader:
    """Background CAN reader that maintains latest joint state for one passive gello.

    Exactly one SocketCAN interface per reader (e.g. ``can_lead_l``). Safe to call
    :meth:`get_joint_pos` from any thread.
    """

    def __init__(self, channel: str, bitrate: int = 1_000_000, alpha: float = 1.0) -> None:
        self._channel = channel
        self._alpha = float(alpha)
        self._bus = can.interface.Bus(interface="socketcan", channel=channel, bitrate=bitrate)
        self._reader = can.BufferedReader()
        self._notifier = can.Notifier(self._bus, [self._reader])

        self._lock = threading.Lock()
        self._positions_rad = np.zeros(NUM_DEVICES, dtype=np.float64)
        self._velocities_rad = np.zeros(NUM_DEVICES, dtype=np.float64)
        self._buttons = 0
        self._seen_devices: set[int] = set()
        self._last_msg_time = time.time()

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"passive_gello[{channel}]", daemon=True)
        self._thread.start()

    def wait_for_all_joints(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if len(self._seen_devices) >= NUM_DEVICES:
                    return
            time.sleep(0.05)
        with self._lock:
            missing = [i for i in range(NUM_DEVICES) if i not in self._seen_devices]
        raise TimeoutError(
            f"passive gello on {self._channel}: only saw devices {sorted(self._seen_devices)} "
            f"within {timeout_s:.1f}s (missing {missing}). Check CAN wiring and power."
        )

    def get_joint_pos(self) -> np.ndarray:
        with self._lock:
            return self._positions_rad.copy()

    def get_joint_vel(self) -> np.ndarray:
        with self._lock:
            return self._velocities_rad.copy()

    def get_buttons(self) -> int:
        with self._lock:
            return self._buttons

    def seconds_since_last_message(self) -> float:
        with self._lock:
            return time.time() - self._last_msg_time

    def close(self) -> None:
        self._stop.set()
        try:
            self._notifier.stop()
        except Exception:
            pass
        try:
            self._bus.shutdown()
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        prev = np.zeros(NUM_DEVICES, dtype=np.float64)
        prev_initialized = np.zeros(NUM_DEVICES, dtype=bool)
        while not self._stop.is_set():
            msg = self._reader.get_message(timeout=0.1)
            if msg is None:
                continue
            if msg.arbitration_id != _ENCODER_REPORT_ID:
                continue
            if len(msg.data) != _ENCODER_STRUCT_SIZE:
                continue
            device_id, pos_ticks, vel_ticks, digital_inputs = struct.unpack(_ENCODER_STRUCT, msg.data)
            if device_id >= NUM_DEVICES:
                continue
            # Ticks wrap at 4096; map into [-pi, pi) centered so zero config lands near zero.
            pos_rad = pos_ticks * _TICKS_TO_RAD
            vel_rad = vel_ticks * _TICKS_TO_RAD
            with self._lock:
                if self._alpha >= 1.0 or not prev_initialized[device_id]:
                    filtered = pos_rad
                else:
                    filtered = (1.0 - self._alpha) * prev[device_id] + self._alpha * pos_rad
                prev[device_id] = filtered
                prev_initialized[device_id] = True
                self._positions_rad[device_id] = filtered
                self._velocities_rad[device_id] = vel_rad
                if device_id == GRIPPER_DEVICE_ID:
                    # Passive gellos carry button state in the gripper device's
                    # digital_inputs field. Harmless if the button isn't wired.
                    self._buttons = int(digital_inputs)
                self._seen_devices.add(device_id)
                self._last_msg_time = time.time()


class PassiveGelloLeaderAgent(Agent):
    """Teleoperation agent backed by a *passive* GELLO leader arm on SocketCAN.

    ``act()`` returns ``{robot_name: {"pos": np.ndarray}}`` with 6 arm joint
    angles in radians, optionally followed by a normalized gripper command
    in ``[0, 1]`` if ``include_gripper`` is True.

    **Gripper output is normalized, not follower radians.** This matches the
    i2rt ``MotorChainRobot`` command convention: the gripper element of
    ``command_joint_pos()`` is a ``[0, 1]`` value that the receiving robot's
    ``JointMapper`` remaps to the follower's ``gripper_limits`` via
    ``out = in * (end - start) + start``. Emitting follower radians here
    would get *re-remapped* on the follower side and land somewhere
    unintended (the lesson that cost us a DM4310 thermal trip).

    Direction-agnostic mapping: the leader's gripper encoder is mechanically
    zeroed at rest and swings by ≈ ``leader_gripper_range_rad`` (default 0.67
    rad) in *either* direction when fully squeezed — the sign depends on
    which way the encoder is mounted. We fold both directions together with
    ``abs()``, so a single symmetric range constant covers any passive gello
    without per-leader calibration::

        t = clip(|encoder_rad| / leader_gripper_range_rad, 0, 1)  # 0=rest, 1=squeeze
        gripper_cmd = 1.0 - t                                     # rest=1=open, squeeze=0=closed

    ``gripper_cmd=0`` will map to ``gripper_limits[0]`` on the follower and
    ``gripper_cmd=1`` will map to ``gripper_limits[1]``. For YAM with
    ``gripper_limits: [0.0, -2.4]``, that's closed=0 at squeeze, open=1 at
    rest. This mirrors lab42/market42's ``xdof/robots/passive_gello.py``.

    Args:
        channel: SocketCAN interface (e.g. ``can_lead_l``).
        robot_name: Key used in the returned action dict.
        joint_signs: Length-6 ±1 per-joint sign multipliers (applied before offsets).
            Defaults to all ``1``.
        joint_offsets_rad: Length-6 radian offsets added after sign flip.
            Defaults to all ``0``.
        include_gripper: If True, append a normalized ``[0, 1]`` gripper
            command as the 7th action element. The receiving robot is
            expected to interpret this as its i2rt-style normalized gripper
            command space (handled automatically by ``MotorChainRobot`` /
            ``SafeMotorChainRobot``).
        leader_gripper_range_rad: Distance (rad) from rest to fully-squeezed
            on the leader encoder, in absolute value. Default 0.67 matches
            lab42's convention. Override only if your hardware differs.
        alpha: Exponential-smoothing factor applied per joint on each new CAN
            sample. 1.0 = no smoothing (default); <1.0 = low-pass filter.
        stale_warn_s: Log a warning if no CAN messages have arrived in this many
            seconds. Set to 0 to disable.
        startup_timeout_s: How long to wait at startup for all 7 encoder devices
            to report at least once before raising.
        bitrate: SocketCAN bitrate. Passive gello encoders run at 1 Mbit/s.
    """

    use_joint_state_as_action: bool = False

    def __init__(
        self,
        channel: str = "can_lead_l",
        robot_name: str = "left",
        joint_signs: Optional[List[int]] = None,
        joint_offsets_rad: Optional[List[float]] = None,
        include_gripper: bool = False,
        leader_gripper_range_rad: float = DEFAULT_LEADER_GRIPPER_RANGE_RAD,
        alpha: float = 1.0,
        stale_warn_s: float = 1.0,
        startup_timeout_s: float = 5.0,
        bitrate: int = 1_000_000,
    ) -> None:
        self.robot_name = robot_name
        self.channel = channel
        self.include_gripper = include_gripper
        self.joint_signs = np.array(joint_signs or [1] * NUM_ARM_JOINTS, dtype=np.float64)
        self.joint_offsets_rad = np.array(joint_offsets_rad or [0.0] * NUM_ARM_JOINTS, dtype=np.float64)
        if self.joint_signs.shape != (NUM_ARM_JOINTS,):
            raise ValueError(f"joint_signs must have length {NUM_ARM_JOINTS}, got {self.joint_signs.shape}")
        if self.joint_offsets_rad.shape != (NUM_ARM_JOINTS,):
            raise ValueError(
                f"joint_offsets_rad must have length {NUM_ARM_JOINTS}, got {self.joint_offsets_rad.shape}"
            )

        self._leader_gripper_range_rad = float(leader_gripper_range_rad)
        if include_gripper and self._leader_gripper_range_rad <= 0.0:
            raise ValueError("leader_gripper_range_rad must be positive when include_gripper=True")

        self._stale_warn_s = float(stale_warn_s)
        self._last_stale_log = 0.0

        self._reader = _PassiveGelloReader(channel=channel, bitrate=bitrate, alpha=alpha)
        try:
            self._reader.wait_for_all_joints(timeout_s=startup_timeout_s)
        except Exception:
            self._reader.close()
            raise
        logger.info("PassiveGelloLeaderAgent connected on %s (robot_name=%s)", channel, robot_name)

    # ------------------------------------------------------------------ #
    # Gripper mapping
    # ------------------------------------------------------------------ #

    def _map_gripper(self, leader_rad: float) -> float:
        """Return a normalized gripper command in [0, 1].

        Rest (|enc| ≈ 0) → 1 (maps to gripper_limits[1] on the follower).
        Squeeze (|enc| ≈ leader_gripper_range_rad) → 0 (maps to gripper_limits[0]).
        """
        t = abs(leader_rad) / self._leader_gripper_range_rad
        t = float(np.clip(t, 0.0, 1.0))
        return 1.0 - t

    # ------------------------------------------------------------------ #
    # Agent protocol
    # ------------------------------------------------------------------ #

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        all_pos = self._reader.get_joint_pos()  # length NUM_DEVICES
        arm_rad = self.joint_signs * all_pos[:NUM_ARM_JOINTS] + self.joint_offsets_rad

        if self.include_gripper:
            gripper_rad = self._map_gripper(float(all_pos[GRIPPER_DEVICE_ID]))
            pos = np.concatenate([arm_rad, [gripper_rad]])
        else:
            pos = arm_rad

        if self._stale_warn_s > 0:
            stale_s = self._reader.seconds_since_last_message()
            now = time.monotonic()
            if stale_s > self._stale_warn_s and (now - self._last_stale_log) > 5.0:
                logger.warning(
                    "PassiveGelloLeaderAgent[%s]: no CAN messages in %.2fs (bus idle?)",
                    self.channel,
                    stale_s,
                )
                self._last_stale_log = now

        return {self.robot_name: {"pos": pos.astype(np.float32)}}

    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        n = NUM_ARM_JOINTS + (1 if self.include_gripper else 0)
        return {self.robot_name: {"pos": Array(shape=(n,), dtype=np.float32)}}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self._reader.close()
        logger.info("PassiveGelloLeaderAgent[%s] disconnected.", self.channel)

    def reset(self) -> None:
        pass
