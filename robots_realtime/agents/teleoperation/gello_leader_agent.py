"""Agent that reads joint positions from a GELLO-style feetech leader device
and outputs joint-position actions for a simulated (or real) follower robot.

The leader device is accessed through the ``YamActiveLeaderTeleoperator``
class from the lerobot plugin.
"""

import logging
import threading
from collections import deque
from typing import Any, Dict, List, Optional

import lerobot.robots  # noqa: F401 — resolve circular import in lerobot
import numpy as np
from dm_env.specs import Array
from lerobot_teleoperator_yamactiveleader import (
    YamActiveLeaderTeleoperator,
    YamActiveLeaderTeleoperatorConfig,
)

from robots_realtime.agents.agent import Agent

logger = logging.getLogger(__name__)

NUM_ARM_JOINTS = 6


class GelloLeaderAgent(Agent):
    """Teleoperation agent backed by a GELLO feetech leader arm.

    Reads the leader's joint positions (in degrees) and converts them
    to radians for output as follower joint-position commands.

    ``act()`` always returns ``{robot_name: {"pos": ...}}``.  When
    ``record_on_intervention=True`` it additionally emits ``{"_record": bool}``
    so the control loop can drive a :class:`TrajectoryLogger` without any
    external trigger — recording starts automatically when the human pushes
    against the DAgger hold and stops when they release.

    ``act()`` always returns ``{robot_name: {"pos": ...}}``.  When
    ``record_on_intervention=True`` it additionally emits ``{"_record": bool}``
    so the control loop can drive a :class:`TrajectoryLogger` without any
    external trigger — recording starts automatically when the human pushes
    against the DAgger hold and stops when they release.

    Args:
        port: Serial port for the feetech bus
            (e.g. ``/dev/tty.usbmodem5AE60805531``).
        robot_name: Key used in the returned action dict
            (must match the robot name in the env config).
        calibrate: Whether to run calibration on connect
            (set to False if already calibrated).
        joint_signs: Per-joint sign multipliers (length 6).
            Use ``-1`` to flip a joint direction.  Defaults to all ``1``.
        joint_offsets_deg: Per-joint offsets in degrees added *after*
            sign flip, *before* conversion to radians.  Defaults to
            all ``0``.
        use_degrees: If True the teleoperator is configured to return
            positions in degrees (default).
        drive_to_zero: If True (default), actively drive all motors to
            the zero-config position on startup, then release the arm
            joints so the operator can move them.
        hold_gripper: If True (default), keep torque enabled on the
            gripper motor so it actively resists changes (holds the
            open position).
        include_gripper: If True, include the gripper value as a 7th
            element in the action output.  Set to False for sim models
            that have no gripper joint.  Defaults to False.
        record_on_intervention: Emit ``_record=True`` whenever the
            DAgger intervention sensor is active (human pushing against
            the held pose).  Useful for automatically labelling
            corrective demonstrations without any external trigger.
        record_on_intervention: Emit ``_record=True`` whenever the
            DAgger intervention sensor is active (human pushing against
            the held pose).  Useful for automatically labelling
            corrective demonstrations without any external trigger.
    """

    use_joint_state_as_action: bool = False

    # Reasonable mock pose for DAgger debugging: slight forward reach with elbow bent.
    # Override via dagger_debug_pose_rad in the constructor or YAML config.
    DAGGER_DEBUG_POSE_RAD: List[float] = [0.2, 0.7, 0.9, -0.7, 0.7, 0.2]

    def __init__(
        self,
        port: str = "/dev/tty.usbmodem5AE60805531",
        robot_name: str = "left",
        device_id: str = "",
        calibrate: bool = True,
        joint_signs: Optional[List[int]] = None,
        joint_offsets_deg: Optional[List[float]] = None,
        use_degrees: bool = True,
        drive_to_zero: bool = True,
        hold_gripper: bool = True,
        include_gripper: bool = False,
        dither: bool = True,
        dagger_debug: bool = False,
        dagger_debug_pose_rad: Optional[List[float]] = False,
        record_on_intervention: bool = False,
    ) -> None:
        self.robot_name = robot_name
        self.joint_signs = np.array(joint_signs or [1] * NUM_ARM_JOINTS, dtype=np.float64)
        self.joint_offsets_deg = np.array(joint_offsets_deg or [0] * NUM_ARM_JOINTS, dtype=np.float64)
        self.include_gripper = include_gripper
        self.record_on_intervention = record_on_intervention
        self._held_action: Optional[Dict[str, Any]] = None

        # device_id determines the calibration file name; fall back to robot_name
        # so a device named "left" automatically picks up left.json.
        effective_id = device_id or robot_name
        config = YamActiveLeaderTeleoperatorConfig(port=port, use_degrees=use_degrees, id=effective_id)
        self.teleop = YamActiveLeaderTeleoperator(config)

        # Calibration requires interactive stdin — not available in a subprocess.
        # Require a pre-existing calibration file; monkey-patch builtins.input to
        # return "" (= keep existing calibration) for the duration of connect().
        # First-time setup: lerobot-calibrate --teleop.type=yam_active_leader
        #                   --teleop.port=<port> --teleop.id=<robot_name>
        if calibrate:
            calib_path = getattr(self.teleop, "calibration_fpath", None)
            if calib_path is None or not calib_path.is_file():
                raise RuntimeError(
                    f"No calibration file found for GELLO '{effective_id}'.\n"
                    f"Expected: {calib_path}\n"
                    f"Run:  lerobot-calibrate --teleop.type=yam_active_leader"
                    f" --teleop.port={port} --teleop.id={effective_id}"
                )
            logger.info("[%s] using calibration file: %s", effective_id, calib_path)

        import builtins
        _orig_input = builtins.input
        builtins.input = lambda *_a, **_kw: ""
        try:
            self.teleop.connect(calibrate=calibrate)
        finally:
            builtins.input = _orig_input
        logger.info("GelloLeaderAgent connected to %s (id=%s)", port, effective_id)

        # ---- Active motor control at startup ---- #
        if drive_to_zero:
            self.teleop.drive_to_zero()
            self.teleop.start_arm_hold()

        if hold_gripper:
            self.teleop.start_gripper_spring()

        if dither:
            self.teleop.start_arm_dither()

        if dagger_debug:
            pose = np.array(dagger_debug_pose_rad or self.DAGGER_DEBUG_POSE_RAD, dtype=np.float64)
            # Invert agent transform: output_rad = deg2rad(signs * leader_deg + offsets)
            # → leader_deg = (rad2deg(output_rad) - offsets) * signs  (signs are ±1)
            target_deg = (np.rad2deg(pose) - self.joint_offsets_deg) * self.joint_signs
            target_dict = {f"joint_{i + 1}": float(target_deg[i]) for i in range(NUM_ARM_JOINTS)}
            arm_pos = pose[:NUM_ARM_JOINTS].astype(np.float32)
            if include_gripper:
                raw = self.teleop.get_action()
                arm_pos = np.concatenate([arm_pos, [raw["gripper.pos"]]]).astype(np.float32)
            self._held_action = {self.robot_name: {"pos": arm_pos}}
            self.teleop.drive_to_config(target_dict)
            self.teleop.start_arm_hold()
            logger.info("DAgger debug hold armed at pose (rad): %s", pose)

    # ------------------------------------------------------------------ #
    # DAgger intervention support
    # ------------------------------------------------------------------ #

    @property
    def is_intervening(self) -> bool:
        """True once the human has actively pushed against the held pose.

        Useful for a DAgger data-collection loop to label transitions
        between policy rollout and human correction.
        """
        return self.teleop.is_arm_hold_intervening

    def clear_intervention(self) -> None:
        """Reset DAgger hold state to begin a new cycle."""
        self.teleop.clear_arm_hold()

    # ------------------------------------------------------------------ #
    # Agent protocol
    # ------------------------------------------------------------------ #

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Read the leader device and return a position action.

        While an arm hold is active (after :pymeth:`drive_to_config` but
        before intervention is detected), returns the held config and polls
        the intervention detector each tick.  Once the human pushes,
        resumes live teleoperation immediately.

        Returns:
            ``{robot_name: {"pos": np.ndarray}}``  —  6 arm-joint angles
            in radians (optionally followed by 1 gripper value if
            *include_gripper* is True).
        """
        # --- Hold mode: return held config and monitor for intervention ---
        # Only active when _held_action was explicitly set (dagger_debug mode).
        # During normal teleop, start_arm_hold() runs on the device but we
        # still return live joint positions.
        if self.teleop.is_arm_hold_active and self._held_action is not None:
            self.teleop.update_arm_hold()
            if not self.teleop.is_arm_hold_intervening:
                return self._held_action

        # --- Normal teleoperation ---
        action = self.teleop.get_action()

        # Adaptive gripper spring — adjusts torque each frame
        # self.teleop.update_gripper_spring(action["gripper.pos"])
        # self.teleop.read_gripper_spring_state()

        # Extract the 6 arm joints (degrees)
        joint_deg = np.array([action[f"joint_{i}.pos"] for i in range(1, NUM_ARM_JOINTS + 1)])

        # Apply sign flips and offsets, then convert to radians
        joint_deg = self.joint_signs * joint_deg + self.joint_offsets_deg
        joint_rad = np.deg2rad(joint_deg)

        if self.include_gripper:
            gripper = action["gripper.pos"]
            # gripper = np.clip(1 - ((gripper - 5) / (85 - 5)), 0, 1)  # normalize gripper position to 0-1 range, w/ some deadzone
            pos = np.concatenate([joint_rad, [gripper]])
        else:
            pos = joint_rad

        out: Dict[str, Any] = {self.robot_name: {"pos": pos.astype(np.float32)}}
        if self.record_on_intervention:
            out["_record"] = bool(self.teleop.is_arm_hold_intervening)
        return out

    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        n = NUM_ARM_JOINTS + (1 if self.include_gripper else 0)
        return {self.robot_name: {"pos": Array(shape=(n,), dtype=np.float32)}}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self.teleop.disconnect()
        logger.info("GelloLeaderAgent disconnected.")

    def reset(self) -> None:
        pass


class _ArmThread:
    """Runs a YamActiveLeaderTeleoperator at maximum bus frequency in a background thread.

    Calls ``get_action()`` (which includes ``update_gripper_spring`` and
    ``update_arm_dither``) and ``update_arm_hold()`` as fast as the serial
    protocol allows.  The main env loop reads the latest buffered action
    without blocking on bus I/O.

    Commands from the main thread (dither start, hold release) are delivered
    via simple boolean flags consumed once at the top of each loop iteration.
    Python's GIL makes single bool writes atomic, so no extra lock is needed
    for these flags.
    """

    def __init__(self, teleop: YamActiveLeaderTeleoperator) -> None:
        self.teleop = teleop
        self._lock = threading.Lock()
        self._latest_raw: Dict[str, float] | None = None
        # Set by the arm thread when hold intervention fires; cleared by main thread.
        self.intervention_event = threading.Event()
        # Flags written by the main thread, consumed once by the arm thread.
        self._pending_dither_start = False
        self._pending_release = False
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"arm-{self.teleop.id}"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while self._running:
            # Consume pending commands from the main thread.
            if self._pending_release:
                self._pending_release = False
                self.teleop.clear_arm_hold()
                self.teleop.bus.disable_torque(self.teleop.ARM_MOTORS)
            if self._pending_dither_start:
                self._pending_dither_start = False
                self.teleop.start_arm_dither()

            # Pump arm hold detector; signal main thread on first intervention.
            if self.teleop.is_arm_hold_active:
                if self.teleop.update_arm_hold():
                    self.intervention_event.set()

            # Read action — includes update_gripper_spring + update_arm_dither.
            try:
                raw = self.teleop.get_action()
                with self._lock:
                    self._latest_raw = raw
            except Exception as e:
                logger.warning("ArmThread %s: get_action failed: %s", self.teleop.id, e)

    @property
    def latest_raw(self) -> Dict[str, float] | None:
        with self._lock:
            return self._latest_raw

    @property
    def is_arm_hold_active(self) -> bool:
        return self.teleop.is_arm_hold_active

    @property
    def is_arm_hold_intervening(self) -> bool:
        return self.teleop.is_arm_hold_intervening

    def request_release(self) -> None:
        """Signal the arm thread to clear its hold and disable arm torque."""
        self._pending_release = True

    def request_dither_start(self) -> None:
        """Signal the arm thread to call start_arm_dither() on the next tick."""
        self._pending_dither_start = True


class BimanualGelloLeaderAgent(Agent):
    """Teleoperation agent backed by two GELLO feetech leader arms (bimanual).

    Reads both leader arms and returns a combined 14-DOF action
    ``[left_j1..6, left_grip, right_j1..6, right_grip]`` under a single
    robot key.  This matches the 14-DOF bimanual layout
    ``[left_j1..6, left_grip, right_j1..6, right_grip]``.

    Recording signal — ``act()`` emits ``{"_record": bool}`` alongside the
    action when ``record_on_intervention=True`` (either arm hold-current
    intervening) or ``record_on_gripper_squeeze=True`` (both grippers
    squeezed past ``gripper_squeeze_threshold``; fully open ≈ 85°, closed ≈ 5°).
    Intervention takes precedence if both flags are set.

    Recording signal — ``act()`` emits ``{"_record": bool}`` alongside the
    action when ``record_on_intervention=True`` (either arm hold-current
    intervening) or ``record_on_gripper_squeeze=True`` (both grippers
    squeezed past ``gripper_squeeze_threshold``; fully open ≈ 85°, closed ≈ 5°).
    Intervention takes precedence if both flags are set.

    Args:
        left_port: Serial port for the left-arm feetech bus.
        right_port: Serial port for the right-arm feetech bus.
        robot_name: Key used in the returned action dict (must match the
            robot key in the env config, e.g. ``"right"``).
        calibrate: Whether to run calibration on connect.
        left_joint_signs: Per-joint sign multipliers for the left arm.
        right_joint_signs: Per-joint sign multipliers for the right arm.
        left_joint_offsets_deg: Per-joint degree offsets for the left arm.
        right_joint_offsets_deg: Per-joint degree offsets for the right arm.
        use_degrees: Pass-through to ``YamActiveLeaderTeleoperatorConfig``.
        drive_to_zero: Drive both arms to zero on startup.
        hold_gripper: Keep gripper torque enabled on both arms.
        include_gripper: Include the gripper value (7th element per arm).
            Defaults to True for bimanual.
        record_on_intervention: Emit ``_record=True`` whenever either arm's
            hold-current sensor detects an intervention (OR semantics).
            Default False.
        hold_delta_threshold: Raw current delta above baseline required to
            trigger intervention detection.  Lower = more sensitive.
            Default 5.0 (library default is 8.0, which is too conservative
            for arms at rest with 0 baseline current).
        hold_filter_alpha: EMA smoothing factor for live current readings.
            Higher = faster response, more noise.  Default 0.3.
            (Library default is 0.1, which is too slow.)
        record_on_gripper_squeeze: Emit ``_record=True`` while both grippers
            are simultaneously squeezed.  Default False.
        gripper_squeeze_threshold: Raw gripper position (degrees) below which
            a gripper is considered "squeezed".  Default 60.0.
        auto_stop_on_static: Override ``_record`` to False when arm joint
            positions have been essentially static (operator let go) for
            ``static_frames`` consecutive ticks.  Works with any trigger.
        static_threshold_rad: Max per-joint delta (rad) per tick below which a
            frame is considered static.  Default 0.003 rad (~0.17°).
        static_frames: Consecutive static ticks required before auto-stop.
            At 30 Hz, 60 frames ≈ 2 s.
        record_on_intervention: Emit ``_record=True`` whenever either arm's
            hold-current sensor detects an intervention (OR semantics).
            Default False.
        hold_delta_threshold: Raw current delta above baseline required to
            trigger intervention detection.  Lower = more sensitive.
            Default 5.0 (library default is 8.0, which is too conservative
            for arms at rest with 0 baseline current).
        hold_filter_alpha: EMA smoothing factor for live current readings.
            Higher = faster response, more noise.  Default 0.3.
            (Library default is 0.1, which is too slow.)
        record_on_gripper_squeeze: Emit ``_record=True`` while both grippers
            are simultaneously squeezed.  Default False.
        gripper_squeeze_threshold: Raw gripper position (degrees) below which
            a gripper is considered "squeezed".  Default 60.0.
        auto_stop_on_static: Override ``_record`` to False when arm joint
            positions have been essentially static (operator let go) for
            ``static_frames`` consecutive ticks.  Works with any trigger.
        static_threshold_rad: Max per-joint delta (rad) per tick below which a
            frame is considered static.  Default 0.003 rad (~0.17°).
        static_frames: Consecutive static ticks required before auto-stop.
            At 30 Hz, 60 frames ≈ 2 s.
    """

    def __init__(
        self,
        left_port: str,
        right_port: str,
        left_id: str,
        right_id: str,
        robot_name: str = "yam_bimanual",
        calibrate: bool = True,
        left_joint_signs: Optional[List[int]] = None,
        right_joint_signs: Optional[List[int]] = None,
        left_joint_offsets_deg: Optional[List[float]] = None,
        right_joint_offsets_deg: Optional[List[float]] = None,
        use_degrees: bool = True,
        drive_to_zero: bool = True,
        hold_gripper: bool = True,
        include_gripper: bool = True,
        record_on_intervention: bool = False,
        hold_delta_threshold: float = 5.0,
        hold_filter_alpha: float = 0.3,
        record_on_gripper_squeeze: bool = False,
        gripper_squeeze_threshold: float = 60.0,
        auto_stop_on_static: bool = False,
        static_threshold_rad: float = 0.0025,
        static_frames: int = 60,
        dither: bool = False,
    ) -> None:
        self.left_id = left_id
        self.right_id = right_id

        self.robot_name = robot_name
        self.include_gripper = include_gripper
        self.record_on_intervention = record_on_intervention
        self._hold_delta_threshold = hold_delta_threshold
        self._hold_filter_alpha = hold_filter_alpha
        self.record_on_gripper_squeeze = record_on_gripper_squeeze
        self.gripper_squeeze_threshold = gripper_squeeze_threshold
        self.auto_stop_on_static = auto_stop_on_static
        self.static_threshold_rad = static_threshold_rad
        self._static_window: deque = deque(maxlen=static_frames)
        self._last_joint_pos: Optional[np.ndarray] = None
        self._recording_locked = False  # set after auto-stop; cleared by reset()
        self.record_on_intervention = record_on_intervention
        self._hold_delta_threshold = hold_delta_threshold
        self._hold_filter_alpha = hold_filter_alpha
        self.record_on_gripper_squeeze = record_on_gripper_squeeze
        self.gripper_squeeze_threshold = gripper_squeeze_threshold
        self.auto_stop_on_static = auto_stop_on_static
        self.static_threshold_rad = static_threshold_rad
        self._static_window: deque = deque(maxlen=static_frames)
        self._last_joint_pos: Optional[np.ndarray] = None
        self._recording_locked = False  # set after auto-stop; cleared by reset()

        self.left_joint_signs = np.array(left_joint_signs or [1] * NUM_ARM_JOINTS, dtype=np.float64)
        self.right_joint_signs = np.array(right_joint_signs or [1] * NUM_ARM_JOINTS, dtype=np.float64)
        self.left_joint_offsets_deg = np.array(left_joint_offsets_deg or [0] * NUM_ARM_JOINTS, dtype=np.float64)
        self.right_joint_offsets_deg = np.array(right_joint_offsets_deg or [0] * NUM_ARM_JOINTS, dtype=np.float64)

        left_cfg = YamActiveLeaderTeleoperatorConfig(port=left_port, use_degrees=use_degrees, id=left_id)
        self.left_teleop = YamActiveLeaderTeleoperator(left_cfg)
        self.left_teleop.connect(calibrate=calibrate)
        logger.info("BimanualGelloLeaderAgent: left arm connected to %s", left_port)

        right_cfg = YamActiveLeaderTeleoperatorConfig(port=right_port, use_degrees=use_degrees, id=right_id)
        self.right_teleop = YamActiveLeaderTeleoperator(right_cfg)
        self.right_teleop.connect(calibrate=calibrate)
        logger.info("BimanualGelloLeaderAgent: right arm connected to %s", right_port)

        if drive_to_zero:
            self.left_teleop.drive_to_zero()
            self.right_teleop.drive_to_zero()
            if record_on_intervention:
                # drive_to_zero() releases arm torque at the end so the operator
                # can move freely.  start_arm_hold() needs torque active to sense
                # current — re-enable it by holding the zero config in place.
                _zero = {f"joint_{i}": 0.0 for i in range(1, NUM_ARM_JOINTS + 1)}
                self.left_teleop.drive_to_config(_zero, settle_time=2.0)
                self.right_teleop.drive_to_config(_zero, settle_time=2.0)
            self.left_teleop.start_arm_hold(
                delta_threshold=hold_delta_threshold,
                filter_alpha=hold_filter_alpha,
            )
            self.right_teleop.start_arm_hold(
                delta_threshold=hold_delta_threshold,
                filter_alpha=hold_filter_alpha,
            )

        if hold_gripper:
            self.left_teleop.start_gripper_spring()
            self.right_teleop.start_gripper_spring()

        self._dither_enabled = dither
        self._dither_started = False

        # Start per-arm background threads — must come after all setup.
        self._left_arm = _ArmThread(self.left_teleop)
        self._right_arm = _ArmThread(self.right_teleop)
        self._left_arm.start()
        self._right_arm.start()
        logger.info("BimanualGelloLeaderAgent: arm reader threads started.")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_pos(
        self,
        raw: Dict[str, Any],
        signs: np.ndarray,
        offsets_deg: np.ndarray,
    ) -> np.ndarray:
        """Convert a raw teleop action dict to joint_rad [+ gripper]."""
        joint_deg = np.array([raw[f"joint_{i}.pos"] for i in range(1, NUM_ARM_JOINTS + 1)])
        joint_deg = signs * joint_deg + offsets_deg
        joint_rad = np.deg2rad(joint_deg)
        if self.include_gripper:
            return np.concatenate([joint_rad, [raw["gripper.pos"]]])
        return joint_rad

    # ------------------------------------------------------------------ #
    # Agent protocol
    # ------------------------------------------------------------------ #

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Return combined 14-DOF (or 12-DOF without gripper) action.

        Layout: ``[left_j1..6, (left_grip,) right_j1..6, (right_grip,)]``

        Arm hold detection and motor writes run in per-arm background threads
        at full bus frequency.  This method only reads the latest buffered
        positions and handles cross-arm coordination.

        Returns:
            ``{robot_name: {"pos": np.ndarray}}``
        """
        # Check for intervention events set by arm threads; handle cross-release.
        left_fired = self._left_arm.intervention_event.is_set()
        right_fired = self._right_arm.intervention_event.is_set()
        if left_fired:
            self._left_arm.intervention_event.clear()
            if self._right_arm.is_arm_hold_active:
                self._right_arm.request_release()
                logger.info("Left arm intervened — releasing right arm hold torque.")
        if right_fired:
            self._right_arm.intervention_event.clear()
            if self._left_arm.is_arm_hold_active:
                self._left_arm.request_release()
                logger.info("Right arm intervened — releasing left arm hold torque.")

        logger.debug(
            "Arm hold — left: active=%s intervening=%s | right: active=%s intervening=%s",
            self._left_arm.is_arm_hold_active,
            self._left_arm.is_arm_hold_intervening,
            self._right_arm.is_arm_hold_active,
            self._right_arm.is_arm_hold_intervening,
        )

        # Start dithering on the first tick after both arm holds have released.
        if self._dither_enabled and not self._dither_started:
            if not self._left_arm.is_arm_hold_active and not self._right_arm.is_arm_hold_active:
                self._left_arm.request_dither_start()
                self._right_arm.request_dither_start()
                self._dither_started = True
                logger.info("Arm holds released — dithering started on both arms.")

        # Read the latest buffered actions from the arm threads.
        left_raw = self._left_arm.latest_raw
        right_raw = self._right_arm.latest_raw
        if left_raw is None or right_raw is None:
            # Threads haven't produced their first reading yet — return zeros.
            n = NUM_ARM_JOINTS + (1 if self.include_gripper else 0)
            return {self.robot_name: {"pos": np.zeros(2 * n, dtype=np.float32)}}

        left_pos = self._build_pos(left_raw, self.left_joint_signs, self.left_joint_offsets_deg)
        right_pos = self._build_pos(right_raw, self.right_joint_signs, self.right_joint_offsets_deg)
        combined = np.concatenate([left_pos, right_pos]).astype(np.float32)

        out: Dict[str, Any] = {self.robot_name: {"pos": combined}}

        if not self._recording_locked:
            if self.record_on_intervention:
                out["_record"] = self._left_arm.is_arm_hold_intervening or self._right_arm.is_arm_hold_intervening

            if self.record_on_gripper_squeeze:
                left_grip = left_raw.get("gripper.pos", 85.0)
                right_grip = right_raw.get("gripper.pos", 85.0)
                out["_record"] = (
                    left_grip < self.gripper_squeeze_threshold and right_grip < self.gripper_squeeze_threshold
                )

        if self.auto_stop_on_static and out.get("_record", False):
            # Only track static frames while actively recording — avoids firing
            # at startup when the arms are idle and haven't been touched yet.
            joints = combined[: NUM_ARM_JOINTS * 2]
            if self._last_joint_pos is not None:
                delta = float(np.max(np.abs(joints - self._last_joint_pos)))
                self._static_window.append(delta < self.static_threshold_rad)
                if len(self._static_window) == self._static_window.maxlen and all(self._static_window):
                    out["_record"] = False
                    self._static_window.clear()
                    self._recording_locked = True
                    logger.info("Auto-stop triggered — recording locked until reset.")
            self._last_joint_pos = joints.copy()
        elif self.auto_stop_on_static:
            # Not recording — keep window and last_pos fresh so the first
            # static check after recording starts has a valid reference.
            self._static_window.clear()
            self._last_joint_pos = combined[: NUM_ARM_JOINTS * 2].copy()

        return out

    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        n_per_arm = NUM_ARM_JOINTS + (1 if self.include_gripper else 0)
        return {self.robot_name: {"pos": Array(shape=(2 * n_per_arm,), dtype=np.float32)}}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self._left_arm.stop()
        self._right_arm.stop()
        self.left_teleop.disconnect()
        self.right_teleop.disconnect()
        logger.info("BimanualGelloLeaderAgent disconnected.")

    def reset(self) -> None:
        self._recording_locked = False
        self._static_window.clear()
        self._last_joint_pos = None
        logger.info("BimanualGelloLeaderAgent reset — recording re-armed.")
        self._recording_locked = False
        self._static_window.clear()
        self._last_joint_pos = None
        logger.info("BimanualGelloLeaderAgent reset — recording re-armed.")
