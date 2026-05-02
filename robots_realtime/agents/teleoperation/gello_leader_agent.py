"""Agent that reads joint positions from a GELLO-style feetech leader device
and outputs joint-position actions for a simulated (or real) follower robot.

The leader device is accessed through the ``YamActiveLeaderTeleoperator``
class from the lerobot plugin.
"""

import logging
import time
from collections import defaultdict
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
        profile: bool = False,
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
            self.teleop.connect(
                calibrate=calibrate,
                drive_to_zero=drive_to_zero,
                hold_gripper=hold_gripper,
                dither=dither,
            )
        finally:
            builtins.input = _orig_input
        logger.info("GelloLeaderAgent connected to %s (id=%s)", port, effective_id)

        self._profile = profile
        self._prof_accum: dict[str, list[float]] = defaultdict(list)
        self._prof_act_times: list[float] = []
        self._prof_last_log = time.perf_counter()
        self._prof_log_interval = 5.0  # log every 5 seconds
        if profile:
            self._install_bus_profiler()

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
    # Bus profiler
    # ------------------------------------------------------------------ #

    def _install_bus_profiler(self) -> None:
        """Monkey-patch bus.sync_read / sync_write to accumulate per-call timings."""
        bus = self.teleop.bus
        accum = self._prof_accum
        _orig_sync_read = bus.sync_read
        _orig_sync_write = bus.sync_write

        def _timed_sync_read(data_name, *args, **kwargs):
            t = time.perf_counter()
            result = _orig_sync_read(data_name, *args, **kwargs)
            accum[f"read:{data_name}"].append((time.perf_counter() - t) * 1e3)
            return result

        def _timed_sync_write(data_name, *args, **kwargs):
            t = time.perf_counter()
            result = _orig_sync_write(data_name, *args, **kwargs)
            accum[f"write:{data_name}"].append((time.perf_counter() - t) * 1e3)
            return result

        bus.sync_read = _timed_sync_read
        bus.sync_write = _timed_sync_write
        logger.info("[%s] bus profiler installed", self.robot_name)

    def _maybe_log_profile(self) -> None:
        now = time.perf_counter()
        if now - self._prof_last_log < self._prof_log_interval:
            return
        self._prof_last_log = now

        lines = [f"[{self.robot_name}] --- act() timing breakdown (last {self._prof_log_interval:.0f}s) ---"]

        if self._prof_act_times:
            ts = self._prof_act_times
            lines.append(
                f"  act() total : n={len(ts):4d}  "
                f"mean={sum(ts)/len(ts):6.2f}ms  "
                f"p50={sorted(ts)[len(ts)//2]:6.2f}ms  "
                f"p95={sorted(ts)[int(len(ts)*0.95)]:6.2f}ms  "
                f"max={max(ts):6.2f}ms"
            )
            self._prof_act_times.clear()

        for key in sorted(self._prof_accum):
            ts = self._prof_accum[key]
            if not ts:
                continue
            lines.append(
                f"  {key:<35s}: n={len(ts):4d}  "
                f"mean={sum(ts)/len(ts):6.2f}ms  "
                f"p50={sorted(ts)[len(ts)//2]:6.2f}ms  "
                f"p95={sorted(ts)[int(len(ts)*0.95)]:6.2f}ms  "
                f"max={max(ts):6.2f}ms"
            )
            ts.clear()

        logger.info("\n".join(lines))

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
        _t0 = time.perf_counter() if self._profile else 0.0

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

        # Extract the 6 arm joints (degrees)
        joint_deg = np.array([action[f"joint_{i}.pos"] for i in range(1, NUM_ARM_JOINTS + 1)])

        # Apply sign flips and offsets, then convert to radians
        joint_deg = self.joint_signs * joint_deg + self.joint_offsets_deg
        joint_rad = np.deg2rad(joint_deg)

        if self.include_gripper:
            gripper = action["gripper.pos"]
            pos = np.concatenate([joint_rad, [gripper]])
        else:
            pos = joint_rad

        out: Dict[str, Any] = {self.robot_name: {"pos": pos.astype(np.float32)}}
        if self.record_on_intervention:
            out["_record"] = bool(self.teleop.is_arm_hold_intervening)

        if self._profile:
            self._prof_act_times.append((time.perf_counter() - _t0) * 1e3)
            self._maybe_log_profile()

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
