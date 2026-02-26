"""Position-position bilateral control with the YAM active leader arm.

Theory of operation
-------------------
Classic position-position bilateral control runs two loops simultaneously:

    leader  ──pos──►  follower
    leader  ◄──pos──  follower

The leader device sends its joint angles to the simulated (or real) follower.
In the return path the *follower's* joint positions are written back as
``Goal_Position`` targets on the Feetech STS3215 arm motors.  The motor's
internal PD controller then generates a restoring torque:

    τ  ∝  Kp × (q_follower − q_leader)

The effective stiffness is tuned via ``Torque_Limit`` (0–1000 on the
STS3215), which caps the peak torque without touching the ``P_Coefficient``
register (EEPROM).  The ``bilateral_kp`` argument (0.0–1.0) linearly scales
this limit:

    torque_limit = int(bilateral_kp × bilateral_torque_max)

A ``warmup_steps`` grace period (default 5 steps) skips bilateral feedback
at startup while the sim follower converges from its initial state to match
the leader, preventing a jerk on the first frame.

Coordinate-frame inversion
--------------------------
The forward transform applied in ``act()`` is::

    q_follower_rad = deg2rad(joint_signs × q_leader_deg + joint_offsets_deg)

The inverse (follower → leader command) is::

    q_leader_cmd_deg = joint_signs × (rad2deg(q_follower_rad) − joint_offsets_deg)

(Valid because every element of ``joint_signs`` is ±1, so 1/sign == sign.)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from dm_env.specs import Array

import lerobot.robots  # noqa: F401 — resolve circular import in lerobot

from lerobot_teleoperator_yamactiveleader import (
    YamActiveLeaderTeleoperator,
    YamActiveLeaderTeleoperatorConfig,
)

from robots_realtime.agents.agent import Agent

logger = logging.getLogger(__name__)

NUM_ARM_JOINTS = 6


class BilateralLeaderAgent(Agent):
    """Teleoperation agent with position-position bilateral haptic feedback.

    Wraps the YAM active leader Feetech bus.  Each call to :py:meth:`act`:

    1. Reads the leader's joint angles.
    2. Writes the follower's last observed joint positions back to the
       leader motors as position targets (bilateral return path).
    3. Returns the leader angles as a follower action dict.

    The bilateral return path is skipped for the first *warmup_steps*
    control iterations so the sim can settle to the leader's initial pose
    before any feedback force is applied.

    Args:
        port: Serial port for the Feetech bus.
        robot_name: Key used in the returned action dict and used to look up
            the follower state in ``obs`` (must match the robot name in the
            env config).
        calibrate: Run motor calibration on connect.
        joint_signs: Per-joint sign multipliers applied to the leader reading
            before converting to radians.  Length 6.
        joint_offsets_deg: Per-joint degree offsets added after sign flip,
            before unit conversion.  Length 6.
        use_degrees: Configure the Feetech bus to return positions in degrees.
        drive_to_zero: Actively drive all motors to zero on startup, then
            release arm joints.
        hold_gripper: Keep gripper torque enabled with the adaptive spring.
        include_gripper: Forward the gripper value as a 7th DOF.
        bilateral_kp: Bilateral stiffness as a fraction of
            *bilateral_torque_max* in [0.0, 1.0].  0 disables feedback
            entirely (pass-through teleoperation).
        bilateral_torque_max: Maximum ``Torque_Limit`` value (0–1000) applied
            to the arm motors during bilateral feedback.  Scale down for
            softer haptics; scale up for stiffer.
        warmup_steps: Number of control steps to skip bilateral feedback at
            startup (lets the sim follower track the leader before engaging
            the return path).
    """

    def __init__(
        self,
        port: str = "/dev/tty.usbmodem5AE60805531",
        robot_name: str = "yam",
        calibrate: bool = True,
        joint_signs: Optional[List[int]] = None,
        joint_offsets_deg: Optional[List[float]] = None,
        use_degrees: bool = True,
        drive_to_zero: bool = True,
        hold_gripper: bool = True,
        include_gripper: bool = True,
        bilateral_kp: float = 0.3,
        bilateral_torque_max: int = 300,
        warmup_steps: int = 5,
    ) -> None:
        self.robot_name = robot_name
        self.joint_signs = np.array(joint_signs or [1] * NUM_ARM_JOINTS, dtype=np.float64)
        self.joint_offsets_deg = np.array(
            joint_offsets_deg or [0.0] * NUM_ARM_JOINTS, dtype=np.float64
        )
        self.include_gripper = include_gripper
        self.bilateral_kp = float(bilateral_kp)
        self._torque_limit = int(np.clip(bilateral_kp * bilateral_torque_max, 0, 1000))
        self._warmup_steps = warmup_steps
        self._step = 0
        self._bilateral_enabled = bilateral_kp > 0.0

        config = YamActiveLeaderTeleoperatorConfig(port=port, use_degrees=use_degrees)
        self.teleop = YamActiveLeaderTeleoperator(config)
        self.teleop.connect(calibrate=calibrate)
        logger.info("BilateralLeaderAgent connected to %s (bilateral_kp=%.2f)", port, bilateral_kp)

        if drive_to_zero:
            self.teleop.drive_to_zero()

        if hold_gripper:
            self.teleop.start_gripper_spring()

        # Pre-compute the set of arm motor names for reuse.
        self._arm_motors: List[str] = self.teleop.ARM_MOTORS  # ["joint_1", ..., "joint_6"]

    # ------------------------------------------------------------------ #
    # Agent protocol
    # ------------------------------------------------------------------ #

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Read leader, apply bilateral feedback, return follower command.

        Returns:
            ``{robot_name: {"pos": np.ndarray}}`` — 6 arm-joint angles in
            radians (+ 1 gripper value when *include_gripper* is True).
        """
        # ── 1. Read leader ──────────────────────────────────────────────
        action = self.teleop.get_action()
        self.teleop.update_gripper_spring(action["gripper.pos"])

        joint_deg = np.array([action[f"joint_{i}.pos"] for i in range(1, NUM_ARM_JOINTS + 1)])
        joint_deg = self.joint_signs * joint_deg + self.joint_offsets_deg
        joint_rad = np.deg2rad(joint_deg)

        # ── 2. Bilateral return path ─────────────────────────────────────
        if self._bilateral_enabled and self._step >= self._warmup_steps:
            follower_rad = self._extract_follower_pos(obs)
            if follower_rad is not None:
                self._send_bilateral_feedback(follower_rad[:NUM_ARM_JOINTS])

        self._step += 1

        # ── 3. Build follower command ─────────────────────────────────────
        if self.include_gripper:
            gripper = action["gripper.pos"]
            gripper = (1 - ((gripper - 5) / (85 - 5))) / 2
            pos = np.concatenate([joint_rad, [gripper]])
        else:
            pos = joint_rad

        return {self.robot_name: {"pos": pos.astype(np.float32)}}

    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        n = NUM_ARM_JOINTS + (1 if self.include_gripper else 0)
        return {self.robot_name: {"pos": Array(shape=(n,), dtype=np.float32)}}

    # ------------------------------------------------------------------ #
    # Bilateral helpers
    # ------------------------------------------------------------------ #

    def _extract_follower_pos(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Pull arm joint positions (radians) from the observation dict."""
        robot_obs = obs.get(self.robot_name)
        if robot_obs is None:
            return None
        return robot_obs.get("joint_pos")

    def _send_bilateral_feedback(self, follower_rad: np.ndarray) -> None:
        """Write follower positions to leader motors as position targets.

        Inverse coordinate transform::

            q_leader_cmd_deg = joint_signs × (rad2deg(q_follower) − joint_offsets_deg)

        The arm motors are enabled at *_torque_limit* so the STS3215 PD
        controller generates a restoring torque proportional to the
        position error (q_follower − q_leader_current).
        """
        # Inverse of: follower_rad = deg2rad(joint_signs * leader_deg + offsets)
        leader_cmd_deg = self.joint_signs * (
            np.rad2deg(follower_rad) - self.joint_offsets_deg
        )

        bus = self.teleop.bus

        # Set bilateral torque limit on all arm motors.
        bus.sync_write(
            "Torque_Limit",
            {m: self._torque_limit for m in self._arm_motors},
        )

        # Ensure torque is enabled (no-op if already on).
        bus.enable_torque(self._arm_motors)

        # Write goal positions (normalized = degrees for MotorNormMode.DEGREES).
        for i, motor in enumerate(self._arm_motors):
            bus.write("Goal_Position", motor, leader_cmd_deg[i], normalize=True)

    def release_bilateral(self) -> None:
        """Disable arm motor torque, returning to free-backdrive mode."""
        self.teleop.bus.disable_torque(self._arm_motors)
        logger.info("Bilateral feedback released — arm motors free.")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self.release_bilateral()
        self.teleop.disconnect()
        logger.info("BilateralLeaderAgent disconnected.")

    def reset(self) -> None:
        self._step = 0
