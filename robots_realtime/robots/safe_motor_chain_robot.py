"""SafeMotorChainRobot - A wrapper around i2rt MotorChainRobot with graceful joint limit handling.

This subclass prevents the system from crashing due to minor joint limit violations
during runtime (e.g., from motor PID overshoots). It logs warnings instead and relies
on command clipping to prevent future violations.
"""

import logging

import numpy as np
from i2rt.robots.motor_chain_robot import MotorChainRobot


class SafeMotorChainRobot(MotorChainRobot):
    """MotorChainRobot with non-fatal runtime joint limit violations.

    This class overrides the joint limit checking behavior to make runtime violations
    non-fatal. During initialization, violations still raise errors (indicating
    miscalibration), but during operation, violations only log warnings.
    """

    def __init__(self, *args, allow_initial_joint_limit_violation: bool = False, **kwargs):
        self._initialization_complete = False
        self._allow_initial_joint_limit_violation = bool(allow_initial_joint_limit_violation)
        super().__init__(*args, **kwargs)
        self._initialization_complete = True

    def _check_current_qpos_in_joint_limits(self, buffer_rad: float = 0.1) -> None:
        """Check if the self._joint_state is in the joint limits.

        During initialization: Raise RuntimeError on violation (indicates miscalibration).
        During runtime: Log warning only and rely on command clipping.

        Args:
            buffer_rad: Buffer in radians to add to joint limits (default 0.1)
        """
        if self._joint_state is None or self._joint_limits is None:
            if not self._initialization_complete:
                raise RuntimeError(
                    f"{self}: Joint limits:{self._joint_limits} or joint state:{self._joint_state} are not set."
                )
            return

        current_pos = self._joint_state.pos

        # Check arm joints (exclude gripper if present)
        if self._gripper_index is not None:
            # Only check arm joints, not the gripper
            arm_pos = current_pos[: self._gripper_index]
            arm_limits = self._joint_limits
        else:
            # Check all joints
            arm_pos = current_pos
            arm_limits = self._joint_limits

        # Check if any joint is outside its limits
        lower_limits = arm_limits[:, 0] - buffer_rad
        upper_limits = arm_limits[:, 1] + buffer_rad

        # Find joints that violate lower limits
        lower_violations = arm_pos < lower_limits
        # Find joints that violate upper limits
        upper_violations = arm_pos > upper_limits

        if np.any(lower_violations) or np.any(upper_violations):
            violation_details = []

            for i, (pos, lower, upper) in enumerate(zip(arm_pos, lower_limits, upper_limits, strict=False)):
                if pos < lower:
                    violation_details.append(f"Joint {i}: {pos:.4f} < {lower:.4f} (lower limit)")
                elif pos > upper:
                    violation_details.append(f"Joint {i}: {pos:.4f} > {upper:.4f} (upper limit)")

            violation_msg = "; ".join(violation_details)

            if not self._initialization_complete and not self._allow_initial_joint_limit_violation:
                # During initialization: fatal error (miscalibrated robot)
                self.motor_chain.running = False
                raise RuntimeError(
                    f"{self}: Joint limit violation detected: {violation_msg}, "
                    "the root reason should be zero position offset. "
                    "possible solution: 1. move the arm to zero position and power cycle the robot. "
                    "2. Recalibrate the motor zero position."
                )
            elif not self._initialization_complete:
                logging.warning(
                    "%s: Initial joint limit violation allowed for immediate homing: %s",
                    self,
                    violation_msg,
                )
            else:
                # During runtime: just log a warning (minor overshoot from PID)
                logging.warning(
                    f"{self}: Joint limit violation (non-fatal): {violation_msg}. Commands will be clipped."
                )
