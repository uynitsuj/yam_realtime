"""MuJoCo simulation robot that implements the i2rt Robot protocol.

Wraps a MuJoCo model and provides forward-kinematics visualization
driven by joint position commands. Optionally launches a passive viewer.
"""

from typing import Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np
from i2rt.robots.robot import Robot


class MujocoSimRobot(Robot):
    """A simulated robot backed by a MuJoCo model.

    Accepts joint-position commands (in radians), updates the model state
    via ``mj_kinematics``, and optionally renders in a passive MuJoCo
    viewer window.

    Args:
        xml_path: Path to the MuJoCo XML model file.
        render: Whether to launch a passive viewer window.
        gripper_index: If set, the last DOF in ``command_joint_pos``
            is treated as a virtual gripper value (not part of the
            MuJoCo model's qpos).
    """

    def __init__(
        self,
        xml_path: str,
        render: bool = True,
        gripper_index: Optional[int] = None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._gripper_index = gripper_index
        self._gripper_pos = np.array([0.0])

        # Total DOFs exposed to the control system
        self._nq = self.model.nq
        self._num_dofs = self._nq + (1 if gripper_index is not None else 0)

        # Optionally launch viewer
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model,
                data=self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)

    # ------------------------------------------------------------------ #
    # Robot protocol
    # ------------------------------------------------------------------ #

    def num_dofs(self) -> int:
        return self._num_dofs

    def get_joint_pos(self) -> np.ndarray:
        qpos = self.data.qpos[: self._nq].copy()
        if self._gripper_index is not None:
            return np.concatenate([qpos, self._gripper_pos])
        return qpos

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Set the sim model's qpos and update kinematics + viewer.

        Args:
            joint_pos: Joint positions in radians.  If ``gripper_index``
                was provided, the last element is the gripper value and
                the preceding elements map to MuJoCo joints.
        """
        self.data.qpos[: self._nq] = joint_pos[: self._nq]
        if self._gripper_index is not None and len(joint_pos) > self._nq:
            self._gripper_pos[0] = joint_pos[self._nq]

        mujoco.mj_kinematics(self.model, self.data)

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def get_observations(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {
            "joint_pos": self.data.qpos[: self._nq].copy(),
            "joint_vel": self.data.qvel[: self._nq].copy(),
        }
        if self._gripper_index is not None:
            obs["gripper_pos"] = self._gripper_pos.copy()
        return obs

    # ------------------------------------------------------------------ #
    # Viewer helpers
    # ------------------------------------------------------------------ #

    def is_viewer_running(self) -> bool:
        """Return True if the viewer window is still open."""
        if self.viewer is None:
            return True  # headless mode — never "closes"
        return self.viewer.is_running()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
