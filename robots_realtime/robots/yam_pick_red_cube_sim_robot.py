"""Robot wrapper for YamEnvPickRedCube bimanual MuJoCo environment.

Bridges the dm_env-based YamEnvPickRedCube into the robots_realtime
Robot protocol so it can be driven by the sim_mode control loop.

Prerequisites:
  - submodules/menagerie must exist with the robot XML files:
      submodules/menagerie/yam_station/station_with_gate.xml
      submodules/menagerie/i2rt_yam/yam.xml
  - On macOS, launch via mjpython for the passive viewer:
      DYLD_LIBRARY_PATH=... .venv/bin/mjpython robots_realtime/envs/launch.py ...
"""

from typing import Dict, Optional

import numpy as np
from i2rt.robots.robot import Robot


class YamPickRedCubeSimRobot(Robot):
    """Wraps YamEnvPickRedCube as a Robot for the sim_mode control loop.

    In right_arm_only mode (default), the robot is registered under the
    key "right" in the YAML robots dict and accepts 7-DOF commands
    (6 joints + 1 gripper) from a single GELLO leader. The left arm is
    held at its zero position.

    Args:
        station_key: Key into STATION_ROBOT_MAP, e.g. "SIM_YAM".
        right_arm_only: If True, only the right arm is commanded; the left
            arm stays at zero. Register this robot under the "right" key.
        render: Launch the passive MuJoCo viewer.
        randomize_scene: Randomize cube and joint start positions on reset.
        control_dt: Control timestep in seconds.
        camera_obs: Whether to render camera observations each step
            (expensive; disable for teleoperation).
    """

    def __init__(
        self,
        station_key: str = "SIM_YAM",
        right_arm_only: bool = True,
        render: bool = True,
        randomize_scene: bool = False,
        control_dt: float = 0.02,
        camera_obs: bool = False,
    ) -> None:
        from robots_realtime.mujoco.envs.schema.robot import STATION_ROBOT_MAP
        from robots_realtime.mujoco.envs.yam_env import YamEnvPickRedCube

        station_spec = STATION_ROBOT_MAP[station_key]
        self._env = YamEnvPickRedCube(
            station_spec_config=station_spec,
            control_dt=control_dt,
            randomize_scene=randomize_scene,
            camera_obs=camera_obs,
        )
        self._right_arm_only = right_arm_only
        arm_dofs = station_spec.robot.num_joint_dofs  # 6
        gripper_dofs = station_spec.robot.num_gripper_dofs  # 1
        self._per_arm_dofs = arm_dofs + gripper_dofs  # 7

        # Left arm neutral command (zeros = default pose from model)
        self._left_cmd = np.zeros(self._per_arm_dofs, dtype=np.float32)

        self._env.reset()

        if render:
            self._env.launch_viewer()

    # ------------------------------------------------------------------ #
    # Robot protocol
    # ------------------------------------------------------------------ #

    def num_dofs(self) -> int:
        if self._right_arm_only:
            return self._per_arm_dofs
        return 2 * self._per_arm_dofs

    def get_joint_pos(self) -> np.ndarray:
        obs = self._env.get_obs()
        right = obs["right"]
        right_pos = np.concatenate([right["joint_pos"], right["gripper_pos"]]).astype(np.float32)
        if self._right_arm_only:
            return right_pos
        left = obs["left"]
        left_pos = np.concatenate([left["joint_pos"], left["gripper_pos"]]).astype(np.float32)
        return np.concatenate([left_pos, right_pos])

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Step the bimanual environment with the commanded joint positions.

        In right_arm_only mode, joint_pos is 7-DOF for the right arm.
        In bimanual mode, joint_pos is 14-DOF (left 7 + right 7).
        """
        if self._right_arm_only:
            left = self._left_cmd
            right = joint_pos[: self._per_arm_dofs]
        else:
            left = joint_pos[: self._per_arm_dofs]
            right = joint_pos[self._per_arm_dofs : 2 * self._per_arm_dofs]

        action_dict: Dict = {
            "left": {"pos": left.astype(np.float32)},
            "right": {"pos": right.astype(np.float32)},
        }
        self._env.step(action_dict)

        if self._env._viewer is not None:
            self._env.viewer_sync()

    def get_observations(self) -> Dict[str, np.ndarray]:
        obs = self._env.get_obs()
        right = obs["right"]
        right_pos = np.concatenate([right["joint_pos"], right["gripper_pos"]]).astype(np.float32)
        if self._right_arm_only:
            return {
                "joint_pos": right_pos,
                "joint_vel": right["joint_vel"].astype(np.float32),
            }
        left = obs["left"]
        left_pos = np.concatenate([left["joint_pos"], left["gripper_pos"]]).astype(np.float32)
        return {
            "joint_pos": np.concatenate([left_pos, right_pos]),
            "joint_vel": np.concatenate([left["joint_vel"], right["joint_vel"]]).astype(np.float32),
        }

    # ------------------------------------------------------------------ #
    # Viewer helpers
    # ------------------------------------------------------------------ #

    def is_viewer_running(self) -> bool:
        viewer = self._env._viewer
        if viewer is None:
            return True  # headless — never "closes"
        return viewer.is_running()

    def close(self) -> None:
        self._env.close_viewer()
