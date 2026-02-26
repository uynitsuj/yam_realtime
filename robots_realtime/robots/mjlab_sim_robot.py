"""MJLab physics-simulation robot for teleoperation.

Wraps a mjlab ``ManagerBasedRlEnv`` and exposes it as a ``Robot``-compatible
interface so the existing ``sim_mode`` launch path in ``launch.py`` can drive
a full physics simulation (with objects, contacts, gravity) instead of the
bare-kinematics ``MujocoSimRobot``.

Key design choices
------------------
* The mjlab env is loaded in *play* mode (infinite episodes, no curriculum,
  no observation corruption).
* Action-space scaling is overridden to ``scale=1.0`` /
  ``use_default_offset=False`` so the leader's joint positions (in radians)
  are passed **directly** as joint-position targets to mjlab's PD controller
  layer - no manual un-scaling required.
* A MuJoCo passive viewer is created by accessing the underlying
  ``sim.mj_model`` / ``sim.mj_data`` from the Warp simulation and copying
  Warp state back to CPU MuJoCo data after each physics step (the same
  mechanism used internally by mjlab's ``NativeMujocoViewer``).
* On macOS the passive viewer requires ``mjpython``; run headless
  (``render: false``) or use ``mjpython`` as documented in the YAML config.
"""

from typing import Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np
from i2rt.robots.robot import Robot


class MjlabSimRobot(Robot):
    """A full-physics YAM simulation backed by a mjlab ``ManagerBasedRlEnv``.

    The environment is loaded in *play* mode and the action processing is
    configured for direct joint-position teleoperation (scale = 1, no default
    offset).  A MuJoCo passive viewer can optionally be launched by reading
    the underlying Warp simulation state after each step.

    Args:
        task_id: mjlab task identifier, e.g. ``"Mjlab-Lift-Cube-Yam"``.
        num_envs: Number of parallel environments.  Use ``1`` for
            teleoperation.
        device: PyTorch / Warp device string, e.g. ``"cpu"`` or ``"cuda:0"``.
        render: Whether to launch a passive MuJoCo viewer window.  On macOS
            this requires running under ``mjpython``; set to ``False`` for
            headless operation.
        num_arm_dofs: Number of arm joints (6 for YAM).
        include_gripper: Whether the command includes a 7th gripper DOF.
        disable_terminations: Clear all episode termination conditions
            (default True).  Teleoperation should never be cut short by
            collision / ground-contact checks that exist for RL training.
        decimation: Override the number of physics substeps per env step.
            Lower values (e.g. 1 or 2) reduce CPU cost at the expense of
            simulation fidelity.  ``None`` keeps the task default (4 for
            the YAM lift-cube task).
    """

    def __init__(
        self,
        task_id: str = "Mjlab-Lift-Cube-Yam",
        num_envs: int = 1,
        device: str = "cpu",
        render: bool = True,
        num_arm_dofs: int = 6,
        include_gripper: bool = True,
        disable_terminations: bool = True,
        decimation: Optional[int] = None,
    ) -> None:
        # Lazy-import mjlab so the rest of robots_realtime works without it.
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.envs.mdp.actions import JointPositionActionCfg
        from mjlab.tasks.registry import load_env_cfg

        self._torch = torch
        self._device = device
        self._num_arm_dofs = num_arm_dofs
        self._include_gripper = include_gripper
        self._num_dofs = num_arm_dofs + (1 if include_gripper else 0)
        self._passive_viewer: Optional[mujoco.viewer.Handle] = None
        self._mj_model: Optional[mujoco.MjModel] = None
        self._mj_data: Optional[mujoco.MjData] = None

        # Load play-mode env config: infinite episodes, no curriculum,
        # no observation corruption.
        env_cfg = load_env_cfg(task_id, play=True)
        env_cfg.scene.num_envs = num_envs

        # Disable all termination conditions for teleoperation — the arm would
        # otherwise reset whenever it triggers a collision/ground contact check
        # (e.g. on startup before reaching the leader's position).
        if disable_terminations:
            env_cfg.terminations = {}

        # Prevent commands from periodically teleporting the cube.
        # LiftingCommand._resample_command() physically moves the cube to a
        # new random position on every resample — every 4 s in play mode.
        # Push the timer to infinity so the scene stays static during teleop.
        for cmd_cfg in env_cfg.commands.values():
            cmd_cfg.resampling_time_range = (int(1e9), int(1e9))

        # Optionally reduce physics substeps per env step for lower CPU cost.
        if decimation is not None:
            env_cfg.decimation = decimation

        # Override action processing for direct joint-position teleoperation.
        # Default mjlab configs use per-joint scale (YAM_ACTION_SCALE) and a
        # default-position offset tuned for RL policy outputs.  For
        # teleoperation we pass absolute joint angles in radians, so we set
        # scale=1 and disable the default offset.
        for action_cfg in env_cfg.actions.values():
            if isinstance(action_cfg, JointPositionActionCfg):
                action_cfg.scale = 1.0
                action_cfg.use_default_offset = False

        # Create the env without offscreen rendering (we handle it below).
        self.env = ManagerBasedRlEnv(env_cfg, device=device, render_mode=None)

        # Set up passive viewer using the underlying MuJoCo model/data.
        if render:
            self._setup_viewer()

        # Reset the env and initialise cached state.
        self.env.reset()
        self._last_joint_pos = np.zeros(self._num_dofs)

    # ------------------------------------------------------------------ #
    # Viewer helpers
    # ------------------------------------------------------------------ #

    def _setup_viewer(self) -> None:
        """Launch a passive MuJoCo viewer attached to the Warp sim model."""
        # Access the underlying Warp simulation through the unwrapped env.
        env_inner = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        sim = env_inner.sim
        self._mj_model = sim.mj_model
        self._mj_data = sim.mj_data

        self._passive_viewer = mujoco.viewer.launch_passive(
            self._mj_model,
            self._mj_data,
            show_left_ui=False,
            show_right_ui=False,
        )
        mujoco.mjv_defaultFreeCamera(self._mj_model, self._passive_viewer.cam)

    def _sync_viewer(self) -> None:
        """Copy Warp state → MuJoCo CPU data → passive viewer.

        Mirrors what ``NativeMujocoViewer.sync_env_to_viewer()`` does so the
        passive viewer reflects the true Warp physics state.
        """
        if self._passive_viewer is None or not self._passive_viewer.is_running():
            return
        env_inner = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        sim = env_inner.sim
        sim_data = sim.data
        if self._mj_model is not None and self._mj_model.nq > 0:
            self._mj_data.qpos[:] = sim_data.qpos[0].cpu().numpy()
            self._mj_data.qvel[:] = sim_data.qvel[0].cpu().numpy()
        if self._mj_model is not None and self._mj_model.nmocap > 0:
            self._mj_data.mocap_pos[:] = sim_data.mocap_pos[0].cpu().numpy()
            self._mj_data.mocap_quat[:] = sim_data.mocap_quat[0].cpu().numpy()
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self._passive_viewer.sync()

    # ------------------------------------------------------------------ #
    # Robot protocol
    # ------------------------------------------------------------------ #

    def num_dofs(self) -> int:
        return self._num_dofs

    def get_joint_pos(self) -> np.ndarray:
        return self._last_joint_pos.copy()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Send a joint-position target to the mjlab physics simulation.

        The action is assembled from the first ``num_arm_dofs`` elements of
        ``joint_pos`` plus an optional gripper value, converted to a
        ``[1, num_dofs]`` float32 tensor, then passed to ``env.step()``.

        Because we set ``scale=1.0`` and ``use_default_offset=False`` in the
        action config, the tensor values are used **directly** as joint
        position targets (radians) by the PD actuator layer in MuJoCo.

        Args:
            joint_pos: Joint positions in radians.  May be length
                ``num_arm_dofs`` (arm only) or ``num_arm_dofs + 1``
                (arm + gripper).
        """
        arm_pos = joint_pos[: self._num_arm_dofs]
        if self._include_gripper and len(joint_pos) > self._num_arm_dofs:
            gripper = joint_pos[self._num_arm_dofs : self._num_arm_dofs + 1]
            full = np.concatenate([arm_pos, gripper])
        else:
            full = arm_pos

        action = self._torch.tensor(
            full, dtype=self._torch.float32, device=self._device
        ).unsqueeze(0)  # [1, num_dofs]

        _obs, _rewards, terminated, truncated, _extras = self.env.step(action)

        # Cache last commanded position as the observable joint state.
        self._last_joint_pos[: len(full)] = full

        # Sync viewer with the post-step Warp state.
        self._sync_viewer()

        # Auto-reset on episode termination so the scene stays alive.
        if terminated[0] or truncated[0]:
            self.env.reset()

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {"joint_pos": self._last_joint_pos.copy()}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_viewer_running(self) -> bool:
        """Return False when the passive viewer window has been closed."""
        if self._passive_viewer is None:
            return True  # headless — never "closes"
        return bool(self._passive_viewer.is_running())

    def close(self) -> None:
        if self._passive_viewer is not None:
            self._passive_viewer.close()
        if hasattr(self.env, "close"):
            self.env.close()
