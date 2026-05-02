"""MuJoCo simulation environment for the bimanual YAM robot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from robots_realtime.sim.config import SimConfig, default_sim_config

_MODELS_DIR = Path(__file__).parent / "models"
_SCENE_XMLS = {
    "bottles":    _MODELS_DIR / "yam_bimanual_scene.xml",
    "robot_only": _MODELS_DIR / "yam_bimanual_robot_only.xml",
}
_DEFAULT_SCENE = "bottles"

# Gripper actuator ctrl range max (from menagerie model)
_GRIPPER_CTRL_MAX = 0.0475


class MuJoCoYAMEnv(gym.Env):
    """MuJoCo-based simulation of the bimanual YAM robot.

    Actions and observations use the same 14D format as the real robot:
    [left_j1..6, left_grip, right_j1..6, right_grip].

    This is a minimal, self-contained re-implementation that does not depend on
    the private xdof-sim package.
    """

    def __init__(
        self,
        config: SimConfig | None = None,
        scene: str = _DEFAULT_SCENE,
        scene_xml: str | Path | None = None,
        chunk_dim: int = 30,
        prompt: str = "",
        render_cameras: bool = True,
        camera_height: int = 480,
        camera_width: int = 640,
        physics_dt: float = 0.002,
        control_decimation: int = 17,  # 0.002 * 17 ≈ 34 ms per control step ≈ 30 Hz
    ):
        super().__init__()
        self.config = config or default_sim_config()
        self.chunk_dim = chunk_dim
        self.prompt = prompt
        self._render_cameras_flag = render_cameras
        self._camera_height = camera_height
        self._camera_width = camera_width
        self._physics_dt = physics_dt
        self._control_decimation = control_decimation

        if scene_xml is not None:
            self._scene_xml = Path(scene_xml)
        elif scene in _SCENE_XMLS:
            self._scene_xml = _SCENE_XMLS[scene]
        else:
            available = list(_SCENE_XMLS.keys())
            raise ValueError(f"Unknown scene '{scene}'. Available: {available}")

        self.camera_names = list(self.config.cameras.keys())
        self.robot_names = list(self.config.robots.keys())
        self.single_timestep_action_dim = 7 * len(self.config.robots)  # 14 for bimanual
        self.state_dim = self.single_timestep_action_dim  # alias

        self.observation_space = spaces.Dict(
            {
                "images": spaces.Dict(
                    {
                        name: spaces.Box(
                            low=0,
                            high=255,
                            shape=(
                                self.config.cameras[name].height,
                                self.config.cameras[name].width,
                                3,
                            ),
                            dtype=np.uint8,
                        )
                        for name in self.camera_names
                    }
                ),
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.state_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.chunk_dim, self.state_dim),
            dtype=np.float32,
        )

        self._setup_model()
        self.cur_step = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_model(self):
        self.model = mujoco.MjModel.from_xml_path(str(self._scene_xml))
        self.model.opt.timestep = self._physics_dt
        self.data = mujoco.MjData(self.model)

        if self._render_cameras_flag:
            self.renderer = mujoco.Renderer(
                self.model,
                height=self._camera_height,
                width=self._camera_width,
            )

        self._build_index_maps()

    def _build_index_maps(self):
        """Map 14D state/action indices → MuJoCo qpos/ctrl indices."""
        self._qpos_indices: list[int] = []
        self._ctrl_indices: list[int] = []
        self._gripper_set: set[int] = set()

        idx = 0
        for robot_name in self.robot_names:
            for j in range(1, 7):
                jnt_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot_name}_joint{j}"
                )
                self._qpos_indices.append(self.model.jnt_qposadr[jnt_id])
                act_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot_name}_joint{j}"
                )
                self._ctrl_indices.append(act_id)
                idx += 1

            finger_jnt_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot_name}_left_finger"
            )
            self._qpos_indices.append(self.model.jnt_qposadr[finger_jnt_id])
            grip_act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot_name}_gripper"
            )
            self._ctrl_indices.append(grip_act_id)
            self._gripper_set.add(idx)
            idx += 1

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._set_qpos(self.get_init_q())
        mujoco.mj_forward(self.model, self.data)
        self.cur_step = 0
        return self.get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(
            self.chunk_dim, self.state_dim
        )
        all_obs = []
        for i in range(self.chunk_dim):
            self.step_single(action[i])
            all_obs.append(self.get_obs())

        final_obs = all_obs[-1]
        chunk_history = _stack_obs(all_obs)
        self.cur_step += 1
        return final_obs, chunk_history, 0.0, False, False, {}

    def close(self):
        if hasattr(self, "renderer"):
            self.renderer.close()

    # ------------------------------------------------------------------
    # Core stepping (also used directly by SimBackend)
    # ------------------------------------------------------------------

    def _step_single(self, action_14d: np.ndarray) -> None:
        """Apply a single 14D action and advance physics by control_decimation steps."""
        ctrl = np.zeros(self.model.nu)
        for i in range(self.single_timestep_action_dim):
            val = float(action_14d[i])
            if i in self._gripper_set:
                val = val * _GRIPPER_CTRL_MAX
            ctrl[self._ctrl_indices[i]] = val
        self.data.ctrl[:] = ctrl
        for _ in range(self._control_decimation):
            mujoco.mj_step(self.model, self.data)

    def step_single(self, action_14d: np.ndarray) -> None:
        """Alias for _step_single (backwards compatibility)."""
        self._step_single(action_14d)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def get_obs(self) -> dict[str, Any]:
        state = np.zeros(self.state_dim, dtype=np.float32)
        for i, qpos_idx in enumerate(self._qpos_indices):
            val = float(self.data.qpos[qpos_idx])
            if i in self._gripper_set:
                val = float(np.clip(val / _GRIPPER_CTRL_MAX, 0.0, 1.0))
            state[i] = val

        if self._render_cameras_flag:
            images = self._render_cameras()
        else:
            images = {
                name: np.zeros(
                    (self._camera_height, self._camera_width, 3), dtype=np.uint8
                )
                for name in self.camera_names
            }

        sim_time = float(self.data.time)
        return {
            "images": images,
            "state": state,
            "prompt": self.prompt,
            "camera_timestamps": {name: sim_time for name in self.camera_names},
            "masks": {name: True for name in self.camera_names},
        }

    def _render_cameras(self) -> dict[str, np.ndarray]:
        """Render all cameras → (H, W, 3) uint8."""
        images = {}
        for name in self.camera_names:
            self.renderer.update_scene(self.data, camera=name)
            images[name] = self.renderer.render().copy()
        return images

    def get_init_q(self) -> np.ndarray:
        return np.concatenate(
            [np.array(self.config.robots[name].init_q) for name in self.robot_names]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_qpos_from_state(self, state: np.ndarray) -> None:
        for i, qpos_idx in enumerate(self._qpos_indices):
            val = float(state[i])
            if i in self._gripper_set:
                val = val * _GRIPPER_CTRL_MAX
            self.data.qpos[qpos_idx] = val

    def _set_qpos(self, state: np.ndarray) -> None:
        """Alias for _set_qpos_from_state (backwards compatibility)."""
        self._set_qpos_from_state(state)


def _stack_obs(obses: list[dict]) -> dict:
    stacked: dict[str, Any] = {}
    for key in obses[0]:
        if isinstance(obses[0][key], str):
            stacked[key] = obses[0][key]
        elif isinstance(obses[0][key], dict):
            stacked[key] = {
                k: np.stack([obs[key][k] for obs in obses])
                for k in obses[0][key]
            }
        else:
            stacked[key] = np.stack([obs[key] for obs in obses])
    return stacked
