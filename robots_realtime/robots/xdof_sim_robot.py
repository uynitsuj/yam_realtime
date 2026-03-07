"""Robot wrapper for the xdof-sim MuJoCoYAMEnv bimanual environment.

Bridges MuJoCoYAMEnv into the robots_realtime Robot protocol so it can be
driven by the sim_mode control loop in launch.py.

The 14D action/state layout used by MuJoCoYAMEnv is:
    [left_j1..6, left_grip, right_j1..6, right_grip]

In right_arm_only mode (default), the Robot accepts 7-DOF commands from a
single GELLO leader; the left arm is held at zeros.

Visualization uses Viser (browser-based), so no mjpython is required on macOS.
Open http://localhost:8080 (or the configured port) in any browser to view.
"""

from typing import Dict, Optional

import mujoco
import numpy as np
from i2rt.robots.robot import Robot


class _ViserSceneManager:
    """Minimal live-teleoperation viser scene backed by a MuJoCo model/data pair.

    Imports scene-building helpers directly from xdof_sim.examples.viser_replay
    (no modifications to the xdof-sim package required).  Only dynamic bodies
    are updated each tick; fixed geometry is uploaded once at construction time.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        port: int = 8080,
        visible_geom_groups: tuple[int, ...] = (0, 1, 2),
    ) -> None:
        import viser
        import viser.transforms as vtf
        from mujoco import mjtGeom, mjtObj
        from mujoco import mj_id2name
        from xdof_sim.examples.viser_replay import (
            _merge_geoms,
            _is_fixed_body,
            _get_body_name,
        )

        self._model = model
        self._data = data
        self._vtf = vtf
        self._mesh_handles: dict[int, viser.MeshHandle] = {}

        self.server = viser.ViserServer(port=port)
        print(f"Viser scene viewer: http://localhost:{port}")

        # Classify visual geoms by body
        body_visual: dict[int, list[int]] = {}
        for i in range(model.ngeom):
            if int(model.geom_group[i]) in visible_geom_groups:
                body_visual.setdefault(int(model.geom_bodyid[i]), []).append(i)

        self.server.scene.configure_environment_map(environment_intensity=0.8)
        self.server.scene.add_frame("/fixed_bodies", show_axes=False)

        for body_id, visual_ids in body_visual.items():
            body_name = _get_body_name(model, body_id)

            if _is_fixed_body(model, body_id):
                # Planes → viser grid; everything else → static trimesh
                nonplane_ids = []
                for gid in visual_ids:
                    if model.geom_type[gid] == mjtGeom.mjGEOM_PLANE:
                        geom_name = mj_id2name(model, mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
                        self.server.scene.add_grid(
                            f"/fixed_bodies/{body_name}/{geom_name}",
                            width=2000.0,
                            height=2000.0,
                            infinite_grid=True,
                            fade_distance=50.0,
                            shadow_opacity=0.2,
                            position=model.geom_pos[gid],
                            wxyz=model.geom_quat[gid],
                        )
                    else:
                        nonplane_ids.append(gid)
                if nonplane_ids:
                    merged = _merge_geoms(model, nonplane_ids)
                    self.server.scene.add_mesh_trimesh(
                        f"/fixed_bodies/{body_name}",
                        merged,
                        cast_shadow=False,
                        receive_shadow=0.2,
                        position=model.body(body_id).pos,
                        wxyz=model.body(body_id).quat,
                    )
            else:
                if visual_ids:
                    merged = _merge_geoms(model, visual_ids)
                    handle = self.server.scene.add_mesh_trimesh(
                        f"/bodies/{body_name}",
                        merged,
                        visible=True,
                    )
                    self._mesh_handles[body_id] = handle

    def update(self) -> None:
        """Push current body poses from MjData to the viser scene."""
        vtf = self._vtf
        with self.server.atomic():
            for body_id, handle in self._mesh_handles.items():
                handle.position = self._data.xpos[body_id]
                xmat = self._data.xmat[body_id].reshape(3, 3)
                handle.wxyz = vtf.SO3.from_matrix(xmat).wxyz
            self.server.flush()

    def stop(self) -> None:
        self.server.stop()


class XdofSimRobot(Robot):
    """Wraps MuJoCoYAMEnv as a Robot for the sim_mode control loop.

    In right_arm_only mode (default), the robot is registered under the key
    "right" in the YAML robots dict and accepts 7-DOF commands (6 joints +
    1 gripper).  The left arm is held at zeros.

    Each call to command_joint_pos() advances physics by one control step
    (physics_dt × control_decimation seconds) via env._step_single(), which
    is the per-tick stepping path.  env.step() is intentionally avoided as
    it is designed for chunked action-sequence inference.

    Visualization uses Viser (browser-based 3D viewer) rather than the
    MuJoCo GUI, so mjpython is not required on macOS.  Open the printed URL
    in any browser after launch.

    Args:
        right_arm_only: If True, only the right arm is commanded; the left
            arm stays at zeros.  Register this robot under the "right" key.
        render: Launch the Viser web viewer.
        render_cameras: Whether to render MuJoCo camera observations each
            step.  Expensive; disable for teleoperation.
        physics_dt: MuJoCo physics timestep in seconds.
        control_decimation: Number of physics steps per control step.
            Effective control rate = 1 / (physics_dt × control_decimation).
            Default 17 × 0.002s ≈ 30 Hz; set to 10 for ~50 Hz.
        scene_variant: Optional scene variant to apply at startup.
            One of "eval", "training", "hybrid".  None leaves the default
            scene as-is.
        viser_port: Port for the Viser web server.
    """

    def __init__(
        self,
        right_arm_only: bool = True,
        render: bool = True,
        render_cameras: bool = False,
        physics_dt: float = 0.002,
        control_decimation: int = 17,
        scene_variant: Optional[str] = None,
        viser_port: int = 8080,
    ) -> None:
        from xdof_sim.config import get_i2rt_sim_config
        from xdof_sim.env import MuJoCoYAMEnv

        config = get_i2rt_sim_config()
        self._env = MuJoCoYAMEnv(
            config=config,
            render_cameras=render_cameras,
            physics_dt=physics_dt,
            control_decimation=control_decimation,
        )
        self._right_arm_only = right_arm_only
        self._per_arm_dofs = 7  # 6 arm joints + 1 gripper
        self._left_cmd = np.zeros(self._per_arm_dofs, dtype=np.float32)

        if scene_variant is not None:
            from xdof_sim.scene_variants import apply_scene_variant
            apply_scene_variant(self._env.model, scene_variant)

        self._env.reset()

        self._viser: Optional[_ViserSceneManager] = None
        if render:
            self._viser = _ViserSceneManager(
                model=self._env.model,
                data=self._env.data,
                port=viser_port,
            )
            # Push initial pose before the control loop starts
            self._viser.update()

    # ------------------------------------------------------------------ #
    # Robot protocol
    # ------------------------------------------------------------------ #

    def num_dofs(self) -> int:
        return self._per_arm_dofs if self._right_arm_only else 2 * self._per_arm_dofs

    def get_joint_pos(self) -> np.ndarray:
        state = self._env.get_obs()["state"]  # 14D: [left_7, right_7]
        if self._right_arm_only:
            return state[self._per_arm_dofs:].copy()
        return state.copy()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Step physics with the commanded joint positions.

        Args:
            joint_pos: 7-DOF array (right arm only) or 14-DOF array
                (bimanual). Values are in radians; gripper in [0, 1].
        """
        if self._right_arm_only:
            left = self._left_cmd
            right = joint_pos[: self._per_arm_dofs]
        else:
            left = joint_pos[: self._per_arm_dofs]
            right = joint_pos[self._per_arm_dofs : 2 * self._per_arm_dofs]

        action_14d = np.concatenate([left, right]).astype(np.float32)
        self._env._step_single(action_14d)

        if self._viser is not None:
            self._viser.update()

    def get_observations(self) -> Dict[str, np.ndarray]:
        state = self._env.get_obs()["state"]  # 14D
        if self._right_arm_only:
            return {"joint_pos": state[self._per_arm_dofs:].copy()}
        return {"joint_pos": state.copy()}

    # ------------------------------------------------------------------ #
    # Viewer helpers
    # ------------------------------------------------------------------ #

    def is_viewer_running(self) -> bool:
        # Viser runs as a background server — always alive until close() is called.
        return True

    def close(self) -> None:
        if self._viser is not None:
            self._viser.stop()
        self._env.close()
