"""Viser + PyRoKi teleoperation agent for one or two YAM arms.

Drag the end-effector gizmos in the browser; PyRoKi solves IK at ``ik_rate`` Hz
on a background thread and :class:`~robots_realtime.runtime.viser_teleop_node.ViserTeleopNode`
publishes the resulting joint targets to the ``RobotNode``s.

Scene contents
--------------
* Opaque URDF  — the IK solution (where the arm is being *commanded* to go).
* Ghost URDF   — the measured joint state fed back from ``{arm}/joint_state``.
* Gizmo        — draggable ``TransformControls`` per arm, offset to the gripper
                 TCP (not link_6) so the gizmo sits where the fingers are.
* Camera panels — one per entry in the node's ``image_topics``.

Conventions this agent bridges
------------------------------
* **Joint order.** The YAM URDF declares ``joint6 … joint1`` (reversed), so the
  IK solution comes back in URDF order and must be ``np.flip``-ed to reach the
  i2rt motor order used on the bus. Same flip applies to the measured state
  going the other way.
* **Gripper.** ``MotorChainRobot``'s ``JointMapper`` expects a *normalized*
  ``[0, 1]`` gripper command (0 = closed, 1 = open) as the 7th element, which
  it remaps onto the follower's calibrated ``gripper_limits``. The sliders are
  therefore normalized, not radians.

Command smoothing
-----------------
Viser delivers gizmo drags at the browser's event rate (~20 Hz measured), so the
raw IK output is a staircase: the same joint vector repeats for ~5 ticks at
100 Hz and then jumps by 80-450 mrad at once. A position-controlled follower
turns each of those jumps into a lunge, which is what "steppy" motion is.

``act()`` therefore publishes a *slew-limited* command that chases the IK
solution at no more than ``max_joint_speed`` rad/s, with an optional
first-order filter (``smoothing_tau_s``) to round the corners. Every tick then
carries a fresh, bounded increment instead of an occasional large jump. Both
knobs are live sliders in the GUI, so they can be tuned against the arm.

Startup handoff
---------------
The gizmos would otherwise spawn at a canned pose and yank the arm there the
instant commands are unpaused. ``sync_on_start`` snaps each gizmo onto the
FK of the *measured* joint state as soon as the first observation arrives, so
the first command equals where the arm already is. The "Sync gizmos to robot"
button re-snaps at any time (useful after an e-stop or a manual re-pose).
Pair with ``session.start_paused: true`` and ``RobotNode.ramp_duration_s``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import viser
import viser.extras
import viser.transforms as vtf
from dm_env.specs import Array

from robots_realtime.agents.agent import Agent
from robots_realtime.robots.inverse_kinematics.yam_pyroki import YamPyroki
from robots_realtime.sensors.cameras.camera_utils import obs_get_rgb, resize_with_pad

# Arm DOF (excludes the gripper, which rides along as the 7th command element).
ARM_DOF = 6
# Link the IK targets and gizmos are attached to.
EE_LINK = "link_6"
BASE_LINK = "base_link"


class YamPyrokiViserAgent(Agent):
    """Interactive Viser IK teleoperation for single-arm or bimanual YAM.

    Args:
        bimanual:            Drive both arms (requires ``right_arm_extrinsic``).
        right_arm_extrinsic: ``{"position": [x, y, z], "rotation": [w, x, y, z]}``
                             of the right base relative to the left base.
        ik_rate:             PyRoKi solve rate (Hz) on the background thread.
        viser_port:          Port for this agent's Viser server. Must differ
                             from any ``ViserMonitorNode`` port in the session.
        gripper_initial:     Initial normalized gripper command (1.0 = open).
        sync_on_start:       Snap the gizmos onto the measured arm pose when the
                             first observation arrives (avoids a startup jump).
        max_joint_speed:     Cap on commanded joint speed (rad/s). The command
                             chases the IK solution at this rate instead of
                             teleporting to it. 0 disables the limiter.
        max_gripper_speed:   Same, for the normalized gripper (units/s).
        smoothing_tau_s:     First-order filter on the slew-limited command
                             (seconds). 0 disables it.
        viz_period_s:        Ghost/camera refresh period.
        preview_size:        Side length of the GUI camera previews.
    """

    def __init__(
        self,
        bimanual: bool = False,
        right_arm_extrinsic: Optional[Dict[str, Any]] = None,
        ik_rate: float = 100.0,
        viser_port: int = 8080,
        gripper_initial: float = 1.0,
        sync_on_start: bool = True,
        max_joint_speed: float = 2.0,
        max_gripper_speed: float = 4.0,
        smoothing_tau_s: float = 0.03,
        viz_period_s: float = 0.05,
        preview_size: int = 224,
        viser_server: Optional["viser.ViserServer"] = None,
        armed: bool = True,
    ) -> None:
        self.bimanual = bimanual
        self.right_arm_extrinsic = right_arm_extrinsic
        if bimanual:
            assert right_arm_extrinsic is not None, "right_arm_extrinsic must be provided for bimanual robot"

        self.arms = ["left", "right"] if bimanual else ["left"]
        self._gripper_initial = float(np.clip(gripper_initial, 0.0, 1.0))
        self._sync_on_start = sync_on_start
        self._sync_pending = sync_on_start
        self._viz_period = viz_period_s
        self._preview_size = preview_size
        self._max_gripper_speed = float(max_gripper_speed)
        self._smoothing_tau_s = float(smoothing_tau_s)
        # Slew-limited command state: last vector published, per arm.
        self._cmd: Dict[str, np.ndarray] = {}
        self._goal_filtered: Dict[str, np.ndarray] = {}
        self._last_act_t: Optional[float] = None

        self._max_joint_speed_init = float(max_joint_speed)

        # An injected server lets this agent share ONE viser page with another
        # node's scene (e.g. ViserMonitorNode's URDF + camera panels) instead of
        # serving a second port. viser handles are namespaced by path, so the
        # two scenes coexist.
        self.viser_server = viser_server if viser_server is not None else viser.ViserServer(port=viser_port)
        # Safety gate: while disarmed act() holds the measured pose, so entering
        # teleop never moves the arm until the operator explicitly arms it.
        self._armed = bool(armed)
        self.ik = YamPyroki(rate=ik_rate, viser_server=self.viser_server, bimanual=bimanual)

        # Private URDF instance for FK during gizmo syncing — mutating the one
        # the IK loop renders from would fight the visualization.
        self._fk_urdf = self.ik.load_urdf()

        self.obs: Optional[Dict[str, Any]] = None
        self._running = True

        self._setup_visualization()

        self.ik_thread = threading.Thread(target=self.ik.run, name="yam_pyroki_ik", daemon=True)
        self.ik_thread.start()
        self.real_vis_thread = threading.Thread(target=self._update_visualization, name="yam_real_vis", daemon=True)
        self.real_vis_thread.start()

    # ── Scene setup ────────────────────────────────────────────────────────────

    def _setup_visualization(self) -> None:
        # Ghost overlay of the *measured* arm, drawn under the IK solution.
        self.base_frame_left_real = self.viser_server.scene.add_frame("/base_left_real", show_axes=False)
        self.urdf_vis_left_real = viser.extras.ViserUrdf(
            self.viser_server,
            self.ik.load_urdf(),
            root_node_name="/base_left_real",
            mesh_color_override=(0.8, 0.5, 0.5),
        )
        for mesh in self.urdf_vis_left_real._meshes:
            mesh.opacity = 0.25  # type: ignore[attr-defined]

        self.gripper_sliders: Dict[str, viser.GuiInputHandle] = {
            "left": self.viser_server.gui.add_slider(
                "Left gripper (0=closed, 1=open)",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=self._gripper_initial,
            )
        }

        if self.bimanual and self.right_arm_extrinsic is not None:
            self.ik.base_frame_right.position = np.array(self.right_arm_extrinsic["position"])
            self.ik.base_frame_right.wxyz = np.array(self.right_arm_extrinsic["rotation"])
            self.base_frame_right_real = self.viser_server.scene.add_frame(
                "/base_left_real/base_right_real", show_axes=False
            )
            self.base_frame_right_real.position = self.ik.base_frame_right.position
            self.base_frame_right_real.wxyz = self.ik.base_frame_right.wxyz
            self.urdf_vis_right_real = viser.extras.ViserUrdf(
                self.viser_server,
                self.ik.load_urdf(),
                root_node_name="/base_left_real/base_right_real",
                mesh_color_override=(0.8, 0.5, 0.5),
            )
            for mesh in self.urdf_vis_right_real._meshes:
                mesh.opacity = 0.25  # type: ignore[attr-defined]
            self.gripper_sliders["right"] = self.viser_server.gui.add_slider(
                "Right gripper (0=closed, 1=open)",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=self._gripper_initial,
            )

        self.speed_slider = self.viser_server.gui.add_slider(
            "Max joint speed (rad/s)", min=0.0, max=6.0, step=0.1, initial_value=self._max_joint_speed_init
        )
        self.smoothing_slider = self.viser_server.gui.add_slider(
            "Smoothing tau (s)", min=0.0, max=0.2, step=0.005, initial_value=self._smoothing_tau_s
        )

        self.sync_button = self.viser_server.gui.add_button("Sync gizmos to robot")

        @self.sync_button.on_click
        def _(_) -> None:
            self._sync_gizmos_to_state()

        # The base class creates this button but leaves it unwired.
        if hasattr(self.ik, "reset_button"):

            @self.ik.reset_button.on_click
            def _(_) -> None:
                self.ik.home()

        self.viser_cam_img_handles: Dict[str, viser.GuiImageHandle] = {}

    # ── Gizmo ↔ robot sync ─────────────────────────────────────────────────────

    def _measured_joints(self, arm: str) -> Optional[np.ndarray]:
        """Measured arm joints (motor order, gripper stripped) for ``arm``."""
        obs = self.obs
        if not isinstance(obs, dict):
            return None
        arm_obs = obs.get(arm)
        if not isinstance(arm_obs, dict):
            return None
        joint_pos = arm_obs.get("joint_pos")
        if joint_pos is None:
            return None
        joint_pos = np.asarray(joint_pos, dtype=np.float64).ravel()
        if joint_pos.size < ARM_DOF:
            return None
        return joint_pos[:ARM_DOF]

    def _fk_ee_pose(self, joints_motor_order: np.ndarray) -> vtf.SE3:
        """FK of the measured joints → link_6 pose in that arm's base frame."""
        self._fk_urdf.update_cfg(np.flip(joints_motor_order))
        return vtf.SE3.from_matrix(np.asarray(self._fk_urdf.get_transform(EE_LINK, BASE_LINK)))

    def _sync_gizmos_to_state(self) -> bool:
        """Snap every gizmo onto the measured pose of its arm.

        Returns True only if every arm was synced (i.e. state had arrived).
        """
        synced_all = True
        for arm in self.arms:
            joints = self._measured_joints(arm)
            handle = self.ik.transform_handles.get(arm)
            if joints is None or handle is None or handle.control is None:
                synced_all = False
                continue
            # get_target_poses() builds the IK target as control @ tcp_offset,
            # so inverting the offset puts the *target* exactly on measured FK.
            tcp_offset = vtf.SE3(
                np.array([*handle.tcp_offset_frame.wxyz, *handle.tcp_offset_frame.position])
            )
            control_tf = self._fk_ee_pose(joints) @ tcp_offset.inverse()
            handle.control.position = np.asarray(control_tf.translation(), dtype=np.float64)
            handle.control.wxyz = np.asarray(control_tf.rotation().wxyz, dtype=np.float64)
            # Restart the slew from where the arm actually is, so the first
            # post-sync command doesn't chase a stale setpoint.
            self._cmd.pop(arm, None)
            self._goal_filtered.pop(arm, None)
        return synced_all

    # ── Visualization loop ─────────────────────────────────────────────────────

    def _update_visualization(self) -> None:
        while self._running and self.obs is None:
            time.sleep(0.025)

        while self._running:
            obs = self.obs
            if obs is None:
                time.sleep(self._viz_period)
                continue

            # One-shot startup sync, as soon as real joint state is available.
            if self._sync_pending and self._sync_gizmos_to_state():
                self._sync_pending = False

            left_joints = self._measured_joints("left")
            if left_joints is not None:
                self.urdf_vis_left_real.update_cfg(np.flip(left_joints))
            if self.bimanual:
                right_joints = self._measured_joints("right")
                if right_joints is not None:
                    self.urdf_vis_right_real.update_cfg(np.flip(right_joints))

            rgb_images = obs_get_rgb(obs)
            if rgb_images:
                for key, image in rgb_images.items():
                    preview = resize_with_pad(image, self._preview_size, self._preview_size)
                    if key not in self.viser_cam_img_handles:
                        self.viser_cam_img_handles[key] = self.viser_server.gui.add_image(preview, label=key)
                    else:
                        self.viser_cam_img_handles[key].image = preview

            time.sleep(self._viz_period)

    # ── Agent interface ────────────────────────────────────────────────────────

    def set_armed(self, armed: bool) -> None:
        """Arm/disarm command output. Disarming also re-syncs the gizmos, so
        re-arming starts from wherever the arm actually is rather than from a
        setpoint the operator may have dragged while disarmed."""
        armed = bool(armed)
        if armed and not self._armed:
            self._sync_pending = True
        self._armed = armed

    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        self.obs = obs

        # Hold position until the startup sync has landed, otherwise the first
        # commands would drive toward the gizmos' canned spawn pose. Disarmed
        # behaves identically: hold measured, publish nothing new.
        if self._sync_pending or not self._armed:
            action = {}
            for arm in self.arms:
                joints = self._measured_joints(arm)
                if joints is None:
                    return {}
                held = np.concatenate([joints, [self.gripper_sliders[arm].value]])
                self._cmd[arm] = held
                self._goal_filtered.pop(arm, None)
                action[arm] = {"pos": held.astype(np.float32)}
            return action

        now = time.monotonic()
        # Clamp dt so a stalled tick can't authorize an unbounded jump.
        dt = 0.01 if self._last_act_t is None else float(np.clip(now - self._last_act_t, 1e-4, 0.05))
        self._last_act_t = now

        action: Dict[str, Dict[str, np.ndarray]] = {}
        for arm in self.arms:
            joints = self.ik.joints.get(arm)
            if joints is None:
                continue
            # IK solves in URDF order (joint6 … joint1) → flip to motor order.
            goal = np.concatenate([np.flip(np.asarray(joints)), [self.gripper_sliders[arm].value]])
            action[arm] = {"pos": self._slew(arm, goal, dt).astype(np.float32)}
        return action

    def _slew(self, arm: str, goal: np.ndarray, dt: float) -> np.ndarray:
        """Advance this arm's command toward *goal* under the speed limit."""
        cur = self._cmd.get(arm)
        if cur is None:
            # Seed from the measured pose when available so the first command
            # is a no-op rather than a jump to wherever the IK happens to be.
            meas = self._measured_joints(arm)
            cur = goal.copy() if meas is None else np.concatenate([meas, goal[-1:]])

        # Filter the *goal*, then rate-limit the approach to it. Doing it in
        # this order keeps the two knobs independent: filtering the output
        # instead would scale each clamped increment by alpha, silently
        # dividing the speed limit by ~tau/dt.
        tau = float(self.smoothing_slider.value)
        smoothed = self._goal_filtered.get(arm)
        if smoothed is None or tau <= 0.0:
            smoothed = goal.copy()
        else:
            smoothed = smoothed + (dt / (tau + dt)) * (goal - smoothed)
        self._goal_filtered[arm] = smoothed

        v_max = float(self.speed_slider.value)
        g_max = self._max_gripper_speed
        if v_max <= 0.0:
            cur = smoothed.copy()
        else:
            limit = np.full(goal.shape, v_max * dt)
            limit[-1] = g_max * dt if g_max > 0 else np.inf
            cur = cur + np.clip(smoothed - cur, -limit, limit)

        self._cmd[arm] = cur
        return cur

    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        """Define the action specification."""
        return {arm: {"pos": Array(shape=(ARM_DOF + 1,), dtype=np.float32)} for arm in self.arms}

    def close(self) -> None:
        self._running = False
        if hasattr(self.ik, "running"):
            self.ik.running = False
        try:
            self.viser_server.stop()
        except Exception:
            pass


__all__ = ["YamPyrokiViserAgent"]
