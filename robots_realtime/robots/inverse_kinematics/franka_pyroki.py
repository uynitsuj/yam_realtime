"""Franka Panda inverse kinematics utilities backed by PyRoKi and Viser."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Literal, Optional

import numpy as np
import pyroki as pk
import viser.extras
import viser.transforms as vtf

from robots_realtime.robots.inverse_kinematics.pyroki_snippets._solve_ik import solve_ik
from robots_realtime.robots.inverse_kinematics.pyroki_snippets._solve_ik_vel_cost import solve_ik as solve_ik_vel_cost
from robots_realtime.robots.viser.viser_base import ViserAbstractBase


class FrankaPyroki(ViserAbstractBase):
    """Interactive Franka Panda inverse kinematics helper.

    This class mirrors :class:`yam_realtime.robots.inverse_kinematics.yam_pyroki.YamPyroki`
    but is specialised for a single (or optionally dual) Franka Panda arm described in
    ``robot_descriptions``. It exposes Viser transform gizmos for the desired end-effector
    target(s), solves IK with PyRoKi, and keeps track of the resulting joint targets.
    """

    DEFAULT_ROBOT_DESCRIPTION = "panda_description"
    DEFAULT_TARGET_LINK = "panda_hand"

    def __init__(
        self,
        *,
        rate: float = 100.0,
        viser_server=None,
        bimanual: bool = False,
        coordinate_frame: Literal["base", "world"] = "base",
        robot_description: Optional[str] = None,
        target_link_name: str = DEFAULT_TARGET_LINK,
    ) -> None:
        self.robot: Optional[pk.Robot] = None
        self.rest_pose: Optional[np.ndarray] = None
        self.target_link_names = [target_link_name]
        if bimanual:
            self.target_link_names = [target_link_name, target_link_name]
        self._joint_count = 7
        self.coordinate_frame = coordinate_frame
        self.has_jitted_left = False
        self.has_jitted_right = False
        self.first_solve = True

        description = robot_description or self.DEFAULT_ROBOT_DESCRIPTION

        super().__init__(
            rate=rate,
            viser_server=viser_server,
            robot_description=description,
            bimanual=bimanual,
            coordinate_frame=coordinate_frame,
        )

        joint_count = self.robot.joints.num_actuated_joints if self.robot is not None else self._joint_count
        self._joint_count = joint_count

        # Override joint storage to match Franka's DOF configuration.
        self.joints["left"] = self.rest_pose
        if self.bimanual:
            self.joints["right"] = self.rest_pose

        # Register reset button callback now that the Viser widgets exist.
        @self.reset_button.on_click
        def _(_event) -> None:  # type: ignore[misc]
            self.home()

        # Ensure the scene starts in the rest pose.
        self.home()

    # ------------------------------------------------------------------
    # ViserAbstractBase hook implementations
    # ------------------------------------------------------------------
    def _setup_visualization(self) -> None:
        super()._setup_visualization()
        if self.bimanual:
            # Add a mirrored URDF for the right arm so both targets are visible.
            self.base_frame_right = self.viser_server.scene.add_frame("/base/base_right", show_axes=False)
            self.base_frame_right.position = (0.0, -0.61, 0.0)
            self.urdf_vis_right = viser.extras.ViserUrdf(
                self.viser_server,
                deepcopy(self.urdf),
                root_node_name="/base/base_right",
            )

    def _setup_solver_specific(self) -> None:
        self.robot = pk.Robot.from_urdf(self.urdf)
        cfg = np.array(self.urdf.cfg, dtype=float)
        # Hardcoded to be close to first solve of IK
        self.rest_pose = np.array([ 4.5195120e-04, -4.1407236e-01, -4.3886289e-04, -2.5841961, -2.1377161e-04,  2.1701238e+00,  7.8556901e-01,  2.0000000e-02])

    def _setup_gui(self) -> None:
        super()._setup_gui()
        self.timing_handle_left = self.viser_server.gui.add_number("Left Arm Time (ms)", 0.01, disabled=True)
        if self.bimanual:
            self.timing_handle_right = self.viser_server.gui.add_number("Right Arm Time (ms)", 0.01, disabled=True)

    def _initialize_transform_handles(self) -> None:
        default_position = (0.4, 0.0, 0.4)
        default_orientation = vtf.SO3.from_rpy_radians(0.0, np.pi, np.pi).wxyz

        left_handle = self.transform_handles.get("left")
        if left_handle and left_handle.control is not None:
            left_handle.control.position = default_position  # type: ignore[assignment]
            left_handle.control.wxyz = default_orientation  # type: ignore[assignment]
        if left_handle:
            left_handle.tcp_offset_frame.position = (0.0, 0.0, 0.0)  # type: ignore[assignment]
            left_handle.tcp_offset_frame.wxyz = (1.0, 0.0, 0.0, 0.0)  # type: ignore[assignment]

        if self.bimanual:
            right_handle = self.transform_handles.get("right")
            if right_handle:
                if right_handle.control is not None:
                    right_handle.control.position = default_position  # type: ignore[assignment]
                    right_handle.control.wxyz = default_orientation  # type: ignore[assignment]
                right_handle.tcp_offset_frame.position = (0.0, 0.0, 0.0)  # type: ignore[assignment]
                right_handle.tcp_offset_frame.wxyz = (1.0, 0.0, 0.0, 0.0)  # type: ignore[assignment]

    def _update_optional_handle_sizes(self) -> None:
        # No auxiliary gizmos to resize.
        return

    def solve_ik(self) -> None:
        if self.robot is None:
            return

        target_poses = self.get_target_poses()
        if not target_poses:
            return

        for side in ("left", "right"):
            if side not in target_poses:
                continue

            target_tf = target_poses[side]
            idx = 0 if side == "left" else 1

            start = time.time()
            # solution = solve_ik(
            #     robot=self.robot,
            #     target_link_name=self.target_link_names[idx],
            #     target_position=target_tf.translation(),
            #     target_wxyz=target_tf.rotation().wxyz,
            # )
            solution = solve_ik_vel_cost(
                robot=self.robot,
                target_link_name=self.target_link_names[idx],
                target_position=target_tf.translation(),
                target_wxyz=target_tf.rotation().wxyz,
                prev_cfg=self.joints[side],
            )
            elapsed_ms = (time.time() - start) * 1000.0

            self.joints[side] = solution

            if side == "left":
                self.timing_handle_left.value = elapsed_ms  # type: ignore[attr-defined]
            elif self.bimanual:
                self.timing_handle_right.value = elapsed_ms  # type: ignore[attr-defined]

    def update_visualization(self) -> None:
        self.urdf_vis_left.update_cfg(self.joints["left"])
        if self.bimanual:
            self.urdf_vis_right.update_cfg(self.joints["right"])  # type: ignore[attr-defined]

    def home(self) -> None:
        if self.rest_pose is None:
            return

        self.joints["left"] = self.rest_pose.copy()
        if self.bimanual:
            self.joints["right"] = self.rest_pose.copy()

        self.update_visualization()
        self._initialize_transform_handles()

    @property
    def joint_count(self) -> int:
        """Number of actuated joints controlled by the IK solver."""

        return self._joint_count


__all__ = ["FrankaPyroki"]

