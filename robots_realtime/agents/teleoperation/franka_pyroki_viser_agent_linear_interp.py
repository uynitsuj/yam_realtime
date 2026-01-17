"""Teleoperation agent that combines PyRoKi IK with Franka OSC control."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import viser
import viser.extras
import viser.transforms as vtf
from dm_env.specs import Array

from robots_realtime.agents.agent import Agent
from robots_realtime.robots.inverse_kinematics.franka_pyroki import FrankaPyroki
from robots_realtime.sensors.cameras.camera_utils import obs_get_rgb, resize_with_center_crop
from robots_realtime.utils.portal_utils import remote
from robots_realtime.utils.depth_utils import depth_color_to_pointcloud

import requests

class FrankaPyrokiViserAgentLinearInterp(Agent):
    """Interactive teleoperation agent for Franka OSC robots.

    The agent exposes Viser transform gizmos (powered by :class:`FrankaPyroki`) that
    continuously solve for joint targets using PyRoKi. The resulting joint targets are fed to
    the :class:`robots_realtime.robots.franka_osc.FrankaPanda` controller through the environment
    interface, making it possible to drive the real robot with Viser while monitoring live
    state feedback.
    """

    def __init__(
        self,
        *,
        bimanual: bool = False,
        right_arm_extrinsic: Optional[Dict[str, Any]] = None,
        robot_description: Optional[str] = None,
        ik_rate: float = 100.0,
        visualize_rgbd: bool = True,
        robotiq_gripper: bool = False,
    ) -> None:
        self.bimanual = bimanual
        self.right_arm_extrinsic = right_arm_extrinsic
        self.visualize_rgbd = visualize_rgbd
        self.robotiq_gripper = robotiq_gripper
        if self.bimanual:
            assert right_arm_extrinsic is not None, (
                "right_arm_extrinsic must be provided for bimanual Franka configuration"
            )

        self.viser_server = viser.ViserServer()
        self.ik = FrankaPyroki(
            rate=ik_rate,
            viser_server=self.viser_server,
            bimanual=bimanual,
            robot_description=robot_description,
        )
        self._curr_joint_target = self.ik.rest_pose
        self.executing_traj = False
        if self.robotiq_gripper:
            self.ik.transform_handles.get("left").tcp_offset_frame.wxyz = vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi/4).wxyz
            self.ik.transform_handles.get("left").tcp_offset_frame.position = (0.0, 0.0, -0.157)
        self.ik_thread = threading.Thread(target=self.ik.run, name="franka_pyroki_ik")
        self.ik_thread.daemon = True
        self.ik_thread.start()

        self.obs: Optional[Dict[str, Any]] = None
        self._update_period = 0.05
        self._setup_visualization()

        self.real_vis_thread = threading.Thread(target=self._update_visualization, name="franka_real_vis")
        self.real_vis_thread.daemon = True
        self.real_vis_thread.start()

        self.traj_interp_thread = threading.Thread(target=self._traj_interp, name="franka_traj_interp")
        self.traj_interp_thread.daemon = True
        self.traj_interp_thread.start()

    def _traj_interp(self) -> None:
        """Interpolate the trajectory between the rest pose and the current joint positions."""
        self.traj_urdfs = []

        while True:
            if self.obs is None:
                time.sleep(0.1)
                continue

            payload = {
                "start_pose_wxyz_xyz": list(self.real_eef_frame_left.wxyz) + list(self.real_eef_frame_left.position),
                "end_pose_wxyz_xyz": list(self.target_eef_frame_left.wxyz) + list(self.target_eef_frame_left.position),
                "prev_cfg": list(self.obs["left"]["joint_pos"]),
                "timesteps": 20,
            }
            response = requests.post("http://127.0.0.1:8116/plan", json=payload, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            # print(data)
            # time.sleep(0.01)
            self.joint_trajectory_waypoints = data["waypoints"]

            mod_factor = 5
            for i, waypoint in enumerate(data["waypoints"]):
                # visualize with viser urdfs 
                # skip if i is not % 5 == 0
                if i % mod_factor != 0:
                    continue
                if int(i / mod_factor) > len(self.traj_urdfs) - 1:
                    self.traj_urdfs.append(
                        viser.extras.ViserUrdf(
                            self.viser_server,
                            deepcopy(self.ik.urdf),
                            root_node_name=f"/traj_interp/waypoint_{i}",
                            mesh_color_override=(0.55, 0.75, 0.35),
                        )
                    )
                    for mesh in self.traj_urdfs[-1]._meshes:
                        mesh.opacity = 0.15  # type: ignore[attr-defined]
                self.traj_urdfs[int(i / mod_factor)].update_cfg(waypoint)

            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def _extract_joint_pos(self, obs: Dict[str, Any], arm: str) -> Optional[np.ndarray]:
        """Best-effort extraction of joint positions for the requested arm from an observation."""

        arm_obs = obs.get(arm)
        if isinstance(arm_obs, dict):
            joint_pos = arm_obs.get("joint_pos")
            if joint_pos is not None:
                return np.asarray(joint_pos)

        # Fall back to top-level fields if the environment exposes single-arm observations.
        if arm == "left" and obs.get("joint_pos") is not None:
            return np.asarray(obs["joint_pos"])

        return None

    def _setup_visualization(self) -> None:
        """Prepare Viser overlays for live robot state and camera feeds."""

        self.base_frame_left_real = self.viser_server.scene.add_frame("/franka_real", show_axes=False)
        self.urdf_vis_left_real = viser.extras.ViserUrdf(
            self.viser_server,
            deepcopy(self.ik.urdf),
            root_node_name="/franka_real",
            mesh_color_override=(0.55, 0.75, 0.95),
        )
        for mesh in self.urdf_vis_left_real._meshes:
            mesh.opacity = 0.3  # type: ignore[attr-defined]
        
        self.real_eef_frame_left = self.viser_server.scene.add_frame("/franka_real/eef", show_axes=True, axes_length=0.1, axes_radius=0.005)
        self.target_eef_frame_left = self.viser_server.scene.add_frame("/target_eef", show_axes=True, axes_length=0.1, axes_radius=0.005)

        self.execute_traj_button = self.viser_server.gui.add_button(label="Execute Trajectory")

        @self.execute_traj_button.on_click
        def _(_) -> None:
            """Execute the trajectory."""
            print("Executing trajectory...")
            self._execute_traj()


        if self.bimanual and self.right_arm_extrinsic is not None:
            self.ik.base_frame_right.position = np.array(self.right_arm_extrinsic["position"])
            self.ik.base_frame_right.wxyz = np.array(self.right_arm_extrinsic["rotation"])

            self.base_frame_right_real = self.viser_server.scene.add_frame("/franka_real/right", show_axes=False)
            self.base_frame_right_real.position = self.ik.base_frame_right.position
            self.urdf_vis_right_real = viser.extras.ViserUrdf(
                self.viser_server,
                deepcopy(self.ik.urdf),
                root_node_name="/franka_real/right",
                mesh_color_override=(0.55, 0.75, 0.95),
            )
            for mesh in self.urdf_vis_right_real._meshes:
                mesh.opacity = 0.3  # type: ignore[attr-defined]

        self.viser_cam_img_handles: Dict[str, viser.GuiImageHandle] = {}

        if self.robotiq_gripper:
            self.left_gripper_slider_handle = self.viser_server.gui.add_slider(
                label="Gripper Width", min=0.0, max=1.0, step=0.005, initial_value=1.0
            )
        else:
            self.left_gripper_slider_handle = self.viser_server.gui.add_slider(
                label="Gripper Width", min=0.0, max=0.1, step=0.001, initial_value=0.1
            )
        if self.bimanual:
            self.right_gripper_slider_handle = self.viser_server.gui.add_slider(
                label="Gripper Width (R)", min=0.0, max=0.1, step=0.001, initial_value=0.1
            )
        
        self.camera_frustum_handles: Dict[str, viser.CameraFrustumHandle] = {}

    def _update_visualization(self) -> None:
        """Continuously sync live robot state and camera frames into Viser."""

        while self.obs is None:
            time.sleep(0.025)

        while True:
            obs_copy = self.obs
            if obs_copy is None:
                time.sleep(self._update_period)
                continue


            self.real_eef_frame_left.position = self.ik.robot.forward_kinematics(obs_copy["left"]["joint_pos"])[-4][4:]
            self.real_eef_frame_left.wxyz = self.ik.robot.forward_kinematics(obs_copy["left"]["joint_pos"])[-4][:4]

            self.target_eef_frame_left.position = self.ik.robot.forward_kinematics(self.ik.joints["left"])[-4][4:]
            self.target_eef_frame_left.wxyz = self.ik.robot.forward_kinematics(self.ik.joints["left"])[-4][:4]


            left_joint_pos = self._extract_joint_pos(obs_copy, "left")
            if left_joint_pos is not None:
                self.urdf_vis_left_real.update_cfg(left_joint_pos)

            if self.bimanual:
                right_joint_pos = self._extract_joint_pos(obs_copy, "right")
                if right_joint_pos is not None:
                    self.urdf_vis_right_real.update_cfg(right_joint_pos)

            rgb_images = obs_get_rgb(obs_copy)
            if rgb_images:
                for key, image in rgb_images.items():
                    if key not in self.viser_cam_img_handles:
                        self.viser_cam_img_handles[key] = self.viser_server.gui.add_image(resize_with_center_crop(image, 224, 224), label=key)
                    if self.visualize_rgbd:
                        self.viser_cam_img_handles[key].image = resize_with_center_crop(image, 224, 224)

                    if key not in self.camera_frustum_handles:
                        self.camera_frustum_handles[key] = self.viser_server.scene.add_camera_frustum(
                            name = f"camera_frustum_{key}",
                            fov = 1.2,
                            aspect = 1.0,
                            scale = 0.05,
                            cast_shadow = False,
                            receive_shadow = False,
                        )
                    if self.visualize_rgbd:
                        self.camera_frustum_handles[key].image = resize_with_center_crop(image, 224, 224)

                    # For now these are hardcoded, TODO: Should attach extrinsics files to sensor class obj and pass extr to obs

                    self.camera_frustum_handles[key].position = (1.0, 0, 0.29)

                    self.camera_frustum_handles[key].wxyz = vtf.SO3.from_rpy_radians(np.pi/2 - np.pi/6, np.pi, -np.pi/2).wxyz

                    if "depth_data" in obs_copy[key] and self.visualize_rgbd:
                        depth_data = obs_copy[key]["depth_data"]
                        points, colors = depth_color_to_pointcloud(
                            depth = depth_data,
                            rgb_img = image,
                            intrinsics = obs_copy[key]["intrinsics"]["left"]["intrinsics_matrix"], # We assume we're taking left camera image from a stereo pair
                            subsample_factor = 4,
                            depth_clip_range = (0.015, 1.2),
                        )
                        self.viser_server.scene.add_point_cloud(name = f"camera_frustum_{key}/point_cloud_{key}", points = points, colors = colors, point_size = 0.002)
                
                time.sleep(self._update_period)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def _execute_traj(self) -> None:
        """Execute the trajectory."""
        print("Executing trajectory...")
        # take self.trajectory_waypoints, deepcopy it, and then iterate over it
        joint_trajectory_waypoints = deepcopy(self.joint_trajectory_waypoints)
        # joint_trajectory_waypoints.append(self.ik.joints["left"])
        self.executing_traj = True
        for i, waypoint in enumerate(joint_trajectory_waypoints):
            print(f"Executing waypoint {i} of {len(joint_trajectory_waypoints)}")
            self._curr_joint_target = np.array(waypoint)
            target_position = self.ik.robot.forward_kinematics(self._curr_joint_target)[-4][4:]
            target_orientation = self.ik.robot.forward_kinematics(self._curr_joint_target)[-4][:4]
            # print(f"Target position: {target_position}, Target orientation: {target_orientation}")
            pos_error = np.linalg.norm(self.real_eef_frame_left.position - target_position)
            ori_error = np.linalg.norm(self.real_eef_frame_left.wxyz - target_orientation)
            if i == len(joint_trajectory_waypoints) - 1:
                while pos_error > 0.015 or ori_error > 0.02:
                    time.sleep(0.15)
                    pos_error = np.linalg.norm(self.real_eef_frame_left.position - target_position)
                    ori_error = np.linalg.norm(self.real_eef_frame_left.wxyz - target_orientation)
                    print(f"Pos error: {pos_error}, Ori error: {ori_error}")
            else:
                time.sleep(0.05)
        self._curr_joint_target = self.obs["left"]["joint_pos"]
        self.executing_traj = False

    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        self.obs = deepcopy(obs)

        # self.ik.joints["left"]

        left_target = np.asarray(self._curr_joint_target, dtype=np.float32)

        left_target[-1] = self.left_gripper_slider_handle.value
        action: Dict[str, Dict[str, np.ndarray]] = {"left": {"pos": left_target}}

        if self.bimanual:
            assert "right" in self.ik.joints, "bimanual mode requires both IK solutions"
            right_target = np.asarray(self.ik.joints["right"], dtype=np.float32)
            right_target[-1] = self.right_gripper_slider_handle.value
            action["right"] = {"pos": right_target}

        return action

    @remote(serialization_needed=True)
    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        """Expose the joint-position action specification for Franka OSC."""

        action_spec = {
            "left": {"pos": Array(shape=(self.ik.joint_count,), dtype=np.float32)},
        }
        if self.bimanual:
            action_spec["right"] = {"pos": Array(shape=(self.ik.joint_count,), dtype=np.float32)}
        return action_spec


__all__ = ["FrankaPyrokiViserAgent"]
