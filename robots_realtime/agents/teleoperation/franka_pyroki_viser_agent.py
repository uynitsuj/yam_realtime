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

class FrankaPyrokiViserAgent(Agent):
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
    ) -> None:
        self.bimanual = bimanual
        self.right_arm_extrinsic = right_arm_extrinsic
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
        self.ik_thread = threading.Thread(target=self.ik.run, name="franka_pyroki_ik")
        self.ik_thread.daemon = True
        self.ik_thread.start()

        self.obs: Optional[Dict[str, Any]] = None
        self._update_period = 0.02
        self._setup_visualization()

        self.real_vis_thread = threading.Thread(target=self._update_visualization, name="franka_real_vis")
        self.real_vis_thread.daemon = True
        self.real_vis_thread.start()

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

        self.left_gripper_slider_handle = self.viser_server.gui.add_slider(
            label="Gripper Width", min=0.0, max=0.08, step=0.001, initial_value=0.08
        )
        if self.bimanual:
            self.right_gripper_slider_handle = self.viser_server.gui.add_slider(
                label="Gripper Width (R)", min=0.0, max=0.08, step=0.001, initial_value=0.08
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
                        self.viser_cam_img_handles[key] = self.viser_server.gui.add_image(image, label=key)
                    self.viser_cam_img_handles[key].image = resize_with_center_crop(image, 224, 224)

                    if key not in self.camera_frustum_handles:
                        self.camera_frustum_handles[key] = self.viser_server.scene.add_camera_frustum(
                            name = f"camera_frustum_{key}",
                            fov = 60.0,
                            aspect = 1.0,
                            scale = 0.01,
                            cast_shadow = False,
                            receive_shadow = False,
                        )
                    self.camera_frustum_handles[key].image = resize_with_center_crop(image, 224, 224)

                    # For now these are hardcoded, but we should attach extrinsics files to sensor and pass to obs

                    self.camera_frustum_handles[key].position = (1.0, 0, 0.28)

                    self.camera_frustum_handles[key].wxyz = vtf.SO3.from_rpy_radians(np.pi/2 + np.pi/6, 0.0, -np.pi/2).wxyz

                    if "depth_data" in obs_copy[key]:
                        depth_data = obs_copy[key]["depth_data"]
                        points, colors = depth_color_to_pointcloud(
                            depth = depth_data,
                            rgb_img = image,
                            intrinsics = obs_copy[key]["intrinsics"]["left"]["intrinsics_matrix"], # We assume we're taking left camera image from a stereo pair
                            subsample_factor = 4,
                            depth_clip_range = (0.015, 1.2),
                        )
                        self.viser_server.scene.add_point_cloud(name = f"camera_frustum_{key}/point_cloud_{key}", points = points, colors = colors, point_size = 0.002, wxyz = vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi).wxyz)
                
                time.sleep(self._update_period)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        self.obs = deepcopy(obs)

        left_target = np.asarray(self.ik.joints["left"], dtype=np.float32)
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
