"""Teleoperation agent that combines PyRoKi IK with Franka OSC control.

This version uses linear interpolation for trajectory execution, triggered
by response changes from the remote msgpack client.
"""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any, Dict, Optional, List

import numpy as np
import viser
import viser.extras
import viser.transforms as vtf
from dm_env.specs import Array

import requests

from robots_realtime.agents.agent import Agent
from robots_realtime.robots.inverse_kinematics.franka_pyroki import FrankaPyroki
from robots_realtime.sensors.cameras.camera_utils import obs_get_rgb, resize_with_center_crop
from robots_realtime.utils.portal_utils import remote
from robots_realtime.utils.depth_utils import depth_color_to_pointcloud
from robots_realtime.utils.server_client_utils import SyncMsgpackNumpyClient


class FrankaOscClientCartesianAgent(Agent):
    """Interactive teleoperation agent for Franka OSC robots with linear interpolation.
    
    Similar to FrankaPyrokiViserAgentLinearInterp but target pose comes from
    msgpack numpy server instead of viser gizmos. Execution is triggered
    automatically when a new different pose arrives.
    """

    def __init__(
        self,
        *,
        bimanual: bool = False,
        right_arm_extrinsic: Optional[Dict[str, Any]] = None,
        robot_description: Optional[str] = None,
        ik_rate: float = 100.0,
        visualize_rgbd: bool = False,
        robotiq_gripper: bool = False,
    ) -> None:
        self.bimanual = bimanual
        self.robotiq_gripper = robotiq_gripper
        self.right_arm_extrinsic = right_arm_extrinsic
        self.visualize_rgbd = visualize_rgbd
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
        
        # Current joint target being sent to robot
        self._curr_joint_target = np.array(self.ik.rest_pose, dtype=np.float32)
        if self.robotiq_gripper:
            self._curr_joint_target[-1] = 1.0
        
        self.executing_traj = False
        self.joint_trajectory_waypoints: List[List[float]] = []
        self.traj_urdfs: List = []
        
        # Target from msgpack server (equivalent to viser gizmo target)
        self.target_joint_pos: Optional[np.ndarray] = None
        self.target_gripper_pos: Optional[float] = None
        
        if self.robotiq_gripper:
            self.ik.transform_handles.get("left").tcp_offset_frame.wxyz = vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi/4).wxyz
            self.ik.transform_handles.get("left").tcp_offset_frame.position = (0.0, 0.0, -0.157)
        
        self.ik_thread = threading.Thread(target=self.ik.run, name="franka_pyroki_ik")
        self.ik_thread.daemon = True
        self.ik_thread.start()

        self.franka_client = SyncMsgpackNumpyClient(host="0.0.0.0", port=9000)

        self.obs: Optional[Dict[str, Any]] = None
        self._update_period = 0.05
        self._setup_visualization()

        self.real_vis_thread = threading.Thread(target=self._update_visualization, name="franka_real_vis")
        self.real_vis_thread.daemon = True
        self.real_vis_thread.start()
        
        # Warmstart JAX JIT compilation
        self._warmstart_planner()

        # Continuous trajectory interpolation thread (like viser agent)
        self.traj_interp_thread = threading.Thread(target=self._traj_interp, name="franka_traj_interp")
        self.traj_interp_thread.daemon = True
        self.traj_interp_thread.start()
        
        self.prev_response = {}

    def _warmstart_planner(self) -> None:
        """Warmstart JAX JIT compilation by running a dummy planning call."""
        print("Warmstarting trajectory planner (JIT compile)...")
        warmstart_joints = np.array(self.ik.rest_pose, dtype=np.float32)
        warmstart_gripper = 1.0 if self.robotiq_gripper else 0.08
        
        # Temporarily set state for warmstart
        self.obs = {"left": {"joint_pos": warmstart_joints}}
        self.target_joint_pos = warmstart_joints[:7]
        self.target_gripper_pos = warmstart_gripper
        
        # Trigger one planning call
        try:
            self._plan_trajectory()
        except Exception as e:
            print(f"Warmstart planning call: {e}")
        
        print("Warmstart complete")

    # ------------------------------------------------------------------
    # Trajectory interpolation (continuous, like viser agent)
    # ------------------------------------------------------------------
    def _traj_interp(self) -> None:
        """Continuously plan trajectories from current pose to target pose.
        
        This runs in a loop, always keeping self.joint_trajectory_waypoints
        up-to-date with the latest planned trajectory.
        """
        while True:
            if self.obs is None or self.target_joint_pos is None:
                time.sleep(0.1)
                continue

            try:
                self._plan_trajectory()
            except Exception as e:
                print(f"Planning error: {e}")

            time.sleep(0.05)  # 20Hz planning rate

    def _plan_trajectory(self) -> None:
        """Plan trajectory from current EEF pose to target pose."""
        gripper_val = self.target_gripper_pos if self.target_gripper_pos is not None else 1.0
        
        # Compute target EEF pose from target joint positions
        target_fk = self.ik.robot.forward_kinematics(
            np.concatenate([self.target_joint_pos, [gripper_val]])
        )[-4]
        target_pose_wxyz_xyz = np.asarray(target_fk[:4]).tolist() + np.asarray(target_fk[4:]).tolist()
        
        payload = {
            "start_pose_wxyz_xyz": list(self.real_eef_frame_left.wxyz) + list(self.real_eef_frame_left.position),
            "end_pose_wxyz_xyz": target_pose_wxyz_xyz,
            "prev_cfg": list(self.obs["left"]["joint_pos"]),
            "timesteps": 25,
        }
        
        response = requests.post("http://127.0.0.1:8116/plan", json=payload, timeout=2.0)
        response.raise_for_status()
        data = response.json()
        
        # Update waypoints with correct gripper value
        waypoints = data["waypoints"]
        for waypoint in waypoints:
            waypoint[-1] = gripper_val
        
        self.joint_trajectory_waypoints = waypoints
        
        # Visualize trajectory
        self._visualize_trajectory(waypoints)

    def _visualize_trajectory(self, waypoints: List[List[float]]) -> None:
        """Visualize trajectory with ghost URDFs."""
        mod_factor = 5
        for i, waypoint in enumerate(waypoints):
            if i % mod_factor != 0:
                continue
            idx = int(i / mod_factor)
            if idx > len(self.traj_urdfs) - 1:
                self.traj_urdfs.append(
                    viser.extras.ViserUrdf(
                        self.viser_server,
                        deepcopy(self.ik.urdf),
                        root_node_name=f"/traj_interp/waypoint_{i}",
                        mesh_color_override=(0.55, 0.75, 0.35),
                    )
                )
                for mesh in self.traj_urdfs[-1]._meshes:
                    mesh.opacity = 0.15
            self.traj_urdfs[idx].update_cfg(waypoint)

    def _execute_traj(self) -> None:
        """Execute the current planned trajectory.
        
        Called when a new target arrives from msgpack server.
        """
        if not self.joint_trajectory_waypoints:
            print("No waypoints to execute")
            return
            
        print(f"Executing trajectory with {len(self.joint_trajectory_waypoints)} waypoints...")
        joint_trajectory_waypoints = deepcopy(self.joint_trajectory_waypoints)
        self.executing_traj = True
        
        for i, waypoint in enumerate(joint_trajectory_waypoints):
            self._curr_joint_target = np.array(waypoint)
            
            # Compute target pose for error checking
            target_position = self.ik.robot.forward_kinematics(self._curr_joint_target)[-4][4:]
            target_orientation = self.ik.robot.forward_kinematics(self._curr_joint_target)[-4][:4]
            
            pos_error = np.linalg.norm(np.array(self.real_eef_frame_left.position) - np.array(target_position))
            ori_error = np.linalg.norm(np.array(self.real_eef_frame_left.wxyz) - np.array(target_orientation))
            
            # Wait longer on last waypoint for convergence
            if i == len(joint_trajectory_waypoints) - 1:
                time.sleep(0.15)
            else:
                time.sleep(0.06)
        
        self._curr_joint_target = np.array(self.obs["left"]["joint_pos"])
        self.executing_traj = False
        print("Trajectory execution complete")

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def _extract_joint_pos(self, obs: Dict[str, Any], arm: str) -> Optional[np.ndarray]:
        """Extract joint positions from observation."""
        arm_obs = obs.get(arm)
        if isinstance(arm_obs, dict):
            joint_pos = arm_obs.get("joint_pos")
            if joint_pos is not None:
                return np.asarray(joint_pos)
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
            mesh.opacity = 0.3

        self.base_frame_left_hyrl = self.viser_server.scene.add_frame("/franka_hyrl", show_axes=False)
        self.urdf_vis_left_hyrl = viser.extras.ViserUrdf(
            self.viser_server,
            deepcopy(self.ik.urdf),
            root_node_name="/franka_hyrl",
            mesh_color_override=(0.55, 0.35, 0.95),
        )
        for mesh in self.urdf_vis_left_hyrl._meshes:
            mesh.opacity = 0.3

        # EEF frames
        self.real_eef_frame_left = self.viser_server.scene.add_frame(
            "/franka_real/eef", show_axes=True, axes_length=0.1, axes_radius=0.005
        )
        self.target_eef_frame_left = self.viser_server.scene.add_frame(
            "/target_eef", show_axes=True, axes_length=0.1, axes_radius=0.005
        )

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
                mesh.opacity = 0.3

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

            # Update real robot visualization
            left_joint_pos = self._extract_joint_pos(obs_copy, "left")
            if left_joint_pos is not None:
                self.urdf_vis_left_real.update_cfg(left_joint_pos)
                real_fk = self.ik.robot.forward_kinematics(left_joint_pos)[-4]
                self.real_eef_frame_left.position = real_fk[4:]
                self.real_eef_frame_left.wxyz = real_fk[:4]

            # Update target EEF frame
            if self.target_joint_pos is not None and self.target_gripper_pos is not None:
                target_fk = self.ik.robot.forward_kinematics(
                    np.concatenate([self.target_joint_pos, [self.target_gripper_pos]])
                )[-4]
                self.target_eef_frame_left.position = target_fk[4:]
                self.target_eef_frame_left.wxyz = target_fk[:4]

            # Update hyrl ghost
            if self.target_joint_pos is not None:
                gripper_vis = self.target_gripper_pos if self.robotiq_gripper else self.target_gripper_pos * 0.08
                self.urdf_vis_left_hyrl.update_cfg(np.concatenate([self.target_joint_pos, [gripper_vis]]))

            if self.bimanual:
                right_joint_pos = self._extract_joint_pos(obs_copy, "right")
                if right_joint_pos is not None:
                    self.urdf_vis_right_real.update_cfg(right_joint_pos)

            # Camera images
            rgb_images = obs_get_rgb(obs_copy)
            if rgb_images:
                for key, image in rgb_images.items():
                    if key not in self.viser_cam_img_handles:
                        self.viser_cam_img_handles[key] = self.viser_server.gui.add_image(
                            resize_with_center_crop(image, 224, 224), label=key
                        )
                    if self.visualize_rgbd:
                        self.viser_cam_img_handles[key].image = resize_with_center_crop(image, 224, 224)
            
            time.sleep(self._update_period)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def _responses_differ(self, resp1: Dict, resp2: Dict) -> bool:
        """Check if two responses have meaningfully different joint targets."""
        if not resp1 or not resp2:
            return bool(resp1) != bool(resp2)
        if b'left' not in resp1 or b'left' not in resp2:
            return False

        left1 = resp1[b'left']
        left2 = resp2[b'left']
        
        joint_pos1 = np.asarray(left1.get(b'joint_pos', []), dtype=np.float32)
        joint_pos2 = np.asarray(left2.get(b'joint_pos', []), dtype=np.float32)
        
        if len(joint_pos1) > 0 and len(joint_pos2) > 0:
            if np.linalg.norm(joint_pos1 - joint_pos2) > 0.01:
                return True
                
        gripper1 = np.asarray(left1.get(b'gripper', []), dtype=np.float32)
        gripper2 = np.asarray(left2.get(b'gripper', []), dtype=np.float32)
        
        if np.linalg.norm(gripper1 - gripper2) > 0.01:
            return True
        
        return False

    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        self.obs = deepcopy(obs)

        # Handle camera extrinsics
        if "top_camera" in self.obs:
            self.obs["top_camera"]["pose"] = np.concatenate([
                np.array([1.0, 0, 0.29]), 
                vtf.SO3.from_rpy_radians(np.pi/2 - np.pi/6, np.pi, -np.pi/2).wxyz
            ])
            self.obs["top_camera"]["pose_mat"] = vtf.SE3(
                wxyz_xyz=np.concatenate([self.obs["top_camera"]["pose"][3:], self.obs["top_camera"]["pose"][:3]])
            ).as_matrix()
        
        # Get response from msgpack server
        response = self.franka_client.send_request(self.obs)
        valid_response = response and isinstance(response, dict) and b'left' in response
        
        if valid_response:
            new_joint_pos = np.asarray(response.get(b'left').get(b'joint_pos'), dtype=np.float32)
            new_gripper_pos = np.asarray(response.get(b'left').get(b'gripper'), dtype=np.float32)

            # Rising edge detection - execute when new different pose arrives
            if self._responses_differ(response, self.prev_response):
                print("New target received - triggering execution")
                self.prev_response = deepcopy(response)
                
                # Update target (planner thread will pick this up)
                self.target_joint_pos = new_joint_pos.copy()
                self.target_gripper_pos = float(new_gripper_pos)
                
                # Execute trajectory in background thread so act() doesn't block
                exec_thread = threading.Thread(target=self._execute_traj, name="traj_exec")
                exec_thread.daemon = True
                exec_thread.start()

        # Return current joint target
        left_target = np.array(self._curr_joint_target, dtype=np.float32)
        action: Dict[str, Dict[str, np.ndarray]] = {"left": {"pos": left_target}}

        if self.bimanual:
            assert "right" in self.ik.joints
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


__all__ = ["FrankaOscClientCartesianAgent"]
