"""Teleoperation agent that combines PyRoKi IK with Franka OSC control.

This version uses linear interpolation for trajectory execution, triggered
by response changes from the remote client.
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
    """Interactive teleoperation agent for Franka OSC robots with linear interpolation."""

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

        self.hyrl_joint_pos = None
        self.hyrl_gripper_pos = None

        self.viser_server = viser.ViserServer()
        self.ik = FrankaPyroki(
            rate=ik_rate,
            viser_server=self.viser_server,
            bimanual=bimanual,
            robot_description=robot_description,
        )
        
        # Thread safety
        self._target_lock = threading.Lock()
        # Initialize target to rest pose so the robot doesn't snap on startup
        self._curr_joint_target = np.array(self.ik.rest_pose, dtype=np.float32)
        if self.robotiq_gripper:
            self._curr_joint_target[-1] = 1.0
        
        self.executing_traj = False
        self.joint_trajectory_waypoints = []
        self.traj_urdfs = []
        
        self.ik_thread = threading.Thread(target=self.ik.run, name="franka_pyroki_ik")
        self.ik_thread.daemon = True
        self.ik_thread.start()

        if self.robotiq_gripper:
            self.ik.transform_handles.get("left").tcp_offset_frame.wxyz = vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi/4).wxyz
            self.ik.transform_handles.get("left").tcp_offset_frame.position = (0.0, 0.0, -0.157)

        self.franka_client = SyncMsgpackNumpyClient(host="0.0.0.0", port=9000)

        self.obs: Optional[Dict[str, Any]] = None
        self._update_period = 0.05
        self._setup_visualization()

        self.real_vis_thread = threading.Thread(target=self._update_visualization, name="franka_real_vis")
        self.real_vis_thread.daemon = True
        self.real_vis_thread.start()
        
        self.prev_response = {}
        self._pending_target_joint_pos = None
        self._pending_target_gripper_pos = None
        
        # Signals the executor to run
        self._execute_request = threading.Event()  
        
        # Single persistent executor thread
        self.executor_thread = threading.Thread(target=self._executor, name="traj_executor")
        self.executor_thread.daemon = True
        self.executor_thread.start()

    # ------------------------------------------------------------------
    # Trajectory planning and execution
    # ------------------------------------------------------------------
    def _executor(self) -> None:
        """Persistent worker thread that waits for execution requests."""
        while True:
            # Wait for a signal to execute
            self._execute_request.wait()
            self._execute_request.clear()
            
            # Skip if data is missing
            if self.obs is None or self._pending_target_joint_pos is None:
                continue
            
            self.executing_traj = True
            
            try:
                # Plan trajectory
                # Note: We pass copies to avoid race conditions if variables change during planning
                target_j = self._pending_target_joint_pos.copy()
                target_g = self._pending_target_gripper_pos
                
                joint_trajectory_waypoints = self._plan_trajectory(target_j, target_g)
                
                if joint_trajectory_waypoints:
                    # Execute trajectory
                    self._execute_waypoints(joint_trajectory_waypoints)
            except Exception as e:
                print(f"Executor error: {e}")
            finally:
                self.executing_traj = False
                print("Trajectory execution complete")

    def _plan_trajectory(self, target_joint_pos: np.ndarray, target_gripper_pos: float) -> List[List[float]]:
        """Plan trajectory from current pose to pending target."""
        # Compute target EEF pose from pending target joint positions
        # Ensure gripper scaling matches your robot hardware specifics
        gripper_val = target_gripper_pos #* 0.08
        
        target_fk = self.ik.robot.forward_kinematics(
            np.concatenate([target_joint_pos, [gripper_val]])
        )[-4]
        
        # Convert JAX arrays to Python floats for JSON serialization
        target_pose_wxyz_xyz = np.asarray(target_fk[:4]).tolist() + np.asarray(target_fk[4:]).tolist()
        
        # Get current start pose
        current_pose = list(self.real_eef_frame_left.wxyz) + list(self.real_eef_frame_left.position)
        
        payload = {
            "start_pose_wxyz_xyz": current_pose,
            "end_pose_wxyz_xyz": target_pose_wxyz_xyz,
            "prev_cfg": np.asarray(self.obs["left"]["joint_pos"]).tolist(),
            "timesteps": 20,
        }
        
        try:
            response = requests.post("http://127.0.0.1:8116/plan", json=payload, timeout=2.0)
            response.raise_for_status()
            data = response.json()
            joint_trajectory_waypoints = data["waypoints"]
        except Exception as e:
            print(f"Planning failed: {e}")
            return []

        # Append the final target to ensure we reach it exactly
        # final_pt = list(np.concatenate([target_joint_pos, [gripper_val]]))
        # joint_trajectory_waypoints.append(final_pt)
        for waypoint in joint_trajectory_waypoints:
            waypoint[-1] = gripper_val
        
        print(f"Trajectory planned with {len(joint_trajectory_waypoints)} waypoints")

        # Visualize trajectory with URDFs (Maintains your existing viz logic)
        self._visualize_trajectory(joint_trajectory_waypoints)
        
        return joint_trajectory_waypoints

    def _visualize_trajectory(self, waypoints: List[List[float]]) -> None:
        """Helper to handle Viser ghosts visualization."""
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

    def _execute_waypoints(self, joint_trajectory_waypoints: List[List[float]]) -> None:
        """Execute the planned trajectory waypoints by updating current target over time."""
        print("Executing trajectory...")
        
        # Execution speed control
        dt = 0.08  # 20Hz update rate for interpolation
        
        for i, waypoint in enumerate(joint_trajectory_waypoints):
            # Update the thread-safe target
            with self._target_lock:
                self._curr_joint_target = np.array(waypoint)
                
            
            # # Sleep to allow the act() loop to pick up this intermediate target
            # # and send it to the robot controller
            time.sleep(dt)
            if i == len(joint_trajectory_waypoints) - 1:
                time.sleep(0.1)

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

        self.base_frame_left_hyrl = self.viser_server.scene.add_frame("/franka_hyrl", show_axes=False)
        self.urdf_vis_left_hyrl = viser.extras.ViserUrdf(
            self.viser_server,
            deepcopy(self.ik.urdf),
            root_node_name="/franka_hyrl",
            mesh_color_override=(0.55, 0.35, 0.95),
        )
        for mesh in self.urdf_vis_left_hyrl._meshes:
            mesh.opacity = 0.3  # type: ignore[attr-defined]

        # EEF frames for trajectory visualization
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
                mesh.opacity = 0.3  # type: ignore[attr-defined]

        self.viser_cam_img_handles: Dict[str, viser.GuiImageHandle] = {}

        if not self.robotiq_gripper:
            self.left_gripper_slider_handle = self.viser_server.gui.add_slider(
                label="Gripper Width", min=0.0, max=0.1, step=0.001, initial_value=0.1
            )
        else:
            self.left_gripper_slider_handle = self.viser_server.gui.add_slider(
                label="Gripper Width", min=0.0, max=1.0, step=0.005, initial_value=1.0
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

            # Update real EEF frame position
            left_joint_pos = self._extract_joint_pos(obs_copy, "left")
            if left_joint_pos is not None:
                self.urdf_vis_left_real.update_cfg(left_joint_pos)
                # Update real EEF frame
                real_fk = self.ik.robot.forward_kinematics(left_joint_pos)[-4]
                self.real_eef_frame_left.position = real_fk[4:]
                self.real_eef_frame_left.wxyz = real_fk[:4]

            # Update target EEF frame from pending target
            if self._pending_target_joint_pos is not None:
                target_fk = self.ik.robot.forward_kinematics(np.concatenate([self._pending_target_joint_pos, [self._pending_target_gripper_pos*0.08]]))[-4]
                self.target_eef_frame_left.position = target_fk[4:]
                self.target_eef_frame_left.wxyz = target_fk[:4]

            if self.hyrl_joint_pos is not None:
                gripper_vis = self.hyrl_gripper_pos * 0.08 if not self.robotiq_gripper else self.hyrl_gripper_pos * 0.04
                self.urdf_vis_left_hyrl.update_cfg(np.concatenate([self.hyrl_joint_pos, [gripper_vis]]))

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
                time.sleep(self._update_period)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def _responses_differ(self, resp1: Dict, resp2: Dict) -> bool:
        """Check if two responses have meaningfully different joint targets."""
        # Check if responses are empty/None
        if not resp1 or not resp2:
            return bool(resp1) != bool(resp2)
            
        # Check if keys exist
        if b'left' not in resp1 or b'left' not in resp2:
            return False

        left1 = resp1[b'left']
        left2 = resp2[b'left']
        
        # Check joint differences
        joint_pos1 = np.asarray(left1.get(b'joint_pos', []), dtype=np.float32)
        joint_pos2 = np.asarray(left2.get(b'joint_pos', []), dtype=np.float32)
        
        if len(joint_pos1) > 0 and len(joint_pos2) > 0:
            if np.linalg.norm(joint_pos1 - joint_pos2) > 0.01:
                return True
                
        # Check gripper differences
        gripper1 = np.asarray(left1.get(b'gripper', []), dtype=np.float32)
        gripper2 = np.asarray(left2.get(b'gripper', []), dtype=np.float32)
        
        if np.linalg.norm(gripper1 - gripper2) > 0.01:
            return True
        
        return False

    def act(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        self.obs = deepcopy(obs)

        # Handle camera extrinsics (Hardcoded per user)
        if "top_camera" in self.obs:
            self.obs["top_camera"]["pose"] = np.concatenate([np.array([1.0, 0, 0.29]), vtf.SO3.from_rpy_radians(np.pi/2 - np.pi/6, np.pi, -np.pi/2).wxyz])
            self.obs["top_camera"]["pose_mat"] = vtf.SE3(wxyz_xyz=np.concatenate([self.obs["top_camera"]["pose"][3:], self.obs["top_camera"]["pose"][:3]])).as_matrix()
        
        # 1. Send observation, get response
        response = self.franka_client.send_request(self.obs)

        # 2. Parse response (Handling empty or malformed MsgPack)
        valid_response = response and isinstance(response, dict) and b'left' in response
        
        if valid_response:
            self.hyrl_joint_pos = np.asarray(response.get(b'left').get(b'joint_pos'), dtype=np.float32)
            self.hyrl_gripper_pos = np.asarray(response.get(b'left').get(b'gripper'), dtype=np.float32)

            # 3. Rising Edge Detection
            # We only trigger if the response differs from previous AND we aren't currently moving
            if self._responses_differ(response, self.prev_response):
                print("New target received (Rising Edge)")
                self.prev_response = deepcopy(response)
                
                # Update pending targets for the executor to pick up
                self._pending_target_joint_pos = self.hyrl_joint_pos.copy()
                self._pending_target_gripper_pos = self.hyrl_gripper_pos.copy()
                print(f"pending target gripper pos: {self._pending_target_gripper_pos}")
                
                # Trigger the executor thread
                # if not self.executing_traj:
                #     print("Triggering trajectory execution...")
                self._execute_request.set()
                # else:
                #     print("Ignoring new target - currently executing trajectory")

        # 4. Construct Action
        # This reads the 'current' target, which is being updated smoothly by the executor thread
        with self._target_lock:
            # We copy to ensure thread safety during concatenation below
            left_target = np.array(self._curr_joint_target, dtype=np.float32)

        # Note: _curr_joint_target already includes the interpolated gripper value
        # from the _execute_waypoints loop, so we don't need to manually override it here
        # unless manual slider control is desired when IDLE.
        
        action: Dict[str, Dict[str, np.ndarray]] = {"left": {"pos": left_target}}

        if self.bimanual:
            assert "right" in self.ik.joints
            right_target = np.asarray(self.ik.joints["right"], dtype=np.float32)
            right_target[-1] = self.right_gripper_slider_handle.value
            action["right"] = {"pos": right_target}

        # print(f"gripper action: {action['left']['pos'][-1]}")

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