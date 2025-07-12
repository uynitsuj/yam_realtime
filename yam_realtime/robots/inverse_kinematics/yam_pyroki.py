"""
Bimanual YAM arms Inverse Kinematics Example using PyRoki with ViserAbstractBase.
"""

import time
from typing import Dict, Optional

import numpy as np
from copy import deepcopy

try:
    import pyroki as pk
except ImportError:
    print("ImportError: pyroki not found:")
    print("pip install git+https://github.com/chungmin99/pyroki.git")
    exit()

import viser
import viser.extras
import viser.transforms as vtf

from yam_realtime.robots.inverse_kinematics.pyroki_snippets._solve_ik import solve_ik

from yam_realtime.robots.viser.viser_base import ViserAbstractBase, TransformHandle


class YamPyroki(ViserAbstractBase):
    """
    Bimanual YAM robot visualization using PyRoki for inverse kinematics.
    Uses a single robot with two target links for bimanual operation.
    """

    def __init__(
        self,
        rate: float = 100.0,
        viser_server: Optional[viser.ViserServer] = None,
        bimanual: bool = False,
    ):
        # For bimanual operation, we'll use a single robot with two target links
        self.robot: Optional[pk.Robot] = None
        self.target_link_names = ["link_6"]
        self.joints = {"left": np.zeros(6)}
        if bimanual:
            self.target_link_names = self.target_link_names * 2
            self.joints["right"] = np.zeros(6)
        super().__init__(rate, viser_server, bimanual=bimanual)
    
    def _setup_visualization(self):
        super()._setup_visualization()
        if self.bimanual:
            self.base_frame_right = self.viser_server.scene.add_frame("/base/base_right", show_axes=False)
            self.base_frame_right.position = (0.0, -0.61, 0.0)
            self.urdf_vis_right = viser.extras.ViserUrdf(self.viser_server, deepcopy(self.urdf), root_node_name="/base/base_right")

    def _setup_solver_specific(self):
        """Setup PyRoki-specific components."""
        # Create a single robot for bimanual operation
        self.robot = pk.Robot.from_urdf(self.urdf)
        
        # Get rest pose from URDF
        self.rest_pose = self.urdf.cfg
        
    def _initialize_transform_handles(self):
        """Initialize transform handle positions for both arms."""
        # Set initial positions for left and right arms
        if self.transform_handles["left"].control is not None:
            self.transform_handles["left"].control.position = (0.25, 0.0, 0.26)
            self.transform_handles["left"].control.wxyz = vtf.SO3.from_rpy_radians(np.pi/2, 0.0, np.pi/2).wxyz
            self.transform_handles["left"].tcp_offset_frame.position = (0.0, 0.04, -0.13) # YAM gripper end is slightly offset from the end of the link_6

        if self.bimanual:
            if self.transform_handles["right"].control is not None:
                self.transform_handles["right"].control.remove()
                self.transform_handles["right"].tcp_offset_frame.remove()
            self.transform_handles["right"] = TransformHandle(
                    tcp_offset_frame=self.viser_server.scene.add_frame(
                        "/base/base_righttarget_right/tcp_offset", show_axes=False, position=(0.0, 0.04, -0.13), wxyz=vtf.SO3.from_rpy_radians(0.0, 0.0, 0.0).wxyz
                    ),
                    control=self.viser_server.scene.add_transform_controls("/base/base_right/target_right", scale=self.tf_size_handle.value, position=(0.25, 0.0, 0.26), wxyz=vtf.SO3.from_rpy_radians(np.pi/2, 0.0, np.pi/2).wxyz),
                )

    def _update_optional_handle_sizes(self):
        """Update optional handle sizes (none for this implementation)."""
        pass

    def solve_ik(self):
        """Solve inverse kinematics for both arms using multiple targets."""
        if self.robot is None:
            return
            
        target_poses = self.get_target_poses()
        
        # Check if we have both left and right targets
        if self.bimanual:
            if "left" not in target_poses or "right" not in target_poses:
                return
        else:
            if "left" not in target_poses:
                return
        
        # Prepare target data for both arms
        target_positions = []
        target_wxyzs = []
        for idx, side in enumerate(self.get_target_poses().keys()):
            target_tf = target_poses[side]
            target_positions.append(target_tf.translation())
            target_wxyzs.append(target_tf.rotation().wxyz)
            
            # Solve IK for all arms
            # try:
            solution = solve_ik(
                robot=self.robot,
                target_link_name=self.target_link_names[idx],
                target_position=target_tf.translation(),
                target_wxyz=target_tf.rotation().wxyz,
            )
            self.joints[side] = solution
            # except Exception as e:
            #     print(f"IK solving failed: {e}")
            #     # Keep previous solution if IK fails
            #     pass
        

    def update_visualization(self):
        """Update visualization with current joint configurations."""
        if self.joints is not None:
            self.urdf_vis_left.update_cfg(self.joints["left"])
            if self.bimanual:
                self.urdf_vis_right.update_cfg(self.joints["right"])

    def home(self):
        """Reset both arms to rest pose."""
        self.joints["left"] = self.rest_pose
        if self.bimanual:
            self.joints["right"] = self.rest_pose
        
        # Reset transform handles to initial positions
        self._initialize_transform_handles()
        
        # Update visualization with rest pose
        self.urdf_vis_left.update_cfg(self.rest_pose)
        if self.bimanual:
            self.urdf_vis_right.update_cfg(self.rest_pose)

    def get_joint_positions(self) -> Optional[np.ndarray]:
        """Get current joint positions for the bimanual robot."""
        if self.bimanual:
            if self.joints["left"] is not None and self.joints["right"] is not None:
                return np.concatenate([self.joints["left"], self.joints["right"]])
            else:
                return None
        else:
            if self.joints["left"] is not None:
                return self.joints["left"]


def main():
    """Main function for YAM IK visualization."""
    # Create visualization
    viz = YamPyroki(rate=100.0)
    
    # Run the visualization loop
    viz.run()


if __name__ == "__main__":
    main()
