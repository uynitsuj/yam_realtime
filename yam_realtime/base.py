"""
Abstract base class for bimanual robot headless visualization.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import viser
import viser.extras
import viser.transforms as vtf

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description as load_urdf_robot_description
except ImportError:
    print("ImportError: robot_descriptions not found:")
    print("pip install git+https://github.com/robot-descriptions/robot_descriptions.py.git@c99c3322c7bfdea65de4c8294573a823d0748bc3")
    exit()


@dataclass
class TransformHandle:
    """Data class to store transform handles."""

    tcp_offset_frame: viser.FrameHandle
    control: Optional[viser.TransformControlsHandle] = None

class ViserAbstractBase(ABC):
    """
    Abstract base class for bimanual robot visualization.
    - This class provides common functionality for different IK solvers
    - Subclasses must implement the solve_ik method with their specific solver
    - robot_description: the name of the robot description to load (default: rby1_description)
    """

    def __init__(
        self,
        rate: float = 100.0,
        viser_server: Optional[viser.ViserServer] = None,
        robot_description: str = "yam_description",
    ):
        self.rate = rate

        # Load URDF for visualization
        self.urdf = load_urdf_robot_description(robot_description)

        # Initialize viser server
        self.viser_server = viser_server if viser_server is not None else viser.ViserServer()

        # Initialize joint configuration
        self.joints = None

        # Allow subclasses to do solver-specific setup first
        self._setup_solver_specific()

        # Setup common components
        self._setup_visualization()

        self._setup_gui()
        self._setup_transform_handles()

    def _setup_visualization(self):
        """Setup basic visualization elements."""
        # Add base frame and robot URDF
        self.base_frame = self.viser_server.scene.add_frame("/base", show_axes=False)
        self.urdf_vis_left = viser.extras.ViserUrdf(self.viser_server, self.urdf, root_node_name="/base")

        # Add ground grid
        self.viser_server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    def _setup_gui(self):
        """Setup GUI elements."""
        # Add timing display
        self.timing_handle = self.viser_server.gui.add_number("Time (ms)", 0.01, disabled=True)

        # Add gizmo size control
        self.tf_size_handle = self.viser_server.gui.add_slider("Gizmo size", min=0.05, max=0.4, step=0.01, initial_value=0.2)

        # Add reset button
        self.reset_button = self.viser_server.gui.add_button("Reset to Rest Pose")

        @self.reset_button.on_click
        def _(_):
            self.home()

    def _setup_transform_handles(self):
        """Setup transform handles for end effectors."""
        self.transform_handles = {
            "left": TransformHandle(
                tcp_offset_frame=self.viser_server.scene.add_frame(
                    "target_left/tcp_offset", show_axes=False, position=(0.0, 0.0, 0.0), wxyz=(1, 0, 0, 0)
                ),
                control=self.viser_server.scene.add_transform_controls("target_left", scale=self.tf_size_handle.value),
            ),
            "right": TransformHandle(
                tcp_offset_frame=self.viser_server.scene.add_frame(
                    "target_right/tcp_offset", show_axes=False, position=(0.0, 0.0, 0.0), wxyz=(1, 0, 0, 0)
                ),
                control=self.viser_server.scene.add_transform_controls("target_right", scale=self.tf_size_handle.value),
            ),
        }

        # Update transform handles when size changes
        @self.tf_size_handle.on_update
        def update_tf_size(_):
            for handle in self.transform_handles.values():
                if handle.control:
                    handle.control.scale = self.tf_size_handle.value
            # Let subclasses handle their own optional handles
            self._update_optional_handle_sizes()

        # Initialize transform handle positions
        self._initialize_transform_handles()

    @property
    def urdf_joint_names(self):
        """Get URDF joint names."""
        return self.urdf.joint_names

    @abstractmethod
    def _update_optional_handle_sizes(self):
        """Override in subclasses to update optional handle sizes."""
        pass

    def update_visualization(self):
        """Update visualization with current state."""
        if self.joints is not None:
            # Update robot configuration
            self.urdf_vis_left.update_cfg(self.joints)

    def get_target_poses(self):
        """Get target poses from transform controls."""
        target_poses = {}

        for side, handle in self.transform_handles.items():
            if handle.control is None:
                continue

            # Combine control handle with TCP offset
            control_tf = vtf.SE3(np.array([*handle.control.wxyz, *handle.control.position]))

            tcp_offset_tf = vtf.SE3(np.array([*handle.tcp_offset_frame.wxyz, *handle.tcp_offset_frame.position]))

            target_poses[side] = control_tf @ tcp_offset_tf

        return target_poses

    def set_ee_targets(self, left_wxyz_xyz: np.ndarray, right_wxyz_xyz: np.ndarray):
        """
        Set end effector targets.
        left_wxyz_xyz: [wxyz, xyz]
        right_wxyz_xyz: [wxyz, xyz]
        """
        self.transform_handles["left"].control.wxyz = left_wxyz_xyz[:4]  # type: ignore
        self.transform_handles["right"].control.wxyz = right_wxyz_xyz[:4]  # type: ignore
        self.transform_handles["left"].control.position = left_wxyz_xyz[4:]  # type: ignore
        self.transform_handles["right"].control.position = right_wxyz_xyz[4:]  # type: ignore

    @abstractmethod
    def home(self):
        """Reset robot to rest pose. Must be implemented by subclasses."""
        pass

    def run(self):
        """Main run loop."""
        while True:
            start_time = time.time()

            self.solve_ik()
            self.update_visualization()

            # Update timing
            elapsed_time = time.time() - start_time
            if hasattr(self, "timing_handle"):
                self.timing_handle.value = 0.99 * self.timing_handle.value + 0.01 * (elapsed_time * 1000)

    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def _setup_solver_specific(self):
        """Setup solver-specific components. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _initialize_transform_handles(self):
        """Initialize transform handle positions. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def solve_ik(self):
        """Solve inverse kinematics. Must be implemented by subclasses."""
        pass
