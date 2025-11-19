# type: ignore
# franka interface is referenced from https://github.com/JeanElsner/panda-py
import logging
import time
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional

import numpy as np
import panda_py
from i2rt.robots.robot import Robot
from i2rt.utils.utils import RateRecorder
from panda_py import controllers

from robots_realtime.robots.utils import Rate

logger = logging.getLogger(__name__)

###############################################################################
# Joint Position Controller Parameters
###############################################################################
# Default stiffness and damping for joint position control
# JOINT_STIFFNESS = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
JOINT_STIFFNESS = np.array([300.0, 300.0, 300.0, 300.0, 125.0, 75.0, 25.0])
JOINT_DAMPING = np.array([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0])
FILTER_COEFF = 1.0

GRIPPER_DEFAULT_SPEED = 10.0
GRIPPER_INITIAL_FORCE = 1.0
GRIPPER_ACTIVE_FORCE = 4.0
GRIPPER_MAX_WIDTH = 0.1
GRIPPER_MOVE_THRESHOLD = 0.055
GRIPPER_COMMAND_EPSILON = 1e-3
GRIPPER_UPDATE_TIMEOUT_S = 0.05


class FrankaPanda(Robot):
    def __init__(
        self,
        host_name: str = "172.16.0.2",
        username: Optional[str] = None,
        password: Optional[str] = None,
        name: Optional[str] = None,
        enable_gripper: bool = False,
    ) -> None:
        """Initialize the Franka Panda robot arm with joint position control.

        Args:
            host_name: str
                The IP address of the robot controller
            username: str
                The username for robot controller login
            password: str
                The password for robot controller login
            enable_gripper: bool
                Whether to enable control for a default Franka Panda gripper
        """
        # Store initialization parameters for reinit
        self._init_params = {
            "host_name": host_name,
            "username": username,
            "password": password,
            "name": name,
        }
        self._initialize(host_name, username, password, name, enable_gripper)

    def _initialize(
        self,
        host_name: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        name: Optional[str] = None,
        enable_gripper: bool = False,
    ) -> None:
        """Internal method to handle the actual initialization."""

        self._joint_state_saver = None
        self.name = name or "franka"
        self.host_name = host_name

        if username and password:
            self.desk = panda_py.Desk(host_name, username, password)
            self.desk.activate_fci()

        self.interface = panda_py.Panda(host_name)
        self.state = self.interface.get_state()
        self.fk = panda_py.fk
        self._num_dofs = 7
        self._stop_event = Event()

        if enable_gripper:
            self.gripper = panda_py.libfranka.Gripper(host_name)
            self.gripper.grasp(0.0, GRIPPER_DEFAULT_SPEED, GRIPPER_INITIAL_FORCE)
            self.gripper.grasp(GRIPPER_MAX_WIDTH, GRIPPER_DEFAULT_SPEED, GRIPPER_INITIAL_FORCE)
            self._num_dofs += 1
            self._gripper_lock = Lock()
            self._gripper_update_event = Event()
            self._gripper_target_width = self.gripper.read_once().width
            self._last_gripper_command = self._gripper_target_width
            self._last_gripper_state = self._last_gripper_command
            self._gripper_thread = Thread(
                target=self._gripper_command_loop,
                name="gripper_command_loop",
                daemon=True,
            )
            self._gripper_thread.start()

        # reduce collision sensitivity for enabling contact rich behavior
        self.interface.get_robot().set_collision_behavior([100.0] * 7, [100.0] * 7, [100.0] * 6, [100.0] * 6)

        # Initialize joint position controller
        self.ctrl = controllers.JointPosition(
            stiffness=JOINT_STIFFNESS,
            damping=JOINT_DAMPING,
            filter_coeff=FILTER_COEFF,
        )
        print("starting controller")

        self.interface.start_controller(self.ctrl)

        print("controller started")

        self._cmd_lock = Lock()
        self._joint_cmd = self.get_joint_pos()
        self._joint_vel_cmd = np.zeros(7)
        self.ctrl_thread_start_time = time.time()
        self._server_thread = Thread(target=self.run, name="control_loop")
        self._server_thread.start()
        self._update_rate = RateRecorder(name="update_rate")
        self._update_rate.start()

    def __repr__(self) -> str:
        return f"FrankaPanda(name={self.name}, host_name={self.host_name})"

    def get_robot_info(self) -> Dict[str, Any]:
        return {
            "joint_stiffness": JOINT_STIFFNESS,
            "joint_damping": JOINT_DAMPING,
            "filter_coeff": FILTER_COEFF,
        }

    def run(self) -> None:
        rate = Rate(300, rate_name="franka_joint_control_loop")
        with RateRecorder(name=self) as rec:
            with self.interface.create_context(frequency=300) as ctx:
                while not self._stop_event.is_set():
                    rate.sleep()
                    rec.track()
                    # extract joint command and current state
                    with self._cmd_lock:
                        if hasattr(self, "gripper"):
                            joint_cmd = self._joint_cmd[:-1].copy()
                        else:
                            joint_cmd = self._joint_cmd.copy()
                        joint_vel_cmd = self._joint_vel_cmd.copy()

                    state = self.interface.get_state()
                    self.state = state

                    # Command the joint position controller
                    self.ctrl.set_control(joint_cmd, joint_vel_cmd)

                    if self._joint_state_saver is not None:
                        self._joint_state_saver.add(
                            time.time(),
                            pos=self.state.q,
                            vel=self.state.dq,
                            eff=self.state.tau_J,
                        )

    def enable_arm(self) -> None:
        self.interface.teaching_mode(False)

    def disable_arm(self) -> None:
        self.interface.teaching_mode(True)

    def num_dofs(self) -> int:
        return self._num_dofs

    def get_joint_pos(self) -> np.ndarray:
        if hasattr(self, "gripper"):
            return np.concatenate([self.interface.q, [self._last_gripper_state]])
        return self.interface.q

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert len(joint_pos) == self._num_dofs, (
            f"Joint position array length mismatch. num_dofs: {self._num_dofs}, joint_pos: {len(joint_pos)}."
        )
        self._update_rate.track()
        with self._cmd_lock:
            self._joint_cmd = joint_pos
            if hasattr(self, "gripper"):
                self._submit_gripper_width(joint_pos[-1])

    def command_joint_vel(self, joint_vel: np.ndarray) -> None:
        """Set desired joint velocities for the position controller.
        
        Args:
            joint_vel: Desired joint velocities (7-element array for arm only)
        """
        assert len(joint_vel) == 7, (
            f"Joint velocity array length mismatch. Expected 7, got {len(joint_vel)}."
        )
        with self._cmd_lock:
            self._joint_vel_cmd = joint_vel

    def get_observations(self) -> Dict[str, np.ndarray]:
        obs = {
            "joint_pos": self.state.q
            if not hasattr(self, "gripper")
            else np.concatenate([self.state.q, [self._last_gripper_state]]),
            "joint_vel": self.state.dq,
            "joint_eff": self.state.tau_J,
        }
        return obs

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the runtime context related to this object."""
        self.close()

    def close(self) -> None:
        """Safely close the robot by stopping the controller and cleaning up resources."""
        # Signal the thread to stop and wait for it to finish
        self._stop_event.set()
        if self._server_thread.is_alive():
            self._server_thread.join()
        if hasattr(self, "_gripper_update_event"):
            self._gripper_update_event.set()
        if hasattr(self, "_gripper_thread") and self._gripper_thread.is_alive():
            self._gripper_thread.join()
        # Stop the controller
        self.interface.stop_controller()
        # Logout from desk
        if hasattr(self, "desk") and self.desk is not None:
            self.desk.logout()

    def reinit(self) -> None:
        """Reinitialize the robot by closing existing connection and creating a new one."""
        logger.info(f"Reinitializing franka panda robot {self.name}")
        self.close()
        # Wait a moment to ensure clean shutdown
        import time

        time.sleep(0.01)

        # Reinitialize with stored parameters
        self._initialize(**self._init_params)
        logger.info(f"Robot {self.name} reinitialized")

    def _gripper_command_loop(self) -> None:
        while not self._stop_event.is_set():
            triggered = self._gripper_update_event.wait(timeout=GRIPPER_UPDATE_TIMEOUT_S)
            if self._stop_event.is_set():
                break
            if not triggered:
                continue
            with self._gripper_lock:
                width = self._gripper_target_width
                self._gripper_update_event.clear()
            if width is None:
                continue
            if (
                self._last_gripper_command is not None
                and abs(self._last_gripper_command - width) < GRIPPER_COMMAND_EPSILON
            ):
                continue
            try:
                if width > GRIPPER_MOVE_THRESHOLD:
                    self.gripper.move(width, GRIPPER_DEFAULT_SPEED)
                else:
                    self.gripper.grasp(width, GRIPPER_DEFAULT_SPEED, GRIPPER_ACTIVE_FORCE)

                self._last_gripper_command = width
                self._last_gripper_state = self.gripper.read_once().width
            except Exception:
                logger.exception("Failed to command gripper to width %s", width)

    def _submit_gripper_width(self, width: float) -> None:
        with self._gripper_lock:
            self._gripper_target_width = width
            self._gripper_update_event.set()

