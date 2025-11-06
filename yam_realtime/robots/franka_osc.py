# type: ignore
# franka interface is referenced from https://github.com/JeanElsner/panda-py
import logging
import time
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional, Tuple

import numpy as np
from i2rt.utils.utils import RateRecorder
from i2rt.robots.robot import Robot
from scipy.spatial.transform import Rotation as R

from yam_realtime.robots.utils import Rate

logger = logging.getLogger(__name__)

###############################################################################
# Single Kp, Kd for both position and orientation in OSC
###############################################################################
KP_OSC = 130.0
KD_OSC = 30.0

# We'll build 6D arrays for the translational + rotational dimensions:
KP_6D = np.full(6, KP_OSC)  # [100, 100, 100, 100, 100, 100]
KD_6D = np.full(6, KD_OSC)  # [ 20,  20,  20,  20,  20,  20]

# Joint impedance for null space (example, can be changed)
# Kp_null = np.array([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
# Kp_null = np.array([30.0, 30.0, 25.0, 25.0, 20.0, 10.0, 10.0])
Kp_null = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
damping_ratio = 8.0
Kd_null = damping_ratio * 2.0 * np.sqrt(Kp_null)


def orientation_error(current_rot: np.ndarray, desired_rot: np.ndarray) -> np.ndarray:
    """
    Compute orientation error in axis-angle form (3-vector).
    """
    q_current = R.from_matrix(current_rot).as_quat()
    q_desired = R.from_matrix(desired_rot).as_quat()
    q_err = R.from_quat(q_desired) * R.from_quat(q_current).inv()
    return q_err.as_rotvec()


class FrankaPanda(Robot):
    def __init__(
        self,
        host_name: str = "172.16.0.2",
        username: Optional[str] = None,
        password: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the Franka Panda robot arm without gripper.

        Args:
            host_name: str
                The IP address of the robot controller
            username: str
                The username for robot controller login
            password: str
                The password for robot controller login
        """
        # Store initialization parameters for reinit
        self._init_params = {
            "host_name": host_name,
            "username": username,
            "password": password,
            "name": name,
        }
        self._initialize(host_name, username, password, name)

    def _initialize(
        self,
        host_name: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Internal method to handle the actual initialization."""
        import panda_py
        from panda_py import controllers


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

        # reduce collision sensitivity for enabling contact rich behavoir
        self.torque_limit = 9.0
        self.interface.get_robot().set_collision_behavior([100.0] * 7, [100.0] * 7, [100.0] * 6, [100.0] * 6)
        self.model = self.interface.get_model()
        self.frame = panda_py.libfranka.Frame.kFlange

        self.ctrl = controllers.AppliedTorque()
        self.interface.start_controller(self.ctrl)

        self._cmd_lock = Lock()
        self._joint_cmd = self.get_joint_pos()
        self._server_thread = Thread(target=self.run, name="control_loop")
        self._stop_event = Event()  # Add a stop event
        self._server_thread.start()
        self._update_rate = RateRecorder(name="update_rate")
        self._update_rate.start()

    def __repr__(self) -> str:
        return f"FrankaPanda(name={self.name}, host_name={self.host_name})"

    def get_robot_info(self) -> Dict[str, Any]:
        return {
            "kp_6d": KP_6D,
            "kd_6d": KD_6D,
            "kp_null": Kp_null,
            "kd_null": Kd_null,
            "damping_ratio": damping_ratio,
        }

    def run(self) -> None:
        rate = Rate(300, rate_name="franka_osc_control_loop")
        with RateRecorder(name=self) as rec:
            with self.interface.create_context(frequency=300) as ctx:
                while not self._stop_event.is_set():
                    rate.sleep()
                    rec.track()
                    # extract joint command and current state
                    with self._cmd_lock:
                        joint_cmd = self._joint_cmd.copy()

                    state = self.interface.get_state()
                    self.state = state

                    q = state.q
                    dq = state.dq

                    # compute current & desired end-effector pose
                    Te = self.fk(q)  # current
                    Tep = self.fk(joint_cmd)  # desired

                    # position error
                    x_cur = Te[:3, 3]
                    x_des = Tep[:3, 3]
                    dx = x_des - x_cur

                    # orientation error
                    R_cur = Te[:3, :3]
                    R_des = Tep[:3, :3]
                    dtheta = orientation_error(R_cur, R_des)

                    # twist error
                    err_6d = np.concatenate([dx, dtheta])

                    # mass matrix and augmentation
                    Mq = self.model.mass(state)
                    Mq = np.array(Mq).reshape(7, 7)
                    # Add 0.15 to last 3 diagonal entries
                    for i in range(4, 7):
                        Mq[i, i] += 0.15
                    Mq_inv = np.linalg.inv(Mq)

                    # task jacobian and inertia
                    J = self.model.zero_jacobian(self.frame, state)
                    J = np.array(J).reshape(7, 6).transpose()

                    J_Minv_JT = J @ Mq_inv @ J.T
                    if abs(np.linalg.det(J_Minv_JT)) >= 1e-2:
                        Mx = np.linalg.inv(J_Minv_JT)
                    else:
                        Mx = np.linalg.pinv(J_Minv_JT, rcond=1e-2)

                    # task-space
                    dX = J @ dq
                    F_task = KP_6D * err_6d - KD_6D * dX
                    tau_task = J.T @ (Mx @ F_task)

                    # null-space
                    dq_null = -Kp_null * (q - joint_cmd) - Kd_null * dq
                    Jbar = Mq_inv @ J.T @ Mx
                    tau_null = (np.eye(self._num_dofs) - J.T @ Jbar.T) @ dq_null

                    # command torque
                    tau = tau_task + tau_null
                    # print(f"tau: {tau}")
                    tau = np.clip(tau, -self.torque_limit, self.torque_limit)
                    print(f"tau: {tau}")
                    self.ctrl.set_control(tau)

                    if self._joint_state_saver is not None:
                        self._joint_state_saver.add(
                            time.time(),
                            pos=self.state.q,
                            vel=self.state.dq,
                            eff=tau,
                        )

    def enable_arm(self) -> None:
        self.interface.teaching_mode(False)

    def disable_arm(self) -> None:
        self.interface.teaching_mode(True)

    def num_dofs(self) -> int:
        return self._num_dofs

    def get_joint_pos(self) -> np.ndarray:
        return self.interface.q

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert len(joint_pos) == self._num_dofs, (
            f"Joint position array length mismatch. num_dofs: {self._num_dofs}, joint_pos: {len(joint_pos)}."
        )
        assert len(joint_pos) == 7, "Joint command must have 7 elements."
        self._update_rate.track()
        with self._cmd_lock:
            self._joint_cmd = joint_pos

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {
            "joint_pos": self.state.q,
            "joint_vel": self.state.dq,
            "joint_eff": self.state.tau_J,
        }


    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the runtime context related to this object."""
        self.close()

    def close(self) -> None:
        """Safely close the robot by setting all torques to zero and cleaning up resources."""
        # Set torques to zero first
        self.ctrl.set_control(np.zeros(self._num_dofs))
        # Signal the thread to stop and wait for it to finish
        self._stop_event.set()
        if self._server_thread.is_alive():
            self._server_thread.join()
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

