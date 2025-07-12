import time
from typing import Dict, List, Optional

import numpy as np
import portal
from i2rt.robots.robot import Robot
from i2rt.robots.utils import JointMapper

# RPC Method Serialization Requirements.
ROBOT_PROTOCOL_METHODS = {
    "num_dofs": False,
    "get_joint_pos": False,
    "get_joint_state": False,
    "command_joint_pos": False,
    "command_joint_state": False,
    "get_observations": False,
    "joint_pos_spec": True,
    "joint_state_spec": True,
    "get_robot_info": True,
    "get_robot_type": True,
    "command_target_vel": False,
}


def start_robot_in_init_state(
    robot: "Robot",
    init_state: np.ndarray,
    t: float = 2.0,
    kp: Optional[np.ndarray] = None,
    kd: Optional[np.ndarray] = None,
) -> "Robot":
    """Helper function to start the robot in the given initial state.

    Args:
        robot (Robot): The robot to start.
        init_state (np.ndarray): The initial state to start the robot in.
        t (float): The time to take to reach the initial state.
        kp (Optional[np.ndarray], optional): The proportional gains for the robot. Defaults to None.
        kd (Optional[np.ndarray], optional): The derivative gains for the robot. Defaults to None.
    """
    assert len(init_state) == robot.num_dofs(), (
        f"Expected target init_state to be of length {robot.num_dofs()}, got {len(init_state)} instead."
    )
    for i in range(100):
        current_state = robot.get_joint_pos()
        cmd_state = current_state + (init_state - current_state) * (i / 100)
        joint_state = {
            "pos": cmd_state,
            "vel": np.zeros_like(current_state),
        }
        if kp is not None:
            joint_state["kp"] = kp
        if kd is not None:
            joint_state["kd"] = kd
        robot.command_joint_state(joint_state)
        time.sleep(t / 100)

    joint_state = {
        "pos": init_state,
        "vel": np.zeros_like(init_state),
    }
    robot.command_joint_state(joint_state)
    return robot


class PrintRobot(Robot):
    """A robot that prints the commanded joint state."""

    def __init__(self, num_dofs: int, dont_print: bool = False):
        self._num_dofs = num_dofs
        self._joint_state = np.zeros((num_dofs,))
        self._dont_print = dont_print

    def num_dofs(self) -> int:
        return self._num_dofs

    def get_joint_pos(self) -> np.ndarray:
        return self._joint_state

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert len(joint_pos) == (self._num_dofs), (
            f"Expected joint state of length {self._num_dofs}, got {len(joint_pos)}."
        )
        self._joint_state = joint_pos
        if not self._dont_print:
            print(self._joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_pos = self.get_joint_pos()
        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_pos,
        }


class BimanualRobot(Robot):
    def __init__(self, robot_l: Robot, robot_r: Robot):
        self._robot_l = robot_l
        self._robot_r = robot_r

    def num_dofs(self) -> int:
        return self._robot_l.num_dofs() + self._robot_r.num_dofs()

    def get_joint_pos(self) -> np.ndarray:
        return np.concatenate((self._robot_l.get_joint_pos(), self._robot_r.get_joint_pos()))

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._robot_l.command_joint_pos(joint_pos[: self._robot_l.num_dofs()])
        self._robot_r.command_joint_pos(joint_pos[self._robot_l.num_dofs() :])

    def get_observations(self) -> Dict[str, np.ndarray]:
        l_obs = self._robot_l.get_observations()
        r_obs = self._robot_r.get_observations()
        assert l_obs.keys() == r_obs.keys()
        return_obs = {}
        for k in l_obs.keys():
            try:
                return_obs[k] = np.concatenate((l_obs[k], r_obs[k]))
            except Exception as e:
                print(e)
                print(k)
                print(l_obs[k])
                print(r_obs[k])
                raise RuntimeError() from e

        return return_obs


DEFAULT_ROBOT_PORT = 6000


class ServerRobot:
    def __init__(
        self,
        robot: Robot,
        port: int = DEFAULT_ROBOT_PORT,
    ) -> None:
        self._robot = robot
        self._server = portal.Server(port)
        print(f"Robot Sever Binding to {port}, Robot: {robot}")

        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)

    def serve(self) -> None:
        """Serve the leader robot."""
        self._server.start()


class ClientRobot(Robot):
    """A class representing a client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._client = portal.Client(f"{host}:{port}")

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        return self._client.num_dofs().result()  # type: ignore

    def get_joint_pos(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        return self._client.get_joint_pos().result()  # type: ignore

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_pos (T): The state to command the leader robot to.
        """
        self._client.command_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (Dict[str, np.ndarray]): The state to command the leader robot to.
        """
        self._client.command_joint_state(joint_state)  # type: ignore

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        return self._client.get_observations().result()  # type: ignore


class ConcatenatedRobot(Robot):
    def __init__(self, robots: List[Robot], remapper: Optional[JointMapper] = None):
        self._robots = robots
        self._remapper = remapper
        self.per_robot_index = np.array([i.num_dofs() for i in self._robots]).cumsum()

    def num_dofs(self) -> int:
        return sum(robot.num_dofs() for robot in self._robots)

    def get_joint_pos(self) -> np.ndarray:
        robot_space_joint_pos = np.concatenate([robot.get_joint_pos() for robot in self._robots])
        if self._remapper is not None:
            return self._remapper.to_command_joint_pos_space(robot_space_joint_pos)
        return robot_space_joint_pos

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        if self._remapper is not None:
            joint_pos = self._remapper.to_robot_joint_pos_space(joint_pos)
        for robot, pos in zip(self._robots, np.split(joint_pos, self.per_robot_index)):
            robot.command_joint_pos(pos)

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        assert self._remapper is None, "Remapper is not supported for command_joint_state"
        for robot, state in zip(self._robots, np.split(joint_state, self.per_robot_index)):  # type: ignore
            robot.command_joint_state(state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        obs = [robot.get_observations() for robot in self._robots]
        obs_dict = {}
        for o in obs:
            for k, v in o.items():
                if k in obs_dict:
                    obs_dict[k] = np.concatenate([obs_dict[k], v])
                else:
                    obs_dict[k] = v
        return obs_dict
