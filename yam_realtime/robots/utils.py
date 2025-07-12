import csv
import logging
import os
import signal
import sys
import time
from typing import List, Optional

import numpy as np

from yam_realtime.robots.robot import Robot

TIMEOUT_INIT = False


class Timeout:
    def __init__(self, seconds: float, name: Optional[str] = None, mode: str = "error"):
        """
        Initialize the Timeout context manager.

        :param seconds: Timeout duration in seconds.
        :param name: Optional name for the operation.
        :param mode: Timeout mode. Either 'error' to raise an exception or 'warning' to print a warning.
        """
        self.seconds = seconds
        self.name = name
        self.mode = mode.lower()
        if self.mode not in {"error", "warning"}:
            raise ValueError("Mode must be either 'error' or 'warning'")

    def handle_timeout(self, signum: int, frame: Optional[object]) -> None:
        """
        Handle the timeout event.
        """
        if self.mode == "error":
            if self.name:
                raise TimeoutError(f"Operation '{self.name}' timed out after {self.seconds} seconds")
            else:
                raise TimeoutError(f"Operation timed out after {self.seconds} seconds")
        elif self.mode == "warning":
            message = "\033[91m[WARNING]\033[0m Operation"
            if self.name:
                message += f" '{self.name}'"
            message += f" exceeded {self.seconds} seconds but continues."
            print(message, file=sys.stderr)

    def __enter__(self):
        """
        Enter the context and set the timeout alarm.
        """
        global TIMEOUT_INIT
        if not TIMEOUT_INIT:
            TIMEOUT_INIT = True
        else:
            raise NotImplementedError("Nested timeouts are not supported")
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)  # type: ignore

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context and clear the timeout alarm.
        """
        global TIMEOUT_INIT
        TIMEOUT_INIT = False
        signal.alarm(0)  # Disable the alarm


class Rate:
    def __init__(self, rate: Optional[float], rate_name: Optional[str] = None):
        self.last = time.time()
        self.rate = rate  # when rate is None, it means we are not using rate control
        self.rate_name = rate_name

    @property
    def dt(self) -> float:
        if self.rate is None:
            return 0.0
        return 1.0 / self.rate

    def sleep(self) -> None:
        if self.rate is None:
            return
        if self.last + self.dt < time.time() - 0.001:
            logging.warning(
                f"Already behind schedule {self.rate_name} by {time.time() - (self.last + self.dt)} seconds"
            )
        else:
            needed_sleep = max(0, self.last + self.dt - time.time() - 0.0001)  # 0.0001 is the time it takes to sleep
            time.sleep(needed_sleep)
        self.last = time.time()


def easeInOutQuad(t):
    t *= 2
    if t < 1:
        return t * t / 2
    else:
        t -= 1
        return -(t * (t - 2) - 1) / 2


def small_joint_motion_control(robot: Robot) -> None:
    steps = 100

    target_joint_poses = np.array([np.pi / 10, 0, 0, np.pi / 10, np.pi / 10, np.pi / 10, 1])
    start_time = time.time()
    for i in range(steps):
        cmd = np.zeros(7)
        cmd = easeInOutQuad(float(i) / steps) * target_joint_poses
        robot.command_joint_pos(cmd)
        time.sleep(0.01)  # Assuming a control loop time step
    print(f"current joint_pos: {robot.get_joint_pos()}")
    for i in range(steps):
        cmd = np.zeros(7)
        cmd = easeInOutQuad(1 - float(i) / steps) * target_joint_poses
        robot.command_joint_pos(cmd)
        time.sleep(0.01)  # Assuming a control loop time step

    end_time = time.time()
    loop_duration = end_time - start_time
    frequency = steps * 2 / loop_duration
    print(f"Loop frequency: {frequency:.2f} Hz")

    # Print final joint positions
    final_joint_positions = robot.get_joint_pos()
    print(f"Final joint positions: {final_joint_positions}")


YAM_ARM_MOTIONS = [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    np.array([2.93, 0.05, 0.60, -0.08, 0.15, 2.07, 0.75]),
    np.array([3.06, 1.68, 3.14, 1.58, 1.57, -2.07, 1.0]),
    np.array([0.06, 0.68, 1.14, 0.58, 0.57, 0.07, 0.75]),
    np.array([-2.60, 0.0, 0.0, 1.58, -1.57, 1.40, 1.0]),
    np.array([-2.61, 0.0, 3.14, -1.59, -1.57, -2.06, 0.75]),
    np.array([0.12, 0.0, 0.74, 1.58, -1.57, 2.07, 1.0]),
]


def aging_motion_control(
    robot: Robot,
    motion_list: List[np.ndarray],
    motion_duration: float = 5.0,
    save_log: bool = True,
    log_trajectories: Optional[str] = None,
) -> None:
    """
    Perform aging motion control on a robot.

    Args:
        robot (Robot): The robot to control.
        motion_duration (float): Duration of each motion in seconds.
        motion_list (List[np.ndarray]): List of target joint positions.
        save_log (bool): If True, save a log file with cycle counts and timestamps.
        log_trajectories (str): Directory path to save trajectory data. If None, no trajectory data will be saved.
                               If provided, trajectory data (position, velocity, effort) will be logged to CSV files.

    This function will:
    1. Execute a series of motions on the robot continuously
    2. Log cycle counts and timestamps
    3. Optionally log trajectory data (joint positions, velocities, efforts) to CSV files
       when log_trajectories is provided. Data is saved every 100 cycles to reduce overhead.
    """
    # Create log directory if specified and doesn't exist
    if log_trajectories and not os.path.exists(log_trajectories):
        os.makedirs(log_trajectories)

    start_timestamp = time.strftime("%Y%m%d_%H%M%S")
    if log_trajectories:
        save_log_path = os.path.join(log_trajectories, f"aging_motion_log_{start_timestamp}.txt")
    else:
        save_log_path = f"aging_motion_log_{start_timestamp}.txt"

    cycle_count = 0
    sleep_time = 0.01
    num_steps = int(motion_duration / sleep_time)

    while True:
        cycle_start_time = time.strftime("%Y%m%d_%H%M%S")

        for motion in motion_list:
            current_joint_pos = robot.get_joint_pos()

            for i in range(num_steps):
                interpolated_motion = current_joint_pos + i / num_steps * (motion - current_joint_pos)
                robot.command_joint_pos(interpolated_motion)

                # Record observations
                if log_trajectories and (cycle_count % 100 == 0):
                    observations = robot.get_observations()
                    csv_filename = os.path.join(log_trajectories, f"cycle_{cycle_count}_{cycle_start_time}.csv")
                    timestamp = time.time()
                    formatted_time = time.ctime(timestamp)

                    file_exists = os.path.isfile(csv_filename)
                    with open(csv_filename, "a", newline="") as csvfile:
                        csv_writer = csv.writer(csvfile)

                        # Write header if file doesn't exist
                        if not file_exists:
                            joint_count = len(observations["joint_pos"])
                            header = ["timestamp"]

                            header += [f"pos_joint_{j}" for j in range(joint_count)]
                            if "gripper_pos" in observations:
                                header += ["pos_gripper"]

                            if "joint_vel" in observations:
                                header += [f"vel_joint_{j}" for j in range(len(observations["joint_vel"]))]

                            if "joint_eff" in observations:
                                header += [f"eff_joint_{j}" for j in range(len(observations["joint_eff"]))]

                            csv_writer.writerow(header)

                        # Prepare row data
                        row_data = [formatted_time]

                        row_data += observations["joint_pos"].tolist()
                        if "gripper_pos" in observations:
                            row_data += observations["gripper_pos"].tolist()

                        if "joint_vel" in observations:
                            row_data += observations["joint_vel"].tolist()

                        if "joint_eff" in observations:
                            row_data += observations["joint_eff"].tolist()

                        csv_writer.writerow(row_data)

                time.sleep(sleep_time)

        # Increment cycle count
        cycle_count += 1
        timestamp = time.ctime()

        # Print the cycle count
        print(f"Cycle: {cycle_count}, Timestamp: {timestamp}")

        if save_log:
            with open(save_log_path, "a") as f:
                f.write(f"Cycle: {cycle_count}, Timestamp: {timestamp}\n")


def apply_offset_and_sign(joint_pos: np.ndarray, joint_offsets: np.ndarray, joint_signs: np.ndarray) -> np.ndarray:
    # to better match human intuition, we first adjust joint direction with signs, then apply the offset
    return joint_offsets + joint_signs * joint_pos
