import logging
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import omegaconf
import portal
import tyro

from yam_realtime.configs.instantiate import instantiate
from yam_realtime.camera.camera import CameraDriver
from yam_realtime.configs.loader import DictLoader
from yam_realtime.envs.robot_env import RobotEnv
from yam_realtime.robots.robot import ROBOT_PROTOCOL_METHODS, Robot
from yam_realtime.robots.utils import Rate, Timeout
from yam_realtime.utils.portal_utils import (
    Client,
    RemoteServer,
    launch_remote_get_local_handler,
)

@dataclass
class LaunchConfig:
    hz: float = 30.0
    cameras: Dict[str, Tuple[CameraDriver, int]] = field(default_factory=dict)
    robots: Dict[str, Union[str, Robot]] = field(default_factory=dict)
    max_steps: Optional[int] = None  # this is for testing
    save_path: Optional[str] = None
    station_metadata: Dict[str, str] = field(default_factory=dict)

@dataclass
class Args:
    config_path: Tuple[str, ...] = ("~/yam_realtime/configs/yam_viser.yaml",)

def main(args: Args) -> None:
    # Setup logging
    server_processes = []
    _launch_remote_get_local_handler = partial(
        launch_remote_get_local_handler,
        launch_remote=True,
        process_pool=server_processes,
    )
    # Load configuration from YAML
    configs_dict = DictLoader.load([os.path.expanduser(x) for x in args.config_path])

    agent_cfg = configs_dict.pop("agent")

    camera_dict = {}
    camera_info = {}

    if "sensors" in configs_dict:
        sensors_cfg = configs_dict.pop("sensors")
        for sensor_name, sensor_cfg in sensors_cfg.items():
            if sensor_name == "cameras":
                if sensor_cfg is not None:
                    for camera_name, camera_config in sensor_cfg.items():
                        camera_config["camera"]["name"] = camera_name
                        _, client = _launch_remote_get_local_handler(camera_config)
                        camera_dict[camera_name] = client
                        if "get_camera_info" in client.supported_remote_methods:  # type: ignore
                            camera_info[camera_name] = client.get_camera_info()
                        else:
                            raise AttributeError("Remote does not implement method 'get_camera_info'!")

    config: LaunchConfig = instantiate(configs_dict)


    # make sure all can interfaces are up.
    subprocess.run(["bash", "yam_realtime/scripts/reset_all_can.sh"], check=True)
    time.sleep(0.5)

    # launch agent after can is up
    if "Client" in agent_cfg["_target_"]:
        agent = instantiate(agent_cfg)
    else:
        # Define the agent methods that need to be remotely accessible
        agent_remote_methods = {
            "act": False, 
            "reset": False, 
            "close": False,
        }
        _, agent = _launch_remote_get_local_handler(agent_cfg, custom_remote_methods=agent_remote_methods)

    robots: Dict[str, Robot] = {}
    for robot_name, robot_path_or_robot in config.robots.items():
        if (
            isinstance(robot_path_or_robot, str)
            or isinstance(robot_path_or_robot, omegaconf.listconfig.ListConfig)
            or isinstance(robot_path_or_robot, list)
        ):
            if isinstance(robot_path_or_robot, omegaconf.listconfig.ListConfig):
                robot_path_or_robot = list(robot_path_or_robot)  # Convert to list of strings
            try:
                robot_dict = DictLoader.load(robot_path_or_robot)
            except Exception as e:
                logging.error(f"Failed to load robot config: {robot_path_or_robot}")
                raise
            if "Client" in robot_dict["_target_"]:
                _robot = instantiate(robot_dict)
            else:
                _, _robot = launch_remote_get_local_handler(
                    robot_dict,
                    process_pool=server_processes,
                    custom_remote_methods=ROBOT_PROTOCOL_METHODS,
                )
        elif isinstance(robot_path_or_robot, Robot):
            port = portal.free_port()

            def _l(robot: Any, por: int) -> None:
                remote_server = RemoteServer(robot, por, custom_remote_methods=ROBOT_PROTOCOL_METHODS)
                remote_server.serve()

            p = portal.Process(partial(_l, robot=robot_path_or_robot, por=port), start=True)
            _robot = Client(port)
        else:
            raise ValueError(f"Invalid robot path or robot: {robot_path_or_robot}")
        robots[robot_name] = _robot  # type: ignore

    frequency = config.hz
    rate = Rate(frequency, rate_name="control_loop")

    env = RobotEnv(
        robot_dict=robots,
        camera_dict=camera_dict,
        control_rate_hz=rate,
    )

    steps = 0

    start_time = time.time()
    loop_count = 0
    # warm up the agent.
    obs = env.reset()
    print(env.action_spec())
    agent.act(obs)
    try:
        while True:
            with Timeout(30, "Agent action"):
                # print(f"sending the policy server the obs")
                action = agent.act(obs)
            with Timeout(1, "Env step", "warning"):
                obs = env.step(action)  # type: ignore

            steps += 1
            loop_count += 1
            # Check if 10 seconds have passed
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1:
                calculated_frequency = loop_count / elapsed_time
                logging.info("Env loop frequency: %.2f iterations per second", calculated_frequency)
                start_time = time.time()  # Reset the timer
                loop_count = 0  # Reset the loop count

            if config.max_steps is not None and steps >= config.max_steps:
                break
    except Exception as e:
        logging.error("Error: %s", e)
        raise e
    finally:
        logging.info("done")

        env.close()  # close camera before kill all spawned portal processes

        # Ensure processes are terminated
        agent.close()
        for server_process in server_processes:
            server_process.kill()

        logging.info("Processes terminated.")


if __name__ == "__main__":
    main(tyro.cli(Args))
