"""
Main launch script for YAM realtime robot control environment.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import tyro

from robots_realtime.agents.agent import Agent
from robots_realtime.envs.configs.instantiate import instantiate
from robots_realtime.envs.configs.loader import DictLoader
from robots_realtime.envs.robot_env import RobotEnv
from robots_realtime.robots.robot import Robot
from robots_realtime.robots.utils import Rate, Timeout
from robots_realtime.sensors.cameras.camera import CameraDriver
from robots_realtime.utils.launch_utils import (
    cleanup_processes,
    initialize_agent,
    initialize_robots,
    initialize_sensors,
    setup_can_interfaces,
    setup_logging,
    run_server_proc,
)


@dataclass
class LaunchConfig:
    hz: float = 30.0
    cameras: Dict[str, Tuple[CameraDriver, int]] = field(default_factory=dict)
    robots: Dict[str, Union[str, Robot]] = field(default_factory=dict)
    max_steps: Optional[int] = None  # this is for testing
    save_path: Optional[str] = None
    station_metadata: Dict[str, str] = field(default_factory=dict)
    sim_mode: bool = False  # skip CAN/sensors, instantiate robots & agent in-process


@dataclass
class Args:
    config_path: Tuple[str, ...] = ("~/yam_realtime/configs/yam_viser_bimanual.yaml",)


def main(args: Args) -> None:
    """
    Main launch entrypoint.

    1. Load configuration from yaml file
    2. Initialize sensors (cameras, force sensors, etc.)
    3. Setup CAN interfaces (for YAM communication)
    4. Initialize robots (hardware interface)
    5. Initialize agent (e.g. teleoperated control, policy control, etc.)
    6. Create environment
    7. Run control loop
    """

    # Setup logging and get logger
    logger = setup_logging()
    logger.info("Starting realtime control system...")

    server_processes = []

    try:
        logger.info("Loading configuration...")
        configs_dict = DictLoader.load([os.path.expanduser(x) for x in args.config_path])

        agent_cfg = configs_dict.pop("agent")
        sensors_cfg = configs_dict.pop("sensors", None)
        api_servers = configs_dict.pop("api_servers", None)

        server_procs = []

        if api_servers is not None:
            for api_server in api_servers:
                server_proc = run_server_proc(api_server)
                print(f"API server {api_server} started")
                server_procs.append(server_proc)
        main_config = instantiate(configs_dict)

        # ----- Sim mode: everything runs in-process, no CAN/portal ----- #
        if main_config.sim_mode:
            logger.info("Running in sim mode (no CAN, no portal RPC)...")

            # Robots are already instantiated by instantiate() since they
            # were _target_ dicts in the YAML.
            robots = main_config.robots
            agent = instantiate(agent_cfg)

            logger.info("Starting sim control loop at %.1f Hz...", main_config.hz)
            _run_sim_control_loop(robots, agent, main_config)
            return

        # ----- Real hardware mode (original path) ----- #
        logger.info("Initializing sensors...")
        camera_dict, camera_info = initialize_sensors(sensors_cfg, server_processes)

        setup_can_interfaces()

        logger.info("Initializing robots...")
        robots = initialize_robots(main_config.robots, server_processes)

        agent = initialize_agent(agent_cfg, server_processes)

        logger.info("Creating robot environment...")
        frequency = main_config.hz
        rate = Rate(frequency, rate_name="control_loop")

        env = RobotEnv(
            robot_dict=robots,
            camera_dict=camera_dict,
            control_rate_hz=rate,
        )

        logger.info("Starting control loop...")
        _run_control_loop(env, agent, main_config)

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise e
    finally:
        # Cleanup
        logger.info("Shutting down...")
        if "env" in locals():
            env.close()
        if "agent" in locals():
            cleanup_processes(agent, server_processes)


def _run_sim_control_loop(
    robots: Dict[str, Robot],
    agent: Agent,
    config: LaunchConfig,
) -> None:
    """Simplified control loop for sim mode (no portal, no cameras).

    Runs entirely in-process so the MuJoCo viewer stays on the main thread.
    """
    logger = logging.getLogger(__name__)
    rate = Rate(config.hz, rate_name="sim_control_loop")
    steps = 0
    start_time = time.time()
    loop_count = 0

    # Build initial observation from robots
    obs = {name: robot.get_observations() for name, robot in robots.items()}
    obs["timestamp"] = time.time()

    try:
        while True:
            # Check if any sim viewer has been closed
            for robot in robots.values():
                if hasattr(robot, "is_viewer_running") and not robot.is_viewer_running():
                    logger.info("Viewer closed, stopping...")
                    return

            action = agent.act(obs)

            # Apply actions directly
            for name, act in action.items():
                if name in robots:
                    robots[name].command_joint_pos(act["pos"])

            rate.sleep()

            # Collect observations
            obs = {name: robot.get_observations() for name, robot in robots.items()}
            obs["timestamp"] = time.time()

            steps += 1
            loop_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1:
                logger.info(f"Sim control loop: {loop_count / elapsed_time:.2f} Hz")
                start_time = time.time()
                loop_count = 0

            if config.max_steps is not None and steps >= config.max_steps:
                logger.info(f"Reached max steps ({config.max_steps}), stopping...")
                break
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        if hasattr(agent, "close"):
            agent.close()
        for robot in robots.values():
            if hasattr(robot, "close"):
                robot.close()


def _run_control_loop(env: RobotEnv, agent: Agent, config: LaunchConfig) -> None:
    """
    Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent instance
        config: Configuration object
    """
    logger = logging.getLogger(__name__)
    steps = 0
    start_time = time.time()
    loop_count = 0

    # Init environment and warm up agent
    obs = env.reset()
    logger.info(f"Action spec: {env.action_spec()}")
    agent.act(obs)

    # Main control loop
    while True:
        # Get action from agent
        with Timeout(30, "Agent action"):
            action = agent.act(obs)

        # Execute action in environment
        with Timeout(1, "Env step", "warning"):
            obs = env.step(action)

        steps += 1
        loop_count += 1

        elapsed_time = time.time() - start_time
        if elapsed_time >= 1:
            calculated_frequency = loop_count / elapsed_time
            logger.info(f"Control loop frequency: {calculated_frequency:.2f} Hz")
            start_time = time.time()
            loop_count = 0

        if config.max_steps is not None and steps >= config.max_steps:
            logger.info(f"Reached max steps ({config.max_steps}), stopping...")
            break


if __name__ == "__main__":
    main(tyro.cli(Args))
