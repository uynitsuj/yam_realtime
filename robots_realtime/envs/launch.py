"""
Main launch script for YAM realtime robot control environment.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
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
    run_server_proc,
    setup_can_interfaces,
    setup_logging,
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
    record_path: Optional[str] = None  # if set, enables trajectory logging


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


def _build_log_step(
    obs: Dict,
    action: Dict,
    robot_names,
) -> Dict[str, np.ndarray]:
    """Extract a flat {key: array} dict for the logger from obs + action.

    Gathers:
    - ``state``     — concatenated joint_pos across all robots (float32)
    - ``action``    — concatenated commanded pos across all robots (float32)
    - ``timestamp`` — wall-clock time as a (1,) float64
    - one key per camera for any robot exposing ``get_camera_images()``

    The resulting dict is embodiment-agnostic: the logger never sees robot
    names or DOF counts, only flat arrays.
    """
    state_parts = [obs[n]["joint_pos"] for n in robot_names if n in obs]
    action_parts = [action[n]["pos"] for n in robot_names if n in action]
    step: Dict[str, np.ndarray] = {}
    if state_parts:
        step["state"] = np.concatenate(state_parts).astype(np.float32)
    if action_parts:
        step["action"] = np.concatenate(action_parts).astype(np.float32)
    step["timestamp"] = np.array([obs.get("timestamp", 0.0)], dtype=np.float64)
    return step


def _apply_agent_record_signal(
    action: Dict,
    traj_logger,
    prev_record: bool,
) -> bool:
    """Start/stop the logger based on the ``_record`` level signal in action.

    The agent emits ``action["_record"] = True`` while it wants to record
    (e.g. both grippers squeezed, or DAgger intervention active).  This
    function detects rising / falling edges and drives the logger accordingly.
    Returns the current record signal value for use as ``prev_record`` next tick.
    """
    if traj_logger is None or "_record" not in action:
        return prev_record
    want = bool(action["_record"])
    if want and not prev_record:
        traj_logger.start_episode()
    elif not want and prev_record and traj_logger.recording:
        traj_logger.end_episode(save=True)
    return want


def _make_logger(config: LaunchConfig):
    """Return a configured TrajectoryLogger if record_path is set, else None.

    Attaches both the file-watcher trigger (``/tmp/record.flag``) and the
    keyboard trigger so recording can be controlled either way.
    """
    if config.record_path is None:
        return None
    from robots_realtime.data.trajectory_logger import (
        TrajectoryLogger,
        attach_file_watcher,
        attach_keyboard_listener,
        attach_signal_handler,
    )

    tl = TrajectoryLogger(config.record_path, fps=config.hz)
    attach_file_watcher(tl)  # headless: touch /tmp/record.flag
    attach_keyboard_listener(tl)  # interactive: r + Enter
    attach_signal_handler(tl)  # scripted:   kill -USR1 <pid>
    return tl


def _run_sim_control_loop(
    robots: Dict[str, Robot],
    agent: Agent,
    config: LaunchConfig,
) -> None:
    """Simplified control loop for sim mode (no portal, no cameras).

    Runs entirely in-process so the MuJoCo viewer stays on the main thread.
    """
    log = logging.getLogger(__name__)
    rate = Rate(config.hz, rate_name="sim_control_loop")
    steps = 0
    start_time = time.time()
    loop_count = 0

    traj_logger = _make_logger(config)
    _prev_record = False  # tracks last _record level for edge detection

    # Build initial observation from robots
    obs = {name: robot.get_observations() for name, robot in robots.items()}
    obs["timestamp"] = time.time()

    try:
        while True:
            # Check if any sim viewer has been closed
            for robot in robots.values():
                if hasattr(robot, "is_viewer_running") and not robot.is_viewer_running():
                    log.info("Viewer closed, stopping...")
                    return

            action = agent.act(obs)
            _prev_record = _apply_agent_record_signal(action, traj_logger, _prev_record)

            # Apply actions directly
            for name, act in action.items():
                if name in robots:
                    robots[name].command_joint_pos(act["pos"])

            rate.sleep()

            # Reset requested via viser button — discard any active episode first.
            for robot in robots.values():
                if hasattr(robot, "consume_reset_request") and robot.consume_reset_request():
                    if traj_logger is not None and traj_logger.recording:
                        traj_logger.end_episode(save=False)
                    _prev_record = False
                    if hasattr(agent, "reset"):
                        agent.reset()
                    break

            # Collect observations
            obs = {name: robot.get_observations() for name, robot in robots.items()}
            obs["timestamp"] = time.time()

            if traj_logger is not None and traj_logger.recording:
                step_data = _build_log_step(obs, action, list(robots.keys()))
                for robot in robots.values():
                    if hasattr(robot, "get_camera_images"):
                        imgs = robot.get_camera_images()
                        if imgs and steps == 1:
                            log.info("Camera keys being logged: %s", list(imgs.keys()))
                        elif not imgs and steps == 1:
                            log.warning("get_camera_images() returned empty — no MP4s will be saved")
                        step_data.update(imgs)
                traj_logger.log_step(step_data)

            steps += 1
            loop_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1:
                log.info(f"Sim control loop: {loop_count / elapsed_time:.2f} Hz")
                start_time = time.time()
                loop_count = 0

            if config.max_steps is not None and steps >= config.max_steps:
                log.info(f"Reached max steps ({config.max_steps}), stopping...")
                break
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        if traj_logger is not None:
            traj_logger.close()
        if hasattr(agent, "close"):
            agent.close()
        for robot in robots.values():
            if hasattr(robot, "close"):
                robot.close()


def _run_control_loop(env: RobotEnv, agent: Agent, config: LaunchConfig) -> None:
    """Run the main real-hardware control loop."""
    log = logging.getLogger(__name__)
    steps = 0
    start_time = time.time()
    loop_count = 0

    traj_logger = _make_logger(config)
    _prev_record = False

    # Init environment and warm up agent
    obs = env.reset()
    log.info(f"Action spec: {env.action_spec()}")
    agent.act(obs)

    robot_names = list(env.get_all_robots().keys())

    try:
        # Main control loop
        while True:
            with Timeout(30, "Agent action"):
                action = agent.act(obs)

            _prev_record = _apply_agent_record_signal(action, traj_logger, _prev_record)

            with Timeout(1, "Env step", "warning"):
                obs = env.step(action)

            if traj_logger is not None and traj_logger.recording:
                step_data = _build_log_step(obs, action, robot_names)
                # Camera images land directly in obs from CameraDriver.read()
                for key, val in obs.items():
                    if isinstance(val, np.ndarray) and _is_image_array(val):
                        step_data[key] = val
                traj_logger.log_step(step_data)

            steps += 1
            loop_count += 1

            elapsed_time = time.time() - start_time
            if elapsed_time >= 1:
                log.info(f"Control loop frequency: {loop_count / elapsed_time:.2f} Hz")
                start_time = time.time()
                loop_count = 0

            if config.max_steps is not None and steps >= config.max_steps:
                log.info(f"Reached max steps ({config.max_steps}), stopping...")
                break
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        if traj_logger is not None:
            traj_logger.close()


def _is_image_array(arr: np.ndarray) -> bool:
    return arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] == 3


if __name__ == "__main__":
    main(tyro.cli(Args))
