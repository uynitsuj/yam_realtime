"""RobotNode — wraps any robot driver and bridges it onto the ZMQ bus.

Works with any robot that implements:
    robot.command_joint_pos(joint_pos: np.ndarray) -> None
    robot.get_observations() -> dict  # must contain "joint_pos"

Examples: i2rt MotorChainRobot (YAM), FrankaPanda (OSC torque control).

Published topics:
    ``{name}/joint_state``  — dict from robot.get_observations()

Subscribed topics (configured at construction):
    ``{cmd_topic}``         — e.g. "gello_left/joint_pos"
"""

from __future__ import annotations

import importlib
import time

import numpy as np
import yaml as _yaml

from robots_realtime.runtime.node import Node, NodeRole


def _resolve(obj):
    """Recursively instantiate any dict containing a ``_target_`` key."""
    if isinstance(obj, dict):
        if "_target_" in obj:
            obj = dict(obj)
            target: str = obj.pop("_target_")
            kwargs = {k: _resolve(v) for k, v in obj.items()}
            module_path, cls_name = target.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            return getattr(mod, cls_name)(**kwargs)
        return {k: _resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(item) for item in obj]
    return obj


def _instantiate_from_target_yaml(config_path: str):
    """Load a YAML config and recursively instantiate all ``_target_`` objects."""
    with open(config_path) as f:
        cfg = _yaml.safe_load(f)
    return _resolve(cfg)


class RobotNode(Node):
    """Generic robot arm node.

    When loaded from YAML, ``robot`` is omitted and must be injected before
    ``setup()`` is called (or a subclass / factory overrides ``setup()``).
    The ``robot_config`` param is stored for reference but robot instantiation
    is left to the caller for hardware configs.

    Args:
        robot:        Any object implementing ``command_joint_pos()`` and
                      ``get_observations()``. Optional when loading from YAML.
        name:      Node name on the bus.
        cmd_topic: Full topic to subscribe to for joint position commands.
                   If None the node runs in read-only mode.
        writer:    Optional Writer injected at construction for recording.
    """

    role = NodeRole.ROBOT
    published_topics: list[str] = ["joint_state"]
    poll_freq: float | None = None
    subscriber_driven: bool = True

    def __init__(
        self,
        robot=None,
        name: str = "robot",
        cmd_topic: str | None = None,
        robot_config: str | None = None,
        poll_freq: float | None = None,
        startup_joint_pos: list[float] | None = None,
        startup_duration_s: float = 2.0,
        # Default to parking at the zero pose on shutdown; override in YAML with
        # a custom list, or set `shutdown_joint_pos: null` to skip parking.
        shutdown_joint_pos: list[float] | None = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        shutdown_duration_s: float = 2.0,
        ramp_duration_s: float = 1.5,
        resume_gap_s: float = 0.2,
        writer=None,
        **kwargs,
    ) -> None:
        self.subscribed_topics = [cmd_topic] if cmd_topic else []
        # Explicitly set poll_freq and subscriber_driven before calling super().__init__
        if poll_freq is not None:
            self.poll_freq = poll_freq
            self.subscriber_driven = False  # switch to fixed_rate mode
        super().__init__(name=name, writer=writer, **kwargs)
        self._robot = robot
        self._cmd_topic = cmd_topic
        self._robot_config = robot_config  # stored for reference; instantiation is caller's job
        self._startup_joint_pos = startup_joint_pos
        self._startup_duration_s = startup_duration_s
        self._shutdown_joint_pos = shutdown_joint_pos
        self._shutdown_duration_s = shutdown_duration_s
        # Safe-handoff ramp state. On the first command and after any gap
        # longer than resume_gap_s, seed _ramp_seed from the robot's actual
        # joint_pos and blend smoothly from seed → target over ramp_duration_s
        # seconds. After the window closes, commands pass through directly so
        # the leader has full tracking authority — unlike a velocity-capped
        # ramp, this is guaranteed to terminate even if the leader is moving
        # faster than the ramp rate during the handoff window.
        self._ramp_duration_s = float(ramp_duration_s)
        self._resume_gap_s = float(resume_gap_s)
        self._ramp_seed: np.ndarray | None = None
        self._ramp_start_time: float = 0.0
        self._ramping: bool = False
        self._last_msg_ts: float = 0.0

    def setup(self) -> None:
        if self._robot is None:
            if self._robot_config is None:
                raise RuntimeError(
                    f"[{self.name}] RobotNode.robot is None — inject a robot driver before starting. "
                    f"(robot_config={self._robot_config!r})"
                )
            self._robot = _instantiate_from_target_yaml(self._robot_config)

        if self._startup_joint_pos is not None:
            print(f"[{self.name}] Moving to startup pose over {self._startup_duration_s:.1f}s")
            self._move_to_pose(self._startup_joint_pos, self._startup_duration_s)
            print(f"[{self.name}] Startup pose reached")

    def step(self) -> None:
        ts = time.time()
        now = time.monotonic()

        # Paused: don't issue joint commands. i2rt's internal control loop keeps
        # the motors at the last commanded position. Skip _last_msg_ts updates
        # so that on resume the gap > resume_gap_s triggers a fresh ramp seed.
        if self._paused:
            self.publish("joint_state", self._robot.get_observations(), ts=ts)
            return

        if self._cmd_topic:
            cmd = self.get_latest(self._cmd_topic)
            cmd_ts = self.get_timestamp(self._cmd_topic) if cmd is not None else None
            if cmd is not None:
                # Use np.array() to ensure a writable copy (np.asarray may return read-only view)
                target = np.array(cmd["joint_pos"], dtype=np.float64)
                is_new = cmd_ts is not None and cmd_ts != self._last_msg_ts

                # Trigger a handoff ramp on first message ever or after a cmd-stream gap.
                # Seed from get_joint_pos() (full 7-element vector in command space) —
                # NOT get_observations()["joint_pos"] which omits the gripper on i2rt
                # MotorChainRobot and would cause a shape mismatch.
                if is_new and (self._last_msg_ts == 0.0 or (cmd_ts - self._last_msg_ts) > self._resume_gap_s):
                    try:
                        seed = np.asarray(self._robot.get_joint_pos(), dtype=np.float64)
                    except (AttributeError, TypeError):
                        seed = None
                    self._ramp_seed = seed.copy() if seed is not None and seed.shape == target.shape else target.copy()
                    self._ramp_start_time = now
                    self._ramping = self._ramp_duration_s > 0.0
                if is_new:
                    self._last_msg_ts = cmd_ts

                if self._ramping and self._ramp_seed is not None:
                    alpha = (now - self._ramp_start_time) / self._ramp_duration_s
                    if alpha >= 1.0:
                        self._ramping = False
                        self._robot.command_joint_pos(target)
                    else:
                        alpha = max(0.0, alpha)
                        blended = (1.0 - alpha) * self._ramp_seed + alpha * target
                        self._robot.command_joint_pos(blended)
                else:
                    self._robot.command_joint_pos(target)

        self.publish("joint_state", self._robot.get_observations(), ts=ts)

    def cleanup(self) -> None:
        if self._shutdown_joint_pos is not None and self._robot is not None:
            print(f"[{self.name}] Parking at shutdown pose over {self._shutdown_duration_s:.1f}s")
            try:
                self._move_to_pose(self._shutdown_joint_pos, self._shutdown_duration_s)
                print(f"[{self.name}] Shutdown pose reached")
            except Exception as exc:
                print(f"[{self.name}] Failed to park at shutdown pose: {exc}")

        if hasattr(self._robot, "stop"):
            self._robot.stop()

    def _move_to_pose(self, target: list[float], duration_s: float) -> None:
        """Smoothly interpolate robot to target joint position."""
        target_arr = np.asarray(target, dtype=np.float64)
        if hasattr(self._robot, "move_joints"):
            self._robot.move_joints(target_arr, time_interval_s=duration_s)
        else:
            current = np.asarray(self._robot.get_joint_pos(), dtype=np.float64)
            steps = max(1, int(duration_s * 25))
            for i in range(steps + 1):
                alpha = i / steps
                self._robot.command_joint_pos((1.0 - alpha) * current + alpha * target_arr)
                time.sleep(duration_s / steps)

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        kwargs = {
            "name": params["name"],
            "cmd_topic": params.get("cmd_topic"),
            "robot_config": params.get("robot_config"),
        }
        # Pass through poll_freq if specified
        if "poll_freq" in params:
            kwargs["poll_freq"] = params["poll_freq"]
        for key in (
            "startup_joint_pos",
            "startup_duration_s",
            "shutdown_joint_pos",
            "shutdown_duration_s",
            "ramp_duration_s",
            "resume_gap_s",
        ):
            if key in params:
                kwargs[key] = params[key]
        return kwargs
