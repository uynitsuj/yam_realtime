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

from robots_realtime.nodes.base import Node, NodeRole


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
        writer=None,
        **kwargs,
    ) -> None:
        self.subscribed_topics = [cmd_topic] if cmd_topic else []
        super().__init__(name=name, writer=writer, **kwargs)
        self._robot = robot
        self._cmd_topic = cmd_topic
        self._robot_config = robot_config  # stored for reference; instantiation is caller's job

    def setup(self) -> None:
        if self._robot is None:
            if self._robot_config is None:
                raise RuntimeError(
                    f"[{self.name}] RobotNode.robot is None — inject a robot driver before starting. "
                    f"(robot_config={self._robot_config!r})"
                )
            self._robot = _instantiate_from_target_yaml(self._robot_config)

    def step(self) -> None:
        ts = time.time()
        if self._cmd_topic:
            cmd = self.get_latest(self._cmd_topic)
            if cmd is not None:
                self._robot.command_joint_pos(np.asarray(cmd["joint_pos"]))

        self.publish("joint_state", self._robot.get_observations(), ts=ts)

    def cleanup(self) -> None:
        if hasattr(self._robot, "stop"):
            self._robot.stop()

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        return {
            "name": params["name"],
            "cmd_topic": params.get("cmd_topic"),
            "robot_config": params.get("robot_config"),
        }
