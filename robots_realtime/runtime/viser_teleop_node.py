"""ViserTeleopNode — Viser IK teleoperation with live sensor visualization.

Combines two concerns that are tightly coupled through a shared ViserServer:
  1. IK teleoperation (publishes joint commands at ik_freq Hz)
  2. Live visualization (feeds joint state + camera frames into the agent's
     obs dict at viz_freq Hz, driving the URDF overlay and camera panels)

Using AgentNode for this would deep-copy camera frames at the full IK rate.
This node decouples the two update rates explicitly.

Session YAML example::

    - type: ViserTeleopNode
      name: franka_viser
      agent_class: robots_realtime.agents.teleoperation.franka_pyroki_viser_agent:FrankaPyrokiViserAgent
      agent_kwargs:
        viser_port: 8765
        ik_rate: 100.0
        robotiq_gripper: true
      arm_key: left
      ik_freq: 100.0
      viz_freq: 20.0
      state_topics:
        left: franka/joint_state
      image_topics:
        camera_top: camera_top/rgb

Bimanual agents omit ``arm_key``; the action dict's per-arm keys then fan out
to ``{name}/{key}_pos`` (e.g. ``yam_viser/left_pos``, ``yam_viser/right_pos``),
matching AgentNode's convention::

    - type: ViserTeleopNode
      name: yam_viser
      agent_class: robots_realtime.agents.teleoperation.yam_pyroki_viser_agent:YamPyrokiViserAgent
      agent_kwargs:
        bimanual: true
        right_arm_extrinsic: {position: [0.0, -0.61, 0.0], rotation: [1.0, 0.0, 0.0, 0.0]}
      state_topics:
        left:  yam_left/joint_state
        right: yam_right/joint_state
"""

from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np

from robots_realtime.runtime.node import Node, NodeRole


class ViserTeleopNode(Node):
    """Viser IK teleoperation node with live sensor visualization.

    Args:
        name:          Node name on the bus.
        agent_class:   Dotted import path, e.g.
                       "robots_realtime.agents.teleoperation.franka_pyroki_viser_agent:FrankaPyrokiViserAgent".
        agent_kwargs:  Keyword arguments forwarded to agent_class().
        arm_key:       Key in the action dict to extract joint positions from
                       (e.g. "left" for single-arm Franka).
        ik_freq:       Rate at which joint commands are published (Hz).
        viz_freq:      Rate at which camera frames are fed into the agent's
                       obs dict for visualization (Hz).  Should be much lower
                       than ik_freq to avoid copying large arrays every tick.
        state_topics:  {obs_key: bus_topic} — joint state inputs.
                       Use the arm key here (e.g. "left") so the agent's
                       _extract_joint_pos can find it.
        image_topics:  {obs_key: bus_topic} — camera RGB inputs.
        writer:        Optional Writer injected at construction for recording.
    """

    role = NodeRole.CONTROLLER
    published_topics: list[str] = ["joint_pos"]
    subscriber_driven: bool = False

    def __init__(
        self,
        name: str = "viser_teleop",
        agent_class: str | None = None,
        agent_kwargs: dict | None = None,
        arm_key: str | None = None,
        ik_freq: float = 100.0,
        viz_freq: float = 20.0,
        state_topics: dict[str, str] | None = None,
        image_topics: dict[str, str] | None = None,
        writer=None,
        **kwargs,
    ) -> None:
        self._state_topics = state_topics or {}
        self._image_topics = image_topics or {}
        self.subscribed_topics = (
            list(self._state_topics.values()) + list(self._image_topics.values())
        )
        self.poll_freq = ik_freq
        # Advertise the topics this node will actually publish: a single-arm
        # agent (or one pinned with arm_key) emits joint_pos; a multi-arm agent
        # fans out to {arm}_pos, one per state topic.
        if arm_key is None and len(self._state_topics) > 1:
            self.published_topics = [f"{key}_pos" for key in self._state_topics]
        super().__init__(name=name, writer=writer, **kwargs)

        self._agent = None
        self._agent_class = agent_class
        self._agent_kwargs = agent_kwargs or {}
        self._arm_key = arm_key
        self._viz_period = 1.0 / viz_freq
        self._last_viz_update: float = 0.0
        self._last_action: dict[str, np.ndarray] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def setup(self) -> None:
        if self._agent_class is None:
            raise RuntimeError(f"[{self.name}] ViserTeleopNode requires 'agent_class'")
        ref = self._agent_class
        if ":" not in ref:
            raise ValueError(
                f"agent_class must be 'module.path:ClassName', got {ref!r}"
            )
        module_path, cls_name = ref.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        self._agent = getattr(mod, cls_name)(**self._agent_kwargs)
        if hasattr(self._agent, "reset"):
            self._agent.reset()

    def step(self) -> None:
        ts = time.time()

        # Always read the latest joint state (cheap — just a dict lookup).
        state_obs: dict = {"timestamp": ts}
        for key, topic in self._state_topics.items():
            data = self.get_latest(topic)
            if data is not None:
                state_obs[key] = data

        # At viz_freq, fold in camera data for the visualization thread.
        # CameraNode publishes {images, depth_data, intrinsics, extrinsics} on
        # {name}/rgb — already in the format obs_get_rgb and _update_visualization expect.
        if ts - self._last_viz_update >= self._viz_period:
            for key, topic in self._image_topics.items():
                cam_msg = self.get_latest(topic)
                if cam_msg is not None:
                    state_obs[key] = cam_msg
            self._last_viz_update = ts

        # agent.act() sets agent.obs (drives visualization) and returns IK targets.
        _t_act = time.perf_counter()
        action = self._agent.act(state_obs)
        self._perf.record("act_ms", (time.perf_counter() - _t_act) * 1e3)

        # How far the IK output moves per tick, per arm. Compare against the
        # same metric on the RobotNode side (cmd_step_mrad): if the agent's
        # steps are smooth but the robot's are chunky, commands are being
        # dropped or coalesced in transit rather than generated stepped.
        if isinstance(action, dict):
            for key, arm in action.items():
                if key.startswith("_"):
                    continue
                pos = self._as_pos(arm)
                if pos is None:
                    continue
                prev = self._last_action.get(key)
                if prev is not None and prev.shape == pos.shape:
                    self._perf.record(f"ik_step_mrad[{key}]", float(np.abs(pos - prev).max()) * 1e3)
                self._last_action[key] = pos

        # Publish joint commands (single-arm → joint_pos, multi-arm → {key}_pos).
        self._publish_commands(action, ts)

    def cleanup(self) -> None:
        if self._agent is not None and hasattr(self._agent, "close"):
            self._agent.close()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _publish_commands(self, action: dict, ts: float) -> None:
        """Publish joint targets, mirroring AgentNode's topic convention.

        ``arm_key`` set              → action[arm_key]["pos"] as ``joint_pos``
        flat ``{"pos": ...}``        → ``joint_pos``
        one arm key                  → ``joint_pos``
        several arm keys (bimanual)  → ``{key}_pos`` for each
        """
        if not isinstance(action, dict) or not action:
            return

        if self._arm_key is not None:
            pos = self._as_pos(action.get(self._arm_key))
            if pos is not None:
                self.publish("joint_pos", {"joint_pos": pos}, ts=ts)
            return

        if "pos" in action:
            pos = self._as_pos(action)
            if pos is not None:
                self.publish("joint_pos", {"joint_pos": pos}, ts=ts)
            return

        arm_keys = [k for k in action if not k.startswith("_")]
        if len(arm_keys) == 1:
            pos = self._as_pos(action[arm_keys[0]])
            if pos is not None:
                self.publish("joint_pos", {"joint_pos": pos}, ts=ts)
            return

        for key in arm_keys:
            pos = self._as_pos(action[key])
            if pos is not None:
                self.publish(f"{key}_pos", {"joint_pos": pos}, ts=ts)

    @staticmethod
    def _as_pos(arm: Any) -> np.ndarray | None:
        if arm is None:
            return None
        if isinstance(arm, dict):
            if "pos" not in arm:
                return None
            arm = arm["pos"]
        return np.asarray(arm, dtype=np.float32)

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        return {
            "name":         params["name"],
            "agent_class":  params.get("agent_class"),
            "agent_kwargs": params.get("agent_kwargs") or {},
            "arm_key":      params.get("arm_key"),
            "ik_freq":      params.get("ik_freq", 100.0),
            "viz_freq":     params.get("viz_freq", 20.0),
            "state_topics": params.get("state_topics"),
            "image_topics": params.get("image_topics"),
        }
