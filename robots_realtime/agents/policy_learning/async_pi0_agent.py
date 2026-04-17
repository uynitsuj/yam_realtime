"""Async diffusion / VLA policy agent backed by an OpenPI websocket server.

Runs as a regular ``Agent`` inside a ZMQ ``AgentNode``:

    AgentNode.step()  (at poll_freq, e.g. 30 Hz)
        └── agent.act(obs)
              └── self.__call__(obs)
                    ├── self._obs = obs_to_model_input(obs)   # lock-protected
                    └── self.select_action()                   # dequeue from chunk buffer

Inference runs entirely independently in a background thread (``_action_loop``)
that reads the latest ``self._obs`` snapshot, fires a websocket ``infer`` call,
and merges the returned action chunk into ``self.last_actions`` with a linear
ramp blend at the chunk boundary. The blend length auto-scales with inference
latency (more blending when the server took longer, since the old chunk is
staler), clamped by ``[min_smoothed_actions, max_smoothed_actions]``.

Three inference modes (``inference_mode`` kwarg):

    sync                — blocking, synchronous. Uses OpenPI's ``ActionChunkBroker``
                          so inference only fires when the last chunk is exhausted
                          (every ``action_horizon`` consumer calls).
    async               — background thread, runs inference flat-out (no sleep
                          between iterations). Fastest buffer refresh.
    async_rate_limited  — background thread, sleeps ``inference_interval_s``
                          seconds between iterations to cap inference rate.

OpenPI client imports are done lazily inside ``__init__`` so this module can be
imported without ``openpi_client`` being installed — it's only required when
the agent is actually instantiated.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple

import numpy as np
from dm_env.specs import Array

from robots_realtime.agents.agent import PolicyAgent
from robots_realtime.agents.constants import ActionSpec
from robots_realtime.robots.utils import Rate

InferenceMode = Literal["sync", "async", "async_rate_limited"]


@dataclass
class ModelIOConfig:
    """What obs keys the policy server expects and what action keys it returns.

    Defaults match the OpenPI bimanual YAM schema (lab42 passive-gello training
    runs). Override via kwargs if your model was trained with different keys.
    """

    action_keys: Tuple[str, ...] = (
        "action-left-pos",
        "action-right-pos",
        "action-left-vel",
        "action-right-vel",
    )
    mlp_keys: Tuple[str, ...] = (
        "left-joint_pos",
        "left-gripper_pos",
        "right-joint_pos",
        "right-gripper_pos",
    )
    image_keys: Tuple[str, ...] = (
        "left_camera-images-rgb",
        "right_camera-images-rgb",
        "top_camera-images-rgb",
    )


def _recursive_flatten(obj: Any, prefix: str = "", sep: str = "-") -> Dict[str, Any]:
    """Flatten a nested dict into {key: value} with ``sep``-joined paths.

    Terminates at arrays, scalars, and non-dict containers — so
    ``{"left": {"joint_pos": arr, "gripper_pos": arr}}`` becomes
    ``{"left-joint_pos": arr, "left-gripper_pos": arr}``.
    """
    flat: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            if isinstance(v, dict):
                flat.update(_recursive_flatten(v, key, sep=sep))
            else:
                flat[key] = v
    else:
        flat[prefix] = obj
    return flat


class AsyncDiffusionAgent(PolicyAgent):
    """OpenPI websocket policy wrapper with chunked async inference."""

    def __init__(
        self,
        use_joint_state_as_action: bool = False,
        ip: str = "0.0.0.0",
        port: int = 8111,
        action_horizon: int = 30,
        inference_mode: InferenceMode = "async_rate_limited",
        inference_interval_s: float | None = 0.5,
        min_smoothed_actions: int = 1,
        max_smoothed_actions: int = 8,
        model_io_config: ModelIOConfig | None = None,
    ) -> None:
        # Validate config first — user-error (bad mode) should surface before
        # env-error (missing openpi_client).
        if inference_mode not in ("sync", "async", "async_rate_limited"):
            raise ValueError(
                f"inference_mode must be one of 'sync', 'async', 'async_rate_limited'; got {inference_mode!r}"
            )
        if inference_mode == "async_rate_limited" and (inference_interval_s is None or inference_interval_s <= 0):
            raise ValueError("inference_mode='async_rate_limited' requires inference_interval_s > 0")
        if min_smoothed_actions < 0 or max_smoothed_actions < 0:
            raise ValueError("min_smoothed_actions and max_smoothed_actions must be non-negative")
        if min_smoothed_actions > max_smoothed_actions:
            raise ValueError(
                f"min_smoothed_actions ({min_smoothed_actions}) cannot exceed "
                f"max_smoothed_actions ({max_smoothed_actions})"
            )

        # Lazy import — openpi_client is an optional dep; keeps this module
        # importable for module registry / tests without the client installed.
        try:
            from openpi_client import action_chunk_broker, image_tools  # noqa: PLC0415
            from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: PLC0415
            from openpi_client.runtime.agents import policy_agent as _policy_agent  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "AsyncDiffusionAgent requires `openpi_client`. Install it into this venv "
                "before instantiating the agent (e.g. `uv pip install openpi-client`)."
            ) from exc

        self._image_tools = image_tools

        self.use_joint_state_as_action = use_joint_state_as_action
        self._websocket_client_policy = _websocket_client_policy.WebsocketClientPolicy(host=ip, port=port)

        self.action_horizon = action_horizon
        self.inference_mode: InferenceMode = inference_mode
        self.inference_interval_s = inference_interval_s
        self.min_smoothed_actions = int(min_smoothed_actions)
        self.max_smoothed_actions = int(max_smoothed_actions)
        self.inference_interval_rate = (
            Rate(1.0 / inference_interval_s, rate_name="inference_interval")
            if inference_mode == "async_rate_limited"
            else None
        )
        self.config = model_io_config or ModelIOConfig()

        self.action_lock = threading.Lock()
        self.last_actions: np.ndarray | None = None
        self.obs_lock = threading.Lock()
        self._obs: Dict[str, Any] | None = None
        self.action_counter = 0
        self._stop = threading.Event()

        if inference_mode in ("async", "async_rate_limited"):
            self.action_thread = threading.Thread(target=self._action_loop, name="AsyncDiffusionAgent_inference", daemon=True)
            self.action_thread.start()
            self._agent = None
        else:
            self._agent = _policy_agent.PolicyAgent(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=self._websocket_client_policy,
                    action_horizon=self.action_horizon,
                )
            )
            self.action_thread = None

    # ------------------------------------------------------------------ #
    # Metadata / specs
    # ------------------------------------------------------------------ #

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "action_horizon": self.action_horizon,
            "inference_mode": self.inference_mode,
            "inference_interval_s": self.inference_interval_s,
            "min_smoothed_actions": self.min_smoothed_actions,
            "max_smoothed_actions": self.max_smoothed_actions,
            **self._websocket_client_policy.get_server_metadata(),
        }

    def action_spec(self) -> ActionSpec:
        if self.use_joint_state_as_action:
            return {
                "left": {
                    "pos": Array(shape=(7,), dtype=np.float32),
                    "vel": Array(shape=(7,), dtype=np.float32),
                },
                "right": {
                    "pos": Array(shape=(7,), dtype=np.float32),
                    "vel": Array(shape=(7,), dtype=np.float32),
                },
            }
        return {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},
            "right": {"pos": Array(shape=(7,), dtype=np.float32)},
        }

    # ------------------------------------------------------------------ #
    # Observation preprocessing
    # ------------------------------------------------------------------ #

    def obs_to_model_input(self, obs: Dict[str, Any]) -> Dict[str, Any] | None:
        """Flatten bus-message obs into the flat ``{key: array}`` shape the server expects.

        AgentNode delivers obs as ``{obs_key: bus_message_dict}``. State messages
        (e.g. yam_left/joint_state) have nested ``{"joint_pos": arr, "gripper_pos": arr, ...}``
        which flattens naturally with the ``obs_key`` prefix. Camera messages wrap
        frames as ``{"images": {"rgb": arr}, "timestamp": ts}`` which flattens to
        ``"<obs_key>-images-rgb"`` — matches ``image_keys`` convention.

        Returns ``None`` if any required mlp / image key is missing. This happens
        early in the session lifetime before producer nodes (RobotNodes, Cameras)
        have published their first message — the consumer should treat this as
        "not ready yet" and simply skip publishing an action for this tick.
        """
        flat = _recursive_flatten(obs)

        required = list(self.config.mlp_keys) + list(self.config.image_keys)
        missing = [k for k in required if k not in flat]
        if missing:
            now = time.monotonic()
            if now - getattr(self, "_last_missing_log_ts", 0.0) > 2.0:
                # Throttled log so the user can see what we're still waiting on.
                preview = ", ".join(missing[:4]) + (" …" if len(missing) > 4 else "")
                print(f"[AsyncDiffusionAgent] obs not ready — waiting on: {preview}")
                self._last_missing_log_ts = now
            return None

        flat_state = [np.asarray(flat[k]).reshape(-1) for k in self.config.mlp_keys]
        state = np.concatenate(flat_state, axis=-1)

        images: Dict[str, Any] = {}
        for k in self.config.image_keys:
            img = flat[k]
            img = self._image_tools.convert_to_uint8(self._image_tools.resize_with_pad(img, 224, 224))
            img = np.transpose(img, (2, 0, 1))
            images[k] = img

        return {"state": state, **images}

    # ------------------------------------------------------------------ #
    # Public act() — called by AgentNode.step() at consumer rate
    # ------------------------------------------------------------------ #

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # PolicyAgent.act is abstract (raises NotImplementedError); don't call super.
        raw = self(obs)
        if raw is None:
            # Obs incomplete (producers still warming up) or the very first
            # inference hasn't landed yet — return an empty action dict so
            # AgentNode's _publish_commands is a no-op for this tick.
            return {}
        # Force a writable copy: chunks sourced from msgpack deserialization
        # (over the websocket) come back as read-only views, and the clip
        # steps below mutate in place.
        a = np.array(raw, dtype=np.float32, copy=True)
        if self.use_joint_state_as_action:
            assert a.shape == (28,), a.shape
            left = a[:14]
            right = a[14:]
            left[6] = np.clip(left[6], 0, 1)
            right[6] = np.clip(right[6], 0, 1)
            return {
                "left": {"pos": left[:7], "vel": left[7:]},
                "right": {"pos": right[:7], "vel": right[7:]},
            }
        assert a.shape == (14,), a.shape
        left = a[:7]
        right = a[7:]
        left[-1] = np.clip(left[-1], 0, 1)
        right[-1] = np.clip(right[-1], 0, 1)
        return {
            "left": {"pos": left},
            "right": {"pos": right},
            # Snapshot of the remaining chunk predictions so the viser monitor
            # (or anyone else) can visualize where the policy expects each arm
            # to be over the next N steps. Underscore prefix → AgentNode treats
            # this as meta and publishes it on a dedicated topic instead of as
            # a joint_pos command.
            "_chunk": self._snapshot_chunk(),
        }

    def _snapshot_chunk(self) -> Dict[str, Any] | None:
        """Return the still-unconsumed tail of the current action chunk, split by arm.

        Shape: {"left": (N, 7), "right": (N, 7)} for the 14-dim bimanual case.
        Returns None if the buffer is not yet populated or is empty. The tail
        starts at `action_counter` so consumers can interpret step 0 of the
        returned array as "the next action we'd dequeue".
        """
        with self.action_lock:
            if self.last_actions is None:
                return None
            remaining = self.last_actions[self.action_counter:]
        if remaining.shape[0] == 0 or remaining.ndim != 2:
            return None
        # For use_joint_state_as_action=False, each row is (14,) = left7 + right7.
        # We only publish the pos-flavoured split here; the velocity flavour (28,)
        # case is rarely used and the visualizer doesn't need vel anyway.
        if remaining.shape[1] == 14:
            return {
                "left":  np.ascontiguousarray(remaining[:, :7], dtype=np.float32),
                "right": np.ascontiguousarray(remaining[:, 7:], dtype=np.float32),
            }
        if remaining.shape[1] == 28:
            return {
                "left":  np.ascontiguousarray(remaining[:, :7],   dtype=np.float32),
                "right": np.ascontiguousarray(remaining[:, 14:21], dtype=np.float32),
            }
        return None

    def __call__(self, obs: Dict[str, Any]) -> np.ndarray | None:
        model_input = self.obs_to_model_input(obs)
        if model_input is None:
            return None
        with self.obs_lock:
            self._obs = model_input
        if self.inference_mode == "sync":
            return self._agent.get_action(self._obs)["actions"]
        # Async modes: if the background thread hasn't produced the first chunk
        # yet, tell the consumer we're not ready rather than blocking step().
        if self.last_actions is None:
            return None
        return self.select_action()

    # ------------------------------------------------------------------ #
    # Async plumbing
    # ------------------------------------------------------------------ #

    def _action_loop(self) -> None:
        while not self._stop.is_set():
            # Wait for the first observation to arrive from the consumer thread.
            if self._obs is None:
                time.sleep(0.01)
                continue

            with self.obs_lock:
                current_obs = self._obs
            with self.action_lock:
                start_inference_action_counter = self.action_counter

            inferred_action = np.asarray(self._websocket_client_policy.infer(current_obs)["actions"])

            with self.action_lock:
                complete_inference_action_counter = self.action_counter
                consumed_during_inference = max(0, complete_inference_action_counter - start_inference_action_counter)

                # Time-align the new chunk: skip its first `consumed_during_inference`
                # actions so index 0 corresponds to the consumer's current tick. If
                # inference took longer than the server's chunk (rare but possible
                # when latency >> action_horizon / poll_freq), we can't time-align;
                # fall back to starting at index 0 rather than producing an empty slice.
                server_chunk_len = inferred_action.shape[0]
                skip = consumed_during_inference
                if skip >= server_chunk_len:
                    print(
                        f"[AsyncDiffusionAgent] inference latency ({skip} ticks) >= server chunk "
                        f"length ({server_chunk_len}); can't time-align, resetting to chunk head"
                    )
                    skip = 0
                new_action = inferred_action[skip:]

                if self.last_actions is None:
                    self.last_actions = new_action
                else:
                    remaining_actions = self.last_actions[self.action_counter :]
                    # Dynamic blend length: scale with how many actions the consumer
                    # dequeued during inference (i.e. with inference latency). Slower
                    # inference → staler old chunk → more blending needed. Clamped by
                    # [min_smoothed_actions, max_smoothed_actions] and by the lengths
                    # of both arrays being blended.
                    target = min(consumed_during_inference, self.max_smoothed_actions)
                    num_smoothed = max(self.min_smoothed_actions, target)
                    num_smoothed = min(num_smoothed, remaining_actions.shape[0], new_action.shape[0])
                    if num_smoothed > 0:
                        weights = np.linspace(1.0 / num_smoothed, 1.0, num_smoothed).reshape(-1, 1)
                        smoothed = weights * new_action[:num_smoothed] + (1.0 - weights) * remaining_actions[:num_smoothed]
                        self.last_actions = np.concatenate([smoothed, new_action[num_smoothed:]], axis=0)
                    else:
                        self.last_actions = new_action
                    self.action_counter = 0

            if self.inference_interval_rate is not None:
                self.inference_interval_rate.sleep()
            # inference_mode == "async": no sleep, loop back immediately (flat-out)

    def select_action(self) -> np.ndarray:
        # Wait for the first inference to land.
        while self.last_actions is None and not self._stop.is_set():
            time.sleep(0.01)
        if self._stop.is_set():
            raise RuntimeError("AsyncDiffusionAgent was closed before the first action became available")
        with self.action_lock:
            idx = min(self.action_counter, self.action_horizon - 1)
            action = self.last_actions[idx]
            if self.action_counter >= self.action_horizon - 1:
                # Inference lagging — hold the final action of the current chunk.
                # A warning is printed at most once per chunk boundary.
                if self.action_counter == self.action_horizon - 1:
                    print(f"[AsyncDiffusionAgent] inference lag — repeating action at counter {self.action_counter}")
            else:
                self.action_counter += 1
        return action

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self._stop.set()
        if self.action_thread is not None and self.action_thread.is_alive():
            self.action_thread.join(timeout=1.0)

    def reset(self) -> None:
        # Drop buffered actions so the next chunk is freshly produced. Retain the
        # background thread — it will infer again as soon as a new obs lands.
        with self.action_lock:
            self.last_actions = None
            self.action_counter = 0
