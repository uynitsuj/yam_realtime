"""Async remote ACT policy agent backed by an act-side websocket server.

Mirrors ``async_pi0_agent.AsyncDiffusionAgent`` (chunk buffering + linear-ramp
smoothing in a background thread) but talks to ``act/scripts/serve_policy.py``
instead of openpi. The wire protocol is the openpi websocket protocol — both
sides go through ``openpi_client.{WebsocketClientPolicy, msgpack_numpy}`` —
because the act server vendors openpi's ``WebsocketPolicyServer``.

Four inference modes:

    sync                -- blocking. ``ActionChunkBroker`` exhausts each chunk
                           one action at a time, then fires the next inference.
                           No client-side smoothing.
    async               -- background thread, runs inference flat-out.
                           Linear-ramp blend at chunk boundaries.
    async_rate_limited  -- background thread, rate-capped (``inference_interval_s``
                           REQUIRED). Same blend.
    temporal_ensemble   -- ACT-paper inference scheme (Zhao et al.) — synchronous
                           and paper-faithful. One inference per consumer tick,
                           blocking. At tick ``t`` every overlapping chunk
                           contributes to the output via exponentially-weighted
                           average ``w_i = exp(-k * age_i)``, where ``age_i`` is
                           the position in the deque of populated predictions
                           for ``t`` (oldest → 0). Default
                           ``temporal_ensemble_k = 0.01`` matches the paper.
                           Requires ``inference_time < 1/poll_freq``; if not,
                           the consumer rate degrades to whatever inference
                           sustains.

Wire schema (matches ``act.policies.policy.ACTPolicy.infer``):
    obs ::= {"state": (D,) float32, "images": {cam: (3, H, W) uint8 CHW}}
    response ::= {"actions": (action_chunk_size, action_dim) float32}
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Literal, Tuple

import numpy as np
from dm_env.specs import Array

from robots_realtime.agents.agent import PolicyAgent
from robots_realtime.agents.constants import ActionSpec
from robots_realtime.robots.utils import Rate

InferenceMode = Literal["sync", "async", "async_rate_limited", "temporal_ensemble"]
_BLEND_MODES = ("async", "async_rate_limited")
_ASYNC_MODES = ("async", "async_rate_limited")
ImagePreprocess = Literal["center_crop", "resize"]


def _center_crop_and_resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop to the largest square that fits, then resize to (target_h, target_w)."""
    import cv2

    h, w = img.shape[:2]
    side = min(h, w)
    h0 = (h - side) // 2
    w0 = (w - side) // 2
    cropped = img[h0 : h0 + side, w0 : w0 + side]
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize preserving aspect ratio with letterboxing."""
    import cv2

    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _recursive_flatten(obj: Any, prefix: str = "", sep: str = "-") -> Dict[str, Any]:
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


@dataclass
class ACTModelIOConfig:
    """What obs keys the ACT server expects (matches ACT's YAMDataConfig)."""

    state_keys: Tuple[str, ...] = (
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
    image_key_to_model_key: Dict[str, str] = field(default_factory=lambda: {
        "left_camera-images-rgb": "wrist_left",
        "right_camera-images-rgb": "wrist_right",
        "top_camera-images-rgb": "exo",
    })


class AsyncRemoteACTAgent(PolicyAgent):
    """ACT policy wrapper that defers inference to a remote websocket server."""

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 8012,
        action_horizon: int = 30,
        encoder_image_size: int | None = None,
        inference_mode: InferenceMode = "async_rate_limited",
        inference_interval_s: float | None = 0.3,
        min_smoothed_actions: int = 1,
        max_smoothed_actions: int = 8,
        # Temporal-ensemble (ACT paper) knob: w_i = exp(-k * age_i), where age_i
        # is the position of the prediction in the list of populated predictions
        # for the current tick (oldest → 0). Higher k = more weight on older
        # predictions (= smoother, but laggier). Paper default: 0.01.
        temporal_ensemble_k: float = 0.01,
        model_io_config: ACTModelIOConfig | None = None,
        image_preprocess: ImagePreprocess = "center_crop",
        use_joint_state_as_action: bool = False,
    ) -> None:
        valid_modes = ("sync", "async", "async_rate_limited", "temporal_ensemble")
        if inference_mode not in valid_modes:
            raise ValueError(
                f"inference_mode must be one of {valid_modes}; got {inference_mode!r}"
            )
        if image_preprocess not in ("center_crop", "resize"):
            raise ValueError(
                f"image_preprocess must be 'center_crop' or 'resize'; got {image_preprocess!r}"
            )
        if inference_mode == "async_rate_limited" and (inference_interval_s is None or inference_interval_s <= 0):
            raise ValueError("inference_mode='async_rate_limited' requires inference_interval_s > 0")
        if min_smoothed_actions > max_smoothed_actions:
            raise ValueError(
                f"min_smoothed_actions ({min_smoothed_actions}) cannot exceed "
                f"max_smoothed_actions ({max_smoothed_actions})"
            )

        # Lazy import — openpi-client is the only wire-format dep.
        try:
            from openpi_client import action_chunk_broker  # noqa: PLC0415
            from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: PLC0415
            from openpi_client.runtime.agents import policy_agent as _policy_agent  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "AsyncRemoteACTAgent requires `openpi_client`. Install it into this venv "
                "before instantiating the agent."
            ) from exc

        self._client = _websocket_client_policy.WebsocketClientPolicy(host=ip, port=port)

        # Auto-resolve encoder_image_size from server metadata. ACTPolicy.metadata
        # exposes it; allow the YAML to override (e.g. to deliberately downsample
        # further for bandwidth). Falls back to 256 if neither is provided so we
        # never silently send raw camera frames.
        server_meta = self._client.get_server_metadata()
        if encoder_image_size is None:
            self._encoder_image_size = int(server_meta.get("encoder_image_size", 256))
        else:
            self._encoder_image_size = int(encoder_image_size)
            srv = server_meta.get("encoder_image_size")
            if srv is not None and int(srv) != self._encoder_image_size:
                print(
                    f"[AsyncRemoteACTAgent] encoder_image_size override ({self._encoder_image_size}) "
                    f"disagrees with server metadata ({srv}); the server will resize on its end."
                )

        self.use_joint_state_as_action = use_joint_state_as_action
        self.action_horizon = int(action_horizon)
        self.inference_mode: InferenceMode = inference_mode
        self.inference_interval_s = inference_interval_s
        self.min_smoothed_actions = int(min_smoothed_actions)
        self.max_smoothed_actions = int(max_smoothed_actions)
        self._te_k = float(temporal_ensemble_k)
        self.inference_interval_rate = (
            Rate(1.0 / inference_interval_s, rate_name="inference_interval")
            if inference_interval_s is not None and inference_interval_s > 0
            else None
        )
        self.config = model_io_config or ACTModelIOConfig()
        self._image_preprocess: ImagePreprocess = image_preprocess

        self.action_lock = threading.Lock()
        self.last_actions: np.ndarray | None = None
        self.obs_lock = threading.Lock()
        self._obs: Dict[str, Any] | None = None
        self.action_counter = 0
        self._stop = threading.Event()

        # Temporal-ensemble state: deque of (emit_tick, chunk) sorted oldest-first.
        # ``emit_tick`` is the absolute consumer tick at which the obs was sampled
        # — chunk[t - emit_tick] is the prediction for tick t. ``self._te_tick``
        # is the absolute consumer tick (advances per select_action call).
        self._te_chunks: Deque[Tuple[int, np.ndarray]] = deque()
        self._te_tick: int = 0
        # Snapshot of the most recent chunk + its emit_tick, kept ONLY for
        # _snapshot_chunk in TE mode (where last_actions/action_counter aren't
        # used at the consumer level).
        self._te_latest_chunk: np.ndarray | None = None
        self._te_latest_emit_tick: int = 0

        # Mode setup. async modes spin a background inference thread; sync
        # uses ActionChunkBroker; temporal_ensemble runs blocking inline (no
        # thread, no broker — chunks are merged into self._te_chunks per call).
        if inference_mode in _ASYNC_MODES:
            self.action_thread = threading.Thread(
                target=self._action_loop,
                name="AsyncRemoteACTAgent_inference",
                daemon=True,
            )
            self.action_thread.start()
            self._agent = None
            self._broker = None
        elif inference_mode == "sync":
            # ActionChunkBroker exhausts each chunk one action at a time
            # before firing the next inference. Same wire protocol; the broker
            # just slices [self._cur_step] off the (chunk, action_dim) array.
            self._broker = action_chunk_broker.ActionChunkBroker(
                policy=self._client,
                action_horizon=self.action_horizon,
            )
            self._agent = _policy_agent.PolicyAgent(policy=self._broker)
            self.action_thread = None
        else:  # temporal_ensemble
            self.action_thread = None
            self._agent = None
            self._broker = None

    # ------------------------------------------------------------------ #
    # Metadata / specs
    # ------------------------------------------------------------------ #

    def get_metadata(self) -> Dict[str, Any]:
        meta = {
            "action_horizon": self.action_horizon,
            "inference_mode": self.inference_mode,
            "inference_interval_s": self.inference_interval_s,
            "min_smoothed_actions": self.min_smoothed_actions,
            "max_smoothed_actions": self.max_smoothed_actions,
            "temporal_ensemble_k": self._te_k,
            "encoder_image_size": self._encoder_image_size,
        }
        meta.update(self._client.get_server_metadata())
        # Restore the client-side encoder_image_size after merging server meta
        # so the agent's effective preprocess size is what shows up.
        meta["encoder_image_size"] = self._encoder_image_size
        return meta

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
        """Flatten bus obs into the wire shape ACTPolicy.infer expects.

        Returns ``None`` when any required key is missing (producers warming up).
        Image preprocessing (crop + resize to ``encoder_image_size``) happens
        here on the client to keep the websocket payload small.
        """
        flat = _recursive_flatten(obs)

        required = list(self.config.state_keys) + list(self.config.image_keys)
        missing = [k for k in required if k not in flat]
        if missing:
            now = time.monotonic()
            if now - getattr(self, "_last_missing_log_ts", 0.0) > 2.0:
                preview = ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")
                print(f"[AsyncRemoteACTAgent] obs not ready -- waiting on: {preview}")
                self._last_missing_log_ts = now
            return None

        flat_state = [np.asarray(flat[k]).reshape(-1) for k in self.config.state_keys]
        state = np.concatenate(flat_state, axis=-1).astype(np.float32)

        target = self._encoder_image_size
        images: Dict[str, np.ndarray] = {}
        for k in self.config.image_keys:
            img = np.asarray(flat[k])
            if self._image_preprocess == "center_crop":
                img = _center_crop_and_resize(img, target, target)
            else:
                img = _resize(img, target, target)
            # Send uint8 CHW over the wire — server divides by 255. Saves ~4x
            # bandwidth vs. shipping float32. Matches openpi pi0 client format.
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))  # HWC -> CHW
            model_key = self.config.image_key_to_model_key.get(k, k)
            images[model_key] = img

        return {"state": state, "images": images}

    # ------------------------------------------------------------------ #
    # Public act() -- called by AgentNode.step() at consumer rate
    # ------------------------------------------------------------------ #

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        raw = self(obs)
        if raw is None:
            return {}
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
            "_chunk": self._snapshot_chunk(),
        }

    def _snapshot_chunk(self) -> Dict[str, Any] | None:
        """Return the still-unconsumed tail of the current chunk, split per-arm.

        - async/async_rate_limited: slice ``last_actions[action_counter:]``.
        - temporal_ensemble:        slice the latest chunk from its last
          consumed offset onward, so the viz shows what the freshest prediction
          intends going forward.
        - sync: ``ActionChunkBroker._last_results["actions"]`` from
          ``self._broker._cur_step:`` (we mirror the broker's internal cursor).
        """
        if self.inference_mode == "temporal_ensemble":
            with self.action_lock:
                if self._te_latest_chunk is None:
                    return None
                offset = self._te_tick - self._te_latest_emit_tick
                if offset < 0 or offset >= self._te_latest_chunk.shape[0]:
                    return None
                remaining = self._te_latest_chunk[offset:]
        elif self.inference_mode == "sync":
            if self._broker is None or self._broker._last_results is None:
                return None
            full = self._broker._last_results.get("actions")
            if full is None or not isinstance(full, np.ndarray) or full.ndim != 2:
                return None
            cur = int(getattr(self._broker, "_cur_step", 0))
            remaining = full[cur:]
        else:
            with self.action_lock:
                if self.last_actions is None:
                    return None
                remaining = self.last_actions[self.action_counter:]

        if remaining.shape[0] == 0 or remaining.ndim != 2:
            return None
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
            # ActionChunkBroker hard-swaps chunks once the previous one is
            # drained — no client-side smoothing. Returns one action per call.
            return self._agent.get_action(self._obs)["actions"]
        if self.inference_mode == "temporal_ensemble":
            # Paper-faithful synchronous TE: blocking inference per consumer
            # tick, then weighted ensemble over all overlapping chunks.
            return self._step_temporal_ensemble(self._obs)
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
            if self._obs is None:
                time.sleep(0.01)
                continue

            with self.obs_lock:
                current_obs = {
                    "state": self._obs["state"],
                    "images": dict(self._obs["images"]),
                }
            with self.action_lock:
                start_inference_action_counter = self.action_counter

            inferred_action = np.asarray(self._client.infer(current_obs)["actions"])
            self._blend_merge(inferred_action, start_inference_action_counter)

            if self.inference_interval_rate is not None:
                self.inference_interval_rate.sleep()

    def _blend_merge(self, inferred_action: np.ndarray, start_inference_action_counter: int) -> None:
        """async/async_rate_limited: linear-ramp blend at chunk boundary.

        Time-aligns the new chunk to the consumer (drop its first
        ``consumed_during_inference`` actions), then blends ``num_smoothed``
        actions across the boundary scaled by inference latency.
        """
        with self.action_lock:
            complete_inference_action_counter = self.action_counter
            consumed_during_inference = max(
                0, complete_inference_action_counter - start_inference_action_counter
            )

            server_chunk_len = inferred_action.shape[0]
            skip = consumed_during_inference
            if skip >= server_chunk_len:
                print(
                    f"[AsyncRemoteACTAgent] inference latency ({skip} ticks) >= chunk "
                    f"length ({server_chunk_len}); can't time-align, resetting to chunk head"
                )
                skip = 0
            new_action = inferred_action[skip:]

            if self.last_actions is None:
                self.last_actions = new_action
            elif new_action.shape[0] < 2 and self.last_actions.shape[0] >= 2:
                print(
                    f"[AsyncRemoteACTAgent] discarding length-{new_action.shape[0]} chunk "
                    f"(consumed_during_inference={consumed_during_inference}) -- keeping old buffer"
                )
            else:
                remaining_actions = self.last_actions[self.action_counter:]
                target = min(consumed_during_inference, self.max_smoothed_actions)
                num_smoothed = max(self.min_smoothed_actions, target)
                num_smoothed = min(num_smoothed, remaining_actions.shape[0], new_action.shape[0])
                if num_smoothed > 0:
                    weights = np.linspace(1.0 / num_smoothed, 1.0, num_smoothed).reshape(-1, 1)
                    smoothed = (
                        weights * new_action[:num_smoothed]
                        + (1.0 - weights) * remaining_actions[:num_smoothed]
                    )
                    self.last_actions = np.concatenate([smoothed, new_action[num_smoothed:]], axis=0)
                else:
                    self.last_actions = new_action
                self.action_counter = 0

    def _step_temporal_ensemble(self, obs: Dict[str, Any]) -> np.ndarray:
        """Paper-faithful synchronous temporal ensembling (Zhao et al. ACT).

        One blocking inference per consumer tick: every call fires a fresh
        chunk, appends it to the deque (emit_tick = current consumer tick),
        and returns the exponentially-weighted average of every chunk that
        covers ``t``. Weights are ``exp(-k * age_index)`` with the deque
        sorted oldest-first → ``age = 0`` at the head, which gets weight 1.

        This blocks the AgentNode consumer for the duration of one inference
        per tick. If ``inference_time > 1/poll_freq`` the consumer rate
        degrades to whatever inference sustains; if it fits in budget, the
        ensemble has exactly ``min(t+1, chunk_size)`` populated entries at
        tick ``t`` (the steady-state count is ``chunk_size``).
        """
        chunk = np.asarray(self._client.infer(obs)["actions"])
        with self.action_lock:
            t = self._te_tick
            self._te_chunks.append((t, chunk))
            # GC: drop any chunk whose [emit, emit + chunk_size) window has
            # already passed. (Cheap because the deque is sorted oldest-first.)
            while self._te_chunks and (
                self._te_chunks[0][0] + self._te_chunks[0][1].shape[0] <= t
            ):
                self._te_chunks.popleft()
            self._te_latest_chunk = chunk
            self._te_latest_emit_tick = t

            # Gather every contribution for tick t. We just appended chunk[0]
            # for this tick so contribs is guaranteed non-empty; the loop
            # exists to also fold in the prior overlapping chunks.
            contribs: list[np.ndarray] = []
            for emit_tick, c in self._te_chunks:
                offset = t - emit_tick
                if 0 <= offset < c.shape[0]:
                    contribs.append(c[offset])
            stacked = np.stack(contribs, axis=0)  # (N, action_dim)
            ages = np.arange(stacked.shape[0], dtype=np.float32)
            weights = np.exp(-self._te_k * ages)
            weights = weights / weights.sum()
            action = (stacked * weights[:, None]).sum(axis=0).astype(np.float32)
            self._te_tick += 1
        return action

    def select_action(self) -> np.ndarray:
        return self._select_action_blend()

    def _select_action_blend(self) -> np.ndarray:
        while self.last_actions is None and not self._stop.is_set():
            time.sleep(0.01)
        if self._stop.is_set():
            raise RuntimeError("AsyncRemoteACTAgent was closed before the first action became available")
        with self.action_lock:
            buf_len = self.last_actions.shape[0]
            idx = min(self.action_counter, buf_len - 1)
            action = self.last_actions[idx]
            if self.action_counter >= buf_len - 1:
                if self.action_counter == buf_len - 1:
                    print(
                        f"[AsyncRemoteACTAgent] inference lag -- repeating action at "
                        f"counter {self.action_counter} (buf_len={buf_len}, "
                        f"action_horizon={self.action_horizon})"
                    )
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
        with self.action_lock:
            self.last_actions = None
            self.action_counter = 0
            self._te_chunks.clear()
            self._te_tick = 0
            self._te_latest_chunk = None
            self._te_latest_emit_tick = 0
        if self._broker is not None:
            self._broker.reset()
