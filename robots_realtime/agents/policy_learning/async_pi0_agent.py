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

Five inference modes (``inference_mode`` kwarg):

    sync                — blocking, synchronous. Uses OpenPI's ``ActionChunkBroker``
                          so inference only fires when the last chunk is exhausted
                          (every ``action_horizon`` consumer calls).
    async               — background thread, runs inference flat-out by default.
    async_rate_limited  — background thread, rate-capped (``inference_interval_s``
                          REQUIRED).
    async_rtc           — background thread + Real-Time Chunking, using
                          the paper-correct SCHEDULED cadence: one inference
                          per chunk cycle, triggered when the buffer has
                          ≈ d_est actions remaining (d_est = rolling-EMA
                          inference latency in consumer ticks). When the
                          chunk lands, the old buffer has drained to ~0
                          remaining and the new chunk's [d_actual:] tail
                          replaces it seamlessly. Matches LeRobot's
                          ActionQueue replace-on-merge semantics.
    temporal_ensemble   — ACT-paper inference scheme (Zhao et al.) — synchronous
                          and paper-faithful. One inference per consumer tick,
                          blocking. At tick ``t`` every overlapping chunk
                          contributes via ``w_i = exp(-k * age_index)`` weighted
                          average; deque sorted oldest-first → ``age = 0`` at
                          the head (highest weight). Default
                          ``temporal_ensemble_k = 0.01`` matches the paper.
                          Requires ``inference_time < 1/poll_freq`` to hold the
                          consumer rate; otherwise the loop runs as fast as
                          inference allows.

``inference_interval_s`` is an **orthogonal** rate cap: REQUIRED for
``async_rate_limited``, OPTIONAL for ``async`` and ``async_rtc`` (``None`` =
flat-out). Useful with fast GPUs where flat-out inference re-infers so quickly
that chunks overlap almost entirely.

OpenPI client imports are done lazily inside ``__init__`` so this module can be
imported without ``openpi_client`` being installed — it's only required when
the agent is actually instantiated.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Literal, Tuple

import numpy as np
from dm_env.specs import Array

from robots_realtime.agents.agent import PolicyAgent
from robots_realtime.agents.constants import ActionSpec
from robots_realtime.robots.utils import Rate

InferenceMode = Literal["sync", "async", "async_rate_limited", "async_rtc", "temporal_ensemble"]
_ASYNC_MODES = ("async", "async_rate_limited", "async_rtc")

ImagePreprocess = Literal["center_crop", "pad"]


def _center_crop_and_resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop to the largest square that fits, then resize to (target_h, target_w).

    Must match the training augmentation for the deployed checkpoint.
    Lab42's OpenPI bimanual YAM runs use center-crop; choose the matching
    strategy per-model via ``AsyncDiffusionAgent(image_preprocess=...)``.

    Strategy: `min(H, W)`-sized square centered on (H/2, W/2), scaled to the
    target. Discards peripheral FOV instead of showing black bars to the model.
    """
    h, w = img.shape[:2]
    side = min(h, w)
    h0 = (h - side) // 2
    w0 = (w - side) // 2
    cropped = img[h0:h0 + side, w0:w0 + side]
    # Already square — resize_with_pad here adds zero padding, just rescales.
    from openpi_client.image_tools import resize_with_pad  # noqa: PLC0415
    return resize_with_pad(cropped, target_h, target_w)


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


class _InferTimingReporter:
    """Rolls up server-reported infer timing and effective rate, logs every ~2 s.

    Ground truth for answering "is inference firing once per chunk, or more?".
    Avoids a per-call print (noisy) while surfacing when something is off.
    """

    def __init__(self, name: str, report_interval_s: float = 2.0) -> None:
        self._name = name
        self._report_interval_s = float(report_interval_s)
        self._calls = 0
        self._sum_ms = 0.0
        self._last_ms = 0.0
        self._window_start = time.monotonic()

    def record(self, server_timing: Dict[str, Any]) -> None:
        ms = float(server_timing.get("infer_ms", 0.0))
        self._last_ms = ms
        self._sum_ms += ms
        self._calls += 1
        now = time.monotonic()
        elapsed = now - self._window_start
        if elapsed >= self._report_interval_s:
            hz = self._calls / elapsed if elapsed > 0 else 0.0
            avg = self._sum_ms / self._calls if self._calls else 0.0
            print(
                f"[{self._name}] server infer: {self._calls} calls in {elapsed:.2f}s "
                f"→ {hz:.2f} Hz, avg {avg:.1f} ms, last {self._last_ms:.1f} ms"
            )
            self._calls = 0
            self._sum_ms = 0.0
            self._window_start = now


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
        # --- Real-Time Chunking (RTC) --------------------------------------- #
        # Only used when ``inference_mode == "async_rtc"``. RTC sends the
        # unexecuted tail of the current chunk as ``action_prefix`` plus an
        # estimated ``inference_delay`` (rolling mean of past inference
        # durations times the consumer tick rate). Server-side flow-matching
        # (see openpi pi0.sample_actions_rtc) keeps the new chunk's early
        # steps coherent with what the robot is already committed to execute.
        rtc_execution_horizon: int | None = None,
        rtc_consumer_rate_hz: float = 30.0,
        # Upper clamp on the RTC guidance weight (server side). Leave at the
        # server's default (1.0) unless you've tuned a specific policy. Higher
        # values amplify the prefix-tracking correction — too high and the
        # guided velocity overshoots, causing chunks to start far off from
        # the observation ("arm shoots forward"). Lower is safer.
        rtc_max_guidance_weight: float | None = None,
        # Server-side per-denoising-step debug prints (gw, |v_t|, |corr|, |err|,
        # |x1_t|, |prefix|, time). Plus a client-side log of the chunk-jump
        # magnitude between the tail of the old chunk and the head of the new
        # one — catches "shoots forward" events where RTC pulled too hard.
        rtc_debug: bool = False,
        # --- Temporal-ensemble (Zhao et al. ACT) ----------------------------- #
        # Only used when ``inference_mode == "temporal_ensemble"``. Synchronous
        # paper-faithful TE: one blocking inference per consumer tick, append
        # chunk to a deque tagged with emit_tick, return the exp(-k * age_index)
        # weighted average of every chunk that covers tick t. Replaces the
        # linear-ramp blend used by the other async modes.
        # Higher k = more weight on older predictions (smoother, laggier).
        temporal_ensemble_k: float = 0.01,
        # MUST match the training-time image augmentation for your checkpoint.
        #   "center_crop": crop to a min(H,W)-sized square from the centre,
        #                  then resize. Discards peripheral FOV. Lab42 / PI
        #                  YAM checkpoints use this.
        #   "pad":         preserve full FOV, letterbox with black bars.
        # Mismatch = the model sees out-of-distribution inputs (black bars
        # or wrong FOV) and can produce unsafe actions.
        image_preprocess: ImagePreprocess = "center_crop",
    ) -> None:
        # Validate config first — user-error (bad mode) should surface before
        # env-error (missing openpi_client).
        valid_modes = ("sync", "async", "async_rate_limited", "async_rtc", "temporal_ensemble")
        if inference_mode not in valid_modes:
            raise ValueError(
                f"inference_mode must be one of {valid_modes}; got {inference_mode!r}"
            )
        if image_preprocess not in ("center_crop", "pad"):
            raise ValueError(
                f"image_preprocess must be 'center_crop' or 'pad'; got {image_preprocess!r}"
            )
        self._image_preprocess: ImagePreprocess = image_preprocess
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

        # Last preprocessed images in HWC uint8 form (pre-transpose) — exposed
        # to consumers via ``act()["_images"]`` so a monitor can display exactly
        # what the model sees without running the preproc pipeline twice.
        self._last_display_images: Dict[str, np.ndarray] = {}

        self.use_joint_state_as_action = use_joint_state_as_action
        self._websocket_client_policy = _websocket_client_policy.WebsocketClientPolicy(host=ip, port=port)
        # Wrap infer() to observe server-reported inference time on every real
        # server hit. In sync mode the ActionChunkBroker calls this once per
        # chunk; in async modes the _action_loop calls it directly. This is the
        # ground-truth inference rate regardless of which path is active.
        self._infer_timer = _InferTimingReporter(name=f"{type(self).__name__}")
        _raw_infer = self._websocket_client_policy.infer
        def _infer_instrumented(obs, _raw=_raw_infer, _rep=self._infer_timer):
            response = _raw(obs)
            _rep.record(response.get("server_timing") or {})
            return response
        self._websocket_client_policy.infer = _infer_instrumented

        self.action_horizon = action_horizon
        self.inference_mode: InferenceMode = inference_mode
        self.inference_interval_s = inference_interval_s
        self.min_smoothed_actions = int(min_smoothed_actions)
        self.max_smoothed_actions = int(max_smoothed_actions)
        # Rate cap is orthogonal to mode: any async mode can be rate-limited by
        # setting inference_interval_s > 0. REQUIRED when mode=async_rate_limited
        # (validated above). OPTIONAL (off by default) when mode=async_rtc —
        # useful when the server is so fast that flat-out inference produces
        # ~100% chunk overlap and you'd rather space requests out.
        self.inference_interval_rate = (
            Rate(1.0 / inference_interval_s, rate_name="inference_interval")
            if inference_interval_s is not None and inference_interval_s > 0
            else None
        )
        self.config = model_io_config or ModelIOConfig()

        self.action_lock = threading.Lock()
        self.last_actions: np.ndarray | None = None
        self.obs_lock = threading.Lock()
        self._obs: Dict[str, Any] | None = None
        self.action_counter = 0
        self._stop = threading.Event()

        # RTC state — derived directly from inference_mode.
        self._rtc_enabled = inference_mode == "async_rtc"
        self._rtc_execution_horizon = rtc_execution_horizon
        self._rtc_consumer_rate_hz = float(rtc_consumer_rate_hz)
        self._rtc_delay_ema_ticks: float = 0.0   # exponential moving average
        self._rtc_delay_ema_alpha: float = 0.3
        self._rtc_max_guidance_weight = rtc_max_guidance_weight
        self._rtc_debug = bool(rtc_debug)
        self._rtc_last_tail: np.ndarray | None = None   # for client-side chunk-jump log

        # Temporal-ensemble state. Deque of (emit_tick, chunk) sorted oldest-first;
        # ``self._te_tick`` is the absolute consumer tick (advances per call to
        # _step_temporal_ensemble). chunk[t - emit_tick] is the prediction for
        # tick t. ``self._te_latest_*`` are kept solely for _snapshot_chunk in
        # TE mode (where last_actions/action_counter aren't used).
        self._te_k = float(temporal_ensemble_k)
        self._te_chunks: Deque[Tuple[int, np.ndarray]] = deque()
        self._te_tick: int = 0
        self._te_latest_chunk: np.ndarray | None = None
        self._te_latest_emit_tick: int = 0

        # Mode wiring. Async modes spin a background inference thread; sync uses
        # ActionChunkBroker; temporal_ensemble runs blocking inline (no thread,
        # no broker — chunks merge into self._te_chunks per call).
        if inference_mode in _ASYNC_MODES:
            self.action_thread = threading.Thread(target=self._action_loop, name="AsyncDiffusionAgent_inference", daemon=True)
            self.action_thread.start()
            self._agent = None
            self._broker = None
        elif inference_mode == "sync":
            self._broker = action_chunk_broker.ActionChunkBroker(
                policy=self._websocket_client_policy,
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
        return {
            "action_horizon": self.action_horizon,
            "inference_mode": self.inference_mode,
            "inference_interval_s": self.inference_interval_s,
            "min_smoothed_actions": self.min_smoothed_actions,
            "max_smoothed_actions": self.max_smoothed_actions,
            "rtc_enabled": self._rtc_enabled,
            "rtc_execution_horizon": self._rtc_execution_horizon,
            "temporal_ensemble_k": self._te_k,
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
        display_images: Dict[str, np.ndarray] = {}
        for k in self.config.image_keys:
            img = flat[k]
            if self._image_preprocess == "center_crop":
                img = _center_crop_and_resize(img, 224, 224)
            else:  # "pad"
                img = self._image_tools.resize_with_pad(img, 224, 224)
            img = self._image_tools.convert_to_uint8(img)
            # Strip the suffix shared across flattened keys ("-images-rgb") so
            # the published topic reads openpi_policy/image/left_camera etc.
            display_label = k.split("-", 1)[0]
            display_images[display_label] = img
            images[k] = np.transpose(img, (2, 0, 1))
        self._last_display_images = display_images

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
            # Exact frames fed to the policy (post center-crop/pad, HWC uint8)
            # so downstream viewers don't duplicate the preprocessing pipeline.
            "_images": dict(self._last_display_images),
        }

    def _snapshot_chunk(self) -> Dict[str, Any] | None:
        """Return the still-unconsumed tail of the current action chunk, split by arm.

        Shape: {"left": (N, 7), "right": (N, 7)} for the 14-dim bimanual case.
        Returns None if the buffer is not yet populated or is empty. The tail
        starts at `action_counter` so consumers can interpret step 0 of the
        returned array as "the next action we'd dequeue".

        - async modes: slice ``last_actions[action_counter:]``.
        - temporal_ensemble: slice the latest chunk from its current offset
          forward, so the viz shows the freshest predicted plan.
        - sync: read the broker's internal ``_last_results["actions"]``.
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
        if self.inference_mode == "temporal_ensemble":
            # Paper-faithful synchronous TE: blocking inference per consumer
            # tick, then weighted ensemble over all overlapping chunks.
            return self._step_temporal_ensemble(self._obs)
        # Async modes: if the background thread hasn't produced the first chunk
        # yet, tell the consumer we're not ready rather than blocking step().
        if self.last_actions is None:
            return None
        return self.select_action()

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
        ensemble has exactly ``min(t+1, action_horizon)`` populated entries
        at tick ``t`` (steady-state count = action_horizon).
        """
        chunk = np.asarray(self._websocket_client_policy.infer(obs)["actions"])
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

    # ------------------------------------------------------------------ #
    # Async plumbing
    # ------------------------------------------------------------------ #

    def _action_loop(self) -> None:
        while not self._stop.is_set():
            # Wait for the first observation to arrive from the consumer thread.
            if self._obs is None:
                time.sleep(0.01)
                continue

            # SCHEDULED trigger for RTC (paper-correct pattern, matches
            # LeRobot's intended ActionQueue cadence):
            #
            # In async_rtc, fire inference exactly ONCE per chunk cycle,
            # at the moment the remaining buffer has ≈ d_est actions left.
            # That way, when the new chunk lands after d_actual ticks, the
            # consumer has drained the old chunk down to ~0 remaining and
            # we swap in the new chunk's [d_actual:] tail seamlessly.
            #
            # Non-RTC async modes stay flat-out / rate-capped (no trigger
            # gate) since they don't send a prefix and have no cycle to
            # align with.
            if (
                self._rtc_enabled
                and self.last_actions is not None
                and not self._stop.is_set()
            ):
                # Always leave a minimum safety margin so a first-inference
                # JIT-compile stall doesn't drain the chunk past empty.
                d_est = max(5, round(self._rtc_delay_ema_ticks))
                while not self._stop.is_set():
                    with self.action_lock:
                        la = self.last_actions
                        buf_len = la.shape[0] if la is not None else 0
                        remaining = buf_len - self.action_counter
                    if remaining <= d_est:
                        break
                    time.sleep(0.005)
                if self._stop.is_set():
                    return

            with self.obs_lock:
                current_obs = dict(self._obs)   # shallow copy so we can add RTC fields
            with self.action_lock:
                start_inference_action_counter = self.action_counter
                # RTC: capture the unexecuted tail of the current chunk as the
                # prefix we want the server to stay coherent with.
                if self._rtc_enabled and self.last_actions is not None:
                    unexecuted_tail = self.last_actions[start_inference_action_counter:]
                    # PAD TO FIXED LENGTH (action_horizon) — the server's
                    # sample_actions_rtc is JIT-compiled, and JAX re-JITs on
                    # every new input shape. If we send a variable-length
                    # prefix (which naturally shrinks as action_counter advances)
                    # we trigger a fresh compile (~10 s!) every inference.
                    # Padding with the tail's last action makes the "unused"
                    # positions request zero-change — server weights these at
                    # execution_horizon and below, so the padded region is
                    # effectively ignored.
                    T = self.action_horizon
                    tail_len = unexecuted_tail.shape[0]
                    if tail_len >= T:
                        rtc_prefix = np.ascontiguousarray(unexecuted_tail[:T], dtype=np.float32)
                    elif tail_len > 0:
                        pad = np.broadcast_to(unexecuted_tail[-1:], (T - tail_len, unexecuted_tail.shape[1]))
                        rtc_prefix = np.ascontiguousarray(
                            np.concatenate([unexecuted_tail, pad], axis=0), dtype=np.float32
                        )
                    else:
                        rtc_prefix = None
                    # Note how many of the T positions are "real" — server
                    # uses execution_horizon to weight only those.
                    real_prefix_len = int(min(max(tail_len, 0), T))
                else:
                    rtc_prefix = None
                    real_prefix_len = 0

            if rtc_prefix is not None and real_prefix_len > 0:
                current_obs["action_prefix"] = rtc_prefix
                current_obs["inference_delay"] = round(self._rtc_delay_ema_ticks)
                # Tell the server how many of the T positions actually represent
                # committed future actions; the rest are padding that should
                # not be guided toward.
                current_obs["execution_horizon"] = (
                    int(self._rtc_execution_horizon)
                    if self._rtc_execution_horizon is not None
                    else real_prefix_len
                )
                if self._rtc_max_guidance_weight is not None:
                    current_obs["max_guidance_weight"] = float(self._rtc_max_guidance_weight)
                if self._rtc_debug:
                    current_obs["rtc_debug"] = True

            # Snapshot the last action we'll be playing before this inference
            # returns — used to measure the chunk-jump at merge time below.
            if self._rtc_debug and self.last_actions is not None:
                with self.action_lock:
                    tail_idx = min(self.action_counter + 1, self.last_actions.shape[0] - 1)
                    self._rtc_last_tail = self.last_actions[tail_idx].copy()
            else:
                self._rtc_last_tail = None

            t_infer_start = time.monotonic()
            inferred_action = np.asarray(self._websocket_client_policy.infer(current_obs)["actions"])
            infer_dt_s = time.monotonic() - t_infer_start

            # Update the RTC latency EMA (in consumer ticks) for the *next*
            # request. This predicts how far ahead the robot will have drifted
            # by the time that next chunk lands.
            if self._rtc_enabled:
                delay_ticks = infer_dt_s * self._rtc_consumer_rate_hz
                a = self._rtc_delay_ema_alpha
                self._rtc_delay_ema_ticks = (1.0 - a) * self._rtc_delay_ema_ticks + a * delay_ticks

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
                elif new_action.shape[0] < 2 and self.last_actions.shape[0] >= 2:
                    # Degenerate new chunk (inference was so slow the time-align
                    # skip ate almost everything — typically the first RTC call
                    # while the server JIT compiles the sample_actions_rtc
                    # graph, ~10s on a cold start). Commit the sacrifice: hold
                    # the existing buffer, repeat its last action until the
                    # next (now-warm) inference lands. Avoids an IndexError in
                    # select_action and a worse 'held empty chunk' behaviour.
                    print(
                        f"[AsyncDiffusionAgent] discarding length-{new_action.shape[0]} chunk "
                        f"(consumed_during_inference={consumed_during_inference}, "
                        f"infer_dt={infer_dt_s*1e3:.0f}ms) — keeping old buffer"
                    )
                    # Don't reset counter; keep repeating the old buffer's tail.
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

                # Client-side RTC diagnostics: how far is the first action of
                # the new chunk from the last action we were about to play?
                # Big jumps are what cause "arm shoots forward" — RTC should
                # keep this small when it's working.
                if self._rtc_debug and self._rtc_last_tail is not None:
                    new_head = np.asarray(new_action[0])
                    jump = new_head - self._rtc_last_tail
                    arm_norm = float(np.linalg.norm(jump[:6])) if jump.shape[0] >= 6 else float("nan")
                    full_norm = float(np.linalg.norm(jump))
                    print(
                        f"[AsyncDiffusionAgent.rtc] chunk-jump  full_norm={full_norm:.4f}  "
                        f"arm_norm={arm_norm:.4f}  consumed_during_inference={consumed_during_inference}  "
                        f"infer_dt={infer_dt_s*1e3:.0f}ms  "
                        f"server_chunk_len={server_chunk_len}  new_len={new_action.shape[0]}  "
                        f"delay_ema_ticks={self._rtc_delay_ema_ticks:.1f}"
                    )

            # Rate cap. async_rtc paces itself via the scheduled trigger at
            # the top of the loop (one inference per chunk cycle), so the
            # rate cap would double-pace — skip it there. Pure async keeps
            # it if configured.
            if self.inference_interval_rate is not None and not self._rtc_enabled:
                self.inference_interval_rate.sleep()
            # else: flat-out, loop back immediately.

    def select_action(self) -> np.ndarray:
        # Wait for the first inference to land.
        while self.last_actions is None and not self._stop.is_set():
            time.sleep(0.01)
        if self._stop.is_set():
            raise RuntimeError("AsyncDiffusionAgent was closed before the first action became available")
        with self.action_lock:
            # Cap by the CURRENT buffer length, not action_horizon — a slow
            # inference (e.g. first-RTC-call JIT compile, 10s) plus the
            # time-align skip can leave self.last_actions much shorter than
            # action_horizon. Capping by action_horizon would index out of
            # bounds on the short buffer and crash the subprocess.
            buf_len = self.last_actions.shape[0]
            idx = min(self.action_counter, buf_len - 1)
            action = self.last_actions[idx]
            if self.action_counter >= buf_len - 1:
                # Inference lagging — hold the final action of the current
                # buffer (we're repeating until a new chunk lands).
                if self.action_counter == buf_len - 1:
                    print(
                        f"[AsyncDiffusionAgent] inference lag — repeating action at "
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
        # Drop buffered actions so the next chunk is freshly produced. Retain the
        # background thread — it will infer again as soon as a new obs lands.
        with self.action_lock:
            self.last_actions = None
            self.action_counter = 0
            self._te_chunks.clear()
            self._te_tick = 0
            self._te_latest_chunk = None
            self._te_latest_emit_tick = 0
        if self._broker is not None:
            self._broker.reset()
