"""Async ACT (Action Chunking Transformer) policy agent for local PyTorch inference.

Runs as a regular ``Agent`` inside a ZMQ ``AgentNode``:

    AgentNode.step()  (at poll_freq, e.g. 30 Hz)
        +-- agent.act(obs)
              +-- self.__call__(obs)
                    |-- self._obs = obs_to_model_input(obs)   # lock-protected
                    +-- self.select_action()                   # dequeue from chunk buffer

Inference runs independently in a background thread (``_action_loop``)
that reads the latest ``self._obs`` snapshot, runs a local ACT forward pass,
and merges the returned action chunk into ``self.last_actions`` with a linear
ramp blend at the chunk boundary.

Two inference modes:

    async               -- background thread, runs inference flat-out by default.
    async_rate_limited  -- background thread, rate-capped (``inference_interval_s``
                           REQUIRED).

ACT imports are done lazily inside ``__init__`` so this module can be imported
without the ``act`` package being installed.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Tuple

import numpy as np
from dm_env.specs import Array

from robots_realtime.agents.agent import PolicyAgent
from robots_realtime.agents.constants import ActionSpec
from robots_realtime.robots.utils import Rate

InferenceMode = Literal["async", "async_rate_limited"]

ImagePreprocess = Literal["center_crop", "resize"]


def _center_crop_and_resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop to the largest square that fits, then resize to (target_h, target_w)."""
    import cv2

    h, w = img.shape[:2]
    side = min(h, w)
    h0 = (h - side) // 2
    w0 = (w - side) // 2
    cropped = img[h0 : h0 + side, w0 : w0 + side]
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized


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
    """Flatten a nested dict into {key: value} with ``sep``-joined paths."""
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
    """What obs keys the ACT model expects and what action keys it returns.

    Defaults match the YAM bimanual schema (matching ACT's YAMDataConfig).
    Override via kwargs if your model was trained with different keys.
    """

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


class AsyncACTAgent(PolicyAgent):
    """Local ACT policy wrapper with chunked async inference."""

    def __init__(
        self,
        use_joint_state_as_action: bool = False,
        checkpoint_path: str = "",
        norm_stats_path: str | None = None,
        use_quantile_norm: bool = True,
        action_dim: int = 14,
        proprio_dim: int = 14,
        action_chunk_size: int = 30,
        backbone: str = "dinov3_vits16",
        encoder_image_size: int = 256,
        hidden_size: int = 384,
        num_heads: int = 12,
        num_blocks: int = 3,
        relative_actions: bool = False,
        freeze_encoder: bool = True,
        device: str = "cuda",
        inference_mode: InferenceMode = "async_rate_limited",
        inference_interval_s: float | None = 0.5,
        min_smoothed_actions: int = 1,
        max_smoothed_actions: int = 8,
        model_io_config: ACTModelIOConfig | None = None,
        image_preprocess: ImagePreprocess = "center_crop",
        # Delta-action conversion (must match training config)
        use_delta_actions: bool = False,
        delta_action_mask: Tuple[bool, ...] | None = None,
    ) -> None:
        if inference_mode not in ("async", "async_rate_limited"):
            raise ValueError(
                f"inference_mode must be 'async' or 'async_rate_limited'; got {inference_mode!r}"
            )
        if image_preprocess not in ("center_crop", "resize"):
            raise ValueError(
                f"image_preprocess must be 'center_crop' or 'resize'; got {image_preprocess!r}"
            )
        self._image_preprocess: ImagePreprocess = image_preprocess
        if inference_mode == "async_rate_limited" and (inference_interval_s is None or inference_interval_s <= 0):
            raise ValueError("inference_mode='async_rate_limited' requires inference_interval_s > 0")
        if min_smoothed_actions > max_smoothed_actions:
            raise ValueError(
                f"min_smoothed_actions ({min_smoothed_actions}) cannot exceed "
                f"max_smoothed_actions ({max_smoothed_actions})"
            )

        try:
            import torch
            from act.model.act_agent import ACTAgent as _ACTAgent
            from act.config import ModelConfig
            from act.transforms import Normalize, Unnormalize, NormStats, AbsoluteActions, make_bool_mask
            from act.data.normalization import load_norm_stats
        except ImportError as exc:
            raise ImportError(
                "AsyncACTAgent requires the `act` package. Install it into this venv "
                "before instantiating the agent."
            ) from exc

        self._torch = torch
        self.use_joint_state_as_action = use_joint_state_as_action
        self.action_chunk_size = action_chunk_size
        self.inference_mode: InferenceMode = inference_mode
        self.inference_interval_s = inference_interval_s
        self.min_smoothed_actions = int(min_smoothed_actions)
        self.max_smoothed_actions = int(max_smoothed_actions)
        self.inference_interval_rate = (
            Rate(1.0 / inference_interval_s, rate_name="inference_interval")
            if inference_interval_s is not None and inference_interval_s > 0
            else None
        )
        self.config = model_io_config or ACTModelIOConfig()

        self._device = device
        self._encoder_image_size = encoder_image_size
        self._use_delta_actions = use_delta_actions
        self._delta_action_mask = delta_action_mask

        model_config = ModelConfig(
            action_dim=action_dim,
            action_chunk_size=action_chunk_size,
            proprio_dim=proprio_dim,
            backbone=backbone,
            encoder_image_size=encoder_image_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_blocks=num_blocks,
            relative_actions=relative_actions,
            freeze_encoder=freeze_encoder,
        )

        self._model = _ACTAgent(model_config).to(device)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self._model.load_state_dict(state_dict)
            print(f"[AsyncACTAgent] loaded checkpoint from {checkpoint_path}")
        self._model.eval()

        norm_stats = load_norm_stats(norm_stats_path) if norm_stats_path else None
        self._normalize = Normalize(norm_stats, use_quantiles=use_quantile_norm)
        self._unnormalize = Unnormalize(norm_stats, use_quantiles=use_quantile_norm)

        if use_delta_actions:
            mask = delta_action_mask or make_bool_mask(action_dim)
            self._absolute_actions = AbsoluteActions(mask)
        else:
            self._absolute_actions = None

        self.action_lock = threading.Lock()
        self.last_actions: np.ndarray | None = None
        self.obs_lock = threading.Lock()
        self._obs: Dict[str, Any] | None = None
        self.action_counter = 0
        self._stop = threading.Event()

        self.action_thread = threading.Thread(
            target=self._action_loop,
            name="AsyncACTAgent_inference",
            daemon=True,
        )
        self.action_thread.start()

    # ------------------------------------------------------------------ #
    # Metadata / specs
    # ------------------------------------------------------------------ #

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "action_chunk_size": self.action_chunk_size,
            "inference_mode": self.inference_mode,
            "inference_interval_s": self.inference_interval_s,
            "min_smoothed_actions": self.min_smoothed_actions,
            "max_smoothed_actions": self.max_smoothed_actions,
            "device": self._device,
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
        """Flatten bus-message obs into the dict shape the ACT model expects.

        Returns None if any required state / image key is missing (producers
        still warming up).
        """
        flat = _recursive_flatten(obs)

        required = list(self.config.state_keys) + list(self.config.image_keys)
        missing = [k for k in required if k not in flat]
        if missing:
            now = time.monotonic()
            if now - getattr(self, "_last_missing_log_ts", 0.0) > 2.0:
                preview = ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")
                print(f"[AsyncACTAgent] obs not ready -- waiting on: {preview}")
                self._last_missing_log_ts = now
            return None

        flat_state = [np.asarray(flat[k]).reshape(-1) for k in self.config.state_keys]
        state = np.concatenate(flat_state, axis=-1).astype(np.float32)

        images: Dict[str, np.ndarray] = {}
        for k in self.config.image_keys:
            img = np.asarray(flat[k])
            target_size = self._encoder_image_size
            if self._image_preprocess == "center_crop":
                img = _center_crop_and_resize(img, target_size, target_size)
            else:
                img = _resize(img, target_size, target_size)
            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0
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
        """Return the still-unconsumed tail of the current action chunk, split by arm."""
        with self.action_lock:
            if self.last_actions is None:
                return None
            remaining = self.last_actions[self.action_counter:]
        if remaining.shape[0] == 0 or remaining.ndim != 2:
            return None
        if remaining.shape[1] == 14:
            return {
                "left": np.ascontiguousarray(remaining[:, :7], dtype=np.float32),
                "right": np.ascontiguousarray(remaining[:, 7:], dtype=np.float32),
            }
        if remaining.shape[1] == 28:
            return {
                "left": np.ascontiguousarray(remaining[:, :7], dtype=np.float32),
                "right": np.ascontiguousarray(remaining[:, 14:21], dtype=np.float32),
            }
        return None

    def __call__(self, obs: Dict[str, Any]) -> np.ndarray | None:
        model_input = self.obs_to_model_input(obs)
        if model_input is None:
            return None
        with self.obs_lock:
            self._obs = model_input
        if self.last_actions is None:
            return None
        return self.select_action()

    # ------------------------------------------------------------------ #
    # Async plumbing
    # ------------------------------------------------------------------ #

    def _action_loop(self) -> None:
        torch = self._torch

        while not self._stop.is_set():
            if self._obs is None:
                time.sleep(0.01)
                continue

            with self.obs_lock:
                current_obs = {k: v if not isinstance(v, dict) else dict(v) for k, v in self._obs.items()}
            with self.action_lock:
                start_inference_action_counter = self.action_counter

            normalized = self._normalize({
                "state": current_obs["state"].copy(),
            })
            state_tensor = torch.from_numpy(normalized["state"]).float().to(self._device)
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D) = (T, B, D)

            images_dict = {}
            for cam_name, img in current_obs["images"].items():
                img_t = torch.from_numpy(img).float().to(self._device)
                if img_t.ndim == 3 and img_t.shape[-1] in (1, 3):
                    img_t = img_t.permute(2, 0, 1)  # HWC -> CHW
                img_t = img_t.unsqueeze(0).unsqueeze(0)  # (B=1, T=1, C, H, W)
                images_dict[cam_name] = img_t

            obs_dict = {
                "proprio": state_tensor,
                "images": images_dict,
            }

            with torch.no_grad():
                actions = self._model(obs_dict)  # (T=1, B=1, chunk_size, action_dim)

            actions_np = actions[0, 0].cpu().numpy().astype(np.float32)  # (chunk_size, action_dim)

            unnorm_data = {"actions": actions_np}
            if self._absolute_actions is not None:
                unnorm_data["state"] = current_obs["state"]
            unnorm_data = self._unnormalize(unnorm_data)
            if self._absolute_actions is not None:
                unnorm_data = self._absolute_actions(unnorm_data)
            inferred_action = unnorm_data["actions"]

            with self.action_lock:
                complete_inference_action_counter = self.action_counter
                consumed_during_inference = max(0, complete_inference_action_counter - start_inference_action_counter)

                server_chunk_len = inferred_action.shape[0]
                skip = consumed_during_inference
                if skip >= server_chunk_len:
                    print(
                        f"[AsyncACTAgent] inference latency ({skip} ticks) >= chunk "
                        f"length ({server_chunk_len}); can't time-align, resetting to chunk head"
                    )
                    skip = 0
                new_action = inferred_action[skip:]

                if self.last_actions is None:
                    self.last_actions = new_action
                elif new_action.shape[0] < 2 and self.last_actions.shape[0] >= 2:
                    print(
                        f"[AsyncACTAgent] discarding length-{new_action.shape[0]} chunk "
                        f"(consumed_during_inference={consumed_during_inference}) -- keeping old buffer"
                    )
                else:
                    remaining_actions = self.last_actions[self.action_counter:]
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

    def select_action(self) -> np.ndarray:
        while self.last_actions is None and not self._stop.is_set():
            time.sleep(0.01)
        if self._stop.is_set():
            raise RuntimeError("AsyncACTAgent was closed before the first action became available")
        with self.action_lock:
            buf_len = self.last_actions.shape[0]
            idx = min(self.action_counter, buf_len - 1)
            action = self.last_actions[idx]
            if self.action_counter >= buf_len - 1:
                if self.action_counter == buf_len - 1:
                    print(
                        f"[AsyncACTAgent] inference lag -- repeating action at "
                        f"counter {self.action_counter} (buf_len={buf_len}, "
                        f"action_chunk_size={self.action_chunk_size})"
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
