"""Side-by-side viewer: LeRobot episode frames vs. live cameras.

Usage:
    uv run python scripts/compare_episode_camera.py \
        --dataset-path /path/to/local/lerobot/dataset \
        --camera-views top left right

Browser controls:
    Episode and Frame sliders browse the local dataset.
    The red move button commands both arms to the displayed frame.
    Ctrl-C in the terminal stops the server.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import importlib
import json
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml as _yaml

from robots_realtime.sensors.cameras.camera_utils import resize_with_pad


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare LeRobot episode frames with live cameras")
    p.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Local LeRobot v3 dataset root. Enables arbitrary-frame scrubbing and motion.",
    )
    p.add_argument("--host", default="0.0.0.0", help="Viser bind host")
    p.add_argument("--port", type=int, default=8091, help="Viser web port")
    p.add_argument(
        "--viser-port",
        type=int,
        default=None,
        help="Embedded Viser port (default: --port + 1)",
    )
    p.add_argument("--opacity", type=float, default=0.5, help="Initial camera overlay opacity")
    p.add_argument(
        "--urdf-path",
        default=None,
        help="Override the YAM URDF path from --camera-config",
    )
    p.add_argument(
        "--s3-path",
        default="xdof-internal-research/repromo/hlm_tshirt_reward_select_lerobot_sarm_8stage",
        help="S3 bucket/prefix (no s3:// prefix)",
    )
    p.add_argument(
        "--camera-serial",
        type=str,
        default=None,
        help="RealSense serial for a single live camera (overrides --camera-config mapping)",
    )
    p.add_argument(
        "--camera-config",
        type=str,
        default=None,
        help="Session YAML used as the source for CameraNode, RobotNode, policy, and Viser settings; maps views -> RealSense serials "
        "(default: auto-detect from configs/ by matching connected serials)",
    )
    p.add_argument(
        "--camera-views",
        nargs="+",
        default=["top"],
        choices=["top", "left", "right"],
        help="Which dataset camera views to show (observation.images.{view})",
    )
    p.add_argument(
        "--cache-dir", type=str, default=None, help="Directory to cache downloaded frames (default: tmpdir)"
    )
    p.add_argument("--no-camera", action="store_true", help="Skip live camera, just browse episodes")
    p.add_argument(
        "--left-robot-config",
        default=None,
        help="YAML config for left arm hardware (default: auto-detect from available CAN channels)",
    )
    p.add_argument(
        "--right-robot-config",
        default=None,
        help="YAML config for right arm hardware (default: auto-detect from available CAN channels)",
    )
    p.add_argument("--move-duration", type=float, default=2.0, help="Seconds to interpolate to target pose")
    p.add_argument("--no-robot", action="store_true", help="Disable robot control (browse-only)")
    p.add_argument(
        "--fov-crop",
        type=float,
        default=None,
        help="Initial top-camera publish_fov_crop fraction before resize (in (0, 1]; "
        "1.0 = off, 0.88 = 12%% tighter FOV). It can be changed live in the webpage; "
        "wrist cameras retain their configured FOV.",
    )
    args = p.parse_args()
    if args.dataset_path is not None:
        dataset_path = Path(args.dataset_path).expanduser()
        if not (dataset_path / "meta" / "info.json").is_file():
            p.error(f"--dataset-path is not a LeRobot dataset root: {dataset_path}")
    if args.fov_crop is not None and not (0.0 < args.fov_crop <= 1.0):
        p.error(f"--fov-crop must be in (0, 1], got {args.fov_crop}")
    if not (0.0 <= args.opacity <= 1.0):
        p.error(f"--opacity must be in [0, 1], got {args.opacity}")
    if args.viser_port is None:
        args.viser_port = args.port + 1
    if args.viser_port == args.port:
        p.error("--viser-port must differ from --port")
    return args


def s3_cp(s3_uri: str, local_path: str) -> bool:
    result = subprocess.run(
        ["aws", "s3", "cp", s3_uri, local_path],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def load_dataset_info(s3_prefix: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        uri = f"s3://{s3_prefix}/meta/info.json"
        if not s3_cp(uri, f.name):
            print(f"Failed to download {uri}", file=sys.stderr)
            sys.exit(1)
        return json.loads(Path(f.name).read_text())


def get_video_s3_key(s3_prefix: str, view: str, episode_idx: int, chunks_size: int) -> str:
    chunk_idx = episode_idx // chunks_size
    file_idx = episode_idx % chunks_size
    return f"s3://{s3_prefix}/videos/observation.images.{view}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"


def extract_first_frame(video_path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def download_first_frame(
    s3_prefix: str, view: str, episode_idx: int, chunks_size: int, cache_dir: Path
) -> np.ndarray | None:
    cache_file = cache_dir / f"ep{episode_idx:06d}_{view}.jpg"
    if cache_file.exists():
        img = cv2.imread(str(cache_file))
        if img is not None:
            return img

    s3_uri = get_video_s3_key(s3_prefix, view, episode_idx, chunks_size)
    print(f"\n  [S3] downloading {view} video: {s3_uri}", flush=True)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    if not s3_cp(s3_uri, tmp_path):
        Path(tmp_path).unlink(missing_ok=True)
        return None

    frame = extract_first_frame(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    if frame is not None:
        cv2.imwrite(str(cache_file), frame)

    return frame


def download_parquet(s3_prefix: str, chunk_idx: int, file_idx: int, cache_dir: Path) -> Path | None:
    cache_file = cache_dir / f"data_chunk{chunk_idx:03d}_file{file_idx:03d}.parquet"
    if cache_file.exists():
        return cache_file
    s3_uri = f"s3://{s3_prefix}/data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
    print(f"\n  [S3] downloading state parquet: {s3_uri}", flush=True)
    if not s3_cp(s3_uri, str(cache_file)):
        return None
    return cache_file


def get_episode_first_state(s3_prefix: str, episode_idx: int, chunks_size: int, cache_dir: Path) -> np.ndarray | None:
    chunk_idx = episode_idx // chunks_size
    file_idx = episode_idx % chunks_size
    parquet_path = download_parquet(s3_prefix, chunk_idx, file_idx, cache_dir)
    if parquet_path is None:
        return None
    table = pq.read_table(str(parquet_path), columns=["observation.state", "episode_index", "frame_index"])
    mask = pc.and_(
        pc.equal(table.column("episode_index"), episode_idx),
        pc.equal(table.column("frame_index"), 0),
    )
    rows = table.filter(mask)
    if len(rows) == 0:
        return None
    return np.array(rows.column("observation.state")[0].as_py(), dtype=np.float64)


def _local_view_name(video_key: str) -> str:
    """Normalize common LeRobot video keys to top/left/right view names."""
    name = video_key.removeprefix("observation.images.").removesuffix("-images-rgb")
    if name.endswith("_camera"):
        name = name[: -len("_camera")]
    if name.startswith("camera_"):
        name = name[len("camera_") :]
    return name


def load_local_episode_metadata(dataset_root: Path) -> dict[int, dict]:
    metadata_files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not metadata_files:
        raise ValueError(f"No episode metadata under {dataset_root / 'meta' / 'episodes'}")
    episodes: dict[int, dict] = {}
    for path in metadata_files:
        for row in pq.read_table(path).to_pylist():
            episodes[int(row["episode_index"])] = row
    return episodes


def resolve_local_video_keys(info: dict, views: list[str]) -> dict[str, str]:
    by_view = {
        _local_view_name(key): key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    }
    missing = [view for view in views if view not in by_view]
    if missing:
        raise ValueError(f"Dataset has no video views {missing}; available views: {sorted(by_view)}")
    return {view: by_view[view] for view in views}


def _format_dataset_path(template: str, chunk_index: int, file_index: int, **extra) -> str:
    return template.format(chunk_index=chunk_index, file_index=file_index, **extra)


def load_local_episode_states(dataset_root: Path, info: dict, episode: dict) -> np.ndarray:
    chunk_index = int(episode["data/chunk_index"])
    file_index = int(episode["data/file_index"])
    relative = _format_dataset_path(info["data_path"], chunk_index, file_index)
    path = dataset_root / relative
    schema_names = set(pq.read_schema(path).names)
    state_key = "observation.state" if "observation.state" in schema_names else "state"
    if state_key not in schema_names:
        raise ValueError(f"No state column in {path}")
    table = pq.read_table(
        path,
        columns=[state_key, "episode_index", "frame_index"],
        filters=[("episode_index", "=", int(episode["episode_index"]))],
    )
    if len(table) == 0:
        raise ValueError(f"No data rows for episode {episode['episode_index']}")
    order = np.argsort(table.column("frame_index").to_numpy())
    states = np.asarray(table.column(state_key).to_pylist(), dtype=np.float64)
    return states[order]


def read_local_video_frame(
    dataset_root: Path,
    info: dict,
    episode: dict,
    video_key: str,
    frame_index: int,
) -> np.ndarray | None:
    prefix = f"videos/{video_key}"
    chunk_index = int(episode[f"{prefix}/chunk_index"])
    file_index = int(episode[f"{prefix}/file_index"])
    relative = _format_dataset_path(info["video_path"], chunk_index, file_index, video_key=video_key)
    video_path = dataset_root / relative
    fps = float(info.get("features", {}).get(video_key, {}).get("info", {}).get("video.fps", info.get("fps", 30)))
    timestamp = float(episode[f"{prefix}/from_timestamp"]) + frame_index / fps
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


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


def connected_realsense_serials() -> set[str]:
    """Serial numbers of RealSense devices currently attached (via rs-enumerate-devices)."""
    try:
        out = subprocess.run(["rs-enumerate-devices", "-s"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return set()
    # Serial column is a long digit run, e.g. 427622273494.
    return set(re.findall(r"\b\d{9,}\b", out.stdout))


def load_camera_serials(config_path: str) -> dict[str, str]:
    """Map dataset view -> RealSense serial from a session config's CameraNode entries.

    CameraNode named 'camera_top' maps to view 'top', 'camera_left' -> 'left', etc.
    """
    with open(config_path) as f:
        cfg = _yaml.safe_load(f) or {}
    mapping: dict[str, str] = {}
    for node in cfg.get("nodes", []) or []:
        if not isinstance(node, dict) or node.get("type") != "CameraNode":
            continue
        name = str(node.get("name", ""))
        dev = node.get("device_id")
        if name.startswith("camera_") and dev:
            mapping[name[len("camera_") :]] = str(dev)
    return mapping


def load_camera_preprocess_specs(config_path: str | None, views: list[str]) -> dict[str, dict]:
    defaults = {view: {"resize": (224, 224), "resize_mode": "center_crop", "fov_crop": 1.0} for view in views}
    if config_path is None:
        return defaults
    with open(config_path) as stream:
        config = _yaml.safe_load(stream) or {}
    for node in config.get("nodes", []):
        if not isinstance(node, dict) or node.get("type") != "CameraNode":
            continue
        name = str(node.get("name", ""))
        if not name.startswith("camera_"):
            continue
        view = name[len("camera_") :]
        if view not in defaults:
            continue
        resize = node.get("publish_resize")
        defaults[view] = {
            "resize": tuple(int(value) for value in resize) if resize is not None else None,
            "resize_mode": str(node.get("publish_resize_mode", "center_crop")),
            "fov_crop": float(node.get("publish_fov_crop", 1.0)),
        }
    return defaults


def preprocess_live_camera_frame(frame: np.ndarray, spec: dict, fov_override: float | None = None) -> np.ndarray:
    fov_crop = float(spec["fov_crop"] if fov_override is None else fov_override)
    output = center_fov_crop(frame, fov_crop)
    resize = spec.get("resize")
    if resize is None:
        return output
    target_h, target_w = resize
    mode = spec.get("resize_mode", "center_crop")
    if mode == "pad":
        return resize_with_pad(output, target_h, target_w)
    if mode == "center_crop":
        return cv2.resize(center_crop_square(output), (target_w, target_h))
    raise ValueError(f"Unsupported camera resize mode: {mode!r}")


def load_robot_node_specs(config_path: str | None) -> dict[str, dict]:
    """Load per-arm robot paths and station overrides from a session config."""
    if config_path is None:
        return {}
    with open(config_path) as stream:
        config = _yaml.safe_load(stream) or {}
    result: dict[str, dict] = {}
    for node in config.get("nodes", []):
        if not isinstance(node, dict) or node.get("type") != "RobotNode":
            continue
        name = str(node.get("name", ""))
        arm = "left" if "left" in name else "right" if "right" in name else None
        if arm is None:
            continue
        result[arm] = {
            "config_path": node.get("robot_config"),
            "overrides": dict(node.get("robot_config_overrides") or {}),
        }
    return result


def load_policy_flip_joint_order(config_path: str | None) -> bool:
    if config_path is None:
        return False
    with open(config_path) as stream:
        config = _yaml.safe_load(stream) or {}
    for node in config.get("nodes", []):
        if not isinstance(node, dict) or node.get("type") != "AgentNode":
            continue
        agent_config = (node.get("agent_kwargs") or {}).get("config") or {}
        if "flip_joint_order" in agent_config:
            return bool(agent_config["flip_joint_order"])
        agent_kwargs = node.get("agent_kwargs") or {}
        if "flip_joint_order" in agent_kwargs:
            return bool(agent_kwargs["flip_joint_order"])
    return False


def flip_bimanual_yam_joint_order(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[-1] != 14:
        raise ValueError(f"Expected a 14-D bimanual YAM vector, got {values.shape}")
    return np.concatenate(
        (values[..., 5::-1], values[..., 6:7], values[..., 12:6:-1], values[..., 13:14]),
        axis=-1,
    )


def resolve_camera_map(camera_config: str | None, views: list[str]) -> dict[str, str]:
    """Resolve {view: serial} either from an explicit config or by auto-detecting the
    configs/ entry whose camera serials match the RealSense devices attached here."""
    if camera_config:
        m = load_camera_serials(camera_config)
        print(f"Camera map from {camera_config}: {m}")
        return m

    connected = connected_realsense_serials()
    if not connected:
        return {}
    best: tuple[int, str, dict[str, str]] | None = None
    for path in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
        try:
            m = load_camera_serials(path)
        except Exception:
            continue
        # Only trust a config whose mapped serials are all physically present, and
        # which actually covers the views we're browsing.
        if m and set(m.values()) <= connected:
            coverage = len(set(views) & set(m))
            if coverage and (best is None or coverage > best[0]):
                best = (coverage, path, m)
    if best is not None:
        print(f"Camera map auto-detected from {best[1]}: {best[2]}")
        return best[2]
    return {}


# Candidate configs per arm, tried in order; the one whose CAN channel is
# actually present on this machine wins. Keeps the script machine-agnostic.
LEFT_ROBOT_CONFIGS = [
    "robot_configs/yam/xdof_hq/left.yaml",
    "robot_configs/yam/left.yaml",
]
RIGHT_ROBOT_CONFIGS = [
    "robot_configs/yam/xdof_hq/right.yaml",
    "robot_configs/yam/right.yaml",
]


def available_can_channels() -> set[str]:
    """Network interfaces currently present (CAN buses show up under /sys/class/net)."""
    net = Path("/sys/class/net")
    if not net.is_dir():
        return set()
    return {p.name for p in net.iterdir()}


def _config_channel(config_path: str) -> str | None:
    with open(config_path) as f:
        cfg = _yaml.safe_load(f)
    return (cfg.get("motor_chain") or {}).get("channel")


def resolve_robot_config(candidates: list[str], side: str) -> str | None:
    """Pick the first candidate config whose CAN channel exists on this machine."""
    avail = available_can_channels()
    for path in candidates:
        if not Path(path).exists():
            continue
        channel = _config_channel(path)
        if channel in avail:
            print(f"  {side}: using {path} (channel '{channel}')")
            return path
    print(
        f"  {side}: no candidate config matches an available CAN channel "
        f"(have: {sorted(avail & _all_candidate_channels(candidates))}); skipping",
        file=sys.stderr,
    )
    return None


def _all_candidate_channels(candidates: list[str]) -> set[str]:
    chans = set()
    for path in candidates:
        if Path(path).exists():
            ch = _config_channel(path)
            if ch:
                chans.add(ch)
    return chans


def instantiate_robot(
    config_path: str,
    overrides: dict | None = None,
    *,
    allow_initial_joint_limit_violation: bool = False,
) -> Any:
    with open(config_path) as f:
        cfg = _yaml.safe_load(f)
    if overrides:
        cfg.update(overrides)
    if allow_initial_joint_limit_violation:
        cfg["allow_initial_joint_limit_violation"] = True
    return _resolve(cfg)


def zero_robots_safely(robots: dict[str, Any], duration_s: float) -> None:
    """Use RobotNode-style interpolation to zero both arms from their measured pose."""
    errors: queue.SimpleQueue[tuple[str, BaseException]] = queue.SimpleQueue()

    def _zero_one(side: str, robot: Any) -> None:
        original_limits = np.asarray(robot._joint_limits, dtype=np.float64).copy()
        current = np.asarray(robot.get_joint_pos(), dtype=np.float64).copy()
        target = current.copy()
        gripper_index = getattr(robot, "_gripper_index", None)
        arm_dof = int(gripper_index) if gripper_index is not None else len(current)
        target[:arm_dof] = 0.0

        # Clipping an already-outside initial pose to the ordinary limit would
        # cause a step on the first interpolation point. Expand only toward the
        # measured pose, then restore the real limits after moving toward zero.
        expanded_limits = original_limits.copy()
        expanded_limits[:, 0] = np.minimum(expanded_limits[:, 0], current[:arm_dof])
        expanded_limits[:, 1] = np.maximum(expanded_limits[:, 1], current[:arm_dof])
        robot._joint_limits = expanded_limits
        try:
            print(f"Zeroing {side} arm over {duration_s:.1f}s from {current[:arm_dof].round(3).tolist()} ...")
            robot.move_joints(target, time_interval_s=duration_s)
        except BaseException as exc:
            errors.put((side, exc))
        finally:
            robot._joint_limits = original_limits

    threads = [
        threading.Thread(target=_zero_one, args=(side, robot), name=f"zero-{side}") for side, robot in robots.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failures: list[str] = []
    while not errors.empty():
        side, exc = errors.get()
        failures.append(f"{side}: {exc}")

    for side, robot in robots.items():
        measured = np.asarray(robot.get_joint_pos(), dtype=np.float64)
        limits = np.asarray(robot._joint_limits, dtype=np.float64)
        arm_dof = limits.shape[0]
        outside = np.any(measured[:arm_dof] < limits[:, 0] - 0.1) or np.any(measured[:arm_dof] > limits[:, 1] + 0.1)
        if outside:
            failures.append(f"{side} remained outside limits: {measured[:arm_dof].round(3).tolist()}")
        else:
            print(f"{side.capitalize()} arm zeroed; measured={measured[:arm_dof].round(3).tolist()}")

    if failures:
        for robot in robots.values():
            try:
                robot.close()
            except Exception:
                pass
        raise RuntimeError("Robot startup zeroing failed: " + "; ".join(failures))


def move_robot_to_state(state: np.ndarray, left_robot: Any, right_robot: Any, duration_s: float) -> None:
    """Directly command both arms to the selected episode-frame joint state.

    state layout: [left_joint(6), left_gripper(1), right_joint(6), right_gripper(1)]
    """
    left_target = state[:7].copy()
    right_target = state[7:14].copy()
    print(f"  Target: left={left_target.round(3).tolist()}, right={right_target.round(3).tolist()}")
    left_robot.move_joints(left_target, time_interval_s=duration_s)
    right_robot.move_joints(right_target, time_interval_s=duration_s)
    print("  Done")


def prefetch_adjacent(
    s3_prefix: str, views: list[str], current_idx: int, total_episodes: int, chunks_size: int, cache_dir: Path
) -> None:
    """Pre-cache frames for the next and previous episodes."""
    for offset in [1, -1, 2, -2]:
        idx = current_idx + offset
        if 0 <= idx < total_episodes:
            for view in views:
                cache_file = cache_dir / f"ep{idx:06d}_{view}.jpg"
                if not cache_file.exists():
                    download_first_frame(s3_prefix, view, idx, chunks_size, cache_dir)


DEFAULT_YAM_URDF = "dependencies/i2rt/i2rt/robot_models/arm/yam/yam.urdf"


def load_yam_viser_layout(config_path: str | None, urdf_override: str | None) -> dict[str, dict]:
    layout: dict[str, dict] = {
        "left": {
            "path": urdf_override or DEFAULT_YAM_URDF,
            "flip_joints": True,
            "extrinsic": {"position": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
        },
        "right": {
            "path": urdf_override or DEFAULT_YAM_URDF,
            "flip_joints": True,
            "extrinsic": {"position": [0.0, -0.61, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
        },
    }
    if config_path is None:
        return layout
    with open(config_path) as stream:
        config = _yaml.safe_load(stream) or {}
    monitor = next(
        (node for node in config.get("nodes", []) if node.get("type") == "ViserMonitorNode"),
        None,
    )
    if monitor is None:
        return layout
    for name, spec in (monitor.get("urdfs") or {}).items():
        arm = "left" if "left" in name else "right" if "right" in name else None
        if arm is None:
            continue
        layout[arm]["path"] = urdf_override or spec.get("path", layout[arm]["path"])
        layout[arm]["flip_joints"] = bool(spec.get("flip_joints", True))
        if spec.get("extrinsic") is not None:
            layout[arm]["extrinsic"] = spec["extrinsic"]
    return layout


def make_episode_panel(
    frames: dict[str, np.ndarray | None],
    episode_idx: int,
    total_episodes: int,
    frame_idx: int = 0,
    frame_count: int = 1,
    fps: float = 30.0,
) -> np.ndarray:
    """Arrange one or more view frames into a labeled panel."""
    valid_frames = {k: v for k, v in frames.items() if v is not None}

    if not valid_frames:
        placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No data", (40, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        valid_frames = {"?": placeholder}

    labeled = []
    for view_name, frame in valid_frames.items():
        f = frame.copy()
        cv2.putText(f, view_name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        labeled.append(f)

    # Stack views horizontally
    max_h = max(f.shape[0] for f in labeled)
    padded = []
    for f in labeled:
        if f.shape[0] < max_h:
            pad = np.zeros((max_h - f.shape[0], f.shape[1], 3), dtype=np.uint8)
            f = np.vstack([f, pad])
        padded.append(f)

    panel = np.hstack(padded)

    # Add episode label bar at top
    bar_h = 28
    bar = np.zeros((bar_h, panel.shape[1], 3), dtype=np.uint8)
    label = (
        f"Episode {episode_idx}/{total_episodes - 1}  "
        f"Frame {frame_idx}/{max(0, frame_count - 1)} ({frame_idx / fps:.2f}s)  "
        "[A/D: episode  J/K: frame  G: goto  M: move  Q: quit]"
    )
    cv2.putText(bar, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([bar, panel])


def show_loading(window_name: str, message: str) -> None:
    """Paint a placeholder and pump the GUI event loop so the window isn't left
    unpainted (which renders as garbage/copied screen content) while we block on S3."""
    splash = np.zeros((252, 640, 3), dtype=np.uint8)
    cv2.putText(splash, message, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(window_name, splash)
    cv2.waitKey(1)


def center_fov_crop(frame: np.ndarray, frac: float) -> np.ndarray:
    """Center-crop to ``frac`` of each dimension to simulate a narrower FOV.

    ``frac`` is the fraction of width/height kept (``(0, 1]``); smaller = tighter
    FOV / more zoom. ``frac >= 1.0`` is a no-op. Mirrors
    ``CameraNode._center_fov_crop`` so the preview matches the deployment path.
    """
    if frac >= 1.0:
        return frame
    h, w = frame.shape[:2]
    ch = max(1, round(h * frac))
    cw = max(1, round(w * frac))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return frame[y0 : y0 + ch, x0 : x0 + cw]


def center_crop_square(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0 : y0 + side, x0 : x0 + side]


def make_camera_panel(
    frame: np.ndarray, target_height: int, label: str = "Live Camera", fov_crop: float = 1.0
) -> np.ndarray:
    """Optionally narrow FOV, then center-crop to square and resize to panel height."""
    cropped = center_crop_square(center_fov_crop(frame, fov_crop))
    resized = cv2.resize(cropped, (target_height, target_height))

    bar_h = 28
    bar = np.zeros((bar_h, resized.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([bar, resized])


COMPARISON_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>YAM Dataset Comparison</title>
<style>
*{box-sizing:border-box}body{height:100vh;overflow:hidden;margin:0;background:#101216;color:#d7dae0;font:13px system-ui;display:flex;flex-direction:column}
header{height:46px;flex:none;display:flex;gap:8px;align-items:center;padding:6px 10px;background:#151820;border-bottom:1px solid #303640}
header>*{flex:none}.camera-toolbar>*{flex:none}
button,input{background:#1d2129;color:#d7dae0;border:1px solid #3a404d;border-radius:6px;padding:4px 8px}input[type=range]{accent-color:#5b9dff;padding:0}
#episode,#frame{width:150px}#fov{width:110px}.muted{color:#929bad;font-size:11px}.workspace{flex:1;min-height:0;display:grid;grid-template-columns:minmax(520px,3fr) minmax(380px,2fr)}
#camera-side{min-width:0;min-height:0;padding:6px;border-right:1px solid #303640;display:flex;flex-direction:column}.camera-toolbar{height:28px;display:flex;gap:8px;align-items:center}
#rows{flex:1;min-height:0;display:grid;grid-template-rows:repeat(3,minmax(0,1fr));gap:5px}.compare{min-height:0;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:18px minmax(0,1fr);gap:3px 6px}
.compare h2{grid-column:1/-1;margin:0;font-size:12px}.pane{min-height:0;position:relative;background:#000;border:1px solid #292d36;border-radius:6px;overflow:hidden}.pane img{width:100%;height:100%;object-fit:contain;display:block}
.tag{position:absolute;top:4px;left:4px;z-index:3;padding:1px 5px;background:#000b;border-radius:4px;font-size:11px}.compare.overlay{grid-template-columns:1fr}.compare.overlay .pane{grid-area:2/1}.compare.overlay .dataset{opacity:var(--opacity,.5);z-index:2;pointer-events:none}
#viser-side{min-width:0;min-height:0;display:flex;flex-direction:column}.viser-title{height:28px;padding:6px 9px;background:#151820}.viser-title .blue{color:#4f8fff}.viser-title .orange{color:#ff6b35}iframe{flex:1;width:100%;border:0;background:#111}
</style></head><body>
<header><strong>Live ↔ LeRobot + URDF</strong><button id="prev-ep">◀ episode</button><input id="episode" type="range" min="0" step="1"><span id="episode-label"></span><button id="next-ep">episode ▶</button><button id="prev-frame">◀ frame</button><input id="frame" type="range" min="0" step="1"><span id="frame-label"></span><button id="next-frame">frame ▶</button><button id="move" style="border-color:#c44;color:#ff8b8b">Move robot</button><span id="status" class="muted"></span></header>
<div class="workspace"><section id="camera-side"><div class="camera-toolbar"><label><input id="overlay" type="checkbox"> overlay cameras</label><span class="muted">dataset opacity</span><input id="opacity" type="range" min="0" max="1" step=".01"><span id="opacity-label"></span><span class="muted">top live FOV</span><input id="fov" type="range" min=".5" max="1" step=".01"><span id="fov-label"></span></div><main id="rows"></main></section><section id="viser-side"><div class="viser-title">URDF overlay — <span class="blue">blue dataset</span> / <span class="orange">orange live</span></div><iframe id="viser" src=""></iframe></section></div>
<script>
const byId=id=>document.getElementById(id);let state=null,selectTimer=null,fovTimer=null;
function setOpacity(v){v=Math.max(0,Math.min(1,Number(v)));byId("opacity").value=v;byId("opacity-label").textContent=v.toFixed(2);document.documentElement.style.setProperty("--opacity",v)}
function build(){byId("rows").innerHTML=state.views.map(view=>"<section class='compare' id='compare-"+view+"'><h2>"+view+"</h2><div class='pane live'><span class='tag'>live</span><img src='/live/"+view+".mjpg'></div><div class='pane dataset'><span class='tag'>dataset</span><img id='dataset-"+view+"'></div></section>").join("");}
function render(){byId("episode").max=state.episode_max;byId("episode").value=state.episode;byId("episode-label").textContent=state.episode+" / "+state.episode_max;byId("frame").max=Math.max(0,state.frame_count-1);byId("frame").value=state.frame;byId("frame-label").textContent=state.frame+" / "+Math.max(0,state.frame_count-1)+" ("+state.time_s.toFixed(2)+"s)";const prep=state.camera_preprocess.top||state.camera_preprocess[state.views[0]]||{};if(document.activeElement!==byId("fov")){byId("fov").value=prep.fov_crop||1;byId("fov-label").textContent=Number(prep.fov_crop||1).toFixed(2)}byId("status").textContent=(state.robot_ready?"robot ready":"viewer only")+" · dataset joints "+(state.dataset_flip_joint_order?"flipped":"native")+" · video "+(prep.resize_mode||"native")+" · top fov "+Number(prep.fov_crop||1).toFixed(2);for(const view of state.views)byId("dataset-"+view).src="/dataset/"+view+".jpg?r="+state.revision;}
async function refresh(){state=await(await fetch("/api/state")).json();render()}
async function select(episode,frame){await fetch("/api/select?episode="+episode+"&frame="+frame,{method:"POST"});setTimeout(refresh,80)}
function schedule(){clearTimeout(selectTimer);selectTimer=setTimeout(()=>select(Number(byId("episode").value),Number(byId("frame").value)),80)}
(async()=>{state=await(await fetch("/api/state")).json();build();setOpacity(state.opacity);byId("viser").src=location.protocol+"//"+location.hostname+":"+state.viser_port;render();setInterval(refresh,1000)})();
byId("episode").oninput=e=>{byId("episode-label").textContent=e.target.value+" / "+state.episode_max};byId("episode").onchange=()=>select(Number(byId("episode").value),0);byId("frame").oninput=e=>{byId("frame-label").textContent=e.target.value+" / "+Math.max(0,state.frame_count-1);schedule()};
byId("prev-ep").onclick=()=>select(Math.max(0,state.episode-1),0);byId("next-ep").onclick=()=>select(Math.min(state.episode_max,state.episode+1),0);byId("prev-frame").onclick=()=>select(state.episode,Math.max(0,state.frame-1));byId("next-frame").onclick=()=>select(state.episode,Math.min(state.frame_count-1,state.frame+1));
byId("move").onclick=()=>fetch("/api/move",{method:"POST"});byId("opacity").oninput=e=>setOpacity(e.target.value);byId("overlay").onchange=e=>document.querySelectorAll(".compare").forEach(row=>row.classList.toggle("overlay",e.target.checked));
byId("fov").oninput=e=>{const v=Number(e.target.value);byId("fov-label").textContent=v.toFixed(2);clearTimeout(fovTimer);fovTimer=setTimeout(()=>fetch("/api/fov?value="+v,{method:"POST"}),40)};
</script></body></html>"""


def make_comparison_app(
    current: dict[str, Any],
    request_queue: queue.SimpleQueue[tuple[str, int, int]],
    live_frames: dict[str, np.ndarray],
    views: list[str],
    episode_ids: list[int],
    fps: float,
    viser_port: int,
    opacity: float,
) -> Any:
    from fastapi import FastAPI, HTTPException  # noqa: PLC0415
    from fastapi.responses import HTMLResponse, Response, StreamingResponse  # noqa: PLC0415

    app = FastAPI(title="YAM dataset comparison")

    async def index() -> HTMLResponse:
        return HTMLResponse(COMPARISON_HTML)

    async def api_state() -> dict:
        return {
            "views": views,
            "episode": int(current["episode_idx"]),
            "episode_max": max(episode_ids),
            "frame": int(current["frame_idx"]),
            "frame_count": int(current["frame_count"]),
            "time_s": int(current["frame_idx"]) / fps,
            "revision": int(current.get("revision", 0)),
            "robot_ready": bool(current.get("robot_ready", False)),
            "dataset_flip_joint_order": bool(current.get("dataset_flip_joint_order", False)),
            "camera_preprocess": current.get("camera_preprocess", {}),
            "robot_config_overrides": current.get("robot_config_overrides", {}),
            "viser_port": viser_port,
            "opacity": opacity,
        }

    async def api_select(episode: int, frame: int = 0) -> dict:
        if episode not in episode_ids:
            raise HTTPException(404, f"Episode {episode} not found")
        request_queue.put(("select", episode, frame))
        return {"accepted": True}

    async def api_move() -> dict:
        request_queue.put(("move", int(current["episode_idx"]), int(current["frame_idx"])))
        return {"accepted": True}

    async def api_fov(value: float) -> dict:
        if not 0.5 <= value <= 1.0:
            raise HTTPException(400, "FOV crop must be in [0.5, 1.0]")
        if "top" not in current["camera_preprocess"]:
            raise HTTPException(404, "Top camera is not enabled")
        current["camera_preprocess"]["top"]["fov_crop"] = float(value)
        return {"accepted": True, "view": "top", "fov_crop": float(value)}

    async def dataset_jpeg(view: str) -> Response:
        if view not in views:
            raise HTTPException(404, view)
        frame = current["frames"].get(view)
        if frame is None:
            raise HTTPException(503, f"Dataset frame unavailable: {view}")
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise HTTPException(500, "JPEG encoding failed")
        return Response(encoded.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def live_mjpeg(view: str) -> StreamingResponse:
        if view not in views:
            raise HTTPException(404, view)

        async def frames():
            while True:
                frame = live_frames.get(view)
                if frame is None:
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    payload = encoded.tobytes()
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(payload)).encode()
                        + b"\r\n\r\n"
                        + payload
                        + b"\r\n"
                    )
                await asyncio.sleep(0.05)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    app.add_api_route("/", index, response_class=HTMLResponse)
    app.add_api_route("/api/state", api_state)
    app.add_api_route("/api/select", api_select, methods=["POST"])
    app.add_api_route("/api/move", api_move, methods=["POST"])
    app.add_api_route("/api/fov", api_fov, methods=["POST"])
    app.add_api_route("/dataset/{view}.jpg", dataset_jpeg)
    app.add_api_route("/live/{view}.mjpg", live_mjpeg)
    return app


def main() -> None:
    args = parse_args()
    shutdown_requested = threading.Event()

    def _request_shutdown(_signum: int, _frame: Any) -> None:
        shutdown_requested.set()

    for shutdown_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(shutdown_signal, _request_shutdown)

    local_root = Path(args.dataset_path).expanduser().resolve() if args.dataset_path else None
    views = args.camera_views

    if local_root is not None:
        info = json.loads((local_root / "meta" / "info.json").read_text())
        local_episodes = load_local_episode_metadata(local_root)
        episode_ids = sorted(local_episodes)
        video_keys = resolve_local_video_keys(info, views)
        dataset_label = str(local_root)
    else:
        s3_prefix = args.s3_path.rstrip("/")
        info = load_dataset_info(s3_prefix)
        episode_ids = list(range(int(info["total_episodes"])))
        local_episodes = {}
        video_keys = {}
        dataset_label = f"s3://{s3_prefix}"
    if not episode_ids:
        raise ValueError(f"Dataset has no episodes: {dataset_label}")

    total_episodes = len(episode_ids)
    chunks_size = int(info["chunks_size"])
    fps = float(info.get("fps", 30))
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp(prefix="ep_compare_"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {dataset_label} ({total_episodes} episodes, views: {views})")

    import trimesh  # noqa: PLC0415
    import viser  # noqa: PLC0415
    import viser.extras  # noqa: PLC0415
    import viser.transforms as vtf  # noqa: PLC0415
    import yourdfpy  # noqa: PLC0415

    server = viser.ViserServer(host=args.host, port=args.viser_port, label="YAM Episode Comparison")
    server.scene.set_up_direction("+z")

    @server.on_client_connect
    def _set_initial_view(client: Any) -> None:
        client.camera.position = np.array([-0.7, -0.31, 0.3], dtype=np.float32)
        client.camera.look_at = np.array([0.45, -0.31, 0.3], dtype=np.float32)

    layout = load_yam_viser_layout(args.camera_config, args.urdf_path)
    dataset_flip_joint_order = load_policy_flip_joint_order(args.camera_config)
    camera_preprocess = load_camera_preprocess_specs(args.camera_config, views)
    if args.fov_crop is not None and "top" in camera_preprocess:
        camera_preprocess["top"]["fov_crop"] = args.fov_crop
    station_robot_specs = load_robot_node_specs(args.camera_config)
    print(f"Dataset joint-order flip: {dataset_flip_joint_order}")
    print(f"Live camera preprocessing: {camera_preprocess}")
    print(f"Station robot specs: {station_robot_specs}")
    from robots_realtime.runtime.viser_monitor_node import GRIPPER_PRESETS  # noqa: PLC0415

    preset = GRIPPER_PRESETS["linear_4310"]
    raw_urdfs: dict[str, dict[str, Any]] = {"dataset": {}, "live": {}}
    grippers: dict[str, dict[str, Any]] = {"dataset": {}, "live": {}}

    def _load_gripper_mesh(path: str) -> Any:
        mesh_path = Path(path).expanduser().resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Linear gripper mesh not found: {mesh_path}")
        return trimesh.load(mesh_path, force="mesh")

    def _add_linear_gripper(source: str, arm: str, arm_root: str, urdf: Any) -> None:
        gripper_root_path = f"{arm_root}/linear_4310"
        root_frame = server.scene.add_frame(gripper_root_path, show_axes=False)
        rgb = tuple(round(channel * 255) for channel in colors[source])
        shell = _load_gripper_mesh(preset["shell_stl"])
        shell_handle = server.scene.add_mesh_simple(
            f"{gripper_root_path}/shell",
            shell.vertices,
            shell.faces,
            color=rgb,
            opacity=opacities[source],
            position=np.asarray(preset["shell_pos"]),
            wxyz=np.asarray(preset["shell_quat_wxyz"]),
        )
        tip_slides: dict[str, Any] = {}
        tip_meshes: list[Any] = [shell_handle]
        for side in ("left", "right"):
            body_path = f"{gripper_root_path}/tip_{side}"
            server.scene.add_frame(
                body_path,
                show_axes=False,
                position=np.asarray(preset[f"tip_{side}_body_pos"]),
                wxyz=np.asarray(preset[f"tip_{side}_body_quat_wxyz"]),
            )
            slide_path = f"{body_path}/slide"
            slide = server.scene.add_frame(slide_path, show_axes=False)
            mesh = _load_gripper_mesh(preset[f"tip_{side}_stl"])
            mesh_handle = server.scene.add_mesh_simple(
                f"{slide_path}/mesh",
                mesh.vertices,
                mesh.faces,
                color=rgb,
                opacity=opacities[source],
                position=np.asarray(preset[f"tip_{side}_mesh_pos"]),
                wxyz=np.asarray(preset[f"tip_{side}_mesh_quat_wxyz"]),
            )
            tip_slides[side] = slide
            tip_meshes.append(mesh_handle)
        body_rotation = vtf.SO3(np.asarray(preset["body_offset_quat_wxyz"], dtype=np.float64)).as_matrix()
        body_offset = np.eye(4, dtype=np.float64)
        body_offset[:3, :3] = body_rotation
        body_offset[:3, 3] = np.asarray(preset["body_offset_pos"], dtype=np.float64)
        grippers[source][arm] = {
            "urdf": urdf,
            "root": root_frame,
            "slides": tip_slides,
            "body_offset": body_offset,
            "meshes": tip_meshes,
        }

    def _update_linear_gripper(source: str, arm: str, cfg: np.ndarray, gripper_pos: float) -> None:
        gripper = grippers[source][arm]
        gripper["urdf"].update_cfg(cfg)
        transform = gripper["urdf"].get_transform("link_6") @ gripper["body_offset"]
        gripper["root"].position = transform[:3, 3].astype(np.float32)
        gripper["root"].wxyz = vtf.SO3.from_matrix(transform[:3, :3]).wxyz.astype(np.float32)
        opening = float(np.clip(gripper_pos, 0.0, 1.0)) * float(preset["slide_range_m"])
        offset = np.asarray(preset["slide_axis"], dtype=np.float32) * opening
        for slide in gripper["slides"].values():
            slide.position = offset

    dataset_urdfs: dict[str, Any] = {}
    live_urdfs: dict[str, Any] = {}
    colors = {"dataset": (0.15, 0.45, 1.0), "live": (1.0, 0.35, 0.08)}
    opacities = {"dataset": 0.45, "live": 0.55}
    for source, handles in (("dataset", dataset_urdfs), ("live", live_urdfs)):
        for arm in ("left", "right"):
            spec = layout[arm]
            urdf_path = Path(spec["path"]).expanduser().resolve()
            if not urdf_path.is_file():
                raise FileNotFoundError(f"YAM URDF not found: {urdf_path}")
            mesh_dir_path = urdf_path.parent / "assets"
            mesh_dir = str(mesh_dir_path) if mesh_dir_path.is_dir() else None
            urdf = (
                yourdfpy.URDF.load(str(urdf_path), mesh_dir=mesh_dir)
                if mesh_dir is not None
                else yourdfpy.URDF.load(str(urdf_path))
            )
            root = f"/{source}/{arm}"
            extrinsic = spec["extrinsic"]
            server.scene.add_frame(
                root,
                show_axes=False,
                position=np.asarray(extrinsic.get("position", [0.0, 0.0, 0.0])),
                wxyz=np.asarray(extrinsic.get("rotation", [1.0, 0.0, 0.0, 0.0])),
            )
            handle = viser.extras.ViserUrdf(server, urdf, root_node_name=root, mesh_color_override=colors[source])
            for mesh in handle._meshes:
                mesh.opacity = opacities[source]
            handles[arm] = handle
            raw_urdfs[source][arm] = urdf
            _add_linear_gripper(source, arm, root, urdf)

    current: dict[str, Any] = {
        "position": 0,
        "episode_idx": episode_ids[0],
        "frame_idx": 0,
        "frame_count": 1,
        "states": None,
        "state": None,
        "frames": {},
        "revision": 0,
        "robot_ready": False,
        "dataset_flip_joint_order": dataset_flip_joint_order,
        "camera_preprocess": camera_preprocess,
        "robot_config_overrides": {arm: spec["overrides"] for arm, spec in station_robot_specs.items()},
    }

    def _load_episode(position: int) -> None:
        position = max(0, min(total_episodes - 1, position))
        episode_idx = episode_ids[position]
        current["position"] = position
        current["episode_idx"] = episode_idx
        current["frame_idx"] = 0
        if local_root is not None:
            episode = local_episodes[episode_idx]
            states = load_local_episode_states(local_root, info, episode)
            current["states"] = states
            current["frame_count"] = min(int(episode.get("length", len(states))), len(states))
        else:
            first_state = get_episode_first_state(s3_prefix, episode_idx, chunks_size, cache_dir)
            current["states"] = None if first_state is None else first_state[None, :]
            current["frame_count"] = 1
        _load_frame(0)

    def _load_frame(frame_idx: int) -> None:
        frame_idx = max(0, min(int(current["frame_count"]) - 1, frame_idx))
        episode_idx = int(current["episode_idx"])
        current["frame_idx"] = frame_idx
        if local_root is not None:
            episode = local_episodes[episode_idx]
            current["frames"] = {
                view: read_local_video_frame(local_root, info, episode, video_keys[view], frame_idx) for view in views
            }
        else:
            current["frames"] = {
                view: download_first_frame(s3_prefix, view, episode_idx, chunks_size, cache_dir) for view in views
            }
        states = current["states"]
        if states is None:
            current["state"] = None
        else:
            dataset_state = np.asarray(states[frame_idx], dtype=np.float64)
            current["state"] = (
                flip_bimanual_yam_joint_order(dataset_state) if dataset_flip_joint_order else dataset_state
            )

    _load_episode(0)

    request_queue: queue.SimpleQueue[tuple[str, int, int]] = queue.SimpleQueue()
    live_frames: dict[str, np.ndarray] = {}
    with server.gui.add_folder("Dataset cameras"):
        dataset_image_handles = {
            view: server.gui.add_image(
                cv2.cvtColor(current["frames"][view], cv2.COLOR_BGR2RGB)
                if current["frames"].get(view) is not None
                else np.zeros((224, 224, 3), dtype=np.uint8),
                label=f"Dataset {view}",
                format="jpeg",
                jpeg_quality=90,
            )
            for view in views
        }
    with server.gui.add_folder("Live cameras"):
        live_image_handles = {
            view: server.gui.add_image(
                np.zeros((224, 224, 3), dtype=np.uint8),
                label=f"Live {view}",
                format="jpeg",
                jpeg_quality=85,
            )
            for view in views
        }
    with server.gui.add_folder("Comparison controls"):
        status_handle = server.gui.add_markdown("")
        episode_slider = server.gui.add_slider(
            "Episode",
            min=min(episode_ids),
            max=max(episode_ids),
            step=1,
            initial_value=episode_ids[0],
        )
        frame_slider_holder: list[Any] = [None]
        previous_episode = server.gui.add_button("Previous episode")
        next_episode = server.gui.add_button("Next episode")
        previous_frame = server.gui.add_button("Previous frame")
        next_frame = server.gui.add_button("Next frame")
        move_button = server.gui.add_button(
            "Move robot to displayed frame",
            color="red",
            hint="Commands both physical YAM arms to the selected dataset state.",
        )
        server.gui.add_markdown(
            "**URDF overlay:** dataset pose is blue; live measured pose is orange. "
            "The red button is the only viewer control that commands arm motion."
        )

    @episode_slider.on_update
    def _episode_changed(_event: Any) -> None:
        if getattr(_event, "client_id", None) is None:
            return
        request_queue.put(("episode_id", int(episode_slider.value), 0))

    @previous_episode.on_click
    def _previous_episode(_event: Any) -> None:
        if getattr(_event, "client_id", None) is None:
            return
        request_queue.put(("episode_position", int(current["position"]) - 1, 0))

    @next_episode.on_click
    def _next_episode(_event: Any) -> None:
        if getattr(_event, "client_id", None) is None:
            return
        request_queue.put(("episode_position", int(current["position"]) + 1, 0))

    @previous_frame.on_click
    def _previous_frame(_event: Any) -> None:
        if getattr(_event, "client_id", None) is None:
            return
        request_queue.put(("frame", int(current["episode_idx"]), int(current["frame_idx"]) - 1))

    @next_frame.on_click
    def _next_frame(_event: Any) -> None:
        if getattr(_event, "client_id", None) is None:
            return
        request_queue.put(("frame", int(current["episode_idx"]), int(current["frame_idx"]) + 1))

    @move_button.on_click
    def _move_to_frame(_event: Any) -> None:
        if getattr(_event, "client_id", None) is None:
            return
        request_queue.put(("move", int(current["episode_idx"]), int(current["frame_idx"])))

    def _replace_frame_slider() -> None:
        if frame_slider_holder[0] is not None:
            frame_slider_holder[0].remove()
        episode_idx = int(current["episode_idx"])
        handle = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(1, int(current["frame_count"]) - 1),
            step=1,
            initial_value=int(current["frame_idx"]),
        )

        @handle.on_update
        def _frame_changed(_event: Any) -> None:
            if getattr(_event, "client_id", None) is None:
                return
            request_queue.put(("frame", episode_idx, int(handle.value)))

        frame_slider_holder[0] = handle

    def _update_dataset_display() -> None:
        for view, frame in current["frames"].items():
            if frame is not None:
                dataset_image_handles[view].image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        state = current["state"]
        if state is not None and len(state) >= 14:
            for arm, joints, gripper_pos in (
                ("left", state[:6], state[6]),
                ("right", state[7:13], state[13]),
            ):
                cfg = np.flip(joints) if layout[arm]["flip_joints"] else np.asarray(joints)
                dataset_urdfs[arm].update_cfg(cfg)
                _update_linear_gripper("dataset", arm, cfg, float(gripper_pos))
        episode_idx = int(current["episode_idx"])
        frame_idx = int(current["frame_idx"])
        frame_count = int(current["frame_count"])
        position = int(current["position"])
        status_handle.content = (
            f"**Episode {episode_idx}** ({position + 1}/{total_episodes})  "
            f"**Frame {frame_idx}**/{frame_count - 1}  "
            f"**Time:** {frame_idx / fps:.2f} s"
        )
        current["revision"] = int(current.get("revision", 0)) + 1

    _replace_frame_slider()
    _update_dataset_display()
    import uvicorn  # noqa: PLC0415

    web_app = make_comparison_app(
        current,
        request_queue,
        live_frames,
        views,
        episode_ids,
        fps,
        args.viser_port,
        args.opacity,
    )
    web_server = uvicorn.Server(uvicorn.Config(web_app, host=args.host, port=args.port, log_level="warning"))
    web_thread = threading.Thread(target=web_server.run, name="comparison-web", daemon=True)
    web_thread.start()
    print(f"Comparison webpage: http://localhost:{args.port}")
    print(f"Embedded Viser: http://localhost:{args.viser_port}")
    print(f"Remote viewer: http://<us05-hostname-or-ip>:{args.port}")

    rs_cams: dict[str, Any] = {}
    if not args.no_camera:
        from robots_realtime.sensors.cameras.realsense_camera import RealSenseCamera  # noqa: PLC0415

        cam_map = (
            {views[0]: args.camera_serial} if args.camera_serial else resolve_camera_map(args.camera_config, views)
        )
        for view in views:
            serial = cam_map.get(view)
            if serial is None:
                print(f"Warning: no live camera serial for {view}", file=sys.stderr)
                continue
            try:
                rs_cams[view] = RealSenseCamera(device_id=serial)
                print(f"Live camera for {view!r} opened (serial {serial})")
            except Exception as exc:
                print(f"Warning: cannot open camera for {view!r}: {exc}", file=sys.stderr)

    left_robot = None
    right_robot = None
    if not args.no_robot:
        left_spec = station_robot_specs.get("left", {})
        right_spec = station_robot_specs.get("right", {})
        left_cfg = (
            args.left_robot_config or left_spec.get("config_path") or resolve_robot_config(LEFT_ROBOT_CONFIGS, "left")
        )
        right_cfg = (
            args.right_robot_config
            or right_spec.get("config_path")
            or resolve_robot_config(RIGHT_ROBOT_CONFIGS, "right")
        )
        if left_cfg is not None:
            left_overrides = left_spec.get("overrides", {})
            print(f"Initializing left arm from {left_cfg} with overrides {left_overrides} ...")
            left_robot = instantiate_robot(
                left_cfg,
                left_overrides,
                allow_initial_joint_limit_violation=True,
            )
        if right_cfg is not None:
            right_overrides = right_spec.get("overrides", {})
            print(f"Initializing right arm from {right_cfg} with overrides {right_overrides} ...")
            right_robot = instantiate_robot(
                right_cfg,
                right_overrides,
                allow_initial_joint_limit_violation=True,
            )
        robots_to_zero = {
            side: robot for side, robot in (("left", left_robot), ("right", right_robot)) if robot is not None
        }
        if robots_to_zero:
            zero_robots_safely(robots_to_zero, args.move_duration)
        ready = left_robot is not None and right_robot is not None
        current["robot_ready"] = ready
        print("Robot control ready" if ready else "Robot control unavailable")

    last_live_update = 0.0
    try:
        while not shutdown_requested.is_set():
            try:
                while True:
                    kind, first, second = request_queue.get_nowait()
                    if kind == "select":
                        nearest = min(episode_ids, key=lambda value: abs(value - first))
                        _load_episode(episode_ids.index(nearest))
                        _load_frame(second)
                        episode_slider.value = nearest
                        _replace_frame_slider()
                        if frame_slider_holder[0] is not None:
                            frame_slider_holder[0].value = int(current["frame_idx"])
                        _update_dataset_display()
                    elif kind == "episode_id":
                        nearest = min(episode_ids, key=lambda value: abs(value - first))
                        _load_episode(episode_ids.index(nearest))
                        episode_slider.value = nearest
                        _replace_frame_slider()
                        _update_dataset_display()
                    elif kind == "episode_position":
                        position = max(0, min(total_episodes - 1, first))
                        _load_episode(position)
                        episode_slider.value = int(current["episode_idx"])
                        _replace_frame_slider()
                        _update_dataset_display()
                    elif kind == "frame" and first == int(current["episode_idx"]):
                        _load_frame(second)
                        if frame_slider_holder[0] is not None:
                            frame_slider_holder[0].value = int(current["frame_idx"])
                        _update_dataset_display()
                    elif (
                        kind == "move" and first == int(current["episode_idx"]) and second == int(current["frame_idx"])
                    ):
                        if left_robot is None or right_robot is None:
                            print("Robot control disabled or initialization failed")
                        elif current["state"] is not None:
                            print(f"Moving to episode {first}, frame {second} over {args.move_duration}s")
                            move_robot_to_state(current["state"], left_robot, right_robot, args.move_duration)
            except queue.Empty:
                pass

            now = time.monotonic()
            if now - last_live_update >= 0.05:
                for arm, robot in (("left", left_robot), ("right", right_robot)):
                    if robot is not None:
                        robot_state = np.asarray(robot.get_joint_pos(), dtype=np.float64)
                        joints = robot_state[:6]
                        cfg = np.flip(joints) if layout[arm]["flip_joints"] else joints
                        live_urdfs[arm].update_cfg(cfg)
                        gripper_pos = float(robot_state[6]) if len(robot_state) > 6 else 0.0
                        _update_linear_gripper("live", arm, cfg, gripper_pos)
                for view, cam in rs_cams.items():
                    try:
                        camera_data = cam.read()
                        live_frame = preprocess_live_camera_frame(camera_data.images["rgb"], camera_preprocess[view])
                        live_image_handles[view].image = live_frame
                        live_frames[view] = live_frame
                    except Exception as exc:
                        print(f"Warning: live camera read failed for {view!r}: {exc}", file=sys.stderr)
                last_live_update = now
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping viewer")
    finally:
        robots_to_zero = {
            side: robot for side, robot in (("left", left_robot), ("right", right_robot)) if robot is not None
        }
        if robots_to_zero:
            current["robot_ready"] = False
            print("Zeroing both arms before viewer shutdown ...")
            try:
                zero_robots_safely(robots_to_zero, args.move_duration)
                print("Shutdown zeroing complete")
            except Exception as exc:
                print(f"Warning: shutdown zeroing failed: {exc}", file=sys.stderr)
        for robot in (left_robot, right_robot):
            if robot is not None and hasattr(robot, "close"):
                robot.close()
        for cam in rs_cams.values():
            if hasattr(cam, "stop"):
                cam.stop()
        web_server.should_exit = True
        web_thread.join(timeout=3.0)
        server.stop()


if __name__ == "__main__":
    main()
