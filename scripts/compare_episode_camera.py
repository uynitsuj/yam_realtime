"""Side-by-side viewer: first frame of S3 LeRobot episodes vs. live camera.

Usage:
    uv run python scripts/compare_episode_camera.py \
        --s3-path my-bucket/path/to/lerobot_dataset \
        --camera-id 0 \
        --camera-views top left right

Keys:
    Right / D  → next episode
    Left  / A  → previous episode
    G          → jump to episode (enter number in terminal)
    M          → move robot to episode's first-frame joint state
    Q / Esc    → quit
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml as _yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare S3 episode first-frames with live camera")
    p.add_argument(
        "--s3-path",
        required=True,
        help="S3 bucket/prefix of the LeRobot dataset (no s3:// prefix)",
    )
    p.add_argument("--camera-serial", type=str, default=None, help="RealSense serial for a single live camera (overrides --camera-config mapping)")
    p.add_argument(
        "--camera-config",
        type=str,
        default=None,
        help="Session YAML with CameraNode entries to map dataset views -> RealSense serials "
        "(default: auto-detect from configs/ by matching connected serials)",
    )
    p.add_argument(
        "--camera-views",
        nargs="+",
        default=["top"],
        choices=["top", "left", "right"],
        help="Which dataset camera views to show (observation.images.{view})",
    )
    p.add_argument("--cache-dir", type=str, default=None, help="Directory to cache downloaded frames (default: tmpdir)")
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
        default=1.0,
        help="Artificially narrow the LIVE camera FOV by center-cropping this fraction of each "
        "axis BEFORE the square crop/resize (in (0, 1]; 1.0 = off, 0.85 = ~15%% tighter FOV). "
        "Mirrors CameraNode.publish_fov_crop so you can preview the deployment FOV against the "
        "dataset frame.",
    )
    args = p.parse_args()
    if not (0.0 < args.fov_crop <= 1.0):
        p.error(f"--fov-crop must be in (0, 1], got {args.fov_crop}")
    return args


def s3_cp(s3_uri: str, local_path: str) -> bool:
    result = subprocess.run(
        ["aws", "s3", "cp", s3_uri, local_path],
        capture_output=True,
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


def download_first_frame(s3_prefix: str, view: str, episode_idx: int, chunks_size: int, cache_dir: Path) -> np.ndarray | None:
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
        out = subprocess.run(["rs-enumerate-devices", "-s"], capture_output=True, text=True)
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


def instantiate_robot(config_path: str):
    with open(config_path) as f:
        cfg = _yaml.safe_load(f)
    cfg["use_gravity_comp"] = False
    cfg.pop("gravity_comp_factor", None)
    robot = _resolve(cfg)
    robot.move_joints(np.zeros(len(robot.motor_chain)), time_interval_s=2.0)
    return robot


def move_robot_to_state(state: np.ndarray, left_robot, right_robot, duration_s: float) -> None:
    """Directly command both arms to the episode's first-frame joint state.

    state layout: [left_joint(6), left_gripper(1), right_joint(6), right_gripper(1)]
    """
    left_target = state[:7].copy()
    right_target = state[7:14].copy()
    print(f"  Target: left={left_target.round(3).tolist()}, right={right_target.round(3).tolist()}")
    left_robot.move_joints(left_target, time_interval_s=duration_s)
    right_robot.move_joints(right_target, time_interval_s=duration_s)
    print("  Done")


def prefetch_adjacent(s3_prefix: str, views: list[str], current_idx: int, total_episodes: int, chunks_size: int, cache_dir: Path) -> None:
    """Pre-cache frames for the next and previous episodes."""
    for offset in [1, -1, 2, -2]:
        idx = current_idx + offset
        if 0 <= idx < total_episodes:
            for view in views:
                cache_file = cache_dir / f"ep{idx:06d}_{view}.jpg"
                if not cache_file.exists():
                    download_first_frame(s3_prefix, view, idx, chunks_size, cache_dir)


def make_episode_panel(frames: dict[str, np.ndarray | None], episode_idx: int, total_episodes: int) -> np.ndarray:
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
    label = f"Episode {episode_idx}/{total_episodes - 1}  [A/D: prev/next  G: goto  M: move robot  Q: quit]"
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


def main() -> None:
    args = parse_args()
    s3_prefix = args.s3_path.rstrip("/")

    print(f"Loading dataset info from s3://{s3_prefix}/meta/info.json ...")
    info = load_dataset_info(s3_prefix)
    total_episodes = info["total_episodes"]
    chunks_size = info["chunks_size"]
    views = args.camera_views
    print(f"Dataset: {total_episodes} episodes, views: {views}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp(prefix="ep_compare_"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Frame cache: {cache_dir}")

    # rs_cams maps dataset view -> live RealSenseCamera so each dataset frame can be
    # shown next to its matching hardware camera.
    rs_cams: dict[str, object] = {}
    if not args.no_camera:
        from robots_realtime.sensors.cameras.realsense_camera import RealSenseCamera

        if args.camera_serial:
            # Single explicit serial: attach it to the first requested view.
            cam_map = {views[0]: args.camera_serial}
        else:
            cam_map = resolve_camera_map(args.camera_config, views)

        if not cam_map:
            print(
                "Warning: no view->serial camera map found; pass --camera-config <session.yaml> "
                "or --camera-serial to see live hardware cameras.",
                file=sys.stderr,
            )
        for view in views:
            serial = cam_map.get(view)
            if serial is None:
                print(f"Warning: no live camera serial for view '{view}'", file=sys.stderr)
                continue
            try:
                rs_cams[view] = RealSenseCamera(device_id=serial)
                print(f"Live camera for '{view}' opened (serial {serial})")
            except Exception as exc:
                print(f"Warning: cannot open camera for '{view}' (serial {serial}): {exc}", file=sys.stderr)

    left_robot = None
    right_robot = None
    if not args.no_robot:
        print("Resolving robot configs from available CAN channels...")
        left_cfg = args.left_robot_config or resolve_robot_config(LEFT_ROBOT_CONFIGS, "left")
        right_cfg = args.right_robot_config or resolve_robot_config(RIGHT_ROBOT_CONFIGS, "right")
        if left_cfg is not None:
            print(f"Initializing left arm from {left_cfg} ...")
            left_robot = instantiate_robot(left_cfg)
        if right_cfg is not None:
            print(f"Initializing right arm from {right_cfg} ...")
            right_robot = instantiate_robot(right_cfg)
        if left_robot is not None or right_robot is not None:
            print("Robots ready")
        else:
            print("No arms available on this machine; continuing browse/camera-only.")

    episode_idx = 0
    need_reload = True
    episode_frames: dict[str, np.ndarray | None] = {}
    episode_state: np.ndarray | None = None
    cam_read_warned: set[str] = set()

    window_name = "Episode vs Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    show_loading(window_name, "Loading episodes from S3...")

    print("Downloading first episode frames...")

    while True:
        if need_reload:
            print(f"Loading episode {episode_idx}...", end=" ", flush=True)
            show_loading(window_name, f"Loading episode {episode_idx}...")
            episode_frames = {}
            for view in views:
                episode_frames[view] = download_first_frame(s3_prefix, view, episode_idx, chunks_size, cache_dir)
            episode_state = get_episode_first_state(s3_prefix, episode_idx, chunks_size, cache_dir)
            print("done")
            need_reload = False
            prefetch_adjacent(s3_prefix, views, episode_idx, total_episodes, chunks_size, cache_dir)

        ep_panel = make_episode_panel(episode_frames, episode_idx, total_episodes)

        panels = [ep_panel]
        for view, cam in rs_cams.items():
            try:
                cam_data = cam.read()
                cam_frame = cv2.cvtColor(cam_data.images["rgb"], cv2.COLOR_RGB2BGR)
                live_label = f"Live {view}" + (f" (fov {args.fov_crop:g})" if args.fov_crop < 1.0 else "")
                panels.append(
                    make_camera_panel(cam_frame, ep_panel.shape[0] - 28, live_label, args.fov_crop)
                )
            except Exception as exc:
                if view not in cam_read_warned:
                    print(f"Warning: live camera read failed for '{view}': {exc}", file=sys.stderr)
                    cam_read_warned.add(view)
        display = np.hstack(panels) if len(panels) > 1 else ep_panel

        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):  # q or Esc
            break
        elif key in (ord("d"), 83, 3):  # d, Right arrow
            if episode_idx < total_episodes - 1:
                episode_idx += 1
                need_reload = True
        elif key in (ord("a"), 81, 2):  # a, Left arrow
            if episode_idx > 0:
                episode_idx -= 1
                need_reload = True
        elif key == ord("g"):
            try:
                target = input(f"\nGo to episode [0-{total_episodes - 1}]: ").strip()
                target_idx = int(target)
                if 0 <= target_idx < total_episodes:
                    episode_idx = target_idx
                    need_reload = True
                else:
                    print(f"Out of range, must be 0-{total_episodes - 1}")
            except (ValueError, EOFError):
                print("Invalid input")
        elif key == ord("m"):
            if left_robot is None or right_robot is None:
                print("\nRobot control disabled (--no-robot or init failed)")
            elif episode_state is not None:
                print(f"\nMoving robot to episode {episode_idx} first-frame state ({args.move_duration}s)...")
                move_robot_to_state(episode_state, left_robot, right_robot, args.move_duration)
            else:
                print(f"\nNo state data for episode {episode_idx}")

    if left_robot is not None and hasattr(left_robot, "close"):
        left_robot.close()
    if right_robot is not None and hasattr(right_robot, "close"):
        right_robot.close()
    for cam in rs_cams.values():
        if hasattr(cam, "stop"):
            cam.stop()
    cv2.destroyAllWindows()
    print(f"\nCached frames at: {cache_dir}")


if __name__ == "__main__":
    main()
