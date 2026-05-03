"""Side-by-side viewer: first frame of S3 LeRobot episodes vs. live camera.

Usage:
    uv run python scripts/compare_episode_camera.py \
        --s3-path xdof-internal-research/repromo/hlm_tshirt_reward_select_lerobot_sarm_8stage \
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
import importlib
import json
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
        default="xdof-internal-research/repromo/hlm_tshirt_reward_select_lerobot_sarm_8stage",
        help="S3 bucket/prefix (no s3:// prefix)",
    )
    p.add_argument("--camera-serial", type=str, default=None, help="RealSense serial number (default: first device)")
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
        default="robot_configs/yam/left.yaml",
        help="YAML config for left arm hardware",
    )
    p.add_argument(
        "--right-robot-config",
        default="robot_configs/yam/right.yaml",
        help="YAML config for right arm hardware",
    )
    p.add_argument("--move-duration", type=float, default=2.0, help="Seconds to interpolate to target pose")
    p.add_argument("--no-robot", action="store_true", help="Disable robot control (browse-only)")
    return p.parse_args()


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


def center_crop_square(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0 : y0 + side, x0 : x0 + side]


def make_camera_panel(frame: np.ndarray, target_height: int) -> np.ndarray:
    """Center-crop to square, then resize to match episode panel height."""
    cropped = center_crop_square(frame)
    resized = cv2.resize(cropped, (target_height, target_height))

    bar_h = 28
    bar = np.zeros((bar_h, resized.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, "Live Camera", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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

    rs_cam = None
    if not args.no_camera:
        from robots_realtime.sensors.cameras.realsense_camera import RealSenseCamera

        try:
            rs_cam = RealSenseCamera(device_id=args.camera_serial)
            print(f"RealSense camera opened: {rs_cam.get_camera_info()}")
        except Exception as exc:
            print(f"Warning: Cannot open RealSense camera: {exc}", file=sys.stderr)

    left_robot = None
    right_robot = None
    if not args.no_robot:
        print(f"Initializing left arm from {args.left_robot_config} ...")
        left_robot = instantiate_robot(args.left_robot_config)
        print(f"Initializing right arm from {args.right_robot_config} ...")
        right_robot = instantiate_robot(args.right_robot_config)
        print("Robots ready")

    episode_idx = 0
    need_reload = True
    episode_frames: dict[str, np.ndarray | None] = {}
    episode_state: np.ndarray | None = None

    window_name = "Episode vs Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Downloading first episode frames...")

    while True:
        if need_reload:
            print(f"Loading episode {episode_idx}...", end=" ", flush=True)
            episode_frames = {}
            for view in views:
                episode_frames[view] = download_first_frame(s3_prefix, view, episode_idx, chunks_size, cache_dir)
            episode_state = get_episode_first_state(s3_prefix, episode_idx, chunks_size, cache_dir)
            print("done")
            need_reload = False
            prefetch_adjacent(s3_prefix, views, episode_idx, total_episodes, chunks_size, cache_dir)

        ep_panel = make_episode_panel(episode_frames, episode_idx, total_episodes)

        if rs_cam is not None:
            try:
                cam_data = rs_cam.read()
                cam_frame = cv2.cvtColor(cam_data.images["rgb"], cv2.COLOR_RGB2BGR)
                cam_panel = make_camera_panel(cam_frame, ep_panel.shape[0] - 28)
                display = np.hstack([ep_panel, cam_panel])
            except Exception:
                display = ep_panel
        else:
            display = ep_panel

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
    if rs_cam is not None:
        rs_cam.stop()
    cv2.destroyAllWindows()
    print(f"\nCached frames at: {cache_dir}")


if __name__ == "__main__":
    main()
