"""Compose a multi-camera view from a robots_realtime episode recording.

Renders the top camera as the full frame, with left and right wrist cameras
inset as picture-in-picture overlays in the top-left and top-right corners.

Episode layout assumed (matches AsyncMp4Writer in robots_realtime/runtime/recording.py):
    camera_top-images-rgb.mp4
    camera_top-rgb-timestamp.npy
    camera_left-images-rgb.mp4
    camera_left-rgb-timestamp.npy
    camera_right-images-rgb.mp4
    camera_right-rgb-timestamp.npy

Usage:
    uv run python scripts/composite_episode_view_renderer.py /path/to/episode_dir
    uv run python scripts/composite_episode_view_renderer.py /path/to/episode_dir --output composite.mp4
    uv run python scripts/composite_episode_view_renderer.py /path/to/episode_dir --pip-scale 0.28 --no-play
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------


def _load_video(path: Path) -> np.ndarray:
    """Load an MP4 as (T, H, W, 3) uint8 using imageio/pyav."""
    import imageio.v3 as iio

    return iio.imread(str(path), plugin="pyav")


def _load_timestamps(episode_dir: Path, camera: str, topic: str = "rgb") -> np.ndarray:
    """Load <camera>-<topic>-timestamp.npy.

    AsyncMp4Writer writes one sidecar per topic; for our camera nodes the
    topic is always "rgb".
    """
    ts_path = episode_dir / f"{camera}-{topic}-timestamp.npy"
    if not ts_path.exists():
        raise FileNotFoundError(f"Timestamp sidecar not found: {ts_path}")
    return np.load(str(ts_path))


def _sample_frame(
    frames: np.ndarray,
    ts: np.ndarray,
    query_t: float,
) -> np.ndarray:
    """Return the nearest frame (sample-and-hold) for query_t."""
    idx = int(np.searchsorted(ts, query_t, side="right")) - 1
    idx = max(0, min(idx, len(frames) - 1))
    return frames[idx]


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


def _resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a (H, W, 3) frame to (height, width, 3) using PIL."""
    from PIL import Image

    img = Image.fromarray(frame)
    img = img.resize((width, height), Image.BILINEAR)
    return np.asarray(img)


def _composite(
    main: np.ndarray,
    left_pip: np.ndarray | None,
    right_pip: np.ndarray | None,
    pip_w: int,
    pip_h: int,
    margin: int = 8,
) -> np.ndarray:
    """Overlay left/right PiP frames on a copy of main.

    Left PiP goes in the top-left corner, right PiP in the top-right.
    A thin dark border is drawn around each inset for clarity.
    """
    out = main.copy()
    fh, fw = main.shape[:2]

    border = 2

    def _paste(inset: np.ndarray, x: int, y: int) -> None:
        bx0 = max(0, x - border)
        by0 = max(0, y - border)
        bx1 = min(fw, x + pip_w + border)
        by1 = min(fh, y + pip_h + border)
        out[by0:by1, bx0:bx1] = 0
        out[y : y + pip_h, x : x + pip_w] = inset

    if left_pip is not None:
        resized = _resize_frame(left_pip, pip_w, pip_h)
        _paste(resized, margin, margin)

    if right_pip is not None:
        resized = _resize_frame(right_pip, pip_w, pip_h)
        _paste(resized, fw - margin - pip_w, margin)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_composite(
    episode_dir: Path,
    output: Path,
    pip_scale: float = 0.25,
    crf: int = 30,
    preset: str = "slow",
    scale: float = 1.0,
) -> None:
    """Load cameras, composite every top-camera frame, write output MP4."""
    top_mp4 = episode_dir / "camera_top-images-rgb.mp4"
    left_mp4 = episode_dir / "camera_left-images-rgb.mp4"
    right_mp4 = episode_dir / "camera_right-images-rgb.mp4"

    if not top_mp4.exists():
        raise FileNotFoundError(f"Top camera video not found: {top_mp4}")

    print("Loading top camera ...")
    top_frames = _load_video(top_mp4)
    top_ts = _load_timestamps(episode_dir, "camera_top")
    T, H, W, _ = top_frames.shape
    print(f"  {W}x{H}, {T} frames")

    pip_w = int(W * pip_scale)
    pip_h = pip_w  # square inset; will be resized to match source AR below

    left_frames: np.ndarray | None = None
    left_ts: np.ndarray | None = None
    right_frames: np.ndarray | None = None
    right_ts: np.ndarray | None = None

    if left_mp4.exists():
        print("Loading left wrist camera ...")
        left_frames = _load_video(left_mp4)
        left_ts = _load_timestamps(episode_dir, "camera_left")
        lh, lw = left_frames.shape[1], left_frames.shape[2]
        pip_h = int(pip_w * lh / lw)
        print(f"  {lw}x{lh}, {len(left_frames)} frames")
    else:
        print("  camera_left-images-rgb.mp4 not found — skipping left PiP")

    if right_mp4.exists():
        print("Loading right wrist camera ...")
        right_frames = _load_video(right_mp4)
        right_ts = _load_timestamps(episode_dir, "camera_right")
        rh, rw = right_frames.shape[1], right_frames.shape[2]
        if left_frames is None:
            pip_h = int(pip_w * rh / rw)
        print(f"  {rw}x{rh}, {len(right_frames)} frames")
    else:
        print("  camera_right-images-rgb.mp4 not found — skipping right PiP")

    # Output dimensions — h264 requires even values for yuv420p chroma subsampling.
    out_w = max(2, (int(round(W * scale)) // 2) * 2)
    out_h = max(2, (int(round(H * scale)) // 2) * 2)

    print(f"\nPiP size: {pip_w}x{pip_h}  (scale={pip_scale:.2f})")
    print(f"Output: {out_w}x{out_h}  (scale={scale:.2f}, crf={crf}, preset={preset})")
    print(f"Writing composite to: {output}")

    fps = round((T - 1) / (top_ts[-1] - top_ts[0])) if T > 1 else 30

    import av

    with av.open(str(output), "w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = out_w
        stream.height = out_h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": preset}

        for i, (frame, t) in enumerate(zip(top_frames, top_ts)):
            left_f = _sample_frame(left_frames, left_ts, t) if left_frames is not None else None
            right_f = _sample_frame(right_frames, right_ts, t) if right_frames is not None else None
            composite = _composite(frame, left_f, right_f, pip_w, pip_h)
            if (out_w, out_h) != (W, H):
                composite = _resize_frame(composite, out_w, out_h)
            av_frame = av.VideoFrame.from_ndarray(composite, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
            if i % 50 == 0:
                print(f"  frame {i}/{T} ...", end="\r")

        for packet in stream.encode():
            container.mux(packet)

    print(f"\nDone. {T} frames written.")


def _play(path: Path) -> None:
    """Open the output video with a system player (ffplay or mpv)."""
    for player in ("ffplay", "mpv", "vlc"):
        if subprocess.run(["which", player], capture_output=True).returncode == 0:
            subprocess.run([player, str(path)])
            return
    print(f"No video player found. Open manually: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose top + wrist camera views from a robots_realtime episode"
    )
    parser.add_argument(
        "episode_dir",
        help="Path to episode directory (contains camera_top-images-rgb.mp4 etc.)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output MP4 path (default: <episode_dir>/composite.mp4)",
    )
    parser.add_argument(
        "--pip-scale",
        type=float,
        default=0.25,
        help="PiP width as fraction of main frame width (default: 0.25)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=30,
        help="x264 CRF (lower=better; 18=archival, 23=default, 28-32=QA-grade) (default: 30)",
    )
    parser.add_argument(
        "--preset",
        default="slow",
        help="x264 preset; slower = smaller at same CRF (default: slow)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Output resolution scale relative to top camera (default: 1.0)",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Write output file but do not open a video player",
    )
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    if not episode_dir.is_dir():
        parser.error(f"Not a directory: {episode_dir}")

    output = Path(args.output) if args.output else episode_dir / "composite.mp4"

    build_composite(
        episode_dir,
        output,
        pip_scale=args.pip_scale,
        crf=args.crf,
        preset=args.preset,
        scale=args.scale,
    )

    if not args.no_play:
        _play(output)


if __name__ == "__main__":
    main()
