from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from PIL import Image

from robots_realtime.sensors.cameras.zed_camera import RESOLUTION_MAP, ZedCamera


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a single frame from a ZED camera and save RGB images as PNG files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where captured PNG files will be stored (default: current directory).",
    )
    parser.add_argument(
        "--filename-prefix",
        default="zed_capture",
        help="Prefix for the saved PNG filenames.",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default=None,
        help="Optional ZED camera serial number. If omitted, the first detected camera is used.",
    )
    parser.add_argument(
        "--resolution",
        choices=tuple(RESOLUTION_MAP.keys()),
        default="HD720",
        help="Capture resolution.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Capture framerate.",
    )
    parser.add_argument(
        "--return-right-image",
        action="store_true",
        help="Also retrieve and save the right RGB image.",
    )
    parser.add_argument(
        "--concat-image",
        action="store_true",
        help="Save a single concatenated RGB image (requires --return-right-image).",
    )
    parser.add_argument(
        "--enable-depth",
        action="store_true",
        help="Enable depth sensing (depth data is not saved by this script).",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Number of attempts to grab a valid frame before giving up.",
    )
    return parser.parse_args()


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    logging.debug("Converting image from dtype %s to uint8", image.dtype)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255)
    return image.astype(np.uint8)


def _save_images(images: Dict[str, np.ndarray], output_dir: Path, filename_prefix: str) -> Iterable[Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    saved_paths = []
    for name, image in images.items():
        if image is None:
            logging.warning("No data returned for image '%s'; skipping.", name)
            continue
        output_path = output_dir / f"{filename_prefix}_{timestamp}_{name}.png"
        Image.fromarray(_ensure_uint8(image)).save(output_path)
        saved_paths.append(output_path)
        logging.info("Saved %s", output_path)
    return saved_paths


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.concat_image and not args.return_right_image:
        raise ValueError("--concat-image requires --return-right-image")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    camera = ZedCamera(
        device_id=args.device_id,
        resolution=args.resolution,
        fps=args.fps,
        concat_image=args.concat_image,
        return_right_image=args.return_right_image,
        enable_depth=args.enable_depth,
    )

    try:
        images = {}
        for attempt in range(1, args.attempts + 1):
            logging.info("Attempt %d/%d to capture image...", attempt, args.attempts)
            data = camera.read()
            images = data.images
            if any(img is not None for img in images.values()):
                break
            logging.warning("Received empty images, retrying...")

        if not any(img is not None for img in images.values()):
            raise RuntimeError("Failed to capture image from ZED camera.")

        saved_paths = list(_save_images(images, args.output_dir, args.filename_prefix))
        if not saved_paths:
            raise RuntimeError("No images were saved; all captures were empty.")

        logging.info("Capture complete. Saved %d file(s).", len(saved_paths))
    finally:
        camera.stop()


if __name__ == "__main__":
    main()

