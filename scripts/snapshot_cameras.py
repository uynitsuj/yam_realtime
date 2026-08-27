#!/usr/bin/env python3
"""Snapshot every camera attached to this machine into one timestamped folder.

Meant for before/after comparisons when swapping or re-mounting cameras: run it
once, change the hardware, run it again, and diff the two folders.

Discovery is shared with ``camera_web_viewer.py`` (RealSense via pyrealsense2,
ZED via pyzed incl. ``--zed-stream``, plain UVC webcams), and frames are read
through the same drivers a session uses, so the PNGs are what a CameraNode
would publish.

Output (``--out-root``/``<YYYYmmdd_HHMMSS>[_<tag>]/``)::

    <label>_<serial>.png             full-resolution RGB frame, lossless
    <label>_<serial>_policy224.jpg   224x224 resize_with_pad (what an OpenPI policy sees)
    <label>_<serial>.json            serial, model, resolution, fps, intrinsics, driver info
    montage.jpg                      every camera side by side, labelled
    manifest.json                    per-camera status and file list

Examples::

    uv run scripts/snapshot_cameras.py --tag before_wrist_swap \\
        --names-from configs/yam/yam_bimanual_openpi_policy_xdof_hq_us05.yaml
    uv run scripts/snapshot_cameras.py --frames 3 --interval 1.0     # short burst per camera
    uv run scripts/snapshot_cameras.py --only camera_left            # substring match on label/serial
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Sibling script, not a package: reuse its discovery and DeviceSpec.build without duplicating them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_web_viewer import DeviceSpec, discover_all, load_name_map, parse_resolution  # noqa: E402

logger = logging.getLogger("snapshot_cameras")

POLICY_INPUT_SIZE = 224
MONTAGE_TILE_HEIGHT = 360


def resize_with_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Letterbox to size x size, same op as the LeRobot converter / OpenPI ResizeImages."""
    try:
        from openpi_client.image_tools import resize_with_pad as _rwp  # noqa: PLC0415

        return _rwp(img, size, size)
    except ImportError:
        h, w = img.shape[:2]
        scale = size / max(h, w)
        resized = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size, 3), dtype=img.dtype)
        y0, x0 = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        return canvas


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _safe(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in label)


def snapshot_camera(
    spec: DeviceSpec,
    out_dir: Path,
    resolution: tuple[int, int] | None,
    fps: int,
    warmup_s: float,
    frames: int,
    interval_s: float,
) -> dict[str, Any]:
    """Open one camera, let auto-exposure settle, save ``frames`` snapshots. Never raises."""
    stem = f"{_safe(spec.label)}_{_safe(spec.detail)}"
    record: dict[str, Any] = {"id": spec.id, "label": spec.label, "kind": spec.kind, "detail": spec.detail, "files": []}
    logger.info("%s (%s %s): opening", spec.label, spec.kind, spec.detail)
    try:
        driver = spec.build(resolution, fps)
    except Exception as exc:  # noqa: BLE001 - one bad camera must not abort the rest
        logger.error("%s: open failed: %s", spec.label, exc)
        record.update(status="error", error=str(exc))
        return record

    try:
        # Read continuously during warm-up so auto-exposure / white balance converge on the scene.
        t_end = time.monotonic() + warmup_s
        n_warm = 0
        while time.monotonic() < t_end:
            driver.read()
            n_warm += 1

        last_rgb = None
        for i in range(frames):
            if i > 0:
                time.sleep(interval_s)
            data = driver.read()
            rgb = np.ascontiguousarray(data.images["rgb"])
            last_rgb = rgb
            suffix = "" if frames == 1 else f"_{i:02d}"
            png = out_dir / f"{stem}{suffix}.png"
            cv2.imwrite(str(png), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            record["files"].append(png.name)
            if i == 0:
                policy_jpg = out_dir / f"{stem}_policy{POLICY_INPUT_SIZE}.jpg"
                cv2.imwrite(str(policy_jpg), cv2.cvtColor(resize_with_pad(rgb, POLICY_INPUT_SIZE), cv2.COLOR_RGB2BGR))
                record["files"].append(policy_jpg.name)

        info: dict[str, Any] = {}
        try:
            info = dict(driver.get_camera_info())
        except Exception as exc:  # noqa: BLE001
            info = {"error": f"get_camera_info failed: {exc}"}
        try:
            info["intrinsics"] = driver.read_calibration_data_intrinsics()
        except Exception as exc:  # noqa: BLE001
            info["intrinsics_error"] = str(exc)
        info.update(
            label=spec.label,
            kind=spec.kind,
            detail=spec.detail,
            extra=spec.extra,
            requested_resolution=list(resolution) if resolution else "native",
            requested_fps=fps,
            frame_shape=list(last_rgb.shape),
            warmup_frames=n_warm,
            captured_at=datetime.now().isoformat(timespec="seconds"),
        )
        (out_dir / f"{stem}.json").write_text(json.dumps(_jsonable(info), indent=2))
        record["files"].append(f"{stem}.json")
        record.update(status="ok", frame_shape=list(last_rgb.shape), warmup_frames=n_warm, _rgb=last_rgb)
        logger.info("%s: saved %s (%dx%d)", spec.label, stem, last_rgb.shape[1], last_rgb.shape[0])
    except Exception as exc:  # noqa: BLE001
        logger.error("%s: capture failed: %s", spec.label, exc)
        record.update(status="error", error=str(exc))
    finally:
        try:
            driver.stop()
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s: stop failed: %s", spec.label, exc)
    return record


def write_montage(records: list[dict[str, Any]], path: Path) -> bool:
    tiles = []
    for rec in records:
        rgb = rec.get("_rgb")
        if rgb is None:
            continue
        h, w = rgb.shape[:2]
        scale = MONTAGE_TILE_HEIGHT / h
        tile = cv2.resize(rgb, (round(w * scale), MONTAGE_TILE_HEIGHT), interpolation=cv2.INTER_AREA)
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        caption = f"{rec['label']}  {rec['detail']}  {w}x{h}"
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 26), (0, 0, 0), thickness=-1)
        cv2.putText(tile, caption, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return False
    cv2.imwrite(str(path), cv2.hconcat(tiles))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", type=Path, default=Path("recordings/camera_snapshots"))
    parser.add_argument("--tag", default=None, help="appended to the folder name, e.g. before_wrist_swap")
    parser.add_argument("--names-from", type=Path, default=None, help="session YAML for friendly camera names")
    parser.add_argument(
        "--resolution", type=parse_resolution, default="640x480", help="capture WxH or `native` (default 640x480)"
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--uvc-fps", type=int, default=None, help="fps requested from UVC cameras only (default: --fps)"
    )
    parser.add_argument("--warmup-s", type=float, default=2.0, help="seconds of frames discarded for AE/AWB to settle")
    parser.add_argument("--frames", type=int, default=1, help="snapshots per camera")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between snapshots when --frames > 1")
    parser.add_argument("--only", action="append", default=[], help="only cameras whose label/serial/id contains this")
    parser.add_argument("--zed-resolution", default=None, help="ZED preset override (VGA|HD720|HD1080|HD2K|...)")
    parser.add_argument("--zed-stream", action="append", default=[], metavar="HOST[:PORT]")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    specs = discover_all(
        load_name_map(args.names_from), zed_streams=tuple(args.zed_stream), zed_resolution=args.zed_resolution
    )
    if args.only:
        specs = [s for s in specs if any(k in f"{s.label} {s.detail} {s.id}" for k in args.only)]
    if args.uvc_fps is not None:
        specs = [replace(s, fps=args.uvc_fps) if s.kind in ("uvc", "uvc-mode") else s for s in specs]
    if not specs:
        logger.error("no cameras found (after --only filter)")
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / (f"{stamp}_{_safe(args.tag)}" if args.tag else stamp)
    out_dir.mkdir(parents=True, exist_ok=False)
    logger.info("snapshotting %d camera(s) into %s", len(specs), out_dir)

    # Sequential on purpose: opening several RealSense/ZED units at once on one USB controller
    # is exactly the kind of contention that produces a bad "before" reference.
    records = [
        snapshot_camera(spec, out_dir, args.resolution, args.fps, args.warmup_s, args.frames, args.interval)
        for spec in specs
    ]
    if write_montage(records, out_dir / "montage.jpg"):
        logger.info("montage: %s", out_dir / "montage.jpg")

    manifest = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "resolution": list(args.resolution) if args.resolution else "native",
        "fps": args.fps,
        "names_from": str(args.names_from) if args.names_from else None,
        "cameras": [{k: v for k, v in r.items() if not k.startswith("_")} for r in records],
    }
    (out_dir / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2))

    n_ok = sum(r["status"] == "ok" for r in records)
    for r in records:
        mark = "OK  " if r["status"] == "ok" else "FAIL"
        detail = "x".join(map(str, r["frame_shape"][1::-1])) if r["status"] == "ok" else r.get("error", "")
        print(f"  {mark} {r['label']:<16} {r['kind']:<10} {r['detail']:<22} {detail}")
    print(f"\n{n_ok}/{len(records)} cameras saved under {out_dir.resolve()}")
    return 0 if n_ok == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
