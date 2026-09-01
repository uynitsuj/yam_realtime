#!/usr/bin/env python3
"""Probe Stereolabs ZED cameras reachable from this machine and emit a CameraNode YAML snippet.

What it checks, in order:
  1. The ZED SDK install is readable by this user (pyzed dlopen()s /usr/local/zed/lib/*.so).
  2. ``pyzed`` imports and its version matches the SDK.
  3. Which ZED cameras are enumerated locally (USB) and which SDK network streams are visible.
  4. (default) Opens every camera, grabs N frames, reports the real resolution / fps / model /
     intrinsics, and saves two JPEGs per camera: the raw frame and the 224x224 resize-with-pad
     version, i.e. exactly what the OpenPI policy will see.

Examples:
    uv run scripts/probe_zed_cameras.py                       # list + open everything at HD720/30
    uv run scripts/probe_zed_cameras.py --list-only
    uv run scripts/probe_zed_cameras.py --serial 12345678 --resolution HD1080 --fps 30 --frames 120
    uv run scripts/probe_zed_cameras.py --stream 10.0.128.50:30000   # ZED X streamed from a Jetson
    uv run scripts/probe_zed_cameras.py --yaml                # print CameraNode entries for the detected serials
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ZED_SDK_DIR = Path("/usr/local/zed")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAVE_DIR = Path("/tmp/zed_probe")
POLICY_INPUT_SIZE = 224
# Order in which detected cameras are assigned node names when printing YAML; re-order by hand
# after checking the saved JPEGs.
NODE_NAMES = ("camera_top", "camera_left", "camera_right")

logger = logging.getLogger("probe_zed")


def sdk_version_from_header() -> str | None:
    header = ZED_SDK_DIR / "include" / "sl" / "Camera.hpp"
    if not header.is_file() or not os.access(header, os.R_OK):
        return None
    text = header.read_text(errors="ignore")
    parts = []
    for macro in ("ZED_SDK_MAJOR_VERSION", "ZED_SDK_MINOR_VERSION", "ZED_SDK_PATCH_VERSION"):
        m = re.search(rf"#define\s+{macro}\s+(\d+)", text)
        if not m:
            return None
        parts.append(m.group(1))
    return ".".join(parts)


def bundled_wheel_version() -> str | None:
    wheels = sorted((REPO_ROOT / "dependencies").glob("pyzed-*.whl"))
    if not wheels:
        return None
    m = re.match(r"pyzed-(\d+\.\d+)", wheels[-1].name)
    return m.group(1) if m else None


def check_sdk_access() -> bool:
    """Return True when the SDK directory is usable; print an actionable fix otherwise."""
    if not ZED_SDK_DIR.is_dir():
        print(f"[FAIL] ZED SDK not found at {ZED_SDK_DIR}. Install ZED SDK 5.x for CUDA 12 from "
              "https://www.stereolabs.com/developers/release and re-run.")
        return False
    lib_dir = ZED_SDK_DIR / "lib"
    if not (os.access(ZED_SDK_DIR, os.R_OK | os.X_OK) and os.access(lib_dir, os.R_OK | os.X_OK)):
        st = ZED_SDK_DIR.stat()
        print(f"[FAIL] {ZED_SDK_DIR} is not readable by uid {os.getuid()} (owner uid {st.st_uid}, "
              f"mode {oct(st.st_mode & 0o777)}). pyzed will fail to load libsl_zed.so.")
        print(f"       Fix (needs sudo):  sudo chmod -R o+rX {ZED_SDK_DIR}")
        return False
    sdk_ver = sdk_version_from_header()
    wheel_ver = bundled_wheel_version()
    print(f"[ OK ] ZED SDK at {ZED_SDK_DIR} (version {sdk_ver or 'unknown'}), bundled pyzed wheel {wheel_ver or 'n/a'}")
    if sdk_ver and wheel_ver and not sdk_ver.startswith(wheel_ver):
        print(f"[WARN] pyzed wheel {wheel_ver} does not match SDK {sdk_ver}. Regenerate with "
              f"`python {ZED_SDK_DIR}/get_python_api.py`, copy the .whl into dependencies/ and update "
              "pyproject.toml [tool.uv.sources].pyzed.")
    return True


def import_sl():
    try:
        from pyzed import sl  # noqa: PLC0415
    except ImportError as exc:
        print(f"[FAIL] `from pyzed import sl` failed: {exc}")
        print("       Install with `uv sync --extra sensors` (uses dependencies/pyzed-*.whl) and make sure "
              f"{ZED_SDK_DIR}/lib is readable.")
        sys.exit(2)
    print(f"[ OK ] pyzed imported, SDK runtime {sl.Camera.get_sdk_version()}")
    return sl


def usb_zed_present() -> bool | None:
    """Best-effort check of the USB bus for a Stereolabs (vendor 0x2b03) device via sysfs."""
    root = Path("/sys/bus/usb/devices")
    if not root.is_dir():
        return None
    for dev in root.iterdir():
        vendor = dev / "idVendor"
        try:
            if vendor.is_file() and vendor.read_text().strip().lower() == "2b03":
                return True
        except OSError:
            continue
    return False


def resize_with_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Letterbox to size x size -- the same op the LeRobot converter and OpenPI apply."""
    try:
        from openpi_client.image_tools import resize_with_pad as _rwp  # noqa: PLC0415

        return _rwp(img, size, size)
    except ImportError:
        import cv2  # noqa: PLC0415

        h, w = img.shape[:2]
        scale = size / max(h, w)
        resized = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size, 3), dtype=img.dtype)
        y0 = (size - resized.shape[0]) // 2
        x0 = (size - resized.shape[1]) // 2
        canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        return canvas


def save_jpeg(path: Path, rgb: np.ndarray) -> None:
    import cv2  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def probe_camera(kind: str, target: str, args: argparse.Namespace) -> dict | None:
    """Open one camera (serial or stream), grab frames, report stats, save previews."""
    from robots_realtime.sensors.cameras.zed_camera import ZedCamera  # noqa: PLC0415

    kwargs = dict(resolution=args.resolution, fps=args.fps, image_key="rgb", check_black_frames=False)
    if kind == "serial":
        kwargs["device_id"] = target
    else:
        kwargs["stream_ip"] = target
    label = f"{kind}={target}"
    print(f"\n--- opening {label} at {args.resolution}/{args.fps} ---")
    try:
        cam = ZedCamera(**kwargs)
    except Exception as exc:  # noqa: BLE001 - report and continue with the next camera
        print(f"[FAIL] {label}: {exc}")
        return None

    try:
        info = cam.get_camera_info()
        first = cam.read().images["rgb"]
        t0 = time.perf_counter()
        n_ok = 0
        for _ in range(args.frames):
            frame = cam.read().images["rgb"]
            n_ok += 1
        elapsed = time.perf_counter() - t0
        measured_fps = n_ok / elapsed if elapsed > 0 else float("nan")
        k = info["intrinsics"]["left"]["intrinsics_matrix"]
        hfov = np.degrees(2 * np.arctan(info["width"] / (2 * k[0, 0])))
        print(f"[ OK ] {label}: model={info['camera_model']} serial={info['device_id']} "
              f"{info['width']}x{info['height']} requested {args.fps} fps, measured {measured_fps:.1f} fps "
              f"over {n_ok} frames, HFOV~{hfov:.0f} deg, fx={k[0, 0]:.1f} cx={k[0, 2]:.1f}")
        mean = float(first.mean())
        if mean < 8:
            print(f"[WARN] {label}: frame is nearly black (mean {mean:.1f}); lens cap / exposure?")
        if args.save_dir:
            stem = f"zed_{info['device_id']}"
            save_jpeg(args.save_dir / f"{stem}_raw.jpg", frame)
            policy_view = resize_with_pad(frame, POLICY_INPUT_SIZE)
            save_jpeg(args.save_dir / f"{stem}_policy{POLICY_INPUT_SIZE}.jpg", policy_view)
            print(f"       saved {args.save_dir}/{stem}_raw.jpg and {stem}_policy{POLICY_INPUT_SIZE}.jpg")
        return {"kind": kind, "target": target, "serial": info["device_id"], "model": info["camera_model"],
                "width": info["width"], "height": info["height"], "fps": measured_fps}
    finally:
        cam.stop()


def print_yaml(cameras: list[dict], args: argparse.Namespace) -> None:
    print("\n# --- CameraNode entries (paste into a session config; re-order names to match the mounts) ---")
    for name, cam in zip(NODE_NAMES, cameras, strict=False):
        print(f"  - type: CameraNode\n    name: {name}\n    driver: ZedCamera")
        if cam["kind"] == "serial":
            print(f'    device_id: "{cam["serial"]}"   # {cam["model"]}')
        else:
            print(f'    stream_ip: "{cam["target"]}"   # {cam["model"]} serial {cam["serial"]}')
        print(f"    resolution: {args.resolution}\n    fps: {args.fps}\n    image_key: rgb\n"
              f"    publish_resize: [{POLICY_INPUT_SIZE}, {POLICY_INPUT_SIZE}]\n    publish_resize_mode: pad\n")
    if len(cameras) > len(NODE_NAMES):
        print(f"# {len(cameras) - len(NODE_NAMES)} more camera(s) detected than node names; add entries by hand.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial", action="append", default=[], help="Only open this serial (repeatable).")
    parser.add_argument("--stream", action="append", default=[], help="ZED SDK stream host[:port] (repeatable).")
    parser.add_argument("--resolution", default="HD720", help="AUTO | HD2K | HD1200 | HD1080 | HD720 | SVGA | VGA")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=60, help="Frames to grab per camera for the fps estimate.")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR, help="Where to write preview JPEGs.")
    parser.add_argument("--no-save", action="store_true", help="Do not write preview JPEGs.")
    parser.add_argument("--list-only", action="store_true", help="Enumerate only; do not open cameras.")
    parser.add_argument("--yaml", action="store_true", help="Print CameraNode YAML for the probed cameras.")
    args = parser.parse_args()
    if args.no_save:
        args.save_dir = None
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if not check_sdk_access():
        return 1
    sl = import_sl()

    usb = usb_zed_present()
    if usb is False:
        print("[INFO] no Stereolabs USB device (vendor 2b03) on the bus right now. USB ZEDs need a USB 3 port; "
              "ZED X (GMSL2) cannot attach to an x86 host -- use --stream from the Jetson that owns it.")

    devices = sl.Camera.get_device_list()
    print(f"\nLocal ZED cameras: {len(devices)}")
    for d in devices:
        print(f"  serial={d.serial_number} model={str(d.camera_model).replace('CAMERA_MODEL.', '')} "
              f"state={str(d.camera_state).replace('CAMERA_STATE.', '')} id={d.id}")
    streams = sl.Camera.get_streaming_device_list()
    print(f"ZED SDK network streams: {len(streams)}")
    for s in streams:
        print(f"  {s.ip}:{s.port} serial={s.serial_number} codec={str(s.codec).replace('STREAMING_CODEC.', '')}")

    if args.list_only:
        return 0

    targets: list[tuple[str, str]] = [("serial", s) for s in args.serial] + [("stream", s) for s in args.stream]
    if not targets:
        targets = [("serial", str(d.serial_number)) for d in devices]
        targets += [("stream", f"{s.ip}:{s.port}") for s in streams]
    if not targets:
        print("\nNothing to open. Plug in a ZED (check `lsusb | grep 2b03`) or pass --stream host:port.")
        return 1

    probed = [r for r in (probe_camera(kind, t, args) for kind, t in targets) if r is not None]
    print(f"\nProbed {len(probed)}/{len(targets)} camera(s) successfully.")
    if args.yaml and probed:
        print_yaml(probed, args)
    return 0 if len(probed) == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())
