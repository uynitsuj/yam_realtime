"""Generic UVC / V4L2 USB camera driver (OpenCV backend) for the ``CameraDriver`` protocol.

Covers any webcam-class camera the kernel exposes as ``/dev/videoN`` — the
YHTek / decxin Realtek-5883 modules on the autolab YAM rig, Logitech C-series,
etc. Only ``read() / stop() / get_camera_info() / read_calibration_data_intrinsics()``
live here; recording and bus publishing are handled upstream by ``CameraNode``.

Why this is more than ``cv2.VideoCapture().read()``:

* **Freshest frame, not oldest.** V4L2 hands frames out FIFO. If the camera
  streams faster than the node polls (the YHTek modules only do 60 fps in
  MJPG at 640x480, and a policy CameraNode runs at 30 Hz), a plain blocking
  ``read()`` returns a frame that has been sitting in the driver queue for a
  full period. With ``threaded=True`` (default) a background thread drains
  the device continuously and ``read()`` hands back the first frame captured
  *after* the call — so a 30 Hz node sampling a 40-60 fps stream sees frames
  ~17 ms old (the camera's own capture→USB latency) instead of 17-40 ms,
  and never the same frame twice. Cost: read() blocks ≤ one device period,
  which CameraNode's deadline-scheduled fixed-rate loop absorbs.
* **Kernel timestamps.** The V4L2 backend exposes each buffer's
  ``CLOCK_MONOTONIC`` capture stamp through ``CAP_PROP_POS_MSEC``. We shift it
  onto the wall clock so it lines up with the RealSense / ZED hardware stamps
  and the robot joint-state stamps, instead of guessing a fixed transfer
  offset. Falls back to ``time.time() - image_transfer_time_offset`` if the
  driver reports nothing usable.
* **Deterministic device addressing.** Pass ``/dev/v4l/by-path/...`` (stable
  across reboots and renumbering) rather than ``/dev/videoN``. Use
  ``discover_usb_cameras()`` / ``python -m ...opencv_camera --list`` to map
  ports to devices. Never probe other nodes at init: sibling CameraNodes open
  theirs concurrently and a probe can steal a device.

Handy shell checks::

    v4l2-ctl --list-devices
    v4l2-ctl -d /dev/video0 --list-formats-ext     # pixel formats x sizes x fps
    v4l2-ctl -d /dev/video0 --list-ctrls           # exposure / gain / wb ranges
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver

logger = logging.getLogger(__name__)


RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "VGA": (640, 480),
    "SVGA": (800, 600),
    "HD720": (1280, 720),
    "HD1080": (1920, 1080),
}


def _resolve_resolution(resolution: Any) -> Tuple[int, int]:
    """Accept ``(w, h)``, ``[w, h]``, ``"WxH"`` or a preset name → ``(w, h)``."""
    if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
        return int(resolution[0]), int(resolution[1])
    if isinstance(resolution, str):
        if "x" in resolution:
            w, h = resolution.split("x", 1)
            return int(w), int(h)
        if resolution in RESOLUTION_PRESETS:
            return RESOLUTION_PRESETS[resolution]
    raise ValueError(
        f"Unknown resolution {resolution!r}. Use 'WxH', (w, h), or a preset: {list(RESOLUTION_PRESETS)}"
    )


def _fourcc_to_str(code: float) -> str:
    try:
        return int(code).to_bytes(4, "little").decode("ascii", errors="replace")
    except (OverflowError, ValueError):
        return "????"


# OpenCV's V4L2 backend maps CAP_PROP_AUTO_EXPOSURE straight onto the UVC
# ``auto_exposure`` menu: 1 = Manual Mode, 3 = Aperture Priority Mode (auto).
_V4L2_EXPOSURE_MANUAL = 1
_V4L2_EXPOSURE_AUTO = 3


@dataclass
class OpencvCamera(CameraDriver):
    """USB / V4L2 camera addressed by device path.

    Args:
        device_path: ``/dev/videoN`` or, preferably, a ``/dev/v4l/by-path/...``
            symlink so the mapping survives renumbering. An int index also works.
        resolution: ``(w, h)``, ``"WxH"`` or preset (``VGA``/``HD720``/...).
        fps: Requested frame rate. UVC cameras only honour rates they advertise
            for the chosen ``fourcc`` x size (``v4l2-ctl --list-formats-ext``);
            the actual rate is logged and exposed via ``get_camera_info()``.
        fourcc: V4L2 pixel format. ``MJPG`` is what gets VGA+ at 30-60 fps over
            USB 2 on most webcams (JPEG decode ≈ 1 ms on a desktop CPU).
            ``YUYV`` is uncompressed — artifact-free, but many cameras cap it at
            30 fps @ VGA or a few fps above that.
        threaded: Run a background grab loop and hand out the newest frame
            (see module docstring). ``False`` = plain blocking read; use with
            ``poll_freq: null`` so the driver paces the node.
        auto_exposure: UVC auto-exposure on/off.
        manual_exposure: When ``auto_exposure=False``, exposure_time_absolute in
            V4L2 units (100 µs steps on UVC; YHTek range 0-10000, default 166).
        manual_gain: Sensor gain (YHTek range 0-128, default 64).
        manual_white_balance_k: Lock white balance to this colour temperature in
            Kelvin (disables auto WB). ``None`` keeps auto WB.
        image_transfer_time_offset: ms subtracted from ``time.time()`` ONLY when
            the driver yields no kernel timestamp.
        read_timeout_s: ``read()`` raises if no new frame arrives in this long.
        camera_type / name: informational, recorded in camera info.
    """

    device_path: Any = "/dev/video0"
    camera_type: str = "usb_camera"
    resolution: Any = (640, 480)
    fps: int = 30
    fourcc: Optional[str] = "MJPG"
    threaded: bool = True
    auto_exposure: bool = True
    manual_exposure: Optional[float] = None
    manual_gain: Optional[float] = None
    manual_white_balance_k: Optional[float] = None
    image_transfer_time_offset: int = 80
    read_timeout_s: float = 2.0
    name: Optional[str] = None

    # Populated in __post_init__; callers should not set these directly.
    intrinsic_data: dict = field(default_factory=dict)  # UVC cams report none; kept for CameraNode parity

    def __repr__(self) -> str:
        return (
            f"OpencvCamera(device_path={self.device_path!r}, name={self.name!r}, "
            f"resolution={self._width}x{self._height}, fps={self.fps}, fourcc={self.fourcc!r})"
        )

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        self._width, self._height = _resolve_resolution(self.resolution)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest: Optional[Tuple[np.ndarray, float]] = None  # (rgb, wall-clock ms)
        self._seq = 0
        self._last_seq_returned = 0
        self._thread_exc: Optional[BaseException] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ts_source = "unknown"

        # Open by path (or index) through the V4L2 backend explicitly. Do NOT probe other
        # /dev/video* nodes here: sibling CameraNodes open theirs concurrently and a probe
        # can grab a device out from under them.
        self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"OpencvCamera: could not open {self.device_path!r}. "
                f"Check `v4l2-ctl --list-devices` / `ls -l /dev/v4l/by-path` and that no other process holds it."
            )
        # Format first, then size, then rate: V4L2 re-validates the frame interval
        # against the (format, size) pair, so setting fps earlier can be reverted.
        if self.fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Keep the driver-side queue shallow so a frame never waits behind others.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._configure_exposure()

        self._actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self._actual_fourcc = _fourcc_to_str(self.cap.get(cv2.CAP_PROP_FOURCC))
        if (self._actual_width, self._actual_height) != (self._width, self._height):
            logger.warning(
                "%s: requested %dx%d but device reports %dx%d",
                self, self._width, self._height, self._actual_width, self._actual_height,
            )
        if self.fourcc and self._actual_fourcc != self.fourcc:
            logger.warning("%s: requested fourcc %s but device reports %s", self, self.fourcc, self._actual_fourcc)
        if self._actual_fps and abs(self._actual_fps - self.fps) > 0.5:
            logger.warning(
                "%s: requested %d fps but device streams %.0f fps for %s@%dx%d "
                "(see `v4l2-ctl -d %s --list-formats-ext`). %s",
                self, self.fps, self._actual_fps, self._actual_fourcc, self._actual_width, self._actual_height,
                self.device_path,
                "threaded=True: read() returns the next captured frame, so pace the node with poll_freq."
                if self.threaded else
                "With threaded=False the node will run at the device rate unless poll_freq caps it "
                "(in which case frames come back stale).",
            )

        if self.threaded:
            self._thread = threading.Thread(
                target=self._capture_loop, name=f"OpencvCamera[{self.name or self.device_path}]", daemon=True
            )
            self._thread.start()

        logger.info(
            "OpencvCamera opened: %s -> %s %dx%d@%.0ffps, threaded=%s, auto_exposure=%s",
            self.device_path, self._actual_fourcc, self._actual_width, self._actual_height,
            self._actual_fps, self.threaded, self.auto_exposure,
        )

    def _configure_exposure(self) -> None:
        def _set(prop: int, value: float, label: str) -> None:
            if not self.cap.set(prop, float(value)):
                logger.warning("%s: %s=%r not accepted by driver", self, label, value)
            else:
                logger.info("%s: %s=%r", self, label, value)

        if self.auto_exposure:
            # Only touch the control if the device is currently in manual mode; some
            # UVC firmwares reset exposure to default when toggled.
            if int(self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)) != _V4L2_EXPOSURE_AUTO:
                _set(cv2.CAP_PROP_AUTO_EXPOSURE, _V4L2_EXPOSURE_AUTO, "auto_exposure(3=auto)")
        else:
            _set(cv2.CAP_PROP_AUTO_EXPOSURE, _V4L2_EXPOSURE_MANUAL, "auto_exposure(1=manual)")
            if self.manual_exposure is not None:
                _set(cv2.CAP_PROP_EXPOSURE, self.manual_exposure, "exposure_time_absolute")
        if self.manual_gain is not None:
            _set(cv2.CAP_PROP_GAIN, self.manual_gain, "gain")
        if self.manual_white_balance_k is not None:
            _set(cv2.CAP_PROP_AUTO_WB, 0, "white_balance_automatic")
            _set(cv2.CAP_PROP_WB_TEMPERATURE, self.manual_white_balance_k, "white_balance_temperature")

    # ------------------------------------------------------------------ #
    # Capture
    # ------------------------------------------------------------------ #

    def _frame_timestamp_ms(self, wall_now_ms: float) -> float:
        """Wall-clock ms for the frame just grabbed.

        Prefers the V4L2 buffer stamp (CLOCK_MONOTONIC, surfaced by OpenCV as
        CAP_PROP_POS_MSEC) shifted onto the wall clock. Rejects it if it is
        missing, non-finite, or lands implausibly far from now (a driver that
        stamps with a different clock, or counts from stream start).
        """
        pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms and math.isfinite(pos_ms) and pos_ms > 0:
            mono_to_wall_ms = (time.time() - time.monotonic()) * 1000.0
            ts = pos_ms + mono_to_wall_ms
            if abs(wall_now_ms - ts) < 2000.0:
                self._ts_source = "v4l2_monotonic"
                return ts
        self._ts_source = "wallclock_minus_offset"
        return wall_now_ms - self.image_transfer_time_offset

    def _grab_one(self) -> Tuple[np.ndarray, float]:
        """Block for the next frame from the device. Raises after repeated failures."""
        failures = 0
        while True:
            ok = self.cap.grab()
            wall_now_ms = time.time() * 1000.0
            if ok:
                ts_ms = self._frame_timestamp_ms(wall_now_ms)
                ok, frame = self.cap.retrieve()
            if ok:
                rgb = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2RGB)
                return rgb, ts_ms
            failures += 1
            if failures >= 50:  # ~0.5 s of nothing → the device is gone
                raise RuntimeError(f"{self}: no frames from device (unplugged or claimed elsewhere?)")
            time.sleep(0.01)

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                rgb, ts_ms = self._grab_one()
                with self._cond:
                    self._latest = (rgb, ts_ms)
                    self._seq += 1
                    self._cond.notify_all()
        except BaseException as exc:  # surface to read() on the node thread
            with self._cond:
                self._thread_exc = exc
                self._cond.notify_all()

    # ------------------------------------------------------------------ #
    # CameraDriver protocol
    # ------------------------------------------------------------------ #

    def read(self) -> CameraData:
        """Block for the next frame the camera captures, then return it.

        Waits for a frame with a sequence number beyond whatever has already
        landed at call time (bounded by one device period), so a flat-out node
        is paced by the camera and a fixed-rate node gets the freshest frame
        physically available — never a queued one, never a repeat.
        """
        if not self.threaded:
            rgb, ts_ms = self._grab_one()
            return CameraData(images={"rgb": rgb}, timestamp=ts_ms)

        deadline = time.monotonic() + self.read_timeout_s
        with self._cond:
            want_seq = max(self._seq, self._last_seq_returned) + 1
            while self._seq < want_seq and self._thread_exc is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"{self}: no new frame within {self.read_timeout_s}s")
                self._cond.wait(remaining)
            if self._thread_exc is not None:
                raise RuntimeError(f"{self}: capture thread died") from self._thread_exc
            assert self._latest is not None
            self._last_seq_returned = self._seq
            rgb, ts_ms = self._latest  # newest frame; seq may have advanced past want_seq, that's fine
        return CameraData(images={"rgb": rgb}, timestamp=ts_ms)

    def get_camera_info(self) -> Dict[str, Any]:
        return {
            "camera_type": self.camera_type,
            "device_id": str(self.device_path),
            "device_path": str(self.device_path),
            "width": self._actual_width,
            "height": self._actual_height,
            "fps": self._actual_fps,
            "requested_fps": self.fps,
            "fourcc": self._actual_fourcc,
            "threaded": self.threaded,
            "auto_exposure": self.auto_exposure,
            "exposure_value": self.manual_exposure,
            "timestamp_source": self._ts_source,
        }

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        # UVC exposes no intrinsics; calibrate offline (OpenCV checkerboard) if needed.
        return dict(self.intrinsic_data)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            if self.cap is not None:
                self.cap.release()

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    @staticmethod
    def list_cameras() -> List[int]:
        """Indices of openable /dev/video* nodes. Diagnostic only; never called at init.

        Prefer ``discover_usb_cameras()`` — it doesn't open anything.
        """
        available = []
        for i in range(20):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                available.append(i)
            cap.release()
        return available


# ---------------------------------------------------------------------- #
# Discovery helper — sysfs only, never opens a device
# ---------------------------------------------------------------------- #


def _read_sysfs(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def discover_usb_cameras() -> List[Dict[str, Any]]:
    """Enumerate V4L2 *capture* nodes with their stable by-path aliases and USB identity.

    UVC cameras expose two ``/dev/videoN`` nodes each (video capture + metadata);
    only the ``video-index0`` one delivers frames. Returns one dict per capture
    node: ``device`` (/dev/videoN), ``by_path`` (list of /dev/v4l/by-path symlinks —
    put one of these in your YAML), ``by_id``, ``name``, ``vendor_id``,
    ``product_id``, ``product``, ``serial``, ``usb_path``.
    """
    aliases: Dict[str, Dict[str, List[str]]] = {}
    for kind in ("by-path", "by-id"):
        d = f"/dev/v4l/{kind}"
        if not os.path.isdir(d):
            continue
        for link in sorted(os.listdir(d)):
            if not link.endswith("video-index0"):
                continue
            target = os.path.realpath(os.path.join(d, link))
            aliases.setdefault(target, {}).setdefault(kind, []).append(os.path.join(d, link))

    cams: List[Dict[str, Any]] = []
    sys_root = "/sys/class/video4linux"
    if not os.path.isdir(sys_root):
        return cams
    for node in sorted(os.listdir(sys_root), key=lambda s: int(s[5:]) if s[5:].isdigit() else 0):
        dev = f"/dev/{node}"
        if dev not in aliases:
            continue  # metadata node or non-USB device without by-path alias
        info: Dict[str, Any] = {
            "device": dev,
            "by_path": aliases[dev].get("by-path", []),
            "by_id": aliases[dev].get("by-id", []),
            "name": _read_sysfs(f"{sys_root}/{node}/name"),
        }
        # Walk up from the interface dir to the USB device dir (has idVendor).
        p = os.path.realpath(f"{sys_root}/{node}/device")
        for _ in range(6):
            if os.path.exists(os.path.join(p, "idVendor")):
                info.update(
                    vendor_id=_read_sysfs(os.path.join(p, "idVendor")),
                    product_id=_read_sysfs(os.path.join(p, "idProduct")),
                    product=_read_sysfs(os.path.join(p, "product")),
                    manufacturer=_read_sysfs(os.path.join(p, "manufacturer")),
                    serial=_read_sysfs(os.path.join(p, "serial")),
                    usb_path=os.path.basename(p),
                )
                break
            p = os.path.dirname(p)
        cams.append(info)
    return cams


if __name__ == "__main__":
    import argparse
    import statistics

    parser = argparse.ArgumentParser(description="OpencvCamera bring-up: list devices or stream one and report timing.")
    parser.add_argument("--list", action="store_true", help="print capture nodes with by-path aliases and USB identity")
    parser.add_argument("--device_path", type=str, default="/dev/video0")
    parser.add_argument("--resolution", type=str, default="640x480")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", type=str, default="MJPG")
    parser.add_argument("--no-threaded", action="store_true")
    parser.add_argument("--poll_hz", type=float, default=None, help="pace read() like CameraNode poll_freq (default: flat-out)")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--show", action="store_true", help="cv2.imshow the stream (Esc to quit)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.list:
        for c in discover_usb_cameras():
            print(f"{c['device']}  {c.get('name')!r}  usb={c.get('vendor_id')}:{c.get('product_id')} "
                  f"serial={c.get('serial')!r}  port={c.get('usb_path')}")
            for bp in c["by_path"]:
                print(f"    {bp}")
        raise SystemExit(0)

    cam = OpencvCamera(device_path=args.device_path, resolution=args.resolution, fps=args.fps,
                       fourcc=args.fourcc, threaded=not args.no_threaded)
    print(cam.get_camera_info())
    try:
        if args.show:
            while True:
                d = cam.read()
                cv2.imshow(args.device_path, d.images["rgb"][..., ::-1])
                if cv2.waitKey(1) == 27:
                    break
        else:
            ages, waits = [], []
            t0 = time.time()
            next_t = time.perf_counter()
            for _ in range(args.frames):
                if args.poll_hz:  # deadline-paced like Node._run_fixed_rate, so read() blocking is absorbed
                    next_t += 1.0 / args.poll_hz
                    time.sleep(max(0.0, next_t - time.perf_counter()))
                t = time.perf_counter()
                d = cam.read()
                waits.append((time.perf_counter() - t) * 1000)
                ages.append(time.time() * 1000 - d.timestamp)
            el = time.time() - t0
            print(f"{args.frames / el:.1f} Hz delivered | read() blocked median {statistics.median(waits):.1f} ms | "
                  f"frame age at return median {statistics.median(ages):.1f} ms, max {max(ages):.1f} ms | "
                  f"ts source {cam.get_camera_info()['timestamp_source']}")
    finally:
        cam.stop()
