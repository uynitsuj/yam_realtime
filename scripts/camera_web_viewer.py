"""Live web viewer for every camera attached to this machine.

Discovers all connected RealSense devices (via ``pyrealsense2``), Stereolabs
ZED cameras (via ``pyzed``: USB ZEDs on this machine plus ZED SDK network
streams, e.g. a ZED X streamed from a Jetson) and all plain UVC / V4L2 webcams
(via ``/sys/class/video4linux``), then serves a single page that shows every
stream at once and lets you tick the ones you want to compare side by side.

Run it::

    uv run scripts/camera_web_viewer.py                       # http://0.0.0.0:8080
    uv run scripts/camera_web_viewer.py --port 9000 --fps 15
    uv run scripts/camera_web_viewer.py --names-from configs/yam/<session>.yaml
    uv run scripts/camera_web_viewer.py --zed-stream 10.0.128.50:30000 --zed-resolution HD1080
    uv run scripts/camera_web_viewer.py --resolution native --uvc-view 800x600   # full sensor + lab42-style tile
    uv run scripts/camera_web_viewer.py --resolution native --uvc-fps 60         # Decxin ~36 fps at 1280x1024

``--resolution native`` opens every device in its largest mode (RealSense: biggest
colour profile, 1280x720 on a D405; UVC: biggest advertised MJPG size, 1280x1024 on
a Decxin) instead of the 640x480 default. Beware that many UVC cameras implement
their lower modes as sensor *windows*: the Decxin's 640x480 is the pixel-for-pixel
centre of its 1280x1024 sensor (half the field of view), 1280x720 is a row window,
while 800x600 is the whole sensor scaled. ``--uvc-view WxH[:crop|scale]`` adds, per
UVC camera, an extra tile that emulates that mode from the native frame, so the
native view and e.g. the 800x600 the lab42 station configs record can be compared
side by side without re-opening the device (V4L2 allows one opener per device).

ZED tiles show the rectified LEFT eye through ``ZedCamera`` (``image_key="rgb"``),
i.e. the same frame a session's CameraNode publishes, so they are directly
comparable with what a policy sees. ``--resolution`` is WxH for RealSense/UVC;
ZEDs only support fixed presets, so it is mapped to the smallest USB-safe preset
that covers the requested size (override with ``--zed-resolution``). If the ZED
SDK is unavailable (missing, or ``/usr/local/zed`` not readable), ZEDs fall back
to raw UVC tiles (unrectified side-by-side stereo frame) with a hint in the log.

Cameras are opened lazily: a device is only claimed while at least one browser
tile is showing it, and it is released ``--idle-timeout`` seconds after the last
viewer goes away. That means this viewer can be started and stopped freely
without fighting a running session for the devices -- but it also means a camera
already claimed by a session will show up here as an error tile rather than
stealing the handle.

Streams are delivered as MJPEG (``multipart/x-mixed-replace``), one HTTP
connection per visible tile. Browsers cap concurrent connections per origin at
around six, so keep a single viewer tab open if you have many cameras.
"""

from __future__ import annotations

import argparse
import asyncio
import grp
import logging
import os
import re
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver
from robots_realtime.sensors.cameras.realsense_camera import (
    RealSenseCamera,
    discover_realsense_cameras,
)

logger = logging.getLogger("camera_web_viewer")

V4L_SYSFS = Path("/sys/class/video4linux")


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #


def _probe_v4l_node(device_path: str) -> None:
    """Open the node directly and translate the errno into a fix the user can act on."""
    try:
        fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
    except FileNotFoundError:
        raise RuntimeError(f"{device_path} does not exist (camera unplugged?)") from None
    except PermissionError:
        raise RuntimeError(f"permission denied on {device_path} — {_permission_hint(device_path)}") from None
    except OSError as exc:
        if exc.errno == 16:  # EBUSY
            raise RuntimeError(f"{device_path} is busy — another process is already streaming it") from None
        raise RuntimeError(f"{device_path}: {exc.strerror}") from None
    os.close(fd)


def _permission_hint(device_path: str) -> str:
    """Name the group that owns the node and whether this user is in it."""
    try:
        owner_gid = os.stat(device_path).st_gid
        owner_group = grp.getgrgid(owner_gid).gr_name
    except (OSError, KeyError):
        return "check the device node's ownership and mode"

    if owner_gid in os.getgroups():
        return f"node is group '{owner_group}' and you are a member; check its mode bits"
    return (
        f"node is owned by group '{owner_group}', which you are not in. "
        f"Fix persistently with a udev rule, or immediately with: "
        f"sudo chmod a+rw {device_path}"
    )


@dataclass
class UvcCamera(CameraDriver):
    """Minimal V4L2 / UVC webcam driver for the ``CameraDriver`` protocol.

    Deliberately *not* ``sensors.cameras.opencv_camera.OpencvCamera``: that class
    probes ``cv2.VideoCapture(0..19)`` in ``__post_init__``, which would open the
    RealSense video nodes this viewer is simultaneously streaming.

    Requests MJPG from the device, since a USB 2.0 webcam cannot sustain VGA30 as
    raw YUYV. ``cv2`` decodes it back to BGR for us; we hand out RGB to match the
    protocol contract.
    """

    device_path: str
    resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    name: Optional[str] = None

    def __post_init__(self) -> None:
        # Probe with a raw open() first: cv2.VideoCapture reports every failure
        # as a closed capture, so a permission problem is otherwise
        # indistinguishable from the device being busy.
        _probe_v4l_node(self.device_path)

        self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"{self.device_path}: opened by the kernel but OpenCV could not start "
                f"a stream (unsupported format, or the device is already streaming)"
            )
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Keep the driver-side queue shallow so read() returns the newest frame
        # rather than draining a backlog when a viewer reconnects.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> CameraData:
        ok, frame = self.cap.read()
        capture_time_ms = time.time() * 1000
        if not ok or frame is None:
            raise RuntimeError(f"{self.device_path}: frame grab failed")
        rgb = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2RGB)
        return CameraData(images={"rgb": rgb}, timestamp=capture_time_ms)

    def read_calibration_data_intrinsics(self) -> dict[str, Any]:
        return {}

    def get_camera_info(self) -> dict[str, Any]:
        return {
            "device_id": self.device_path,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
        }

    def stop(self) -> None:
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None


class SharedUvcSource:
    """One opener per UVC device, fanning frames out to several ``UvcView`` drivers.

    V4L2 allows a single streaming owner per device, so a native tile and an
    emulated-mode tile of the same camera cannot both open it. The source opens the
    device once (at the resolution the first view asks for), reads it in one thread,
    and every view picks up the newest frame. The device is released when the last
    view stops.
    """

    _registry: dict[str, "SharedUvcSource"] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def get(cls, device_path: str) -> "SharedUvcSource":
        with cls._registry_lock:
            src = cls._registry.get(device_path)
            if src is None:
                src = cls._registry[device_path] = cls(device_path)
            return src

    def __init__(self, device_path: str) -> None:
        self.device_path = device_path
        self._cond = threading.Condition()
        self._refs = 0
        self._camera: Optional[UvcCamera] = None
        self._thread: Optional[threading.Thread] = None
        self._frame: Optional[np.ndarray] = None
        self._ts = 0.0
        self._seq = 0
        self._error: Optional[Exception] = None

    @property
    def resolution(self) -> Optional[tuple[int, int]]:
        cam = self._camera
        return None if cam is None else tuple(cam.resolution)

    def acquire(self, resolution: tuple[int, int], fps: int, label: Optional[str]) -> None:
        with self._cond:
            if self._camera is None:
                self._camera = UvcCamera(device_path=self.device_path, resolution=resolution, fps=fps, name=label)
                self._frame, self._seq, self._error = None, 0, None
                self._thread = threading.Thread(target=self._run, name=f"uvc-src-{self.device_path}", daemon=True)
                self._thread.start()
            elif tuple(self._camera.resolution) != tuple(resolution):
                logger.warning(
                    "%s is already open at %s; a view asked for %s and will share the open stream",
                    self.device_path, self._camera.resolution, resolution,
                )
            self._refs += 1

    def release(self) -> None:
        with self._cond:
            self._refs -= 1
            if self._refs > 0:
                return
            cam, self._camera = self._camera, None
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if cam is not None:
            cam.stop()

    def _run(self) -> None:
        while True:
            with self._cond:
                cam = self._camera
            if cam is None:
                return
            try:
                data = cam.read()
            except Exception as exc:  # noqa: BLE001 - surfaced to every view via wait_frame
                with self._cond:
                    self._error = exc
                    self._seq += 1
                    self._cond.notify_all()
                time.sleep(0.05)
                continue
            with self._cond:
                self._frame = data.images["rgb"]
                self._ts = data.timestamp
                self._error = None
                self._seq += 1
                self._cond.notify_all()

    def wait_frame(self, after_seq: int, timeout_s: float = 2.0) -> tuple[np.ndarray, float, int]:
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > after_seq or self._camera is None, timeout=timeout_s):
                raise RuntimeError(f"{self.device_path}: no frame within {timeout_s:.0f}s")
            if self._camera is None:
                raise RuntimeError(f"{self.device_path}: source closed")
            if self._error is not None:
                raise RuntimeError(f"{self.device_path}: {self._error}")
            return self._frame, self._ts, self._seq


def emulate_uvc_mode(frame: np.ndarray, mode: tuple[int, int], kind: str = "auto") -> np.ndarray:
    """Reproduce what the camera firmware delivers for a lower UVC mode, from a native frame.

    Measured on the Decxin (Realtek 0bda:5883, 1280x1024 sensor), correlation against the
    real hardware mode: 640x480 = 1:1 centre window (0.997), 1280x720 = 1:1 row window
    (0.999), 800x600 = whole sensor scaled to 800x640 then 20 rows trimmed top/bottom
    (0.997; a plain squish only reaches 0.90). ``auto`` therefore crops when the mode is
    full-width or at most half the sensor width, and scales otherwise.
    """
    h, w = frame.shape[:2]
    mw, mh = mode
    if kind == "auto":
        kind = "crop" if (mw == w or 2 * mw <= w) else "scale"
    if kind == "crop":
        y0, x0 = max(0, (h - mh) // 2), max(0, (w - mw) // 2)
        return frame[y0 : y0 + mh, x0 : x0 + mw]
    scale = max(mw / w, mh / h)
    scaled = cv2.resize(frame, (max(mw, round(w * scale)), max(mh, round(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = scaled.shape[:2]
    y0, x0 = (sh - mh) // 2, (sw - mw) // 2
    return scaled[y0 : y0 + mh, x0 : x0 + mw]


@dataclass
class UvcView(CameraDriver):
    """A ``CameraDriver`` onto a ``SharedUvcSource``: passthrough, or an emulated lower mode."""

    device_path: str
    resolution: tuple[int, int]  # device (source) mode to open
    fps: int = 30
    name: Optional[str] = None
    mode: Optional[tuple[int, int]] = None  # emulate this device mode from the source frame; None = passthrough
    mode_kind: str = "auto"  # auto | crop | scale

    def __post_init__(self) -> None:
        self._source = SharedUvcSource.get(self.device_path)
        self._source.acquire(tuple(self.resolution), self.fps, self.name)
        self._seq = 0
        self._released = False

    def read(self) -> CameraData:
        frame, ts, self._seq = self._source.wait_frame(self._seq)
        if self.mode is not None:
            frame = emulate_uvc_mode(frame, tuple(self.mode), self.mode_kind)
        return CameraData(images={"rgb": frame}, timestamp=ts)

    def read_calibration_data_intrinsics(self) -> dict[str, Any]:
        return {}

    def get_camera_info(self) -> dict[str, Any]:
        src = self._source.resolution or tuple(self.resolution)
        out = tuple(self.mode) if self.mode is not None else src
        return {
            "device_id": self.device_path,
            "source_width": src[0],
            "source_height": src[1],
            "width": out[0],
            "height": out[1],
            "fps": self.fps,
            "emulated_mode": f"{out[0]}x{out[1]} ({self.mode_kind})" if self.mode is not None else None,
        }

    def stop(self) -> None:
        if not self._released:
            self._released = True
            self._source.release()


_V4L_SIZE_RE = re.compile(r"Size: Discrete (\d+)x(\d+)")


def uvc_native_resolution(device_path: str) -> tuple[int, int]:
    """Largest MJPG frame size the UVC device advertises (via ``v4l2-ctl``); 640x480 if unknown."""
    try:
        out = subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--list-formats-ext"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("%s: v4l2-ctl unavailable (%s); assuming 640x480 native", device_path, exc)
        return (640, 480)
    best: Optional[tuple[int, int]] = None
    in_mjpg = False
    for line in out.splitlines():
        if line.strip().startswith("["):
            in_mjpg = "MJPG" in line
            continue
        m = _V4L_SIZE_RE.search(line)
        if m and in_mjpg:
            cand = (int(m.group(1)), int(m.group(2)))
            if best is None or cand[0] * cand[1] > best[0] * best[1]:
                best = cand
    return best or (640, 480)


def realsense_native_resolution(serial: str, fps: int) -> tuple[int, int]:
    """Largest colour profile the RealSense offers at >= ``fps`` (D405: 1280x720 @ 30)."""
    try:
        import pyrealsense2 as rs  # noqa: PLC0415

        for dev in rs.context().query_devices():
            if dev.get_info(rs.camera_info.serial_number) != serial:
                continue
            best: Optional[tuple[int, int]] = None
            for sensor in dev.query_sensors():
                for prof in sensor.get_stream_profiles():
                    if not prof.is_video_stream_profile() or prof.stream_type() != rs.stream.color or prof.fps() < fps:
                        continue
                    video = prof.as_video_stream_profile()
                    cand = (video.width(), video.height())
                    if best is None or cand[0] * cand[1] > best[0] * best[1]:
                        best = cand
            if best is not None:
                return best
    except Exception as exc:  # noqa: BLE001
        logger.debug("realsense native lookup failed for %s: %s", serial, exc)
    return (1280, 720)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


STEREOLABS_USB_VENDOR = "2b03"
ZED_SDK_DIR = Path("/usr/local/zed")
# USB ZED (ZED / Mini / 2 / 2i) presets, smallest first. SVGA/HD1200 are GMSL-only
# (ZED X) and would fail to open on a USB unit, so they are never auto-selected.
ZED_USB_PRESETS: tuple[tuple[str, int, int], ...] = (
    ("VGA", 672, 376),
    ("HD720", 1280, 720),
    ("HD1080", 1920, 1080),
    ("HD2K", 2208, 1242),
)


def zed_resolution_for(resolution: tuple[int, int]) -> str:
    """Smallest USB-safe ZED preset that covers the requested WxH (``--resolution`` is WxH)."""
    width, height = resolution
    for preset, w, h in ZED_USB_PRESETS:
        if w >= width and h >= height:
            return preset
    return ZED_USB_PRESETS[-1][0]


@dataclass
class DeviceSpec:
    """A camera we know how to open, before it has actually been opened."""

    id: str
    label: str
    kind: str  # "realsense" | "uvc" | "uvc-mode" | "zed" | "zed-stream"
    detail: str  # serial number, /dev path, or host:port
    extra: dict[str, str] = field(default_factory=dict)
    # ZED preset override ("HD720", ...). None = derive from the requested WxH.
    zed_resolution: Optional[str] = None
    # Per-device capture resolution; overrides the global one (used to pin UVC sources to native).
    resolution: Optional[tuple[int, int]] = None
    # Per-device frame-rate request; overrides the global one (see --uvc-fps).
    fps: Optional[int] = None
    # "uvc-mode" tiles: emulate this device mode from the shared native frame.
    uvc_mode: Optional[tuple[int, int]] = None
    uvc_mode_kind: str = "auto"

    def build(self, resolution: Optional[tuple[int, int]], fps: int) -> CameraDriver:
        """Open the device. ``resolution=None`` means the device's largest native mode."""
        resolution = self.resolution or resolution
        fps = self.fps or fps
        if self.kind == "realsense":
            res = resolution or realsense_native_resolution(self.detail, fps)
            return RealSenseCamera(device_id=self.detail, resolution=res, fps=fps)
        if self.kind in ("zed", "zed-stream"):
            return self._build_zed(resolution, fps)
        res = resolution or uvc_native_resolution(self.detail)
        return UvcView(
            device_path=self.detail, resolution=res, fps=fps, name=self.label,
            mode=self.uvc_mode, mode_kind=self.uvc_mode_kind,
        )

    def _build_zed(self, resolution: Optional[tuple[int, int]], fps: int) -> CameraDriver:
        # Lazy: zed_camera imports pyzed at module level and raises an actionable ImportError
        # when the SDK cannot be loaded; that message becomes the tile's error text.
        from robots_realtime.sensors.cameras.zed_camera import ZedCamera  # noqa: PLC0415

        # AUTO lets the SDK pick the camera's native mode when no WxH was requested.
        preset = self.zed_resolution or (zed_resolution_for(resolution) if resolution else "AUTO")
        source = {"stream_ip": self.detail} if self.kind == "zed-stream" else {"device_id": self.detail}
        # image_key="rgb": same key the session CameraNodes publish, so the tile shows the exact
        # (rectified, left-eye) frame a policy would consume. Black-frame check off: a viewer
        # should show a capped lens, not die on it.
        return ZedCamera(
            resolution=preset, fps=fps, image_key="rgb", check_black_frames=False, name=self.label, **source
        )


def _zed_sdk_hint() -> str:
    if not ZED_SDK_DIR.is_dir():
        return f"ZED SDK not installed at {ZED_SDK_DIR}"
    if not (os.access(ZED_SDK_DIR, os.R_OK | os.X_OK) and os.access(ZED_SDK_DIR / "lib", os.R_OK | os.X_OK)):
        return f"{ZED_SDK_DIR} is not readable by this user; fix: sudo chmod -R o+rX {ZED_SDK_DIR}"
    return "pyzed is not installed in this venv; run `uv sync --extra sensors` (see scripts/setup_zed.sh)"


def discover_zed_cameras() -> tuple[list[dict[str, str]], bool]:
    """Enumerate USB ZED cameras and ZED SDK network streams.

    Returns ``(cameras, sdk_available)``. ``sdk_available`` is False when ``pyzed`` cannot be
    imported (SDK missing / unreadable / not installed in the venv); callers then leave any
    Stereolabs UVC node in the plain-webcam list so the camera is at least viewable raw.
    """
    try:
        from pyzed import sl  # noqa: PLC0415
    except ImportError as exc:
        logger.info("ZED discovery skipped: %s (%s)", _zed_sdk_hint(), exc)
        return [], False

    cameras: list[dict[str, str]] = []
    try:
        for dev in sl.Camera.get_device_list():
            cameras.append(
                {
                    "kind": "zed",
                    "serial": str(dev.serial_number),
                    "model": str(dev.camera_model).replace("CAMERA_MODEL.", ""),
                    "state": str(dev.camera_state).replace("CAMERA_STATE.", ""),
                }
            )
        for props in sl.Camera.get_streaming_device_list():
            cameras.append(
                {
                    "kind": "zed-stream",
                    "serial": str(props.serial_number),
                    "endpoint": f"{props.ip}:{props.port}",
                    "codec": str(props.codec).replace("STREAMING_CODEC.", ""),
                }
            )
    except Exception as exc:
        logger.warning("ZED discovery failed: %s", exc)
    return cameras, True


def discover_uvc_cameras(exclude_stereolabs: bool = False) -> list[dict[str, str]]:
    """Enumerate non-RealSense V4L2 capture nodes from sysfs.

    UVC devices publish one node per function; the metadata node carries a
    non-zero ``index``, so ``index == 0`` selects the capture node. RealSense
    cameras are excluded here because ``pyrealsense2`` owns them; Stereolabs
    nodes are excluded when ``pyzed`` owns them (``exclude_stereolabs``).
    """
    found: list[dict[str, str]] = []
    if not V4L_SYSFS.is_dir():
        return found

    for node in sorted(V4L_SYSFS.iterdir(), key=lambda p: int(p.name.removeprefix("video") or 0)):
        try:
            name = (node / "name").read_text().strip()
            index = int((node / "index").read_text().strip())
        except (OSError, ValueError):
            continue
        if index != 0 or "realsense" in name.lower():
            continue

        usb = (node / "device").resolve().parent
        vid = _read_or_blank(usb / "idVendor")
        pid = _read_or_blank(usb / "idProduct")
        if exclude_stereolabs and vid.lower() == STEREOLABS_USB_VENDOR:
            continue
        product = _read_or_blank(usb / "product") or name
        found.append(
            {
                "path": f"/dev/{node.name}",
                "name": name,
                "product": product,
                "usb_id": f"{vid}:{pid}" if vid and pid else "",
                # Generic bridges ship one serial for every unit (Decxin: "YHTek"), so the
                # physical USB port is the only stable way to tell two of them apart.
                "serial": _read_or_blank(usb / "serial"),
                "port": usb.name,
            }
        )
    return found


def _read_or_blank(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def load_name_map(config_path: Optional[Path]) -> dict[str, str]:
    """Map ``device_id`` -> ``name`` from a session config's CameraNode entries."""
    if config_path is None:
        return {}
    import yaml  # noqa: PLC0415 -- optional, only needed for --names-from

    try:
        cfg = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("could not read names from %s: %s", config_path, exc)
        return {}

    names: dict[str, str] = {}
    for node in (cfg or {}).get("nodes", []) or []:
        if not isinstance(node, dict) or node.get("type") != "CameraNode":
            continue
        name = node.get("name")
        if not name:
            continue
        # RealSense/ZED: device_id is the serial. ZED-over-network: stream_ip is host[:port].
        for key in ("device_id", "stream_ip"):
            if node.get(key):
                names[str(node[key])] = str(name)
    return names


def discover_all(
    name_map: dict[str, str],
    zed_streams: tuple[str, ...] = (),
    zed_resolution: Optional[str] = None,
) -> list[DeviceSpec]:
    specs: list[DeviceSpec] = []

    for cam in discover_realsense_cameras():
        serial = cam["serial"]
        specs.append(
            DeviceSpec(
                id=f"rs-{serial}",
                label=name_map.get(serial, f"{cam['name'].replace('Intel RealSense ', '')} {serial[-4:]}"),
                kind="realsense",
                detail=serial,
                extra={"model": cam["name"], "firmware": cam.get("firmware", "")},
            )
        )

    zed_cams, zed_sdk_available = discover_zed_cameras()
    seen_streams: set[str] = set()
    for cam in zed_cams:
        if cam["kind"] == "zed":
            specs.append(
                DeviceSpec(
                    id=f"zed-{cam['serial']}",
                    label=name_map.get(cam["serial"], f"{cam['model']} {cam['serial'][-4:]}"),
                    kind="zed",
                    detail=cam["serial"],
                    extra={"model": cam["model"], "state": cam["state"]},
                    zed_resolution=zed_resolution,
                )
            )
        else:
            seen_streams.add(cam["endpoint"])
            specs.append(
                DeviceSpec(
                    id=f"zedstream-{cam['endpoint'].replace(':', '-').replace('.', '-')}",
                    label=name_map.get(
                        cam["endpoint"], name_map.get(cam["endpoint"].split(":")[0], f"ZED stream {cam['serial']}")
                    ),
                    kind="zed-stream",
                    detail=cam["endpoint"],
                    extra={"serial": cam["serial"], "codec": cam["codec"]},
                    zed_resolution=zed_resolution,
                )
            )
    # Explicit --zed-stream endpoints (SDK multicast discovery does not cross subnets).
    for endpoint in zed_streams:
        if endpoint in seen_streams:
            continue
        specs.append(
            DeviceSpec(
                id=f"zedstream-{endpoint.replace(':', '-').replace('.', '-')}",
                label=name_map.get(endpoint, name_map.get(endpoint.split(":")[0], f"ZED stream {endpoint}")),
                kind="zed-stream",
                detail=endpoint,
                zed_resolution=zed_resolution,
            )
        )

    for cam in discover_uvc_cameras(exclude_stereolabs=zed_sdk_available):
        path = cam["path"]
        specs.append(
            DeviceSpec(
                id=f"uvc-{path.rsplit('/', 1)[-1]}",
                label=name_map.get(path, f"{cam['product']} {cam['port']}".strip()),
                kind="uvc",
                detail=path,
                extra={
                    "usb_id": cam["usb_id"], "v4l_name": cam["name"], "serial": cam["serial"], "usb_port": cam["port"]
                },
            )
        )
        if cam["usb_id"].lower().startswith(f"{STEREOLABS_USB_VENDOR}:"):
            logger.warning(
                "%s is a Stereolabs ZED but pyzed is unavailable (%s); showing the raw UVC stereo frame",
                path,
                _zed_sdk_hint(),
            )

    return specs


def parse_uvc_view(value: str) -> tuple[tuple[int, int], str]:
    """``WxH`` or ``WxH:crop`` / ``WxH:scale`` -> ((w, h), kind)."""
    mode, _, kind = value.partition(":")
    kind = kind or "auto"
    if kind not in ("auto", "crop", "scale"):
        raise argparse.ArgumentTypeError(f"--uvc-view kind must be crop|scale|auto, got {kind!r}")
    res = parse_resolution(mode)
    if res is None:
        raise argparse.ArgumentTypeError("--uvc-view needs an explicit WxH")
    return res, kind


def expand_uvc_views(specs: list[DeviceSpec], views: list[tuple[tuple[int, int], str]]) -> list[DeviceSpec]:
    """Per UVC camera: pin the native tile to the device's largest mode and add one emulated tile per view."""
    out: list[DeviceSpec] = []
    for spec in specs:
        if spec.kind != "uvc":
            out.append(spec)
            continue
        native = spec.resolution or uvc_native_resolution(spec.detail)
        native_txt = f"{native[0]}x{native[1]}"
        out.append(replace(spec, resolution=native, extra={**spec.extra, "mode": f"{native_txt} native"}))
        for (mw, mh), kind in views:
            out.append(
                replace(
                    spec,
                    id=f"{spec.id}-{mw}x{mh}",
                    label=f"{spec.label} @{mw}x{mh}",
                    kind="uvc-mode",
                    resolution=native,
                    uvc_mode=(mw, mh),
                    uvc_mode_kind=kind,
                    extra={**spec.extra, "mode": f"{mw}x{mh} emulated ({kind}) from {native_txt}"},
                )
            )
    return out


# --------------------------------------------------------------------------- #
# Capture workers
# --------------------------------------------------------------------------- #


class _Subscriber:
    """One live viewer of a camera, bridging the capture thread to an event loop.

    Frames are pushed rather than polled so that an ``asyncio`` client is parked
    on ``queue.get()``: cancellation on client disconnect then lands on a real
    await point, and the stream generator's ``finally`` actually runs. A blocking
    wait inside a threadpool would leave the subscriber registered forever and
    the device claimed after the browser tab closed.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, maxsize: int = 2) -> None:
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    def offer(self, frame: Optional[bytes]) -> None:
        """Hand a frame (or ``None`` to close) to the loop. Never blocks capture."""

        def _put() -> None:
            if self.queue.full():
                # Drop the stalest frame: a slow client should fall behind, not
                # throttle the capture thread or every other viewer.
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self.queue.put_nowait(frame)

        try:
            self.loop.call_soon_threadsafe(_put)
        except RuntimeError:
            pass  # loop already closed; the subscriber is on its way out


class CameraWorker:
    """Owns one device: opens it on demand, encodes JPEGs, fans them out.

    Encoding happens once per frame in the capture thread no matter how many
    browser tiles are watching, so a second viewer costs bandwidth but not CPU.
    """

    def __init__(
        self,
        spec: DeviceSpec,
        resolution: Optional[tuple[int, int]],
        fps: int,
        jpeg_quality: int,
        idle_timeout_s: float,
    ) -> None:
        self.spec = spec
        self.resolution = resolution
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.idle_timeout_s = idle_timeout_s

        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._shutdown = False

        self._subs: set[_Subscriber] = set()
        self._last_unsubscribe: Optional[float] = None

        self._frame: Optional[bytes] = None
        self._seq = 0
        self._status = "idle"  # idle | opening | streaming | error
        self._error: Optional[str] = None
        self._width = 0
        self._height = 0
        self._measured_fps = 0.0

    # -- lifecycle ---------------------------------------------------------- #

    def add_subscriber(self, sub: _Subscriber) -> None:
        with self._cond:
            self._subs.add(sub)
            self._last_unsubscribe = None
            if self._thread is None or not self._thread.is_alive():
                self._status = "opening"
                self._error = None
                self._frame = None
                self._thread = threading.Thread(
                    target=self._run, name=f"cam-{self.spec.id}", daemon=True
                )
                self._thread.start()
            self._cond.notify_all()

    def remove_subscriber(self, sub: _Subscriber) -> None:
        with self._cond:
            self._subs.discard(sub)
            if not self._subs:
                self._last_unsubscribe = time.monotonic()
            self._cond.notify_all()

    def _broadcast(self, frame: Optional[bytes]) -> None:
        """Fan a frame out to every viewer. Caller must hold ``self._cond``."""
        for sub in self._subs:
            sub.offer(frame)

    def shutdown(self) -> None:
        """Stop capture and close every open stream.

        The broadcast is essential: subscribers park on ``queue.get()``, so
        without an explicit ``None`` their MJPEG responses never finish. Uvicorn
        waits for in-flight requests before running lifespan shutdown, so a
        parked stream would deadlock against the very code meant to release it.
        """
        with self._cond:
            self._shutdown = True
            self._broadcast(None)
            self._cond.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)

    # -- capture thread ----------------------------------------------------- #

    def _should_keep_running(self) -> bool:
        if self._shutdown:
            return False
        if self._subs:
            return True
        # Linger briefly so a page reload or a compare-mode toggle does not
        # force a full device re-open (RealSense takes ~1s to restart).
        if self._last_unsubscribe is None:
            return True
        return (time.monotonic() - self._last_unsubscribe) < self.idle_timeout_s

    def _run(self) -> None:
        driver: Optional[CameraDriver] = None
        try:
            driver = self.spec.build(self.resolution, self.fps)
        except Exception as exc:
            logger.warning("%s: open failed: %s", self.spec.id, exc)
            with self._cond:
                self._status = "error"
                self._error = str(exc)
                self._seq += 1
                self._broadcast(None)
                self._cond.notify_all()
            return

        with self._cond:
            self._status = "streaming"
            self._cond.notify_all()

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        window_start = time.monotonic()
        window_frames = 0
        consecutive_errors = 0

        try:
            while True:
                with self._cond:
                    if not self._should_keep_running():
                        break

                try:
                    data = driver.read()
                    consecutive_errors = 0
                except Exception as exc:
                    consecutive_errors += 1
                    logger.debug("%s: read failed: %s", self.spec.id, exc)
                    if consecutive_errors >= 10:
                        with self._cond:
                            self._status = "error"
                            self._error = f"read failed: {exc}"
                            self._seq += 1
                            self._broadcast(None)
                            self._cond.notify_all()
                        break
                    time.sleep(0.05)
                    continue

                rgb = data.images["rgb"]
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), encode_params)
                if not ok:
                    continue

                window_frames += 1
                elapsed = time.monotonic() - window_start
                measured = self._measured_fps
                if elapsed >= 1.0:
                    measured = window_frames / elapsed
                    window_start, window_frames = time.monotonic(), 0

                with self._cond:
                    self._frame = buf.tobytes()
                    self._seq += 1
                    self._height, self._width = rgb.shape[:2]
                    self._measured_fps = measured
                    self._broadcast(self._frame)
                    self._cond.notify_all()
        finally:
            try:
                driver.stop()
            except Exception as exc:
                logger.debug("%s: stop failed: %s", self.spec.id, exc)
            with self._cond:
                if self._status != "error":
                    self._status = "idle"
                self._measured_fps = 0.0
                self._frame = None
                self._thread = None
                self._seq += 1
                self._broadcast(None)
                self._cond.notify_all()

    # -- consumers ---------------------------------------------------------- #

    async def mjpeg_stream(self) -> Any:
        """Yield multipart MJPEG chunks until the client disconnects."""
        sub = _Subscriber(asyncio.get_running_loop())
        self.add_subscriber(sub)
        try:
            while True:
                frame = await sub.queue.get()
                if frame is None:  # device closed or errored
                    return
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
        finally:
            self.remove_subscriber(sub)

    async def snapshot(self, timeout_s: float = 5.0) -> Optional[bytes]:
        sub = _Subscriber(self._loop_or_current(), maxsize=1)
        self.add_subscriber(sub)
        try:
            return await asyncio.wait_for(sub.queue.get(), timeout_s)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return None
        finally:
            self.remove_subscriber(sub)

    @staticmethod
    def _loop_or_current() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    def status(self) -> dict[str, Any]:
        with self._cond:
            return {
                "id": self.spec.id,
                "label": self.spec.label,
                "kind": self.spec.kind,
                "detail": self.spec.detail,
                "extra": self.spec.extra,
                "status": self._status,
                "error": self._error,
                "width": self._width,
                "height": self._height,
                "fps": round(self._measured_fps, 1),
                "viewers": len(self._subs),
            }


# --------------------------------------------------------------------------- #
# Web app
# --------------------------------------------------------------------------- #


def build_app(workers: dict[str, CameraWorker]) -> "FastAPI":  # noqa: F821 -- imported lazily below
    from contextlib import asynccontextmanager  # noqa: PLC0415

    from fastapi import FastAPI, HTTPException  # noqa: PLC0415
    from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse  # noqa: PLC0415

    @asynccontextmanager
    async def lifespan(_app: "FastAPI") -> Any:
        yield
        for worker in workers.values():
            worker.shutdown()

    app = FastAPI(title="Camera Web Viewer", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/cameras")
    def api_cameras() -> JSONResponse:
        return JSONResponse([w.status() for w in workers.values()])

    @app.get("/stream/{cam_id}")
    async def stream(cam_id: str) -> StreamingResponse:
        worker = workers.get(cam_id)
        if worker is None:
            raise HTTPException(status_code=404, detail=f"unknown camera {cam_id!r}")
        return StreamingResponse(
            worker.mjpeg_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    @app.get("/snapshot/{cam_id}")
    async def snapshot(cam_id: str) -> Response:
        worker = workers.get(cam_id)
        if worker is None:
            raise HTTPException(status_code=404, detail=f"unknown camera {cam_id!r}")
        frame = await worker.snapshot()
        if frame is None:
            raise HTTPException(status_code=503, detail="no frame available")
        return Response(content=frame, media_type="image/jpeg")

    return app


def _lan_ip() -> str:
    """Best-effort outward-facing IP, for printing a reachable URL."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def parse_resolution(value: str) -> Optional[tuple[int, int]]:
    """``WxH`` -> (w, h); ``native`` -> None (each device's largest advertised mode)."""
    if value.strip().lower() == "native":
        return None
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8080, help="bind port (default: %(default)s)")
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default="640x480",
        help="capture WxH, or `native` for each device's largest mode (default: 640x480)",
    )
    parser.add_argument(
        "--uvc-view",
        action="append",
        type=parse_uvc_view,
        default=[],
        metavar="WxH[:crop|scale]",
        help="per UVC camera, add a tile emulating that device mode from the native frame "
        "(forces native capture on UVC devices; repeatable)",
    )
    parser.add_argument("--fps", type=int, default=30, help="requested capture fps (default: %(default)s)")
    parser.add_argument(
        "--uvc-fps",
        type=int,
        default=None,
        help="fps requested from UVC cameras only (default: --fps). The Decxin delivers ~20 fps when asked for 30 "
        "at 1280x1024 but ~36 fps when asked for 60, while the D405 has no 60 fps profile at 1280x720.",
    )
    parser.add_argument(
        "--zed-resolution",
        default=None,
        choices=["VGA", "HD720", "HD1080", "HD2K", "SVGA", "HD1200", "AUTO"],
        help="ZED preset override; default maps --resolution to the smallest USB-safe preset that covers it",
    )
    parser.add_argument(
        "--zed-stream",
        action="append",
        default=[],
        metavar="HOST[:PORT]",
        help="ZED SDK network stream to add (e.g. a ZED X on a Jetson; port defaults to 30000). Repeatable.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=80, help="1-100 (default: %(default)s)")
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=5.0,
        help="seconds to hold a device open after the last viewer leaves (default: %(default)s)",
    )
    parser.add_argument(
        "--names-from",
        type=Path,
        default=None,
        help="session YAML to pull friendly CameraNode names from (matched on device_id)",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    specs = discover_all(
        load_name_map(args.names_from), zed_streams=tuple(args.zed_stream), zed_resolution=args.zed_resolution
    )
    if args.uvc_view:
        specs = expand_uvc_views(specs, args.uvc_view)
    if args.uvc_fps is not None:
        # Both the native tile and its emulated-mode tiles share one SharedUvcSource, so they must agree.
        specs = [replace(s, fps=args.uvc_fps) if s.kind in ("uvc", "uvc-mode") else s for s in specs]
    if not specs:
        raise SystemExit(
            "No cameras found. Check `lsusb`, and that this user can read /dev/video* "
            "(RealSense needs the librealsense udev rules; plain webcams need group `video`; "
            f"ZED: `lsusb | grep {STEREOLABS_USB_VENDOR}` and {_zed_sdk_hint()})."
        )

    workers = {
        spec.id: CameraWorker(spec, args.resolution, args.fps, args.jpeg_quality, args.idle_timeout)
        for spec in specs
    }

    res_txt = "native" if args.resolution is None else f"{args.resolution[0]}x{args.resolution[1]}"
    fps_txt = f"{args.fps} fps" + (f", UVC {args.uvc_fps} fps" if args.uvc_fps is not None else "")
    print(f"\nDiscovered {len(specs)} camera tile(s) (capture: {res_txt}, {fps_txt}):")
    for spec in specs:
        mode = spec.extra.get("mode", "")
        print(f"  - {spec.label:<28} [{spec.kind}] {spec.detail} {mode}")
    host_display = _lan_ip() if args.host == "0.0.0.0" else args.host
    # flush: stdout is block-buffered when redirected to a log, and uvicorn.run
    # never returns, so the banner would otherwise never appear.
    print(f"\n  Open  http://{host_display}:{args.port}\n", flush=True)

    import uvicorn  # noqa: PLC0415

    def _handle_signal(signum: int, _frame: Any) -> None:
        # Runs before uvicorn starts draining connections, so the streams are
        # already closed by the time it waits on them.
        logger.info("signal %d received, closing camera streams", signum)
        for worker in workers.values():
            worker.shutdown()
        signal.signal(signum, previous.get(signum, signal.SIG_DFL))

    previous = {
        sig: signal.signal(sig, _handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)
    }

    config = uvicorn.Config(
        build_app(workers),
        host=args.host,
        port=args.port,
        log_level="warning",
        timeout_graceful_shutdown=5,  # backstop if a client still will not drain
    )
    uvicorn.Server(config).run()


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera Viewer</title>
<style>
  :root {
    --bg: #0e1116; --panel: #161b22; --panel-2: #1c222c; --line: #2a323d;
    --text: #e6edf3; --muted: #8b98a5; --accent: #4a9eff; --ok: #3fb950;
    --warn: #d29922; --err: #f85149;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  header {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 12px 18px; background: var(--panel); border-bottom: 1px solid var(--line);
    position: sticky; top: 0; z-index: 10;
  }
  h1 { font-size: 15px; font-weight: 600; margin: 0; letter-spacing: .02em; }
  h1 span { color: var(--muted); font-weight: 400; margin-left: 8px; }
  .spacer { flex: 1; }
  .seg { display: flex; background: var(--panel-2); border: 1px solid var(--line); border-radius: 7px; overflow: hidden; }
  .seg button {
    background: none; border: 0; color: var(--muted); padding: 6px 14px;
    font: inherit; font-size: 13px; cursor: pointer;
  }
  .seg button.on { background: var(--accent); color: #fff; }
  .seg button:disabled { opacity: .4; cursor: not-allowed; }
  .ctl { display: flex; align-items: center; gap: 7px; color: var(--muted); font-size: 13px; }
  select, .btn {
    background: var(--panel-2); color: var(--text); border: 1px solid var(--line);
    border-radius: 6px; padding: 5px 10px; font: inherit; font-size: 13px; cursor: pointer;
  }
  .btn:hover, select:hover { border-color: var(--accent); }

  main { padding: 16px 18px 32px; }
  .grid { display: grid; gap: 14px; }

  .tile {
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    overflow: hidden; display: flex; flex-direction: column;
  }
  .tile.sel { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .tile-head {
    display: flex; align-items: center; gap: 9px; padding: 8px 11px;
    border-bottom: 1px solid var(--line); background: var(--panel-2);
  }
  .tile-head input[type=checkbox] { width: 15px; height: 15px; accent-color: var(--accent); cursor: pointer; flex: none; }
  .name { font-weight: 600; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .badge {
    flex: none; font-size: 10px; text-transform: uppercase; letter-spacing: .06em;
    padding: 2px 6px; border-radius: 4px; background: #243040; color: var(--muted);
  }
  .badge.realsense { background: #1d3a5c; color: #7cc0ff; }
  .badge.uvc { background: #3d2f1a; color: #e5b567; }
  .badge.zed, .badge.zed-stream { background: #1f3d2a; color: #7fe0a0; }
  .badge.uvc-mode { background: #2a2a3d; color: #b8a6ff; }
  .stats { margin-left: auto; display: flex; gap: 9px; align-items: center; font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); flex: none; }
  .dot.streaming { background: var(--ok); }
  .dot.opening { background: var(--warn); }
  .dot.error { background: var(--err); }

  .view { position: relative; background: #000; aspect-ratio: 4 / 3; display: flex; align-items: center; justify-content: center; }
  .view img { width: 100%; height: 100%; object-fit: contain; display: block; cursor: zoom-in; }
  .msg { color: var(--muted); font-size: 12px; text-align: center; padding: 20px; max-width: 90%; }
  .msg.error { color: var(--err); }
  .msg .retry {
    display: inline-block; margin-top: 12px; background: var(--panel-2); color: var(--text);
    border: 1px solid var(--line); border-radius: 6px; padding: 5px 12px; font: inherit;
    font-size: 12px; cursor: pointer;
  }
  .msg .retry:hover { border-color: var(--accent); }

  .tile-foot { display: flex; gap: 10px; align-items: center; padding: 6px 11px; font-size: 11px; color: var(--muted); border-top: 1px solid var(--line); }
  .tile-foot a { color: var(--muted); text-decoration: none; margin-left: auto; }
  .tile-foot a:hover { color: var(--accent); }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }

  .empty { color: var(--muted); text-align: center; padding: 80px 20px; }

  .zoom {
    position: fixed; inset: 0; background: rgba(0,0,0,.94); z-index: 100;
    display: flex; align-items: center; justify-content: center; cursor: zoom-out;
  }
  .zoom img { max-width: 96vw; max-height: 92vh; object-fit: contain; }
  .zoom .cap { position: absolute; top: 16px; left: 20px; font-size: 13px; color: var(--text); }
</style>
</head>
<body>
<header>
  <h1>Camera Viewer<span id="count"></span></h1>
  <div class="seg">
    <button id="mode-all" class="on">All</button>
    <button id="mode-cmp">Compare</button>
  </div>
  <div class="ctl">
    <label for="cols">Columns</label>
    <select id="cols">
      <option value="auto" selected>Auto</option>
      <option value="1">1</option><option value="2">2</option>
      <option value="3">3</option><option value="4">4</option>
    </select>
  </div>
  <div class="spacer"></div>
  <button class="btn" id="sel-all">Select all</button>
  <button class="btn" id="sel-none">Clear</button>
</header>

<main>
  <div id="grid" class="grid"></div>
  <div id="empty" class="empty" hidden></div>
</main>

<script>
const state = {
  cams: [],
  mode: "all",
  cols: "auto",
  selected: new Set(JSON.parse(localStorage.getItem("cam-selected") || "[]")),
  builtKey: null,
};

const $ = (id) => document.getElementById(id);
const grid = $("grid");

function saveSelection() {
  localStorage.setItem("cam-selected", JSON.stringify([...state.selected]));
}

function visibleIds() {
  const all = state.cams.map((c) => c.id);
  return state.mode === "all" ? all : all.filter((id) => state.selected.has(id));
}

function applyColumns(n) {
  grid.style.gridTemplateColumns =
    state.cols === "auto"
      ? `repeat(auto-fit, minmax(${n <= 2 ? 460 : 340}px, 1fr))`
      : `repeat(${state.cols}, minmax(0, 1fr))`;
}

/* Tiles are rebuilt only when the visible set or layout changes: recreating an
   <img> restarts its MJPEG connection, which re-opens the device server-side. */
function render() {
  const ids = visibleIds();
  const key = `${state.mode}|${state.cols}|${ids.join(",")}`;

  if (key !== state.builtKey) {
    state.builtKey = key;
    grid.textContent = "";
    applyColumns(ids.length);

    if (ids.length === 0) {
      $("empty").hidden = false;
      $("empty").textContent =
        state.cams.length === 0
          ? "No cameras discovered."
          : "No cameras ticked — tick some in All view, then switch back to Compare.";
    } else {
      $("empty").hidden = true;
      for (const id of ids) grid.appendChild(buildTile(state.cams.find((c) => c.id === id)));
    }
  }

  for (const cam of state.cams) updateTile(cam);

  $("count").textContent = `${state.cams.length} device${state.cams.length === 1 ? "" : "s"}` +
    (state.selected.size ? ` · ${state.selected.size} selected` : "");
  $("mode-cmp").disabled = state.selected.size === 0;
  $("mode-all").classList.toggle("on", state.mode === "all");
  $("mode-cmp").classList.toggle("on", state.mode === "compare");
}

function buildTile(cam) {
  const tile = document.createElement("div");
  tile.className = "tile" + (state.selected.has(cam.id) ? " sel" : "");
  tile.dataset.id = cam.id;

  const check = document.createElement("input");
  check.type = "checkbox";
  check.checked = state.selected.has(cam.id);
  check.title = "Include in Compare view";
  check.addEventListener("change", () => {
    check.checked ? state.selected.add(cam.id) : state.selected.delete(cam.id);
    saveSelection();
    render();
  });

  const head = document.createElement("div");
  head.className = "tile-head";
  head.append(check, el("div", "name", cam.label), el("span", `badge ${cam.kind}`, cam.kind));

  const stats = el("div", "stats");
  stats.append(el("span", "fps"), el("span", "res"), el("span", "dot"));
  head.append(stats);

  const view = el("div", "view");
  const img = document.createElement("img");
  img.alt = cam.label;
  img.src = `/stream/${encodeURIComponent(cam.id)}?t=${Date.now()}`;
  img.addEventListener("click", () => openZoom(cam));
  view.append(img, el("div", "msg"));

  const foot = el("div", "tile-foot");
  const code = document.createElement("code");
  code.textContent = cam.detail;
  const snap = document.createElement("a");
  snap.href = `/snapshot/${encodeURIComponent(cam.id)}`;
  snap.download = `${cam.label.replace(/\W+/g, "_")}.jpg`;
  snap.textContent = "snapshot ↓";
  foot.append(code, snap);

  tile.append(head, view, foot);
  return tile;
}

function updateTile(cam) {
  const tile = grid.querySelector(`.tile[data-id="${CSS.escape(cam.id)}"]`);
  if (!tile) return;

  tile.classList.toggle("sel", state.selected.has(cam.id));
  tile.querySelector(".dot").className = `dot ${cam.status}`;
  tile.querySelector(".fps").textContent = cam.status === "streaming" ? `${cam.fps.toFixed(1)} fps` : "";
  tile.querySelector(".res").textContent = cam.width ? `${cam.width}×${cam.height}` : "";

  const img = tile.querySelector("img");
  const msg = tile.querySelector(".msg");
  const failed = cam.status === "error";
  img.hidden = failed;
  msg.hidden = !failed && cam.status === "streaming";
  msg.className = "msg" + (failed ? " error" : "");

  if (failed) {
    // Only rebuild the message when the text changes, so the retry button does
    // not get torn out from under a click on every 1s poll.
    if (msg.dataset.err !== cam.error) {
      msg.dataset.err = cam.error || "";
      msg.textContent = cam.error || "camera error";
      const retry = el("button", "retry", "Retry");
      retry.addEventListener("click", () => {
        img.src = `/stream/${encodeURIComponent(cam.id)}?t=${Date.now()}`;
        msg.dataset.err = "";
        msg.textContent = "retrying…";
      });
      msg.append(document.createElement("br"), retry);
    }
  } else {
    msg.dataset.err = "";
    if (cam.status !== "streaming") msg.textContent = "opening…";
  }
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function openZoom(cam) {
  const overlay = el("div", "zoom");
  const img = document.createElement("img");
  img.src = `/stream/${encodeURIComponent(cam.id)}?t=${Date.now()}`;
  overlay.append(el("div", "cap", `${cam.label} — ${cam.detail}`), img);
  const close = () => { img.removeAttribute("src"); overlay.remove(); };
  overlay.addEventListener("click", close);
  document.addEventListener("keydown", function esc(e) {
    if (e.key === "Escape") { close(); document.removeEventListener("keydown", esc); }
  });
  document.body.append(overlay);
}

$("mode-all").addEventListener("click", () => { state.mode = "all"; render(); });
$("mode-cmp").addEventListener("click", () => { state.mode = "compare"; render(); });
$("cols").addEventListener("change", (e) => { state.cols = e.target.value; render(); });
$("sel-all").addEventListener("click", () => {
  state.cams.forEach((c) => state.selected.add(c.id));
  saveSelection(); render();
});
$("sel-none").addEventListener("click", () => {
  state.selected.clear(); saveSelection();
  if (state.mode === "compare") state.mode = "all";
  render();
});

async function poll() {
  try {
    const res = await fetch("/api/cameras", { cache: "no-store" });
    state.cams = await res.json();
    render();
  } catch (err) {
    console.warn("poll failed", err);
  }
}

poll();
setInterval(poll, 1000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
