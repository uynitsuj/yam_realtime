"""Intel RealSense camera driver (pyrealsense2) for the ``CameraDriver`` protocol.

Adapted from market42-bair (``market42/nodes/cameras/realsense.py``) to fit
robots_realtime's simpler driver interface — this class only implements
``read() / stop() / get_camera_info() / read_calibration_data_intrinsics()``;
recording and ZMQ publishing are handled upstream by ``CameraNode``.

D405 note: D400-series *depth* cameras without a dedicated RGB sensor expose
only stereo infrared streams. We detect that at setup and transparently fall
back to the left-infrared stream with ``rgb8`` format, so callers always get
an (H, W, 3) uint8 array on the ``"rgb"`` key of ``CameraData.images``.

``pyrealsense2`` is imported lazily so this module stays importable in envs
that don't have it installed (e.g. the CI test runner).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver

logger = logging.getLogger(__name__)


RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "VGA": (640, 480),
    "SVGA": (960, 600),
    "HD720": (1280, 720),
    "HD1080": (1920, 1080),
}


def _resolve_resolution(resolution: Any) -> Tuple[int, int]:
    if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
        return int(resolution[0]), int(resolution[1])
    if isinstance(resolution, str):
        if "x" in resolution:
            w, h = resolution.split("x", 1)
            return int(w), int(h)
        if resolution in RESOLUTION_PRESETS:
            return RESOLUTION_PRESETS[resolution]
    raise ValueError(
        f"Unknown resolution {resolution!r}. "
        f"Use 'WxH', (w, h), or a preset: {list(RESOLUTION_PRESETS.keys())}"
    )


@dataclass
class RealSenseCamera(CameraDriver):
    """Intel RealSense camera driver indexed by serial number.

    Args:
        device_id: RealSense serial number. ``None`` = first enumerated device.
        resolution: ``"WxH"``, preset name, or ``(w, h)`` tuple. Default ``"VGA"``.
        fps: Frame rate. Default 30.
        auto_exposure: Enable auto-exposure on the stereo / color sensor.
        manual_exposure_us: When ``auto_exposure=False``, set the color sensor's
            exposure to this value in microseconds (D405 default 33000μs).
            Ignored if ``auto_exposure=True``. Useful for locking exposure to a
            known-good value across cameras when AE drifts per-cam.
        manual_gain: When ``auto_exposure=False``, set the color sensor's gain
            (D405 range [16, 248], default 16). Ignored if ``auto_exposure=True``.
        manual_white_balance_k: If set (Kelvin, e.g. 4600), disables auto WB and
            locks the color temperature. ``None`` keeps auto WB.
        enable_depth: Also stream the depth channel. Emitted under
            ``CameraData.other_sensors["depth"]`` when available.
    """

    device_id: Optional[str] = None
    resolution: Any = "VGA"
    fps: int = 30
    auto_exposure: bool = True
    manual_exposure_us: Optional[float] = None
    manual_gain: Optional[float] = None
    manual_white_balance_k: Optional[float] = None
    enable_depth: bool = False

    # Populated in __post_init__; callers should not set these directly.
    intrinsic_data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            import pyrealsense2 as rs  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "RealSenseCamera requires `pyrealsense2`. Install it into this venv "
                "(e.g. `uv pip install pyrealsense2`)."
            ) from exc

        self._rs = rs
        self._width, self._height = _resolve_resolution(self.resolution)
        self._pipeline: Any = None
        self._profile: Any = None
        self._use_infrared: bool = False
        self._lock = threading.Lock()

        self._open_with_retries()
        self._configure_exposure()
        self.intrinsic_data = self._read_intrinsics()

        logger.info(
            "RealSenseCamera opened: serial=%s, %dx%d@%dfps, infrared=%s, depth=%s",
            self.device_id or "auto",
            self._width,
            self._height,
            self.fps,
            self._use_infrared,
            self.enable_depth,
        )

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _open_with_retries(self, max_retries: int = 5) -> None:
        """Open the pipeline; retry on transient 'device busy' errors."""
        rs = self._rs
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                # Detect whether the target device has a dedicated color sensor.
                # D405 does not — its stereo module exposes rgb8 infrared only.
                has_color = self._device_has_color_sensor()

                pipe = rs.pipeline()
                cfg = rs.config()
                if self.device_id is not None:
                    cfg.enable_device(self.device_id)

                stream = rs.stream.color if has_color else rs.stream.infrared
                cfg.enable_stream(stream, self._width, self._height, rs.format.rgb8, self.fps)
                if self.enable_depth:
                    cfg.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self.fps)

                self._profile = pipe.start(cfg)
                self._pipeline = pipe
                self._use_infrared = not has_color
                return
            except RuntimeError as exc:
                last_exc = exc
                if attempt < max_retries - 1 and "busy" in str(exc).lower():
                    wait_s = 0.5 * (attempt + 1)
                    logger.warning(
                        "RealSense device %s busy, retrying in %.1fs (%d/%d)",
                        self.device_id or "auto",
                        wait_s,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait_s)
                else:
                    raise
        if last_exc is not None:
            raise last_exc

    def _device_has_color_sensor(self) -> bool:
        """Whether the target device exposes ``rs.stream.color``.

        Checks the actual stream profiles published by every sensor on the
        device, not the sensor *name*. Required for the D405: that camera has
        a single ``'Stereo Module'`` sensor that hosts BOTH ``stream.infrared``
        AND ``stream.color`` profiles, so a name keyword check ("RGB" /
        "Color") falsely returns False and the driver falls back to streaming
        a monochrome IR frame as 3-channel — which is wildly off-distribution
        from the color RGB feed used during training.
        """
        rs = self._rs
        ctx = rs.context()
        for dev in ctx.query_devices():
            if self.device_id is None or dev.get_info(rs.camera_info.serial_number) == self.device_id:
                for sensor in dev.query_sensors():
                    for sp in sensor.get_stream_profiles():
                        try:
                            if sp.stream_type() == rs.stream.color:
                                return True
                        except Exception:
                            continue
                return False
        # Device not (yet) enumerable — assume color and let pipeline.start() surface a real error.
        return True

    def _configure_exposure(self) -> None:
        rs = self._rs
        if self._profile is None:
            return
        device = self._profile.get_device()
        for sensor in device.query_sensors():
            try:
                name = sensor.get_info(rs.camera_info.name)
            except Exception:
                continue
            if "stereo" in name.lower() or "rgb" in name.lower():
                # Auto-exposure first; some sensors require AE off before
                # accepting an explicit exposure value.
                try:
                    sensor.set_option(rs.option.enable_auto_exposure, bool(self.auto_exposure))
                except Exception as exc:
                    logger.debug("auto_exposure set failed on %s: %s", name, exc)
                if not self.auto_exposure:
                    if self.manual_exposure_us is not None and sensor.supports(rs.option.exposure):
                        try:
                            sensor.set_option(rs.option.exposure, float(self.manual_exposure_us))
                            logger.info(
                                "RealSenseCamera %s: manual exposure %.0fμs",
                                self.device_id or "auto", self.manual_exposure_us,
                            )
                        except Exception as exc:
                            logger.warning("manual exposure set failed on %s: %s", name, exc)
                    if self.manual_gain is not None and sensor.supports(rs.option.gain):
                        try:
                            sensor.set_option(rs.option.gain, float(self.manual_gain))
                            logger.info(
                                "RealSenseCamera %s: manual gain %.1f",
                                self.device_id or "auto", self.manual_gain,
                            )
                        except Exception as exc:
                            logger.warning("manual gain set failed on %s: %s", name, exc)
                if self.manual_white_balance_k is not None:
                    if sensor.supports(rs.option.enable_auto_white_balance):
                        try:
                            sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
                        except Exception as exc:
                            logger.debug("disable AWB failed on %s: %s", name, exc)
                    if sensor.supports(rs.option.white_balance):
                        try:
                            sensor.set_option(rs.option.white_balance, float(self.manual_white_balance_k))
                            logger.info(
                                "RealSenseCamera %s: manual white_balance %.0fK",
                                self.device_id or "auto", self.manual_white_balance_k,
                            )
                        except Exception as exc:
                            logger.warning("manual white_balance set failed on %s: %s", name, exc)
                break

    def _read_intrinsics(self) -> dict:
        rs = self._rs
        if self._profile is None:
            return {}
        try:
            # Find the active video stream (color or infrared)
            streams = self._profile.get_streams()
            video_stream = None
            for s in streams:
                if s.stream_type() in (rs.stream.color, rs.stream.infrared):
                    video_stream = s
                    break
            if video_stream is None:
                return {}
            intr = video_stream.as_video_stream_profile().get_intrinsics()
            return {
                "fx": intr.fx,
                "fy": intr.fy,
                "cx": intr.ppx,
                "cy": intr.ppy,
                "disto": list(intr.coeffs),
                "distortion_model": str(intr.model).replace("distortion.", ""),
                "width": self._width,
                "height": self._height,
            }
        except Exception as exc:
            logger.warning("RealSenseCamera intrinsics read failed: %s", exc)
            return {}

    # ------------------------------------------------------------------ #
    # CameraDriver protocol
    # ------------------------------------------------------------------ #

    def read(self) -> CameraData:
        if self._pipeline is None:
            raise RuntimeError("RealSenseCamera.read() called after stop() or before open")
        frames = self._pipeline.wait_for_frames(timeout_ms=1000)

        color_frame = frames.get_infrared_frame() if self._use_infrared else frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("RealSenseCamera: no color/infrared frame in pipeline output")

        rgb = np.asanyarray(color_frame.get_data())
        # Infrared RGB8 from D405 returns a 3-channel frame already; no conversion needed.

        ts_ms = float(frames.get_timestamp())  # camera-side timestamp in ms

        other: dict = {}
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                other["depth"] = np.asanyarray(depth_frame.get_data())

        return CameraData(
            images={"rgb": rgb},
            timestamp=ts_ms,
            other_sensors=other if other else None,
        )

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        return dict(self.intrinsic_data)

    def get_camera_info(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "width": self._width,
            "height": self._height,
            "fps": self.fps,
            "auto_exposure": self.auto_exposure,
            "infrared_fallback": self._use_infrared,
        }

    def stop(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception as exc:
                    logger.debug("RealSenseCamera.stop() failed: %s", exc)
                self._pipeline = None
                self._profile = None


# ---------------------------------------------------------------------- #
# Discovery helper — useful from scripts and the CameraNode registry
# ---------------------------------------------------------------------- #


def discover_realsense_cameras() -> list[dict[str, str]]:
    """Enumerate connected RealSense devices. Returns [] if pyrealsense2 missing."""
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
    except ImportError:
        return []

    cameras: list[dict[str, str]] = []
    try:
        for dev in rs.context().query_devices():
            cameras.append(
                {
                    "serial": dev.get_info(rs.camera_info.serial_number),
                    "name": dev.get_info(rs.camera_info.name),
                    "firmware": dev.get_info(rs.camera_info.firmware_version),
                }
            )
    except Exception as exc:
        logger.warning("RealSense discovery failed: %s", exc)
    return cameras
