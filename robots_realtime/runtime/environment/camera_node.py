"""CameraNode — wraps any CameraDriver and publishes frames to the bus.

Hardware timestamp from the driver (RealSense / ZED SDK) is used directly,
giving sub-millisecond accurate per-frame timestamps for post-hoc alignment.

poll_freq is None by default: the driver's blocking read() call paces the loop
at the hardware frame rate.  Set poll_freq only for drivers (e.g. bare OpenCV)
where read() returns immediately and you want an explicit rate cap.
"""

from __future__ import annotations

import importlib
import time

from robots_realtime.runtime.node import Node, NodeRole
from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver


_CAMERA_DRIVER_REGISTRY: dict[str, str] = {
    "ZedCamera":        "robots_realtime.sensors.cameras.zed_camera:ZedCamera",
    "OpenCVCamera":     "robots_realtime.sensors.cameras.opencv_camera:OpenCVCamera",
    "RealSenseCamera":  "robots_realtime.sensors.cameras.realsense_camera:RealSenseCamera",
}

_NODE_ONLY_KEYS = {"name", "type", "poll_freq"}


def _instantiate_camera_driver(spec: dict) -> CameraDriver:
    """Instantiate a camera driver from a spec dict (driver name + kwargs)."""
    driver_name: str = spec["driver"]
    if driver_name in _CAMERA_DRIVER_REGISTRY:
        ref = _CAMERA_DRIVER_REGISTRY[driver_name]
    elif ":" in driver_name:
        ref = driver_name
    else:
        raise ValueError(
            f"Unknown camera driver '{driver_name}'. "
            f"Known drivers: {list(_CAMERA_DRIVER_REGISTRY.keys())}"
        )
    module_path, cls_name = ref.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    kwargs = {k: v for k, v in spec.items() if k != "driver"}
    return getattr(mod, cls_name)(**kwargs)


class CameraNode(Node):
    """Publish camera frames from any CameraDriver onto the bus.

    Published topics:
        ``{name}/rgb``    — dict with ``frame`` (H,W,3 uint8) and ``ts`` float
        ``{name}/info``   — camera info dict (published once on setup)

    Optionally also publishes:
        ``{name}/depth``  — if driver provides it in CameraData.other_sensors
        ``{name}/imu``    — if driver provides IMUData

    Args:
        driver:    Camera driver implementing read() -> CameraData.
        name:      Node name on the bus.
        poll_freq: Optional rate cap for drivers where read() is non-blocking.
        writer:    Optional Writer injected at construction for recording.
    """

    role = NodeRole.SENSOR
    published_topics: list[str] = ["rgb"]
    poll_freq: float | None = None

    def __init__(
        self,
        driver: CameraDriver | None = None,
        name: str = "camera",
        poll_freq: float | None = None,
        writer=None,
        _driver_spec: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, writer=writer, **kwargs)
        self._driver = driver
        self._driver_spec = _driver_spec
        self.poll_freq = poll_freq

    def setup(self) -> None:
        if self._driver is None:
            if self._driver_spec is None:
                raise RuntimeError(
                    f"[{self.name}] CameraNode.driver is None — inject a camera driver before starting."
                )
            self._driver = _instantiate_camera_driver(self._driver_spec)

    def step(self) -> None:
        data: CameraData = self._driver.read()

        # Hardware timestamp from driver (ms) → seconds
        ts = data.timestamp / 1000.0 if data.timestamp else time.time()

        # Publish one consolidated message that matches the format agents and
        # visualization code already expect (same shape as old CameraNode._get_latest_data).
        msg: dict = {"images": data.images, "timestamp": ts}

        # Depth: support both other_sensors["depth"] (standard) and the
        # dynamic depth_data attribute that ZedCamera sets directly.
        depth = (data.other_sensors or {}).get("depth") or getattr(data, "depth_data", None)
        if depth is not None:
            msg["depth_data"] = depth

        # Intrinsics and extrinsics from the driver if available.
        intrinsics = getattr(self._driver, "intrinsic_data", None)
        if intrinsics is not None:
            msg["intrinsics"] = intrinsics
        extrinsics = getattr(self._driver, "extrinsics", None)
        if extrinsics is not None:
            msg["extrinsics"] = extrinsics

        self.publish("rgb", msg, ts=ts)

        if data.imu_data is not None:
            imu = data.imu_data
            self.publish("imu", {
                "accel": imu.acceleration,
                "gyro": imu.gyroscope,
                "ts": imu.timestamp,
            }, ts=ts)

    def cleanup(self) -> None:
        if hasattr(self._driver, "stop"):
            self._driver.stop()

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        kwargs: dict = {
            "name": params["name"],
            "poll_freq": params.get("poll_freq"),
        }
        if "driver" in params:
            driver_kwargs = {k: v for k, v in params.items() if k not in _NODE_ONLY_KEYS}
            kwargs["_driver_spec"] = driver_kwargs
        return kwargs
