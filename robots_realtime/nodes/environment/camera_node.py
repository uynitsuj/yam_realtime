"""CameraNode — wraps any CameraDriver and publishes frames to the bus.

Hardware timestamp from the driver (RealSense / ZED SDK) is used directly,
giving sub-millisecond accurate per-frame timestamps for post-hoc alignment.

poll_freq is None by default: the driver's blocking read() call paces the loop
at the hardware frame rate.  Set poll_freq only for drivers (e.g. bare OpenCV)
where read() returns immediately and you want an explicit rate cap.
"""

from __future__ import annotations

import time

from robots_realtime.nodes.base import Node, NodeRole
from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver


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
        **kwargs,
    ) -> None:
        super().__init__(name=name, writer=writer, **kwargs)
        self._driver = driver
        self.poll_freq = poll_freq

    def setup(self) -> None:
        if self._driver is None:
            raise RuntimeError(
                f"[{self.name}] CameraNode.driver is None — inject a camera driver before starting."
            )

    def step(self) -> None:
        data: CameraData = self._driver.read()

        # Hardware timestamp from driver (ms) → seconds
        ts = data.timestamp / 1000.0 if data.timestamp else time.time()

        for cam_name, frame in data.images.items():
            topic = "rgb" if cam_name == self.name else f"rgb/{cam_name}"
            self.publish(topic, {"frame": frame}, ts=ts)

        if data.imu_data is not None:
            imu = data.imu_data
            self.publish("imu", {
                "accel": imu.acceleration,
                "gyro": imu.gyroscope,
                "ts": imu.timestamp,
            }, ts=ts)

        if data.other_sensors:
            depth = data.other_sensors.get("depth")
            if depth is not None:
                self.publish("depth", {"frame": depth}, ts=ts)

    def cleanup(self) -> None:
        if hasattr(self._driver, "stop"):
            self._driver.stop()

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        # Camera driver is instantiated separately; this sets node params only.
        return {
            "name": params["name"],
            "poll_freq": params.get("poll_freq"),
        }
