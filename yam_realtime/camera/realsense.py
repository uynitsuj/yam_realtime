import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyrealsense2 as rs

from xdof.camera.camera import CameraData, CameraDriver, plot_camera_read
from xdof.utils.general import compare_versions


def get_device_ids() -> List[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    time.sleep(2)
    return device_ids


def get_device_info() -> Dict[str, str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    device_info = {}
    for dev in devices:
        serial_number = dev.get_info(rs.camera_info.serial_number)
        firmware_version = dev.get_info(rs.camera_info.firmware_version)
        device_info[serial_number] = firmware_version
    return device_info


@dataclass
class RealSenseCamera(CameraDriver):
    """RealSense RGB camera driver with exposure control."""

    device_id: Optional[str] = None
    resolution: Tuple[int, int] = (640, 480)  # Resolution as (width, height)
    fps: int = 30
    image_transfer_time_offset: Union[str, int] = (
        80  # ms typical transfer time, can change based on computer type and load
    )
    auto_exposure: bool = True  # Enable or disable auto exposure
    brightness: int = 10  # Brightness value, default is 0, range from -64 to 64. it controls the brightness when auto_exposure is True.
    exposure_value: Optional[int] = None  # Manual exposure value (if auto_exposure is False)
    name: Optional[str] = None

    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self.device_id!r}, name={self.name!r}, resolution={self.resolution}, fps={self.fps})"

    def __post_init__(self):
        if self.exposure_value is not None:
            logging.info(f"Exposure value has been set to {self.exposure_value}, disable auto exposure")
            self.auto_exposure = False
        # clip the brightness value to the range of -64 to 64
        self.brightness = max(-64, min(self.brightness, 64))
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            logging.error("No RealSense devices found")
            raise ValueError("No RealSense devices found")
        logging.info(f"RealSense devices: {devices}")

        if self.device_id is None:
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)
            self._pipeline = rs.pipeline()
            config = rs.config()
        else:
            assert self.device_id in get_device_info(), f"In {self}, Device {self.device_id} not found"

            # if the firmware version is too new, auto exposure will not work, see https://xdofai.slack.com/archives/C07BF68VCTZ/p1737007412638339?thread_ts=1736700561.077479&cid=C07BF68VCTZ
            firmware_version = get_device_info()[self.device_id]
            # Parse the firmware version and assert it is no newer than 5.13.0.50
            assert compare_versions(firmware_version, "5.13.0.50") != 1, (
                f"Firmware version {firmware_version} might be too new, the auto exposure might not work."
            )

            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.device_id)

        config.enable_stream(
            rs.stream.color,
            self.resolution[0],
            self.resolution[1],
            rs.format.rgb8,
            self.fps,
        )

        # Start the pipeline
        self.profile = self._pipeline.start(config)

        # Configure exposure settings
        self._configure_exposure()

    def _configure_exposure(self) -> None:
        """Configure the exposure settings for the RealSense camera (D405 compatible)."""
        device = self.profile.get_device()
        sensors = device.query_sensors()

        # Ensure we are working with the Stereo Module
        stereo_sensor = None
        for sensor in sensors:
            if sensor.get_info(rs.camera_info.name) == "Stereo Module":
                stereo_sensor = sensor
                break

        if stereo_sensor is None:
            raise ValueError("No Stereo Module sensor found on the device")

        if self.auto_exposure:
            stereo_sensor.set_option(rs.option.enable_auto_exposure, True)
            stereo_sensor.set_option(rs.option.brightness, self.brightness)
            logging.info(f"Auto exposure enabled, brightness set to {self.brightness}")
        else:
            if self.exposure_value is None:
                raise ValueError("Exposure value must be set when auto_exposure is False")
            stereo_sensor.set_option(rs.option.enable_auto_exposure, False)
            stereo_sensor.set_option(rs.option.exposure, self.exposure_value)
            logging.info(f"Manual exposure set to {self.exposure_value}")

    def read(self) -> CameraData:
        result = {}
        # todo: find a better way to handle this, because when I unplug the camera, it will raise an error after around 10 seconds
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError as e:
            mess = f"{self}: {e}"
            logging.error(mess)
            raise RuntimeError(mess) from e

        color_frame = frames.get_color_frame()
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())  # Convert to NumPy array
            ts_image = frames.get_timestamp()
            # for what do those tiemstamps mean, checkout realsense_cam_latency_test.ipynb
            result = CameraData(
                images={"rgb": color_image},
                timestamp=ts_image - self.image_transfer_time_offset,
            )
        else:
            logging.warning("Failed to grab image from RealSense camera")
            result = CameraData(images={"rgb": None}, timestamp=-1.0)  # type: ignore

        return result

    def set_exposure(self, auto_exposure: bool, exposure_value: Optional[int] = None) -> None:
        """Update the exposure settings during runtime."""
        self.auto_exposure = auto_exposure
        self.exposure_value = exposure_value
        self._configure_exposure()

    def stop(self) -> None:
        self._pipeline.stop()

    def get_camera_info(self) -> dict:
        info = {}
        info.update(
            {
                "camera_type": "realsense",
                "device_id": self.device_id,
                "width": self.resolution[0],
                "height": self.resolution[1],
                "fps": self.fps,
                "auto_exposure": self.auto_exposure,
                "brightness": self.brightness,
                "exposure_value": self.exposure_value,
            }
        )
        return info

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Calibration data reading is not implemented for {self}")


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--device_id", type=str, default=None)
    args.add_argument("--info-only", action="store_true", help="Only print device info and exit")

    args = args.parse_args()
    device_info = get_device_info()
    print("Device INFO:")
    print(device_info)

    # Exit early if info-only flag is set
    if args.info_only:
        import sys

        sys.exit(0)

    realsense = RealSenseCamera(device_id=args.device_id)
    plot_camera_read(realsense)
