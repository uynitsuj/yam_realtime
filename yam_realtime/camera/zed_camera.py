import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np
from pyzed import sl

from xdof.camera.camera import CameraData, CameraDriver

RESOLUTION_MAP = {
    "HD2K": sl.RESOLUTION.HD2K,  # only support 15 fps
    "HD1080": sl.RESOLUTION.HD1080,  # only support 30 fps
    "HD720": sl.RESOLUTION.HD720,
    "VGA": sl.RESOLUTION.VGA,
}


@dataclass
class ZedCamera(CameraDriver):
    """Zed RGB camera driver"""

    resolution: str = "HD1080"
    fps: int = 30
    zed_id: Optional[str] = None
    image_transfer_time_offset: float = 100  # unit: ms, see https://linear.app/xdof/issue/SWE-201/use-gpu-machine-for-xmi-station-to-use-autoexposure-for-zed-camera-for#comment-33d5c228
    concat_image: bool = True
    name: Optional[str] = None

    def __repr__(self) -> str:
        return f"ZedCamera(zed_id={self.zed_id!r}, name={self.name!r}, resolution={self.resolution}, fps={self.fps})"

    @classmethod
    def check_available_cameras(cls: type["ZedCamera"]) -> None:
        print(f"available cameras: {sl.Camera.get_device_list()}")
        logging.info("Checking available ZED cameras...")
        for c in sl.Camera.get_device_list():
            logging.info(f"Camera serial number: {c.serial_number}")

    def __post_init__(self):
        # Create a Camera object
        self.zed = sl.Camera()
        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        if self.zed_id is not None:
            init_params.set_from_serial_number(int(self.zed_id))
        init_params.camera_resolution = RESOLUTION_MAP[
            self.resolution
        ]  # Use HD720 opr HD1200 video mode, depending on camera type.
        init_params.camera_fps = self.fps  # Set fps at 30
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            err_code = "Camera Open : " + repr(err) + ". Exit program."
            logging.error(err_code)
            raise RuntimeError(err_code)

        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN)

        self.image_left = sl.Mat()
        self.image_right = sl.Mat()
        self.runtime_parameters = sl.RuntimeParameters()

        logging.info(f"Successfully opened ZED camera with parameters: {self}")

    def read(self) -> CameraData:
        result = {}
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
            # for zed, timestamp is the timestamp of the image arrives to the computer memory
            ts_image = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_microseconds()

            # Convert BGR to RGB for both left and right images
            left_rgb = cv2.cvtColor(self.image_left.get_data(), cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(self.image_right.get_data(), cv2.COLOR_BGR2RGB)

            if self.concat_image:
                result = CameraData(
                    images={"rgb": np.concatenate([left_rgb, right_rgb], axis=1)},
                    timestamp=ts_image - self.image_transfer_time_offset,
                )
            else:
                result = CameraData(
                    images={"left": left_rgb, "right": right_rgb},
                    timestamp=ts_image - self.image_transfer_time_offset,
                )
        else:
            logging.warning(f"{self}: Failed to grab image from ZED camera")
            result = CameraData(images={"left": None, "right": None}, timestamp=-1.0)  # type: ignore

        return result

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        """Read calibration data from the camera.

        Returns:
            dict: The calibration data.
        """
        # This is a placeholder implementation
        return {
            "left": {"K": None, "D": None},
            "right": {"K": None, "D": None},
        }

    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information.

        Returns:
            dict: Camera information.
        """
        return {
            "camera_type": "zed",
            "device_id": self.zed_id,
            "resolution": self.resolution,
            "fps": self.fps,
        }

    def stop(self) -> None:
        """Stop the camera."""
        self.zed.close()
        logging.info(f"Stopping ZED camera: {self}")


if __name__ == "__main__":
    from xdof.camera.camera import plot_camera_read

    ZedCamera.check_available_cameras()
    zed = ZedCamera()

    plot_camera_read(zed)
