import logging
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
from pyzed import sl

from robots_realtime.sensors.cameras.camera import (
    CameraData,
    CameraDriver,
)

RESOLUTION_MAP = {
    "HD2K": sl.RESOLUTION.HD2K,  # only support 15 fps
    "HD1200": sl.RESOLUTION.HD1200,
    "HD1080": sl.RESOLUTION.HD1080,  # only support 30 fps
    "HD720": sl.RESOLUTION.HD720,
    "VGA": sl.RESOLUTION.VGA,
    "SVGA": sl.RESOLUTION.SVGA,
}
RESOLUTION_TO_VALID_FPS = {
    "HD2K": [15],
    "HD1200": [15, 30, 60],
    "HD1080": [15, 30, 60],
    "HD720": [15, 30, 60],
    "VGA": [15, 30, 60, 100],
}
# used for logging and saved in metadata.
RESOLUTION_SIZE_MAP = {
    "HD2K": (2560, 1440),
    "HD1200": (1920, 1200),
    "HD1080": (1920, 1080),
    "HD720": (1280, 720),
    "VGA": (640, 480),
    "SVGA": (960, 600),
}

"""
Zed X supported resolution and fps:
1200p: 15, 30, 60
1080p: 15, 30, 60
720p: 15, 30, 60
600p(SVGA): 15, 30, 60, 120

Zed 2 supported resolution and fps:
2k: 15
1080p: 15, 30
720p: 15, 30, 60
376p (VGA): 15, 30, 60, 100
"""


class STEREO_OR_MONO(Enum):
    STEREO = "stereo"
    MONO = "mono"


@dataclass
class ZedCamera(CameraDriver):
    """Zed RGB camera driver"""

    resolution: str = "SVGA"
    fps: int = 60
    device_id: str | None = None
    image_transfer_time_offset_ms: float = 70  # unit: ms,
    concat_image: bool = False  # if True, concat the left and right image, it might slow down the read frequency.
    return_right_image: bool = False
    name: str | None = None
    enable_depth: bool = False

    def __repr__(self) -> str:
        return f"ZedCamera(device_id={self.device_id!r}, name={self.name!r}, resolution={self.resolution}, fps={self.fps})"

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
        if self.device_id:
            init_params.set_from_serial_number(int(self.device_id))
        init_params.camera_resolution = RESOLUTION_MAP[self.resolution]
        self.width, self.height = RESOLUTION_SIZE_MAP[self.resolution]
        # Use HD720 opr HD1200 video mode, depending on camera type.
        # if self.fps not in RESOLUTION_TO_VALID_FPS[self.resolution]:
        #     raise ValueError(f"Invalid fps for resolution {self.resolution}. Valid fps are {RESOLUTION_TO_VALID_FPS[self.resolution]}")
        init_params.camera_fps = self.fps  # Set fps at 30
        if self.enable_depth:
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
            init_params.coordinate_units = sl.UNIT.METER
        else:
            init_params.depth_mode = sl.DEPTH_MODE.NONE

        # init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            err_code = "Camera Open : " + repr(err) + ". Exit program."
            logging.error(err_code)
            raise RuntimeError(err_code)

        logging.info(f"Zed camera opened with device id {self.device_id}")

        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN)

        self.image_left = sl.Mat()
        self.image_right = sl.Mat()
        if self.enable_depth:
            self.depth_map = sl.Mat()

        self.camera_info = self.zed.get_camera_information()
        self.runtime_parameters = sl.RuntimeParameters()
        self.camera_type = self.camera_info.camera_model.name

        self.intrinsic_data = {
            "left": self._load_intrinsic_data("left"),
            "right": self._load_intrinsic_data("right"),
        }

        # Extract and save camera information once
        self.serial_number: int = self.camera_info.serial_number if self.device_id is None else int(self.device_id)

        logging.info(f"Successfully opened ZED camera with parameters: {self}")

    def _load_intrinsic_data(self, camera_side: str, raw: bool = False) -> dict:
        """Load camera calibration parameters for specified camera side (left/right) and return dict."""
        if raw:
            calib_params = self.camera_info.camera_configuration.calibration_parameters_raw
        else:
            calib_params = self.camera_info.camera_configuration.calibration_parameters

        cam = getattr(calib_params, f"{camera_side}_cam")
        intrinsics_matrix = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
        return {
            "intrinsics_matrix": intrinsics_matrix,
            "distortion_coefficients": list(cam.disto),
            "distortion_model": "zed_rectified",  # Zed gives rectified distortion coefficients
        }

    def read_depth(self) -> np.ndarray:
        """Read only depth map from ZED camera.

        Returns:
            np.ndarray: The depth map.
        """
        assert self.enable_depth, "Depth is not enabled"
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
        else:
            logging.warning(f"{self}: Failed to grab depth map from ZED camera")
            return np.zeros((0, 0))

        return self.depth_map.get_data()

    def read(self) -> CameraData:
        start_time = time.time()
        result = {}
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            if self.return_right_image:
                self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
            # for zed, timestamp is the timestamp of the image arrives to the computer memory
            ts_image = int(self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_microseconds() / 1000)

            left_bgra = self.image_left.get_data()
            if self.return_right_image:
                right_bgra = self.image_right.get_data()
            else:
                right_bgra = None
            # check if left or right is all black, if so raise runtime error. see SWE-381.
            # Downsample the image to 1% of the original size to speed up the check. On some station, this checker might take a longer time.
            if np.all(left_bgra[::10, ::10, :3] < 8):
                raise RuntimeError(f"Zed camera {self.device_id} left camera is all black")
            if self.return_right_image and np.all(right_bgra[::10, ::10, :3] < 8):
                raise RuntimeError(f"Zed camera {self.device_id} right camera is all black")

            # np.ascontiguousarray will slow down the read function, but it can speed up video saving.
            # Video saving is the current bottleneck, to speed it up, we make this trade-off.

            if self.concat_image:
                left_rgb = np.ascontiguousarray(left_bgra[:, :, :3][:, :, ::-1])
                if not self.return_right_image:
                    raise RuntimeError("concat_image is True, but return_right_image is False")
                right_rgb = np.ascontiguousarray(right_bgra[:, :, :3][:, :, ::-1])
                result = CameraData(
                    images={"rgb": np.concatenate([left_rgb, right_rgb], axis=1)},
                    timestamp=ts_image - self.image_transfer_time_offset_ms,
                )
            else:
                left_rgb = np.ascontiguousarray(left_bgra[:, :, :3][:, :, ::-1])
                if self.return_right_image:
                    right_rgb = np.ascontiguousarray(right_bgra[:, :, :3][:, :, ::-1])
                    result = CameraData(
                        images={"left_rgb": left_rgb, "right_rgb": right_rgb},
                        timestamp=ts_image - self.image_transfer_time_offset_ms,
                    )
                else:
                    result = CameraData(
                        images={"left_rgb": left_rgb},
                        timestamp=ts_image - self.image_transfer_time_offset_ms,
                    )
                
            if self.enable_depth:
                self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
                depth_map_data = self.depth_map.get_data()
                result.depth_data = np.ascontiguousarray(depth_map_data)
        else:
            logging.warning(f"{self}: Failed to grab image from ZED camera")
            # Return empty images on failure to maintain type consistency
            if self.concat_image:
                result = CameraData(images={"rgb": None}, timestamp=-1.0)
            else:
                result = CameraData(images={"left_rgb": None, "right_rgb": None}, timestamp=-1.0)  # type: ignore

        end_time = time.time()
        print(f"time taken to read camera data: {(end_time - start_time) * 1000} ms")
        return result

    def read_calibration_data_intrinsics(self) -> dict:
        return self.intrinsic_data

    def get_camera_info(self) -> dict:
        """Get camera information as a dict instance."""
        # Prepare intrinsic data in the required format
        info = {
            "camera_type": "zed",
            "device_id": str(self.device_id),
            "width": self.width,
            "height": self.height,
            "polling_fps": self.fps,
            "name": self.name if self.name is not None else "zed_camera",
            "image_transfer_time_offset_ms": self.image_transfer_time_offset_ms,
            "intrinsics": self.intrinsic_data,
            "concat_image": self.concat_image,
        }
        return info

    def stop(self) -> None:
        """Stop the camera."""
        self.zed.close()
        logging.info(f"Stopping ZED camera: {self}")


if __name__ == "__main__":
    ZedCamera.check_available_cameras()
    # hd720p mode is supported by both zed x and zed 2i.
    zed = ZedCamera(resolution="HD720")
    print(zed.get_camera_info().model_dump())
    t_start = time.time()
    while True:
        data = zed.read()
        print(f"frequency: {1 / (time.time() - t_start)}")
        t_start = time.time()

    # plot_camera_read(zed)
