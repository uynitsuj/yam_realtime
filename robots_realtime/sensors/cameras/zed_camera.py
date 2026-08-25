"""Stereolabs ZED camera driver (pyzed) for the ``CameraDriver`` protocol.

Supported inputs (pick one):
    device_id     USB ZED / ZED Mini / ZED 2 / ZED 2i attached to this machine,
                  selected by serial number. ``None`` = first enumerated camera.
    stream_ip     A ZED SDK *network stream* (``sl.InputType.set_from_stream``),
                  e.g. a ZED X on a Jetson / ZED Box running the SDK streaming
                  sender. ZED X is GMSL2-only and cannot be plugged into an x86
                  box directly -- streaming is how it reaches this driver.
    svo_path      An SVO/SVO2 recording, for offline pipeline tests.

Frame naming: the left (or mono) frame is published under ``image_key``. The
historical default is ``"left_rgb"``; OpenPI policies flatten camera messages
to ``"<obs_key>-images-<image_key>"`` and expect ``"<obs_key>-images-rgb"``, so
policy-deployment configs set ``image_key: rgb`` (same key RealSense/OpenCV
drivers use, and same on-disk name ``<node>-images-rgb.mp4``).

Requires the ZED SDK (``/usr/local/zed``) plus the matching ``pyzed`` wheel
(``uv sync --extra sensors``). See ``scripts/setup_zed.sh``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import viser.transforms as vtf
import yaml

from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver

try:
    from pyzed import sl
except ImportError as exc:  # pragma: no cover - depends on the host install
    raise ImportError(
        "pyzed (ZED SDK Python API) could not be imported. Either the `sensors` extra is not "
        "installed (`uv sync --extra sensors`), or the SDK shared libraries under /usr/local/zed "
        "are missing / not readable by this user (fix: `sudo chmod -R o+rX /usr/local/zed`), or the "
        "pyzed wheel version does not match the installed SDK. Run scripts/setup_zed.sh for a "
        f"guided check. Original error: {exc}"
    ) from exc

logger = logging.getLogger(__name__)

RESOLUTION_MAP = {
    "AUTO": sl.RESOLUTION.AUTO,
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
# Nominal sizes, used only as a fallback when the SDK does not report the real
# resolution (it always does after a successful open()).
RESOLUTION_SIZE_MAP = {
    "AUTO": (0, 0),
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

Zed 2 / 2i (USB) supported resolution and fps:
2k: 15
1080p: 15, 30
720p: 15, 30, 60
376p (VGA): 15, 30, 60, 100
"""

# ZED SDK streaming sender default port.
DEFAULT_STREAM_PORT = 30000


class STEREO_OR_MONO(Enum):
    STEREO = "stereo"
    MONO = "mono"


@dataclass
class ZedCamera(CameraDriver):
    """Zed RGB camera driver"""

    resolution: str = "SVGA"
    fps: int = 60
    device_id: str | None = None
    # ZED SDK network stream source ("host" or "host:port"); mutually exclusive with device_id/svo_path.
    stream_ip: str | None = None
    stream_port: int = DEFAULT_STREAM_PORT
    # SVO/SVO2 file to play back instead of a live camera.
    svo_path: str | None = None
    image_transfer_time_offset_ms: float = 70  # unit: ms,
    concat_image: bool = False  # if True, concat the left and right image, it might slow down the read frequency.
    return_right_image: bool = False
    # Key the left/mono frame is published under. Use "rgb" for OpenPI policy configs.
    image_key: str = "left_rgb"
    right_image_key: str = "right_rgb"
    name: str | None = None
    enable_depth: bool = False
    extrinsics_file: str | None = None  # path to a camera extrinsics YAML (see configs/camera_extrinsics/)
    # Consecutive grab() failures tolerated before read() raises. A raise takes the CameraNode down
    # loudly instead of silently publishing None frames into the recording + the policy.
    grab_retries: int = 5
    # Raise if a frame is (near) all black -- catches a dead sensor or capped lens at startup.
    check_black_frames: bool = True
    # Extra ZED SDK verbosity (prints to stdout from the C++ side).
    sdk_verbose: bool = False

    def __repr__(self) -> str:
        source = self._source_description()
        return f"ZedCamera({source}, name={self.name!r}, resolution={self.resolution}, fps={self.fps})"

    def _source_description(self) -> str:
        if self.svo_path:
            return f"svo={self.svo_path!r}"
        if self.stream_ip:
            host, port = self._parse_stream_endpoint()
            return f"stream={host}:{port}"
        return f"device_id={self.device_id!r}"

    def _parse_stream_endpoint(self) -> tuple[str, int]:
        assert self.stream_ip is not None
        host, sep, port = self.stream_ip.partition(":")
        return host, int(port) if sep else int(self.stream_port)

    @classmethod
    def list_devices(cls) -> list[dict]:
        """Enumerate locally attached ZED cameras (USB / GMSL) as plain dicts."""
        devices = []
        for dev in sl.Camera.get_device_list():
            devices.append(
                {
                    "serial_number": int(dev.serial_number),
                    "model": str(dev.camera_model).replace("CAMERA_MODEL.", ""),
                    "state": str(dev.camera_state).replace("CAMERA_STATE.", ""),
                    "id": int(dev.id),
                    "path": str(getattr(dev, "path", "")),
                }
            )
        return devices

    @classmethod
    def list_streams(cls) -> list[dict]:
        """Enumerate ZED SDK streaming senders visible on the local network."""
        streams = []
        for props in sl.Camera.get_streaming_device_list():
            streams.append(
                {
                    "ip": str(props.ip),
                    "port": int(props.port),
                    "serial_number": int(props.serial_number),
                    "current_bitrate": int(props.current_bitrate),
                    "codec": str(props.codec).replace("STREAMING_CODEC.", ""),
                }
            )
        return streams

    @classmethod
    def check_available_cameras(cls: type["ZedCamera"]) -> None:
        devices = cls.list_devices()
        print(f"available cameras: {devices}")
        logger.info("Checking available ZED cameras...")
        for dev in devices:
            logger.info("Camera serial number: %s (%s, %s)", dev["serial_number"], dev["model"], dev["state"])
        streams = cls.list_streams()
        if streams:
            logger.info("ZED SDK streams on the network: %s", streams)

    def _build_init_params(self) -> "sl.InitParameters":
        init_params = sl.InitParameters()
        sources = [bool(self.device_id), bool(self.stream_ip), bool(self.svo_path)]
        if sum(sources) > 1:
            raise ValueError(f"{self}: device_id, stream_ip and svo_path are mutually exclusive")

        if self.svo_path:
            init_params.set_from_svo_file(str(self.svo_path))
            init_params.svo_real_time_mode = True
        elif self.stream_ip:
            host, port = self._parse_stream_endpoint()
            init_params.set_from_stream(host, port)
        elif self.device_id:
            init_params.set_from_serial_number(int(self.device_id))

        if self.resolution not in RESOLUTION_MAP:
            raise ValueError(f"Unknown resolution {self.resolution!r}. Options: {list(RESOLUTION_MAP)}")
        init_params.camera_resolution = RESOLUTION_MAP[self.resolution]
        # Resolution/fps are properties of the sender for streams; the SDK ignores them there.
        init_params.camera_fps = self.fps
        init_params.sdk_verbose = 1 if self.sdk_verbose else 0
        init_params.coordinate_units = sl.UNIT.METER
        if self.enable_depth:
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        else:
            init_params.depth_mode = sl.DEPTH_MODE.NONE
        return init_params

    def __post_init__(self):
        # Create a Camera object
        self.zed = sl.Camera()
        init_params = self._build_init_params()
        self.width, self.height = RESOLUTION_SIZE_MAP[self.resolution]

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            err_code = f"Camera Open ({self._source_description()}): {err!r}. Exit program."
            logger.error(err_code)
            raise RuntimeError(err_code)

        logger.info("Zed camera opened (%s)", self._source_description())

        # Streams / SVO files are read-only w.r.t. sensor settings.
        if not (self.stream_ip or self.svo_path):
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN)

        self.image_left = sl.Mat()
        self.image_right = sl.Mat()
        if self.enable_depth:
            self.depth_map = sl.Mat()

        self.camera_info = self.zed.get_camera_information()
        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.confidence_threshold = 75
        self.camera_type = self.camera_info.camera_model.name

        # Trust what the SDK reports over the nominal table (AUTO, streams and SVO files
        # only know their real size after open()).
        cam_res = self.camera_info.camera_configuration.resolution
        if cam_res.width > 0 and cam_res.height > 0:
            self.width, self.height = int(cam_res.width), int(cam_res.height)
        reported_fps = float(self.camera_info.camera_configuration.fps)
        if reported_fps > 0 and abs(reported_fps - self.fps) > 0.5:
            logger.warning("%s: requested %s fps but camera runs at %s fps", self, self.fps, reported_fps)
            self.fps = int(round(reported_fps))

        self.intrinsic_data = {
            "left": self._load_intrinsic_data("left"),
            "right": self._load_intrinsic_data("right"),
        }

        # Extract and save camera information once
        self.serial_number: int = (
            int(self.device_id) if self.device_id else int(self.camera_info.serial_number)
        )

        self.extrinsics: dict | None = self._load_extrinsics() if self.extrinsics_file else None
        self._consecutive_failures = 0

        logger.info("Successfully opened ZED camera with parameters: %s (%sx%s)", self, self.width, self.height)

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

    def _load_extrinsics(self) -> dict | None:
        """Load camera-to-world extrinsics from a YAML file.

        The YAML must contain:
            position: [x, y, z]        # metres, in world/base frame
            rpy_radians: [roll, pitch, yaw]

        Returns a dict with pre-computed ``position`` (np.ndarray), ``wxyz``
        (quaternion, np.ndarray) and ``pose_mat`` (4x4 SE3, np.ndarray) so
        callers never need to redo the trigonometry.
        """
        path = Path(self.extrinsics_file)
        if not path.exists():
            logger.warning("ZedCamera: extrinsics file not found: %s. Extrinsics will be unavailable.", path)
            return None

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        position = np.array(data["position"], dtype=np.float64)
        rpy = data["rpy_radians"]
        wxyz = vtf.SO3.from_rpy_radians(*rpy).wxyz
        pose_mat = vtf.SE3(wxyz_xyz=np.concatenate([wxyz, position])).as_matrix()

        logger.info("ZedCamera: loaded extrinsics from %s", path)
        return {"position": position, "wxyz": wxyz, "pose_mat": pose_mat}

    def read_depth(self) -> np.ndarray:
        """Read only depth map from ZED camera.

        Returns:
            np.ndarray: The depth map.
        """
        assert self.enable_depth, "Depth is not enabled"
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
        else:
            logger.warning("%s: Failed to grab depth map from ZED camera", self)
            return np.zeros((0, 0))

        return self.depth_map.get_data()

    def _grab(self) -> None:
        """Block until a new frame is available; raise after ``grab_retries`` consecutive failures."""
        while True:
            err = self.zed.grab(self.runtime_parameters)
            if err == sl.ERROR_CODE.SUCCESS:
                self._consecutive_failures = 0
                return
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                raise RuntimeError(f"{self}: end of SVO file reached")
            self._consecutive_failures += 1
            logger.warning(
                "%s: grab failed (%r), attempt %d/%d", self, err, self._consecutive_failures, self.grab_retries
            )
            if self._consecutive_failures >= self.grab_retries:
                raise RuntimeError(f"{self}: {self.grab_retries} consecutive grab failures, last error {err!r}")
            time.sleep(0.01)

    def _assert_not_black(self, bgra: np.ndarray, side: str) -> None:
        # Downsample to 1% of the pixels: on some stations a full-frame check costs real time.
        if self.check_black_frames and np.all(bgra[::10, ::10, :3] < 8):
            raise RuntimeError(f"Zed camera {self._source_description()} {side} image is all black")

    def read(self) -> CameraData:
        self._grab()
        self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
        if self.return_right_image:
            self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
        # for zed, timestamp is the timestamp of the image arrives to the computer memory
        ts_image = int(self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_microseconds() / 1000)
        timestamp = ts_image - self.image_transfer_time_offset_ms

        left_bgra = self.image_left.get_data()
        self._assert_not_black(left_bgra, "left")
        # np.ascontiguousarray slows read() slightly but speeds up video encoding, which is the
        # real bottleneck on recording stations.
        left_rgb = np.ascontiguousarray(left_bgra[:, :, :3][:, :, ::-1])

        right_rgb = None
        if self.return_right_image:
            right_bgra = self.image_right.get_data()
            self._assert_not_black(right_bgra, "right")
            right_rgb = np.ascontiguousarray(right_bgra[:, :, :3][:, :, ::-1])

        if self.concat_image:
            if right_rgb is None:
                raise RuntimeError("concat_image is True, but return_right_image is False")
            result = CameraData(images={"rgb": np.concatenate([left_rgb, right_rgb], axis=1)}, timestamp=timestamp)
        elif right_rgb is not None:
            result = CameraData(
                images={self.image_key: left_rgb, self.right_image_key: right_rgb}, timestamp=timestamp
            )
        else:
            result = CameraData(images={self.image_key: left_rgb}, timestamp=timestamp)

        if self.enable_depth:
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            result.other_sensors = {"depth": np.ascontiguousarray(self.depth_map.get_data())}

        return result

    def read_calibration_data_intrinsics(self) -> dict:
        return self.intrinsic_data

    def get_camera_info(self) -> dict:
        """Get camera information as a dict instance."""
        info = {
            "camera_type": "zed",
            "camera_model": self.camera_type,
            "device_id": str(self.serial_number),
            "source": self._source_description(),
            "width": self.width,
            "height": self.height,
            "polling_fps": self.fps,
            "name": self.name if self.name is not None else "zed_camera",
            "image_transfer_time_offset_ms": self.image_transfer_time_offset_ms,
            "intrinsics": self.intrinsic_data,
            "concat_image": self.concat_image,
            "image_key": self.image_key,
        }
        return info

    def stop(self) -> None:
        """Stop the camera."""
        self.zed.close()
        logger.info("Stopping ZED camera: %s", self)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ZedCamera.check_available_cameras()
    # hd720p mode is supported by both zed x and zed 2i.
    zed = ZedCamera(resolution="HD720", fps=30, image_key="rgb")
    print(zed.get_camera_info())
    t_start = time.time()
    while True:
        data = zed.read()
        print(f"frequency: {1 / (time.time() - t_start):.1f} Hz, shape={data.images['rgb'].shape}")
        t_start = time.time()
