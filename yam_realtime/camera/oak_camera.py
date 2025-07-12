from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import depthai as dai
import numpy as np
import tyro

from xdof.camera.camera import CameraData, CameraDriver

"""
The OAK camera's 'device_id' should be set to the camera's MXID, which can be obtained by running:

python -c 'import depthai as dai; [print(f"Name: {d.name}, MXID: {d.mxid}, State: {d.state}") for d in dai.Device.getAllConnectedDevices()]'
"""


@dataclass
class OakCamera(CameraDriver):
    device_id: Optional[str] = None  # MXID of the OAK camera
    camera_type: str = "oak_camera"
    resolution: Tuple[int, int] = (800, 480)  # (width, height)
    fps: int = 60
    image_transfer_time_offset: float = 130.0  # ms, could vary by system
    name: Optional[str] = None

    def __repr__(self) -> str:
        return f"OakCamera(resolution={self.resolution}, fps={self.fps}, name={self.name!r})"

    def __post_init__(self):
        if self.device_id is None:
            self.device_id = self.name or f"oak_{id(self)}"

        # open pipeline
        self.pipeline = dai.Pipeline()

        # configure color camera node
        cam = self.pipeline.createColorCamera()
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setPreviewSize(self.resolution[0], self.resolution[1])
        cam.setInterleaved(False)
        cam.setFps(self.fps)

        # send frames to host
        xout = self.pipeline.createXLinkOut()
        xout.setStreamName("color")
        cam.preview.link(xout.input)

        # start device and pipeline
        self.device = dai.Device(self.pipeline)
        # retrieve output queue for color frames
        self.queue = self.device.getOutputQueue(name="color", maxSize=4, blocking=False)  # type: ignore

    def read(self) -> CameraData:
        packet = self.queue.get()

        if packet is None:
            raise RuntimeError("No packet available in the queue. The queue might be empty.")

        frame = packet.getCvFrame()

        if frame is None:
            raise RuntimeError("Received empty frame from OAK device")

        frame = np.ascontiguousarray(frame[:, :, ::-1])  # BGR to RGB

        ts_ms = packet.getTimestamp().total_seconds() * 1000
        adjusted_ts = ts_ms - self.image_transfer_time_offset

        return CameraData(images={"rgb": frame}, timestamp=adjusted_ts)

    def get_camera_info(self) -> dict:
        return {
            "camera_type": self.camera_type,
            "device_id": self.device_id,
            "width": self.resolution[0],
            "height": self.resolution[1],
            "fps": self.fps,
            "name": self.name,
        }

    def stop(self) -> None:
        self.device.close()

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Calibration data reading is not implemented for {self}")


@dataclass
class Args:
    width: int = 800
    height: int = 480
    fps: int = 60
    name: Optional[str] = None
    device_id: Optional[str] = None


if __name__ == "__main__":
    from xdof.camera.camera import plot_camera_read

    args = tyro.cli(Args)
    oak_cam = OakCamera(
        resolution=(args.width, args.height),
        fps=args.fps,
        name=args.name,
        device_id=args.device_id,
    )

    plot_camera_read(oak_cam)
