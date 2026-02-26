from typing import Dict

import mujoco
from pydantic import (
    BaseModel,
    ConfigDict,
)

from robots_realtime import ROOT_PATH
from xdof_sdk.data.schema.types import Transform3D

# Path constants for robot assets
MENAGERIE_ROOT = ROOT_PATH / "robots_realtime/mujoco/assets"

# Standard DOF counts
SIX_DOF = 6
SEVEN_DOF = 7
ONE_DOF = 1


class MujocoSpecConfig(BaseModel):
    """Base specification for MuJoCo XML-based models."""

    model_config = ConfigDict(extra="forbid")

    xml_path: str

    def get_mj_spec(self) -> mujoco.MjSpec:
        """Load MuJoCo specification from the configured XML path.

        Returns:
            mujoco.MjSpec: Loaded MuJoCo specification ready for compilation.

        Raises:
            FileNotFoundError: If the XML file doesn't exist.
            mujoco.MjError: If the XML file is malformed.
        """
        spec = mujoco.MjSpec.from_file(self.xml_path)
        # Ensure copy during attach is enabled
        spec.copy_during_attach = True
        return spec


class RobotSpecConfig(MujocoSpecConfig):
    num_joint_dofs: int
    num_gripper_dofs: int
    last_link: str
    wrist_cam_in_last_link: Transform3D = Transform3D()
    flange_in_last_link: Transform3D = Transform3D()


class BimanualWorkstationSpecConfig(MujocoSpecConfig):
    """Standard workstation environment specification."""

    left_arm_T_right_arm: Transform3D
    xml_path: str = str(MENAGERIE_ROOT / "production" / "oreo_urdf" / "station" / "station.xml")
    left_arm_T_top_left_camera: Transform3D = Transform3D()
    world_T_left_arm: Transform3D = Transform3D()


class CameraSpecConfig(BaseModel):
    # We actually just generate these by making mujoco model in the code lol

    # Name of the camera body in the MuJoCo model
    body_name: str

    main_camera_frame_name: str
    baseline_meters: float | None = None
    left_camera_T_right_camera: Transform3D | None = None
    stereo_camera_frame_name: str | None = None


class XMISpecConfig(MujocoSpecConfig):
    num_gripper_dofs: int
    top_camera_in_tracker: Transform3D = (
        Transform3D()
    )  # the main view of the top camera. If stereo, this is the left camera by default.
    gripper_in_left_tracker: Transform3D = Transform3D()
    gripper_in_right_tracker: Transform3D = Transform3D()
    gripper_camera_in_gripper: Transform3D = Transform3D()


INTEL_D405_CAMERA_SPEC_CONFIG = CameraSpecConfig(
    body_name="d405",
    main_camera_frame_name="camera_frame",
)
OAK_D_CAMERA_SPEC_CONFIG = CameraSpecConfig(
    body_name="oak_d",
    main_camera_frame_name="camera_frame",
)
ZED_X_ONE_GS_CAMERA_SPEC_CONFIG = CameraSpecConfig(
    body_name="zed_x_one_gs",
    main_camera_frame_name="camera_frame",
)
ZED2_CAMERA_SPEC_CONFIG = CameraSpecConfig(
    body_name="zed2",
    main_camera_frame_name="left_camera_frame",
    left_camera_T_right_camera=Transform3D(
        position=[0.12, 0.0, 0.0],
        quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
    ),
    stereo_camera_frame_name="right_camera_frame",
)
ZED_X_CAMERA_SPEC_CONFIG = CameraSpecConfig(
    body_name="zed_x",
    main_camera_frame_name="left_camera_frame",
    left_camera_T_right_camera=Transform3D(
        position=[0.12, 0.0, 0.0],
        quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
    ),
    stereo_camera_frame_name="right_camera_frame",
)


class BimanualStationSpecConfig(BaseModel):
    """Base class for robot station specifications."""

    robot: RobotSpecConfig
    workstation: BimanualWorkstationSpecConfig
    wrist_camera: CameraSpecConfig
    top_camera: CameraSpecConfig


class XMIStationSpecConfig(BaseModel):
    """Base class for robot station specifications."""

    xmi: XMISpecConfig
    wrist_camera: CameraSpecConfig
    top_camera: CameraSpecConfig


_base_yam_config = BimanualStationSpecConfig(
    robot=RobotSpecConfig(
        xml_path=str(MENAGERIE_ROOT / "i2rt_yam" / "yam.xml"),
        num_joint_dofs=SIX_DOF,
        num_gripper_dofs=ONE_DOF,
        last_link="link_6",
        # Transform3D https://cad.onshape.com/documents/42d1361a8dcbb2f186683920/w/cde8cc7d54786faf08864399/e/1077b5eec0767259a9ddf7b5
        wrist_cam_in_last_link=Transform3D(
            position=[-0.0017, 0.079729, 0.066021],
            quaternion_wxyz=[0.0, 0.0, 0.4226, -0.9063],
        ),
        flange_in_last_link=Transform3D(
            position=[0.0, 0.0, 0],
            quaternion_wxyz=[0.0, 0.0, 0.0, -1.0],
        ),
    ),
    workstation=BimanualWorkstationSpecConfig(
        left_arm_T_right_arm=Transform3D(
            position=[0.0, -0.61, 0.0],  # right arm is 0.61m in -Y direction from left arm
            quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],  # no rotation
        ),
        left_arm_T_top_left_camera=Transform3D(
            position=[0.08600512 - 0.2525, -0.305, 0.95432053],
            quaternion_wxyz=[0.183, -0.683, 0.683, -0.183],
        ),
    ),
    wrist_camera=INTEL_D405_CAMERA_SPEC_CONFIG,
    top_camera=INTEL_D405_CAMERA_SPEC_CONFIG,
)

_yam_zed_config = _base_yam_config.model_copy(deep=True)
_yam_zed_config.top_camera = ZED_X_CAMERA_SPEC_CONFIG
_yam_zed_config.wrist_camera = ZED_X_ONE_GS_CAMERA_SPEC_CONFIG
_yam_zed_config.robot.xml_path = str(MENAGERIE_ROOT / "production" / "oreo_urdf" / "yam_lw" / "yam_lw_gripper.xml")

_yam_oak_linear_gripper_config = _base_yam_config.model_copy(deep=True)
_yam_oak_linear_gripper_config.wrist_camera = OAK_D_CAMERA_SPEC_CONFIG
_yam_oak_linear_gripper_config.top_camera = OAK_D_CAMERA_SPEC_CONFIG
_yam_oak_linear_gripper_config.robot.xml_path = str(
    MENAGERIE_ROOT / "production" / "oreo_urdf" / "yam_lw" / "yam_lw_gripper.xml"
)

_yam_realsense_linear_gripper_config = _base_yam_config.model_copy(deep=True)
_yam_realsense_linear_gripper_config.wrist_camera = INTEL_D405_CAMERA_SPEC_CONFIG
_yam_realsense_linear_gripper_config.top_camera = INTEL_D405_CAMERA_SPEC_CONFIG
_yam_realsense_linear_gripper_config.robot.xml_path = str(
    MENAGERIE_ROOT / "production" / "oreo_urdf" / "yam_lw" / "yam_lw_gripper.xml"
)

# Station type to robot specification mapping
# Follows the standard of Z front, Y down, X right (for both gripper and cameras)
# Values here are read directly from the CAD models
STATION_ROBOT_MAP: Dict[str, BimanualStationSpecConfig] = {
    "SIM_YAM": BimanualStationSpecConfig(
        robot=RobotSpecConfig(
            xml_path=str(MENAGERIE_ROOT / "i2rt_yam" / "yam.xml"),
            num_joint_dofs=SIX_DOF,
            num_gripper_dofs=ONE_DOF,
            last_link="link_6",
            # Transform3D https://cad.onshape.com/documents/42d1361a8dcbb2f186683920/w/cde8cc7d54786faf08864399/e/1077b5eec0767259a9ddf7b5
            wrist_cam_in_last_link=Transform3D(
                position=[0.0, 0.092783, 0.064977],
                quaternion_wxyz=[0.0, 0.0, 0.4226, -0.9063],
            ),
            flange_in_last_link=Transform3D(
                position=[0.0, 0.0, 0],
                quaternion_wxyz=[0.0, 0.0, 0.0, -1.0],
            ),
        ),
        workstation=BimanualWorkstationSpecConfig(
            xml_path=str(MENAGERIE_ROOT / "yam_station" / "station_with_gate.xml"),
            world_T_left_arm=Transform3D(position=[0.2525, 0.31, 0.75], quaternion_wxyz=[1.0, 0.0, 0.0, 0.0]),
            left_arm_T_right_arm=Transform3D(
                position=[0.0, -0.61, 0.0],  # right arm is 0.61m in -Y direction from left arm
                quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],  # no rotation
            ),
            left_arm_T_top_left_camera=Transform3D(
                position=[0.08600512 - 0.2525, -0.305, 0.95432053],
                quaternion_wxyz=[0.183, -0.683, 0.683, -0.183],
            ),
        ),
        wrist_camera=INTEL_D405_CAMERA_SPEC_CONFIG,
        top_camera=INTEL_D405_CAMERA_SPEC_CONFIG,
    ),
    "YAM_0_61": _base_yam_config,
    "YAM_OAK_LINEAR_GRIPPER": _yam_oak_linear_gripper_config,
    "YAM_REALSENSE_LINEAR_GRIPPER": _yam_realsense_linear_gripper_config,
    "YAM_ZED_0_61": _yam_zed_config,
    "FRANKA_STANDARD": BimanualStationSpecConfig(
        robot=RobotSpecConfig(
            xml_path=str(MENAGERIE_ROOT / "production" / "oreo_urdf" / "franka" / "panda_nohand.xml"),
            num_joint_dofs=SEVEN_DOF,
            num_gripper_dofs=ONE_DOF,
            last_link="link7",
            # Transform3D values measured from CAD, https://cad.onshape.com/documents/ccf73a611a9fa11493261964/w/77ae15aa30c9eb44b0a92be4/e/a31857ca33ca5a3b6104d48c
            wrist_cam_in_last_link=Transform3D(
                position=[0.009, 0.092889, 0.107 + 0.004598],
                quaternion_wxyz=[0.0, 0.0, 0.21636, -0.97645],
            ),
            # flange z offset read from xml
            #  https://github.com/xdofai/menagerie/blob/d667da08441d625e4acfd5a35d5e57abbf8ae7a0/franka_emika_panda/panda.xml#L211
            flange_in_last_link=Transform3D(
                position=[0.0, 0.0, 0.107],
                quaternion_wxyz=[0.0, 0.0, 0.0, 1.0],
            ),
        ),
        workstation=BimanualWorkstationSpecConfig(
            left_arm_T_right_arm=Transform3D(
                position=[0.0, -0.7868, 0.0],  # right arm is 0.7868m in -Y direction from left arm
                quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],  # no rotation
            ),
            left_arm_T_top_left_camera=Transform3D(
                position=[0.2282, (-0.7868 / 2) + 0.06, 0.915 + 0.065334],
                quaternion_wxyz=[0.22415285, -0.67033723, 0.67033723, -0.22415285],
            ),
        ),
        wrist_camera=INTEL_D405_CAMERA_SPEC_CONFIG,
        top_camera=ZED2_CAMERA_SPEC_CONFIG,
    ),
}


XMI_STATION_MAP: Dict[str, XMIStationSpecConfig] = {
    "XMI_ROBOTIQ": XMIStationSpecConfig(
        xmi=XMISpecConfig(
            # This XML path is not actually used.
            xml_path=str(MENAGERIE_ROOT / "xmi_robotique" / "xmi.xml"),
            num_gripper_dofs=ONE_DOF,
            # values read from metadata here /nfs/data/sz_xmi_01/20250724/episode_20250724_030256_20ac83df.npy.mp4
            top_camera_in_tracker=Transform3D(
                position=[0.083002, 0.06422473, 0.05830567],
                quaternion_wxyz=[0.2777717255528629, -0.6502644862942537, 0.6477349362892192, -0.2836166755644964],
            ),
            gripper_in_left_tracker=Transform3D(
                position=[0.01732262, 4.181e-05, -0.07813787],
                quaternion_wxyz=[0.002435159999709972, -0.6542473099220941, 0.7561531799099596, 0.013670989998372159],
            ),
            gripper_in_right_tracker=Transform3D(
                position=[0.03051299, 0.00825025, -0.07254365],
                quaternion_wxyz=[0.01810433501018543, -0.7436692454183826, 0.6682627853759595, -0.007289455004101041],
            ),
            gripper_camera_in_gripper=Transform3D(
                position=[0.0, -0.086465, -0.01125],
                quaternion_wxyz=[-0.99939083, 0.0348995, -0.0, -0.0],
            ),
        ),
        wrist_camera=INTEL_D405_CAMERA_SPEC_CONFIG,
        top_camera=ZED2_CAMERA_SPEC_CONFIG,
    ),
    "XMI_PASSIVE_LINEAR_GRIPPER": XMIStationSpecConfig(
        xmi=XMISpecConfig(
            # This XML path is not actually used.
            xml_path=str(MENAGERIE_ROOT / "xmi_dummy" / "xmi.xml"),
            num_gripper_dofs=ONE_DOF,
            # values read from metadata here /nfs/data/sz_xmi_01/20250724/episode_20250724_030256_20ac83df.npy.mp4
            top_camera_in_tracker=Transform3D(
                position=[0.01026, -0.06816211, 0.08698146],
                quaternion_wxyz=[0.879787, 0.4753045, 0.0061976, 0.00469645],
            ),
            gripper_in_left_tracker=Transform3D(
                position=[-0.0, -0.01511252, 0.04749972],
                quaternion_wxyz=[0.84037059, 0.54148486, 0.01276508, 0.02021065],
            ),
            gripper_in_right_tracker=Transform3D(
                position=[-0.0, -0.01511252, 0.04749972],
                quaternion_wxyz=[0.84037059, 0.54148486, 0.01276508, 0.02021065],
            ),
            gripper_camera_in_gripper=Transform3D(
                position=[0.0, -0.0802, 0.021529],
                quaternion_wxyz=[-0.97958455, 0.19976, 0.0207, 0.0087],
            ),
        ),
        wrist_camera=INTEL_D405_CAMERA_SPEC_CONFIG,
        top_camera=INTEL_D405_CAMERA_SPEC_CONFIG,
    ),
}