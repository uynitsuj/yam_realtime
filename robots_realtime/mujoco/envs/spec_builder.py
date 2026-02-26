import os
import tempfile
from typing import Sequence
from xml.etree import ElementTree as ET

import mujoco
import mujoco.viewer
import tyro

from robots_realtime.mujoco.envs.schema.robot import (
    STATION_ROBOT_MAP,
    XMI_STATION_MAP,
    BimanualStationSpecConfig,
    CameraSpecConfig,
    RobotSpecConfig,
    XMIStationSpecConfig,
)
from xdof_sdk.data.schema.types import Transform3D, WorldFrame

WORLD_FRAME_NAME = "left_arm"
FLANGE_FRAME_NAME = "flange_frame"


def collapse_defaults(xml_string: str) -> str:
    """
    Condense two consecutive <default> tags (parent→child) into one
    when the inner <default> lacks attributes.
    """
    root = ET.fromstring(xml_string)

    def _collapse(node):
        i = 0
        while i < len(node):
            child = node[i]
            _collapse(child)  # depth-first

            # Parent and child are both <default>, and child has no attrs ➜ flatten
            if node.tag == "default" and child.tag == "default" and not child.attrib:
                # splice grandchildren at current position
                for grand in list(child):
                    node.insert(i, grand)
                    i += 1
                node.remove(child)  # drop the redundant tag
                continue  # re-check same index
            i += 1

    _collapse(root)
    return ET.tostring(root, encoding="unicode")


def compile_station_spec(
    station_spec_config: BimanualStationSpecConfig,
) -> mujoco.MjSpec:
    station_spec = station_spec_config.workstation.get_mj_spec()
    robot_spec_config = station_spec_config.robot

    world_T_left = station_spec_config.workstation.world_T_left_arm
    world_T_top_left_camera = world_T_left @ station_spec_config.workstation.left_arm_T_top_left_camera
    attach_camera(
        parent=station_spec.worldbody,
        name="top_camera",
        camera_spec_config=station_spec_config.top_camera,
        transform=world_T_top_left_camera,
    )

    # build arm with the wrist camera
    arm_spec = build_arm_with_camera(
        robot=robot_spec_config,
        camera=station_spec_config.wrist_camera,
    )

    # Left arm with respect to world frame
    left_site = station_spec.worldbody.add_site(
        pos=world_T_left.position,
        quat=world_T_left.quaternion_wxyz,  # identity quaternion
    )

    # Right arm positioned using left_T_right transform
    left_T_right = station_spec_config.workstation.left_arm_T_right_arm
    world_T_right = world_T_left @ left_T_right
    right_site = station_spec.worldbody.add_site(pos=world_T_right.position, quat=world_T_right.quaternion_wxyz)

    left_site.attach_body(arm_spec.body("arm"), "left_", "")
    right_site.attach_body(arm_spec.body("arm"), "right_", "")
    return station_spec


def build_arm_with_camera(robot: RobotSpecConfig, camera: CameraSpecConfig) -> mujoco.MjSpec:
    arm_spec = robot.get_mj_spec()
    attach_camera(
        parent=arm_spec.body(robot.last_link),
        name="camera",
        camera_spec_config=camera,
        transform=robot.wrist_cam_in_last_link,
    )

    arm_spec.body(robot.last_link).add_body(
        name=FLANGE_FRAME_NAME,
        pos=robot.flange_in_last_link.position,
        quat=robot.flange_in_last_link.quaternion_wxyz,
    )

    return arm_spec


def get_bimanual_station_spec_xml(
    station_spec: mujoco.MjSpec,
    world_frame: WorldFrame = WorldFrame.LEFT_ARM,
) -> str:
    assert world_frame == WorldFrame.LEFT_ARM, "Only LEFT_ARM world frame is supported for URDF generation. For now"

    # move the station to the origin as defined by the WORLD_FRAME_NAME
    model = station_spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    world_pos = data.body(
        WORLD_FRAME_NAME
    ).xpos  # get the position of the world frame, which is the left arm (see assert)

    adjusted_spec = mujoco.MjSpec()
    frame = adjusted_spec.worldbody.add_frame(
        pos=-world_pos
    )  # move the station to the origin as defined by the the WorldFrame
    body = station_spec.worldbody
    body.name = "station_worldbody"
    frame.attach_body(body, "", "")

    xml = adjusted_spec.to_xml()
    xml = collapse_defaults(xml)
    return xml


def add_body_with_cube(
    parent: mujoco.MjsBody,
    name: str,
    pos: Sequence[float] = (0, 0, 0),
    quat: Sequence[float] = (1, 0, 0, 0),
    size: Sequence[float] = (0.01, 0.01, 0.01),
    rgba: Sequence[float] = (0.5, 0.5, 0.5, 0.5),
    add_freejoint: bool = False,
) -> mujoco.MjsBody:
    body = parent.add_body(name=name, pos=pos, quat=quat, mass=0.001)
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        rgba=rgba,
    )
    if add_freejoint:
        body.add_freejoint()
    return body


def _add_camera_to_body(body: mujoco.MjsBody, name: str):
    body.add_camera(
        name=name,
        pos=[0, 0, 0],
        quat=[0, 1, 0, 0],
        focal_length=[1.93e-3, 1.93e-3],
        resolution=[1280, 720],
        sensor_size=[3896e-6, 2140e-6],
    )


def _get_camera_mjsbody(camera_spec_config: CameraSpecConfig) -> mujoco.MjsBody:
    camera = mujoco.MjSpec()
    # just to keep track of what camera it is
    main_camera = add_body_with_cube(
        parent=camera.worldbody,
        name=camera_spec_config.main_camera_frame_name,
        pos=[0, 0, 0],
        quat=[1, 0, 0, 0],
    )
    _add_camera_to_body(main_camera, camera_spec_config.main_camera_frame_name)

    if camera_spec_config.left_camera_T_right_camera is not None:
        assert camera_spec_config.stereo_camera_frame_name is not None
        body = add_body_with_cube(
            parent=main_camera,
            name=camera_spec_config.stereo_camera_frame_name,
            # We assume we're already in the X right, Y down, Z forward coordinate system
            # since the world transform to top_left_camera should take us there already.
            pos=camera_spec_config.left_camera_T_right_camera.position,
            quat=camera_spec_config.left_camera_T_right_camera.quaternion_wxyz,
        )
        _add_camera_to_body(body, camera_spec_config.stereo_camera_frame_name)

    # This is really dumb, we need a reference to worldbody before we rename it
    # because `worldbody` is a property function and looks for worldbody by searching for "world" wtf!!!
    worldbody = camera.worldbody
    camera.worldbody.name = camera_spec_config.body_name
    return worldbody


def attach_camera(
    parent: mujoco.MjsBody,
    name: str,
    camera_spec_config: CameraSpecConfig,
    transform: Transform3D,
) -> None:
    camera_body = _get_camera_mjsbody(camera_spec_config)

    site = parent.add_site(
        name=name,
        pos=transform.position,
        quat=transform.quaternion_wxyz,
    )
    site.attach_body(camera_body, name + "_", "")


def get_xmi_station_spec_xml(xmi_station_spec_config: XMIStationSpecConfig) -> str:
    world = mujoco.MjSpec()

    xmi_head = add_body_with_cube(
        parent=world.worldbody,
        name="xmi_head",
        pos=[0, 0, 0.2],  # offset for easier visualization
        quat=[1, 0, 0, 0],
        add_freejoint=True,
    )

    attach_camera(
        parent=xmi_head,
        name="top_camera",
        camera_spec_config=xmi_station_spec_config.top_camera,
        transform=xmi_station_spec_config.xmi.top_camera_in_tracker,
    )

    for side in ["left", "right"]:
        if side == "left":
            color = [0, 0.5, 0.5, 0.5]
            gripper_in_controller = xmi_station_spec_config.xmi.gripper_in_left_tracker
            pos = [0.0, 0.2, 0.0]  # offset for easier visualization
        else:
            gripper_in_controller = xmi_station_spec_config.xmi.gripper_in_right_tracker
            color = [0.5, 0.5, 0, 0.5]
            pos = [0.0, -0.2, 0.0]  # offset for easier visualization

        xmi_controller = add_body_with_cube(
            parent=world.worldbody,
            name=f"{side}_xmi_controller_frame",
            pos=pos,
            quat=[1, 0, -1, 0],
            rgba=color,
            add_freejoint=True,
        )
        gripper_body = add_body_with_cube(
            parent=xmi_controller,
            name=f"{side}_gripper_frame",
            pos=gripper_in_controller.position,
            quat=gripper_in_controller.quaternion_wxyz,
            rgba=color,
        )
        attach_camera(
            parent=gripper_body,
            name=f"{side}_gripper_camera",
            camera_spec_config=xmi_station_spec_config.wrist_camera,
            transform=xmi_station_spec_config.xmi.gripper_camera_in_gripper,
        )

    return world.to_xml()


def get_standard_urdf_str(
    station_type: str,
    world_frame: WorldFrame = WorldFrame.LEFT_ARM,
) -> str:
    """Generate URDF string for the specified robot station configuration."""
    # get mjcf of the station
    if station_type in STATION_ROBOT_MAP:
        _station_spec = compile_station_spec(STATION_ROBOT_MAP[station_type])
        mjcf_xml = get_bimanual_station_spec_xml(_station_spec, world_frame)
    elif station_type in XMI_STATION_MAP:
        _xmi_spec_config = XMI_STATION_MAP[station_type]
        mjcf_xml = get_xmi_station_spec_xml(_xmi_spec_config)
    else:
        raise ValueError(f"Unsupported station type: {station_type}")

    # Generate the URDF string
    return get_urdf_str(mjcf_xml)


def get_urdf_str(
    mjcf_xml_str: str,
) -> str:
    """Convert MJCF XML string to URDF string."""
    from robots_realtime.mujoco.convert_urdf import convert

    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate paths within temporary directory
        xml_path = os.path.join(temp_dir, "station.xml")
        urdf_path = os.path.join(temp_dir, "station.urdf")

        # Save XML to file
        with open(xml_path, "w") as xml_file:
            xml_file.write(mjcf_xml_str)

        # Export environment as XML, then convert to URDF
        convert(xml_path, urdf_path, no_mesh=True)

        # Read and return URDF content
        with open(urdf_path, "r") as urdf_file:
            return urdf_file.read()


def main(station_type: str) -> None:
    urdf_content = get_standard_urdf_str(station_type)
    if station_type in STATION_ROBOT_MAP:
        _station_spec = compile_station_spec(STATION_ROBOT_MAP[station_type])
        xml = get_bimanual_station_spec_xml(_station_spec)
    elif station_type in XMI_STATION_MAP:
        _xmi_spec_config = XMI_STATION_MAP[station_type]
        xml = get_xmi_station_spec_xml(_xmi_spec_config)
    else:
        raise ValueError(f"Unsupported station type: {station_type}")

    with tempfile.TemporaryDirectory() as temp_dir:
        from robots_realtime.mujoco.convert_urdf import create_meshfree_mjcf

        # Generate paths within temporary directory
        xml_path = os.path.join(temp_dir, "station.xml")
        with open(xml_path, "w") as xml_file:
            xml_file.write(xml)
        new_xml_path = create_meshfree_mjcf(xml_path)

        with open(new_xml_path, "r") as xml_file:
            xml = xml_file.read()

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.viewer.launch(m, d)


if __name__ == "__main__":
    tyro.cli(main)