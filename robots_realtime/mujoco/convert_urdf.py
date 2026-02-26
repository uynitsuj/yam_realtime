# https://github.com/Yasu31/mjcf_urdf_simple_converter/blob/main/mjcf_urdf_simple_converter/mjcf_urdf_simple_converter.py
# modified from here

import os
import tempfile
from dataclasses import dataclass
from typing import Optional
from xml.dom import minidom
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation
from stl import mesh

CONVERT_DICT = {
    # Conversion names for OREO,
    "top_camera_camera_frame": "top-camera",
    "top_camera_left_camera_frame": "top-left-camera",
    "top_camera_right_camera_frame": "top-right-camera",
    "left_camera_camera_frame": "left-wrist-camera",
    "right_camera_camera_frame": "right-wrist-camera",
    # Arm frames
    "left_arm": "left-arm-base",
    "right_arm": "right-arm-base",
    "left_flange_frame": "left-flange-frame",
    "right_flange_frame": "right-flange-frame",
    # XMI_ROBOTIQ
    "left_gripper_frame": "left-arm-proprio",
    "right_gripper_frame": "right-arm-proprio",
    "left_gripper_camera_camera_frame": "left-wrist-camera",
    "right_gripper_camera_camera_frame": "right-wrist-camera",
}


def array2str(arr):
    return " ".join([str(x) for x in arr])


def create_meshfree_mjcf(mjcf_file):
    """Create a temporary MJCF file with all mesh geoms removed"""
    tree = ET.parse(mjcf_file)
    root = tree.getroot()

    # Remove all mesh assets
    for asset in root.findall(".//asset"):
        for mesh_elem in asset.findall("mesh"):
            asset.remove(mesh_elem)

    # Remove or replace all mesh geoms with simple primitives
    for geom in root.findall('.//geom[@type="mesh"]'):
        # Replace mesh geoms with small spheres to maintain structure
        geom.set("type", "sphere")
        geom.set("size", "0.01")
        if "mesh" in geom.attrib:
            del geom.attrib["mesh"]

    # Remove mesh references from any geom elements
    for geom in root.findall(".//geom[@mesh]"):
        geom.set("type", "sphere")
        geom.set("size", "0.01")
        del geom.attrib["mesh"]

    # Create temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".xml")
    try:
        with os.fdopen(temp_fd, "w") as temp_file:
            temp_file.write(ET.tostring(root, encoding="unicode"))
        return temp_path
    except:
        os.close(temp_fd)
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def create_body(xml_root, name, inertial_pos, inertial_rpy, mass, inertia_tensor):
    """
    create a body with given mass and inertia
    """
    # create XML element for this body
    body = ET.SubElement(xml_root, "link", {"name": name})

    # add inertial element
    inertial = ET.SubElement(body, "inertial")
    ET.SubElement(
        inertial,
        "origin",
        {"xyz": array2str(inertial_pos), "rpy": array2str(inertial_rpy)},
    )
    ET.SubElement(inertial, "mass", {"value": str(mass)})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": str(inertia_tensor[0, 0]),
            "iyy": str(inertia_tensor[1, 1]),
            "izz": str(inertia_tensor[2, 2]),
            "ixy": str(inertia_tensor[0, 1]),
            "ixz": str(inertia_tensor[0, 2]),
            "iyz": str(inertia_tensor[1, 2]),
        },
    )
    return body


def quat_mj_to_scipy(quat_mj):
    """Convert MuJoCo quaternion [w,x,y,z] to SciPy format [x,y,z,w]"""
    return [quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]]


def get_inertial_from_xml(mjcf_root, body_name):
    """Extract inertial information from XML for a given body"""
    # Find the body element in the XML
    body_elem = None
    for body in mjcf_root.iter("body"):
        if body.get("name") == body_name:
            body_elem = body
            break

    if body_elem is None:
        # Return default values if body not found
        return {
            "mass": 0.0,
            "pos": np.array([0.0, 0.0, 0.0]),
            "quat": np.array([1.0, 0.0, 0.0, 0.0]),  # w, x, y, z
            "inertia": np.array([0.001, 0.001, 0.001, 0.0, 0.0, 0.0]),
        }

    # Look for inertial element
    inertial_elem = body_elem.find("inertial")
    if inertial_elem is None:
        # Return default values if no inertial element
        return {
            "mass": 0.000001,
            "pos": np.array([0.0, 0.0, 0.0]),
            "quat": np.array([1.0, 0.0, 0.0, 0.0]),  # w, x, y, z
            "inertia": np.array([0.001, 0.001, 0.001, 0.0, 0.0, 0.0]),
        }

    # Parse mass
    mass = float(inertial_elem.get("mass", "0.0"))

    # Parse position
    pos_str = inertial_elem.get("pos", "0 0 0")
    pos = np.array([float(x) for x in pos_str.split()])

    # Parse quaternion (MuJoCo format: w x y z)
    quat_str = inertial_elem.get("quat", "1 0 0 0")
    quat = np.array([float(x) for x in quat_str.split()])

    # Parse inertia (MuJoCo format: fullinertia="ixx ixy ixz iyy iyz izz")
    fullinertia_str = inertial_elem.get("fullinertia", None)
    if fullinertia_str:
        # Full 6-element inertia tensor
        inertia_values = [float(x) for x in fullinertia_str.split()]
        inertia = np.array(inertia_values)
    else:
        # Try diagonal inertia (diaginertia="ixx iyy izz")
        diaginertia_str = inertial_elem.get("diaginertia", "0.001 0.001 0.001")
        diag_values = [float(x) for x in diaginertia_str.split()]
        inertia = np.array([diag_values[0], diag_values[1], diag_values[2], 0.0, 0.0, 0.0])

    return {"mass": mass, "pos": pos, "quat": quat, "inertia": inertia}


def create_geometry_element(parent, element_type, name, pos, rpy, geometry_data):
    """Create visual or collision element with geometry"""
    element = ET.SubElement(parent, element_type, {"name": name})
    ET.SubElement(element, "origin", {"xyz": array2str(pos), "rpy": array2str(rpy)})
    geometry = ET.SubElement(element, "geometry")

    if geometry_data["type"] == "mesh":
        ET.SubElement(geometry, "mesh", {"filename": geometry_data["filename"]})
        if element_type == "visual":
            ET.SubElement(element, "material", {"name": "white"})
    elif geometry_data["type"] == "capsule":
        ET.SubElement(
            geometry,
            "capsule",
            {
                "radius": str(geometry_data["radius"]),
                "length": str(geometry_data["length"]),
            },
        )
    elif geometry_data["type"] == "box":
        ET.SubElement(geometry, "box", {"size": array2str(geometry_data["size"])})
    elif geometry_data["type"] == "sphere":
        ET.SubElement(geometry, "sphere", {"radius": str(geometry_data["radius"])})
    elif geometry_data["type"] == "cylinder":
        ET.SubElement(
            geometry,
            "cylinder",
            {
                "radius": str(geometry_data["radius"]),
                "length": str(geometry_data["length"]),
            },
        )

    return element


def create_mesh_stl(model, geom_dataid, output_dir, mesh_name):
    """Create STL file from MuJoCo mesh data"""
    vertadr = model.mesh_vertadr[geom_dataid]
    vertnum = model.mesh_vertnum[geom_dataid]
    vert = model.mesh_vert[vertadr : vertadr + vertnum]
    faceadr = model.mesh_faceadr[geom_dataid]
    facenum = model.mesh_facenum[geom_dataid]
    face = model.mesh_face[faceadr : faceadr + facenum]

    data = np.zeros(facenum, dtype=mesh.Mesh.dtype)
    for i in range(facenum):
        data["vectors"][i] = vert[face[i]]

    m = mesh.Mesh(data, remove_empty_areas=False)
    mesh_save_path = os.path.join(output_dir, f"converted_{mesh_name}.stl")
    m.save(mesh_save_path)
    return mesh_save_path


def create_joint(xml_root, name, parent, child, pos, rpy, axis=None, jnt_range=None, joint_type=None):
    """Create a joint element (fixed, revolute, or floating)"""
    if joint_type is None:
        joint_type = "fixed" if axis is None else "revolute"

    jnt_element = ET.SubElement(xml_root, "joint", {"type": joint_type, "name": name})
    ET.SubElement(jnt_element, "parent", {"link": parent})
    ET.SubElement(jnt_element, "child", {"link": child})
    ET.SubElement(jnt_element, "origin", {"xyz": array2str(pos), "rpy": array2str(rpy)})

    if joint_type == "floating":
        # Floating joints don't need axis or limits
        pass
    elif axis is not None:
        ET.SubElement(jnt_element, "axis", {"xyz": array2str(axis)})
        ET.SubElement(
            jnt_element,
            "limit",
            {
                "lower": str(jnt_range[0]),  # type: ignore
                "upper": str(jnt_range[1]),  # type: ignore
                "effort": "100",
                "velocity": "100",
            },
        )
    return jnt_element


def convert(mjcf_file, urdf_file, asset_file_prefix="", mesh_collision=False, no_mesh=False):
    """Convert MJCF to URDF

    Args:
        mjcf_file: Input MJCF file path
        urdf_file: Output URDF file path
        asset_file_prefix: Prefix for STL filenames
        mesh_collision: Create collision elements for mesh geoms
        no_mesh: If True, exclude all geometry elements from URDF
    """
    assert mjcf_file.endswith(".xml"), f"{mjcf_file=} should end with .xml"
    assert urdf_file.endswith(".urdf"), f"{urdf_file=} should end with .urdf"
    output_dir = os.path.dirname(urdf_file)
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parse XML directly for inertial information
    mjcf_tree = ET.parse(mjcf_file)
    mjcf_root = mjcf_tree.getroot()

    # check if first default has a class
    # Create mesh-free version if no_mesh is True to avoid asset loading issues
    if no_mesh:
        temp_mjcf_path = create_meshfree_mjcf(mjcf_file)
        try:
            model = mujoco.MjModel.from_xml_path(temp_mjcf_path)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_mjcf_path):
                os.unlink(temp_mjcf_path)
    else:
        model = mujoco.MjModel.from_xml_path(mjcf_file)
    root = ET.Element("robot", {"name": "converted_robot"})
    root.append(ET.Comment(f"Auto generated from {os.path.basename(mjcf_file)}"))

    for id in range(model.nbody):
        child_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, id)

        # Check if body name should be renamed using CONVERT_DICT
        if child_name in CONVERT_DICT:
            child_name = CONVERT_DICT[child_name]
        parent_id = model.body_parentid[id]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
        # Check if parent body name should be renamed using CONVERT_DICT
        if parent_name in CONVERT_DICT:
            parent_name = CONVERT_DICT[parent_name]

        # Get body transform from parent to child
        parentbody2childbody_pos = model.body_pos[id]
        parentbody2childbody_quat = quat_mj_to_scipy(model.body_quat[id])
        parentbody2childbody_Rot = Rotation.from_quat(parentbody2childbody_quat).as_matrix()
        parentbody2childbody_rpy = Rotation.from_quat(parentbody2childbody_quat).as_euler("xyz")

        # Get inertial properties from XML
        original_child_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, id)

        # Handle world body specially
        if original_child_name == "world":
            mass = 0.001  # Minimal mass for world body
            childbody2childinertia_pos = np.array([0.0, 0.0, 0.0])
            childbody2childinertia_rpy = np.array([0.0, 0.0, 0.0])
            # Use minimal dummy inertia for world body
            inertia_tensor = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])
        else:
            inertial_info = get_inertial_from_xml(mjcf_root, original_child_name)

            mass = inertial_info["mass"]
            childbody2childinertia_pos = inertial_info["pos"]
            # Set RPY to zero - we'll handle rotation in the inertia tensor itself
            childbody2childinertia_rpy = np.array([0.0, 0.0, 0.0])

            # Create diagonal inertia tensor from XML inertia data
            inertia_data = inertial_info["inertia"]
            diagonal_inertia = np.array([[inertia_data[0], 0, 0], [0, inertia_data[1], 0], [0, 0, inertia_data[2]]])

            # Apply rotation to transform diagonal inertia to body frame
            # This gives us the proper off-diagonal terms (ixy, ixz, iyz)
            childbody2childinertia_quat = quat_mj_to_scipy(inertial_info["quat"])
            if not np.allclose(childbody2childinertia_quat, [0, 0, 0, 1]):
                # Get rotation matrix from inertial frame to body frame
                rotation_matrix = Rotation.from_quat(childbody2childinertia_quat).as_matrix()
                # Transform inertia tensor: I_body = R * I_diagonal * R^T
                inertia_tensor = rotation_matrix @ diagonal_inertia @ rotation_matrix.T
            else:
                # No rotation needed
                inertia_tensor = diagonal_inertia

            # Validate inertia tensor - check for positive eigenvalues
            eigenvalues = np.linalg.eigvals(inertia_tensor)
            if np.any(eigenvalues <= 0):
                print(f"WARNING: Body {child_name} has invalid inertia tensor with eigenvalues {eigenvalues}")
                print("         Using default diagonal inertia")
                # Use default minimal inertia
                inertia_tensor = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])

        # Create child body
        body_element = create_body(
            root,
            child_name,
            childbody2childinertia_pos,
            childbody2childinertia_rpy,
            mass,
            inertia_tensor,
        )

        # Process geometry elements (skip if no_mesh is True)
        if not no_mesh:
            for geomnum_i in range(model.body_geomnum[id]):
                geomid = model.body_geomadr[id] + geomnum_i
                geom_pos = model.geom_pos[geomid]
                geom_quat = quat_mj_to_scipy(model.geom_quat[geomid])
                geom_rpy = Rotation.from_quat(geom_quat).as_euler("xyz")

                if model.geom_type[geomid] == mujoco.mjtGeom.mjGEOM_MESH:
                    # Mesh geoms become visual elements (and optionally collision)
                    geom_dataid = model.geom_dataid[geomid]
                    mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(geom_dataid))

                    # Create STL file
                    create_mesh_stl(model, geom_dataid, output_dir, mesh_name)

                    # Create visual element
                    mesh_filename = f"{asset_file_prefix}converted_{mesh_name}.stl"
                    create_geometry_element(
                        body_element,
                        "visual",
                        mesh_name,
                        geom_pos,
                        geom_rpy,
                        {"type": "mesh", "filename": mesh_filename},
                    )

                    # Optionally create collision element for mesh
                    if mesh_collision:
                        create_geometry_element(
                            body_element,
                            "collision",
                            f"{mesh_name}_collision",
                            geom_pos,
                            geom_rpy,
                            {"type": "mesh", "filename": mesh_filename},
                        )
                else:
                    # Non-mesh geoms become collision elements
                    geom_name = f"{child_name}_geom_{geomnum_i}"
                    geom_size = model.geom_size[geomid]

                    # Determine geometry type and parameters
                    if model.geom_type[geomid] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                        geom_data = {
                            "type": "capsule",
                            "radius": geom_size[0],
                            "length": geom_size[1] * 2,
                        }
                    elif model.geom_type[geomid] == mujoco.mjtGeom.mjGEOM_BOX:
                        geom_data = {
                            "type": "box",
                            "size": geom_size * 2,
                        }  # mujoco uses half-sizes
                    elif model.geom_type[geomid] == mujoco.mjtGeom.mjGEOM_SPHERE:
                        geom_data = {"type": "sphere", "radius": geom_size[0]}
                    elif model.geom_type[geomid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        geom_data = {
                            "type": "cylinder",
                            "radius": geom_size[0],
                            "length": geom_size[1] * 2,
                        }
                    else:
                        print(f"Warning: Unsupported geom type {model.geom_type[geomid]}, using sphere fallback")
                        geom_data = {"type": "sphere", "radius": 0.01}

                    create_geometry_element(body_element, "collision", geom_name, geom_pos, geom_rpy, geom_data)

        # Create joint connecting parent to child
        if original_child_name == "world":
            continue  # World body has no parent joint

        jntnum = model.body_jntnum[id]

        if jntnum == 0:
            # No joints - create fixed joint
            jnt_name = f"{parent_name}2{child_name}_fixed"
            create_joint(
                root,
                jnt_name,
                parent_name,
                child_name,
                parentbody2childbody_pos,
                parentbody2childbody_rpy,
            )
        elif jntnum == 1:
            # Single joint - handle revolute or fixed
            jntid = model.body_jntadr[id]
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jntid) or f"joint_{jntid}"
            # Check if joint name should be renamed using CONVERT_DICT
            if jnt_name in CONVERT_DICT:
                jnt_name = CONVERT_DICT[jnt_name]

            if jnt_name.startswith("joint_"):
                print(f"WARNING: Using auto-generated joint name {jnt_name}")

            if model.jnt_type[jntid] == mujoco.mjtJoint.mjJNT_HINGE:
                # Revolute joint
                childbody2jnt_pos = model.jnt_pos[jntid]
                parentbody2jnt_pos = parentbody2childbody_pos + parentbody2childbody_Rot @ childbody2jnt_pos

                create_joint(
                    root,
                    jnt_name,
                    parent_name,
                    child_name,
                    parentbody2jnt_pos,
                    parentbody2childbody_rpy,
                    model.jnt_axis[jntid],
                    model.jnt_range[jntid],
                )
            elif model.jnt_type[jntid] == mujoco.mjtJoint.mjJNT_FREE:
                jnt_name = f"floating_{child_name}"

                # Free joint becomes floating joint in URDF
                print(f"Converting freejoint {jnt_name} to floating joint")
                create_joint(
                    root,
                    jnt_name,
                    parent_name,
                    child_name,
                    parentbody2childbody_pos,
                    parentbody2childbody_rpy,
                    joint_type="floating",
                )
            else:
                # Other joint types treated as fixed
                print(f"Unsupported joint type {model.jnt_type[jntid]}, treating as fixed")
                create_joint(
                    root,
                    jnt_name,
                    parent_name,
                    child_name,
                    parentbody2childbody_pos,
                    parentbody2childbody_rpy,
                )
        else:
            # Multiple joints - create fixed joint with warning
            print(f"WARNING: Body {child_name} has {jntnum} joints, creating fixed joint")
            jnt_name = f"{parent_name}2{child_name}_fixed_multi"
            create_joint(
                root,
                jnt_name,
                parent_name,
                child_name,
                parentbody2childbody_pos,
                parentbody2childbody_rpy,
            )

    # define white material
    material_element = ET.SubElement(root, "material", {"name": "white"})
    ET.SubElement(material_element, "color", {"rgba": "1 1 1 1"})

    # write to file with pretty printing
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(urdf_file, "w") as f:
        f.write(xmlstr)


@dataclass
class Args:
    """Example usage:
    >>> python convert_urdf.py --mjcf-file mjmodel.xml --urdf-file urdf/robot.urdf --mesh-collision

    This generates a folder called urdf, with the mesh files and urdf in the specified folder.
    The --mesh-collision flag creates collision elements for mesh geoms, which can be omitted if you want the meshes only as visual geoms
    """

    mjcf_file: str
    urdf_file: str
    package: Optional[str] = None
    mesh_collision: bool = False  # Create collision elements for mesh geoms
    no_mesh: bool = False  # if true, inclulde no geometry elemtents in the URDF, only joints and bodies


def main(args: Args):
    package_prefix = "package://" if args.package else f"package://{args.package}/"
    convert(
        args.mjcf_file,
        args.urdf_file,
        asset_file_prefix=package_prefix,
        mesh_collision=args.mesh_collision,
        no_mesh=args.no_mesh,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))