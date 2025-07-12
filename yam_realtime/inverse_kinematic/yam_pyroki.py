"""
Simplest Inverse Kinematics Example using PyRoki.
"""

import time

import numpy as np

try:
    import pyroki as pk
except ImportError:
    print("ImportError: pyroki not found:")
    print("pip install git+https://github.com/chungmin99/pyroki.git")
    exit()

import viser

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except ImportError:
    print("ImportError: robot_descriptions not found, for now:")
    print(
        "pip install git+https://github.com/uynitsuj/robot_descriptions.py.git@e2502f3d4d8aa38d18b72527c5baae2b19a1182e"
    )
    print("[INFO] Will be changed to official repo once YAM and RBY1 are added and released to upstream")
    exit()

from viser.extras import ViserUrdf

from xdof.pyroki.pyroki_snippets import solve_ik as solve_ik


def main():
    """Main function for basic IK."""

    urdf = load_robot_description("yam_description")
    target_link_name = "link_6"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    rest_pose = urdf_vis._urdf.cfg

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.31, 0.0, 0.26), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    solution = None
    while True:
        # Solve IK.
        if solution is None:
            prev_cfg = rest_pose
        else:
            prev_cfg = solution
        start_time = time.time()
        solution = solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
        )

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(solution)


if __name__ == "__main__":
    main()
