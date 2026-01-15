"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""

import time

import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

import robots_realtime.robots.inverse_kinematics.pyroki_snippets as pks


def main():
    """Main function for basic IK."""

    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )
    # ik_target_prev = server.scene.add_transform_controls(
    #     "/ik_target_prev", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    # )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    solution_init = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8, 0.0])
    solution = None

    while True:
        # Solve IK.
        if solution is None:
            solution = solution_init
        start_time = time.time()
        solution = pks.solve_ik_vel_cost(
            robot=robot,
            target_link_name=target_link_name,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
            prev_cfg=np.array(solution),
        )


        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(solution)


if __name__ == "__main__":
    main()