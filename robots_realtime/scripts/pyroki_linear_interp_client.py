import time
import numpy as np
import requests
import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description

def main():
    # 1. Setup Viser and Robot
    server = viser.ViserServer()
    robot_name = "panda"
    urdf = load_robot_description(f"{robot_name}_description")
    urdf_vis = ViserUrdf(server, urdf)
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)

    # 2. State Management
    # We store the trajectory in a dict so the callback can update it
    state = {
        "traj": np.zeros((10, 8)), # Placeholder (10 timesteps, 8-DOF Panda)
        "needs_update": True,
        "last_request_time": 0
    }

    # 3. Interactive UI Elements
    # Start position is fixed for this example
    start_pos = [0.4, -0.3, 0.3]
    start_wxyz = [0.0, 0.0, 1.0, 0.0]
    server.scene.add_frame("/start", position=np.array(start_pos), wxyz=np.array(start_wxyz), axes_length=0.1)

    # Interactive End Target
    ik_target = server.scene.add_transform_controls(
        "/ik_target", 
        scale=0.2, 
        position=(0.4, 0.3, 0.3), 
        wxyz=(0, 0, 1, 0)
    )

    # UI Feedback
    solve_time_gui = server.gui.add_number("Solve Time (ms)", initial_value=0.0, disabled=True)
    status_gui = server.gui.add_text("Status", initial_value="Idle", disabled=True)

    # 4. The Request Logic
    def update_trajectory(_=None):
        """Sends the current target to the TrajOpt server."""
        now = time.time()
        # Rate limit: Don't request more than 10 times per second
        if now - state["last_request_time"] < 0.1:
            return

        payload = {
            "start_pose_wxyz_xyz": start_wxyz + start_pos,
            "end_pose_wxyz_xyz": list(ik_target.wxyz) + list(ik_target.position),
            "timesteps": 20,
        }

        try:
            status_gui.value = "Optimizing..."
            response = requests.post("http://127.0.0.1:8116/plan", json=payload, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            state["traj"] = np.array(data["waypoints"])
            state["last_request_time"] = now
            solve_time_gui.value = data["compute_time"] * 1000
            status_gui.value = "Ready"
        except Exception as e:
            status_gui.value = "Error"
            print(f"Request failed: {e}")

    # Trigger update when the gizmo is moved
    ik_target.on_update(update_trajectory)

    # Initial solve
    update_trajectory()

    # 5. Animation / Playback Loop
    slider = server.gui.add_slider("Playback", min=0, max=20, step=1, initial_value=0)
    playing = server.gui.add_checkbox("Loop Trajectory", initial_value=True)

    counter = 0
    while True:
        if playing.value:
            # Cycle through the current best trajectory
            idx = counter % len(state["traj"])
            urdf_vis.update_cfg(state["traj"][idx])
            slider.value = idx
            counter += 1
        else:
            # Scrub manually with slider
            urdf_vis.update_cfg(state["traj"][slider.value])
        
        time.sleep(0.05)

if __name__ == "__main__":
    main()