import time
import numpy as np
import requests
import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description

def main():
    # 1. Setup Viser and Robot
    server = viser.ViserServer()
    robot_name = "panda" # Ensure this matches your server
    urdf = load_robot_description(f"{robot_name}_description")
    urdf_vis = ViserUrdf(server, urdf)
    
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)

    # 2. Define Problem (matches the example "over the wall" scenario)
    start_pos = [0.5, -0.3, 0.2]
    end_pos = [0.5, 0.3, 0.2]
    down_wxyz = [0.0, 0.0, 1.0, 0.0] # Panda downward orientation
    
    # Define the wall obstacle for the server and for local visualization
    wall_config = {
        "type": "box",
        "extent": [0.4, 0.1, 0.4],
        "position": [0.5, 0.0, 0.2]
    }

    # Add visual markers to the scene
    for name, pos in zip(["start", "end"], [start_pos, end_pos]):
        server.scene.add_frame(f"/{name}", position=np.array(pos), wxyz=np.array(down_wxyz), axes_length=0.05)
    
    # Add the wall to the visualizer
    # import trimesh
    # server.scene.add_mesh_simple(
    #     "/wall",
    #     mesh=trimesh.creation.box(extents=wall_config["extent"]),
    #     position=np.array(wall_config["position"]),
    #     color=(200, 50, 50),
    # )

    # 3. Request Trajectory from Server
    payload = {
        "start_pose_wxyz_xyz": down_wxyz + start_pos,
        "end_pose_wxyz_xyz": down_wxyz + end_pos,
        "obstacles": [],
        "timesteps": 10,
        "dt": 0.1
    }

    print("Requesting trajectory from server...")
    try:
        response = requests.post("http://127.0.0.1:8116/solve", json=payload)
        response.raise_for_status()
        data = response.json()
        traj = np.array(data["trajectory"])
        print(f"Trajectory received! Solve time: {data['compute_time']:.4f}s")
    except Exception as e:
        print(f"Failed to get trajectory: {e}")
        return

    print("Requesting trajectory from server again...")
    try:
        response = requests.post("http://127.0.0.1:8116/solve", json=payload)
        response.raise_for_status()
        data = response.json()
        traj = np.array(data["trajectory"])
        print(f"Trajectory received! Solve time: {data['compute_time']:.4f}s")
    except Exception as e:
        print(f"Failed to get trajectory: {e}")
        return

    # 4. Playback Loop
    timesteps = len(traj)
    slider = server.gui.add_slider("Timestep", min=0, max=timesteps - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps
        
        urdf_vis.update_cfg(traj[slider.value])
        time.sleep(0.05)

if __name__ == "__main__":
    main()