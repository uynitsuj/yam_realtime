import logging
import time
from pathlib import Path
from typing import Any, List, Literal

import numpy as np
import pyroki as pk
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tyro

# Assuming these are available in your environment as per your examples
import robots_realtime.robots.inverse_kinematics.pyroki_snippets as pks
from functools import lru_cache
import json

# =====================================================
# Cached Solver Factory
# =====================================================

@lru_cache(maxsize=10)
def get_compiled_solver(timesteps: int, dt: float, num_obstacles: int):
    """
    This function will be cached. It returns a JIT-compiled 
    trajectory optimization function specific to these dimensions.
    """
    logger.info(f"JIT Compiling TrajOpt for: "
                f"timesteps={timesteps}, obstacles={num_obstacles}")
    
    # We return the solver function itself. 
    # Note: pks.solve_trajopt usually handles JIT internally, 
    # but we ensure the robot/coll objects are consistent.
    return pks.solve_trajopt

# =====================================================
# Logging and FastAPI Setup
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyroki_trajopt_server")

app = FastAPI(title="PyRoKi TrajOpt Server")

# Global state for the robot model
_ROBOT = None
_ROBOT_COLL = None
_TARGET_LINK = None

# =====================================================
# Pydantic Schemas
# =====================================================

class ObstacleEntry(BaseModel):
    type: Literal["halfspace", "sphere", "capsule", "box"]
    point: List[float] | None = None
    normal: List[float] | None = None
    center: List[float] | None = None
    radius: float | None = None
    position: List[float] | None = None
    height: float | None = None
    extent: List[float] | None = None

class TrajOptRequest(BaseModel):
    start_pose_wxyz_xyz: List[float]  # [w, x, y, z, x, y, z]
    end_pose_wxyz_xyz: List[float]    # [w, x, y, z, x, y, z]
    obstacles: List[ObstacleEntry] | None = None
    timesteps: int = 25
    dt: float = 0.02

class TrajOptResponse(BaseModel):
    trajectory: List[List[float]]
    compute_time: float

# =====================================================
# Obstacle Builder
# =====================================================

def _build_world_coll(obstacles: List[ObstacleEntry] | None) -> List[Any]:
    if not obstacles:
        return []
    
    world = []
    for obj in obstacles:
        if obj.type == "halfspace":
            world.append(pk.collision.HalfSpace.from_point_and_normal(
                np.array(obj.point), np.array(obj.normal)
            ))
        elif obj.type == "sphere":
            world.append(pk.collision.Sphere.from_center_and_radius(
                np.array(obj.center), np.array([obj.radius])
            ))
        elif obj.type == "capsule":
            world.append(pk.collision.Capsule.from_radius_height(
                position=np.array(obj.position),
                radius=np.array([obj.radius]),
                height=np.array([obj.height]),
            ))
        elif obj.type == "box":
            world.append(pk.collision.Box.from_extent(
                extent=np.array(obj.extent),
                position=np.array(obj.position or [0, 0, 0]),
            ))
    return world

# =====================================================
# Initialization
# =====================================================

def init_server(robot_name: str, target_link_name: str):
    global _ROBOT, _ROBOT_COLL, _TARGET_LINK
    
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    
    logger.info(f"Initializing {robot_name}...")
    urdf = load_robot_description(f"{robot_name}_description")
    
    # Initialize Robot
    _ROBOT = pk.Robot.from_urdf(urdf)
    
    # Initialize Collision Model
    # Note: You can swap this for sphere decomposition if you have the JSON
    # _ROBOT_COLL = pk.collision.RobotCollision.from_urdf(urdf)
    sphere_json_path = Path(__file__).parent / "assets" / "panda_spheres.json"


    with open(sphere_json_path, "r") as f:
        sphere_decomposition = json.load(f)
    _ROBOT_COLL = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_decomposition,
        urdf=urdf,
    )
    
    _TARGET_LINK = target_link_name
    logger.info("Server Ready.")

# =====================================================
# Endpoints
# =====================================================
@app.post("/solve", response_model=TrajOptResponse)
async def solve_trajectory(req: TrajOptRequest):
    if _ROBOT is None:
        raise HTTPException(503, "Robot not initialized")

    start_wxyz = np.array(req.start_pose_wxyz_xyz[:4])
    start_pos = np.array(req.start_pose_wxyz_xyz[4:])
    end_wxyz = np.array(req.end_pose_wxyz_xyz[:4])
    end_pos = np.array(req.end_pose_wxyz_xyz[4:])
    
    world_coll = _build_world_coll(req.obstacles)
    
    # 1. Access the solver. The lru_cache handles the overhead.
    # We include num_obstacles because changing the list length triggers re-JIT.
    solver = get_compiled_solver(
        req.timesteps, 
        req.dt, 
        len(world_coll), 
    )

    try:
        start_time = time.time()
        
        # 2. Call the solver. 
        # After the first call for a specific config, this will be much faster.
        traj = solver(
            robot=_ROBOT,
            robot_coll=_ROBOT_COLL,
            world_coll=world_coll,
            target_link_name=_TARGET_LINK,
            start_position=start_pos,
            start_wxyz=start_wxyz,
            end_position=end_pos,
            end_wxyz=end_wxyz,
            timesteps=req.timesteps,
            dt=req.dt,
        )
        
        # Ensure it's a numpy array for serialization
        traj_np = np.array(traj)
        
        duration = time.time() - start_time
        return TrajOptResponse(
            trajectory=traj_np.tolist(),
            compute_time=duration
        )

    except Exception as e:
        logger.exception("Trajectory optimization failed")
        raise HTTPException(500, f"Solver error: {str(e)}")

# =====================================================
# Main
# =====================================================

def main(
    robot: str = "panda",
    target_link: str = "panda_hand",
    port: int = 8116,
    host: str = "127.0.0.1",
):
    init_server(robot, target_link)
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    tyro.cli(main)