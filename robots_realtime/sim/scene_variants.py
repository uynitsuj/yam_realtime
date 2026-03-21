"""Runtime scene configuration for MuJoCo YAM simulation.

Applies visual changes (colors, visibility) to a loaded MuJoCo model without
needing separate XML files per variant. Call apply_scene_variant() after
loading the model but before the first render.

Scene variants:
    eval     — cage walls + blue-grey bin
    training — no walls + orange bucket
    hybrid   — cage walls + orange bucket (default)
"""

from __future__ import annotations

import mujoco
import numpy as np

OFF_WHITE = np.array([0.95, 0.93, 0.96, 1.0], dtype=np.float32)
BRIGHT_WHITE = np.array([0.97, 0.97, 0.97, 1.0], dtype=np.float32)
HIDDEN = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
DARK_BLUE_GREY = np.array([0.15, 0.25, 0.35, 1.0], dtype=np.float32)
ORANGE_BUCKET = np.array([0.95, 0.55, 0.15, 1.0], dtype=np.float32)

_WALL_GEOMS = ["back_wall", "left_wall", "right_wall"]
_BIN_GEOMS = ["bin_bottom"] + [f"bin_wall_{a}" for a in range(0, 360, 30)]

VARIANTS = {
    "eval": {
        "floor_rgba": OFF_WHITE,
        "table_rgba": BRIGHT_WHITE,
        "wall_rgba": OFF_WHITE,
        "bin_rgba": DARK_BLUE_GREY,
        "gate_rgba": np.array([0.85, 0.85, 0.85, 1.0]),
    },
    "training": {
        "floor_rgba": OFF_WHITE,
        "table_rgba": BRIGHT_WHITE,
        "wall_rgba": HIDDEN,
        "bin_rgba": ORANGE_BUCKET,
        "gate_rgba": np.array([0.85, 0.85, 0.85, 1.0]),
    },
    "hybrid": {
        "floor_rgba": OFF_WHITE,
        "table_rgba": BRIGHT_WHITE,
        "wall_rgba": OFF_WHITE,
        "bin_rgba": ORANGE_BUCKET,
        "gate_rgba": np.array([0.85, 0.85, 0.85, 1.0]),
    },
}


def apply_scene_variant(model: mujoco.MjModel, variant: str = "hybrid") -> None:
    """Apply a named scene variant to a loaded MuJoCo model."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(VARIANTS)}")
    v = VARIANTS[variant]

    def _set(name: str, rgba: np.ndarray) -> None:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            model.geom_rgba[gid] = rgba

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_matid[floor_id] = -1
        model.geom_rgba[floor_id] = v["floor_rgba"]

    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "play_table")
    if table_body_id >= 0:
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == table_body_id:
                model.geom_rgba[gid] = v["table_rgba"]

    for name in _WALL_GEOMS:
        _set(name, v["wall_rgba"])
    for name in _BIN_GEOMS:
        _set(name, v["bin_rgba"])

    gate_mesh_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_MESH, "base_visual_gate"
    )
    if gate_mesh_id >= 0:
        for gid in range(model.ngeom):
            if model.geom_dataid[gid] == gate_mesh_id:
                model.geom_rgba[gid] = v["gate_rgba"]
                break


def apply_bottle_rgba(model: mujoco.MjModel, rgba: tuple[float, ...]) -> None:
    color = np.array(rgba, dtype=np.float32)
    for i in range(1, 7):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"b{i}_body")
        if gid >= 0:
            model.geom_rgba[gid] = color


def apply_bin_position(
    model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float
) -> None:
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bin_joint")
    if jnt_id >= 0:
        addr = model.jnt_qposadr[jnt_id]
        data.qpos[addr] = x
        data.qpos[addr + 1] = y


def list_variants() -> list[str]:
    return list(VARIANTS)
