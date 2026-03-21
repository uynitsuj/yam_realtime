"""Robot and camera configuration dataclasses for the YAM bimanual sim."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CameraConfig:
    serial: str = ""
    height: int = 480
    width: int = 640
    fps: int = 30
    socket: Optional[str] = None


@dataclass
class RobotConfig:
    root_pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    root_ori: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    init_q: List[float] = field(default_factory=lambda: [0.0] * 7)


@dataclass
class SimConfig:
    """Full system configuration for the MuJoCo YAM sim."""

    cameras: Dict[str, CameraConfig] = field(default_factory=dict)
    robots: Dict[str, RobotConfig] = field(default_factory=dict)


def default_sim_config() -> SimConfig:
    """Default config matching the i2rt bimanual station layout."""
    return SimConfig(
        cameras={
            "top": CameraConfig(height=480, width=640, fps=30),
            "left": CameraConfig(height=480, width=640, fps=30),
            "right": CameraConfig(height=480, width=640, fps=30),
        },
        robots={
            "left": RobotConfig(
                root_pos=[0.0, 0.3, 0.0],
                root_ori=[1.0, 0.0, 0.0, 0.0],
                init_q=[
                    -0.20656902,
                    0.47283894,
                    0.99431604,
                    -0.7043946,
                    -0.30842298,
                    -0.32864118,
                    0.9987507,
                ],
            ),
            "right": RobotConfig(
                root_pos=[0.0, -0.3, 0.0],
                root_ori=[1.0, 0.0, 0.0, 0.0],
                init_q=[
                    0.20160982,
                    0.39005876,
                    1.1182956,
                    -0.8726253,
                    0.13332571,
                    0.42629892,
                    0.9895317,
                ],
            ),
        },
    )


# Symmetric flat init_q — good for interactive IK / visualisation
FLAT_INIT_Q = [0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 1.0]


def flat_init_config() -> SimConfig:
    """Config with both arms in a symmetric upright pose."""
    cfg = default_sim_config()
    for robot in cfg.robots.values():
        robot.init_q = list(FLAT_INIT_Q)
    return cfg
