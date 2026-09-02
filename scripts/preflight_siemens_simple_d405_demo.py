#!/usr/bin/env python3
"""Read-only preflight for the Siemens simple-D405 policy demo."""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/yam/yam_bimanual_openpi_policy_xdof_hq_us05_siemens_simple_d405.yaml")
DEFAULT_OPENPI = Path("/nfs_us_2/karim/worktrees/openpi-industrial-packing-v3")
DEFAULT_CHECKPOINT = Path(
    "/nfs_us_2/siemens/policy_ckpts/pi05_siemens_simple_d405_bs128/siemens_simple_d405_pi05_20260901/14999"
)
EXPECTED_CAMERAS = {
    "camera_top": "427622273494",
    "camera_left": "427622271411",
    "camera_right": "427622273554",
}


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"PASS  {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"FAIL  {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"WARN  {message}")

    def require(self, condition: bool, message: str) -> None:
        (self.ok if condition else self.fail)(message)


def _one(nodes: list[dict[str, Any]], type_name: str, report: Report) -> dict[str, Any]:
    matches = [node for node in nodes if node.get("type") == type_name]
    report.require(len(matches) == 1, f"exactly one {type_name} is configured")
    return matches[0] if len(matches) == 1 else {}


def check_config(config_path: Path, report: Report) -> tuple[str, int]:
    if not config_path.is_file():
        report.fail(f"session config exists: {config_path}")
        return "127.0.0.1", 8012

    config = yaml.safe_load(config_path.read_text())
    report.require(config.get("version") == "1", "session config schema version is 1")
    session = config.get("session", {})
    report.require(session.get("start_paused") is True, "session starts paused")
    report.require(session.get("record_on_unpause") is True, "recording starts on unpause")

    nodes: list[dict[str, Any]] = config.get("nodes", [])
    agent = _one(nodes, "AgentNode", report)
    kwargs = agent.get("agent_kwargs", {})
    expected_kwargs = {
        "action_horizon": 30,
        "inference_mode": "sync",
        "image_preprocess": "pad",
        "use_joint_state_as_action": False,
        "prompt": "industrial packing",
        "flip_joint_order": True,
    }
    for key, expected in expected_kwargs.items():
        report.require(kwargs.get(key) == expected, f"agent {key}={expected!r}")

    camera_nodes = {node.get("name"): node for node in nodes if node.get("type") == "CameraNode"}
    report.require(
        set(camera_nodes) == set(EXPECTED_CAMERAS), "exactly three expected D405 camera nodes are configured"
    )
    for name, serial in EXPECTED_CAMERAS.items():
        camera = camera_nodes.get(name, {})
        expected = {
            "driver": "RealSenseCamera",
            "device_id": serial,
            "resolution": "VGA",
            "fps": 30,
            "publish_resize": [224, 224],
            "publish_resize_mode": "pad",
            "publish_fov_crop": 1.0,
        }
        report.require(
            all(camera.get(key) == value for key, value in expected.items()),
            f"{name} matches the D405 training image contract",
        )

    root = config_path.resolve().parents[2]
    for robot in (node for node in nodes if node.get("type") == "RobotNode"):
        path = root / robot.get("robot_config", "")
        report.require(path.is_file(), f"robot config exists: {path.relative_to(root)}")

    monitor = _one(nodes, "ViserMonitorNode", report)
    for urdf in monitor.get("urdfs", {}).values():
        path = root / urdf.get("path", "")
        report.require(path.is_file(), f"URDF exists: {path.relative_to(root)}")

    return str(kwargs.get("ip", "127.0.0.1")), int(kwargs.get("port", 8012))


def check_openpi(openpi_root: Path, report: Report) -> None:
    config_py = openpi_root / "src/openpi/training/config.py"
    report.require(config_py.is_file(), f"matching OpenPI worktree exists: {openpi_root}")
    if config_py.is_file():
        text = config_py.read_text()
        report.require(
            'name="pi05_siemens_simple_d405_bs128"' in text,
            "OpenPI contains pi05_siemens_simple_d405_bs128",
        )
        report.require(
            'repo_id="siemens_simple_d405"' in text,
            "OpenPI config selects the Siemens simple-D405 dataset statistics",
        )


def check_checkpoint(checkpoint: Path, report: Report) -> None:
    required = (
        "_CHECKPOINT_METADATA",
        "params/_METADATA",
        "params/manifest.ocdbt",
        "assets/siemens_simple_d405/norm_stats.json",
    )
    for relative in required:
        report.require((checkpoint / relative).is_file(), f"checkpoint contains {relative}")
    param_files = [path for path in (checkpoint / "params").rglob("*") if path.is_file()]
    param_bytes = sum(path.stat().st_size for path in param_files)
    report.require(
        param_bytes > 1_000_000_000, f"checkpoint parameter payload is present ({param_bytes / 1e9:.1f} GB)"
    )


def check_server(host: str, port: int, *, required: bool, report: Report) -> None:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            report.ok(f"inference server is reachable at {host}:{port}")
    except OSError as exc:
        message = f"inference server is not reachable at {host}:{port}: {exc}"
        (report.fail if required else report.warn)(message)


def check_hardware(*, required: bool, report: Report) -> None:
    missing_can = [name for name in ("can_left", "can_right") if not (Path("/sys/class/net") / name).exists()]
    if missing_can:
        message = f"missing CAN interfaces: {', '.join(missing_can)}"
        (report.fail if required else report.warn)(message)
    else:
        report.ok("can_left and can_right interfaces exist")

    try:
        import pyrealsense2 as rs  # noqa: PLC0415
    except ImportError:
        message = "pyrealsense2 is not installed in this environment"
        (report.fail if required else report.warn)(message)
        return

    try:
        context = rs.context()
        connected = {device.get_info(rs.camera_info.serial_number) for device in context.query_devices()}
    except RuntimeError as exc:
        message = f"unable to enumerate RealSense devices: {exc}"
        (report.fail if required else report.warn)(message)
        return
    missing_cameras = sorted(set(EXPECTED_CAMERAS.values()) - connected)
    if missing_cameras:
        message = f"configured D405 serials are not connected: {', '.join(missing_cameras)}"
        (report.fail if required else report.warn)(message)
    else:
        report.ok("all configured D405 serials are connected")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--require-server", action="store_true")
    parser.add_argument("--require-hardware", action="store_true")
    args = parser.parse_args()

    report = Report()
    host, port = check_config(args.config, report)
    check_openpi(args.openpi_root, report)
    check_checkpoint(args.checkpoint, report)
    check_server(host, port, required=args.require_server, report=report)
    check_hardware(required=args.require_hardware, report=report)

    print(f"\nSummary: {len(report.failures)} failure(s), {len(report.warnings)} warning(s)")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
