#!/usr/bin/env python3
"""Visually verify a top-camera extrinsics estimate in viser.

The check: deproject the camera's own depth into the ROBOT BASE frame using the
extrinsics under test, and draw it in the same scene as the URDF arms driven by
live joint states. If the extrinsics are right, the depth cloud's table lands
flat at z=0, the arm mounts sit under the URDF bases, and the moving arms in the
cloud sit inside their URDF meshes. If they are wrong, it is obvious instantly —
which is the point, because reprojection RMS in pixels hides a lot.

The camera frustum is drawn at the estimated pose too, so the viewpoint itself
can be sanity-checked against the room.

Nothing here commands the robot. Cameras are opened read-only, joint states are
read off the bus.

Usage
-----
  # autodetect camera_top from the session YAML, load the solved extrinsics
  uv run scripts/verify_top_cam_extrinsics.py \
      --extrinsics /path/to/top_cam_extrinsics.npz

  # explicit serial, static arm pose (no session running)
  uv run scripts/verify_top_cam_extrinsics.py \
      --device-id 427622273855 --joint-source static

  # take depth off the bus instead of opening the device (needs the session's
  # camera_top CameraNode to be configured with `enable_depth: true`)
  uv run scripts/verify_top_cam_extrinsics.py --depth-source bus

Then open http://localhost:8081. Nudge the pose with the GUI sliders until the
cloud lines up, and hit "save extrinsics" to write the corrected values.

Extrinsics file formats
-----------------------
  .npz   keys R_cam_world (3x3) and t_cam_world (3,), the OpenCV WORLD->CAMERA
         convention: p_cam = R @ p_world + t. This is what cv2.solvePnP returns.
  .yaml  the repo convention (see configs/camera_extrinsics/): `position` and
         `rpy_radians` giving the CAMERA->WORLD pose. This is what gets saved.

Both are normalised internally to camera->world.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_SESSION = REPO / "configs/yam/yam_bimanual_openpi_policy_xdof_hq_us07_yam_box_abc.yaml"

logger = logging.getLogger("verify_extrinsics")


# --------------------------------------------------------------------------- #
# rotation helpers
# --------------------------------------------------------------------------- #
def rpy_from_matrix(R: np.ndarray) -> np.ndarray:
    """Invert ``viser.transforms.SO3.from_rpy_radians`` numerically.

    The repo's extrinsics YAMLs are consumed with ``SO3.from_rpy_radians(*rpy)``
    (see ZedCamera._load_extrinsics), so this must be that function's exact
    inverse. Rather than hard-code an Euler convention and hope, try the
    plausible ones and keep whichever round-trips — then assert it did.
    """
    import viser.transforms as vtf

    best = None
    for order in ("zyx", "xyz"):
        if order == "zyx":
            # R = Rz(y) Ry(p) Rx(r)
            p = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
            if abs(np.cos(p)) > 1e-8:
                r = np.arctan2(R[2, 1], R[2, 2])
                y = np.arctan2(R[1, 0], R[0, 0])
            else:  # gimbal lock
                r, y = np.arctan2(-R[1, 2], R[1, 1]), 0.0
        else:
            p = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
            if abs(np.cos(p)) > 1e-8:
                r = np.arctan2(-R[1, 2], R[2, 2])
                y = np.arctan2(-R[0, 1], R[0, 0])
            else:
                r, y = np.arctan2(R[2, 1], R[1, 1]), 0.0
        rpy = np.array([r, p, y])
        err = np.abs(vtf.SO3.from_rpy_radians(*rpy).as_matrix() - R).max()
        if best is None or err < best[0]:
            best = (err, rpy)
    assert best is not None and best[0] < 1e-6, (
        f"could not invert from_rpy_radians (residual {best[0]:.2e}) — "
        "the saved YAML would not round-trip, refusing to guess"
    )
    return best[1]


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    import viser.transforms as vtf

    return np.asarray(vtf.SO3.from_matrix(R).wxyz, dtype=np.float64)


def rot_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Small-angle-friendly Rz @ Ry @ Rx, used for the GUI nudge deltas."""
    cx, sx, cy, sy, cz, sz = (
        np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    )
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# --------------------------------------------------------------------------- #
# config / extrinsics IO
# --------------------------------------------------------------------------- #
def load_session(path: Path) -> dict:
    if not path.exists():
        logger.warning("session YAML not found: %s (falling back to defaults)", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_node(session: dict, ntype: str, name: str | None = None) -> dict | None:
    for n in session.get("nodes", []) or []:
        if n.get("type") == ntype and (name is None or n.get("name") == name):
            return n
    return None


def autodetect_device(session: dict, camera_node: str) -> str:
    """Prefer the session YAML's device_id; else the sole attached RealSense."""
    node = find_node(session, "CameraNode", camera_node)
    if node and node.get("device_id"):
        dev = str(node["device_id"])
        logger.info("camera %s -> serial %s (from session YAML)", camera_node, dev)
        return dev
    import pyrealsense2 as rs

    serials = [d.get_info(rs.camera_info.serial_number) for d in rs.context().devices]
    if len(serials) == 1:
        logger.info("single RealSense attached -> serial %s", serials[0])
        return serials[0]
    raise SystemExit(
        f"could not autodetect the top camera. Attached serials: {serials}. "
        "Pass --device-id explicitly."
    )


def load_extrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_cw, t_cw): p_world = R_cw @ p_cam + t_cw."""
    if path.suffix == ".npz":
        z = np.load(path)
        R = np.asarray(z["R_cam_world"], float)     # world -> cam
        t = np.asarray(z["t_cam_world"], float).reshape(3)
        return R.T, -R.T @ t
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    import viser.transforms as vtf

    R_cw = np.asarray(vtf.SO3.from_rpy_radians(*d["rpy_radians"]).as_matrix(), float)
    return R_cw, np.asarray(d["position"], float)


def save_extrinsics(path: Path, R_cw: np.ndarray, t_cw: np.ndarray, note: str) -> None:
    rpy = rpy_from_matrix(R_cw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# Top-camera extrinsics, CAMERA->WORLD in the robot base frame.\n"
            "#\n"
            "# position:    [x, y, z] metres, camera origin in the base frame\n"
            "# rpy_radians: [roll, pitch, yaw], consumed via SO3.from_rpy_radians\n"
            f"#\n# {note}\n\n"
        )
        yaml.safe_dump(
            {"position": [float(v) for v in t_cw],
             "rpy_radians": [float(v) for v in rpy]},
            f, default_flow_style=False, sort_keys=False,
        )
    logger.info("wrote %s", path)


# --------------------------------------------------------------------------- #
# depth sources
# --------------------------------------------------------------------------- #
class DeviceDepth:
    """Open the RealSense directly (read-only). Depth aligned to colour."""

    def __init__(self, serial: str, width: int, height: int, fps: int):
        import pyrealsense2 as rs

        self._rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            self.profile = self.pipe.start(cfg)
        except RuntimeError as exc:
            raise SystemExit(
                f"could not open RealSense {serial}: {exc}\n"
                "If a session is already streaming this camera, either stop it or "
                "use --depth-source bus (needs `enable_depth: true` on the "
                "camera_top CameraNode)."
            ) from exc
        self.scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        i = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], float)
        self.size = (i.width, i.height)
        self._spatial = rs.spatial_filter()

    def read(self) -> tuple[np.ndarray, np.ndarray] | None:
        frames = self.align.process(self.pipe.wait_for_frames())
        d, c = frames.get_depth_frame(), frames.get_color_frame()
        if not d or not c:
            return None
        depth = np.asanyarray(self._spatial.process(d).get_data()).astype(np.float32) * self.scale
        color = np.asanyarray(c.get_data())[:, :, ::-1].copy()          # BGR -> RGB
        return depth, color

    def close(self) -> None:
        try:
            self.pipe.stop()
        except Exception:
            pass


class BusDepth:
    """Take depth + rgb off the running session's ZMQ bus."""

    def __init__(self, node: str, serial: str):
        from robots_realtime.runtime.transport.subscriber import Subscriber

        self.topics = [f"{node}/depth", f"{node}/rgb", f"{node}/camera_info"]
        self.sub = Subscriber(self.topics)
        self.K: np.ndarray | None = None
        self.size = (0, 0)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and self.K is None:
            self._refresh_intrinsics()
            time.sleep(0.2)
        if self.K is None:
            raise SystemExit(
                f"no camera_info on {self.topics[2]} after 10 s — is the session running, "
                "and does the camera_top CameraNode have `enable_depth: true`?"
            )

    def _refresh_intrinsics(self) -> None:
        info = self.sub.get_data(self.topics[2]) if hasattr(self.sub, "get_data") else None
        if not info:
            return
        intr = (info.get("intrinsics") or {}) if isinstance(info, dict) else {}
        m = intr.get("intrinsics_matrix")
        if m is not None:
            self.K = np.asarray(m, float).reshape(3, 3)

    def read(self) -> tuple[np.ndarray, np.ndarray] | None:
        d = self.sub.get_data(self.topics[0])
        c = self.sub.get_data(self.topics[1])
        if not d or not c:
            return None
        depth = np.asarray(d.get("depth_data"), np.float32)
        if depth.dtype != np.float32 or depth.max() > 100:              # raw z16 millimetres
            depth = depth.astype(np.float32) * 1e-3
        color = np.asarray((c.get("images") or {}).get("rgb"))
        return depth, color

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# joint sources
# --------------------------------------------------------------------------- #
class BusJoints:
    def __init__(self, arms: dict[str, str]):
        from robots_realtime.runtime.transport.subscriber import Subscriber

        self.arms = arms
        self.sub = Subscriber(list(arms.values()))

    def read(self) -> dict[str, np.ndarray]:
        out = {}
        for arm, topic in self.arms.items():
            msg = self.sub.get_data(topic)
            if msg and "joint_pos" in msg:
                out[arm] = np.asarray(msg["joint_pos"], float).reshape(-1)
        return out


class StaticJoints:
    def __init__(self, poses: dict[str, np.ndarray]):
        self.poses = poses

    def read(self) -> dict[str, np.ndarray]:
        return dict(self.poses)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-yaml", type=Path, default=DEFAULT_SESSION,
                    help="session config used to autodetect the camera + URDFs")
    ap.add_argument("--camera-node", default="camera_top", help="CameraNode name to verify")
    ap.add_argument("--device-id", default=None, help="RealSense serial (overrides autodetect)")
    ap.add_argument("--extrinsics", type=Path,
                    default=REPO / "configs/camera_extrinsics/us07_yam_top_d405.yaml",
                    help=".npz (world->cam, solvePnP convention) or .yaml (cam->world). "
                         "Pass 'none' to start from identity and align by hand.")
    ap.add_argument("--save-to", type=Path,
                    default=REPO / "configs/camera_extrinsics/us07_yam_top_d405.yaml",
                    help="where the 'save extrinsics' button writes")
    ap.add_argument("--depth-source", choices=("device", "bus"), default="device")
    ap.add_argument("--joint-source", choices=("bus", "static"), default="bus")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--rate", type=float, default=10.0, help="scene update Hz")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import viser
    import viser.extras
    import yourdfpy

    session = load_session(args.session_yaml)
    serial = args.device_id or autodetect_device(session, args.camera_node)

    # ---- extrinsics under test ------------------------------------------- #
    if args.extrinsics and str(args.extrinsics).lower() != "none" and args.extrinsics.exists():
        R0, t0 = load_extrinsics(args.extrinsics)
        logger.info("loaded extrinsics from %s", args.extrinsics)
    else:
        R0, t0 = np.eye(3), np.zeros(3)
        logger.warning("no --extrinsics given; starting from identity")
    logger.info("camera position in base frame: [%+.3f %+.3f %+.3f] m", *t0)

    # ---- sources ---------------------------------------------------------- #
    depth_src: Any = (
        DeviceDepth(serial, args.width, args.height, args.fps)
        if args.depth_source == "device"
        else BusDepth(args.camera_node, serial)
    )

    viz = find_node(session, "ViserMonitorNode") or {}
    urdf_specs: dict[str, dict] = (viz.get("urdfs") or {})
    if not urdf_specs:
        urdf_specs = {
            "yam_left": {"path": "dependencies/i2rt/i2rt/robot_models/arm/yam/yam.urdf",
                         "state_topic": "yam_left/joint_state", "flip_joints": True},
            "yam_right": {"path": "dependencies/i2rt/i2rt/robot_models/arm/yam/yam.urdf",
                          "state_topic": "yam_right/joint_state", "flip_joints": True,
                          "extrinsic": {"position": [0.0, -0.61, 0.0]}},
        }

    if args.joint_source == "bus":
        joints: Any = BusJoints({k: v["state_topic"] for k, v in urdf_specs.items()})
    else:
        static = {}
        for arm in urdf_specs:
            node = find_node(session, "RobotNode", arm)
            q = (node or {}).get("startup_joint_pos") or [0.0] * 6
            static[arm] = np.asarray(q, float)[:6]
        joints = StaticJoints(static)
        logger.info("static joint pose: %s", {k: np.round(v, 3).tolist() for k, v in static.items()})

    # ---- scene ------------------------------------------------------------ #
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.1,
                          position=(0.0, -0.3, 0.0))
    server.scene.add_frame("/base", axes_length=0.15, axes_radius=0.005)

    urdfs: dict[str, Any] = {}
    for arm, spec in urdf_specs.items():
        path = REPO / spec["path"]
        if not path.is_file():
            raise SystemExit(f"URDF for {arm!r} not found at {path}")
        # The i2rt URDFs reference meshes as package://assets/*.stl; yourdfpy needs
        # to be told where that is. Same default as ViserMonitorNode: the sibling
        # assets/ dir next to the URDF.
        mesh_dir = spec.get("mesh_dir")
        mesh_dir = Path(mesh_dir).expanduser() if mesh_dir else path.parent / "assets"
        ext = spec.get("extrinsic") or {}
        root = f"/{arm}"
        server.scene.add_frame(
            root, show_axes=False,
            position=tuple(float(v) for v in ext.get("position", (0, 0, 0))),
            wxyz=tuple(float(v) for v in ext.get("rotation", (1, 0, 0, 0))),
        )
        urdf = (yourdfpy.URDF.load(str(path), mesh_dir=str(mesh_dir))
                if mesh_dir.is_dir() else yourdfpy.URDF.load(str(path)))
        vis = viser.extras.ViserUrdf(server, urdf, root_node_name=root)
        n_act = len(urdf.actuated_joint_names)
        # ViserUrdf leaves every link at the origin until the first update_cfg,
        # which renders as a collapsed arm. Seed it immediately so the scene is
        # never wrong-looking, even before any joint state arrives.
        vis.update_cfg(np.zeros(n_act))
        urdfs[arm] = dict(vis=vis, flip=bool(spec.get("flip_joints", True)), n=n_act)
        missing = len(urdf.link_map) - len(urdf.scene.geometry)
        logger.info("loaded URDF %s at %s (%d/%d link meshes%s)", arm,
                    ext.get("position", (0, 0, 0)), len(urdf.scene.geometry),
                    len(urdf.link_map),
                    f"; {missing} MISSING from {mesh_dir}" if missing else "")

    # ---- GUI -------------------------------------------------------------- #
    with server.gui.add_folder("pose nudge (applied on top of the estimate)"):
        g_dx = server.gui.add_slider("dx (m)", -0.30, 0.30, 0.002, 0.0)
        g_dy = server.gui.add_slider("dy (m)", -0.30, 0.30, 0.002, 0.0)
        g_dz = server.gui.add_slider("dz (m)", -0.30, 0.30, 0.002, 0.0)
        g_rr = server.gui.add_slider("roll (deg)", -20.0, 20.0, 0.1, 0.0)
        g_rp = server.gui.add_slider("pitch (deg)", -20.0, 20.0, 0.1, 0.0)
        g_ry = server.gui.add_slider("yaw (deg)", -20.0, 20.0, 0.1, 0.0)
        g_reset = server.gui.add_button("reset nudge")
        g_save = server.gui.add_button("save extrinsics")
    with server.gui.add_folder("point cloud"):
        g_show = server.gui.add_checkbox("show cloud", True)
        g_color = server.gui.add_dropdown("colour by", ("rgb", "height z"), initial_value="rgb")
        g_stride = server.gui.add_slider("stride", 1, 6, 1, 2)
        g_zmin = server.gui.add_slider("depth min (m)", 0.1, 2.0, 0.01, 0.25)
        g_zmax = server.gui.add_slider("depth max (m)", 0.2, 5.0, 0.01, 1.60)
        g_size = server.gui.add_slider("point size", 0.001, 0.02, 0.0005, 0.004)
        g_frustum = server.gui.add_checkbox("show frustum", True)
    g_status = server.gui.add_text("status", "starting…", disabled=True)

    @g_reset.on_click
    def _(_) -> None:
        for s in (g_dx, g_dy, g_dz, g_rr, g_rp, g_ry):
            s.value = 0.0

    def current_pose() -> tuple[np.ndarray, np.ndarray]:
        """Estimate composed with the GUI nudge, in the WORLD frame."""
        dR = rot_xyz(np.deg2rad(g_rr.value), np.deg2rad(g_rp.value), np.deg2rad(g_ry.value))
        return dR @ R0, dR @ t0 + np.array([g_dx.value, g_dy.value, g_dz.value])

    @g_save.on_click
    def _(_) -> None:
        R_cw, t_cw = current_pose()
        save_extrinsics(
            args.save_to, R_cw, t_cw,
            f"camera_node={args.camera_node} serial={serial}; "
            f"nudge dxyz=({g_dx.value:+.3f},{g_dy.value:+.3f},{g_dz.value:+.3f}) m "
            f"drpy=({g_rr.value:+.1f},{g_rp.value:+.1f},{g_ry.value:+.1f}) deg",
        )
        g_status.value = f"saved -> {args.save_to}"

    # ---- persistent handles ------------------------------------------------ #
    # PointCloudHandle exposes no settable points in viser 1.0.x, so the cloud is
    # re-added by name each frame (viser replaces it). The frustum does expose
    # position/wxyz/image, so it is created once and mutated.
    h_frustum = server.scene.add_camera_frustum(
        "/top_cam",
        fov=2 * np.arctan(depth_src.size[1] / 2 / depth_src.K[1, 1]),
        aspect=depth_src.size[0] / depth_src.size[1],
        scale=0.12, color=(255, 190, 60),
        wxyz=quat_from_matrix(R0), position=tuple(t0),
    )
    h_axes = server.scene.add_frame("/top_cam_axes", axes_length=0.08, axes_radius=0.003,
                                    wxyz=quat_from_matrix(R0), position=tuple(t0))
    h_cloud: Any = None

    # ---- loop ------------------------------------------------------------- #
    logger.info("viser up on http://localhost:%d", args.port)
    stop = threading.Event()
    uu = vv = None
    last = 0.0
    t_start = time.monotonic()
    warned_no_joints = False
    try:
        while not stop.is_set():
            t_loop = time.monotonic()

            live = joints.read()
            if not live and not warned_no_joints and time.monotonic() - t_start > 5.0:
                warned_no_joints = True
                logger.warning(
                    "no joint states after 5 s (source=%s) — the arms are drawn at the "
                    "zero pose and will NOT match the depth cloud. Start a session, or "
                    "re-run with --joint-source static.", args.joint_source)
                g_status.value = "NO JOINT DATA — arms drawn at zero pose"
            for arm, u in urdfs.items():
                q = live.get(arm)
                if q is None:
                    continue
                cfg = np.asarray(q, float)[:u["n"]]
                if u["flip"]:
                    cfg = np.flip(cfg)
                try:
                    u["vis"].update_cfg(cfg)
                except Exception as exc:                      # noqa: BLE001
                    logger.debug("urdf %s update failed: %s", arm, exc)

            R_cw, t_cw = current_pose()
            q_cw = quat_from_matrix(R_cw)
            h_frustum.visible = h_axes.visible = bool(g_frustum.value)
            if g_frustum.value:
                h_frustum.position = h_axes.position = tuple(float(v) for v in t_cw)
                h_frustum.wxyz = h_axes.wxyz = q_cw

            got = depth_src.read()
            if got is not None and g_show.value:
                depth, color = got
                s = int(g_stride.value)
                d = depth[::s, ::s]
                if uu is None or uu.shape != d.shape:
                    H, W = depth.shape
                    gy, gx = np.mgrid[0:H:s, 0:W:s]
                    K = depth_src.K
                    uu = (gx - K[0, 2]) / K[0, 0]
                    vv = (gy - K[1, 2]) / K[1, 1]
                m = np.isfinite(d) & (d > g_zmin.value) & (d < g_zmax.value)
                z = d[m]
                pts_cam = np.stack([uu[m] * z, vv[m] * z, z], 1)
                pts = pts_cam @ R_cw.T + t_cw
                if g_color.value == "rgb" and color is not None and color.shape[:2] == depth.shape:
                    cols = color[::s, ::s][m]
                else:
                    zz = pts[:, 2]
                    lo, hi = -0.05, 0.45
                    tt = np.clip((zz - lo) / (hi - lo), 0, 1)
                    cols = (np.stack([tt, 0.45 * np.ones_like(tt), 1 - tt], 1) * 255).astype(np.uint8)
                h_cloud = server.scene.add_point_cloud(
                    "/cloud", points=pts.astype(np.float32),
                    colors=np.asarray(cols, np.uint8), point_size=float(g_size.value))
                if color is not None and g_frustum.value:
                    h_frustum.image = color[::4, ::4]
                if time.monotonic() - last > 0.5:
                    last = time.monotonic()
                    on_table = np.abs(pts[:, 2]) < 0.02
                    g_status.value = (
                        f"{len(pts):,} pts | cam [{t_cw[0]:+.3f} {t_cw[1]:+.3f} {t_cw[2]:+.3f}] m | "
                        f"{100*on_table.mean():.0f}% within 2 cm of z=0"
                    )
            elif h_cloud is not None:
                h_cloud.visible = bool(g_show.value)

            time.sleep(max(0.0, 1.0 / args.rate - (time.monotonic() - t_loop)))
    except KeyboardInterrupt:
        logger.info("bye")
    finally:
        depth_src.close()


if __name__ == "__main__":
    main()
