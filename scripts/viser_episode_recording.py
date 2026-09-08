"""Build a standalone Viser recording (.viser) from a robots_realtime episode.

Reconstructs the bimanual YAM station in 3D from a real-robot episode:

  * both arms rendered from the i2rt YAM **URDF** (``viser.extras.ViserUrdf``),
    driven by the recorded ``/yam_{left,right}/joint_state`` joint angles;
  * the ``left`` / ``right`` wrist D405s as **camera frustums rigidly attached
    to the end effectors** (frustum pose = FK(link_6) ∘ T_link6_cam), each
    textured with the frame that camera actually recorded at that instant;
  * the ``top`` D405 as a static frustum at its calibrated gate-mounted pose;
  * the static cell — gate weldment, side panels, play table — pulled straight
    out of the MuJoCo station model so the geometry and every camera extrinsic
    come from a single source of truth.

All extrinsics are read from the MuJoCo model at
``robots_realtime/sim/models/yam_bimanual_scene.xml`` rather than hardcoded, so
the visualization tracks the sim station definition. Per frame the recorded
joint state is written into ``mjData.qpos`` and MuJoCo FK supplies the camera
and gripper-finger poses; the URDF is posed independently and agrees with
MuJoCo FK to <10 um / <0.001 deg (see ``--self-test``).

Usage
-----
    # one-shot: build the recording, host it, print the playback URL
    uv run scripts/viser_episode_recording.py <episode_dir> --view

    # just write the .viser file
    uv run scripts/viser_episode_recording.py <episode_dir> -o episode.viser

    # live server with a scrub slider (nothing written to disk)
    uv run scripts/viser_episode_recording.py <episode_dir> --serve

    # check URDF FK against MuJoCo FK and dump the extrinsics
    uv run scripts/viser_episode_recording.py --self-test

Hosting a .viser file yourself
------------------------------
``--view`` does this for you, but the pieces are just static files:

    viser-build-client --out-dir viser-client/
    python -m http.server 8000
    # then open:
    #   http://localhost:8000/viser-client/?playbackPath=http://localhost:8000/episode.viser

The same URL form works for static hosting (GitHub Pages, S3, ...) and can be
dropped into an <iframe>. See https://viser.studio/main/embedded_visualizations/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import viser
import viser.extras
import viser.transforms as vtf
import yourdfpy

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MJCF = REPO_ROOT / "robots_realtime" / "sim" / "models" / "yam_bimanual_scene.xml"
DEFAULT_URDF_DIR = REPO_ROOT / "dependencies" / "i2rt" / "i2rt" / "robot_models" / "arm" / "yam"

SIDES = ("left", "right")
CAMERAS = ("top", "left", "right")

# MuJoCo cameras look down -Z with +Y up (OpenGL); viser frustums look down +Z
# with +Y down (OpenCV). 180 deg about the camera X axis converts between them.
T_GL_CV = np.diag([1.0, -1.0, -1.0, 1.0])

# Recorded gripper_pos is normalized [0, 1] (0 = closed, 1 = open). The station
# model drives each jaw with a slide joint; scale into that joint's range.
GRIPPER_JOINT_TRAVEL = 0.0475

# MuJoCo geom groups that make up the visible station. Group 2 is the visual
# layer, group 3 is collision-only, group 0 holds the sim-only task props.
VISUAL_GEOM_GROUPS = (2,)
PROP_GEOM_GROUPS = (0,)


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------


@dataclass
class ArmStream:
    """Recorded proprioception for one arm."""

    ts: np.ndarray  # (N,) unix seconds
    q: np.ndarray  # (N, 6) joint angles, radians
    gripper: np.ndarray  # (N,) normalized [0, 1]


@dataclass
class CameraStream:
    """Decoded (and downscaled) RGB frames for one camera."""

    ts: np.ndarray  # (T,) unix seconds
    frames: np.ndarray  # (T, h, w, 3) uint8


def _read_mcap_json(path: Path, topic: str) -> list[tuple[float, dict]]:
    """Return [(log_time_seconds, payload), ...] for one topic of a JSON MCAP."""
    from mcap.reader import make_reader

    out: list[tuple[float, dict]] = []
    with open(path, "rb") as f:
        for _, channel, msg in make_reader(f).iter_messages():
            if channel.topic == topic:
                out.append((msg.log_time / 1e9, json.loads(msg.data)))
    return out


def load_arm_streams(episode_dir: Path) -> dict[str, ArmStream]:
    """Load ``/yam_{side}/joint_state`` for both arms.

    These are the *measured* joint angles (~200 Hz), which is what we want for
    visualization -- ``openpi_policy.mcap`` holds commanded targets instead.
    """
    streams: dict[str, ArmStream] = {}
    for side in SIDES:
        path = episode_dir / f"yam_{side}.mcap"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; this script needs real-robot joint_state streams")
        msgs = _read_mcap_json(path, f"/yam_{side}/joint_state")
        if not msgs:
            raise RuntimeError(f"{path} has no /yam_{side}/joint_state messages")
        streams[side] = ArmStream(
            ts=np.array([t for t, _ in msgs], dtype=np.float64),
            q=np.array([d["joint_pos"] for _, d in msgs], dtype=np.float64),
            gripper=np.array([d["gripper_pos"][0] for _, d in msgs], dtype=np.float64),
        )
        s = streams[side]
        dur = s.ts[-1] - s.ts[0]
        print(f"  yam_{side}: {len(s.ts)} samples, {dur:.2f}s at ~{(len(s.ts) - 1) / dur:.0f} Hz")
    return streams


def load_camera_streams(
    episode_dir: Path,
    wanted_ts: np.ndarray,
    image_width: int,
) -> dict[str, CameraStream]:
    """Decode only the frames needed for ``wanted_ts``, downscaled to ``image_width``.

    Videos are streamed frame by frame and thrown away unless needed -- decoding
    a full 640x480 episode into memory costs ~1.4 GB per camera.
    """
    import imageio.v3 as iio
    from PIL import Image

    streams: dict[str, CameraStream] = {}
    for cam in CAMERAS:
        mp4 = episode_dir / f"camera_{cam}-images-rgb.mp4"
        npy = episode_dir / f"camera_{cam}-rgb-timestamp.npy"
        if not mp4.exists() or not npy.exists():
            print(f"  camera_{cam}: missing mp4/timestamps, frustum will be untextured")
            continue

        ts = np.load(npy).astype(np.float64)
        keep = np.unique(_nearest_index(ts, wanted_ts))
        keep_set = set(keep.tolist())

        decoded: dict[int, np.ndarray] = {}
        for i, frame in enumerate(iio.imiter(str(mp4), plugin="pyav")):
            if i in keep_set:
                img = Image.fromarray(frame)
                if image_width and img.width != image_width:
                    h = max(1, round(img.height * image_width / img.width))
                    img = img.resize((image_width, h), Image.BILINEAR)
                decoded[i] = np.asarray(img, dtype=np.uint8)
            if i >= keep[-1]:
                break

        # Timestamp arrays and frame counts can disagree by a frame or two; keep
        # the prefix both agree on so index lookups stay in bounds.
        n = min(len(ts), max(decoded) + 1 if decoded else 0)
        stacked = np.stack([decoded[min(i, n - 1)] for i in sorted(decoded)])
        index_of = {orig: new for new, orig in enumerate(sorted(decoded))}
        streams[cam] = CameraStream(ts=ts[:n], frames=stacked)
        streams[cam]._index_of = index_of  # type: ignore[attr-defined]
        print(
            f"  camera_{cam}: {len(ts)} recorded frames, decoded {len(decoded)} "
            f"at {stacked.shape[2]}x{stacked.shape[1]}"
        )
    return streams


def _nearest_index(src_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Index of the nearest ``src_ts`` sample for each ``query_ts`` (clamped)."""
    idx = np.searchsorted(src_ts, query_ts)
    idx = np.clip(idx, 1, len(src_ts) - 1)
    left, right = src_ts[idx - 1], src_ts[idx]
    return np.where(query_ts - left < right - query_ts, idx - 1, idx)


# ---------------------------------------------------------------------------
# Station kinematics (MuJoCo)
# ---------------------------------------------------------------------------


def _pose(xpos: np.ndarray, xmat: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = xmat.reshape(3, 3)
    T[:3, 3] = xpos
    return T


@dataclass
class StaticGeom:
    """One visual geom of the cell, resolved to world coordinates."""

    name: str
    kind: str  # "mesh" | "box" | "cylinder" | "sphere"
    T_world_geom: np.ndarray
    color: tuple[int, int, int]
    opacity: float
    size: np.ndarray  # MuJoCo geom_size semantics
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None


class Station:
    """The MuJoCo YAM station: extrinsics, cell geometry, and FK."""

    def __init__(self, mjcf_path: Path):
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

        self.arm_qpos_adr = {}
        self.finger_qpos_adr = {}
        for side in SIDES:
            self.arm_qpos_adr[side] = np.array(
                [self.model.jnt_qposadr[self._jid(f"{side}_joint{k}")] for k in range(1, 7)]
            )
            self.finger_qpos_adr[side] = (
                self.model.jnt_qposadr[self._jid(f"{side}_left_finger")],
                self.model.jnt_qposadr[self._jid(f"{side}_right_finger")],
            )

        self.set_state({s: np.zeros(6) for s in SIDES}, {s: 0.0 for s in SIDES})

    # -- name helpers -------------------------------------------------------

    def _jid(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _bid(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _cid(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)

    # -- state / FK ---------------------------------------------------------

    def set_state(self, q: dict[str, np.ndarray], gripper: dict[str, float]) -> None:
        """Write joint angles + normalized gripper openings, then run FK."""
        for side in SIDES:
            self.data.qpos[self.arm_qpos_adr[side]] = q[side]
            travel = float(np.clip(gripper[side], 0.0, 1.0)) * GRIPPER_JOINT_TRAVEL
            lo, hi = self.finger_qpos_adr[side]
            # The two jaws are mirrored (MuJoCo joint equality, ratio -1).
            self.data.qpos[lo] = travel
            self.data.qpos[hi] = -travel
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def body_pose(self, name: str) -> np.ndarray:
        b = self._bid(name)
        return _pose(self.data.xpos[b], self.data.xmat[b])

    def camera_pose_cv(self, name: str) -> np.ndarray:
        """Camera pose in world, converted to the OpenCV convention viser wants."""
        c = self._cid(name)
        return _pose(self.data.cam_xpos[c], self.data.cam_xmat[c]) @ T_GL_CV

    def camera_fov_y(self, name: str) -> float:
        return float(np.deg2rad(self.model.cam_fovy[self._cid(name)]))

    # -- geometry -----------------------------------------------------------

    def _mesh_arrays(self, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
        m = self.model
        va, vn = m.mesh_vertadr[mesh_id], m.mesh_vertnum[mesh_id]
        fa, fn = m.mesh_faceadr[mesh_id], m.mesh_facenum[mesh_id]
        verts = np.asarray(m.mesh_vert[va : va + vn], dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(m.mesh_face[fa : fa + fn], dtype=np.int32).reshape(-1, 3)
        return verts, faces

    def _geom_color(self, gid: int, force_visible: bool) -> tuple[tuple[int, int, int], float]:
        m = self.model
        rgba = m.mat_rgba[m.geom_matid[gid]] if m.geom_matid[gid] >= 0 else m.geom_rgba[gid]
        rgb = tuple(round(c * 255) for c in rgba[:3])
        alpha = float(rgba[3])
        if force_visible and alpha == 0.0:
            # The D405 shells are alpha-0 in sim (they only exist for collision
            # and camera mounting). Show them here -- seeing the physical camera
            # body is the point of this visualization.
            return (60, 60, 66), 1.0
        return rgb, alpha  # type: ignore[return-value]

    def _is_arm_body(self, bid: int) -> bool:
        roots = {self._bid(f"{s}_arm") for s in SIDES}
        while bid != 0:
            if bid in roots:
                return True
            bid = self.model.body_parentid[bid]
        return False

    def static_geoms(self, include_props: bool, show_camera_bodies: bool) -> list[StaticGeom]:
        """Visual geoms of the cell (everything not attached to an arm)."""
        m = self.model
        groups = set(VISUAL_GEOM_GROUPS) | (set(PROP_GEOM_GROUPS) if include_props else set())
        kinds = {
            mujoco.mjtGeom.mjGEOM_MESH: "mesh",
            mujoco.mjtGeom.mjGEOM_BOX: "box",
            mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
            mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
        }

        out: list[StaticGeom] = []
        for gid in range(m.ngeom):
            if self._is_arm_body(m.geom_bodyid[gid]):
                continue
            gtype = mujoco.mjtGeom(m.geom_type[gid])
            if gtype not in kinds:
                continue  # planes become a viser grid; capsules are collision-only here

            is_camera_body = m.geom_dataid[gid] >= 0 and (
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[gid]) == "camera_d405"
            )
            if is_camera_body and not show_camera_bodies:
                continue
            if not is_camera_body and m.geom_group[gid] not in groups:
                continue

            color, opacity = self._geom_color(gid, force_visible=is_camera_body)
            if opacity == 0.0:
                continue

            verts = faces = None
            if kinds[gtype] == "mesh":
                verts, faces = self._mesh_arrays(m.geom_dataid[gid])

            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid])
            out.append(
                StaticGeom(
                    name=f"{body}_g{gid}",
                    kind=kinds[gtype],
                    T_world_geom=_pose(self.data.geom_xpos[gid], self.data.geom_xmat[gid]),
                    color=color,
                    opacity=opacity,
                    size=np.array(m.geom_size[gid], dtype=np.float64),
                    vertices=verts,
                    faces=faces,
                )
            )
        return out

    def arm_mesh_geoms(self, body: str) -> list[StaticGeom]:
        """Visual mesh geoms of one arm body, expressed in that body's frame."""
        m = self.model
        bid = self._bid(body)
        T_world_body = self.body_pose(body)
        out: list[StaticGeom] = []
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] != bid:
                continue
            if mujoco.mjtGeom(m.geom_type[gid]) != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if m.geom_group[gid] not in VISUAL_GEOM_GROUPS:
                continue
            color, opacity = self._geom_color(gid, force_visible=False)
            if opacity == 0.0:
                continue
            verts, faces = self._mesh_arrays(m.geom_dataid[gid])
            out.append(
                StaticGeom(
                    name=f"{body}_g{gid}",
                    kind="mesh",
                    T_world_geom=np.linalg.inv(T_world_body) @ _pose(self.data.geom_xpos[gid], self.data.geom_xmat[gid]),
                    color=color,
                    opacity=opacity,
                    size=np.array(m.geom_size[gid], dtype=np.float64),
                    vertices=verts,
                    faces=faces,
                )
            )
        return out


# ---------------------------------------------------------------------------
# URDF
# ---------------------------------------------------------------------------


def load_yam_urdf(urdf_dir: Path) -> yourdfpy.URDF:
    """Load the i2rt YAM URDF, resolving its ``package://assets/...`` meshes."""

    def resolve(fname: str) -> str:
        if fname.startswith("package://"):
            fname = fname[len("package://") :]
        return str(urdf_dir / fname)

    return yourdfpy.URDF.load(
        str(urdf_dir / "yam.urdf"),
        filename_handler=resolve,
        load_collision_meshes=False,
    )


def urdf_cfg(urdf: yourdfpy.URDF, q: np.ndarray) -> np.ndarray:
    """Reorder ``q`` (joint1..joint6) into the URDF's actuated-joint order.

    The i2rt URDF declares joints from the wrist down, so ``actuated_joint_names``
    is ('joint6', ..., 'joint1'). Passing ``q`` positionally silently produces a
    mirrored arm; always index by name.
    """
    by_name = {f"joint{k + 1}": q[k] for k in range(6)}
    return np.array([by_name[n] for n in urdf.actuated_joint_names], dtype=np.float64)


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def _add_static_geom(server: viser.ViserServer, prefix: str, g: StaticGeom) -> None:
    T = vtf.SE3.from_matrix(g.T_world_geom)
    kwargs = dict(wxyz=T.rotation().wxyz, position=T.translation())
    opacity = None if g.opacity >= 1.0 else g.opacity
    name = f"{prefix}/{g.name}"

    if g.kind == "mesh":
        assert g.vertices is not None and g.faces is not None
        server.scene.add_mesh_simple(
            name, g.vertices, g.faces, color=g.color, opacity=opacity, side="double", **kwargs
        )
    elif g.kind == "box":
        server.scene.add_box(name, color=g.color, dimensions=2.0 * g.size[:3], opacity=opacity, **kwargs)
    elif g.kind == "cylinder":
        server.scene.add_cylinder(
            name, radius=float(g.size[0]), height=2.0 * float(g.size[1]), color=g.color, opacity=opacity, **kwargs
        )
    elif g.kind == "sphere":
        server.scene.add_icosphere(name, radius=float(g.size[0]), color=g.color, opacity=opacity, **kwargs)


@dataclass
class SceneHandles:
    urdf_vis: dict[str, viser.extras.ViserUrdf]
    arm_frames: dict[str, viser.FrameHandle]
    wrist_frames: dict[str, viser.FrameHandle]  # link_6 pose per side
    finger_frames: dict[str, dict[str, viser.FrameHandle]]
    frustums: dict[str, viser.CameraFrustumHandle]
    ee_axes: dict[str, viser.FrameHandle]


def build_scene(
    server: viser.ViserServer,
    station: Station,
    urdf_dir: Path,
    *,
    include_props: bool,
    show_camera_bodies: bool,
    show_cell: bool,
    frustum_scale: float,
    wrist_frustum_scale: float,
    image_aspect: float,
) -> SceneHandles:
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=3.0, height=3.0, cell_size=0.25, position=(0.3, 0.0, 0.0))

    if show_cell:
        for g in station.static_geoms(include_props, show_camera_bodies):
            _add_static_geom(server, "/cell", g)

    urdf_vis: dict[str, viser.extras.ViserUrdf] = {}
    arm_frames: dict[str, viser.FrameHandle] = {}
    wrist_frames: dict[str, viser.FrameHandle] = {}
    finger_frames: dict[str, dict[str, viser.FrameHandle]] = {}
    frustums: dict[str, viser.CameraFrustumHandle] = {}
    ee_axes: dict[str, viser.FrameHandle] = {}

    for side in SIDES:
        T = vtf.SE3.from_matrix(station.body_pose(f"{side}_arm"))
        arm_frames[side] = server.scene.add_frame(
            f"/{side}_arm", show_axes=False, wxyz=T.rotation().wxyz, position=T.translation()
        )
        # Each ViserUrdf needs its own URDF instance -- update_cfg mutates it.
        urdf_vis[side] = viser.extras.ViserUrdf(
            server, load_yam_urdf(urdf_dir), root_node_name=f"/{side}_arm"
        )

        # The i2rt URDF ships no link_6 mesh (assets/link_6_visual.stl is absent),
        # and has no gripper joints at all. Borrow the wrist + jaw meshes from the
        # MuJoCo station model and drive them off MuJoCo FK so the end effector
        # and its camera mount are actually visible.
        wrist_frames[side] = server.scene.add_frame(f"/{side}_wrist", show_axes=False)
        for g in station.arm_mesh_geoms(f"{side}_link_6"):
            _add_static_geom(server, f"/{side}_wrist", g)

        finger_frames[side] = {}
        for finger in ("link_left_finger", "link_right_finger"):
            body = f"{side}_{finger}"
            finger_frames[side][finger] = server.scene.add_frame(f"/{side}_{finger}", show_axes=False)
            for g in station.arm_mesh_geoms(body):
                _add_static_geom(server, f"/{side}_{finger}", g)

        ee_axes[side] = server.scene.add_frame(
            f"/{side}_wrist/ee_axes", show_axes=True, axes_length=0.06, axes_radius=0.0025
        )

    for cam in CAMERAS:
        # Wrist frustums sit right on the jaws, so keep them smaller than the
        # top one or they swallow the end effector they are attached to.
        frustums[cam] = server.scene.add_camera_frustum(
            f"/camera_{cam}",
            fov=station.camera_fov_y(cam),
            aspect=image_aspect,
            scale=frustum_scale if cam == "top" else wrist_frustum_scale,
            color=(255, 170, 60) if cam == "top" else (80, 190, 255),
            line_width=2.0,
        )

    return SceneHandles(urdf_vis, arm_frames, wrist_frames, finger_frames, frustums, ee_axes)


def apply_frame(
    station: Station,
    handles: SceneHandles,
    q: dict[str, np.ndarray],
    gripper: dict[str, float],
    images: dict[str, np.ndarray | None],
    urdfs: dict[str, yourdfpy.URDF],
) -> None:
    """Pose everything in the scene for one timestep."""
    station.set_state(q, gripper)

    for side in SIDES:
        handles.urdf_vis[side].update_cfg(urdf_cfg(urdfs[side], q[side]))

        T6 = vtf.SE3.from_matrix(station.body_pose(f"{side}_link_6"))
        handles.wrist_frames[side].wxyz = T6.rotation().wxyz
        handles.wrist_frames[side].position = T6.translation()

        for finger, handle in handles.finger_frames[side].items():
            Tf = vtf.SE3.from_matrix(station.body_pose(f"{side}_{finger}"))
            handle.wxyz = Tf.rotation().wxyz
            handle.position = Tf.translation()

    for cam in CAMERAS:
        Tc = vtf.SE3.from_matrix(station.camera_pose_cv(cam))
        f = handles.frustums[cam]
        f.wxyz = Tc.rotation().wxyz
        f.position = Tc.translation()
        img = images.get(cam)
        if img is not None:
            f.image = img


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


def self_test(station: Station, urdf_dir: Path) -> bool:
    """Check URDF FK against MuJoCo FK and print the extrinsics we rely on."""
    urdf = load_yam_urdf(urdf_dir)
    missing = [n for n in ("joint1", "joint6") if n not in urdf.actuated_joint_names]
    if missing:
        print(f"FAIL: URDF is missing joints {missing}")
        return False

    print(f"URDF actuated joint order: {urdf.actuated_joint_names}")
    rng = np.random.default_rng(0)
    max_p = max_r = 0.0
    for _ in range(32):
        q = rng.uniform(-0.8, 0.8, 6)
        station.set_state({s: q for s in SIDES}, {s: 0.5 for s in SIDES})
        urdf.update_cfg(urdf_cfg(urdf, q))
        for side in SIDES:
            A = np.linalg.inv(station.body_pose(f"{side}_arm")) @ station.body_pose(f"{side}_link_6")
            B = urdf.get_transform("link_6", "base_link")
            max_p = max(max_p, float(np.linalg.norm(A[:3, 3] - B[:3, 3])))
            R = A[:3, :3].T @ B[:3, :3]
            max_r = max(max_r, float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))))
    print(f"URDF vs MuJoCo FK at link_6 over 32 random configs: "
          f"max pos err {max_p * 1e6:.1f} um, max rot err {max_r:.5f} deg")

    station.set_state({s: np.zeros(6) for s in SIDES}, {s: 0.0 for s in SIDES})
    np.set_printoptions(precision=5, suppress=True)
    for side in SIDES:
        print(f"\nT_world_{side}armbase =\n{station.body_pose(f'{side}_arm')}")
        T6 = station.body_pose(f"{side}_link_6")
        print(f"T_{side}link6_cam (OpenCV) =\n{np.linalg.inv(T6) @ station.camera_pose_cv(side)}")
    print(f"\nT_world_topcam (OpenCV) =\n{station.camera_pose_cv('top')}")
    for cam in CAMERAS:
        print(f"camera_{cam}: vertical fov {np.degrees(station.camera_fov_y(cam)):.1f} deg")

    ok = max_p < 1e-4 and max_r < 0.01
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Static playback hosting
# ---------------------------------------------------------------------------


def ensure_viser_client(dest: Path) -> Path:
    """Drop viser's prebuilt static client at ``dest`` (no-op if already there).

    This is what ``viser-build-client`` does: the wheel ships a built client, so
    it is a directory copy, not an npm build.
    """
    import shutil

    from viser._client_autobuild import build_dir

    if (dest / "index.html").exists():
        return dest
    if not (build_dir / "index.html").exists():
        raise RuntimeError(f"viser ships no prebuilt client at {build_dir}; run viser-build-client manually")
    shutil.copytree(build_dir, dest, dirs_exist_ok=True)
    print(f"Copied viser client -> {dest}", flush=True)
    return dest


def serve_playback(recording: Path, port: int) -> None:
    """Serve ``recording``'s directory over http and print the playback URL."""
    import http.server
    import socketserver
    import subprocess

    root = recording.parent
    ensure_viser_client(root / "viser-client")
    url = f"http://localhost:{port}/viser-client/?playbackPath=http://localhost:{port}/{recording.name}"

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a) -> None:  # keep the console readable
            pass

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    # Walk forward a few ports so a busy default doesn't sink the whole run.
    httpd = None
    for candidate in range(port, port + 10):
        try:
            httpd = Server(("127.0.0.1", candidate), Handler)
        except OSError:
            continue
        if candidate != port:
            print(f"port {port} was busy, using {candidate} instead")
            port = candidate
            url = f"http://localhost:{port}/viser-client/?playbackPath=http://localhost:{port}/{recording.name}"
        break
    if httpd is None:
        print(f"\nNo free port in {port}..{port + 9}. Retry with --view-port <other>.")
        raise SystemExit(1)

    print(f"\n  Viser playback:  {url}\n", flush=True)
    print("  Space = play/pause, drag the timeline to scrub, scroll = zoom.", flush=True)
    print("  Ctrl-C to stop.\n", flush=True)
    if os.environ.get("DISPLAY"):
        subprocess.Popen(
            ["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode_dir", type=Path, nargs="?", help="episode directory to visualize")
    p.add_argument("-o", "--output", type=Path, default=None, help="output .viser path")
    p.add_argument("--fps", type=float, default=20.0, help="playback/sample rate (default: 20)")
    p.add_argument("--image-width", type=int, default=320, help="frustum texture width, 0 to keep native")
    p.add_argument("--frustum-scale", type=float, default=0.08, help="top-camera frustum size in meters")
    p.add_argument(
        "--wrist-frustum-scale", type=float, default=0.05, help="wrist-camera frustum size in meters"
    )
    p.add_argument("--start", type=float, default=0.0, help="trim start, seconds into the episode")
    p.add_argument("--duration", type=float, default=None, help="trim length in seconds")
    p.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="station MJCF")
    p.add_argument("--urdf-dir", type=Path, default=DEFAULT_URDF_DIR, help="dir holding yam.urdf + assets/")
    p.add_argument("--no-cell", action="store_true", help="skip static cell geometry")
    p.add_argument(
        "--camera-bodies",
        action="store_true",
        help="also draw the D405 shells (alpha-0 in sim; the frustums already mark the cameras)",
    )
    p.add_argument("--show-sim-props", action="store_true", help="also draw the MJCF bottles/bin (sim-only)")
    p.add_argument("--serve", action="store_true", help="serve interactively instead of writing a file")
    p.add_argument("--port", type=int, default=8080, help="viser port for --serve")
    p.add_argument(
        "--view",
        action="store_true",
        help="after writing the .viser, drop the static client next to it and serve the playback URL",
    )
    p.add_argument("--view-port", type=int, default=8123, help="static http port for --view (walks up if busy)")
    p.add_argument("--self-test", action="store_true", help="validate FK/extrinsics and exit")
    args = p.parse_args(argv)

    station = Station(args.mjcf)

    if args.self_test:
        return 0 if self_test(station, args.urdf_dir) else 1

    if args.episode_dir is None:
        p.error("episode_dir is required unless --self-test is given")
    episode_dir = args.episode_dir.expanduser().resolve()
    if not episode_dir.is_dir():
        p.error(f"not a directory: {episode_dir}")

    print(f"Episode: {episode_dir}")
    arms = load_arm_streams(episode_dir)

    # Shared timeline: the window all proprioception streams cover.
    t0 = max(s.ts[0] for s in arms.values())
    t1 = min(s.ts[-1] for s in arms.values())
    t0 += max(0.0, args.start)
    if args.duration is not None:
        t1 = min(t1, t0 + args.duration)
    if t1 <= t0:
        p.error(f"empty time window ({t0:.3f} -> {t1:.3f}); check --start/--duration")

    dt = 1.0 / args.fps
    grid = np.arange(t0, t1, dt)
    print(f"Timeline: {t1 - t0:.2f}s, {len(grid)} frames at {args.fps:g} fps")

    cams = load_camera_streams(episode_dir, grid, args.image_width)
    arm_idx = {side: _nearest_index(arms[side].ts, grid) for side in SIDES}
    cam_idx = {cam: _nearest_index(cams[cam].ts, grid) for cam in cams}

    aspect = 4.0 / 3.0
    for cam in cams:
        aspect = cams[cam].frames.shape[2] / cams[cam].frames.shape[1]
        break

    server = viser.ViserServer(port=args.port) if args.serve else viser.ViserServer()
    handles = build_scene(
        server,
        station,
        args.urdf_dir,
        include_props=args.show_sim_props,
        show_camera_bodies=args.camera_bodies,
        show_cell=not args.no_cell,
        frustum_scale=args.frustum_scale,
        wrist_frustum_scale=args.wrist_frustum_scale,
        image_aspect=aspect,
    )
    urdfs = {side: handles.urdf_vis[side]._urdf for side in SIDES}

    server.initial_camera.position = (1.9, -1.5, 1.75)
    server.initial_camera.look_at = (0.45, 0.0, 0.95)
    server.initial_camera.up = (0.0, 0.0, 1.0)

    def frame_payload(k: int) -> tuple[dict, dict, dict]:
        q = {side: arms[side].q[arm_idx[side][k]] for side in SIDES}
        grip = {side: float(arms[side].gripper[arm_idx[side][k]]) for side in SIDES}
        images: dict[str, np.ndarray | None] = {}
        for cam, stream in cams.items():
            orig = int(cam_idx[cam][k])
            mapped = stream._index_of.get(orig)  # type: ignore[attr-defined]
            images[cam] = stream.frames[mapped] if mapped is not None else None
        return q, grip, images

    if args.serve:
        slider = server.gui.add_slider("Frame", min=0, max=len(grid) - 1, step=1, initial_value=0)
        playing = server.gui.add_checkbox("Play", True)
        time_readout = server.gui.add_number("t (s)", 0.0, disabled=True)

        @slider.on_update
        def _(_) -> None:
            k = int(slider.value)
            apply_frame(station, handles, *frame_payload(k), urdfs)
            time_readout.value = round(float(grid[k] - grid[0]), 3)

        apply_frame(station, handles, *frame_payload(0), urdfs)
        print(f"\nServing on http://localhost:{args.port}  (ctrl-c to stop)")
        while True:
            if playing.value:
                slider.value = (int(slider.value) + 1) % len(grid)
            time.sleep(dt)

    out = args.output or episode_dir.parent / f"{episode_dir.name}.viser"
    out = out.expanduser().resolve()
    print(f"\nRecording {len(grid)} frames -> {out}")

    serializer = server.get_scene_serializer()
    started = time.time()
    for k in range(len(grid)):
        apply_frame(station, handles, *frame_payload(k), urdfs)
        serializer.insert_sleep(dt)
        if k % 100 == 0 or k == len(grid) - 1:
            print(f"  frame {k + 1}/{len(grid)}  ({time.time() - started:.1f}s elapsed)", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    data = serializer.serialize()
    out.write_bytes(data)
    print(f"\nWrote {out}  ({len(data) / 1e6:.1f} MB, {len(grid) * dt:.1f}s of playback)")

    if args.view:
        # Free the viser websocket port before taking a second one; the scene is
        # already serialized, so the live server has nothing left to do.
        server.stop()
        serve_playback(out, args.view_port)
        return 0

    print("View it with:")
    print("  viser-build-client --out-dir viser-client/")
    print(f"  python -m http.server 8000 --directory {out.parent}")
    print(f"  open http://localhost:8000/viser-client/?playbackPath=http://localhost:8000/{out.name}")
    print("...or just re-run with --view to do all of that for you.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
