"""XdofSimNode — unified bimanual YAM sim node (physics + cameras + recording).

Mirrors market42's XdofSimNode pattern:
- Physics steps flat-out in main thread
- Cameras rendered + viser previews in background thread at camera_fps
- Viser browser viewer (http://localhost:viser_port) with live body poses and
  wrist camera panels — optional, enabled by default
- VR streaming to Quest headset via Three.js WebXR — auto-starts when a
  Quest/Meta device is detected over ADB, otherwise silently skipped
- Recording: {name}-left.mcap + {name}-right.mcap + {name}-sim_state.mcap +
  per-camera MP4s

Published topics:
    {name}/left_state   — 7D left arm [j1..6, grip_norm]
    {name}/right_state  — 7D right arm

Subscribed topics (from cmd_topics):
    e.g. "gello_left/joint_pos", "gello_right/joint_pos"
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import numpy as np

from robots_realtime.nodes.base import Node, NodeRole
from robots_realtime.recording.writer import McapWriter, AsyncMp4Writer, NullWriter

logger = logging.getLogger(__name__)

_DOFS_PER_ARM = 7  # 6 joints + 1 gripper
_GRIPPER_CTRL_MAX = 0.0475


# ---------------------------------------------------------------------------
# Quest ADB detection
# ---------------------------------------------------------------------------

def _detect_quest_device() -> str | None:
    """Return ADB serial of a connected Quest/Meta device, or None."""
    import shutil
    import subprocess

    if shutil.which("adb") is None:
        return None
    try:
        result = subprocess.run(
            ["adb", "devices", "-l"], capture_output=True, text=True, timeout=5.0
        )
        for line in result.stdout.splitlines()[1:]:
            line = line.strip()
            parts = line.split()
            if len(parts) < 2 or parts[1] == "offline":
                continue
            device_id = parts[0]
            prop = subprocess.run(
                ["adb", "-s", device_id, "shell", "getprop", "ro.product.model"],
                capture_output=True, text=True, timeout=3.0,
            )
            model = prop.stdout.strip().lower()
            if any(kw in model for kw in ("quest", "oculus", "meta")):
                return device_id
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# VR streamer (Three.js WebXR over WebSocket)
# ---------------------------------------------------------------------------

class _VrStreamer:
    """Streams MuJoCo body transforms to a Three.js WebXR client (Quest headset).

    Exports per-body GLB meshes once at construction, then serves them and a
    binary WebSocket transform stream from an aiohttp server in a background
    thread.  Reads body poses directly from the live MjData pointer — no lock
    needed for visualization-quality reads.

    Open ``http://<server-ip>:<port>`` in the Quest Browser to connect.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        port: int = 8012,
        stream_rate: float = 60.0,
        task: str = "",
        adb_device: str | None = None,
    ) -> None:
        import shutil
        from pathlib import Path
        from scipy.spatial.transform import Rotation
        from robots_realtime.nodes.sim._mujoco_viser import VR_HTML, export_body_glbs

        self._data = data
        self._port = port
        self._stream_rate = stream_rate
        self._adb_device = adb_device

        mesh_dir = Path(f"/tmp/rr_vr_{task or 'sim'}")
        if mesh_dir.exists():
            shutil.rmtree(mesh_dir)
        self._body_info = export_body_glbs(model, mesh_dir)
        self._mesh_dir = mesh_dir
        self._all_ids = list(self._body_info.keys())
        logger.info("[VrStreamer] exported %d body meshes → %s", len(self._body_info), mesh_dir)

        # MuJoCo Z-up → Three.js Y-up: (x, y, z) → (x, z, -y)
        self._R_conv = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
        self._Rotation = Rotation
        self._html = VR_HTML

        self._stop_event = threading.Event()
        self._loop: Any = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="VrStreamer"
        )
        self._thread.start()

    def _build_frame(self) -> bytes:
        """Pack all body world transforms into a flat float32 buffer (Y-up)."""
        R_conv = self._R_conv
        buf = np.zeros(len(self._all_ids) * 8, dtype=np.float32)
        for i, bid in enumerate(self._all_ids):
            xpos = self._data.xpos[bid]
            xmat = self._data.xmat[bid].reshape(3, 3)
            pos_yup = R_conv @ xpos
            det = np.linalg.det(xmat)
            if abs(det) > 1e-6:
                q = self._Rotation.from_matrix(R_conv @ xmat @ R_conv.T).as_quat()
            else:
                q = np.array([0.0, 0.0, 0.0, 1.0])
            offset = i * 8
            buf[offset] = float(bid)
            buf[offset + 1:offset + 4] = pos_yup
            buf[offset + 4:offset + 8] = q
        return buf.tobytes()

    def _run(self) -> None:
        import asyncio
        from aiohttp import web

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        app = web.Application()
        ws_clients: list[Any] = []

        async def handle_index(request: Any) -> Any:
            return web.Response(text=self._html, content_type="text/html")

        async def handle_bodies(request: Any) -> Any:
            return web.json_response({
                str(k): {"file": v["file"], "is_fixed": bool(v["is_fixed"])}
                for k, v in self._body_info.items()
            })

        async def handle_mesh(request: Any) -> Any:
            path = self._mesh_dir / request.match_info["filename"]
            if not path.exists():
                return web.Response(status=404)
            return web.FileResponse(path, headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache",
                "Content-Type": "model/gltf-binary",
            })

        async def handle_ws(request: Any) -> Any:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            ws_clients.append(ws)
            logger.info("[VrStreamer] client connected (%d total)", len(ws_clients))
            try:
                async for _ in ws:
                    pass
            finally:
                ws_clients.remove(ws)
            return ws

        async def stream_loop() -> None:
            dt = 1.0 / self._stream_rate
            while not self._stop_event.is_set():
                if ws_clients:
                    frame = self._build_frame()
                    for client in list(ws_clients):
                        try:
                            await client.send_bytes(frame)
                        except Exception:
                            pass
                await asyncio.sleep(dt)

        async def on_startup(app_: Any) -> None:
            asyncio.create_task(stream_loop())

        app.on_startup.append(on_startup)
        app.router.add_get("/", handle_index)
        app.router.add_get("/api/bodies", handle_bodies)
        app.router.add_get("/meshes/{filename}", handle_mesh)
        app.router.add_get("/ws", handle_ws)

        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "0.0.0.0", self._port)
        loop.run_until_complete(site.start())
        logger.info("[VrStreamer] serving at http://0.0.0.0:%d", self._port)

        if self._adb_device:
            import subprocess as _sp
            url = f"http://localhost:{self._port}"
            try:
                _sp.run(["adb", "-s", self._adb_device, "reverse",
                         f"tcp:{self._port}", f"tcp:{self._port}"],
                        capture_output=True, timeout=5.0)
                logger.info("[VrStreamer] ADB reverse forwarding set up")
            except Exception as exc:
                logger.warning("[VrStreamer] ADB reverse failed: %s", exc)
            try:
                _sp.run(["adb", "-s", self._adb_device, "shell", "am", "start",
                         "-a", "android.intent.action.VIEW", "-d", url],
                        capture_output=True, timeout=5.0)
                logger.info("[VrStreamer] opened Quest Browser → %s", url)
            except Exception as exc:
                logger.warning("[VrStreamer] failed to open Quest Browser: %s", exc)

        loop.run_forever()
        loop.run_until_complete(runner.cleanup())

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# Viser scene manager
# ---------------------------------------------------------------------------

class _ViserSceneManager:
    """Live viser browser visualization backed by a running MuJoCo model/data.

    Uploads fixed geometry once at construction; pushes dynamic body poses
    each call to ``update_poses_only()`` (fast, called from physics thread).
    Camera frames are rendered at full resolution in the background render
    thread via ``update_cam_previews()`` and downsampled for the sidebar.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        cam_names: list[str],
        port: int = 8765,
        visible_geom_groups: tuple[int, ...] = (0, 1, 2),
        record_camera_width: int = 640,
        record_camera_height: int = 480,
        preview_camera_width: int = 320,
        preview_camera_height: int = 240,
    ) -> None:
        import mujoco
        import viser
        import viser.transforms as vtf
        from mujoco import mj_id2name, mjtGeom, mjtObj
        from robots_realtime.nodes.sim._mujoco_viser import (
            _get_body_name,
            _is_fixed_body,
            _merge_geoms,
            configure_default_camera,
        )

        self._model = model
        self._data = data
        self._vtf = vtf
        self._mesh_handles: dict[int, Any] = {}
        self._reset_requested = False
        self._mujoco = mujoco
        self._renderer: Any = None
        self._renderer_created = False
        self._record_camera_width = record_camera_width
        self._record_camera_height = record_camera_height
        self._preview_camera_width = preview_camera_width
        self._preview_camera_height = preview_camera_height
        self._cam_names = cam_names
        self._cam_handles: dict[str, Any] = {}

        self.server = viser.ViserServer(port=port)
        configure_default_camera(self.server)
        try:
            self.actual_port = self.server._websock_server._port
        except Exception:
            self.actual_port = port
        logger.info("Viser scene viewer: http://localhost:%d", self.actual_port)

        # Camera sidebar panels
        if cam_names:
            placeholder = np.zeros(
                (preview_camera_height, preview_camera_width, 3), dtype=np.uint8
            )
            with self.server.gui.add_folder("Cameras"):
                for name in cam_names:
                    self._cam_handles[name] = self.server.gui.add_image(
                        placeholder,
                        label=name,
                        format="jpeg",
                        jpeg_quality=80,
                    )

        # Reset button
        reset_btn = self.server.gui.add_button("Reset Environment", color="red")

        @reset_btn.on_click
        def _(_) -> None:
            self._reset_requested = True

        # Static geometry upload
        body_visual: dict[int, list[int]] = {}
        for i in range(model.ngeom):
            if int(model.geom_group[i]) in visible_geom_groups:
                body_visual.setdefault(int(model.geom_bodyid[i]), []).append(i)

        self.server.scene.add_frame("/fixed_bodies", show_axes=False)

        for body_id, visual_ids in body_visual.items():
            body_name = _get_body_name(model, body_id)
            if _is_fixed_body(model, body_id):
                nonplane_ids = []
                for gid in visual_ids:
                    if model.geom_type[gid] == mjtGeom.mjGEOM_PLANE:
                        geom_name = mj_id2name(model, mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
                        self.server.scene.add_grid(
                            f"/fixed_bodies/{body_name}/{geom_name}",
                            width=2000.0, height=2000.0,
                            position=model.geom_pos[gid],
                            wxyz=model.geom_quat[gid],
                        )
                    else:
                        nonplane_ids.append(gid)
                if nonplane_ids:
                    merged = _merge_geoms(model, nonplane_ids)
                    handle = self.server.scene.add_mesh_trimesh(
                        f"/fixed_bodies/{body_name}", merged,
                        position=model.body(body_id).pos,
                        wxyz=model.body(body_id).quat,
                    )
                    self._mesh_handles[body_id] = handle
            elif visual_ids:
                merged = _merge_geoms(model, visual_ids)
                handle = self.server.scene.add_mesh_trimesh(
                    f"/bodies/{body_name}", merged, visible=True
                )
                self._mesh_handles[body_id] = handle

    def pop_reset_requested(self) -> bool:
        if self._reset_requested:
            self._reset_requested = False
            return True
        return False

    def update_poses_only(self) -> None:
        """Push body poses to viser (fast, no rendering)."""
        vtf = self._vtf
        with self.server.atomic():
            for body_id, handle in self._mesh_handles.items():
                handle.position = self._data.xpos[body_id]
                xmat = self._data.xmat[body_id].reshape(3, 3)
                handle.wxyz = vtf.SO3.from_matrix(xmat).wxyz
            self.server.flush()

    def update_cam_previews(self) -> dict[str, Any]:
        """Render cameras at record resolution, push downsampled previews to sidebar.

        Returns full-resolution frames for MP4 recording — no second render needed.
        Called from the background render thread only.
        """
        if not self._renderer_created and self._cam_names:
            self._renderer_created = True
            try:
                self._renderer = self._mujoco.Renderer(
                    self._model,
                    height=self._record_camera_height,
                    width=self._record_camera_width,
                )
                logger.info(
                    "Viser renderer: %dx%d → %dx%d preview",
                    self._record_camera_width, self._record_camera_height,
                    self._preview_camera_width, self._preview_camera_height,
                )
            except Exception as exc:
                logger.warning("Could not create viser renderer: %s", exc)

        if self._renderer is None:
            return {}

        scale_y = max(1, self._record_camera_height // self._preview_camera_height)
        scale_x = max(1, self._record_camera_width // self._preview_camera_width)

        frames: dict[str, Any] = {}
        for cam_name in self._cam_names:
            try:
                self._renderer.update_scene(self._data, camera=cam_name)
                frame = self._renderer.render().copy()
                frames[cam_name] = frame
                if cam_name in self._cam_handles:
                    self._cam_handles[cam_name].image = frame[::scale_y, ::scale_x]
            except Exception:
                pass
        return frames

    def stop(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        try:
            self.server.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# XdofSimNode
# ---------------------------------------------------------------------------

class XdofSimNode(Node):
    """Unified bimanual YAM simulation node.

    Owns the MuJoCo environment directly; manages physics stepping in the main
    loop and camera rendering in a background thread.

    Args:
        name:               Node name on the bus.
        scene:              Scene variant (e.g. "hybrid").
        task:               Scene XML (e.g. "bottles").
        physics_dt:         MuJoCo timestep in seconds.
        control_decimation: Physics steps per control step.
        camera_fps:         Camera render / viser update rate in Hz.
        cmd_topics:         Dict mapping arm key → subscribed topic.
        viser_port:         Viser server port. None to disable.
        vr_port:            VR streaming port. None to disable; auto-starts
                            only when a Quest device is detected via ADB.
        writer:             Ignored — XdofSimNode manages its own writers.
    """

    role = NodeRole.ROBOT
    published_topics: list[str] = ["left_state", "right_state"]
    poll_freq: float | None = None  # flat-out

    def __init__(
        self,
        name: str = "yam",
        scene: str = "hybrid",
        task: str = "bottles",
        physics_dt: float = 0.0001,
        control_decimation: int = 17,
        camera_fps: float = 30.0,
        cmd_topics: dict | None = None,
        viser_port: int | None = 8765,
        vr_port: int | None = 8012,
        writer=None,
        **kwargs,
    ) -> None:
        self._scene = scene
        self._task = task
        self._physics_dt = physics_dt
        self._control_decimation = control_decimation
        self._camera_fps = camera_fps
        self._cmd_topics: dict[str, str] = cmd_topics or {}
        self._viser_port = viser_port
        self._vr_port = vr_port

        self.subscribed_topics = list(self._cmd_topics.values())
        super().__init__(name=name, writer=NullWriter(), **kwargs)

        self._env = None
        self._cmd: np.ndarray | None = None

        # Threading objects — created in setup() to stay picklable
        self._render_thread: threading.Thread | None = None
        self._render_stop: threading.Event | None = None
        self._cmd_lock: threading.Lock | None = None

        # Viser / VR — created in setup()
        self._viser: _ViserSceneManager | None = None
        self._vr_streamer: _VrStreamer | None = None

        # Stale-message skip: track last-seen timestamp per subscribed topic
        self._last_cmd_ts: dict[str, float] = {}
        # Throttle viser pose + state publish to ~30 Hz
        self._last_obs_ts: float = 0.0
        self._obs_interval: float = 1.0 / 30.0

        # Per-arm and per-camera writers
        self._left_writer: McapWriter | None = None
        self._right_writer: McapWriter | None = None
        self._sim_state_writer: McapWriter | None = None
        self._cam_writers: dict[str, AsyncMp4Writer] = {}
        self._recording_save_dir: str = ""
        self._pending_save_dir: str = ""

    # ------------------------------------------------------------------
    # Node interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        self._render_stop = threading.Event()
        self._cmd_lock = threading.Lock()

        os.environ.setdefault("MUJOCO_GL", "egl")
        import robots_realtime.sim as sim

        self._env = sim.make_env(
            scene_variant=self._scene,
            task=self._task,
            render_cameras=False,
            physics_dt=self._physics_dt,
            control_decimation=self._control_decimation,
        )
        with self._cmd_lock:
            self._cmd = np.array(self._env.get_init_q(), dtype=np.float64)
        self._env.reset()

        cam_names = getattr(self._env, "camera_names", [])
        self.published_topics = ["left_state", "right_state"] + list(cam_names)

        # Viser browser viewer
        if self._viser_port is not None:
            try:
                self._viser = _ViserSceneManager(
                    model=self._env.model,
                    data=self._env.data,
                    cam_names=cam_names,
                    port=self._viser_port,
                )
            except Exception:
                logger.exception("[%s] failed to start Viser — continuing without it", self.name)
                self._viser = None

        # VR streaming — only if a Quest device is detected
        if self._vr_port is not None:
            device = _detect_quest_device()
            if device:
                logger.info("[%s] Quest detected (%s) — starting VR streamer", self.name, device)
                try:
                    self._vr_streamer = _VrStreamer(
                        model=self._env.model,
                        data=self._env.data,
                        port=self._vr_port,
                        task=self._task,
                        adb_device=device,
                    )
                except Exception:
                    logger.exception("[%s] failed to start VR streamer", self.name)
                    self._vr_streamer = None
            else:
                logger.info(
                    "[%s] No Quest device detected via ADB — VR streamer skipped "
                    "(connect Quest via USB ADB to auto-start, or pass vr_port=None to silence)",
                    self.name,
                )

        # Background render thread (always started — drives viser previews + MP4 recording)
        self._render_stop.clear()
        self._render_thread = threading.Thread(
            target=self._render_loop,
            daemon=True,
            name=f"SimRender-{self.name}",
        )
        self._render_thread.start()

        if self._pending_save_dir:
            self._open_cam_writers(self._pending_save_dir)
            self._pending_save_dir = ""

    def step(self) -> None:
        # 1. Absorb latest commands — skip stale messages
        for arm_key, topic in self._cmd_topics.items():
            ts = self.get_timestamp(topic)
            if ts is None:
                continue
            if ts == self._last_cmd_ts.get(topic, 0.0):
                continue  # stale
            self._last_cmd_ts[topic] = ts

            latest = self.get_latest(topic)
            if latest is None:
                continue
            jp = latest.get("joint_pos")
            if jp is None:
                continue
            arr = np.asarray(jp, dtype=np.float64)

            with self._cmd_lock:
                if arm_key == "left":
                    self._cmd[:_DOFS_PER_ARM] = arr[:_DOFS_PER_ARM]
                elif arm_key == "right":
                    self._cmd[_DOFS_PER_ARM: _DOFS_PER_ARM * 2] = arr[:_DOFS_PER_ARM]

        # 2. GUI reset from viser
        if self._viser is not None and self._viser.pop_reset_requested():
            self._env.reset(seed=int(time.time() * 1000) & 0xFFFFFFFF)
            with self._cmd_lock:
                self._cmd[:] = self._env.get_init_q()

        # 3. Step physics
        with self._cmd_lock:
            cmd_snapshot = self._cmd.copy()
        self._env._step_single(cmd_snapshot)

        # 4. Record sim state at every physics step
        now = time.time()
        if self._recording:
            state = self._read_joint_state()
            left_state = state[:_DOFS_PER_ARM].tolist()
            right_state = state[_DOFS_PER_ARM:].tolist()
            if self._left_writer is not None and self._left_writer.is_open:
                self._left_writer.write("joint_state", now, {"joint_pos": left_state})
            if self._right_writer is not None and self._right_writer.is_open:
                self._right_writer.write("joint_state", now, {"joint_pos": right_state})
            if self._sim_state_writer is not None and self._sim_state_writer.is_open:
                self._sim_state_writer.write("sim_qpos", now, {"qpos": self._env.data.qpos.copy().tolist()})

        # 5. Publish state + viser pose update at ~30 Hz
        if now - self._last_obs_ts >= self._obs_interval:
            self._last_obs_ts = now
            if not self._recording:
                state = self._read_joint_state()
                left_state = state[:_DOFS_PER_ARM].tolist()
                right_state = state[_DOFS_PER_ARM:].tolist()

            self.publish("left_state", {"joint_pos": left_state}, ts=now)
            self.publish("right_state", {"joint_pos": right_state}, ts=now)

            if self._viser is not None:
                self._viser.update_poses_only()

    @property
    def web_endpoints(self) -> list[str]:
        urls = []
        if self._viser_port is not None:
            urls.append(f"http://localhost:{self._viser_port}  (viser)")
        if self._vr_port is not None:
            urls.append(f"http://localhost:{self._vr_port}  (vr)")
        return urls

    def cleanup(self) -> None:
        if self._render_stop is not None:
            self._render_stop.set()
        if self._render_thread is not None:
            self._render_thread.join(timeout=3.0)
            self._render_thread = None
        if self._viser is not None:
            self._viser.stop()
            self._viser = None
        if self._vr_streamer is not None:
            self._vr_streamer.stop()
            self._vr_streamer = None
        if self._env is not None and hasattr(self._env, "close"):
            try:
                self._env.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def start_recording(self, save_dir: str) -> None:
        self._recording_save_dir = save_dir
        self._left_writer = McapWriter()
        self._left_writer.open(save_dir, f"{self.name}-left")
        self._right_writer = McapWriter()
        self._right_writer.open(save_dir, f"{self.name}-right")
        self._sim_state_writer = McapWriter()
        self._sim_state_writer.open(save_dir, f"{self.name}-sim_state")

        if self._env is not None:
            self._open_cam_writers(save_dir)
        else:
            self._pending_save_dir = save_dir
        self._recording = True

    def _open_cam_writers(self, save_dir: str) -> None:
        cam_names = getattr(self._env, "camera_names", [])
        self._cam_writers = {}
        for cam_name in cam_names:
            w = AsyncMp4Writer(fps=self._camera_fps)
            w.open(save_dir, f"{self.name}-{cam_name}")
            self._cam_writers[cam_name] = w

    def stop_recording(self) -> str:
        self._recording = False
        for w in [self._left_writer, self._right_writer, self._sim_state_writer]:
            if w is not None and w.is_open:
                w.close()
        for w in self._cam_writers.values():
            if w.is_open:
                w.close()
        self._cam_writers = {}
        self._left_writer = self._right_writer = self._sim_state_writer = None
        return self._recording_save_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_joint_state(self) -> np.ndarray:
        env = self._env
        dim = env.single_timestep_action_dim
        state = np.zeros(dim, dtype=np.float32)
        for i, qpos_idx in enumerate(env._qpos_indices):
            val = float(env.data.qpos[qpos_idx])
            if i in env._gripper_set:
                val = float(np.clip(val / _GRIPPER_CTRL_MAX, 0.0, 1.0))
            state[i] = val
        return state

    def _render_loop(self) -> None:
        """Background thread: viser camera previews + MP4 recording frames."""
        interval = 1.0 / self._camera_fps

        while not self._render_stop.is_set():
            t0 = time.monotonic()
            ts = time.time()

            if self._viser is not None:
                # Viser handles rendering at record resolution + sidebar downsampling.
                # Returns full-res frames for MP4 recording.
                frames = self._viser.update_cam_previews()
            else:
                # No viser — create standalone renderer lazily for recording.
                frames = self._render_standalone(ts)

            if self._recording and frames:
                for cam_name, frame in frames.items():
                    w = self._cam_writers.get(cam_name)
                    if w is not None and w.is_open:
                        w.write("rgb", ts, {"frame": frame})
                    self.publish(cam_name, {"frame": frame}, ts=ts)

            elapsed = time.monotonic() - t0
            remaining = interval - elapsed
            if remaining > 3e-4:
                time.sleep(remaining - 1e-4)

        if self._standalone_renderer is not None:
            try:
                self._standalone_renderer.close()
            except Exception:
                pass

    _standalone_renderer: Any = None

    def _render_standalone(self, ts: float) -> dict[str, Any]:
        """Render cameras without viser (used when viser_port=None)."""
        env = self._env
        if env is None:
            return {}
        if self._standalone_renderer is None:
            try:
                import mujoco
                self._standalone_renderer = mujoco.Renderer(
                    env.model,
                    height=env._camera_height,
                    width=env._camera_width,
                )
            except Exception as exc:
                logger.warning("[%s] could not create renderer: %s", self.name, exc)
                return {}
        frames: dict[str, Any] = {}
        for cam_name in getattr(env, "camera_names", []):
            try:
                self._standalone_renderer.update_scene(env.data, camera=cam_name)
                frames[cam_name] = self._standalone_renderer.render().copy()
            except Exception:
                pass
        return frames

    # ------------------------------------------------------------------
    # YAML config
    # ------------------------------------------------------------------

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        return {
            "name": params["name"],
            "scene": params.get("scene", "hybrid"),
            "task": params.get("task", "bottles"),
            "physics_dt": params.get("physics_dt", 0.0001),
            "control_decimation": params.get("control_decimation", 17),
            "camera_fps": params.get("camera_fps", 30.0),
            "cmd_topics": params.get("cmd_topics", {}),
            "viser_port": params.get("viser_port", 8765),
            "vr_port": params.get("vr_port", 8012),
        }
