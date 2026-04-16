"""ViserMonitorNode — pure-observation visualization node.

Subscribes to joint-state and camera-RGB topics, renders:
  * one ``viser.extras.ViserUrdf`` per arm, configured from the latest joint_pos
  * one viser GUI image panel per camera, resized to a small preview size

No IK, no command publishing, no agent — this node exists only to let a human
see what the hardware is doing from any browser pointed at its port.

Auto-opens the viser URL in a browser when a display is detected. Prefers
``chromium`` / ``google-chrome`` kiosk mode for a fullscreen, chromeless
window that we can gracefully kill on ``cleanup()``. Falls back to
``webbrowser.open()`` (standard tab) if no Chromium-family browser is
installed, and skips entirely on headless machines (no ``DISPLAY`` /
``WAYLAND_DISPLAY``).

YAML example::

    - type: ViserMonitorNode
      name: viz
      port: 8080
      viz_freq: 20
      preview_size: [320, 240]
      auto_open_browser: true
      fullscreen: true
      urdfs:
        yam_left:
          path: dependencies/i2rt/i2rt/robot_models/arm/yam/yam.urdf
          state_topic: yam_left/joint_state
          flip_joints: true      # YAM motor order is reversed vs URDF joint order
          extrinsic:             # optional; omit for an identity pose
            position: [0.0, 0.0, 0.0]
            rotation: [1.0, 0.0, 0.0, 0.0]   # wxyz
        yam_right:
          path: dependencies/i2rt/i2rt/robot_models/arm/yam/yam.urdf
          state_topic: yam_right/joint_state
          flip_joints: true
          extrinsic:
            position: [0.0, 0.5, 0.0]
            rotation: [1.0, 0.0, 0.0, 0.0]
      image_topics:
        top:   camera_top/rgb
        left:  camera_left/rgb
        right: camera_right/rgb
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import webbrowser
from typing import Any

import numpy as np

from robots_realtime.runtime.node import Node, NodeRole
from robots_realtime.sensors.cameras.camera_utils import resize_with_pad

logger = logging.getLogger(__name__)


class ViserMonitorNode(Node):
    """Read-only visualization — URDF overlays and camera panels via viser."""

    role = NodeRole.SENSOR
    poll_freq: float | None = None
    subscriber_driven: bool = False

    def __init__(
        self,
        name: str = "viz",
        port: int = 8080,
        urdfs: dict[str, dict] | None = None,
        image_topics: dict[str, str] | None = None,
        viz_freq: float = 20.0,
        preview_size: tuple[int, int] = (224, 224),
        auto_open_browser: bool = True,
        fullscreen: bool = True,
        # Initial 3D camera view — "+x is the direction the arms point", "+z is up".
        # Default places the viewer ~1.5 m behind the robots, slightly elevated,
        # looking at the approximate workspace centre.
        initial_camera_position: tuple[float, float, float] = (-1.3, 0.3, 0.9),
        initial_camera_look_at: tuple[float, float, float] = (0.45, 0.3, 0.3),
        up_axis: str = "+z",
        writer=None,
        **kwargs,
    ) -> None:
        self._urdfs_spec = urdfs or {}
        self._image_topics = image_topics or {}
        self.subscribed_topics = (
            [spec["state_topic"] for spec in self._urdfs_spec.values() if "state_topic" in spec]
            + list(self._image_topics.values())
        )
        # poll_freq drives how often the URDF/image GUI updates — lower is cheaper.
        self.poll_freq = float(viz_freq)
        super().__init__(name=name, writer=writer, **kwargs)

        self._port = int(port)
        # preview_size is (width, height) — user convention. Converted to
        # (height, width) when calling resize_with_pad which takes H, W.
        if len(preview_size) != 2:
            raise ValueError(f"preview_size must be (width, height); got {preview_size!r}")
        self._preview_w = int(preview_size[0])
        self._preview_h = int(preview_size[1])
        self._auto_open_browser = bool(auto_open_browser)
        self._fullscreen = bool(fullscreen)
        self._initial_camera_position = tuple(float(v) for v in initial_camera_position)
        self._initial_camera_look_at = tuple(float(v) for v in initial_camera_look_at)
        self._up_axis = up_axis

        # Initialised in setup()
        self._server: Any = None
        self._urdf_vis: dict[str, Any] = {}
        self._image_handles: dict[str, Any] = {}
        self._browser_proc: subprocess.Popen | None = None

    # Surfaced on the session TUI (see tui.py:_endpoints_text).
    @property
    def web_endpoints(self) -> list[str]:
        return [f"viser: http://localhost:{self._port}"]

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        import viser  # noqa: PLC0415
        import viser.extras  # noqa: PLC0415
        import yourdfpy  # noqa: PLC0415

        self._server = viser.ViserServer(port=self._port)
        logger.info("[%s] viser server listening on http://localhost:%d", self.name, self._port)

        # World-up direction — determines the natural axis for the orbit
        # controls (e.g. rolling around pitches, not yaws).
        try:
            self._server.scene.set_up_direction(self._up_axis)
        except Exception as exc:
            logger.debug("[%s] set_up_direction(%r) failed: %s", self.name, self._up_axis, exc)

        # Set the initial camera pose for every client that connects.
        init_pos = np.asarray(self._initial_camera_position, dtype=np.float32)
        init_target = np.asarray(self._initial_camera_look_at, dtype=np.float32)

        @self._server.on_client_connect
        def _set_initial_view(client) -> None:  # noqa: ANN001 — viser client type
            client.camera.position = init_pos
            client.camera.look_at = init_target

        for arm_key, spec in self._urdfs_spec.items():
            urdf_path = os.path.abspath(os.path.expanduser(spec["path"]))
            if not os.path.isfile(urdf_path):
                raise FileNotFoundError(f"[{self.name}] URDF for {arm_key!r} not found at {urdf_path}")
            mesh_dir = spec.get("mesh_dir")
            if mesh_dir is not None:
                mesh_dir = os.path.abspath(os.path.expanduser(mesh_dir))
            else:
                # Default sibling "assets/" next to the URDF — matches i2rt layout.
                default_mesh = os.path.join(os.path.dirname(urdf_path), "assets")
                if os.path.isdir(default_mesh):
                    mesh_dir = default_mesh

            urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir) if mesh_dir else yourdfpy.URDF.load(urdf_path)

            root = spec.get("root_node_name", f"/{arm_key}")
            frame = self._server.scene.add_frame(root, show_axes=bool(spec.get("show_axes", False)))
            extrinsic = spec.get("extrinsic")
            if extrinsic is not None:
                if "position" in extrinsic:
                    frame.position = np.asarray(extrinsic["position"], dtype=np.float32)
                if "rotation" in extrinsic:
                    frame.wxyz = np.asarray(extrinsic["rotation"], dtype=np.float32)

            urdf_kwargs: dict = {"root_node_name": root}
            if "mesh_color" in spec and spec["mesh_color"] is not None:
                urdf_kwargs["mesh_color_override"] = tuple(spec["mesh_color"])
            urdf_vis = viser.extras.ViserUrdf(self._server, urdf, **urdf_kwargs)

            opacity = spec.get("opacity")
            if opacity is not None:
                for mesh in urdf_vis._meshes:
                    try:
                        mesh.opacity = float(opacity)
                    except Exception:
                        pass

            self._urdf_vis[arm_key] = urdf_vis
            logger.info("[%s] URDF loaded: %s (root=%s, meshes=%d)", self.name, arm_key, root, len(urdf_vis._meshes))

        if self._auto_open_browser:
            self._open_browser()

    def step(self) -> None:
        # Update URDF configs from the latest joint state.
        for arm_key, spec in self._urdfs_spec.items():
            topic = spec.get("state_topic")
            if topic is None:
                continue
            data = self.get_latest(topic)
            if data is None:
                continue
            jp = data.get("joint_pos")
            if jp is None:
                continue
            cfg = np.asarray(jp, dtype=np.float64)
            if spec.get("flip_joints", True):
                cfg = np.flip(cfg)
            try:
                # ViserUrdf.update_cfg accepts whatever length matches the URDF's
                # actuated joints. If the bus publishes more entries (e.g. gripper),
                # trim to the URDF's joint count.
                expected = len(self._urdf_vis[arm_key]._urdf.actuated_joint_names)  # type: ignore[attr-defined]
                self._urdf_vis[arm_key].update_cfg(cfg[:expected])
            except Exception as exc:
                logger.debug("[%s] URDF update for %s failed: %s", self.name, arm_key, exc)

        # Update camera panels.
        for label, topic in self._image_topics.items():
            msg = self.get_latest(topic)
            if msg is None:
                continue
            img = self._extract_rgb(msg)
            if img is None:
                continue
            # resize_with_pad takes (image, height, width) — not (w, h). The
            # preview_size kwarg is (width, height) for ergonomic YAML config,
            # so we transpose here.
            thumb = resize_with_pad(img, self._preview_h, self._preview_w)
            if label not in self._image_handles:
                self._image_handles[label] = self._server.gui.add_image(thumb, label=label)
            else:
                self._image_handles[label].image = thumb

    def cleanup(self) -> None:
        # Close the browser we spawned (best-effort).
        if self._browser_proc is not None:
            try:
                self._browser_proc.terminate()
                try:
                    self._browser_proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._browser_proc.kill()
            except Exception as exc:
                logger.debug("[%s] browser terminate failed: %s", self.name, exc)
            self._browser_proc = None

        # Stop the viser server.
        if self._server is not None:
            try:
                self._server.stop()
            except Exception as exc:
                logger.debug("[%s] viser stop failed: %s", self.name, exc)
            self._server = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_rgb(msg: dict) -> np.ndarray | None:
        """CameraNode publishes ``{"images": {"rgb": ndarray}, ...}``; also
        tolerate older ``{"frame": ndarray}`` and ``{"rgb": ndarray}`` shapes."""
        if not isinstance(msg, dict):
            return None
        images = msg.get("images")
        if isinstance(images, dict):
            arr = images.get("rgb")
            if arr is None and images:
                arr = next(iter(images.values()))
            if arr is not None:
                return np.asarray(arr)
        for key in ("frame", "rgb"):
            arr = msg.get(key)
            if arr is not None:
                return np.asarray(arr)
        return None

    @staticmethod
    def _active_local_graphical_session_for_uid(uid: int) -> dict[str, str] | None:
        """Return env overrides for the current user's active local X/Wayland session, or None.

        Uses ``loginctl`` to find a session where:
          * ``State=active`` (currently in front of the user)
          * ``Remote=no``   (it's a local seat, not SSH)
          * ``Type`` is ``x11`` or ``wayland``
          * ``User`` matches ``uid`` (so we have read access to its Xauthority)

        This is far more reliable than ``/sys/class/drm/*/status`` — DRM reports
        "disconnected" when the monitor is in DPMS standby or the driver hasn't
        polled recently, giving false negatives even when a display is plugged in.
        A real graphical session in loginctl implies a real physical display.

        Returns the env dict needed to spawn a child process into that session
        (DISPLAY, WAYLAND_DISPLAY, XAUTHORITY, XDG_RUNTIME_DIR as applicable),
        or None if no such session exists.
        """
        if shutil.which("loginctl") is None:
            return None
        try:
            out = subprocess.check_output(
                ["loginctl", "list-sessions", "--no-legend"],
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            ).decode("utf-8", "replace")
        except Exception:
            return None

        for line in out.splitlines():
            parts = line.split()
            if not parts:
                continue
            sid = parts[0]
            try:
                props_out = subprocess.check_output(
                    [
                        "loginctl",
                        "show-session",
                        sid,
                        "--property=Type",
                        "--property=State",
                        "--property=Remote",
                        "--property=User",
                        "--property=Display",
                    ],
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                ).decode("utf-8", "replace")
            except Exception:
                continue
            props: dict[str, str] = {}
            for prop_line in props_out.splitlines():
                if "=" in prop_line:
                    k, v = prop_line.split("=", 1)
                    props[k.strip()] = v.strip()

            if props.get("State") != "active":
                continue
            if props.get("Remote") == "yes":
                continue
            stype = props.get("Type")
            if stype not in ("x11", "wayland"):
                continue
            try:
                session_uid = int(props.get("User", "-1"))
            except ValueError:
                continue
            if session_uid != uid:
                continue

            env_overrides: dict[str, str] = {}
            runtime_dir = f"/run/user/{uid}"
            if os.path.isdir(runtime_dir):
                env_overrides["XDG_RUNTIME_DIR"] = runtime_dir

            if stype == "x11":
                # Prefer the DISPLAY the session advertised (e.g. ":1"); fall back
                # to any local X socket under /tmp/.X11-unix.
                disp = props.get("Display")
                if not disp:
                    for sock in sorted(glob.glob("/tmp/.X11-unix/X*")):
                        num = sock.rsplit("X", 1)[-1]
                        if num.isdigit():
                            disp = f":{num}"
                            break
                if not disp:
                    continue
                env_overrides["DISPLAY"] = disp
                # gdm writes the user's X cookies to /run/user/<uid>/gdm/Xauthority;
                # fall back to the user's $HOME/.Xauthority (pre-systemd).
                for candidate in (
                    f"/run/user/{uid}/gdm/Xauthority",
                    os.path.expanduser(f"~{os.getlogin()}/.Xauthority"),
                ):
                    if os.path.isfile(candidate):
                        env_overrides["XAUTHORITY"] = candidate
                        break
            else:  # wayland
                for sock in sorted(glob.glob(f"{runtime_dir}/wayland-*")):
                    if sock.endswith(".lock"):
                        continue
                    env_overrides["WAYLAND_DISPLAY"] = os.path.basename(sock)
                    break
                if "WAYLAND_DISPLAY" not in env_overrides:
                    continue

            return env_overrides
        return None

    @staticmethod
    def _fallback_display_from_sockets() -> dict[str, str]:
        """Best-effort DISPLAY discovery when loginctl isn't available or finds nothing.

        Just points at the first X socket or Wayland socket we can find —
        works when the current user has a running session on the local seat
        but loginctl isn't exposing it cleanly.
        """
        for sock in sorted(glob.glob("/tmp/.X11-unix/X*")):
            num = sock.rsplit("X", 1)[-1]
            if num.isdigit():
                return {"DISPLAY": f":{num}"}
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.geteuid()}"
        if os.path.isdir(runtime_dir):
            for sock in sorted(glob.glob(f"{runtime_dir}/wayland-*")):
                if sock.endswith(".lock"):
                    continue
                return {"WAYLAND_DISPLAY": os.path.basename(sock), "XDG_RUNTIME_DIR": runtime_dir}
        return {}

    def _open_browser(self) -> None:
        url = f"http://localhost:{self._port}"

        # Only auto-launch when BOTH of these are true:
        #   1. This terminal has no DISPLAY / WAYLAND_DISPLAY set (we're in SSH
        #      or a bare tty) — because if the user is sitting at the machine
        #      running the session from a graphical terminal, chromium --kiosk
        #      would cover the terminal and kiosk mode traps input, leaving
        #      them no way to stop the session.
        #   2. A local graphical session (monitor + logged-in desktop) exists
        #      for our uid — so there's actually a physical display to pop up on.
        this_terminal_is_graphical = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        if this_terminal_is_graphical:
            logger.info(
                "[%s] launched from a graphical terminal (DISPLAY set) — "
                "skipping auto-browser so the console stays reachable. Visit %s",
                self.name, url,
            )
            return

        env_overrides = self._active_local_graphical_session_for_uid(os.geteuid())
        source = "loginctl"
        if not env_overrides:
            env_overrides = self._fallback_display_from_sockets()
            source = "socket_probe"
        if not env_overrides:
            logger.info(
                "[%s] no local graphical session found for uid=%d — skipping auto-browser; visit %s",
                self.name, os.geteuid(), url,
            )
            return

        display_repr = (
            env_overrides.get("DISPLAY") or env_overrides.get("WAYLAND_DISPLAY") or "?"
        )

        # A display is targetable. Prefer a Chromium-family browser in kiosk
        # mode so the window is fullscreen and we can kill the process on
        # cleanup. Firefox also has a --kiosk flag. Fall back to the stdlib
        # webbrowser module if none of those are installed.
        kiosk_candidates: list[list[str]] = []
        if self._fullscreen:
            for browser in (
                "chromium",
                "chromium-browser",
                "google-chrome",
                "google-chrome-stable",
                "firefox",
            ):
                bin_path = shutil.which(browser)
                if bin_path is None:
                    continue
                kiosk_candidates.append([bin_path, "--kiosk", url])

        child_env = os.environ.copy()
        child_env.update(env_overrides)

        for cmd in kiosk_candidates:
            try:
                self._browser_proc = subprocess.Popen(
                    cmd,
                    env=child_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                logger.info(
                    "[%s] opened %s in kiosk mode (via %s, display=%s) -> %s",
                    self.name, os.path.basename(cmd[0]), source, display_repr, url,
                )
                return
            except Exception as exc:
                logger.warning("[%s] failed to spawn %s: %s", self.name, cmd[0], exc)

        # Fallback: regular tab (can't close on exit, but at least opens).
        try:
            webbrowser.open(url)
            logger.info("[%s] opened default browser (fallback) -> %s", self.name, url)
        except Exception as exc:
            logger.warning("[%s] could not open browser at %s: %s", self.name, url, exc)

    # ------------------------------------------------------------------ #
    # YAML wiring
    # ------------------------------------------------------------------ #

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        return {
            "name":                    params["name"],
            "port":                    params.get("port", 8080),
            "urdfs":                   params.get("urdfs") or {},
            "image_topics":            params.get("image_topics") or {},
            "viz_freq":                params.get("viz_freq", 20.0),
            "preview_size":            tuple(params.get("preview_size", (240, 180))),
            "auto_open_browser":       params.get("auto_open_browser", True),
            "fullscreen":              params.get("fullscreen", True),
            "initial_camera_position": tuple(params.get("initial_camera_position", (-1.3, 0.3, 0.9))),
            "initial_camera_look_at":  tuple(params.get("initial_camera_look_at",  (0.45, 0.3, 0.3))),
            "up_axis":                 params.get("up_axis", "+z"),
        }
