"""Viser replay viewer for robots_realtime sim episode recordings.

Loads an episode directory recorded by robots_realtime and replays it in an
interactive Viser browser UI.  Two modes are available:

  qpos    — directly restore the recorded MuJoCo qpos at each step (exact replay,
             requires yam-sim_state.mcap; preferred for sim episodes).
  physics — feed the recorded gello actions back through step_single (useful for
             checking policy/action consistency).

Both modes run at the control rate (~30 Hz with default physics_dt=0.002, decimation=17).

Usage:
    rr-replay recordings/20260323/episode_*/
    rr-replay <episode_dir> --port 8080 --speed 2.0
    rr-replay <episode_dir> --scene hybrid --task bottles
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# MCAP / data loading
# ---------------------------------------------------------------------------


def _read_mcap_json(path: Path) -> list[tuple[float, dict]]:
    """Return [(ts_seconds, data_dict), ...] from a JSON-encoded MCAP file."""
    from mcap.reader import make_reader

    results: list[tuple[float, dict]] = []
    with open(path, "rb") as f:
        for _, _, msg in make_reader(f).iter_messages():
            ts = msg.log_time / 1e9  # nanoseconds → seconds
            results.append((ts, json.loads(msg.data)))
    return results


def _load_episode(episode_dir: Path) -> dict:
    """Load all streams from a robots_realtime sim episode directory.

    Returns:
        actions_left, ts_left   — (N, 7) float64, (N,) float64
        actions_right, ts_right — (N, 7) float64, (N,) float64
        camera_frames           — {cam: (T, H, W, 3) uint8}
        camera_ts               — {cam: (T,) float64}
        episode_dir             — Path
    """
    print(f"Loading episode: {episode_dir}")

    # --- Actions from gello MCAP files ---
    def _load_arm(fname: str, label: str):
        path = episode_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing {fname} in {episode_dir}")
        msgs = _read_mcap_json(path)
        arr = np.array([d["joint_pos"] for _, d in msgs], dtype=np.float64)
        ts  = np.array([t for t, _ in msgs], dtype=np.float64)
        dur = ts[-1] - ts[0] if len(ts) > 1 else 0.0
        hz  = (len(ts) - 1) / dur if dur > 0 else 0.0
        print(f"  {label}: {len(arr)} frames, {dur:.1f}s at ~{hz:.0f} Hz  (shape {arr.shape})")
        return arr, ts

    actions_left,  ts_left  = _load_arm("gello_left.mcap",  "gello_left")
    actions_right, ts_right = _load_arm("gello_right.mcap", "gello_right")

    # --- Camera videos ---
    camera_frames: dict[str, np.ndarray] = {}
    camera_ts:     dict[str, np.ndarray] = {}
    for cam in ("left", "right", "top"):
        mp4 = episode_dir / f"yam-{cam}-images-rgb.mp4"
        npy = episode_dir / f"yam-{cam}-rgb-timestamp.npy"
        if not mp4.exists():
            continue
        try:
            import imageio.v3 as iio
            frames = iio.imread(str(mp4), plugin="pyav")  # (T, H, W, 3)
            ts_arr = np.load(str(npy)) if npy.exists() else np.linspace(ts_left[0], ts_left[-1], len(frames))
            camera_frames[cam] = frames
            camera_ts[cam] = ts_arr
            h, w = frames.shape[1], frames.shape[2]
            print(f"  camera '{cam}': {len(frames)} frames ({w}x{h}), {ts_arr[-1]-ts_arr[0]:.1f}s")
        except Exception as exc:
            print(f"  Warning: could not load camera '{cam}': {exc}")

    return {
        "actions_left":  actions_left,
        "ts_left":       ts_left,
        "actions_right": actions_right,
        "ts_right":      ts_right,
        "camera_frames": camera_frames,
        "camera_ts":     camera_ts,
        "episode_dir":   episode_dir,
    }


def _load_sim_states(episode_dir: Path):
    """Load full qpos timeline from yam-sim_state.mcap.

    Returns (qposes, timestamps) or None if file is absent.
    """
    mcap_file = episode_dir / "yam-sim_state.mcap"
    if not mcap_file.exists():
        return None
    msgs = _read_mcap_json(mcap_file)
    qposes = np.array([d["qpos"] for _, d in msgs], dtype=np.float64)
    timestamps = np.array([t for t, _ in msgs], dtype=np.float64)
    print(f"  sim_state: {len(qposes)} frames (nq={qposes.shape[1]}), {timestamps[-1]-timestamps[0]:.1f}s")
    return qposes, timestamps


def _read_sim_config(episode_dir: Path) -> dict:
    """Read scene/task from session_meta.json if stored by the sim node."""
    meta_file = episode_dir / "session_meta.json"
    if not meta_file.exists():
        return {}
    with open(meta_file) as f:
        meta = json.load(f)
    for node in meta.get("nodes", []):
        cfg = node.get("sim_config", {})
        if "scene" in cfg or "task" in cfg:
            return cfg
    return {}


# ---------------------------------------------------------------------------
# Timeline helpers
# ---------------------------------------------------------------------------


def _sample_hold(values: np.ndarray, ts: np.ndarray, grid_ts: np.ndarray) -> np.ndarray:
    """Sample-and-hold: for each grid timestamp pick the latest value at or before it."""
    out = np.empty((len(grid_ts),) + values.shape[1:], dtype=values.dtype)
    for i, t in enumerate(grid_ts):
        idx = max(0, int(np.searchsorted(ts, t, side="right")) - 1)
        out[i] = values[min(idx, len(values) - 1)]
    return out


def _build_action_timeline(
    actions_left: np.ndarray, ts_left: np.ndarray,
    actions_right: np.ndarray, ts_right: np.ndarray,
    control_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample async left/right streams onto a regular control grid.

    Returns:
        actions: (T, 14) float32 — [left_7, right_7]
        grid_ts: (T,) float64
    """
    t_start = max(ts_left[0],  ts_right[0])
    t_end   = min(ts_left[-1], ts_right[-1])
    grid_ts = np.arange(t_start, t_end, 1.0 / control_hz)

    left_a  = _sample_hold(actions_left,  ts_left,  grid_ts).astype(np.float32)
    right_a = _sample_hold(actions_right, ts_right, grid_ts).astype(np.float32)
    actions = np.concatenate([left_a, right_a], axis=1)
    return actions, grid_ts


def _get_camera_frame(
    cam_name: str,
    query_ts: float,
    camera_frames: dict[str, np.ndarray],
    camera_ts: dict[str, np.ndarray],
) -> np.ndarray | None:
    frames = camera_frames.get(cam_name)
    ts_arr = camera_ts.get(cam_name)
    if frames is None or ts_arr is None or len(frames) == 0:
        return None
    idx = max(0, int(np.searchsorted(ts_arr, query_ts, side="right")) - 1)
    return frames[min(idx, len(frames) - 1)]


# ---------------------------------------------------------------------------
# Viser replay viewer
# ---------------------------------------------------------------------------


class EpisodeReplayViewer:
    """Interactive Viser 3D viewer for replaying robots_realtime sim episodes.

    Holds two timelines (both at ~30 Hz with physics_dt=0.002, decimation=17):
      qpos mode   — directly restores recorded MuJoCo qpos (exact replay).
      physics mode — feeds actions through step_single (re-simulation).
    """

    def __init__(
        self,
        env,
        *,
        actions_physics: np.ndarray,
        grid_ts_physics: np.ndarray,
        qpos_actions: np.ndarray | None = None,
        grid_ts_qpos: np.ndarray | None = None,
        camera_frames: dict[str, np.ndarray],
        camera_ts: dict[str, np.ndarray],
        sim_states: np.ndarray | None = None,
        port: int = 8080,
        speed: float = 1.0,
    ):
        import viser
        import viser.transforms as vtf
        from robots_realtime.nodes.sim._mujoco_viser import (
            _get_body_name,
            _is_fixed_body,
            _merge_geoms,
            configure_default_camera,
        )
        from mujoco import mj_id2name, mjtGeom, mjtObj

        self.env    = env
        self.model  = env.model
        self.data   = env.data
        self.camera_frames = camera_frames
        self.camera_ts     = camera_ts
        self._speed = speed
        self._vtf   = vtf
        self._get_body_name = _get_body_name
        self._is_fixed_body = _is_fixed_body
        self._merge_geoms   = _merge_geoms
        self._mj_id2name    = mj_id2name
        self._mjtGeom       = mjtGeom
        self._mjtObj        = mjtObj

        # Two timelines — active one selected by mode.
        self._sim_states      = sim_states       # (T_q, nq)  qpos at recorded rate
        self._qpos_actions    = qpos_actions     # (T_q, 14)  actions aligned to qpos grid
        self._grid_ts_qpos    = grid_ts_qpos     # (T_q,)     ~30 Hz
        self._actions_physics = actions_physics  # (T_p, 14)  actions at control rate
        self._grid_ts_physics = grid_ts_physics  # (T_p,)     ~588 Hz

        self._step_idx = 0
        self._paused   = True
        self._mode     = "qpos" if sim_states is not None else "physics"
        self._mesh_handles: dict[int, object] = {}

        self.server = viser.ViserServer(port=port)
        configure_default_camera(self.server)
        print(f"Viser: http://localhost:{port}")

        self._build_scene()

        self._cam_handles: dict[str, object] = {}
        if camera_frames:
            with self.server.gui.add_folder("Recorded Cameras"):
                for cam_name, frames in camera_frames.items():
                    h, w = frames.shape[1], frames.shape[2]
                    self._cam_handles[cam_name] = self.server.gui.add_image(
                        np.zeros((h, w, 3), dtype=np.uint8),
                        label=cam_name, format="jpeg", jpeg_quality=85,
                    )

        self._build_gui()
        self._update_scene()
        self._update_cameras(0)

    # ------------------------------------------------------------------
    # Active-timeline accessors (switch on self._mode)
    # ------------------------------------------------------------------

    @property
    def _actions(self) -> np.ndarray:
        if self._mode == "qpos" and self._qpos_actions is not None:
            return self._qpos_actions
        return self._actions_physics

    @property
    def _grid_ts(self) -> np.ndarray:
        if self._mode == "qpos" and self._grid_ts_qpos is not None:
            return self._grid_ts_qpos
        return self._grid_ts_physics

    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        model = self.model
        _get_body_name = self._get_body_name
        _is_fixed_body = self._is_fixed_body
        _merge_geoms   = self._merge_geoms
        mjtGeom = self._mjtGeom
        mjtObj  = self._mjtObj
        mj_id2name = self._mj_id2name

        visible_groups = (0, 1, 2)
        body_visual: dict[int, list[int]] = {}
        for i in range(model.ngeom):
            if int(model.geom_group[i]) in visible_groups:
                body_visual.setdefault(int(model.geom_bodyid[i]), []).append(i)

        self.server.scene.add_frame("/fixed_bodies", show_axes=False)
        for body_id, visual_ids in body_visual.items():
            body_name = _get_body_name(model, body_id)
            if _is_fixed_body(model, body_id):
                plane_ids    = [g for g in visual_ids if model.geom_type[g] == mjtGeom.mjGEOM_PLANE]
                nonplane_ids = [g for g in visual_ids if model.geom_type[g] != mjtGeom.mjGEOM_PLANE]
                for gid in plane_ids:
                    name = mj_id2name(model, mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
                    self.server.scene.add_grid(
                        f"/fixed_bodies/{body_name}/{name}",
                        width=2000.0, height=2000.0,
                        position=model.geom_pos[gid], wxyz=model.geom_quat[gid],
                    )
                if nonplane_ids:
                    merged = _merge_geoms(model, nonplane_ids)
                    self.server.scene.add_mesh_trimesh(
                        f"/fixed_bodies/{body_name}", merged,
                        position=model.body(body_id).pos,
                        wxyz=model.body(body_id).quat,
                    )
            elif visual_ids:
                merged = _merge_geoms(model, visual_ids)
                handle = self.server.scene.add_mesh_trimesh(f"/bodies/{body_name}", merged, visible=True)
                self._mesh_handles[body_id] = handle

    def _build_gui(self) -> None:
        import viser

        with self.server.gui.add_folder("Info"):
            self._status_html = self.server.gui.add_markdown("")
            self._update_status()

        with self.server.gui.add_folder("Playback"):
            self._play_btn = self.server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)

            @self._play_btn.on_click
            def _(_) -> None:
                self._paused = not self._paused
                self._play_btn.label = "Play" if self._paused else "Pause"
                self._play_btn.icon = (
                    viser.Icon.PLAYER_PLAY if self._paused else viser.Icon.PLAYER_PAUSE
                )
                self._update_status()

            step_btn = self.server.gui.add_button("Step", icon=viser.Icon.PLAYER_TRACK_NEXT)

            @step_btn.on_click
            def _(_) -> None:
                self._paused = True
                self._play_btn.label = "Play"
                self._play_btn.icon = viser.Icon.PLAYER_PLAY
                self._sim_step()
                self._update_status()

            reset_btn = self.server.gui.add_button("Reset", icon=viser.Icon.REFRESH)

            @reset_btn.on_click
            def _(_) -> None:
                self._paused = True
                self._play_btn.label = "Play"
                self._play_btn.icon = viser.Icon.PLAYER_PLAY
                self._reset()

            self._speed_slider = self.server.gui.add_slider(
                "Speed", min=0.1, max=5.0, step=0.1, initial_value=self._speed
            )

            @self._speed_slider.on_update
            def _(_) -> None:
                self._speed = self._speed_slider.value

            mode_opts    = ["qpos (exact)", "physics (re-step)"]
            initial_mode = mode_opts[0] if self._sim_states is not None else mode_opts[1]
            self._mode_dropdown = self.server.gui.add_dropdown(
                "Replay mode", options=mode_opts, initial_value=initial_mode
            )
            if self._sim_states is None:
                self._mode_dropdown.disabled = True

            @self._mode_dropdown.on_update
            def _(_) -> None:
                self._mode = "qpos" if "qpos" in self._mode_dropdown.value else "physics"
                self._reset()  # step_idx=0, re-renders at new rate

    def _update_scene(self) -> None:
        vtf = self._vtf
        with self.server.atomic():
            for body_id, handle in self._mesh_handles.items():
                handle.position = self.data.xpos[body_id]
                xmat = self.data.xmat[body_id].reshape(3, 3)
                handle.wxyz = vtf.SO3.from_matrix(xmat).wxyz
            self.server.flush()

    def _update_cameras(self, step_idx: int) -> None:
        if not self._cam_handles:
            return
        grid = self._grid_ts
        ts = grid[min(step_idx, len(grid) - 1)]
        for cam_name, handle in self._cam_handles.items():
            frame = _get_camera_frame(cam_name, ts, self.camera_frames, self.camera_ts)
            if frame is not None:
                handle.image = frame

    def _sim_step(self) -> None:
        actions = self._actions
        if self._step_idx >= len(actions):
            return
        if self._mode == "qpos" and self._sim_states is not None:
            qpos = self._sim_states[min(self._step_idx, len(self._sim_states) - 1)]
            nq = min(len(qpos), len(self.data.qpos))
            self.data.qpos[:nq] = qpos[:nq]
            mujoco.mj_forward(self.model, self.data)
        else:
            self.env.step_single(actions[self._step_idx].astype(np.float32))
        self._step_idx += 1
        self._update_scene()
        self._update_cameras(self._step_idx)

    def _reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        q0 = self.env.get_init_q()
        self.data.qpos[:len(q0)] = q0
        mujoco.mj_forward(self.model, self.data)
        self._step_idx = 0
        self._update_scene()
        self._update_cameras(0)
        self._update_status()

    def _update_status(self) -> None:
        total = len(self._actions)
        grid  = self._grid_ts
        pct   = f"{100 * self._step_idx / total:.0f}%" if total > 0 else "0%"
        dur   = (grid[-1] - grid[0]) if len(grid) > 1 else 0.0
        elap  = (grid[min(self._step_idx, len(grid) - 1)] - grid[0]) if self._step_idx > 0 else 0.0
        done  = " ✓ DONE" if self._step_idx >= total else ""
        self._status_html.content = (
            f"**{'Paused' if self._paused else 'Playing'}**{done}  \n"
            f"Step: {self._step_idx}/{total} ({pct})  \n"
            f"Time: {elap:.1f}s / {dur:.1f}s  \n"
            f"Mode: {self._mode}  \n"
            f"Speed: {self._speed:.1f}x"
        )

    def run(self) -> None:
        print("Replay viewer running — open the URL above in a browser.")
        print("Press Ctrl-C to quit.")
        try:
            while True:
                t0 = time.monotonic()
                if not self._paused and self._step_idx < len(self._actions):
                    self._sim_step()
                    self._update_status()
                grid = self._grid_ts
                dt = (grid[1] - grid[0]) if len(grid) > 1 else 1.0 / 30.0
                elapsed  = time.monotonic() - t0
                to_sleep = (dt / self._speed) - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
        except KeyboardInterrupt:
            print("\nShutting down.")
            self.server.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a robots_realtime sim episode through MuJoCo + Viser"
    )
    parser.add_argument("episode_dir", help="Path to episode directory")
    parser.add_argument("--port",  type=int,   default=8080, help="Viser server port (default: 8080)")
    parser.add_argument("--speed", type=float, default=1.0,  help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--scene", type=str,   default=None, help="Scene name (default: from session_meta or 'hybrid')")
    parser.add_argument("--task",  type=str,   default=None, help="Task name (default: from session_meta or 'bottles')")
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir).resolve()

    # Scene / task: prefer CLI args, then session_meta, then defaults.
    sim_cfg = _read_sim_config(episode_dir)
    scene = args.scene or sim_cfg.get("scene", "hybrid")
    task  = args.task  or sim_cfg.get("task",  "bottles")

    # Load episode data
    data = _load_episode(episode_dir)
    print()

    # Load sim states for exact qpos replay
    raw_states = _load_sim_states(episode_dir)

    # Build MuJoCo environment (scene_variant = visual style, task = scene XML)
    print(f"Creating environment (scene_variant={scene}, task={task}) ...")
    import robots_realtime.sim as sim

    env = sim.make_env(scene_variant=scene, task=task, render_cameras=False)
    env.reset()

    control_hz = 1.0 / (env.model.opt.timestep * env._control_decimation)
    print(f"Control Hz: {control_hz:.1f}")

    # Build the physics-rate action timeline (needed for re-step mode).
    actions_physics, grid_ts_physics = _build_action_timeline(
        data["actions_left"],  data["ts_left"],
        data["actions_right"], data["ts_right"],
        control_hz=control_hz,
    )
    print(f"Physics timeline: {len(actions_physics)} steps at {control_hz:.0f} Hz, {grid_ts_physics[-1]-grid_ts_physics[0]:.1f}s")

    # Align sim_states to control grid (same rate as physics mode).
    sim_states: np.ndarray | None = None
    if raw_states is not None:
        qposes_raw, ts_raw = raw_states
        sim_states = _sample_hold(qposes_raw, ts_raw, grid_ts_physics)
        print(f"Sim states:      {len(sim_states)} steps aligned to {control_hz:.1f} Hz grid")
    else:
        print("No yam-sim_state.mcap — only physics re-step mode available")

    viewer = EpisodeReplayViewer(
        env,
        actions_physics=actions_physics,
        grid_ts_physics=grid_ts_physics,
        qpos_actions=actions_physics,
        grid_ts_qpos=grid_ts_physics,
        camera_frames=data["camera_frames"],
        camera_ts=data["camera_ts"],
        sim_states=sim_states,
        port=args.port,
        speed=args.speed,
    )
    viewer.run()


if __name__ == "__main__":
    main()
