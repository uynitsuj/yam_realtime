"""Stage A — reconstruct the observations a deployed OpenPI policy actually saw.

Reads a ``robots_realtime`` episode directory (recorded by an ``AgentNode``
running ``AsyncDiffusionAgent`` against an OpenPI websocket server) and rebuilds,
for a strided grid of policy ticks, the exact ``{state, 3 images}`` observation
that was shipped over the ZMQ bus to the policy.

Why this needs care
-------------------
The on-disk MP4s keep the FULL-resolution, FULL-FOV frame, but the frames that
reached the policy went through ``CameraNode``'s bus-payload path:

    publish_fov_crop  ->  publish_resize_mode(publish_resize)

and then through ``AsyncDiffusionAgent._center_crop_and_resize(224, 224)``
(a no-op on an already-square 224x224 frame). We replay that exact chain here by
importing the same functions from ``camera_node``, so the reconstruction differs
from the live bus frame only by MP4 compression loss.

State layout follows ``ModelIOConfig.mlp_keys``:

    [left joint_pos (6), left gripper_pos (1), right joint_pos (6), right gripper_pos (1)]

Usage
-----
    uv run python scripts/uq_vla/extract_obs.py <episode_dir> --out obs.npz
    uv run python scripts/uq_vla/extract_obs.py <session_root> --glob --out-dir out/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots_realtime.runtime.environment.camera_node import (  # noqa: E402
    _center_crop_and_resize,
    _center_fov_crop,
)

# Camera role -> (recording prefix, publish_fov_crop) from the deployment config
# (configs/yam/yam_bimanual_openpi_policy_xdof_hq_us0{5,7}_yam_box_abc.yaml).
# The top camera is FOV-cropped to 0.88 on the bus; the wrists are not.
CAMERAS: dict[str, tuple[str, float]] = {
    "top_camera": ("camera_top", 0.88),
    "left_camera": ("camera_left", 1.0),
    "right_camera": ("camera_right", 1.0),
}
ARMS = ("left", "right")
POLICY_INPUT_HW = (224, 224)


def _read_mcap_json(path: Path, topic: str | None = None) -> list[tuple[float, dict]]:
    """Return [(ts_seconds, payload), ...] from a JSON-encoded MCAP file."""
    from mcap.reader import make_reader

    out: list[tuple[float, dict]] = []
    with open(path, "rb") as f:
        for _, channel, msg in make_reader(f).iter_messages():
            if topic is not None and channel.topic != topic:
                continue
            out.append((msg.log_time / 1e9, json.loads(msg.data)))
    out.sort(key=lambda r: r[0])
    return out


def _nearest(query: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each t in `query`, index of the nearest entry in sorted `ref` and |dt|."""
    idx = np.searchsorted(ref, query)
    idx = np.clip(idx, 1, len(ref) - 1)
    left, right = ref[idx - 1], ref[idx]
    take_left = (query - left) < (right - query)
    idx = np.where(take_left, idx - 1, idx)
    return idx, np.abs(ref[idx] - query)


def _decode_frames(mp4: Path, wanted: np.ndarray, fov_crop: float) -> np.ndarray:
    """Stream-decode only `wanted` frame indices, applying the bus preprocessing.

    Sequential decode with a keep-set — an episode's 3 cameras at full VGA would
    be ~4.5 GB in memory if fully decoded, and the strided grid needs ~10% of it.
    """
    import av

    keep = {int(i): k for k, i in enumerate(wanted)}
    out = np.zeros((len(wanted), *POLICY_INPUT_HW, 3), dtype=np.uint8)
    seen = np.zeros(len(wanted), dtype=bool)

    with av.open(str(mp4)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            slot = keep.get(i)
            if slot is None:
                continue
            img = frame.to_ndarray(format="rgb24")
            img = _center_fov_crop(img, fov_crop)
            out[slot] = _center_crop_and_resize(img, *POLICY_INPUT_HW)
            seen[slot] = True
            if seen.all():
                break

    if not seen.all():
        raise RuntimeError(f"{mp4.name}: could not decode {int((~seen).sum())} requested frames")
    return out


def extract(episode_dir: Path, stride: int, max_dt: float) -> dict[str, np.ndarray]:
    """Rebuild policy observations on a strided grid of policy command ticks."""
    policy_mcap = episode_dir / "openpi_policy.mcap"
    if not policy_mcap.exists():
        raise FileNotFoundError(f"{episode_dir}: no openpi_policy.mcap")

    # The command stream is the policy's consumer tick — the true observation
    # timeline. Inference fired less often, but our stride is our own choice.
    cmd = {side: _read_mcap_json(policy_mcap, f"/openpi_policy/{side}_pos") for side in ARMS}
    tick_ts = np.array([t for t, _ in cmd["left"]], dtype=np.float64)
    grid = tick_ts[::stride]

    # --- Proprioception -----------------------------------------------------
    arm_state: dict[str, np.ndarray] = {}
    for side in ARMS:
        msgs = _read_mcap_json(episode_dir / f"yam_{side}.mcap", f"/yam_{side}/joint_state")
        ts = np.array([t for t, _ in msgs], dtype=np.float64)
        jp = np.array([d["joint_pos"] for _, d in msgs], dtype=np.float32)
        gp = np.array([d["gripper_pos"] for _, d in msgs], dtype=np.float32).reshape(len(msgs), -1)
        idx, dt = _nearest(grid, ts)
        if dt.max() > max_dt:
            raise RuntimeError(f"{episode_dir.name}: yam_{side} gap {dt.max():.3f}s > {max_dt}s")
        arm_state[side] = np.concatenate([jp[idx], gp[idx]], axis=-1)  # (N, 7)

    state = np.concatenate([arm_state["left"], arm_state["right"]], axis=-1)  # (N, 14)

    # --- Images -------------------------------------------------------------
    images: dict[str, np.ndarray] = {}
    cam_dt: dict[str, np.ndarray] = {}
    for role, (prefix, fov_crop) in CAMERAS.items():
        mp4 = episode_dir / f"{prefix}-images-rgb.mp4"
        ts = np.load(episode_dir / f"{prefix}-rgb-timestamp.npy").astype(np.float64)
        idx, dt = _nearest(grid, ts)
        if dt.max() > max_dt:
            raise RuntimeError(f"{episode_dir.name}: {prefix} gap {dt.max():.3f}s > {max_dt}s")
        images[role] = _decode_frames(mp4, idx, fov_crop)
        cam_dt[role] = dt

    # The action actually commanded at each grid tick — ground truth to plot
    # the uncertainty signal against.
    cmd_action = np.concatenate(
        [
            np.array([d["joint_pos"] for _, d in cmd[side]], dtype=np.float32)[::stride]
            for side in ARMS
        ],
        axis=-1,
    )[: len(grid)]

    return {
        "ts": grid,
        "t_rel": grid - tick_ts[0],
        "state": state.astype(np.float32),
        "cmd_action": cmd_action,
        "stride": np.int32(stride),
        "episode": np.array(episode_dir.name),
        **{f"image_{role}": arr for role, arr in images.items()},
        **{f"cam_dt_{role}": arr for role, arr in cam_dt.items()},
    }


def _episode_dirs(root: Path) -> list[Path]:
    if (root / "openpi_policy.mcap").exists():
        return [root]
    return sorted(p.parent for p in root.glob("**/openpi_policy.mcap"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="Episode dir, or a session root to walk.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=10, help="Policy ticks between eval points (30 Hz ticks).")
    ap.add_argument("--max-dt", type=float, default=0.1, help="Max allowed nearest-neighbour time gap (s).")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N episodes.")
    args = ap.parse_args()

    episodes = _episode_dirs(args.root)
    if args.limit:
        episodes = episodes[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(episodes)} episode(s) -> {args.out_dir}")

    for i, ep in enumerate(episodes, 1):
        out = args.out_dir / f"{ep.name}.npz"
        if out.exists():
            print(f"[{i}/{len(episodes)}] {ep.name}: cached")
            continue
        try:
            data = extract(ep, args.stride, args.max_dt)
        except Exception as e:  # noqa: BLE001 — one bad episode shouldn't kill the sweep
            print(f"[{i}/{len(episodes)}] {ep.name}: SKIPPED ({type(e).__name__}: {e})")
            continue
        np.savez_compressed(out, **data)
        worst = max(float(data[f"cam_dt_{r}"].max()) for r in CAMERAS)
        print(
            f"[{i}/{len(episodes)}] {ep.name}: {len(data['ts'])} frames, "
            f"{data['t_rel'][-1]:.1f}s, worst cam align {worst * 1e3:.0f} ms"
        )


if __name__ == "__main__":
    main()
