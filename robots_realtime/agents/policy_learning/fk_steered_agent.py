"""FK-objective steering of a frozen VLA at the sampler level (anchor P0).

The idea, in one paragraph: pi0 turns an initial noise sample into an action
chunk deterministically, so the noise is a handle on *which* behaviour you get.
Draw K noises in ONE batched server call, decode each candidate chunk's joint
angles through forward kinematics into a hand path, score that path by its
closest approach to a commanded 3-D anchor, and execute the best candidate. No
training, no change to the policy.

Integration
-----------
Everything here hangs off a second wrapper around
``self._websocket_client_policy.infer``. Every inference path in
``AsyncDiffusionAgent`` — the sync ActionChunkBroker, the async thread, temporal
ensemble — funnels through that one call and receives ``{"actions": (H, 14)}``
back, so wrapping it makes steering transparent to all of them and none of the
chunking / smoothing / RTC machinery needs to know this exists.

Safety and bring-up
-------------------
``steer_mode`` has three settings and the default is the safe one:

    off     no steering, no batching. Byte-identical to AsyncDiffusionAgent.
    dryrun  draws K candidates and scores them, logs which one selection WOULD
            have picked, and executes candidate 0 anyway. Nothing changes at the
            arm. This is a genuine paired comparison — same batch, same state,
            the only difference is the choice — so the log answers "would this
            have helped?" before it is ever allowed to steer.
    on      executes the argmin candidate.

Command lifecycle
-----------------
A command is not for the whole episode. P0_FKRAY measured ~1 bottle/episode lost
to a stale anchor: after the target is grasped, the anchor still points at the
now-empty table and selection starts ranking candidates by distance to nothing,
causing set-downs and bin-clips. ``release_at_grasp`` retires the anchor when the
commanded arm's gripper closes near it, after which inference reverts to the
unsteered path until a new anchor arrives.

Requires the openpi fork with the K-batching seam in ``policies/policy.py``
(noise leading dim declares K). With stock openpi the ``noise`` key is silently
ignored and every candidate is identical — which is why ``_assert_seam`` runs
once at the first steered inference rather than trusting it.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from robots_realtime.agents.policy_learning.async_pi0_agent import AsyncDiffusionAgent

logger = logging.getLogger(__name__)

# ── YAM forward kinematics ──────────────────────────────────────────────────
# (body pos xyz, body quat wxyz) for link_1..link_6, chain order. Transcribed
# from the i2rt YAM MJCF and verified against the deployed URDF
# (dependencies/i2rt/i2rt/robot_models/arm/yam/yam.urdf) to 0.00 mm over 16
# random configurations, with NO joint sign flips. Every hinge is +z (MuJoCo's
# default axis); quats are unnormalised in the source and normalised on load.
_CHAIN = [
    ((0.0, 0.0, 0.0631), (1.0, 0.0, 0.0, 1.0)),
    ((2.5e-05, -0.02, 0.0409), (1.0, 1.0, 1.0, 1.0)),
    ((0.264, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
    ((-0.245, -0.06, 0.0), (1.0, 0.0, 0.0, 0.0)),
    ((-0.074, -0.0395, 2.5e-05), (1.0, -1.0, 1.0, 1.0)),
    ((0.0, 0.0353, 0.0395), (1.0, -1.0, 0.0, 0.0)),
]
# grasp_site in link_6. This offset is what makes FK agree with the URDF; it is
# the point between the fingers, which is the thing that must reach the anchor.
_GRASP_SITE = np.array([0.0, 0.0, 0.1347, 1.0])
NUM_ARM_JOINTS = 6

# 14-d action layout: left 6 joints + left gripper, right 6 joints + right gripper.
_ARM_SLICE = {"left": slice(0, 6), "right": slice(7, 13)}
_GRIP_IDX = {"left": 6, "right": 13}


def _quat_to_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _fixed_transforms() -> np.ndarray:
    out = np.zeros((NUM_ARM_JOINTS, 4, 4))
    for i, (pos, quat) in enumerate(_CHAIN):
        out[i, :3, :3] = _quat_to_mat(*quat)
        out[i, :3, 3] = pos
        out[i, 3, 3] = 1.0
    return out


_FIXED = _fixed_transforms()


def fk_grasp_site(q: np.ndarray) -> np.ndarray:
    """Batched FK. ``q``: (..., >=6) joint angles -> (..., 3) grasp-site position
    in that arm's own base frame. Trailing columns (the gripper) are ignored: a
    finger joint does not move the site, which is a property of the arm chain."""
    lead = q.shape[:-1]
    qq = np.ascontiguousarray(q[..., :NUM_ARM_JOINTS]).reshape(-1, NUM_ARM_JOINTS)
    n = qq.shape[0]
    T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    for i in range(NUM_ARM_JOINTS):
        c, s = np.cos(qq[:, i]), np.sin(qq[:, i])
        Rz = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s
        Rz[:, 1, 0] = s; Rz[:, 1, 1] = c
        T = T @ _FIXED[i] @ Rz
    return (T @ _GRASP_SITE)[:, :3].reshape(*lead, 3)


class FkSteeredAgent(AsyncDiffusionAgent):
    """AsyncDiffusionAgent + K-candidate FK-objective selection.

    Args (beyond AsyncDiffusionAgent's):
        steer_mode:        off | dryrun | on. Default "dryrun" — steering is
                           opt-in, and the arm cannot be affected until asked.
        n_candidates:      K per steered inference. K=16 is the real-time ceiling
                           on one RTX 5090 at a 500 ms chunk budget (~25 ms per
                           candidate once the shared image prefix amortises).
        n_candidates_first: K for the first ``first_chunks`` inferences of a
                           command. The arm is stationary at episode start so
                           latency is free there, and that is exactly when the
                           target choice is made (P0_FKRAY: K=64-early beats
                           uniform K). None -> same as n_candidates.
        first_chunks:      how many inferences count as "early".
        right_base_offset: right arm base relative to left, metres. MUST match
                           the session's URDF extrinsic or the right arm's score
                           is computed 0.61 m from where the anchor really is.
        arm_gate_band:     anchors further than this from the bimanual midline
                           score ONLY the near arm. Kills the transit-credit
                           hole where the far arm's unrelated sweep crosses the
                           sight line. Within the band, min over both arms.
        anchor_obs_key:    obs key carrying {"point": [x, y, z]} in the LEFT-ARM
                           BASE frame. Wire it through the node's state_topics.
        release_at_grasp:  retire the anchor once the commanded gripper closes
                           within release_radius_m of it.
        noise_action_dim:  pi0 denoises in a PADDED action space (32); the client
                           only sees 14 after the output transform unpads. Noise
                           must be model-space or the server raises on shapes.
    """

    def __init__(
        self,
        *args: Any,
        steer_mode: str = "dryrun",
        n_candidates: int = 16,
        n_candidates_first: Optional[int] = 64,
        first_chunks: int = 3,
        right_base_offset: tuple = (0.0, -0.61, 0.0),
        arm_gate_band: float = 0.10,
        anchor_obs_key: str = "anchor",
        release_at_grasp: bool = True,
        release_radius_m: float = 0.12,
        grasp_closed_below: float = 0.35,
        noise_action_dim: int = 32,
        steer_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # YAML 1.1 parses bare `on`/`off` as booleans, so a config that says
        # `steer_mode: on` arrives here as True. Accept that rather than raising
        # on something that reads perfectly correct in the YAML.
        if isinstance(steer_mode, bool):
            steer_mode = "on" if steer_mode else "off"
        steer_mode = str(steer_mode).lower()
        if steer_mode not in ("off", "dryrun", "on"):
            raise ValueError(f"steer_mode must be off|dryrun|on, got {steer_mode!r}")
        self.steer_mode = steer_mode
        self.n_candidates = int(n_candidates)
        self.n_candidates_first = int(n_candidates_first or n_candidates)
        self.first_chunks = int(first_chunks)
        self.right_base_offset = np.asarray(right_base_offset, dtype=np.float64)
        self.arm_gate_band = float(arm_gate_band)
        self.anchor_obs_key = anchor_obs_key
        self.release_at_grasp = bool(release_at_grasp)
        self.release_radius_m = float(release_radius_m)
        self.grasp_closed_below = float(grasp_closed_below)
        self.noise_action_dim = int(noise_action_dim)

        self._anchor_lock = threading.Lock()
        self._anchor: np.ndarray | None = None
        self._released = False
        self._steer_calls = 0
        self._seam_checked = False
        self._rng = np.random.default_rng(steer_seed)
        self.last_steer_info: Dict[str, Any] = {}

        # Second wrapper. super().__init__ already replaced .infer with a timing
        # instrument; wrapping that keeps the timing intact and steers underneath
        # every inference path at once.
        self._infer_upstream = self._websocket_client_policy.infer
        self._websocket_client_policy.infer = self._infer_steered
        if self._broker is not None:
            # sync mode holds its own reference, captured before we swapped.
            self._broker._policy = self._websocket_client_policy  # type: ignore[attr-defined]
        logger.info(
            "[FkSteeredAgent] steer_mode=%s K=%d (first %d chunks K=%d) "
            "release_at_grasp=%s anchor_key=%r",
            self.steer_mode, self.n_candidates, self.first_chunks,
            self.n_candidates_first, self.release_at_grasp, self.anchor_obs_key,
        )

    # ── command interface ────────────────────────────────────────────────
    def set_anchor(self, point) -> None:
        """Command a 3-D anchor in the LEFT-ARM BASE frame. None clears it."""
        with self._anchor_lock:
            self._anchor = None if point is None else np.asarray(point, dtype=np.float64).reshape(3)
            self._released = False
            self._steer_calls = 0
        logger.info("[FkSteeredAgent] anchor set: %s", None if point is None else np.round(point, 3))

    def _current_anchor(self) -> np.ndarray | None:
        with self._anchor_lock:
            if self._released:
                return None
            return None if self._anchor is None else self._anchor.copy()

    def __call__(self, obs: Dict[str, Any]) -> np.ndarray | None:
        # Pick the anchor off the raw obs before obs_to_model_input drops
        # everything that is not state or an image.
        cmd = obs.get(self.anchor_obs_key)
        if isinstance(cmd, dict):
            pt = cmd.get("point", cmd.get("anchor"))
            if pt is not None:
                pt = np.asarray(pt, dtype=np.float64).reshape(-1)
                if pt.size == 3:
                    with self._anchor_lock:
                        if self._anchor is None or not np.allclose(self._anchor, pt, atol=1e-6):
                            self._anchor = pt
                            self._released = False
                            self._steer_calls = 0
                            logger.info("[FkSteeredAgent] anchor <- %s", np.round(pt, 3))
        return super().__call__(obs)

    # ── scoring ──────────────────────────────────────────────────────────
    def _hand_paths(self, chunks: np.ndarray) -> Dict[str, np.ndarray]:
        """(K, H, 14) chunks -> per-arm (K, H, 3) hand paths in the LEFT base frame."""
        out = {}
        for arm, sl in _ARM_SLICE.items():
            p = fk_grasp_site(chunks[:, :, sl])
            if arm == "right":
                p = p + self.right_base_offset
            out[arm] = p
        return out

    def _score(self, chunks: np.ndarray, anchor: np.ndarray) -> tuple:
        """Closest approach of each candidate's hand path to the anchor.

        Returns (scores (K,), arms_used). Closest-approach over the chunk rather
        than mean: the mean punishes candidates that reach late, and P0_FKRAY
        measured arrive-and-stay scoring as a null against it.
        """
        paths = self._hand_paths(chunks)
        d = {arm: np.linalg.norm(p - anchor, axis=-1) for arm, p in paths.items()}  # (K,H)
        # Arm gate on the bimanual midline: an anchor clearly on one arm's side
        # scores that arm only.
        midline_y = 0.5 * self.right_base_offset[1]
        dy = anchor[1] - midline_y
        if dy > self.arm_gate_band:
            return d["left"].min(axis=1), ("left",)
        if dy < -self.arm_gate_band:
            return d["right"].min(axis=1), ("right",)
        return np.minimum(d["left"].min(axis=1), d["right"].min(axis=1)), ("left", "right")

    def _maybe_release(self, chunk: np.ndarray, anchor: np.ndarray, arms) -> bool:
        """Retire the anchor once the commanded gripper closes near it."""
        if not self.release_at_grasp:
            return False
        paths = self._hand_paths(chunk[None])
        for arm in arms:
            near = np.linalg.norm(paths[arm][0] - anchor, axis=-1) < self.release_radius_m
            closed = chunk[:, _GRIP_IDX[arm]] < self.grasp_closed_below
            if bool(np.any(near & closed)):
                with self._anchor_lock:
                    self._released = True
                logger.info("[FkSteeredAgent] anchor RELEASED (%s gripper closed within "
                            "%.0f cm) — reverting to unsteered inference",
                            arm, 100 * self.release_radius_m)
                return True
        return False

    # ── the steered inference ────────────────────────────────────────────
    def _infer_steered(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        anchor = self._current_anchor()
        if self.steer_mode == "off" or anchor is None:
            return self._infer_upstream(obs)

        horizon = getattr(self, "_server_horizon", None)
        if horizon is None:
            resp = self._infer_upstream(obs)
            self._server_horizon = int(np.asarray(resp["actions"]).shape[0])
            return resp

        k = self.n_candidates_first if self._steer_calls < self.first_chunks else self.n_candidates
        noise = self._rng.standard_normal((k, horizon, self.noise_action_dim)).astype(np.float32)

        t0 = time.perf_counter()
        resp = self._infer_upstream({**obs, "noise": noise})
        acts = np.asarray(resp["actions"], dtype=np.float64)
        dt = time.perf_counter() - t0

        if acts.ndim != 3 or acts.shape[0] != k:
            # Stock openpi ignores the noise key and returns a single chunk. Fail
            # loudly: silently executing an unsteered chunk while logging
            # "steering" is exactly the class of bug that voided earlier results.
            raise RuntimeError(
                f"server returned {acts.shape} for K={k}; the openpi K-batching seam "
                "is missing. Steering cannot work — see policies/policy.py."
            )
        self._assert_seam(acts)

        scores, arms = self._score(acts, anchor)
        best = int(np.argmin(scores))
        chosen = best if self.steer_mode == "on" else 0
        self._steer_calls += 1

        self.last_steer_info = {
            "k": k, "best": best, "chosen": chosen, "arms": arms,
            "score_best": float(scores[best]), "score_exec": float(scores[chosen]),
            "score_median": float(np.median(scores)), "score_worst": float(scores.max()),
            "margin": float(scores[0] - scores[best]), "infer_ms": 1e3 * dt,
            "anchor": anchor.tolist(), "mode": self.steer_mode,
        }
        logger.info(
            "[steer] K=%d %s best=%d d=%.3fm exec=%d d=%.3fm (median %.3f worst %.3f) "
            "margin_vs_cand0=%+.3fm %.0fms%s",
            k, "/".join(arms), best, scores[best], chosen, scores[chosen],
            np.median(scores), scores.max(), scores[0] - scores[best], 1e3 * dt,
            "" if self.steer_mode == "on" else "  [DRYRUN: executing candidate 0]",
        )

        out = dict(resp)
        out["actions"] = acts[chosen]
        self._maybe_release(acts[chosen], anchor, arms)
        return out

    def _assert_seam(self, acts: np.ndarray) -> None:
        """Once per process: prove the candidates actually differ.

        If the server ignores injected noise the K chunks come back identical and
        the argmin is over copies of one behaviour — steering that silently does
        nothing. Cheap to check, and the exact failure the anchor ledger records
        as having voided a whole batch of results.
        """
        if self._seam_checked:
            return
        self._seam_checked = True
        spread = float(np.abs(acts[0] - acts[1:]).max()) if acts.shape[0] > 1 else 0.0
        if spread < 1e-6:
            raise RuntimeError(
                "K candidates are identical (max|delta| < 1e-6): the server is ignoring "
                "the injected noise, so selection is a no-op. Fix the noise seam in "
                "openpi policies/policy.py before trusting any steered result."
            )
        logger.info("[FkSteeredAgent] noise seam OK — candidate spread %.3f", spread)
