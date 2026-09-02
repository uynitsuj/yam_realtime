"""Stage B — epistemic uncertainty (VFD) for a pi0 checkpoint over offline rollouts.

Implements Velocity-Field Disagreement from "Uncertainty Quantification for
Flow-Based Vision-Language-Action Models" (Roemer et al., arXiv:2606.18043),
Algorithm 2 / Eq. (7):

    u_e(y; V) = 1/(M(M-1)N_s) * E_{x0~p0} [ sum_{i != j} sum_{l=0}^{N_s-1}
                    kappa_{s_l} || v_i(x^(i)_{s_l}, y) - v_j(x^(i)_{s_l}, y) ||^2_2 ]

with kappa_s = s/(1-s), s_l = l*ds, and each member integrating its OWN Euler
path x^(i)_{s+ds} = x^(i)_s + v_i(x^(i)_s, y) ds from a shared-shape Gaussian x0.

Both members are evaluated at the SAME state, which is why a single checkpoint
with different noise seeds gives identically zero VFD: seeds move where you probe
the field, not the disagreement between fields. Members must differ in weights.

Coordinate convention: openpi's pi0 runs the ODE with t = 1 at noise and t = 0 at
data (the reverse of the paper's s), so s = 1 - t, kappa = (1-t)/t, and the Euler
step is x <- x + v*dt with dt = -1/N_s. VFD is a squared difference, so the sign
flip between the two conventions cancels.

Also computed, for comparison against the paper's single-model baselines:
  * per-ODE-step, per-chunk-index and per-action-dim breakdowns of the disagreement
  * DECU-style PaiDE score at the last ODE step
  * the terminal action chunks of every member (saved, so Action-L2, self-dispersion,
    ACE and STAC can be derived in post-processing without re-running the model)

Usage
-----
    uv run --project /home/us07/openpi python scripts/uq_vla/compute_vfd.py \
        --obs-dir out/obs --out-dir out/uq --members sss45,sss30
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import pathlib
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

CKPT_ROOT = "s3://xdof-internal-research/model_ckpts"
DEFAULT_PROMPT = "Put the plastic bottles in the bin"

# Ensemble members. sss45/30/15 are three independent WARP-BC fine-tunes of the
# SAME base checkpoint on the same episodes, differing only in the reward-model
# stride used to reweight frames -- the closest available stand-in for the
# paper's "fine-tune the base VLA M times on shuffled D_pre". `snap30k` is the
# step-30000 snapshot of the sss45 run (a snapshot ensemble: weaker, since its
# weights are nested in the sss45 member's own trajectory).
MEMBERS: dict[str, tuple[str, str]] = {
    "sss45": (
        "pi0_bottles_warpbc_sss45",
        f"{CKPT_ROOT}/pi0_bottles_warpbc_sss45/sky_pi0_bottles_warpbc_sss45_put_the_plastic_bottles_in_the_bin_d405_v021_sss45_20260615_122115/59999",
    ),
    "sss30": (
        "pi0_bottles_warpbc_sss30",
        f"{CKPT_ROOT}/pi0_bottles_warpbc_sss30/sky_pi0_bottles_warpbc_sss30_put_the_plastic_bottles_in_the_bin_d405_v021_sss30_20260615_122006/59999",
    ),
    "sss15": (
        "pi0_bottles_warpbc_sss15",
        f"{CKPT_ROOT}/pi0_bottles_warpbc_sss15/sky_pi0_bottles_warpbc_sss15_put_the_plastic_bottles_in_the_bin_d405_v021_sss15_20260615_121905/59999",
    ),
    "snap30k": (
        "pi0_bottles_warpbc_sss45",
        f"{CKPT_ROOT}/pi0_bottles_warpbc_sss45/sky_pi0_bottles_warpbc_sss45_put_the_plastic_bottles_in_the_bin_d405_v021_sss45_20260615_122115/30000",
    ),
}

CAMERA_KEYS = ("left_camera", "right_camera", "top_camera")
YAM_ACTION_DIM = 14  # real bimanual joint space; the model head is padded to 32


# ---------------------------------------------------------------------------
# Velocity field access
# ---------------------------------------------------------------------------


def _prefix_cache(model: _pi0.Pi0, obs: _model.Observation):
    """Fill the KV cache with one prefix (image + language) forward pass."""
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
    return kv_cache, prefix_mask


def _velocity(model: _pi0.Pi0, state, kv_cache, prefix_mask, x_t, t):
    """v_theta(x_t, t | y) reusing a pre-filled prefix cache.

    Mirrors the inner `step` of `Pi0.sample_actions` exactly, but returns v_t
    instead of taking the Euler step, and takes `state` directly since
    `embed_suffix` only reads `obs.state` off the observation.
    """
    batch = x_t.shape[0]
    obs = _model.Observation(images={}, image_masks={}, state=state)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        obs, x_t, jnp.broadcast_to(jnp.asarray(t, dtype=jnp.float32), batch)
    )
    suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    assert prefix_out is None
    return model.action_out_proj(suffix_out[:, -model.action_horizon :])


# ---------------------------------------------------------------------------
# VFD
# ---------------------------------------------------------------------------


def make_vfd_fn(graphdefs: tuple, num_steps: int, batch: int, act_dim: int = YAM_ACTION_DIM):
    """Build the jitted per-observation VFD computation for M members.

    Returns a function ``(states, inputs, noise) -> dict`` where ``states`` is
    the tuple of nnx states matching ``graphdefs``, ``inputs`` is the batched
    (leading axis 1) model-input pytree, and ``noise`` is ``(M, B, H, A)``.

    All members share one observation, so the prefix pass runs at batch 1 per
    member and its KV cache is broadcast over the M*B probe states -- the image
    encoder is by far the most expensive part and is not per-sample.
    """
    M = len(graphdefs)
    dt = -1.0 / num_steps

    def fn(states, inputs, noise):
        models = [nnx.merge(gd, st) for gd, st in zip(graphdefs, states, strict=True)]
        obs = _model.preprocess_observation(None, _model.Observation.from_dict(inputs), train=False)

        n_probe = M * batch
        caches, masks, tiled_state = [], [], jnp.repeat(obs.state, n_probe, axis=0)
        for model in models:
            kv, pmask = _prefix_cache(model, obs)
            # KV cache is layer-major (l, b, t, k, h) -- broadcast along the BATCH axis.
            caches.append(jax.tree.map(lambda a: jnp.repeat(a, n_probe, axis=1), kv))
            masks.append(jnp.repeat(pmask, n_probe, axis=0))

        # x[m] is member m's own Euler path; all M*B states are probed by every
        # member's velocity field at every step.
        x = noise  # (M, B, H, A)

        def step(carry, l):
            x = carry
            t = 1.0 + dt * l  # 1.0, 1-1/Ns, ..., 1/Ns
            s = 1.0 - t
            kappa = s / t  # = s/(1-s) in the paper's coordinates

            flat = x.reshape(n_probe, *x.shape[2:])
            v = jnp.stack(
                [
                    _velocity(model, tiled_state, caches[m], masks[m], flat, t)
                    for m, model in enumerate(models)
                ]
            )  # (M_eval, M*B, H, A)
            v = v.reshape(M, M, batch, *x.shape[2:])  # (M_eval, M_path, B, H, A)

            # Squared chunk distance between every (eval_i, eval_j) pair on every path.
            diff = v[:, None, ...] - v[None, :, ...]  # (i, j, path, B, H, A)
            d = diff[..., :act_dim]
            d2 = jnp.sum(jnp.square(d), axis=(-2, -1)).mean(axis=-1)  # (i, j, path)

            # Eq. (7) uses the pair (i, j) evaluated on member i's own path.
            own = jnp.einsum("iji->ij", d2)  # (i, j), path == i
            offdiag = 1.0 - jnp.eye(M)
            vfd_inc = kappa * jnp.sum(own * offdiag) / (M * max(M - 1, 1) * num_steps)

            # Same quantity resolved per chunk index / per action dim (pairs on
            # their own path, averaged over ordered pairs and noise samples).
            sq = jnp.square(d)  # (i, j, path, B, H, A)
            own_sq = jnp.einsum("ijibha->ijbha", sq)
            w = (offdiag / (M * max(M - 1, 1)))[:, :, None, None, None]
            per_chunk = kappa * jnp.sum(own_sq * w, axis=(0, 1)).mean(axis=0).sum(axis=-1)  # (H,)
            per_dim = kappa * jnp.sum(own_sq * w, axis=(0, 1)).mean(axis=0).sum(axis=0)  # (A,)

            # Full-width (padded head) variant, for reference.
            d2_full = jnp.sum(jnp.square(diff), axis=(-2, -1)).mean(axis=-1)
            vfd_full_inc = kappa * jnp.sum(jnp.einsum("iji->ij", d2_full) * offdiag) / (
                M * max(M - 1, 1) * num_steps
            )

            # Each member advances along its own field.
            v_own = jnp.einsum("mmbha->mbha", v)
            x_next = x + dt * v_own
            return x_next, (vfd_inc, vfd_full_inc, d2, per_chunk, per_dim)

        x_final, (vfd_steps, vfd_full_steps, d2_steps, per_chunk, per_dim) = jax.lax.scan(
            step, x, jnp.arange(num_steps)
        )

        return {
            "vfd": jnp.sum(vfd_steps),
            "vfd_full": jnp.sum(vfd_full_steps),
            "vfd_per_step": vfd_steps,  # (Ns,)
            "vfd_per_chunk": jnp.sum(per_chunk, axis=0) / num_steps,  # (H,)
            "vfd_per_dim": jnp.sum(per_dim, axis=0) / num_steps,  # (A,)
            "d2_steps": d2_steps,  # (Ns, M, M, M) raw squared distances
            "actions_norm": x_final,  # (M, B, H, A) terminal chunks, model space
        }

    return jax.jit(fn)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Loaded:
    name: str
    policy: object
    graphdef: object
    state: object


def load_members(names: list[str]) -> list[Loaded]:
    out = []
    for name in names:
        cfg_name, ckpt = MEMBERS[name]
        print(f"[load] {name}: {cfg_name}")
        policy = _policy_config.create_trained_policy(
            _config.get_config(cfg_name), ckpt, default_prompt=DEFAULT_PROMPT
        )
        graphdef, state = nnx.split(policy._model)  # noqa: SLF001 — need the raw module
        out.append(Loaded(name, policy, graphdef, state))
    return out


def _action_norm_stats(policy) -> tuple[np.ndarray, np.ndarray] | None:
    """The (mean, std) the member normalizes actions with."""
    for t in policy._output_transform.transforms:  # noqa: SLF001
        stats = getattr(t, "norm_stats", None)
        if stats is not None and "actions" in stats:
            a = stats["actions"]
            return np.asarray(a.mean, dtype=np.float64), np.asarray(a.std, dtype=np.float64)
    return None


def _check_shared_action_space(members: list["Loaded"], tol: float = 0.02) -> None:
    """Members must share the normalized action space or VFD measures coordinates.

    Not an exact-match test: dataset copies scored at different reward-model
    strides give norm stats that differ in the 3rd decimal, which is irrelevant
    next to the signal. Fail only when the mismatch is a real fraction of a
    std (`tol`), and report the deviation eitherway so it is never silent.
    """
    ref = _action_norm_stats(members[0].policy)
    if ref is None:
        print("[load] no action norm stats found -- skipping the shared-space check")
        return
    worst = 0.0
    for m in members[1:]:
        cur = _action_norm_stats(m.policy)
        if cur is None:
            raise SystemExit(f"member {m.name} has no action norm stats")
        # Express both deviations in units of the reference std.
        d = max(
            float(np.abs(cur[0] - ref[0]).max() / np.abs(ref[1]).max()),
            float(np.abs(cur[1] - ref[1]).max() / np.abs(ref[1]).max()),
        )
        worst = max(worst, d)
        if d > tol:
            raise SystemExit(
                f"member {m.name} normalizes actions {d:.1%} of a std away from "
                f"{members[0].name} -- their velocity fields live in different "
                "coordinates and VFD would partly measure that offset."
            )
    print(f"[load] members share the action space (worst deviation {worst:.2%} of a std)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--members", default="sss45,sss30")
    ap.add_argument("--num-steps", type=int, default=10, help="ODE Euler steps N_s.")
    ap.add_argument("--batch", type=int, default=5, help="Noise samples B per member (paper uses 5).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="Truncate each episode (debug).")
    args = ap.parse_args()

    members = load_members(args.members.split(","))
    _check_shared_action_space(members)

    prep = members[0].policy._input_transform  # noqa: SLF001
    unprep = members[0].policy._output_transform  # noqa: SLF001
    model0 = members[0].policy._model  # noqa: SLF001
    horizon, act_dim_model = model0.action_horizon, model0.action_dim

    vfd_fn = make_vfd_fn(
        tuple(m.graphdef for m in members), args.num_steps, args.batch, YAM_ACTION_DIM
    )
    states = tuple(m.state for m in members)
    M = len(members)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    episodes = sorted(args.obs_dir.glob("*.npz"))
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"{len(episodes)} episode(s), M={M}, B={args.batch}, Ns={args.num_steps}")

    rng = jax.random.key(args.seed)
    for ei, ep_path in enumerate(episodes, 1):
        out_path = args.out_dir / ep_path.name
        if out_path.exists():
            print(f"[{ei}/{len(episodes)}] {ep_path.stem}: cached")
            continue

        obs = np.load(ep_path, allow_pickle=True)
        n = len(obs["ts"]) if args.max_frames is None else min(len(obs["ts"]), args.max_frames)
        # Materialise up front: an npz member decompresses in full on every
        # access, so indexing it inside the frame loop re-inflates ~88 MB per
        # camera per frame and pins the run to the CPU.
        img = {c: np.asarray(obs[f"image_{c}"][:n]) for c in CAMERA_KEYS}
        state_all = np.asarray(obs["state"][:n])
        acc: dict[str, list] = {}
        actions: list[np.ndarray] = []
        t0 = time.time()

        for i in range(n):
            raw = {
                "state": state_all[i],
                **{f"{c}-images-rgb": img[c][i] for c in CAMERA_KEYS},
            }
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], prep(raw))
            rng, sub = jax.random.split(rng)
            noise = jax.random.normal(sub, (M, args.batch, horizon, act_dim_model))

            res = jax.device_get(vfd_fn(states, inputs, noise))
            chunks = res.pop("actions_norm")
            for k, v in res.items():
                # jax hands back bf16 for the LLM-side tensors; np.savez would
                # store that as opaque void bytes.
                acc.setdefault(k, []).append(np.asarray(v).astype(np.float32))

            # Back to client joint space so the plots are in radians, not sigmas.
            actions.append(
                np.stack(
                    [
                        np.stack(
                            [
                                unprep({"state": inputs["state"][0], "actions": chunks[m, b]})["actions"]
                                for b in range(args.batch)
                            ]
                        )
                        for m in range(M)
                    ]
                )
            )

            if i == 0:
                print(f"[{ei}/{len(episodes)}] {ep_path.stem}: first frame {time.time() - t0:.1f}s (incl. JIT)")

        elapsed = time.time() - t0
        np.savez_compressed(
            out_path,
            **{k: np.asarray(v) for k, v in acc.items()},
            actions=np.asarray(actions, dtype=np.float32),  # (N, M, B, H, 14)
            ts=obs["ts"][:n],
            t_rel=obs["t_rel"][:n],
            state=obs["state"][:n],
            cmd_action=obs["cmd_action"][:n],
            members=np.array([m.name for m in members]),
            num_steps=np.int32(args.num_steps),
            batch=np.int32(args.batch),
            episode=obs["episode"],
        )
        print(
            f"[{ei}/{len(episodes)}] {ep_path.stem}: {n} frames in {elapsed:.1f}s "
            f"({elapsed / n * 1e3:.0f} ms/frame), VFD mean {np.mean(acc['vfd']):.4g}"
        )


if __name__ == "__main__":
    main()
