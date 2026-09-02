"""Headline numbers for the VFD report, as JSON on stdout."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot_uq import _spearman, load_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uq-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()

    eps = load_all(args.uq_dir)
    vfd = np.concatenate([e["vfd"] for e in eps])
    per_step = np.concatenate([e["vfd_per_step"] for e in eps])
    per_dim = np.concatenate([e["vfd_per_dim"] for e in eps])
    names = ["L1", "L2", "L3", "L4", "L5", "L6", "Lgrip",
             "R1", "R2", "R3", "R4", "R5", "R6", "Rgrip"]
    share = per_dim.mean(axis=0)
    share = share / share.sum() * 100
    ep_mean = np.array([e["vfd"].mean() for e in eps])

    # Fraction of VFD accumulated in the second half of the ODE (s >= 0.5).
    ns = per_step.shape[1]
    late = per_step[:, ns // 2:].sum(axis=1) / np.maximum(per_step.sum(axis=1), 1e-9)

    out = {
        "episodes": len(eps),
        "frames": int(len(vfd)),
        "members": eps[0]["members"],
        "num_steps": eps[0]["num_steps"],
        "vfd": {
            "median": float(np.median(vfd)),
            "p90": float(np.percentile(vfd, 90)),
            "p99": float(np.percentile(vfd, 99)),
            "max": float(vfd.max()),
            "min": float(vfd.min()),
            "ratio_p99_median": float(np.percentile(vfd, 99) / np.median(vfd)),
        },
        "episode_mean": {
            "min": float(ep_mean.min()), "max": float(ep_mean.max()),
            "spread_ratio": float(ep_mean.max() / ep_mean.min()),
            "most_uncertain": eps[int(np.argmax(ep_mean))]["name"],
            "least_uncertain": eps[int(np.argmin(ep_mean))]["name"],
        },
        "late_ode_share_mean": float(late.mean()),
        "top_joints": [
            {"joint": names[i], "share_pct": float(share[i])}
            for i in np.argsort(share)[::-1][:4]
        ],
        "spearman_vs_vfd": {
            "self_dispersion": _spearman(np.concatenate([e["self_dispersion"] for e in eps]), vfd),
            "action_l2": _spearman(np.concatenate([e["action_l2"] for e in eps]), vfd),
            "stac": _spearman(np.concatenate([e["stac"] for e in eps]), vfd),
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
