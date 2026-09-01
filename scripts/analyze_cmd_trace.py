#!/usr/bin/env python3
"""Diagnose stepped/jerky arm motion from a RobotNode ``debug_trace`` CSV.

Usage:
    uv run python scripts/analyze_cmd_trace.py debug_traces/yam_left_*.csv

Answers, in order:
  1. How fast is the node actually polling?           (poll rate)
  2. How fast do *fresh* commands arrive?             (command rate)
     A command rate well below the poll rate means the arm is being held at
     each setpoint for several ticks — a staircase, which reads as "steppy".
  3. How big is each step when a command does land?   (mrad per fresh command)
     Big, infrequent steps = visible stepping. Small, frequent = smooth.
  4. How well does the arm follow?                    (tracking error + lag)
     Large error with a smooth command stream points at the follower (gains,
     gravity comp, friction) rather than at the command pipeline.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    cols: dict[str, np.ndarray] = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        cols[key] = np.array([np.nan if v == "" else float(v) for v in vals], dtype=np.float64)
    return cols


def _stack(cols: dict[str, np.ndarray], prefix: str, n: int) -> np.ndarray:
    return np.column_stack([cols[f"{prefix}{i}"] for i in range(n) if f"{prefix}{i}" in cols])


def _lag_samples(cmd: np.ndarray, meas: np.ndarray, max_lag: int) -> int:
    """Cross-correlation lag (in samples) that best aligns measured to command."""
    c = cmd - cmd.mean()
    m = meas - meas.mean()
    if np.allclose(c, 0) or np.allclose(m, 0):
        return 0
    best, best_lag = -np.inf, 0
    for lag in range(max_lag + 1):
        a = c[: len(c) - lag]
        b = m[lag:]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            continue
        score = float(a @ b / denom)
        if score > best:
            best, best_lag = score, lag
    return best_lag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", type=Path, help="CSV written by RobotNode debug_trace")
    ap.add_argument("--n-arm", type=int, default=6, help="arm DOF, excluding gripper")
    args = ap.parse_args()

    cols = _load(args.trace)
    t = cols["t"]
    dur = t[-1] - t[0]
    if dur <= 0:
        raise SystemExit("trace covers no time")

    applied = _stack(cols, "applied_", 7)
    meas = _stack(cols, "meas_", args.n_arm)
    fresh = cols["cmd_fresh"] > 0.5
    age = cols["cmd_age_ms"]

    # Ticks before the first command carry no target/applied — drop them from
    # the tracking stats (they would poison every percentile with NaN).
    have_cmd = ~np.isnan(applied[:, 0])
    if not have_cmd.any():
        raise SystemExit("no commanded samples in trace")
    if (~have_cmd).any():
        print(f"(skipping {int((~have_cmd).sum())} tick(s) before the first command)")

    print(f"\n=== {args.trace} ===")
    print(f"samples {len(t)} over {dur:.1f}s")

    # 1. Poll rate.
    dt = np.diff(t)
    print(
        f"\npoll        {len(t) / dur:7.1f} Hz   "
        f"(period p50 {np.median(dt) * 1e3:.2f} ms, p95 {np.percentile(dt, 95) * 1e3:.2f} ms, "
        f"max {dt.max() * 1e3:.2f} ms)"
    )

    # 2. Fresh-command rate — the number that matters for stepping.
    n_fresh = int(fresh.sum())
    cmd_hz = n_fresh / dur
    print(f"commands    {cmd_hz:7.1f} Hz   ({n_fresh} fresh of {len(t)} ticks)")
    if n_fresh > 1:
        gaps = np.diff(t[fresh])
        print(
            f"  inter-arrival  p50 {np.median(gaps) * 1e3:6.2f} ms   "
            f"p95 {np.percentile(gaps, 95) * 1e3:6.2f} ms   max {gaps.max() * 1e3:6.2f} ms"
        )
        holds = np.diff(np.flatnonzero(fresh))
        print(f"  ticks per command (hold length)  p50 {np.median(holds):.1f}  max {holds.max()}")
    valid_age = age[~np.isnan(age)]
    if valid_age.size:
        print(
            f"  command age at use  p50 {np.median(valid_age):6.2f} ms   "
            f"p95 {np.percentile(valid_age, 95):6.2f} ms   max {valid_age.max():6.2f} ms"
        )

    # 3. Step size per fresh command. What matters is how often the command
    #    VALUE changes — repeated identical messages are still a held setpoint.
    if n_fresh > 1:
        fa = applied[fresh & have_cmd]
        ft = t[fresh & have_cmd]
        steps = np.abs(np.diff(fa, axis=0)).max(axis=1) * 1e3
        changed = steps > 1e-6
        n_changed = int(changed.sum())
        span = ft[-1] - ft[0] if len(ft) > 1 else dur
        print(
            f"  distinct setpoints  {n_changed} ({n_changed / span:.1f} changes/s); "
            f"{int((~changed).sum())} repeats of the previous value"
        )
        if n_changed > 1:
            cg = np.diff(ft[1:][changed])
            print(
                f"  time between changes  p50 {np.median(cg) * 1e3:6.1f} ms   "
                f"p90 {np.percentile(cg, 90) * 1e3:6.1f} ms   max {cg.max() * 1e3:6.0f} ms"
            )
        moving = steps[changed]
        print("\nper-command joint step (max over joints, mrad)")
        if moving.size:
            print(
                f"  while moving:  p50 {np.median(moving):6.2f}   p95 {np.percentile(moving, 95):6.2f}   "
                f"max {moving.max():6.2f}   (n={moving.size})"
            )
            implied = np.median(moving) * cmd_hz / 1e3
            print(f"  implied joint speed at p50 step × command rate: {implied:.3f} rad/s")
        else:
            print("  (arm was stationary for the whole trace)")

    # 4. Tracking.
    if meas.size:
        n = min(meas.shape[1], applied.shape[1])
        err = np.abs(applied[have_cmd, :n] - meas[have_cmd, :n])
        print("\ntracking error |applied - measured| (mrad)")
        print(
            f"  overall  p50 {np.median(err) * 1e3:6.2f}   p95 {np.percentile(err, 95) * 1e3:6.2f}   "
            f"max {err.max() * 1e3:6.2f}"
        )
        worst = int(np.argmax(err.mean(axis=0)))
        print(f"  worst joint: j{worst} (mean {err[:, worst].mean() * 1e3:.2f} mrad)")
        max_lag = max(1, int(0.2 * len(t) / dur))  # up to 200 ms
        lag = _lag_samples(applied[have_cmd, worst], meas[have_cmd, worst], max_lag)
        print(f"  best-fit lag on j{worst}: {lag} samples ≈ {lag * dur / len(t) * 1e3:.1f} ms")

    # Verdict.
    poll_hz = len(t) / dur
    print("\ninterpretation")
    if cmd_hz < 0.6 * poll_hz:
        print(
            f"  ! command stream ({cmd_hz:.0f} Hz) is well below the poll rate ({poll_hz:.0f} Hz):"
            f" each setpoint is held ~{poll_hz / max(cmd_hz, 1e-9):.1f} ticks. This is the staircase."
        )
    else:
        print(f"  command stream keeps up with the poll rate ({cmd_hz:.0f} vs {poll_hz:.0f} Hz).")
    if n_fresh > 1 and moving.size and np.percentile(moving, 95) > 20:
        print(
            f"  ! large per-command jumps (p95 {np.percentile(moving, 95):.0f} mrad):"
            " smooth/interpolate the target, or drag the gizmo more slowly."
        )
    if meas.size and np.percentile(err, 95) * 1e3 > 50:
        print(
            f"  ! large tracking error (p95 {np.percentile(err, 95) * 1e3:.0f} mrad) —"
            " follower side: gains, gravity comp, or joint friction."
        )


if __name__ == "__main__":
    sys.exit(main())
