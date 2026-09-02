"""Stage C — figures for VFD uncertainty over offline pi0 rollouts.

Consumes the per-episode npz files written by ``compute_vfd.py`` and renders the
report figures plus an optional camera+trace overlay video.

Derived comparison metrics (computed here from the stored terminal action chunks,
so they cost nothing extra):

  self-dispersion  mean pairwise L2 between the B chunks of ONE member. Pure
                   sampling/multimodality spread -- the quantity a single
                   checkpoint with different seeds can measure.
  Action-L2        the paper's baseline: mean pairwise L2 between the sampler
                   member's chunks and each other member's chunks.
  STAC             mean pairwise L2 between the chunk distributions at
                   consecutive query times, over the part of the horizon they
                   both cover. A temporal-consistency signal.

ACE is deliberately absent: it is defined on end-effector position deltas, and
this policy emits absolute joint positions, so a faithful port needs forward
kinematics rather than a re-binning.

Usage
-----
    uv run python scripts/uq_vla/plot_uq.py --uq-dir out/uq_vla/uq --out-dir out/uq_vla/figs
    uv run python scripts/uq_vla/plot_uq.py ... --video-episode episode_163301_c0228c78 --obs-dir out/uq_vla/obs
"""

from __future__ import annotations

import argparse
import itertools
import pathlib

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Validated categorical slots + sequential blue ramp (dataviz reference palette).
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e6e5e1"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ)

JOINT_NAMES = [f"L{i}" for i in range(1, 7)] + ["Lgrip"] + [f"R{i}" for i in range(1, 7)] + ["Rgrip"]

mpl.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": INK3, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "grid.color": GRID, "grid.linewidth": 0.7,
    "font.size": 9, "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
})


# ---------------------------------------------------------------------------
# Loading + derived metrics
# ---------------------------------------------------------------------------


def _pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean pairwise chunk L2 between sample sets a (N,C,...) and b (N,C,...)."""
    n, ca = a.shape[0], a.shape[1]
    cb = b.shape[1]
    flat_a = a.reshape(n, ca, -1)
    flat_b = b.reshape(n, cb, -1)
    d = np.linalg.norm(flat_a[:, :, None, :] - flat_b[:, None, :, :], axis=-1)
    return d.reshape(n, -1).mean(axis=1)


def load_episode(path: pathlib.Path) -> dict:
    d = np.load(path, allow_pickle=True)
    a = d["actions"]  # (N, M, B, H, 14)
    n, M, B, H, _ = a.shape

    out = {k: np.asarray(d[k]) for k in
           ("vfd", "vfd_full", "vfd_per_step", "vfd_per_chunk", "vfd_per_dim",
            "t_rel", "state", "cmd_action")}
    out["name"] = str(d["episode"])
    out["members"] = [str(x) for x in d["members"]]
    out["actions"] = a
    out["num_steps"] = int(d["num_steps"])

    # Self-dispersion of the sampler member (member 0).
    m0 = a[:, 0]
    pairs = list(itertools.combinations(range(B), 2))
    out["self_dispersion"] = np.mean(
        [np.linalg.norm((m0[:, i] - m0[:, j]).reshape(n, -1), axis=-1) for i, j in pairs], axis=0
    )

    # Action-L2: sampler member vs each other member, averaged.
    out["action_l2"] = (
        np.mean([_pairwise_l2(m0, a[:, m]) for m in range(1, M)], axis=0)
        if M > 1 else np.zeros(n)
    )

    # STAC over the horizon two consecutive queries share. Queries are `stride`
    # policy ticks apart; with H=30 and stride=10 they overlap by 20 steps.
    dt = np.median(np.diff(out["t_rel"])) if n > 1 else 1.0
    lag = int(round(dt * 30.0))  # policy ticks between queries (30 Hz consumer)
    ov = max(H - lag, 1)
    stac = np.full(n, np.nan)
    if n > 1 and ov > 1:
        cur = m0[:-1, :, lag:lag + ov]   # what query k predicted for the shared window
        nxt = m0[1:, :, :ov]             # what query k+1 predicts for the same window
        stac[1:] = _pairwise_l2(cur, nxt)
    out["stac"] = stac
    return out


def load_all(uq_dir: pathlib.Path) -> list[dict]:
    eps = [load_episode(p) for p in sorted(uq_dir.glob("*.npz"))]
    if not eps:
        raise SystemExit(f"no npz files in {uq_dir}")
    return eps


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _finish(fig, path: pathlib.Path, note: str | None = None) -> None:
    if note:
        fig.text(0.005, 0.004, note, fontsize=7, color=INK3, ha="left", va="bottom")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_timelines(eps: list[dict], out: pathlib.Path, members: str) -> None:
    """Small multiples: VFD through each rollout, on a shared scale."""
    pooled = np.concatenate([e["vfd"] for e in eps])
    hi = np.percentile(pooled, 99.5)
    p90 = np.percentile(pooled, 90)
    ncol = 6
    nrow = int(np.ceil(len(eps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 1.5 * nrow),
                             sharex=False, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, e in zip(axes, eps):
        ax.axhline(p90, color=INK3, lw=0.7, ls=(0, (3, 3)), zorder=1)
        ax.fill_between(e["t_rel"], 0, np.clip(e["vfd"], 0, hi), color=C1, alpha=0.13, lw=0, zorder=2)
        ax.plot(e["t_rel"], np.clip(e["vfd"], 0, hi), color=C1, lw=1.1, zorder=3)
        ax.set_ylim(0, hi)
        ax.set_xlim(0, max(e["t_rel"][-1], 1))
        ax.set_title(e["name"].replace("episode_", ""), fontsize=7, color=INK2, pad=2)
        ax.grid(axis="y", alpha=0.6)
        ax.tick_params(labelsize=6.5, length=2)
    for ax in axes[len(eps):]:
        ax.set_visible(False)
    for ax in axes[len(eps) - ncol if len(eps) > ncol else 0:len(eps)]:
        ax.set_xlabel("time in rollout (s)", fontsize=7)
    for r in range(nrow):
        axes[r * ncol].set_ylabel("VFD", fontsize=7)

    # Reserve the header band first, then place both lines inside it -- letting
    # tight_layout run afterwards is what made the two lines collide.
    head = 0.42 / fig.get_figheight()  # ~0.42 inch of header, whatever the row count
    fig.tight_layout(rect=[0, 0, 1, 1 - head])
    fig.text(0.005, 1 - head * 0.30, "Epistemic uncertainty (VFD) through each rollout",
             fontsize=13, color=INK, ha="left", va="center")
    fig.text(0.005, 1 - head * 0.78,
             f"dashed line = 90th percentile of all frames pooled ({p90:.0f})   ·   "
             f"y clipped at the 99.5th percentile ({hi:.0f})   ·   ensemble: {members}",
             fontsize=8, color=INK2, ha="left", va="center")
    _finish(fig, out / "fig1_timelines.png")


def fig_ranking(eps: list[dict], out: pathlib.Path) -> None:
    """Which rollouts were the model most unsure about, start to finish."""
    order = np.argsort([e["vfd"].mean() for e in eps])
    names = [eps[i]["name"].replace("episode_", "") for i in order]
    mean = np.array([eps[i]["vfd"].mean() for i in order])
    peak = np.array([eps[i]["vfd"].max() for i in order])
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.2, 0.23 * len(order) + 1.4))
    ax.hlines(y, mean, peak, color=GRID, lw=2.4, zorder=1)
    ax.scatter(peak, y, s=26, color="#ffffff", zorder=2, linewidths=0)
    ax.scatter(peak, y, s=22, facecolor="none", edgecolor=C2, linewidths=1.4, zorder=3, label="peak")
    ax.scatter(mean, y, s=26, color=C1, zorder=4, linewidths=0, label="mean")
    ax.set_yticks(y, names, fontsize=6.5)
    ax.set_xlabel("VFD")
    ax.set_ylim(-1, len(order))
    ax.grid(axis="x", alpha=0.6)
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("Rollouts ranked by mean epistemic uncertainty", loc="left", pad=8)
    _finish(fig, out / "fig2_ranking.png",
            "Long bars = uncertainty spiked somewhere in an otherwise confident rollout.")


def fig_heatmap(eps: list[dict], out: pathlib.Path) -> None:
    """Every rollout as one row, time normalised, so shared structure shows up."""
    nb = 25
    grid = np.full((len(eps), nb), np.nan)
    order = np.argsort([e["vfd"].mean() for e in eps])
    for r, i in enumerate(order):
        e = eps[i]
        frac = e["t_rel"] / max(e["t_rel"][-1], 1e-9)
        idx = np.clip((frac * nb).astype(int), 0, nb - 1)
        for b in range(nb):
            sel = idx == b
            if sel.any():
                grid[r, b] = e["vfd"][sel].mean()
        # Short episodes leave empty bins; fill them from their neighbours so the
        # row reads as a continuous trace rather than a dotted one.
        row = grid[r]
        miss = np.isnan(row)
        if miss.any() and not miss.all():
            row[miss] = np.interp(np.flatnonzero(miss), np.flatnonzero(~miss), row[~miss])

    vmin, vmax = np.nanpercentile(grid, [5, 95])
    fig, (ax, cax) = plt.subplots(
        1, 2, figsize=(8.6, 0.19 * len(eps) + 1.7),
        gridspec_kw={"width_ratios": [40, 1], "wspace": 0.03})
    im = ax.imshow(grid, aspect="auto", cmap=SEQ_CMAP, vmin=vmin, vmax=vmax,
                   interpolation="nearest", extent=[0, 100, len(eps) - 0.5, -0.5])
    ax.set_yticks(range(len(order)),
                  [eps[i]["name"].replace("episode_", "") for i in order], fontsize=6)
    ax.set_xlabel("progress through rollout (%)")
    ax.set_title("VFD across all rollouts, time-normalised", loc="left", pad=8)
    ax.grid(False)
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("VFD", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _finish(fig, out / "fig3_heatmap.png",
            "Rows sorted by mean VFD (least to most uncertain). Darker = higher epistemic uncertainty. "
            "Colour spans the 5th-95th percentile of bin means.")


def fig_flowtime(eps: list[dict], out: pathlib.Path) -> None:
    """Where along the generative ODE the disagreement actually lives."""
    per_step = np.concatenate([e["vfd_per_step"] for e in eps])  # (frames, Ns)
    ns = per_step.shape[1]
    s = np.arange(ns) / ns  # paper's s: 0 = noise, 1 = data
    kappa = np.where(s < 1, s / np.maximum(1 - s, 1e-9), np.nan)
    mean = per_step.mean(axis=0)
    lo, hi = np.percentile(per_step, [25, 75], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    ax = axes[0]
    ax.fill_between(s, lo, hi, color=C1, alpha=0.16, lw=0)
    ax.plot(s, mean, color=C1, marker="o", ms=4.5)
    ax.set_xlabel("flow-matching time $s$   (0 = noise, 1 = data)")
    ax.set_ylabel("contribution to VFD")
    ax.set_title("Disagreement concentrates near the data end", loc="left", pad=8)
    ax.grid(axis="y", alpha=0.6)

    ax = axes[1]
    ax.plot(s, kappa, color=C2, marker="o", ms=4.5)
    ax.set_xlabel("flow-matching time $s$")
    ax.set_ylabel(r"$\kappa_s = s/(1-s)$")
    ax.set_title("...partly by construction: the $\\kappa_s$ weight", loc="left", pad=8)
    ax.grid(axis="y", alpha=0.6)

    fig.tight_layout()
    _finish(fig, out / "fig4_flowtime.png",
            "Left: shaded band = interquartile range over all frames of all rollouts. "
            "The s=0 term is exactly zero because kappa_0 = 0.")


def fig_joints(eps: list[dict], out: pathlib.Path) -> None:
    """Which joints the members disagree about, and where in the chunk."""
    per_dim = np.concatenate([e["vfd_per_dim"] for e in eps])      # (frames, 14)
    per_chunk = np.concatenate([e["vfd_per_chunk"] for e in eps])  # (frames, H)

    share = per_dim.mean(axis=0)
    share = share / share.sum() * 100
    fig = plt.figure(figsize=(8.6, 3.2))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1], wspace=0.28)

    ax = fig.add_subplot(gs[0])
    colors = [C2 if "grip" in n else C1 for n in JOINT_NAMES]
    ax.bar(JOINT_NAMES, share, color=colors, width=0.68)
    for i, v in enumerate(share):
        if v > share.max() * 0.55:
            ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_ylabel("share of VFD (%)")
    ax.set_title("Which joints the ensemble disagrees about", loc="left", pad=8)
    ax.grid(axis="y", alpha=0.6)
    ax.tick_params(axis="x", labelsize=7.5)
    handles = [mpl.patches.Patch(color=C1, label="arm joint"),
               mpl.patches.Patch(color=C2, label="gripper")]
    ax.legend(handles=handles, loc="upper right", ncol=1)

    ax = fig.add_subplot(gs[1])
    m = per_chunk.mean(axis=0)
    q1, q3 = np.percentile(per_chunk, [25, 75], axis=0)
    k = np.arange(len(m))
    ax.fill_between(k, q1, q3, color=C1, alpha=0.16, lw=0)
    ax.plot(k, m, color=C1)
    ax.set_xlabel("index within the 30-step action chunk")
    ax.set_ylabel("contribution to VFD")
    ax.set_title("...and where in the chunk", loc="left", pad=8)
    ax.grid(axis="y", alpha=0.6)

    _finish(fig, out / "fig5_joints.png",
            "Pooled over every evaluated frame of every rollout.")


def fig_baselines(eps: list[dict], out: pathlib.Path) -> None:
    """Does epistemic VFD say anything the single-model signals don't?"""
    vfd = np.concatenate([e["vfd"] for e in eps])
    cand = [
        ("self-dispersion", np.concatenate([e["self_dispersion"] for e in eps]),
         "spread of one member's own samples\n(a single checkpoint can measure this)"),
        ("Action-L2", np.concatenate([e["action_l2"] for e in eps]),
         "sampler vs other members' chunks\n(the paper's ensemble baseline)"),
        ("STAC", np.concatenate([e["stac"] for e in eps]),
         "chunk distributions at consecutive\nqueries, over their shared horizon"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
    for ax, (name, x, sub) in zip(axes, cand):
        rho = _spearman(x, vfd)
        ok = np.isfinite(x) & np.isfinite(vfd)
        ax.scatter(x[ok], vfd[ok], s=5, color=C1, alpha=0.18, linewidths=0)
        ax.set_xlabel(f"{name}  (rad)")
        ax.set_title(f"{name}   ·   Spearman $\\rho$ = {rho:.2f}", loc="left", pad=26, fontsize=9.5)
        # Above the axes, not inside it -- the cloud fills the upper-left corner.
        ax.annotate(sub, xy=(0, 1.015), xycoords="axes fraction", fontsize=7,
                    color=INK3, va="bottom", ha="left")
        ax.grid(alpha=0.6)
    axes[0].set_ylabel("VFD (epistemic)")
    fig.suptitle("Epistemic VFD vs the sampling-based signals", x=0.005, ha="left",
                 fontsize=13, color=INK, y=1.02)
    fig.tight_layout()
    _finish(fig, out / "fig6_baselines.png",
            "One point per evaluated frame, all rollouts pooled. Weak correlation = VFD is not a "
            "restatement of sample spread.")


def fig_fan(eps: list[dict], out: pathlib.Path, joint: int = 1) -> None:
    """What a high-VFD moment actually looks like in action space."""
    all_vfd = np.concatenate([e["vfd"] for e in eps])
    owner = np.concatenate([[i] * len(e["vfd"]) for i, e in enumerate(eps)])
    local = np.concatenate([np.arange(len(e["vfd"])) for e in eps])
    lo_i, hi_i = int(np.argmin(all_vfd)), int(np.argmax(all_vfd))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3), sharey=True)
    for ax, gi, label in ((axes[0], lo_i, "lowest VFD"), (axes[1], hi_i, "highest VFD")):
        e = eps[owner[gi]]
        n = local[gi]
        a = e["actions"][n]  # (M, B, H, 14)
        M = a.shape[0]
        k = np.arange(a.shape[2])
        for m, col in zip(range(M), (C1, C2, C3)):
            for b in range(a.shape[1]):
                ax.plot(k, a[m, b, :, joint], color=col, alpha=0.55, lw=1.1,
                        label=e["members"][m] if b == 0 else None)
        ax.set_xlabel("index within action chunk")
        ax.set_title(f"{label} = {all_vfd[gi]:.0f}\n{e['name'].replace('episode_', '')}, "
                     f"t = {e['t_rel'][n]:.1f}s", loc="left", pad=8, fontsize=9.5)
        ax.grid(alpha=0.6)
        ax.legend(loc="best", ncol=M)
    axes[0].set_ylabel(f"{JOINT_NAMES[joint]} command (rad)")
    fig.suptitle("Sampled action chunks at the least and most uncertain frames",
                 x=0.005, ha="left", fontsize=13, color=INK, y=1.02)
    fig.tight_layout()
    _finish(fig, out / "fig7_chunk_fan.png",
            f"Every sampled chunk from every member, joint {JOINT_NAMES[joint]}. "
            "High VFD = the members' fields pull toward different futures.")


def make_video(eps: list[dict], obs_dir: pathlib.Path, name: str, out: pathlib.Path,
               fps: float) -> None:
    """Camera triptych with the live VFD trace underneath."""
    import imageio.v3 as iio

    ep = next((e for e in eps if e["name"] == name), None)
    if ep is None:
        print(f"  video: no episode named {name}")
        return
    obs = np.load(obs_dir / f"{name}.npz", allow_pickle=True)
    vfd, t = ep["vfd"], ep["t_rel"]
    n = min(len(vfd), len(obs["ts"]))
    hi = float(np.percentile(np.concatenate([e["vfd"] for e in eps]), 99))
    p90 = float(np.percentile(np.concatenate([e["vfd"] for e in eps]), 90))

    frames = []
    for i in range(n):
        strip = np.concatenate(
            [obs[f"image_{c}"][i] for c in ("top_camera", "left_camera", "right_camera")], axis=1)
        fig = plt.figure(figsize=(strip.shape[1] / 100, (strip.shape[0] + 190) / 100), dpi=100)
        ax_i = fig.add_axes([0, 0.475, 1, 0.525])
        ax_i.imshow(strip)
        ax_i.axis("off")
        ax_p = fig.add_axes([0.065, 0.11, 0.90, 0.30])
        ax_p.axhline(p90, color=INK3, lw=0.7, ls=(0, (3, 3)))
        ax_p.fill_between(t[:i + 1], 0, np.clip(vfd[:i + 1], 0, hi), color=C1, alpha=0.16, lw=0)
        ax_p.plot(t[:i + 1], np.clip(vfd[:i + 1], 0, hi), color=C1, lw=1.4)
        ax_p.scatter([t[i]], [min(vfd[i], hi)], s=30,
                     color=C2 if vfd[i] > p90 else C1, zorder=5, linewidths=0)
        ax_p.set_xlim(0, t[n - 1])
        ax_p.set_ylim(0, hi)
        ax_p.set_ylabel("VFD", fontsize=8)
        ax_p.set_xlabel("time in rollout (s)", fontsize=8)
        ax_p.grid(axis="y", alpha=0.6)
        ax_p.tick_params(labelsize=7)
        fig.text(0.065, 0.965, "top", fontsize=8, color="#ffffff")
        fig.text(0.399, 0.965, "left wrist", fontsize=8, color="#ffffff")
        fig.text(0.733, 0.965, "right wrist", fontsize=8, color="#ffffff")
        fig.text(0.065, 0.045, f"{name}   ·   ensemble {' + '.join(ep['members'])}   ·   "
                 f"VFD = {vfd[i]:.0f}" + ("   ABOVE 90th PERCENTILE" if vfd[i] > p90 else ""),
                 fontsize=8, color=C2 if vfd[i] > p90 else INK2)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    # libx264 + yuv420p needs even dimensions, and macro_block_size=1 turns off
    # imageio's auto-padding, so trim to even here.
    stack = np.stack(frames)
    stack = stack[:, : stack.shape[1] // 2 * 2, : stack.shape[2] // 2 * 2]
    path = out / f"overlay_{name}.mp4"
    iio.imwrite(path, stack, fps=fps, codec="libx264",
                pixelformat="yuv420p", macro_block_size=1)
    print(f"  wrote {path.name} ({len(frames)} frames @ {fps} fps)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uq-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--obs-dir", type=pathlib.Path, default=None)
    ap.add_argument("--video-episode", default=None, help="Episode name, or 'max' for the most uncertain.")
    ap.add_argument("--video-fps", type=float, default=3.0)
    args = ap.parse_args()

    eps = load_all(args.uq_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    members = " + ".join(eps[0]["members"])
    print(f"{len(eps)} episode(s), {sum(len(e['vfd']) for e in eps)} frames, ensemble {members}")

    fig_timelines(eps, args.out_dir, members)
    fig_ranking(eps, args.out_dir)
    fig_heatmap(eps, args.out_dir)
    fig_flowtime(eps, args.out_dir)
    fig_joints(eps, args.out_dir)
    fig_baselines(eps, args.out_dir)
    fig_fan(eps, args.out_dir)

    if args.video_episode and args.obs_dir:
        name = args.video_episode
        if name == "max":
            name = max(eps, key=lambda e: e["vfd"].mean())["name"]
        make_video(eps, args.obs_dir, name, args.out_dir, args.video_fps)


if __name__ == "__main__":
    main()
