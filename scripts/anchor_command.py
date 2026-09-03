#!/usr/bin/env python3
"""Turn a natural-language command into 3-D anchors on the bus for FkSteeredAgent.

Two steps, deliberately separate because they contend for the camera:

  1. ground   Opens the top D405 directly, grounds the command with Qwen2.5-VL at
              FULL resolution, deprojects each hit through depth + the calibrated
              extrinsics into the LEFT-ARM BASE frame, and writes a JSON.
              *** Run this with the session STOPPED — it needs the camera. ***

  2. publish  Reads that JSON and publishes one anchor at a time on
              ``anchor_cmd/point`` for the running session. Advances to the next
              anchor when the commanded arm's gripper closes near the current one
              (same FK + release rule the agent uses), or on ENTER.

Splitting it this way also means the VLM runs once on a static scene rather than
every tick — grounding is a command-level act, not a control-loop one.

Usage
-----
    # scene is static, session not yet running
    uv run scripts/anchor_command.py ground --command "green bottles only"

    # start the session (agent_class: FkSteeredAgent, steer_mode: on), then
    uv run scripts/anchor_command.py publish

Wire the session YAML's AgentNode with:
    state_topics:
      anchor: anchor_cmd/point
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_EXTRINSICS = REPO / "configs/camera_extrinsics/us07_yam_top_d405.yaml"
DEFAULT_OUT = REPO / "out/anchors.json"
TOP_SERIAL = "427622273855"
MIDLINE_Y = -0.305          # halfway between the two arm bases
RIGHT_BASE_OFFSET = np.array([0.0, -0.61, 0.0])

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("anchor_command")


# --------------------------------------------------------------------------- #
def ground(args: argparse.Namespace) -> None:
    import cv2
    import pyrealsense2 as rs
    import torch
    from PIL import Image
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)
    sys.path.insert(0, str(REPO / "scripts"))
    from verify_top_cam_extrinsics import load_extrinsics

    # ── capture full-res colour + temporally-medianed depth ──────────────── #
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(args.device_id)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    try:
        prof = pipe.start(cfg)
    except RuntimeError as exc:
        raise SystemExit(
            f"could not open the top camera ({exc}).\n"
            "`ground` needs the camera to itself — stop the session first."
        ) from exc
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    acc = []
    for i in range(40):
        f = align.process(pipe.wait_for_frames())
        if i >= 10:
            acc.append(np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * scale)
    bgr = np.asanyarray(f.get_color_frame().get_data())
    intr = f.get_color_frame().profile.as_video_stream_profile().get_intrinsics()
    pipe.stop()
    D = np.stack(acc)
    depth = np.nanmedian(np.where(D > 0, D, np.nan), axis=0)
    H, W = depth.shape
    logger.info("captured %dx%d, depth valid %.1f%%", W, H, 100 * np.isfinite(depth).mean())

    R_cw, t_cw = load_extrinsics(Path(args.extrinsics))

    # ── ground the command ───────────────────────────────────────────────── #
    M = "Qwen/Qwen2.5-VL-7B-Instruct"
    proc = AutoProcessor.from_pretrained(M, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        M, dtype=torch.bfloat16, device_map={"": args.gpu},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)).eval()
    img = Image.fromarray(bgr[:, :, ::-1])

    def ask(q: str):
        msg = [{"role": "user", "content": [{"type": "image", "image": img},
                                            {"type": "text", "text": q}]}]
        enc = proc(text=[proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)],
                   images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=640, do_sample=False)
        a = proc.batch_decode(o[:, enc.input_ids.shape[1]:], skip_special_tokens=True)[0]
        gh, gw = enc["image_grid_thw"][0][1].item() * 14, enc["image_grid_thw"][0][2].item() * 14
        try:
            items = json.loads(re.search(r"\[.*\]", a, re.S).group(0))
        except Exception:
            logger.warning("could not parse VLM reply: %s", a[:200]); return []
        out = []
        for e in items:
            b = e.get("bbox_2d") or e.get("bbox")
            if b and len(b) == 4:
                out.append(((b[0] + b[2]) / 2 * W / gw, (b[1] + b[3]) / 2 * H / gh, str(e.get("label", ""))))
        return out

    # The spatial-role prompt has the best measured recall; a referring
    # expression then selects the subset the command actually names. Asking for
    # both at once loses bottles (see US07_REAL_CELL.md §4).
    every = ask("Locate all bottles on the wooden table. Output JSON with bbox_2d and a "
                "label saying whether each is leftmost, middle, or rightmost.")
    logger.info("enumerated %d bottle(s)", len(every))
    chosen = every
    if args.command.strip().lower() not in ("", "all", "all bottles"):
        hits = ask(f"Locate every object matching this description: {args.command}. "
                   "Output JSON with bbox_2d and label. Only objects on the wooden table.")
        logger.info("command %r matched %d", args.command, len(hits))
        # Trust the REFERRING boxes as the anchors. An earlier version snapped each
        # hit onto the nearest ENUMERATED box, which silently produced wrong
        # anchors: enumeration recall is imperfect, so a correctly-located green
        # bottle would snap onto whichever clear bottle happened to be enumerated
        # nearby — boxing the wrong object while looking entirely confident.
        chosen = hits or [(u, v, l) for u, v, l in every]

    def lift(u, v, r=9):
        p = depth[max(0, int(v) - r):int(v) + r + 1, max(0, int(u) - r):int(u) + r + 1]
        p = p[np.isfinite(p)]
        if p.size == 0:
            return None
        z = float(np.median(p))
        Pc = np.array([(u - intr.ppx) / intr.fx * z, (v - intr.ppy) / intr.fy * z, z])
        return R_cw @ Pc + t_cw

    # Coarse sanity gate — GROSS errors only (floor, ceiling, a wild depth hole).
    #
    # Deliberately NOT tight: anchor z varies legitimately with object POSE. A
    # bottle lying flat lifts to z ~ +0.05; the same bottle standing upright lifts
    # to z ~ +0.18, because depth sees its cap ~14 cm nearer the camera. An
    # earlier version of this gate used z_max=0.12 and would have rejected every
    # upright bottle as "arm occlusion" — a fixed height threshold cannot tell an
    # occluder from an upright object, so do not try to make it. The overlay image
    # is the real check; this only catches nonsense.
    anchors, rejected = [], []
    for u, v, label in chosen:
        P = lift(u, v)
        if P is None:
            logger.warning("no depth at (%.0f, %.0f) — skipping", u, v); continue
        if not (args.z_min <= P[2] <= args.z_max):
            rejected.append((u, v, label, float(P[2])))
            continue
        anchors.append({"point": [float(x) for x in P], "uv": [float(u), float(v)], "label": label})
    for u, v, label, z in rejected:
        logger.warning("REJECTED %r at (%.0f, %.0f): z=%+.3f m outside [%+.2f, %+.2f] — "
                       "grossly off the table. Check the overlay: the box is probably on "
                       "the floor, the bin rim, or a depth hole.",
                       label, u, v, z, args.z_min, args.z_max)
    if not anchors:
        raise SystemExit("every candidate anchor failed the height gate — retract the arms "
                         "clear of the table and re-run `ground`.")
    anchors.sort(key=lambda a: a["uv"][0], reverse=(args.order == "right_to_left"))

    logger.info("\ncommand %r -> %d anchor(s), execution order (%s):", args.command, len(anchors), args.order)
    for i, a in enumerate(anchors):
        P = a["point"]
        arm = "left" if P[1] > MIDLINE_Y else "right"
        logger.info("  %d. base [%+.3f %+.3f %+.3f] m  (%s arm)  %s", i + 1, *P, arm, a["label"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"command": args.command, "order": args.order, "anchors": anchors},
              open(args.out, "w"), indent=1)
    logger.info("wrote %s", args.out)

    vis = bgr.copy()
    for i, a in enumerate(anchors):
        u, v = int(a["uv"][0]), int(a["uv"][1])
        cv2.circle(vis, (u, v), 12, (0, 0, 255), 3)
        cv2.putText(vis, f"#{i+1}", (u + 14, v), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imwrite(str(Path(args.out).with_suffix(".png")), vis)


# --------------------------------------------------------------------------- #
def publish(args: argparse.Namespace) -> None:
    from robots_realtime.agents.policy_learning.fk_steered_agent import (
        _GRIP_IDX, fk_grasp_site)
    from robots_realtime.runtime.transport.publisher import Publisher
    from robots_realtime.runtime.transport.subscriber import Subscriber

    data = json.load(open(args.anchors))
    anchors = [np.asarray(a["point"], float) for a in data["anchors"]]
    if not anchors:
        raise SystemExit(f"{args.anchors} has no anchors")
    logger.info("command %r: %d anchor(s)", data.get("command"), len(anchors))

    pub = Publisher("anchor_cmd", port=5555)
    sub = Subscriber(["yam_left/joint_state", "yam_right/joint_state"])
    idx = 0
    advance = threading.Event()

    def keys() -> None:
        for _ in sys.stdin:
            advance.set()
    threading.Thread(target=keys, daemon=True).start()
    logger.info("publishing on anchor_cmd/point — ENTER to skip to the next anchor, Ctrl-C to stop")

    def grasped(anchor: np.ndarray) -> bool:
        """Same release rule the agent uses, on MEASURED joints."""
        arm = "left" if anchor[1] > MIDLINE_Y else "right"
        st = sub.get_data(f"yam_{arm}/joint_state")
        if not st or "joint_pos" not in st:
            return False
        q = np.asarray(st["joint_pos"], float).reshape(-1)
        if q.size < 6:
            return False
        p = fk_grasp_site(q[None])[0]
        if arm == "right":
            p = p + RIGHT_BASE_OFFSET
        grip = st.get("gripper_pos")
        grip = float(np.asarray(grip).reshape(-1)[0]) if grip is not None else 1.0
        return bool(np.linalg.norm(p - anchor) < args.release_radius and grip < args.grip_closed)

    hold_until = 0.0
    try:
        while idx < len(anchors):
            a = anchors[idx]
            pub.publish("point", {"point": a.astype(np.float32)}, record=False)
            now = time.time()
            if advance.is_set() or (now > hold_until and hold_until and grasped(a)):
                if advance.is_set():
                    logger.info("anchor %d/%d skipped by operator", idx + 1, len(anchors))
                else:
                    logger.info("anchor %d/%d GRASPED -> advancing", idx + 1, len(anchors))
                advance.clear()
                idx += 1
                hold_until = time.time() + args.settle_s
                continue
            if not hold_until:
                hold_until = time.time() + args.settle_s
                logger.info("anchor %d/%d -> base [%+.3f %+.3f %+.3f] (%s arm)", idx + 1,
                            len(anchors), *a, "left" if a[1] > MIDLINE_Y else "right")
            time.sleep(0.05)
        logger.info("all anchors done; publishing nothing further (agent reverts to unsteered)")
    except KeyboardInterrupt:
        logger.info("stopped")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    g = sp.add_parser("ground", help="VLM-ground a command into base-frame anchors (session STOPPED)")
    g.add_argument("--command", default="green bottles only")
    g.add_argument("--order", choices=("left_to_right", "right_to_left"), default="left_to_right")
    g.add_argument("--device-id", default=TOP_SERIAL)
    g.add_argument("--extrinsics", default=str(DEFAULT_EXTRINSICS))
    g.add_argument("--out", default=str(DEFAULT_OUT))
    g.add_argument("--gpu", type=int, default=0)
    g.add_argument("--z-min", type=float, default=-0.06,
                   help="reject anchors below this base-frame z (table is ~-0.009)")
    g.add_argument("--z-max", type=float, default=0.35,
                   help="reject anchors above this base-frame z. Loose on purpose: an "
                        "UPRIGHT bottle legitimately lifts to ~+0.18")
    g.set_defaults(func=ground)

    p = sp.add_parser("publish", help="publish anchors on the bus for a running session")
    p.add_argument("--anchors", default=str(DEFAULT_OUT))
    p.add_argument("--release-radius", type=float, default=0.12)
    p.add_argument("--grip-closed", type=float, default=0.35)
    p.add_argument("--settle-s", type=float, default=3.0,
                   help="grace period after switching anchors before grasp-advance can fire")
    p.set_defaults(func=publish)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
