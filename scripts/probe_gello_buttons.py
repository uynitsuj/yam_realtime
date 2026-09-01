#!/usr/bin/env python3
"""Dump passive-GELLO button bits so the DAgger takeover button can be identified.

Read-only: listens on the leader's SocketCAN bus and prints the gripper
encoder's ``digital_inputs`` byte as it changes. Nothing is transmitted, so this
is safe to run with the arms powered.

The byte packs two momentary switches, decoded the same way i2rt does in
``dm_driver.PassiveEncoderReader._parse_encoder_message``::

    button_state = [digital_inputs % 2, digital_inputs // 2]   # [top, grip]

On this station's passive gellos the two switches are:

    index 0 = bit 0 = TOP button    = YELLOW
    index 1 = bit 1 = LOWER button  = WHITE

The bit order is i2rt's; the top/lower naming comes from i2rt labelling the pair
``[button_top, button_grip]``, which describes the teaching handle on the active
leader — so treat the position/colour mapping below as the thing this script
exists to confirm. Press each button in turn and check that yellow lights up
index 0 and white lights up index 1. If they're reversed, swap
``takeover_button_index`` and ``episode_button_index`` in the DAgger config. If
neither ever changes, the switches aren't wired on this leader.

Usage:
    uv run scripts/probe_gello_buttons.py                    # both leaders
    uv run scripts/probe_gello_buttons.py --channels can_lead_l
"""

from __future__ import annotations

import argparse
import struct
import time

import can

ENCODER_REPORT_ID = 0x50F
ENCODER_STRUCT = "!B h h B"
ENCODER_STRUCT_SIZE = struct.calcsize(ENCODER_STRUCT)
GRIPPER_DEVICE_ID = 6


def probe(channels: list[str], bitrate: int, duration_s: float) -> None:
    buses = {}
    for ch in channels:
        try:
            buses[ch] = can.interface.Bus(interface="socketcan", channel=ch, bitrate=bitrate)
        except Exception as exc:
            print(f"[{ch}] could not open: {exc}")
    if not buses:
        raise SystemExit("no CAN interfaces opened — is can_lead_l/can_lead_r up?")

    notifiers = {}
    readers = {}
    for ch, bus in buses.items():
        readers[ch] = can.BufferedReader()
        notifiers[ch] = can.Notifier(bus, [readers[ch]])

    print(f"Listening on {', '.join(buses)} for {duration_s:.0f}s — press each button now.")
    print("Reporting only on CHANGE of the digital_inputs byte.\n")
    print(f"{'time':>7}  {'channel':<12} {'raw':>4}  {'bin':>10}  "
          f"{'index0 TOP/YELLOW':^18}  {'index1 LOWER/WHITE':^18}")
    print("-" * 88)

    last: dict[str, int | None] = {ch: None for ch in buses}
    seen_any = {ch: False for ch in buses}
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < duration_s:
            for ch, reader in readers.items():
                msg = reader.get_message(timeout=0.02)
                if msg is None or msg.arbitration_id != ENCODER_REPORT_ID:
                    continue
                if len(msg.data) != ENCODER_STRUCT_SIZE:
                    continue
                dev, _pos, _vel, digital_inputs = struct.unpack(ENCODER_STRUCT, msg.data)
                if dev != GRIPPER_DEVICE_ID:
                    continue
                seen_any[ch] = True
                if digital_inputs == last[ch]:
                    continue
                last[ch] = digital_inputs
                b0, b1 = digital_inputs % 2, digital_inputs // 2
                print(
                    f"{time.monotonic() - t0:7.2f}  {ch:<12} {digital_inputs:>4}  "
                    f"{digital_inputs:>10_b}  {('PRESSED' if b0 else '-'):^18}  "
                    f"{('PRESSED' if b1 else '-'):^18}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        for n in notifiers.values():
            n.stop()
        for b in buses.values():
            b.shutdown()

    print()
    for ch in buses:
        if not seen_any[ch]:
            print(f"[{ch}] WARNING: no frames from gripper device {GRIPPER_DEVICE_ID} — check power/wiring.")
        elif last[ch] in (0, None):
            print(f"[{ch}] digital_inputs never left 0 — buttons likely not wired on this leader.")
        else:
            print(f"[{ch}] saw digital_inputs change; last value {last[ch]}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channels", nargs="+", default=["can_lead_l", "can_lead_r"])
    ap.add_argument("--bitrate", type=int, default=1_000_000)
    ap.add_argument("--duration", type=float, default=60.0)
    args = ap.parse_args()
    probe(args.channels, args.bitrate, args.duration)


if __name__ == "__main__":
    main()
