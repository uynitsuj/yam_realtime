#!/usr/bin/env python3
"""Read passive-GELLO buttons without transmitting on the CAN bus.

Each CAN channel is consumed by its own callback thread. This avoids the
unbounded BufferedReader backlog that occurred when one loop alternated a
single read from buses with different report rates. Buttons are read from
device 6's 0x50F digital_inputs byte and debounced using kernel CAN timestamps.

Usage:
    uv run scripts/probe_gello_buttons.py
    uv run scripts/probe_gello_buttons.py --channels can_lead_l --duration 60
"""

from __future__ import annotations

import argparse
import collections
import threading
import time

import can

from robots_realtime.agents.teleoperation.passive_gello_protocol import (
    DEFAULT_BUTTON_DEVICE_ID,
    ENCODER_REPORT_ID,
    ButtonTracker,
    decode_encoder_report,
)


class ChannelButtonListener(can.Listener):
    """Process one channel immediately, without retaining an unbounded queue."""

    def __init__(
        self,
        channel: str,
        device_id: int,
        debounce_s: float,
        start_monotonic: float,
        print_lock: threading.Lock,
    ) -> None:
        self.channel = channel
        self.device_id = device_id
        self.tracker = ButtonTracker(debounce_s=debounce_s)
        self.start_monotonic = start_monotonic
        self.print_lock = print_lock
        self.frame_count = 0
        self.encoder_device_counts: collections.Counter[int] = collections.Counter()
        self.seen_button_device = False

    def on_message_received(self, msg: can.Message) -> None:
        self.frame_count += 1
        if msg.arbitration_id != ENCODER_REPORT_ID:
            return
        report = decode_encoder_report(msg.data)
        if report is None:
            return
        report_device = report[0]
        self.encoder_device_counts[report_device] += 1
        if report_device != self.device_id:
            return

        self.seen_button_device = True
        changed = self.tracker.update(report[3], now=msg.timestamp)
        if not changed:
            return

        stable = self.tracker.value
        button_0 = stable & 0x01
        button_1 = (stable >> 1) & 0x01
        with self.print_lock:
            print(
                f"{time.monotonic() - self.start_monotonic:7.2f}  "
                f"{self.channel:<12} {stable:>4}  {stable:>10_b}  "
                f"{('PRESSED' if button_0 else '-'):^18}  "
                f"{('PRESSED' if button_1 else '-'):^18}",
                flush=True,
            )


def probe(
    channels: list[str],
    bitrate: int,
    duration_s: float,
    device_id: int,
    debounce_s: float,
) -> None:
    buses: dict[str, can.BusABC] = {}
    listeners: dict[str, ChannelButtonListener] = {}
    notifiers: dict[str, can.Notifier] = {}
    start = time.monotonic()
    print_lock = threading.Lock()

    for channel in channels:
        try:
            bus = can.interface.Bus(interface="socketcan", channel=channel, bitrate=bitrate)
        except Exception as exc:
            print(f"[{channel}] could not open: {exc}")
            continue
        listener = ChannelButtonListener(channel, device_id, debounce_s, start, print_lock)
        buses[channel] = bus
        listeners[channel] = listener
        notifiers[channel] = can.Notifier(bus, [listener])

    if not buses:
        raise SystemExit("no CAN interfaces opened - are can_lead_l/can_lead_r up?")

    print(f"Listening on {', '.join(buses)} for {duration_s:.0f}s - press each button now.")
    print(f"Independent callback readers; CAN-timestamp debounce={debounce_s * 1_000:.0f} ms.\n")
    print(
        f"{'time':>7}  {'channel':<12} {'raw':>4}  {'bin':>10}  "
        f"{'index0 TOP/YELLOW':^18}  {'index1 LOWER/WHITE':^18}"
    )
    print("-" * 89)

    try:
        deadline = start + duration_s
        while time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        pass
    finally:
        for notifier in notifiers.values():
            notifier.stop()
        for bus in buses.values():
            bus.shutdown()

    elapsed = max(time.monotonic() - start, 1e-9)
    print()
    for channel, listener in listeners.items():
        device_rates = ", ".join(
            f"{device}:{count / elapsed:.1f} Hz" for device, count in sorted(listener.encoder_device_counts.items())
        )
        print(f"[{channel}] consumed {listener.frame_count / elapsed:.1f} CAN frames/s; devices: {device_rates}")
        if listener.seen_button_device:
            print(f"[{channel}] OK: reading buttons from device {device_id}'s 0x50F digital inputs.")
        else:
            print(f"[{channel}] WARNING: no 0x50F reports from button device {device_id}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", nargs="+", default=["can_lead_l", "can_lead_r"])
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--device-id", type=int, default=DEFAULT_BUTTON_DEVICE_ID)
    parser.add_argument("--debounce-ms", type=float, default=20.0)
    args = parser.parse_args()
    probe(args.channels, args.bitrate, args.duration, args.device_id, args.debounce_ms / 1_000.0)


if __name__ == "__main__":
    main()
