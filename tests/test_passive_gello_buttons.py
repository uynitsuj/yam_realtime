import struct

from robots_realtime.agents.teleoperation.passive_gello_protocol import (
    ButtonTracker,
    decode_encoder_report,
)


def encoder_report(device: int, inputs: int) -> bytes:
    return struct.pack("!B h h B", device, 123, -45, inputs)


def test_encoder_report_decodes_device_and_buttons() -> None:
    assert decode_encoder_report(encoder_report(6, 0b11)) == (6, 123, -45, 0b11)


def test_encoder_decoder_rejects_bad_payload_size() -> None:
    assert decode_encoder_report(b"short") is None


def test_button_tracker_debounces_using_supplied_timestamps() -> None:
    tracker = ButtonTracker(debounce_s=0.02)
    assert not tracker.update(1, now=1.000)
    assert not tracker.update(0, now=1.005)
    assert not tracker.update(1, now=1.010)
    assert not tracker.update(1, now=1.029)
    assert tracker.update(1, now=1.030)
    assert tracker.value == 1


def test_zero_debounce_publishes_immediately() -> None:
    tracker = ButtonTracker(debounce_s=0)
    assert tracker.update(2, now=1.0)
    assert tracker.value == 2
    assert tracker.update(0, now=2.0)
    assert tracker.value == 0
