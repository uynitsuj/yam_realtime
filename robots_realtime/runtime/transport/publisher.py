"""ZMQ publisher with per-topic publish-rate throttling and optional writer recording."""

from __future__ import annotations

import time

import zmq

from robots_realtime.runtime.transport.message_bus import DEFAULT_PUB_PORT
from robots_realtime.runtime.transport.serialization import pack


class Publisher:
    """Publishes msgpack-encoded messages to the XPUB/XSUB broker.

    Also records every message to an injected Writer at the full call rate,
    independent of the ZMQ publish throttle.

    Args:
        node_name:    Prepended to every topic as ``"{node_name}/{topic}"``.
        writer:       Optional Writer instance.  If provided and open, every
                      publish() call writes to it before throttle check.
        publish_freq: If set, caps how often each topic is actually sent on the
                      bus.  None sends on every call.
        host:         Broker host (default localhost).
        port:         Broker XSUB port (publishers connect here).
    """

    def __init__(
        self,
        node_name: str,
        writer=None,
        publish_freq: float | None = None,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PUB_PORT,
    ) -> None:
        self._node_name = node_name
        self._writer = writer  # Writer | None
        self._publish_freq = publish_freq
        self._min_interval = (1.0 / publish_freq) if publish_freq else 0.0
        self._last_sent: dict[str, float] = {}

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.connect(f"tcp://{host}:{port}")
        # Give the slow-joiner a moment to let subscriptions propagate
        time.sleep(0.01)

    def publish(
        self,
        topic_suffix: str,
        data: dict,
        ts: float | None = None,
        record: bool = True,
        record_data: dict | None = None,
    ) -> bool:
        """Send ``data`` on ``"{node_name}/{topic_suffix}"``.

        Records to the writer (if open) at the full call rate, unless ``record``
        is False or the topic is internal (prefixed with ``_``).

        ``record_data`` lets a node split bus and disk payloads — write the
        full-fidelity copy while shipping a smaller one over the wire (e.g.
        CameraNode resizes frames for the bus but keeps full-res on disk).
        Defaults to ``data`` when None.

        Returns True if the message was sent on the bus, False if throttled.
        """
        now = time.time()
        ts_val = ts if ts is not None else now

        if (
            record
            and self._writer is not None
            and self._writer.is_open
            and not topic_suffix.startswith("_")
        ):
            self._writer.write(topic_suffix, ts_val, data if record_data is None else record_data)

        # Throttle ZMQ bus sends
        if self._min_interval:
            last = self._last_sent.get(topic_suffix, 0.0)
            if now - last < self._min_interval:
                return False

        self._last_sent[topic_suffix] = now

        topic = f"{self._node_name}/{topic_suffix}"
        envelope = {"ts": ts_val, "src": self._node_name, "data": data}
        self._sock.send_multipart([topic.encode(), pack(envelope)])
        return True

    def close(self) -> None:
        self._sock.close(linger=0)
