"""ZMQ subscriber — keeps only the latest message per topic.

Drains the socket in a background thread so callers never block on
socket I/O.  get_data() is always O(1).
"""

from __future__ import annotations

import threading

import zmq

from robots_realtime.transport.message_bus import DEFAULT_SUB_PORT
from robots_realtime.transport.serialization import unpack


class Subscriber:
    """Subscribes to one or more topic prefixes on the XPUB/XSUB broker.

    Maintains a latest-per-topic buffer updated by a background drain thread.
    Intermediate messages from fast producers are silently dropped — callers
    always get the freshest data.

    Args:
        topics: List of full topic strings to subscribe to,
                e.g. ``["gello_left/joint_pos", "camera_0/rgb"]``.
                An empty list subscribes to everything (use with care).
        host: Broker host.
        port: Broker XPUB port (subscribers connect here).
    """

    def __init__(
        self,
        topics: list[str],
        host: str = "127.0.0.1",
        port: int = DEFAULT_SUB_PORT,
    ) -> None:
        self._latest: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(f"tcp://{host}:{port}")

        if not topics:
            self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        else:
            for t in topics:
                self._sock.setsockopt(zmq.SUBSCRIBE, t.encode())

        self._thread = threading.Thread(
            target=self._drain_loop, daemon=True, name="SubDrain"
        )
        self._thread.start()

    def _drain_loop(self) -> None:
        """Background thread: drain socket and keep latest per topic."""
        while not self._stop.is_set():
            if self._sock.poll(5):  # 5 ms timeout
                try:
                    parts = self._sock.recv_multipart(zmq.NOBLOCK)
                    if len(parts) >= 2:
                        envelope = unpack(parts[1])
                        with self._lock:
                            self._latest[parts[0].decode()] = envelope
                except zmq.Again:
                    pass

    def drain(self) -> int:
        """No-op — draining is handled by the background thread."""
        return 0

    def drain_one(self, timeout_ms: int = 50) -> bool:
        """Block up to *timeout_ms* waiting for any new message."""
        import time
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest:
                    return True
            time.sleep(0.001)
        return False

    def get_latest(self, topic: str) -> dict | None:
        """Return the most recently received envelope for *topic*, or None."""
        with self._lock:
            return self._latest.get(topic)

    def get_data(self, topic: str) -> dict | None:
        """Convenience: return just the ``data`` field of the latest envelope."""
        with self._lock:
            env = self._latest.get(topic)
        return env["data"] if env is not None else None

    def get_timestamp(self, topic: str) -> float | None:
        """Return the hardware timestamp of the latest message on *topic*."""
        with self._lock:
            env = self._latest.get(topic)
        return env["ts"] if env is not None else None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close(linger=0)
