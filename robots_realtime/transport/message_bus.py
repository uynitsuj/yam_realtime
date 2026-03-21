"""XPUB/XSUB message broker.

Runs in its own subprocess so its GIL and GC pauses are isolated from all
node processes.  Publishers connect to the XSUB frontend; subscribers connect
to the XPUB backend.

Usage (typically via Session):
    bus = MessageBus()
    bus.start()           # spawns subprocess
    ...
    bus.stop()
"""

from __future__ import annotations

import multiprocessing as mp
import time


DEFAULT_PUB_PORT = 5555   # nodes publish  → connect here
DEFAULT_SUB_PORT = 5556   # nodes subscribe → connect here


def _broker_worker(pub_port: int, sub_port: int, ready_event: mp.Event) -> None:
    import zmq

    ctx = zmq.Context()
    xsub = ctx.socket(zmq.XSUB)   # receives from publishers
    xsub.bind(f"tcp://*:{pub_port}")

    xpub = ctx.socket(zmq.XPUB)   # sends to subscribers
    xpub.bind(f"tcp://*:{sub_port}")

    ready_event.set()
    zmq.proxy(xsub, xpub)         # blocks forever; proxy handles all routing


class MessageBus:
    """XPUB/XSUB broker running in a dedicated subprocess."""

    def __init__(
        self,
        pub_port: int = DEFAULT_PUB_PORT,
        sub_port: int = DEFAULT_SUB_PORT,
    ) -> None:
        self.pub_port = pub_port
        self.sub_port = sub_port
        self._proc: mp.Process | None = None

    def start(self, timeout: float = 5.0) -> None:
        ready = mp.Event()
        self._proc = mp.Process(
            target=_broker_worker,
            args=(self.pub_port, self.sub_port, ready),
            daemon=True,
            name="MessageBus",
        )
        self._proc.start()
        if not ready.wait(timeout):
            raise RuntimeError("MessageBus failed to start within timeout")

    def stop(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)
            self._proc = None

    def __enter__(self) -> "MessageBus":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
