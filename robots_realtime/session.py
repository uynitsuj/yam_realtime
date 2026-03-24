"""Session — top-level orchestrator.

The Session's monitor thread subscribes to every topic on the bus and measures
live publish-Hz per node.  All recording is delegated to the nodes themselves
via start_recording(save_dir) / stop_recording().

  - Nodes own their writers — no MCAP/video writing in the monitor thread.
  - Session.start_episode() creates the episode directory and calls
    host.start_recording(save_dir) for all hosts.
  - Session.end_episode() calls host.stop_recording() for all hosts.
"""

from __future__ import annotations

import datetime
import json
import signal
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import zmq

from robots_realtime.nodes.base import Node, ProcessHost
from robots_realtime.transport.message_bus import MessageBus, DEFAULT_SUB_PORT
from robots_realtime.transport.serialization import unpack


_HZ_WINDOW = 30


def _node_descriptor(node) -> dict:
    d: dict = {
        "name": node.name,
        "published_topics": list(getattr(node, "published_topics", [])),
        "subscribed_topics": list(getattr(node, "subscribed_topics", [])),
    }
    # Include sim node config so replay tools can auto-detect scene/task.
    sim_cfg: dict = {}
    if getattr(node, "_scene", None) is not None:
        sim_cfg["scene"] = node._scene
    if getattr(node, "_task", None) is not None:
        sim_cfg["task"] = node._task
    if sim_cfg:
        d["sim_config"] = sim_cfg
    return d


@dataclass
class NodeStatus:
    name: str
    alive: bool = True
    hz: float = 0.0
    _timestamps: dict[str, deque] = field(default_factory=dict, repr=False)

    def record_message(self, topic_suffix: str) -> None:
        buf = self._timestamps.setdefault(topic_suffix, deque(maxlen=_HZ_WINDOW))
        buf.append(time.perf_counter())
        best = max(self._timestamps.values(), key=len)
        if len(best) >= 2:
            span = best[-1] - best[0]
            self.hz = (len(best) - 1) / span if span > 0 else 0.0


class Session:
    """Orchestrates a graph of Nodes.

    Each node runs in its own subprocess via ProcessHost.  Recording is
    delegated to the nodes via start_recording() / stop_recording().

    Args:
        nodes:                List of Node instances to run.
        save_root:            Root directory for episode recordings.
        record_node_names:    Subset of node names to record; defaults to all.
        record_topic:         Full bus topic carrying the boolean record signal
                              (e.g. "gello_left/record").
        auto_record_duration: If set, automatically start recording on
                              session start and stop after this many seconds.
        pub_port:             MessageBus XSUB port.
        sub_port:             MessageBus XPUB port.
    """

    def __init__(
        self,
        nodes: list,
        save_root: str | Path = "recordings",
        record_node_names: list[str] | None = None,
        record_topic: str | None = None,
        auto_record_duration: float | None = None,
        pub_port: int = 5555,
        sub_port: int = DEFAULT_SUB_PORT,
    ) -> None:
        self._pub_port = pub_port
        self._sub_port = sub_port
        self._save_root = Path(save_root)
        self._record_topic = record_topic
        self._auto_record_duration = auto_record_duration
        self._session_start_time = time.time()

        self._hosts: list[ProcessHost] = []
        all_node_names: list[str] = []
        self._node_descriptors: list[dict] = []

        for item in nodes:
            if isinstance(item, ProcessHost):
                # Accept pre-built hosts (advanced usage)
                self._hosts.append(item)
                all_node_names.extend(item.node_names)
            elif isinstance(item, Node):
                self._hosts.append(ProcessHost(item))
                all_node_names.extend([item.name])
                self._node_descriptors.append(_node_descriptor(item))
            else:
                raise TypeError(f"Unexpected item in nodes list: {type(item)}")

        self._record_node_names: list[str] = record_node_names or all_node_names

        self._bus = MessageBus(pub_port=pub_port, sub_port=sub_port)

        self._status: dict[str, NodeStatus] = {
            name: NodeStatus(name=name) for name in all_node_names
        }

        self._stop_event = threading.Event()
        self._prev_record_signal = False
        self._monitor_thread: threading.Thread | None = None
        self._log_dir: Path | None = None

        # Recording state
        self._is_recording: bool = False
        self._episode_dir: Path | None = None
        self._episode_start_time: float | None = None
        self._recording_lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        import tempfile
        self._log_dir = Path(tempfile.mkdtemp(prefix="rr_logs_"))

        self._bus.start()
        time.sleep(0.1)

        for host in self._hosts:
            log_path = self._log_dir / f"{host.node_name}.log"
            host.start(log_path=log_path)
        for host in self._hosts:
            host.send_start()

        self._setup_signal_handlers()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="SessionMonitor"
        )
        self._monitor_thread.start()

        if self._auto_record_duration is not None:
            t = threading.Thread(
                target=self._auto_record_timer,
                args=(self._auto_record_duration,),
                daemon=True,
            )
            t.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._is_recording:
            self.end_episode(save=True)
        threads = [
            threading.Thread(target=host.stop, daemon=True)
            for host in self._hosts
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=8.0)
        self._bus.stop()

    def wait(self) -> None:
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    # ── Recording controls ────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def episode_start_time(self) -> float | None:
        return self._episode_start_time

    @property
    def save_root(self) -> Path:
        return self._save_root

    def start_episode(self) -> None:
        with self._recording_lock:
            if self._is_recording:
                return
            save_dir = self._make_episode_dir()
            self._episode_dir = Path(save_dir)
            self._episode_start_time = time.time()
            self._is_recording = True

        # Delegate recording to all hosts
        for host in self._hosts:
            try:
                host.start_recording(save_dir)
            except Exception:
                pass

    def end_episode(self, save: bool = True) -> Path | None:
        with self._recording_lock:
            if not self._is_recording:
                return None
            self._is_recording = False
            episode_dir = self._episode_dir
            self._episode_dir = None
            self._episode_start_time = None

        # Delegate stop_recording to all hosts
        for host in self._hosts:
            try:
                host.stop_recording()
            except Exception:
                pass

        if not save and episode_dir is not None:
            import shutil
            shutil.rmtree(episode_dir, ignore_errors=True)
            return None

        return episode_dir

    def toggle_recording(self) -> None:
        if self._is_recording:
            self.end_episode(save=True)
        else:
            self.start_episode()

    @property
    def log_dir(self) -> Path | None:
        return self._log_dir

    @property
    def web_endpoints(self) -> list[str]:
        urls = []
        for host in self._hosts:
            urls.extend(host._node.web_endpoints)
        return urls

    # ── Status ────────────────────────────────────────────────────────────────

    def node_statuses(self) -> list[NodeStatus]:
        return list(self._status.values())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _make_episode_dir(self) -> str:
        """Create the episode directory and write session_meta.json."""
        now = datetime.datetime.now()
        uid = uuid.uuid4().hex[:8]
        path = (
            self._save_root
            / now.strftime("%Y%m%d")
            / f"episode_{now.strftime('%H%M%S')}_{uid}"
        )
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "session_start_time": self._session_start_time,
            "episode_start_time": time.time(),
            "episode_dir": str(path),
            "nodes": self._node_descriptors,
            "record_topic": self._record_topic,
            "save_root": str(self._save_root),
        }
        try:
            (path / "session_meta.json").write_text(
                json.dumps(meta, indent=2, default=str)
            )
        except Exception:
            pass

        return str(path)

    def _monitor_loop(self) -> None:
        """Subscribe to all bus topics; measure Hz per node.

        Also watches record_topic for start/stop signals.
        """
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.connect(f"tcp://127.0.0.1:{self._sub_port}")
        sock.setsockopt(zmq.SUBSCRIBE, b"")

        node_names = set(self._status)

        while not self._stop_event.is_set():
            while sock.poll(0):
                try:
                    parts = sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if len(parts) < 2:
                    continue

                topic_b, payload_b = parts[0], parts[1]
                topic = topic_b.decode()
                parts = topic.split("/", 1)
                if not parts:
                    continue
                node_name = parts[0]
                topic_suffix = parts[1] if len(parts) > 1 else ""

                if node_name not in node_names:
                    continue

                # Measure Hz
                self._status[node_name].record_message(topic_suffix)

                # Handle record signal from gello (or any configured topic)
                if self._record_topic and topic == self._record_topic:
                    try:
                        envelope = unpack(payload_b)
                        want = bool(envelope.get("data", {}).get("record", False))
                        if want and not self._prev_record_signal:
                            self.start_episode()
                        elif not want and self._prev_record_signal:
                            self.end_episode(save=True)
                        self._prev_record_signal = want
                    except Exception:
                        pass

            time.sleep(0.005)

        sock.close(linger=0)

    def _auto_record_timer(self, duration: float) -> None:
        # Brief warmup so ZMQ sockets connect and nodes start publishing
        time.sleep(0.3)
        self.start_episode()
        time.sleep(duration)
        self.end_episode(save=True)
        self._stop_event.set()

    def _setup_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGUSR1, lambda *_: self.toggle_recording())
            signal.signal(signal.SIGUSR2, lambda *_: self.end_episode(save=False))
        except (OSError, ValueError):
            pass
