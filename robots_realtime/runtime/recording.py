"""Writer abstractions for recording node data (aligned with market42/lab42 format).

Each node owns its writer; recording is started/stopped via start_recording() / stop_recording().

Writers:
    McapWriter     — MCAP with protobuf (RobotState/GripperState) or JSON fallback
    AsyncMp4Writer — per-topic MP4 with background encoder thread + timestamp sidecar
    NullWriter     — no-op (for nodes that manage their own multi-file writers)
"""

from __future__ import annotations

import json
import queue
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


# ── Abstract base ─────────────────────────────────────────────────────────────


class Writer(ABC):
    """Abstract base for all node data writers."""

    @abstractmethod
    def open(self, save_dir: str, node_name: str) -> None:
        """Open / create output files in save_dir for the given node."""

    @abstractmethod
    def write(self, topic: str, timestamp: float, data: dict) -> None:
        """Write one sample.

        Args:
            topic:     Topic suffix (e.g. "joint_state", "rgb").
            timestamp: Unix-epoch seconds (hardware clock or sim time).
            data:      Payload dict — numpy arrays are supported.
        """

    @abstractmethod
    def close(self) -> str:
        """Flush and close all output files.

        Returns:
            Path to the primary output file (or directory) as a string.
        """

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """True between open() and close()."""


# ── McapWriter ────────────────────────────────────────────────────────────────


class McapWriter(Writer):
    """Writes one MCAP file per node.

    Uses xdof_sdk protobuf schemas for well-known topics:
        joint_pos / joint_vel / joint_eff  →  RobotState (position/velocity/torque)
        gripper_pos                        →  GripperState (position)

    Everything else is JSON-encoded as a generic object schema.

    Topic naming in the MCAP file:
        /{node_name}-robot-state          — RobotState messages
        /{node_name}-gripper-state        — GripperState messages
        /{node_name}/{topic}              — JSON fallback for all other topics
    """

    def __init__(self) -> None:
        self._path: Path | None = None
        self._node_name: str = ""
        self._fh = None
        self._writer = None
        self._schema_ids: dict[str, int] = {}
        self._channel_ids: dict[str, int] = {}
        self._open: bool = False
        # Try to import protobuf schemas; fall back to JSON-only mode if unavailable
        self._proto_available = False
        self._RobotState = None
        self._GripperState = None

    def _try_import_proto(self) -> None:
        try:
            from xdof_sdk.proto.robot_state_pb2 import RobotState
            from xdof_sdk.proto.gripper_state_pb2 import GripperState
            self._RobotState = RobotState
            self._GripperState = GripperState
            self._proto_available = True
        except ImportError:
            self._proto_available = False

    def open(self, save_dir: str, node_name: str) -> None:
        from mcap.writer import Writer as McapBaseWriter

        self._try_import_proto()
        self._node_name = node_name
        self._path = Path(save_dir) / f"{node_name}.mcap"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "wb")
        self._writer = McapBaseWriter(self._fh)
        self._writer.start()
        self._schema_ids = {}
        self._channel_ids = {}
        self._open = True

    @property
    def is_open(self) -> bool:
        return self._open

    def _get_json_channel(self, topic_suffix: str) -> int:
        """Register a JSON schema+channel for an arbitrary topic."""
        key = f"json:{topic_suffix}"
        if key not in self._channel_ids:
            full_topic = f"/{self._node_name}/{topic_suffix}"
            schema_id = self._schema_ids.get(key)
            if schema_id is None:
                schema_id = self._writer.register_schema(
                    name=full_topic,
                    encoding="jsonschema",
                    data=json.dumps({"type": "object"}).encode(),
                )
                self._schema_ids[key] = schema_id
            channel_id = self._writer.register_channel(
                schema_id=schema_id,
                topic=full_topic,
                message_encoding="json",
            )
            self._channel_ids[key] = channel_id
        return self._channel_ids[key]

    def _get_proto_channel(self, proto_topic: str, proto_cls) -> int:
        """Register a protobuf schema+channel."""
        key = f"proto:{proto_topic}"
        if key not in self._channel_ids:
            schema_id = self._schema_ids.get(key)
            if schema_id is None:
                descriptor = proto_cls.DESCRIPTOR
                schema_id = self._writer.register_schema(
                    name=descriptor.full_name,
                    encoding="protobuf",
                    data=_file_descriptor_set_bytes(descriptor),
                )
                self._schema_ids[key] = schema_id
            channel_id = self._writer.register_channel(
                schema_id=schema_id,
                topic=proto_topic,
                message_encoding="protobuf",
            )
            self._channel_ids[key] = channel_id
        return self._channel_ids[key]

    def write(self, topic: str, timestamp: float, data: dict) -> None:
        if not self._open or self._writer is None:
            return

        now_ns = int(timestamp * 1e9)

        # Map well-known topics to protobuf if available
        if self._proto_available:
            if topic in ("joint_pos", "joint_vel", "joint_eff", "joint_state"):
                self._write_robot_state(topic, data, now_ns)
                return
            if topic in ("gripper_pos", "gripper_state"):
                self._write_gripper_state(data, now_ns)
                return

        # JSON fallback
        channel_id = self._get_json_channel(topic)
        payload = _serialize_json(data)
        self._writer.add_message(
            channel_id=channel_id,
            log_time=now_ns,
            publish_time=now_ns,
            data=payload,
        )

    def _write_robot_state(self, topic: str, data: dict, now_ns: int) -> None:
        msg = self._RobotState()
        # Map fields
        jp = data.get("joint_pos") or data.get("position")
        jv = data.get("joint_vel") or data.get("velocity")
        je = data.get("joint_eff") or data.get("torque")
        if jp is not None:
            msg.position[:] = _to_float_list(jp)
        if jv is not None:
            msg.velocity[:] = _to_float_list(jv)
        if je is not None:
            msg.torque[:] = _to_float_list(je)
        proto_topic = f"/{self._node_name}-robot-state"
        channel_id = self._get_proto_channel(proto_topic, self._RobotState)
        self._writer.add_message(
            channel_id=channel_id,
            log_time=now_ns,
            publish_time=now_ns,
            data=msg.SerializeToString(),
        )

    def _write_gripper_state(self, data: dict, now_ns: int) -> None:
        msg = self._GripperState()
        gp = data.get("gripper_pos") or data.get("position")
        if gp is not None:
            try:
                msg.position = float(gp)
            except (TypeError, ValueError):
                pass
        proto_topic = f"/{self._node_name}-gripper-state"
        channel_id = self._get_proto_channel(proto_topic, self._GripperState)
        self._writer.add_message(
            channel_id=channel_id,
            log_time=now_ns,
            publish_time=now_ns,
            data=msg.SerializeToString(),
        )

    def close(self) -> str:
        if not self._open:
            return str(self._path or "")
        self._open = False
        try:
            self._writer.finish()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        return str(self._path or "")

    def __enter__(self) -> "McapWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def _to_float_list(val) -> list[float]:
    """Convert numpy array or list to list of Python floats."""
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            return val.astype(float).tolist()
    except ImportError:
        pass
    if hasattr(val, "tolist"):
        return [float(x) for x in val.tolist()]
    return [float(x) for x in val]


def _serialize_json(data: dict) -> bytes:
    """JSON-serialize a dict, converting numpy arrays to lists."""
    def default(obj):
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
        except ImportError:
            pass
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(data, default=default).encode()


def _file_descriptor_set_bytes(descriptor) -> bytes:
    """Serialize a protobuf FileDescriptorSet for MCAP schema registration."""
    try:
        from google.protobuf import descriptor_pb2, descriptor_pool
        fds = descriptor_pb2.FileDescriptorSet()
        _collect_file_descriptors(descriptor.file, fds, set())
        return fds.SerializeToString()
    except Exception:
        return b""


def _collect_file_descriptors(file_desc, fds, seen: set) -> None:
    if file_desc.name in seen:
        return
    seen.add(file_desc.name)
    for dep in file_desc.dependencies:
        _collect_file_descriptors(dep, fds, seen)
    file_desc.CopyToProto(fds.file.add())


# ── AsyncMp4Writer ─────────────────────────────────────────────────────────────


_MP4_SENTINEL = None


class AsyncMp4Writer(Writer):
    """Writes per-topic MP4 files with background encoder threads.

    For each unique topic, a separate encoder thread is spawned on the first
    write call.  Frames are enqueued non-blocking from the caller thread.

    File naming:
        {node_name}-images-{topic}.mp4
        {node_name}-timestamp.npy          (one sidecar per topic)

    Args:
        fps: Output video frame rate.
        crf: H.264 constant-rate factor (lower = higher quality).
    """

    def __init__(self, fps: float = 30.0, crf: int = 18) -> None:
        self._fps = fps
        self._crf = crf
        self._save_dir: Path | None = None
        self._node_name: str = ""
        self._open: bool = False
        # Per-topic: queue, timestamps list, writer handle, thread
        self._queues: dict[str, queue.Queue] = {}
        self._timestamps: dict[str, list] = {}
        self._imageio_writers: dict[str, Any] = {}
        self._threads: dict[str, threading.Thread] = {}

    def open(self, save_dir: str, node_name: str) -> None:
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._node_name = node_name
        self._queues = {}
        self._timestamps = {}
        self._imageio_writers = {}
        self._threads = {}
        self._open = True

    @property
    def is_open(self) -> bool:
        return self._open

    def _ensure_topic(self, topic: str) -> None:
        """Lazily create the encoder thread for a topic on first write."""
        if topic in self._queues:
            return
        q: queue.Queue = queue.Queue()
        self._queues[topic] = q
        self._timestamps[topic] = []
        self._imageio_writers[topic] = None
        t = threading.Thread(
            target=self._encode_loop,
            args=(topic, q),
            daemon=True,
            name=f"Mp4Enc-{self._node_name}-{topic}",
        )
        self._threads[topic] = t
        t.start()

    def write(self, topic: str, timestamp: float, data: dict) -> None:
        if not self._open:
            return
        # CameraNode publishes ``{"images": {topic: frame}, "timestamp": ts, ...}``
        # so the frame lives under data["images"][topic]. Fall back to a flat
        # ``data["frame"]`` for callers that pre-unwrap, and to data[topic]
        # for any other shapes someone might wire up in the future.
        # NB: ndarrays raise on bool() truthiness checks, so use explicit
        # ``is None`` chains rather than ``a or b`` short-circuit.
        frame = None
        images = data.get("images")
        if isinstance(images, dict):
            frame = images.get(topic)
            if frame is None:
                frame = images.get("rgb")
        if frame is None:
            frame = data.get("frame")
        if frame is None:
            cand = data.get(topic)
            if cand is not None and hasattr(cand, "ndim"):
                frame = cand
        if frame is None:
            return
        self._ensure_topic(topic)
        self._queues[topic].put((frame, timestamp))

    def close(self) -> str:
        if not self._open:
            return str(self._save_dir or "")
        self._open = False
        # Send sentinel to all encoder threads
        for topic, q in self._queues.items():
            q.put(_MP4_SENTINEL)
        # Wait for all threads (bounded — encoder threads are daemon so process can exit anyway)
        for topic, t in self._threads.items():
            t.join(timeout=10.0)
        # Save timestamp sidecars
        import numpy as np
        for topic, ts_list in self._timestamps.items():
            ts_path = self._save_dir / f"{self._node_name}-{topic}-timestamp.npy"
            np.save(str(ts_path), np.array(ts_list, dtype=np.float64))
        return str(self._save_dir)

    def _encode_loop(self, topic: str, q: queue.Queue) -> None:
        import imageio
        import os
        writer = None
        mp4_path = self._save_dir / f"{self._node_name}-images-{topic}.mp4"
        ts_list = self._timestamps[topic]

        while True:
            item = q.get()
            if item is _MP4_SENTINEL:
                break
            frame, ts = item
            if writer is None:
                h, w = frame.shape[:2]
                cpu_count = os.cpu_count() or 1
                writer = imageio.get_writer(
                    str(mp4_path),
                    format="FFMPEG",
                    mode="I",
                    fps=self._fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    output_params=[
                        "-profile:v", "baseline",
                        "-level", "3.0",
                        "-preset", "fast",
                        "-crf", str(self._crf),
                        "-threads", str(max(cpu_count // 5, 1)),
                    ],
                )
            writer.append_data(frame)
            ts_list.append(ts)

        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass


# ── NullWriter ────────────────────────────────────────────────────────────────


class NullWriter(Writer):
    """No-op writer — for nodes that manage their own multi-file writers."""

    def __init__(self) -> None:
        self._open: bool = False

    def open(self, save_dir: str, node_name: str) -> None:
        self._open = True

    def write(self, topic: str, timestamp: float, data: dict) -> None:
        pass

    def close(self) -> str:
        self._open = False
        return ""

    @property
    def is_open(self) -> bool:
        return self._open
