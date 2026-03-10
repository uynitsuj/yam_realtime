"""Trajectory logger for robot teleoperation episodes.

Works with both sim and real-world control loops — it only cares about the
flat ``{key: np.ndarray}`` dict passed to :meth:`TrajectoryLogger.log_step`.
The caller decides what to include; the logger decides how to persist it.

**Streaming I/O** — no growing in-memory buffer:

* Numeric arrays → raw float32 bytes appended to a ``.bin`` file each step;
  converted to ``.npy`` at episode end (one small final write).
* Image arrays (uint8, HxWx3) → frames written immediately to MP4 via
  ``cv2.VideoWriter``; the file is a valid MP4 throughout recording.

Saved layout::

    <save_path>/
      YYYYMMDD/
        episode_YYYYMMDD_HHMMSS_<uid>/
          state.npy        — (N, D) float32
          action.npy       — (N, D) float32
          timestamp.npy    — (N, 1) float64
          <key>.mp4        — one MP4 per image key
          <key>.npy        — one npy per extra numeric key

**Trigger mechanisms** — the logger itself only exposes ``start_episode()``
and ``end_episode()``.  How recording is triggered is a separate concern:

* ``attach_keyboard_listener(logger)`` — stdin commands (interactive terminal)
* ``attach_file_watcher(logger, flag_file)`` — create the file to start,
  delete it to stop; works headless / from any IDE terminal or script
* ``attach_signal_handler(logger)`` — SIGUSR1 toggles recording
* Direct calls to ``logger.start_episode()`` / ``logger.end_episode()``
  from any external code, thread, or API

Example — headless usage::

    logger = TrajectoryLogger("/data/recordings", fps=30.0)
    attach_file_watcher(logger, "/tmp/record.flag")
    # now: touch /tmp/record.flag  → start, rm /tmp/record.flag → stop

Example — programmatic control::

    logger = TrajectoryLogger("/data/recordings", fps=30.0)
    logger.start_episode()
    for obs, action in loop:
        logger.log_step({"state": obs, "action": action})
    logger.end_episode()
    logger.close()
"""

import contextlib
import hashlib
import logging
import os
import signal
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

_log = logging.getLogger(__name__)


@contextlib.contextmanager
def _silence_c_stderr():
    """Temporarily redirect C-level stderr to /dev/null.

    Used to suppress OpenCV's FFMPEG codec-probe error messages, which are
    printed directly to the C stderr fd and cannot be silenced via Python's
    sys.stderr or logging filters.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull_fd)


# ---------------------------------------------------------------------------
# Streaming primitives
# ---------------------------------------------------------------------------


class _NumericStream:
    """Appends float32 rows to a ``.bin`` scratch file; finalises to ``.npy``."""

    def __init__(self, path: Path) -> None:
        self._npy_path = path.with_suffix(".npy")
        self._bin_path = path.with_suffix(".bin")
        self._f = open(self._bin_path, "wb")
        self._rows = 0
        self._cols: Optional[int] = None

    def write(self, arr: np.ndarray) -> None:
        flat = np.asarray(arr, dtype=np.float32).ravel()
        if self._cols is None:
            self._cols = flat.size
        self._f.write(flat.tobytes())
        self._rows += 1

    def close(self) -> None:
        self._f.close()
        if self._rows > 0 and self._cols:
            raw = np.frombuffer(self._bin_path.read_bytes(), dtype=np.float32)
            np.save(self._npy_path, raw.reshape(self._rows, self._cols))
        self._bin_path.unlink(missing_ok=True)


# class _VideoStream:
#     """Streams uint8 RGB frames to an MP4 via cv2.VideoWriter."""

#     def __init__(self, path: Path, fps: float, h: int, w: int) -> None:
#         import cv2
#         # Prefer H.264 (avc1) for broad platform compatibility; fall back to mp4v.
#         # Codec probing is done with C-level stderr suppressed because OpenCV
#         # prints FFMPEG errors even for expected probe failures.
#         self._writer = None
#         for fourcc_str in ("avc1", "mp4v"):
#             fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
#             with _silence_c_stderr():
#                 writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
#             if writer.isOpened():
#                 self._writer = writer
#                 break
#         if self._writer is None:
#             raise RuntimeError(
#                 f"cv2.VideoWriter could not open {path} with avc1 or mp4v "
#                 f"(fps={fps}, size={w}x{h}). "
#                 "Check that opencv-python is built with video support and the path is writable."
#             )

#     def write(self, frame: np.ndarray) -> None:
#         import cv2
#         self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

#     def close(self) -> None:
#         self._writer.release()


class _VideoStream:
    """Streams uint8 RGB frames using imageio-ffmpeg for better UI compatibility."""

    def __init__(self, path: Path, fps: float, h: int, w: int) -> None:
        import imageio

        # 'libx264' is the gold standard for H.264
        # 'pix_fmt="yuv420p"' for compatibility
        self._writer = imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=8,  # Helps with odd-dimension resolutions
        )

    def write(self, frame: np.ndarray) -> None:
        # imageio expects RGB, so no need for cvtColor(BGR)!
        self._writer.append_data(frame)

    def close(self) -> None:
        self._writer.close()


def _is_image(arr: np.ndarray) -> bool:
    return arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] == 3


def _short_uid(n: int = 8) -> str:
    return hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:n]


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class TrajectoryLogger:
    """Records teleoperation steps and streams them directly to disk.

    The logger is trigger-agnostic: call ``start_episode()`` / ``end_episode()``
    directly, or attach one of the standalone trigger helpers below.

    ``log_step`` is a no-op while not recording, so it is safe to call every
    control tick unconditionally.
    """

    def __init__(self, save_path: str, fps: float = 30.0) -> None:
        self.save_path = Path(save_path)
        self.fps = fps
        self.save_path.mkdir(parents=True, exist_ok=True)

        self._recording = False
        self._ep_dir: Optional[Path] = None
        self._streams: Dict[str, Union[_NumericStream, _VideoStream]] = {}
        self._n_steps = 0
        self._finalise_threads: List[threading.Thread] = []

    @property
    def recording(self) -> bool:
        return self._recording

    def start_episode(self) -> None:
        """Open a new episode directory and begin streaming to disk."""
        if self._recording:
            _log.warning("start_episode() called while already recording — ignored.")
            return
        now = datetime.now()
        self._ep_dir = (
            self.save_path / now.strftime("%Y%m%d") / f"episode_{now.strftime('%Y%m%d_%H%M%S')}_{_short_uid()}"
        )
        self._ep_dir.mkdir(parents=True, exist_ok=True)
        self._streams = {}
        self._n_steps = 0
        self._recording = True
        _log.info("Recording started → %s", self._ep_dir)
        print(f"\n[logger] Recording started → {self._ep_dir}\n")

    def log_step(self, data: Dict[str, np.ndarray]) -> None:
        """Stream one control step to disk.  No-op when not recording."""
        if not self._recording or self._ep_dir is None:
            return
        for key, arr in data.items():
            arr = np.asarray(arr)
            if key not in self._streams:
                self._streams[key] = self._open_stream(key, arr)
            self._streams[key].write(arr)
        self._n_steps += 1

    def end_episode(self, save: bool = True) -> None:
        """Stop recording and finalise writers in a background thread."""
        if not self._recording:
            return
        self._recording = False
        n, ep_dir, streams = self._n_steps, self._ep_dir, self._streams
        self._streams = {}
        self._ep_dir = None

        if not save or n == 0 or ep_dir is None:
            for s in streams.values():
                try:
                    s.close()
                except Exception:
                    pass
            print(f"\n[logger] Episode discarded ({n} steps).\n")
            return

        print(f"\n[logger] Stopping — {n} steps. Finalising in background …\n")
        t = threading.Thread(
            target=self._finalise_worker,
            args=(streams, ep_dir, n),
            daemon=True,
        )
        t.start()
        self._finalise_threads.append(t)

    def close(self, timeout: float = 120.0) -> None:
        """Flush any in-progress episode and wait for all background saves."""
        if self._recording:
            self.end_episode(save=True)
        for t in self._finalise_threads:
            t.join(timeout=timeout)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _open_stream(self, key: str, arr: np.ndarray) -> Union[_NumericStream, _VideoStream]:
        assert self._ep_dir is not None
        if _is_image(arr):
            h, w = arr.shape[:2]
            return _VideoStream(self._ep_dir / f"{key}.mp4", self.fps, h, w)
        return _NumericStream(self._ep_dir / key)

    @staticmethod
    def _finalise_worker(
        streams: Dict[str, Union[_NumericStream, _VideoStream]],
        ep_dir: Path,
        n_steps: int,
    ) -> None:
        try:
            for stream in streams.values():
                stream.close()
            _log.info("Saved: %s  (%d steps)", ep_dir, n_steps)
            print(f"[logger] Saved: {ep_dir}  ({n_steps} steps)\n")
        except Exception:
            _log.exception("Failed to finalise episode %s", ep_dir)


# ---------------------------------------------------------------------------
# Trigger helpers  (all optional — use whichever fits your workflow)
# ---------------------------------------------------------------------------


def attach_keyboard_listener(logger: TrajectoryLogger) -> None:
    """Spawn a daemon thread reading stdin commands.

    Commands (type then press Enter):
      r — start / stop recording
      d — discard current episode without saving
      q — quit (raises KeyboardInterrupt in the main thread)

    Suitable for interactive terminal sessions.
    """

    def _loop() -> None:
        import sys

        print("[logger] Keyboard trigger active —  r=record/stop  d=discard  q=quit")
        while True:
            try:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == "r":
                    if logger.recording:
                        logger.end_episode(save=True)
                    else:
                        logger.start_episode()
                elif cmd == "d":
                    logger.end_episode(save=False)
                elif cmd == "q":
                    print("[logger] Quit requested.")
                    import ctypes

                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(threading.main_thread().ident),
                        ctypes.py_object(KeyboardInterrupt),
                    )
                    break
            except Exception:
                break

    threading.Thread(target=_loop, daemon=True).start()


def attach_file_watcher(
    logger: TrajectoryLogger,
    flag_file: Union[str, Path] = "/tmp/record.flag",
    poll_interval: float = 0.25,
) -> None:
    """Watch a flag file to control recording — no terminal needed.

    * ``touch <flag_file>``  (or create it any way) → start recording
    * ``rm   <flag_file>``                           → stop and save
    * ``echo discard > <flag_file>``                 → stop and discard

    Suitable for headless / IDE workflows where stdin is not accessible.
    The watcher runs as a daemon thread and polls every ``poll_interval``
    seconds (default 250 ms).
    """
    flag = Path(flag_file)

    def _loop() -> None:
        print(f"[logger] File-watcher trigger active — flag: {flag}")
        prev_exists = flag.exists()
        while True:
            try:
                exists = flag.exists()
                if exists and not prev_exists:
                    # File appeared → start
                    logger.start_episode()
                elif not exists and prev_exists:
                    # File removed → stop and save
                    logger.end_episode(save=True)
                elif exists and prev_exists:
                    # File present: check for "discard" content
                    try:
                        content = flag.read_text().strip().lower()
                    except OSError:
                        content = ""
                    if content == "discard" and logger.recording:
                        flag.unlink(missing_ok=True)
                        logger.end_episode(save=False)
                prev_exists = flag.exists()
                time.sleep(poll_interval)
            except Exception:
                _log.exception("File watcher error")
                time.sleep(poll_interval)

    threading.Thread(target=_loop, daemon=True).start()


def attach_signal_handler(logger: TrajectoryLogger) -> None:
    """Toggle recording on SIGUSR1, discard on SIGUSR2.

    Usage from any shell::

        kill -USR1 <pid>   # start or stop recording
        kill -USR2 <pid>   # discard current episode

    The PID is printed to stdout when this function is called.
    Suitable for scripted or CI-driven data collection.
    """
    import os

    def _toggle(_signum, _frame) -> None:
        if logger.recording:
            logger.end_episode(save=True)
        else:
            logger.start_episode()

    def _discard(_signum, _frame) -> None:
        logger.end_episode(save=False)

    signal.signal(signal.SIGUSR1, _toggle)
    signal.signal(signal.SIGUSR2, _discard)
    print(f"[logger] Signal trigger active — PID {os.getpid()}  (SIGUSR1=toggle, SIGUSR2=discard)")
