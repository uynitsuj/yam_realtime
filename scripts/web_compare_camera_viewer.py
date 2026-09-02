"""Web viewer: live camera feed side-by-side with recorded episode videos.

Serves a browser page with a live session-bus camera view on the left and the
matching dataset video (with native scrubbing controls) on the right. Browse
episodes with prev/next buttons, the dropdown, A/D or arrow keys, or the mouse
wheel over the video pane.

Usage:
    uv run python scripts/web_compare_camera_viewer.py \
        --dataset-path /path/to/dataset \
        --live-source bus \
        --port 8090

Then open http://<this-machine>:8090

By default the live image is subscribed from ``camera_<view>/rgb`` on the
robots_realtime ZMQ bus, so this viewer can run beside an active robot session
without reopening its RealSense device. ``--live-source camera`` retains direct
camera capture for use when no session is running.
Recorded videos keep the full camera FOV on disk, so the live pane defaults to
the raw feed; the FOV slider center-crops BOTH panes equally to preview the
policy's cropped view (e.g. 0.88 for the j4_zoom configs).
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RECORDINGS_DIR = "/nfs_us_2/karim/warp/rr_recordings/original_j4stiff_zoomed_oldlighting"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Web viewer: live camera vs recorded episode videos")
    p.add_argument(
        "--dataset-path", "--recordings-dir", dest="dataset_path",
        default=DEFAULT_RECORDINGS_DIR,
        help="Dataset root (robots_realtime recordings or a LeRobot video dataset)",
    )
    p.add_argument("--view", default="top", choices=["top", "left", "right"], help="Camera view to compare")
    p.add_argument("--video-glob", default=None, help="Override dataset MP4 discovery with a glob relative to the dataset root")
    p.add_argument("--live-source", choices=["bus", "camera", "none"], default="bus", help="Read live frames from the running session bus (default), directly from a camera, or disable live frames")
    p.add_argument("--bus-host", default="127.0.0.1", help="robots_realtime message-bus host")
    p.add_argument("--bus-port", type=int, default=5556, help="robots_realtime subscriber port")
    p.add_argument("--live-topic", default=None, help="Bus topic (default: camera_<view>/rgb)")
    p.add_argument("--camera-serial", default=None, help="RealSense serial for the live camera (default: auto-detect from configs/)")
    p.add_argument("--no-camera", action="store_true", help="Deprecated alias for --live-source none")
    p.add_argument("--opacity", type=float, default=0.5, help="Initial dataset opacity in overlay mode (0..1)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for the live MJPEG stream")
    return p.parse_args()


# ------------------------------------------------------------------ #
# Episode discovery
# ------------------------------------------------------------------ #


def discover_episodes(root: Path, view: str, video_glob: str | None = None) -> list[dict]:
    """Find this view in robots_realtime recordings or a LeRobot video tree."""
    if video_glob:
        videos = sorted(root.glob(video_glob))
    else:
        needles = (
            f"camera_{view}",
            f"images.{view}",
            f"images.camera_{view}",
            f"images.{view}_camera",
        )
        videos = sorted(
            p for p in root.rglob("*.mp4")
            if any(needle in p.as_posix().lower() for needle in needles)
        )
    return [
        {"label": str(p.relative_to(root)), "path": p}
        for p in videos
    ]


# ------------------------------------------------------------------ #
# Live camera capture
# ------------------------------------------------------------------ #


def connected_realsense_serials() -> set[str]:
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
    except ImportError:
        return set()
    try:
        return {d.get_info(rs.camera_info.serial_number) for d in rs.context().devices}
    except Exception:
        return set()


def autodetect_serial(view: str) -> str | None:
    """Serial of the CameraNode named camera_<view> in any config whose device is attached."""
    connected = connected_realsense_serials()
    if not connected:
        return None
    for path in sorted((REPO_ROOT / "configs").rglob("*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text()) or {}
        except Exception:
            continue
        for node in cfg.get("nodes") or []:
            if not (isinstance(node, dict) and node.get("type") == "CameraNode"):
                continue
            if node.get("name") == f"camera_{view}" and str(node.get("device_id") or "") in connected:
                print(f"Live camera serial auto-detected from {path.relative_to(REPO_ROOT)}: {node['device_id']}")
                return str(node["device_id"])
    return None


class LiveCamera:
    """Background capture thread holding the latest JPEG-encoded frame.

    Reconnects automatically if the camera drops (unplug, device busy)."""

    def __init__(self, serial: str, jpeg_quality: int) -> None:
        self.serial = serial
        self.status = "connecting"
        self._jpeg_quality = jpeg_quality
        self._latest: bytes | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def latest(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._latest, self._seq

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        from robots_realtime.sensors.cameras.realsense_camera import RealSenseCamera  # noqa: PLC0415

        while self._running:
            cam = None
            try:
                cam = RealSenseCamera(device_id=self.serial)
                self.status = "live"
                while self._running:
                    data = cam.read()
                    bgr = cv2.cvtColor(data.images["rgb"], cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
                    if ok:
                        with self._lock:
                            self._latest = buf.tobytes()
                            self._seq += 1
            except Exception as exc:
                self.status = f"error: {exc}"
                print(f"Live camera error (will retry): {exc}", file=sys.stderr)
                time.sleep(2.0)
            finally:
                if cam is not None:
                    try:
                        cam.stop()
                    except Exception:
                        pass
        self.status = "stopped"


class BusCamera:
    """Latest-frame adapter for a running robots_realtime camera topic."""

    def __init__(self, topic: str, host: str, port: int, jpeg_quality: int) -> None:
        from robots_realtime.runtime.transport.subscriber import Subscriber  # noqa: PLC0415

        self.serial = f"bus:{topic}"
        self.status = "waiting for frames"
        self._topic = topic
        self._jpeg_quality = jpeg_quality
        self._latest: bytes | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._running = True
        self._subscriber = Subscriber(topics=[topic], host=host, port=port)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def latest(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._latest, self._seq

    def stop(self) -> None:
        self._running = False
        self._subscriber.close()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        last_ts: float | None = None
        while self._running:
            data = self._subscriber.get_data(self._topic)
            ts = self._subscriber.get_timestamp(self._topic)
            if data is None or ts == last_ts:
                time.sleep(0.01)
                continue
            last_ts = ts
            try:
                rgb = data["images"]["rgb"]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
                )
                if ok:
                    with self._lock:
                        self._latest = buf.tobytes()
                        self._seq += 1
                    self.status = "live"
            except Exception as exc:
                self.status = f"error: malformed {self._topic}: {exc}"
        self.status = "stopped"


# ------------------------------------------------------------------ #
# Web app
# ------------------------------------------------------------------ #

app = FastAPI()

EPISODES: list[dict] = []
CAMERA: LiveCamera | BusCamera | None = None
VIEW = "top"
INITIAL_OPACITY = 0.5


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/state")
def api_state() -> dict:
    cam = {"status": "disabled", "serial": None}
    if CAMERA is not None:
        cam = {"status": CAMERA.status, "serial": CAMERA.serial}
    return {
        "view": VIEW,
        "camera": cam,
        "opacity": INITIAL_OPACITY,
        "episodes": [{"label": e["label"]} for e in EPISODES],
    }


@app.get("/live.mjpg")
async def live_mjpg() -> StreamingResponse:
    if CAMERA is None:
        raise HTTPException(404, "live camera disabled")

    async def gen():
        last_seq = -1
        while True:
            frame, seq = CAMERA.latest()
            if frame is not None and seq != last_seq:
                last_seq = seq
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            await asyncio.sleep(0.02)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video/{idx}")
def video(idx: int) -> FileResponse:
    if not 0 <= idx < len(EPISODES):
        raise HTTPException(404, f"episode index out of range (0-{len(EPISODES) - 1})")
    # Starlette's FileResponse handles Range requests, so <video> scrubbing works.
    return FileResponse(EPISODES[idx]["path"], media_type="video/mp4")


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>live vs recordings</title>
<style>
  * { box-sizing: border-box; margin: 0; }
  body { background: #101216; color: #d7dae0; font: 14px/1.4 system-ui, sans-serif; height: 100vh;
         display: flex; flex-direction: column; overflow: hidden; }
  header { display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
           padding: 10px 16px; border-bottom: 1px solid #262a33; }
  h1 { font-size: 15px; font-weight: 600; margin-right: 8px; }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px;
         background: #888; vertical-align: baseline; }
  .dot.live { background: #4cd964; } .dot.err { background: #ff5252; }
  .group { display: flex; align-items: center; gap: 6px; }
  button, select { background: #1d2129; color: #d7dae0; border: 1px solid #333947;
                   border-radius: 6px; padding: 5px 10px; font: inherit; cursor: pointer; }
  button:hover { background: #262c38; }
  select { max-width: 340px; }
  input[type=range] { accent-color: #5b9dff; }
  .muted { color: #7c8494; font-size: 12px; }
  main { flex: 1; display: flex; align-items: center; justify-content: center; gap: 14px;
         padding: 14px; min-height: 0; }
  .pane { display: flex; flex-direction: column; gap: 6px; min-width: 0;
          flex: 0 1 min(46vw, calc((100vh - 130px) * 4 / 3)); }
  .pane h2 { font-size: 13px; font-weight: 500; color: #9aa3b2; }
  .crop { position: relative; overflow: hidden; aspect-ratio: 4/3; width: 100%;
          background: #000; border: 1px solid #262a33; border-radius: 8px; }
  .crop img, .crop video { position: absolute; inset: 0; width: 100%; height: 100%;
                           transform: scale(calc(1 / var(--fov, 1))); }
  .crop .ph { position: absolute; inset: 0; display: flex; align-items: center;
              justify-content: center; color: #5c6474; }
  /* Overlay mode: stack the two panes in one grid cell, video blended on top */
  main.overlay { display: grid; place-items: center; }
  main.overlay .pane { grid-area: 1 / 1; width: min(92vw, calc((100vh - 130px) * 4 / 3)); }
  main.overlay .pane h2 { visibility: hidden; }
  main.overlay #pane-video { opacity: var(--blend, 0.5); z-index: 2; }
  main.overlay #pane-video h2 { visibility: visible; }
</style>
</head>
<body>
<header>
  <h1>camera_<span id="view">top</span></h1>
  <span class="group"><span id="cam-dot" class="dot"></span><span id="cam-status" class="muted">connecting…</span></span>
  <span class="group">
    <button id="prev" title="A / ←">◀</button>
    <select id="ep-select"></select>
    <button id="next" title="D / →">▶</button>
    <span id="ep-count" class="muted"></span>
  </span>
  <span class="group">
    <label class="muted" for="fov">FOV crop</label>
    <input id="fov" type="range" min="0.5" max="1" step="0.01" value="1">
    <span id="fov-val" class="muted">1.00</span>
    <button id="fov-policy" title="Match publish_fov_crop of the j4_zoom policy configs">0.88</button>
    <button id="fov-reset">reset</button>
  </span>
  <span class="group">
    <label><input id="overlay" type="checkbox"> overlay</label>
    <input id="blend" type="range" min="0" max="1" step="0.02" value="0.5" disabled>
    <span id="blend-val" class="muted">0.50</span>
  </span>
  <span class="muted">A/D or ←/→: episode · wheel over recording: scroll episodes</span>
</header>
<main id="panes">
  <section class="pane" id="pane-live">
    <h2 id="live-label">Live</h2>
    <div class="crop"><div class="ph">no live camera</div><img id="live" alt=""></div>
  </section>
  <section class="pane" id="pane-video">
    <h2 id="ep-label">Recording</h2>
    <div class="crop"><div class="ph">no episodes</div><video id="vid" controls autoplay muted loop playsinline></video></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
let eps = [], idx = -1;

function load(i) {
  if (!eps.length) return;
  i = Math.max(0, Math.min(eps.length - 1, i));
  if (i === idx) return;
  idx = i;
  $('ep-select').value = i;
  $('ep-count').textContent = (i + 1) + '/' + eps.length;
  $('ep-label').textContent = 'Recording — ' + eps[i].label;
  $('vid').src = '/video/' + i;
  $('vid').play().catch(() => {});
}

function setCamStatus(cam) {
  const dot = $('cam-dot'), txt = $('cam-status');
  dot.className = 'dot' + (cam.status === 'live' ? ' live' : cam.status.startsWith('error') ? ' err' : '');
  txt.textContent = cam.status + (cam.serial ? ' (' + cam.serial + ')' : '');
}

async function pollState(first) {
  try {
    const st = await (await fetch('/api/state')).json();
    setCamStatus(st.camera);
    if (first) {
      $('view').textContent = st.view;
      $('live-label').textContent = 'Live camera_' + st.view;
      eps = st.episodes;
      $('ep-select').innerHTML = eps.map((e, i) => `<option value="${i}">${e.label}</option>`).join('');
      if (st.camera.status !== 'disabled') $('live').src = '/live.mjpg';
      setOpacity(st.opacity);
      load(0);
    }
  } catch (e) { $('cam-status').textContent = 'server unreachable'; }
}

$('prev').onclick = () => load(idx - 1);
$('next').onclick = () => load(idx + 1);
$('ep-select').onchange = e => load(+e.target.value);
$('pane-video').addEventListener('wheel', e => { e.preventDefault(); load(idx + Math.sign(e.deltaY)); }, { passive: false });

document.addEventListener('keydown', e => {
  if (['INPUT', 'SELECT', 'VIDEO'].includes(e.target.tagName)) return;
  if (e.key === 'ArrowRight' || e.key === 'd') { e.preventDefault(); load(idx + 1); }
  if (e.key === 'ArrowLeft'  || e.key === 'a') { e.preventDefault(); load(idx - 1); }
});

function setFov(v) {
  v = Math.max(0.5, Math.min(1, v));
  $('fov').value = v;
  $('fov-val').textContent = (+v).toFixed(2);
  document.documentElement.style.setProperty('--fov', v);
}
$('fov').oninput = e => setFov(+e.target.value);
$('fov-policy').onclick = () => setFov(0.88);
$('fov-reset').onclick = () => setFov(1.0);

$('overlay').onchange = e => {
  $('panes').classList.toggle('overlay', e.target.checked);
  $('blend').disabled = !e.target.checked;
};
function setOpacity(v) {
  $('blend').value = v;
  $('blend-val').textContent = (+v).toFixed(2);
  document.documentElement.style.setProperty('--blend', v);
}
$('blend').oninput = e => setOpacity(+e.target.value);

pollState(true);
setInterval(() => pollState(false), 3000);
</script>
</body>
</html>
"""


def main() -> None:
    global EPISODES, CAMERA, VIEW, INITIAL_OPACITY
    args = parse_args()
    VIEW = args.view
    if not 0.0 <= args.opacity <= 1.0:
        sys.exit("--opacity must be between 0 and 1")
    INITIAL_OPACITY = float(args.opacity)
    live_source = "none" if args.no_camera else args.live_source

    root = Path(args.dataset_path).expanduser()
    if not root.is_dir():
        sys.exit(f"Dataset path not found: {root}")
    EPISODES = discover_episodes(root, args.view, args.video_glob)
    if not EPISODES:
        hint = f" matching --video-glob {args.video_glob!r}" if args.video_glob else ""
        sys.exit(f"No {args.view} camera MP4s{hint} found under {root}")
    print(f"Found {len(EPISODES)} matching videos under {root}:")
    for i, e in enumerate(EPISODES):
        print(f"  [{i}] {e['label']}")

    if live_source == "bus":
        topic = args.live_topic or f"camera_{args.view}/rgb"
        CAMERA = BusCamera(topic, args.bus_host, args.bus_port, args.jpeg_quality)
        print(f"Subscribing to tcp://{args.bus_host}:{args.bus_port} topic {topic}")
    elif live_source == "camera":
        serial = args.camera_serial or autodetect_serial(args.view)
        if serial is None:
            print(
                f"Warning: no connected camera found for view '{args.view}'; "
                "pass --camera-serial or use --live-source none.",
                file=sys.stderr,
            )
        else:
            CAMERA = LiveCamera(serial, args.jpeg_quality)


    print(f"\nOpen http://{socket.gethostname()}:{args.port}  (or http://localhost:{args.port})\n")
    try:
        # timeout_graceful_shutdown: open MJPEG streams never close on their own,
        # so without this Ctrl-C would hang waiting for clients to disconnect.
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=2)
    finally:
        if CAMERA is not None:
            CAMERA.stop()


if __name__ == "__main__":
    main()
