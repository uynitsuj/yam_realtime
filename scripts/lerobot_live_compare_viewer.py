#!/usr/bin/env python3
"""Compare all live camera views with episodes in a LeRobot v3 dataset.

Each camera gets a live-vs-dataset row. The selected episode is sought and
looped using its exact timestamps inside LeRobot's chunk MP4 files.

Usage:
    uv run python scripts/lerobot_live_compare_viewer.py
    uv run python scripts/lerobot_live_compare_viewer.py \
        --dataset-path /path/to/dataset --opacity 0.5 --port 8090
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import pyarrow as pa
import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from robots_realtime.runtime.transport.subscriber import Subscriber

DEFAULT_DATASET = Path("/nfs_us_2/siemens/datasets/siemens_simple_d405_v2")
_VIEW_ORDER = {"top": 0, "left": 1, "right": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bus-host", default="127.0.0.1")
    parser.add_argument("--bus-port", type=int, default=5556)
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--opacity", type=float, default=0.5)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    return parser.parse_args()


def view_name(video_key: str) -> str:
    name = video_key.removesuffix("-images-rgb")
    return name.removesuffix("_camera").removesuffix("-camera")


def load_dataset(root: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"LeRobot metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected LeRobot v3.0, got {info.get('codebase_version')!r}"
        )

    keys = [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]
    views = {view_name(key): key for key in keys}
    views = dict(
        sorted(views.items(), key=lambda item: (_VIEW_ORDER.get(item[0], 99), item[0]))
    )
    if not views:
        raise ValueError("Dataset has no video features")

    metadata_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not metadata_files:
        raise ValueError("Dataset has no episode metadata parquet files")
    rows = pa.concat_tables([pq.read_table(path) for path in metadata_files]).to_pylist()
    template = info["video_path"]

    episodes: list[dict[str, Any]] = []
    for row in rows:
        index = int(row["episode_index"])
        tasks = row.get("tasks") or []
        episode: dict[str, Any] = {
            "index": index,
            "label": f"episode {index:04d}" + (f" — {tasks[0]}" if tasks else ""),
            "views": {},
        }
        for view, key in views.items():
            base = f"videos/{key}"
            relative = template.format(
                video_key=key,
                chunk_index=int(row[f"{base}/chunk_index"]),
                file_index=int(row[f"{base}/file_index"]),
            )
            path = root / relative
            if not path.is_file():
                raise ValueError(f"Missing episode video: {path}")
            episode["views"][view] = {
                "path": path,
                "start": float(row[f"{base}/from_timestamp"]),
                "end": float(row[f"{base}/to_timestamp"]),
            }
        episodes.append(episode)
    return views, episodes


class BusCamera:
    def __init__(
        self, topic: str, host: str, port: int, jpeg_quality: int
    ) -> None:
        self.topic = topic
        self.status = "waiting for frames"
        self._quality = jpeg_quality
        self._subscriber = Subscriber([topic], host=host, port=port)
        self._latest: bytes | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def latest(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._latest, self._seq

    def _run(self) -> None:
        last_timestamp: float | None = None
        while self._running:
            data = self._subscriber.get_data(self.topic)
            timestamp = self._subscriber.get_timestamp(self.topic)
            if data is None or timestamp == last_timestamp:
                time.sleep(0.01)
                continue
            last_timestamp = timestamp
            try:
                rgb = data["images"]["rgb"]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, encoded = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
                )
                if ok:
                    with self._lock:
                        self._latest = encoded.tobytes()
                        self._seq += 1
                    self.status = "live"
            except Exception as exc:
                self.status = f"error: {exc}"
        self.status = "stopped"

    def stop(self) -> None:
        self._running = False
        self._subscriber.close()
        self._thread.join(timeout=1.0)


app = FastAPI()
DATASET_ROOT = DEFAULT_DATASET
VIEWS: dict[str, str] = {}
EPISODES: list[dict[str, Any]] = []
CAMERAS: dict[str, BusCamera] = {}
INITIAL_OPACITY = 0.5


def get_episode(position: int) -> dict[str, Any]:
    if not 0 <= position < len(EPISODES):
        raise HTTPException(404, f"episode position out of range: {position}")
    return EPISODES[position]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/state")
def api_state() -> dict[str, Any]:
    return {
        "dataset": str(DATASET_ROOT),
        "views": list(VIEWS),
        "cameras": {
            view: {"status": camera.status, "topic": camera.topic}
            for view, camera in CAMERAS.items()
        },
        "opacity": INITIAL_OPACITY,
        "episodes": [
            {"index": episode["index"], "label": episode["label"]}
            for episode in EPISODES
        ],
    }


@app.get("/api/episode/{position}")
def api_episode(position: int) -> dict[str, Any]:
    episode = get_episode(position)
    return {
        "index": episode["index"],
        "label": episode["label"],
        "views": {
            view: {"start": segment["start"], "end": segment["end"]}
            for view, segment in episode["views"].items()
        },
    }


@app.get("/video/{view}/{position}")
def video(view: str, position: int) -> FileResponse:
    segment = get_episode(position)["views"].get(view)
    if segment is None:
        raise HTTPException(404, f"unknown view: {view}")
    return FileResponse(segment["path"], media_type="video/mp4")


@app.get("/live/{view}.mjpg")
async def live(view: str) -> StreamingResponse:
    camera = CAMERAS.get(view)
    if camera is None:
        raise HTTPException(404, f"live camera disabled: {view}")

    async def frames():
        last_seq = -1
        while True:
            frame, seq = camera.latest()
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

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Live vs LeRobot</title>
<style>
*{box-sizing:border-box}body{height:100vh;overflow:hidden;margin:0;background:#101216;color:#d7dae0;
font:13px system-ui;display:flex;flex-direction:column}
header{flex:none;z-index:10;display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:6px 10px;
background:#101216;border-bottom:1px solid #292d36}
button,select{background:#1d2129;color:#d7dae0;border:1px solid #3a404d;border-radius:6px;padding:4px 8px}
select{width:min(380px,34vw)}input[type=range]{accent-color:#5b9dff}.muted{color:#8b93a3;font-size:11px}
#scrub{flex:1;min-width:180px}#rows{flex:1;min-height:0;display:grid;grid-template-rows:repeat(3,minmax(0,1fr));
gap:5px;padding:5px}.compare{min-height:0;display:grid;grid-template-columns:1fr 1fr;
grid-template-rows:auto minmax(0,1fr);gap:4px 8px}
.compare h2{grid-column:1/-1;margin:0;font-size:12px;line-height:14px}.pane{min-height:0;position:relative;
background:#000;border:1px solid #292d36;border-radius:6px;overflow:hidden}
.pane img,.pane video{width:100%;height:100%;object-fit:contain;display:block}
.tag{position:absolute;top:4px;left:4px;z-index:3;padding:1px 5px;background:#000a;border-radius:4px;font-size:11px}
.compare.overlay{grid-template-columns:minmax(0,1fr)}.compare.overlay .pane{grid-area:2/1}
.compare.overlay .recorded{opacity:var(--opacity,.5);z-index:2;pointer-events:none}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#777;margin-right:4px}
.dot.live{background:#4cd964}.dot.error{background:#ff5252}
</style></head><body>
<header><strong>Live ↔ LeRobot</strong><button id="prev">◀</button><select id="episodes"></select>
<button id="next">▶</button><span id="count" class="muted"></span><button id="play">❚❚</button>
<input id="scrub" type="range" min="0" max="1" step=".01" value="0">
<span id="time" class="muted">0:00.0 / 0:00.0</span>
<label><input id="overlay" type="checkbox"> overlay</label><span class="muted">opacity</span>
<input id="opacity" type="range" min="0" max="1" step=".01"><span id="opacity-value" class="muted"></span>
<span id="dataset" class="muted"></span></header><main id="rows"></main>
<script>
const byId=id=>document.getElementById(id);
let state,position=-1,token=0,episode=null,episodeDuration=0,playing=true,scrubbing=false;
function opacity(value){value=Math.max(0,Math.min(1,Number(value)));byId('opacity').value=value;
byId('opacity-value').textContent=value.toFixed(2);document.documentElement.style.setProperty('--opacity',value)}
function formatTime(value){value=Math.max(0,Number(value)||0);const minutes=Math.floor(value/60);
return minutes+':'+(value-minutes*60).toFixed(1).padStart(4,'0')}
function showTime(value){byId('time').textContent=formatTime(value)+' / '+formatTime(episodeDuration)}
function videos(){return state.views.map(view=>byId('video-'+view))}
function setPlaying(value){playing=value;byId('play').textContent=playing?'❚❚':'▶';
for(const video of videos()){if(playing)video.play().catch(()=>{});else video.pause()}}
function seek(value){if(!episode)return;value=Math.max(0,Math.min(episodeDuration,Number(value)));
for(const view of state.views){const video=byId('video-'+view),segment=episode.views[view];
video.currentTime=segment.start+value}byId('scrub').value=value;showTime(value)}
function updateFromPrimary(){if(!episode||scrubbing)return;const primary=state.views[0];
const video=byId('video-'+primary),segment=episode.views[primary];
let relative=video.currentTime-segment.start;if(relative>=episodeDuration-.02){seek(0);if(playing)setPlaying(true);return}
relative=Math.max(0,relative);byId('scrub').value=relative;showTime(relative);
for(const view of state.views.slice(1)){const other=byId('video-'+view),otherSegment=episode.views[view];
const otherRelative=other.currentTime-otherSegment.start;if(Math.abs(otherRelative-relative)>.12)other.currentTime=otherSegment.start+relative}}
function build(){byId('rows').innerHTML=state.views.map(view=>{const camera=state.cameras[view]||{status:'disabled',topic:''};
const klass=camera.status==='live'?'live':camera.status.startsWith('error')?'error':'';
const image=state.cameras[view]?'<img src="/live/'+view+'.mjpg">':'';
return '<section class="compare" id="compare-'+view+'"><h2>'+view+' <span class="muted"><span id="dot-'+view+
'" class="dot '+klass+'"></span><span id="status-'+view+'">'+camera.status+' '+camera.topic+
'</span></span></h2><div class="pane live"><span class="tag">live</span>'+image+
'</div><div class="pane recorded"><span class="tag">dataset</span><video id="video-'+view+
'" muted playsinline preload="metadata"></video></div></section>'}).join('');
for(const video of videos())video.onclick=()=>setPlaying(!playing)}
async function load(next){next=Math.max(0,Math.min(state.episodes.length-1,next));const mine=++token;
const loaded=await(await fetch('/api/episode/'+next)).json();if(mine!==token)return;episode=loaded;position=next;
episodeDuration=Math.min(...state.views.map(view=>loaded.views[view].end-loaded.views[view].start));
byId('scrub').max=episodeDuration;byId('scrub').value=0;showTime(0);
byId('episodes').value=position;byId('count').textContent=(position+1)+'/'+state.episodes.length;
let pending=state.views.length;for(const view of state.views){const video=byId('video-'+view),segment=loaded.views[view];
video.src='/video/'+view+'/'+position;video.onloadedmetadata=()=>{video.currentTime=segment.start;
if(--pending===0)setPlaying(true)};video.ontimeupdate=view===state.views[0]?updateFromPrimary:null}}
async function refresh(){const fresh=await(await fetch('/api/state')).json();for(const view of state.views){
const camera=fresh.cameras[view]||{status:'disabled',topic:''},dot=byId('dot-'+view);
dot.className='dot '+(camera.status==='live'?'live':camera.status.startsWith('error')?'error':'');
byId('status-'+view).textContent=camera.status+' '+camera.topic}}
(async()=>{state=await(await fetch('/api/state')).json();byId('dataset').textContent=state.dataset;
byId('episodes').innerHTML=state.episodes.map((ep,i)=>'<option value="'+i+'">'+ep.label+'</option>').join('');
opacity(state.opacity);build();await load(0);setInterval(refresh,3000)})();
byId('prev').onclick=()=>load(position-1);byId('next').onclick=()=>load(position+1);
byId('play').onclick=()=>setPlaying(!playing);byId('episodes').onchange=e=>load(Number(e.target.value));
byId('opacity').oninput=e=>opacity(e.target.value);byId('scrub').oninput=e=>seek(e.target.value);
byId('scrub').onpointerdown=()=>{scrubbing=true};byId('scrub').onchange=()=>{scrubbing=false};
byId('overlay').onchange=e=>document.querySelectorAll('.compare').forEach(row=>row.classList.toggle('overlay',e.target.checked));
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','VIDEO'].includes(e.target.tagName))return;
if(e.key==='ArrowRight'||e.key==='d')load(position+1);if(e.key==='ArrowLeft'||e.key==='a')load(position-1);
if(e.key===' '){e.preventDefault();setPlaying(!playing)}});
</script></body></html>"""


def main() -> None:
    global DATASET_ROOT, VIEWS, EPISODES, INITIAL_OPACITY

    args = parse_args()
    DATASET_ROOT = args.dataset_path.expanduser().resolve()
    if not 0.0 <= args.opacity <= 1.0:
        sys.exit("--opacity must be between 0 and 1")
    INITIAL_OPACITY = args.opacity
    try:
        VIEWS, EPISODES = load_dataset(DATASET_ROOT)
    except ValueError as exc:
        sys.exit(str(exc))

    print(
        f"Loaded {len(EPISODES)} episodes and views "
        f"{', '.join(VIEWS)} from {DATASET_ROOT}"
    )
    if not args.no_live:
        for view in VIEWS:
            topic = f"camera_{view}/rgb"
            CAMERAS[view] = BusCamera(
                topic, args.bus_host, args.bus_port, args.jpeg_quality
            )
        print(f"Live bus: tcp://{args.bus_host}:{args.bus_port}")

    print(
        f"Open http://{socket.gethostname()}:{args.port} "
        f"(or http://localhost:{args.port})"
    )
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="warning",
            timeout_graceful_shutdown=2,
        )
    finally:
        for camera in CAMERAS.values():
            camera.stop()


if __name__ == "__main__":
    main()
