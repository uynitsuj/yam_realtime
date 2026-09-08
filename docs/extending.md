# Extending robots_realtime

## Adding a new agent

Implement the `Agent` protocol — just `act(obs: dict) -> dict`:

```python
# robots_realtime/agents/my_agent.py
class MyAgent:
    def reset(self) -> None: ...
    def act(self, obs: dict) -> dict:
        # obs contains whatever state_topics / image_topics you subscribed to
        return {"pos": joint_positions}          # single arm
        # or {"left": {"pos": ...}, "right": {"pos": ...}}  # multi-arm
```

Reference it in YAML — no node code needed:

```yaml
- type: AgentNode
  name: my_agent
  agent_class: robots_realtime.agents.my_agent:MyAgent
  agent_kwargs:
    checkpoint: /path/to/weights.pt
  loop_mode: subscriber_driven
  state_topics:
    left: yam_left/joint_state
    right: yam_right/joint_state
  image_topics:
    top: camera_top/rgb
```

## Adding a new robot

Implement two methods:

```python
class MyRobot:
    def command_joint_pos(self, joint_pos: np.ndarray) -> None: ...
    def get_observations(self) -> dict: ...  # must contain "joint_pos"
```

## Adding a new camera

Implement `read() -> CameraData` from `robots_realtime.sensors.cameras.camera`.

Built-in drivers (`driver:` key on a `CameraNode`): `RealSenseCamera`, `ZedCamera`,
`OpenCVCamera`. Any other class can be referenced as `module.path:ClassName`.

### Plain USB (UVC) cameras with `OpenCVCamera`

Webcam-class cameras (decxin/YHTek modules, Logitech, ...) need no SDK — the
kernel exposes them as `/dev/videoN` and `OpenCVCamera` reads them through
OpenCV's V4L2 backend. Bring-up:

```bash
# 1. Find the cameras and their stable by-path aliases (never opens a device)
uv run python -m robots_realtime.sensors.cameras.opencv_camera --list
# 2. Check what each one can actually do: pixel format × size × fps
v4l2-ctl -d /dev/video0 --list-formats-ext
# 3. Stream one and check rate / latency / timestamps, paced like a 30 Hz node
uv run python -m robots_realtime.sensors.cameras.opencv_camera \
    --device_path /dev/v4l/by-path/pci-0000:0b:00.3-usb-0:1:1.0-video-index0 --poll_hz 30
```

```yaml
  - type: CameraNode
    name: camera_top
    driver: OpenCVCamera
    device_path: /dev/v4l/by-path/pci-0000:08:00.0-usb-0:6:1.0-video-index0  # not /dev/videoN
    resolution: [640, 480]     # or "WxH" / VGA / HD720
    fps: 30
    fourcc: MJPG               # MJPG for 30-60 fps at VGA+ over USB 2; YUYV = uncompressed
    poll_freq: 30              # pace the node if the device won't run at the requested fps
    # auto_exposure: false / manual_exposure: 166 / manual_gain: 64 / manual_white_balance_k: 4600
```

Notes:

- Use `/dev/v4l/by-path/...-video-index0` paths. `/dev/videoN` numbering changes
  across reboots, and cheap modules often share one USB serial so `by-id` is
  ambiguous. Each UVC camera has two nodes; only `index0` delivers frames.
- Put each camera on its own USB host controller when possible so they don't
  compete for bandwidth (`--list` prints the controller in the by-path name).
- UVC cameras only honour frame rates they advertise for the chosen
  `fourcc` × size. The driver logs the real rate; if it's above what you want,
  set `poll_freq` — the driver's background grab thread makes `read()` return
  the next frame captured after the call, so the node stays at `poll_freq`
  with no stale-queue latency.
- Timestamps are the V4L2 kernel capture stamps mapped onto the wall clock, so
  they line up with RealSense/ZED hardware stamps and joint-state stamps.

## Session config reference

```yaml
version: "1"

session:
  save_root: recordings
  record_topic: gello_left/record   # bus topic that triggers record start/stop
  auto_record_duration: 10.0        # auto-record for N seconds then exit

nodes:
  - type: AgentNode
    name: gello_left
    agent_class: robots_realtime.agents.teleoperation.gello_leader_agent:GelloLeaderAgent
    agent_kwargs:
      port: /dev/ttyUSB0
      robot_name: left
    arm_key: left
    loop_mode: flat_out

  - type: RobotNode
    name: yam_left
    robot_config: robot_configs/yam/left.yaml
    cmd_topic: gello_left/joint_pos

  - type: CameraNode
    name: camera_top
    fps: 30
```
