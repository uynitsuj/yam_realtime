# Deploying an OpenPI policy with ZED cameras (us05 / pi0.5 Siemens industrial packing)

End-to-end path from "I have a checkpoint and some ZED cameras" to a live rollout:

```
ZED cameras --pyzed--> CameraNode (224x224 pad) --ZMQ--> AsyncDiffusionAgent --ws--> openpi serve_policy.py (RTX 5090)
YAM arms   <--CAN----- RobotNode <------------------------ left_pos / right_pos <--------- 30-step action chunks
```

## 0. What was found on us05 (2026-08-25)

| Item | State |
|---|---|
| USB bus | 3x Intel RealSense D405, 4x CAN adapters. **No Stereolabs device (vendor 2b03) enumerated.** |
| PCIe | no GMSL2 capture card. ZED X / ZED X One are GMSL2 and only attach to a Jetson. |
| ZED SDK | installed at `/usr/local/zed` (libs registered in `ldconfig`, tools symlinked in `/usr/local/bin`), **but `drwxrwx--- dguo:dguo`**: your user cannot traverse it, so `import pyzed.sl` fails until `sudo chmod -R o+rX /usr/local/zed`. |
| pyzed | `dependencies/pyzed-5.1-cp311-cp311-linux_x86_64.whl`, wired as the `sensors` extra. Must match the SDK major.minor (`scripts/setup_zed.sh` checks). |
| GPU / CUDA | RTX 5090 32 GB, CUDA 12.8, driver 575. |
| Disk | root filesystem ~100 % full (12 GB free). Watch it when creating venvs. |

So the ZEDs are either USB models (ZED 2i / ZED Mini / ZED 2) that were not plugged in at probe time, or ZED X units on a
Jetson. Both are supported by the driver: `device_id` (serial) for USB, `stream_ip` for a ZED SDK network stream.

Kimmy's lab42 ZED work, for reference: merged PR #5277 "[Eval] ZedX right eye support in eval service", and the
unmerged branches `kimmy/market42-zedx-standalone` / `kimmy/market42-zedx-frontend` (a *Remote ZED X bridge*:
`xdof/camera/zed_bridge_server.py` runs on the Jetson, market42's `nodes/cameras/remote_zedx.py` consumes it). That
bridge is market42-specific; for robots_realtime the plain ZED SDK streaming sender on the Jetson + `stream_ip` here is
the equivalent.

## 1. Environment

```bash
cd ~/robots_realtime
sudo chmod -R o+rX /usr/local/zed          # once; see table above
./scripts/setup_zed.sh                     # SDK/wheel version check, submodules, uv sync --extra sensors --extra realsense
```

## 2. Find the cameras

```bash
uv run scripts/probe_zed_cameras.py --yaml                 # USB ZEDs: lists serials, opens each, saves /tmp/zed_probe/*.jpg
uv run scripts/probe_zed_cameras.py --stream <jetson-ip>:30000 --yaml   # ZED X streamed from a Jetson
```

Look at `/tmp/zed_probe/zed_<serial>_policy224.jpg`: that letterboxed 224x224 image is exactly what the policy will see.
Use it to decide which serial is top / left / right, then paste the printed `CameraNode` entries into
`configs/yam/yam_bimanual_openpi_policy_xdof_hq_us05_zed.yaml` (replace the `ZED_SERIAL_*` placeholders).

## 3. Serve the checkpoint

```bash
cd ~/openpi
uv run scripts/serve_policy.py \
  --config pi05_siemens_industrial_packing_bs128 \
  --checkpoint-dir /nfs_us_2/siemens/policy_ckpts/pi05_siemens_industrial_packing_bs128/siemens_packing_pi05_lerobot_20260825/14999 \
  --default-prompt "<task_name the episodes were recorded with>" \
  --port 8012
```

Add `--smoke-test` to load the checkpoint, run two inferences on a synthetic YAM observation and exit (verifies the
config/param shapes and prints compile + steady-state latency).

## 4. Roll out

```bash
cd ~/robots_realtime
uv run rr-session configs/yam/yam_bimanual_openpi_policy_xdof_hq_us05_zed.yaml
```

The session starts paused (arms ramp to the startup pose and hold). Press `[space]` to hand control to the policy;
recording starts automatically. Viser at <http://localhost:8080> shows the URDFs, the predicted chunk, and the three
224x224 policy inputs.

## 5. Things that must match training (and why the config is set the way it is)

The checkpoint was trained on a LeRobot dataset produced by lab42's converter
(`xdof/learning/datasets/lerobot/default_configs.py`, `scripts/convert_yam_data.py`):

| Training fact | Deployment setting |
|---|---|
| Images: `resize_and_pad` to 224x224 (letterbox, full FOV, no centre crop) | `publish_resize_mode: pad`, `image_preprocess: pad`, `publish_fov_crop: 1.0` |
| Camera keys `left_camera-images-rgb`, `right_camera-images-rgb`, `top_camera-images-rgb`; top = ZED **left eye** | `ZedCamera.image_key: rgb` so the bus key flattens to `<cam>-images-rgb`; driver publishes the left eye |
| 30 fps, state = 14 absolute joints (6+gripper per arm), actions absolute joints | `poll_freq: 30`, `use_joint_state_as_action: false` |
| Prompt = each episode's `metadata.task_name` (`prompt_from_task=True`) | `--default-prompt` on the server. Not recoverable from the checkpoint; take it from the Siemens job's task name in DataEngine. |
| Model: pi0.5, `action_dim=32`, norm stats 32-dim quantile | `Pi0Config(pi05=True)` + `LeRobotYamDataConfig(repo_id="industrial_packing_yam")` |
| `action_horizon` | **Assumed 30** (matches every other pi05 YAM config in this fork; the training launcher config was not found in any local openpi checkout or branch). If the run used another value, pass `--action-horizon N` when serving. The wandb run id is `yw7woibo` (`wandb_id.txt` next to the checkpoints) if you want to confirm. |

## 6. Troubleshooting

* `ImportError: libsl_zed.so: cannot open shared object file` -> `/usr/local/zed` permissions (step 1).
* `Camera Open (...): CAMERA_NOT_DETECTED` -> `lsusb | grep 2b03` shows nothing: USB 3 port / cable, or the camera is a
  ZED X on a Jetson (use `stream_ip`).
* `... is all black` at startup -> lens cap, or a ZED still ramping exposure; set `check_black_frames: false` to debug.
* Three USB ZEDs at HD1080 saturate one USB 3 root hub; use HD720 (the config default) or spread the cameras across
  controllers.
* `[AsyncDiffusionAgent] obs not ready -- waiting on: top_camera-images-rgb` -> a camera node is publishing under
  `left_rgb` (missing `image_key: rgb`) or has died; check the TUI node table.
