# VR Streaming (Quest headset)

`MujocoSimNode` can stream the live MuJoCo scene to a Meta Quest headset over WebXR. It auto-starts when a Quest is detected — no extra setup beyond USB ADB.

## Setup

1. Enable **Developer Mode** on the Quest (Meta app → headset → Developer Mode).
2. Connect the headset via USB and accept the ADB prompt on the headset.
3. Verify detection:
   ```bash
   adb devices
   # should show something like: 1WMHH123456789  device
   ```
4. Run a sim session as normal:
   ```bash
   uv run rr-session configs/sessions/yam_sim_gello_teleop.yaml
   ```

That's it. When a Quest is detected at startup, the node:
- Exports all MuJoCo body meshes as GLB files
- Sets up ADB reverse port forwarding (`tcp:8012`)
- Opens the Three.js WebXR client in the Quest Browser automatically

Put the headset on and you'll see the live sim scene. Press **Enter VR** to go immersive.

## How it works

Body poses (position + quaternion) are streamed as binary WebSocket frames at 60 Hz. Only transforms are sent after the initial mesh load — no per-frame mesh data. The Three.js client reconstructs the scene from the GLB files served at startup and animates each body from the transform stream.

The viser browser viewer (`http://localhost:8765`) continues to work in parallel on the desktop.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| No auto-launch | Check `adb devices` — headset must show `device`, not `unauthorized` |
| Quest Browser opens but scene is empty | Wait a few seconds for mesh loading; check the headset for a loading indicator |
| Port conflict on 8012 | Set `vr_port: <other>` in the session YAML |
| Want to disable VR entirely | Set `vr_port: null` in the session YAML |
