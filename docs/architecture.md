# Architecture

## Node graph

Every session is a graph of **nodes**, each running in its own subprocess with its own MCAP writer. All nodes communicate over a ZMQ XPUB/XSUB message bus. The bus runs in its own subprocess so its GIL pauses don't affect control latency.

```
                    ┌─────────────────────────┐
                    │     MessageBus (ZMQ)    │
                    │   XPUB/XSUB broker      │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┼─────────────────────┐
          │                      │                     │
   ┌──────▼──────┐       ┌───────▼──────┐      ┌───────▼──────┐
   │  AgentNode  │       │  AgentNode   │      │  RobotNode   │
   │ gello_left  │       │ gello_right  │      │  yam_left    │
   │ (MCAP)      │       │ (MCAP)       │      │  (MCAP)      │
   └─────────────┘       └──────────────┘      └──────────────┘
```

## Node types

**Agent nodes** — produce commands (`joint_pos`) from observations:
- `AgentNode` wrapping `GelloLeaderAgent` — GELLO leader arm
- `AgentNode` wrapping `FrankaPyrokiViserAgent` — browser IK gizmo for Franka
- `AgentNode` wrapping `DiffusionPolicyAgent` / `AsyncPi0Agent` — learned policies
- `AgentNode` wrapping `DummyAgent` — synthetic random targets for testing

**Environment nodes** — consume commands, produce observations:
- `RobotNode` — any robot with `command_joint_pos()` / `get_observations()`
- `CameraNode` — any camera with a `read() -> CameraData` driver
- `XdofSimNode` — bimanual YAM MuJoCo simulation with live Viser viewer and optional Quest VR streaming

## Loop modes

`AgentNode` supports three loop modes:

| `loop_mode`         | Use case |
|---------------------|----------|
| `flat_out`          | Hardware leader arms — paced by serial/CAN I/O |
| `fixed_rate`        | Viser IK solver — runs at a configured Hz |
| `subscriber_driven` | Learned policies — triggered by new observations |

## Recording format

Each node owns its writer. Recording is started and stopped via control signals to each subprocess. Output per episode:

```
recordings/20260321/episode_150034_abc123/
  gello_left.mcap          # agent commands, per-arm
  gello_right.mcap
  yam_left.mcap            # robot joint states
  yam_right.mcap
  camera_top-images-rgb.mp4
  camera_top-rgb-timestamp.npy
  session_meta.json
```

MCAP files use JSON encoding. `session_meta.json` records node descriptors, sim config, and episode timing.
