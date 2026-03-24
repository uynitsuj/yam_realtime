# robots_realtime

A research codebase for real-time robot teleoperation, data collection, and policy deployment.

The collection stack is as modular as the policy itself — agents (GELLO arms, learned policies, interactive IK gizmos) and environments (physical robots, sensors, cameras, MuJoCo sim) are composed at runtime from a YAML config. Swapping a GELLO for a trained policy, or real hardware for sim, is a one-line change. The recording format (MCAP + MP4) is identical regardless — so training pipelines don't need to change when the data source does.

<img src="media/yam_realtime.gif" width="500">
<img src="media/franka_realtime2.gif" width="500">
<img src="media/yam_active_leader_dagger.gif" width="500">

For building YAM active leader arms: [lerobot_teleoperator_yamactiveleader](https://github.com/uynitsuj/lerobot_teleoperator_yamactiveleader)

## Misc. Documentation
[Architecture & recording format](docs/architecture.md) 

[Extending (new agents, robots, cameras)](docs/extending.md)


---

## Installation

```bash
git clone --recurse-submodules https://github.com/uynitsuj/robots_realtime.git
cd robots_realtime
uv venv --python 3.11 && uv pip install -e .
```

---

## Usage

### Run a session (Teleop Data Collection)

```bash
uv run rr-session configs/sessions/yam_sim_gello_teleop.yaml
```

| Config | Description |
|--------|-------------|
| `yam_sim_dummy.yaml` | Synthetic agents → MuJoCo sim. No hardware needed. |
| `yam_sim_gello_teleop.yaml` | GELLO arms → MuJoCo sim + Viser viewer |
| `yam_bimanual_gello_teleop.yaml` | GELLO arms → physical YAM arms + cameras |
| `franka_viser_teleop.yaml` | Browser IK gizmo → Franka Panda + camera |

### Replay a sim episode

```bash
uv run rr-replay recordings/20260323/episode_175805_0473b1bc/
```

Opens a Viser viewer at `http://localhost:8080`. Two modes: **qpos** (exact, restores recorded state) and **physics** (re-simulates from actions).

> Commands run against the project venv via `uv run`. Alternatively: `source .venv/bin/activate` and drop the prefix.
