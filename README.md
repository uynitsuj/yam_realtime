# Robot Realtime Control Interfaces

Robots Realtime is a research codebase supporting modular software stacks for realtime control, teleoperation, and policy integration on real-world robot embodiments including bi-manual I2RT YAM arms, Franka Panda, (more to come...).

It provides extensible pythonic infrastructure for low-latency joint command streaming, agent-based policy control, visualization, and integration with inverse kinematics solvers like [pyroki](https://github.com/chungmin99/pyroki) developed by [Chung-Min Kim](https://chungmin99.github.io/)! 

Examples:

<img src="media/yam_realtime.gif" width="500">
<img src="media/franka_realtime2.gif" width="500">
<img src="media/yam_active_leader_dagger.gif" width="500">

For details on how to build and assemble your own YAM active leader arms see its [github repo](https://github.com/uynitsuj/lerobot_teleoperator_yamactiveleader)!


## Installation
Clone the repository and initialize submodules:
```bash
git clone --recurse-submodules https://github.com/uynitsuj/robots_realtime.git
# Or if already cloned without --recurse-submodules, run:
git submodule update --init --recursive
```
Install the main package and I2RT repo for CAN driver interface using uv:
```bash
cd robots_realtime
curl -LsSf https://astral.sh/uv/install.sh | sh
source .venv/bin/activate

uv venv --python 3.11
uv pip install -e .
```
## Configuration
If using YAM arms, configure YAM arms CAN chain according to instructions from the [I2RT repo](https://github.com/i2rt-robotics/i2rt)

## Launch
Then run the launch entrypoint script with an appropriate robot config file.
For Real-World Bimanual YAMS:
```bash
uv run robots_realtime/envs/launch.py --config_path configs/yam/yam_viser_bimanual.yaml
```
For Real-World Franka Panda (with default panda gripper):
```bash
uv sync --extra sensors --extra franka_panda
uv run robots_realtime/envs/launch.py --config_path configs/franka/franka_viser_osc.yaml
```
or for robotiq gripper instead of default panda grippers (ensure flange orientation is correct):
```bash
uv run robots_realtime/envs/launch.py --config_path configs/franka/franka_robotiq_viser.yaml
```
For testing YAM Active Leaders with mujoco simulation. If on mac, use:
```bash
DYLD_LIBRARY_PATH=~/.local/share/uv/python/cpython-3.11.14-macos-aarch64-none/lib .venv/bin/mjpython robots_realtime/envs/launch.py --config-path configs/yam/yam_gello_pick_red_cube_sim.yaml
```
Otherwise:
```bash
uv run python robots_realtime/envs/launch.py --config-path configs/yam/yam_gello_pick_red_cube_sim.yaml
```

## Extending with Custom Agents
To integrate your own controller or policy:

Subclass the base agent interface:
```python
from robots_realtime.agents.agent import Agent

class MyAgent(Agent):
    ...
```
Add your agent to your YAML config so the launcher knows which controller to instantiate.

Examples of agents you might implement:
- Leader arm or VR controller teleoperation
- Learned policy (e.g., Diffusion Policy, ACT, PI0)
- Offline motion-planner + scripted trajectory player

## Linting
If contributing, please use ruff (automatically installed) for linting (https://docs.astral.sh/ruff/tutorial/#getting-started)
```bash
ruff check # lint
ruff check --fix # lint and fix anything fixable
ruff format # code format
```
