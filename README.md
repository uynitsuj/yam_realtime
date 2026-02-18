# Robot Realtime Control Interfaces

Robots Realtime is a research codebase supporting modular software stacks for realtime control, teleoperation, and policy integration on real-world robot embodiments including bi-manual I2RT YAM arms, Franka Panda, (more to come...).

It provides extensible pythonic infrastructure for low-latency joint command streaming, agent-based policy control, visualization, and integration with inverse kinematics solvers like [pyroki](https://github.com/chungmin99/pyroki) developed by [Chung-Min Kim](https://chungmin99.github.io/)! 

![yam_realtime](media/yam_realtime.gif)
<!-- ![franka_realtime](media/franka_realtime.gif) -->
![franka_realtime2](media/franka_realtime2.gif)

Shown is a headless-capable web-based real-time visualizer and controller for viewing commanded joint state and actual robot state, built with [Viser](https://viser.studio/main/)

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
For Bimanual YAMS:
```bash
uv run robots_realtime/envs/launch.py --config_path configs/yam_viser_bimanual.yaml
```
For Franka Panda (with default panda gripper):
```bash
uv sync --extra sensors --extra franka_panda
uv run robots_realtime/envs/launch.py --config_path configs/franka_viser_osc.yaml
```
or for robotiq gripper (ensure flange orientation is correct):
```bash
uv run robots_realtime/envs/launch.py --config_path configs/franka_robotiq_client.yaml
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

## Roadmap/Todos

- [ ] Add data logging infrastructure
- [ ] Implement a [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) agent controller
- [ ] Implement a [Physical Intelligence π0](https://www.physicalintelligence.company/blog/pi0) agent controller
