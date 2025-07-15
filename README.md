# YAM Realtime Control Interfaces

YAM Realtime is a modular software stack for realtime control, teleoperation, and policy integration on bi-manual I2RT YAM arms.

It provides infrastructure for low-latency joint command streaming, extensible agent-based policy control, visualization, and integration with inverse kinematics solvers like [pyroki](https://github.com/chungmin99/pyroki) developed by [Chung-Min Kim](https://chungmin99.github.io/)! 

![yam_realtime](media/yam_realtime.gif)

## Installation
Clone the repository and initialize submodules:
```
git clone --recurse-submodules https://github.com/uynitsuj/yam_realtime.git
```
If you already cloned it without --recurse-submodules, run:
```
git submodule update --init --recursive
```
Install the main package and I2RT repo for CAN driver interface:
```
cd yam_realtime
python -m pip install -e .
python -m pip install dependencies/i2rt/
```
# Configuration
First configure YAM arms CAN chain according to instructions from the [I2RT repo](https://github.com/i2rt-robotics/i2rt)

Your robot-specific configuration (joint limits, CAN IDs, kinematics parameters) should be defined in a YAML file under `configs/`.

# Launch
Then run the launch entrypoint script with an appropriate robot config file:
```
python yam_realtime/envs/launch.py --config_path configs/yam_viser_bimanual.yaml
```
# Extending with Custom Agents
To integrate your own controller or policy:

Subclass the base agent interface:
```python
from yam_realtime.agents.agent import Agent

class MyAgent(Agent):
    ...
```
Add your agent to your YAML config so the launcher knows which controller to instantiate.

Examples of agents you might implement:

- Teleoperation controller
- Learned policy (e.g., Diffusion Policy, ACT, PI0)
- Scripted trajectory player

## TODOS

- [ ] Add data logging infrastructure
- [ ] Implement a [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) agent controller
- [ ] Implement a [Physical Intelligence π0](https://www.physicalintelligence.company/blog/pi0) agent controller
