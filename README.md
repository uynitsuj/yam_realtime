# YAM Realtime Control Interfaces

A collection of realtime control interfaces for bi-manual I2RT YAM arms.

Differential IK solving handled by the [pyroki](https://github.com/chungmin99/pyroki) library developed by [Chung-Min Kim](https://chungmin99.github.io/)! 

![yam_realtime](media/yam_realtime.gif)

## Installation
```
git clone --recurse-submodules https://github.com/uynitsuj/yam_realtime.git

# Or if you already cloned the repo:
git submodule update --init --recursive

cd yam_realtime
python -m pip install -e .
python -m pip install dependencies/i2rt/
```
First configure YAM arms CAN chain according to instructions from the [I2RT repo](https://github.com/i2rt-robotics/i2rt)

Then run the launch entrypoint script with an appropriate robot config file:
```
python yam_realtime/envs/launch.py --config_path configs/yam_viser_bimanual.yaml
```

Add your own agents (e.g. robot policy controllers) by extending the base class `yam_realtime.agents.agent Agent` and adding an appropriate config file.

## TODOS

- [ ] Add data logging infrastructure
- [ ] Implement a [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) agent controller
- [ ] Implement a [Physical Intelligence π0](https://www.physicalintelligence.company/blog/pi0) agent controller
