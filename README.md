# YAM Realtime Control Interfaces

A collection of realtime control interfaces for bi-manual I2RT YAM arms.

Differential IK solving handled by the [pyroki](https://github.com/chungmin99/pyroki) library developed by [Chung-Min Kim](https://chungmin99.github.io/)! 

```
git clone --recurse-submodules https://github.com/uynitsuj/yam_realtime.git

# Or if you already cloned the repo:
git submodule update --init --recursive
cd yam_realtime
python -m pip install -e .
python -m pip install /dependencies/i2rt/

python yam_realtime/envs/launch.py --config_path configs/yam_viser.yaml
```