# type: ignore
import collections
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import tyro
from dm_env.specs import Array

from yam_realtime.agents.agent import PolicyAgent
from yam_realtime.agents.constants import ActionSpec
from yam_realtime.agents.replay_agent import ReplayBimanualGelloAgent
from yam_realtime.data.data_utils import open_trajectory, recusive_flatten, reverse_flatten
from yam_realtime.utils.portal_utils import remote


class AsyncDiffusionAgent(PolicyAgent):
    def __init__(
        self,
    ) -> None:
        
    def load_model(self, folder_path: Union[str, Path], step: Optional[int] = None, bfloat16: bool = False) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def obs_to_model_input(self):
        raise NotImplementedError

    @remote()
    def act(self, obs):
        action = reverse_flatten(self(obs))["action"]
        
        return {
            "left": {"pos": action["left"]["pos"]},
            "right": {"pos": action["right"]["pos"]},
        }

    @remote(serialization_needed=True)
    def action_spec(self) -> ActionSpec:
        """Define the action specification."""
        if self.use_joint_state_as_action:
            return {
                "left": {
                    "pos": Array(shape=(7,), dtype=np.float32),
                    "vel": Array(shape=(7,), dtype=np.float32),
                },
                "right": {
                    "pos": Array(shape=(7,), dtype=np.float32),
                    "vel": Array(shape=(7,), dtype=np.float32),
                },
            }
        else:
            return {
                "left": {"pos": Array(shape=(7,), dtype=np.float32)},
                "right": {"pos": Array(shape=(7,), dtype=np.float32)},
            }

    def __call__(self, obs):
        obs = recusive_flatten(obs)
        with self.obs_deque_lock:
            if self.obs_deque is None:
                self.obs_deque = collections.deque([obs] * self.obs_horizon, maxlen=self.obs_horizon)
            # add latest observation to deque
            self.obs_deque.append(obs)
        raise NotImplementedError

