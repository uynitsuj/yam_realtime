from yam_realtime.agents.agent import Agent
from yam_realtime.inverse_kinematics.yam_pyroki import BimanualYamPyroki
from typing import Dict, Any
import threading
from yam_realtime.utils.portal_utils import remote
from yam_realtime.agents.constants import ActionSpec
import numpy as np
from dm_env.specs import Array

class ViserPyrokiAgent(Agent):
    def __init__(self):
        self.ik = BimanualYamPyroki()
        self.thread = threading.Thread(target=self.ik.solve_ik)
        self.thread.start()

    # def _setup_visualization(self):
    #     self.ik.viser_server

    def act(self, obs: Dict[str, Any]) -> Any:
        action = {
            "left": {
                "pos": np.concatenate([self.ik.joints["left"], [0.0]]),
            },
            "right": {
                "pos": np.concatenate([self.ik.joints["right"], [0.0]]),
            },
        }
        return action

    @remote(serialization_needed=True)
    def action_spec(self) -> ActionSpec:
        """Define the action specification."""
        return {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},
            "right": {"pos": Array(shape=(7,), dtype=np.float32)},
        }
