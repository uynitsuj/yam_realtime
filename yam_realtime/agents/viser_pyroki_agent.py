from yam_realtime.agents.agent import Agent
from yam_realtime.inverse_kinematics.yam_pyroki import BimanualYamPyroki
from typing import Dict, Any
import threading
from yam_realtime.utils.portal_utils import remote
from yam_realtime.agents.constants import ActionSpec
import numpy as np
from dm_env.specs import Array
import viser
import viser.extras
from copy import deepcopy
import time

class ViserPyrokiAgent(Agent):

    def __init__(self, right_arm_extrinsic: Dict[str, Any]):
        self.right_arm_extrinsic = right_arm_extrinsic
        self.viser_server = viser.ViserServer()
        self.ik = BimanualYamPyroki(viser_server=self.viser_server)
        self.ik_thread = threading.Thread(target=self.ik.run)
        self.ik_thread.start()
        self.obs = None
        self.real_vis_thread = threading.Thread(target=self._update_visualization)
        self.real_vis_thread.start()
        self._setup_visualization()


    def _setup_visualization(self):
        self.ik.base_frame_right.position = np.array(self.right_arm_extrinsic["position"])
        self.ik.base_frame_right.wxyz = np.array(self.right_arm_extrinsic["rotation"])

        self.base_frame_left_real = self.viser_server.scene.add_frame("/base_left_real", show_axes=False)
        self.base_frame_right_real = self.viser_server.scene.add_frame("/base_left_real/base_right_real", show_axes=False)
        self.base_frame_right_real.position = self.ik.base_frame_right.position

        self.urdf_vis_left_real = viser.extras.ViserUrdf(self.viser_server, deepcopy(self.ik.urdf), root_node_name="/base_left_real", mesh_color_override=(0.8, 0.5, 0.5))
        self.urdf_vis_right_real = viser.extras.ViserUrdf(self.viser_server, deepcopy(self.ik.urdf), root_node_name="/base_left_real/base_right_real", mesh_color_override=(0.8, 0.5, 0.5))

        for mesh in self.urdf_vis_left_real._meshes:
            mesh.opacity = 0.25
        for mesh in self.urdf_vis_right_real._meshes:
            mesh.opacity = 0.25

        # self.cam_image = self.viser_server.gui.add_image(np.zeros((100, 100, 3)), label="camera")

    def _update_visualization(self):
        while self.obs is None:
            time.sleep(0.025)
        while True:
            self.urdf_vis_right_real.update_cfg(np.flip(self.obs["right"]["joint_pos"]))
            self.urdf_vis_left_real.update_cfg(np.flip(self.obs["left"]["joint_pos"]))
            # self.cam_image.image = self.obs["top_camera"]["images"]["rgb"]
            time.sleep(0.02)

    def act(self, obs: Dict[str, Any]) -> Any:        
        self.obs = deepcopy(obs)

        action = {
            "left": {
                "pos": np.concatenate([np.flip(self.ik.joints["left"]), [0.0]]),
            },
            "right": {
                "pos": np.concatenate([np.flip(self.ik.joints["right"]), [0.0]]),
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
