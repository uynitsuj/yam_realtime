
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import dm_env
import mediapy
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from dm_env import specs

from robots_realtime.mujoco.envs.schema.robot import (
    MENAGERIE_ROOT,
    STATION_ROBOT_MAP,
    BimanualStationSpecConfig,
)
from robots_realtime.mujoco.envs.spec_builder import compile_station_spec


class YamEnv(dm_env.Environment):
    def _add_others(self, station_spec):
        return station_spec

    def _build_model(self):
        station_spec = compile_station_spec(self._station_spec_config)

        cameras = list(station_spec.cameras)
        top_camera = next(x for x in cameras if "top" in x.name).name
        left_camera = next(x for x in cameras if "left" in x.name).name
        right_camera = next(x for x in cameras if "right" in x.name).name
        self.camera_ids = {
            "top": top_camera,
            "left": left_camera,
            "right": right_camera,
        }
        station_spec = self._add_others(station_spec)

        # add floating camera
        station_spec.worldbody.add_camera(name="movie_camera", pos=[0, 0, 0], quat=[1, 0, 0, 0])
        model = station_spec.compile()

        self.xml = station_spec.to_xml()
        self.assets = station_spec.assets

        model.opt.timestep = 0.0001
        model.opt.integrator = 3  # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtintegrator

        self.actuator_names = [x.name for x in station_spec.actuators]
        self.actuator_ids = np.array([model.actuator(name).id for name in self.actuator_names])

        self.joint_names = [x.name for x in station_spec.joints]

        self.left_joint_names = [x.name for x in station_spec.joints if x.name.startswith("left_")]
        self.left_joint_ids = np.array([model.joint(name).id for name in self.left_joint_names])

        self.right_joint_names = [x.name for x in station_spec.joints if x.name.startswith("right_")]
        self.right_joint_ids = np.array([model.joint(name).id for name in self.right_joint_names])
        return model

    def set_movie_camera(self, pos: np.ndarray, quat: np.ndarray) -> None:
        camera = self._model.camera("movie_camera")
        camera.pos = pos
        camera.quat = quat
        mujoco.mj_kinematics(self._model, self._data)

    def save_xml(self, path: str = "station.xml") -> None:
        # save as xml
        with open(path, "w") as f:
            f.write(self.xml)

    def load_state(self, state: np.ndarray) -> None:
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(self._model, self._data, state, spec)
        mujoco.mj_forward(self._model, self._data)

    def __init__(
        self,
        station_spec_config: BimanualStationSpecConfig,
        seed: Optional[int] = None,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = np.inf,
        randomize_scene: bool = True,
        dm_env: bool = False,
        camera_obs: bool = True,
    ) -> None:
        self._station_spec_config = station_spec_config
        self._robot_spec_config = station_spec_config.robot
        self._model = self._build_model()
        self._data = mujoco.MjData(self._model)
        self._camera_obs = camera_obs

        self._height = 480
        self._width = 640

        self._randomize_scene = randomize_scene
        self._dm_env = dm_env  # TODO unify this at some point

        self._model.opt.timestep = physics_dt
        self.control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._terminated_already = False

        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)

        self._renderer: Optional[mujoco.Renderer] = None
        self._scene_option = mujoco.MjvOption()
        self._info = {}

        # Get initial state.
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        size = mujoco.mj_stateSize(self._model, spec)
        self.state = np.empty(size, np.float64)

        # viewer
        self._viewer_type = None
        self._viewer = None

    def time_limit_exceeded(self) -> bool:
        """Returns True if the simulation time has exceeded the time limit."""
        return self._data.time >= self._time_limit

    def render(self, camera_name: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                model=self._model,
                height=self._height,
                width=self._width,
            )
            self._renderer.disable_depth_rendering()
            self._renderer.disable_segmentation_rendering()
        self._renderer.update_scene(self._data, camera=camera_name)
        return self._renderer.render()

    def reset(self) -> dm_env.TimeStep:
        self._terminated_already = False
        mujoco.mj_resetData(self._model, self._data)

        # Forward simulation to update the data
        mujoco.mj_forward(self._model, self._data)

        obs = self._compute_observation()
        if self._dm_env:
            return dm_env.TimeStep(
                step_type=dm_env.StepType.FIRST,
                reward=None,
                discount=None,
                observation=obs,
            )
        else:
            return obs  # type: ignore

    def slow_move(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """For simulation, we can skip the slove move and directly apply the action."""
        return self.step(action_dict)

    def step(  # type: ignore
        self,
        action_dict: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if action_dict == {}:
            return self._compute_observation()
        # parse action dict into raw action
        raw_action = np.concatenate([action_dict["left"]["pos"], action_dict["right"]["pos"]])
        return self.step_np_action(raw_action).observation  # type: ignore

    def step_np_action(self, action: np.ndarray) -> dm_env.TimeStep:
        if self._terminated_already:
            raise ValueError("The environment has already terminated. Please reset the environment.")

        action[6] = action[6] * 0.041
        action[13] = action[13] * 0.041
        self._data.ctrl[self.actuator_ids] = action
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()

        terminated = self.time_limit_exceeded()
        discount = 1.0
        if terminated:
            step_type = dm_env.StepType.LAST
            self._terminated_already = True
        else:
            step_type = dm_env.StepType.MID

        return dm_env.TimeStep(
            step_type=step_type,
            reward=rew,
            discount=discount,
            observation=obs,
        )

    def observation_spec(self) -> Dict[str, Any]:  # type: ignore
        spec = {}
        for side in ["left", "right"]:
            _spec = {}
            _spec["joint_pos"] = specs.Array(shape=(6,), dtype=np.float32)
            _spec["joint_vel"] = specs.Array(shape=(6,), dtype=np.float32)
            _spec["gripper_pos"] = specs.Array(shape=(1,), dtype=np.float32)
            spec[f"{side}"] = _spec

        if self._camera_obs:
            for camera in ["top", "left", "right"]:
                spec[f"{camera}_camera"] = {
                    "images": {"rgb": specs.Array(shape=(self._height, self._width, 3), dtype=np.uint8)},
                    "timestamp": specs.Array(shape=(1,), dtype=np.float32),
                }
        spec["state"] = specs.Array(shape=self.state.shape, dtype=np.float32)
        spec["timestamp"] = specs.Array(shape=(1,), dtype=np.float32)
        return spec

    def action_spec(self) -> Dict[str, Any]:  # type: ignore
        """Return the action specification for the robot, which includes the gripper."""
        dofs = self._robot_spec_config.num_joint_dofs + self._robot_spec_config.num_gripper_dofs
        return {
            "left": {
                "pos": specs.Array(
                    shape=(dofs,),
                    dtype=np.float32,
                ),
            },
            "right": {
                "pos": specs.Array(
                    shape=(dofs,),
                    dtype=np.float32,
                ),
            },
        }

    def np_action_spec(self):
        dofs = 2 * (self._robot_spec_config.num_joint_dofs + self._robot_spec_config.num_gripper_dofs)
        return specs.BoundedArray(
            shape=(dofs,),
            dtype=np.float32,
            minimum=-1 * np.ones(dofs),
            maximum=np.ones(dofs),
        )

    def get_obs(self):
        return self._compute_observation()

    # Helper methods.
    def _compute_observation(self) -> dict:
        obs = {}
        sides = {"left": self.left_joint_ids, "right": self.right_joint_ids}
        for side_name, joint_ids in sides.items():
            _obs = {}
            qpos = self._data.qpos[joint_ids].astype(np.float32)
            _obs["joint_pos"] = qpos[:6]
            _obs["gripper_pos"] = qpos[6:]

            qvel = self._data.qvel[joint_ids].astype(np.float32)
            _obs["joint_vel"] = qvel[:6]
            obs[f"{side_name}"] = _obs

        if self._camera_obs:
            for camera in ["top", "left", "right"]:
                cam_name = self.camera_ids[camera]
                obs[f"{camera}_camera"] = {
                    "images": {"rgb": self.render(cam_name)},
                    "timestamp": time.time(),
                }

        mujoco.mj_getState(self._model, self._data, self.state, mujoco.mjtState.mjSTATE_INTEGRATION)
        obs["state"] = self.state.copy()
        obs["timestamp"] = time.time()
        return obs

    def _compute_reward(self) -> float:
        return 0

    def launch_viewer(self, type: Literal["mj", "cv"] = "mj") -> None:
        self._viewer_type = type
        if self._viewer_type == "mj":
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data, show_left_ui=True, show_right_ui=True)
        # todo: find a way to visualize the all images.
        elif self._viewer_type == "cv":
            raise NotImplementedError("CV viewer is not implemented")
        else:
            raise ValueError(f"Invalid viewer type: {self._viewer_type}")

    def viewer_sync(self) -> None:
        if self._viewer_type == "mj":
            self._viewer.sync()  # type: ignore
        elif self._viewer_type == "cv":
            pass
        else:
            raise ValueError(f"Invalid viewer type: {self._viewer_type}")

    def close_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


class YamEnvCabinateMug(YamEnv):
    def _add_others(self, station_spec):
        cabinate_spec = mujoco.MjSpec.from_file(str(MENAGERIE_ROOT / "kitchen" / "wall_cabinet_600.xml"))
        mug_spec = mujoco.MjSpec.from_file(str(MENAGERIE_ROOT / "mug" / "mug.xml"))

        cabinate_site = station_spec.worldbody.add_site(pos=[0.9, 0.60, -0.65], euler=[0, 0, -np.pi / 2 + np.pi / 6])
        cabinate_site.attach_body(cabinate_spec.worldbody, "cabinates_", "")

        mug_spawn_site = station_spec.worldbody.add_site(pos=[0.65, -0.3, 0.752])
        mug_body = mug_spawn_site.attach_body(mug_spec.worldbody, "mug_", "")
        self.mug_freejoint_name = "mug_joint"
        mug_body.add_freejoint(name=self.mug_freejoint_name)
        return station_spec

    def reset(self) -> dm_env.TimeStep:
        self._terminated_already = False
        mujoco.mj_resetData(self._model, self._data)

        if self._randomize_scene:
            # Add randomization to the initial position of the mug (only x and y and rotation around z)
            random_offset = self._random.uniform(-0.05, 0.05, size=2)  # Random offset in x, y
            random_z = self._random.uniform(-np.pi, np.pi)

            mug_joint = self._data.jnt(self.mug_freejoint_name)
            mug_joint.qpos[:2] += random_offset
            # sample a random direction
            perturb_axis = np.array([0.0, 0.0, 1.0])
            perturb_theta = self._random.uniform(-np.pi, np.pi)
            mujoco.mju_axisAngle2Quat(mug_joint.qpos[3:], perturb_axis, perturb_theta)

        # Forward simulation to update the data
        mujoco.mj_forward(self._model, self._data)

        obs = self._compute_observation()
        if self._dm_env:
            return dm_env.TimeStep(
                step_type=dm_env.StepType.FIRST,
                reward=None,
                discount=None,
                observation=obs,
            )
        else:
            return obs  # type: ignore


class YamEnvPickRedCube(YamEnv):
    def _add_others(self, station_spec):
        # Add a red cube to the station
        size = 0.015
        cube_spawn_site = station_spec.worldbody.add_site(pos=[0.6, -0.3, 0.753 + size])
        cube_spec = mujoco.MjSpec()
        body = cube_spec.worldbody.add_body(name="cube_body")
        body.add_geom(
            name="red_box",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[size, size, size],
            rgba=[1, 0, 0, 1],
        )
        cube_body = cube_spawn_site.attach_body(cube_spec.worldbody, "cube_", "")
        self.cup_freejoint_name = "cup_joint"
        cube_body.add_freejoint(name=self.cup_freejoint_name)

        # add a floating transparent light green region as the goal region
        worldbody = station_spec.worldbody  # .add_site(pos=[0.6, -0.3, 0.753 + size / 2])
        worldbody.add_geom(
            pos=[0.6, -0.3, 0.753 + 0.3],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.25, 0.25, 0.1],
            rgba=[0.5, 1, 0.5, 0.05],
            contype=0,  # no collision with the robot
            conaffinity=0,  # no collision with the robot
            group=2,
            mass=0,
        )
        return station_spec

    def reset(self) -> dm_env.TimeStep:
        self._terminated_already = False
        mujoco.mj_resetData(self._model, self._data)

        if self._randomize_scene:
            # Add randomization to the initial position of the mug (only x and y and rotation around z)
            random_offset = self._random.uniform(-0.15, 0.15, size=2)  # Random offset in x, y
            random_z = self._random.uniform(-np.pi, np.pi)

            cup_joint = self._data.jnt(self.cup_freejoint_name)
            cup_joint.qpos[:2] += random_offset
            # sample a random direction
            perturb_axis = np.array([0.0, 0.0, 1.0])
            perturb_theta = self._random.uniform(-np.pi, np.pi)
            mujoco.mju_axisAngle2Quat(cup_joint.qpos[3:], perturb_axis, perturb_theta)

            # add small random jitter to starting qpos state
            joint_names = []
            for j in self.right_joint_names + self.left_joint_names:
                if "finger" not in j:
                    joint_names.append(j)

            for joint_name in joint_names:
                joint = self._data.jnt(joint_name)
                joint.qpos += self._random.uniform(-0.1, 0.1, size=joint.qpos.shape)

        # Forward simulation to update the data
        mujoco.mj_forward(self._model, self._data)

        obs = self._compute_observation()
        if self._dm_env:
            return dm_env.TimeStep(
                step_type=dm_env.StepType.FIRST,
                reward=None,
                discount=None,
                observation=obs,
            )
        else:
            return obs  # type: ignore


@dataclass
class Args:
    policy: str = "random"
    debug: bool = False
    load_path: Optional[str] = None
    camera_obs: bool = True
    save_video: bool = False
    save_xml_path: Optional[str] = None


class KeyReset:
    def __init__(self):
        self.reset = False

    def key_callback(self, keycode: int) -> None:
        from dm_control.viewer import user_input

        if keycode == user_input.KEY_SPACE:
            self.reset = True


def main(args: Args) -> None:
    from robots_realtime.data.data_utils import flatten_dict, open_trajectory, reverse_flatten
    from robots_realtime.viewer.b_mujoco import MujocoViewerBackend
    from robots_realtime.viewer.core import Viewer
    from robots_realtime.viewer.utils import generate_spiral_camera

    # station_spec = STATION_ROBOT_MAP["FRANKA_STANDARD"]
    station_spec = STATION_ROBOT_MAP["SIM_YAM"]
    env = YamEnvPickRedCube(station_spec, dm_env=True, camera_obs=args.camera_obs)

    if args.save_xml_path:
        env.save_xml(args.save_xml_path)
        exit()

    action_spec = env.np_action_spec()

    m = env._model
    d = env._data
    reset = KeyReset()
    t = env.reset()

    def print_dict(d: dict, str: str = "") -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                print_dict(v, str=f"{str}{k}-")
            elif hasattr(v, "shape"):
                print(f"{str}{k}: {v.shape}")
            else:
                print(f"{str}{k}: {type(v)}")

    print_dict(t.observation)

    if args.load_path:
        trajectory = open_trajectory(args.load_path)

        if not trajectory:
            raise ValueError(f"Trajectory {args.load_path} not found")

        trajectory = reverse_flatten(trajectory)
        loaded_actions = trajectory["action"]
        if "state" in trajectory:
            state = trajectory["state"]
            env.reset()
            env.load_state(state[0])
    # if args.policy == "single_gello":
    #     from robots_realtime.agents.gello_agent import RainbowGelloAgent
    #     from robots_realtime.robots.generic_robot import GenericRobot, GenericRobotConfig

        config = GenericRobotConfig(
            driver_type="dynamixel",
            port="/dev/ttyUSB0",
            baudrate=2000000,
            joint_ids=(7, 8, 9, 10, 11, 12, 13),
            joint_offsets=np.array(
                [
                    2.87621398,
                    1.86225268,
                    -1.23638852,
                    0.03067962,
                    -0.18714566,
                    0.10584467,
                    0.08590292,
                ]
            ),
            joint_signs=[1, 1, 1, 1, 1, 1, 1],
            gripper_config=None,
            start_driver_read_thread=False,
        )
        robot = GenericRobot.from_config(config)
        agent = RainbowGelloAgent(robot=robot, handle_id=0x81, use_joint_state_as_action=False)

        def action(t):  # type: ignore
            a_ = np.zeros(16)

            state = agent.get_state()
            assert state is not None, "Agent state is None, make sure the agent is initialized correctly."
            one_hand_joints = state["joint_angles"]
            one_gripper = state["trigger"]
            a_[:7] = one_hand_joints
            a_[7] = one_gripper
            return a_

    elif args.policy == "random":

        def action(t):
            a = np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
            return a.astype(action_spec.dtype)

    elif args.policy == "none":

        def action(t):
            a = np.zeros(action_spec.shape)
            # a = np.array([1.5] * 14)
            a = np.array([0] * 14)
            return a.astype(action_spec.dtype)

    elif args.policy == "replay":
        assert args.load_path is not None

        class ReplayActions:
            def __init__(self, actions: dict):
                self.actions = actions
                self.i = 0

            def __call__(self, t):
                ret_action = {}
                for k in self.actions:
                    ret_action[k] = {"pos": self.actions[k]["pos"][self.i]}
                self.i += 1
                return np.concatenate(
                    [
                        ret_action["left"]["pos"],
                        ret_action["right"]["pos"],
                    ]
                )

        action = ReplayActions(loaded_actions)  # type: ignore

    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    mujoco.viewer.launch(
        model=m,
        data=d,
    )

    if args.debug:
        import cv2

        while True:
            t = env.step_np_action(action(t))
            flatten_obs = flatten_dict(t.observation)

            images = []
            for k, v in flatten_obs.items():
                if "images" in k or "rgb" in k:
                    images.append(v)
            stacked = np.hstack(images)
            cv2.imshow("images", stacked[:, :, ::-1])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        backends = []
        # render 1080p
        # renderer = MujocoRendererBackend(model=m, data=d, width=640, height=1080)
        # backends.append(renderer)
        backends.append(MujocoViewerBackend(model=m, data=d, key_callback=reset.key_callback))
        viewer = Viewer(backends=backends)
        waypoints = generate_spiral_camera(
            center=(0, 0, 1),
            start_radius=3.5,
            start_height=0.2,
            spiral_rate=-0.01,
            revolve_speed=-2 * np.pi / 20.0,  # one revolution every 5 seconds
            vertical_speed=0.01,
            total_time=5000.0,
            frames_per_second=30,
        )
        print("starting")
        with viewer:
            viewer.sync()
            images = []
            top_images = []
            try:
                for pos, quat in waypoints:
                    if reset.reset:
                        reset.reset = False
                        env.reset()
                        if args.policy == "replay":
                            env.load_state(state[0])  # type: ignore
                            action.i = 0  # type: ignore
                    else:
                        step_start = time.time()
                        t = env.step_np_action(action(t))
                        viewer.sync()
                        env.set_movie_camera(pos, quat)
                        img = env.render("movie_camera")
                        #  img = viewer.render()
                        images.append(img)
                        top_images.append(t.observation["top_camera"]["images"]["rgb"])

                        time_until_next_step = env.control_dt - (time.time() - step_start)
                        # if time_until_next_step > 0:
                        #     time.sleep(time_until_next_step)
            except KeyboardInterrupt:
                pass
            finally:
                if not args.save_video:
                    exit()
                mediapy.write_video(
                    "render.mp4",
                    images[1:],
                    fps=30,
                )
                mediapy.write_video(
                    "top_render.mp4",
                    top_images[1:],
                    fps=30,
                )


if __name__ == "__main__":
    """
    replay command:
    mjpython robots_realtime/mujoco/envs/yam_env.py --load_path  ~/nfs/data/sz_03/20250124/episode_20250124_170930_87ab75c1.npy.mp4 --policy replay
    """
    main(tyro.cli(Args))
