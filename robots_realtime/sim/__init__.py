"""Minimal self-contained MuJoCo simulation for the bimanual YAM robot.

Does not depend on the private xdof-sim package. Assets (XML + STL meshes)
are bundled under robots_realtime/sim/models/.

Quick start:
    import robots_realtime.sim as sim
    env = sim.make_env(scene="hybrid")
    obs, _ = env.reset()
    obs, history, *_ = env.step(env.action_space.sample())
"""

from robots_realtime.sim.env import MuJoCoYAMEnv
from robots_realtime.sim.config import SimConfig, default_sim_config, flat_init_config
from robots_realtime.sim.scene_variants import apply_scene_variant, list_variants


def make_env(
    scene_variant: str = "hybrid",
    task: str = "bottles",
    render_cameras: bool = True,
    prompt: str = "",
    chunk_dim: int = 30,
    config: SimConfig | None = None,
    **kwargs,
) -> MuJoCoYAMEnv:
    """Create and configure a MuJoCo YAM environment.

    Args:
        scene_variant: Visual variant — "eval", "training", or "hybrid".
        task: Scene XML to load — currently "bottles".
        render_cameras: Whether to render camera images in observations.
        prompt: Task description included in obs["prompt"].
        chunk_dim: Timesteps per action chunk.
        config: Optional SimConfig; defaults to default_sim_config().
        **kwargs: Passed through to MuJoCoYAMEnv.
    """
    env = MuJoCoYAMEnv(
        config=config,
        scene=task,
        render_cameras=render_cameras,
        prompt=prompt,
        chunk_dim=chunk_dim,
        **kwargs,
    )
    apply_scene_variant(env.model, scene_variant)
    return env


__all__ = [
    "MuJoCoYAMEnv",
    "SimConfig",
    "default_sim_config",
    "flat_init_config",
    "apply_scene_variant",
    "list_variants",
    "make_env",
]
