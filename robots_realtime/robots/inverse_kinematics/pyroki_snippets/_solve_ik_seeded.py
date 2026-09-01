"""IK seeded from the previous solution.

Why this exists alongside :mod:`_solve_ik`
------------------------------------------
``solve_ik`` exposes no initial guess, and pyroki's ``JointVar`` defaults to the
*midpoint of the joint limits* (see ``pyroki/_robot.py``: ``default_joint_cfg =
(lower + upper) / 2``). Every call therefore starts the optimizer from mid-range
and converges to whichever local minimum that basin leads to. For a one-shot
gizmo drag that is fine. For a *continuously tracked* target it is not: two
nearly-identical targets can land on opposite IK branches (elbow up vs down),
and a position-controlled follower turns that discontinuity into a lunge.

This variant pins the solve to the branch the arm is already on, two ways:

* ``initial_vals`` starts the optimizer at ``seed_cfg`` instead of mid-range.
* a weak ``rest_cost`` toward ``seed_cfg`` breaks ties between branches that fit
  the pose target equally well, without meaningfully degrading pose accuracy.

Pass the *previous commanded configuration* as ``seed_cfg``. Both arguments and
the return value are in **URDF joint order** — for YAM that is reversed relative
to the i2rt motor order, so callers must ``np.flip`` on the way in and out (the
same convention as ``YamPyrokiViserAgent``).
"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk

# Weak relative to pose_cost's pos_weight=50 / ori_weight=10: enough to pick a
# branch, not enough to visibly pull the end-effector off target.
DEFAULT_SEED_WEIGHT = 0.5


def solve_ik_seeded(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: onp.ndarray,
    target_position: onp.ndarray,
    seed_cfg: onp.ndarray,
    seed_weight: float = DEFAULT_SEED_WEIGHT,
) -> onp.ndarray:
    """Solve IK, staying on the branch nearest ``seed_cfg``.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: Target orientation, shape (4,).
        target_position: Target position, shape (3,).
        seed_cfg: Previous configuration, in URDF joint order. Used both as the
            optimizer's starting point and as the rest-cost target.
        seed_weight: Weight on the pull toward ``seed_cfg``. 0 disables the
            rest cost but keeps the warm start.

    Returns:
        cfg: shape (robot.joints.num_actuated_joints,), URDF joint order.
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    seed_cfg = onp.asarray(seed_cfg, dtype=onp.float64)
    assert seed_cfg.shape == (robot.joints.num_actuated_joints,), (
        f"seed_cfg must have shape ({robot.joints.num_actuated_joints},), got {seed_cfg.shape}"
    )
    target_link_index = robot.links.names.index(target_link_name)
    cfg = _solve_ik_seeded_jax(
        robot,
        jnp.array(target_link_index),
        jnp.array(target_wxyz),
        jnp.array(target_position),
        jnp.array(seed_cfg),
        jnp.array(seed_weight),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return onp.array(cfg)


@jdc.jit
def _solve_ik_seeded_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    seed_cfg: jax.Array,
    seed_weight: jax.Array,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_constraint(
            robot,
            joint_var,
        ),
        pk.costs.rest_cost(
            joint_var,
            seed_cfg,
            weight=seed_weight,
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([joint_var.with_value(seed_cfg)]),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]
