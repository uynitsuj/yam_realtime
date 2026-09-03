"""DAgger intervention arbiter for bimanual YAM.

Runs an OpenPI policy and two *passive* GELLO leaders in the same session, and
arbitrates which of them commands the followers. The operator toggles takeover
with a button on either leader, corrects the policy by hand, then toggles back —
as many times as they like inside a single recorded episode. Every published
command carries a label saying who produced it.

Why the correction is relative, not absolute
--------------------------------------------
The obvious implementation — switch the followers' command source to the
leaders' joint positions — is unsafe here. A passive GELLO is unactuated and
unconstrained, so at the moment of takeover it is wherever the operator left it,
which bears no relation to where the policy has driven the follower. Commanding
its absolute position makes the follower snap across that arbitrary delta, and
"arbitrary" can mean most of the joint range.

So takeover *anchors* instead. We latch the leader's TCP pose and the follower's
TCP pose at the switch instant, and thereafter command::

    T_follower(t) = anchor_follower ∘ (motion of the leader since anchor_leader)

At t = takeover the leader's motion is identity, so the first intervention
command equals the pose the follower was already being commanded to — the switch
is continuous by construction rather than by tuning. Getting from that Cartesian
target back to joint commands is what the IK is for.

Deltas compose in the *base* frame: a hand translation in +x moves the follower
in +x whatever the wrist is doing. Body-frame composition would rotate the
operator's displacement by the current end-effector orientation, which is much
harder to fly.

Episode control
---------------
Recording is driven entirely from this agent, through Session's ``record_topic``.
That monitor loop is *level*-triggered on a latched boolean — False→True starts
an episode, True→False ends and saves it — so the agent owns the latch and is the
single source of truth for episode boundaries:

* **Unpause starts an episode.** Unpausing *is* the handoff: the policy begins
  driving the real scene at that instant, so anything not recorded from that
  moment is lost rollout. The agent watches ``obs["_paused"]`` and raises the
  latch on the paused→unpaused edge.
* **Pause ends it** (saving). Keeps the latch in lockstep with Session no matter
  how an episode ended — including ``episode_timeout``, which ends the episode
  *and* re-pauses.
* **The white button runs the rollout cycle**, which is the normal way to work:
  press to end the episode and send the arms home, press again to start the next
  one. See below.

Session's own ``record_on_unpause`` must stay **off**, and this agent replaces
it. Leaving it on would have Session call ``start_episode()`` directly without
advancing its ``_prev_record_signal``, desynchronising the latch: the operator's
next press would resolve to a ``start_episode()`` on an already-recording session
and be silently swallowed.

The rollout cycle
-----------------
After the initial ``[space]``, a whole collection session runs off the two
leader buttons::

    IDLE ──white──> POLICY <──yellow──> INTERVENTION
    (parked            │                     │
     at home)          │                  (yellow)
        ↑              │                     ↓
        └──HOMING──white┘                HANDBACK ──> POLICY

Pressing white during a rollout ends the episode *and* walks the arms back to
``home_joint_pos``, then parks them there — so the operator gets a safe,
unrecorded window to reset the scene without the policy flailing at it. Pressing
white again starts the next episode and hands the arms back to the policy.

Homing is done by *commanding* it through the same slew limiter the intervention
path uses, not by any special robot call. That matters: Session's pause gates
``RobotNode`` outright, so a "parked" state built on pause could not be commanded
home in the first place, and ``RobotNode._move_to_pose`` blocks, which would
stall its 200 Hz loop mid-session. Commanding it keeps the motion rate-limited,
non-blocking, and continuous from wherever the arms happened to be.

Both routes back into the policy — white from IDLE, and unpause — go through
HANDBACK rather than switching straight over, so the policy's chunk is flushed
and the command blended from the parked pose instead of stepping to it.

Handback has the mirror problem
-------------------------------
Going the other way, the policy's action chunk was computed from an observation
taken up to one chunk ago and may command a pose well away from where the
operator left the arm. Handback therefore (a) asks the policy node to flush its
chunk buffer via ``dagger/policy_reset``, (b) holds the operator's final pose
until a command inferred from a *post-flush* observation actually arrives, and
(c) blends joint-space from that pose to the policy's over ``handback_blend_s``.

Conventions
-----------
* **Joint order.** The YAM URDF declares ``joint6 … joint1``, reversed relative
  to the i2rt motor order used on the bus and by the policy. Every crossing
  needs ``np.flip`` — same as :class:`YamPyrokiViserAgent`.
* **TCP, not link_6.** Anchors and targets are the gripper TCP
  (``TCP_OFFSET_POS`` from ``link_6``), because that is the point the operator's
  hand actually controls. The offset is inverted before the target reaches IK.
* **Gripper.** ``PassiveGelloLeaderAgent`` already emits a normalized ``[0, 1]``
  trigger value matching i2rt's ``JointMapper`` convention, and the policy's 7th
  element is the same normalized quantity. No unit conversion anywhere; the
  gripper is passed through absolutely (slew-limited), so squeezing always means
  closing.
* **Same kinematics both sides.** The passive GELLO is a kinematically-matched
  YAM skeleton and ``PassiveGelloLeaderAgent`` applies ``joint_signs`` to put its
  output in YAM joint space, so one URDF serves leader FK and follower FK/IK.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import viser.transforms as vtf
from dm_env.specs import Array

from robots_realtime.agents.agent import Agent
from robots_realtime.agents.teleoperation.passive_gello_leader_agent import (
    PassiveGelloLeaderAgent,
)

logger = logging.getLogger(__name__)

ARM_DOF = 6
CMD_DOF = ARM_DOF + 1          # 6 arm joints + normalized gripper
EE_LINK = "link_6"
BASE_LINK = "base_link"
DEFAULT_URDF = "dependencies/i2rt/i2rt/robot_models/arm/yam/yam.urdf"

# Gripper TCP relative to link_6 — same constant the viser IK gizmos use
# (robots/inverse_kinematics/yam_pyroki.py: tcp_offset_frame.position).
TCP_OFFSET_POS = (0.0, 0.04, -0.13)

# ── Leader buttons ───────────────────────────────────────────────────────────
# The gripper encoder's `digital_inputs` byte carries two momentary switches,
# decoded LSB-first the same way i2rt does in
# dm_driver.PassiveEncoderReader._parse_encoder_message:
#
#     button_state = [digital_inputs % 2, digital_inputs // 2]
#
#   index 0 = bit 0 = TOP button    = YELLOW
#   index 1 = bit 1 = LOWER button  = WHITE   (i2rt calls it "grip")
#
# Provenance: the bit order is i2rt's decoding, and the top/lower naming comes
# from i2rt labelling the pair [button_top, button_grip]
# (utils/mujoco_control_interface.py, and its btn_top_geom / btn_grip_geom
# indicator spheres). Those labels describe the *teaching handle* on the active
# YAM leader, so on a passive gello the position naming is inherited rather than
# independently confirmed — the colours are ours. If takeover ever fires off the
# wrong button, `scripts/probe_gello_buttons.py` is the authority: it prints the
# live byte with both indices labelled.
BUTTON_YELLOW_TOP = 0
BUTTON_WHITE_LOWER = 1
BUTTON_COLOURS = {BUTTON_YELLOW_TOP: "yellow/top", BUTTON_WHITE_LOWER: "white/lower"}

MODE_IDLE = "idle"                  # parked at home, policy ignored, not recording
MODE_HOMING = "homing"              # driving back to home after a rollout
MODE_POLICY = "policy"
MODE_INTERVENTION = "intervention"
MODE_HANDBACK = "handback"

# Modes in which a rollout is underway (episode recording, policy in the loop).
_LIVE_MODES = (MODE_POLICY, MODE_INTERVENTION, MODE_HANDBACK)


class _ArmAnchor:
    """SE(3) reference frames latched at the instant of takeover, for one arm."""

    __slots__ = ("T_fol_0", "T_lead_0")

    def __init__(self, T_lead_0: vtf.SE3, T_fol_0: vtf.SE3) -> None:
        self.T_lead_0 = T_lead_0
        self.T_fol_0 = T_fol_0


class DaggerInterventionAgent(Agent):
    """Arbitrates between an OpenPI policy and two passive GELLO leaders.

    Reads the policy's commands off the bus (via ``state_topics``) rather than
    running inference itself, so the policy node keeps publishing its own
    commands throughout — during intervention those become the recorded
    counterfactual "what the policy would have done", which is the signal DAgger
    training needs.

    Expected ``state_topics`` on the owning AgentNode::

        left:         yam_left/joint_state        # measured follower state
        right:        yam_right/joint_state
        policy_left:  openpi_policy/left_pos      # policy command, motor order
        policy_right: openpi_policy/right_pos

    Returns ``{"left": {"pos": (7,)}, "right": {"pos": (7,)}, "_extras": {...}}``.

    Args:
        left_channel:  SocketCAN interface for the left passive leader.
        right_channel: SocketCAN interface for the right passive leader.
        joint_signs:   Per-joint ±1 sign map, forwarded to both leaders. Must
                       match the teleop config for this station.
        takeover_button_index: Which leader button toggles intervention.
                       0 = YELLOW (top), 1 = WHITE (lower). Defaults to yellow:
                       it's the action used most within a rollout, and a mispress
                       is cheap (press again to hand back), whereas a mispressed
                       episode button splits or ends the recording.
        episode_button_index: Which leader button toggles the episode (start /
                       end-and-save), or None to disable. 0 = YELLOW (top),
                       1 = WHITE (lower); defaults to white. Must differ from
                       ``takeover_button_index``.
        episode_button_arm: Limit the episode button to ``left`` or ``right``;
                       None accepts it from either leader.
        pause_button_index: Which leader button toggles the Session pause gate,
                       or None to disable. The configured arm/index pair must
                       not overlap the episode-button mapping.
        pause_button_arm: Limit the pause button to ``left`` or ``right``;
                       None accepts it from either leader.
        record_on_unpause: Start an episode on the paused→unpaused edge and end
                       it on the unpaused→paused edge. On by default, because
                       unpausing hands the arms to the policy and a rollout that
                       isn't recorded from that instant has lost its start.
                       Requires ``record_topic: <node>/record`` and Session's own
                       ``record_on_unpause: false`` — see the module docstring.
        home_joint_pos: Parked pose the arms return to when a rollout ends —
                       either one length-7 list for both arms, or
                       ``{"left": [...], "right": [...]}``. Should match the
                       ``startup_joint_pos`` on the RobotNodes so the arms park
                       where they started; there is no way for this agent to read
                       that off the robot nodes, so it is stated twice. Required
                       when ``home_on_episode_end`` is True.
        home_on_episode_end: Walk the arms back to ``home_joint_pos`` and park
                       there whenever a rollout ends, instead of leaving them
                       wherever the policy or operator left them. Set False to
                       get a plain episode toggle with no motion.
        homing_max_joint_speed: Joint-speed cap while homing (rad/s). Separate
                       from ``max_joint_speed`` so the unattended return to home
                       can be gentler than live teleop.
        home_tol_rad:  Per-joint tolerance for declaring home reached and
                       switching from HOMING to IDLE.
        joint_limits:   Optional motor-order ``[[lower, upper], ...]`` limits
                       for the six follower arm joints. When provided, these
                       replace the wider URDF limits used by follower IK.
        joint_limit_margin: Symmetric soft margin inside ``joint_limits``.
                       Seeds and final IK results are explicitly guarded too.
        urdf_path:     YAM URDF used for both leader FK and follower FK/IK.
        handback_blend_s: Duration of the joint-space blend from the operator's
                       final pose to the policy command.
        handback_fresh_timeout_s: How long to wait for a post-flush policy
                       command before blending toward whatever is latest anyway.
                       Guards against a wedged or disconnected policy server.
        max_joint_speed:   Cap on commanded arm-joint speed (rad/s). 0 disables.
                       This is a *glitch* guard, not a comfort knob: it bounds how
                       far one tick can move the arm if a leader encoder reports
                       garbage. Setting it low doesn't feel safer, it feels
                       broken. Measured tracking gain at a 1.5 rad/s hand speed:
                       0.83 at 2.0 (reads as wrong scale), 0.98 at 3.0, 0.99 at
                       6.0. The URDF joint velocity limit is 10 rad/s, so 6.0
                       keeps headroom while capping one 100 Hz tick at 60 mrad.
        max_gripper_speed: Cap on normalized gripper speed (units/s). 0 disables.
        smoothing_tau_s:   First-order filter on the slew target (seconds), 0 to
                       disable. Costs lag directly — ~4 mm at 0.03, ~1 mm at 0.01
                       — so keep it only large enough to round off the leader's
                       encoder quantisation.
        ik_seed_weight:    Weight pulling the IK solution toward the previous
                       command, to hold one IK branch. See ``_solve_ik_seeded``.
        ik_pos_tol_m / ik_ori_tol_rad: *Quality* thresholds. Missing by more than
                       this reports ``ik_ok: false`` in the control-mode label so
                       the stretch can be filtered out in post, but the arm keeps
                       tracking the closest reachable pose. It has to: the target
                       goes unreachable whenever the operator pushes past a joint
                       limit or out of the workspace, and freezing the arm on the
                       first millimetre of overshoot reads as the rig being
                       broken. Tracking the reachable component is what every
                       teleop system does — the operator feels the axis stop
                       while the others keep following.
        ik_reject_pos_m: *Sanity* threshold, much larger. A solution this far off
                       is a solver blow-up rather than a workspace edge, so the
                       last accepted solution is held instead.
        gello_stale_s: Treat a leader as dead after this long without a CAN
                       frame. While intervening on a dead leader we hold the last
                       command rather than handing back, because silently
                       re-arming the policy mid-correction is the worse failure.
        include_gripper: Forwarded to the leaders; keep True for YAM.
        button_debounce_s: Stable time required for button changes, using CAN timestamps.
    """

    use_joint_state_as_action: bool = False

    def __init__(
        self,
        left_channel: str = "can_lead_l",
        right_channel: str = "can_lead_r",
        joint_signs: Optional[List[int]] = None,
        takeover_button_index: int = BUTTON_YELLOW_TOP,
        episode_button_index: Optional[int] = BUTTON_WHITE_LOWER,
        episode_button_arm: Optional[str] = None,
        pause_button_index: Optional[int] = None,
        pause_button_arm: Optional[str] = None,
        record_on_unpause: bool = True,
        home_joint_pos: Optional[Any] = None,
        home_on_episode_end: bool = True,
        homing_max_joint_speed: float = 1.0,
        home_tol_rad: float = 0.03,
        joint_limits: Optional[Any] = None,
        joint_limit_margin: float = 0.0,
        urdf_path: str = DEFAULT_URDF,
        handback_blend_s: float = 1.0,
        handback_fresh_timeout_s: float = 1.5,
        max_joint_speed: float = 6.0,
        max_gripper_speed: float = 4.0,
        smoothing_tau_s: float = 0.01,
        ik_seed_weight: float = 0.5,
        ik_pos_tol_m: float = 0.02,
        ik_ori_tol_rad: float = 0.2,
        ik_reject_pos_m: float = 0.15,
        gello_stale_s: float = 0.5,
        include_gripper: bool = True,
        leader_gripper_range_rad: Optional[float] = None,
        startup_timeout_s: float = 5.0,
        button_debounce_s: float = 0.02,
    ) -> None:
        if takeover_button_index not in (0, 1):
            raise ValueError(f"takeover_button_index must be 0 or 1, got {takeover_button_index}")
        if episode_button_index is not None:
            if episode_button_index not in (0, 1):
                raise ValueError(f"episode_button_index must be 0, 1 or null, got {episode_button_index}")
            if episode_button_index == takeover_button_index:
                raise ValueError(
                    f"episode_button_index and takeover_button_index are both "
                    f"{takeover_button_index} — the gello only has two buttons, so they must differ"
                )

        if episode_button_arm not in (None, "left", "right"):
            raise ValueError(
                f"episode_button_arm must be left, right or null, got {episode_button_arm!r}"
            )
        if pause_button_index is not None and pause_button_index not in (0, 1):
            raise ValueError(f"pause_button_index must be 0, 1 or null, got {pause_button_index}")
        if pause_button_arm not in (None, "left", "right"):
            raise ValueError(
                f"pause_button_arm must be left, right or null, got {pause_button_arm!r}"
            )
        if pause_button_index is not None and pause_button_index == episode_button_index:
            mappings_overlap = (
                pause_button_arm is None
                or episode_button_arm is None
                or pause_button_arm == episode_button_arm
            )
            if mappings_overlap:
                raise ValueError(
                    "pause and episode buttons overlap; use different indices or distinct arms"
                )

        self.arms = ["left", "right"]
        self._button_index = int(takeover_button_index)
        self._episode_button_index = (
            None if episode_button_index is None else int(episode_button_index)
        )
        self._episode_button_arm = episode_button_arm
        self._pause_button_index = (
            None if pause_button_index is None else int(pause_button_index)
        )
        self._pause_button_arm = pause_button_arm
        self._handback_blend_s = float(handback_blend_s)
        self._handback_fresh_timeout_s = float(handback_fresh_timeout_s)
        self._max_joint_speed = float(max_joint_speed)
        self._max_gripper_speed = float(max_gripper_speed)
        self._smoothing_tau_s = float(smoothing_tau_s)
        self._ik_seed_weight = float(ik_seed_weight)
        self._ik_pos_tol = float(ik_pos_tol_m)
        self._ik_ori_tol = float(ik_ori_tol_rad)
        self._ik_reject_pos = float(ik_reject_pos_m)
        if self._ik_reject_pos <= self._ik_pos_tol:
            raise ValueError(
                f"ik_reject_pos_m ({self._ik_reject_pos}) must exceed ik_pos_tol_m "
                f"({self._ik_pos_tol}) — the first is a blow-up guard, the second a "
                f"quality flag."
            )
        self._gello_stale_s = float(gello_stale_s)
        self._home_on_episode_end = bool(home_on_episode_end)
        self._homing_speed = float(homing_max_joint_speed)
        self._home_tol = float(home_tol_rad)
        self._joint_guards = self._parse_joint_limits(joint_limits, joint_limit_margin)
        self._home = self._parse_home(home_joint_pos)
        if self._home_on_episode_end and self._home is None:
            raise ValueError(
                "home_on_episode_end=True requires home_joint_pos — set it to the same "
                "pose as the RobotNodes' startup_joint_pos, or pass "
                "home_on_episode_end=false for a plain episode toggle."
            )

        # ── Leaders (composition — reuse their CAN reader, sign map, gripper map)
        leader_kwargs: Dict[str, Any] = {
            "joint_signs": joint_signs,
            "include_gripper": include_gripper,
            "startup_timeout_s": startup_timeout_s,
            "button_debounce_s": button_debounce_s,
        }
        if leader_gripper_range_rad is not None:
            leader_kwargs["leader_gripper_range_rad"] = leader_gripper_range_rad
        self._leaders: Dict[str, PassiveGelloLeaderAgent] = {}
        try:
            self._leaders["left"] = PassiveGelloLeaderAgent(
                channel=left_channel, robot_name="left", **leader_kwargs
            )
            self._leaders["right"] = PassiveGelloLeaderAgent(
                channel=right_channel, robot_name="right", **leader_kwargs
            )
        except Exception:
            for leader in self._leaders.values():
                leader.close()
            raise

        # ── Kinematics. yourdfpy for FK (cheap, no jax), pyroki for IK.
        import pyroki as pk  # noqa: PLC0415 — heavy import, keep it off module load
        import yourdfpy  # noqa: PLC0415

        from robots_realtime.robots.inverse_kinematics.pyroki_snippets._solve_ik_seeded import (  # noqa: PLC0415
            solve_ik_seeded,
        )

        self._solve_ik = solve_ik_seeded
        # Separate URDF handles per role: update_cfg() mutates the scene graph,
        # so sharing one would make leader FK clobber follower FK mid-tick.
        self._urdf_lead = yourdfpy.URDF.load(urdf_path, load_meshes=False, build_scene_graph=True)
        self._urdf_fol = yourdfpy.URDF.load(urdf_path, load_meshes=False, build_scene_graph=True)
        self._apply_ik_joint_limits(self._urdf_fol)
        self._robot = pk.Robot.from_urdf(self._urdf_fol)
        self._tcp_offset = vtf.SE3.from_rotation_and_translation(
            vtf.SO3.identity(), np.asarray(TCP_OFFSET_POS, dtype=np.float64)
        )

        # ── Mode state
        # Parked at home is the resting state: RobotNode.setup() has already moved
        # the arms to startup_joint_pos and Session starts paused, so IDLE is
        # where we actually are and commanding home is a no-op.
        self._mode = MODE_IDLE if self._home_on_episode_end else MODE_POLICY
        # Per (arm, button index) previous level, for rising-edge detection.
        self._prev_button = {(arm, i): False for arm in self.arms for i in (0, 1)}
        self._button_edges_by_arm = {(arm, i): False for arm in self.arms for i in (0, 1)}
        # Latched record request. Session edge-detects it on record_topic; this
        # agent is the only thing that drives it (see the module docstring).
        self._record_latch = False
        self._record_on_unpause = bool(record_on_unpause)
        # None until the first observation, so we never mistake "first tick" for
        # a pause transition.
        self._paused_prev: Optional[bool] = None
        self._last_rehome_request_ts: Optional[float] = None
        self._anchors: Dict[str, _ArmAnchor] = {}
        self._takeover_count = 0
        self._switch_t = time.monotonic()
        self._ik_ok = True
        self._last_ik_warn_t = 0.0

        # Handback bookkeeping
        self._handback_from: Dict[str, np.ndarray] = {}
        self._handback_t0: Optional[float] = None       # set when a fresh policy cmd lands
        self._handback_requested_t: float = 0.0
        self._policy_reset_pending = False
        self._policy_ts_at_handback: Dict[str, float] = {}

        # Last ACCEPTED IK solution per arm (arm joints, motor order). This is the
        # IK loop's own state and must stay separate from the slew-limited
        # command: seeding the solver from the command couples the two loops and
        # latches up — the limiter holds the command back, so the seed lags, so
        # the target looks unreachable, so the solve is rejected, so the command
        # never advances and the gap grows without bound. Seeding from the last
        # solution keeps the per-tick step small (a few mm at hand speeds), which
        # is both what the warm start wants and what keeps rest_cost near-free.
        self._ik_q: Dict[str, np.ndarray] = {}

        # ── Command state (slew limiter), keyed by arm; motor order, length 7
        self._cmd: Dict[str, np.ndarray] = {}
        self._goal_filtered: Dict[str, np.ndarray] = {}
        self._last_act_t: Optional[float] = None

        self._warm_up_ik()
        logger.info(
            "DaggerInterventionAgent ready (leaders %s/%s) — takeover: %s button "
            "(index %d), episode: %s",
            left_channel,
            right_channel,
            BUTTON_COLOURS.get(self._button_index, "?"),
            self._button_index,
            "disabled"
            if self._episode_button_index is None
            else f"{BUTTON_COLOURS.get(self._episode_button_index, '?')} button "
                 f"(index {self._episode_button_index})",
        )

    def _parse_home(self, home: Any) -> Optional[Dict[str, np.ndarray]]:
        """Normalise home_joint_pos into {arm: (7,) array}, or None."""
        if home is None:
            return None
        if isinstance(home, dict):
            missing = [a for a in self.arms if a not in home]
            if missing:
                raise ValueError(f"home_joint_pos is missing arm(s) {missing}")
            per_arm = {a: np.asarray(home[a], dtype=np.float64).ravel() for a in self.arms}
        else:
            shared = np.asarray(home, dtype=np.float64).ravel()
            per_arm = {a: shared.copy() for a in self.arms}
        for arm, q in per_arm.items():
            if q.shape != (CMD_DOF,):
                raise ValueError(
                    f"home_joint_pos[{arm}] must have {CMD_DOF} elements "
                    f"(6 arm joints + normalized gripper), got {q.shape[0]}"
                )
        return per_arm


    def _parse_joint_limits(self, limits: Any, margin: float) -> Optional[np.ndarray]:
        """Return guarded six-joint limits in physical motor order."""
        margin = float(margin)
        if margin < 0.0:
            raise ValueError(f"joint_limit_margin must be non-negative, got {margin}")
        if limits is None:
            return None
        raw = np.asarray(limits, dtype=np.float64)
        if raw.shape != (ARM_DOF, 2):
            raise ValueError(f"joint_limits must have shape ({ARM_DOF}, 2), got {raw.shape}")
        if not np.all(np.isfinite(raw)) or np.any(raw[:, 0] >= raw[:, 1]):
            raise ValueError("joint_limits must contain finite lower < upper pairs")
        guarded = raw.copy()
        guarded[:, 0] += margin
        guarded[:, 1] -= margin
        if np.any(guarded[:, 0] >= guarded[:, 1]):
            raise ValueError("joint_limit_margin leaves one or more joints with no valid range")
        return guarded

    def _apply_ik_joint_limits(self, urdf: Any) -> None:
        """Install motor-order follower guards into the reversed-order YAM URDF."""
        if self._joint_guards is None:
            return
        by_name = {joint.name: joint for joint in urdf.robot.joints}
        for motor_idx, (lower, upper) in enumerate(self._joint_guards):
            name = f"joint{motor_idx + 1}"
            joint = by_name.get(name)
            if joint is None or joint.limit is None:
                raise ValueError(f"Cannot apply follower limit: {name} is missing from the IK URDF")
            joint.limit.lower = float(lower)
            joint.limit.upper = float(upper)

    def _guard_arm_joints(self, joints: np.ndarray) -> np.ndarray:
        """Clamp one motor-order IK vector to the configured soft guards."""
        q = np.asarray(joints, dtype=np.float64).reshape(ARM_DOF).copy()
        if self._joint_guards is not None:
            q = np.clip(q, self._joint_guards[:, 0], self._joint_guards[:, 1])
        return q

    # ── Kinematics helpers ────────────────────────────────────────────────────

    def _warm_up_ik(self) -> None:
        """Force the IK JIT compile now (~1.7 s) instead of at first takeover.

        Without this the first intervention tick blocks for seconds with the arm
        mid-rollout, which is exactly when a stall is least acceptable.
        """
        t0 = time.perf_counter()
        seed = self._guard_arm_joints(np.zeros(ARM_DOF))
        self._solve_ik(
            self._robot, EE_LINK, np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.3, 0.0, 0.3]), seed
        )
        logger.info("IK warm-up (JIT compile) took %.2f s", time.perf_counter() - t0)

    def _fk_tcp(self, urdf: Any, q_motor: np.ndarray) -> vtf.SE3:
        """TCP pose in the arm's base frame, from joints in i2rt motor order."""
        urdf.update_cfg(np.flip(np.asarray(q_motor[:ARM_DOF], dtype=np.float64)))
        T_link6 = vtf.SE3.from_matrix(np.asarray(urdf.get_transform(EE_LINK, BASE_LINK)))
        return T_link6 @ self._tcp_offset

    def _ik_tcp(
        self, T_tcp_target: vtf.SE3, seed_motor: np.ndarray
    ) -> tuple[np.ndarray, bool, bool]:
        """Solve for joints (motor order) putting the TCP at *T_tcp_target*.

        Returns ``(q_motor, on_target, sane)``:

        * ``on_target`` — within ``ik_pos_tol_m`` / ``ik_ori_tol_rad``. False
          means the operator is pushing past a joint limit or out of the
          workspace; the solution is still the closest reachable pose and is
          worth tracking, but the stretch is flagged in the recorded label.
        * ``sane`` — within ``ik_reject_pos_m``. False means the solver blew up
          rather than hit a workspace edge, and the caller should hold.
        """
        # get_target_poses() builds the IK target as control @ tcp_offset, so
        # undo the offset to get the link_6 pose the solver wants.
        T_link6 = T_tcp_target @ self._tcp_offset.inverse()
        seed_motor = self._guard_arm_joints(seed_motor)
        sol_urdf = self._solve_ik(
            self._robot,
            EE_LINK,
            np.asarray(T_link6.rotation().wxyz, dtype=np.float64),
            np.asarray(T_link6.translation(), dtype=np.float64),
            np.flip(seed_motor),
            self._ik_seed_weight,
        )
        # Pyroki already solves against these bounds. Keep an explicit guard,
        # as Market42 does, so numerical tolerance can never leak out-of-range
        # commands to the follower controller.
        q_motor = self._guard_arm_joints(np.flip(np.asarray(sol_urdf, dtype=np.float64)))

        achieved = self._fk_tcp(self._urdf_fol, q_motor)
        pos_err = float(np.linalg.norm(achieved.translation() - T_tcp_target.translation()))
        ori_err = float(
            np.linalg.norm((achieved.rotation().inverse() @ T_tcp_target.rotation()).log())
        )
        on_target = pos_err <= self._ik_pos_tol and ori_err <= self._ik_ori_tol
        sane = pos_err <= self._ik_reject_pos
        if not sane:
            logger.warning(
                "IK solution rejected: pos_err=%.0f mm exceeds the %.0f mm sanity bound "
                "— holding last solution",
                pos_err * 1e3,
                self._ik_reject_pos * 1e3,
            )
        elif not on_target:
            # Throttled: this fires continuously while the operator leans on a
            # joint limit, which is a normal thing to do.
            now = time.monotonic()
            if now - self._last_ik_warn_t > 2.0:
                self._last_ik_warn_t = now
                logger.info(
                    "IK at workspace edge: pos_err=%.0f mm ori_err=%.0f mrad — tracking "
                    "the closest reachable pose",
                    pos_err * 1e3,
                    ori_err * 1e3,
                )
        return q_motor, on_target, sane

    # ── Observation helpers ───────────────────────────────────────────────────

    @staticmethod
    def _measured(obs: Dict[str, Any], arm: str) -> Optional[np.ndarray]:
        """Measured follower joints (motor order, arm only) for *arm*."""
        arm_obs = obs.get(arm)
        if not isinstance(arm_obs, dict):
            return None
        jp = arm_obs.get("joint_pos")
        if jp is None:
            return None
        jp = np.asarray(jp, dtype=np.float64).ravel()
        return jp[:ARM_DOF] if jp.size >= ARM_DOF else None

    @staticmethod
    def _policy_cmd(obs: Dict[str, Any], arm: str) -> Optional[np.ndarray]:
        """Policy's commanded 7-vector for *arm*, as published on the bus."""
        msg = obs.get(f"policy_{arm}")
        if not isinstance(msg, dict):
            return None
        jp = msg.get("joint_pos")
        if jp is None:
            return None
        jp = np.asarray(jp, dtype=np.float64).ravel()
        return jp[:CMD_DOF] if jp.size >= CMD_DOF else None

    @staticmethod
    def _policy_ts(obs: Dict[str, Any], arm: str) -> float:
        return float(obs.get("_topic_ts", {}).get(f"policy_{arm}", 0.0))

    def _leader_cmd(self, arm: str) -> np.ndarray:
        """Leader's 7-vector (6 arm joints in YAM space + normalized gripper)."""
        return np.asarray(self._leaders[arm].act({})[arm]["pos"], dtype=np.float64)

    def _leader_stale(self, arm: str) -> bool:
        return self._leaders[arm].seconds_since_last_message() > self._gello_stale_s

    # ── Mode transitions ──────────────────────────────────────────────────────

    def _poll_pause_edge(self, obs: Dict[str, Any]) -> None:
        """Start an episode when the arms go live, end it when they're gated.

        Unpause is the handoff to the policy, so the episode has to begin there
        rather than on a separate operator action. Ending on pause is what keeps
        the latch honest: ``episode_timeout`` ends the episode *and* re-pauses,
        and a manual [space] mid-rollout should not leave a dangling episode that
        the next press would try to "end" a second time.
        """
        paused = obs.get("_paused")
        if paused is None or not self._record_on_unpause:
            return
        paused = bool(paused)
        prev, self._paused_prev = self._paused_prev, paused
        if prev is None or prev == paused:
            return
        if not paused:                      # paused → running: episode starts
            if not self._record_latch:
                self._record_latch = True
                logger.info("episode START (unpause — handing arms to the policy)")
            if self._mode not in _LIVE_MODES:
                # Blend out of the parked pose rather than stepping to whatever
                # the policy currently wants.
                self._enter_handback(obs)
        else:                               # running → paused: episode ends
            if self._record_latch:
                self._record_latch = False
                logger.info("episode END (pause, saving)")
            # Park the mode, but do NOT drive a homing motion: RobotNode is gated
            # while paused, so the arms cannot move. Drop the command cache so the
            # next resume re-seeds from the measured pose instead of a stale one.
            self._mode = MODE_IDLE if self._home_on_episode_end else MODE_POLICY
            self._cmd.clear()
            self._goal_filtered.clear()
            self._anchors.clear()
            self._ik_q.clear()

    def _poll_rehome_request(self, obs: Dict[str, Any]) -> bool:
        """Consume a new session/rehome request exactly once."""
        request = obs.get("rehome_request")
        request_ts = obs.get("_topic_ts", {}).get("rehome_request")
        if not isinstance(request, dict) or not request.get("request") or request_ts is None:
            return False
        if request_ts == self._last_rehome_request_ts:
            return False
        self._last_rehome_request_ts = float(request_ts)
        self._record_latch = False
        self._policy_reset_pending = True
        logger.info("episode END (saving) — TUI save+home request")
        self._enter_homing(obs)
        return True

    def _poll_button_edges(self) -> Dict[int, bool]:
        """Record per-leader rising edges and return their per-index OR."""
        edges = {0: False, 1: False}
        self._button_edges_by_arm = {(arm, i): False for arm in self.arms for i in (0, 1)}
        for arm in self.arms:
            try:
                levels = self._leaders[arm].get_buttons()
            except Exception:
                levels = (False, False)
            for i in (0, 1):
                pressed = bool(levels[i])
                rising = pressed and not self._prev_button[(arm, i)]
                self._button_edges_by_arm[(arm, i)] = rising
                edges[i] = edges[i] or rising
                self._prev_button[(arm, i)] = pressed
        return edges

    def _enter_intervention(self, obs: Dict[str, Any]) -> None:
        """Latch leader + follower anchors so the correction starts at identity."""
        self._anchors.clear()
        for arm in self.arms:
            # Anchor the follower on the last command we PUBLISHED, not on the
            # measured state: the follower lags its command under load, and
            # anchoring on the lagging value would step the command backwards by
            # the tracking error at the instant of takeover.
            q_ref = self._cmd.get(arm)
            if q_ref is None:
                q_ref = self._policy_cmd(obs, arm)
            if q_ref is None:
                meas = self._measured(obs, arm)
                if meas is None:
                    logger.error("Cannot take over %s: no command or state yet", arm)
                    return
                q_ref = np.concatenate([meas, [1.0]])
            self._cmd.setdefault(arm, np.asarray(q_ref, dtype=np.float64).copy())
            self._anchors[arm] = _ArmAnchor(
                T_lead_0=self._fk_tcp(self._urdf_lead, self._leader_cmd(arm)),
                T_fol_0=self._fk_tcp(self._urdf_fol, q_ref),
            )
            # Start the IK loop on the anchor pose: at t=0 the delta is identity,
            # so this is exactly the solution and the first solve is a no-op.
            self._ik_q[arm] = np.asarray(q_ref, dtype=np.float64)[:ARM_DOF].copy()
            self._goal_filtered.pop(arm, None)

        self._mode = MODE_INTERVENTION
        self._takeover_count += 1
        self._switch_t = time.monotonic()
        logger.info(
            "TAKEOVER #%d (%s button) — operator has control",
            self._takeover_count,
            BUTTON_COLOURS.get(self._button_index, "?"),
        )

    def _enter_homing(self, obs: Dict[str, Any]) -> None:
        """End the rollout and walk the arms back to the parked pose."""
        if not self._home_on_episode_end or self._home is None:
            self._mode = MODE_IDLE
            self._switch_t = time.monotonic()
            return
        # Seed the slew from the last command so homing starts continuous with
        # whatever the policy or the operator was doing a tick ago.
        for arm in self.arms:
            if arm not in self._cmd:
                meas = self._measured(obs, arm)
                if meas is not None:
                    self._cmd[arm] = np.concatenate([meas, self._home[arm][-1:]])
            self._goal_filtered.pop(arm, None)
        self._anchors.clear()
        self._mode = MODE_HOMING
        self._switch_t = time.monotonic()
        logger.info("HOMING — returning to parked pose at %.1f rad/s", self._homing_speed)

    def _step_homing(self, obs: Dict[str, Any], dt: float) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        reached = True
        for arm in self.arms:
            goal = self._home[arm]
            if arm not in self._cmd:
                meas = self._measured(obs, arm)
                if meas is None:
                    reached = False
                    continue
                self._cmd[arm] = np.concatenate([meas, goal[-1:]])
            q = self._slew(arm, goal, dt, max_speed=self._homing_speed)
            out[arm] = q
            # Arm joints only: the gripper is normalized units, not radians, and
            # it reaches its endpoint on its own schedule.
            if np.abs(q[:ARM_DOF] - goal[:ARM_DOF]).max() > self._home_tol:
                reached = False
        if reached:
            self._mode = MODE_IDLE
            self._switch_t = time.monotonic()
            logger.info("PARKED at home — safe to reset the scene; white button starts the next episode")
        return out

    def _step_idle(self) -> Dict[str, np.ndarray]:
        """Hold the parked pose.

        Keeps *commanding* home rather than going silent, so RobotNode sees a
        continuous stream and doesn't treat the next rollout as a resume-after-gap
        (which would re-trigger its handoff ramp on top of our own blend).
        """
        out: Dict[str, np.ndarray] = {}
        if self._home is None:
            return out
        for arm in self.arms:
            self._cmd[arm] = self._home[arm].copy()
            self._goal_filtered.pop(arm, None)
            out[arm] = self._cmd[arm]
        return out

    def _enter_handback(self, obs: Dict[str, Any]) -> None:
        """Request a policy chunk flush and hold pose until a fresh command lands."""
        self._handback_from = {
            arm: self._cmd[arm].copy() for arm in self.arms if arm in self._cmd
        }
        self._policy_ts_at_handback = {arm: self._policy_ts(obs, arm) for arm in self.arms}
        self._handback_t0 = None
        self._handback_requested_t = time.monotonic()
        self._policy_reset_pending = True
        self._mode = MODE_HANDBACK
        self._switch_t = time.monotonic()
        logger.info("HANDBACK — flushing policy chunk, blending over %.2f s", self._handback_blend_s)

    # ── Per-mode command construction ─────────────────────────────────────────

    def _step_intervention(self, obs: Dict[str, Any], dt: float) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        self._ik_ok = True
        for arm in self.arms:
            anchor = self._anchors.get(arm)
            held = self._cmd.get(arm)
            if anchor is None or held is None:
                continue
            if self._leader_stale(arm):
                # Hold, don't hand back: re-arming the policy without the
                # operator asking is a worse outcome than a frozen arm.
                logger.warning("leader %s CAN stale — holding last command", arm)
                out[arm] = held
                self._ik_ok = False
                continue

            q_lead = self._leader_cmd(arm)
            T_lead_t = self._fk_tcp(self._urdf_lead, q_lead)

            # Base-frame delta: translation adds, rotation left-composes.
            p_target = anchor.T_fol_0.translation() + (
                T_lead_t.translation() - anchor.T_lead_0.translation()
            )
            R_target = (
                T_lead_t.rotation() @ anchor.T_lead_0.rotation().inverse()
            ) @ anchor.T_fol_0.rotation()
            T_target = vtf.SE3.from_rotation_and_translation(R_target, p_target)

            # Seed from the last accepted SOLUTION, not from the command — see
            # the _ik_q comment in __init__. A rejected solve holds the solution
            # (so the arm parks at the reachable boundary) but the seed stays
            # there too, so the operator can always move back and recover.
            seed = self._ik_q.get(arm, held[:ARM_DOF])
            q_ik, on_target, sane = self._ik_tcp(T_target, seed)
            if sane:
                # Track even when off-target: that is the operator leaning on a
                # joint limit, and the solution is the closest reachable pose.
                self._ik_q[arm] = q_ik
                if not on_target:
                    self._ik_ok = False
            else:
                self._ik_ok = False
                q_ik = self._ik_q.get(arm, held[:ARM_DOF])

            gripper = q_lead[ARM_DOF] if q_lead.size > ARM_DOF else held[-1]
            goal = np.concatenate([q_ik, [gripper]])
            out[arm] = self._slew(arm, goal, dt)
        return out

    def _step_handback(self, obs: Dict[str, Any], dt: float) -> Dict[str, np.ndarray]:
        now = time.monotonic()

        # Start the blend only once the policy has published a command derived
        # from a post-flush observation; until then hold the operator's pose.
        if self._handback_t0 is None:
            fresh = all(
                self._policy_ts(obs, arm) > self._policy_ts_at_handback.get(arm, 0.0)
                and self._policy_cmd(obs, arm) is not None
                for arm in self.arms
            )
            timed_out = (now - self._handback_requested_t) > self._handback_fresh_timeout_s
            if fresh or timed_out:
                if timed_out and not fresh:
                    logger.warning(
                        "no fresh policy command %.2f s after handback — blending to latest anyway",
                        self._handback_fresh_timeout_s,
                    )
                self._handback_t0 = now
            else:
                return {arm: q for arm, q in self._handback_from.items()}

        alpha = 1.0 if self._handback_blend_s <= 0 else (now - self._handback_t0) / self._handback_blend_s
        alpha = float(np.clip(alpha, 0.0, 1.0))

        out: Dict[str, np.ndarray] = {}
        for arm in self.arms:
            start = self._handback_from.get(arm)
            target = self._policy_cmd(obs, arm)
            if start is None:
                if target is not None:
                    out[arm] = target
                continue
            if target is None:
                out[arm] = start
                continue
            # Blend toward the LIVE policy command, so the policy is already
            # driving by the time alpha reaches 1 and there is no second step.
            blended = (1.0 - alpha) * start + alpha * target
            out[arm] = blended
            self._cmd[arm] = blended
            self._goal_filtered.pop(arm, None)

        if alpha >= 1.0:
            self._mode = MODE_POLICY
            self._switch_t = now
            logger.info("policy has control")
        return out

    def _step_policy(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for arm in self.arms:
            cmd = self._policy_cmd(obs, arm)
            if cmd is None:
                continue
            # Track the passthrough so a takeover can anchor on it, and so the
            # slew limiter resumes from the right place.
            self._cmd[arm] = cmd.copy()
            self._goal_filtered.pop(arm, None)
            out[arm] = cmd
        return out

    # ── Slew limiter ──────────────────────────────────────────────────────────

    def _slew(
        self, arm: str, goal: np.ndarray, dt: float, max_speed: Optional[float] = None
    ) -> np.ndarray:
        """Advance this arm's command toward *goal* under the speed limit.

        Filter the *goal*, then rate-limit the approach to it. That ordering
        keeps the two knobs independent — filtering the output instead would
        scale each clamped increment by alpha, silently dividing the speed limit
        by ~tau/dt. (Same reasoning as YamPyrokiViserAgent._slew.)
        """
        cur = self._cmd.get(arm)
        if cur is None:
            cur = goal.copy()

        tau = self._smoothing_tau_s
        smoothed = self._goal_filtered.get(arm)
        if smoothed is None or tau <= 0.0:
            smoothed = goal.copy()
        else:
            smoothed = smoothed + (dt / (tau + dt)) * (goal - smoothed)
        self._goal_filtered[arm] = smoothed

        v_max = self._max_joint_speed if max_speed is None else max_speed
        if v_max <= 0.0:
            cur = smoothed.copy()
        else:
            limit = np.full(goal.shape, v_max * dt)
            limit[-1] = self._max_gripper_speed * dt if self._max_gripper_speed > 0 else np.inf
            cur = cur + np.clip(smoothed - cur, -limit, limit)

        self._cmd[arm] = cur
        return cur

    # ── Agent protocol ────────────────────────────────────────────────────────

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        now = time.monotonic()
        # Clamp dt so a stalled tick can't authorize an unbounded jump.
        dt = 0.01 if self._last_act_t is None else float(np.clip(now - self._last_act_t, 1e-4, 0.05))
        self._last_act_t = now

        self._poll_pause_edge(obs)
        rehome_requested = self._poll_rehome_request(obs)

        edges = self._poll_button_edges()

        # A dedicated hardware equivalent of the TUI space bar. The AgentNode
        # publishes this one-shot request and Session owns the actual gate.
        pause_toggle_requested = False
        if self._pause_button_index is not None:
            if self._pause_button_arm is None:
                pause_toggle_requested = edges[self._pause_button_index]
            else:
                pause_toggle_requested = self._button_edges_by_arm[
                    (self._pause_button_arm, self._pause_button_index)
                ]
            if pause_toggle_requested:
                logger.info(
                    "session play/pause toggle requested — %s %s button",
                    self._pause_button_arm or "either",
                    BUTTON_COLOURS.get(self._pause_button_index, "?"),
                )

        # ── Yellow: takeover / hand back. Only meaningful mid-rollout; ignored
        # while parked, where there is no episode to correct.
        if not rehome_requested and edges[self._button_index]:
            if self._mode == MODE_INTERVENTION:
                self._enter_handback(obs)
            elif self._mode in (MODE_POLICY, MODE_HANDBACK):
                self._enter_intervention(obs)
            else:
                logger.info(
                    "takeover ignored — arms are %s; press the %s button to start an episode",
                    self._mode,
                    BUTTON_COLOURS.get(self._episode_button_index, "episode"),
                )

        # ── Left white: rollout save/home. The right white switch is kept out
        # of this path because it owns the separate Session pause toggle.
        episode_edge = False
        if self._episode_button_index is not None:
            if self._episode_button_arm is None:
                episode_edge = edges[self._episode_button_index]
            else:
                episode_edge = self._button_edges_by_arm[(self._episode_button_arm, self._episode_button_index)]
        if not rehome_requested and self._episode_button_index is not None and episode_edge:
            colour = BUTTON_COLOURS.get(self._episode_button_index, "?")
            if self._mode in _LIVE_MODES:
                self._record_latch = False
                self._policy_reset_pending = True
                logger.info("episode END (saving) — %s button", colour)
                self._enter_homing(obs)
            else:
                self._record_latch = True
                logger.info("episode START — %s button, handing arms to the policy", colour)
                # Route via HANDBACK so the parked→policy step is flushed and
                # blended rather than jumped, same as a mid-rollout hand back.
                self._enter_handback(obs)

        if self._mode == MODE_INTERVENTION:
            cmds = self._step_intervention(obs, dt)
        elif self._mode == MODE_HANDBACK:
            cmds = self._step_handback(obs, dt)
        elif self._mode == MODE_HOMING:
            cmds = self._step_homing(obs, dt)
        elif self._mode == MODE_IDLE:
            cmds = self._step_idle()
        else:
            cmds = self._step_policy(obs)

        action: Dict[str, Any] = {
            arm: {"pos": np.asarray(q, dtype=np.float32)} for arm, q in cmds.items()
        }

        extras: Dict[str, Any] = {
            "control_mode": {
                "mode": self._mode,
                # Redundant scalar, but it makes filtering an episode down to
                # operator-authored timesteps a one-liner downstream.
                "intervention": self._mode == MODE_INTERVENTION,
                # True through the whole rollout (policy + intervention +
                # handback), False while parked. Lets a consumer drop the homing
                # and parked stretches without enumerating mode strings.
                "live": self._mode in _LIVE_MODES,
                "recording": self._record_latch,
                "takeover_count": self._takeover_count,
                "since_switch_s": now - self._switch_t,
                "ik_ok": bool(self._ik_ok),
            }
        }
        # One-shot: the consumer edge-detects on message timestamp, so repeating
        # reset=True would flush the policy's chunk on every tick of the blend.
        if self._policy_reset_pending:
            extras["policy_reset"] = {"reset": True}
            self._policy_reset_pending = False
        if pause_toggle_requested:
            extras["pause_toggle"] = {"toggle": True}
        action["_extras"] = extras
        # Session's monitor loop edge-detects this latch on record_topic:
        # False→True starts an episode, True→False ends and saves it.
        if self._episode_button_index is not None or self._record_on_unpause:
            action["_record"] = self._record_latch
        return action

    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        return {arm: {"pos": Array(shape=(CMD_DOF,), dtype=np.float32)} for arm in self.arms}

    def reset(self) -> None:
        self._mode = MODE_IDLE if self._home_on_episode_end else MODE_POLICY
        self._anchors.clear()
        self._ik_q.clear()
        self._cmd.clear()
        self._goal_filtered.clear()
        self._handback_from.clear()
        self._handback_t0 = None
        self._policy_reset_pending = False
        self._last_act_t = None
        self._prev_button = {(arm, i): False for arm in self.arms for i in (0, 1)}
        # Deliberately NOT clearing _record_latch or _paused_prev: reset() runs on
        # node startup and could be re-triggered later, and dropping either would
        # end the operator's episode out from under them (or re-fire a stale
        # pause edge).
        self._ik_ok = True

    def close(self) -> None:
        for leader in self._leaders.values():
            leader.close()


__all__ = ["DaggerInterventionAgent"]
