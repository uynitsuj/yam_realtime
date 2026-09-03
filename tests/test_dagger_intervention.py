"""Offline check of DaggerInterventionAgent: FSM + relative-SE(3) continuity.

No CAN, no arms. Fake leaders let us drive the button and the leader joints
directly, so we can measure the command step at every mode transition -- the
thing that would show up as a lunge on real hardware.
"""
import numpy as np, time, sys
import robots_realtime.agents.teleoperation.dagger_intervention_agent as D

class FakeLeader:
    def __init__(self, channel, robot_name, **kw):
        self.robot_name = robot_name
        self.q = np.array([0.1, 0.9, 0.5, 0.2, -0.1, 0.3, 1.0])
        self.buttons = (False, False)
        self.stale = 0.0
    def act(self, obs): return {self.robot_name: {"pos": self.q.copy()}}
    def get_buttons(self): return self.buttons
    def seconds_since_last_message(self): return self.stale
    def close(self): pass

D.PassiveGelloLeaderAgent = FakeLeader
HOME = np.array([0.0, 0.4, 0.35, 0.35, 0.0, 0.0, 1.0])
FOLLOWER_LIMITS = np.array([
    [-2.09, 3.14], [0.0, 3.14], [0.05, 3.14],
    [-1.35, 1.35], [-1.50, 1.50], [-2.00, 2.00],
])
agent = D.DaggerInterventionAgent(episode_button_arm="left",
                                  handback_blend_s=0.3, handback_fresh_timeout_s=0.5,
                                  home_joint_pos=list(HOME), home_on_episode_end=True,
                                  homing_max_joint_speed=8.0, home_tol_rad=0.03,
                                  joint_limits=FOLLOWER_LIMITS, joint_limit_margin=0.1)
L = agent._leaders

POLICY = {"left": np.array([0.0, 0.4, 0.35, 0.35, 0.0, 0.0, 1.0]),
          "right": np.array([0.0, 0.5, 0.30, 0.40, 0.1, 0.0, 0.5])}
ts = [1000.0]
paused = [True]          # session starts paused (start_paused: true)
def obs(drift=0.0):
    ts[0] += 0.01
    return {
        "_paused": paused[0],
        "left":  {"joint_pos": POLICY["left"][:6]},
        "right": {"joint_pos": POLICY["right"][:6]},
        "policy_left":  {"joint_pos": POLICY["left"] + drift},
        "policy_right": {"joint_pos": POLICY["right"] + drift},
        "_topic_ts": {"policy_left": ts[0], "policy_right": ts[0]},
    }

CTRL_KEYS = ("_extras", "_record")

def arms_only(action):
    """Strip the control keys, leaving {arm: pos_array}."""
    return {k: v["pos"].copy() for k, v in action.items() if k not in CTRL_KEYS}

def run(n, drift=0.0, sleep=0.002):
    out = []
    for _ in range(n):
        a = agent.act(obs(drift))
        out.append((arms_only(a), a["_extras"]["control_mode"]["mode"]))
        time.sleep(sleep)
    return out

def press(index, n=1):
    """Pulse a button: hold for one act(), then release."""
    for a in ("left", "right"):
        L[a].buttons = tuple(i == index for i in (0, 1))
    r = run(n)
    for a in ("left", "right"):
        L[a].buttons = (False, False)
    return r

def latch():
    for a in ("left", "right"): L[a].buttons = (False, False)
    return agent.act(obs()).get("_record")

fail = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond: fail.append(name)

def arm_mrad(a, b):
    """Max ARM joint step, mrad. Excludes element 6: the gripper is a normalized
    [0,1] command, not radians, so folding it in mixes units."""
    return max(np.abs(a[k][:6] - b[k][:6]).max() for k in a) * 1e3

def grip_step(a, b):
    return max(abs(a[k][6] - b[k][6]) for k in a)

print("\n[0a] follower limits constrain IK with a soft margin")
expected_lower = FOLLOWER_LIMITS[:, 0] + 0.1
expected_upper = FOLLOWER_LIMITS[:, 1] - 0.1
check("motor-order guards use the configured 0.1 rad margin",
      np.allclose(agent._joint_guards[:, 0], expected_lower)
      and np.allclose(agent._joint_guards[:, 1], expected_upper))
check("Pyroki receives the same limits in reversed URDF order",
      np.allclose(np.asarray(agent._robot.joints.lower_limits), np.flip(expected_lower))
      and np.allclose(np.asarray(agent._robot.joints.upper_limits), np.flip(expected_upper)))
check("post-solve guard clips numerical leakage",
      np.allclose(agent._guard_arm_joints(np.full(6, 99.0)), expected_upper))

def go_live():
    """Leave the parked state and settle into POLICY (via the handback blend)."""
    paused[0] = False
    agent._paused_prev = False
    agent._mode = "policy"
    agent._handback_t0 = None
    run(3)

print("\n[0] starts parked at home, not recording")
check("initial mode is idle", agent._mode == "idle")
check("not recording while parked", agent.act(obs()).get("_record") is False)
parked = arms_only(agent.act(obs()))
check("commands the home pose while parked",
      all(np.allclose(parked[a], HOME, atol=1e-9) for a in ("left", "right")))

print("\n[1] policy passthrough")
go_live()
h = run(5)
check("mode is policy", all(m == "policy" for _, m in h))
check("command == policy command", np.allclose(h[-1][0]["left"], POLICY["left"], atol=1e-6))

print("\n[2] takeover with the leader FAR from the follower (the naive-impl failure)")
# Leader parked somewhere unrelated -- absolute passthrough would snap here.
for a in ("left", "right"):
    L[a].q = np.array([1.4, 2.2, 1.1, -0.9, 0.8, -1.2, 0.2])
last_policy = h[-1][0]
naive = max(np.abs(L[a].q - last_policy[a]).max() for a in ("left", "right")) * 1e3
for a in ("left", "right"): L[a].buttons = (True, False)   # takeover = index 0
h2 = run(1)
for a in ("left", "right"): L[a].buttons = (False, False)
check("mode is intervention", h2[0][1] == "intervention")
jump = arm_mrad(h2[0][0], last_policy)
check("no arm snap at takeover", jump < 1.0,
      f"arm step={jump:.4f} mrad  (absolute passthrough would have been {naive:.0f} mrad)")
gj = grip_step(h2[0][0], last_policy)
check("gripper moves only at its slew rate", gj <= 4.0 * 0.05 + 1e-9, f"grip step={gj:.4f}")

print("\n[3] relative tracking: move the leader, follower should follow the DELTA")
T_fol_0 = {a: agent._anchors[a].T_fol_0 for a in ("left", "right")}
T_lead_0 = {a: agent._anchors[a].T_lead_0 for a in ("left", "right")}
for a in ("left", "right"): L[a].q = L[a].q + np.array([0.0, -0.12, 0.05, 0.0, 0.0, 0.0, 0.0])
h3 = run(400, sleep=0.0)   # let the slew limiter converge
for a in ("left", "right"):
    lead_d = agent._fk_tcp(agent._urdf_lead, L[a].q).translation() - T_lead_0[a].translation()
    fol_d = agent._fk_tcp(agent._urdf_fol, h3[-1][0][a]).translation() - T_fol_0[a].translation()
    err = np.linalg.norm(lead_d - fol_d)
    check(f"{a}: follower TCP delta tracks leader TCP delta", err < 3e-3,
          f"|lead_d|={np.linalg.norm(lead_d)*1e3:.1f}mm err={err*1e3:.2f}mm")

print("\n[4] gripper is absolute but slew-limited")
for a in ("left", "right"): L[a].q[6] = 0.0     # full squeeze from 0.2
g0 = h3[-1][0]["left"][6]
h4 = run(1, sleep=0.0)
dg = abs(h4[0][0]["left"][6] - g0)
check("gripper does not jump in one tick", dg < 0.1, f"dg={dg:.4f} per tick")
h4b = run(300, sleep=0.0)
check("gripper reaches the commanded value", abs(h4b[-1][0]["left"][6] - 0.0) < 0.02,
      f"final={h4b[-1][0]['left'][6]:.4f}")

print("\n[5] handback: flush requested, pose held until fresh policy cmd, then blended")
before = h4b[-1][0]
for a in ("left", "right"): L[a].buttons = (True, False)   # takeover = index 0
a_hb = agent.act(obs())
for a in ("left", "right"): L[a].buttons = (False, False)
check("mode is handback", a_hb["_extras"]["control_mode"]["mode"] == "handback")
check("policy_reset published exactly once", "policy_reset" in a_hb["_extras"])
check("reset not repeated on next tick", "policy_reset" not in agent.act(obs())["_extras"])
step = arm_mrad(arms_only(a_hb), before)
check("no arm step at handback instant", step < 1e-6, f"step={step:.6f} mrad")
# Policy target deliberately far away, to prove the blend is what protects us.
h5 = run(250, drift=0.6, sleep=0.002)   # >0.45 s wall > handback_blend_s=0.3
worst = max(arm_mrad(h5[i][0], h5[i-1][0]) for i in range(1, len(h5)))
check("mode returned to policy", h5[-1][1] == "policy")
check("blend is monotone with no jump", worst < 40.0, f"worst per-tick step={worst:.2f} mrad")
check("ends on the policy command",
      np.allclose(h5[-1][0]["left"], POLICY["left"] + 0.6, atol=1e-3))

print("\n[6] stale leader holds instead of handing back")
for a in ("left", "right"): L[a].buttons = (True, False)   # takeover = index 0
run(1); 
for a in ("left", "right"): L[a].buttons = (False, False)
held = run(1)[-1][0]
for a in ("left", "right"): L[a].stale = 5.0
for a in ("left", "right"): L[a].q += 0.5      # leader moves while "dead"
h6 = run(10)
check("still in intervention", h6[-1][1] == "intervention")
check("command frozen while CAN is stale", arm_mrad(h6[-1][0], held) < 1e-9
                                                  and grip_step(h6[-1][0], held) < 1e-9)
check("ik_ok flagged false", agent._ik_ok is False)
for a in ("left", "right"): L[a].stale = 0.0   # revive the leaders for later sections

print("\n[7] episode button (index 1) toggles the record latch")
agent.reset(); agent._record_latch = False
agent._paused_prev = None; paused[0] = False; latch()   # settle the pause edge
agent._mode = "policy"                                  # rollout underway
check("latch starts False", latch() is False)
press(1); check("press 1 in a rollout -> latch False (end+save)", latch() is False)
check("...and the arms start homing", agent._mode in ("homing", "idle"))
press(1); check("press 1 while parked -> latch True (next episode)", latch() is True)
agent._mode = "policy"
press(1); check("press 1 again -> latch False", latch() is False)
# A held button must toggle exactly ONCE, on the press. The switch is momentary,
# so level-triggering would flip the latch on every tick it stays down.
was = latch()
for a in ("left", "right"): L[a].buttons = (False, True)
run(10)                                     # press and hold
mid = agent.act(obs()).get("_record")       # still held
check("holding toggles exactly once", mid is (not was), f"{was} -> {mid}")
run(20)                                     # keep holding
still = agent.act(obs()).get("_record")
check("holding longer does not toggle again", still is mid)
for a in ("left", "right"): L[a].buttons = (False, False)
check("release alone does not toggle", latch() is mid)

print("\n[8] button responsibilities don't overlap")
# White owns the episode + parking; yellow owns who is driving. White DOES move
# the mode -- that's the rollout cycle -- but it must never hand control to the
# operator, and yellow must never touch the recording.
agent.reset(); agent._record_latch = False
agent._paused_prev = None; paused[0] = False; latch()

agent._mode = "idle"
press(1)
check("white from parked starts recording", latch() is True)
check("...and does NOT hand control to the operator", agent._mode != "intervention")

agent._mode = "policy"
press(0)
check("yellow takes over", agent._mode == "intervention")
check("yellow does not touch the latch", latch() is True)

press(1)
check("white ends the episode even mid-intervention", latch() is False)
check("...and takes control back for homing", agent._mode in ("homing", "idle"))

print("\n[10] unpause starts an episode, pause ends it")
agent.reset(); agent._record_latch = False; agent._paused_prev = None
paused[0] = True
check("nothing recorded while paused", latch() is False)
check("first tick is a baseline, not an edge", latch() is False)
paused[0] = False                                  # [space] -> arms live
check("unpause starts the episode", latch() is True)
paused[0] = True                                   # [space] -> arms gated
check("pause ends the episode", latch() is False)
paused[0] = False
check("unpause starts the next episode", latch() is True)

print("\n[11] pause/unpause mid-episode does not double-fire")
# Re-pausing while already recording must end exactly once, and re-unpausing
# must start exactly once -- no transition on a repeated level.
for _ in range(5): latch()
check("holding unpaused keeps the latch True", latch() is True)
paused[0] = True; check("pause ends once", latch() is False)
for _ in range(5): latch()
check("holding paused keeps it False", latch() is False)

print("\n[12] episode_timeout leaves the latch synced")
# Session's _on_episode_timeout does end_episode(save=True) THEN pause(). The
# agent sees only the pause edge; the latch must land False so the next [space]
# is a clean start rather than a no-op.
paused[0] = False; latch()                         # episode running
check("episode running", agent._record_latch is True)
paused[0] = True                                   # Session ends + pauses
check("latch dropped to match Session", latch() is False)
paused[0] = False
check("next unpause starts a fresh episode", latch() is True)

print("\n[13] white button and pause edges cooperate")
paused[0] = False; agent._record_latch = True; agent._paused_prev = False
press(1); check("white ends episode while unpaused", latch() is False)
check("...without changing pause state", paused[0] is False)
press(1); check("white starts the next one", latch() is True)
paused[0] = True; check("pause still ends it", latch() is False)

print("\n[14] white button runs the full rollout cycle")
agent.reset(); agent._record_latch = False
agent._paused_prev = False; paused[0] = False
agent._cmd.clear(); agent._goal_filtered.clear()
run(2)
check("parked at start", agent._mode == "idle")
press(1)
check("white starts the episode", agent._record_latch is True)
check("...and hands off via handback", agent._mode in ("handback", "policy"))
run(250, sleep=0.002)
check("settles into policy", agent._mode == "policy")
# Take over FIRST, then move the leader: the correction is relative, so a leader
# repositioned before the press produces no motion at all (that is the point).
press(0)
check("intervened away from home", agent._mode == "intervention")
for a in ("left", "right"): L[a].q = L[a].q + np.array([0.0, -0.45, 0.30, 0.0, 0.0, 0.0, 0.0])
# Real wall-clock: the slew limiter advances by max_joint_speed * dt, and with
# sleep=0 the measured dt is ~0.2 ms, so no-sleep ticks barely move the arms.
run(200, sleep=0.002)
away = arms_only(agent.act(obs()))
dist = max(np.abs(away[a][:6] - HOME[:6]).max() for a in ("left", "right"))
check("arms are away from home", dist > 0.3, f"max|q-home|={dist:.3f} rad")

press(1)
check("white ends the episode", agent._record_latch is False)
check("...and starts homing", agent._mode == "homing")
h = run(400, sleep=0.002)      # 0.8 s at 8 rad/s -- ample travel
check("reaches home and parks", agent._mode == "idle", f"mode={agent._mode}")
final = h[-1][0]
err = max(np.abs(final[a][:6] - HOME[:6]).max() for a in ("left", "right"))
check("parked pose == home", err < 0.03, f"max|q-home|={err*1e3:.1f} mrad")
# Homing must be rate-limited, not a jump.
worst = max(arm_mrad(h[i][0], h[i-1][0]) for i in range(1, len(h)))
check("homing is rate-limited", worst <= 8.0 * 0.05 * 1e3 + 1.0, f"worst step={worst:.1f} mrad")
check("not recording while parked", latch() is False)

print("\n[15] yellow is inert while parked")
before = agent._mode
press(0)
check("takeover ignored when parked", agent._mode == before)
check("...and does not start recording", latch() is False)

print("\n[16] home_on_episode_end requires a home pose")
try:
    D.DaggerInterventionAgent(home_on_episode_end=True, home_joint_pos=None)
    check("missing home_joint_pos raises", False)
except ValueError:
    check("missing home_joint_pos raises", True)
try:
    D.DaggerInterventionAgent(home_on_episode_end=True, home_joint_pos=[0.0] * 6)
    check("wrong-length home_joint_pos raises", False)
except ValueError:
    check("wrong-length home_joint_pos raises", True)

print("\n[9] misconfiguration is rejected")
try:
    D.DaggerInterventionAgent(takeover_button_index=1, episode_button_index=1)
    check("same index for both buttons raises", False)
except ValueError as e:
    check("same index for both buttons raises", True, f"({e.__class__.__name__})")


print("\n[18] right lower button is inert; TUI rehome request saves and homes")
agent.reset(); agent._mode = "policy"; agent._record_latch = True
agent._paused_prev = False; paused[0] = False
L["right"].buttons = (False, True)
run(1)
L["right"].buttons = (False, False)
run(1)
check("right lower button does not end the episode",
      agent._record_latch is True and agent._mode == "policy")
request_obs = obs()
request_obs["rehome_request"] = {"request": True}
request_obs["_topic_ts"]["rehome_request"] = 12345.0
request_action = agent.act(request_obs)
check("TUI request drops the recording latch", request_action.get("_record") is False)
check("TUI request starts homing", agent._mode in ("homing", "idle"))
check("TUI request flushes policy state", "policy_reset" in request_action["_extras"])
repeat_action = agent.act(request_obs)
check("same TUI request is consumed once", "policy_reset" not in repeat_action["_extras"])


# ── [17] regression: the IK loop must not latch up ───────────────────────────
# Seeding the solver from the slew-limited COMMAND (instead of from the last
# accepted solution) couples the two loops: the limiter holds the command back,
# so the seed lags, so the target looks unreachable, so the solve is rejected,
# so the command never advances and the gap grows without bound. The symptom was
# a follower frozen mid-motion and a tracking gain that fell as hand speed rose
# -- which reads as "the IK scale is wrong" rather than as a control bug.
print("\n[17] IK tracks 1:1 and does not latch up")
ag2 = D.DaggerInterventionAgent(home_joint_pos=list(HOME), home_on_episode_end=False,
                                max_joint_speed=6.0, smoothing_tau_s=0.01,
                                handback_blend_s=0.1)
L2 = ag2._leaders
ts2 = [2000.0]
def obs2():
    ts2[0] += 0.01
    return {"_paused": False,
            "left": {"joint_pos": HOME[:6]}, "right": {"joint_pos": HOME[:6]},
            "policy_left": {"joint_pos": HOME}, "policy_right": {"joint_pos": HOME},
            "_topic_ts": {"policy_left": ts2[0], "policy_right": ts2[0]}}
ag2._paused_prev = False; ag2._mode = "policy"
for _ in range(3): ag2.act(obs2()); time.sleep(0.01)
for a in ("left", "right"): L2[a].buttons = (True, False)
ag2.act(obs2()); time.sleep(0.01)
for a in ("left", "right"): L2[a].buttons = (False, False)
anchor = {a: (ag2._anchors[a].T_lead_0, ag2._anchors[a].T_fol_0) for a in ("left", "right")}
q0 = {a: L2[a].q.copy() for a in ("left", "right")}

# Real wall-clock at the configured 100 Hz, driving joint 2 UP -- away from its
# 0.0 lower limit -- so the workspace edge is not what binds.
t0 = time.monotonic(); act = None
while True:
    el = time.monotonic() - t0
    if el > 1.0: break
    for a in ("left", "right"):
        L2[a].q = q0[a].copy(); L2[a].q[1] = q0[a][1] + 1.5 * el
    act = ag2.act(obs2()); time.sleep(0.01)

a = "left"
lead_d = np.linalg.norm(ag2._fk_tcp(ag2._urdf_lead, L2[a].q).translation()
                        - anchor[a][0].translation())
fol_d = np.linalg.norm(ag2._fk_tcp(ag2._urdf_fol, act[a]["pos"]).translation()
                       - anchor[a][1].translation())
gain = fol_d / max(lead_d, 1e-9)
check("leader actually moved a long way", lead_d > 0.20, f"{lead_d*1e3:.0f} mm")
check("tracking gain is ~1:1 at 1.5 rad/s hand speed", gain > 0.95,
      f"gain={gain:.3f}  lag={(lead_d-fol_d)*1e3:.1f} mm")
check("follower did not freeze", fol_d > 0.20, f"{fol_d*1e3:.0f} mm")
ag2.close()

print("\n" + ("ALL CHECKS PASSED" if not fail else f"FAILED: {fail}"))
sys.exit(1 if fail else 0)
