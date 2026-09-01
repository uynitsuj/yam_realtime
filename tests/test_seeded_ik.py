"""FK -> IK roundtrip + zero-delta check for the seeded IK."""
import numpy as np, time
import yourdfpy, pyroki as pk, viser.transforms as vtf
from robots_realtime.robots.inverse_kinematics.pyroki_snippets._solve_ik_seeded import solve_ik_seeded

URDF = "dependencies/i2rt/i2rt/robot_models/arm/yam/yam.urdf"
EE, BASE = "link_6", "base_link"
urdf = yourdfpy.URDF.load(URDF, load_meshes=False, build_scene_graph=True)
robot = pk.Robot.from_urdf(urdf)
print("actuated:", robot.joints.num_actuated_joints, "urdf order:", urdf.actuated_joint_names)

def fk(q_motor):
    """q in i2rt motor order (6,) -> link_6 SE3 in base frame."""
    urdf.update_cfg(np.flip(np.asarray(q_motor, dtype=np.float64)))
    return vtf.SE3.from_matrix(np.asarray(urdf.get_transform(EE, BASE)))

rng = np.random.default_rng(0)
# Sample inside the follower's configured joint_limits (robot_configs/yam/xdof_hq/left.yaml)
LIM = np.array([[-2.09,3.14],[0.0,3.14],[0.05,3.14],[-1.35,1.35],[-1.50,1.50],[-2.00,2.00]])

t0=time.perf_counter(); _=solve_ik_seeded(robot, EE, np.array([1.,0,0,0]), np.array([0.3,0.,0.3]), np.zeros(6)); 
print(f"JIT warmup: {time.perf_counter()-t0:.2f}s")

# --- roundtrip: FK(q) -> IK seeded near q -> q' should match q
errs_q, errs_p, times = [], [], []
for _ in range(30):
    q = rng.uniform(LIM[:,0], LIM[:,1])
    T = fk(q)
    seed = np.flip(q + rng.normal(0, 0.05, 6))   # URDF order, perturbed
    t0=time.perf_counter()
    sol = solve_ik_seeded(robot, EE, T.rotation().wxyz, T.translation(), seed)
    times.append((time.perf_counter()-t0)*1e3)
    q2 = np.flip(sol)
    errs_q.append(np.abs(q2-q).max())
    errs_p.append(np.linalg.norm(fk(q2).translation()-T.translation()))
print(f"roundtrip  max|dq| median={np.median(errs_q)*1e3:.2f} mrad  p95={np.percentile(errs_q,95)*1e3:.2f} mrad")
print(f"roundtrip  pos err  median={np.median(errs_p)*1e3:.3f} mm    max={max(errs_p)*1e3:.3f} mm")
print(f"solve time median={np.median(times):.2f} ms  p95={np.percentile(times,95):.2f} ms")

# --- zero-delta: target == FK(seed) must return the seed exactly-ish (the takeover instant)
worst = 0.0
for _ in range(20):
    q = rng.uniform(LIM[:,0], LIM[:,1])
    T = fk(q)
    sol = solve_ik_seeded(robot, EE, T.rotation().wxyz, T.translation(), np.flip(q))
    worst = max(worst, np.abs(np.flip(sol)-q).max())
print(f"zero-delta max|dq| = {worst*1e3:.4f} mrad  (must be ~0 -> no jump at takeover)")
