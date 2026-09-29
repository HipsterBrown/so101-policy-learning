"""Phase 0a gate: limits vs URDF and Viam, zero-pose render, FK cross-check vs Viam, reachability.

Usage: python scripts/check_model.py [--gui]
Exits non-zero if any gate fails.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import so101.envs as E
from so101.agent import TCP_P
from so101.ik import fk, grasp_pose, ik, load_chain, tilt_deg

ROOT = Path(__file__).resolve().parents[1]
chain = load_chain()
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


env = gym.make("SO101Lift-v1", num_envs=1, obs_mode="state",
               render_mode="human" if "--gui" in sys.argv else "rgb_array")
env.reset(seed=0)
u = env.unwrapped
robot = u.agent.robot

# 1. Limits: sim as loaded vs TheRobotStudio URDF (all 6 incl. gripper) vs Viam so101.json (arm joints 1-5)
# ponytail: no real-arm readings yet; compare against a hand-jogged real_arm.json before Phase 5 sim-to-real.
sim_lim = np.degrees(robot.get_qlimits()[0].cpu().numpy())  # (6, 2)
viam_lim = np.degrees(chain.limits)
urdf = ET.parse(ROOT / "assets/so101/so101_new_calib.urdf").getroot()
urdf_lim = {j.get("name"): np.degrees([float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))])
            for j in urdf.findall("joint") if j.get("type") == "revolute"}
print("joint          sim[min,max]        urdf[min,max]       viam[min,max]")
for j, name in enumerate(u.agent.arm_joint_names + u.agent.gripper_joint_names):
    v = viam_lim[j] if j < 5 else np.array([np.nan, np.nan])
    print(f"{name:>13} {sim_lim[j].round(1)!s:>18} {urdf_lim[name].round(1)!s:>18} {v.round(1)!s:>18}")
    check(f"{name} sim == URDF limits", np.allclose(sim_lim[j], urdf_lim[name], atol=0.1))
    if j < 5:
        check(f"{name} sim == Viam limits", np.allclose(sim_lim[j], v, atol=0.1))

# 2. Zero-pose render (compare with a photo of the real arm at its set_homing pose)
robot.set_qpos(np.zeros(6))
if "--gui" in sys.argv:
    print("Viewer open: tick 'collision' in the Render panel to inspect the jaw boxes. Close the window to continue.")
    while not u.render_human().closed:
        pass
else:
    imageio.imwrite(ROOT / "videos/zero_pose.png", u.render()[0].cpu().numpy())
    print("wrote videos/zero_pose.png")

# 3. FK cross-check: SAPIEN gripper_link relative to base_link vs Viam fk(q) * C
# (SAPIEN's pinocchio model needs the extra `pin` package on macOS; on CPU, set_qpos updates link poses directly.)
links = robot.links_map


def link_T(name):
    """Pose of a link relative to base_link, as a 4x4 numpy matrix."""
    return (links["base_link"].pose.inv() * links[name].pose).to_transformation_matrix()[0].cpu().numpy()


def sapien_gl(q5):
    robot.set_qpos(torch.tensor(np.r_[q5, 0.0], dtype=torch.float32)[None])
    return link_T("gripper_link")


C = np.linalg.inv(fk(chain, np.zeros(5), tcp=None)) @ sapien_gl(np.zeros(5))
R_tool_inv = np.linalg.inv(chain.fixed[-1][:3, :3])
c_err = np.degrees(np.linalg.norm(R.from_matrix(R_tool_inv.T @ C[:3, :3]).as_rotvec()))
check("C equals inverse of Viam tool rotation", c_err < 0.5 and np.linalg.norm(C[:3, 3]) < 1e-4,
      f"(rot err {c_err:.3f} deg, trans {np.linalg.norm(C[:3, 3]) * 1000:.3f} mm)")
rng = np.random.default_rng(0)
pos_err, rot_err = [], []
for q in rng.uniform(chain.limits[:, 0], chain.limits[:, 1], size=(50, 5)):
    A, B = sapien_gl(q), fk(chain, q, tcp=None) @ C
    pos_err.append(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000)
    rot_err.append(np.degrees(np.linalg.norm(R.from_matrix(B[:3, :3].T @ A[:3, :3]).as_rotvec())))
check("FK cross-check", max(pos_err) < 1.0 and max(rot_err) < 0.5,
      f"(max pos {max(pos_err):.3f} mm, max rot {max(rot_err):.3f} deg)")
robot.set_qpos(torch.zeros(1, 6))
gfl_in_gl = np.linalg.inv(link_T("gripper_link")) @ link_T("gripper_frame_link")
print(f"GripperTCPPose vs upstream gripper_frame_link: {np.linalg.norm(gfl_in_gl[:3, 3] - TCP_P) * 1000:.1f} mm (info, expect ~2)")

# 4. Reach-box coverage, position-only
lo, hi = np.array(E.REACH_LO), np.array(E.REACH_HI)
hits = 0
for p in rng.uniform(lo, hi, size=(1000, 3)):
    T = np.eye(4)
    T[:3, 3] = p
    hits += ik(chain, T, np.zeros(5), rot_weight=0)[1]
check("reach box >= 99% position-reachable", hits >= 990, f"({hits / 10:.1f}%)")

# 5. Lift grasp coverage: grasp + pre-grasp top-down, lift with tilt <= 15 deg
nx, ny = E.LIFT_NOMINAL_XY
r = E.LIFT_XY_RANGE
bad = []
for x in np.linspace(nx - r, nx + r, 5):
    for y in np.linspace(ny - r, ny + r, 5):
        for yaw in np.radians([-45, 0, 45]):
            g = np.array([x, y, E.CUBE_HALF])
            qg, okg = ik(chain, grasp_pose(g, yaw), np.zeros(5))
            _, okp = ik(chain, grasp_pose(g + [0, 0, 0.03], yaw), qg)
            ql, okl = ik(chain, grasp_pose(g + [0, 0, 0.06], yaw), qg, rot_weight=0.1)
            okl = okl and tilt_deg(fk(chain, ql)) <= 15
            if not (okg and okp and okl):
                bad.append((round(x, 3), round(y, 3), round(np.degrees(yaw)), okg, okp, okl))
check("lift spawn region 100% graspable", not bad, f"({len(bad)} failing of 75)")
for b in bad[:10]:
    print("   fail x,y,yaw,grasp,pre,lift:", b)

env.close()
print("\nALL PASS" if not failures else f"\nFAILED: {failures}")
sys.exit(1 if failures else 0)
