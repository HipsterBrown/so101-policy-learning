"""Phase 0b gate: IK-scripted top-down grasp and lift of the block, 10 seeds, >= 8 must succeed.

Usage: python scripts/scripted_grasp.py
Writes one MP4 per seed to videos/scripted_grasp/.
"""
import sys

import gymnasium as gym
import numpy as np
import torch

from mani_skill.utils.wrappers.record import RecordEpisode

import so101.envs  # noqa: F401
from so101.agent import GRIPPER_CLOSED, GRIPPER_OPEN
from so101.ik import fk, grasp_pose, ik, load_chain, tilt_deg

SEEDS = 10
PASS_AT = 8
MAX_LIFT_TILT = 15.0
YAW_TRIM = 0.0  # Viam tool frame is yawed 2.8 deg from the jaw face
# (name, steps) per segment; 125 total, well under the 200-step episode below.
SEGMENTS = [("open", 10), ("pre", 30), ("descend", 20), ("close", 15), ("lift", 30), ("hold", 20)]

chain = load_chain()
env = gym.make("SO101Lift-v1", num_envs=1, obs_mode="state", control_mode="pd_joint_pos",
               render_mode="rgb_array", max_episode_steps=200)
env = RecordEpisode(env, output_dir="videos/scripted_grasp", save_trajectory=False, video_fps=20)


def plan(block_p, block_yaw, q_now):
    """Waypoints (pre, grasp, lift) or None. Tries the 4 yaws equivalent under the cube's 90 deg symmetry."""
    best = None
    for k in range(4):
        yaw = (block_yaw + k * np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
        qp, okp = ik(chain, grasp_pose(block_p + [0, 0, 0.03], yaw), q_now)
        qg, okg = ik(chain, grasp_pose(block_p, yaw), qp)
        ql, okl = ik(chain, grasp_pose(block_p + [0, 0, 0.06], yaw), qg, rot_weight=0.1)
        if okp and okg and okl and tilt_deg(fk(chain, ql)) <= MAX_LIFT_TILT:
            cost = abs(qg[4] - q_now[4])  # least wrist-roll travel
            if best is None or cost < best[0]:
                best = (cost, qp, qg, ql)
    return None if best is None else best[1:]


successes = 0
for seed in range(SEEDS):
    env.reset(seed=seed)
    u = env.unwrapped
    p = u.block.pose.p[0].cpu().numpy()
    w, _, _, z = u.block.pose.q[0].cpu().numpy()
    q_now = u.agent.robot.get_qpos()[0].cpu().numpy()[:5]
    wp = plan(p, 2 * np.arctan2(z, w), q_now)
    if wp is None:
        print(f"seed {seed}: IK failed for every yaw -> FAIL")
        continue
    qp, qg, ql = wp
    targets = {"open": (q_now, GRIPPER_OPEN), "pre": (qp, GRIPPER_OPEN), "descend": (qg, GRIPPER_OPEN),
               "close": (qg, GRIPPER_CLOSED), "lift": (ql, GRIPPER_CLOSED), "hold": (ql, GRIPPER_CLOSED)}
    prev = np.r_[q_now, GRIPPER_OPEN]
    ever_success = ever_grasped = False
    for name, steps in SEGMENTS:
        goal = np.r_[targets[name][0], targets[name][1]]
        for s in range(1, steps + 1):
            action = prev + (goal - prev) * s / steps
            _, _, _, _, info = env.step(torch.tensor(action, dtype=torch.float32)[None])
            ever_success |= bool(info["success"].item())
            ever_grasped |= bool(info["is_grasped"].item())
        prev = goal
    lift_cm = (u.block.pose.p[0, 2].item() - p[2]) * 100
    print(f"seed {seed}: success={ever_success} grasped={ever_grasped} block_raised={lift_cm:.1f} cm")
    successes += ever_success

env.close()
print(f"\n{successes}/{SEEDS} succeeded (gate: >= {PASS_AT})")
sys.exit(0 if successes >= PASS_AT else 1)
