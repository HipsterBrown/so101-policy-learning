"""Scripted top-down grasp: waypoint planning plus a fast oracle in the training action space.

`plan` is shared by scripts/scripted_grasp.py (pd_joint_pos, Phase 0b gate) and `fast_grasp`
(pd_joint_target_delta_pos, Phase 2 episode-length check and reward-trace tests).
"""
import numpy as np
import torch

from so101.agent import GRIPPER_CLOSED, GRIPPER_OPEN
from so101.ik import fk, grasp_pose, ik, load_chain, tilt_deg

CHAIN = load_chain()
MAX_LIFT_TILT = 15.0
MAX_DELTA = np.array([0.05] * 5 + [0.2])  # pd_joint_target_delta_pos bounds in agent.py
ARM_TOL = 0.002  # m, TCP FK distance to the waypoint, on actual qpos
GRIP_STALL = 0.005  # rad per step; 2 steps below this after the target reaches CLOSED ends the close
PHASE_TIMEOUT = 40


def plan(block_p, block_yaw, q_now):
    """Waypoints (pre, grasp, lift) or None. Tries the 4 yaws equivalent under the cube's 90 deg symmetry."""
    best = None
    for k in range(4):
        yaw = (block_yaw + k * np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
        qp, okp = ik(CHAIN, grasp_pose(block_p + [0, 0, 0.03], yaw), q_now)
        qg, okg = ik(CHAIN, grasp_pose(block_p, yaw), qp)
        ql, okl = ik(CHAIN, grasp_pose(block_p + [0, 0, 0.06], yaw), qg, rot_weight=0.1)
        if okp and okg and okl and tilt_deg(fk(CHAIN, ql)) <= MAX_LIFT_TILT:
            cost = abs(qg[4] - q_now[4])  # least wrist-roll travel
            if best is None or cost < best[0]:
                best = (cost, qp, qg, ql)
    return None if best is None else best[1:]


def fast_grasp(env, seed, max_steps=100):
    """Max-rate scripted grasp under pd_joint_target_delta_pos on a num_envs=1 SO101Lift-v1 env.

    Returns {success, first_success_step, steps}; steps[i] = dict(reward, is_grasped, lifted, success,
    grasp_err) for env step i+1, where grasp_err = |grasp_center - block| in metres.
    """
    env.reset(seed=seed)
    u = env.unwrapped
    p = u.block.pose.p[0].cpu().numpy()
    w, _, _, z = u.block.pose.q[0].cpu().numpy()
    q0 = u.agent.robot.get_qpos()[0].cpu().numpy()
    out = dict(success=False, first_success_step=None, steps=[])
    wp = plan(p, 2 * np.arctan2(z, w), q0[:5])
    if wp is None:
        return out
    qp, qg, ql = wp
    phases = [("pre", qp, GRIPPER_OPEN), ("grasp", qg, GRIPPER_OPEN), ("close", qg, GRIPPER_CLOSED),
              ("lift", ql, GRIPPER_CLOSED), ("hold", ql, GRIPPER_CLOSED)]
    ph, in_phase, stall, prev_g = 0, 0, 0, q0[5]
    for t in range(1, max_steps + 1):
        name, q_arm, g = phases[ph]
        target = u.agent.controller._target_qpos[0].cpu().numpy()
        action = np.clip((np.r_[q_arm, g] - target) / MAX_DELTA, -1, 1)
        _, reward, _, trunc, info = env.step(torch.tensor(action, dtype=torch.float32)[None])
        success = bool(info["success"].item())
        grasp_err = torch.linalg.norm(u.agent.grasp_center - u.block.pose.p, axis=1)[0].item()
        out["steps"].append(dict(reward=float(reward.item()), is_grasped=bool(info["is_grasped"].item()),
                                 lifted=bool(info["lifted"].item()), success=success, grasp_err=grasp_err))
        if success and not out["success"]:
            out["success"], out["first_success_step"] = True, t
        q = u.agent.robot.get_qpos()[0].cpu().numpy()
        in_phase += 1
        if name in ("pre", "grasp"):
            done = np.linalg.norm(fk(CHAIN, q[:5])[:3, 3] - fk(CHAIN, q_arm)[:3, 3]) <= ARM_TOL
        elif name == "close":
            closed = u.agent.controller._target_qpos[0, 5].item() <= GRIPPER_CLOSED + 1e-6
            stall = stall + 1 if closed and abs(q[5] - prev_g) < GRIP_STALL else 0
            done = stall >= 2
        elif name == "lift":
            done = success  # the lift waypoint is never reached within ARM_TOL: wrist_flex saturates
        else:
            done = False
        prev_g = q[5]
        if (done or in_phase >= PHASE_TIMEOUT) and ph < len(phases) - 1:
            ph, in_phase, stall = ph + 1, 0, 0
        if bool(trunc.item()):
            break
    return out
