import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.structs import Pose

import so101.envs  # noqa: F401  registers the envs
from so101.ik import ik, load_chain


def _make(env_id, **kw):
    return gym.make(env_id, num_envs=1, obs_mode="state", **kw)


def test_obs_is_32_floats():
    # qpos 6, qvel 6, controller target qpos 6, tcp 3, block pose 7, target 3, is_grasped 1
    for env_id in ["SO101Reach-v1", "SO101Lift-v1"]:
        env = _make(env_id)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (1, 32), (env_id, obs.shape)
        obs, *_ = env.step(env.action_space.sample())
        assert obs.shape == (1, 32)
        env.close()


def test_reach_success_when_ik_puts_tcp_on_target():
    # Ties the Viam IK chain to the sim TCP: a frame mismatch > 2 cm fails this.
    env = _make("SO101Reach-v1", control_mode="pd_joint_pos")
    env.reset(seed=0)
    u = env.unwrapped
    T = np.eye(4)
    T[:3, 3] = u.goal.pose.p[0].cpu().numpy()
    q, ok = ik(load_chain(), T, np.zeros(5), rot_weight=0)
    assert ok
    u.agent.robot.set_qpos(torch.tensor(np.r_[q, 0.0], dtype=torch.float32)[None])
    info = u.get_info()
    assert info["success"].item(), info["dist"].item()
    env.close()


def test_lift_not_successful_at_reset():
    env = _make("SO101Lift-v1")
    _, info = env.reset(seed=0)
    assert not info["success"].item()
    env.close()


def test_lift_hold_counts_once_per_step():
    # Extra get_info()/get_obs() calls (as in vector-env resets) must not advance the hold counter.
    env = _make("SO101Lift-v1")
    env.reset(seed=0)
    u = env.unwrapped
    u.agent.is_grasping = lambda obj, **kw: torch.ones(1, dtype=torch.bool)
    up = Pose.create_from_pq(p=torch.tensor([[0.2, 0.0, 0.015 + 0.15]]))
    for step in range(1, 11):
        u.block.set_pose(up)
        u.block.set_linear_velocity(torch.zeros(1, 3))  # else gravity speeds it up and it falls out of range
        _, _, _, _, info = env.step(torch.zeros(1, 6))
        u.get_info()
        u.get_info()
        assert info["success"].item() == (step == 10), step
    env.close()


def test_lift_reward_trace_with_fast_oracle():
    # Staged reward climbs through the scripted grasp and success pays exactly R_SUCCESS (spec §3).
    from so101.envs import R_SUCCESS, W_ACTION
    from so101.oracle import fast_grasp
    env = _make("SO101Lift-v1", reward_mode="dense")
    out = fast_grasp(env, seed=0)
    env.close()
    assert out["success"], out["first_success_step"]
    steps = out["steps"]
    pre = [s for s in steps[: out["first_success_step"]] if not s["is_grasped"]]
    assert pre[-1]["reward"] > pre[0]["reward"]
    min_action = -W_ACTION * 6  # clipped actions: ||a||^2 <= 6
    for s in steps:
        if s["success"]:
            assert s["reward"] == R_SUCCESS
            assert s["grasp_err"] < 0.005  # grasp_center sits on the held block's centre
        elif s["is_grasped"] and s["lifted"]:
            assert s["reward"] >= 2.5 + min_action  # reach + grasp + lift near goal (+ static when still)
        elif s["is_grasped"]:
            assert s["reward"] >= 1.5 + min_action  # reach + grasp + lift at rest (1 - tanh(0.3) = 0.71): fails if a term drops
        assert min_action / R_SUCCESS <= s["reward"] / R_SUCCESS <= 1
