import gymnasium as gym
import numpy as np
import torch

import so101.envs  # noqa: F401  registers the envs
from so101.ik import ik, load_chain


def _make(env_id, **kw):
    return gym.make(env_id, num_envs=1, obs_mode="state", **kw)


def test_obs_is_26_floats():
    for env_id in ["SO101Reach-v1", "SO101Lift-v1"]:
        env = _make(env_id)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (1, 26), (env_id, obs.shape)
        obs, *_ = env.step(env.action_space.sample())
        assert obs.shape == (1, 26)
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
