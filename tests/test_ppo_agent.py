import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

_spec = importlib.util.spec_from_file_location("ppo", Path(__file__).resolve().parents[1] / "scripts/ppo.py")
ppo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ppo)


class _Envs:  # Agent only reads the single spaces
    single_observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
    single_action_space = gym.spaces.Box(-1, 1, (2,), np.float32)


def test_obs_stats_normalize_a_known_batch():
    agent = ppo.Agent(_Envs())
    x = torch.randn(5000, 4) * torch.tensor([1.0, 2.0, 0.5, 3.0]) + torch.tensor([5.0, -1.0, 0.0, 2.0])
    agent.update_obs_stats(x)
    z = agent.normalize(x)
    assert torch.allclose(z.mean(0), torch.zeros(4), atol=0.05)
    assert torch.allclose(z.std(0), torch.ones(4), atol=0.05)


def test_obs_stats_merge_matches_full_batch():
    a, b = ppo.Agent(_Envs()), ppo.Agent(_Envs())
    x = torch.randn(2000, 4) * 3 + 1
    a.update_obs_stats(x)
    b.update_obs_stats(x[:700])
    b.update_obs_stats(x[700:])
    assert torch.allclose(a.obs_mean, b.obs_mean, atol=1e-4) and torch.allclose(a.obs_var, b.obs_var, atol=1e-3)


def test_obs_stats_merge_of_shifted_batches_matches_ground_truth():
    agent = ppo.Agent(_Envs())
    x = torch.randn(2000, 4)
    agent.update_obs_stats(x[:700])
    agent.update_obs_stats(x[700:] + 10)  # large delta between batches exercises the delta**2 term
    full = torch.cat([x[:700], x[700:] + 10])
    assert torch.allclose(agent.obs_mean, full.mean(0), rtol=1e-4)
    assert torch.allclose(agent.obs_var, full.var(0, unbiased=False), rtol=1e-3)


def test_obs_stats_survive_state_dict_round_trip():
    a = ppo.Agent(_Envs())
    a.update_obs_stats(torch.randn(100, 4) + 3)
    b = ppo.Agent(_Envs())
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.obs_mean, b.obs_mean) and torch.equal(a.obs_var, b.obs_var)


def test_norm_off_is_identity():
    agent = ppo.Agent(_Envs(), norm_obs=False)
    x = torch.randn(10, 4) * 50
    agent.update_obs_stats(x)
    assert torch.equal(agent.normalize(x), x)
