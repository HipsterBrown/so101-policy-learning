"""Replay a trained ppo.py checkpoint on the Mac (CPU sim), in the SAPIEN viewer or to MP4s.

Usage:
  python scripts/watch_policy.py path/to/final_ckpt.pt                  # viewer window
  python scripts/watch_policy.py path/to/final_ckpt.pt --video videos/policy --episodes 5
Options: --env SO101Reach-v1 (default) | SO101Lift-v1, --episodes N, --seed S.
"""
import argparse
import importlib.util
from pathlib import Path

import gymnasium as gym
import torch

from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import so101.envs  # noqa: F401  registers the envs

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--env", default="SO101Reach-v1")
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--video", help="write MP4s to this dir instead of opening the viewer")
args = p.parse_args()

# Agent (the MLP) lives in the vendored ppo.py; its training code only runs under __main__.
spec = importlib.util.spec_from_file_location("ppo", ROOT / "scripts/ppo.py")
ppo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppo)

env = gym.make(args.env, num_envs=1, obs_mode="state", render_mode="rgb_array" if args.video else "human")
if args.video:
    env = RecordEpisode(env, output_dir=args.video, save_trajectory=False, video_fps=20)
env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)
agent = ppo.Agent(env)
agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))  # obs dim mismatch here = checkpoint from another control mode; missing obs_mean/obs_var = Phase 1 checkpoint (pre obs-norm), retrain

for ep in range(args.episodes):
    obs, _ = env.reset(seed=args.seed + ep)
    done, hit = False, False
    while not done:
        with torch.no_grad():
            obs, _, term, trunc, info = env.step(agent.get_action(obs, deterministic=True))
        hit |= bool(info["success"].item())
        done = bool((term | trunc).item())
        if not args.video:
            env.unwrapped.render()
    print(f"episode {ep}: success={hit}")
env.close()
