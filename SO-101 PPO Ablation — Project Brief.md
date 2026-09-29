# SO-101 PPO Ablation — Project Brief

Sep 28, 2026 · @Nick Hehr

## Overview

Train a PPO policy in simulation that lifts a block with the SO-101, then run an ablation study to learn which PPO details and design choices actually drive performance. Real-arm deployment is the stretch goal.

- **Goal:** understand PPO by building it, then measure what each component contributes.
- **Scope (weekend or two):** sim setup, reach task, block-lift task, single-file PPO rewrite, ablation matrix.
- **Stretch:** sim-to-real on the physical SO-101 with an AprilTag-tracked block.
- **Non-goals:** vision-based policies (a later phase), Isaac Lab, pencils or other thin objects, production-quality code.

## Stack and compute

ManiSkill3 officially ships only the SO-100, not the SO-101, so the SO-101 model has to come from a community project or be imported from TheRobotStudio's URDF. That decision gates Phase 0.

| Path | SO-101 model | GPU parallel | PPO baseline | Notes |
| --- | --- | --- | --- | --- |
| ManiSkill3 official | SO-100 only ([robot list](https://maniskill.readthedocs.io/en/latest/robots/index.html)) | Yes, T4 works | CleanRL-style single file | Kinematics differ from SO-101; fine for learning, weaker for sim-to-real |
| ManiSkill3 + community SO-101 | Squint project (SO-101, ManiSkill3, sim-to-real visual RL) ([topic](https://github.com/topics/so-101)) | Yes | Squint's own | Built for vision; check how much is reusable for state-based PPO |
| ManiSkill3 + custom import | SO-101 URDF from [TheRobotStudio](https://github.com/TheRobotStudio/SO-ARM100) ([custom robot guide](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_robots.html)) | Yes | CleanRL-style single file | You define controllers and gripper friction yourself, which takes about half a day |
| [SO101-Nexus](https://github.com/johnsutor/so101-nexus) (MuJoCo) | Native SO-101, PickLift task included | Experimental MuJoCo Warp backend, NVIDIA only | BC and PPO baselines, Colab notebook | Beta; the current README is MuJoCo-only |

**Compute plan:** develop and watch replays on the Mac, train on a Colab T4. Keep training in a `.py` script with CLI flags; the notebook only calls it. Save checkpoints to Google Drive so a disconnect doesn't lose a run.

**Deployment:** the Jetson hosts the overhead camera, AprilTag detection and policy inference, and sends joint targets to the Go servo module.

## Environment spec

Two tasks share one observation and action design, so only the reward and success test change between them.

|  | Reach | Lift block |
| --- | --- | --- |
| Goal | Gripper tip within 2 cm of a random target point | Block (about 3 cm cube) raised 5 cm above the table |
| Episode length | 50 steps | 100 steps |
| Success | Distance < 2 cm | Block height > 5 cm and held for 10 steps |
| Randomized at reset | Target position in the workspace | Block position ±10 cm, yaw ±90° |

**Observations (privileged state):** 6 joint positions, 6 joint velocities, gripper tip position, block position and orientation, target position, and whether the block is between the fingers. Each episode is the same fixed-size vector.

**Actions (baseline):** a small change to each joint's target angle per step, clipped to a safe range, sent to a simulated position controller. This maps directly onto the goal-position commands your Go servo module sends.

**Reward for lift (staged, baseline):**

1. **Reach:** bigger the closer the gripper is to the block.
2. **Grasp:** a bonus while the block is between closed fingers.
3. **Lift:** bigger the higher the block, only while grasped.
4. **Success:** a large one-time bonus.
5. **Penalty:** a small cost for large actions, so movements stay smooth enough for real servos.

Start from the task's built-in reward if the chosen stack provides one, and write down each term's weight before changing anything.

## Ablation matrix

Seven variants × 3 seeds = 21 runs on the lift task, each changing exactly one thing from the baseline. Each is a CLI flag in the single-file PPO script.

| Variant | Change from baseline | Question it answers |
| --- | --- | --- |
| Baseline | None | Reference point |
| Sparse reward | Only the success bonus | How much does reward shaping carry learning? |
| No grasp term | Drop reward stage 2 | Does the policy need to be told grasping matters? |
| No obs normalization | Raw observations | How sensitive is PPO to input scale? |
| No advantage normalization | Raw advantages | Does per-batch scaling of the learning signal matter here? |
| GAE λ = 1.0 | No smoothing of the "was this action good" estimate | Noisy vs smoothed credit assignment |
| No entropy bonus | Entropy coefficient 0 | Does it still explore enough to find the grasp? |

**Metrics per run:** final success rate over 100 eval episodes with deterministic actions, and environment steps until 80% success (sample efficiency). Report the mean and the min–max across seeds.

**Reading results:** treat a difference as real only when the seed ranges don't overlap. If they overlap, add 2 more seeds before concluding.

**Optional eighth variant:** end-effector position actions via your IK instead of joint changes. This is a bigger change, so run it after the core seven.

## Phases

Phases run in order; each one's success criterion gates the next. Phases 0–4 fit the weekend budget, Phase 5 likely spills over.

- [ ] **Phase 0: Sim check (half a day).** Pick the SO-101 model path from the stack table, load it, and compare its joint limits and zero pose with your zeroed real arm.
  - Gate: a scripted grasp using your IK lifts the block in sim.
- [ ] **Phase 1: Reach with a library (a few hours).** Train SB3 PPO or the stack's baseline on the reach task.
  - Gate: ≥95% success on 100 eval episodes.
- [ ] **Phase 2: Lift baseline (half a day).** Staged reward, 3 seeds on Colab.
  - Gate: ≥80% success on all 3 seeds.
- [ ] **Phase 3: Single-file PPO (a day).** Write or adapt a CleanRL-style `ppo.py` with every ablation as a flag.
  - Gate: reproduces Phase 2 within the seed range.
- [ ] **Phase 4: Ablations (a day, mostly waiting).** Run the 21-run matrix and build the results table.
  - Gate: every variant has 3 finished seeds and both metrics.
- [ ] **Phase 5: Sim-to-real (stretch).** Add randomization of block mass, friction and servo delay; run on the real arm via the Jetson with AprilTag tracking.
  - Gate: success rate measured over 20 real trials.

* [ ] **Phase 6: PPO vs SAC + HER (optional, about a day).** Same lift env and Colab T4. ManiSkill's SAC baseline is CleanRL-based like its PPO ([baselines](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/baselines.html)); HER isn't included, so add goal relabeling yourself: put the target lift height in the observation and recompute the reward for relabeled goals.
  - Runs: PPO sparse, SAC sparse, SAC + HER sparse, SAC dense, 3 seeds each.
  - Gate: a plot of success rate vs environment steps and vs wall-clock time for all four.

## Risks and open questions

| Risk | Fallback |
| --- | --- |
| Lift never learns (common with PPO grasping) | Switch to push-to-goal, or warm-start from a few demos as SO101-Nexus does |
| Simulated gripper slips or jitters | Tune finger friction in Phase 0 before any training |
| Colab disconnects mid-matrix | Resumable checkpoints on Drive; a rented cloud GPU VM for the full matrix |
| Sim servos are faster and stiffer than STS3215s | Action penalty, servo delay randomization in Phase 5 |

- [ ] Which SO-101 model path: community ManiSkill3, custom import, or SO101-Nexus on MuJoCo?
- [ ] Is the Jetson an Xavier NX or Orin NX? It affects camera and tag-detection throughput, not training.
- [ ] Where do results live: W&B, TensorBoard, or a CSV per run?

## Sources

- [ManiSkill robots list](https://maniskill.readthedocs.io/en/latest/robots/index.html)
- [ManiSkill custom robots tutorial](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_robots.html)
- [SO101-Nexus](https://github.com/johnsutor/so101-nexus)
- [GitHub so-101 topic (Squint)](https://github.com/topics/so-101)
- [TheRobotStudio SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
