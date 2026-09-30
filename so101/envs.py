"""SO101Reach-v1 and SO101Lift-v1: same 32-float state obs and joint target-delta action, different reset/success/reward."""
from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig

import so101.agent  # noqa: F401  registers "so101"
from so101.agent import GRIPPER_OPEN

# Reach targets, arm base frame (m). Shrink if check_model.py reports < 99% position-reachable.
REACH_LO = (0.15, -0.15, 0.02)
REACH_HI = (0.30, 0.15, 0.20)
# Lift block spawn (spec: +/-5 cm around the nominal point; the brief's +/-10 cm doesn't fit the reachable band).
CUBE_HALF = 0.015
LIFT_NOMINAL_XY = (0.20, 0.0)
LIFT_XY_RANGE = 0.05
LIFT_HEIGHT = 0.05
HOLD_STEPS = 10
# Lift reward saturates LIFT_GOAL above spawn: 1 cm over the `lifted` line, below the scripted lift's 6.2-6.3 cm
# (wrist_flex saturates near its limit there). A two-sided goal on the line would reward hovering where `lifted` flickers.
LIFT_GOAL = 0.06
# Reward weights (Phase 2 spec §1), fixed before tuning; log every change in the spec's Change log.
W_REACH, W_GRASP, W_LIFT, W_STATIC, W_ACTION = 1.0, 1.0, 1.0, 1.0, 0.01
R_SUCCESS = W_REACH + W_GRASP + W_LIFT + W_STATIC + 1.0  # success always pays the most; also the normalizer
QPOS_NOISE = 0.02


class SO101BaseEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["so101"]
    block = None  # Lift sets this; Reach reports zeros in its place

    def __init__(self, *args, robot_uids="so101", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.45, 0.35, 0.35], [0.2, 0.0, 0.05])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose())  # base_link at the origin

    def _load_scene(self, options: dict):
        # Base mesh dips 2.4 mm below base_link into the ground; harmless, the root is fixed.
        self.ground = build_ground(self.scene)

    def _reset_robot(self, b):
        qpos = torch.zeros(b, 6)
        qpos[:, :5] += torch.randn(b, 5) * QPOS_NOISE
        qpos[:, 5] = GRIPPER_OPEN
        self.agent.reset(qpos)

    def _get_obs_extra(self, info: dict):
        b = self.num_envs
        return dict(
            tcp_pos=self.agent.tcp_pos,
            block_pose=self.block.pose.raw_pose if self.block is not None else torch.zeros(b, 7, device=self.device),
            target_pos=self.target_pos,
            is_grasped=info["is_grasped"].float(),
        )


@register_env("SO101Reach-v1", max_episode_steps=50)
class SO101ReachEnv(SO101BaseEnv):
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.goal = actors.build_sphere(
            self.scene, radius=0.01, color=[0, 1, 0, 1], name="goal",
            body_type="kinematic", add_collision=False, initial_pose=sapien.Pose(),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_robot(b)
            lo, hi = torch.tensor(REACH_LO), torch.tensor(REACH_HI)
            self.goal.set_pose(Pose.create_from_pq(p=lo + torch.rand(b, 3) * (hi - lo)))

    @property
    def target_pos(self):
        return self.goal.pose.p

    def evaluate(self):
        dist = torch.linalg.norm(self.agent.tcp_pos - self.goal.pose.p, axis=1)
        return dict(success=dist < 0.02, dist=dist,
                    is_grasped=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reward = 1 - torch.tanh(5 * info["dist"])
        reward[info["success"]] = 2.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 2.0


@register_env("SO101Lift-v1", max_episode_steps=100)
class SO101LiftEnv(SO101BaseEnv):
    @property
    def _default_sim_config(self):
        # The 5 g cube pinched between two drive-held links needs more solver iterations than the default 15,
        # or the jaws sink into it (~2 mm per control step at 15; ~0.1 mm steady state at 50).
        return SimConfig(sim_freq=100, control_freq=20, scene_config=SceneConfig(solver_position_iterations=50))

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # Density and friction from ManiSkill's SO100GraspCube-v1 digital twin.
        material = sapien.pysapien.physx.PhysxMaterial(static_friction=0.3, dynamic_friction=0.3, restitution=0)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[CUBE_HALF] * 3, material=material, density=200)
        builder.add_box_visual(half_size=[CUBE_HALF] * 3,
                               material=sapien.render.RenderMaterial(base_color=[0.05, 0.16, 0.63, 1]))
        builder.initial_pose = sapien.Pose(p=[*LIFT_NOMINAL_XY, CUBE_HALF])
        self.block = builder.build(name="block")
        self.hold = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._hold_at = torch.full((self.num_envs,), -1, dtype=torch.int32, device=self.device)
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_robot(b)
            xyz = torch.zeros(b, 3)
            xyz[:, 0] = LIFT_NOMINAL_XY[0] + (torch.rand(b) * 2 - 1) * LIFT_XY_RANGE
            xyz[:, 1] = LIFT_NOMINAL_XY[1] + (torch.rand(b) * 2 - 1) * LIFT_XY_RANGE
            xyz[:, 2] = CUBE_HALF
            yaw = (torch.rand(b) * 2 - 1) * (np.pi / 2)
            q = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], dim=-1)
            self.block.set_pose(Pose.create_from_pq(p=xyz, q=q))
            self.goal_pos[env_idx] = xyz + torch.tensor([0.0, 0.0, LIFT_GOAL])
            self.hold[env_idx] = 0
            self._hold_at[env_idx] = -1

    @property
    def target_pos(self):
        return self.goal_pos

    def evaluate(self):
        is_grasped = self.agent.is_grasping(self.block)
        lifted = self.block.pose.p[:, 2] - CUBE_HALF > LIFT_HEIGHT
        # evaluate() also runs on reset() and on extra get_info()/get_obs() calls, so count once per env step.
        step = self._elapsed_steps.to(torch.int32)
        new = step != self._hold_at
        self.hold = torch.where(new, torch.where(lifted & is_grasped, self.hold + 1, torch.zeros_like(self.hold)), self.hold)
        self._hold_at = step.clone()
        return dict(success=self.hold >= HOLD_STEPS, is_grasped=is_grasped, lifted=lifted)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # PickCube-style staged reward (spec §1). Terms go into `info` (returned by step) for per-term logging.
        b = self.block.pose.p
        G, L = info["is_grasped"].float(), info["lifted"].float()
        info["r_reach"] = W_REACH * (1 - torch.tanh(5 * torch.linalg.norm(self.agent.grasp_center - b, axis=1)))
        info["r_grasp"] = W_GRASP * G
        info["r_lift"] = W_LIFT * (1 - torch.tanh(5 * (self.goal_pos[:, 2] - b[:, 2]).clamp_min(0))) * G
        qvel_arm = self.agent.robot.get_qvel()[:, :5]
        info["r_static"] = W_STATIC * (1 - torch.tanh(5 * torch.linalg.norm(qvel_arm, axis=1))) * G * L
        info["r_action"] = -W_ACTION * (action ** 2).sum(-1)
        reward = info["r_reach"] + info["r_grasp"] + info["r_lift"] + info["r_static"] + info["r_action"]
        return torch.where(info["success"], torch.full_like(reward, R_SUCCESS), reward)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / R_SUCCESS
