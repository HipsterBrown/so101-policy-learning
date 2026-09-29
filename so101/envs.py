"""SO101Reach-v1 and SO101Lift-v1: same 26-float state obs and joint-delta action, different reset/success/reward."""
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
            self.goal_pos[env_idx] = xyz + torch.tensor([0.0, 0.0, LIFT_HEIGHT])
            self.hold[env_idx] = 0

    @property
    def target_pos(self):
        return self.goal_pos

    def evaluate(self):
        is_grasped = self.agent.is_grasping(self.block)
        lifted = self.block.pose.p[:, 2] - CUBE_HALF > LIFT_HEIGHT
        # ponytail: hold counter lives in evaluate(), which ManiSkill calls once per step (and once at reset,
        # when the block is on the ground). Move to _after_control_step if anything starts calling evaluate() extra.
        self.hold = torch.where(lifted & is_grasped, self.hold + 1, torch.zeros_like(self.hold))
        return dict(success=self.hold >= HOLD_STEPS, is_grasped=is_grasped, lifted=lifted)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)  # staged reward is Phase 2

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info)
