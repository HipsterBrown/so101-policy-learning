"""ManiSkill agent for the SO-101 (TheRobotStudio so101_new_calib.urdf), modeled on ManiSkill's SO-100."""
import copy
from pathlib import Path

import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation as R

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from so101.ik import GRIPPER_TCP, load_chain

URDF = Path(__file__).resolve().parents[1] / "assets/so101/so101_new_calib.urdf"

# TCP in the gripper_link frame: gripper_link -> Viam tool (fixed rotation) -> GripperTCPPose.
_R_TOOL = load_chain().fixed[-1][:3, :3]
TCP_P = _R_TOOL @ GRIPPER_TCP
_xyzw = R.from_matrix(_R_TOOL).as_quat()
TCP_Q = np.array([_xyzw[3], *_xyzw[:3]])  # sapien uses wxyz

# Gripper joint targets (rad). Calibration knobs: OPEN must clear a 3 cm cube, CLOSED squeezes past contact.
GRIPPER_OPEN = 0.5  # finger pad stays level with a 3 cm cube and clears it by ~54 mm (Task 3 geometry review)
GRIPPER_CLOSED = -0.17

# Jaw opening directions in each jaw link's own frame, for is_grasping. The fixed jaw sits at -x of the
# moving jaw in gripper_link. Knobs: if the scripted grasp lifts the block but is_grasped stays False,
# flip these signs.
FIXED_OPEN_AXIS = (-1.0, 0.0, 0.0)
MOVING_OPEN_AXIS = (1.0, 0.0, 0.0)


@register_agent()
class SO101(BaseAgent):
    uid = "so101"
    urdf_path = str(URDF)
    urdf_config = dict(
        _materials=dict(gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)),
        link=dict(
            gripper_link=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            moving_jaw_so101_v1_link=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )
    keyframes = dict(zero=Keyframe(qpos=np.zeros(6), pose=sapien.Pose()))
    arm_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    gripper_joint_names = ["gripper"]

    @property
    def _controller_configs(self):
        joints = [j.name for j in self.robot.active_joints]
        pd_joint_pos = PDJointPosControllerConfig(
            joints, lower=None, upper=None, stiffness=[1e3] * 6, damping=[1e2] * 6,
            force_limit=100, normalize_action=False,
        )
        # Same bounds as SO-100: 0.05 rad/step on the arm keeps motion servo-friendly.
        pd_joint_delta_pos = PDJointPosControllerConfig(
            joints, [-0.05] * 5 + [-0.2], [0.05] * 5 + [0.2], stiffness=[1e3] * 6, damping=[1e2] * 6,
            force_limit=100, use_delta=True, use_target=False,
        )
        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True
        return deepcopy_dict(dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
        ))

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["gripper_link"]  # fixed jaw is part of this link
        self.finger2_link = self.robot.links_map["moving_jaw_so101_v1_link"]

    @property
    def tcp_pose(self) -> Pose:
        offset = Pose.create_from_pq(p=torch.tensor(TCP_P, dtype=torch.float32),
                                     q=torch.tensor(TCP_Q, dtype=torch.float32), device=self.device)
        return self.finger1_link.pose * offset

    @property
    def tcp_pos(self):
        return self.tcp_pose.p

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
        l_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        lforce = torch.linalg.norm(l_forces, axis=1)
        rforce = torch.linalg.norm(r_forces, axis=1)
        l_axis = torch.tensor(FIXED_OPEN_AXIS, device=self.device, dtype=torch.float32)
        r_axis = torch.tensor(MOVING_OPEN_AXIS, device=self.device, dtype=torch.float32)
        ldir = self.finger1_link.pose.to_transformation_matrix()[..., :3, :3] @ l_axis
        rdir = self.finger2_link.pose.to_transformation_matrix()[..., :3, :3] @ r_axis
        langle = common.compute_angle_between(ldir, l_forces)
        rangle = common.compute_angle_between(rdir, r_forces)
        lflag = (lforce >= min_force) & (torch.rad2deg(langle) <= max_angle)
        rflag = (rforce >= min_force) & (torch.rad2deg(rangle) <= max_angle)
        return lflag & rflag

    def is_static(self, threshold=0.2):
        qvel = self.robot.get_qvel()[:, :-1]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold
