"""Utilities specific to pybullet environment, skills, etc."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.human import Human
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot


@dataclass(frozen=True)
class HumanSpec:
    """Defines the spec for a human user in the pybullet environment."""

    base_pose: Pose = Pose(position=(2.0, 0.53, 0.51))
    grasp_transform: Pose = Pose((0, 0, 0), (-np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2))
    # TODO
    # reading_pose: Pose = multiply_poses(base_pose, Pose.from_rpy((-0.3, -0.5, 0.0), (np.pi / 2, -np.pi / 6, np.pi / 4)))


@dataclass(frozen=True)
class AssistiveHumanSpec(HumanSpec):
    """Avatar derived from assistive gym."""

    # Default arm joints.
    init_joints: JointPositions = field(
        default_factory=lambda: [0.0, 0.0, 0.0, np.pi / 10, 0.0, -np.pi / 4, -np.pi / 2]
    )

    # Joints for the rest of the human body that remain static.
    setup_joints: dict[str, float] = field(
        default_factory=lambda: {
            "left_elbow": -np.pi / 3,
            "right_hip_x": -np.pi / 2,
            "left_hip_x": -np.pi / 2,
            "head_z": -np.pi / 3,
        }
    )

    # Used by some ROM models.
    impairment: str = "none"
    gender: str = "male"
    subject_id: int = 1
    condition: str = "limit_4"

    def get_joint_urdf_name(self, human_readable_name: str) -> str:
        """Look up known joints in the URDF."""
        # Use https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/
        # or another online URDF viewer to add new ones.
        known_names = {
            "right_hip_x": "joint29",
            "left_hip_x": "joint36",
            "left_elbow": "joint21",
            "head_z": "joint10",
            "shoulder_x": "joint1",
            "shoulder_y": "joint2",
            "shoulder_z": "joint3",
            "right_elbow": "joint14",
        }
        return known_names[human_readable_name]


@dataclass(frozen=True)
class SmoothHumanSpec(HumanSpec):
    """Smoother robot with right arm animated."""

    base_pose: Pose = Pose(position=(2.0, 0.53, 0.25))

    # right_arm_kwargs: dict[str, Any] = field(
    #     default_factory=lambda: {
    #         "home_joint_positions": [-0.5, 1.0, 0.56, -0.48, -0.31, 0.0, -1.44],
    #     }
    # )

    right_leg_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "home_joint_positions": [-np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    left_leg_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "home_joint_positions": [-np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def create_human_from_spec(
    human_spec: HumanSpec, physics_client_id: int
) -> SingleArmPyBulletRobot:
    """Create a human in pybullet from a specification."""
    if isinstance(human_spec, AssistiveHumanSpec):
        return create_assistive_human_from_spec(human_spec, physics_client_id)
    if isinstance(human_spec, SmoothHumanSpec):
        return create_smooth_human_from_spec(human_spec, physics_client_id)
    raise NotImplementedError(
        f"Creating a human from spec of type {type(human_spec)} is not implemented."
    )


def create_assistive_human_from_spec(
    human_spec: AssistiveHumanSpec, physics_client_id: int
) -> SingleArmPyBulletRobot:
    """Create a human in pybullet from a specification."""
    human = create_pybullet_robot(
        "assistive-human", physics_client_id, base_pose=human_spec.base_pose
    )
    human.set_joints(human_spec.init_joints)
    for joint_name, joint_value in human_spec.setup_joints.items():
        urdf_name = human_spec.get_joint_urdf_name(joint_name)
        joint_idx = human.joint_from_name(urdf_name)
        p.resetJointState(
            human.robot_id, joint_idx, joint_value, physicsClientId=physics_client_id
        )
    return human


def create_smooth_human_from_spec(
    human_spec: SmoothHumanSpec, physics_client_id: int
) -> SingleArmPyBulletRobot:
    """Create a smoother human in pybullet from a specification."""
    human = Human(
        physics_client_id,
        base_pose=human_spec.base_pose,
        # right_arm_kwargs=human_spec.right_arm_kwargs,
        right_leg_kwargs=human_spec.right_leg_kwargs,
        left_leg_kwargs=human_spec.left_leg_kwargs,
    )
    return human.right_arm
