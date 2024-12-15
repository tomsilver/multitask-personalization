"""Utilities specific to pybullet environment, skills, etc."""

from dataclasses import dataclass, field

import numpy as np
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions
import pybullet as p

# TODO remove assistive-gym dependency and change pybullet version


@dataclass(frozen=True)
class HumanSpec:
    """Defines the spec for a human user in the pybullet environment."""

    base_pose: Pose = Pose(position=(2.0, 0.53, 0.51))

    # Default arm joints.
    init_joints: JointPositions = field(
        default_factory=lambda: [0.0, 0.0, 0.0, np.pi / 10, 0.0, 0.0, -np.pi / 2]
    )
    # Arm joints during reading.
    reading_joints: JointPositions = field(
        default_factory=lambda: [0.0, 0.0, 0.0, np.pi / 10, 0.0, 0.0, -np.pi / 2]
    )
    # Arm joints for reverse handover.
    reverse_handover_joints: JointPositions = field(
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


def create_human_from_spec(
    human_spec: HumanSpec, physics_client_id: int
) -> SingleArmPyBulletRobot:
    """Create a human in pybullet from a specification."""
    human = create_pybullet_robot("assistive-human", physics_client_id, base_pose=human_spec.base_pose)
    human.set_joints(human_spec.init_joints)
    for joint_name, joint_value in human_spec.setup_joints.items():
        urdf_name = human_spec.get_joint_urdf_name(joint_name)
        joint_idx = human.joint_from_name(urdf_name)
        p.resetJointState(human.robot_id, joint_idx, joint_value,
                            physicsClientId=physics_client_id)
    return human
