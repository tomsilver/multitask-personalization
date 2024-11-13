"""Utilities specific to pybullet environment, skills, etc."""

from dataclasses import dataclass, field

import numpy as np
from assistive_gym.envs.agents.human import Human
from assistive_gym.envs.human_creation import HumanCreation
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.joint import JointPositions


@dataclass(frozen=True)
class HumanSpec:
    """Defines the spec for a human user in the pybullet environment."""

    impairment: str = "none"
    gender: str = "male"
    subject_id: int = 1
    condition: str = "limit_4"
    base_pose: Pose = Pose(position=(1.0, 0.53, 0.39))
    init_joints: JointPositions = field(
        default_factory=lambda: [
            0.0,
            0.0,
            0.0,
            0.08726646,
            0.0,
            0.0,
            -1.57079633,
            0.0,
            0.0,
            0.0,
        ]
    )
    setup_joints: dict[str, float] = field(
        default_factory=lambda: {
            "right_elbow": -90,
            "left_elbow": -90,
            "right_hip_x": -90,
            "right_knee": 80,
            "left_hip_x": -90,
            "left_knee": 80,
            "head_x": 0,
            "head_y": 0,
            "head_z": 0,
        }
    )


def create_human_from_spec(
    human_spec: HumanSpec, rng: np.random.Generator, physics_client_id: int
) -> Human:
    """Create a human in pybullet from a specification."""
    human_creation = HumanCreation(physics_client_id, np_random=rng, cloth=False)
    human = Human([], controllable=False)
    human.init(
        human_creation,
        static_human_base=True,
        impairment=human_spec.impairment,
        gender=human_spec.gender,
        config=None,
        id=physics_client_id,
        np_random=rng,
    )
    set_pose(human.body, human_spec.base_pose, physics_client_id)

    # Use some default joint positions from assistive gym first.
    joints_positions = [
        (getattr(human, f"j_{name}"), value)
        for name, value in human_spec.setup_joints.items()
    ]
    human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)
    # Now set arm joints using scene spec.
    human.set_joint_angles(
        human.right_arm_joints,
        human_spec.init_joints,
    )

    return human
