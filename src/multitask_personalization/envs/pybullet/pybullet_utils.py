"""Utilities specific to pybullet environment, skills, etc."""

import numpy as np
from assistive_gym.envs.agents.human import Human
from assistive_gym.envs.human_creation import HumanCreation
from pybullet_helpers.geometry import set_pose

from multitask_personalization.envs.pybullet.pybullet_task_spec import HumanSpec


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
    # Now set arm joints using task spec.
    human.set_joint_angles(
        human.right_arm_joints,
        human_spec.init_joints,
    )

    return human
