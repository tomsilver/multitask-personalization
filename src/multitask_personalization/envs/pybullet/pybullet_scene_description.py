"""Defines the initialization of a scene in a pybullet environment."""

from __future__ import annotations

from dataclasses import dataclass, field

from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions


@dataclass(frozen=True)
class PyBulletSceneDescription:
    """Container for default hyperparameters."""

    robot_name: str = "kinova-gen3"  # must be 7-dof and have fingers
    robot_base_pose: Pose = Pose((0.0, 0.0, 0.0))
    initial_joints: JointPositions = field(
        default_factory=lambda: [
            -4.3,
            -1.6,
            -4.8,
            -1.8,
            -1.4,
            -1.1,
            1.6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    robot_max_joint_delta: float = 0.5

    robot_stand_pose: Pose = Pose((0.0, 0.0, -0.2))
    robot_stand_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    robot_stand_half_extents: tuple[float, float, float] = (0.1, 0.1, 0.2)

    human_base_pose: Pose = Pose(position=(1.0, 0.53, 0.39))
    human_joints: JointPositions = field(
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

    wheelchair_base_pose: Pose = Pose(position=(1.0, 0.5, -0.46))

    table_pose: Pose = Pose(position=(-0.5, 0.0, -0.2))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.1, 0.3, 0.2)

    object_pose: Pose = Pose(position=(-0.5, 0.0, 0.05))
    object_rgba: tuple[float, float, float, float] = (0.9, 0.6, 0.3, 1.0)
    object_radius: float = 0.025
    object_length: float = 0.1

    camera_distance: float = 2.0
