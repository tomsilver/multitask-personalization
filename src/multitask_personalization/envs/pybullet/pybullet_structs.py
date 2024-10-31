"""States, actions, and other data structures for the pybullet environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions


@dataclass(frozen=True)
class PyBulletState:
    """A state in the pybullet environment."""

    robot_base: Pose
    robot_joints: JointPositions
    human_base: Pose
    human_joints: JointPositions
    cup_pose: Pose
    book_poses: list[Pose]
    book_descriptions: list[str]
    grasp_transform: Pose | None
    held_object: str | None = None


class GripperAction(Enum):
    """Open or close the gripper."""

    OPEN = 1
    CLOSE = 2


PyBulletAction: TypeAlias = tuple[int, JointPositions | GripperAction | None]  # OneOf
